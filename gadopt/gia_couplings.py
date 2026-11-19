r"""Global scalar fields and surface loads of the self-gravitating GIA solver.

A coupling is one piece of physics that the solver adds to the displacement
and the potential: a fluid core, the rotational closure, the centre-of-mass
frame, the sea-level equation, or a prescribed surface load. The user gives
them as a list,

    solver = SelfGravitatingGIASolver(
        u, psi, approximation, ...,
        couplings=[FluidCore(boundary=RC, rho_core=rho_c),
                   Rotation(C=C), CentreOfMass(), SeaLevel(...)])

and the solver does the rest. A coupling owns its `Real` unknowns, its
residual rows and its diagnostics. It does not own a space, a solver or a
time step.

## The protocol

The solver runs three ordered stages over every coupling, so the order of
the list does not change the result:

1. **Potentials.** `Rotation` publishes the rotational potential `psi_rot`
   in the context. Nothing else writes a potential.
2. **Sheets.** Every coupling that moves mass on a surface returns a
   `MassSheet`: the fluid core its CMB sheet, the sea level the ocean and ice
   sheet (which reads `psi_rot`), a surface load its prescribed sheet.
3. **Residuals.** Every coupling writes its rows. `Rotation` and
   `CentreOfMass` read the whole list of sheets.

The residual of a coupling receives the whole mixed solution and all test
functions through one `CouplingContext`, because three couplings write into
rows that are not their own: `FluidCore` into the displacement and potential
rows, `SeaLevel` into the displacement and potential rows, and `CentreOfMass`
into the potential, `Shift` and rotation rows through the variation of the
sea-level sheet.

## What a sheet is used for, and what it is not used for

A sheet is consumed by the rotation rows (the sheet's part of `dI_i3`), the
centre-of-mass rows (its part of the mass dipole, and the variation of that
part in the frame column), the 2-D monopole datum and the scale of the net
mass check. **The potential row does not sum the sheets.** Each coupling
writes its own sheet into the potential row as part of its own residual, and
a second entry would count it twice.

## Rules every coupling follows

- Unknowns and coefficients are UFL. A coupling exposes UFL expressions and
  `Real` Functions, never Python numbers, so that a functional built from
  them keeps its dependency on the tape.
- Every derivative is expanded at once (`expand_derivatives`). An
  unevaluated derivative keeps a parent-mesh integral in every sub-block of
  the Jacobian after Firedrake splits the form, and the Slate compiler of the
  block-0 preconditioners then refuses the block.
- A `Real` field of the parent, its test function, a constant and the row
  scalings `c`, `B_mu` and `Lambda` (which live on the parent or have no
  mesh; the solver refuses another mesh) on a facet of the mechanics mesh go
  through `CouplingContext.on_parent_facet`, which restricts them where the
  facet is an interior facet of the parent.
- A coupling parameter that the caller gives (a number, a `Constant`, or a
  `Real` Function of either mesh) goes through
  `CouplingContext.on_coupling_facet` in a facet integrand. That method
  restricts the parameter if it lives on the parent and leaves it alone if
  it lives on the mechanics mesh, so the caller can give it on either mesh.
- Each coupling refuses an invalid combination in `check`, before any form
  is built.
"""

import abc
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from firedrake import (Constant, FacetNormal, Form, Measure,
                       SpatialCoordinate, TestFunction, assemble, dot,
                       ds, grad, split, sqrt)
from mpi4py import MPI
from ufl import derivative as ufl_derivative
from ufl.algorithms import extract_coefficients
from ufl.algorithms.ad import expand_derivatives
from ufl.domain import extract_domains
from ufl.sobolevspace import H1, H2, HInf

from .momentum_equation import rotational_potential
from .sea_level_sheet import ocean_fraction, softplus_column, water_fraction
from .utility import (CombinedSurfaceMeasure, ensure_constant,
                      interior_facet_integrand, scalar_value)

__all__ = [
    "CentreOfMass",
    "Coupling",
    "CouplingContext",
    "FluidCore",
    "MassSheet",
    "Rotation",
    "SeaLevel",
    "SurfaceLoad",
    "scalar_value",
]


@dataclass(frozen=True)
class MassSheet:
    r"""A surface mass density on one boundary of the mechanics mesh.

    The record every consumer of a sheet reads. All sheets of the coupled
    solver sit on facets of the mechanics mesh, so a position polynomial in
    a sheet integral is written in the mechanics mesh's coordinates.

    Attributes:
      name: a label for messages.
      density: `sigma` at the current solution, UFL on `measure`. Positive
        where mass is added.
      test_variation: the Gateaux derivative of `density` with respect to the
        mixed solution, in the direction of the mixed test function, expanded.
        `None` for a prescribed sheet, whose density does not depend on the
        solution. The centre-of-mass column reads it.
      measure: the facet measure, already restricted to the sheet's tag.
      in_monopole_datum: whether the 2-D monopole datum counts this sheet's
        mass. The datum is evaluated at the start of a solve, so a sheet that
        depends on the solution enters it one solve late. That is accepted
        for the fluid-core sheet and refused for the sea-level sheet, whose
        net mass the `Shift` row sets to zero (`SeaLevel`).
    """

    name: str
    density: Any
    test_variation: Any
    measure: Measure
    in_monopole_datum: bool


class CouplingContext:
    """What the couplings of one solver share: fields, measures, scalings, sheets.

    Built by the solver once its mixed space and solution exist, and passed
    to every stage of every coupling. A coupling reads its own fields and the
    fields of the others by name, never by index.

    The attributes and methods below are the whole interface a coupling may
    rely on. Of the solver itself a coupling uses only `real_field(name)`,
    for the `Real` Functions its diagnostics return.

    Attributes:
      solution: the mixed solution `Function`.
      space: the mixed function space.
      fields: `{name: component of split(solution)}`.
      tests: `{name: component of the test functions}`.
      approximation: the solver's approximation.
      mesh: the mechanics mesh.
      potential_mesh: the parent mesh of the potential and the `Real` fields.
      cross_mesh: whether the two meshes differ.
      dx_m: the mantle volume measure, intersected with the parent.
      boundary_quad_degree: the quadrature degree of every facet integral of
        the couplings, the degree of the DtN boundary form.
      mechanics_quad_degree: the quadrature degree of the momentum
        equation, which its weak boundary terms also use.
      scaling_factor: `c`, the factor of the whole momentum residual.
      row_scale_B_mu: `B_mu`, floored to one where it is exactly zero.
      theta_psi: `c B_mu / Lambda`, the scaling of the potential row.
      Lambda: the self-gravity number.
      four_pi_G: `4 pi G` of the boundary form, which equals `Lambda`.
      bcs: the caller's mechanics boundary conditions, for the checks.
      rotational_potential: `None`, or after stage 1 a callable
        `(mesh, facet=False, split=None) -> psi_rot`.
      sheets: the list of `MassSheet`s after stage 2.
    """

    def __init__(self, *, solution, fields, tests, approximation, mesh,
                 potential_mesh, dx_m, boundary_quad_degree, scaling_factor,
                 row_scale_B_mu, Lambda, four_pi_G, bcs, field_names,
                 mechanics_quad_degree=None):
        self.solution = solution
        self.space = solution.function_space()
        self.fields = fields
        self.tests = tests
        self.approximation = approximation
        self.mesh = mesh
        self.potential_mesh = potential_mesh
        self.cross_mesh = mesh is not potential_mesh
        self.dx_m = dx_m
        self.boundary_quad_degree = boundary_quad_degree
        self.mechanics_quad_degree = mechanics_quad_degree
        self.scaling_factor = scaling_factor
        self.row_scale_B_mu = row_scale_B_mu
        self.Lambda = Lambda
        self.theta_psi = scaling_factor * row_scale_B_mu / Lambda
        self.four_pi_G = four_pi_G
        self.bcs = bcs
        #: The field names in mixed-space order, for `split_of`.
        self.field_names = tuple(field_names)
        self.rotational_potential = None
        self.sheets = []

    def field_degree(self, name) -> int:
        """The polynomial degree of the element of the field `name`."""
        index = self.field_names.index(name)
        return self.space[index].ufl_element().embedded_superdegree

    def split_of(self, function) -> dict:
        """`{name: component}` of `split(function)`, for a copy of the solution."""
        return dict(zip(self.field_names, split(function)))

    def on_parent_facet(self, value):
        """`value`, a `Real` field of the parent or a constant, on a coupling facet.

        The coupling facets are exterior facets of the mechanics mesh and, in
        a cross-mesh system, interior facets of the parent. UFL requires every
        `Argument` of the parent to carry a restriction there, including the
        test and trial functions of a `Real` field. `avg` is exact for a
        field that is continuous across the facet, which a `Real` field is,
        and it does not depend on which side UFL calls `'+'`. A field of the
        mechanics mesh must not go through here: UFL refuses a restriction on
        its exterior facets.
        """
        return interior_facet_integrand(value) if self.cross_mesh else value

    def parent_trace(self, value):
        """A continuous field of the parent (the potential) on a coupling facet."""
        return interior_facet_integrand(value) if self.cross_mesh else value

    def on_coupling_facet(self, value):
        """A coupling parameter on a coupling facet, restricted as its mesh requires.

        A coupling parameter (`FluidCore(rho_core=, g=)`, `Rotation(Omega_sq=)`,
        `SeaLevel(g_surface=)`, `SurfaceLoad(g_surface=)`) can be a number, a
        `Constant`, or a `Real` Function of the mechanics mesh or of the
        parent. In a cross-mesh system a coupling facet is an exterior facet
        of the mechanics mesh and an interior facet of the parent, and UFL
        has a different rule for each mesh there: a quantity of the parent
        must be restricted (`interior_facet_integrand`), because the
        derivative with respect to a `Real` Function of the parent is an
        `Argument` that UFL refuses unrestricted on an interior facet; a
        quantity of the mechanics mesh must not be restricted, because UFL
        refuses a restriction on an exterior facet (`Inconsistent
        restrictions`). This method reads the mesh of `value` from its
        terminals and applies the rule of that mesh. It does not change the
        form; each facet integrand calls it on each parameter where the
        integrand is built.

        A parameter with no mesh (a number or a `Constant`) is returned as it
        is. A parameter whose terminals live on both meshes is refused,
        because no single restriction is valid for it.

        `avg` gives the value on the facet only for a parent field that is
        continuous across it: a `Real` Function, or a field of an
        H1-conforming element (continuous Lagrange). For a discontinuous
        parent field (DG, Quadrature) `avg` is the mean of the mantle side
        and the side outside the mantle, which is not the mantle's value, so
        such a field is refused. The same parameter on the mechanics mesh
        has one side only and is accepted.

        Args:
          value: a UFL expression of the parameter.

        Returns:
          `value`, restricted with `avg` if it lives on the parent.

        Raises:
          ValueError: If `value` mixes quantities of the two meshes, lives
            on a mesh that is neither of them, or holds a discontinuous
            field of the parent.
        """
        value = ensure_constant(value)
        if not self.cross_mesh:
            return value
        meshes = set(extract_domains(value))
        if not meshes or meshes == {self.mesh}:
            return value
        if meshes == {self.potential_mesh}:
            for coefficient in extract_coefficients(value):
                element = coefficient.ufl_element()
                if element.sobolev_space not in (H1, H2, HInf):
                    raise ValueError(
                        f"The coupling parameter {value} holds {coefficient}, a "
                        f"field of the parent in the discontinuous element "
                        f"{element}. A coupling facet is an interior facet of "
                        "the parent, and the average over its two sides is not "
                        "the mantle's value of a discontinuous field. Give the "
                        "field on the mechanics mesh, or in a continuous or "
                        "Real element of the parent.")
            return interior_facet_integrand(value)
        raise ValueError(
            "A coupling parameter must be a number, a Constant, or a Real "
            "Function of one mesh: the mechanics mesh or the parent of the "
            f"potential. The value {value} lives on {len(meshes)} meshes, or on "
            "a mesh that is neither. Give each parameter as a Function of one "
            "of those two meshes.")

    def surface_measure(self, tag, degree=None) -> Measure:
        """The facet measure of a coupling on `tag`, at the boundary degree.

        The mechanics mesh's own `ds`, intersected with the parent's interior
        facet measure `dS` in a cross-mesh system. The facet-to-facet pairing
        is the one that evaluates a parent field at the right points:
        pairing the `ds` with the parent's cell measure instead assembles
        without an error and is 21 percent wrong for a parent field.
        `FacetNormal` of the mechanics mesh is then the outward normal of the
        mantle, which every coupling uses.

        On an extruded mechanics mesh the CMB is the bottom surface,
        `"bottom"`, and `CombinedSurfaceMeasure` maps the tag. An extruded
        mesh is never cross-mesh, because `CombinedSurfaceMeasure` takes no
        `intersect_measures`.

        Args:
          tag: the boundary tag on the mechanics mesh.
          degree: the quadrature degree; `None` takes `boundary_quad_degree`.
        """
        degree = self.boundary_quad_degree if degree is None else degree
        if self.mesh.extruded:
            if self.cross_mesh:
                raise NotImplementedError(
                    "An extruded mechanics mesh with a separate potential mesh "
                    "is not supported: CombinedSurfaceMeasure takes no "
                    "intersect_measures, so a coupling facet would lose the "
                    "facet-to-facet pairing with the parent.")
            return CombinedSurfaceMeasure(domain=self.mesh, degree=degree)(tag)
        kwargs = {"domain": self.mesh, "degree": degree}
        if self.cross_mesh:
            kwargs["intersect_measures"] = (
                Measure("dS", domain=self.potential_mesh),)
        return ds(**kwargs)(tag)

    def plain_surface_measure(self, tag) -> Measure:
        """The mechanics mesh's own `ds(tag)`, for a geometric check only."""
        if self.mesh.extruded:
            return CombinedSurfaceMeasure(
                domain=self.mesh, degree=self.boundary_quad_degree)(tag)
        return Measure("ds", domain=self.mesh)(tag)

    def check_closed_surface(self, tag, label) -> None:
        """Refuse a coupling boundary that is empty or only part of a circle or sphere.

        A tag of the wrong facet kind assembles to zero with at most a
        warning, so a coupling written on it would be absent from the
        residual with no error. The facet extent is compared with `2 pi R`
        (`4 pi R^2` in 3-D) at its own mean radius, to one percent, which
        catches a missing or a partial boundary and does not grade the mesh.
        """
        dss = self.plain_surface_measure(tag)
        try:
            extent = assemble(Constant(1.0) * dss)
        except (KeyError, LookupError):
            # A tag the mesh does not know raises as a bare `KeyError`.
            extent = 0.0
        if extent <= 0.0:
            raise ValueError(
                f"{label}: boundary {tag} has an empty measure on the mechanics "
                "mesh. A tag alone does not say whether its facets are interior "
                "or exterior; the coupling boundaries are exterior facets of "
                "the mechanics mesh.")
        X = SpatialCoordinate(self.mesh)
        radius = assemble(sqrt(dot(X, X)) * dss) / extent
        dim = self.mesh.geometric_dimension
        expected = 2 * np.pi * radius if dim == 2 else 4 * np.pi * radius ** 2
        if abs(extent - expected) > 1e-2 * expected:
            raise ValueError(
                f"{label}: boundary {tag} measures {extent:.6e} at mean radius "
                f"{radius:.6g}, against {expected:.6e} for a complete circle or "
                "sphere of that radius.")

    def has_strong_or_weak_displacement_bc(self, tag) -> bool:
        """Whether the caller's mechanics `bcs` constrain the displacement on `tag`."""
        return bool({"u", "un"} & set(self.bcs.get(tag, {})))


class Coupling(abc.ABC):
    """One coupling of the self-gravitating GIA solver.

    A coupling binds to one solver and refuses a second, because it keeps
    the context of that solver for its diagnostics.

    Subclasses set `order`, which fixes where their `Real` fields sit among
    the `Real` fields of the mixed space (after the DtN multipliers), so that
    the field order does not depend on the order of the `couplings` list.
    """

    #: Position of this coupling's `Real` fields among the `Real` fields.
    order: ClassVar[int]
    #: Whether a solver may hold more than one coupling of this class.
    repeatable: ClassVar[bool] = False

    def __init__(self) -> None:
        self._ctx = None
        self._solver = None

    @property
    def context(self) -> CouplingContext:
        """The context of the solver this coupling is bound to."""
        if self._ctx is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to a solver yet; pass it "
                "to SelfGravitatingGIASolver(couplings=[...]).")
        return self._ctx

    def real_field_names(self, dim: int) -> tuple[str, ...]:
        """The names of this coupling's `Real` unknowns, in their space order."""
        return ()

    def bind(self, solver, ctx: CouplingContext) -> None:
        """Bind to one solver. A second solver is refused."""
        if self._solver is not None and self._solver is not solver:
            raise RuntimeError(
                f"This {type(self).__name__} is already bound to another "
                "solver. Build one coupling object per solver.")
        self._solver = solver
        self._ctx = ctx

    def check(self, couplings) -> None:
        """Refuse an invalid combination with the boundaries or the other couplings."""

    def potentials(self, ctx: CouplingContext) -> None:
        """Stage 1: publish a potential in the context."""

    def sheets(self, ctx: CouplingContext) -> list[MassSheet]:
        """Stage 2: the mass sheets this coupling moves."""
        return []

    @abc.abstractmethod
    def residual(self, ctx: CouplingContext) -> Form:
        """Stage 3: this coupling's rows, as a form in the whole test function."""


def _find(couplings, cls):
    """The couplings of class `cls` in `couplings`."""
    return [c for c in couplings if isinstance(c, cls)]


class FluidCore(Coupling):
    r"""An inviscid, homogeneous, hydrostatic core, eliminated onto its boundary.

    The alternative to a rigid core (`un = 0` at the CMB in `bcs`). The core
    is not meshed. What it does to the mantle is push on it and carry mass
    across the displaced interface, and both are boundary terms, the
    variation of one energy on the CMB:

        E = c int_CMB [ B_mu rho_core (u.n) psi
                        + 0.5 B_mu rho_core g_0 (u.n)^2
                        + beta_core p_core (u.n) ] ds

    with `n` the mantle's outward normal (inward at the CMB, so `u.n = -u_r`),
    `c` the residual scaling, `p_core` one `Real` pressure and
    `beta_core = B_mu` floored to one at `B_mu = 0`. The `p_core` variation is
    `int_CMB u.n ds = 0`, so the incompressible core keeps its volume.

    **The spring carries `rho_core` alone, not the density contrast.** The
    hydrostatic prestress volume term already supplies the mantle half
    `-0.5 rho_0 g_0 u_r^2` at the CMB, so the net is the physical contrast
    spring. `buoyancy_density="contrast"` writes the old, too soft
    coefficient and exists only to reproduce baseline runs.

    With rotation, the rotational potential presses on the CMB as the
    gravitational one does, `+c B_mu rho_core (w.n) psi_rot`. That term is
    written by hand and not added to the energy, because the energy would
    then also give the sheet's part of the rotation row, which `Rotation`
    already writes.

    The CMB mass sheet `sigma_core = -rho_core (u.n)` is published for the
    rotation rows, the centre-of-mass rows and the 2-D monopole datum, which
    counts it one solve late.

    Args:
      boundary: the CMB tag on the mechanics mesh, or `"bottom"` on an
        extruded mesh.
      rho_core: the core density.
      rho_mantle: the mantle density at the CMB, used only by
        `buoyancy_density="contrast"`; `None` takes the approximation's.
      g: the reference gravity at the CMB; `None` takes the approximation's.
      buoyancy_density: `"core"` (the physical spring) or `"contrast"`, which
        writes the density contrast and is too soft by the mantle half; it
        exists to reproduce runs made with that coefficient.
    """

    order = 10

    def __init__(self, *, boundary, rho_core, rho_mantle=None, g=None,
                 buoyancy_density: str = "core") -> None:
        super().__init__()
        if buoyancy_density not in ("core", "contrast"):
            raise ValueError(
                f"FluidCore.buoyancy_density must be 'core' or 'contrast', not "
                f"{buoyancy_density!r}")
        self.boundary = boundary
        self.rho_core = rho_core
        self.rho_mantle = rho_mantle
        self.g = g
        self.buoyancy_density = buoyancy_density

    def real_field_names(self, dim):
        return ("core_pressure",)

    def check(self, couplings) -> None:
        ctx = self.context
        if ctx.has_strong_or_weak_displacement_bc(self.boundary):
            raise ValueError(
                f"Boundary {self.boundary} carries both a FluidCore and a 'u' or "
                "'un' condition. The fluid core is a traction proportional to "
                "the normal displacement, which 'un' pins; they are "
                "alternatives, and 'un = 0' is the rigid core.")
        ctx.check_closed_surface(self.boundary, "FluidCore")

    def measure(self) -> Measure:
        """The CMB measure, restricted to the boundary."""
        return self.context.surface_measure(self.boundary)

    def sheet_density(self, u):
        """`sigma_core = rho_core (u . rhat) = -rho_core (u . n)` for a displacement `u`.

        A facet quantity: `rho_core` is restricted as its mesh requires
        (`CouplingContext.on_coupling_facet`).
        """
        ctx = self.context
        return -ctx.on_coupling_facet(self.rho_core) * dot(u, FacetNormal(ctx.mesh))

    def sheets(self, ctx):
        return [MassSheet(
            name="fluid core",
            density=self.sheet_density(ctx.fields["displacement"]),
            # Linear in the displacement, so the variation is the density at
            # the displacement test function.
            test_variation=self.sheet_density(ctx.tests["displacement"]),
            measure=self.measure(),
            in_monopole_datum=True)]

    def energy(self, ctx) -> Form:
        """The CMB energy whose variation is the fluid-core condition."""
        approximation = ctx.approximation
        u = ctx.fields["displacement"]
        psi = ctx.fields["potential"]
        p_core = ctx.fields["core_pressure"]
        un = dot(u, FacetNormal(ctx.mesh))
        # The parameters of the CMB, each restricted as its mesh requires.
        rho_core = ctx.on_coupling_facet(self.rho_core)
        rho_mantle = ctx.on_coupling_facet(
            approximation.density if self.rho_mantle is None else self.rho_mantle)
        g0 = ctx.on_coupling_facet(approximation.g if self.g is None else self.g)
        B_mu = approximation.B_mu
        dss = self.measure()
        c = ctx.scaling_factor
        # Each integrand is written with its parent quantities, the row
        # scalings among them, under one restriction (`on_parent_facet`):
        # `c` and `B_mu` can be `Real` controls on the parent.
        # `-c B_mu sigma psi`, written through the sheet so that the sheet the
        # other rows read and the one the potential row carries are one
        # expression.
        E = -ctx.on_parent_facet(c * B_mu * psi) * self.sheet_density(u) * dss
        spring_rho = (rho_core - rho_mantle if self.buoyancy_density == "contrast"
                      else rho_core)
        E += ctx.on_parent_facet(c * B_mu) * 0.5 * spring_rho * g0 * un * un * dss
        # The uniform pressure that keeps the core volume. The floored B_mu
        # keeps the row non-empty at B_mu = 0.
        E += ctx.on_parent_facet(c * ctx.row_scale_B_mu * p_core) * un * dss
        return E

    def residual(self, ctx) -> Form:
        # One energy gives the (u, psi), (psi, u), (u, u), (u, p) and (p, u)
        # blocks, so they are transposes by construction.
        # In the direction of the mixed test function, given explicitly:
        # `ufl.derivative` would otherwise create a plain UFL argument, which
        # does not compare equal to Firedrake's test functions of the other
        # rows and makes the summed form invalid.
        F = expand_derivatives(ufl_derivative(
            self.energy(ctx), ctx.solution, TestFunction(ctx.space)))
        if ctx.rotational_potential is not None:
            w = ctx.tests["displacement"]
            n = FacetNormal(ctx.mesh)
            psi_rot = ctx.rotational_potential(ctx.mesh, facet=True)
            F += (ctx.on_parent_facet(ctx.scaling_factor * ctx.approximation.B_mu)
                  * ctx.on_coupling_facet(self.rho_core) * dot(w, n) * psi_rot
                  * self.measure())
        return F

    def pressure(self):
        """The core pressure, a `Real` Function (a view of the solution)."""
        return self._solver.real_field("core_pressure")


class Rotation(Coupling):
    r"""The rotational closure: the rotation-vector change `m` and its feedback.

    The rotation rows close

        K_i m_i = s_i dI_i3[u, sheets],   K = (C - A, C - A, C),  s = (+1, +1, -1)

    with `dI_i3 = int rho_0 grad(p_i) . u dx + sum_sheets int sigma p_i dS`,
    `p_3 = x^2 + y^2`, `p_1 = -x z`, `p_2 = -y z`. The rotational potential
    `psi_rot = Omega_sq sum_i m_i p_i` (negated centrifugal potential, the
    sign convention of `psi`) acts on the mantle as a body force, on the CMB
    through `FluidCore` and in the sea level through `SeaLevel`. Each row is
    scaled by `theta_rot_i = s_i c B_mu Omega_sq`, which makes the body force
    and the row transposes. A 2-D disc has the rotation-rate change `m_3`
    only.

    `C - A` is the dynamical ellipticity. It is tied to the fluid-limit Love
    number by MacCullagh's relation `C - A = Q k_s`, with
    `Q = 4 pi ahat^5 Omega_sq / (3 Lambda)` in the solver's units. With `k_s`
    given, `C - A` is that UFL expression of `Lambda` and `Omega_sq`, so a
    control on either reaches the closure on the tape. With both given, they
    must agree to `CONSISTENCY_RTOL`.

    Args:
      C: the polar moment of the reference body, `int rho_0 r^2 dV` in 2-D.
      C_minus_A: the dynamical ellipticity (3-D).
      k_s: the secular Love number; needs `surface_radius`.
      Omega_sq: the squared rotation rate in the units of the solver,
        `Omega^2 L / g_bar` (1.566e-3 for the Earth with `L` the mantle depth
        and `g_bar = 9.815`; `gadopt.gia_presets.OMEGA_SQ_EARTH`). Required,
        because the value depends on the non-dimensionalisation.
      surface_radius: the non-dimensional radius of the free surface,
        checked against the mechanics mesh's outer radius to two percent.
    """

    order = 20
    CLOSURE_SIGNS = (1.0, 1.0, -1.0)
    NAMES = ("m1", "m2", "m3")
    #: Relative tolerance of a given `C_minus_A` against `Q k_s`. An error
    #: `eps` in `C - A` changes the polar motion by about
    #: `eps (C - A) / [(C - A) - Q k_T]`, an amplification near 100 at 20 kyr
    #: on the Spada benchmark, so 1e-4 bounds that error near one percent.
    CONSISTENCY_RTOL = 1e-4
    #: Relative tolerance of `surface_radius` against the mesh's outer radius.
    SURFACE_RADIUS_RTOL = 2e-2

    def __init__(self, *, Omega_sq, C=None, C_minus_A=None, k_s=None,
                 surface_radius=None) -> None:
        super().__init__()
        self.C = C
        self.C_minus_A_given = C_minus_A
        self.k_s = k_s
        self.Omega_sq = ensure_constant(Omega_sq)
        self.surface_radius = surface_radius
        #: The closure constants `K_i`, UFL, set by `check`.
        self.closure_constants = {}

    def real_field_names(self, dim):
        return ("m3",) if dim == 2 else self.NAMES

    def check(self, couplings) -> None:
        ctx = self.context
        dim = ctx.mesh.geometric_dimension
        if self.surface_radius is not None:
            coords = ctx.mesh.coordinates.dat.data_ro
            local = (float(np.sqrt((coords ** 2).sum(axis=1)).max())
                     if coords.size else 0.0)
            measured = ctx.mesh.comm.allreduce(local, MPI.MAX)
            ahat = scalar_value(self.surface_radius)
            if abs(ahat - measured) > self.SURFACE_RADIUS_RTOL * measured:
                raise ValueError(
                    f"surface_radius={ahat:.6g} against the mechanics mesh's "
                    f"outer radius {measured:.6g}. Give the non-dimensional "
                    "radius of the free surface, a / L.")
        elif self.k_s is not None:
            raise ValueError(
                "Rotation(k_s=...) needs surface_radius: C - A = Q k_s with "
                "Q = a^5 Omega^2 / (3 G).")
        if self.C is None:
            raise ValueError("Rotation needs C, the polar moment, for the m3 row.")
        self.closure_constants[2] = ensure_constant(self.C)
        if dim == 2:
            return
        consistent = None
        if self.k_s is not None:
            # UFL, not a number: `Lambda` and `Omega_sq` may be controls.
            Q = (4 * np.pi * scalar_value(self.surface_radius) ** 5
                 * self.Omega_sq / (3 * ctx.Lambda))
            consistent = Q * ensure_constant(self.k_s)
        given = self.C_minus_A_given
        if given is None and consistent is None:
            raise ValueError(
                "Rotation in 3-D needs C_minus_A, or k_s and surface_radius.")
        if given is not None and consistent is not None:
            # The check reads numbers, computed from the factors: `float()`
            # of the composite raises when a factor is a `Real` Function.
            Q_value = (4 * np.pi * scalar_value(self.surface_radius) ** 5
                       * scalar_value(self.Omega_sq) / (3 * scalar_value(ctx.Lambda)))
            q = Q_value * scalar_value(self.k_s)
            relative = abs(scalar_value(given) - q) / abs(q)
            if relative > self.CONSISTENCY_RTOL:
                # Both readings of the mismatch: the C - A that k_s implies,
                # and the k_s that the given C - A implies, so the caller
                # sees which constant to change and to what.
                raise ValueError(
                    f"Rotation gives C_minus_A={scalar_value(given)!r} and "
                    f"k_s={scalar_value(self.k_s)!r}, which disagree by "
                    f"{relative:.3%} against C - A = Q k_s = {q!r}; the given "
                    f"C_minus_A implies k_s = (C - A) / Q = "
                    f"{scalar_value(given) / Q_value!r}. Give one of them, or "
                    "reconcile them.")
        constant = consistent if consistent is not None else ensure_constant(given)
        self.closure_constants[0] = self.closure_constants[1] = constant

    def slots(self, ctx):
        """`(m1, m2, m3)` field names, `None` where the space has no component."""
        names = self.real_field_names(ctx.mesh.geometric_dimension)
        return tuple(name if name in names else None for name in self.NAMES)

    def potentials(self, ctx) -> None:
        def psi_rot(mesh, facet=False, split=None):
            """`psi_rot` in the coordinates of `mesh`, from the rotation scalars."""
            fields = ctx.fields if split is None else split
            n_rot = 1 if mesh.geometric_dimension == 2 else 3
            values = [Constant(0.0) if name is None else fields[name]
                      for name in self.slots(ctx)]
            Omega_sq = self.Omega_sq
            if facet:
                values = [ctx.on_parent_facet(value) for value in values]
                Omega_sq = ctx.on_coupling_facet(Omega_sq)
            return rotational_potential(values[-n_rot:], mesh, Omega_sq=Omega_sq)
        ctx.rotational_potential = psi_rot

    @staticmethod
    def inertia_polynomial(i: int, X):
        """`p_i(x)`: `x^2 + y^2` for `i = 2`, `-x_i z` otherwise.

        `p_i` is the polynomial whose integral against the density gives the
        inertia component `I_i3`: `R^2 - z^2 = x^2 + y^2` for `i = 2`, and
        `-x_i z` for the two polar-wander components. Those two need the
        third coordinate, so in 2-D only the rotation-rate component (`i = 2`)
        exists.

        Args:
          i: The component, 0 or 1 for polar wander and 2 for the rotation rate.
          X: The `SpatialCoordinate` of the mesh.

        Returns:
          The UFL polynomial `p_i(X)`.

        Raises:
          ValueError: If `i` is 0 or 1 and `X` has fewer than three components.
        """
        if i == 2:
            return X[0] ** 2 + X[1] ** 2
        # Without this check, UFL would raise `Index out of bounds` at `X[2]`,
        # which does not say what is wrong.
        if len(X) < 3:
            raise ValueError(
                f"The inertia component {i} needs a third coordinate; a 2-D "
                "disc has only the rotation-rate component m_3 (index 2).")
        return -X[i] * X[2]

    def inertia_form(self, ctx, i, test=None, scale=None, parameter=None):
        """`dI_i3` at the current solution, weighted by `parameter * scale * test`.

        Args:
          ctx: the `CouplingContext`.
          i: the component of `dI_i3`, 0 to 2.
          test: a test function of a `Real` field of the parent, or `None`
            for 1.
          scale: a factor made of quantities of the parent (row scalings),
            or `None` for 1.
          parameter: a coupling parameter of either mesh (`Omega_sq`), or
            `None` for 1. It is restricted on the sheets as its own mesh
            requires, so it is kept apart from `scale`.

        Returns:
          The form of `dI_i3`: the mantle term on `dx_m` and one term per
          sheet on its facet measure.
        """
        weight = Constant(1.0) if test is None else test
        scale = Constant(1.0) if scale is None else scale
        parameter = Constant(1.0) if parameter is None else parameter
        # The weight and the scale, both of the parent, under one restriction
        # on the sheets; the parameter under the restriction of its mesh.
        facet_weight = (ctx.on_parent_facet(scale * weight)
                        * ctx.on_coupling_facet(parameter))
        p = self.inertia_polynomial(i, SpatialCoordinate(ctx.mesh))
        u = ctx.fields["displacement"]
        form = (parameter * scale * weight * ctx.approximation.density
                * dot(grad(p), u) * ctx.dx_m)
        for sheet in ctx.sheets:
            form += facet_weight * p * sheet.density * sheet.measure
        return form

    def row_scale(self, ctx, i):
        """`s_i c B_mu`: the part of the scaling of row `i` made of parent quantities."""
        return self.CLOSURE_SIGNS[i] * ctx.scaling_factor * ctx.row_scale_B_mu

    def theta_rot(self, ctx, i):
        """`s_i c B_mu Omega_sq`: the scaling of row `i`."""
        return self.row_scale(ctx, i) * self.Omega_sq

    def residual(self, ctx) -> Form:
        F = Form([])
        # `nu_i` is a `Real` test function, whose basis function is the
        # constant 1, so `K_i m_i` is divided by the volume of the measure it
        # is integrated on: the row means `K_i m_i - s_i dI_i3 = 0`.
        volume = assemble(Constant(1.0) * ctx.dx_m)
        for i, name in enumerate(self.slots(ctx)):
            if name is None:
                continue
            m_i, nu_i = ctx.fields[name], ctx.tests[name]
            theta = self.theta_rot(ctx, i)
            F += theta * (self.closure_constants[i] * m_i / volume) * nu_i * ctx.dx_m
            # `theta_rot` split into its parent part and `Omega_sq`, which
            # can be a `Real` Function of either mesh.
            F -= self.CLOSURE_SIGNS[i] * self.inertia_form(
                ctx, i, test=nu_i, scale=self.row_scale(ctx, i),
                parameter=self.Omega_sq)
        return F

    def values(self) -> dict:
        """The rotation scalars, `{name: Real Function}` (views of the solution)."""
        return {name: self._solver.real_field(name)
                for name in self.real_field_names(self.context.mesh.geometric_dimension)}

    def inertia_perturbation(self) -> dict:
        """`{"dI_13": form, ...}`: the 0-forms of `dI_i3` at the current solution."""
        ctx = self.context
        out = {}
        for i, name in enumerate(("dI_13", "dI_23", "dI_33")):
            if i < 2 and ctx.mesh.geometric_dimension == 2:
                continue
            out[name] = self.inertia_form(ctx, i)
        return out


class CentreOfMass(Coupling):
    r"""Lagrange multipliers that hold the centre of mass at the origin.

    The first mass moment of the perturbation is

        D_i = int_mantle rho_0 u_i dx + sum_sheets int x_i sigma dS

    and the multipliers `lambda_i` enforce `D = 0`. The rows are the
    variation of `c beta sum_i lambda_i D_i` (`beta = B_mu`, floored at zero):
    the constraint `beta D_i = 0` and the column
    `c beta lambda . (int rho_0 w dx + sum_sheets int x (d sigma)[test] dS)`,
    a uniform acceleration of all moving mass. The column carries the
    variation of every sheet that depends on the solution, so with sea level
    it reaches the potential, `Shift` and rotation rows.

    A rigid translation is a near-kernel of the coupled operator only with a
    fluid core, so this coupling requires a `FluidCore`.
    """

    order = 30

    def real_field_names(self, dim):
        return tuple(f"centre_of_mass_{i}" for i in range(dim))

    def check(self, couplings) -> None:
        if not _find(couplings, FluidCore):
            raise ValueError(
                "CentreOfMass requires a FluidCore. A rigid core (un = 0 at the "
                "CMB) already fixes the translation of the mantle, so the "
                "constraint would over-determine the system.")

    def _beta(self, ctx):
        return ctx.scaling_factor * ctx.row_scale_B_mu

    def dipole_form(self, ctx, i, weight=None, scale=None):
        """`D_i` at the current solution, as a form in `weight` (1 by default)."""
        weight = Constant(1.0) if weight is None else weight
        scale = Constant(1.0) if scale is None else scale
        facet_weight = ctx.on_parent_facet(scale * weight)
        X = SpatialCoordinate(ctx.mesh)
        u = ctx.fields["displacement"]
        form = scale * weight * ctx.approximation.density * u[i] * ctx.dx_m
        for sheet in ctx.sheets:
            form += facet_weight * X[i] * sheet.density * sheet.measure
        return form

    def residual(self, ctx) -> Form:
        beta = self._beta(ctx)
        X = SpatialCoordinate(ctx.mesh)
        w = ctx.tests["displacement"]
        F = Form([])
        for i, name in enumerate(self.real_field_names(ctx.mesh.geometric_dimension)):
            lam_i, nu_i = ctx.fields[name], ctx.tests[name]
            # The constraint row: the whole moment, every sheet included.
            F += self.dipole_form(ctx, i, weight=nu_i, scale=beta)
            # The column: the variation of D_i in the direction of the test
            # functions. The mantle term is linear in u.
            F += beta * lam_i * ctx.approximation.density * w[i] * ctx.dx_m
            for sheet in ctx.sheets:
                if sheet.test_variation is None:
                    continue
                F += (ctx.on_parent_facet(beta * lam_i) * X[i]
                      * sheet.test_variation * sheet.measure)
        return F

    def multipliers(self) -> list:
        """The multipliers `lambda_i`, `Real` Functions (views of the solution)."""
        names = self.real_field_names(self.context.mesh.geometric_dimension)
        return [self._solver.real_field(name) for name in names]

    def dipole(self) -> list:
        """`D_i` at the current solution, one 0-form per direction."""
        ctx = self.context
        return [self.dipole_form(ctx, i) for i in range(ctx.mesh.geometric_dimension)]


class SeaLevel(Coupling):
    r"""The sea-level equation on the outer surface, with a gravitationally consistent ocean.

    The surface load is the sheet of the ocean and the grounded ice, and the
    ocean follows the solution through the sea level

        SL = SL_init + (N - N_init) - (u_r - ur_init) + Shift

    with `N = (psi + psi_rot) / g_surface`, `u_r = u . n` and `Shift` a
    `Real` unknown, the uniform change that conserves the mass of water and
    ice. The live sheet is `S(I, SL) - S(I_init, SL_init)` with
    `S = rho_w SL + rho_i eps softplus((I_+ - r SL) / eps)`
    (`gadopt.sea_level_sheet`); with `fixed_ocean=True` it is
    `rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)`,
    `C0 = H(r SL_init / eps)`, which is affine in the unknowns.

    The rows are the `u`, `psi` and `Shift` variations of the energy
    `-c B_mu g_s int G(Delta) dS` with `dG/dDelta = sigma`, written term by
    term: the weight of the load on the surface, the mass sheet in the
    potential row, and the net mass of water and ice. The sheet enters the
    rotation row through `Rotation` and the frame rows through
    `CentreOfMass`; it is left out of the 2-D monopole datum, because the
    `Shift` row sets its net mass to zero and a datum evaluated one solve
    late would hold the mass of an ice change the ocean has not balanced.

    Args:
      boundary: the outer surface tag on the mechanics mesh.
      rho_w, rho_i: water and ice densities.
      g_surface: the surface gravity in the geoid and the load weight.
      SL_init: the reference sea level, on the mechanics mesh.
      I, I_init: the current and the reference ice thickness.
      N_init, ur_init: geoid and uplift of the reference state.
      softplus_width: `eps`, the rounding width, positive.
      fixed_ocean: hold the ocean function at the reference state.
    """

    order = 40

    def __init__(self, *, boundary, rho_w, rho_i, g_surface, SL_init, I,
                 I_init, N_init, ur_init, softplus_width,
                 fixed_ocean: bool = False) -> None:
        super().__init__()
        if scalar_value(softplus_width) <= 0.0:
            raise ValueError(
                f"SeaLevel.softplus_width must be positive, got "
                f"{scalar_value(softplus_width)}.")
        self.boundary = boundary
        self.rho_w = rho_w
        self.rho_i = rho_i
        self.g_surface = g_surface
        self.SL_init = SL_init
        self.I = I
        self.I_init = I_init
        self.N_init = N_init
        self.ur_init = ur_init
        self.softplus_width = softplus_width
        self.fixed_ocean = fixed_ocean

    def real_field_names(self, dim):
        return ("sea_level_shift",)

    def check(self, couplings) -> None:
        ctx = self.context
        if not _find(couplings, CentreOfMass):
            raise ValueError(
                "SeaLevel requires CentreOfMass. The ocean load has degree-1 "
                "content, and only the centre-of-mass multipliers fix the "
                "translation it drives.")
        if "normal_stress" in ctx.bcs.get(self.boundary, {}):
            raise ValueError(
                f"Boundary {self.boundary} carries both the sea-level load and a "
                "normal_stress. The sea-level sheet already enters the momentum "
                "row; give the ice thickness to SeaLevel.I instead.")
        if any(load.boundary == self.boundary for load in _find(couplings, SurfaceLoad)):
            raise ValueError(
                f"Boundary {self.boundary} carries both SeaLevel and a "
                "SurfaceLoad. With sea level on, the ice and ocean load is the "
                "sea-level sheet.")
        if ctx.mesh.extruded:
            raise NotImplementedError(
                "The sea-level equation is implemented on a non-extruded "
                "mechanics mesh only.")
        p = max(ctx.field_degree("displacement"), ctx.field_degree("potential"))
        q = ctx.boundary_quad_degree
        if q < 2 * p:
            raise ValueError(
                f"The sea-level measure uses the boundary quadrature degree {q}, "
                f"below 2 p = {2 * p} for polynomial degree p = {p} of the "
                "displacement and the potential. Pass a larger dtn_quad_degree, "
                "or None for the calibrated default.")
        ctx.check_closed_surface(self.boundary, "SeaLevel")

    def measure(self) -> Measure:
        """The sea-level surface measure, restricted to the boundary."""
        return self.context.surface_measure(self.boundary)

    def sea_level(self, split=None):
        """`SL` as UFL on `measure()`, at the solution or at the fields `split`."""
        ctx = self.context
        fields = ctx.fields if split is None else split
        u = fields["displacement"]
        psi = fields["potential"]
        shift = ctx.on_parent_facet(fields["sea_level_shift"])
        g_surface = ctx.on_coupling_facet(self.g_surface)
        # Bruns: N = +(psi + psi_rot) / g.
        N = ctx.parent_trace(psi) / g_surface
        if ctx.rotational_potential is not None:
            N = N + ctx.rotational_potential(ctx.mesh, facet=True,
                                             split=split) / g_surface
        u_r = dot(u, FacetNormal(ctx.mesh))
        return self.SL_init + (N - self.N_init) - (u_r - self.ur_init) + shift

    def ocean_mask(self):
        """`C0 = H(r SL_init / eps)`, the ocean function of the reference state."""
        return ocean_fraction(self.SL_init, ensure_constant(self.softplus_width),
                              self.rho_w, self.rho_i)

    def sheet_of(self, SL):
        """The load sheet `sigma` for the sea level `SL`."""
        if self.fixed_ocean:
            C0 = self.ocean_mask()
            water = self.rho_w * C0 * (SL - self.SL_init)
            ice = self.rho_i * ((1 - C0) * self.I - (1 - C0) * self.I_init)
            return water + ice
        eps = ensure_constant(self.softplus_width)
        now = softplus_column(self.I, SL, eps, self.rho_w, self.rho_i)
        reference = softplus_column(self.I_init, self.SL_init, eps, self.rho_w,
                                    self.rho_i)
        return now - reference

    def sheet(self):
        """`sigma`, the surface mass density of ocean and ice, UFL on `measure()`."""
        return self.sheet_of(self.sea_level())

    def water_fraction(self):
        """`d sigma / d SL / rho_w`: 1 over open water, 0 on land and grounded ice."""
        if self.fixed_ocean:
            return self.ocean_mask()
        return water_fraction(self.I, self.sea_level(),
                              ensure_constant(self.softplus_width),
                              self.rho_w, self.rho_i)

    def sheets(self, ctx):
        sigma = self.sheet()
        # Expanded at once; see the module docstring.
        variation = expand_derivatives(ufl_derivative(
            sigma, ctx.solution, TestFunction(ctx.space)))
        return [MassSheet(name="sea level", density=sigma,
                          test_variation=variation, measure=self.measure(),
                          in_monopole_datum=False)]

    def _scale(self, ctx):
        """`c B_mu`, the part of the factor `c B_mu g_s` of the sea-level energy
        that is made of parent quantities."""
        return ctx.scaling_factor * ctx.approximation.B_mu

    def residual(self, ctx) -> Form:
        # `g_s` can be a `Real` Function of either mesh, so it is restricted
        # as its own mesh requires and kept apart from the parent factors.
        g_surface = ctx.on_coupling_facet(self.g_surface)
        sigma = self.sheet()
        dss = self.measure()
        scale = self._scale(ctx)
        w = ctx.tests["displacement"]
        v = ctx.tests["potential"]
        s = ctx.tests["sea_level_shift"]
        # The parent scale and every parent test function under one
        # restriction. The displacement row: the weight of the load,
        # `c B_mu g_s sigma (w . n)`.
        F = (ctx.on_parent_facet(scale) * g_surface * sigma
             * dot(w, FacetNormal(ctx.mesh)) * dss)
        # The potential row: the mass sheet, with the sign of every sheet,
        # `-c B_mu sigma v`. The `g_s` of the energy cancels against the
        # `1 / g_s` of the geoid `N = psi / g_s`.
        F -= ctx.on_parent_facet(scale * v) * sigma * dss
        # The Shift row: the net mass of water and ice is zero, up to the
        # centre-of-mass column, `-c B_mu g_s sigma s`.
        F -= ctx.on_parent_facet(scale * s) * g_surface * sigma * dss
        return F

    def energy(self) -> Form:
        r"""The sea-level energy, with the live sheet linearised at the current state.

            E_sl = -c B_mu g_s int_Re G(Delta) dS,   dG/dDelta = sigma

        with `Delta = SL - SL_init = (N - N_init) - (u_r - ur_init) + Shift`.
        A diagnostic: its variations are the three sea-level rows of
        `residual` at the state where it is built.

        - `u`, direction `w`: `+c B_mu g_s int sigma (w . n) dS`, the weight of
          the load, the same as a `normal_stress` of `B_mu g_s sigma`;
        - `psi`, direction `v`: `-c B_mu int sigma v dS`, the same as a sheet
          of density `sigma` in the potential row, which carries
          `-theta_psi 4 pi G sigma v = -c B_mu sigma v`;
        - `Shift`, direction `s`: `-c B_mu g_s int sigma dS`, the conservation
          of the mass of water and ice. With `CentreOfMass`, its energy adds
          `c B_mu sum_i lambda_i int x_i (dsigma/dShift) s dS` to the same
          row, because the sheet moves the centre of mass; that column keeps
          the residual the gradient of one Lagrangian and the Jacobian
          symmetric.

        The sign of `E_sl` is fixed by the first two. `g_s` equals the
        reference gravity at the surface, so the potential variation cancels
        it and the displacement variation keeps it.

        The exact `G` of the softplus sheet needs the dilogarithm, which UFL
        does not have, so the live sheet is linearised at a copy `SL_f` of
        the solution made at the call:

            sigma_lin(SL) = sigma(SL_f) + rho_w W_f (SL - SL_f),
            G(Delta)      = sigma_lin(SL_init) Delta + 0.5 rho_w W_f Delta^2

        with `W_f` the water fraction at the frozen state. Then
        `dG/dDelta = sigma_lin(SL)`, which equals `sigma(SL)` at the frozen
        state, so the derivative of the energy equals the written residual
        there. The fixed coastline needs no linearisation.
        """
        ctx = self.context
        delta = self.sea_level() - self.SL_init
        if self.fixed_ocean:
            C0 = self.ocean_mask()
            ice = self.rho_i * ((1 - C0) * self.I - (1 - C0) * self.I_init)
            G = ice * delta + 0.5 * self.rho_w * C0 * delta * delta
            return (-ctx.on_parent_facet(self._scale(ctx))
                    * ctx.on_coupling_facet(self.g_surface) * G * self.measure())
        frozen = ctx.solution.copy(deepcopy=True)
        SL_f = self.sea_level(ctx.split_of(frozen))
        eps = ensure_constant(self.softplus_width)
        W_f = water_fraction(self.I, SL_f, eps, self.rho_w, self.rho_i)
        sigma_0 = self.sheet_of(SL_f) + self.rho_w * W_f * (self.SL_init - SL_f)
        G = sigma_0 * delta + 0.5 * self.rho_w * W_f * delta * delta
        return (-ctx.on_parent_facet(self._scale(ctx))
                * ctx.on_coupling_facet(self.g_surface) * G * self.measure())

    def shift(self):
        """`Shift`, a `Real` Function (a view of the solution)."""
        return self._solver.real_field("sea_level_shift")


class SurfaceLoad(Coupling):
    r"""A prescribed surface mass density, given once for every use of it.

    The load `sigma` on one boundary of the mechanics mesh enters

    - the momentum equation as the normal traction `B_mu g_surface sigma`,
      the term `c B_mu g_surface sigma (w . n)` that a weak `normal_stress`
      condition would add, written in this coupling's residual;
    - the potential row as a mass sheet, `-theta_psi 4 pi G int sigma v dS`;
    - the rotation rows and the centre-of-mass rows through its `MassSheet`;
    - the 2-D monopole datum, with no lag, because it does not depend on the
      solution.

    Giving the load once leaves no second copy (a traction on the mechanics
    mesh and an `interior_sigma` on the parent) that could disagree. Make
    `sigma` a `Function` updated with `assign` for a time-dependent load, so
    every use follows it. It has no `Real` field.

    Args:
      boundary: the loaded surface tag on the mechanics mesh.
      sigma: the surface density, a UFL expression or `Function` on the
        mechanics mesh.
      g_surface: the gravity that weighs the load in the traction.
    """

    order = 50
    repeatable = True

    def __init__(self, *, boundary, sigma, g_surface) -> None:
        super().__init__()
        self.boundary = boundary
        self.sigma = sigma
        self.g_surface = g_surface

    def check(self, couplings) -> None:
        ctx = self.context
        if "normal_stress" in ctx.bcs.get(self.boundary, {}):
            raise ValueError(
                f"Boundary {self.boundary} carries both a SurfaceLoad and a "
                "normal_stress in bcs. The SurfaceLoad already writes the "
                "traction; remove the normal_stress.")
        others = [c for c in _find(couplings, SurfaceLoad)
                  if c is not self and c.boundary == self.boundary]
        if others:
            raise ValueError(
                f"Two SurfaceLoads on boundary {self.boundary}; add the loads "
                "into one sigma.")
        ctx.check_closed_surface(self.boundary, "SurfaceLoad")

    def measure(self) -> Measure:
        """The load measure at the boundary degree, restricted to the boundary."""
        return self.context.surface_measure(self.boundary)

    def sheets(self, ctx):
        return [MassSheet(name="surface load", density=self.sigma,
                          test_variation=None, measure=self.measure(),
                          in_monopole_datum=True)]

    def residual(self, ctx) -> Form:
        # The momentum row: the weight of the load, the term a weak
        # `normal_stress` of `B_mu g_s sigma` adds to the momentum equation,
        # at the momentum equation's quadrature degree. It is written here and
        # not as a condition of the momentum equation because that equation
        # pairs its facets with the parent's cells, where the derivative with
        # respect to a `Real` Function of the parent (`B_mu`, `g_s`, `c`) is
        # assembled twice. On the coupling facet measure the row scaling
        # `c B_mu`, which lives on the parent or has no mesh, takes the
        # restriction of every parent quantity, and the coupling parameter
        # `g_s` the restriction of its own mesh.
        w = ctx.tests["displacement"]
        traction = (ctx.on_parent_facet(ctx.scaling_factor * ctx.approximation.B_mu)
                    * ctx.on_coupling_facet(self.g_surface) * self.sigma)
        F = (traction * dot(w, FacetNormal(ctx.mesh))
             * ctx.surface_measure(self.boundary, degree=ctx.mechanics_quad_degree))
        # The potential row of a prescribed sheet, `-theta_psi 4 pi G sigma v`.
        # `theta_psi 4 pi G` is `c B_mu` for every nonzero `B_mu`; at
        # `B_mu = 0` the potential still sees the load while the mechanics
        # does not, which is the null-coupling configuration.
        v = ctx.tests["potential"]
        F -= (ctx.on_parent_facet(ctx.theta_psi * ctx.four_pi_G * v)
              * self.sigma * self.measure())
        return F


def ordered(couplings) -> list:
    """`couplings` sorted by `order`, refusing a duplicate of a single coupling.

    The order fixes the position of the `Real` fields, so it must not depend
    on the order of the list the caller wrote.
    """
    couplings = list(couplings)
    for c in couplings:
        if not isinstance(c, Coupling):
            raise TypeError(f"{c!r} is not a Coupling.")
    for cls in {type(c) for c in couplings}:
        same = [c for c in couplings if type(c) is cls]
        if len(same) > 1 and not cls.repeatable:
            raise ValueError(f"More than one {cls.__name__} in couplings.")
    if len({id(c) for c in couplings}) != len(couplings):
        raise ValueError("The same coupling object appears twice in couplings.")
    return sorted(couplings, key=lambda c: c.order)
