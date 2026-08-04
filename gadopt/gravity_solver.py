r"""Gravitational Poisson solver with Dirichlet-to-Neumann boundary treatment.

This module solves the gravitational Poisson equation

$$
  nabla^2 psi = -4 pi G rho,
$$

whose correct boundary condition lives at infinity: psi must match the decaying
free-space solution. On the finite domains G-ADOPT works with (annuli, spherical
shells and their extensions), that condition is imposed exactly (up to mode
truncation) through Dirichlet-to-Neumann (DtN) maps on circular/spherical
boundaries: each angular eigenmode of the boundary trace extends uniquely into
the source-free exterior (or interior core) as the decaying (regular) harmonic
solution, whose normal derivative is proportional to its own trace. The map
needs the trace of psi only - never the density - so the solver discovers the
mode content by itself. The trace coefficients are global scalar unknowns in
R spaces, defined by scalar constraint rows and fed back as boundary flux;
everything is a plain UFL form, so the system remains differentiable, adjoint-
capable, and embeddable in larger coupled residuals.

The DtN terms use the Robin-shifted formulation: the flux is split as

$$
  (d psi)/(d n) = -(alpha/R) psi + sum_k (alpha - lambda_k R)/R \, c_k e_k,
$$

which is algebraically identical for the treated modes but places a pointwise
Robin term in the psi-psi block, keeping it symmetric positive definite for the
Schur fieldsplit (the naive form leaves that block a singular pure-Neumann
Laplacian). Untreated modes above the truncation see Robin(alpha/R) - a
decaying far field - instead of a reflecting homogeneous Neumann condition.

Mass sheets sitting exactly on a DtN boundary (e.g. dynamic topography) enter
through the flux jump $[(d psi)/(d r)] = -4 pi G sigma$, which reduces to the
standard surface-source term $-4 pi G int sigma v \, ds$ with the DtN machinery
unchanged.

A sheet does not have to sit on a boundary of the mesh. The coupled
self-gravitating problem stands its DtN boundaries off from the sources, which
puts the Earth's surface *inside* the domain, so the surface load is a mass
sheet on a tagged **interior** facet and the same term is written against `dS`
rather than `ds`. `sigma` and `interior_sigma` are the two spellings; they
differ only in the measure and in the restriction the interior one needs. The
distinction cannot be inferred from the tag, because Firedrake reports every
physical-group label in both `exterior_facets.unique_markers` and
`interior_facets.unique_markers` regardless of which set actually holds those
facets, and asking a measure for the wrong kind of tag returns zero with a
warning rather than raising - a sheet written the wrong way is silently absent
and the solve converges without it. `DtNGravityForm.check_sheet_measures`
therefore refuses an empty sheet measure at construction.

In 2-D the monopole needs one extra ingredient, because the exterior expansion
carries a logarithmic branch $C_0 ln r$ whose coefficient the boundary trace
does not determine. The DtN map is complete for every $m >= 1$ and blind at
$m = 0$, so a 2-D exterior boundary additionally receives the flux datum

$$
  (d psi)/(d n)|_{m = 0} = -2 G M / R,
$$

with $M$ the total enclosed mass. That leaves the additive constant free -
there is no "psi -> 0 at infinity" to fix it - and the Robin shift above fixes
it for us: testing the residual with $v = 1$, the volume source and the datum
cancel exactly and leave $oint psi \, ds = 0$ on the boundary. So the system
stays nonsingular with no nullspace handling, and the potential is returned in
the gauge $oint_{dB} psi \, ds = 0$, i.e. $psi = -2 G M ln(r/R)$ outside the
sources. Only `grad psi` is gauge-independent; anyone comparing against a
closed form that fixes the constant differently will see a constant offset.
3-D has no analogue - the $l = 0$ exterior decays as $1/r$ and the trace map
reproduces it exactly - so nonzero net mass has always been legal there.

Users instantiate the `GravitySolver` class by providing a solution function,
the density, and boundary conditions keyed by boundary marker, then call the
`solve` method to update the solution.
"""

from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np
from firedrake import *
from firedrake.ufl_expr import extract_unique_domain

from firedrake.petsc import PETSc

from .dtn_adjoint import annotate_lowrank_solve, taped_trace_coefficients
from .dtn_form import (  # noqa: F401 - re-exported; see the note below
    QUADRATURE_ELEMENT_DEGREE_COST, QUADRATURE_RULE_COEFFICIENTS, BaseDtN,
    CylindricalDtN, DtNGravityForm, DtNMode, QuadratureCalibration,
    QuadratureRuleReport, SphericalDtN)
from .dtn_lowrank import LowRankDtNOperator, apply_dirichlet_to_rows
from .solver_options_manager import (ConfigType, SolverConfigurationMixin,
                                     gamg_parameters)
from .utility import DEBUG, INFO, ensure_constant, log_level

# The boundary treatment, the DtN descriptors and the calibrated quadrature
# apparatus live in `gadopt.dtn_form`, so that a coupled residual can use them
# without instantiating a solver. They are re-exported here because that is
# where they were, and `from gadopt.gravity_solver import ...` is written in
# the tests and in the scaling scripts.

__all__ = ["CylindricalDtN", "DtNGravityForm", "SphericalDtN", "GravitySolver"]


#: For the one-by-one systems on a `Real` space that define the enclosed-mass
#: scalars. The operator is a nonzero number, so `preonly` + Jacobi is an exact
#: solve rather than a preconditioning choice, and it must not inherit the
#: gravity solver's own fieldsplit options.
real_scalar_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

direct_gravity_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Direct solve for the psi (potential) block.

MUMPS LU. Exact and robust, but the sparse factors of a 3-D volume operator
suffer fill-in that outgrows the matrix, so this is the default only in 2-D
(and for small 3-D problems selected explicitly). See `iterative_gravity_solver_parameters`
for the scalable alternative.
"""

iterative_gravity_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "pc_type": "python",
    "pc_python_type": "gadopt.SPDAssembledPC",
    **gamg_parameters("assembled_"),
}
"""Iterative solve for the psi (potential) block.

Conjugate gradients preconditioned by algebraic multigrid (GAMG). The Robin
shift makes the psi block strictly SPD, so `gadopt.SPDAssembledPC` sets the
PETSc MAT_SPD flag (CG eigen-estimates in the Chebyshev and GAMG setup) and no
near-nullspace is needed. These are the same parameters G-ADOPT's 3-D spherical
Stokes solver uses for its velocity block, which inverts a strictly harder SPD
operator on the same extruded cubed-sphere meshes at production scale. AMG has
O(N) memory with no fill-in, so it replaces MUMPS as the default psi solver in
3-D.
"""


def _from_form(name: str) -> property:
    """A read-only forward from `GravitySolver` to its `DtNGravityForm`.

    The boundary treatment moved to `gadopt.dtn_form` so that a coupled
    residual could use it without a solver; every name it took with it is
    still reachable here, because the tests, the demos and the scaling scripts
    read seventeen of them off the solver object and none of that is worth
    churning. The forward returns the form's *own* object rather than a copy,
    so `solver.ds is solver.form.ds` and the two can never disagree.
    """
    def get(self):
        return getattr(self.form, name)
    get.__name__ = name
    return property(
        get, doc=f"Forwarded to `DtNGravityForm.{name}` on `self.form`.")


def _flatten_options(options, prefix=""):
    """PETSc-style flat options from the nested G-ADOPT dictionary."""
    flat = {}
    for key, value in options.items():
        if isinstance(value, Mapping):
            flat.update(_flatten_options(value, prefix + key + "_"))
        else:
            flat[prefix + key] = value
    return flat


lowrank_gravity_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-11,
    "ksp_max_it": 1000,
    **gamg_parameters(),
}
"""CG on `A + B`, algebraic multigrid built from the Robin-shifted `A` alone.

The same GAMG settings the multiplier path gives its `fieldsplit_0`, applied
without the `AssembledPC` indirection because here `A` is already an assembled
`aij` matrix and PETSc can be handed it directly as `Pmat`. `ksp_rtol` matches
the multiplier path's OUTER tolerance, not its inner one: there is no outer
Krylov cleaning up after this solve, so this rtol is the solution accuracy.

Note when comparing iteration counts against the multiplier path: its
`fieldsplit_0` runs at `ksp_rtol` 1e-8, so a comparison against this preset's
1e-11 is not like-for-like. Pinned equal, both paths take 8 CG iterations.

PETSc will report

    Option left: name:-Gravity_lowrank_mg_levels_pc_type value: sor

and it is a **false alarm**: the option is in effect. Interrogating the
preconditioner directly shows the level smoother is `chebyshev/sor` as
requested, and forcing anything else makes it worse (11 CG iterations as
shipped, 15 with `chebyshev/jacobi`, 15 with `richardson/sor`). A genuinely
unused option looks identical to this one, and both natural responses to the
warning - "fixing" it, or discounting a comparison because of it - are wrong.
"""


class GravitySolver(SolverConfigurationMixin):
    """Solver for the gravitational Poisson equation with DtN boundaries.

    Solves $nabla^2 psi = -4 pi G rho$ for the perturbation potential psi with
    the infinite-domain behaviour imposed through Dirichlet-to-Neumann maps on
    the boundaries specified in `bcs`. The solution function is updated in
    place by `solve`.

    The density may live on the solver mesh or on a `Submesh` of it (the
    extended-domain configuration, where psi extends beyond the region
    carrying the density); the cross-mesh coupling is set up automatically.

    **The 2-D gauge.** A 2-D problem with an exterior DtN boundary and nonzero
    net mass has a logarithmic far field and therefore no natural zero. The
    solver returns the potential normalised so that its mean over that boundary
    vanishes, i.e. `psi = -2 G M ln(r/R)` outside the sources with `M` the
    total enclosed mass. Only `grad psi` is gauge-independent, so a closed-form
    reference that fixes the constant some other way will disagree by an
    offset that is not an error. Two configurations are refused rather than
    solved in a gauge that cannot exist: a strong `psi` condition alongside a
    nonzero enclosed mass, and more than one exterior DtN boundary on the same
    2-D mesh. 3-D is unaffected throughout.

    Args:
      solution:
        Firedrake function for the potential, on a scalar function space
      rho:
        Density: a Firedrake function (on the solution mesh or a Submesh of
        it), a UFL expression, or a number
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      gravitational_constant:
        Value of G in the chosen non-dimensionalisation
      quad_degree:
        Integer denoting the quadrature degree applied on boundary integrals
        involving the DtN eigenfunctions. UFL's automatic estimate is useless
        for these - it adds a flat two degrees per elementary function, so
        `cos(m phi)` estimates the same at m = 1 and m = 40 - and the degree is
        therefore explicit. The default is mesh-aware: what decides whether a
        rule resolves a mode is how much of one oscillation falls inside one
        facet, `L h / R`, so the default follows that rather than the
        truncation alone, and stops growing once the mesh resolves the mode.
        See `default_quad_degree`. `__init__` warns for free when that rule is
        capped or is extrapolating beyond the meshes it was calibrated on, but
        does **not** measure whether the degree works - call
        `check_boundary_quadrature` for that, and see
        `warn_on_quadrature_rule_limits` for why it is not automatic
      source_quad_degree:
        Optional quadrature degree for the volume source term; useful when
        rho is an analytic UFL expression (e.g. an angular mode restricted to
        a mesh-conforming shell by a conditional) whose exact integration
        removes the source-representation error from convergence studies
      solver_parameters:
        Either a complete dictionary of PETSc solver options, or the string
        "direct"/"iterative" to select a G-ADOPT default. Omitting it (None)
        picks the default by dimension: "direct" in 2-D, "iterative" in 3-D,
        mirroring the Stokes solver. The two presets differ only in how the psi
        (potential) block is inverted - MUMPS versus CG + algebraic multigrid;
        direct does not scale to large 3-D meshes. Passing a full dictionary
        means you own the description of the two Schur blocks that the class
        otherwise sets up.
      solver_parameters_extra:
        Dictionary of PETSc solver options used to update the default G-ADOPT
        options. The safe way to tweak the default (e.g. override only the psi
        block, via a `{"dtn": {"fieldsplit_0": ...}}` entry) while keeping the
        Schur structure.
      dtn_representation:
        How the DtN terms are represented. "multiplier" (default) gives every
        treated mode a scalar Real unknown constrained by a row of a mixed
        system, which keeps every term in plain UFL. "lowrank" eliminates those
        multipliers by hand, leaving one scalar space and a rank-n update to
        the Robin-shifted stiffness applied in factored form; it solves the
        same discrete system - measured agreement 5e-15 on potentials and
        3e-15 on trace coefficients - and is dramatically faster at any
        appreciable truncation, 480x on the first solve and 122x on a repeat
        solve at 162 modes. Its derivative is hand-written in
        `gadopt.dtn_adjoint` rather than taped; the portable adjoint tests pass
        on it at unweakened tolerances. See `NOTES/bench/STEP7-VERDICT.md`.

    ### Valid keys for boundary conditions
    | Condition      |  Type  |                 Description                        |
    | :------------- | :----- | :------------------------------------------------: |
    | psi            | Strong | Prescribed potential                               |
    | flux           | Weak   | Prescribed outward normal derivative (d psi)/(d n) |
    | dtn            | Weak   | DtN map (`CylindricalDtN` or `SphericalDtN`)       |
    | sigma          | Weak   | Surface mass density on an exterior facet (`ds`)   |
    | interior_sigma | Weak   | Surface mass density on a tagged interior facet    |

    A boundary with no entry keeps the natural (homogeneous Neumann)
    condition. The orientation of a DtN boundary (exterior map enclosing the
    sources vs interior map enclosing a source-free core) and its radius are
    measured from the marked boundary, not specified by the user. DtN maps
    are only valid on origin-centred coordinate circles/spheres; a boundary
    deviating from constant radius triggers a warning, but the solve
    proceeds with the mean radius and correspondingly degraded accuracy.

    `sigma` and `interior_sigma` are the same physics on the two kinds of
    facet, and which one a tag needs is a property of the mesh rather than of
    the condition - so it is the caller's to state, because Firedrake will not
    tell them (see the module docstring). Every other condition applies to a
    boundary of the mesh and takes no such choice: a DtN map is a statement
    about the exterior of the domain, and a strong condition and a prescribed
    flux both need a boundary to be applied on.
    """

    name = "Gravity"
    #: Robin shift; see the module docstring. Read here and handed to the
    #: boundary form, so that a subclass can vary it - which is how the gauge
    #: test varies it - and the form sees the value the subclass chose.
    alpha = 1.0

    def __init__(
        self,
        solution: Function,
        rho,
        /,
        *,
        bcs: dict[int | str, dict[str, Any]],
        gravitational_constant: float | Constant = 1.0,
        quad_degree: int | None = None,
        source_quad_degree: int | None = None,
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
        dtn_representation: str = "multiplier",
    ) -> None:
        if dtn_representation not in ("multiplier", "lowrank"):
            raise ValueError(
                f"dtn_representation must be 'multiplier' or 'lowrank', got "
                f"{dtn_representation!r}.")
        self.dtn_representation = dtn_representation
        self.solution = solution
        self.rho = ensure_constant(rho)
        self.bcs = bcs
        self.G = ensure_constant(gravitational_constant)

        self.solution_space = solution.function_space()
        if self.solution_space.value_size != 1:
            raise ValueError("solution must live on a scalar function space")
        self.mesh = self.solution_space.mesh()
        self.X = SpatialCoordinate(self.mesh)

        rho_mesh = extract_unique_domain(self.rho)
        self.cross_mesh = rho_mesh is not None and rho_mesh is not self.mesh
        self.rho_mesh = rho_mesh if self.cross_mesh else self.mesh

        #: The boundary treatment. Sorts `bcs`, chooses the boundary
        #: quadrature rule, measures every DtN boundary, and writes the
        #: boundary bilinear form and the boundary source. `alpha` is handed
        #: over rather than read off the form's own class attribute, because
        #: the Robin shift is a knob of *this* class - a test subclasses
        #: `GravitySolver` and overrides it - and a form that read its own
        #: default would ignore the override without failing.
        self.form = DtNGravityForm(
            self.solution_space, bcs,
            gravitational_constant=self.G, quad_degree=quad_degree,
            alpha=self.alpha)
        self.set_volume_measures(source_quad_degree)
        self.set_monopole_datum()
        if self.dtn_representation == "lowrank":
            self.set_lowrank_operator()
        else:
            self.set_function_spaces()
            self.set_form()
        self.warn_on_quadrature_rule_limits()
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.set_solver()

    # -- Forwarded to the boundary form -----------------------------------
    #
    # Everything the DtN treatment owns is read off `self.form` rather than
    # copied, so the two cannot drift apart: the day someone rebuilds a measure
    # on one side, every reader on the other side sees the new one. The
    # accompanying test asserts identity - `solver.ds is solver.form.ds` - and
    # not merely that the attribute exists, because a shim that only forwards
    # turns "every existing test passes" into a statement about `getattr`.
    #
    # Three names deliberately do *not* forward. `alpha` is a class attribute
    # of this class, passed into the form by `__init__`, because a test
    # subclasses `GravitySolver` to vary the Robin shift and a form reading its
    # own default would ignore the override silently. `n_multipliers` is zero
    # on the low-rank path, where the form still knows the true count.
    # `_multiplier_offset` indexes *this* class's mixed space and means nothing
    # to a boundary form that has no space of its own.
    #
    # What the shim forwards is what is *read*, not what constructs: the form's
    # own build steps (`set_boundary_conditions`, `set_measures`,
    # `check_sheet_measures`, `set_boundary_geometry`) are reachable on
    # `self.form` and are not given a second name here, since re-running one of
    # them after the residual is built would rebuild half the state the
    # residual already refers to.
    ds = _from_form("ds")
    dS = _from_form("dS")
    quad_degree = _from_form("quad_degree")
    quad_rule_report = _from_form("quad_rule_report")
    dtn_boundaries = _from_form("dtn_boundaries")
    sigma_bcs = _from_form("sigma_bcs")
    flux_bcs = _from_form("flux_bcs")
    dirichlet_bcs = _from_form("dirichlet_bcs")
    boundary_geometry = _from_form("boundary_geometry")
    boundary_area = _from_form("boundary_area")
    mode_rows = _from_form("mode_rows")
    _multiplier_keys = _from_form("multiplier_keys")
    element_degree = _from_form("element_degree")
    surface_measure = _from_form("surface_measure")
    geometry_probe_measure = _from_form("geometry_probe_measure")
    boundary_facet_scale = _from_form("boundary_facet_scale")
    default_quad_degree = _from_form("default_quad_degree")
    quadrature_rule_report = _from_form("quadrature_rule_report")
    warn_on_quadrature_rule_limits = _from_form("warn_on_quadrature_rule_limits")
    sheet_integral = _from_form("sheet_integral")
    boundary_quadrature_rule = _from_form("boundary_quadrature_rule")
    check_boundary_quadrature = _from_form("check_boundary_quadrature")
    _mode_norms = _from_form("_mode_norms")
    _validate_recovered_rule = _from_form("_validate_recovered_rule")

    def set_volume_measures(self, source_quad_degree: int | None) -> None:
        """Sets the volume measures. The boundary ones belong to the form.

        `dx` integrates over the potential's mesh and `dx_rho` over the
        density's, which are the same object unless the density lives on a
        `Submesh`; the cross-mesh pair carries the intersection that makes
        Firedrake build the entity maps.
        """
        if self.cross_mesh:
            self.dx = Measure(
                "dx", domain=self.mesh,
                intersect_measures=(Measure("dx", domain=self.rho_mesh),))
            self.dx_rho = Measure(
                "dx", domain=self.rho_mesh,
                intersect_measures=(Measure("dx", domain=self.mesh),))
        else:
            self.dx = self.dx_rho = dx(domain=self.mesh)
        if source_quad_degree is not None:
            self.dx_rho = self.dx_rho(degree=source_quad_degree)

    def set_monopole_datum(self) -> None:
        """Prepares the 2-D exterior monopole flux datum; see the module docstring.

        The enclosed mass is held in `Real` `Function`s rather than baked into
        the form as numbers, and rather than promoted to unknowns with
        constraint rows. Both alternatives were measured and both are worse: a
        number severs the tape (§ `update_total_mass`), while an unknown costs
        a field, a constraint row and a Schur dimension, and does not transfer
        to the low-rank path at all, which has no multipliers. A `Real`
        `Function` assigned from an assembled `AdjFloat` is differentiable *and*
        is an ordinary UFL coefficient, so one expression serves both
        representations and `gadopt.dtn_adjoint` picks it up with no new code.

        Two scalars rather than one, because the flux conditions carry mass
        whose contribution to the datum has no `G` in it - see `monopole_flux`.
        """
        #: The 2-D exterior DtN boundaries, which are the only ones whose
        #: monopole the trace fails to determine. Empty in 3-D.
        self.monopole_boundaries = [
            bc_id for bc_id, _ in self.dtn_boundaries
            if self.mesh.geometric_dimension == 2
            and self.boundary_geometry[bc_id][0] == "exterior"]

        self._real_space = None
        self.mesh_volume = None
        self.source_mass = None
        self.cavity_flux = None
        if not self.monopole_boundaries:
            # Nothing is built in 3-D, not even the Real space and the volume
            # assemble: "the 3-D path pays nothing for this" should be true of
            # the code and not only of the per-solve cost.
            return

        if len(self.monopole_boundaries) > 1:
            raise ValueError(
                f"2-D mesh with {len(self.monopole_boundaries)} exterior DtN "
                f"boundaries {self.monopole_boundaries}: the monopole datum "
                "needs the mass enclosed by the exterior boundary, which is "
                "not defined when there is more than one of them.")
        clash = sorted(
            {bc_id for bc_id, _ in self.flux_bcs}.intersection(
                self.monopole_boundaries),
            key=str)
        if clash:
            raise ValueError(
                f"Boundary {clash[0]}: 'flux' and a 2-D exterior 'dtn' both "
                "prescribe the m = 0 normal derivative there - the DtN map "
                "through the enclosed mass, the flux condition directly - so "
                "the two are a double specification of one quantity. Move the "
                "flux condition to an interior boundary or drop it.")

        self._real_space = FunctionSpace(self.mesh, "R", 0)
        #: The measure the one-by-one system's operator assembles to; plain
        #: `dx`, never `self.dx`, which in the cross-mesh case is intersected
        #: with the density's mesh and is therefore a different number.
        self.mesh_volume = assemble(Constant(1.0) * dx(domain=self.mesh))
        #: `int rho dx + sum_sheets int sigma ds`, refreshed by `solve`.
        self.source_mass = Function(self._real_space, name="source_mass")
        #: `sum_flux int (d psi)/(d n) ds` over the prescribed-flux boundaries,
        #: which is `4 pi G` times the mass they imply inside their cavities.
        self.cavity_flux = Function(self._real_space, name="cavity_flux")

    def enclosed_mass_forms(self) -> tuple:
        """`(identity, mass, flux)`: the one-by-one systems on the Real space.

        `mu` and `nu` are globally constant, so dividing the left-hand side by
        `mesh_volume` - the value that same measure assembles to, from the same
        measure - makes the system the identity, and `source_mass` holds the
        mass its name claims rather than a mass per unit volume. See
        `update_total_mass` for why the mass is routed through a solve at all,
        and why the solve is rebuilt every time rather than cached.

        `ensure_constant` is applied to the boundary values because a Python
        `0.0` folds the whole integral away, and a right-hand side that folds
        to nothing is not a linear form at all - which for a lone
        `{"flux": 0.0}` condition meant a `ValueError` from deep inside
        Firedrake rather than the zero it obviously means.
        """
        mu = TestFunction(self._real_space)
        nu = TrialFunction(self._real_space)
        identity = (nu / self.mesh_volume) * mu * dx(domain=self.mesh)

        mass = self.rho * mu * self.dx_rho
        for bc_id, sigma, integral_type in self.sigma_bcs:
            mass += self.sheet_integral(
                ensure_constant(sigma) * mu, bc_id, integral_type)

        flux = None
        for bc_id, value in self.flux_bcs:
            term = ensure_constant(value) * mu * self.ds(bc_id)
            flux = term if flux is None else flux + term
        return identity, mass, flux

    def monopole_flux(self, bc_id):
        """`(d psi)/(d n)` of the m = 0 mode on a 2-D exterior DtN boundary.

        `-2 G M / R`, with `M` everything the boundary encloses: the volumetric
        density, the sheets, and the cavities implied by prescribed-flux
        boundaries. By Gauss the last of those is `int g ds / (4 pi G)` for a
        flux datum `g`, whose `G` cancels against the `2 G` in front - so that
        piece is written without `G` rather than as a `G/G`, which keeps it
        exactly differentiable when the gravitational constant is itself a
        control.

        Returned as a flux rather than as a form so that both representations
        can feed it through the same idiom they already use for the `flux`
        boundary conditions, where the two sign conventions are visible side by
        side (`set_form` builds a residual, `set_lowrank_operator` a
        right-hand side).
        """
        _, R = self.boundary_geometry[bc_id]
        flux = -2.0 * self.G * self.source_mass / R
        if self.flux_bcs:
            # Left out entirely rather than added as a term that is
            # structurally zero, so that a configuration with no flux
            # conditions does not carry a tape dependency that can never move.
            flux -= self.cavity_flux / (2.0 * np.pi * R)
        return flux

    def monopole_fluxes(self) -> dict:
        """`{bc_id: flux}` for `DtNGravityForm.boundary_source`'s `extra_flux`.

        The one thing the mass bookkeeping sends across to the boundary form.
        Everything that computes it - the `Real` space, the two `Real` fields,
        `enclosed_mass_forms`, `update_total_mass` and `check_net_mass` - stays
        on this class, because it is the only taped code here and the only code
        whose breakage a passing Taylor test cannot see; see
        `update_total_mass`. What crosses is a UFL expression in two `Real`
        coefficients, which the form treats exactly as it treats a prescribed
        `flux` condition, so a tape that is watching those coefficients follows
        it through with no new code.

        Empty in 3-D and in 2-D without an exterior DtN boundary.
        """
        return {bc_id: self.monopole_flux(bc_id)
                for bc_id in self.monopole_boundaries}

    def set_function_spaces(self) -> None:
        """Builds the internal mixed space: psi, cross-mesh dummy, multipliers.

        Vector-valued R-space unknowns are unsupported in Firedrake, so each
        trace coefficient is a separate scalar R field.
        """
        spaces = [self.solution_space]
        if self.cross_mesh:
            spaces.append(FunctionSpace(self.rho_mesh, "DG", 0))

        R_scalar = FunctionSpace(self.mesh, "R", 0)
        self.n_multipliers = self.form.n_multipliers
        if self.n_multipliers > 200:
            warn(
                f"{self.n_multipliers} DtN multipliers requested; the "
                "per-field bookkeeping of scalar R spaces makes large "
                "truncation degrees expensive. Consider a lower truncation "
                "combined with a buffer region.")
        self._multiplier_offset = len(spaces)
        spaces.extend([R_scalar] * self.n_multipliers)

        self.mixed_space = MixedFunctionSpace(spaces)
        self.mixed_solution = Function(self.mixed_space)

    def set_form(self) -> None:
        """Sets the weak form: Poisson, DtN maps, sheets, and flux conditions.

        Only the volume half is written here. The boundary half comes from
        `self.form`, in two pieces rather than one, because the volume source
        sits between them in the term order this form has always had.
        """
        trials = split(self.mixed_solution)
        tests = TestFunctions(self.mixed_space)
        psi, v = trials[0], tests[0]
        i_R = self._multiplier_offset
        multipliers = list(zip(trials[i_R:], tests[i_R:]))

        F = dot(grad(psi), grad(v)) * self.dx
        F -= 4 * np.pi * self.G * self.rho * v * self.dx_rho
        if self.cross_mesh:
            # The dummy field makes Firedrake set up the cross-mesh entity
            # maps; its equation is trivial.
            lam, mu_lam = trials[1], tests[1]
            F += lam * mu_lam * Measure("dx", domain=self.rho_mesh)

        F += self.form.boundary_bilinear(psi, v, multipliers)
        F -= self.form.boundary_source(v, extra_flux=self.monopole_fluxes())

        self.F = F
        self.strong_bcs = [
            DirichletBC(self.mixed_space.sub(0), val, bc_id)
            for bc_id, val in self.dirichlet_bcs]

    def set_lowrank_operator(self) -> None:
        """Builds the Robin-shifted stiffness, the right-hand side and `C`.

        The multipliers are eliminated by hand (see `gadopt.dtn_lowrank`), so
        there is no mixed space, no Schur complement and no matrix-free
        operator: one scalar space, one assembled SPD matrix, and a rank-`n`
        update applied in factored form. `mixed_solution` still exists and is
        still the thing to reset before a timed solve, but it is now a plain
        `Function` on the potential space rather than a mixed one.
        """
        V = self.solution_space
        u, v = TrialFunction(V), TestFunction(V)
        self.mixed_space = V
        self.mixed_solution = Function(V)
        # Zero, and the boundary form's own count is not: there is no field to
        # enumerate on this path, which is exactly what `set_solver_options`
        # and the 128-field tests read this number for.
        self.n_multipliers = 0
        self._multiplier_offset = 0

        a = dot(grad(u), grad(v)) * self.dx
        a += self.form.boundary_bilinear(u, v)

        F = 4 * np.pi * self.G * self.rho * v * self.dx_rho
        F += self.form.boundary_source(v, extra_flux=self.monopole_fluxes())

        self.a_form, self.rhs_form = a, F
        self.strong_bcs = [
            DirichletBC(V, val, bc_id) for bc_id, val in self.dirichlet_bcs]

        self.form.build_mode_rows()
        self.build_time = sum(rows.build_time for rows in self.mode_rows)

        constrained = set()
        for bc in self.strong_bcs:
            constrained.update(np.asarray(bc.nodes, dtype=np.int64).tolist())
        apply_dirichlet_to_rows(self.mode_rows, constrained)

    def set_lowrank_solver(self) -> None:
        """Drives a PETSc KSP directly: CG on `A + B`, preconditioned by `A`.

        `NonlinearVariationalSolver` offers no hook to hand PETSc a custom
        `Amat` while taking `Pmat` from a form, so this path owns its KSP. That
        is also why the taped variational solve disappears here.

        `ksp.setOperators(N, A)` with different `Amat` and `Pmat` is exactly the
        paper's "build the multigrid on the stiffness alone"; our `A` is
        Robin-shifted rather than `epsilon`-shifted, so it is SPD by
        construction with nothing to tune.
        """
        self.A = assemble(self.a_form, bcs=self.strong_bcs, mat_type="aij")
        petsc_A = self.A.petscmat
        petsc_A.setOption(petsc_A.Option.SPD, True)

        comm = self.mesh.comm
        self.operator_context = LowRankDtNOperator(petsc_A, self.mode_rows, comm)
        self.N = PETSc.Mat().createPython(
            petsc_A.getSizes(), self.operator_context, comm=comm)
        self.N.setUp()
        self.N.setOption(PETSc.Mat.Option.SYMMETRIC, True)

        self.ksp = PETSc.KSP().create(comm=comm)
        self.ksp.setOperators(self.N, petsc_A)
        self.ksp.setOptionsPrefix(self.name + "_lowrank_")
        options = PETSc.Options()
        for key, value in _flatten_options(self.solver_parameters).items():
            options[self.name + "_lowrank_" + key] = value
        self.ksp.setFromOptions()

    #: Preconditioners that belong to `SelfGravitatingGIASolver` and cannot
    #: work here. Refused by NAME rather than left to fail inside PETSc,
    #: following `SelfGravitatingGIASolver._check_block0_split_matches_layout`:
    #: a configuration mistake whose only symptom is an opaque PETSc code is
    #: exactly the class of failure that pattern exists to convert into a
    #: sentence. This one is not exotic - it is the most likely misuse, because
    #: the natural act on seeing the coupled system's multiplier block get a
    #: speedup is to copy the three block-1 lines into a standalone solve.
    _COUPLED_ONLY_PC_NAMES = ("DtNMultiplierDiagPC",)

    def _refuse_coupled_only_preconditioners(self, config) -> None:
        """Raises on a preconditioner that only `SelfGravitatingGIASolver` can use.

        Scans the flattened options for a `pc_python_type` naming one of
        `_COUPLED_ONLY_PC_NAMES`. Left to itself the misuse produces
        `PETSc.Error: error code 101` - measured - which names neither the
        preconditioner nor the reason, because PETSc flattens a Python
        exception raised inside a python PC.
        """
        if not isinstance(config, Mapping):
            return
        for key, value in _flatten_options(config).items():
            if not (isinstance(value, str) and key.endswith("pc_python_type")):
                continue
            if value.rsplit(".", 1)[-1] in self._COUPLED_ONLY_PC_NAMES:
                raise ValueError(
                    f"{key} = {value!r} is scoped to SelfGravitatingGIASolver "
                    "and cannot be used with GravitySolver.\n"
                    "Two independent reasons, neither of them fixable by "
                    "options:\n"
                    "  1. The block-1 rows differ. The coupled solver scales "
                    "every DtN constraint row by theta_psi; GravitySolver "
                    "leaves them unscaled, so the (c,c) diagonal that "
                    "preconditioner inverts is wrong here by exactly that "
                    "factor.\n"
                    "  2. GravitySolver supplies no 'dtn_block1_diagonal' in "
                    "its appctx, and wiring one would put an appctx-carried "
                    "object on the path that owns the shipped, verified "
                    "adjoint - while pyadjoint drops appctx from the kwargs of "
                    "the adjoint solve (see DtNTwoBlockSchurPC).\n"
                    "Leave block 1 at pc_type: 'none' here. The speedup you "
                    "are copying belongs to the coupled system.")

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None,
    ) -> None:
        """Sets PETSc solver options.

        The psi (potential) block is the only thing that varies between the two
        presets. Matrices with R-space blocks cannot be assembled
        monolithically, so with DtN multipliers present both presets share a
        full Schur complement eliminating onto the R fields: the psi block is
        assembled and inverted (directly by MUMPS, or iteratively by CG + AMG),
        and the tiny dense multiplier Schur complement is handled by GMRES.
        Without multipliers the psi solver is applied to the whole assembled
        system.

        The split itself is set up by `gadopt.DtNTwoBlockSchurPC`, which
        describes the two blocks to PETSc as index sets rather than as lists of
        field numbers; that is what keeps the solve working past the 128-field
        limit PETSc imposes on field enumeration, which one scalar R space per
        angular mode otherwise crosses at a spherical truncation of L = 7 on a
        two-boundary shell (L = 11 with a single DtN boundary). Its sub-solvers
        are configured under the `dtn` entry.

        A full solver-options dictionary is used verbatim. The strings "direct"
        and "iterative" select a G-ADOPT preset; None picks one by dimension
        (direct in 2-D, iterative in 3-D), mirroring the Stokes solver, so the
        scalable path is the default at production scale.
        """
        self._refuse_coupled_only_preconditioners(solver_preset)
        self._refuse_coupled_only_preconditioners(solver_extras)

        if isinstance(solver_preset, Mapping):
            if (self.n_multipliers > 0
                    and solver_preset.get("mat_type") in ("aij", "baij", "sbaij")):
                raise ValueError(
                    "Monolithic matrix assembly is unsupported with DtN "
                    "R-space multipliers. Provide a fieldsplit configuration "
                    "(see the G-ADOPT default) or adjust the default via "
                    "solver_parameters_extra instead.")
            self.add_to_solver_config(solver_preset)
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.set_solver)
            return

        if solver_preset is not None and solver_preset not in ("direct", "iterative"):
            raise ValueError(
                f"solver_parameters must be a dictionary or one of 'direct', "
                f"'iterative', got {solver_preset!r}.")
        if solver_preset is None:
            solver_preset = (
                "direct" if self.mesh.topological_dimension == 2 else "iterative")

        if self.dtn_representation == "lowrank":
            # This path drives a PETSc KSP directly, so the options are plain
            # KSP/PC options with no mat_type or snes_type: there is no
            # Firedrake solver object to interpret those. "direct" inverts the
            # PRECONDITIONER A by LU, not the operator A + B - the low-rank
            # update is never assembled, so no direct method applies to it, and
            # CG is still doing the work.
            if solver_preset == "direct":
                self.add_to_solver_config({
                    "ksp_type": "cg", "ksp_rtol": 1e-11, "ksp_max_it": 1000,
                    **{f"pc_{k}": v for k, v in (
                        ("type", "lu"),
                        ("factor_mat_solver_type", "mumps"))},
                })
            else:
                self.add_to_solver_config(dict(lowrank_gravity_solver_parameters))
            if INFO >= log_level:
                self.add_to_solver_config({"ksp_converged_reason": None})
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.set_solver)
            return

        if self.n_multipliers == 0:
            # No DtN multipliers: the psi solver acts on the whole assembled
            # system. GAMG wants the plain (unprefixed) options here, since
            # there is no fieldsplit_0 AssembledPC to hand them to. This also
            # means no SPDAssembledPC and hence no MAT_SPD flag - the eigen-
            # estimates fall back to GMRES; benign, and not worth the
            # AssembledPC indirection on a monolithic aij solve. Unlike the
            # multiplier path there is no outer Krylov cleaning up the psi
            # solve, so ksp_rtol here IS the final solution accuracy (the same
            # situation as the Stokes preonly-outer path): tighten it if a
            # no-multiplier 3-D config ever needs gradient-quality output.
            base = {"mat_type": "aij", "snes_type": "ksponly"}
            if solver_preset == "direct":
                self.add_to_solver_config({**base, **direct_gravity_solver_parameters})
            else:
                self.add_to_solver_config({
                    **base,
                    "ksp_type": "cg",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                    **gamg_parameters(),
                })
                if INFO >= log_level:
                    self.add_to_solver_config({"ksp_converged_reason": None})
        else:
            if solver_preset == "direct":
                fieldsplit_0 = {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled": dict(direct_gravity_solver_parameters),
                }
            else:
                fieldsplit_0 = dict(iterative_gravity_solver_parameters)
            self.add_to_solver_config({
                "mat_type": "matfree",
                "snes_type": "ksponly",
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-11,
                "pc_type": "python",
                "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
                "dtn": {
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0": fieldsplit_0,
                    "fieldsplit_1": {
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-6,
                        "pc_type": "none",
                    },
                },
            })
            if INFO >= log_level:
                self.add_to_solver_config({"ksp_converged_reason": None})
            if DEBUG >= log_level:
                self.add_to_solver_config({
                    "dtn": {
                        "fieldsplit_0": {"ksp_converged_reason": None},
                        "fieldsplit_1": {"ksp_converged_reason": None},
                    },
                })

        self.add_to_solver_config(solver_extras)
        self.warn_on_stale_fieldsplit_options()
        self.register_update_callback(self.set_solver)

    def warn_on_stale_fieldsplit_options(self) -> None:
        """Warns about fieldsplit options left at the top level of the preset.

        The two Schur blocks used to be described to PETSc by field number, so
        their sub-solvers sat directly under the solver options. They now live
        inside `gadopt.DtNTwoBlockSchurPC` under its own `dtn` prefix, and a
        `fieldsplit_0` entry passed at the top level of
        `solver_parameters_extra` silently stops matching anything rather than
        failing - the one way this change can bite an existing script.
        """
        if self.n_multipliers == 0:
            return
        stale = sorted(
            key for key in self.solver_parameters
            if key.startswith(("fieldsplit_", "pc_fieldsplit_")))
        if stale:
            warn(
                f"Solver options {stale} sit at the top level, where the DtN "
                "preset no longer reads them: the two Schur blocks are now "
                "described by index sets inside gadopt.DtNTwoBlockSchurPC and "
                "its sub-solvers live under the 'dtn' entry. Pass them as "
                "solver_parameters_extra={'dtn': {'fieldsplit_0': ...}} "
                "instead.")

    def set_solver(self) -> None:
        """Sets up the Firedrake variational problem and solver."""
        if self.dtn_representation == "lowrank":
            self.set_lowrank_solver()
            return
        self.problem = NonlinearVariationalProblem(
            self.F, self.mixed_solution, bcs=self.strong_bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            options_prefix=self.name,
        )

    def update_total_mass(self) -> None:
        """Refreshes the enclosed-mass scalars the monopole datum reads.

        Called from `solve` and not from `__init__`, so that the assembles and
        the assigns land on the tape. That placement is the whole design: the
        mass has to be differentiable without ever becoming an unknown.

        **Do not freeze these as `float`s.** Doing so severs the channel in a
        way no Taylor test can see: the taped map is then self-consistent, so
        `taylor_test` converges at rate 2.0000 while the gradient of the actual
        forward map is wrong - measured at 325% on both representations, the
        adjoint returning 4.948 where a central difference of fresh solves gives
        1.163. The instrument that catches it is a finite difference of *fresh
        solves*, which is why the tests carry one.

        **And do not reach for the obvious spelling either.** Getting an
        assembled scalar into a form differentiably is narrower than it looks;
        of the five ways to write it, two work. Measured on a stand-alone
        problem whose right-hand side depends on the control only through the
        scalar, against a reference gradient of 6.168:

        | spelling                                   | adjoint | tangent |
        | ------------------------------------------ | ------- | ------- |
        | `Constant.assign(assemble(...))`           | **0.0** | fails   |
        | `assemble(...).riesz_representation("l2")` | **0.0** | fails   |
        | `Function(R).assign(assemble(...))`        | 6.168   | **fails** |
        | `assemble(...).riesz_representation("L2")` | 6.168   | 6.168   |
        | a small variational problem on `R`         | 6.168   | 6.168   |

        The first two sever silently, which is the failure mode this whole item
        exists to avoid. The third is the tempting one and is what the plan note
        proposed: its adjoint is exact, because `assemble` of a 0-form returns
        an `AdjFloat` and the assign is taped, but its *tangent* value arrives
        as a bare `numpy.float64` that `derivative` rejects, so it fails the
        TLM-adjoint dot-product identity while every Taylor test still passes.
        The last two are equivalent - the `L2` Riesz map is a mass-matrix solve
        and emits the same block - so the explicit one is written out, because
        it says what it does without an argument about which volume factor
        cancels which.

        **The solve is rebuilt on every call, and must be.** Hoisting it into a
        cached `LinearVariationalSolver` is the obvious optimisation - it saves
        3.2 ms of the datum's 3.8 ms, since the one-by-one operator is a fixed
        number - and it is wrong in parallel. A cached solver whose right-hand
        side carries a *facet* integral into a `Real` space returns the correct
        value on its first solve and garbage afterwards: measured on a unit
        disc, `1.881574` then `0.0` alternating on two ranks, and
        `1.881574, -3.763149, 13.171020, -37.631486` on four. The KSP still
        converges, so the failure is a silently wrong potential in exactly the
        configuration this item exists to enable - a mass sheet on a 2-D
        exterior boundary, solved more than once, in parallel.

        Every other spelling is stable at 1, 2 and 4 ranks: plain repeated
        1-form assembly of the same form object, of a fresh one, assembly into
        a pre-zeroed tensor, 0-form assembly, and the fresh solve used here.
        Only the cached solver reuses whatever state is at fault. This looks
        like a Firedrake defect rather than a misuse and is worth reporting
        upstream, alongside the `R`-row reproducer `SOLVE-STRATEGIES.md`
        already owes them.

        A no-op outside 2-D, and in 2-D without an exterior DtN boundary, so
        the 3-D path pays nothing for any of this.
        """
        if not self.monopole_boundaries:
            return

        identity, mass, flux = self.enclosed_mass_forms()
        solve(identity == mass, self.source_mass,
              solver_parameters=real_scalar_solver_parameters)
        if flux is not None:
            solve(identity == flux, self.cavity_flux,
                  solver_parameters=real_scalar_solver_parameters)
        self.check_net_mass()

    def total_enclosed_mass(self) -> float:
        """Everything the 2-D exterior DtN boundary encloses, as a number.

        Density, sheets, and the cavity masses implied by prescribed-flux
        boundaries. Valid after `solve`; zero when there is no such boundary,
        which includes every 3-D configuration - there the scalars are never
        built, because the 3-D `l = 0` exterior mode is determined by the trace
        and needs none of this.

        Diagnostic only - the differentiable quantities are `source_mass` and
        `cavity_flux`, and reading a float off them is the sever documented in
        `update_total_mass`.
        """
        if not self.monopole_boundaries:
            return 0.0
        return (self._real_value(self.source_mass)
                + self._real_value(self.cavity_flux)
                / (4.0 * np.pi * self._real_value(self.G)))

    @staticmethod
    def _real_value(value) -> float:
        """A number from a `Constant`, a `Real` `Function` or a plain scalar.

        `Real` data is replicated on every rank, so reading entry 0 is a local
        operation that needs no reduction and agrees everywhere.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(value.dat.data_ro[0])

    def check_net_mass(self) -> None:
        """Warns about monopole leakage, and refuses a doubly-anchored gauge.

        This used to refuse any nonzero net mass in 2-D, because the monopole
        datum did not exist. It does now (see the module docstring), so nonzero
        mass is ordinary, and what is left is the check's second job: a density
        that *should* integrate to zero but does not usually has DG0 leakage
        from an interface that does not conform to cell edges, which is a
        genuine accuracy problem whatever the boundary treatment.

        "Nonzero" can no longer be the trigger for that, so the trigger is
        "nonzero but implausibly small": a leaked monopole is a discretisation
        residue a few orders above round-off, while a mass anyone meant to be
        there sits far above the band. The band errs quiet - it will miss
        leakage on a density whose own mass is genuinely tiny - which is the
        right direction for a warning that would otherwise fire on every
        legitimate nonzero-mass problem and teach users to filter it.

        The refusal that replaces the old one is the gauge argument's caveat.
        With a strong `psi` condition present, `v = 1` is not admissible, so
        the Robin monopole relation and the Dirichlet condition become two
        competing anchors for the same constant and the system is
        over-constrained. Measured on an annulus, that costs 1.1555 relative
        error, identical at two refinements - a real over-constraint, not a
        convergence artefact - so it is refused rather than warned about.
        """
        if not self.monopole_boundaries:
            return

        scale = assemble(abs(self.rho) * self.dx_rho)
        for bc_id, sigma, integral_type in self.sigma_bcs:
            # `ensure_constant` for the same reason as in `enclosed_mass_forms`:
            # a Python `0.0` folds `abs(sigma) * ds` away entirely, and
            # assembling nothing raises `IndexError: tuple index out of range`
            # from inside UFL rather than returning the zero it means.
            scale += assemble(self.sheet_integral(
                abs(ensure_constant(sigma)), bc_id, integral_type))
        mass = abs(self._real_value(self.source_mass))
        # Against the density's own scale and nothing else. An earlier
        # `max(scale, 1.0)` floor turned this into an absolute test below
        # scale 1, so uniformly rescaling rho - a transformation the whole
        # linear problem is invariant under - switched the refusal below off:
        # rho = 1e-9 was refused and rho = 1e-10 solved silently and wrongly.
        relative = mass / scale if scale > 0.0 else float(mass > 0.0)
        # The flux conditions are user data rather than a discretised field, so
        # they cannot leak; they can still anchor a monopole, hence the second
        # term here and nowhere in the leakage band below.
        anchored = relative > 1e-8 or self._real_value(self.cavity_flux) != 0.0

        if anchored and self.dirichlet_bcs:
            raise ValueError(
                f"Net enclosed mass {self.total_enclosed_mass():.3e} is "
                "nonzero on a 2-D mesh that has both an exterior DtN boundary "
                f"and a strong 'psi' condition on boundary "
                f"{self.dirichlet_bcs[0][0]}. The 2-D monopole datum fixes the "
                "potential's additive constant through the Robin term, and the "
                "Dirichlet condition fixes it again: the two anchors disagree "
                "and the system is over-constrained. Drop one of them, or use "
                "a zero-net-mass density.")

        if 1e-8 < relative < 1e-4:
            warn(
                f"Net mass {self._real_value(self.source_mass):.3e} is "
                f"{relative:.1e} of the density's own scale - nonzero, but too "
                "small to look deliberate. This is the signature of monopole "
                "leakage from a density interface that does not conform to "
                "cell edges, which is an accuracy problem in its own right; "
                "align the mesh with the density interfaces. If the mass is "
                "meant to be there, ignore this.")

    def solve(self) -> None:
        """Solves the system and updates the solution function in place."""
        self.update_total_mass()
        if self.dtn_representation == "lowrank":
            self._solve_lowrank()
            annotate_lowrank_solve(self)
        else:
            self.solver.solve()
        self.solution.assign(self.mixed_solution.subfunctions[0])

    def _solve_lowrank(self, rhs_form=None) -> None:
        """CG on `A + B` with the right-hand side reassembled each call.

        `rho` and the sheet densities may have changed since construction;
        `A + B` cannot have, since every coefficient in it is geometry.

        Strong conditions are lifted by hand, and **both** halves of that are
        necessary. `assemble(L, bcs=...)` *zeroes* the constrained entries -
        the homogeneous convention - rather than setting them to the
        prescribed values, so the write-back below is not redundant tidying:
        it is what makes the answer right. And it subtracts none of the
        coupling of the prescribed values into the free rows. Measured against
        Firedrake's own solver on a plain Dirichlet Poisson problem with no
        DtN anywhere:

            raw KSP, b = assemble(L, bcs=bcs)   relative error 1.0
            fd.solve(A, w, b), same b           relative error 3.0e-16
            the construction below              relative error 4.4e-16

        Both the lifting and the value-setting live in Firedrake's `solve`
        wrapper rather than in assembly. **Any Firedrake code that drives a
        KSP directly has to do both by hand** - which is why the ordinary
        idiom works everywhere else in this codebase and fails only here.

        `lift` must be the ZERO EXTENSION of the boundary data, not the data
        function itself: `action(a, g)` with `g` nonzero in the interior
        over-subtracts by the interior block and gives a relative error of 1.0
        again. The wrong version looks more natural.

        The lifting needs `A` only: the columns of `C` are zeroed at
        constrained degrees of freedom, so `B` applied to a field supported
        only there is zero.
        """
        rhs_form = self.rhs_form if rhs_form is None else rhs_form
        if self.strong_bcs:
            lift = Function(self.solution_space)
            for bc in self.strong_bcs:
                bc.apply(lift)
            b = assemble(rhs_form - action(self.a_form, lift))
            # `bc.nodes` carries ghost entries too, while `dat.data_wo` is the
            # owned block; the vector handed to the KSP is owned-only, so the
            # ghosts are both unnecessary and out of range.
            owned = self.solution_space.dof_dset.size
            for bc in self.strong_bcs:
                nodes = bc.nodes[bc.nodes < owned]
                b.dat.data_wo[nodes] = lift.dat.data_ro[nodes]
        else:
            b = assemble(rhs_form)
        with b.dat.vec_ro as rhs, self.mixed_solution.dat.vec as x:
            self.ksp.solve(rhs, x)
        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"The low-rank DtN solve did not converge: PETSc reason "
                f"{self.ksp.getConvergedReason()} after "
                f"{self.ksp.getIterationNumber()} iterations.")

    def coefficients(self) -> dict[int | str, dict[str, float]]:
        """Solved trace coefficients of every DtN boundary, keyed by marker.

        For each DtN boundary, maps mode labels (e.g. "cos3", "Y2,-1") to the
        solved trace coefficients - the spectrum of psi on that boundary,
        i.e. the geoid coefficients when evaluated at the surface.
        """
        out = {bc_id: {} for bc_id, _ in self.dtn_boundaries}
        if self.dtn_representation == "lowrank":
            # Taped, not read off .dat.data_ro and float()ed: c_k is a linear
            # functional of psi with a constant vector, so it is trivially
            # differentiable, and the trace spectrum is the geoid rather than a
            # diagnostic. The values are AdjFloat, which is a float subclass,
            # so callers that only want a number are unaffected.
            return taped_trace_coefficients(self)
        i_R = self._multiplier_offset
        for (bc_id, key), f in zip(
                self._multiplier_keys, self.mixed_solution.subfunctions[i_R:]):
            out[bc_id][key] = float(f)
        return out
