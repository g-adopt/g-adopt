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
from .dtn_lowrank import LowRankDtNOperator
from .gravity_potential import (  # noqa: F401 - re-exported; see the note below
    GravityPotential, real_scalar_solver_parameters)
from .solver_options_manager import (ConfigType, SolverConfigurationMixin,
                                     gamg_parameters)
from .utility import (DEBUG, INFO, ensure_constant, log_level, scalar_value)

# The boundary treatment, the DtN descriptors and the calibrated quadrature
# apparatus live in `gadopt.dtn_form`, and the potential's measures, residual,
# 2-D monopole datum and low-rank mode rows in `gadopt.gravity_potential`, so
# that a coupled residual can use them without instantiating this solver.
# The names are re-exported here, because `from gadopt.gravity_solver import
# ...` is written in the tests and in the scaling scripts.

__all__ = ["CylindricalDtN", "DtNGravityForm", "SphericalDtN", "GravitySolver"]


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


def _forwarded(part: str, name: str, target: str | None = None) -> property:
    """A read-only forward from `GravitySolver` to an attribute of one of its parts.

    The boundary treatment is `self.form` (`DtNGravityForm`), the measures
    and the 2-D monopole datum are `self.gravity` (`GravityPotential`). The
    solver gives the names that the tests, the demos and the scaling scripts
    read a second name on itself. The forward returns the part's own object
    and not a copy, so `solver.ds is solver.form.ds` and the two cannot
    disagree.

    Args:
      part: the name of the part on the solver, `"form"` or `"gravity"`.
      name: the name on the solver.
      target: the name on the part, when it differs from `name`.
    """
    target = target or name

    def get(self):
        return getattr(getattr(self, part), target)
    get.__name__ = name
    return property(get, doc=f"Forwarded to `self.{part}.{target}`.")


def _from_form(name: str) -> property:
    """A read-only forward to the `DtNGravityForm` of the solver."""
    return _forwarded("form", name)


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

    The measures, the residual of the potential, the 2-D monopole datum and
    the low-rank mode rows belong to `GravityPotential` (`self.gravity`),
    the part that the coupled self-gravitating GIA solver composes too. This
    class adds its own mixed space, the solver options, the low-rank KSP and
    the recorded solve.

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

        #: The potential part: the boundary treatment (`self.form`), the
        #: volume measures, the residual with the volume density as its
        #: source, the 2-D monopole datum and the low-rank mode rows. It
        #: refuses an unknown `dtn_representation`. `alpha` is handed over
        #: instead of being left to the form's own class attribute, because
        #: the Robin shift is a setting of this class: a test subclasses
        #: `GravitySolver` and overrides it, and a form that read its own
        #: default would ignore the override without failing.
        self.gravity = GravityPotential(
            self.solution_space, self.rho_mesh, bcs,
            gravitational_constant=self.G, volume_density=self.rho,
            source_quad_degree=source_quad_degree, quad_degree=quad_degree,
            alpha=self.alpha, representation=dtn_representation)
        #: The boundary treatment, `DtNGravityForm`: sorts `bcs`, chooses the
        #: boundary quadrature rule, measures every DtN boundary, and writes
        #: the boundary bilinear form and the boundary source.
        self.form = self.gravity.form
        if self.dtn_representation == "lowrank":
            self.set_lowrank_operator()
        else:
            self.set_function_spaces()
            self.set_form()
        self.warn_on_quadrature_rule_limits()
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.set_solver()

    # -- Forwarded to the parts -------------------------------------------
    #
    # Everything the DtN treatment and the potential part own is read off
    # `self.form` and `self.gravity` and not copied, so the two cannot drift
    # apart: if a measure is rebuilt on one side, every reader on the other
    # side sees the new one. `tests/unit/test_dtn_form.py` asserts identity
    # for every forward - `solver.ds is solver.form.ds` - and not merely that
    # the attribute exists, because a forward that only exists turns "every
    # existing test passes" into a statement about `getattr`.
    #
    # Three names do not forward. `alpha` is a class attribute of this class,
    # passed into the form by `__init__`, because a test subclasses
    # `GravitySolver` to vary the Robin shift and a form reading its own
    # default would ignore the override silently. `n_multipliers` is set by
    # the construction of each representation. `_multiplier_offset` indexes
    # this class's mixed space and means nothing to a boundary form that has
    # no space of its own.
    #
    # What is forwarded is what is read, not what constructs: the build steps
    # of the parts (`set_boundary_conditions`, `set_measures`,
    # `check_sheet_measures`, `set_boundary_geometry`, `set_monopole_datum`)
    # are reachable on the parts and are not given a second name here, since
    # running one of them again after the residual is built would rebuild
    # state the residual already refers to.
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
    #: `dx` integrates over the potential's mesh. When the density lives on
    #: a `Submesh` it carries the intersection with the density's mesh that
    #: makes Firedrake build the entity maps.
    dx = _forwarded("gravity", "dx", "dx_g")
    #: The 2-D monopole datum (`GravityPotential.set_monopole_datum`):
    #: the exterior DtN boundaries whose monopole the trace does not
    #: determine, and the two `Real` Functions of the enclosed mass, which
    #: are the differentiable quantities behind `total_enclosed_mass`. The
    #: Functions are `None` without such a boundary, which includes 3-D.
    monopole_boundaries = _forwarded("gravity", "monopole_boundaries")
    source_mass = _forwarded("gravity", "source_mass")
    cavity_flux = _forwarded("gravity", "cavity_flux")

    def set_function_spaces(self) -> None:
        """Builds the internal mixed space: psi, cross-mesh dummy, multipliers.

        Vector-valued R-space unknowns are unsupported in Firedrake, so each
        trace coefficient is a separate scalar R field.
        """
        spaces = [self.solution_space]
        if self.cross_mesh:
            spaces.append(FunctionSpace(self.rho_mesh, "DG", 0))

        R_scalar = FunctionSpace(self.mesh, "R", 0)
        self.n_multipliers = self.gravity.n_multipliers
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

        The potential row and the DtN constraint rows come from
        `GravityPotential.residual`, with the volume density as the source.
        A sheet on an interior facet restricts its whole integrand, `G`
        included, where the boundary form builds it (`sheet_integral`).
        """
        trials = split(self.mixed_solution)
        tests = TestFunctions(self.mixed_space)
        psi, v = trials[0], tests[0]
        i_R = self._multiplier_offset
        multipliers = list(zip(trials[i_R:], tests[i_R:]))

        F = self.gravity.residual(psi, v, multipliers=multipliers)
        if self.cross_mesh:
            # The dummy field makes Firedrake set up the cross-mesh entity
            # maps; its equation is trivial.
            lam, mu_lam = trials[1], tests[1]
            F += lam * mu_lam * Measure("dx", domain=self.rho_mesh)

        self.F = F
        self.strong_bcs = self.gravity.dirichlet_bcs(self.mixed_space.sub(0))

    def set_lowrank_operator(self) -> None:
        """Builds the Robin-shifted stiffness, the right-hand side and `C`.

        The multipliers are eliminated by hand (see `gadopt.dtn_lowrank`), so
        there is no mixed space, no Schur complement and no matrix-free
        operator: one scalar space, one assembled SPD matrix, and a rank-`n`
        update applied in factored form. `mixed_solution` is a plain
        `Function` on the potential space here, not a mixed one; the KSP
        writes its solution into it, and `solve` copies it into `solution`.

        The stiffness and the right-hand side are `GravityPotential.operator`
        without multipliers (the Robin shift alone) and
        `GravityPotential.source`; the mode rows `C`, with the rows of the
        strong conditions removed, are `GravityPotential.build_mode_rows`.
        """
        V = self.solution_space
        u, v = TrialFunction(V), TestFunction(V)
        self.mixed_space = V
        self.mixed_solution = Function(V)
        # Zero, and the boundary form's own count is not: there is no field to
        # enumerate on this path, which is what `set_solver_options` and the
        # 128-field tests read this number for.
        self.n_multipliers = 0
        self._multiplier_offset = 0

        self.a_form = self.gravity.operator(u, v)
        self.rhs_form = self.gravity.source(v)
        self.strong_bcs = self.gravity.dirichlet_bcs(V)

        mode_rows = self.gravity.build_mode_rows(V)
        self.build_time = sum(rows.build_time for rows in mode_rows)

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
    #: because a configuration mistake whose only symptom is an opaque PETSc code is
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

        The two Schur blocks are described to PETSc by index sets inside
        `gadopt.DtNTwoBlockSchurPC`, and their sub-solvers live under its own
        `dtn` prefix. A `fieldsplit_0` entry passed at the top level of
        `solver_parameters_extra` matches nothing there, and PETSc ignores it
        without an error, so this method warns.
        """
        if self.n_multipliers == 0:
            return
        stale = sorted(
            key for key in self.solver_parameters
            if key.startswith(("fieldsplit_", "pc_fieldsplit_")))
        if stale:
            warn(
                f"Solver options {stale} sit at the top level, where the DtN "
                "preset does not read them: the two Schur blocks are "
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

        Called from `solve` and not from `__init__`, so that the solves land
        on the tape: the mass has to be differentiable without becoming an
        unknown. The solvers are built once, by the constructor of
        `GravityPotential`. `GravityPotential.update_total_mass` gives the
        reasons for the recorded solve, the reset to zero before it, and the
        measurements behind them. Then `GravityPotential.check_net_mass` runs
        with the volume density and the sheets of `bcs` as the scale, off the
        tape. A no-op outside 2-D, and in 2-D without an exterior DtN
        boundary.

        This is a method of the solver so that a test can override it, for
        example to take the enclosed mass off the tape.
        """
        self.gravity.update_total_mass()
        self.gravity.check_net_mass()

    def total_enclosed_mass(self) -> float:
        """Everything the 2-D exterior DtN boundary encloses, as a number.

        Density, sheets, and the cavity masses implied by prescribed-flux
        boundaries. Valid after `solve`; zero when there is no such boundary,
        which includes every 3-D configuration - there the scalars are never
        built, because the 3-D `l = 0` exterior mode is determined by the trace
        and needs none of this.

        Diagnostic only - the differentiable quantities are `source_mass` and
        `cavity_flux`, and a float read off them is not on the tape
        (`GravityPotential.update_total_mass`).
        """
        if not self.monopole_boundaries:
            return 0.0
        return (scalar_value(self.source_mass)
                + scalar_value(self.cavity_flux)
                / (4.0 * np.pi * scalar_value(self.G)))

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
            # `bc.nodes` carries ghost entries too, while `dat.data_rw` is the
            # owned block; the vector handed to the KSP is owned-only, so the
            # ghosts are both unnecessary and out of range. The write is
            # `data_rw` and not `data_wo`: it changes only some entries of a
            # freshly assembled vector, and a write-only access in pyop3 drops
            # the halo reduction that the assembly may still have pending, which
            # would lose the ghost contributions on partition boundaries.
            owned = self.solution_space.nodes.owned.local_size
            for bc in self.strong_bcs:
                nodes = bc.nodes[bc.nodes < owned]
                b.dat.data_rw[nodes] = lift.dat.data_ro[nodes]
        else:
            b = assemble(rhs_form)
        with b.dat.vec_ro as rhs, self.mixed_solution.dat.vec_rw as x:
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
