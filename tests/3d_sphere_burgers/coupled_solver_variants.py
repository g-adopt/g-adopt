"""Solver variants for the Burgers internal-variable comparison.

One problem builder, one nullspace builder and one solver constructor per
configuration. The benchmark script and the Gadi scaling driver both import
this module so that every comparison uses the same problem and the same
solver dictionaries. The configuration names and the layout of their option
dictionaries live in ``solver_configs.py``, which needs no Firedrake.

The five configurations are:

- ``substituted``: `InternalVariableSolver`, the displacement-only reference.
- ``multiplicative``: `CoupledInternalVariableSolver` with the symmetric
  multiplicative fieldsplit over displacement and internal variables that was
  the shipped preset before static condensation
  (`multiplicative_gia_solver_parameters`). The baseline this work started
  from; it needs many sweeps and can fail, so it runs with an outer iteration
  cap.
- ``schur-a11``: Schur fieldsplit that eliminates the internal variables
  exactly and preconditions the displacement Schur complement with GAMG on
  the elastic block. PETSc options only.
- ``schur-substituted``: same layout, with GAMG on the substituted
  (``eta_eff``) operator assembled by `gadopt.SubstitutedDisplacementPC`.
- ``static-condensation``: Slate static condensation of the internal variables
  through `gadopt.InternalVariableSCPC`, then CG and GAMG on the exact
  condensed displacement operator. This is the shipped preset of
  `CoupledInternalVariableSolver`; the subclass here only exposes the Krylov
  method and the nullspace callables of the comparison.

Every coupled configuration keeps both Maxwell elements in one internal-variable
field of shape ``(2, 3, 3)`` (see `gadopt.internal_variable_equation`), so the
internal variables are always field 1 of the mixed space.

Every solver modifies its configuration through `add_to_solver_config` and
`DeleteParam`, never by assigning `solver_parameters` directly, so that the
GPU and monitoring hooks of `StokesSolverBase` keep working. The two coupled
subclasses hand the base class their own iterative preset and name the
prefix of their displacement block (`displacement_block_prefix`), so the
base class places the Krylov and GAMG options where each layout needs them.
"""

import numpy as np

from gadopt import (
    CompressibleInternalVariableApproximation,
    Constant,
    CoupledInternalVariableSolver,
    CubedSphereMesh,
    DeleteParam,
    ExtrudedMesh,
    Function,
    FunctionSpace,
    InternalVariableSolver,
    MixedVectorSpaceBasis,
    SpatialCoordinate,
    VectorFunctionSpace,
    atan2,
    get_boundary_ids,
    pi,
    rigid_body_modes,
    sqrt,
    tanh,
)
from gadopt.internal_variable_equation import internal_variable_space
from gadopt.stokes_integrators import (
    multiplicative_gia_solver_parameters,
    newton_stokes_solver_parameters,
)
from gadopt.utility import extruded_layer_heights, initialise_background_field
from solver_configs import CONFIGURATIONS, displacement_block_key

# Outer radius, base of the lithosphere, 420 km and 670 km discontinuities,
# core-mantle boundary. Metres.
RADII_M = np.asarray([6371e3, 6301e3, 5951e3, 5701e3, 3480e3])
YEAR_S = 8.64e4 * 365.25
DENSITY_SCALE = 4500.0
SHEAR_SCALE = 1e11
VISCOSITY_SCALE = 1e21
MAXWELL_SCALE = VISCOSITY_SCALE / SHEAR_SCALE


def _material_field(space, values, radii, coordinate, name):
    field = Function(space, name=name)
    initialise_background_field(field, values, coordinate, radii)
    return field


def burgers_problem(reflevel, dg0_layers, dt_years=1000.0, symmetric_tensor=True):
    """Build the Burgers sphere problem of ``3d_sphere_burgers.py``.

    Args:
      reflevel: cubed-sphere refinement level (surface cells = 6 * 4**reflevel).
      dg0_layers: radial cells per rheological shell, one integer for all four
        shells or a list of four integers (lithosphere, upper mantle, transition
        zone, lower mantle).
      dt_years: time step in years. The Maxwell times of the mantle shells are
        290 to 1270 years, so this sets ``dt/tau`` and with it the difficulty of
        the coupled solve.
      symmetric_tensor: store the internal variables as symmetric tensors
        (the default of `internal_variable_space`).

    Returns:
      ``(mesh, V, S, dt, bcs, make_approximation)`` where ``make_approximation``
      builds a fresh approximation each time it is called, because every solver
      assigns its own shear coefficient to ``approximation.mu``.
    """
    radii = RADII_M / (RADII_M[0] - RADII_M[-1])
    depth = RADII_M[0] - RADII_M[-1]

    surface_mesh = CubedSphereMesh(
        radii[-1], refinement_level=reflevel, degree=2, name="surface_mesh"
    )
    # extruded_layer_heights wants one entry per radius; the last is unused.
    if not isinstance(dg0_layers, int):
        dg0_layers = [*dg0_layers, 0] if len(dg0_layers) == 4 else list(dg0_layers)
    layer_heights = extruded_layer_heights(dg0_layers, list(radii))
    mesh = ExtrudedMesh(
        surface_mesh,
        layers=len(layer_heights),
        layer_height=layer_heights,
        extrusion_type="radial",
    )
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    # One field for both Maxwell elements, shape (2, 3, 3).
    S = internal_variable_space(mesh, 2, symmetric=symmetric_tensor)
    DG0 = FunctionSpace(mesh, "DG", 0)
    DG1 = FunctionSpace(mesh, "DG", 1)
    X = SpatialCoordinate(mesh)

    dt = Constant(dt_years * YEAR_S / MAXWELL_SCALE)

    density = _material_field(
        DG0, np.asarray([3037, 3438, 3871, 4978]) / DENSITY_SCALE, radii, X, "density"
    )
    shear_values = 0.5 * np.asarray([0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]) / SHEAR_SCALE
    shear_1 = _material_field(DG0, shear_values, radii, X, "shear modulus 1")
    shear_2 = _material_field(DG0, shear_values, radii, X, "shear modulus 2")
    bulk = _material_field(DG0, 2 * shear_values, radii, X, "bulk modulus")
    viscosity_log = np.log10(0.5 * np.asarray([1e40, 1e21, 1e21, 2e21]) / VISCOSITY_SCALE)
    viscosity_1 = _material_field(DG1, viscosity_log, radii, X, "viscosity 1")
    viscosity_2 = _material_field(DG1, viscosity_log, radii, X, "viscosity 2")
    viscosity_1.interpolate(10**viscosity_1)
    viscosity_2.interpolate(10**viscosity_2)

    B_mu = Constant(DENSITY_SCALE * depth * 9.815 / SHEAR_SCALE)
    distance_from_axis = sqrt(X[0] ** 2 + X[1] ** 2)
    colatitude = atan2(distance_from_axis, X[2])
    disc_halfwidth = (2 * pi / 360) * 10
    surface_dx = 200e3
    surface_resolution = 2 * pi / (2 * pi * radii[0] * depth / surface_dx)
    disc = 0.5 * (1 - tanh((abs(colatitude) - disc_halfwidth) / (2 * surface_resolution)))
    ice_load = B_mu * (931 / DENSITY_SCALE) * (1000 / depth) * disc
    bcs = {
        boundary.bottom: {"un": 0},
        boundary.top: {"normal_stress": ice_load, "free_surface": {}},
    }

    def make_approximation():
        return CompressibleInternalVariableApproximation(
            bulk_modulus=bulk,
            density=density,
            shear_modulus=[shear_1, shear_2],
            viscosity=[viscosity_1, viscosity_2],
            B_mu=B_mu,
            bulk_shear_ratio=1.94,
        )

    return mesh, V, S, dt, bcs, make_approximation


def make_nullspaces(V, Z=None):
    """Rigid-body nullspace and near-nullspace on V, lifted to Z if given."""
    exact_u = rigid_body_modes(V, rotational=True)
    near_u = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
    if Z is None:
        return exact_u, near_u
    others = [Z.sub(i) for i in range(1, len(Z))]
    return (
        MixedVectorSpaceBasis(Z, [exact_u, *others]),
        MixedVectorSpaceBasis(Z, [near_u, *others]),
    )


def _internal_variable_options(internal_solver, internal_rtol):
    """Options for one internal-variable block.

    ``local-lu`` is a rank-local LU, exact for the DG mass matrix because it
    has no off-rank coupling. ``cg-sor`` is the shipped iterative solve.
    """
    if internal_solver == "local-lu":
        return {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "bjacobi",
            "assembled_sub_ksp_type": "preonly",
            "assembled_sub_pc_type": "lu",
        }
    if internal_solver == "cg-sor":
        return {
            "ksp_type": "cg",
            "ksp_rtol": internal_rtol,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "sor",
        }
    raise ValueError(f"Unknown internal_solver {internal_solver!r}")


def _rigid_body_near_nullspace(V):
    return rigid_body_modes(V, rotational=True, translations=[0, 1, 2])


class MultiplicativeCoupledInternalVariableSolver(CoupledInternalVariableSolver):
    """The symmetric multiplicative fieldsplit that preceded static condensation.

    Kept as the baseline of the comparison. The displacement block is
    `fieldsplit_0`, a matrix-free block that `SPDAssembledPC` assembles for
    GAMG; the internal-variable block is `fieldsplit_1`, whose options the
    driver supplies.
    """

    displacement_block_prefix = "fieldsplit_0"
    displacement_block_assembled = True

    def set_solver_options(self, solver_preset, solver_extras, gpu_extras):
        snes = {"snes_type": "ksponly"} if self._newtonian_rheology() else newton_stokes_solver_parameters
        super().set_solver_options(
            solver_preset, solver_extras, gpu_extras,
            iterative_preset=multiplicative_gia_solver_parameters | snes,
        )


class SchurCoupledInternalVariableSolver(CoupledInternalVariableSolver):
    """Schur fieldsplit that eliminates the internal variables exactly.

    Split 0 holds the internal-variable field; its block is cell-local (a DG
    mass matrix per Maxwell element) and is inverted exactly. Split 1 is the
    displacement, whose KSP sees the exact Schur complement.
    ``schur_precondition`` selects the matrix GAMG runs on: ``a11`` (the
    elastic block, shear coefficient ``mu0``) or ``substituted``
    (`gadopt.SubstitutedDisplacementPC`, coefficient ``eta_eff``).

    The history equations carry the boundary term that makes the exact Schur
    complement symmetric at weak-normal boundaries, so the Schur field can
    run CG; GMRES stays the default of the comparison so that old and new
    runs read the same. The outer method is ``preonly`` by default: with the
    internal-variable block inverted exactly, one application of the full
    Schur factorisation is the whole linear solve, which is the same
    accounting as the substituted solver's single CG solve.

    Args:
      schur_precondition: ``a11`` or ``substituted``, see above.
      schur_ksp: Krylov method on the displacement Schur complement.
      schur_outer: Krylov method of the outer solve, ``preonly`` or ``fgmres``.
      internal_solver: ``local-lu`` or ``cg-sor`` for the internal-variable block.
      internal_rtol: tolerance of the ``cg-sor`` internal solve.

    The remaining arguments are those of `CoupledInternalVariableSolver`.
    """

    displacement_block_prefix = "fieldsplit_1"
    # The displacement split is a matrix-free block that `SPDAssembledPC`
    # (or `SubstitutedDisplacementPC`) assembles for GAMG.
    displacement_block_assembled = True

    def __init__(
        self,
        solution,
        approximation,
        /,
        *,
        schur_precondition="a11",
        schur_ksp="gmres",
        schur_outer="preonly",
        internal_solver="local-lu",
        internal_rtol=1e-5,
        **kwargs,
    ):
        if schur_precondition not in ("a11", "substituted"):
            raise ValueError(f"Unknown schur_precondition {schur_precondition!r}")
        self.schur_precondition = schur_precondition
        self.schur_ksp = schur_ksp
        self.schur_outer = schur_outer
        self.internal_solver = internal_solver
        self.internal_rtol = internal_rtol
        super().__init__(solution, approximation, **kwargs)

    def _iterative_preset(self):
        """The Schur layout, without the displacement split.

        The base class fills ``fieldsplit_1`` (this class's
        ``displacement_block_prefix``) with the Krylov and GAMG options, so it
        must be absent here or the base class treats the block as configured.
        """
        internal = _internal_variable_options(self.internal_solver, self.internal_rtol)
        snes = {"snes_type": "ksponly"} if self._newtonian_rheology() else newton_stokes_solver_parameters
        return {
            "mat_type": "matfree",
            "ksp_type": self.schur_outer,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_schur_precondition": "a11",
            "pc_fieldsplit_0_fields": "1",
            "pc_fieldsplit_1_fields": "0",
            "fieldsplit_0": internal,
        } | snes

    def set_solver_options(self, solver_preset, solver_extras, gpu_extras):
        super().set_solver_options(
            solver_preset, solver_extras, gpu_extras, iterative_preset=self._iterative_preset()
        )
        # GAMG needs the rigid-body modes on the matrix it assembles; the
        # fieldsplit hands no near-nullspace to a python preconditioner.
        self.appctx["displacement_near_nullspace"] = _rigid_body_near_nullspace

    def _configure_iterative_solver(self, device_type, gpu_extras):
        super()._configure_iterative_solver(device_type, gpu_extras)
        # The base class configures a CG solve on the block. The Schur
        # complement is nonsymmetric, so switch the Krylov method, and swap
        # the assembled elastic block for the substituted operator if asked.
        block = self.solver_parameters[self.displacement_block_prefix]
        updates = {"ksp_type": self.schur_ksp}
        if self.schur_ksp != "cg":
            updates["ksp_gmres_restart"] = 50
        if self.schur_precondition == "substituted":
            updates |= {
                "pc_python_type": "gadopt.SubstitutedDisplacementPC",
                "aux": block["assembled"],
                "assembled": DeleteParam,
            }
        self.add_to_solver_config({self.displacement_block_prefix: updates})


class StaticCondensationCoupledInternalVariableSolver(CoupledInternalVariableSolver):
    """Eliminate the DG internal variables cell by cell with Slate.

    `gadopt.InternalVariableSCPC` assembles the exact condensed displacement
    operator as a sparse matrix and solves it with the options under
    ``condensed_field``, then recovers the internal variables locally. The
    outer KSP is ``preonly``: the condensation is the whole linear solve.
    This is the shipped preset of `CoupledInternalVariableSolver`.

    Args:
      condensed_ksp: Krylov method on the condensed displacement operator.
        The operator is symmetric, so CG by default.

    The remaining arguments are those of `CoupledInternalVariableSolver`.
    """

    def __init__(self, solution, approximation, /, *, condensed_ksp="cg", **kwargs):
        self.condensed_ksp = condensed_ksp
        super().__init__(solution, approximation, **kwargs)

    def _configure_iterative_solver(self, device_type, gpu_extras):
        super()._configure_iterative_solver(device_type, gpu_extras)
        updates = {"ksp_type": self.condensed_ksp}
        if self.condensed_ksp != "cg":
            updates["ksp_gmres_restart"] = 50
        self.add_to_solver_config({self.displacement_block_prefix: updates})


def construct_solver(
    config,
    problem,
    *,
    outer_rtol=1e-5,
    internal_rtol=1e-5,
    internal_solver="local-lu",
    show_ksp_reasons=True,
    constant_jacobian=False,
    gpu_extra_parameters=None,
    schur_ksp="gmres",
    schur_outer="preonly",
    multiplicative_max_it=40,
):
    """Build the solver for one configuration.

    ``outer_rtol`` is the tolerance of the displacement KSP, wherever the
    configuration keeps it. Returns ``(solver, displacement)`` where
    ``displacement`` is the Function (or sub-function) that holds the
    displacement after each solve.
    """
    if config not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration {config!r}")
    mesh, V, S, dt, bcs, make_approximation = problem
    block_key = displacement_block_key(config)
    displacement_extras = {"ksp_rtol": outer_rtol}
    if show_ksp_reasons:
        displacement_extras["ksp_converged_reason"] = None
    extras = {
        "snes_monitor": DeleteParam,
        "ksp_monitor": DeleteParam,
        "ksp_converged_reason": None if show_ksp_reasons else DeleteParam,
    }
    if block_key is None:
        extras |= displacement_extras
    else:
        extras[block_key] = displacement_extras
    if config == "multiplicative":
        internal = _internal_variable_options(internal_solver, internal_rtol)
        # The outer GMRES must converge as tightly as the reference's CG for
        # the comparison to be fair. The block sweep can need dozens of full
        # elastic solves per step, so cap the outer iterations instead of
        # letting a job run out.
        extras |= {
            "ksp_rtol": outer_rtol,
            "fieldsplit_1": internal,
            "ksp_max_it": multiplicative_max_it,
        }
    common = dict(
        dt=dt,
        bcs=bcs,
        solver_parameters="iterative",
        solver_parameters_extra=extras,
        gpu_extra_parameters=gpu_extra_parameters or {},
        constant_jacobian=constant_jacobian,
    )

    if config == "substituted":
        u = Function(V, name="substituted displacement")
        internal = Function(S, name="substituted internal variables")
        exact, near = make_nullspaces(V)
        solver = InternalVariableSolver(
            u,
            make_approximation(),
            internal_variables=internal,
            nullspace=exact,
            transpose_nullspace=exact,
            near_nullspace=near,
            **common,
        )
        return solver, u

    Z = V * S
    z = Function(Z, name=f"{config} solution")
    exact, near = make_nullspaces(V, Z)
    common |= dict(nullspace=exact, transpose_nullspace=exact, near_nullspace=near)
    approximation = make_approximation()

    if config == "multiplicative":
        solver = MultiplicativeCoupledInternalVariableSolver(z, approximation, **common)
    elif config.startswith("schur-"):
        solver = SchurCoupledInternalVariableSolver(
            z,
            approximation,
            schur_precondition=config.removeprefix("schur-"),
            schur_ksp=schur_ksp,
            schur_outer=schur_outer,
            internal_solver=internal_solver,
            internal_rtol=internal_rtol,
            **common,
        )
    else:
        solver = StaticCondensationCoupledInternalVariableSolver(z, approximation, **common)
    return solver, z.subfunctions[0]
