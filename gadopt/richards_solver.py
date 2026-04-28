"""Solver class for the Richards equation governing variably saturated flow.

This module provides the `RichardsSolver` class for solving the Richards equation,
which describes movement of fluid phase in a two-phase flow system, if we ignore the
compressibility of the dry phase. The solver uses time integrators and together with
soil curve models for hydraulic properties.

Typical usage:
    >>> from gadopt import *
    >>> # Define soil curve
    >>> soil_params = {
    ...     'theta_r': 0.102, 'theta_s': 0.368,
    ...     'alpha': 0.0335, 'n': 2.0,
    ...     'Ks': 0.00922, 'Ss': 1e-5
    ... }
    >>> soil_curve = VanGenuchtenCurve(soil_params)
    >>> # Setup solver
    >>> richards_solver = RichardsSolver(
    ...     h, soil_curve, delta_t, DIRK22,
    ...     bcs={1: {'h': 0.0}, 2: {'flux': -0.01}}
    ... )
    >>> # Time loop
    >>> richards_solver.solve()
"""

from collections.abc import Mapping
from numbers import Number
from typing import Any
from warnings import warn

from firedrake import *

from . import richards_equation as richards_eq
from .equations import Equation
from .solver_options_manager import SolverConfigurationMixin, ConfigType
from .soil_curves import SoilCurve
from .time_stepper import IrksomeIntegrator
from .utility import DEBUG, INFO, is_continuous, log_level

__all__ = [
    "RichardsSolver",
    "direct_richards_solver_parameters",
    "iterative_richards_solver_parameters",
    "vlumping_richards_solver_parameters",
    "vlumping_hmg_richards_solver_parameters",
]


_newton_common: dict[str, Any] = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_max_it": 50,
}


direct_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1e-15,
}
"""Direct solver preset: LU factorisation via MUMPS.

G-ADOPT uses this preset by default in 2-D.
"""


iterative_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "ksp_max_it": 200,
    "ksp_gmres_restart": 30,

    # BoomerAMG via Hypre -- strong classical AMG for scalar elliptic
    # operators. Requires PETSc to be built with --download-hypre (or
    # equivalent). If Hypre is unavailable, RichardsSolver raises an
    # informative error before the solve starts.
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.7,
    "pc_hypre_boomeramg_agg_nl": 1,
    "pc_hypre_boomeramg_agg_num_paths": 2,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
    **_newton_common,
}
"""Iterative solver preset: Newton + GMRES + BoomerAMG (Hypre).

G-ADOPT uses this preset by default in 3-D for non-extruded meshes. Requires
PETSc to be built with Hypre support; otherwise an informative error is
raised at solve time.
"""


vlumping_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    # Flexible GMRES because the vertically-lumped 2-level MG is a
    # variable preconditioner (the Galerkin coarse solve is itself
    # iterative in practice).
    "ksp_type": "fgmres",
    # Inexact Newton: the 1e-4 linear tolerance consistently wins against
    # the 1e-5 baseline across the scaling studies (Cockett and
    # Murrumbidgee). Tighter linear solves don't reduce Newton counts.
    "ksp_rtol": 1e-4,
    "ksp_max_it": 200,
    "ksp_gmres_restart": 30,

    "pc_type": "python",
    "pc_python_type": "gadopt.VerticallyLumpedPC",

    # Fine-level smoother: Chebyshev wrapping block-Jacobi ILU handles
    # the stiff vertical modes at modest cost. Iteration counts are
    # insensitive to vertical resolution.
    "lumped_mg_levels_ksp_type": "chebyshev",
    "lumped_mg_levels_ksp_max_it": 2,
    "lumped_mg_levels_ksp_convergence_test": "skip",
    "lumped_mg_levels_pc_type": "bjacobi",
    "lumped_mg_levels_sub_pc_type": "ilu",
    "lumped_mg_levels_sub_pc_factor_levels": 0,

    # Coarse solve: the vertically lumped coarse operator is 2D-like
    # (~n_horiz DOFs) so MUMPS LU is cheap and robust.
    "lumped_mg_coarse_ksp_type": "preonly",
    "lumped_mg_coarse_pc_type": "lu",
    "lumped_mg_coarse_pc_factor_mat_solver_type": "mumps",

    **_newton_common,
}
"""Iterative solver preset: vertically lumped 2-level MG.

Two-level multigrid tailored for high-aspect-ratio extruded meshes
(Kramer et al. 2010). Collapses the vertical dimension onto a 2D-like
coarse problem via Galerkin projection; the coarse solve is MUMPS LU.
Inexact Newton (ksp_rtol=1e-4) is baked into the preset.

G-ADOPT selects this preset by default in 3-D on extruded Cartesian
meshes. It requires an extruded fine mesh but does not require a
MeshHierarchy.
"""


vlumping_hmg_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-4,
    "ksp_max_it": 200,
    "ksp_gmres_restart": 30,

    "pc_type": "python",
    "pc_python_type": "gadopt.VerticallyLumpedHMGPC",

    # Fine-level smoother: ASM line smoother (one patch per vertical
    # column, factorised by LU). Exact column solves damp vertical modes
    # in one sweep.
    "lumped_mg_levels_ksp_type": "chebyshev",
    "lumped_mg_levels_ksp_max_it": 1,
    "lumped_mg_levels_ksp_convergence_test": "skip",
    "lumped_mg_levels_pc_type": "python",
    "lumped_mg_levels_pc_python_type": "firedrake.ASMLinesmoothPC",
    "lumped_mg_levels_pc_linesmooth_codims": "0",
    # The outer ASM PC's prefix ends in "_sub_", so the patch-local
    # sub-PC lives under "_sub_sub_". A single sub_ downgrades ASM
    # itself to LU on the rank-local matrix -- an instant OOM.
    "lumped_mg_levels_pc_linesmooth_sub_sub_pc_type": "lu",

    # Coarse solve: geometric MG descending the 2D base MeshHierarchy.
    "lumped_mg_coarse_pc_type": "mg",
    "lumped_mg_coarse_mg_levels_ksp_type": "chebyshev",
    "lumped_mg_coarse_mg_levels_ksp_max_it": 2,
    "lumped_mg_coarse_mg_levels_pc_type": "bjacobi",
    "lumped_mg_coarse_mg_levels_sub_pc_type": "ilu",
    "lumped_mg_coarse_mg_coarse_pc_type": "lu",
    "lumped_mg_coarse_mg_coarse_pc_factor_mat_solver_type": "mumps",

    **_newton_common,
}
"""Iterative solver preset: vlumping + geometric MG on the 2D base.

Same two-level structure as ``vlumping_richards_solver_parameters``, but
the coarse solve descends a geometric multigrid on a 2D base
``MeshHierarchy`` instead of MUMPS. Fine-level smoother is
``ASMLinesmoothPC`` (exact per-column solves).

Requires the extruded fine mesh's base mesh to live in a
``MeshHierarchy`` with at least one refinement level. If that hierarchy
is absent, the solver raises ``RuntimeError`` at setup time.
"""


class RichardsSolver(SolverConfigurationMixin):
    """Advances Richards equation in time for variably saturated flow.

    The Richards equation describes water movement in variably saturated porous media:

    $$
    (S_s S + C) \\frac{\\partial h}{\\partial t} + \\nabla \\cdot (K \\nabla h) + \\nabla \\cdot (K \\nabla z) = 0
    $$

    where:
    - $h$ is the pressure head
    - $S_s$ is the specific storage coefficient (from soil curve)
    - $S(h)$ is the effective saturation
    - $C(h) = d\\theta/dh$ is the specific moisture capacity
    - $K(h)$ is the hydraulic conductivity
    - $z$ is the vertical coordinate

    **Note**: The solution field is updated in place.

    Args:
      solution:
        Firedrake function for pressure head $h$
      soil_curve:
        gadopt.SoilCurve instance defining hydraulic properties (from soil_curves module)
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator for an implicit or explicit numerical scheme
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      solver_parameters_extra:
        Dictionary of PETSc solver options used to update the default G-ADOPT options
      quad_degree:
        Integer specifying the quadrature degree. If omitted, it is set to `2p + 1`,
        where p is the polynomial degree of the trial space
      interior_penalty:
        Penalty parameter for SIPG method in DG discretizations. Default is 2.0.
        Smaller values give weaker boundary enforcement.
      nullspace:
        ``VectorSpaceBasis`` spanning the nullspace of the Jacobian. Relevant
        for pure-Neumann problems (all fluxes specified, no Dirichlet on h),
        where the constant is a true nullspace.
      transpose_nullspace:
        Nullspace of the transposed Jacobian. Usually equal to ``nullspace``
        for Richards because the advective part is small.
      near_nullspace:
        Near-nullspace basis passed to the AMG preconditioner. For scalar
        Richards the near-nullspace is the constants, so if left ``None``
        on an iterative preset ``RichardsSolver`` auto-builds it.
      timestepper_kwargs:
        Dictionary of additional keyword arguments passed to the timestepper constructor.
        Useful for parameterized schemes (e.g., {'order': 5} for IrksomeRadauIIA) or
        adaptive time-stepping (e.g., {'adaptive_parameters': {'tol': 1e-3}})

    ### Valid keys for boundary conditions
    |  Condition  |  Type  |              Description               |
    | :---------- | :----- | :------------------------------------: |
    | h           | Strong | Pressure head (Dirichlet)              |
    | flux        | Weak   | Total flux (Neumann)                   |

    Examples:
        >>> # Dirichlet BC: fixed pressure head at top
        >>> bcs = {'top': {'h': 0.0}}
        >>> # Neumann BC: prescribed flux at bottom
        >>> bcs = {'bottom': {'flux': -0.01}}
        >>> # Mixed BCs
        >>> bcs = {'top': {'h': 0.0}, 'bottom': {'flux': -0.01}}
    """

    name = "Richards"

    def __init__(
        self,
        solution: Function,
        soil_curve: SoilCurve,
        /,
        delta_t: Constant,
        timestepper: type[IrksomeIntegrator],
        *,
        bcs: dict[int | str, dict[str, Number]] = {},
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
        quad_degree: int | None = None,
        interior_penalty: float | None = None,
        nullspace: "VectorSpaceBasis | None" = None,
        transpose_nullspace: "VectorSpaceBasis | None" = None,
        near_nullspace: "VectorSpaceBasis | None" = None,
        timestepper_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.solution = solution
        self.soil_curve = soil_curve
        self.delta_t = delta_t
        self.timestepper = timestepper
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.interior_penalty = interior_penalty
        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self.near_nullspace = near_nullspace
        self.timestepper_kwargs = timestepper_kwargs or {}
        self._preset_name: str | None = None

        # Default to stage_type="value" for mass-conservative time discretisation.
        # The stage value stepper discretises Dt(theta(h)) as the exact finite
        # difference (theta(h_new) - theta(h_old))/dt rather than applying the
        # chain rule C(h)*dh/dt, which introduces systematic mass balance error.
        # Requires a stiffly accurate Butcher tableau (BackwardEuler, RadauIIA,
        # DIRK22, etc.). Adaptive time stepping is not yet supported with
        # stage_type="value" in Irksome, so we only set the default when
        # adaptive_parameters is not requested.
        if ('stage_type' not in self.timestepper_kwargs
                and 'adaptive_parameters' not in self.timestepper_kwargs):
            self.timestepper_kwargs['stage_type'] = 'value'
        elif (self.timestepper_kwargs.get('stage_type') == 'value'
                and 'adaptive_parameters' in self.timestepper_kwargs):
            raise ValueError(
                "stage_type='value' is incompatible with adaptive_parameters. "
                "Irksome does not yet support adaptive time stepping with the "
                "stage value stepper."
            )
        elif self.timestepper_kwargs.get('stage_type', None) != 'value':
            warn(
                "RichardsSolver is not using stage_type='value'. Mass will not "
                "be conserved exactly. The nonlinear mass term Dt(theta(h)) will "
                "be discretised via the chain rule C(h)*dh/dt, which introduces "
                "systematic mass balance error that accumulates over time. To "
                "enable mass conservation, use stage_type='value' with a stiffly "
                "accurate Butcher tableau (e.g. BackwardEuler, RadauIIA, DIRK22).",
                stacklevel=2,
            )

        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.test = TestFunction(self.solution_space)

        self.continuous_solution = is_continuous(self.solution)

        self.set_boundary_conditions()
        self.set_equation()
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.setup_solver()

    def set_boundary_conditions(self) -> None:
        """Sets up strong and weak boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        for bc_id, bc in self.bcs.items():
            weak_bc = {}

            for bc_type, value in bc.items():
                if bc_type == 'h':
                    # Pressure head BC
                    if self.continuous_solution:
                        # Strong Dirichlet for CG
                        strong_bc = DirichletBC(self.solution_space, value, bc_id)
                        self.strong_bcs.append(strong_bc)
                    else:
                        # Weak Dirichlet for DG (handled in equation)
                        weak_bc['h'] = value
                elif bc_type == 'flux':
                    # Flux BC (always weak)
                    weak_bc['flux'] = value
                else:
                    raise ValueError(
                        f"Unknown boundary condition type: {bc_type}. "
                        f"Valid types are 'h' (pressure head) and 'flux'."
                    )

            if weak_bc:
                self.weak_bcs[bc_id] = weak_bc

    def set_equation(self) -> None:
        """Sets up the Richards equation with all terms."""
        eq_attrs = {
            'soil_curve': self.soil_curve,
        }

        if self.interior_penalty is not None:
            eq_attrs['interior_penalty'] = self.interior_penalty

        self.equation = Equation(
            self.test,
            self.solution_space,
            residual_terms=[
                richards_eq.richards_mass_term,
                richards_eq.richards_diffusion_term,
                richards_eq.richards_gravity_term,
            ],
            eq_attrs=eq_attrs,
            bcs=self.weak_bcs,
            quad_degree=self.quad_degree,
        )

    _PRESETS: dict[str, dict[str, Any]] = {
        "direct": direct_richards_solver_parameters,
        "iterative": iterative_richards_solver_parameters,
        "vlumping": vlumping_richards_solver_parameters,
        "vlumping_hmg": vlumping_hmg_richards_solver_parameters,
    }

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None = None,
    ) -> None:
        """Sets PETSc solver parameters."""
        if isinstance(solver_preset, Mapping):
            self._preset_name = None
            self.add_to_solver_config(solver_preset)
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.setup_solver)
            return

        if solver_preset is None:
            solver_preset = self._auto_select_preset()

        if solver_preset not in self._PRESETS:
            valid = "', '".join(self._PRESETS)
            raise ValueError(
                f"Unknown solver_parameters preset '{solver_preset}'. "
                f"Valid options: '{valid}'."
            )

        self._preset_name = solver_preset
        self._validate_preset(solver_preset)
        self.add_to_solver_config(self._PRESETS[solver_preset])

        if DEBUG >= log_level:
            self.add_to_solver_config({"ksp_monitor": None, "snes_monitor": None})
        elif INFO >= log_level:
            self.add_to_solver_config({"ksp_converged_reason": None, "snes_converged_reason": None})

        self.add_to_solver_config(solver_extras)
        self.register_update_callback(self.setup_solver)

    def _auto_select_preset(self) -> str:
        """Pick a default preset based on mesh type.

        * 2-D: ``"direct"`` (MUMPS LU).
        * 3-D extruded Cartesian: ``"vlumping"``, which exploits the
          vertical anisotropy typical of water-table problems.
        * 3-D otherwise: ``"iterative"`` (BoomerAMG).
        """
        if self.mesh.topological_dimension == 2:
            return "direct"
        extruded = getattr(self.mesh, "extruded", False)
        cartesian = getattr(self.mesh, "cartesian", False)
        if extruded and cartesian:
            return "vlumping"
        return "iterative"

    def _validate_preset(self, name: str) -> None:
        """Fail fast for presets whose structural prerequisites aren't met."""
        if name == "iterative":
            # BoomerAMG needs PETSc built with Hypre.
            from firedrake.petsc import PETSc as _PETSc
            if not _PETSc.Sys.hasExternalPackage("hypre"):
                raise RuntimeError(
                    "solver_parameters='iterative' requires PETSc to be "
                    "built with Hypre (BoomerAMG). This PETSc build does "
                    "not include Hypre. Rebuild PETSc with --download-hypre "
                    "or choose a different preset (e.g. 'vlumping' on "
                    "extruded Cartesian meshes, or 'direct')."
                )

        if name in ("vlumping", "vlumping_hmg"):
            if not getattr(self.mesh, "extruded", False):
                raise RuntimeError(
                    f"solver_parameters='{name}' requires an extruded mesh. "
                    f"Use 'iterative' (BoomerAMG) or 'direct' for "
                    f"non-extruded meshes."
                )

        if name == "vlumping_hmg":
            from firedrake.mg.utils import get_level
            base = getattr(self.mesh, "_base_mesh", None)
            hierarchy, level = (None, None) if base is None else get_level(base)
            if hierarchy is None or level is None:
                raise RuntimeError(
                    "solver_parameters='vlumping_hmg' requires the extruded "
                    "mesh's base mesh to live in a MeshHierarchy so the "
                    "coarse PCMG can descend a geometric hierarchy. Build "
                    "the mesh with\n"
                    "    base_h = MeshHierarchy(base, L)\n"
                    "    mesh = ExtrudedMeshHierarchy(base_h, layers, "
                    "layer_height=h)[-1]\n"
                    "with L >= 1, or fall back to 'vlumping'."
                )

    def setup_near_nullspace(self) -> "VectorSpaceBasis":
        """Build the constant near-nullspace for AMG.

        For scalar Richards the near-nullspace is the constants -- the
        single mode the diffusion operator nearly annihilates. Providing
        it to BoomerAMG / vlumping's coarse AMG ensures the prolongation
        preserves constants, which is critical for optimal AMG convergence.
        """
        ones = Function(self.solution_space).assign(1.0)
        basis = VectorSpaceBasis([ones])
        basis.orthonormalize()
        return basis

    def setup_solver(self) -> None:
        """Sets up the timestepper using specified parameters."""
        # Auto-build the constant near-nullspace on iterative presets if
        # the user hasn't supplied one. The constant is the only non-
        # trivial near-nullspace mode for scalar Richards, so there is no
        # ambiguity in the default.
        if (self.near_nullspace is None
                and self._preset_name in ("iterative", "vlumping", "vlumping_hmg")):
            self.near_nullspace = self.setup_near_nullspace()

        if self.nullspace is not None:
            self.timestepper_kwargs.setdefault('nullspace', self.nullspace)
        if self.transpose_nullspace is not None:
            self.timestepper_kwargs.setdefault('transpose_nullspace', self.transpose_nullspace)
        if self.near_nullspace is not None:
            self.timestepper_kwargs.setdefault('near_nullspace', self.near_nullspace)

        self.ts = self.timestepper(
            self.equation,
            self.solution,
            self.delta_t,
            solver_parameters=self.solver_parameters,
            strong_bcs=self.strong_bcs,
            **self.timestepper_kwargs,
        )

    @property
    def solver(self):
        """Underlying Firedrake NonlinearVariationalSolver.

        Convenience accessor that reaches through the Irksome stepper to
        the SNES/KSP plumbing, used by tests and diagnostics that need
        to inspect the preconditioner tree after a solve.
        """
        return self.ts.stepper.solver

    def solve(self, t: float | None = None) -> tuple[float, float] | None:
        """Advances solver in time.

        Args:
          t:
            Current simulation time (optional)

        Returns:
          When adaptive time-stepping is enabled: tuple (error, dt_used) where:
              - error: Error estimate from the adaptive stepper
              - dt_used: Actual time step used (may differ from initial dt)
          When adaptive time-stepping is not enabled: None
        """
        return self.ts.advance(t)
