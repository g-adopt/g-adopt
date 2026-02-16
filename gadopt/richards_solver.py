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
]

iterative_richards_solver_parameters: dict[str, Any] = {
    # ===========================================================================
    # Matrix type
    # ===========================================================================
    "mat_type": "aij",  # Assembled sparse matrix (required for GAMG)
    # Alternative: "matfree" - matrix-free, but requires AssembledPC wrapper for GAMG

    # ===========================================================================
    # Nonlinear solver (SNES) configuration
    # ===========================================================================
    "snes_type": "newtonls",  # Newton with line search
    # Alternative: "newtontr" - Newton trust region (more robust, slower)
    # Alternative: "ngmres" - Nonlinear GMRES (for slow convergence)

    "snes_rtol": 1e-8,  # Relative tolerance for nonlinear residual
    "snes_atol": 1e-8,  # Absolute tolerance for nonlinear residual
    # Alternative: 1e-10 for tighter convergence, 1e-6 for faster solves

    "snes_max_it": 50,  # Maximum Newton iterations
    # Alternative: 100 for very difficult problems

    # --- Line search ---
    "snes_linesearch_type": "bt",  # Backtracking (cubic interpolation)
    # Alternative: "l2" - L2 norm minimization (more robust for difficult problems)
    # Alternative: "basic" - full Newton step (faster but less robust)
    # Alternative: "cp" - critical point (for optimization-like problems)

    # ===========================================================================
    # Linear solver (KSP) configuration
    # ===========================================================================
    "ksp_type": "gmres",  # GMRES - robust for general matrices
    # Alternative: "fgmres" - flexible GMRES (if preconditioner varies)
    # Alternative: "cg" - conjugate gradient (only if Jacobian is SPD)
    # Alternative: "bcgs" - BiCGSTAB (sometimes faster, less robust)
    # Alternative: "richardson" - only as smoother, not main solver

    "ksp_rtol": 1e-5,  # Relative tolerance for linear solve
    # Alternative: 1e-3 to 1e-4 for inexact Newton (faster, may need more Newton iters)
    # Alternative: 1e-8 for very tight linear solves

    "ksp_max_it": 100,  # Maximum Krylov iterations
    # Alternative: 200-500 for difficult problems

    # --- GMRES-specific ---
    "ksp_gmres_restart": 30,  # Restart after 30 iterations
    # Alternative: 50-100 for difficult problems (more memory)
    # Alternative: 10-20 for memory-constrained problems

    # ===========================================================================
    # Preconditioner (PC) configuration - GAMG
    # ===========================================================================
    "pc_type": "gamg",  # PETSc's native algebraic multigrid
    # Alternative: "hypre" - Hypre BoomerAMG (sometimes better for 3D)
    # Alternative: "mg" - geometric multigrid (requires mesh hierarchy)
    # Alternative: "ilu" - incomplete LU (not mesh-independent, avoid!)
    # Alternative: "bjacobi" - block Jacobi (not mesh-independent, avoid!)

    # --- GAMG algorithm type ---
    "pc_gamg_type": "agg",  # Smoothed aggregation (default, good for scalar problems)
    # Alternative: "classical" - classical AMG (Ruge-Stuben style)
    # Alternative: "geo" - geometric (requires coordinates, experimental)

    # --- GAMG coarsening threshold ---
    "pc_gamg_threshold": 0.02,  # Strength of connection threshold
    # Alternative: 0.0 - keep all connections (more robust, more expensive)
    # Alternative: 0.05 - more aggressive coarsening (faster, may degrade)
    # Note: Higher values = more aggressive coarsening, fewer levels

    # --- GAMG aggregation parameters ---
    "pc_gamg_agg_nsmooths": 1,  # Smoothing steps for prolongation
    # Alternative: 0 - no smoothing (faster, less accurate)
    # Alternative: 2 - more smoothing (more robust for difficult problems)

    "pc_gamg_square_graph": 1,  # Square graph for aggressive coarsening
    # Alternative: 0 - no squaring
    # Alternative: 2 - more aggressive

    # --- GAMG coarse grid parameters ---
    "pc_gamg_coarse_eq_limit": 1000,  # Max equations on coarsest grid
    # Alternative: 500 - coarsen more aggressively
    # Alternative: 2000 - less aggressive, may be more robust

    # "pc_gamg_repartition": True,  # Repartition coarse grids (parallel only)
    # Note: Can improve parallel scaling but adds overhead

    # ===========================================================================
    # Multigrid levels configuration
    # ===========================================================================
    # --- Smoother KSP type ---
    "mg_levels_ksp_type": "chebyshev",  # Chebyshev polynomial smoother
    # Alternative: "richardson" - Richardson iteration (simpler, may need damping)
    # Alternative: "gmres" - GMRES smoother (more robust, more expensive)

    "mg_levels_ksp_max_it": 2,  # Smoother iterations per level
    # Alternative: 1 - faster but may degrade convergence
    # Alternative: 3-4 - more robust for difficult problems

    "mg_levels_ksp_convergence_test": "skip",  # Don't check convergence in smoother

    # --- Smoother PC type ---
    "mg_levels_pc_type": "jacobi",  # Point Jacobi (simple, parallel)
    # Alternative: "sor" - SOR/Gauss-Seidel (better smoothing, less parallel)
    # Alternative: "bjacobi" with "sub_pc_type": "ilu" - block Jacobi + ILU
    # Alternative: "asm" - additive Schwarz (more robust, more expensive)
    # Alternative: "pbjacobi" - point block Jacobi

    # --- Chebyshev-specific (when using chebyshev smoother) ---
    # "mg_levels_ksp_chebyshev_esteig": "0,0.1,0,1.1",  # Eigenvalue estimates
    # Note: Usually auto-computed, but can be tuned if convergence issues

    # ===========================================================================
    # Coarse grid solver configuration
    # ===========================================================================
    "mg_coarse_ksp_type": "preonly",  # Direct solve on coarse grid
    # Alternative: "gmres" - iterative on coarse (for very large coarse grids)

    "mg_coarse_pc_type": "lu",  # LU factorization on coarse grid
    # Alternative: "cholesky" - if coarse system is SPD
    # Alternative: "redundant" with "redundant_pc_type": "lu" - redundant LU (parallel)
    # Alternative: "telescope" - reduce to subset of processors

    # "mg_coarse_pc_factor_mat_solver_type": "mumps",  # Use MUMPS for coarse LU
    # Alternative: "superlu_dist" - SuperLU distributed
    # Note: Only needed for large coarse grids or parallel efficiency
}
"""Default iterative solver parameters for solution of Richards equation.

Configured to use Newton's method with GMRES Krylov solver and GAMG (algebraic
multigrid) preconditioning. This provides mesh-independent convergence for the
diffusion-dominated Richards equation.

The configuration is optimized for DIRK (diagonally implicit Runge-Kutta) time
stepping methods where each stage is solved independently.

Key features:
- GAMG with smoothed aggregation for mesh-independent convergence
- Chebyshev smoother on multigrid levels (2 iterations)
- Backtracking line search for nonlinear robustness
- Direct coarse grid solve for robustness

Note:
  G-ADOPT defaults to iterative solvers in 3-D.

References:
  - PETSc GAMG documentation: https://petsc.org/release/manualpages/PC/PCGAMG/
  - MATH-COURSE.md: Solver parameter patterns from Firedrake/Irksome demos
"""

direct_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_monitor": None,
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1e-15,
}
"""Default direct solver parameters for solution of Richards equation.

Configured to use Newton's method with LU factorisation performed via the MUMPS library.

Note:
  G-ADOPT defaults to direct solvers in 2-D.
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
        timestepper_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.solution = solution
        self.soil_curve = soil_curve
        self.delta_t = delta_t
        self.timestepper = timestepper
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.interior_penalty = interior_penalty
        self.timestepper_kwargs = timestepper_kwargs or {}

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

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None = None,
    ) -> None:
        """Sets PETSc solver parameters."""
        if isinstance(solver_preset, Mapping):
            self.add_to_solver_config(solver_preset)
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.setup_solver)
            return

        if solver_preset is not None:
            match solver_preset:
                case "direct":
                    self.add_to_solver_config(direct_richards_solver_parameters)
                case "iterative":
                    self.add_to_solver_config(iterative_richards_solver_parameters)
                case _:
                    raise ValueError("Solver type must be 'direct' or 'iterative'.")
        elif self.mesh.topological_dimension() == 2:
            self.add_to_solver_config(direct_richards_solver_parameters)
        else:
            self.add_to_solver_config(iterative_richards_solver_parameters)

        if DEBUG >= log_level:
            self.add_to_solver_config({"ksp_monitor": None, "snes_monitor": None})
        elif INFO >= log_level:
            self.add_to_solver_config({"ksp_converged_reason": None, "snes_converged_reason": None})

        self.add_to_solver_config(solver_extras)
        self.register_update_callback(self.setup_solver)

    def setup_solver(self) -> None:
        """Sets up the timestepper using specified parameters."""
        self.ts = self.timestepper(
            self.equation,
            self.solution,
            self.delta_t,
            solver_parameters=self.solver_parameters,
            strong_bcs=self.strong_bcs,
            **self.timestepper_kwargs,
        )

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
