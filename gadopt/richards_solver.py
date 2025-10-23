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
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import DEBUG, INFO, is_continuous, log_level

__all__ = [
    "RichardsSolver",
    "direct_richards_solver_parameters",
    "iterative_richards_solver_parameters",
]

iterative_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "bcgs",
    "pc_type": "bjacobi",
    "ksp_rtol": 1e-5,
}
"""Default iterative solver parameters for solution of Richards equation.

Configured to use Newton's method with BiConjugate Gradient Stabilized (BiCGStab)
Krylov scheme and Block Jacobi preconditioning. This configuration is suitable
for 3D problems.

Note:
  G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_richards_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
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
        timestepper: type[RungeKuttaTimeIntegrator],
        *,
        bcs: dict[int | str, dict[str, Number]] = {},
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
        quad_degree: int | None = None,
    ) -> None:
        self.solution = solution
        self.soil_curve = soil_curve
        self.delta_t = delta_t
        self.timestepper = timestepper
        self.bcs = bcs
        self.quad_degree = quad_degree

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

        self.equation = Equation(
            self.test,
            self.solution_space,
            residual_terms=[
                richards_eq.richards_diffusion_term,
                richards_eq.richards_gravity_term,
            ],
            mass_term=richards_eq.richards_mass_term,
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
        )

    def solve(self, update_forcings=None, t: float | None = None) -> None:
        """Advances solver in time.

        Args:
          update_forcings:
            Optional callable to update time-dependent forcings. Called with current time.
          t:
            Current simulation time (optional)
        """
        self.ts.advance(update_forcings, t)
