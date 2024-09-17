r"""This module provides a fine-tuned solver class for the energy conservation equation.
Users instantiate the `EnergySolver` class by providing relevant parameters and call
the `solve` method to request a solver update.

"""

from numbers import Number
from typing import Any, Callable, Optional

import firedrake as fd

from .approximations import EquationSystem
from .equations import Equation
from .scalar_equation import mass_term_energy
from .scalar_equation import residual_terms_advection_diffusion as terms_energy
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import DEBUG, INFO, absv, is_continuous, log, log_level, su_nubar

__all__ = [
    "iterative_energy_solver_parameters",
    "direct_energy_solver_parameters",
    "EnergySolver",
]

iterative_energy_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}
"""Default iterative solver parameters for solution of energy equation.

Configured to use the GMRES Krylov scheme with Successive Over Relaxation (SOR)
preconditioning. Note that default energy solver parameters can be augmented or adjusted
by accessing the solver_parameter dictionary, for example:
energy_solver.solver_parameters['ksp_converged_reason'] = None
energy_solver.solver_parameters['ksp_rtol'] = 1e-4

Note:
  G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_energy_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of energy equation.

Configured to use LU factorisation performed via the MUMPS library.

Note:
  G-ADOPT defaults to direct solvers in 2-D.
"""


class EnergySolver:
    """Timestepper and solver for the energy equation.

    The temperature, T, is updated in place.

    Arguments:
      approximation:
        G-ADOPT base approximation describing the system of equations
      T:
        Firedrake function for temperature
      u:
        Firedrake function for velocity
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      bcs:
        Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters:
        Solver parameters provided to PETSc
      su_advection:
        Boolean indicating whether or not to use the streamline-upwind stabilisation scheme

    """

    def __init__(
        self,
        approximation: EquationSystem,
        T: fd.Function,
        u: fd.Function,
        delta_t: fd.Constant,
        timestepper: RungeKuttaTimeIntegrator,
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        su_advection: bool = False,
    ) -> None:
        self.solution = T
        self.delta_t = delta_t
        self.timestepper = timestepper
        self.bcs = bcs

        self.Q = T.function_space()
        self.mesh = self.Q.mesh()
        self.solution_old = fd.Function(self.Q)

        self.set_boundary_conditions()

        if isinstance(solver_parameters, dict):
            self.solver_parameters = solver_parameters
        else:
            self.set_solver_parameters(solver_parameters)

        terms_kwargs = {
            "absorption_coefficient": approximation.linearized_energy_sink(u),
            "advective_velocity_scaling": approximation.rho * approximation.cp,
            "reference_for_conduction": approximation.T,
            "source": approximation.energy_source(u),
            "u": u,
        }

        if su_advection:
            if not is_continuous(self.Q):
                raise TypeError("SU advection requires a continuous function space.")

            log("Using SU advection")
            # SU(PG) à la Donea & Huerta (2003)
            # Columns of Jacobian J are the vectors that span the quad/hex and can be
            # seen as unit vectors scaled with the dx/dy/dz in that direction (assuming
            # physical coordinates x, y, z aligned with local coordinates).
            # Thus u^T J is (dx * u , dy * v), and following (2.44c), Pe = u^T J / 2 kappa.
            # beta(Pe) is the xibar vector in (2.44a)
            # then we get artifical viscosity nubar from (2.49)

            J = fd.Function(
                fd.TensorFunctionSpace(self.mesh, "DQ", 1), name="Jacobian"
            ).interpolate(fd.Jacobian(self.mesh))
            # Set lower bound for diffusivity in case zero diffusivity specified for
            # pure advection.
            kappa = approximation.kappa + 1e-12
            Pe = absv(fd.dot(u, J)) / 2 / kappa  # Calculate grid Peclet number
            nubar = su_nubar(u, J, Pe)  # Calculate SU artificial diffusion

            terms_kwargs["su_nubar"] = nubar

        self.equation = Equation(
            fd.TestFunction(self.Q),
            self.Q,
            terms_energy,
            mass_term=mass_term_energy,
            terms_kwargs=terms_kwargs,
            approximation=approximation,
            bcs=self.weak_bcs,
        )

        # Solver object is set up later to permit editing default solver parameters
        # specified above.
        self._solver_setup = False

    def set_boundary_conditions(self) -> None:
        """Sets up boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        apply_strongly = is_continuous(self.solution)

        for bc_id, bc in self.bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == "T":
                    if apply_strongly:
                        self.strong_bcs.append(fd.DirichletBC(self.Q, value, bc_id))
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[bc_id] = weak_bc

    def set_solver_parameters(self, solver_parameters) -> None:
        """Sets PETSc solver parameters."""
        if isinstance(solver_parameters, str):
            match solver_parameters:
                case "direct":
                    self.solver_parameters = direct_energy_solver_parameters.copy()
                case "iterative":
                    self.solver_parameters = iterative_energy_solver_parameters.copy()
                case _:
                    raise ValueError(
                        f"Solver type '{solver_parameters}' not implemented."
                    )
        elif self.mesh.topological_dimension() == 2:
            self.solver_parameters = direct_energy_solver_parameters.copy()
        else:
            self.solver_parameters = iterative_energy_solver_parameters.copy()

        if DEBUG >= log_level:
            self.solver_parameters["ksp_monitor"] = None
        elif INFO >= log_level:
            self.solver_parameters["ksp_converged_reason"] = None

    def setup_solver(self) -> None:
        """Sets up timestepper and associated solver, using specified solver parameters"""
        self.ts = self.timestepper(
            self.equation,
            self.solution,
            self.delta_t,
            solution_old=self.solution_old,
            solver_parameters=self.solver_parameters,
            strong_bcs=self.strong_bcs,
        )
        self._solver_setup = True

    def solve(self, t: Number = 0, update_forcings: Optional[Callable] = None) -> None:
        """Advances solver in time."""
        if not self._solver_setup:
            self.setup_solver()

        self.ts.advance(t, update_forcings)
