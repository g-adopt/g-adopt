from numbers import Number
from typing import Any, Optional

from firedrake import Constant, DirichletBC, Function, Jacobian, TensorFunctionSpace, dot

from .approximations import BaseApproximation
from .scalar_equation import EnergyEquation
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import is_continuous, ensure_constant
from .utility import log_level, INFO, DEBUG, log
from .utility import absv, su_nubar

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
"""Default iterative solver parameters for solution of energy equation. Configured to use the GMRES Krylov scheme
   with Successive Over Relaxation (SOR) preconditioning. Note that default energy solver parameters
   can be augmented or adjusted by accessing the solver_parameter dictionary, for example:
   energy_solver.solver_parameters['ksp_converged_reason'] = None
   energy_solver.solver_parameters['ksp_rtol'] = 1e-4
   G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_energy_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of energy equation. Configured to use LU factorisation,
   using the MUMPS library. G-ADOPT defaults to direct solvers in 2-D.
"""


class EnergySolver:
    """Timestepper and solver for the energy equation. The temperature, T, is updated in place.

    Arguments:
      T:                 Firedrake function for temperature
      u:                 Firedrake function for velocity
      approximation:     G-ADOPT base approximation describing the system of equations
      delta_t:           Simulation time step
      timestepper:       Runge-Kutta time integrator implementing an explicit or implicit numerical scheme
      bcs:               Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters: Solver parameters provided to PETSc
      su_advection:      Boolean specifying whether or not to use the streamline-upwind stabilisation scheme

    """

    def __init__(
        self,
        T: Function,
        u: Function,
        approximation: BaseApproximation,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        bcs: Optional[dict[int, dict[str, Number]]] = None,
        solver_parameters: Optional[dict[str, Any]] = None,
        su_advection: bool = False,
    ):
        self.T = T
        self.Q = T.function_space()
        self.mesh = self.Q.mesh()
        self.delta_t = delta_t
        rhocp = approximation.rhocp()
        self.eq = EnergyEquation(self.Q, self.Q, rhocp=rhocp)
        self.fields = {
            'diffusivity': ensure_constant(approximation.kappa()),
            'reference_for_diffusion': approximation.Tbar,
            'source': approximation.energy_source(u),
            'velocity': u,
            'advective_velocity_scaling': rhocp
        }
        sink = approximation.linearized_energy_sink(u)
        if sink:
            self.fields['absorption_coefficient'] = sink

        if su_advection:
            if not is_continuous(self.Q):
                raise TypeError("SU advection requires a continuous function space.")

            log("Using SU advection")
            # SU(PG) ala Donea & Huerta:
            # Columns of Jacobian J are the vectors that span the quad/hex
            # which can be seen as unit-vectors scaled with the dx/dy/dz in that direction (assuming physical coordinates x,y,z aligned with local coordinates)
            # thus u^T J is (dx * u , dy * v)
            # and following (2.44c) Pe = u^T J / 2 kappa
            # beta(Pe) is the xibar vector in (2.44a)
            # then we get artifical viscosity nubar from (2.49)

            J = Function(TensorFunctionSpace(self.mesh, 'DQ', 1), name='Jacobian').interpolate(Jacobian(self.mesh))
            kappa = self.fields['diffusivity'] + 1e-12  # Set lower bound for diffusivity in case zero diffusivity specified for pure advection.
            vel = self.fields['velocity']
            Pe = absv(dot(vel, J)) / (2*kappa)  # Calculate grid peclet number
            nubar = su_nubar(vel, J, Pe)  # Calculate SU artifical diffusion

            self.fields['su_nubar'] = nubar

        if solver_parameters is None:
            if self.mesh.topological_dimension() == 2:
                self.solver_parameters = direct_energy_solver_parameters.copy()
                if INFO >= log_level:
                    # not really "informative", but at least we get a 1-line message saying we've passed the energy solve
                    self.solver_parameters['ksp_converged_reason'] = None
            else:
                self.solver_parameters = iterative_energy_solver_parameters.copy()
                if DEBUG >= log_level:
                    self.solver_parameters['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['ksp_converged_reason'] = None
        else:
            self.solver_parameters = solver_parameters
        apply_strongly = is_continuous(T)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == 'T':
                    if apply_strongly:
                        self.strong_bcs.append(DirichletBC(self.Q, value, id))
                    else:
                        weak_bc['q'] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.timestepper = timestepper
        self.T_old = Function(self.Q)
        # solver is setup only at the end, so users
        # can overwrite or augment default parameters specified above
        self._solver_setup = False

    def setup_solver(self):
        """Sets up timestepper and associated solver, using specified solver parameters"""
        self.ts = self.timestepper(self.eq, self.T, self.fields, self.delta_t,
                                   bnd_conditions=self.weak_bcs, solution_old=self.T_old,
                                   strong_bcs=self.strong_bcs,
                                   solver_parameters=self.solver_parameters)
        self._solver_setup = True

    def solve(self, t=0, update_forcings=None):
        """Advances solver in time."""
        if not self._solver_setup:
            self.setup_solver()
        self.ts.advance(t, update_forcings)
