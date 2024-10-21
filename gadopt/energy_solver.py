r"""This module provides a minimal solver for general scalar equations including
advection and diffusion terms and a fine-tuned solver class for the energy conservation
equation. Users instantiate the `EnergySolver` class by providing relevant parameters
and call the `solve` method to request a solver update.

"""

import abc
from numbers import Number
from typing import Any, Callable, Optional

import firedrake as fd

from . import scalar_equation as scal_eq
from .approximations import BaseApproximation
from .equations import Equation
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import DEBUG, INFO, absv, is_continuous, log, log_level, su_nubar

__all__ = [
    "GenericTransportSolver",
    "EnergySolver",
    "direct_energy_solver_parameters",
    "iterative_energy_solver_parameters",
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


class GenericTransportBase(abc.ABC):
    """Timestepper and solver for an equation involving advection and diffusion terms.

    All combinations of advection, diffusion, sink, and source terms are handled.

    Note: The scalar field is updated in place.

    Arguments:
      scalar_field:
        Firedrake function for scalar field of interest
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      solution_old:
        Firedrake function holding the previous solution
      bcs:
        Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters:
        Dicitionary of solver parameters or a string specifying a default configuration
        provided to PETSc

    """

    terms_mapping = {
        "advection": scal_eq.scalar_advection_term,
        "diffusion": scal_eq.scalar_diffusion_term,
        "sink": scal_eq.scalar_absorption_term,
        "source": scal_eq.scalar_source_term,
    }

    def __init__(
        self,
        scalar_field: fd.Function,
        delta_t: fd.Constant,
        timestepper: RungeKuttaTimeIntegrator,
        *,
        solution_old: Optional[fd.Function] = None,
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
    ) -> None:
        self.solution = scalar_field
        self.delta_t = delta_t
        self.timestepper = timestepper

        self.Q = scalar_field.function_space()
        self.mesh = self.Q.mesh()
        self.solution_old = solution_old or fd.Function(self.Q)

        self.set_boundary_conditions(bcs)

        if isinstance(solver_parameters, dict):
            self.solver_parameters = solver_parameters
        else:
            self.set_solver_parameters(solver_parameters)

        # Solver object is set up later to permit editing default solver parameters.
        self._solver_setup = False

    def set_boundary_conditions(self, bcs: dict[int, dict[str, Number]]) -> None:
        """Sets up boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        apply_strongly = is_continuous(self.solution)

        for bc_id, bc in bcs.items():
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

    def su_nubar(self, u: fd.Function, su_diffusivity: float) -> fd.ufl.algebra.Product:
        """Sets up the streamline-upwind scheme."""
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
        kappa = su_diffusivity + 1e-12
        Pe = absv(fd.dot(u, J)) / 2 / kappa  # Calculate grid Peclet number
        nubar = su_nubar(u, J, Pe)  # Calculate SU artificial diffusion

        return nubar

    @abc.abstractmethod
    def set_equation(self):
        """Sets up the equation."""
        raise NotImplementedError

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


class GenericTransportSolver(GenericTransportBase):
    """Timestepper and solver for an advection-diffusion equation.

    All combinations of advection, diffusion, sink, and source terms are valid.

    The scalar field is updated in place.

    Arguments:
      terms:
        List of equation terms defined in scalar_equation.py
      scalar_field:
        Firedrake function for scalar field of interest
      u:
        Firedrake function for velocity
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      terms_kwargs:
        Dicitionary of terms arguments and their values
      solution_old:
        Firedrake function holding the previous solution
      bcs:
        Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters:
        Dicitionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      su_diffusivity:
        Float activating the streamline-upwind stabilisation scheme and specifying the
        corresponding diffusivity

    """

    _terms_kwargs = ["diffusivity", "sink", "source"]

    def __init__(
        self,
        terms: str | list[str],
        scalar_field: fd.Function,
        u: fd.Function,
        delta_t: fd.Constant,
        timestepper: RungeKuttaTimeIntegrator,
        *,
        terms_kwargs: Optional[dict[str, float]] = None,
        solution_old: Optional[fd.Function] = None,
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        su_diffusivity: Optional[float] = None,
    ) -> None:
        super().__init__(
            scalar_field,
            delta_t,
            timestepper,
            solution_old=solution_old,
            bcs=bcs,
            solver_parameters=solver_parameters,
        )

        if isinstance(terms, str):
            terms = [terms]

        if terms_kwargs is None:
            terms_kwargs = {}
        assert all(term_kwarg in self._terms_kwargs for term_kwarg in terms_kwargs)

        self.set_equation(terms, u, terms_kwargs, su_diffusivity)

    def set_equation(
        self,
        terms: list[str],
        u: fd.Function,
        terms_kwargs: dict[str, float],
        su_diffusivity: Optional[float],
    ) -> None:
        terms_kwargs["u"] = u

        if su_diffusivity is not None:
            terms_kwargs["su_nubar"] = self.su_nubar(u, su_diffusivity)

        eq_terms = [self.terms_mapping[term] for term in terms]

        self.equation = Equation(
            fd.TestFunction(self.Q),
            self.Q,
            eq_terms,
            mass_term=scal_eq.mass_term,
            terms_kwargs=terms_kwargs,
            bcs=self.weak_bcs,
        )


class EnergySolver(GenericTransportBase):
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
      solution_old:
        Firedrake function holding the previous solution
      bcs:
        Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters:
        Dicitionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      su_diffusivity:
        Boolean indicating whether or not to use the streamline-upwind stabilisation scheme

    """

    def __init__(
        self,
        T: fd.Function,
        u: fd.Function,
        approximation: BaseApproximation,
        delta_t: fd.Constant,
        timestepper: RungeKuttaTimeIntegrator,
        *,
        solution_old: Optional[fd.Function] = None,
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        su_diffusivity: Optional[float] = None,
    ) -> None:
        super().__init__(
            T,
            delta_t,
            timestepper,
            solution_old=solution_old,
            bcs=bcs,
            solver_parameters=solver_parameters,
        )

        self.set_equation(self.terms_mapping.keys(), approximation, u, su_diffusivity)

    def set_equation(
        self,
        terms: list[str],
        approximation: BaseApproximation,
        u: fd.Function,
        su_diffusivity: Optional[float],
    ) -> None:
        rho_cp = approximation.rhocp()

        terms_kwargs = {
            "advective_velocity_scaling": rho_cp,
            "diffusivity": approximation.kappa(),
            "reference_for_diffusion": approximation.Tbar,
            "sink": approximation.linearized_energy_sink(u),
            "source": approximation.energy_source(u),
            "u": u,
        }

        if su_diffusivity is not None:
            terms_kwargs["su_nubar"] = self.su_nubar(u, su_diffusivity)

        eq_terms = [self.terms_mapping[term] for term in terms]

        self.equation = Equation(
            fd.TestFunction(self.Q),
            self.Q,
            eq_terms,
            mass_term=lambda eq, trial: scal_eq.mass_term(eq, rho_cp * trial),
            terms_kwargs=terms_kwargs,
            approximation=approximation,
            bcs=self.weak_bcs,
        )
