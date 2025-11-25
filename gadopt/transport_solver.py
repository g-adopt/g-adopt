r"""This module provides a minimal solver, independent of any approximation, for generic
transport equations, which may include advection, diffusion, sink, and source terms, and
a fine-tuned solver class for the energy conservation equation. Users instantiate the
`GenericTransportSolver` or `EnergySolver` classes by providing the appropriate
documented parameters and call the `solve` method to request a solver update.

"""

import abc
from collections.abc import Mapping
from numbers import Number
from typing import Any, Callable

from firedrake import *

from . import scalar_equation as scalar_eq
from .approximations import BaseApproximation
from .equations import Equation
from .solver_options_manager import SolverConfigurationMixin, ConfigType
from .time_stepper import BackwardEuler, RungeKuttaTimeIntegrator
from .utility import DEBUG, INFO, absv, ensure_constant, is_continuous, log, log_level

__all__ = [
    "GenericTransportSolver",
    "EnergySolver",
    "DiffusiveSmoothingSolver",
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
by accessing the solver_parameter dictionary.

Examples:
    >>> energy_solver.solver_parameters['ksp_converged_reason'] = None
    >>> energy_solver.solver_parameters['ksp_rtol'] = 1e-4

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


class GenericTransportBase(SolverConfigurationMixin, abc.ABC):
    """Base class for advancing a generic transport equation in time.

    All combinations of advection, diffusion, sink, and source terms are handled.

    **Note**: The solution field is updated in place.

    Args:
      solution:
        Firedrake function for the field of interest
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      solution_old:
        Firedrake function holding the solution field at the previous time step
      eq_attrs:
        Dictionary of terms arguments and their values
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      solver_parameters_extra:
        Dictionary of PETSc solver options used to update the default G-ADOPT options
      timestepper_kwargs:
        Dictionary of additional keyword arguments passed to the timestepper constructor.
        Useful for parameterized schemes (e.g., {'order': 5} for IrksomeRadauIIA)
      su_advection:
        Boolean activating the streamline-upwind stabilisation scheme when using
        continuous finite elements

    """

    terms_mapping = {
        "advection": scalar_eq.advection_term,
        "diffusion": scalar_eq.diffusion_term,
        "sink": scalar_eq.sink_term,
        "source": scalar_eq.source_term,
    }

    def __init__(
        self,
        solution: Function,
        /,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        *,
        solution_old: Function | None = None,
        eq_attrs: dict[str, float] = {},
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
        timestepper_kwargs: dict[str, Any] | None = None,
        su_advection: bool = False,
    ) -> None:
        self.solution = solution
        self.delta_t = delta_t
        self.timestepper = timestepper
        self.timestepper_kwargs = timestepper_kwargs or {}
        self.solution_old = solution_old or Function(solution)
        self.eq_attrs = eq_attrs
        self.bcs = bcs
        self.su_advection = su_advection

        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.test = TestFunction(self.solution_space)

        self.continuous_solution = is_continuous(self.solution)

        self.set_boundary_conditions()
        self.set_su_nubar()
        self.set_equation()
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.setup_solver()

    def set_boundary_conditions(self) -> None:
        """Sets up boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        for bc_id, bc in self.bcs.items():
            weak_bc = {}

            for bc_type, value in bc.items():
                if bc_type == self.strong_bcs_tag:
                    if self.continuous_solution:
                        strong_bc = DirichletBC(self.solution_space, value, bc_id)
                        self.strong_bcs.append(strong_bc)
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[bc_type] = value

            self.weak_bcs[bc_id] = weak_bc

    def set_su_nubar(self) -> None:
        """Sets up the advection streamline-upwind scheme (Donea & Huerta, 2003).

        Columns of Jacobian J are the vectors that span the quad/hex and can be seen as
        unit vectors scaled with the dx/dy/dz in that direction (assuming physical
        coordinates x, y, z aligned with local coordinates).
        Thus u^T J is (dx * u , dy * v). Following (2.44c), Pe = u^T J / 2 kappa, and
        beta(Pe) is the xibar vector in (2.44a). Finally, we get the artificial
        diffusion nubar from (2.49).

        Donea, J., & Huerta, A. (2003).
        Finite element methods for flow problems.
        John Wiley & Sons.
        """
        if not self.su_advection:
            return

        if (u := getattr(self, "u", self.eq_attrs.get("u"))) is None:
            raise ValueError(
                "'u' must be included into `eq_attrs` if `su_advection` is given."
            )

        if not self.continuous_solution:
            raise TypeError("SU advection requires a continuous function space.")

        log("Using SU advection")

        J = Function(TensorFunctionSpace(self.mesh, "DQ", 1), name="Jacobian")
        J.interpolate(Jacobian(self.mesh))
        # Calculate grid Peclet number. Note the use of a lower bound for diffusivity if
        # a pure advection scenario is considered.
        kappa = self.eq_attrs.get("diffusivity", 0.0)
        Pe = absv(dot(u, J)) / 2 / (kappa + 1e-12)
        beta_Pe = as_vector([1 / tanh(Pe_i + 1e-6) - 1 / (Pe_i + 1e-6) for Pe_i in Pe])
        nubar = dot(absv(dot(u, J)), beta_Pe) / 2  # Calculate SU artificial diffusion

        self.eq_attrs["su_nubar"] = nubar

    @abc.abstractmethod
    def set_equation(self):
        """Sets up the term contributions in the equation."""
        raise NotImplementedError

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
                    self.add_to_solver_config(direct_energy_solver_parameters)
                case "iterative":
                    self.add_to_solver_config(iterative_energy_solver_parameters)
                case _:
                    raise ValueError("Solver type must be 'direct' or 'iterative'.")
        elif self.mesh.topological_dimension == 2:
            self.add_to_solver_config(direct_energy_solver_parameters)
        else:
            self.add_to_solver_config(iterative_energy_solver_parameters)

        if DEBUG >= log_level:
            self.add_to_solver_config({"ksp_monitor": None})
        elif INFO >= log_level:
            self.add_to_solver_config({"ksp_converged_reason": None})

        self.add_to_solver_config(solver_extras)
        self.register_update_callback(self.setup_solver)

    def setup_solver(self) -> None:
        """Sets up the timestepper using specified parameters."""
        self.ts = self.timestepper(
            self.equation,
            self.solution,
            self.delta_t,
            solution_old=self.solution_old,
            solver_parameters=self.solver_parameters,
            strong_bcs=self.strong_bcs,
            **self.timestepper_kwargs,
        )

    def solver_callback(self) -> None:
        """Optional instructions to execute right after a solve."""
        pass

    def solve(self, t: float | None = None) -> None:
        """Advances solver in time."""
        self.ts.advance(t=t)

        self.solver_callback()


class GenericTransportSolver(GenericTransportBase):
    """Advances in time a generic transport equation.

    **Note**: The solution field is updated in place.

    Terms and Attributes:
        This solver handles all combinations of advection, diffusion, sink, and source
        terms. Depending on the included terms, specific attributes must be provided
        according to:

        |   Term    | Required attribute(s) |           Optional attribute(s)           |
        | --------- | --------------------- | ----------------------------------------- |
        | advection | u                     | advective_velocity_scaling, su_nubar      |
        | diffusion | diffusivity           | reference_for_diffusion, interior_penalty |
        | source    | source                |                                           |
        | sink      | sink_coeff            |                                           |

    Args:
      terms:
        List of equation terms to include (a string for a single term is accepted)
      solution:
        Firedrake function for the field of interest
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      solution_old:
        Firedrake function holding the solution field at the previous time step
      eq_attrs:
        Dictionary of terms arguments and their values
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      timestepper_kwargs:
        Dictionary of additional keyword arguments passed to the timestepper constructor.
        Useful for parameterized schemes (e.g., {'order': 5} for IrksomeRadauIIA)
      su_advection:
        Boolean activating the streamline-upwind stabilisation scheme when using
        continuous finite elements

    """

    strong_bcs_tag = "g"

    def __init__(
        self,
        terms: str | list[str],
        solution: Function,
        /,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        **kwargs,
    ) -> None:
        self.terms = [terms] if isinstance(terms, str) else terms

        super().__init__(solution, delta_t, timestepper, **kwargs)

    def set_equation(self) -> None:
        self.equation = Equation(
            self.test,
            self.solution_space,
            residual_terms=[self.terms_mapping[term] for term in self.terms],
            mass_term=scalar_eq.mass_term,
            eq_attrs=self.eq_attrs,
            bcs=self.weak_bcs,
        )


class EnergySolver(GenericTransportBase):
    """Advances in time the energy conservation equation.

    **Note**: The solution field is updated in place.

    Args:
      solution:
        Firedrake function for temperature
      u:
        Firedrake function for velocity
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      delta_t:
        Simulation time step
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      solution_old:
        Firedrake function holding the solution field at the previous time step
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      timestepper_kwargs:
        Dictionary of additional keyword arguments passed to the timestepper constructor.
        Useful for parameterized schemes (e.g., {'order': 5} for IrksomeRadauIIA)
      su_advection:
        Boolean activating the streamline-upwind stabilisation scheme when using
        continuous finite elements

    """

    strong_bcs_tag = "T"

    def __init__(
        self,
        solution: Function,
        u: Function,
        approximation: BaseApproximation,
        /,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        **kwargs,
    ) -> None:
        self.u = u
        self.approximation = approximation

        super().__init__(solution, delta_t, timestepper, **kwargs)

        self.T_old = self.solution_old

    def set_equation(self) -> None:
        rho_cp = self.approximation.rhocp()

        self.eq_attrs |= {
            "advective_velocity_scaling": rho_cp,
            "diffusivity": self.approximation.kappa(),
            "reference_for_diffusion": self.approximation.Tbar,
            "sink_coeff": self.approximation.linearized_energy_sink(self.u),
            "source": self.approximation.energy_source(self.u),
            "u": self.u,
        }

        self.equation = Equation(
            self.test,
            self.solution_space,
            residual_terms=self.terms_mapping.values(),
            mass_term=lambda eq, trial: scalar_eq.mass_term(eq, rho_cp * trial),
            eq_attrs=self.eq_attrs,
            approximation=self.approximation,
            bcs=self.weak_bcs,
        )


class DiffusiveSmoothingSolver(GenericTransportSolver):
    """A class to perform diffusive smoothing by inheriting from GenericTransportSolver.

    This class provides functionality to solve a diffusion equation for smoothing
    a scalar function, using clean inheritance from GenericTransportSolver.

    Args:
        solution (firedrake.Function): The solution function to store the smoothed result.
        wavelength (Number): The wavelength for diffusion.
        K (Union[firedrake.Function, Number], optional): Diffusion tensor. Defaults to 1.
        bcs (dict[int, dict[str, int | float]] | None): Boundary conditions to impose on the solution.
        solver_parameters (dict[str, str | float] | None): Solver parameters for the solver. Defaults to None.
        integration_quad_degree (int | None): Quadrature degree for integrating the diffusion tensor. If None, defaults to 2p+1 where p is the polynomial degree.
        **kwargs: Additional keyword arguments to pass to the solver.
    """

    def __init__(
        self,
        solution: Function,
        wavelength: Number,
        K: Function | Number = 1,
        bcs: dict[int, dict[str, int | float]] | None = None,
        solver_parameters: dict[str, str | float] | None = None,
        integration_quad_degree: int | None = None,
        **kwargs
    ):
        # Extract function space from solution
        function_space = solution.function_space()

        # Calculate diffusive time step
        dt = self._calculate_diffusive_time_step(function_space, wavelength, K, integration_quad_degree)

        # Initialise the parent GenericTransportSolver
        super().__init__(
            "diffusion",
            solution,
            dt,
            BackwardEuler,
            eq_attrs={"diffusivity": ensure_constant(K)},
            bcs=bcs,
            solver_parameters=solver_parameters,
            **kwargs
        )

    def _calculate_diffusive_time_step(
        self,
        function_space: FunctionSpace,
        wavelength: Number,
        K: Function | Number,
        integration_quad_degree: int | None
    ) -> Constant:
        """Calculate the diffusive time step based on wavelength and diffusivity."""
        mesh = function_space.mesh()

        # Determine quadrature degree for tensor integration
        if integration_quad_degree is None:
            p = function_space.ufl_element().degree()
            if not isinstance(p, int):  # Tensor-product element
                p = max(p)
            integration_quad_degree = 2 * p + 1

        # For anisotropic diffusion, use average diffusivity
        if hasattr(K, 'ufl_shape') and len(K.ufl_shape) == 2:
            # Tensor diffusivity (2D tensor, e.g., (2,2) or (3,3))
            K_avg = (
                assemble(sqrt(inner(K, K)) * dx(mesh, degree=integration_quad_degree)) /
                assemble(Constant(1) * dx(mesh, degree=integration_quad_degree))
            )
        else:
            # Scalar diffusivity (Number, Constant, or scalar Function)
            K_avg = K

        return Constant(wavelength**2 / (4 * K_avg))

    def action(self, field: Function) -> None:
        """Apply smoothing action to an input field.

        Args:
            field (firedrake.Function): The input field to be smoothed.

        Note:
            The smoothed result is stored in the solution function passed to the constructor.
        """
        # Start with the input field
        self.solution.assign(field)

        # Solve the diffusion equation (inherited from GenericTransportSolver)
        self.solve()
