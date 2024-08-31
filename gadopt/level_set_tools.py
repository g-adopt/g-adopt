r"""This module provides a set of classes and functions enabling multi-material
capabilities. Users initialise materials by instantiating the `Material` class and
define the physical properties of material interfaces using `field_interface`. They
instantiate the `LevelSetSolver` class by providing relevant parameters and call the
`solve` method to request a solver update. Finally, they may call the `entrainment`
function to calculate material entrainment in the simulation.

"""

from numbers import Number
from typing import Optional

import firedrake as fd
from firedrake.ufl_expr import extract_unique_domain

from .equations import BaseEquation, BaseTerm
from .scalar_equation import ScalarAdvectionEquation
from .time_stepper import RungeKuttaTimeIntegrator, eSSPRKs3p3

__all__ = ["LevelSetSolver", "entrainment", "material_field"]


# Default solver options for level-set advection and reinitialisation
solver_params_default = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu",
}
# Default parameters used to set up level-set reinitialisation
reini_params_default = {
    "tstep": 5e-2,
    "tstep_alg": eSSPRKs3p3,
    "frequency": 5,
    "iterations": 1,
}


class ReinitialisationTerm(BaseTerm):
    """Term for the conservative level set reinitialisation equation.

    Implements terms on the right-hand side of Equation 17 from
    Parameswaran, S., & Mandal, J. C. (2023).
    A stable interface-preserving reinitialization equation for conservative level set
    method.
    European Journal of Mechanics-B/Fluids, 98, 40-63.
    """

    def residual(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument,
        trial_lagged: fd.ufl_expr.Argument,
        fields: dict,
        bcs: dict,
    ) -> fd.ufl.core.expr.Expr:
        """Residual contribution expressed through UFL.

        Args:
            test:
              UFL test function.
            trial:
              UFL trial function.
            trial_lagged:
              UFL trial function from the previous time step.
            fields:
              A dictionary of provided UFL expressions.
            bcs:
              A dictionary of boundary conditions.
        """
        level_set_grad = fields["level_set_grad"]
        epsilon = fields["epsilon"]

        sharpen_term = -trial * (1 - trial) * (1 - 2 * trial) * test * self.dx
        balance_term = (
            epsilon
            * (1 - 2 * trial)
            * fd.sqrt(level_set_grad[0] ** 2 + level_set_grad[1] ** 2)
            * test
            * self.dx
        )

        return sharpen_term + balance_term


class ReinitialisationEquation(BaseEquation):
    """Equation for conservative level set reinitialisation.

    Attributes:
        terms:
          A list of equation terms that contribute to the system's residual.
    """

    terms = [ReinitialisationTerm]


class LevelSetSolver:
    """Solver for the conservative level-set approach.

    Solves the advection and reinitialisation equations for a level set function.

    Attributes:
        mesh:
          The UFL mesh where values of the level set function exist.
        level_set:
          The Firedrake function for the level set.
        func_space_lsgp:
          The UFL function space where values of the projected level-set gradient are
          calculated.
        level_set_grad_proj:
          The Firedrake function for the projected level-set gradient.
        proj_solver:
          An integer or a float representing the reference density.
        reini_params:
          A dictionary containing parameters used in the reinitialisation approach.
        ls_ts:
          The G-ADOPT timestepper object for the advection equation.
        reini_ts:
          The G-ADOPT timestepper object for the reinitialisation equation.
        subcycles:
          An integer specifying the number of advection solves to perform.
    """

    def __init__(
        self,
        level_set: fd.Function,
        velocity: fd.ufl.tensors.ListTensor,
        tstep: fd.Constant,
        tstep_alg: type[RungeKuttaTimeIntegrator],
        epsilon: fd.Constant,
        subcycles: int = 1,
        reini_params: Optional[dict] = None,
        solver_params: Optional[dict] = None,
    ):
        """Initialises the solver instance.

        Args:
            level_set:
              The Firedrake function for the level set.
            velocity:
              The UFL expression for the velocity.
            tstep:
              The Firedrake function over the Real space for the simulation time step.
            tstep_alg:
              The class for the timestepping algorithm used in the advection solver.
            subcycles:
              An integer specifying the number of advection solves to perform.
            epsilon:
              A UFL constant denoting the thickness of the hyperbolic tangent profile.
            reini_params:
              A dictionary containing parameters used in the reinitialisation approach.
            solver_params:
              A dictionary containing solver parameters used in the advection and
              reinitialisation approaches.
        """
        self.level_set = level_set
        self.tstep = tstep
        self.tstep_alg = tstep_alg
        self.subcycles = subcycles

        self.reini_params = reini_params or reini_params_default
        self.solver_params = solver_params or solver_params_default

        self.mesh = extract_unique_domain(level_set)
        self.func_space = level_set.function_space()

        self.func_space_lsgp = fd.VectorFunctionSpace(
            self.mesh, "CG", self.func_space.finat_element.degree
        )
        self.level_set_grad_proj = fd.Function(
            self.func_space_lsgp,
            name=f"Level-set gradient (projection) #{level_set.name()[-1]}",
        )

        self.proj_solver = self.gradient_L2_proj()

        self.ls_fields = {"velocity": velocity}
        self.reini_fields = {
            "level_set_grad": self.level_set_grad_proj,
            "epsilon": epsilon,
        }

        self.solvers_ready = False

    def gradient_L2_proj(self) -> fd.variational_solver.LinearVariationalSolver:
        """Constructs a projection solver.

        Projects the level-set gradient from a discontinuous function space to the
        equivalent continuous one.

        Returns:
            A Firedrake solver capable of projecting a discontinuous gradient field on
            a continuous function space.
        """
        test_function = fd.TestFunction(self.func_space_lsgp)
        trial_function = fd.TrialFunction(self.func_space_lsgp)

        mass_term = fd.inner(test_function, trial_function) * fd.dx(domain=self.mesh)
        residual_element = (
            self.level_set * fd.div(test_function) * fd.dx(domain=self.mesh)
        )
        residual_boundary = (
            self.level_set
            * fd.dot(test_function, fd.FacetNormal(self.mesh))
            * fd.ds(domain=self.mesh)
        )
        residual = -residual_element + residual_boundary
        problem = fd.LinearVariationalProblem(
            mass_term, residual, self.level_set_grad_proj
        )

        solver_params = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        return fd.LinearVariationalSolver(problem, solver_parameters=solver_params)

    def update_level_set_gradient(self, *args, **kwargs):
        """Calls the gradient projection solver.

        Can be used as a callback.
        """
        self.proj_solver.solve()

    def set_up_solvers(self):
        """Sets up the time steppers for advection and reinitialisation."""
        self.ls_ts = self.tstep_alg(
            ScalarAdvectionEquation(self.func_space, self.func_space),
            self.level_set,
            self.ls_fields,
            self.tstep / self.subcycles,
            solver_parameters=self.solver_params,
        )
        self.reini_ts = self.reini_params["tstep_alg"](
            ReinitialisationEquation(self.func_space, self.func_space),
            self.level_set,
            self.reini_fields,
            self.reini_params["tstep"],
            solver_parameters=self.solver_params,
        )

    def solve(self, step: int):
        """Updates the level-set function.

        Calls advection and reinitialisation solvers within a subcycling loop.
        The reinitialisation solver can be iterated and might not be called at each
        simulation time step.

        Args:
            step:
              An integer representing the current simulation step.
        """
        if not self.solvers_ready:
            self.set_up_solvers()
            self.solvers_ready = True

        for subcycle in range(self.subcycles):
            self.ls_ts.advance(0)

            if step >= self.reini_params["frequency"]:
                self.reini_ts.solution_old.assign(self.level_set)

            if step % self.reini_params["frequency"] == 0:
                for reini_step in range(self.reini_params["iterations"]):
                    self.reini_ts.advance(
                        0, update_forcings=self.update_level_set_gradient
                    )

                self.ls_ts.solution_old.assign(self.level_set)


def material_field_recursive(
    level_set: fd.Function | list[fd.Function],
    field_values: list[fd.ufl.core.expr.Expr],
    interface: str,
) -> fd.ufl.core.expr.Expr:
    """Sets physical property expressions for each material.

    Ensures that the correct expression is assigned to each material based on the
    level-set functions.
    Property transition across material interfaces are expressed according to the
    provided averaging scheme.

    Args:
        level_set:
          A Firedrake function for the level set (or a list of these).
        field_values:
          A list of physical property values relevant for each material.
        interface:
          A string specifying how property transitions between materials are calculated.

    Returns:
        A UFL expression representing the physical property throughout the domain.

    Raises:
        ValueError: Incorrect interface strategy supplied.

    """
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match interface:
            case "sharp":
                return fd.conditional(
                    ls > 0.5,
                    field_values.pop(),
                    material_field_recursive(level_set, field_values, interface),
                )
            case "arithmetic":
                return field_values.pop() * ls + material_field_recursive(
                    level_set, field_values, interface
                ) * (1 - ls)
            case "geometric":
                return field_values.pop() ** ls * material_field_recursive(
                    level_set, field_values, interface
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / field_values.pop()
                    + (1 - ls)
                    / material_field_recursive(level_set, field_values, interface)
                )
            case _:
                raise ValueError(
                    "Method must be sharp, arithmetic, geometric, or harmonic."
                )
    else:  # Final level set; specify values for both sides of the interface
        match interface:
            case "sharp":
                return fd.conditional(ls < 0.5, *field_values)
            case "arithmetic":
                return field_values[0] * (1 - ls) + field_values[1] * ls
            case "geometric":
                return field_values[0] ** (1 - ls) * field_values[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / field_values[0] + ls / field_values[1])
            case _:
                raise ValueError(
                    "Interface must be sharp, arithmetic, geometric, or harmonic."
                )


def material_field(
    level_set: fd.Function | list[fd.Function],
    field_values: list[fd.ufl.core.expr.Expr],
    interface: str,
) -> fd.ufl.core.expr.Expr:
    """Executes material_field_recursive providing a copy of the level-set list.

    Providing a copy of the level-set list prevents the original one from being consumed
    by the function call.

    Args:
        level_set:
          A Firedrake function for the level set (or a list of these).
        field_values:
          A list of physical property values relevant for each material.
        interface:
          A string specifying how property transitions between materials are calculated.

    Returns:
        A UFL expression representing the physical property throughout the domain.
    """
    if not isinstance(level_set, list):
        level_set = [level_set]

    return material_field_recursive(level_set.copy(), field_values, interface)


def entrainment(
    level_set: fd.Function, material_area: Number, entrainment_height: Number
):
    """Calculates entrainment diagnostic.

    Determines the proportion of a material that is located above a given height.

    Args:
        level_set:
          A level-set Firedrake function.
        material_area:
          An integer or a float representing the total area occupied by a material.
        entrainment_height:
          An integer or a float representing the height above which the entrainment
          diagnostic is determined.

    Returns:
        A float corresponding to the calculated entrainment diagnostic.
    """
    mesh_coords = fd.SpatialCoordinate(level_set.function_space().mesh())
    target_region = mesh_coords[1] >= entrainment_height
    material_entrained = fd.conditional(level_set < 0.5, 1, 0)

    return (
        fd.assemble(fd.conditional(target_region, material_entrained, 0) * fd.dx)
        / material_area
    )
