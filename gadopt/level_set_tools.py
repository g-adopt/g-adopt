r"""This module provides a set of classes and functions enabling multi-material
capabilities. The `material_field` function provides a simple way to define a UFL
expression for a physical field depending on material properties. The `LevelSetSolver`
class instantiates the advection and reinitialisation systems for a single level set.
The `entrainment` function enables users to easily calculate material entrainment.

"""

import operator
from math import ceil
from numbers import Number
from typing import Any

import firedrake as fd
import numpy as np
from mpi4py import MPI

from . import scalar_equation as scalar_eq
from .equations import Equation
from .time_stepper import RungeKuttaTimeIntegrator, eSSPRKs3p3
from .transport_solver import GenericTransportSolver

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
    "tstep": 0.1,
    "tstep_alg": eSSPRKs3p3,
    "frequency": None,
    "iterations": 1,
}


def reinitialisation_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    """Term for the conservative level set reinitialisation equation.

    Implements terms on the right-hand side of Equation 17 from
    Parameswaran, S., & Mandal, J. C. (2023).
    A stable interface-preserving reinitialization equation for conservative level set
    method.
    European Journal of Mechanics-B/Fluids, 98, 40-63.
    """
    sharpen_term = -trial * (1 - trial) * (1 - 2 * trial) * eq.test * eq.dx
    balance_term = (
        eq.epsilon
        * (1 - 2 * trial)
        * fd.sqrt(fd.inner(eq.level_set_grad, eq.level_set_grad))
        * eq.test
        * eq.dx
    )

    return sharpen_term + balance_term


class LevelSetSolver:
    """Solver for the conservative level-set approach.

    Solves the advection and reinitialisation equations for a level set function.

    Attributes:
        mesh:
          The UFL mesh where values of the level set function exist.
        solution:
          The Firedrake function for the level set.
        func_space_lsgp:
          The UFL function space for the projected level-set gradient.
        ls_grad_proj:
          The Firedrake function for the projected level-set gradient.
        proj_solver:
          A Firedrake LinearVariationalSolver to project the level-set gradient.
        reini_params:
          A dictionary containing parameters used in the reinitialisation approach.
        ls_solver:
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
        epsilon: float,
        subcycles: int = 1,
        bcs: dict = {},
        reini_params: dict[str, Any] | None = None,
        solver_params: dict[str, str] | None = None,
    ) -> None:
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
            epsilon:
              A float denoting the thickness of the hyperbolic tangent profile.
            subcycles:
              An integer specifying the number of advection solves to perform.
            bcs:
              Dictionary of identifier-value pairs specifying boundary conditions
            reini_params:
              A dictionary containing parameters used in the reinitialisation approach.
            solver_params:
              A dictionary containing solver parameters used in the advection and
              reinitialisation approaches.
        """
        self.solution = level_set
        self.tstep = tstep
        self.tstep_alg = tstep_alg
        self.subcycles = subcycles
        self.bcs = bcs

        self.reini_params = reini_params or reini_params_default.copy()
        self.solver_params = solver_params or solver_params_default.copy()

        self.func_space = level_set.function_space()
        self.mesh = self.func_space.mesh()
        self.solution_old = fd.Function(self.func_space)

        if self.reini_params["frequency"] is None:
            max_coords = self.mesh.coordinates.dat.data.max(axis=0)
            min_coords = self.mesh.coordinates.dat.data.min(axis=0)
            for i in range(len(max_coords)):
                max_coords[i] = level_set.comm.allreduce(max_coords[i], MPI.MAX)
                min_coords[i] = level_set.comm.allreduce(min_coords[i], MPI.MIN)
            max_mesh_dimension = max(max_coords - min_coords)

            self.reini_params["frequency"] = ceil(0.02 / epsilon * max_mesh_dimension)

        self.func_space_lsgp = fd.VectorFunctionSpace(
            self.mesh, "CG", self.func_space.finat_element.degree
        )
        self.ls_grad_proj = fd.Function(
            self.func_space_lsgp,
            name=f"Level-set gradient (projection) #{level_set.name()[-1]}",
        )

        self.proj_solver = self.gradient_L2_proj()

        self.ls_eq_attrs = {"u": velocity}
        self.reini_eq_attrs = {"level_set_grad": self.ls_grad_proj, "epsilon": epsilon}

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
            self.solution * fd.div(test_function) * fd.dx(domain=self.mesh)
        )
        residual_boundary = (
            self.solution
            * fd.dot(test_function, fd.FacetNormal(self.mesh))
            * fd.ds(domain=self.mesh)
        )
        residual = -residual_element + residual_boundary
        problem = fd.LinearVariationalProblem(mass_term, residual, self.ls_grad_proj)

        solver_params = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        return fd.LinearVariationalSolver(problem, solver_parameters=solver_params)

    def update_level_set_gradient(self, *args, **kwargs) -> None:
        """Calls the gradient projection solver.

        Can be used as a callback.
        """
        self.proj_solver.solve()

    def set_up_solvers(self) -> None:
        """Sets up the time steppers for advection and reinitialisation."""
        self.ls_solver = GenericTransportSolver(
            "advection",
            self.solution,
            self.tstep / self.subcycles,
            self.tstep_alg,
            solution_old=self.solution_old,
            eq_attrs=self.ls_eq_attrs,
            bcs=self.bcs,
            solver_parameters=self.solver_params,
        )

        reinitialisation_equation = Equation(
            fd.TestFunction(self.func_space),
            self.func_space,
            reinitialisation_term,
            mass_term=scalar_eq.mass_term,
            eq_attrs=self.reini_eq_attrs,
        )

        self.reini_ts = self.reini_params["tstep_alg"](
            reinitialisation_equation,
            self.solution,
            self.reini_params["tstep"],
            solution_old=self.solution_old,
            solver_parameters=self.solver_params,
        )

    def solve(self, step: int, equation: str | None = None) -> None:
        """Updates the level-set function.

        Calls advection and reinitialisation solvers within a subcycling loop.
        The reinitialisation solver can be iterated and may not be called at each
        simulation time step.

        Args:
            step:
              An integer representing the current time-loop iteration.
            equation:
              An optional string specifying which equation to solve if not both.
        """
        if not self.solvers_ready:
            self.set_up_solvers()
            self.solvers_ready = True

        for subcycle in range(self.subcycles):
            if equation != "reinitialisation":
                self.ls_solver.solve()

            if equation != "advection" and step % self.reini_params["frequency"] == 0:
                for reini_step in range(self.reini_params["iterations"]):
                    self.reini_ts.advance(
                        t=0, update_forcings=self.update_level_set_gradient
                    )


def material_field_recursive(
    level_set: fd.Function | list[fd.Function],
    field_values: list[fd.ufl.core.expr.Expr],
    interface: str,
) -> fd.ufl.core.expr.Expr:
    """Sets physical property expressions reflecting material distribution.

    Ensures that the correct expression is assigned to each material based on the
    level-set functions. Property transition across material interfaces are expressed
    according to the provided averaging scheme.

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
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match interface:
            case "sharp":
                heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                return field_values.pop() * heaviside + material_field_recursive(
                    level_set, field_values, interface
                ) * (1 - heaviside)
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
    else:  # Final level set; specify values for both sides of the interface
        match interface:
            case "sharp":
                heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                return field_values[0] * (1 - heaviside) + field_values[1] * heaviside
            case "arithmetic":
                return field_values[0] * (1 - ls) + field_values[1] * ls
            case "geometric":
                return field_values[0] ** (1 - ls) * field_values[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / field_values[0] + ls / field_values[1])


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

    Raises:
        ValueError: Incorrect interface strategy supplied.
    """
    if not isinstance(level_set, list):
        level_set = [level_set]

    _impl_interface = ["sharp", "arithmetic", "geometric", "harmonic"]
    if interface not in _impl_interface:
        raise ValueError(f"Interface must be one of {_impl_interface}.")

    return material_field_recursive(level_set.copy(), field_values, interface)


def entrainment(
    level_set: fd.Function, material_area: Number, entrainment_height: Number
) -> float:
    """Calculates the entrainment diagnostic.

    Determines the proportion of a material that is located above a given height.

    Args:
        level_set:
          A Firedrake function for the level set field.
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


def min_max_height(
    level_set: fd.Function, epsilon: fd.Constant, side: int, mode: str
) -> float:
    """Calculates the maximum or minimum height of the level set.

    Determines the location.

    Args:
        level_set:
          A Firedrake function for the level set field.
        epsilon:
          A Firedrake constant for the thickness of the hyperbolic tangent profile.
        side:
          An integer indicating the level set side making up the geometric object.
        mode:
          A string indicating whether the maximum or minimum height is sought.

    Returns:
        A float corresponding to the level set maximum or minimum height.
    """

    match side:
        case 0:
            comparison = operator.le
        case 1:
            comparison = operator.ge
        case _:
            raise ValueError("'side' must be 0 or 1.")

    match mode:
        case "min":
            arg_finder = np.argmin
            irrelevant_data = np.inf
            mpi_comparison = MPI.MIN
        case "max":
            arg_finder = np.argmax
            irrelevant_data = -np.inf
            mpi_comparison = MPI.MAX
        case _:
            raise ValueError("'mode' must be 'min' or 'max'.")

    mesh = level_set.ufl_domain()
    if not mesh.cartesian:
        raise ValueError("Only Cartesian meshes are currently supported.")

    coords_space = fd.VectorFunctionSpace(mesh, level_set.ufl_element())
    coords = fd.Function(coords_space).interpolate(mesh.coordinates)
    coords_data = coords.dat.data_ro_with_halos
    ls_data = level_set.dat.data_ro_with_halos

    mask_ls = comparison(ls_data, 0.5)
    if mask_ls.any():
        ind_inside = arg_finder(coords_data[mask_ls, -1])
        height_inside = coords_data[mask_ls, -1][ind_inside]

        if not mask_ls.all():
            hor_coords = coords_data[mask_ls, :-1][ind_inside]
            hor_dist_vec = coords_data[~mask_ls, :-1] - hor_coords
            hor_dist = np.sqrt(np.sum(hor_dist_vec**2, axis=1))

            epsilon = float(epsilon)
            mask_hor_coords = hor_dist < epsilon

            if mask_hor_coords.any():
                ind_outside = abs(
                    coords_data[~mask_ls, -1][mask_hor_coords] - height_inside
                ).argmin()
                height_outside = coords_data[~mask_ls, -1][mask_hor_coords][ind_outside]

                ls_inside = ls_data[mask_ls][ind_inside]
                sdls_inside = epsilon * np.log(ls_inside / (1 - ls_inside))

                ls_outside = ls_data[~mask_ls][mask_hor_coords][ind_outside]
                sdls_outside = epsilon * np.log(ls_outside / (1 - ls_outside))

                sdls_dist = sdls_outside / (sdls_outside - sdls_inside)
                height = sdls_dist * height_inside + (1 - sdls_dist) * height_outside
            else:
                height = height_inside

        else:
            height = height_inside

    else:
        height = irrelevant_data

    height_global = level_set.comm.allreduce(height, mpi_comparison)

    return height_global


reinitialisation_term.required_attrs = {"epsilon", "level_set_grad"}
reinitialisation_term.optional_attrs = set()
