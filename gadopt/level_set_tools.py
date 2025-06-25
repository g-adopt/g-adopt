"""This module provides a set of classes and functions enabling multi-material
capabilities. Users initialise level-set fields using the `interface_thickness` and
`assign_level_set_values` functions. Given the level-set function(s), users can define
material-dependent physical properties via the `material_field` function. To evolve
level-set fields, users instantiate the `LevelSetSolver` class, choosing if they require
advection, reinitialisation, or both, and then call the `solve` method to request a
solver update. Finally, they may call the `material_entrainment` and `min_max_height`
functions to calculate useful simulation diagnostics.

"""

import operator
import re
from typing import Any, Callable
from warnings import warn

import firedrake as fd
import numpy as np
import shapely as sl
from mpi4py import MPI
from numpy.testing import assert_allclose
from ufl.core.expr import Expr

from .equations import Equation
from .scalar_equation import mass_term
from .time_stepper import eSSPRKs3p3, eSSPRKs10p3
from .transport_solver import GenericTransportSolver
from .utility import node_coordinates

__all__ = [
    "LevelSetSolver",
    "assign_level_set_values",
    "interface_thickness",
    "material_entrainment",
    "material_field",
    "min_max_height",
]

# Default parameters for level-set advection
adv_params_default = {
    "time_integrator": eSSPRKs10p3,
    "bcs": {},
    "solver_params": {"pc_type": "bjacobi", "sub_pc_type": "ilu"},
    "subcycles": 1,
}
# Default parameters for level-set reinitialisation
reini_params_default = {
    "timestep": 0.02,
    "time_integrator": eSSPRKs3p3,
    "solver_params": {"pc_type": "bjacobi", "sub_pc_type": "ilu"},
    "steps": 1,
}


def interface_thickness(
    level_set_space: fd.functionspaceimpl.WithGeometry, scale: float = 0.35
) -> fd.Function:
    """Default strategy for the thickness of the conservative level set profile.

    Args:
      level_set_space:
        The Firedrake function space of the level-set field
      scale:
        A float to control interface thickness values relative to cell sizes

    Returns:
      A Firedrake function holding the interface thickness values
    """
    epsilon = fd.Function(level_set_space, name="Interface thickness")
    epsilon.interpolate(scale * fd.MinCellEdgeLength(level_set_space.mesh()))

    return epsilon


def assign_level_set_values(
    level_set: fd.Function,
    epsilon: float | fd.Function,
    /,
    interface_geometry: str,
    interface_coordinates: list[list[float]] | list[list[float], float] | None = None,
    *,
    interface_callable: Callable | str | None = None,
    interface_args: tuple[Any] | None = None,
    boundary_coordinates: list[list[float]] | np.ndarray | None = None,
):
    """Updates level-set field given interface thickness and signed-distance function.

    Generates signed-distance function values at level-set nodes and overwrites
    level-set data according to the conservative level-set method using the provided
    interface thickness. By convention, the 1-side of the conservative level set is set
    above the curve or inside the polygon or circle.

    Three scenarios are currently implemented to generate the signed-distance function:
    - The material interface is described by a mathematical function y = f(x). In this
      case, `interface_geometry` should be `curve` and `interface_callable` must be
      provided along with any `interface_args` to implement the aforementioned
      mathematical function.
    - The material interface is a polygon, and `interface_geometry` takes the value
      `polygon`. In this case, `interface_coordinates` must exclude the polygon sides
      that do not act as a material interface and coincide with domain boundaries. The
      coordinates of these sides should be provided using the `boundary_coordinates`
      argument such that the concatenation of the two coordinate objects describes a
      closed polygonal chain.
    - The material interface is a circle, and `interface_geometry` takes the value
      `circle`. In this case, `interface_coordinates` is a list holding the coordinates
      of the circle's centre and radius. No other arguments are required.

    Geometrical objects underpinning material interfaces are generated using Shapely.

    Implemented interface geometry presets and associated arguments:
    | Interface |                     Arguments                      |
    | :-------- | :------------------------------------------------: |
    | line      | slope, intercept                                   |
    | cosine    | amplitude, wavelength, vertical_shift, phase_shift |
    | rectangle | ref_vertex_coords, edge_sizes                      |

    Args:
      level_set:
        A Firedrake function for the targeted level-set field
      epsilon:
        A float or Firedrake function representing the interface thickness
      interface_geometry:
        A string specifying the geometry to create
      interface_coordinates:
        A sequence or an array-like with shape (N, 2) of numeric coordinate pairs
        defining the interface or a list containing centre coordinates and radius
      interface_callable:
        A callable implementing the mathematical function depicting the interface or a
        string matching an implemented callable preset
      interface_args:
        A tuple of arguments provided to the interface callable
      boundary_coordinates:
        A sequence of numeric coordinate pairs or an array-like with shape (N, 2)
    """

    def stack_coordinates(func: Callable) -> Callable:
        """Decorator to stack coordinates when the material interface is a curve.

        Args:
          func:
            A callable implementing the mathematical function depicting the interface

        Returns:
          A callable that can stack interface coordinates
        """

        def wrapper(*args) -> float | np.ndarray:
            if isinstance(interface_coords_x := args[0], (int, float)):
                return func(*args)
            else:
                return np.column_stack((interface_coords_x, func(*args)))

        return wrapper

    def line(x, slope, intercept) -> float | np.ndarray:
        """Straight line equation"""
        return slope * x + intercept

    def cosine(
        x, amplitude, wavelength, vertical_shift, phase_shift=0
    ) -> float | np.ndarray:
        """Cosine function with an amplitude and a vertical shift."""
        cosine = np.cos(2 * np.pi / wavelength * x + phase_shift)

        return amplitude * cosine + vertical_shift

    def rectangle(
        ref_vertex_coords: tuple[float], edge_sizes: tuple[float]
    ) -> list[tuple[float]]:
        """Material interface defined by a rectangle.

        Edges are aligned with Cartesian directions and do not overlap domain boundaries.

        Args:
          ref_vertex_coords:
            A tuple holding the coordinates of the lower-left vertex
          edge_sizes:
            A tuple holding the edge sizes

        Returns:
          A list of tuples representing the coordinates of the rectangle's vertices
        """
        interface_coords = [
            (ref_vertex_coords[0], ref_vertex_coords[1]),
            (ref_vertex_coords[0] + edge_sizes[0], ref_vertex_coords[1]),
            (
                ref_vertex_coords[0] + edge_sizes[0],
                ref_vertex_coords[1] + edge_sizes[1],
            ),
            (ref_vertex_coords[0], ref_vertex_coords[1] + edge_sizes[1]),
            (ref_vertex_coords[0], ref_vertex_coords[1]),
        ]

        return interface_coords

    callable_presets = {"cosine": cosine, "line": line, "rectangle": rectangle}
    if isinstance(interface_callable, str):
        interface_callable = callable_presets[interface_callable]

    if interface_callable is not None:
        if interface_geometry == "curve":
            interface_callable = stack_coordinates(interface_callable)
        interface_coordinates = interface_callable(*interface_args)

    match interface_geometry:
        case "curve":
            interface = sl.LineString(interface_coordinates)

            signed_distance = [
                (1 if y > interface_callable(x, *interface_args[1:]) else -1)
                * interface.distance(sl.Point(x, y))
                for x, y in node_coordinates(level_set).dat.data
            ]
        case "polygon":
            if boundary_coordinates is None:
                interface = sl.Polygon(interface_coordinates)
                sl.prepare(interface)

                signed_distance = [
                    (1 if interface.contains(sl.Point(x, y)) else -1)
                    * interface.boundary.distance(sl.Point(x, y))
                    for x, y in node_coordinates(level_set).dat.data
                ]
            else:
                interface = sl.LineString(interface_coordinates)
                interface_with_boundaries = sl.Polygon(
                    np.vstack((interface_coordinates, boundary_coordinates))
                )
                sl.prepare(interface_with_boundaries)

                signed_distance = [
                    (1 if interface_with_boundaries.intersects(sl.Point(x, y)) else -1)
                    * interface.distance(sl.Point(x, y))
                    for x, y in node_coordinates(level_set).dat.data
                ]
        case "circle":
            centre, radius = interface_coordinates
            interface = sl.Point(centre).buffer(radius)
            sl.prepare(interface)

            signed_distance = [
                (1 if interface.contains(sl.Point(x, y)) else -1)
                * interface.boundary.distance(sl.Point(x, y))
                for x, y in node_coordinates(level_set).dat.data
            ]
        case _:
            raise ValueError(
                "'interface_geometry' must be 'curve', 'polygon', or 'circle'."
            )

    if isinstance(epsilon, fd.Function):
        epsilon = epsilon.dat.data

    level_set.dat.data[:] = (1 + np.tanh(np.asarray(signed_distance) / 2 / epsilon)) / 2


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

    Advects and reinitialises a level-set field.

    Attributes:
      solution:
        The Firedrake function holding level-set values
      solution_grad:
        The Firedrake function holding level-set gradient values
      solution_space:
        The Firedrake function space where the level set lives
      mesh:
        The Firedrake mesh representing the numerical domain
      advection:
        A boolean specifying whether advection is set up
      reinitialisation:
        A boolean specifying whether reinitialisation is set up
      adv_kwargs:
        A dictionary holding the parameters used to set up advection
      reini_kwargs:
        A dictionary holding the parameters used to set up reinitialisation
      adv_solver:
        A G-ADOPT GenericTransportSolver tackling advection
      gradient_solver:
        A Firedrake LinearVariationalSolver to calculate the level-set gradient
      reini_integrator:
        A G-ADOPT time integrator tackling reinitialisation
      step:
        An integer representing the number of advection steps already made
    """

    def __init__(
        self,
        level_set: fd.Function,
        /,
        *,
        adv_kwargs: dict[str, Any] | None = None,
        reini_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialises the solver instance.

        Valid keys for `adv_kwargs`:
        |    Argument     | Required |                   Description                   |
        | :-------------- | :------: | :---------------------------------------------: |
        | u               | True     | Velocity field (`Function`)                     |
        | timestep        | True     | Integration time step (`Constant`)              |
        | time_integrator | False    | Time integrator (class in `time_stepper.py`)    |
        | bcs             | False    | Boundary conditions (`dict`, G-ADOPT API)       |
        | solver_params   | False    | Solver parameters (`dict`, PETSc API)           |
        | subcycles       | False    | Advection iterations in one time step (`int`)   |

        Valid keys for `reini_kwargs`:
        |    Argument     | Required |                   Description                   |
        | :-------------- | :------: | :---------------------------------------------: |
        | epsilon         | True     | Interface thickness (`float` or `Function`)     |
        | timestep        | False    | Integration step in pseudo-time (`int`)         |
        | time_integrator | False    | Time integrator (class in `time_stepper.py`)    |
        | solver_params   | False    | Solver parameters (`dict`, PETSc API)           |
        | steps           | False    | Pseudo-time integration steps (`int`)           |
        | frequency       | False    | Advection steps before reinitialisation (`int`) |

        Args:
          level_set:
            The Firedrake function holding the level-set field
          adv_kwargs:
            A dictionary with parameters used to set up advection
          reini_kwargs:
            A dictionary with parameters used to set up reinitialisation
        """
        self.solution = level_set
        self.solution_old = fd.Function(self.solution)
        self.solution_space = level_set.function_space()
        self.mesh = self.solution.ufl_domain()
        self.advection = False
        self.reinitialisation = False

        self.set_gradient_solver()

        if isinstance(adv_kwargs, dict):
            if not all(param in adv_kwargs for param in ["u", "timestep"]):
                raise KeyError("'u' and 'timestep' must be present in 'adv_kwargs'")

            self.advection = True
            self.adv_kwargs = adv_params_default | adv_kwargs

        if isinstance(reini_kwargs, dict):
            if "epsilon" not in reini_kwargs:
                raise KeyError("'epsilon' must be present in 'reini_kwargs'")

            self.reinitialisation = True
            self.reini_kwargs = reini_params_default | reini_kwargs
            if "frequency" not in self.reini_kwargs:
                self.reini_kwargs["frequency"] = self.reinitialisation_frequency()

        if not any([self.advection, self.reinitialisation]):
            raise ValueError("Advection or reinitialisation must be initialised")

        self._solvers_ready = False

    def reinitialisation_frequency(self) -> int:
        """Implements default strategy for the reinitialisation frequency.

        Reinitialisation becomes less frequent as mesh resolution increases, with the
        underlying assumption that the minimum cell size occurs along the material
        interface. The current strategy is to apply reinitialisation at every time step
        up to a certain cell size and then scale the frequency with the decrease in cell
        size.
        """
        epsilon = self.reini_kwargs["epsilon"]
        if isinstance(epsilon, fd.Function):
            epsilon = self.mesh.comm.allreduce(epsilon.dat.data.min(), MPI.MIN)

        if self.mesh.cartesian:
            max_coords = self.mesh.coordinates.dat.data.max(axis=0)
            min_coords = self.mesh.coordinates.dat.data.min(axis=0)
            for i in range(len(max_coords)):
                max_coords[i] = self.mesh.comm.allreduce(max_coords[i], MPI.MAX)
                min_coords[i] = self.mesh.comm.allreduce(min_coords[i], MPI.MIN)
            domain_size = np.sqrt(np.sum((max_coords - min_coords) ** 2))

            return max(1, round(4.9e-3 * domain_size / epsilon - 0.25))
        else:
            warn(
                "No frequency strategy implemented for reinitialisation in "
                "non-rectangular/cuboidal domains; applying reinitialisation at every "
                "time step"
            )

            return 1

    def set_gradient_solver(self) -> None:
        """Constructs a solver to determine the level-set gradient.

        The weak form is derived through integration by parts and includes a term
        accounting for boundary flux.
        """
        grad_name = "Level-set gradient"
        if number_match := re.search(r"\s#\d+$", self.solution.name()):
            grad_name += number_match.group()

        gradient_space = fd.VectorFunctionSpace(
            self.mesh, "Q", self.solution.ufl_element().degree()
        )
        self.solution_grad = fd.Function(gradient_space, name=grad_name)

        test = fd.TestFunction(gradient_space)
        trial = fd.TrialFunction(gradient_space)

        bilinear_form = fd.inner(test, trial) * fd.dx(domain=self.mesh)
        ibp_element = -self.solution * fd.div(test) * fd.dx(domain=self.mesh)
        ibp_boundary = (
            self.solution
            * fd.dot(test, fd.FacetNormal(self.mesh))
            * fd.ds(domain=self.mesh)
        )
        linear_form = ibp_element + ibp_boundary

        problem = fd.LinearVariationalProblem(
            bilinear_form, linear_form, self.solution_grad
        )
        self.gradient_solver = fd.LinearVariationalSolver(problem)

    def set_up_solvers(self) -> None:
        """Sets up time integrators for advection and reinitialisation as required."""
        if self.advection:
            self.adv_solver = GenericTransportSolver(
                "advection",
                self.solution,
                self.adv_kwargs["timestep"] / self.adv_kwargs["subcycles"],
                self.adv_kwargs["time_integrator"],
                solution_old=self.solution_old,
                eq_attrs={"u": self.adv_kwargs["u"]},
                bcs=self.adv_kwargs["bcs"],
                solver_parameters=self.adv_kwargs["solver_params"],
            )

        if self.reinitialisation:
            reinitialisation_equation = Equation(
                fd.TestFunction(self.solution_space),
                self.solution_space,
                reinitialisation_term,
                mass_term=mass_term,
                eq_attrs={
                    "level_set_grad": self.solution_grad,
                    "epsilon": self.reini_kwargs["epsilon"],
                },
            )

            self.reini_integrator = self.reini_kwargs["time_integrator"](
                reinitialisation_equation,
                self.solution,
                self.reini_kwargs["timestep"],
                solution_old=self.solution_old,
                solver_parameters=self.reini_kwargs["solver_params"],
            )

        self.step = 0
        self._solvers_ready = True

    def update_gradient(self, *args, **kwargs) -> None:
        """Calls the gradient solver.

        Can be provided as a forcing to time integrators.
        """
        self.gradient_solver.solve()

    def reinitialise(self) -> None:
        """Performs reinitialisation steps."""
        for _ in range(self.reini_kwargs["steps"]):
            self.reini_integrator.advance(t=0, update_forcings=self.update_gradient)

    def solve(
        self,
        disable_advection: bool = False,
        disable_reinitialisation: bool = False,
    ) -> None:
        """Updates the level-set function by means of advection and reinitialisation.

        Args:
          disable_advection:
            A boolean to disable the advection solve.
          disable_reinitialisation:
            A boolean to disable the reinitialisation solve.
        """
        if not self._solvers_ready:
            self.set_up_solvers()

        if self.advection and not disable_advection:
            for _ in range(self.adv_kwargs["subcycles"]):
                self.adv_solver.solve()
                self.step += 1

                if self.reinitialisation and not disable_reinitialisation:
                    if self.step % self.reini_kwargs["frequency"] == 0:
                        self.reinitialise()

        elif self.reinitialisation and not disable_reinitialisation:
            if self.step % self.reini_kwargs["frequency"] == 0:
                self.reinitialise()


def material_interface(
    level_set: fd.Function, field_value: float, other_side: float | Expr, interface: str
):
    """Generates UFL algebra describing a physical property across a material interface.

    Ensures that the correct expression is assigned to each material based on the
    level-set field.

    Args:
      level_set:
        A Firedrake function representing the level-set field
      field_values:
        A float corresponding to the value of a physical property for a given material
      other_side:
        A float or UFL instance expressing the field value outside the material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL instance for a term of the physical-property algebraic expression

    """
    match interface:
        case "sharp":
            return fd.conditional(level_set > 0.5, field_value, other_side)
        case "sharp_adjoint":
            ls_shift = level_set - 0.5
            heaviside = (ls_shift + abs(ls_shift)) / 2 / ls_shift

            return field_value * heaviside + other_side * (1 - heaviside)
        case "arithmetic":
            return field_value * level_set + other_side * (1 - level_set)
        case "geometric":
            return field_value**level_set * other_side ** (1 - level_set)
        case "harmonic":
            return 1 / (level_set / field_value + (1 - level_set) / other_side)


def material_field_from_copy(
    level_set: list[fd.Function], field_values: list[float], interface: str
) -> Expr:
    """Generates UFL algebra by consuming `level_set` and `field_values` lists.

    Args:
      level_set:
        A list of one or multiple Firedrake level-set functions
      field_values:
        A list of physical property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL algebra representing the physical property throughout the domain

    """
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        return material_interface(
            ls,
            field_values.pop(),
            material_field_from_copy(level_set, field_values, interface),
            interface,
        )
    else:  # Final level set; specify values for both sides of the interface
        return material_interface(ls, field_values.pop(), field_values.pop(), interface)


def material_field(
    level_set: fd.Function | list[fd.Function],
    field_values: list[float],
    interface: str,
) -> Expr:
    """Generates UFL algebra describing a physical property across the domain.

    Calls `material_field_from_copy` using a copy of the level-set list, preventing the
    potential original one from being consumed by the function call.

    Ordering of `field_values` must be consistent with `level_set`, such that the first
    element in the list corresponds to the field value on the 0-side of the first
    level-set function and the last element in the list to the field value on the 1-side
    of the last level-set function.

    **Note**: When requesting the `sharp_adjoint` interface, calling Stokes solver may
    raise `DIVERGED_FNORM_NAN` if the nodal level-set value is exactly 0.5 (i.e.
    denoting the location of the material interface).

    Args:
      level_set:
        A Firedrake function for the level set (or a list thereof)
      field_values:
        A list of physical-property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL algebra representing the physical property throughout the domain

    Raises:
      ValueError: Incorrect interface strategy supplied
    """
    level_set = level_set.copy() if isinstance(level_set, list) else [level_set]

    impl_interface = ["sharp", "sharp_adjoint", "arithmetic", "geometric", "harmonic"]
    if interface not in impl_interface:
        raise ValueError(f"Interface must be one of {impl_interface}.")

    return material_field_from_copy(level_set, field_values, interface)


def material_entrainment(
    level_set: fd.Function,
    /,
    *,
    material_size: float,
    entrainment_height: float,
    side: int,
    direction: str,
    skip_material_size_check: bool = False,
) -> float:
    """Calculates the proportion of a material located above or below a given height.

    For the diagnostic calculation to be meaningful, the level-set side provided must
    spatially isolate the target material.

    **Note**: This function checks if the total volume or area occupied by the target
    material matches the `material_size` value.

    Args:
      level_set:
        A Firedrake function for the level-set field
      material_size:
        A float representing the total volume or area occupied by the target material
      entrainment_height:
        A float representing the height above which to calculate entrainment
      side:
        An integer (`0` or `1`) denoting the level-set value on the target material side
      direction:
        A string (`above` or `below`) denoting the target entrainment direction
      skip_material_size_check:
        A boolean enabling to skip the consistency check of the material volume or area

    Returns:
      A float corresponding to the material fraction above or below the target height

    Raises:
      AssertionError: Material volume or area notably different from `material_size`
    """
    if not level_set.ufl_domain().cartesian:
        raise ValueError("Only Cartesian meshes are currently supported.")

    match side:
        case 0:
            material_check = operator.le
        case 1:
            material_check = operator.ge
        case _:
            raise ValueError("'side' must be 0 or 1.")

    match direction:
        case "above":
            region_check = operator.ge
        case "below":
            region_check = operator.le
        case _:
            raise ValueError("'direction' must be 'above' or 'below'.")

    material = fd.conditional(material_check(level_set, 0.5), 1, 0)
    if not skip_material_size_check:
        assert_allclose(
            fd.assemble(material * fd.dx),
            material_size,
            rtol=5e-2,
            err_msg="Material volume or area notably different from 'material_size'",
        )

    *_, vertical_coord = node_coordinates(level_set)
    target_region = region_check(vertical_coord, entrainment_height)
    is_entrained = fd.conditional(target_region, material, 0)

    return fd.assemble(is_entrained * fd.dx) / material_size


def min_max_height(
    level_set: fd.Function, epsilon: float | fd.Function, *, side: int, mode: str
) -> float:
    """Calculates the maximum or minimum height of a material interface.

    Args:
      level_set:
        A Firedrake function for the level set field
      epsilon:
        A float or Firedrake function denoting the thickness of the material interface
      side:
        An integer (`0` or `1`) denoting the level-set value on the target material side
      mode:
        A string ("min" or "max") specifying which extremum height is sought

    Returns:
      A float corresponding to the material interface extremum height
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
            extremum = np.min
            ls_arg_extremum = np.argmax
            irrelevant_data = np.inf
            mpi_comparison = MPI.MIN
        case "max":
            extremum = np.max
            ls_arg_extremum = np.argmin
            irrelevant_data = -np.inf
            mpi_comparison = MPI.MAX
        case _:
            raise ValueError("'mode' must be 'min' or 'max'.")

    if not level_set.ufl_domain().cartesian:
        raise ValueError("Only Cartesian meshes are currently supported.")

    coords = node_coordinates(level_set)

    coords_data = coords.dat.data_ro
    ls_data = level_set.dat.data_ro
    if isinstance(epsilon, float):
        eps_data = epsilon * np.ones_like(ls_data)
    else:
        eps_data = epsilon.dat.data_ro

    mask_ls = comparison(ls_data, 0.5)
    if mask_ls.any():
        coords_inside = coords_data[mask_ls, -1]
        ind_coords_inside = np.flatnonzero(coords_inside == extremum(coords_inside))

        if ind_coords_inside.size == 1:
            ind_inside = ind_coords_inside.item()
        else:
            ind_min_ls_inside = ls_arg_extremum(ls_data[mask_ls][ind_coords_inside])
            ind_inside = ind_coords_inside[ind_min_ls_inside]

        height_inside = coords_inside[ind_inside]

        hor_coords = coords_data[mask_ls, :-1][ind_inside]
        hor_dist_vec = coords_data[~mask_ls, :-1] - hor_coords
        hor_dist = np.sqrt(np.sum(hor_dist_vec**2, axis=1))

        mask_hor_coords = hor_dist < eps_data[~mask_ls]

        if mask_hor_coords.any():
            ind_outside = abs(
                coords_data[~mask_ls, -1][mask_hor_coords] - height_inside
            ).argmin()
            height_outside = coords_data[~mask_ls, -1][mask_hor_coords][ind_outside]

            ls_inside = ls_data[mask_ls][ind_inside]
            eps_inside = eps_data[mask_ls][ind_inside]
            sdls_inside = eps_inside * np.log(ls_inside / (1 - ls_inside))

            ls_outside = ls_data[~mask_ls][mask_hor_coords][ind_outside]
            eps_outside = eps_data[~mask_ls][mask_hor_coords][ind_outside]
            sdls_outside = eps_outside * np.log(ls_outside / (1 - ls_outside))

            sdls_dist = sdls_outside / (sdls_outside - sdls_inside)
            height = sdls_dist * height_inside + (1 - sdls_dist) * height_outside
        else:
            height = height_inside
    else:
        height = irrelevant_data

    height_global = level_set.comm.allreduce(height, mpi_comparison)

    return height_global


reinitialisation_term.required_attrs = {"epsilon", "level_set_grad"}
reinitialisation_term.optional_attrs = set()
