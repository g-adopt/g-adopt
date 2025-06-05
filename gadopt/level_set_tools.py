r"""This module provides a set of classes and functions enabling multi-material
capabilities. Users initialise materials by instantiating the `Material` class and
define the physical properties of material interfaces using `field_interface`. They
instantiate the `LevelSetSolver` class by providing relevant parameters and call the
`solve` method to request a solver update. Finally, they may call the `entrainment`
function to calculate material entrainment in the simulation.

"""

import abc
from dataclasses import dataclass, fields
from numbers import Number
from typing import Any, Callable, Optional

import firedrake as fd
import numpy as np
import shapely as sl
from firedrake.ufl_expr import extract_unique_domain

from . import scalar_equation as scalar_eq
from .equations import Equation
from .time_stepper import eSSPRKs3p3
from .transport_solver import GenericTransportSolver
from .utility import node_coordinates

__all__ = [
    "LevelSetSolver",
    "Material",
    "assign_level_set_values",
    "entrainment",
    "interface_thickness",
    "material_field",
]

# Default solver options for level-set advection and reinitialisation
solver_params_default = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu",
}
# Default parameters used to set up level-set reinitialisation
reini_params_default = {
    "tstep": 1e-2,
    "tstep_alg": eSSPRKs3p3,
    "frequency": 5,
    "iterations": 1,
}


@dataclass(kw_only=True)
class Material:
    """A material with physical properties for the level-set approach.

    Expects material buoyancy to be defined using a value for either the reference
    density, buoyancy number, or compositional Rayleigh number.

    Contains static methods to calculate the physical properties of a material.
    Methods implemented here describe properties in the simplest non-dimensional
    simulation setup and must be overriden for more complex scenarios.

    Attributes:
        density:
          An integer or a float representing the reference density.
        B:
          An integer or a float representing the buoyancy number.
        RaB:
          An integer or a float representing the compositional Rayleigh number.
        density_B_RaB:
          A string to notify how the buoyancy term is calculated.
    """

    density: Optional[Number] = None
    B: Optional[Number] = None
    RaB: Optional[Number] = None

    def __post_init__(self):
        """Checks instance field values.

        Raises:
            ValueError:
              Incorrect field types.
        """
        count_None = 0
        for field_var in fields(self):
            field_var_value = getattr(self, field_var.name)
            if isinstance(field_var_value, Number):
                self.density_B_RaB = field_var.name
            elif field_var_value is None:
                count_None += 1
            else:
                raise ValueError(
                    "When provided, density, B, and RaB must have type int or float."
                )
        if count_None != 2:
            raise ValueError(
                "One, and only one, of density, B, and RaB must be provided, and it "
                "must be an integer or a float."
            )

    @staticmethod
    def viscosity(*args, **kwargs):
        """Calculates dynamic viscosity (Pa s)."""
        return 1.0

    @staticmethod
    def thermal_expansion(*args, **kwargs):
        """Calculates volumetric thermal expansion coefficient (K^-1)."""
        return 1.0

    @staticmethod
    def thermal_conductivity(*args, **kwargs):
        """Calculates thermal conductivity (W m^-1 K^-1)."""
        return 1.0

    @staticmethod
    def specific_heat_capacity(*args, **kwargs):
        """Calculates specific heat capacity at constant pressure (J kg^-1 K^-1)."""
        return 1.0

    @staticmethod
    def internal_heating_rate(*args, **kwargs):
        """Calculates internal heating rate per unit mass (W kg^-1)."""
        return 0.0

    @classmethod
    def thermal_diffusivity(cls, *args, **kwargs):
        """Calculates thermal diffusivity (m^2 s^-1)."""
        return cls.thermal_conductivity() / cls.density() / cls.specific_heat_capacity()


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
    interface thickness.

    Three scenarios are currently implemented to generate the signed-distance function:
    - The material interface is described by a mathematical function y = f(x). In this
      case, `interface_geometry` should be "curve" and `interface_callable` must be
      provided along with any `interface_args` to implement the aforementioned
      mathematical function.
    - The material interface is a polygon, and `interface_geometry` takes the value
      "polygon". In this case, `interface_coordinates` must exclude the polygon sides
      that do not act as a material interface and coincide with domain boundaries. The
      coordinates of these sides should be provided using the `boundary_coordinates`
      argument such that the concatenation of the two coordinate objects describes a
      closed polygonal chain.
    - The material interface is a circle, and `interface_geometry` takes the value
      "circle". In this case, `interface_coordinates` is a list holding the coordinates
      of the circle's centre and the circle radius. No other arguments are required.

    Geometrical objects underpinning material interfaces are generated using Shapely.

    Implemented interface geometry presets and associated arguments:
    | Curve     |                     Arguments                      |
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
                "`interface_geometry` must be 'curve', 'polygon', or 'circle'."
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
        tstep_alg: abc.ABCMeta,
        subcycles: int,
        epsilon: fd.Constant,
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
        self.u = velocity
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

        self.ls_eq_attrs = {"u": velocity}
        self.reini_eq_attrs = {
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

    def update_level_set_gradient(self):
        """Calls the gradient projection solver.

        Can be used as a callback.
        """
        self.proj_solver.solve()

    def set_up_solvers(self):
        """Sets up the time steppers for advection and reinitialisation."""
        test = fd.TestFunction(self.func_space)

        self.ls_solver = GenericTransportSolver(
            "advection",
            self.level_set,
            self.tstep / self.subcycles,
            self.tstep_alg,
            eq_attrs={"u": self.u},
            solver_parameters=self.solver_params,
        )

        reinitialisation_equation = Equation(
            test,
            self.func_space,
            reinitialisation_term,
            mass_term=scalar_eq.mass_term,
            eq_attrs=self.reini_eq_attrs,
        )
        self.reini_ts = self.reini_params["tstep_alg"](
            reinitialisation_equation,
            self.level_set,
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
            self.ls_solver.solve()

            if step >= self.reini_params["frequency"]:
                self.reini_ts.solution_old.assign(self.level_set)

            if step % self.reini_params["frequency"] == 0:
                for reini_step in range(self.reini_params["iterations"]):
                    self.reini_ts.advance(
                        update_forcings=self.update_level_set_gradient
                    )

                self.ls_solver.solution_old.assign(self.level_set)


def material_field_from_copy(
    level_set: fd.Function | list[fd.Function],
    field_values: list,
    interface: str,
    adjoint: bool = False,
) -> fd.ufl.algebra.Sum | fd.ufl.algebra.Product | fd.ufl.algebra.Division:
    """Generates UFL algebra describing a physical property across the domain.

    Ensures that the correct expression is assigned to each material based on the
    level-set functions. Property transitions across material interfaces are expressed
    according to the provided strategy.

    Args:
      level_set:
        A Firedrake function for the level set (or a list of these)
      field_values:
        A list of physical property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated
      adjoint:
        A boolean indicating an adjoint-compatible implementation of the sharp interface

    Note:
      When requesting the sharp interface with the adjoint flag, Stokes solver may raise
      DIVERGED_FNORM_NAN if the nodal level-set value is exactly 0.5 (i.e. denoting the
      location of the material interface).

    Returns:
      UFL algebra representing the physical property throughout the domain

    """
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match interface:
            case "sharp":
                if adjoint:
                    heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                    return field_values.pop() * heaviside + material_field_from_copy(
                        level_set, field_values, interface
                    ) * (1 - heaviside)
                else:
                    return fd.conditional(
                        ls > 0.5,
                        field_values.pop(),
                        material_field_from_copy(level_set, field_values, interface),
                    )
            case "arithmetic":
                return field_values.pop() * ls + material_field_from_copy(
                    level_set, field_values, interface
                ) * (1 - ls)
            case "geometric":
                return field_values.pop() ** ls * material_field_from_copy(
                    level_set, field_values, interface
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / field_values.pop()
                    + (1 - ls)
                    / material_field_from_copy(level_set, field_values, interface)
                )
    else:  # Final level set; specify values for both sides of the interface
        match interface:
            case "sharp":
                if adjoint:
                    heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                    return (
                        field_values[0] * (1 - heaviside) + field_values[1] * heaviside
                    )
                else:
                    return fd.conditional(ls < 0.5, *field_values)
            case "arithmetic":
                return field_values[0] * (1 - ls) + field_values[1] * ls
            case "geometric":
                return field_values[0] ** (1 - ls) * field_values[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / field_values[0] + ls / field_values[1])


def material_field(
    level_set: fd.Function | list[fd.Function],
    field_values: list,
    interface: str,
) -> fd.ufl.algebra.Sum | fd.ufl.algebra.Product | fd.ufl.algebra.Division:
    """Generates UFL algebra describing a physical property across the domain.

    Calls `material_field_from_copy` using a copy of the level-set list, preventing the
    original one from being consumed by the function call.

    Args:
      level_set:
        A Firedrake function for the level set (or a list of these)
      field_values:
        A list of physical property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL algebra representing the physical property throughout the domain

    Raises:
      ValueError: Incorrect interface strategy supplied
    """
    if not isinstance(level_set, list):
        level_set = [level_set]

    _impl_interface = ["sharp", "arithmetic", "geometric", "harmonic"]
    if interface not in _impl_interface:
        raise ValueError(f"Interface must be one of {_impl_interface}.")

    return material_field_from_copy(level_set.copy(), field_values, interface)


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


reinitialisation_term.required_attrs = {"epsilon", "level_set_grad"}
reinitialisation_term.optional_attrs = set()
