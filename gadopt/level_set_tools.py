r"""This module provides a set of classes and functions enabling multi-material
capabilities. Users initialise materials by instantiating the `Material` class and
define the physical properties of material interfaces using `field_interface`. They
instantiate the `LevelSetSolver` class by providing relevant parameters and call the
`solve` method to request a solver update. Finally, they may call the `entrainment`
function to calculate material entrainment in the simulation.

"""

import re
from dataclasses import dataclass, fields
from numbers import Number
from typing import Any, Callable, Optional
from warnings import warn

import firedrake as fd
import numpy as np
import shapely as sl
from mpi4py import MPI

from . import scalar_equation as scalar_eq
from .equations import Equation
from .time_stepper import eSSPRKs3p3, eSSPRKs10p3
from .transport_solver import GenericTransportSolver
from .utility import node_coordinates

__all__ = [
    "LevelSetSolver",
    "Material",
    "assign_level_set_values",
    "density_RaB",
    "entrainment",
    "field_interface",
    "interface_thickness",
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
                mass_term=scalar_eq.mass_term,
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


def field_interface_recursive(
    level_set: list, material_value: list, method: str
) -> fd.ufl.core.expr.Expr:
    """Sets physical property expressions for each material.

    Ensures that the correct expression is assigned to each material based on the
    level-set functions.
    Property transition across material interfaces are expressed according to the
    provided method.

    Args:
        level_set:
          A list of level-set UFL functions.
        material_value:
          A list of physical property values applicable to each material.
        method:
          A string specifying the nature of property transitions between materials.

    Returns:
        A UFL expression to calculate physical property values throughout the domain.

    Raises:
      ValueError: Incorrect method name supplied.

    """
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match method:
            case "sharp":
                return fd.conditional(
                    ls > 0.5,
                    material_value.pop(),
                    field_interface_recursive(level_set, material_value, method),
                )
            case "arithmetic":
                return material_value.pop() * ls + field_interface_recursive(
                    level_set, material_value, method
                ) * (1 - ls)
            case "geometric":
                return material_value.pop() ** ls * field_interface_recursive(
                    level_set, material_value, method
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / material_value.pop()
                    + (1 - ls)
                    / field_interface_recursive(level_set, material_value, method)
                )
            case _:
                raise ValueError(
                    "Method must be sharp, arithmetic, geometric, or harmonic."
                )
    else:  # Final level set; specify values for both sides of the interface
        match method:
            case "sharp":
                return fd.conditional(ls < 0.5, *material_value)
            case "arithmetic":
                return material_value[0] * (1 - ls) + material_value[1] * ls
            case "geometric":
                return material_value[0] ** (1 - ls) * material_value[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / material_value[0] + ls / material_value[1])
            case _:
                raise ValueError(
                    "Method must be sharp, arithmetic, geometric, or harmonic."
                )


def field_interface(
    level_set: list, material_value: list, method: str
) -> fd.ufl.core.expr.Expr:
    """Executes field_interface_recursive with a modified argument.

    Calls field_interface_recursive using a copy of the level-set list to ensure the
    original one is not consumed by the function call.

    Args:
        level_set:
          A list of level-set UFL functions.
        material_value:
          A list of physical property values applicable to each material.
        method:
          A string specifying the nature of property transitions between materials.

    Returns:
        A UFL expression to calculate physical property values throughout the domain.
    """
    return field_interface_recursive(level_set.copy(), material_value, method)


def density_RaB(
    Simulation,
    level_set: list,
    func_space_interp: fd.functionspaceimpl.WithGeometry,
    method: Optional[str] = "sharp",
) -> tuple[
    fd.Constant,
    fd.Constant | fd.ufl.core.expr.Expr,
    fd.Function,
    fd.Constant | fd.ufl.core.expr.Expr,
    fd.Function,
    bool,
]:
    """Sets up buoyancy-related fields.

    Assigns UFL expressions to buoyancy-related fields based on the way the Material
    class was initialised.

    Args:
        Simulation:
          A class representing the current simulation.
        level_set:
          A list of level-set UFL functions.
        func_space_interp:
          A continuous UFL function space where material fields are calculated.
        method:
          An optional string specifying the nature of property transitions between
          materials.

    Returns:
        A tuple containing the reference density field, the density difference field,
        the density field, the UFL expression for the compositional Rayleigh number,
        the compositional Rayleigh number field, and a boolean indicating if the
        simulation is expressed in dimensionless form.

    Raises:
        ValueError: Inconsistent buoyancy-related field across materials.
    """
    density = fd.Function(func_space_interp, name="Density")
    RaB = fd.Function(func_space_interp, name="RaB")
    # Identify if the governing equations are written in dimensional form or not and
    # define accordingly relevant variables for the buoyancy term
    if all(material.density_B_RaB == "density" for material in Simulation.materials):
        dimensionless = False
        RaB_ufl = fd.Constant(1)
        ref_dens = fd.Constant(Simulation.reference_material.density)
        dens_diff = field_interface(
            level_set,
            [material.density - ref_dens for material in Simulation.materials],
            method=method,
        )
        density.interpolate(dens_diff + ref_dens)
    else:
        dimensionless = True
        ref_dens = fd.Constant(1)
        dens_diff = fd.Constant(1)
        if all(material.density_B_RaB == "B" for material in Simulation.materials):
            RaB_ufl = field_interface(
                level_set,
                [Simulation.Ra * material.B for material in Simulation.materials],
                method=method,
            )
        elif all(material.density_B_RaB == "RaB" for material in Simulation.materials):
            RaB_ufl = field_interface(
                level_set,
                [material.RaB for material in Simulation.materials],
                method=method,
            )
        else:
            raise ValueError(
                "All materials must share a common buoyancy-defining parameter."
            )
        RaB.interpolate(RaB_ufl)

    return ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless


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
