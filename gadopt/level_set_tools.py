import abc
from dataclasses import dataclass, fields
from numbers import Number
from typing import Optional

import firedrake as fd
from firedrake.ufl_expr import extract_unique_domain

from .equations import BaseEquation, BaseTerm
from .scalar_equation import ScalarAdvectionEquation
from .time_stepper import eSSPRKs3p3

__all__ = [
    "LevelSetSolver",
    "Material",
    "density_RaB",
    "entrainment",
    "field_interface",
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
