import abc

from firedrake import (
    Constant,
    FacetNormal,
    Function,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    conditional,
    div,
    dot,
    ds,
    dx,
    inner,
    max_value,
    min_value,
    sqrt,
)

from .equations import BaseEquation, BaseTerm
from .scalar_equation import ScalarAdvectionEquation


class AbstractMaterial(abc.ABC):
    """Abstract class to specify relevant material properties that affect simulation's
    evolution. Each material property is provided as a method to allow for temporal and
    spatial dependency."""

    @abc.abstractmethod
    def B():
        """Buoyancy number"""
        pass

    @abc.abstractmethod
    def RaB():
        """Compositional Rayleigh number, expressed as the product of the Rayleigh and
        Buoyancy numbers"""
        pass

    @abc.abstractmethod
    def density():
        """Dimensional density"""
        pass

    @abc.abstractmethod
    def viscosity():
        """Dimensional dynamic viscosity"""
        pass

    @abc.abstractmethod
    def thermal_expansion():
        """Dimensional coefficient of thermal expansion"""
        pass

    @abc.abstractmethod
    def thermal_conductivity():
        """Dimensional thermal conductivity"""
        pass

    @abc.abstractmethod
    def specific_heat_capacity():
        """Dimensional specific heat capacity at constant pressure"""
        pass

    @abc.abstractmethod
    def internal_heating_rate():
        """Dimensional internal heating rate per unit mass"""
        pass

    @classmethod
    def thermal_diffusivity(cls):
        """Dimensional thermal diffusivity"""
        return cls.thermal_conductivity() / cls.density() / cls.specific_heat_capacity()


class ReinitialisationTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        level_set_grad = fields["level_set_grad"]
        epsilon = fields["epsilon"]

        sharpen_term = -trial * (1 - trial) * (1 - 2 * trial) * test * self.dx
        balance_term = (
            epsilon
            * (1 - 2 * trial)
            * sqrt(level_set_grad[0] ** 2 + level_set_grad[1] ** 2)
            * test
            * self.dx
        )

        return sharpen_term + balance_term


class ReinitialisationEquation(BaseEquation):
    terms = [ReinitialisationTerm]


class LevelSetSolver:
    def __init__(
        self,
        level_set,
        velocity,
        tstep,
        tstep_alg,
        subcycles,
        reini_params,
        solver_params=None,
    ):
        func_space = level_set.function_space()
        self.mesh = func_space.mesh()
        self.level_set = level_set
        ls_fields = {"velocity": velocity}

        self.func_space_lsgp = VectorFunctionSpace(
            self.mesh, "CG", func_space.finat_element.degree
        )
        self.level_set_grad_proj = Function(
            self.func_space_lsgp,
            name=f"Level-set gradient (projection) #{level_set.name()[-1]}",
        )
        self.proj_solver = self.gradient_L2_proj()

        self.reini_params = reini_params
        reini_fields = {
            "level_set_grad": self.level_set_grad_proj,
            "epsilon": reini_params["epsilon"],
        }

        ls_eq = ScalarAdvectionEquation(func_space, func_space)
        reini_eq = ReinitialisationEquation(func_space, func_space)

        if solver_params is None:
            solver_params = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }

        self.ls_ts = tstep_alg(
            ls_eq,
            self.level_set,
            ls_fields,
            tstep / subcycles,
            solver_parameters=solver_params,
        )
        self.reini_ts = reini_params["tstep_alg"](
            reini_eq,
            self.level_set,
            reini_fields,
            reini_params["tstep"],
            solver_parameters=solver_params,
        )

        self.subcycles = subcycles

    def gradient_L2_proj(self):
        test_function = TestFunction(self.func_space_lsgp)
        trial_function = TrialFunction(self.func_space_lsgp)

        mass_term = inner(test_function, trial_function) * dx(domain=self.mesh)
        residual_element = self.level_set * div(test_function) * dx(domain=self.mesh)
        residual_boundary = (
            self.level_set
            * dot(test_function, FacetNormal(self.mesh))
            * ds(domain=self.mesh)
        )
        residual = -residual_element + residual_boundary
        problem = LinearVariationalProblem(
            mass_term, residual, self.level_set_grad_proj
        )

        solver_params = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        return LinearVariationalSolver(problem, solver_parameters=solver_params)

    def update_level_set_gradient(self, *args):
        self.proj_solver.solve()

    def solve(self, step):
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


def sharp_interface(level_set, material_value, method):
    ls = level_set.pop()

    if level_set:  # Directly specify material value on only one side of the interface
        return conditional(
            ls > 0.5,
            material_value.pop(),
            sharp_interface(level_set, material_value, method),
        )
    else:  # Final level set; specify values for both sides of the interface
        return conditional(ls < 0.5, *material_value)


def diffuse_interface(level_set, material_value, method):
    ls = max_value(min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match method:
            case "arithmetic":
                return material_value.pop() * ls + diffuse_interface(
                    level_set, material_value, method
                ) * (1 - ls)
            case "geometric":
                return material_value.pop() ** ls * diffuse_interface(
                    level_set, material_value, method
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / material_value.pop()
                    + (1 - ls) / diffuse_interface(level_set, material_value, method)
                )
    else:  # Final level set; specify values for both sides of the interface
        match method:
            case "arithmetic":
                return material_value[0] * (1 - ls) + material_value[1] * ls
            case "geometric":
                return material_value[0] ** (1 - ls) * material_value[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / material_value[0] + ls / material_value[1])


def density_RaB(Simulation, level_set, func_space_interp):
    density = Function(func_space_interp, name="Density")
    RaB = Function(func_space_interp, name="RaB")
    # Identify if the equations are written in dimensional form or not and define relevant
    # variables accordingly
    B_is_None = all(material.B() is None for material in Simulation.materials)
    RaB_is_None = all(material.RaB() is None for material in Simulation.materials)
    if B_is_None and RaB_is_None:
        RaB_ufl = Constant(1)
        ref_dens = Constant(Simulation.reference_material.density())
        dens_diff = sharp_interface(
            level_set.copy(),
            [material.density() - ref_dens for material in Simulation.materials],
            method="arithmetic",
        )
        density.interpolate(dens_diff + ref_dens)
        dimensionless = False
    elif RaB_is_None:
        ref_dens = Constant(1)
        dens_diff = Constant(1)
        RaB_ufl = diffuse_interface(
            level_set.copy(),
            [Simulation.Ra * material.B() for material in Simulation.materials],
            method="arithmetic",
        )
        RaB.interpolate(RaB_ufl)
        dimensionless = True
    elif B_is_None:
        ref_dens = Constant(1)
        dens_diff = Constant(1)
        RaB_ufl = sharp_interface(
            level_set.copy(),
            [material.RaB() for material in Simulation.materials],
            method="arithmetic",
        )
        RaB.interpolate(RaB_ufl)
        dimensionless = True
    else:
        raise ValueError("Providing B and RaB is redundant.")

    return ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless
