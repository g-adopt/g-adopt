from dataclasses import dataclass, fields
from typing import Optional

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


@dataclass(kw_only=True)
class Material:
    """A material defined by physical properties and compatible with a level set."""

    density: Optional[float] = None  # Reference density
    B: Optional[float] = None  # Buoyancy number
    RaB: Optional[float] = None  # Compositional Rayleigh number

    def __post_init__(self):
        count_float = 0
        count_None = 0
        for field_var in fields(self):
            if isinstance(field_var_value := getattr(self, field_var.name), float):
                count_float += 1
                self.density_B_RaB = field_var.name
            elif field_var_value is None:
                count_None += 1
        if count_float != 1 and count_None != 2:
            raise ValueError(
                "One, and only one, of density, B, and RaB must be provided, and it"
                " must be a float"
            )

    @staticmethod
    def viscosity(*args, **kwargs):
        """Dynamic viscosity (Pa s)"""
        return 1.0

    @staticmethod
    def thermal_expansion(*args, **kwargs):
        """Volumetric thermal expansion coefficient (K^-1)"""
        return 1.0

    @staticmethod
    def thermal_conductivity(*args, **kwargs):
        """Thermal conductivity (W m^-1 K^-1)"""
        return 1.0

    @staticmethod
    def specific_heat_capacity(*args, **kwargs):
        """Specific heat capacity at constant pressure (J kg^-1 K^-1)"""
        return 1.0

    @staticmethod
    def internal_heating_rate(*args, **kwargs):
        """Internal heating rate per unit mass (W kg^-1)"""
        return 0.0

    @classmethod
    def thermal_diffusivity(cls, *args, **kwargs):
        "Thermal diffusivity (m^2 s^-1)"
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
    # Identify if the governing equations are written in dimensional form or not and
    # define accordingly relevant variables for the buoyancy term
    if all(material.density_B_RaB == "density" for material in Simulation.materials):
        RaB_ufl = Constant(1)
        ref_dens = Constant(Simulation.reference_material.density)
        dens_diff = sharp_interface(
            level_set.copy(),
            [material.density - ref_dens for material in Simulation.materials],
            method="arithmetic",
        )
        density.interpolate(dens_diff + ref_dens)
        dimensionless = False
    elif all(material.density_B_RaB == "B" for material in Simulation.materials):
        ref_dens = Constant(1)
        dens_diff = Constant(1)
        RaB_ufl = diffuse_interface(
            level_set.copy(),
            [Simulation.Ra * material.B for material in Simulation.materials],
            method="arithmetic",
        )
        RaB.interpolate(RaB_ufl)
        dimensionless = True
    elif all(material.density_B_RaB == "RaB" for material in Simulation.materials):
        ref_dens = Constant(1)
        dens_diff = Constant(1)
        RaB_ufl = sharp_interface(
            level_set.copy(),
            [material.RaB for material in Simulation.materials],
            method="arithmetic",
        )
        RaB.interpolate(RaB_ufl)
        dimensionless = True
    else:
        raise ValueError(
            "All materials must be initialised using the same buoyancy-related variable"
        )

    return ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless
