"""This module provides the EquationSystem class that emulates physical approximations
of fluid dynamics systems by exposing methods to calculate specific terms in the
corresponding mathematical equations. Users instantiate the appropriate class by
providing relevant parameters and pass the instance to other objects, such as solvers.
Under the hood, G-ADOPT queries attributes and methods from the approximation.

"""

from firedrake import Identity, div, grad, inner, sym
from firedrake.ufl_expr import extract_unique_domain

from .utility import ensure_constant, vertical_component


class EquationSystem:
    """Constructs the system of equations to be solved by defining relevant terms.

    The system of conservation equations implemented is:
        - mass -> div(rho * u) = 0;
        - momentum -> -grad(p) + div(stress(u)) + buoyancy(p, T) * khat = 0;
        - energy -> rho * cp * DT/Dt + linearized_energy_sink(u) * T
          = div(k * grad(Tbar + T)) + energy_source(u);

    where the following terms are provided as methods:
        - linearized_energy_sink(u) = 0 (BA), Di * rhobar * alphabar * g * w (EBA),
          or Di * rhobar * alphabar * w (TALA/ALA)
        - Tbar is 0 or reference temperature profile (ALA)
        - stress(u) depends on the compressible attribute:
            - if compressible then stress(u) = mu * [sym(grad(u) - 2/3 div(u)]
            - if not compressible then stress(u) = mu * sym(grad(u))

    *Parameters (name: physical meaning)*
        1. Reference dimensional parameters
            # Treatise on Geophysics makes use of incompressibility (i.e. bulk modulus).
            chi: isothermal compressibility
            H: specific heat source
            kappa: thermal diffusivity
            rho_diff: compositional density contrast
            rho_diff_fs: free-surface density contrast
        2. Non-dimensional parameters
            Di: dissipation number
            Gamma: Grüneisen parameter
            Q: non-dimensional heat source
            Ra: Rayleigh number
            Ra_c: compositional Rayleigh number
            Ra_fs: free-surface Rayleigh number
        3. Reference profiles
            alpha: coefficient of thermal expansion
            cp: isobaric specific heat capacity
            g: acceleration of gravity
            k: thermal conductivity
            mu: dynamic viscosity
            rho: densiy
            T: temperature

    Reference profiles represent parameters that admit depth-dependent variations along
    the hydrostatic adiabat. These parameters arise both in the dimensional and
    non-dimensional equation systems. Depending on the selected approximation, some
    parameters should have constant reference profiles. For example, only the
    acceleration of gravity, the dynamic viscosity, and the thermal conductivity should
    have depth-dependent variations in the Boussinesq approximation.
    """

    _approximations = ["BA", "EBA", "TALA", "ALA"]
    _impl_buoyancy_terms = ["compositional", "free_surface", "thermal"]

    def __init__(
        self,
        approximation: str,
        dimensional: bool,
        *,
        parameters: dict[str, float] = {},
        buoyancy_terms: list[str] = [],
    ):
        assert approximation in self._approximations
        assert all(term in self._impl_buoyancy_terms for term in buoyancy_terms)

        self.approximation = approximation
        self.dimensional = dimensional
        self.buoyancy_terms = buoyancy_terms
        for parameter, value in parameters.items():
            setattr(self, parameter, ensure_constant(value))

        self.compressible = "ALA" in approximation

        self.check_reference_profiles()
        if "thermal" in buoyancy_terms:
            self.check_thermal_diffusion()

        self.set_buoyancy()
        self.set_adiabatic_compression()
        self.set_viscous_dissipation_factor()
        self.set_heat_source()

        self.compressible_stress = 2 / 3 * self.mu if self.compressible else 0

    def check_reference_profiles(self):
        """Ensures reference profiles are defined."""
        if not self.dimensional:
            for attribute in ["alpha", "chi", "cp", "g", "k", "mu", "rho"]:
                if not hasattr(self, attribute):
                    setattr(self, attribute, ensure_constant(1))
            if not hasattr(self, "T"):
                setattr(self, "T", ensure_constant(0))

    def check_thermal_diffusion(self):
        """Ensures compatibility between terms defining thermal diffusivity."""
        if hasattr(self, "kappa"):
            if hasattr(self, "k") and not hasattr(self, "cp"):
                self.cp = self.k / self.rho / self.kappa
            elif not hasattr(self, "k") and hasattr(self, "cp"):
                self.k = self.kappa * self.rho * self.cp
            elif self.approximation == "BA":
                self.k = self.kappa
                self.cp = 1 / self.rho
            else:
                assert self.kappa == self.k / self.rho / self.cp
        else:
            self.kappa = self.k / self.rho / self.cp

    def set_adiabatic_compression(self):
        """Defines the adiabatic compression term in the energy conservation."""
        if self.approximation == "BA":
            self.adiabatic_compression = 0
        else:
            self.adiabatic_compression = self.rho * self.alpha * self.g
            if not self.dimensional:
                self.adiabatic_compression *= self.Di

    def set_buoyancy(self):
        """Defines buoyancy terms in the momentum conservation."""
        if "compositional" in self.buoyancy_terms:
            if self.dimensional:
                self.compositional_buoyancy = self.rho_diff * self.g
            else:
                self.compositional_buoyancy = self.Ra_c
        else:
            self.compositional_buoyancy = 0

        if "free_surface" in self.buoyancy_terms:
            if self.dimensional:
                self.free_surface_buoyancy = self.rho_diff_fs * self.g
            else:
                self.free_surface_buoyancy = self.Ra_fs
        else:
            self.free_surface_buoyancy = 0

        if "thermal" in self.buoyancy_terms:
            self.thermal_buoyancy = self.rho * self.alpha * self.g
            if not self.dimensional:
                self.thermal_buoyancy *= self.Ra
        else:
            self.thermal_buoyancy = 0

        if self.approximation == "ALA":
            # How is this contribution calculated? I cannot seem to recover it using
            # chapter 7.02 of Treatise on Geophysics. Relevant discussion there starts
            # from page 18. The article also mentions that cp and cv should be equal
            # under the anelastic liquid approximation.
            self.compressible_buoyancy = self.rho * self.chi * self.g
            if not self.dimensional:
                self.compressible_buoyancy *= self.Di / self.Gamma
        else:
            self.compressible_buoyancy = 0

    def set_heat_source(self):
        """Defines the heat source term in the energy conservation."""
        if self.dimensional and hasattr(self, "H"):
            self.heat_source = self.rho * self.H
        elif not self.dimensional and hasattr(self, "Q"):
            self.heat_source = self.Q
        else:
            self.heat_source = 0

    def set_viscous_dissipation_factor(self):
        """Defines the viscous dissipation factor used in the energy conservation."""
        if self.approximation == "BA":
            self.visc_dis_factor = 0
        else:
            self.visc_dis_factor = 1 if self.dimensional else self.Di / self.Ra

    def buoyancy(self, p: float, T: float) -> float:
        """Calculates the buoyancy term in the momentum conservation.

        Note: In a dimensional system, T represents the difference between temperature
        and reference temperature.
        """
        if self.dimensional and "thermal" in self.buoyancy_terms:
            T -= self.T
        return (
            self.thermal_buoyancy * T
            - self.compressible_buoyancy * p
            - self.compositional_buoyancy
        )

    def energy_source(self, u: float) -> float:
        """Calculates the energy source term in the energy conservation.

        Note: Here, the energy source term includes the viscous dissipation
        contribution."""
        return self.viscous_dissipation(u) + self.heat_source

    def free_surface_normal_stress(
        self, p: float, T: float, eta: float, *, variable_rho_fs: bool
    ) -> float:
        """Calculates the free-surface normal stress in the momentum conservation.

        Note: This term only affects the momentum balance at physical boundaries
        representing a free surface."""
        normal_stress = self.free_surface_buoyancy * eta
        if variable_rho_fs:
            normal_stress -= self.buoyancy(p, T) * eta

        return normal_stress

    def linearized_energy_sink(self, u: float) -> float:
        """Calculates the energy sink term in the energy conservation."""
        return self.adiabatic_compression * vertical_component(u)

    def stress(self, u: float) -> float:
        """Calculates the stress term in momentum and energy conservations."""
        identity = Identity(extract_unique_domain(u).topological_dimension())
        return 2 * self.mu * sym(grad(u)) - self.compressible_stress * div(u) * identity

    def viscous_dissipation(self, u: float) -> float:
        """Calculates the viscous dissipation term in the energy conservation."""
        return self.visc_dis_factor * inner(self.stress(u), grad(u))
