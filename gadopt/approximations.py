"""This module provides the Approximation class that emulates physical approximations
of fluid dynamics systems by exposing methods to calculate specific term components in
the corresponding mathematical equations. Users instantiate the class by providing
appropriate parameters and pass the instance to other objects, usually solvers. Under
the hood, G-ADOPT queries attributes and methods from the approximation.

"""

from numbers import Number

import firedrake as fd
from firedrake.ufl_expr import extract_unique_domain

from .utility import ensure_constant, vertical_component


class Approximation:
    """Constructs the equation approximation by defining relevant term components.

    Implemented approximations include Boussinesq (BA), Extended Boussinesq (EBA),
    Truncated Anelastic Liquid (TALA), and Anelastic Liquid (ALA).

    The system of conservation equations implemented is:
        - mass -> div(rho * u) = 0;
        - momentum -> -grad(p) + div(stress(u)) + buoyancy(p, T) * khat = 0;
        - energy -> rho * cp * DT/Dt + linearized_energy_sink(u) * T
          = div(k * grad(Tbar + T)) + energy_source(u);

    where the following terms are provided as methods:
        - buoyancy(p, T)
            = thermal_buoyancy * T - compressible_buoyancy * p - compositional_buoyancy
        - stress(u)
            = mu * sym(grad(u)) (BA/EBA)
            = mu * [sym(grad(u) - 2/3 div(u)] (TALA/ALA)
        - linearized_energy_sink(u)
            = 0 (BA)
            = Di * rho * alpha * g * w (EBA/TALA/ALA)
        - viscous_dissipation(u)
            = 0 (BA)
            = Di / Ra * inner(stress(u) * grad(u)) (EBA/TALA/ALA)
        - energy_source(u)
            = viscous_dissipation(u) + Q

    *Parameters (name: physical meaning)*
        1. Reference dimensional parameters
            H: specific heat source
            kappa: thermal diffusivity
            rho_diff: compositional density contrast
        2. Non-dimensional parameters
            Di: dissipation number
            Gamma: Grüneisen parameter
            Q: non-dimensional heat source
            Ra: Rayleigh number
            Ra_c: compositional Rayleigh number
        3. Reference profiles
            alpha: coefficient of thermal expansion (varies only in TALA and ALA)
            # Treatise on Geophysics uses incompressibility (i.e. bulk modulus).
            chi: isothermal compressibility
            cp: isobaric specific heat capacity (varies only in TALA and ALA)
            G: shear modulus (only used in VE)
            g: acceleration of gravity
            k: thermal conductivity
            mu: dynamic viscosity
            rho: densiy (varies only in TALA and ALA)
            T: temperature (varies only in TALA and ALA)

    Reference profiles represent parameters that admit depth-dependent variations along
    the hydrostatic adiabat. These parameters arise both in the dimensional and
    non-dimensional equation systems. Depending on the selected approximation, some
    parameters should have constant reference profiles. For example, only the
    acceleration of gravity, the dynamic viscosity, and the thermal conductivity should
    have depth-dependent variations in the Boussinesq approximation.
    """

    _presets = ["BA", "EBA", "TALA", "ALA", "VE"]
    _components = ["momentum", "mass", "energy"]

    _momentum_components = {"BA": ["compositional_buoyancy", "thermal_buoyancy"]}
    _momentum_components["EBA"] = _momentum_components["BA"] + []
    _momentum_components["TALA"] = _momentum_components["EBA"] + ["compressible_stress"]
    _momentum_components["ALA"] = _momentum_components["TALA"] + [
        "compressible_buoyancy"
    ]
    _momentum_components["VE"] = ["viscoelastic_buoyancy"]

    _mass_components = {"BA": []}
    _mass_components["EBA"] = _mass_components["BA"] + []
    _mass_components["TALA"] = _mass_components["EBA"] + ["mass_advection"]
    _mass_components["ALA"] = _mass_components["TALA"] + []
    _mass_components["VE"] = []

    _energy_components = {"BA": ["heat_source"]}
    _energy_components["EBA"] = _energy_components["BA"] + [
        "adiabatic_compression",
        "viscous_dissipation",
    ]
    _energy_components["TALA"] = _energy_components["EBA"] + []
    _energy_components["ALA"] = _energy_components["TALA"] + []

    def __init__(
        self,
        preset_or_components: str | dict[str, list[str]],
        *,
        dimensional: bool,
        parameters: dict[str, Number | fd.Constant | fd.Function] = {},
    ) -> None:
        if isinstance(preset_or_components, str):
            assert preset_or_components in self._presets, "Unknown preset provided."
            self.preset = preset_or_components

            if self.preset in self._momentum_components:
                self.momentum_components = self._momentum_components[self.preset]
            if self.preset in self._mass_components:
                self.mass_components = self._mass_components[self.preset]
            if self.preset in self._energy_components:
                self.energy_components = self._energy_components[self.preset]
        else:
            self.preset = None

            assert all(
                component in self._components for component in preset_or_components
            ), "Unknown component provided."

            if "momentum" in preset_or_components:
                self.momentum_components = preset_or_components["momentum"]
            if "mass" in preset_or_components:
                self.mass_components = preset_or_components["mass"]
            if "energy" in preset_or_components:
                self.energy_components = preset_or_components["energy"]

        self.dimensional = dimensional
        for parameter, value in parameters.items():
            setattr(self, parameter, ensure_constant(value))

        if not self.dimensional:
            self.check_reference_profiles()

        if hasattr(self, "momentum_components"):
            self.set_buoyancy()
            self.set_compressible_stress()

        if hasattr(self, "mass_components"):
            self.set_mass_advection()

        if hasattr(self, "energy_components"):
            self.check_thermal_diffusion()

            self.set_adiabatic_compression()
            self.set_heat_source()
            self.set_viscous_dissipation()

    def check_reference_profiles(self) -> None:
        """Ensures reference profiles are defined for non-dimensional systems."""
        for attribute in ["alpha", "chi", "cp", "G", "g", "k", "mu", "rho"]:
            if not hasattr(self, attribute):
                setattr(self, attribute, fd.Constant(1.0))
        if not hasattr(self, "T"):
            setattr(self, "T", fd.Constant(0.0))

    def check_thermal_diffusion(self) -> None:
        """Ensures compatibility between terms defining thermal diffusivity."""
        if hasattr(self, "kappa"):
            if not hasattr(self, "energy_components") or self.energy_components == [
                "heat_source"
            ]:
                self.k = self.kappa
                self.cp = 1 / self.rho

            if hasattr(self, "k") and not hasattr(self, "cp"):
                self.cp = self.k / self.rho / self.kappa
            elif not hasattr(self, "k") and hasattr(self, "cp"):
                self.k = self.kappa * self.rho * self.cp
            else:
                assert self.kappa == self.k / self.rho / self.cp
        elif hasattr(self, "k") and hasattr(self, "cp"):
            self.kappa = self.k / self.rho / self.cp

    def set_adiabatic_compression(self) -> None:
        """Defines the adiabatic compression term in the energy conservation."""
        if "adiabatic_compression" in self.energy_components:
            self.adiabatic_compression = self.rho * self.alpha * self.g
            if not self.dimensional:
                self.adiabatic_compression *= self.Di
        else:
            self.adiabatic_compression = 0.0

    def set_buoyancy(self) -> None:
        """Defines buoyancy terms in the momentum conservation."""
        self.compositional_buoyancy = 0.0
        if "compositional_buoyancy" in self.momentum_components:
            if self.dimensional and hasattr(self, "rho_diff"):
                self.compositional_buoyancy = self.rho_diff * self.g
            elif not self.dimensional and hasattr(self, "Ra_c"):
                self.compositional_buoyancy = self.Ra_c

        self.compressible_buoyancy = 0.0
        if "compressible_buoyancy" in self.momentum_components:
            # This term looks similar to the thermal contribution, but I cannot seem to
            # recover it using chapter 7.02 of Treatise on Geophysics. Relevant
            # discussion there starts from page 18. The article also mentions that cp
            # and cv should be equal under the anelastic liquid approximation.
            self.compressible_buoyancy = self.rho * self.chi * self.g
            if not self.dimensional:
                self.compressible_buoyancy *= self.Di / self.Gamma

        self.thermal_buoyancy = 0.0
        if "thermal_buoyancy" in self.momentum_components:
            if self.dimensional and hasattr(self, "alpha"):
                self.thermal_buoyancy = self.rho * self.alpha * self.g
            elif not self.dimensional and hasattr(self, "Ra"):
                self.thermal_buoyancy = self.Ra * self.rho * self.alpha * self.g

        self.viscoelastic_buoyancy = 0.0
        if "viscoelastic_buoyancy" in self.momentum_components:
            self.viscoelastic_buoyancy = fd.grad(self.rho) * self.g

    def set_compressible_stress(self) -> None:
        """Defines the compressible stress term in the momentum conservation."""
        if "compressible_stress" in self.momentum_components:
            self.compressible_stress = 2 / 3 * self.mu
        else:
            self.compressible_stress = 0.0

    def set_heat_source(self) -> None:
        """Defines the heat source term in the energy conservation."""
        self.heat_source = 0.0
        if "heat_source" in self.energy_components:
            if self.dimensional and hasattr(self, "H"):
                self.heat_source = self.rho * self.H
            elif not self.dimensional and hasattr(self, "Q"):
                self.heat_source = self.Q

    def set_mass_advection(self) -> None:
        """Defines the advection term in the mass conservation."""
        self.rho_mass = self.rho if "mass_advection" in self.mass_components else 1.0

    def set_viscous_dissipation(self) -> None:
        """Defines the viscous dissipation factor used in the energy conservation."""
        if "viscous_dissipation" in self.energy_components:
            self.viscous_dissipation_factor = 1.0
            if not self.dimensional:
                self.viscous_dissipation_factor *= self.Di / self.Ra
        else:
            self.viscous_dissipation_factor = 0.0

    def buoyancy(
        self,
        p: fd.ufl.indexed.Indexed,
        T: fd.Constant | fd.Function = fd.Constant(0),
        displ: fd.Constant | fd.Function = fd.Constant(0),
    ) -> fd.ufl.algebra.Sum:
        """Calculates the buoyancy term in the momentum conservation.

        Note: In a dimensional system, T represents the difference between actual
        temperature and reference temperature.
        """
        if self.dimensional and self.thermal_buoyancy != 0:
            T -= self.T
        return (
            self.thermal_buoyancy * T
            - self.compressible_buoyancy * p
            - self.compositional_buoyancy
            + fd.inner(self.viscoelastic_buoyancy, displ)
        )

    def energy_source(self, u: fd.ufl.indexed.Indexed) -> fd.ufl.algebra.Sum:
        """Calculates the energy source term in the energy conservation.

        Note: The energy source term includes the viscous dissipation contribution.
        """
        return self.viscous_dissipation(u) + self.heat_source

    def linearized_energy_sink(
        self, u: fd.ufl.indexed.Indexed
    ) -> fd.ufl.algebra.Product:
        """Calculates the energy sink term in the energy conservation."""
        return self.adiabatic_compression * vertical_component(u)

    def stress(
        self, u: fd.ufl.indexed.Indexed, stress_old: float | fd.Function = 0.0
    ) -> fd.ufl.algebra.Sum:
        """Calculates the stress term in momentum and energy conservations."""
        identity = fd.Identity(extract_unique_domain(u).geometric_dimension())

        stokes_part = 2 * self.mu * fd.sym(fd.grad(u))
        compressible_part = -self.compressible_stress * fd.div(u) * identity
        if isinstance(stress_old, float):
            stress_old *= identity

        return stokes_part + compressible_part + stress_old

    def viscoelastic_rheology(
        self, dt: float | fd.Constant | fd.Function
    ) -> fd.ufl.algebra.Division:
        maxwell_time = self.mu / self.G
        self.mu = self.mu / (maxwell_time + dt / 2)
        stress_scale = (maxwell_time - dt / 2) / (maxwell_time + dt / 2)

        return stress_scale

    def viscous_dissipation(self, u: fd.ufl.indexed.Indexed) -> fd.ufl.algebra.Product:
        """Calculates the viscous dissipation term in the energy conservation."""
        return self.viscous_dissipation_factor * fd.inner(self.stress(u), fd.grad(u))
