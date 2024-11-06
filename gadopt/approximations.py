"""This module provides the Approximation class that emulates physical approximations
of fluid dynamics systems by exposing methods to calculate specific term components in
the corresponding mathematical equations. Users instantiate the class by providing
appropriate parameters and pass the instance to other objects, usually solvers. Under
the hood, G-ADOPT queries attributes and methods from the approximation.

"""

from numbers import Number
from typing import Any

import firedrake as fd
from firedrake.ufl_expr import extract_unique_domain

from .utility import ensure_constant, vertical_component


class Approximation:
    """Constructs the equation approximation by defining relevant term components.

    Implemented approximations include Boussinesq (BA), Extended Boussinesq (EBA),
    Truncated Anelastic Liquid (TALA), and Anelastic Liquid (ALA). Additionally, an
    approximation for the small displacement viscoelastic limit (SDVA) is included (see
    below).

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

    Small Displacement Viscoelastic Approximation
    ---------------------------------------------
    By assuming a small displacement, we can linearise the problem, assuming a
    perturbation away from a reference state. We assume a Maxwell viscoelastic rheology,
    i.e. stress is the same but viscous and elastic strains combine linearly. We follow
    the approach by Zhong et al. (2003) redefining the problem in terms of incremental
    displacement, i.e. velocity * dt, where dt is the timestep. This produces a mixed
    Stokes system for incremental displacement and pressure which can be solved in the
    same way as mantle convection (where unknowns are velocity and pressure), with a
    modified viscosity and a stress term accounting for the deviatoric stress at the
    previous timestep.

    When using SDVA in a non-dimensional system, one must define a Weissenberg number
    (Wei). In the current G-ADOPT implementation of viscoelasticity, this number is
    expressed as:
    Wei = rho * g * d * (1 / G + dt / 2 / mu),
    where d is the characteristic length scale, dt is the simulation time step, and all
    other parameters are listed below.

    Zhong, S., Paulson, A., & Wahr, J. (2003).
    Three-dimensional finite-element modelling of Earth's viscoelastic deformation:
    effects of lateral variations in lithospheric thickness.
    Geophysical Journal International, 155(2), 679-695.

    *Parameters (name: physical meaning)*
        1. Reference dimensional parameters
            H: specific heat source
            kappa: thermal diffusivity
            rho_material: material (compositional) density
        2. Non-dimensional parameters
            Di: dissipation number
            Gamma: Grüneisen parameter
            Q: non-dimensional heat source
            Ra: Rayleigh number
            Ra_c: compositional Rayleigh number
            Wei: Weissenberg number (only used in SDVA)
        3. Reference profiles
            alpha: coefficient of thermal expansion (considered constant under BA)
            chi: isothermal compressibility
            cp: isobaric specific heat capacity (considered constant under BA)
            G: shear modulus (only used in SDVA)
            g: acceleration of gravity
            k: thermal conductivity
            mu: dynamic viscosity
            rho: density (considered constant under BA)
            T: temperature (considered constant under BA)

    Reference profiles represent parameters that admit depth-dependent variations along
    the hydrostatic adiabat. These parameters arise both in the dimensional and
    non-dimensional equation systems. Depending on the selected approximation, some
    parameters should have constant reference profiles. For example, only the
    acceleration of gravity, the dynamic viscosity, and the thermal conductivity should
    have depth-dependent variations under the Boussinesq approximation.
    """

    _presets = ["BA", "EBA", "TALA", "ALA", "SDVA"]
    _equations = ["momentum", "mass", "energy"]

    _momentum_components = {"BA": ["compositional_buoyancy", "thermal_buoyancy"]}
    _momentum_components["EBA"] = _momentum_components["BA"] + []
    _momentum_components["TALA"] = _momentum_components["EBA"] + ["compressible_stress"]
    _momentum_components["ALA"] = _momentum_components["TALA"] + [
        "compressible_buoyancy"
    ]
    _momentum_components["SDVA"] = ["viscoelastic_buoyancy"]

    _mass_components = {"BA": ["volume_continuity"]}
    _mass_components["EBA"] = _mass_components["BA"] + []
    _mass_components["TALA"] = ["mass_continuity"]
    _mass_components["ALA"] = _mass_components["TALA"] + []
    _mass_components["SDVA"] = ["volume_continuity"]

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
        /,
        *,
        dimensional: bool,
        parameters: dict[str, Number | fd.Constant | fd.Function] = {},
    ) -> None:
        if isinstance(preset_or_components, str):
            assert preset_or_components in self._presets, "Unknown preset provided."
            self.preset = preset_or_components

            for equation in self._equations:
                setattr(
                    self,
                    f"{equation}_components",
                    getattr(self, f"_{equation}_components").get(self.preset),
                )
        else:
            self.preset = None

            for equation, components in preset_or_components.items():
                assert equation in self._equations, "Unknown equation provided."
                setattr(self, f"{equation}_components", components)

        self.dimensional = dimensional
        for parameter, value in parameters.items():
            setattr(self, parameter, ensure_constant(value))

        if not self.dimensional:
            self.check_reference_profiles()

        if getattr(self, "momentum_components", None) is not None:
            self.set_buoyancy()
            self.set_compressible_stress()

        if getattr(self, "mass_components", None) is not None:
            self.set_flux_divergence()

        if getattr(self, "energy_components", None) is not None:
            self.check_thermal_diffusion()

            self.set_adiabatic_compression()
            self.set_heat_source()
            self.set_viscous_dissipation()

    def check_reference_profiles(self) -> None:
        """Ensures reference profiles are defined for non-dimensional systems."""
        for attribute in ["alpha", "chi", "cp", "G", "g", "k", "mu", "rho"]:
            if not hasattr(self, attribute):
                setattr(self, attribute, fd.Constant(1.0))

        for attribute in ["T"]:
            if not hasattr(self, attribute):
                setattr(self, attribute, fd.Constant(0.0))

    def check_thermal_diffusion(self) -> None:
        """Ensures compatibility between parameters defining thermal diffusivity."""
        if hasattr(self, "kappa"):
            if all(
                component not in self.energy_components
                for component in ["adiabatic_compression", "viscous_dissipation"]
            ):
                self.k = self.kappa
                self.cp = 1 / self.rho
            elif hasattr(self, "k") and not hasattr(self, "cp"):
                self.cp = self.k / self.rho / self.kappa
            elif not hasattr(self, "k") and hasattr(self, "cp"):
                self.k = self.kappa * self.rho * self.cp
            else:
                assert self.kappa == self.k / self.rho / self.cp

    def set_adiabatic_compression(self) -> None:
        """Defines the adiabatic compression factor in the energy conservation."""
        self.adiabatic_compression = 0.0
        if "adiabatic_compression" in self.energy_components:
            self.adiabatic_compression = self.rho * self.alpha * self.g
            if not self.dimensional:
                self.adiabatic_compression *= self.Di

    def set_buoyancy(self) -> None:
        """Defines buoyancy factors in the momentum conservation."""
        self.compositional_buoyancy = 0.0
        if "compositional_buoyancy" in self.momentum_components:
            if self.dimensional and hasattr(self, "rho_material"):
                self.compositional_buoyancy = (self.rho_material - self.rho) * self.g
            elif not self.dimensional and hasattr(self, "Ra_c"):
                self.compositional_buoyancy = self.Ra_c * self.g

        self.compressible_buoyancy = 0.0
        if "compressible_buoyancy" in self.momentum_components:
            # How does one obtain this term using chapter 7.02 of Treatise on
            # Geophysics? The article mentions that specific heats should be equal under
            # the anelastic liquid approximation.
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
            if not self.dimensional:
                self.viscoelastic_buoyancy *= self.Wei

    def set_compressible_stress(self) -> None:
        """Defines the compressible stress factor in the momentum conservation."""
        self.compressible_stress = 0.0
        if "compressible_stress" in self.momentum_components:
            self.compressible_stress = 2 / 3 * self.mu

    def set_heat_source(self) -> None:
        """Defines the heat source term in the energy conservation."""
        self.heat_source = 0.0
        if "heat_source" in self.energy_components:
            if self.dimensional and hasattr(self, "H"):
                self.heat_source = self.rho * self.H
            elif not self.dimensional and hasattr(self, "Q"):
                self.heat_source = self.Q

    def set_flux_divergence(self) -> None:
        """Defines the density contribution to the flux in the mass conservation."""
        if "volume_continuity" in self.mass_components:
            self.rho_flux = 1.0
        elif "mass_continuity" in self.mass_components:
            self.rho_flux = self.rho
        else:
            raise ValueError("Unknown continuity equation.")

    def set_viscous_dissipation(self) -> None:
        """Defines the viscous dissipation factor used in the energy conservation."""
        self.viscous_dissipation_factor = 0.0
        if "viscous_dissipation" in self.energy_components:
            self.viscous_dissipation_factor = 1.0
            if not self.dimensional:
                self.viscous_dissipation_factor *= self.Di / self.Ra

    def buoyancy(
        self,
        *,
        p: fd.ufl.indexed.Indexed | float = 0.0,
        T: fd.Function | float = 0.0,
        displ: fd.Function | float = 0.0,
        params_fs: dict[str, Any] | None = None,
    ) -> fd.ufl.algebra.Sum:
        """Calculates the buoyancy term in the momentum conservation.

        Either returns the buoyancy term for the interior of the domain or that of the
        free surface described by `params_fs`.

        Note: In a dimensional system, T represents the difference between actual
        temperature and reference temperature.
        """
        if self.dimensional and self.thermal_buoyancy != 0:
            T -= self.T

        buoyancy = (  # Calculates domain interior buoyancy
            self.thermal_buoyancy * T
            - self.compressible_buoyancy * p
            - self.compositional_buoyancy
            + fd.inner(self.viscoelastic_buoyancy, displ)
        )

        if params_fs is not None:  # Calculates free-surface buoyancy
            buoyancy_free_surface = (self.rho - params_fs.get("rho_ext", 0)) * self.g
            if not self.dimensional:
                buoyancy_free_surface *= params_fs["Ra_fs"]
            if params_fs.get("include_buoyancy_effects", True):
                buoyancy_free_surface -= buoyancy

            return buoyancy_free_surface

        return buoyancy

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
        """Calculates the stress term in momentum and energy conservations.

        Note: In the SDVA, the stress term includes its value at the previous time step.
        """
        identity = fd.Identity(extract_unique_domain(u).geometric_dimension())

        stokes_part = 2 * self.mu * fd.sym(fd.grad(u))
        compressible_part = -self.compressible_stress * fd.div(u) * identity
        if isinstance(stress_old, float):
            stress_old *= identity

        return stokes_part + compressible_part + stress_old

    def viscous_dissipation(self, u: fd.ufl.indexed.Indexed) -> fd.ufl.algebra.Product:
        """Calculates the viscous dissipation term in the energy conservation."""
        return self.viscous_dissipation_factor * fd.inner(self.stress(u), fd.grad(u))
