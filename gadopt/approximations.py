import abc
from firedrake import sym, grad, inner, div, Identity
from .utility import ensure_constant, vertical_component


class BaseApproximation(abc.ABC):
    """
    Base class to provide expressions in (Navier?)-Stokes + energy equations

    Basic assumption is that we are solving (to be extended when needed)

    div(dev_stress) + grad p + buoyancy(T, p) * khat = 0

    div(rho_continuity * u) = 0

    rhocp DT/Dt + linearized_energy_sink(u) * T = div(kappa*grad(Tbar + T)) + energy_source(u)

    where the following are provided by Approximation methods:

    linearized_energy_sink(u) = 0 (BA/EBA) or Di*rhobar*alphabar*w (ALA)
    kappa() is diffusivity or conductivity depending on rhocp()
    Tbar (property) is 0 or reference temperature profile (ALA)
    compressible (property) False or True
    if compressible then dev_stress=mu*[sym(grad(u)-2/3 div(u()]
    if not compressible then dev_stress=mu*sym(grad(u)) and rho_continuity is assumed to be 1
    """
    @property
    @abc.abstractmethod
    def compressible(self):
        "Whether approximation is compressible (True/False)"
        pass

    @abc.abstractmethod
    def buoyancy(self, p, T):
        "UFL expression for buoyancy (momentum source in gravity direction)"
        pass

    @abc.abstractmethod
    def rho_continuity(self):
        "UFL expression for density in mass continuity equation (=1 for incompressible"
        pass

    @abc.abstractmethod
    def rhocp(self):
        "UFL expression expression for coefficient in front of DT/dt in energy eequation"
        pass

    @abc.abstractmethod
    def kappa(self):
        "UFL expression for diffusivity/conductivity"
        pass

    @property
    @abc.abstractmethod
    def Tbar(self):
        "Reference temperature profile"
        pass

    @abc.abstractmethod
    def linearized_energy_sink(self, u):
        "UFL expression for (temperature dependent) sink terms in energy equation."
        pass

    @abc.abstractmethod
    def energy_source(self, u):
        "UFL expression for any other terms (not dependent on T or u) in energy equation"
        pass


class BoussinesqApproximation(BaseApproximation):
    """Boussinesq approximation:

    Small density variation linear in Temperature only, only taken into account in buoyancy term.
    All references rho, cp, alpha are constant and typically incorporated in Ra
    Viscous dissipation is neglected (Di << 1)."""
    compressible = False

    def __init__(self, Ra, kappa=1, g=1, rho=1, alpha=1):
        """
        :arg Ra:   Rayleigh number
        :arg kappa, g, rho, alpha:  Diffusivity, gravitational acceleration, reference density and thermal expansion coefficient
                                    Normally kept at 1 when non-dimensionalised."""
        self.Ra = ensure_constant(Ra)
        self._kappa = ensure_constant(kappa)
        self.g = ensure_constant(g)
        self.rho = ensure_constant(rho)
        self.alpha = ensure_constant(alpha)

    def buoyancy(self, p, T):
        return self.Ra * self.g * self.alpha * self.rho * T

    def rho_continuity(self):
        return 1

    def rhocp(self):
        return 1

    def kappa(self):
        return self._kappa

    Tbar = 0

    def linearized_energy_sink(self, u):
        return 0

    def energy_source(self, u):
        return 0


class ExtendedBoussinesqApproximation(BoussinesqApproximation):
    """
    Extended Boussinesq

    As Boussinesq but includes viscous dissipation and work against gravity (both scaled with Di)."""
    compressible = False

    def __init__(self, Ra, Di, mu=1, H=None, cartesian=True, **kwargs):
        """
        :arg Ra: Rayleigh number
        :arg Di: Dissipation number
        :arg mu: Viscosity used in viscous dissipation
        :arg H:  Volumetric heat production
        :arg cartesian:  True: gravity points in negative z-direction, False: gravity points radially inward
        :arg kappa, g, rho, alpha:  Diffusivity, gravitational acceleration, reference density and thermal expansion coefficient
                                    Normally kept at 1 when non-dimensionalised."""

        super().__init__(Ra, **kwargs)
        self.Di = Di
        self.mu = mu
        self.H = H
        self.cartesian = cartesian

    def viscous_dissipation(self, u):
        stress = 2 * self.mu * sym(grad(u))
        if self.compressible:  # (used in AnelasticLiquidApproximations below)
            stress -= 2/3 * self.mu * div(u) * Identity(u.ufl_shape[0])
        phi = inner(stress, grad(u))
        return phi * self.Di / self.Ra

    def work_against_gravity(self, u, T):
        w = vertical_component(u, self.cartesian)
        return self.Di * self.alpha * self.rho * self.g * w * T

    def linearized_energy_sink(self, u):
        return self.work_against_gravity(u, 1)

    def energy_source(self, u):
        source = self.viscous_dissipation(u)
        if self.H:
            source += self.H * self.rho

        return source


class TruncatedAnelasticLiquidApproximation(ExtendedBoussinesqApproximation):
    """
    Truncated Anelastic Liquid Approximation

    Compressible approximation. Excludes linear dependence of density on pressure (chi)"""
    compressible = True

    def __init__(self, Ra, Di,
                 Tbar=0, chi=1, cp=1,
                 gamma0=1, cp0=1, cv0=1,
                 **kwargs):
        """
        :arg Ra:   Rayleigh number
        :arg Di:   Dissipation number
        Reference values that may be depth-dependent (default to 1 if not supplied):
        :arg rho:  reference density
        :arg alpha: reference thermal expansion coefficient
        :arg Tbar: reference temperature. In the diffusion term we use Tbar + T (i.e. T is the pertubartion) - default 0
        :arg chi:  reference isothermal compressibility
        :arg cp:   reference specific heat at constant pressure
        :arg mu:   Viscosity used in viscous dissipation
        :arg H:    Volumetric heat production - default 0
        :arg cartesian:  True: gravity points in negative z-direction, False: gravity points radially inward
        :arg kappa, g:  Diffusivity, gravitational acceleration
        Constant coefficients in pressure dependent buoyancy term::w
        :arg gamma0:   Gruneisen number
        :arg cp0, cv0: specific heat at constant pressure and volume reference for entire Mantle (all references above may be depth-dependent)."""
        super().__init__(Ra, Di, **kwargs)
        self.Tbar = Tbar
        # Equation of State:
        self.chi = chi
        self.cp = cp
        assert 'g' not in kwargs
        self.gamma0, self.cp0, self.cv0 = gamma0, cp0, cv0

    def rho_continuity(self):
        return self.rho

    def rhocp(self):
        return self.rho * self.cp

    def linearized_energy_sink(self, u):
        w = vertical_component(u, self.cartesian)
        return self.Di * self.rho * self.alpha * w


class AnelasticLiquidApproximation(TruncatedAnelasticLiquidApproximation):
    """
    Anelastic Liquid Approximation

    Compressible approximation. Includes linear dependence of density on pressure (chi)"""

    def buoyancy(self, p, T):
        pressure_part = -self.Di * self.cp0 / self.cv0 / self.gamma0 * self.g * self.rho * self.chi * p
        temperature_part = super().buoyancy(p, T)
        return pressure_part + temperature_part
