import abc
from .utility import ensure_constant

class BaseApproximation:
    """
    Base class to provide expressions in (Navier?)-Stokes + energy equations

    Basic assumption is that we are solving (to be extended when needed)

    div(dev_stress) + grad p + buoyancy(T, p) * khat = 0

    div(rho_continuity * u) = 0

    rhocp DT/Dt + linearized_energy_sink * T = div(kappa*grad(T)) + energy_source

    where the following are provided by Approximation methods:

    linearized_energy_sink = 0 (BA/EBA) or Di*rhobar*alphabar*w*T (ALA)
    kappa(), rho_continuity(), rhocp(), energy_source() are expressions independent of u, p and T'
    kappa() is diffusivity or conductivity depending on rhocp()
    compressible (property) False or True
    if compressible then dev_stess=mu*[sym(grad(u)-2/3 div(u()]
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

    @abc.abstractmethod
    def linearized_energy_sink(self):
        "UFL expression for (temperature dependent) sink terms in energy equation."
        pass

    @abc.abstractmethod
    def energy_source(self):
        "UFL expression for any other terms (not dependent on T or u) in energy equation"
        pass


class BoussinesqApproximation(BaseApproximation):
    compressible = False

    def __init__(self, Ra, kappa=1, g=1):
        self.Ra = ensure_constant(Ra)
        self._kappa = ensure_constant(kappa)
        self.g = ensure_constant(g)

    def buoyancy(self, p, T):
        return self.Ra * self.g * T

    def rho_continuity(self):
        return 1

    def rhocp(self):
        return 1

    def kappa(self):
        return self._kappa

    def linearized_energy_sink(self):
        return 0

    def energy_source(self):
        return 0
