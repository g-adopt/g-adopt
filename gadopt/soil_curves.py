"""
Soil curve models for Richards equation.

This module implements various soil-water retention and hydraulic conductivity
relationships commonly used in unsaturated flow modelling. These models describe
the relationship between hydraulic pressure head (h) and soil properties such
as moisture content and hydraulic conductivity.

References:
    - Haverkamp, R., et al. (1977). A comparison of numerical simulation models
      for one-dimensional infiltration. Soil Science Society of America Journal.
    - van Genuchten, M. T. (1980). A closed-form equation for predicting the
      hydraulic conductivity of unsaturated soils. Soil Science Society of
      America Journal.
"""

import firedrake as fd
import ufl
from abc import ABC, abstractmethod

from .utility import ensure_constant


class SoilCurve(ABC):
    """
    Abstract base class for soil curve models.

    Soil curves describe the relationship between hydraulic pressure head (h)
    and soil hydraulic properties. The hydraulic pressure head h is defined
    as the pressure head relative to atmospheric pressure, where:
    - h <= 0: unsaturated conditions (tension)
    - h > 0: saturated conditions (clamped to theta_s, K_s)

    All soil curve models must implement methods for:
    - moisture_content: $\theta(h)$ - volumetric water content
    - hydraulic_conductivity: $K(h)$ - hydraulic conductivity
    - water_retention: $C(h)$ - specific moisture capacity ($d\theta/dh$)

    All models require a specific storage coefficient Ss parameter.

    The van Genuchten and Haverkamp branches raise a saturation magnitude
    (|alpha h| or |h|) to a non-integer power. Differentiated branch-wise by
    UFL, that magnitude produces a derivative term that blows up at exactly
    h = 0 (pow(0, negative) = Inf, sign(0)*Inf = NaN), which would make the
    Newton Jacobian and the pyadjoint gradient NaN whenever a DOF sits exactly
    at the water table. The `reg_eps` parameter smoothly regularises the
    magnitude with sqrt(.^2 + eps^2) so the derivative stays bounded; the
    forward value is unchanged to O(reg_eps^2) away from h = 0. Set reg_eps=0
    to recover the exact but adjoint-singular form.
    """

    #: Default dimensionless smoothing of the saturation magnitude near h = 0.
    DEFAULT_REG_EPS = 1e-6

    def __init__(self, theta_r, theta_s, Ks, Ss, reg_eps=DEFAULT_REG_EPS, **kwargs):
        """
        Initialise soil curve with model parameters.

        Args:
            theta_r: Residual water content [-]
            theta_s: Saturated water content [-]
            Ks: Saturated hydraulic conductivity [L/T]
            Ss: Specific storage coefficient [1/L]
            reg_eps: Dimensionless smoothing of the saturation-magnitude near
                h = 0 that keeps the Jacobian and adjoint finite at the water
                table. Defaults to 1e-6. Set to 0 to disable (recovers the
                exact but adjoint-singular form). Unused by the exponential
                model, which is already smooth.
            **kwargs: Additional model-specific parameters

        Example:
            soil_curve = ExponentialCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5,
                                         Ss=0.0, alpha=0.328)
        """
        self.reg_eps = ensure_constant(reg_eps)

        # Build parameters dictionary from positional and keyword arguments.
        # Route Ss through ensure_constant so a bare float/int becomes a Constant
        # while a Function or Function(R) (e.g. a spatially varying storage or an
        # inversion control) passes through untouched and is never float()'d.
        params = {
            'theta_r': theta_r,
            'theta_s': theta_s,
            'Ks': Ks,
            'Ss': ensure_constant(Ss)
        }
        params.update(kwargs)

        # Wrap scalar values in fd.Constant for UFL compatibility;
        # pass through values that are already UFL expressions (e.g.
        # spatially varying fields for heterogeneous soil properties).
        self.parameters = {
            key: value if isinstance(value, ufl.core.expr.Expr) else fd.Constant(value)
            for key, value in params.items()
        }
        self._validate_parameters()

    @property
    def theta_r(self):
        """Residual water content [L^3/L^3]."""
        return self.parameters['theta_r']

    @property
    def theta_s(self):
        """Saturated water content [L^3/L^3]."""
        return self.parameters['theta_s']

    @property
    def Ks(self):
        """Saturated hydraulic conductivity [L/T]."""
        return self.parameters['Ks']

    @property
    def Ss(self):
        """Specific storage coefficient [1/L]."""
        return self.parameters['Ss']

    def _reg_abs(self, x, scale=1):
        r"""Smoothly regularised magnitude $\sqrt{x^2 + (\epsilon\,\text{scale})^2}$.

        This replaces ``abs(x)`` where ``x`` is raised to a non-integer power so
        that the branch-wise UFL derivative stays bounded at ``x = 0`` (no
        ``pow(0, negative) = Inf`` and no ``sign(0)`` factor). ``scale`` carries
        the natural magnitude of ``x`` so the dimensionless ``reg_eps`` produces
        an offset consistent with the units of ``x`` (e.g. a head scale ``1/alpha``
        when ``x`` is a head). When ``reg_eps`` is exactly zero this collapses back
        to the exact ``abs(x)``, letting users opt out of the regularisation.
        """
        # float() here only ever sees the default/opt-out Constant or number, so
        # it does not preclude reg_eps being a control: the regularised branch
        # below keeps reg_eps symbolic in the form.
        if float(self.reg_eps) == 0.0:
            return abs(x)
        eps = self.reg_eps * scale
        return fd.sqrt(x * x + eps * eps)

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate that all required parameters are provided."""
        pass

    @abstractmethod
    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Calculate volumetric moisture content $\theta(h)$.

        Args:
            h: Hydraulic pressure head $[L]$

        Returns:
            Volumetric moisture content $\theta$ $[L^3/L^3]$
        """
        pass

    @abstractmethod
    def hydraulic_conductivity(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Calculate relative hydraulic conductivity $K(h)$.

        Args:
            h: Hydraulic pressure head $[L]$

        Returns:
            Hydraulic conductivity $K$ $[L/T]$
        """
        pass

    @abstractmethod
    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""
        Calculate specific moisture capacity $C(h) = d\theta/dh$.

        Args:
            h: Hydraulic pressure head $[L]$

        Returns:
            Specific moisture capacity C $[L^{-1}]$
        """
        pass


class HaverkampCurve(SoilCurve):
    """
    Haverkamp soil curve model.

    The Haverkamp model provides empirical relationships for soil-water
    retention and hydraulic conductivity based on fitting parameters.

    Parameters:
        theta_r: Residual water content $[L^3/L^3]$
        theta_s: Saturated water content $[L^3/L^3]$
        alpha: Fitting parameter $[L^{-\\beta}]$
        beta: Fitting parameter [dimensionless]
        Ks: Saturated hydraulic conductivity $[L/T]$
        A: Fitting parameter $[L^{\\gamma}]$
        gamma: Fitting parameter [dimensionless]
        Ss: Specific storage coefficient $[L^{-1}]$
    """

    @property
    def alpha(self):
        """Haverkamp alpha fitting parameter."""
        return self.parameters['alpha']

    @property
    def beta(self):
        """Haverkamp beta fitting parameter."""
        return self.parameters['beta']

    @property
    def A(self):
        """Haverkamp A fitting parameter."""
        return self.parameters['A']

    @property
    def gamma(self):
        """Haverkamp gamma fitting parameter."""
        return self.parameters['gamma']

    def _validate_parameters(self) -> None:
        """Validate Haverkamp model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'beta', 'Ks', 'A', 'gamma', 'Ss']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def _abs_h_theta(self, h):
        r"""Regularised $|h|$ for the moisture branch (head scale $\alpha^{1/\beta}$)."""
        return self._reg_abs(h, scale=self.alpha**(1 / self.beta))

    def _abs_h_K(self, h):
        r"""Regularised $|h|$ for the conductivity branch (head scale $A^{1/\gamma}$)."""
        return self._reg_abs(h, scale=self.A**(1 / self.gamma))

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp moisture content relationship.

        $\theta(h) = \theta_r + \alpha(\theta_s - \theta_r) / (\alpha + |h|^{\beta})$
        """
        abs_h = self._abs_h_theta(h)
        theta = self.theta_r + self.alpha * (self.theta_s - self.theta_r) / (self.alpha + abs_h**self.beta)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp hydraulic conductivity relationship.

        $K(h) = K_s \\cdot A / (A + |h|^{\\gamma})$
        """
        abs_h = self._abs_h_K(h)
        K = self.Ks * (self.A / (self.A + abs_h**self.gamma))
        return fd.conditional(h <= 0, K, self.Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp specific moisture capacity.

        $C(h) = -\\text{sign}(h) \\cdot \\alpha \\cdot \\beta \\cdot (\\theta_s - \\theta_r) \\cdot |h|^{(\\beta-1)} / (\\alpha + |h|^{\\beta})^2$
        """
        # d theta/dh of the regularised moisture branch: with m(h) = sqrt(h^2+eps^2)
        # the chain rule gives dm/dh = h/m, so the singular sign(h)*|h|^(beta-1)
        # is replaced by the bounded (h/abs_h)*abs_h^(beta-1).
        abs_h = self._abs_h_theta(h)
        C = (-(h / abs_h) * self.alpha * self.beta * (self.theta_s - self.theta_r) *
             abs_h**(self.beta - 1) / (self.alpha + abs_h**self.beta)**2)
        return fd.conditional(h <= 0, C, 0)


class VanGenuchtenCurve(SoilCurve):
    """
    van Genuchten soil curve model.

    The van Genuchten model is widely used for describing soil-water retention
    and hydraulic conductivity relationships. It provides smooth, continuous
    curves with good physical interpretation.

    Parameters:
        theta_r: Residual water content $[L^3/L^3]$
        theta_s: Saturated water content $[L^3/L^3]$
        alpha: Inverse of air-entry pressure $[L^{-1}]$
        n: Pore-size distribution parameter [dimensionless]
        Ks: Saturated hydraulic conductivity $[L/T]$
        Ss: Specific storage coefficient $[L^{-1}]$
    """

    @property
    def alpha(self):
        """van Genuchten alpha parameter (inverse air-entry pressure)."""
        return self.parameters['alpha']

    @property
    def n(self):
        """van Genuchten n parameter (pore-size distribution)."""
        return self.parameters['n']

    def _validate_parameters(self) -> None:
        """Validate van Genuchten model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'n', 'Ks', 'Ss']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate parameter ranges when n is a scalar Constant
        n = self.parameters['n']
        if isinstance(n, fd.Constant) and n.values()[0] <= 1.0:
            raise ValueError("Parameter n must be > 1.0")

    def _abs_alpha_h(self, h):
        r"""Regularised dimensionless magnitude $|\alpha h|$.

        $\alpha h$ is already nondimensional, so the smoothing offset is the bare
        dimensionless ``reg_eps`` (``scale=1``).
        """
        return self._reg_abs(self.alpha * h)

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten moisture content relationship.

        $\theta(h) = \theta_r + (\theta_s - \theta_r) / (1 + |\alpha h|^n)^m$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n
        ah = self._abs_alpha_h(h)

        theta = self.theta_r + (self.theta_s - self.theta_r) / ((1 + ah**self.n)**m)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten hydraulic conductivity relationship.

        $K(h) = K_s \\cdot (1 - |\\alpha h|^{(n-1)} \\cdot (1 + |\\alpha h|^n)^{(-m)})^2 / (1 + |\\alpha h|^n)^{(m/2)}$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n
        ah = self._abs_alpha_h(h)

        term1 = 1 - ah**(self.n - 1) * (1 + ah**self.n)**(-m)
        term2 = (1 + ah**self.n)**(m/2)
        K = self.Ks * (term1**2 / term2)
        return fd.conditional(h <= 0, K, self.Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten specific moisture capacity.

        $C(h) = -(\\theta_s - \\theta_r) \\cdot n \\cdot m \\cdot h \\cdot \\alpha^n \\cdot |h|^{(n-2)} \\cdot (\\alpha^n \\cdot |h|^n + 1)^{(-m-1)}$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n
        ah = self._abs_alpha_h(h)

        # The singular factor is |h|^(n-2); rewrite it via the regularised
        # ah = |alpha h| using |h|^(n-2) = ah^(n-2) / alpha^(n-2). Then
        # h * alpha^n * |h|^(n-2) = (alpha*h) * alpha * ah^(n-2), which stays
        # bounded at h = 0. The leading (alpha*h) carries the correct sign.
        C = (-(self.theta_s - self.theta_r) * self.n * m * (self.alpha * h) *
             self.alpha * ah**(self.n - 2) * ((ah**self.n + 1)**(-m - 1)))
        return fd.conditional(h <= 0, C, 0)


class ExponentialCurve(SoilCurve):
    """
    Exponential soil curve model.

    The exponential model provides simple analytical relationships for
    soil-water retention and hydraulic conductivity. It is often used
    for preliminary analysis or when detailed soil characterization
    data is not available.

    Parameters:
        theta_r: Residual water content $[L^3/L^3]$
        theta_s: Saturated water content $[L^3/L^3]$
        alpha: Exponential decay parameter $[L^{-1}]$
        Ks: Saturated hydraulic conductivity $[L/T]$
        Ss: Specific storage coefficient $[L^{-1}]$
    """

    @property
    def alpha(self):
        """Exponential alpha decay parameter."""
        return self.parameters['alpha']

    def _validate_parameters(self) -> None:
        """Validate exponential model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'Ks', 'Ss']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential moisture content relationship.

        $\\theta(h) = \\theta_r + (\\theta_s - \\theta_r) \\cdot \\exp(\\alpha h)$
        """
        theta = self.theta_r + (self.theta_s - self.theta_r) * fd.exp(h * self.alpha)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential hydraulic conductivity relationship.

        $K(h) = K_s \\cdot \\exp(\\alpha h)$
        """
        K = self.Ks * fd.exp(h * self.alpha)
        return fd.conditional(h <= 0, K, self.Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential specific moisture capacity.

        $C(h) = (\\theta_s - \\theta_r) \\cdot \\alpha \\cdot \\exp(\\alpha h)$
        """
        C = (self.theta_s - self.theta_r) * fd.exp(h * self.alpha) * self.alpha
        return fd.conditional(h <= 0, C, 0)
