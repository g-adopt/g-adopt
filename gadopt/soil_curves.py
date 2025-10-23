"""
Soil curve models for Richards equation.

This module implements various soil-water retention and hydraulic conductivity
relationships commonly used in unsaturated flow modelling. These models describe
the relationship between hydraulic pressure head (h) and soil properties such
as moisture content and relative permeability.

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
from typing import Dict, Any


class SoilCurve(ABC):
    """
    Abstract base class for soil curve models.

    Soil curves describe the relationship between hydraulic pressure head (h)
    and soil hydraulic properties. The hydraulic pressure head h is defined
    as the pressure head relative to atmospheric pressure, where:
    - h < 0: unsaturated conditions (tension)
    - h <= 0: saturated conditions

    All soil curve models must implement methods for:
    - moisture_content: $\theta(h)$ - volumetric water content
    - relative_permeability: $K(h)$ - hydraulic conductivity
    - water_retention: $C(h)$ - specific moisture capacity ($d\theta/dh$)

    All models require a specific storage coefficient Ss parameter.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialise soil curve with model parameters.

        Args:
            parameters: Dictionary containing model-specific parameters.
                       Must include 'Ss' (specific storage coefficient).
        """
        # Convert all parameters to fd.Constant for UFL compatibility
        self.parameters = {key: fd.Constant(value) for key, value in parameters.items()}
        self._validate_parameters()

        # Ensure Ss is provided
        if 'Ss' not in self.parameters:
            raise ValueError("Parameter 'Ss' (specific storage coefficient) is required")

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
    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
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
        f"""
        Calculate specific moisture capacity $ C(h) = d\theta/dh $.

        Args:
            h: Hydraulic pressure head $[ L ]$

        Returns:
            Specific moisture capacity C $ [ L^{-1} ] $
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
            if not isinstance(self.parameters[param], fd.Constant):
                raise ValueError(f"Parameter {param} must be a fd.Constant")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp moisture content relationship.

        $\theta(h) = \theta_r + \alpha(\theta_s - \theta_r) / (\alpha + |h|^{\beta})$
        """
        theta = self.theta_r + self.alpha * (self.theta_s - self.theta_r) / (self.alpha + abs(h)**self.beta)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp relative permeability relationship.

        $K(h) = K_s \\cdot A / (A + |h|^{\\gamma})$
        """
        K = self.Ks * (self.A / (self.A + abs(h)**self.gamma))
        return fd.conditional(h <= 0, K, self.Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp specific moisture capacity.

        $C(h) = -\\text{sign}(h) \\cdot \\alpha \\cdot \\beta \\cdot (\\theta_s - \\theta_r) \\cdot |h|^{(\\beta-1)} / (\\alpha + |h|^{\\beta})^2$
        """
        C = (-fd.sign(h) * self.alpha * self.beta * (self.theta_s - self.theta_r) *
             abs(h)**(self.beta - 1) / (self.alpha + abs(h)**self.beta)**2)
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
            if not isinstance(self.parameters[param], fd.Constant):
                raise ValueError(f"Parameter {param} must be a fd.Constant")

        # Validate parameter ranges (check the value of the Constant)
        if self.parameters['n'].values()[0] <= 1.0:
            raise ValueError("Parameter n must be > 1.0")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten moisture content relationship.

        $\theta(h) = \theta_r + (\theta_s - \theta_r) / (1 + |\alpha h|^n)^m$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n

        theta = self.theta_r + (self.theta_s - self.theta_r) / ((1 + abs(self.alpha * h)**self.n)**m)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten relative permeability relationship.

        $K(h) = K_s \\cdot (1 - |\\alpha h|^{(n-1)} \\cdot (1 + |\\alpha h|^n)^{(-m)})^2 / (1 + |\\alpha h|^n)^{(m/2)}$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n

        term1 = 1 - abs(self.alpha * h)**(self.n - 1) * (1 + abs(self.alpha * h)**self.n)**(-m)
        term2 = (1 + abs(self.alpha * h)**self.n)**(m/2)
        K = self.Ks * (term1**2 / term2)
        return fd.conditional(h <= 0, K, self.Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten specific moisture capacity.

        $C(h) = -(\\theta_s - \\theta_r) \\cdot n \\cdot m \\cdot h \\cdot \\alpha^n \\cdot |h|^{(n-2)} \\cdot (\\alpha^n \\cdot |h|^n + 1)^{(-m-1)}$
        where $m = 1 - 1/n$
        """
        m = 1 - 1/self.n

        C = (-(self.theta_s - self.theta_r) * self.n * m * h * (self.alpha**self.n) *
             abs(h)**(self.n - 2) * ((self.alpha**self.n * abs(h)**self.n + 1)**(-m - 1)))
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
            if not isinstance(self.parameters[param], fd.Constant):
                raise ValueError(f"Parameter {param} must be a fd.Constant")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential moisture content relationship.

        $\\theta(h) = \\theta_r + (\\theta_s - \\theta_r) \\cdot \\exp(\\alpha h)$
        """
        theta = self.theta_r + (self.theta_s - self.theta_r) * fd.exp(h * self.alpha)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential relative permeability relationship.

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
