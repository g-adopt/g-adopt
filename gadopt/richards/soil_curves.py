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
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialise soil curve with model parameters.

        Args:
            parameters: Dictionary containing model-specific parameters
        """
        # Convert all parameters to fd.Constant for UFL compatibility
        self.parameters = {key: fd.Constant(value) for key, value in parameters.items()}
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate that all required parameters are provided."""
        pass

    @abstractmethod
    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Calculate volumetric moisture content θ(h).

        Args:
            h: Hydraulic pressure head [L]

        Returns:
            Volumetric moisture content θ [L³/L³]
        """
        pass

    @abstractmethod
    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Calculate relative hydraulic conductivity K(h).

        Args:
            h: Hydraulic pressure head [L]

        Returns:
            Hydraulic conductivity K [L/T]
        """
        pass

    @abstractmethod
    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Calculate specific moisture capacity C(h) = dθ/dh.

        Args:
            h: Hydraulic pressure head [L]

        Returns:
            Specific moisture capacity C [L⁻¹]
        """
        pass


class HaverkampCurve(SoilCurve):
    """
    Haverkamp soil curve model.

    The Haverkamp model provides empirical relationships for soil-water
    retention and hydraulic conductivity based on fitting parameters.

    Parameters:
        theta_r: Residual water content $[L^3/L^3]$
        theta_s: Saturated water content $[L^3/L^]3
        alpha: Fitting parameter [L⁻ᵝ]
        beta: Fitting parameter [dimensionless]
        Ks: Saturated hydraulic conductivity [L/T]
        A: Fitting parameter [Lᵞ]
        gamma: Fitting parameter [dimensionless]
    """

    def _validate_parameters(self) -> None:
        """Validate Haverkamp model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'beta', 'Ks', 'A', 'gamma']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(self.parameters[param], fd.Constant):
                raise ValueError(f"Parameter {param} must be a fd.Constant")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp moisture content relationship.

        θ(h) = θᵣ + α(θₛ - θᵣ) / (α + |h|^β)
        """
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']

        theta = theta_r + alpha * (theta_s - theta_r) / (alpha + abs(h)**beta)
        return fd.conditional(h <= 0, theta, theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp relative permeability relationship.

        K(h) = Kₛ * A / (A + |h|^γ)
        """
        Ks = self.parameters['Ks']
        A = self.parameters['A']
        gamma = self.parameters['gamma']

        K = Ks * (A / (A + abs(h)**gamma))
        return fd.conditional(h <= 0, K, Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Haverkamp specific moisture capacity.

        C(h) = -sign(h) * α * β * (θₛ - θᵣ) * |h|^(β-1) / (α + |h|^β)²
        """
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']

        C = (-fd.sign(h) * alpha * beta * (theta_s - theta_r) *
             abs(h)**(beta - 1) / (alpha + abs(h)**beta)**2)
        return fd.conditional(h <= 0, C, 0)


class VanGenuchtenCurve(SoilCurve):
    """
    van Genuchten soil curve model.

    The van Genuchten model is widely used for describing soil-water retention
    and hydraulic conductivity relationships. It provides smooth, continuous
    curves with good physical interpretation.

    Parameters:
        theta_r: Residual water content [L³/L³]
        theta_s: Saturated water content [L³/L³]
        alpha: Inverse of air-entry pressure [L⁻¹]
        n: Pore-size distribution parameter [dimensionless]
        Ks: Saturated hydraulic conductivity [L/T]
    """

    def _validate_parameters(self) -> None:
        """Validate van Genuchten model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'n', 'Ks']
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

        θ(h) = θᵣ + (θₛ - θᵣ) / (1 + |αh|^n)^m
        where m = 1 - 1/n
        """
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']
        alpha = self.parameters['alpha']
        n = self.parameters['n']
        m = 1 - 1/n

        theta = theta_r + (theta_s - theta_r) / ((1 + abs(alpha * h)**n)**m)
        return fd.conditional(h <= 0, theta, theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten relative permeability relationship.

        K(h) = Kₛ * (1 - |αh|^(n-1) * (1 + |αh|^n)^(-m))² / (1 + |αh|^n)^(m/2)
        where m = 1 - 1/n
        """
        Ks = self.parameters['Ks']
        alpha = self.parameters['alpha']
        n = self.parameters['n']
        m = 1 - 1/n

        term1 = 1 - abs(alpha * h)**(n - 1) * (1 + abs(alpha * h)**n)**(-m)
        term2 = (1 + abs(alpha * h)**n)**(m/2)
        K = Ks * (term1**2 / term2)
        return fd.conditional(h <= 0, K, Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        van Genuchten specific moisture capacity.

        C(h) = -(θₛ - θᵣ) * n * m * h * α^n * |h|^(n-2) * (α^n * |h|^n + 1)^(-m-1)
        where m = 1 - 1/n
        """
        alpha = self.parameters['alpha']
        n = self.parameters['n']
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']
        m = 1 - 1/n

        C = (-(theta_s - theta_r) * n * m * h * (alpha**n) *
             abs(h)**(n - 2) * ((alpha**n * abs(h)**n + 1)**(-m - 1)))
        return fd.conditional(h <= 0, C, 0)


class ExponentialCurve(SoilCurve):
    """
    Exponential soil curve model.

    The exponential model provides simple analytical relationships for
    soil-water retention and hydraulic conductivity. It is often used
    for preliminary analysis or when detailed soil characterization
    data is not available.

    Parameters:
        theta_r: Residual water content [L³/L³]
        theta_s: Saturated water content [L³/L³]
        alpha: Exponential decay parameter [L⁻¹]
        Ks: Saturated hydraulic conductivity [L/T]
    """

    def _validate_parameters(self) -> None:
        """Validate exponential model parameters."""
        required_params = ['theta_r', 'theta_s', 'alpha', 'Ks']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(self.parameters[param], fd.Constant):
                raise ValueError(f"Parameter {param} must be a fd.Constant")

    def moisture_content(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential moisture content relationship.

        θ(h) = θᵣ + (θₛ - θᵣ) * exp(αh)
        """
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']
        alpha = self.parameters['alpha']

        theta = theta_r + (theta_s - theta_r) * fd.exp(h * alpha)
        return fd.conditional(h <= 0, theta, theta_s)

    def relative_permeability(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential relative permeability relationship.

        K(h) = Kₛ * exp(αh)
        """
        Ks = self.parameters['Ks']
        alpha = self.parameters['alpha']

        K = Ks * fd.exp(h * alpha)
        return fd.conditional(h <= 0, K, Ks)

    def water_retention(self, h: fd.Function | ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """
        Exponential specific moisture capacity.

        C(h) = (θₛ - θᵣ) * α * exp(αh)
        """
        alpha = self.parameters['alpha']
        theta_r = self.parameters['theta_r']
        theta_s = self.parameters['theta_s']

        C = (theta_s - theta_r) * fd.exp(h * alpha) * alpha
        return fd.conditional(h <= 0, C, 0)
