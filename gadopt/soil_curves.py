r"""Soil curve models for the Richards equation.

This module implements soil-water retention and hydraulic conductivity
relationships for unsaturated flow. Each model maps the pressure head $h$ to the
volumetric moisture content $\theta(h)$, the hydraulic conductivity $K(h)$ and
the specific moisture capacity $C(h) = d\theta/dh$. All three are returned as
UFL expressions, so they enter variational forms directly and are
differentiated by UFL for the Newton Jacobian and by pyadjoint for gradients.

References:
    Haverkamp, R., Vauclin, M., Touma, J., Wierenga, P. J. and Vachaud, G.
        (1977). A comparison of numerical simulation models for one-dimensional
        infiltration. Soil Science Society of America Journal 41, 285-294.
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the
        hydraulic conductivity of unsaturated soils. Soil Science Society of
        America Journal 44, 892-898.
    Mualem, Y. (1976). A new model for predicting the hydraulic conductivity
        of unsaturated porous media. Water Resources Research 12, 513-522.
    Gardner, W. R. (1958). Some steady-state solutions of the unsaturated
        moisture flow equation with application to evaporation from a water
        table. Soil Science 85, 228-232.
    Tracy, F. T. (2006). Clean two- and three-dimensional analytical solutions
        of Richards' equation for testing numerical solvers. Water Resources
        Research 42, W08503.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import firedrake as fd
import numpy as np
import ufl

__all__ = [
    "SoilCurve",
    "HaverkampCurve",
    "VanGenuchtenCurve",
    "ExponentialCurve",
]


class SoilCurve(ABC):
    r"""Abstract base class for soil curve models.

    The pressure head $h$ is the pressure relative to atmospheric pressure,
    expressed as a height of water. For $h \le 0$ the soil is unsaturated and the
    model's closed forms apply. For $h > 0$ the soil is saturated and every curve
    is clamped to $\theta = \theta_s$, $K = K_s$ and $C = 0$.

    Every model provides `moisture_content`, `hydraulic_conductivity` and
    `specific_moisture_capacity`, plus `effective_saturation`, which is derived
    from the moisture content.

    Parameter types. Each physical parameter (in this class and in the
    subclasses) accepts three kinds of value, which decide how it can be used:

    - A Python or NumPy number is stored as a `firedrake.Constant`. It is a
      fixed value of the model and cannot be a pyadjoint control.
    - A `firedrake.Function` on a Real ("R") space is a single scalar that can
      be a pyadjoint control, for example to invert for one soil parameter.
    - A `firedrake.Function` on a mesh function space, or any scalar UFL
      expression, describes a spatially varying (heterogeneous) soil. It is
      stored by reference and can also be a control.

    Range checks (for example $K_s > 0$, or $n > 1$ for van Genuchten) apply to
    the first two kinds, whose values are known when the curve is built. A
    spatially varying parameter is not checked.

    Regularisation at the water table. The van Genuchten and Haverkamp curves
    raise a magnitude ($|\alpha h|$ or $|h|$) to a non-integer power. The UFL
    derivative of that power is unbounded at exactly $h = 0$ (it evaluates to
    `Inf`, and to `NaN` after multiplication by `sign(0)`), which would make the
    Newton Jacobian and the pyadjoint gradient `NaN` whenever a degree of freedom
    sits exactly on the water table. To keep the derivative finite, the
    magnitude $|x|$ is replaced by $\sqrt{x^2 + (\epsilon s)^2}$, where
    $\epsilon$ is `reg_eps` and $s$ is the natural scale of $x$ for that curve.
    Away from $h = 0$ the curves change by $O(\epsilon^2)$ relative to the exact
    forms. At $h = 0$ itself the regularised value differs from the saturated
    clamp by an amount that depends on the exponent. For $\theta$ of both models
    and $K$ of Haverkamp the difference is $O(\epsilon^{\beta})$, $O(\epsilon^n)$
    or $O(\epsilon^{\gamma})$ and negligible. For the van Genuchten conductivity
    it is $K(0)/K_s \approx (1 - \epsilon^{\,n-1})^2$, which is not negligible for
    $n$ close to 1: with the default $\epsilon = 10^{-6}$ it is 0.51 at
    $n = 1.09$, 0.88 at $n = 1.2$, 0.973 at $n = 1.31$ and 0.998 at $n = 1.5$.
    This is a property of the van Genuchten-Mualem conductivity, whose exact
    derivative at $h = 0$ is unbounded for $n < 2$: any regularisation that
    bounds the derivative must change the value near $h = 0$. Set
    `reg_eps=0` to recover the exact forms, with unbounded derivatives at
    $h = 0$.

    Attributes:
        theta_r: Residual water content $\theta_r$ $[L^3/L^3]$.
        theta_s: Saturated water content $\theta_s$ $[L^3/L^3]$.
        Ks: Saturated hydraulic conductivity $K_s$ $[L/T]$.
        Ss: Specific storage coefficient $S_s$ $[L^{-1}]$.
        reg_eps: Dimensionless regularisation $\epsilon$ of the magnitude at the
            water table (a Python float, fixed when the curve is built).
    """

    #: Default dimensionless regularisation of the magnitude near h = 0.
    DEFAULT_REG_EPS = 1e-6

    def __init__(
        self,
        *,
        theta_r,
        theta_s,
        Ks,
        Ss=0.0,
        reg_eps: float = DEFAULT_REG_EPS,
    ):
        """Store and check the parameters common to every soil model.

        The arguments are keyword-only, so a parameter cannot be bound to the
        wrong name by its position.

        Args:
            theta_r: Residual water content $[L^3/L^3]$, in [0, 1).
            theta_s: Saturated water content $[L^3/L^3]$, in (theta_r, 1].
            Ks: Saturated hydraulic conductivity $[L/T]$, > 0.
            Ss: Specific storage coefficient $[L^{-1}]$, >= 0. Defaults to 0.
            reg_eps: Dimensionless regularisation of the magnitude at the water
                table, >= 0 (see the class docstring). It is converted to a
                Python float when the curve is built, so it is not a UFL
                coefficient and cannot be changed afterwards. Defaults to
                `DEFAULT_REG_EPS`. The exponential model does not use it.

        Raises:
            ValueError: If a parameter whose value is known (a number or a
                Real-space Function) is out of range, if theta_s <= theta_r, or
                if a parameter is not a scalar.
        """
        # reg_eps is a numerical setting of the discretisation, not a physical
        # parameter, so it stays a plain float. Keeping it out of the UFL form
        # makes the zero test in _reg_abs exact and honest: the choice between
        # the regularised and the exact magnitude is made here, once.
        reg_eps = float(reg_eps)
        if reg_eps < 0.0:
            raise ValueError(f"{type(self).__name__}: reg_eps must be >= 0, got {reg_eps}")
        self.reg_eps = reg_eps

        # Volumetric contents are fractions of the bulk volume.
        self.theta_r = self._parameter("theta_r", theta_r, lambda v: (v >= 0.0) & (v < 1.0), "in [0, 1)")
        self.theta_s = self._parameter("theta_s", theta_s, lambda v: (v > 0.0) & (v <= 1.0), "in (0, 1]")
        self.Ks = self._parameter("Ks", Ks, lambda v: v > 0.0, "> 0")
        self.Ss = self._parameter("Ss", Ss, lambda v: v >= 0.0, ">= 0")

        # theta_s > theta_r makes the retention curve decrease from theta_s to
        # theta_r as the soil dries; it is also the denominator of the effective
        # saturation. Only checked when both values are known.
        theta_r_values = self._known_values(self.theta_r)
        theta_s_values = self._known_values(self.theta_s)
        if theta_r_values is not None and theta_s_values is not None:
            if np.any(theta_s_values <= theta_r_values):
                raise ValueError(
                    f"{type(self).__name__}: theta_s must be greater than theta_r, "
                    f"got theta_s={theta_s_values} and theta_r={theta_r_values}"
                )

    def _parameter(
        self,
        name: str,
        value,
        condition: Callable[[np.ndarray], np.ndarray],
        requirement: str,
    ) -> ufl.core.expr.Expr:
        """Convert one physical parameter to a UFL expression and check its range.

        Args:
            name: Parameter name, used in error messages.
            value: A number, a Real-space Function, a mesh Function or a UFL
                expression (see "Parameter types" in the class docstring).
            condition: Function of a NumPy array of values that returns a boolean
                array, True where the value is valid.
            requirement: Human-readable form of the condition for the error
                message, for example "> 0".

        Returns:
            The parameter as a UFL expression: the original object if it already
            was one, else a new `firedrake.Constant`.

        Raises:
            ValueError: If the parameter is not a scalar, or if its value is known
                and fails the condition.
        """
        # Anything symbolic (Function, Constant, UFL expression) is kept by
        # reference, so a heterogeneous field or a pyadjoint control is not copied.
        # Everything else is a number. Testing for ufl Expr instead of for float
        # also wraps NumPy scalars and 0-d arrays.
        expr = value if isinstance(value, ufl.core.expr.Expr) else fd.Constant(value)

        # The curves are written for scalar parameters. A vector Constant would
        # broadcast silently in some operations and fail in others.
        if expr.ufl_shape != ():
            raise ValueError(
                f"{type(self).__name__}: {name} must be a scalar, got shape {expr.ufl_shape}"
            )

        values = self._known_values(expr)
        if values is not None and not np.all(condition(values)):
            raise ValueError(f"{type(self).__name__}: {name} must be {requirement}, got {values}")
        return expr

    @staticmethod
    def _known_values(expr: ufl.core.expr.Expr) -> np.ndarray | None:
        """Return the values of a parameter that is known when the curve is built.

        A `firedrake.Constant` and a Function on a Real space hold one value each,
        the same on every MPI rank. A spatially varying Function or a general
        UFL expression has no single value, and the range checks skip it.

        Args:
            expr: The parameter as a UFL expression.

        Returns:
            A 1-D NumPy array of values, or None if the values are not known.
        """
        if isinstance(expr, fd.Constant):
            return np.atleast_1d(np.asarray(expr.values(), dtype=float))
        if isinstance(expr, fd.Function) and expr.function_space().ufl_element().family() == "Real":
            return np.atleast_1d(np.asarray(expr.dat.data_ro, dtype=float))
        return None

    def _reg_abs(self, x: ufl.core.expr.Expr, scale=1.0) -> ufl.core.expr.Expr:
        r"""Regularised magnitude $\sqrt{x^2 + (\epsilon\,s)^2}$, with $s$ = `scale`.

        Replaces `abs(x)` where `x` is raised to a non-integer power, so the UFL
        derivative stays bounded at `x = 0` (see the class docstring). `scale`
        is the natural magnitude of `x` for the curve, which makes the
        dimensionless `reg_eps` an offset in the units of `x`. With
        `reg_eps = 0` this is the exact `abs(x)`.

        Args:
            x: The quantity whose magnitude is needed, a UFL expression.
            scale: Natural magnitude of `x`, in the units of `x`. A number or a
                UFL expression.

        Returns:
            The regularised magnitude as a UFL expression.
        """
        if self.reg_eps == 0.0:
            return abs(x)
        eps = self.reg_eps * scale
        return fd.sqrt(x * x + eps * eps)

    @abstractmethod
    def moisture_content(self, h: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Volumetric moisture content $\theta(h)$.

        Args:
            h: Pressure head $[L]$, a UFL expression.

        Returns:
            Volumetric moisture content $\theta$ $[L^3/L^3]$, between
            $\theta_r$ and $\theta_s$.
        """

    @abstractmethod
    def hydraulic_conductivity(self, h: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Hydraulic conductivity $K(h) = K_s\,k_r(h)$.

        $k_r$ is the dimensionless relative permeability, between 0 and 1.

        Args:
            h: Pressure head $[L]$, a UFL expression.

        Returns:
            Hydraulic conductivity $K$ $[L/T]$, between 0 and $K_s$.
        """

    @abstractmethod
    def specific_moisture_capacity(self, h: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Specific moisture capacity $C(h) = d\theta/dh$.

        Each model writes $C$ out by hand. It is the exact derivative of the
        model's (regularised) `moisture_content`, and is cheaper to evaluate
        than the expression UFL would build by differentiating $\theta$.

        Args:
            h: Pressure head $[L]$, a UFL expression.

        Returns:
            Specific moisture capacity $C$ $[L^{-1}]$, >= 0, and 0 for $h > 0$.
        """

    def effective_saturation(self, h: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Effective saturation $S_e(h) = (\theta(h) - \theta_r)/(\theta_s - \theta_r)$.

        $S_e$ is 0 at residual water content and 1 at saturation. It weights the
        specific storage term of the Richards equation.

        Args:
            h: Pressure head $[L]$, a UFL expression.

        Returns:
            Effective saturation $S_e$ [dimensionless], between 0 and 1.
        """
        return (self.moisture_content(h) - self.theta_r) / (self.theta_s - self.theta_r)


class HaverkampCurve(SoilCurve):
    r"""Haverkamp et al. (1977) soil curve model.

    Empirical retention and conductivity curves:

    $$\theta(h) = \theta_r + \frac{\alpha\,(\theta_s - \theta_r)}{\alpha + |h|^{\beta}},
    \qquad K(h) = K_s\,\frac{A}{A + |h|^{\gamma}}.$$

    $\alpha$ and $A$ carry units of head to the powers $\beta$ and $\gamma$, so
    their values depend on the length unit of $h$. Haverkamp's sand, for example,
    has $\alpha = 1.611 \times 10^6$ cm$^{3.96}$ and $\beta = 3.96$ with $h$ in cm.

    Attributes:
        alpha: Retention fitting parameter $\alpha$ $[L^{\beta}]$.
        beta: Retention exponent $\beta$ [dimensionless].
        A: Conductivity fitting parameter $A$ $[L^{\gamma}]$.
        gamma: Conductivity exponent $\gamma$ [dimensionless].
        theta_r, theta_s, Ks, Ss, reg_eps: See `SoilCurve`.
    """

    def __init__(
        self,
        *,
        theta_r,
        theta_s,
        Ks,
        alpha,
        beta,
        A,
        gamma,
        Ss=0.0,
        reg_eps: float = SoilCurve.DEFAULT_REG_EPS,
    ):
        r"""Build a Haverkamp curve.

        Args:
            theta_r: Residual water content $[L^3/L^3]$.
            theta_s: Saturated water content $[L^3/L^3]$.
            Ks: Saturated hydraulic conductivity $[L/T]$.
            alpha: Retention fitting parameter $[L^{\beta}]$, > 0.
            beta: Retention exponent [dimensionless], > 0.
            A: Conductivity fitting parameter $[L^{\gamma}]$, > 0.
            gamma: Conductivity exponent [dimensionless], > 0.
            Ss: Specific storage coefficient $[L^{-1}]$. Defaults to 0.
            reg_eps: Regularisation at the water table, see `SoilCurve`.

        Raises:
            ValueError: If a parameter with a known value is out of range.
        """
        super().__init__(theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss, reg_eps=reg_eps)
        # Positive alpha and A keep the denominators positive; positive
        # exponents make theta and K decrease monotonically as |h| grows.
        self.alpha = self._parameter("alpha", alpha, lambda v: v > 0.0, "> 0")
        self.beta = self._parameter("beta", beta, lambda v: v > 0.0, "> 0")
        self.A = self._parameter("A", A, lambda v: v > 0.0, "> 0")
        self.gamma = self._parameter("gamma", gamma, lambda v: v > 0.0, "> 0")

    def _abs_h_theta(self, h):
        r"""Regularised $|h|$ for the retention curve.

        The natural head scale is $\alpha^{1/\beta}$, the head at which
        $|h|^\beta = \alpha$ and the soil is half way between residual and
        saturated water content.
        """
        return self._reg_abs(h, scale=self.alpha**(1 / self.beta))

    def _abs_h_K(self, h):
        r"""Regularised $|h|$ for the conductivity curve.

        The natural head scale is $A^{1/\gamma}$, the head at which $K = K_s/2$.
        """
        return self._reg_abs(h, scale=self.A**(1 / self.gamma))

    def moisture_content(self, h):
        r"""Haverkamp moisture content.

        $\theta(h) = \theta_r + \alpha(\theta_s - \theta_r)/(\alpha + |h|^{\beta})$
        for $h \le 0$, and $\theta_s$ for $h > 0$.
        """
        abs_h = self._abs_h_theta(h)
        theta = self.theta_r + self.alpha * (self.theta_s - self.theta_r) / (self.alpha + abs_h**self.beta)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h):
        r"""Haverkamp hydraulic conductivity.

        $K(h) = K_s\,A/(A + |h|^{\gamma})$ for $h \le 0$, and $K_s$ for $h > 0$.
        """
        abs_h = self._abs_h_K(h)
        K = self.Ks * (self.A / (self.A + abs_h**self.gamma))
        return fd.conditional(h <= 0, K, self.Ks)

    def specific_moisture_capacity(self, h):
        r"""Haverkamp specific moisture capacity.

        $C(h) = -\mathrm{sign}(h)\,\alpha\beta(\theta_s - \theta_r)\,|h|^{\beta-1}
        / (\alpha + |h|^{\beta})^2$ for $h \le 0$, and 0 for $h > 0$.
        """
        # d theta/dh of the regularised retention curve. With m(h) the
        # regularised magnitude, dm/dh = h/m, so the factor sign(h)*|h|^(beta-1)
        # of the exact form becomes (h/m)*m^(beta-1), which is bounded at h = 0.
        abs_h = self._abs_h_theta(h)
        C = (-(h / abs_h) * self.alpha * self.beta * (self.theta_s - self.theta_r) *
             abs_h**(self.beta - 1) / (self.alpha + abs_h**self.beta)**2)
        return fd.conditional(h <= 0, C, 0)


class VanGenuchtenCurve(SoilCurve):
    r"""van Genuchten (1980) retention curve with the Mualem (1976) conductivity.

    With $u = |\alpha h|$ and $m = 1 - 1/n$:

    $$\theta(h) = \theta_r + \frac{\theta_s - \theta_r}{(1 + u^n)^m},
    \qquad K(h) = K_s\,\frac{\bigl(1 - u^{n-1}(1 + u^n)^{-m}\bigr)^2}{(1 + u^n)^{m/2}}.$$

    The conductivity has an unbounded derivative at $h = 0$ for $n < 2$. See
    `SoilCurve` for what the regularisation does to its value there.

    Attributes:
        alpha: Inverse of the air-entry head $\alpha$ $[L^{-1}]$.
        n: Pore-size distribution exponent $n$ [dimensionless].
        theta_r, theta_s, Ks, Ss, reg_eps: See `SoilCurve`.
    """

    def __init__(
        self,
        *,
        theta_r,
        theta_s,
        Ks,
        alpha,
        n,
        Ss=0.0,
        reg_eps: float = SoilCurve.DEFAULT_REG_EPS,
    ):
        r"""Build a van Genuchten curve.

        $m$ is always $1 - 1/n$ (the Mualem constraint), so it is not a
        parameter.

        Args:
            theta_r: Residual water content $[L^3/L^3]$.
            theta_s: Saturated water content $[L^3/L^3]$.
            Ks: Saturated hydraulic conductivity $[L/T]$.
            alpha: Inverse of the air-entry head $[L^{-1}]$, > 0.
            n: Pore-size distribution exponent [dimensionless], > 1.
            Ss: Specific storage coefficient $[L^{-1}]$. Defaults to 0.
            reg_eps: Regularisation at the water table, see `SoilCurve`.

        Raises:
            ValueError: If a parameter with a known value is out of range.
        """
        super().__init__(theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss, reg_eps=reg_eps)
        self.alpha = self._parameter("alpha", alpha, lambda v: v > 0.0, "> 0")
        # n > 1 gives m > 0, without which theta would not decrease with |h|.
        self.n = self._parameter("n", n, lambda v: v > 1.0, "> 1")

    def _abs_alpha_h(self, h):
        r"""Regularised dimensionless magnitude $|\alpha h|$.

        $\alpha h$ is already dimensionless, so the regularisation offset is
        `reg_eps` itself (scale 1).
        """
        return self._reg_abs(self.alpha * h)

    def moisture_content(self, h):
        r"""van Genuchten moisture content.

        $\theta(h) = \theta_r + (\theta_s - \theta_r)/(1 + |\alpha h|^n)^m$ for
        $h \le 0$, and $\theta_s$ for $h > 0$.
        """
        m = 1 - 1 / self.n
        ah = self._abs_alpha_h(h)
        theta = self.theta_r + (self.theta_s - self.theta_r) / ((1 + ah**self.n)**m)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h):
        r"""van Genuchten-Mualem hydraulic conductivity.

        $K(h) = K_s\,(1 - |\alpha h|^{n-1}(1 + |\alpha h|^n)^{-m})^2
        / (1 + |\alpha h|^n)^{m/2}$ for $h \le 0$, and $K_s$ for $h > 0$.
        """
        m = 1 - 1 / self.n
        ah = self._abs_alpha_h(h)
        # Mualem's integral, evaluated in closed form for m = 1 - 1/n: the
        # numerator is (1 - (1 - S_e^(1/m))^m)^2 and the denominator S_e^(-1/2),
        # both written in terms of u = |alpha h|.
        numerator = (1 - ah**(self.n - 1) * (1 + ah**self.n)**(-m))**2
        denominator = (1 + ah**self.n)**(m / 2)
        K = self.Ks * numerator / denominator
        return fd.conditional(h <= 0, K, self.Ks)

    def specific_moisture_capacity(self, h):
        r"""van Genuchten specific moisture capacity.

        $C(h) = -(\theta_s - \theta_r)\,n m\,\alpha^n h\,|h|^{n-2}
        (1 + |\alpha h|^n)^{-m-1}$ for $h \le 0$, and 0 for $h > 0$.
        """
        m = 1 - 1 / self.n
        ah = self._abs_alpha_h(h)
        # The singular factor of the exact form is |h|^(n-2). Writing it with
        # the regularised u = |alpha h| as u^(n-2)/alpha^(n-2) gives
        # alpha^n h |h|^(n-2) = (alpha h) alpha u^(n-2), which is bounded at
        # h = 0. The leading (alpha h) carries the sign.
        C = (-(self.theta_s - self.theta_r) * self.n * m * (self.alpha * h) *
             self.alpha * ah**(self.n - 2) * ((ah**self.n + 1)**(-m - 1)))
        return fd.conditional(h <= 0, C, 0)


class ExponentialCurve(SoilCurve):
    r"""Exponential (Gardner 1958) soil curve model.

    $$\theta(h) = \theta_r + (\theta_s - \theta_r)\,e^{\alpha h},
    \qquad K(h) = K_s\,e^{\alpha h}.$$

    The exponential conductivity makes the steady Richards equation linear in
    $K$, which is what gives Tracy's (2006) closed-form solutions. It is used
    mostly for verification against those solutions. The curves are smooth for
    $h < 0$, so `reg_eps` has no effect. $\theta$ and $K$ are continuous at
    $h = 0$, but their slopes are not: $C$ jumps from $\alpha(\theta_s -
    \theta_r)$ just below the water table to 0 above it, and $dK/dh$ from
    $\alpha K_s$ to 0.

    Attributes:
        alpha: Exponential decay rate $\alpha$ $[L^{-1}]$.
        theta_r, theta_s, Ks, Ss, reg_eps: See `SoilCurve`.
    """

    def __init__(
        self,
        *,
        theta_r,
        theta_s,
        Ks,
        alpha,
        Ss=0.0,
        reg_eps: float = SoilCurve.DEFAULT_REG_EPS,
    ):
        r"""Build an exponential curve.

        Args:
            theta_r: Residual water content $[L^3/L^3]$.
            theta_s: Saturated water content $[L^3/L^3]$.
            Ks: Saturated hydraulic conductivity $[L/T]$.
            alpha: Exponential decay rate $[L^{-1}]$, > 0.
            Ss: Specific storage coefficient $[L^{-1}]$. Defaults to 0.
            reg_eps: Accepted for a uniform interface. Not used.

        Raises:
            ValueError: If a parameter with a known value is out of range.
        """
        super().__init__(theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss, reg_eps=reg_eps)
        self.alpha = self._parameter("alpha", alpha, lambda v: v > 0.0, "> 0")

    def moisture_content(self, h):
        r"""Exponential moisture content.

        $\theta(h) = \theta_r + (\theta_s - \theta_r)\,e^{\alpha h}$ for
        $h \le 0$, and $\theta_s$ for $h > 0$.
        """
        theta = self.theta_r + (self.theta_s - self.theta_r) * fd.exp(self.alpha * h)
        return fd.conditional(h <= 0, theta, self.theta_s)

    def hydraulic_conductivity(self, h):
        r"""Exponential hydraulic conductivity.

        $K(h) = K_s\,e^{\alpha h}$ for $h \le 0$, and $K_s$ for $h > 0$.
        """
        K = self.Ks * fd.exp(self.alpha * h)
        return fd.conditional(h <= 0, K, self.Ks)

    def specific_moisture_capacity(self, h):
        r"""Exponential specific moisture capacity.

        $C(h) = \alpha(\theta_s - \theta_r)\,e^{\alpha h}$ for $h \le 0$, and 0
        for $h > 0$.
        """
        C = self.alpha * (self.theta_s - self.theta_r) * fd.exp(self.alpha * h)
        return fd.conditional(h <= 0, C, 0)
