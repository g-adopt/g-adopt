"""
Unit tests for soil curve models in the Richards equation module.

These cover the Haverkamp, van Genuchten and exponential constitutive models
for unsaturated flow. The suite follows these principles:

- Constitutive values are checked against independent NumPy implementations of
  the published closed forms, at parameter values where every parameter changes
  the result. The van Genuchten conductivity oracle uses Mualem's form in terms
  of the effective saturation, which is different algebra from the gadopt code.
  Each oracle is anchored to hand-computed values, so an error in an oracle also
  fails.
- The specific moisture capacity C is checked to be the derivative of the
  implemented moisture content theta by a central difference, at heads where
  the curves are steep.
- The behaviour at the water table h = 0 is checked against the formulas the
  documentation gives, including the size of the regularisation's effect on
  each curve and its dependence on the curve's head scale.
- The reg_eps regularisation that keeps the UFL derivative (the Newton Jacobian
  and the pyadjoint adjoint) finite at h = 0 is guarded both ways: the
  regularised form stays finite, and the unregularised form goes singular.
- The adjoint is exercised with pyadjoint Taylor tests over the head field
  (including a field that touches h = 0) and over a soil parameter.
- Heterogeneous (spatially varying) parameters and the parameter checks of the
  constructors are tested directly.

References:
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the
        hydraulic conductivity of unsaturated soils. Soil Sci. Soc. Am. J. 44,
        892-898. Retention Eq. [21], conductivity Eq. [9], m = 1 - 1/n Eq. [22],
        derivative Eq. [23]. Example parameters from Fig. 1 (alpha = 0.005/cm,
        n = 2, theta_r = 0.10, theta_s = 0.50) and Fig. 3 (Ks = 100 cm/day).
    Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of
        unsaturated porous media. Water Resour. Res. 12, 513-522.
    Haverkamp, R., et al. (1977). A comparison of numerical simulation models
        for one-dimensional infiltration. Soil Sci. Soc. Am. J. 41, 285-294.
        Sand parameters: theta_r = 0.075, theta_s = 0.287, Ks = 34 cm/h,
        alpha = 1.611e6, beta = 3.96, A = 1.175e6, gamma = 4.74 (h in cm).
"""

import contextlib
import io

import firedrake as fd
import numpy as np
import pytest
import ufl
from gadopt.soil_curves import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)

PROPERTY_NAMES = (
    "moisture_content",
    "hydraulic_conductivity",
    "specific_moisture_capacity",
)
ALL_FAMILIES = (VanGenuchtenCurve, HaverkampCurve, ExponentialCurve)


# ---------------------------------------------------------------------------
# Independent NumPy oracles for the published closed forms.
#
# These share no code with gadopt.soil_curves. Each is the unsaturated branch
# (h <= 0); the saturated branch is a separate clamp.
# ---------------------------------------------------------------------------
def _vg_oracle(h, theta_r, theta_s, Ks, alpha, n):
    """van Genuchten 1980 retention (Eq. [21]) with Mualem's conductivity.

    K is written in Mualem's form in terms of the effective saturation,
        K = Ks Se^(1/2) (1 - (1 - Se^(1/m))^m)^2,
    which is algebraically equal to van Genuchten's Eq. [9] in terms of
    |alpha h| but shares no intermediate quantity with it. C is the derivative
    of Eq. [21] for h < 0:
        C = (theta_s - theta_r) m n alpha u^(n-1) (1 + u^n)^(-m-1),  u = |alpha h|.
    """
    m = 1.0 - 1.0 / n
    u = np.abs(alpha * h)
    Se = (1.0 + u**n) ** (-m)
    theta = theta_r + (theta_s - theta_r) * Se
    K = Ks * np.sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    C = (theta_s - theta_r) * m * n * alpha * u ** (n - 1) * (1.0 + u**n) ** (-m - 1.0)
    return theta, K, C


def _haverkamp_oracle(h, theta_r, theta_s, Ks, alpha, beta, A, gamma):
    """Haverkamp et al. 1977 closed forms.

    C = d(theta)/dh = alpha beta (theta_s - theta_r) |h|^(beta-1) / (alpha + |h|^beta)^2.
    """
    w = np.abs(h)
    theta = theta_r + alpha * (theta_s - theta_r) / (alpha + w**beta)
    K = Ks * A / (A + w**gamma)
    C = alpha * beta * (theta_s - theta_r) * w ** (beta - 1) / (alpha + w**beta) ** 2
    return theta, K, C


def _exp_oracle(h, theta_r, theta_s, Ks, alpha):
    """Exponential (Gardner) model."""
    e = np.exp(alpha * h)
    theta = theta_r + (theta_s - theta_r) * e
    K = Ks * e
    C = (theta_s - theta_r) * alpha * e
    return theta, K, C


# Parameter sets for the oracle comparisons. The van Genuchten set is the
# published Fig. 1 / Fig. 3 example (h in cm, K in cm/day). The Haverkamp set is
# the 1977 sand (h in cm, K in cm/h): alpha differs from A and beta from gamma,
# so swapping or dropping any of them changes the result.
VG_PAPER_PARAMS = dict(theta_r=0.10, theta_s=0.50, Ks=100.0, alpha=0.005, n=2.0)
HAVERKAMP_SAND_PARAMS = dict(
    theta_r=0.075, theta_s=0.287, Ks=34.0, alpha=1.611e6, beta=3.96, A=1.175e6, gamma=4.74,
)
EXP_PARAMS = dict(theta_r=0.15, theta_s=0.45, Ks=1e-5, alpha=0.5)


def _paper_curve(cls, **overrides):
    """Build one family from its published parameter set, with overrides."""
    params = {
        VanGenuchtenCurve: VG_PAPER_PARAMS,
        HaverkampCurve: HAVERKAMP_SAND_PARAMS,
        ExponentialCurve: EXP_PARAMS,
    }[cls]
    return cls(**{**params, **overrides})


# ---------------------------------------------------------------------------
# Mesh / evaluation helpers.
# ---------------------------------------------------------------------------
def create_test_mesh():
    """A single-cell interval mesh; constitutive curves are pointwise."""
    return fd.UnitIntervalMesh(1)


def create_pressure_head_function(mesh, value):
    """A CG1 pressure-head function holding a uniform value."""
    V = fd.FunctionSpace(mesh, "CG", 1)
    h = fd.Function(V)
    h.interpolate(fd.Constant(value))
    return h


def evaluate_soil_curve(model, h, property_name):
    """Interpolate one soil-curve property of ``model`` at head field ``h``."""
    expr = getattr(model, property_name)(h)
    result = fd.Function(h.function_space())
    result.interpolate(expr)
    return result


def _eval_scalar(model, property_name, hval):
    """Evaluate a property at a uniform head and return the (scalar) value."""
    h = create_pressure_head_function(create_test_mesh(), hval)
    return evaluate_soil_curve(model, h, property_name).dat.data[0]


def _jacobian_diag(model, property_name, hval, mesh=None):
    """Assemble d(property)/dh at a uniform head and return the matrix entries.

    This is the same UFL differentiation that feeds the Newton Jacobian and the
    pyadjoint adjoint, so finiteness here is finiteness of both.
    """
    if mesh is None:
        mesh = create_test_mesh()
    V = fd.FunctionSpace(mesh, "CG", 1)
    h = fd.Function(V)
    h.interpolate(fd.Constant(hval))

    expr = getattr(model, property_name)(h)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    J = fd.derivative(expr * v * fd.dx, h, u)
    A = fd.assemble(J)
    return np.array(A.M.values).ravel()


def _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS, n=1.5, **overrides):
    """Build a curve with a NON-INTEGER exponent, which exposes the |.|^p singularity.

    A non-integer n, beta or gamma is what makes the unsaturated-branch
    derivative unbounded at h = 0. For Haverkamp, alpha differs from A and beta
    from gamma, so the two branches use different head scales. The exponential
    model has no such exponent; n is ignored for it.
    """
    common = dict(theta_r=0.15, theta_s=0.45, Ks=1e-5, reg_eps=reg_eps)
    if cls is VanGenuchtenCurve:
        params = dict(alpha=0.5, n=n)
    elif cls is HaverkampCurve:
        params = dict(alpha=2.0, beta=n, A=3.0, gamma=n + 0.3)
    elif cls is ExponentialCurve:
        params = dict(alpha=0.5)
    else:
        raise ValueError(cls)
    return cls(**{**common, **params, **overrides})


# ---------------------------------------------------------------------------
# Construction: keyword arguments, defaults and parameter checks.
# ---------------------------------------------------------------------------
class TestConstruction:
    """The constructors' signatures and parameter checks."""

    def test_abstract_base_class(self):
        """SoilCurve is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SoilCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5)

    @pytest.mark.parametrize("cls,missing", [
        (HaverkampCurve, "gamma"),
        (VanGenuchtenCurve, "n"),
        (ExponentialCurve, "alpha"),
        (VanGenuchtenCurve, "Ks"),
    ])
    def test_missing_parameter_raises(self, cls, missing):
        """A missing model parameter is a TypeError that names it."""
        params = {
            VanGenuchtenCurve: VG_PAPER_PARAMS,
            HaverkampCurve: HAVERKAMP_SAND_PARAMS,
            ExponentialCurve: EXP_PARAMS,
        }[cls]
        kwargs = {k: v for k, v in params.items() if k != missing}
        with pytest.raises(TypeError, match=missing):
            cls(**kwargs)

    @pytest.mark.parametrize("cls,extra", [
        (VanGenuchtenCurve, {"m": 0.3}),
        (HaverkampCurve, {"n": 2.0}),
        (ExponentialCurve, {"bogus": 1.0}),
    ])
    def test_unknown_parameter_raises(self, cls, extra):
        """An unknown keyword (for example an independent van Genuchten m) is rejected."""
        with pytest.raises(TypeError):
            _paper_curve(cls, **extra)

    def test_positional_arguments_rejected(self):
        """Parameters are keyword-only, so they cannot bind to the wrong name."""
        with pytest.raises(TypeError):
            ExponentialCurve(0.15, 0.45, 1e-5, 0.5)

    @pytest.mark.parametrize("cls", ALL_FAMILIES)
    def test_Ss_defaults_to_zero(self, cls):
        """Specific storage is optional and defaults to 0."""
        assert float(_paper_curve(cls).Ss) == 0.0

    @pytest.mark.parametrize("cls,overrides,match", [
        (VanGenuchtenCurve, {"n": 0.5}, "n must be > 1"),
        (VanGenuchtenCurve, {"n": 1.0}, "n must be > 1"),
        (VanGenuchtenCurve, {"alpha": -0.005}, "alpha must be > 0"),
        (HaverkampCurve, {"alpha": -1.0}, "alpha must be > 0"),
        (HaverkampCurve, {"A": 0.0}, "A must be > 0"),
        (HaverkampCurve, {"beta": -1.0}, "beta must be > 0"),
        (HaverkampCurve, {"gamma": 0.0}, "gamma must be > 0"),
        (ExponentialCurve, {"alpha": -0.5}, "alpha must be > 0"),
        (ExponentialCurve, {"Ks": -1e-5}, "Ks must be > 0"),
        (ExponentialCurve, {"Ss": -1e-3}, "Ss must be >= 0"),
        (ExponentialCurve, {"theta_r": -0.1}, r"theta_r must be in \[0, 1\)"),
        (ExponentialCurve, {"theta_s": 1.2}, r"theta_s must be in \(0, 1\]"),
        (ExponentialCurve, {"theta_r": 0.45, "theta_s": 0.15}, "theta_s must be greater than theta_r"),
        (ExponentialCurve, {"reg_eps": -1e-6}, "reg_eps must be >= 0"),
    ])
    def test_out_of_range_parameter_raises(self, cls, overrides, match):
        """A known parameter value outside its valid range is a ValueError."""
        with pytest.raises(ValueError, match=match):
            _paper_curve(cls, **overrides)

    def test_valid_boundary_values_accepted(self):
        """Values just inside the ranges are accepted."""
        _paper_curve(VanGenuchtenCurve, n=1.0001)
        _paper_curve(ExponentialCurve, theta_r=0.0, theta_s=1.0, Ss=0.0, reg_eps=0.0)

    def test_real_space_parameter_is_checked(self):
        """A Real-space Function has a known value, so it is range-checked."""
        R = fd.FunctionSpace(create_test_mesh(), "R", 0)
        with pytest.raises(ValueError, match="n must be > 1"):
            _paper_curve(VanGenuchtenCurve, n=fd.Function(R).assign(0.5))

    def test_spatially_varying_parameter_is_not_checked(self):
        """A spatially varying field has no single value and is not range-checked."""
        V = fd.FunctionSpace(fd.UnitIntervalMesh(4), "CG", 1)
        n_field = fd.Function(V).assign(0.5)  # would be rejected as a number
        model = _paper_curve(VanGenuchtenCurve, n=n_field)
        assert model.n is n_field

    def test_vector_parameter_rejected(self):
        """A non-scalar parameter is rejected instead of broadcast."""
        with pytest.raises(ValueError, match="n must be a scalar"):
            _paper_curve(VanGenuchtenCurve, n=np.array([2.0, 0.5]))

    def test_numbers_become_constants(self):
        """Python and NumPy numbers become firedrake Constants."""
        model = _paper_curve(VanGenuchtenCurve, n=np.float32(2.0), Ss=0)
        for value in (model.theta_r, model.Ks, model.alpha, model.n, model.Ss):
            assert isinstance(value, fd.Constant)
        assert float(model.n) == 2.0

    def test_reg_eps_is_a_float(self):
        """reg_eps is a numerical setting stored as a Python float."""
        model = _paper_curve(VanGenuchtenCurve, reg_eps=np.float32(1e-4))
        assert type(model.reg_eps) is float
        assert model.reg_eps == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Constitutive values vs independent published oracle.
# ---------------------------------------------------------------------------
class TestConstitutiveValues:
    """theta, K, C against independent NumPy implementations of the papers.

    With the default reg_eps the regularisation perturbs the result only at
    O(reg_eps^2) away from h = 0, so the exact-form oracles agree to ~1e-6 here.
    """

    def _check(self, model, oracle_values, h, rtol):
        for property_name, expected in zip(PROPERTY_NAMES, oracle_values):
            np.testing.assert_allclose(_eval_scalar(model, property_name, h), expected, rtol=rtol)

    def test_van_genuchten_matches_paper(self):
        # At h = -200 cm, |alpha h| = 1 exactly. Hand values from the van
        # Genuchten 1980 closed forms (Eq. [21], [9], [23]) at the Fig. 1/3
        # parameters: theta = 0.1 + 0.4/sqrt(2), K = 100 (1 - 2^-1/2)^2 / 2^(1/4),
        # C = 0.4 * 0.5 * 2 * 0.005 * 2^(-3/2).
        h = -200.0
        oracle = _vg_oracle(h, **VG_PAPER_PARAMS)
        np.testing.assert_allclose(oracle, (0.382842712, 7.213750788, 7.0710678e-4), rtol=1e-8)
        self._check(VanGenuchtenCurve(**VG_PAPER_PARAMS), oracle, h, rtol=1e-6)

    def test_haverkamp_matches_closed_form(self):
        # Haverkamp sand at h = -30 cm, hand-evaluated from the 1977 closed forms.
        h = -30.0
        oracle = _haverkamp_oracle(h, **HAVERKAMP_SAND_PARAMS)
        np.testing.assert_allclose(oracle, (0.222341077, 3.563507830, 5.93185297e-3), rtol=1e-8)
        self._check(HaverkampCurve(**HAVERKAMP_SAND_PARAMS), oracle, h, rtol=1e-6)

    def test_exponential_matches_closed_form(self):
        h = -2.0
        oracle = _exp_oracle(h, **EXP_PARAMS)
        np.testing.assert_allclose(
            oracle, (0.15 + 0.30 * np.exp(-1.0), 1e-5 * np.exp(-1.0), 0.15 * np.exp(-1.0)), rtol=1e-12,
        )
        self._check(ExponentialCurve(**EXP_PARAMS), oracle, h, rtol=1e-10)


# ---------------------------------------------------------------------------
# C is the true derivative of the implemented theta.
# ---------------------------------------------------------------------------
class TestDerivativeConsistency:
    """specific_moisture_capacity(h) == d/dh moisture_content(h), by central difference.

    This ties C to the *implemented* theta independently of whether the theta
    formula is itself correct. The heads are in each parameter set's own unit and
    span the steep part of each curve, where theta_s - theta is not small.
    """

    @pytest.mark.parametrize("cls,h0", [
        (VanGenuchtenCurve, -50.0),
        (VanGenuchtenCurve, -200.0),
        (VanGenuchtenCurve, -1000.0),
        (HaverkampCurve, -10.0),
        (HaverkampCurve, -30.0),
        (HaverkampCurve, -60.0),
        (ExponentialCurve, -0.3),
        (ExponentialCurve, -1.0),
        (ExponentialCurve, -3.0),
    ])
    def test_C_is_fd_of_theta(self, cls, h0):
        model = _paper_curve(cls)
        # Step relative to the head, so the truncation error is the same
        # fraction of C for every unit system.
        delta = 1e-6 * abs(h0)
        theta_p = _eval_scalar(model, "moisture_content", h0 + delta)
        theta_m = _eval_scalar(model, "moisture_content", h0 - delta)
        fd_C = (theta_p - theta_m) / (2 * delta)
        analytic_C = _eval_scalar(model, "specific_moisture_capacity", h0)
        np.testing.assert_allclose(analytic_C, fd_C, rtol=1e-5)


# ---------------------------------------------------------------------------
# Effective saturation.
# ---------------------------------------------------------------------------
class TestEffectiveSaturation:
    """effective_saturation is (theta - theta_r)/(theta_s - theta_r) and lies in [0, 1]."""

    @pytest.mark.parametrize("cls,h0", [
        (VanGenuchtenCurve, -200.0),
        (HaverkampCurve, -30.0),
        (ExponentialCurve, -2.0),
    ])
    def test_matches_definition(self, cls, h0):
        model = _paper_curve(cls)
        theta = _eval_scalar(model, "moisture_content", h0)
        expected = (theta - float(model.theta_r)) / (float(model.theta_s) - float(model.theta_r))
        Se = _eval_scalar(model, "effective_saturation", h0)
        np.testing.assert_allclose(Se, expected, rtol=1e-12)
        assert 0.0 < Se < 1.0

    @pytest.mark.parametrize("cls", ALL_FAMILIES)
    def test_saturated_is_one(self, cls):
        np.testing.assert_allclose(_eval_scalar(_paper_curve(cls), "effective_saturation", 0.5), 1.0, rtol=1e-14)


# ---------------------------------------------------------------------------
# Behaviour at the water table.
# ---------------------------------------------------------------------------
class TestWaterTable:
    """The saturated clamp, continuity at h = 0, and the documented values at h = 0."""

    @pytest.mark.parametrize("cls", ALL_FAMILIES)
    def test_saturated_branch_is_exact_clamp(self, cls):
        """For h > 0 the curve clamps exactly to (theta_s, Ks, 0)."""
        model = _paper_curve(cls)
        h = 0.5
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", h), float(model.theta_s), rtol=1e-14)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", h), float(model.Ks), rtol=1e-14)
        np.testing.assert_allclose(_eval_scalar(model, "specific_moisture_capacity", h), 0.0, atol=1e-14)

    @pytest.mark.parametrize("cls", ALL_FAMILIES)
    def test_moisture_continuous_at_seam(self, cls):
        """theta -> theta_s as h -> 0^-, for every family.

        theta has no jump at h = 0 even with the regularisation (the documented
        differences are O(reg_eps^beta) or O(reg_eps^n)).
        """
        model = _paper_curve(cls)
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", -1e-7), float(model.theta_s), rtol=1e-6)

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    def test_capacity_continuous_at_seam(self, cls):
        """C -> 0 as h -> 0^- for van Genuchten and Haverkamp.

        The exponential model is excluded: its C jumps from alpha (theta_s -
        theta_r) to 0 at h = 0, as its class docstring states.
        """
        model = _paper_curve(cls)
        np.testing.assert_allclose(_eval_scalar(model, "specific_moisture_capacity", -1e-7), 0.0, atol=1e-6)

    def test_exponential_capacity_jumps_at_seam(self):
        """The exponential C is alpha (theta_s - theta_r) just below h = 0 and 0 above."""
        model = _paper_curve(ExponentialCurve)
        expected = EXP_PARAMS["alpha"] * (EXP_PARAMS["theta_s"] - EXP_PARAMS["theta_r"])
        np.testing.assert_allclose(_eval_scalar(model, "specific_moisture_capacity", -1e-12), expected, rtol=1e-9)

    @pytest.mark.parametrize("cls", [HaverkampCurve, ExponentialCurve])
    def test_conductivity_continuous_at_seam(self, cls):
        """K -> Ks as h -> 0^- for Haverkamp and exponential."""
        model = _paper_curve(cls)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", -1e-7), float(model.Ks), rtol=1e-6)

    @pytest.mark.parametrize("n,expected_ratio", [(1.2, 0.8778), (1.5, 0.9980), (2.0, 1.0)])
    def test_van_genuchten_conductivity_at_zero_matches_documentation(self, n, expected_ratio):
        """K(0)/Ks for van Genuchten with the default reg_eps, as the class docstring states.

        The value is (1 - eps^(n-1) (1 + eps^n)^-m)^2 / (1 + eps^n)^(m/2) with
        eps = reg_eps, which is far from 1 when n is close to 1. The documented
        rounded ratios are checked too.
        """
        eps = SoilCurve.DEFAULT_REG_EPS
        m = 1.0 - 1.0 / n
        formula = (1.0 - eps ** (n - 1) * (1.0 + eps**n) ** (-m)) ** 2 / (1.0 + eps**n) ** (m / 2)
        model = _paper_curve(VanGenuchtenCurve, n=n)
        ratio = _eval_scalar(model, "hydraulic_conductivity", 0.0) / float(model.Ks)
        np.testing.assert_allclose(ratio, formula, rtol=1e-10)
        np.testing.assert_allclose(ratio, expected_ratio, atol=1e-4)

    def test_regularisation_is_relative_to_the_head_scale(self):
        """At h = 0 the regularised magnitude is reg_eps times the curve's head scale.

        For Haverkamp the head scales are alpha^(1/beta) and A^(1/gamma), so
        theta(0) = theta_r + (theta_s - theta_r)/(1 + eps^beta) and
        K(0) = Ks/(1 + eps^gamma), independent of the units of alpha and A. A
        large reg_eps makes the offset visible. The sand set has alpha and A of
        order 1e6, so an offset that ignored the scale would give values close
        to theta_s and Ks.
        """
        eps = 0.1
        p = HAVERKAMP_SAND_PARAMS
        model = HaverkampCurve(**p, reg_eps=eps)
        theta0 = p["theta_r"] + (p["theta_s"] - p["theta_r"]) / (1.0 + eps ** p["beta"])
        K0 = p["Ks"] / (1.0 + eps ** p["gamma"])
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", 0.0), theta0, rtol=1e-12)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", 0.0), K0, rtol=1e-12)

        # van Genuchten regularises the dimensionless |alpha h| directly.
        p = VG_PAPER_PARAMS
        m = 1.0 - 1.0 / p["n"]
        model = VanGenuchtenCurve(**p, reg_eps=eps)
        theta0 = p["theta_r"] + (p["theta_s"] - p["theta_r"]) * (1.0 + eps ** p["n"]) ** (-m)
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", 0.0), theta0, rtol=1e-12)


# ---------------------------------------------------------------------------
# The three families are distinct curves.
# ---------------------------------------------------------------------------
class TestModelDistinctness:
    """Guards against an accidental collapse of one family onto another."""

    def test_families_give_distinct_curves(self):
        common = dict(theta_r=0.15, theta_s=0.45, Ks=1e-5)
        haverkamp = HaverkampCurve(alpha=1.0, beta=2.0, A=1.0, gamma=2.0, **common)
        vg = VanGenuchtenCurve(alpha=0.5, n=2.0, **common)
        exp = ExponentialCurve(alpha=0.5, **common)

        for hval in (-0.5, -1.0, -2.0, -5.0):
            thetas = [_eval_scalar(m, "moisture_content", hval) for m in (haverkamp, vg, exp)]
            Ks_vals = [_eval_scalar(m, "hydraulic_conductivity", hval) for m in (haverkamp, vg, exp)]
            # No two of the three may coincide at this head.
            for i in range(3):
                for j in range(i + 1, 3):
                    assert not np.isclose(thetas[i], thetas[j], rtol=1e-3), (
                        f"theta collapsed for families {i},{j} at h={hval}"
                    )
                    assert not np.isclose(Ks_vals[i], Ks_vals[j], rtol=1e-3), (
                        f"K collapsed for families {i},{j} at h={hval}"
                    )


# ---------------------------------------------------------------------------
# Regularisation of the singular magnitude derivative at the water table.
# ---------------------------------------------------------------------------
class TestRegularisationAtWaterTable:
    """Regularisation of the singular |.|^p derivative at exactly h = 0.

    For van Genuchten and Haverkamp with a non-integer exponent, the unsaturated
    branch contains a magnitude raised to a non-integer power. Differentiated by
    UFL this produces a term that is Inf (and NaN after sign(0)*Inf or 0*Inf) at
    exactly h = 0, which would make the Newton Jacobian and the pyadjoint
    gradient NaN. The reg_eps regularisation must keep every property AND its
    derivative finite at h = 0.
    """

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_values_and_derivatives_finite_at_zero(self, cls, property_name):
        """With the default reg_eps, value and derivative are finite at h = 0."""
        model = _make_curve(cls)

        value = _eval_scalar(model, property_name, 0.0)
        assert np.all(np.isfinite(value)), (
            f"{cls.__name__}.{property_name} value not finite at h=0"
        )

        deriv = _jacobian_diag(model, property_name, 0.0)
        assert np.all(np.isfinite(deriv)), (
            f"d({cls.__name__}.{property_name})/dh not finite at h=0"
        )

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    def test_unregularised_form_is_singular_at_zero(self, cls):
        """With reg_eps=0 the exact form goes NaN/Inf at h = 0.

        At least one of the three property derivatives must be non-finite at
        h = 0 when the regularisation is disabled. If this stops holding, the
        test above no longer guards anything.
        """
        model = _make_curve(cls, reg_eps=0.0)
        any_singular = False
        for property_name in PROPERTY_NAMES:
            deriv = _jacobian_diag(model, property_name, 0.0)
            if not np.all(np.isfinite(deriv)):
                any_singular = True
        assert any_singular, (
            f"{cls.__name__} unexpectedly finite at h=0 with reg_eps=0; "
            "the singular-derivative case is no longer reproduced"
        )

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_forward_unchanged_away_from_zero(self, cls, property_name):
        """Forward value matches the exact (reg_eps=0) form away from h = 0.

        The regularisation is O(reg_eps^2) there and must not change the
        constitutive relations in the unsaturated region to several digits.
        """
        model_reg = _make_curve(cls)
        model_exact = _make_curve(cls, reg_eps=0.0)
        v_reg = _eval_scalar(model_reg, property_name, -1.0)
        v_exact = _eval_scalar(model_exact, property_name, -1.0)
        np.testing.assert_allclose(v_reg, v_exact, rtol=1e-7, atol=1e-30)


# ---------------------------------------------------------------------------
# pyadjoint Taylor tests.
# ---------------------------------------------------------------------------
def _taylor_rate(build_functional, control, perturbation):
    """Annotate a functional, check that replay reproduces it, and return the Taylor rate.

    Args:
        build_functional: Callable that builds the curve and returns the assembled
            functional. It is called while annotation is on.
        control: The Function to differentiate with respect to.
        perturbation: Direction for the Taylor test.

    Returns:
        The smallest convergence rate reported by pyadjoint's taylor_test.
    """
    from firedrake.adjoint import (
        continue_annotation, pause_annotation, Control,
        ReducedFunctional, taylor_test, get_working_tape,
    )

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        J = build_functional()
        rf = ReducedFunctional(J, Control(control))
    finally:
        pause_annotation()

    try:
        # Replay at the recorded control must reproduce the recorded value; a
        # mismatch means the tape did not capture the forward evaluation.
        np.testing.assert_allclose(rf(control), J, rtol=1e-12)
        with contextlib.redirect_stdout(io.StringIO()):
            rate = taylor_test(rf, control, perturbation)
    finally:
        tape.clear_tape()
    return rate


class TestAdjointTaylor:
    """End-to-end pyadjoint gradient checks on a functional of a soil curve.

    The functional is the integral of one property. It is nonlinear in the head
    and in the parameters, so second-order convergence of the Taylor remainder
    checks the adjoint-propagated gradient through the regularised forms. The
    quadrature degree is fixed: with automatic degree selection the functional
    and its derivative are integrated at different degrees, and the remainder
    stops converging at second order for reasons unrelated to the gradient.
    """

    QUADRATURE_DEGREE = 8

    @staticmethod
    def _head_field(V, case):
        """Head field and perturbation direction for a Taylor test over h.

        "unsaturated" stays in [-1.3, -0.3]. "touching" is h = -x, which is zero
        at the node x = 0, where the regularisation acts. Its perturbation is
        non-positive, so h + eps dh stays in the unsaturated branch: the branch
        switch at h > 0 is not differentiable and would spoil the rate for
        reasons unrelated to the gradient.
        """
        x, = fd.SpatialCoordinate(V.mesh())
        if case == "unsaturated":
            return (fd.Function(V).interpolate(-0.3 - x),
                    fd.Function(V).interpolate(0.1 * fd.cos(7.0 * x) + 0.1))
        return (fd.Function(V).interpolate(-x),
                fd.Function(V).interpolate(-(0.1 * fd.cos(7.0 * x) + 0.1)))

    @pytest.mark.parametrize("case", ["unsaturated", "touching"])
    @pytest.mark.parametrize("cls", ALL_FAMILIES)
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_head_control(self, cls, property_name, case):
        """Gradient with respect to the head field."""
        V = fd.FunctionSpace(fd.UnitIntervalMesh(4), "CG", 1)
        h, dh = self._head_field(V, case)
        model = _make_curve(cls)
        dx = fd.dx(degree=self.QUADRATURE_DEGREE)

        rate = _taylor_rate(lambda: fd.assemble(getattr(model, property_name)(h) * dx), h, dh)
        assert rate > 1.9, f"{cls.__name__}.{property_name} ({case}) Taylor rate {rate} not ~2"

    @pytest.mark.parametrize("cls,parameter,property_name", [
        (VanGenuchtenCurve, "n", "moisture_content"),
        (VanGenuchtenCurve, "n", "hydraulic_conductivity"),
        (VanGenuchtenCurve, "n", "specific_moisture_capacity"),
        (HaverkampCurve, "beta", "moisture_content"),
        (HaverkampCurve, "gamma", "hydraulic_conductivity"),
        (HaverkampCurve, "beta", "specific_moisture_capacity"),
        (ExponentialCurve, "alpha", "moisture_content"),
        (ExponentialCurve, "alpha", "hydraulic_conductivity"),
        (ExponentialCurve, "alpha", "specific_moisture_capacity"),
    ])
    def test_parameter_control(self, cls, parameter, property_name):
        """Gradient with respect to a Real-space soil parameter.

        The head field touches h = 0, so the regularised forms are exercised.
        Each property is paired with a parameter it depends on (Haverkamp K
        does not depend on beta).
        """
        mesh = fd.UnitIntervalMesh(4)
        V = fd.FunctionSpace(mesh, "CG", 1)
        R = fd.FunctionSpace(mesh, "R", 0)
        h, _ = self._head_field(V, "touching")
        nominal = float(getattr(_make_curve(cls), parameter))
        control = fd.Function(R).assign(nominal)
        dx = fd.dx(degree=self.QUADRATURE_DEGREE)

        def functional():
            model = _make_curve(cls, **{parameter: control})
            return fd.assemble(getattr(model, property_name)(h) * dx)

        rate = _taylor_rate(functional, control, fd.Function(R).assign(0.1 * nominal))
        assert rate > 1.9, f"{cls.__name__}.{property_name} w.r.t. {parameter}: Taylor rate {rate} not ~2"


# ---------------------------------------------------------------------------
# Heterogeneous (spatially varying) soil parameters.
# ---------------------------------------------------------------------------
class TestHeterogeneousParameters:
    """Spatially varying parameters are supported.

    A Function parameter must be stored by reference and produce a spatially
    varying, finite theta/K/C with a finite Jacobian, including at the water
    table.
    """

    def _vg_heterogeneous(self):
        mesh = fd.UnitIntervalMesh(4)
        P = fd.FunctionSpace(mesh, "CG", 1)
        x, = fd.SpatialCoordinate(mesh)
        alpha = fd.Function(P).interpolate(0.3 + 0.4 * x)     # 0.3 .. 0.7
        theta_s = fd.Function(P).interpolate(0.40 + 0.10 * x)  # 0.40 .. 0.50
        Ss = fd.Function(P).interpolate(0.01 + 0.0 * x)
        model = VanGenuchtenCurve(
            theta_r=0.10, theta_s=theta_s, Ks=1e-5, Ss=Ss, alpha=alpha, n=2.0,
        )
        return mesh, P, model, alpha, theta_s, Ss

    def test_fields_stored_by_reference(self):
        """A Function parameter is stored as-is, never collapsed to a scalar."""
        _, _, model, alpha, theta_s, Ss = self._vg_heterogeneous()
        assert model.alpha is alpha
        assert model.theta_s is theta_s
        assert model.Ss is Ss

    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_values_finite_and_spatially_varying(self, property_name):
        _, P, model, _, _, _ = self._vg_heterogeneous()
        h = fd.Function(P).interpolate(fd.Constant(-1.0))
        out = fd.Function(P).interpolate(getattr(model, property_name)(h))
        data = out.dat.data
        assert np.all(np.isfinite(data)), f"{property_name} not finite for heterogeneous params"
        assert np.ptp(data) > 0.0, f"{property_name} did not vary in space despite varying params"

    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_jacobian_finite_at_water_table(self, property_name):
        """The regularisation must hold pointwise across a heterogeneous field."""
        mesh, _, model, _, _, _ = self._vg_heterogeneous()
        deriv = _jacobian_diag(model, property_name, 0.0, mesh=mesh)
        assert np.all(np.isfinite(deriv)), (
            f"heterogeneous {property_name} Jacobian not finite at h=0"
        )

    def test_real_space_parameter_stored_by_reference(self):
        """A Real-space Function (a scalar control) is preserved by identity."""
        R = fd.FunctionSpace(create_test_mesh(), "R", 0)
        ss = fd.Function(R).assign(0.02)
        model = _paper_curve(VanGenuchtenCurve, Ss=ss)
        assert model.Ss is ss
        assert isinstance(model.Ss, ufl.core.expr.Expr)


# ---------------------------------------------------------------------------
# Robustness across the parameter envelope.
# ---------------------------------------------------------------------------
class TestRobustnessEnvelope:
    """The regularisation must hold beyond the nominal n = 1.5 test point."""

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("n", [1.01, 1.5])
    @pytest.mark.parametrize("hval", [0.0, -1e3])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_value_and_jacobian_finite(self, cls, n, hval, property_name):
        """n close to 1 and large |h| keep value and Jacobian finite."""
        model = _make_curve(cls, n=n)
        assert np.all(np.isfinite(_eval_scalar(model, property_name, hval)))
        assert np.all(np.isfinite(_jacobian_diag(model, property_name, hval)))
