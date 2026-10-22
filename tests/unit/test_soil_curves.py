"""
Unit tests for soil curve models in the Richards equation module.

These cover the Haverkamp, van Genuchten and exponential constitutive models
for unsaturated flow. The suite is built around a few deliberate principles:

- Constitutive values are checked against an *independent* numpy reimplementation
  of the published closed forms (van Genuchten 1980; Haverkamp et al. 1977), not
  against a restatement of the gadopt code, so a transcription error in
  ``soil_curves.py`` actually fails a test.
- The specific moisture capacity C is checked to be the true derivative of the
  *implemented* moisture content theta via a finite difference of theta.
- The reg_eps regularisation that keeps the branch-wise UFL derivative (the
  Newton Jacobian and the pyadjoint adjoint) finite at the water table h = 0 is
  guarded both ways: the regularised form stays finite, and the unregularised
  form is shown to genuinely go singular (so the finiteness test is not vacuous).
- The adjoint pipeline itself is exercised with a real pyadjoint Taylor test.
- Heterogeneous (spatially varying) soil parameters are a first-class supported
  capability and are checked across theta, K, C and the Jacobian.

References:
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the
        hydraulic conductivity of unsaturated soils. Soil Sci. Soc. Am. J. 44,
        892-898. Retention Eq. [21], conductivity Eq. [9], m = 1 - 1/n Eq. [22],
        derivative Eq. [23]. Canonical parameters from Fig. 1 (alpha = 0.005/cm,
        n = 2, theta_r = 0.10, theta_s = 0.50) and Fig. 3 (Ks = 100 cm/day).
    Haverkamp, R., et al. (1977). A comparison of numerical simulation models
        for one-dimensional infiltration. Soil Sci. Soc. Am. J. 41, 285-294.
"""

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
    "water_retention",
)


# ---------------------------------------------------------------------------
# Independent numpy oracles for the published closed forms.
#
# These are written straight from the papers' equations and share no code with
# gadopt.soil_curves, so they form a genuine cross-check: a wrong exponent or a
# dropped factor in the UFL implementation will disagree with these. Each is the
# unsaturated branch (h <= 0); the saturated branch is a separate clamp.
# ---------------------------------------------------------------------------
def _vg_oracle(h, theta_r, theta_s, Ks, alpha, n):
    """van Genuchten 1980, unsaturated branch. theta Eq. [21], K Eq. [9].

    C = d(theta)/dh is derived from Eq. [21] with |h| = -h for h < 0:
        C = (theta_s - theta_r) m n alpha u^(n-1) (1 + u^n)^(-m-1),  u = |alpha h|.
    """
    m = 1.0 - 1.0 / n
    u = np.abs(alpha * h)
    theta = theta_r + (theta_s - theta_r) * (1.0 + u**n) ** (-m)
    term1 = 1.0 - u ** (n - 1) * (1.0 + u**n) ** (-m)
    term2 = (1.0 + u**n) ** (m / 2.0)
    K = Ks * term1**2 / term2
    C = (theta_s - theta_r) * m * n * alpha * u ** (n - 1) * (1.0 + u**n) ** (-m - 1.0)
    return theta, K, C


def _haverkamp_oracle(h, theta_r, theta_s, Ks, alpha, beta, A, gamma):
    """Haverkamp et al. 1977 closed forms (the forms in the module docstring).

    C = d(theta)/dh = alpha beta (theta_s - theta_r) |h|^(beta-1) / (alpha + |h|^beta)^2.
    """
    w = np.abs(h)
    theta = theta_r + alpha * (theta_s - theta_r) / (alpha + w**beta)
    K = Ks * A / (A + w**gamma)
    C = alpha * beta * (theta_s - theta_r) * w ** (beta - 1) / (alpha + w**beta) ** 2
    return theta, K, C


def _exp_oracle(h, theta_r, theta_s, Ks, alpha):
    """Exponential model (smooth, no magnitude term)."""
    e = np.exp(alpha * h)
    theta = theta_r + (theta_s - theta_r) * e
    K = Ks * e
    C = (theta_s - theta_r) * alpha * e
    return theta, K, C


# Parameter sets used for the oracle comparisons. The van Genuchten set is the
# published Fig. 1 / Fig. 3 example so the test is anchored to a real curve.
VG_PAPER_PARAMS = dict(theta_r=0.10, theta_s=0.50, Ks=100.0, alpha=0.005, n=2.0)
HAVERKAMP_PARAMS = dict(
    theta_r=0.15, theta_s=0.45, Ks=1e-5, alpha=1.0, beta=2.0, A=1.0, gamma=2.0
)
EXP_PARAMS = dict(theta_r=0.15, theta_s=0.45, Ks=1e-5, alpha=0.5)


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

    This mirrors exactly the branch-wise UFL differentiation that feeds the
    Newton Jacobian and the pyadjoint adjoint, so finiteness here is finiteness
    of both.
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


def _make_curve(cls, reg_eps, n=1.5):
    """Build a curve with a NON-INTEGER exponent that exposes the |.|^p singularity.

    A non-integer n/beta/gamma is what makes the unsaturated-branch derivative
    blow up at h = 0. The exponential model has no such exponent and is excluded
    from the regularisation tests.
    """
    if cls is VanGenuchtenCurve:
        return VanGenuchtenCurve(
            theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
            alpha=0.5, n=n, reg_eps=reg_eps,
        )
    if cls is HaverkampCurve:
        return HaverkampCurve(
            theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
            alpha=1.0, beta=n, A=1.0, gamma=n, reg_eps=reg_eps,
        )
    raise ValueError(cls)


# ---------------------------------------------------------------------------
# Base class and parameter validation.
# ---------------------------------------------------------------------------
class TestSoilCurveBase:
    """Base-class contract and parameter validation paths."""

    def test_abstract_base_class(self):
        """SoilCurve is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SoilCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0)

    def test_missing_parameter_raises_each_family(self):
        """Every family reports a missing required parameter."""
        with pytest.raises(ValueError, match="Missing required parameter"):
            HaverkampCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                           alpha=1.0, beta=2.0)  # missing A, gamma
        with pytest.raises(ValueError, match="Missing required parameter"):
            VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                              alpha=0.5)  # missing n
        with pytest.raises(ValueError, match="Missing required parameter"):
            ExponentialCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0)  # missing alpha

    def test_van_genuchten_n_guard(self):
        """The scalar n > 1 guard fires at and below the boundary, not above."""
        with pytest.raises(ValueError, match="n must be > 1.0"):
            VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                              alpha=0.5, n=0.5)
        # n exactly 1.0 is the boundary of the strict ">" guard and must raise.
        with pytest.raises(ValueError, match="n must be > 1.0"):
            VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                              alpha=0.5, n=1.0)
        # n just above the boundary is accepted.
        VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                          alpha=0.5, n=1.0001)

    def test_van_genuchten_n_guard_bypassed_for_ufl_n(self):
        """A UFL-expression n intentionally bypasses the scalar range guard.

        The guard only validates a scalar Constant; a spatially varying or
        control n is a UFL expression and is left to the user, so construction
        must not raise even with a nominally out-of-range value.
        """
        mesh = create_test_mesh()
        R = fd.FunctionSpace(mesh, "R", 0)
        n_field = fd.Function(R).assign(0.5)  # would be rejected as a scalar
        VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                          alpha=0.5, n=n_field)


# ---------------------------------------------------------------------------
# Constitutive values vs independent published oracle.
# ---------------------------------------------------------------------------
class TestConstitutiveValues:
    """theta, K, C against an independent numpy reimplementation of the papers.

    With the default reg_eps the regularisation perturbs the result only at
    O(reg_eps^2) away from h = 0, so an exact-form oracle agrees to ~1e-6 here.
    """

    def test_van_genuchten_matches_paper(self):
        # Evaluated at h = -200 cm so |alpha h| = 1 exactly. Hand values from the
        # van Genuchten 1980 closed forms (Eq. [21], [9], [23]) at the Fig. 1/3
        # parameters: theta = 0.382842712, K = 7.213750788, C = 7.0710678e-4.
        h = -200.0
        model = VanGenuchtenCurve(Ss=0.0, **VG_PAPER_PARAMS)
        theta_o, K_o, C_o = _vg_oracle(h, **VG_PAPER_PARAMS)
        np.testing.assert_allclose(theta_o, 0.382842712, rtol=1e-8)  # anchor
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", h), theta_o, rtol=1e-6)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", h), K_o, rtol=1e-6)
        np.testing.assert_allclose(_eval_scalar(model, "water_retention", h), C_o, rtol=1e-6)

    def test_haverkamp_matches_closed_form(self):
        h = -2.0
        model = HaverkampCurve(Ss=0.0, **HAVERKAMP_PARAMS)
        theta_o, K_o, C_o = _haverkamp_oracle(h, **HAVERKAMP_PARAMS)
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", h), theta_o, rtol=1e-6)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", h), K_o, rtol=1e-6)
        np.testing.assert_allclose(_eval_scalar(model, "water_retention", h), C_o, rtol=1e-6)

    def test_exponential_matches_closed_form(self):
        h = -2.0
        model = ExponentialCurve(Ss=0.0, **EXP_PARAMS)
        theta_o, K_o, C_o = _exp_oracle(h, **EXP_PARAMS)
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", h), theta_o, rtol=1e-10)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", h), K_o, rtol=1e-10)
        np.testing.assert_allclose(_eval_scalar(model, "water_retention", h), C_o, rtol=1e-10)


# ---------------------------------------------------------------------------
# C is the true derivative of the implemented theta.
# ---------------------------------------------------------------------------
class TestDerivativeConsistency:
    """water_retention(h) == d/dh moisture_content(h), by central difference.

    This ties C to the *implemented* theta independently of whether the theta
    formula is itself correct (the value of theta is checked separately against
    the published closed forms).
    """

    @pytest.mark.parametrize("model", [
        VanGenuchtenCurve(Ss=0.0, **VG_PAPER_PARAMS),
        HaverkampCurve(Ss=0.0, **HAVERKAMP_PARAMS),
        ExponentialCurve(Ss=0.0, **EXP_PARAMS),
    ], ids=["vanGenuchten", "Haverkamp", "exponential"])
    @pytest.mark.parametrize("h0", [-0.3, -1.0, -3.0])
    def test_C_is_fd_of_theta(self, model, h0):
        delta = 1e-6
        theta_p = _eval_scalar(model, "moisture_content", h0 + delta)
        theta_m = _eval_scalar(model, "moisture_content", h0 - delta)
        fd_C = (theta_p - theta_m) / (2 * delta)
        analytic_C = _eval_scalar(model, "water_retention", h0)
        np.testing.assert_allclose(analytic_C, fd_C, rtol=1e-5)


# ---------------------------------------------------------------------------
# Branch seam at the water table.
# ---------------------------------------------------------------------------
class TestBranchSeam:
    """Continuity at h = 0 and the exact saturated clamp for h > 0."""

    @pytest.mark.parametrize("model,theta_s,Ks", [
        (VanGenuchtenCurve(Ss=0.0, **VG_PAPER_PARAMS), 0.50, 100.0),
        (HaverkampCurve(Ss=0.0, **HAVERKAMP_PARAMS), 0.45, 1e-5),
    ], ids=["vanGenuchten", "Haverkamp"])
    def test_saturated_branch_is_exact_clamp(self, model, theta_s, Ks):
        """For h > 0 the curve clamps exactly to (theta_s, Ks, 0)."""
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", 0.5), theta_s, rtol=1e-12)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", 0.5), Ks, rtol=1e-12)
        np.testing.assert_allclose(_eval_scalar(model, "water_retention", 0.5), 0.0, atol=1e-14)

    @pytest.mark.parametrize("model,theta_s,Ks", [
        (VanGenuchtenCurve(Ss=0.0, **VG_PAPER_PARAMS), 0.50, 100.0),
        (HaverkampCurve(Ss=0.0, **HAVERKAMP_PARAMS), 0.45, 1e-5),
    ], ids=["vanGenuchten", "Haverkamp"])
    def test_unsaturated_branch_is_continuous_at_seam(self, model, theta_s, Ks):
        """The unsaturated branch limits to the saturated clamp as h -> 0^-.

        theta -> theta_s, K -> Ks (to O(reg_eps)) and C -> 0, so there is no
        jump at the water table. Guards against a flipped/displaced seam.
        """
        h = -1e-7
        np.testing.assert_allclose(_eval_scalar(model, "moisture_content", h), theta_s, rtol=1e-6)
        np.testing.assert_allclose(_eval_scalar(model, "hydraulic_conductivity", h), Ks, rtol=1e-4)
        np.testing.assert_allclose(_eval_scalar(model, "water_retention", h), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# The three families are genuinely distinct curves.
# ---------------------------------------------------------------------------
class TestModelDistinctness:
    """Guards against an accidental collapse of one family onto another."""

    def test_families_give_distinct_curves(self):
        common = dict(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0)
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
    branch contains a magnitude raised to a non-integer/negative power.
    Differentiated branch-wise by UFL this produces a term that is Inf (and NaN
    after sign(0)*Inf or 0*Inf) at exactly h = 0, poisoning the Newton Jacobian
    and the pyadjoint gradient. The reg_eps smoothing must keep all of
    moisture_content, hydraulic_conductivity and water_retention AND their
    derivatives finite at h = 0.
    """

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_values_and_derivatives_finite_at_zero(self, cls, property_name):
        """With the default reg_eps, value and derivative are finite at h = 0."""
        model = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)

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
        """Sanity check: with reg_eps=0 the old form really does go NaN/Inf.

        At least one of the three property derivatives must be non-finite at
        h = 0 when the regularisation is disabled. This is the failure mode the
        fix removes; if this assertion ever stops holding the test above is no
        longer guarding anything.
        """
        model = _make_curve(cls, reg_eps=0.0)
        any_singular = False
        for property_name in PROPERTY_NAMES:
            deriv = _jacobian_diag(model, property_name, 0.0)
            if not np.all(np.isfinite(deriv)):
                any_singular = True
        assert any_singular, (
            f"{cls.__name__} unexpectedly finite at h=0 with reg_eps=0; "
            "the singular-derivative regression is no longer reproduced"
        )

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_forward_unchanged_away_from_zero(self, cls, property_name):
        """Forward value matches the exact (reg_eps=0) form away from h = 0.

        The regularisation is O(reg_eps^2) and must not perturb the constitutive
        relations in the bulk unsaturated region to several digits.
        """
        model_reg = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)
        model_exact = _make_curve(cls, reg_eps=0.0)
        v_reg = _eval_scalar(model_reg, property_name, -1.0)
        v_exact = _eval_scalar(model_exact, property_name, -1.0)
        np.testing.assert_allclose(v_reg, v_exact, rtol=1e-7, atol=1e-30)


# ---------------------------------------------------------------------------
# A pyadjoint Taylor test of the adjoint pipeline.
# ---------------------------------------------------------------------------
class TestAdjointTaylor:
    """End-to-end pyadjoint gradient check on a functional of a soil curve.

    Unlike the UFL-derivative checks above, this annotates the forward
    evaluation, builds a ReducedFunctional over the head field as control, and
    runs pyadjoint's own taylor_test: second-order convergence confirms the
    adjoint-propagated gradient through the regularised constitutive form is
    correct. This is the capability the regularisation exists to protect.
    """

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_pyadjoint_taylor_second_order(self, cls, property_name):
        from firedrake.adjoint import (
            continue_annotation, pause_annotation, Control,
            ReducedFunctional, taylor_test, get_working_tape,
        )

        model = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)
        mesh = fd.UnitIntervalMesh(4)
        V = fd.FunctionSpace(mesh, "CG", 1)
        x, = fd.SpatialCoordinate(mesh)

        # Strictly unsaturated head field (range -0.3 .. -1.3), set before
        # annotation so it is a clean leaf control.
        h = fd.Function(V).interpolate(-0.3 - x)

        tape = get_working_tape()
        tape.clear_tape()
        continue_annotation()
        try:
            J = fd.assemble(getattr(model, property_name)(h) * fd.dx)
            rf = ReducedFunctional(J, Control(h))
        finally:
            pause_annotation()

        dh = fd.Function(V).interpolate(0.1 * fd.cos(7.0 * x) + 0.1)
        rate = taylor_test(rf, h, dh)
        tape.clear_tape()

        assert rate > 1.9, (
            f"{cls.__name__}.{property_name} pyadjoint Taylor rate {rate} not ~2"
        )


# ---------------------------------------------------------------------------
# Heterogeneous (spatially varying) soil parameters.
# ---------------------------------------------------------------------------
class TestHeterogeneousParameters:
    """Spatially varying parameters are a supported, exercised capability.

    A UFL-expression parameter (e.g. a Function over the mesh) must pass through
    untouched and produce a genuinely spatially varying, finite theta/K/C with a
    finite Jacobian, including at the water table.
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

    def test_fields_pass_through_untouched(self):
        """A Function parameter is stored as-is, never collapsed to a scalar."""
        _, _, model, alpha, theta_s, Ss = self._vg_heterogeneous()
        assert model.parameters["alpha"] is alpha
        assert model.parameters["theta_s"] is theta_s
        # Ss is routed through ensure_constant, which passes Functions through.
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


# ---------------------------------------------------------------------------
# Ss / ensure_constant routing.
# ---------------------------------------------------------------------------
class TestParameterRouting:
    """Ss is wrapped if scalar and passed through if already a field/control."""

    def test_scalar_Ss_becomes_symbolic_constant(self):
        """A bare float Ss is wrapped so it is usable in a UFL form."""
        model = VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                                  alpha=0.5, n=2.0)
        assert isinstance(model.Ss, ufl.core.expr.Expr)
        assert not isinstance(model.Ss, float)
        assert float(model.Ss) == 0.0

    def test_field_Ss_passes_through_unchanged(self):
        """A Function/control Ss is preserved by identity, never float()'d."""
        mesh = create_test_mesh()
        R = fd.FunctionSpace(mesh, "R", 0)
        ss = fd.Function(R).assign(0.02)
        model = VanGenuchtenCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=ss,
                                  alpha=0.5, n=2.0)
        assert model.Ss is ss


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
        """n close to 1 and large |h| keep value and Jacobian finite at the seam."""
        model = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS, n=n)
        assert np.all(np.isfinite(_eval_scalar(model, property_name, hval)))
        assert np.all(np.isfinite(_jacobian_diag(model, property_name, hval)))
