"""Tests of `eb_fit.py` (design handout section 12.3, items 1 and 4).

Serial and fast (no Firedrake). Run with

    python -m pytest demos/glacial_isostatic_adjustment/giamip/eb_fit
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eb_fit  # noqa: E402


# ---------------------------------------------------------------------------
# The EB modulus
# ---------------------------------------------------------------------------


def test_short_time_limit_is_the_unrelaxed_modulus():
    """mu_EB -> 1 as s' -> infinity: the instant response is mu_inf.

    The band integral decays as (s' tau_H)^(-alpha), slowly for alpha =
    0.274, so the limit is approached only at very large s'.
    """
    assert abs(eb_fit.mu_eb(1e30) - 1.0) < 1e-6
    # Monotone approach from below on the way there.
    values = eb_fit.mu_eb(np.logspace(2, 30, 15))
    assert np.all(np.diff(values) > 0) and np.all(values < 1.0)


def test_long_time_limit_is_the_dashpot():
    """mu_EB -> s' as s' -> 0: the steady-state dashpot of viscosity eta_0."""
    for s in (1e-6, 1e-8):
        assert abs(eb_fit.mu_eb(s) / s - 1.0) < 1e-4


def test_quadrature_matches_the_closed_form():
    """The quadrature of the band integral equals 2F1 to 1e-10, real and complex s'."""
    for s in [1e-3, 0.1, 1.0, 10.0, 1e3, 1e5,
              0.1j, 1j, 10j, 1e3j, 0.5 + 2j]:
        quad = eb_fit.band_integral_quadrature(s)
        closed = eb_fit.band_integral_closed(s)
        assert abs(quad - closed) <= 1e-10 * abs(closed), (s, quad, closed)
        mu_q = eb_fit.mu_eb(s, closed=False)
        mu_c = eb_fit.mu_eb(s, closed=True)
        assert abs(mu_q - mu_c) <= 1e-10 * abs(mu_c)


def test_reference_values_of_the_handout():
    """mu_EB near 0.276 at s' = 1 and 0.76 at s' = 1e3 (real s', handout 12.3).

    The handout quotes these to two or three digits; they rule out the sign
    error of Lau (2023) Eq. 20 as printed (short-time limit about
    1 / (1 + 2 Delta) = 0.21). This code gives 0.27555 and 0.75460 with
    tau_L = 0, and 0.27556 and 0.75506 with tau_L = 1e-12. The handout's
    0.76 is the second number rounded up, so the comparison allows 6e-3.
    """
    assert abs(float(eb_fit.mu_eb(1.0)) - 0.276) < 1e-3
    assert abs(float(eb_fit.mu_eb(1e3)) - 0.76) < 6e-3
    # Pinned values of this code, so a change of the formula is seen.
    assert abs(float(eb_fit.mu_eb(1.0)) - 0.2755472364602175) < 1e-12
    assert abs(float(eb_fit.mu_eb(1e3)) - 0.7546007854636978) < 1e-12


def test_lower_end_of_the_band_is_negligible():
    """tau_L = 1e-12 instead of 0 changes mu_EB by less than 1e-3 relative."""
    for s in (1.0, 1j, 1e3):
        a = eb_fit.mu_eb(s, tau_l=0.0, closed=False)
        b = eb_fit.mu_eb(s, tau_l=1e-12, closed=False)
        assert abs(a - b) < 1e-3 * abs(a)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_one_maxwell_element_is_recovered():
    """With Delta = 0 and the explicit grid tau_bar = [1], the fit is one Maxwell element.

    EB without its band is the Maxwell model (J = 1 + 1/s'). With one element
    of time 1 the constraint alone forces mu_bar = 1, so the assertion that
    carries the test is the zero misfit: mu_EB with Delta = 0 is exactly one
    Maxwell element.
    """
    f = eb_fit.fit(tau=[1.0], delta=0.0)
    assert f["n"] == 1 and f["tau_bar"] == [1.0]
    assert abs(f["mu_bar"][0] - 1.0) < 1e-12
    assert f["max_rel_error"] < 1e-12


@pytest.mark.parametrize("n", [3, 4])
def test_the_fit_meets_its_conditions(n):
    """Sum condition to 1e-12, weights above the floor, error and instant modulus."""
    f = eb_fit.fit(n)
    assert f["n"] == n and not f["dropped"]
    assert f["tau_bar_max"] == pytest.approx(1.0 + eb_fit.DELTA)
    # The long-term viscosity of the fit is eta_0.
    assert abs(f["sum_mu_tau"] - 1.0) < 1e-12
    assert min(f["mu_bar"]) >= eb_fit.MU_FLOOR
    # The element times are the log grid from 1e-2 to 1 + Delta.
    assert np.allclose(f["tau_bar"], np.logspace(-2, np.log10(2.9), n))
    # The instant modulus of the fit: the handout expects 0.65-0.76 (6.3).
    assert 0.65 < f["sum_mu_bar"] < 0.77
    # A rough fit over three decades with 3-4 elements (S17): this code
    # gives 3.2 % (N = 3) and 3.4 % (N = 4) on the grid end 2.9.
    assert f["max_rel_error"] < 0.05
    assert f["l2_rel_error"] <= f["max_rel_error"]


def test_an_element_below_the_floor_is_dropped():
    """A grid with an element far outside the band loses that element.

    An element of tau_bar = 1e4 carries a tiny modulus at best (its
    viscosity share would be 1e4 mu, and the constraint makes the fit push
    it to zero or below), so it is dropped and the fit is repeated.
    """
    f = eb_fit.fit(tau=[0.01, 0.17, 2.9, 1e4])
    assert 1e4 in f["dropped"]
    assert f["n"] == 3
    assert abs(f["sum_mu_tau"] - 1.0) < 1e-12
    assert min(f["mu_bar"]) >= eb_fit.MU_FLOOR


def test_main_writes_the_json_and_the_plot(tmp_path):
    """`main` writes `eb_fit.json` with the selected fit and `eb_fit.png`."""
    record = eb_fit.main(["--n", "3", "4", "--select", "3", "--no_free",
                          "--output_dir", str(tmp_path)])
    path = tmp_path / "eb_fit.json"
    png = tmp_path / "eb_fit.png"
    assert path.exists() and png.exists() and png.stat().st_size > 10_000
    loaded = json.loads(path.read_text())
    assert loaded["selected"]["n"] == 3
    assert set(loaded["fits"]) == {"3", "4"}
    assert loaded["selected"] == record["selected"]
    assert len(loaded["script_sha256"]) == 64


def test_a_small_positive_modulus_is_dropped(monkeypatch):
    """The floor applies to a small positive modulus too, not only a negative one.

    The N = 4 grid fit has the modulus 0.085 at tau_bar = 0.438. With the
    floor raised to 0.1 that element is dropped first and the fit is
    repeated; the refit can drop a further element. The remaining elements
    keep the sum condition and are all above the floor.
    """
    monkeypatch.setattr(eb_fit, "MU_FLOOR", 0.1)
    f = eb_fit.fit(4)
    assert f["dropped"][0] == pytest.approx(0.43813, rel=1e-4)
    assert f["n"] == 4 - len(f["dropped"])
    assert abs(f["sum_mu_tau"] - 1.0) < 1e-12
    assert min(f["mu_bar"]) >= 0.1


def test_constrained_lsq_against_a_hand_solution():
    """`constrained_lsq` on a problem with a known answer.

    Minimise (x1 - 1)^2 + (x2 - 3)^2 subject to x1 + x2 = 2: the solution is
    the projection of (1, 3) onto the line, (0, 2).
    """
    x = eb_fit.constrained_lsq(np.eye(2), np.array([1.0, 3.0]),
                               np.array([1.0, 1.0]), 2.0)
    assert np.allclose(x, [0.0, 2.0], rtol=0, atol=1e-14)


def test_the_json_beside_the_script_is_current():
    """`eb_fit.json` was written by this script, and its selected fit is `fit(n)`.

    The prepare scripts of Exp09 and Exp12 read this file. The SHA-256 of
    the script in the file must equal the script on disk, and the selected
    fit must be reproduced by `fit` to 1e-12 (the grid fit on the grid end
    of the file, or the free-time fit when it was selected).
    """
    here = os.path.dirname(os.path.abspath(eb_fit.__file__))
    with open(os.path.join(here, "eb_fit.json")) as handle:
        record = json.load(handle)
    assert record["script_sha256"] == eb_fit.sha256(eb_fit.__file__)
    selected = record["selected"]
    if selected.get("free_times"):
        again = eb_fit.fit(tau=selected["tau_bar"])
    else:
        again = eb_fit.fit(selected["n"], tau_max=selected["tau_bar_max"])
        assert np.allclose(again["tau_bar"], selected["tau_bar"], rtol=1e-12,
                           atol=0)
    assert np.allclose(again["mu_bar"], selected["mu_bar"], rtol=1e-12,
                       atol=0)


def test_relaxation_of_the_maxwell_limit():
    """With Delta = 0 the EB relaxation modulus is exp(-t / tau_M)."""
    t = np.array([0.01, 0.3, 1.0, 3.0])
    g = eb_fit.relaxation_eb(t, delta=0.0)
    assert np.allclose(g, np.exp(-t), rtol=1e-10, atol=1e-14)
