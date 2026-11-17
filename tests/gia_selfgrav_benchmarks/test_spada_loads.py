"""Table 4 load and plot-reference checks; no Firedrake required."""

import json
from pathlib import Path

import numpy as np
import pytest
from spada_loads import (
    CAP_RADIUS_DEG,
    CAP_THICKNESS_M,
    DISC_THICKNESS_M,
    RHO_ICE,
    cap_load,
    disc_load,
    load_coefficients,
    polar_motion_scale,
)


@pytest.mark.parametrize("shape", ["cap", "disc"])
def test_coefficients_against_surface_density_integral(shape):
    # sigma_n = (2n+1)/2 int_{cos(alpha)}^1 sigma(x) P_n(x) dx.
    # x = cos(alpha) + (1-cos(alpha))*u^2 removes the cap's square root.
    nodes, weights = np.polynomial.legendre.leggauss(160)
    u, weights = (nodes + 1) / 2, weights / 2
    delta = 1 - np.cos(np.deg2rad(CAP_RADIUS_DEG))
    x = 1 - delta + delta * u**2
    density = (RHO_ICE * CAP_THICKNESS_M * u if shape == "cap"
               else np.full_like(u, RHO_ICE * DISC_THICKNESS_M))
    n = np.arange(257)
    integral = (np.polynomial.legendre.legvander(x, 256).T
                @ (weights * density * 2 * delta * u))
    np.testing.assert_allclose(load_coefficients(256, shape),
                               (2*n + 1)/2 * integral, rtol=2e-9, atol=2e-7)


def test_disc_mass_and_polar_excitation():
    # Table 4 gives equal total mass, but different quadrupoles, for these loads.
    np.testing.assert_allclose(disc_load(0)[0], cap_load(0)[0], rtol=1e-11)
    c = np.cos(np.deg2rad(CAP_RADIUS_DEG))
    expected_sigma2 = 5 * RHO_ICE * DISC_THICKNESS_M * (c - c**3) / 4
    np.testing.assert_allclose(disc_load(2)[2], expected_sigma2, rtol=1e-13)
    assert polar_motion_scale("cap") == 1
    np.testing.assert_allclose(polar_motion_scale("disc"),
                               expected_sigma2 / cap_load(2)[2])
    assert 0.99 < polar_motion_scale("disc") < 1


def test_invalid_shape():
    with pytest.raises(ValueError, match="Unknown load shape"):
        load_coefficients(10, "unknown")


def test_plot_reference_changes_only_the_load():
    from plot_spada_profiles import taboo_coefficients

    with np.load(Path(__file__).with_name("reference.npz")) as reference:
        for t in (0., 1., 20.):
            cap = taboo_coefficients(reference, t)
            disc = taboo_coefficients(reference, t, "disc")
            ratio = disc_load(256)[2:] / cap_load(256)[2:]
            for q in ("U", "V", "N"):
                np.testing.assert_allclose(disc[q][2:], cap[q][2:] * ratio)
                np.testing.assert_array_equal(disc[q][:2], 0)


def test_convergence_does_not_mix_load_shapes(tmp_path):
    from plot_spada_load_convergence import load_runs

    for directory, shape in [("legacy", None), ("disc", "disc")]:
        path = tmp_path / directory
        path.mkdir()
        summary = {"case": "cap", "smoke": False, "nmax": 10,
                   "epochs": [{"t_kyr": 0}]}
        if shape is not None:
            summary["load_shape"] = shape
        (path / "summary_cap.json").write_text(json.dumps(summary))
        (path / "profiles_cap_0kyr.npz").touch()
    for shape, directory in [("cap", "legacy"), ("disc", "disc")]:
        runs = load_runs(tmp_path, shape)
        assert len(runs) == 1
        assert runs[0][3] == tmp_path / directory
