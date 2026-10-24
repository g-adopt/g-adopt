"""Profile geometry and rotating plots, independent of Firedrake."""

import json

import matplotlib
import numpy as np
import pytest
from spada_profile_geometry import profile_geometry, profile_settings

matplotlib.use("Agg")


def test_default_meridians_preserve_legacy_sampling():
    geometry = profile_geometry("cap", (0., 0.))
    assert geometry["profile_kind"] == "meridian"
    np.testing.assert_array_equal(geometry["colatitude_deg"], np.linspace(0, 180, 1801))
    np.testing.assert_array_equal(geometry["longitude_deg"], [0, 90, 180])
    assert geometry["unit_vectors"].shape == (3, 1801, 3)
    np.testing.assert_allclose(geometry["tangent_vectors"], geometry["south_vectors"])


def test_rotating_profiles_cross_load_with_perpendicular_tangents():
    geometry = profile_geometry("polar-motion", (25., 75.))
    coordinate = geometry["coordinate_deg"]
    zero = np.flatnonzero(coordinate == 0).item()
    tc, pc = np.deg2rad([25., 75.])
    axis = [np.sin(tc)*np.cos(pc), np.sin(tc)*np.sin(pc), np.cos(tc)]
    np.testing.assert_allclose(geometry["unit_vectors"][:, zero], [axis, axis])
    np.testing.assert_allclose(geometry["point_colatitude_deg"][:, zero], 25.)
    np.testing.assert_allclose(geometry["point_longitude_deg"][:, zero], 75.)
    north, east = geometry["tangent_vectors"][:, zero]
    np.testing.assert_allclose(north @ east, 0, atol=1e-15)
    assert north[2] > 0
    np.testing.assert_allclose(east, [-np.sin(pc), np.cos(pc), 0], atol=1e-15)
    np.testing.assert_allclose(geometry["right_vectors"][0, zero], east, atol=1e-15)
    for basis in ("unit_vectors", "tangent_vectors", "right_vectors"):
        np.testing.assert_allclose(np.linalg.norm(geometry[basis], axis=-1), 1)
    np.testing.assert_allclose(np.sum(geometry["unit_vectors"]*geometry["tangent_vectors"], axis=-1),
                               0, atol=1e-15)
    # Signed distance is the great-circle angular distance from the centre.
    np.testing.assert_allclose(geometry["unit_vectors"] @ np.array(axis),
                               np.broadcast_to(np.cos(np.deg2rad(coordinate)), (2, coordinate.size)),
                               atol=1e-15)


def test_off_axis_controls_use_identical_profile_geometry():
    off = profile_geometry("cap", (25., 75.))
    on = profile_geometry("polar-motion", (25., 75.))
    assert off["profile_kind"] == on["profile_kind"] == "load-centred"
    for key in off:
        np.testing.assert_array_equal(off[key], on[key])


def test_custom_locations_and_grid():
    geometry = profile_geometry("polar-motion", (25, 75), centre=(60, -20),
                                bearings=[45, 135, 225], extent_deg=25, spacing_deg=.3)
    assert geometry["coordinate_deg"][0] == -25
    assert geometry["coordinate_deg"][-1] == 25
    assert np.max(np.diff(geometry["coordinate_deg"])) <= .3 + 1e-12
    np.testing.assert_array_equal(geometry["centre_deg"], [60, 340])
    assert profile_settings(geometry, 1e-6)["depth_fraction"] == 1e-6
    meridians = profile_geometry("polar-motion", (25, 75), longitudes=[75, 165],
                                 extent_deg=60, spacing_deg=1)
    assert meridians["unit_vectors"].shape == (2, 61, 3)


@pytest.mark.parametrize("kwargs", [dict(extent_deg=0), dict(extent_deg=181),
    dict(spacing_deg=0), dict(spacing_deg=float("nan")), dict(bearings=[float("inf")]),
    dict(centre=(190, 75)), dict(longitudes=[75], bearings=[0]),
    dict(longitudes=[75], centre=(25, 75))])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        profile_geometry("polar-motion", (25, 75), **kwargs)


def test_rotating_plot_needs_no_taboo_reference(tmp_path, monkeypatch):
    from plot_spada_profiles import main

    geometry = profile_geometry("polar-motion", (25, 75), extent_deg=25, spacing_deg=5)
    shape = geometry["unit_vectors"].shape[:2]
    summary = {"case": "polar-motion", "load_shape": "disc", "smoke": True,
               "epochs": [{"t_kyr": 0}, {"t_kyr": .1}]}
    path = tmp_path / "summary_polar-motion.json"
    path.write_text(json.dumps(summary))
    for epoch in summary["epochs"]:
        np.savez(tmp_path / f"profiles_polar-motion_{epoch['t_kyr']:g}kyr.npz",
                 **geometry, t_kyr=epoch["t_kyr"], found=np.ones(shape, dtype=bool),
                 U_m=np.zeros(shape), along_m=np.ones(shape), across_m=np.zeros(shape),
                 N_m=np.ones(shape))
    output = tmp_path / "profiles.png"
    monkeypatch.setattr("sys.argv", ["plot_spada_profiles.py", str(path),
                                     "--reference", str(tmp_path / "absent.npz"),
                                     "--max-distance", "20", "--output", str(output)])
    main()
    assert output.stat().st_size > 0


def test_reference_rates_are_time_derivatives():
    from pathlib import Path

    from plot_spada_profiles import taboo_coefficients

    with np.load(Path(__file__).with_name("reference.npz")) as reference:
        for shape in ("cap", "disc"):
            a = taboo_coefficients(reference, 1. - 1e-5, shape)
            b = taboo_coefficients(reference, 1. + 1e-5, shape)
            rates = taboo_coefficients(reference, 1., shape, rate=True)
            for key in rates:
                np.testing.assert_allclose(rates[key], (b[key] - a[key]) / .02,
                                           rtol=2e-7, atol=2e-11)


def test_off_axis_reference_and_rate_plot(tmp_path, monkeypatch):
    from pathlib import Path

    from plot_spada_profiles import main, reference_profiles

    geometry = profile_geometry("cap", (25., 75.), extent_deg=20, spacing_deg=2)
    summary = {"case": "cap", "load_shape": "disc", "load_centre_deg": [25., 75.],
               "load_representation": "spatial", "smoke": True,
               "epochs": [{"t_kyr": 0}, {"t_kyr": .1}]}
    shape = geometry["unit_vectors"].shape[:2]
    path = tmp_path / "summary_cap.json"
    path.write_text(json.dumps(summary))
    with np.load(Path(__file__).with_name("reference.npz")) as reference:
        for epoch in summary["epochs"]:
            t = epoch["t_kyr"]
            fields = reference_profiles(reference, t, geometry, summary, rate=True)
            np.testing.assert_allclose(fields["across"], 0., atol=1e-14)
            np.testing.assert_allclose(fields["along"], -fields["along"][:, ::-1], atol=1e-12)
            np.savez(tmp_path / f"profiles_cap_{t:g}kyr.npz", **geometry, t_kyr=t,
                     found=np.ones(shape, dtype=bool), rate_valid=t > 0,
                     rate_dt_yr=10., **{k+"_m_per_yr": v for k, v in fields.items()})
    output = tmp_path / "rates.pdf"
    monkeypatch.setattr("sys.argv", ["plot_spada_profiles.py", str(path), "--rates",
                                     "--max-distance", "20", "--output", str(output)])
    main()
    assert output.stat().st_size > 0
