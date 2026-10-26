"""Scientific comparison rules and output auditing, without Firedrake."""

import json
from pathlib import Path

import numpy as np
import pytest

import analyse_spada_study as analysis
from plot_spada_profiles import reference_profiles
from spada_profile_geometry import profile_geometry
from spada_study import BASELINE, EPOCHS, run_matrix


def profile(t):
    geometry = profile_geometry("cap", (0, 0), longitudes=[0], extent_deg=20, spacing_deg=10)
    shape = geometry["unit_vectors"].shape[:2]
    result = {**geometry, "depth_fraction": np.array(1e-6), "t_kyr": np.array(t),
              "found": np.ones(shape, dtype=bool), "rate_valid": np.array(t > 0),
              "rate_interval_start_kyr": np.array(max(0, t-.1)),
              "rate_interval_end_kyr": np.array(t), "rate_dt_yr": np.array(100. if t else 0.)}
    for q in ("U", "V", "W", "along", "across", "N"):
        result[q+"_m"] = np.ones(shape)
        result[q+"_m_per_yr"] = np.full(shape, .001 if t else np.nan)
    return result


def test_relative_norm_uses_profile_peak_and_handles_zero_crossings():
    ref = np.array([[0.], [100.]])
    score = analysis.norms(np.array([[1.], [100.]]), ref)
    assert score["max_abs"] == 1
    assert score["max_percent"] == 1
    zero = analysis.norms(np.ones((3, 1)), np.zeros((3, 1)))
    assert zero["max_percent"] is None
    assert zero["rms_percent"] is None


def test_vector_difference_is_euclidean_and_invariant_under_basis_rotation():
    ref = np.array([[3., 4.], [4., -3.]])
    model = ref + np.array([[3., 4.], [-3., -4.]])
    original = analysis.norms(model, ref)
    rotation = np.array([[0., -1.], [1., 0.]])
    assert analysis.norms(model @ rotation, ref @ rotation) == original
    assert original["max_abs"] == 5
    assert original["max_percent"] == 100


def test_low_mode_diagnostic_recovers_dipole_without_changing_input():
    geometry = profile_geometry("cap", (0, 0), spacing_deg=5)
    directions = geometry["unit_vectors"]
    coefficients = np.array([.2, -.3, .4])
    values = .1 + directions @ coefficients
    original = values.copy()
    result = analysis.low_mode_fit(values, directions)
    np.testing.assert_allclose([result[k] for k in ("constant", "dipole_x", "dipole_y", "dipole_z")],
                               [.1, *coefficients], atol=1e-14)
    assert result["fraction_squared_norm_explained"] == pytest.approx(1)
    np.testing.assert_array_equal(values, original)


def test_zero_low_mode_difference_has_no_fractional_score():
    directions = profile_geometry("cap", (0, 0), spacing_deg=10)["unit_vectors"]
    result = analysis.low_mode_fit(np.zeros(directions.shape[:2]), directions)
    assert result["fraction_squared_norm_explained"] is None


def test_phase_difference_wraps_across_180_degrees():
    assert analysis.wrapped_degrees(179 - (-179)) == -2


def test_profile_matching_refuses_different_locations():
    left, right = profile(.1), profile(.1)
    right["unit_vectors"] = right["unit_vectors"].copy()
    right["unit_vectors"][0, 1, 0] += .01
    with pytest.raises(ValueError, match="unit_vectors"):
        analysis.check_geometry(left, right)


def test_physical_match_refuses_rotation_and_load_location_changes():
    left = {"case": "cap", "load_centre_deg": [0, 0]}
    with pytest.raises(ValueError, match="load_centre_deg"):
        analysis.same_physics(left, {**left, "load_centre_deg": [25, 75]})


def test_reference_rate_is_interval_average_of_instantaneous_reference():
    data = profile(1.)
    data.update(rate_interval_start_kyr=np.array(.5), rate_interval_end_kyr=np.array(1.),
                rate_dt_yr=np.array(500.))
    summary = {"load_centre_deg": [0, 0], "load_shape": "cap", "load_representation": "spatial"}
    nodes, weights = np.polynomial.legendre.leggauss(12)
    with np.load(Path(__file__).with_name("reference.npz")) as ref:
        averaged = analysis.reference_fields(ref, 1., data, summary, rates=True)
        instantaneous = [reference_profiles(ref, .75+.25*x, data, summary, rate=True) for x in nodes]
    for q in ("U", "along", "across", "N"):
        quadrature = sum(w*row[q] for w, row in zip(weights/2, instantaneous))
        np.testing.assert_allclose(averaged[q], quadrature, atol=2e-14, rtol=1e-9)


def test_saved_controls_not_directory_name_determine_validity():
    expected = next(r for r in run_matrix() if r["id"] == "cap-off/mesh500")
    run = analysis.Run(Path("ignored"), expected["id"], {
                       **BASELINE, "case": "cap", "rotation": False,
                       "load_shape": "cap", "load_centre_deg": [0, 0],
                       "load_representation": "spatial", "diagnostic_nmax": 10})
    analysis.validate_controls(run, expected)
    assert any("mesh_h_km" in error for error in run.errors)


@pytest.fixture
def saved_run(tmp_path):
    directory = tmp_path / "cap-off/baseline"
    directory.mkdir(parents=True)
    summary = {**BASELINE, "case": "cap", "rotation": False, "load_shape": "cap",
               "load_centre_deg": [0, 0], "load_representation": "spatial",
               "load_nmax": 10, "diagnostic_nmax": 10,
               "mesh": {"flat_cells": 0, "folded_cells_parent": 0, "folded_cells_mantle": 0},
               "epochs": [{"t_kyr": t} for t in EPOCHS]}
    path = directory / "summary_cap.json"
    path.write_text(json.dumps(summary))
    for t in EPOCHS:
        np.savez(directory / f"profiles_cap_{t:g}kyr.npz", **profile(t))
    (directory / "params_cap.log").write_text("step t_kyr\n0 0\n1 20\n")
    (directory / "pbs.stdout").write_text(
        "Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_ solve did not converge due to DIVERGED_ITS iterations 200\n"
        "Linear SelfGravitatingGIA_ solve converged due to CONVERGED_RTOL iterations 6\n"
        "=== finished test, exit 0 ===\n")
    return tmp_path, path


def test_inner_iteration_caps_are_separate_from_coupled_failure(saved_run):
    root, path = saved_run
    run = analysis.read_run(path, root, {})
    assert not run.errors
    assert run.solver["inner_caps"] == 1
    assert run.solver["outer_max_iterations"] == 6


def test_audit_detects_outer_failure_even_with_successful_process_exit(saved_run):
    root, path = saved_run
    log = path.parent / "pbs.stdout"
    log.write_text(log.read_text().replace("CONVERGED_RTOL", "DIVERGED_ITS"))
    run = analysis.read_run(path, root, {})
    assert "Outer solver failure" in run.errors


def test_audit_accepts_undefined_initial_rates_but_rejects_later_nan(saved_run):
    root, path = saved_run
    data = profile(20)
    data["N_m_per_yr"][0, 1] = np.nan
    np.savez(path.parent / "profiles_cap_20kyr.npz", **data)
    run = analysis.read_run(path, root, {})
    assert run.errors == ["Invalid N rate at 20.0 kyr"]


@pytest.mark.parametrize("aligned", [False, True])
def test_mesh_order_requires_decreasing_and_aligned_differences(aligned):
    runs = {}
    for name, h, offset in (("mesh500", 500., 4.), ("baseline", 250., 1.), ("mesh125", 125., .25)):
        summary = {**BASELINE, "mesh_h_km": h, "case": "cap", "rotation": False,
                   "load_shape": "cap", "load_centre_deg": [0, 0]}
        data = {t: profile(t) for t in EPOCHS}
        for d in data.values():
            d["N_m"][:] = offset
        if not aligned and name == "mesh500":
            for d in data.values():
                d["N_m"][0] = [4, -2, 1]
        identity = "cap-off/"+name
        runs[identity] = analysis.Run(Path(identity), identity, summary, profiles=data)
    rows = [r for r in analysis.mesh_rows(runs) if r["quantity"] == "N" and r["kind"] == "value"]
    if aligned:
        assert all(r["apparent_order_if_aligned"] == pytest.approx(2) for r in rows)
    else:
        assert all(r["apparent_order_if_aligned"] is None for r in rows)
