"""Prevent misleading variant labels and preserve the off-axis vector basis."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from plot_spada_baselines import read_profile, validate_comparison, validate_sampling
from plot_spada_profiles import reference_profiles
from spada_profile_geometry import profile_geometry
from spada_study import BASELINE


@pytest.fixture
def baseline():
    return {**BASELINE, "case": "cap", "rotation": False, "smoke": False,
            "load_shape": "disc", "load_centre_deg": [0., 0.],
            "load_representation": "spatial", "load_nmax": None, "diagnostic_nmax": 10,
            "epochs": [{"t_kyr": 5.}]}


def test_advertised_variant_requires_the_saved_setting_to_change(baseline):
    validate_comparison(baseline, {**baseline, "timestep_scale": 2.}, "dt2")
    with pytest.raises(ValueError, match="timestep_scale"):
        validate_comparison(baseline, baseline, "dt2")


@pytest.mark.parametrize("unexpected", [
    {"rotation": True}, {"bulk_shear_ratio": 100.}, {"load_centre_deg": [25., 75.]},
])
def test_timestep_panel_refuses_other_physical_or_numerical_changes(baseline, unexpected):
    with pytest.raises(ValueError, match="expected"):
        validate_comparison(baseline, {**baseline, "timestep_scale": 2., **unexpected}, "dt2")


def test_offaxis_control_changes_location_without_enabling_rotation(baseline):
    changed = {**baseline, "load_centre_deg": [25., 75.]}
    validate_comparison(baseline, changed, "offaxis")
    with pytest.raises(ValueError, match="rotation"):
        validate_comparison(baseline, {**changed, "rotation": True}, "offaxis")


def saved_profile(geometry):
    size = geometry["unit_vectors"].shape[:2]
    return {**geometry, "t_kyr": np.array(5.), "depth_fraction": np.array(1e-6),
            "found": np.ones(size, dtype=bool),
            **{key: np.ones(size) for key in ("U_m", "along_m", "N_m")}}


def test_offaxis_positive_branch_preserves_outward_tangential_reference(tmp_path, baseline):
    pole = saved_profile(profile_geometry("cap", [0., 0.], longitudes=[0.],
                                          extent_deg=25, spacing_deg=1))
    centre = [25., 75.]
    offaxis = saved_profile(profile_geometry("cap", centre, bearings=[0.],
                                             extent_deg=25, spacing_deg=1))
    summary = {**baseline, "load_centre_deg": centre}
    np.savez(tmp_path / "profiles_cap_5kyr.npz", **offaxis)
    selected, index, keep = read_profile(tmp_path, summary, 5., 0., 25., {}, bearing=0.)
    np.testing.assert_allclose(selected["coordinate_deg"][keep], pole["coordinate_deg"])
    validate_sampling(pole, selected, 0, index, np.ones(26, dtype=bool), keep, offaxis=True)
    with np.load(Path(__file__).with_name("reference.npz"), allow_pickle=False) as reference:
        a = reference_profiles(reference, 5., pole, baseline)
        b = reference_profiles(reference, 5., selected, summary)
    assert a["along"][0, 5] > 0
    for quantity in ("U", "along", "N"):
        np.testing.assert_allclose(b[quantity][index, keep], a[quantity][0], atol=1e-10, rtol=1e-11)


def test_same_location_variants_cannot_silently_compare_different_bases_or_depths():
    a = saved_profile(profile_geometry("cap", [0., 0.], longitudes=[0.],
                                       extent_deg=25, spacing_deg=1))
    keep = np.ones(26, dtype=bool)
    b = deepcopy(a)
    b["depth_fraction"] = np.array(1e-5)
    with pytest.raises(ValueError, match="depth_fraction"):
        validate_sampling(a, b, 0, 0, keep, keep)
    b = deepcopy(a)
    b["tangent_vectors"] *= -1
    with pytest.raises(ValueError, match="tangent_vectors"):
        validate_sampling(a, b, 0, 0, keep, keep)
