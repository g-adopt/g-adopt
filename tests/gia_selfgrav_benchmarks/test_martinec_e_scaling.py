"""Check integral objectives against physical errors on non-unit surfaces."""
import math

import pytest

from martinec_e import objective_from_rms, objective_rms


@pytest.mark.parametrize("area", [0.25, 4.0, 61.0279801978])
def test_objective_matches_weighted_surface_error(area):
    # Two constant-error patches cover 1/4 and 3/4 of the surface. In
    # control units their errors are 1 and 2, giving integrals A/4 and 3A.
    rms_m = math.sqrt(0.25 * 1000**2 + 0.75 * 2000**2)
    integral_objective = 0.5 * (area / 4 + 3 * area)
    assert objective_from_rms(rms_m, area) == pytest.approx(integral_objective)
    assert objective_rms(integral_objective, area) == pytest.approx(rms_m)
    assert objective_rms(0.0, area) == 0.0


@pytest.mark.parametrize("area", [0.25, 4.0, 61.0279801978])
def test_objective_stopping_threshold_preserves_physical_tolerance(area):
    tolerance_m = 0.01
    # Uniform errors make the integral independent of any RMS conversion.
    goal = objective_from_rms(tolerance_m, area)
    assert objective_rms(goal, area) == pytest.approx(tolerance_m)
    for error_m in (0.009, 0.011):
        objective = 0.5 * area * (error_m / 1000)**2
        assert (objective < goal) == (error_m < tolerance_m)
