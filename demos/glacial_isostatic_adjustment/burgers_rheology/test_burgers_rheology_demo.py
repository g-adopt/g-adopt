"""Check the demo against continuous creep solutions, rather than saved baselines."""

from pathlib import Path

import numpy as np


def test_analytical_creep_and_recovery():
    data = np.genfromtxt(Path(__file__).with_name("strains.csv"), delimiter=",", names=True)
    t = data["time"]
    rise, unload_start, unload_end = 0.1, 0.9, 1.0
    amplitude, tau = 1e-5, 0.3
    duration = unload_end - unload_start
    loading_time = np.minimum(t, rise)
    holding_time = np.clip(t - rise, 0, unload_start - rise)
    unloading_time = np.clip(t - unload_start, 0, duration)
    integrated_stress = amplitude * (
        loading_time**2 / (2 * rise) + holding_time
        + unloading_time - unloading_time**2 / (2 * duration))
    maxwell = (data["stress"] + integrated_stress) / 2

    # Response of a Kelvin element to a unit stress ramp starting at t=0.
    def ramp_response(time):
        positive = np.maximum(time, 0)
        return positive + tau * np.expm1(-positive / tau)

    transient = amplitude / 2 * (
        (ramp_response(t) - ramp_response(t - rise)) / rise
        - (ramp_response(t - unload_start) - ramp_response(t - unload_end)) / duration)
    burgers = maxwell + transient
    # Bounds allow the first-order integration error at the demo's dt=0.005.
    np.testing.assert_allclose(data["maxwell"], maxwell, rtol=0, atol=3e-8)
    np.testing.assert_allclose(data["transient"], transient, rtol=0, atol=5e-8)
    np.testing.assert_allclose(data["total"], burgers, rtol=0, atol=8e-8)
    np.testing.assert_allclose(data["permanent"][-1], 4.5e-6, rtol=1e-8)


def test_strain_partition_and_unloading():
    data = np.genfromtxt(Path(__file__).with_name("strains.csv"), delimiter=",", names=True)
    np.testing.assert_allclose(data["total"], data["elastic"] + data["transient"] + data["permanent"], atol=1e-13)
    np.testing.assert_allclose(data["elastic"], data["stress"] / 2, atol=1e-13)
    np.testing.assert_allclose(data["total"] - data["maxwell"], data["transient"], atol=1e-13)
    unloaded = data["time"] >= 1
    np.testing.assert_allclose(data["permanent"][unloaded], data["permanent"][unloaded][0], atol=1e-13)
    np.testing.assert_allclose(data["maxwell"][unloaded], data["maxwell"][unloaded][0], atol=1e-13)
    assert np.all(np.diff(data["transient"][unloaded]) < 0)
    np.testing.assert_allclose(data[0].tolist(), 0, atol=1e-15)
