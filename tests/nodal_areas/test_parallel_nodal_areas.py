from pathlib import Path

import numpy as np


def test_parallel_spherical_layer_areas():
    radii, layer_sums, pointwise_errors, comm_sizes = np.loadtxt(
        Path(__file__).with_name("layer_sums.dat"), unpack=True
    )
    expected = 4 * np.pi * radii**2

    assert np.all(np.diff(radii) < 0)
    np.testing.assert_allclose(layer_sums, expected, rtol=2.0e-3)
    np.testing.assert_allclose(
        layer_sums / layer_sums[0], expected / expected[0], rtol=1.0e-12
    )
    np.testing.assert_allclose(pointwise_errors, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(comm_sizes, 2)
