"""Tests for finite-element stress-glut assembly."""

import firedrake as fd
import numpy as np

from gadopt.earthquake_sources import build_stress_glut, normalised_gaussian_delta


def test_stress_glut_conserves_moment_and_keeps_patch_locations():
    """Accumulate overlapping patches on a mesh with boundary truncation."""
    mesh = fd.UnitCubeMesh(3, 3, 3)
    centres = [(0.1, 0.4, 0.6), (0.6, 0.4, 0.6), (0.4, 0.4, 0.6)]
    tensors = np.array([
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    width = 0.2

    glut, integrals = build_stress_glut(mesh, centres, tensors, width)
    integrated = np.array([[fd.assemble(glut[i, j] * fd.dx) for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(integrated, tensors.sum(axis=0), atol=1e-12, rtol=0)

    deltas, expected_integrals = zip(*[normalised_gaussian_delta(mesh, centre, width) for centre in centres])
    assert integrals[0] < integrals[1]  # The near-edge Gaussian is more truncated.
    np.testing.assert_allclose(integrals, expected_integrals, atol=1e-12, rtol=0)
    values = glut.dat.data_ro
    np.testing.assert_allclose(values[:, 0, 0], deltas[0].dat.data_ro, atol=1e-12, rtol=0)
    np.testing.assert_allclose(values[:, 0, 1], 2 * deltas[1].dat.data_ro - 0.5 * deltas[2].dat.data_ro,
                               atol=1e-12, rtol=0)
