"""
Test case for smoothing
"""

from pathlib import Path

import numpy as np
import pytest

from gadopt import *
from gadopt.utility import upward_normal


# Fixture for loading cylindrical field data
@pytest.fixture
def load_field():
    checkpoint_base = (
        Path(__file__).parent.parent / "demos/mantle_convection/adjoint_2d_cylindrical"
    )
    # Start with a previously initialised temperature field
    with CheckpointFile(str(checkpoint_base / "Checkpoint230.h5"), mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = False

        T = f.load_function(mesh, "Temperature")

    temp_bcs = {"bottom": {"g": 1.0}, "top": {"g": 0.0}}

    # Compute layer average for initial stage:
    T_avg = Function(T, name="Temperature (layer average)")
    averager = LayerAveraging(mesh, quad_degree=6)
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

    return T, T_avg, temp_bcs


def test_isotropic_smoothing(load_field):
    # load field and its average for comparison
    T, _, temp_bcs = load_field

    solution = Function(T, name="Temperature (smoothed)")

    kappa = Constant(1)
    wavelength = 0.1
    delta_t = wavelength**2 / 4 / kappa

    smoother = GenericTransportSolver(
        "diffusion",
        solution,
        delta_t,
        BackwardEuler,
        eq_attrs={"diffusivity": kappa},
        bcs=temp_bcs,
    )
    smoother.solve()

    result = assemble((solution - T) ** 2 * dx)
    np.testing.assert_allclose(result, 0.06243541054145882)


def test_anisotropic_smoothing(load_field):
    # load field and its average for comparison
    T, T_avg, temp_bcs = load_field

    solution = Function(T, name="Temperature (smoothed)")

    # Unit radial and tangential vectors
    mesh = T.function_space().mesh()
    e_r = upward_normal(mesh)
    e_t = as_vector((-e_r[1], e_r[0]))
    # Define the radial and tangential diffusivity values
    kappa_r = Constant(0.0)  # Radial diffusivity
    kappa_t = Constant(1.0)  # Tangential diffusivity
    # Construct the anisotropic diffusivity tensor
    kappa = kappa_r * outer(e_r, e_r) + kappa_t * outer(e_t, e_t)

    wavelength = 1e3
    kappa_avg = assemble(sqrt(inner(kappa, kappa)) * dx(mesh)) / assemble(1 * dx(mesh))
    delta_t = wavelength**2 / 4 / kappa_avg

    smoother = GenericTransportSolver(
        "diffusion",
        solution,
        delta_t,
        BackwardEuler,
        eq_attrs={"diffusivity": kappa},
        bcs=temp_bcs,
    )
    smoother.solve()

    np.testing.assert_array_less(assemble((T_avg - solution) ** 2 * dx), 1e-7)
