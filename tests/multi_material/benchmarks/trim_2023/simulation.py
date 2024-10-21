"""Thermochemical benchmark.
Trim, S. J., Butler, S. L., McAdam, S. S., & Spiteri, R. J. (2023).
Manufacturing an exact solution for 2D thermochemical mantle convection models.
Geochemistry, Geophysics, Geosystems, 24(4), e2022GC010807.
"""

from functools import partial

import firedrake as fd
import flib
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga

from .materials import bottom_material, top_material


def C0(mesh_coord_y):
    return 1 / (1 + fd.exp(-2 * k * (intercept - mesh_coord_y)))


def f(t):
    return a * fd.sin(fd.pi * b * t)


def initialise_temperature(temperature):
    λ = domain_dims[0]
    x, y = fd.SpatialCoordinate(temperature.function_space().mesh())

    temperature.interpolate(
        (
            -(fd.pi**3)
            * (λ**2 + 1) ** 2
            / λ**3
            * fd.cos(fd.pi * x / λ)
            * fd.sin(fd.pi * y)
            * f(0)
            + Ra * B * C0(y)
            + (Ra - Ra * B) * (1 - y)
        )
        / Ra
    )


def internal_heating_rate(int_heat_rate, simu_time):
    # flib can be obtained from https://github.com/seantrim/exact-thermochem-solution
    analytical_values = []
    for x, y in ga.node_coordinates(int_heat_rate):
        analytical_values.append(
            flib.h_python(
                x, y, float(simu_time), domain_dims[0], k, intercept, Ra, Ra * B
            )
        )

    int_heat_rate.dat.data[:] = analytical_values


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    λ = domain_dims[0]
    rms_velocity_analytical = fd.pi * fd.sqrt(λ**2 + 1) / 2 / λ * abs(f(simu_time))

    diag_fields["output_time"].append(simu_time)
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["rms_velocity_analytical"].append(rms_velocity_analytical)
    diag_fields["entrainment"].append(
        ga.entrainment(diag_vars["level_set"][0], material_area, entrainment_height)
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

        ax[0].set_ylim(0, 250)
        ax[1].set_ylim(0, 0.8)

        ax[0].set_xlabel("Time (non-dimensional)")
        ax[1].set_xlabel("Time (non-dimensional)")
        ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
        ax[1].set_ylabel("Entrainment (non-dimensional)")

        ax[0].plot(
            diag_fields["output_time"],
            diag_fields["rms_velocity"],
            label="Simulation",
        )
        ax[0].plot(
            diag_fields["output_time"],
            diag_fields["rms_velocity_analytical"],
            label="Analytical",
        )
        ax[1].plot(diag_fields["output_time"], diag_fields["entrainment"])

        ax[0].legend()

        fig.savefig(
            f"{output_path}/rms_velocity_and_entrainment.pdf",
            dpi=300,
            bbox_inches="tight",
        )


# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (1, 1)
mesh_gen = "firedrake"
mesh_elements = (128, 128)

# Parameters to initialise level sets
slope = 0
intercept = 0.5
# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [
    partial(isd.isd_simple_curve, domain_dims[0], isd.straight_line)
]
isd_params = [(slope, intercept)]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [bottom_material, top_material]

# Approximation parameters
dimensional = False
Ra = 1e5
B = 0.5

# Temperature parameters
a = 100
b = 100
k = 35

# Boundary conditions
temp_bc_bot = B * (C0(0) - 1) + 1
temp_bc_top = B * C0(1)
temp_bcs = {3: {"T": temp_bc_bot}, 4: {"T": temp_bc_top}}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e-6
dump_period = 1e-4
checkpoint_period = 5
time_end = 0.01

# Diagnostic objects
diag_fields = {
    "output_time": [],
    "rms_velocity": [],
    "rms_velocity_analytical": [],
    "entrainment": [],
}
material_area = domain_dims[0] * intercept
entrainment_height = 0.5
