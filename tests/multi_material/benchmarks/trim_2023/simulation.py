"""Thermochemical benchmark.
Trim, S. J., Butler, S. L., McAdam, S. S., & Spiteri, R. J. (2023).
Manufacturing an exact solution for 2D thermochemical mantle convection models.
Geochemistry, Geophysics, Geosystems, 24(4), e2022GC010807.
"""

import firedrake as fd
import flib
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from gadopt import material_entrainment, node_coordinates

from .materials import bottom_material, top_material


def C0(mesh_coord_y):
    return 1 / (1 + fd.exp(-2 * k * (interface_coord_y - mesh_coord_y)))


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
    # flib can be obtained from
    # https://github.com/seantrim/exact-thermochem-solution
    analytical_values = []
    for x, y in node_coordinates(int_heat_rate).dat.data:
        analytical_values.append(
            flib.h_python(
                x, y, float(simu_time), domain_dims[0], k, interface_coord_y, Ra, Ra * B
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
        material_entrainment(
            diag_vars["level_set"][0],
            material_size=material_area,
            entrainment_height=entrainment_height,
            side=0,
            direction="above",
        )
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
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
            f"{output_path}/rms_velocity_and_entrainment_{tag}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (1, 1)
mesh_gen = "firedrake"
mesh_elements = (128, 128)

# Degree of the function space on which the level-set function is defined.
level_set_func_space_deg = 2

# Parameters to initialise level set
callable_args = (
    curve_parameter := np.array([0.0, domain_dims[0]]),
    interface_slope := 0,
    interface_coord_y := 0.5,
)
boundary_coordinates = [domain_dims, (0.0, domain_dims[1]), (0.0, interface_coord_y)]
signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": "line",
    "interface_args": callable_args,
    "boundary_coordinates": boundary_coordinates,
}
# The following list must be ordered such that, unpacking from the end, each dictionary
# contains the keyword arguments required to initialise the signed-distance array
# corresponding to the interface between a given material and the remainder of the
# numerical domain (all previous materials excluded). By convention, the material thus
# isolated occupies the positive side of the signed-distance array.
signed_distance_kwargs_list = [signed_distance_kwargs]

# Material ordering must follow the logic implemented in the above list. In other words,
# the last material in the below list must correspond to the portion of the numerical
# domain isolated by the signed-distance array calculated using the last dictionary in
# the above list. The first material in the below list will, therefore, occupy the
# negative side of the signed-distance array calculated from the first dictionary above.
materials = [bottom_material, top_material]

# Approximation parameters
dimensional = False
Ra, g = 1e5, 1
B = 0.5

# Parameters to initialise temperature
a = 100
b = 100
k = 35

# Boundary conditions
temp_bc_bot = B * (C0(0) - 1) + 1
temp_bc_top = B * C0(1)
temp_bcs = {3: {"T": temp_bc_bot}, 4: {"T": temp_bc_top}}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Stokes solver options
stokes_nullspace_args = {}
stokes_solver_params = None

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
material_area = domain_dims[0] * interface_coord_y
entrainment_height = 0.5
