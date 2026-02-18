"""Thermal benchmark.
Davies, D. R., Kramer, S. C., Ghelichkhan, S., & Gibson, A. (2022).
Automating Finite Element Methods for Geodynamics via Firedrake.
Geoscientific Model Development Discussions, 2022, 1-50.
"""

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .materials import material


def initialise_temperature(temperature):
    mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())
    radial_coord = fd.sqrt(mesh_coords[0] ** 2 + mesh_coords[1] ** 2)

    temperature.interpolate(
        r_max
        - radial_coord
        + A
        * fd.cos(4.0 * fd.atan2(mesh_coords[1], mesh_coords[0]))
        * fd.sin((radial_coord - r_min) * fd.pi)
    )


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    diag_fields["output_time"].append(simu_time)
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["nusselt_bottom"].append(geo_diag.Nu_bottom() * Nu_bottom_scaling)
    diag_fields["nusselt_top"].append(geo_diag.Nu_top() * Nu_top_scaling)
    diag_fields["energy_conservation"].append(
        abs(
            abs(diag_fields["nusselt_top"][-1]) - abs(diag_fields["nusselt_bottom"][-1])
        )
    )
    diag_fields["avg_temperature"].append(geo_diag.T_avg())
    diag_fields["min_temperature"].append(geo_diag.T_min())
    diag_fields["max_temperature"].append(geo_diag.T_max())

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        for axis, (diag_name, diag_label) in zip(ax.flatten(), diag_names.items()):
            axis.grid()
            axis.set_xlabel("Time (non-dimensional)")
            axis.set_ylabel(diag_label)

            axis.plot(diag_fields["output_time"], diag_fields[diag_name])

        fig.savefig(
            f"{output_path}/diagnostics_{tag}.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


# Simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
r_min, r_max = 1.22, 2.22
n_cells, n_layers = 128, 32
mesh_gen = "extruded_annulus"

# Parameters to initialise level set
signed_distance_kwargs = {
    "interface_geometry": "circle",
    "interface_coordinates": ((0.0, 0.0), r_min + (r_max - r_min) / 3.0),
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
materials = [material, material]

# Approximation parameters
dimensional = False
Ra = 1e5

# Parameters to initialise temperature
A = 0.02

# Boundary conditions with mapping {"bottom": r_min, "top": r_max}
temp_bcs = {"bottom": {"T": 1.0}, "top": {"T": 0.0}}
stokes_bcs = {"bottom": {"un": 0.0}, "top": {"un": 0.0}}

# Timestepping objects
initial_timestep = 1e-7
dump_period = 5e-4
checkpoint_period = 5
steady_state_threshold = 1e-4

# Diagnostic objects
r_ratio = r_min / r_max
Nu_top_scaling = fd.ln(r_ratio) / (r_ratio - 1.0)
Nu_bottom_scaling = r_ratio * fd.ln(r_ratio) / (r_ratio - 1.0)

diag_names = {
    "rms_velocity": "Root-mean-square velocity (non-dimensional)",
    "nusselt_bottom": "Bottom Nusselt number (non-dimensional)",
    "nusselt_top": "Top Nusselt number (non-dimensional)",
    "energy_conservation": "Difference between top and bottom Nusselt numbers",
    "avg_temperature": "Average temperature (non-dimensional)",
    "min_temperature": "Minimum temperature (non-dimensional)",
    "max_temperature": "Maximum temperature (non-dimensional)",
}
diag_fields = {field: [] for field in ("output_time", *diag_names.keys())}
