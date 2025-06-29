"""Thermochemical benchmark.
Van Keken, P. E., King, S. D., Schmeling, H., Christensen, U. R., Neumeister, D., &
Doin, M. P. (1997).
A comparison of methods for the modeling of thermochemical convection.
Journal of Geophysical Research: Solid Earth, 102(B10), 22477-22495.
"""

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from gadopt import material_entrainment

from .materials import dense_material, reference_material


def initialise_temperature(temperature):
    mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

    u0 = (
        domain_dims[0] ** (7 / 3)
        / (1 + domain_dims[0] ** 4) ** (2 / 3)
        * (Ra / 2 / fd.sqrt(fd.pi)) ** (2 / 3)
    )
    v0 = u0
    Q = 2 * fd.sqrt(domain_dims[0] / fd.pi / u0)
    Tu = fd.erf((1 - mesh_coords[1]) / 2 * fd.sqrt(u0 / mesh_coords[0])) / 2
    Tl = 1 - 1 / 2 * fd.erf(
        mesh_coords[1] / 2 * fd.sqrt(u0 / (domain_dims[0] - mesh_coords[0]))
    )
    Tr = 1 / 2 + Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(v0 / (mesh_coords[1] + 1)) * fd.exp(
        -(mesh_coords[0] ** 2) * v0 / (4 * mesh_coords[1] + 4)
    )
    Ts = 1 / 2 - Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(v0 / (2 - mesh_coords[1])) * fd.exp(
        -((domain_dims[0] - mesh_coords[0]) ** 2) * v0 / (8 - 4 * mesh_coords[1])
    )

    temperature.interpolate(fd.max_value(fd.min_value(Tu + Tl + Tr + Ts - 3 / 2, 1), 0))
    fd.DirichletBC(temperature.function_space(), 1, 3).apply(temperature)
    fd.DirichletBC(temperature.function_space(), 0, 4).apply(temperature)


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    diag_fields["output_time"].append(simu_time)
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["entrainment"].append(
        material_entrainment(
            diag_vars["level_set"][0],
            material_size=diag_params["domain_dim_x"]
            * diag_params["interface_coord_y"],
            entrainment_height=diag_params["entrainment_height"],
            side=0,
            direction="above",
            skip_material_size_check=True,
        )
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        rms_vel_schmeling = np.loadtxt("data/HSvrms4800.dat")
        entr_schmeling = np.loadtxt("data/HSentr4800.dat")
        rms_vel_van_keken = np.loadtxt("data/pvk120_003.vrms.dat")
        entr_van_keken = np.loadtxt("data/pvk120_003.entr.dat")
        rms_vel_christensen = np.loadtxt("data/URC125x40_240k.vrms.dat")
        entr_christensen = np.loadtxt("data/URC125x40_240k.entr.dat")
        rms_vel_christensen_chain = np.loadtxt("data/URCchain.vrms.dat")
        entr_christensen_chain = np.loadtxt("data/URCchain.entr.dat")

        fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

        ax[0].set_xlabel("Time (non-dimensional)")
        ax[1].set_xlabel("Time (non-dimensional)")
        ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
        ax[1].set_ylabel("Entrainment (non-dimensional)")

        ax[0].plot(
            rms_vel_schmeling[:, 0],
            rms_vel_schmeling[:, 1],
            linestyle="dotted",
            label="HS (van Keken et al., 1997)",
        )
        ax[1].plot(
            entr_schmeling[:, 0],
            entr_schmeling[:, 1],
            linestyle="dotted",
            label="HS (van Keken et al., 1997)",
        )
        ax[0].plot(
            rms_vel_van_keken[:, 0],
            rms_vel_van_keken[:, 1],
            linestyle="dotted",
            label="PvK (van Keken et al., 1997)",
        )
        ax[1].plot(
            entr_van_keken[:, 0],
            entr_van_keken[:, 1],
            linestyle="dotted",
            label="PvK (van Keken et al., 1997)",
        )
        ax[0].plot(
            rms_vel_christensen[:, 0],
            rms_vel_christensen[:, 1],
            linestyle="dotted",
            label="CND (van Keken et al., 1997)",
        )
        ax[1].plot(
            entr_christensen[:, 0],
            entr_christensen[:, 1],
            linestyle="dotted",
            label="CND (van Keken et al., 1997)",
        )
        ax[0].plot(
            rms_vel_christensen_chain[:, 0],
            rms_vel_christensen_chain[:, 1],
            linestyle="dotted",
            label="CND Markerchain (van Keken et al., 1997)",
        )
        ax[1].plot(
            entr_christensen_chain[:, 0],
            entr_christensen_chain[:, 1],
            linestyle="dotted",
            label="CND Markerchain (van Keken et al., 1997)",
        )

        ax[0].plot(
            diag_fields["output_time"],
            diag_fields["rms_velocity"],
            label="Conservative level set",
        )
        ax[1].plot(
            diag_fields["output_time"],
            diag_fields["entrainment"],
            label="Conservative level set",
        )

        ax[0].legend(fontsize=12, fancybox=True, shadow=True)
        ax[1].legend(fontsize=12, fancybox=True, shadow=True)

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
domain_dims = (2, 1)
mesh_gen = "firedrake"
mesh_elements = (128, 64)

# Degree of the function space on which the level-set function is defined.
level_set_func_space_deg = 2

# Parameters to initialise level set
callable_args = (
    curve_parameter := np.array([0.0, domain_dims[0]]),
    interface_slope := 0,
    interface_coord_y := 0.025,
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
materials = [dense_material, reference_material]

# Approximation parameters
dimensional = False
Ra, g = 3e5, 1

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Stokes solver options
stokes_nullspace_args = {}
stokes_solver_params = None

# Timestepping objects
initial_timestep = 1e-6
dump_period = 1e-4
checkpoint_period = 5
time_end = 0.05

# Diagnostic objects
diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
entrainment_height = 0.2
diag_params = {
    "domain_dim_x": domain_dims[0],
    "interface_coord_y": interface_coord_y,
    "entrainment_height": entrainment_height,
}
