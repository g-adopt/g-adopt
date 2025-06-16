"""Compositional benchmark.
Schmalholz, S. M. (2011).
A simple analytical solution for slab detachment.
Earth and Planetary Science Letters, 304(1-2), 45-54.
"""

import firedrake as fd
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.io import loadmat

from .materials import lithosphere, mantle


def generate_mesh(mesh_path):
    gmsh.initialize()
    gmsh.model.add("mesh")

    point_1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_vert_res)
    point_2 = gmsh.model.geo.addPoint(0, domain_dims[1], 0, mesh_vert_res)

    line_1 = gmsh.model.geo.addLine(point_1, point_2)

    gmsh.model.geo.extrude(
        [(1, line_1)],
        mesh_fine_layer_min_x,
        0,
        0,
        numElements=[21],
        recombine=True,
    )  # Horizontal resolution: 20 km

    num_layers = int(mesh_fine_layer_width / mesh_fine_layer_hor_res)
    gmsh.model.geo.extrude(
        [(1, line_1 + 1)],
        mesh_fine_layer_width,
        0,
        0,
        numElements=[num_layers],
        recombine=True,
    )

    gmsh.model.geo.extrude(
        [(1, line_1 + 5)],
        domain_dims[0] - mesh_fine_layer_min_x - mesh_fine_layer_width,
        0,
        0,
        numElements=[21],
        recombine=True,
    )  # Horizontal resolution: 20 km

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [line_1], tag=1)
    gmsh.model.addPhysicalGroup(1, [line_1 + 9], tag=2)
    gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [2, 6, 10]], tag=3)
    gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [3, 7, 11]], tag=4)

    gmsh.model.addPhysicalGroup(2, [5, 9, 13], tag=1)

    gmsh.model.mesh.generate(2)

    gmsh.write(f"{mesh_path}/mesh.msh")
    gmsh.finalize()


def initialise_temperature(temperature):
    pass


def steady_state_condition(stokes_solver):
    pass


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    epsilon = diag_vars["epsilon"]
    eps_data = epsilon.dat.data_ro
    level_set = diag_vars["level_set"][0]
    level_set_data = level_set.dat.data_ro
    coords_data = (
        fd.Function(
            fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
        )
        .interpolate(fd.SpatialCoordinate(level_set))
        .dat.data_ro
    )

    mask_ls_outside = (
        (coords_data[:, 0] <= domain_dims[0] / 2)
        & (coords_data[:, 1] < domain_dims[1] - lithosphere_thickness - 2e4)
        & (coords_data[:, 1] > domain_dims[1] - lithosphere_thickness - slab_length)
        & (level_set_data < 0.5)
    )
    mask_ls_inside = (
        (coords_data[:, 0] <= domain_dims[0] / 2)
        & (coords_data[:, 1] < domain_dims[1] - lithosphere_thickness - 2e4)
        & (coords_data[:, 1] > domain_dims[1] - lithosphere_thickness - slab_length)
        & (level_set_data >= 0.5)
    )
    if mask_ls_outside.any():
        ind_outside = coords_data[mask_ls_outside, 0].argmax()
        hor_coord_outside = coords_data[mask_ls_outside, 0][ind_outside]
        if not mask_ls_outside.all():
            ver_coord_outside = coords_data[mask_ls_outside, 1][ind_outside]
            mask_ver_coord = (
                abs(coords_data[mask_ls_inside, 1] - ver_coord_outside) < 1e3
            )
            if mask_ver_coord.any():
                ind_inside = coords_data[mask_ls_inside, 0][mask_ver_coord].argmin()
                hor_coord_inside = coords_data[mask_ls_inside, 0][mask_ver_coord][
                    ind_inside
                ]

                ls_outside = max(1e-2, level_set_data[mask_ls_outside][ind_outside])
                eps_outside = eps_data[mask_ls_outside][ind_outside]
                sdls_outside = eps_outside * np.log(ls_outside / (1 - ls_outside))

                ls_inside = min(
                    1 - 1e-2,
                    level_set_data[mask_ls_inside][mask_ver_coord][ind_inside],
                )
                eps_inside = eps_data[mask_ls_inside][mask_ver_coord][ind_inside]
                sdls_inside = eps_inside * np.log(ls_inside / (1 - ls_inside))

                ls_dist = sdls_inside / (sdls_inside - sdls_outside)
                hor_coord_interface = (
                    ls_dist * hor_coord_outside + (1 - ls_dist) * hor_coord_inside
                )
                min_width = domain_dims[0] - 2 * hor_coord_interface
            else:
                min_width = domain_dims[0] - 2 * hor_coord_outside
        else:
            min_width = domain_dims[0] - 2 * hor_coord_outside
    else:
        min_width = np.inf

    min_width_global = level_set.comm.allreduce(min_width, MPI.MIN)

    diag_fields["normalised_time"].append(simu_time / characteristic_time)
    diag_fields["slab_necking"].append(min_width_global / slab_width)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        slab_necking_schmalholz = loadmat("data/DET_FREE_NEW_TOP_R100.mat")

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.set_xlabel("Normalised time")
        ax.set_ylabel("Slab necking")

        ax.plot(
            slab_necking_schmalholz["Time"][0] / characteristic_time,
            slab_necking_schmalholz["Thickness"][0] / slab_width * 1e3,
            label="Schmalholz (2011)",
        )

        ax.plot(
            diag_fields["normalised_time"],
            diag_fields["slab_necking"],
            label="Conservative level set",
        )

        ax.legend(fontsize=12, fancybox=True, shadow=True)

        fig.savefig(
            f"{output_path}/slab_necking_{tag}.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (1e6, 6.6e5)
mesh_gen = "gmsh"
mesh_vert_res = 1.5e4
mesh_fine_layer_min_x = 4.2e5
mesh_fine_layer_width = 1.6e5
mesh_fine_layer_hor_res = 8e3

# Degree of the function space on which the level-set function is defined.
level_set_func_space_deg = 2

# Parameters to initialise level set
interface_coords = [
    (0, 5.8e5),
    (4.6e5, 5.8e5),
    (4.6e5, 3.3e5),
    (5.4e5, 3.3e5),
    (5.4e5, 5.8e5),
    (domain_dims[0], 5.8e5),
]

boundary_coords = [
    (domain_dims[0], domain_dims[1]),
    (0, domain_dims[1]),
    (0, 5.8e5),
]
# Keyword arguments to define the signed-distance function
signed_distance_kwargs = {
    "interface_geometry": "polygon",
    "interface_coordinates": interface_coords,
    "boundary_coordinates": boundary_coords,
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
materials = [mantle, lithosphere]

# Approximation parameters
dimensional = True
Ra, g = 1, 9.81

# Boundary conditions
temp_bcs = {}
stokes_bcs = {
    1: {"ux": 0, "uy": 0},
    2: {"ux": 0, "uy": 0},
    3: {"uy": 0},
    4: {"uy": 0},
}

# Stokes solver options
stokes_nullspace_args = {}
stokes_solver_params = None

# Timestepping objects
initial_timestep = 1e11
subcycles = 1
dump_period = 5e5 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 25e6 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"normalised_time": [], "slab_necking": []}
lithosphere_thickness = 8e4
slab_length = 2.5e5
slab_width = 8e4
characteristic_time = (
    4 * lithosphere.visc_coeff / (lithosphere.rho - mantle.rho) / g / slab_length
) ** lithosphere.stress_exponent
