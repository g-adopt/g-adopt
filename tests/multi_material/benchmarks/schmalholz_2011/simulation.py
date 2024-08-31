"""Compositional benchmark.
Schmalholz, S. M. (2011).
A simple analytical solution for slab detachment.
Earth and Planetary Science Letters, 304(1-2), 45-54.
"""

import firedrake as fd
import gmsh
import initial_signed_distance as isd
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
        [(1, line_1)], mesh_fine_layer_min_x, 0, 0, numElements=[21], recombine=True
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


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    epsilon = float(diag_vars["epsilon"])
    level_set = diag_vars["level_set"][0]
    level_set_data = level_set.dat.data_ro_with_halos
    coords_data = (
        fd.Function(
            fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
        )
        .interpolate(fd.SpatialCoordinate(level_set))
        .dat.data_ro_with_halos
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

                ls_outside = max(
                    1e-6, min(1 - 1e-6, level_set_data[mask_ls_outside][ind_outside])
                )
                sdls_outside = epsilon * np.log(ls_outside / (1 - ls_outside))

                ls_inside = max(
                    1e-6,
                    min(
                        1 - 1e-6,
                        level_set_data[mask_ls_inside][mask_ver_coord][ind_inside],
                    ),
                )
                sdls_inside = epsilon * np.log(ls_inside / (1 - ls_inside))

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
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
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

        ax.legend()

        fig.savefig(f"{output_path}/slab_necking.pdf", dpi=300, bbox_inches="tight")


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

# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [isd.isd_schmalholz]
isd_params = [None]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [mantle, lithosphere]

# Approximation parameters
dimensional = True
buoyancy_terms = ["compositional"]
g = 9.81

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0, "uy": 0}, 2: {"ux": 0, "uy": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e11
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
