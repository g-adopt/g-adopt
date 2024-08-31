"""Compositional benchmark.
Crameri, F., Schmeling, H., Golabek, G. J., Duretz, T., Orendt, R.,
Buiter, S. J. H., ... & Tackley, P. J. (2012).
A comparison of numerical surface topography calculations in geodynamic modelling:
an evaluation of the 'sticky air' method.
Geophysical Journal International, 189(1), 38-54.
"""

from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .materials import air, lithosphere, mantle


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    max_topography_analytical = (
        top_interface_deflection / 1e3 * np.exp(relaxation_rate * simu_time)
    )

    epsilon = float(diag_vars["epsilon"])
    level_set = diag_vars["level_set"][1]
    level_set_data = level_set.dat.data_ro_with_halos
    coords_data = (
        fd.Function(
            fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
        )
        .interpolate(fd.SpatialCoordinate(level_set))
        .dat.data_ro_with_halos
    )

    mask_ls = level_set_data <= 0.5
    if mask_ls.any():
        ind_lower_bound = coords_data[mask_ls, 1].argmax()
        max_topo_lower_bound = coords_data[mask_ls, 1][ind_lower_bound]
        if not mask_ls.all():
            hor_coord_lower_bound = coords_data[mask_ls, 0][ind_lower_bound]
            mask_hor_coord = abs(coords_data[~mask_ls, 0] - hor_coord_lower_bound) < 1e3
            if mask_hor_coord.any():
                ind_upper_bound = coords_data[~mask_ls, 1][mask_hor_coord].argmin()
                max_topo_upper_bound = coords_data[~mask_ls, 1][mask_hor_coord][
                    ind_upper_bound
                ]

                ls_lower_bound = level_set_data[mask_ls][ind_lower_bound]
                sdls_lower_bound = epsilon * np.log(
                    ls_lower_bound / (1 - ls_lower_bound)
                )

                ls_upper_bound = level_set_data[~mask_ls][mask_hor_coord][
                    ind_upper_bound
                ]
                sdls_upper_bound = epsilon * np.log(
                    ls_upper_bound / (1 - ls_upper_bound)
                )

                ls_dist = sdls_upper_bound / (sdls_upper_bound - sdls_lower_bound)
                max_topo = (
                    ls_dist * max_topo_lower_bound
                    + (1 - ls_dist) * max_topo_upper_bound
                )
            else:
                max_topo = max_topo_lower_bound
        else:
            max_topo = max_topo_lower_bound
    else:
        max_topo = -np.inf

    max_topo_global = level_set.comm.allreduce(max_topo, MPI.MAX)

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
    diag_fields["max_topography"].append(
        (max_topo_global - top_material_interface_y) / 1e3
    )
    diag_fields["max_topography_analytical"].append(max_topography_analytical)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.set_xlabel("Time (kyr)")
        ax.set_ylabel("Maximum topography (km)")

        ax.plot(
            diag_fields["output_time"],
            diag_fields["max_topography_analytical"],
            label="Analytical (Crameri et al., 2012)",
        )
        ax.plot(
            diag_fields["output_time"],
            diag_fields["max_topography"],
            label="Conservative level set",
        )

        ax.legend()

        fig.savefig(
            f"{output_path}/maximum_topography.pdf", dpi=300, bbox_inches="tight"
        )


# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (2.8e6, 8e5)
mesh_gen = "gmsh"

# Parameters to initialise level sets
bottom_material_interface_y = 6e5
bottom_interface_slope = 0
top_material_interface_y = 7e5
top_interface_deflection = 7e3
# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [
    partial(isd.isd_simple_curve, domain_dims[0], isd.straight_line),
    partial(isd.isd_simple_curve, domain_dims[0], isd.cosine_curve),
]
isd_params = [
    (bottom_interface_slope, bottom_material_interface_y),
    (top_interface_deflection, domain_dims[0], top_material_interface_y),
]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [mantle, lithosphere, air]

# Approximation parameters
dimensional = True
buoyancy_terms = ["compositional"]
g = 10

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e10
dump_period = 2e3 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 1e5 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"output_time": [], "max_topography": [], "max_topography_analytical": []}
relaxation_rate = -0.2139e-11
