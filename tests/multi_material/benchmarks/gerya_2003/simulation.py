"""Compositional benchmark.
Gerya, T. V., & Yuen, D. A. (2003).
Characteristics-based marker-in-cell method with conservative finite-differences
schemes for modeling geological flows with strongly variable transport properties.
Physics of the Earth and Planetary Interiors, 140(4), 293-318.
"""

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .materials import buoyant_material, dense_material


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    level_set = diag_vars["level_set"][0]

    block_area = fd.assemble(fd.conditional(level_set >= 0.5, 1, 0) * fd.dx)

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
    diag_fields["block_area"].append(block_area / 1e10)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Relative block area")

        ax.plot(
            diag_fields["output_time"],
            diag_fields["block_area"],
            label="Conservative level set",
        )

        ax.legend(fontsize=12, fancybox=True, shadow=True)

        fig.savefig(f"{output_path}/block_area.pdf", dpi=300, bbox_inches="tight")


# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (5e5, 5e5)
mesh_gen = "firedrake"
mesh_elements = (64, 64)

# Parameters to initialise level sets
ref_vertex_x = 2e5
ref_vertex_y = 3.5e5
edge_length = 1e5
# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [isd.isd_rectangle]
isd_params = [(ref_vertex_x, ref_vertex_y, edge_length)]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [buoyant_material, dense_material]

# Approximation parameters
dimensional = True
g = 9.8

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e11
dump_period = 1e5 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 9.886e6 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"output_time": [], "block_area": []}
