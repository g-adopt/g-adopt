"""Compositional benchmark.
Crameri, F., Schmeling, H., Golabek, G. J., Duretz, T., Orendt, R.,
Buiter, S. J. H., ... & Tackley, P. J. (2012).
A comparison of numerical surface topography calculations in geodynamic modelling:
an evaluation of the 'sticky air' method.
Geophysical Journal International, 189(1), 38-54.
"""

from functools import partial

import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from gadopt.level_set_tools import min_max_height

from .materials import air, lithosphere, mantle


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    max_topo = min_max_height(diag_vars["level_set"][1], diag_vars["epsilon"], 0, "max")
    max_topo_analytical = (
        top_interface_deflection / 1e3 * np.exp(relaxation_rate * simu_time)
    )

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
    diag_fields["max_topography"].append((max_topo - top_material_interface_y) / 1e3)
    diag_fields["max_topography_analytical"].append(max_topo_analytical)

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
