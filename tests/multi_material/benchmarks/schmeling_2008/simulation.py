"""Compositional benchmark.
Schmeling, H., Babeyko, A. Y., Enns, A., Faccenna, C., Funiciello, F., Gerya, T.,
... & Van Hunen, J. (2008).
A benchmark comparison of spontaneous subduction models—Towards a free surface.
Physics of the Earth and Planetary Interiors, 171(1-4), 198-223.
"""

from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pandas import read_excel

from .materials import mantle, air, lithosphere


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    level_set = diag_vars["level_set"][1]
    function_space = level_set.function_space()

    depth_per_core = fd.Function(function_space).interpolate(
        fd.conditional(
            level_set >= 0.5,
            domain_dims[1] - function_space.mesh().coordinates[1],
            np.nan,
        )
    )
    max_depth_per_core = np.nanmax(depth_per_core.dat.data_ro_with_halos, initial=0)
    max_depth = level_set.comm.allreduce(max_depth_per_core, MPI.MAX)

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
    diag_fields["slab_tip_depth"].append((max_depth - 5e4) / 1e3)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_check", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        datafile = "data/zslab-case1-best-reformatted.xlsx"

        model_names_schmeling = read_excel(
            datafile, sheet_name="Tabelle1", header=None, skiprows=2, nrows=1
        )

        geom_cols = []
        geom_col_names = []
        for col in model_names_schmeling:
            model_name = model_names_schmeling[col].item()
            if isinstance(model_name, str) and "geom" in model_name:
                geom_cols.append(col)
                geom_col_names.append(model_name)

        cols_to_read = np.repeat(geom_cols, 2)
        cols_to_read[1::2] += 1

        model_data = read_excel(
            datafile,
            sheet_name="Tabelle1",
            header=None,
            usecols=cols_to_read,
            skiprows=4,
        )

        fdcon_data = read_excel(
            datafile, sheet_name="zslabmodel5", header=None, usecols=[1, 2]
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.invert_yaxis()
        ax.set_xlim(right=65)

        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Slab tip depth (km)")

        ax.plot(
            fdcon_data[1].dropna().values,
            fdcon_data[2].dropna().values,
            linestyle="dotted",
            label="FDCON 561x141 geom",
        )

        for col, col_name in zip(geom_cols, geom_col_names):
            ax.plot(
                model_data[col].dropna().values,
                model_data[col + 1].dropna().values,
                linestyle="dotted",
                label=col_name,
            )

        ax.plot(
            diag_fields["output_time"],
            diag_fields["slab_tip_depth"],
            color="black",
            label="Conservative level set",
        )

        ax.legend()

        fig.savefig(f"{output_path}/slab_tip_depth.pdf", dpi=300, bbox_inches="tight")


# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (3e6, 7.5e5)
mesh_gen = "gmsh"

# Parameters to initialise level sets
material_interface_y = 7e5
interface_slope = 0
# The following two lists must be ordered such that, unpacking from the end, each
# pair of arguments enables initialising a level set whose 0-contour corresponds to
# the entire interface between a given material and the remainder of the numerical
# domain. By convention, the material thereby isolated occupies the positive side
# of the signed-distance level set.
initialise_signed_distance = [
    partial(isd.isd_simple_curve, domain_dims[0], isd.straight_line),
    isd.isd_schmeling,
]
isd_params = [(interface_slope, material_interface_y), None]

# Material ordering must follow the logic implemented in the above two lists. In
# other words, the last material in the below list corresponds to the portion of
# the numerical domain entirely isolated by the level set initialised using the
# last pair of arguments in the above two lists. The first material in the below list
# must, therefore, occupy the negative side of the signed-distance level set initialised
# from the first pair of arguments above.
materials = [mantle, air, lithosphere]

# Approximation parameters
dimensional = True
buoyancy_terms = ["compositional"]
g = 9.81

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e11
dump_period = 8e5 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 6e7 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"output_time": [], "slab_tip_depth": []}
