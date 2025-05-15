from dataclasses import dataclass

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pandas import read_excel

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


@dataclass
class Lithosphere(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e23


@dataclass
class Air(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e19


class Simulation:
    """Compositional benchmark.
    Schmeling, H., Babeyko, A. Y., Enns, A., Faccenna, C., Funiciello, F., Gerya, T.,
    ... & Van Hunen, J. (2008).
    A benchmark comparison of spontaneous subduction models—Towards a free surface.
    Physics of the Earth and Planetary Interiors, 171(1-4), 198-223.
    """

    name = "Schmeling_2008"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (3e6, 7.5e5)
    domain_origin = (0, 0)
    mesh_file = "benchmarks/schmeling_2008.msh"

    # Degree of the function space on which the level-set function @classmethod
    # is defined.cls
    level_set_func_space_deg = 2

    # Parameters to initialise surface level set
    interface_coords_x = np.array([0.0, domain_dims[0]])
    callable_args = (surface_slope := 0, surface_coord_y := 7e5)
    surface_signed_distance_kwargs = {
        "interface_geometry": "curve",
        "interface_callable": "line",
        "interface_args": (interface_coords_x, *callable_args),
    }
    # Parameters to initialise slab level set
    slab_interface_coords = [
        (domain_dims[0], 7e5),
        (1e6, 7e5),
        (1e6, 5e5),
        (1.1e6, 5e5),
        (1.1e6, 6e5),
        (domain_dims[0], 6e5),
    ]
    slab_boundary_coords = [(domain_dims[0], 7e5)]
    slab_signed_distance_kwargs = {
        "interface_geometry": "polygon",
        "interface_coordinates": slab_interface_coords,
        "boundary_coordinates": slab_boundary_coords,
    }
    # The following list must be ordered such that, unpacking from the end, each dictionary
    # contains the keyword arguments required to initialise the signed-distance array
    # corresponding to the interface between a given material and the remainder of the
    # numerical domain (all previous materials excluded). By convention, the material thus
    # isolated occupies the positive side of the signed-distance array.
    signed_distance_kwargs_list = [
        surface_signed_distance_kwargs,
        slab_signed_distance_kwargs,
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    mantle = Mantle(density=3200)
    lithosphere = Lithosphere(density=3300)
    air = Air(density=0)
    materials = [mantle, air, lithosphere]
    reference_material = mantle

    # Physical parameters
    Ra, g = 1, 9.81

    # Boundary conditions
    temp_bcs = {}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e11
    subcycles = 1
    dump_period = 8e5 * 365.25 * 8.64e4
    checkpoint_period = 5
    time_end = 6e7 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"output_time": [], "slab_tip_depth": []}

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        level_set = diag_vars["level_set"][1]
        function_space = level_set.function_space()

        depth_per_core = fd.Function(function_space).interpolate(
            fd.conditional(
                level_set >= 0.5,
                cls.domain_dims[1] - function_space.mesh().coordinates[1],
                np.nan,
            )
        )
        max_depth_per_core = np.nanmax(depth_per_core.dat.data_ro_with_halos, initial=0)
        max_depth = level_set.comm.allreduce(max_depth_per_core, MPI.MAX)

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
        cls.diag_fields["slab_tip_depth"].append((max_depth - 5e4) / 1e3)

        if MPI.COMM_WORLD.rank == 0:
            np.savez(
                f"{cls.name.lower()}/output_{Simulation.restart_from_checkpoint}_check",
                diag_fields=cls.diag_fields,
            )

    @classmethod
    def plot_diagnostics(cls):
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
                cls.diag_fields["output_time"],
                cls.diag_fields["slab_tip_depth"],
                color="black",
                label="Conservative level set",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/slab_tip_depth.pdf".lower(), dpi=300, bbox_inches="tight"
            )
