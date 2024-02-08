from dataclasses import dataclass

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


@dataclass
class BuoyantMaterial(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


@dataclass
class DenseMaterial(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


class Simulation:
    name = "Gerya_2003"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (5e5, 5e5)
    mesh_elements = (128, 128)

    # Parameters to initialise level sets
    ref_vertex_x = 2e5
    ref_vertex_y = 3.5e5
    edge_length = 1e5
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(ref_vertex_x, ref_vertex_y, edge_length)]
    initialise_signed_distance = [isd.isd_rectangle]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    buoyant_material = BuoyantMaterial(density=3200)
    dense_material = DenseMaterial(density=3300)
    materials = [buoyant_material, dense_material]
    reference_material = buoyant_material

    # Physical parameters
    Ra, g = 1, 9.8

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e11
    subcycles = 1
    time_end = 9.886e6 * 365.25 * 8.64e4
    dump_period = 1e5 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"output_time": [], "max_depth": [], "min_depth": []}

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        level_set = variables["level_set"][0]
        function_space = level_set.function_space()

        depth_per_core = fd.Function(function_space).interpolate(
            fd.conditional(
                level_set >= 0.5,
                cls.domain_dimensions[1] - function_space.mesh().coordinates[1],
                np.nan,
            )
        )
        max_depth_per_core = np.nanmax(depth_per_core.dat.data_ro_with_halos, initial=0)
        min_depth_per_core = np.nanmin(
            depth_per_core.dat.data_ro_with_halos, initial=cls.domain_dimensions[1]
        )
        max_depth = level_set.comm.allreduce(max_depth_per_core, MPI.MAX)
        min_depth = level_set.comm.allreduce(min_depth_per_core, MPI.MIN)

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
        cls.diag_fields["max_depth"].append(max_depth / 1e3)
        cls.diag_fields["min_depth"].append(min_depth / 1e3)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.set_xlabel("Time (Myr)")
            ax.set_ylabel("Depth (km)")

            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["max_depth"],
                label="Deepest node within the block",
            )
            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["min_depth"],
                label="Shallowest node within the block",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/block_depth.pdf".lower(), dpi=300, bbox_inches="tight"
            )
