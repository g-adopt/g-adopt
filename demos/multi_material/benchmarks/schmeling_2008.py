from dataclasses import dataclass
from functools import partial

import firedrake as fd
import gadopt as ga
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


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
    name = "Schmeling_2008"

    # Degree of the function space on which the level-set function @classmethod
    # is defined.cls
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (3e6, 7.5e5)
    mesh_file = "benchmarks/schmeling_2008.msh"

    # Parameters to initialise level sets
    material_interface_y = 7e5
    interface_slope = 0
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(interface_slope, material_interface_y), None]
    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
        isd.isd_schmeling,
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
    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e11
    subcycles = 1
    time_end = 6e7 * 365.25 * 8.64e4
    dump_period = 8e5 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"output_time": [], "slab_tip_depth": []}

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        level_set = variables["level_set"][1]
        function_space = level_set.function_space()

        depth_per_core = fd.Function(function_space).interpolate(
            fd.conditional(
                level_set >= 0.5,
                cls.domain_dimensions[1] - function_space.mesh().coordinates[1],
                np.nan,
            )
        )
        max_depth_per_core = np.nanmax(depth_per_core.dat.data_ro_with_halos, initial=0)
        max_depth = level_set.comm.allreduce(max_depth_per_core, MPI.MAX)

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
        cls.diag_fields["slab_tip_depth"].append((max_depth - 5e4) / 1e3)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.grid()

            ax.invert_yaxis()

            ax.set_xlabel("Time (Myr)")
            ax.set_ylabel("Slab tip depth (km)")

            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["slab_tip_depth"],
                label="Conservative level set",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/slab_tip_depth.pdf".lower(), dpi=300, bbox_inches="tight"
            )
