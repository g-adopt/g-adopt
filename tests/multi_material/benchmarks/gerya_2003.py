from dataclasses import dataclass

import firedrake as fd
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
    """Compositional benchmark.
    Gerya, T. V., & Yuen, D. A. (2003).
    Characteristics-based marker-in-cell method with conservative finite-differences
    schemes for modeling geological flows with strongly variable transport properties.
    Physics of the Earth and Planetary Interiors, 140(4), 293-318.
    """

    name = "Gerya_2003"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (5e5, 5e5)
    domain_origin = (0, 0)
    mesh_elements = (64, 64)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level set
    callable_args = (ref_vertex_coords := (2e5, 3.5e5), edge_sizes := (1e5, 1e5))
    signed_distance_kwargs = {
        "interface_geometry": "polygon",
        "interface_callable": "rectangle",
        "interface_args": callable_args,
    }
    # The following list must be ordered such that, unpacking from the end, each dictionary
    # contains the keyword arguments required to initialise the signed-distance array
    # corresponding to the interface between a given material and the remainder of the
    # numerical domain (all previous materials excluded). By convention, the material thus
    # isolated occupies the positive side of the signed-distance array.
    signed_distance_kwargs_list = [signed_distance_kwargs]

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

    # Approximation parameters
    dimensional = True
    Ra, g = 1, 9.8

    # Boundary conditions
    temp_bcs = {}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e11
    subcycles = 1
    dump_period = 1e5 * 365.25 * 8.64e4
    checkpoint_period = 5
    time_end = 9.886e6 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"output_time": [], "block_area": []}

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        level_set = diag_vars["level_set"][0]

        block_area = fd.assemble(fd.conditional(level_set >= 0.5, 1, 0) * fd.dx)

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
        cls.diag_fields["block_area"].append(block_area / 1e10)

        if MPI.COMM_WORLD.rank == 0:
            np.savez(
                f"{cls.name.lower()}/output_{Simulation.restart_from_checkpoint}_check",
                diag_fields=cls.diag_fields,
            )

    @classmethod
    def plot_diagnostics(cls):
        if MPI.COMM_WORLD.rank == 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.grid()

            ax.set_xlabel("Time (Myr)")
            ax.set_ylabel("Relative block area")

            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["block_area"],
                label="Conservative level set",
            )

            ax.legend(fontsize=12, fancybox=True, shadow=True)

            fig.savefig(
                f"{cls.name}/block_area.pdf".lower(), dpi=300, bbox_inches="tight"
            )
