from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


class Simulation:
    """Thermochemical benchmark.
    Robey, J. M., & Puckett, E. G. (2019).
    Implementation of a volume-of-fluid method in a finite element code with
    applications to thermochemical convection in a density stratified fluid in the
    Earth's mantle.
    Computers & Fluids, 190, 217-253.
    """

    name = "Robey_2019"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (3, 1)
    domain_origin = (0, 0)
    mesh_elements = (96, 32)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level sets
    material_interface_y = 0.5
    interface_slope = 0
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(interface_slope, material_interface_y)]
    initialise_signed_distance = [
        partial(
            isd.isd_simple_curve,
            domain_origin[0],
            domain_dims[0],
            isd.straight_line,
        )
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    top_material = ga.Material(B=0)
    bottom_material = ga.Material(B=0.2)
    materials = [bottom_material, top_material]
    reference_material = None

    # Physical parameters
    Ra, g = 1e5, 1

    # Parameters to initialise temperature
    A = 0.05
    k = 1.5

    # Boundary conditions
    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e-6
    subcycles = 1
    dump_period = 2e-4
    checkpoint_period = 5
    time_end = 0.0236

    # Diagnostic objects
    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.5
    diag_params = {
        "domain_dim_x": domain_dims[0],
        "material_interface_y": material_interface_y,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        bottom_tbl = (
            1
            - 5 * mesh_coords[1]
            + cls.A
            * fd.sin(10 * fd.pi * mesh_coords[1])
            * (1 - fd.cos(2 / 3 * cls.k * fd.pi * mesh_coords[0]))
        )
        top_tbl = (
            5
            - 5 * mesh_coords[1]
            + cls.A
            * fd.sin(10 * fd.pi * mesh_coords[1])
            * (1 - fd.cos(2 / 3 * cls.k * fd.pi * mesh_coords[0] + fd.pi))
        )

        initial_temperature = fd.conditional(
            mesh_coords[1] <= 0.1,
            bottom_tbl,
            fd.conditional(mesh_coords[1] >= 0.9, top_tbl, 0.5),
        )
        temperature.interpolate(initial_temperature)

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        cls.diag_fields["output_time"].append(simu_time)
        cls.diag_fields["rms_velocity"].append(geo_diag.u_rms())
        cls.diag_fields["entrainment"].append(
            ga.entrainment(
                diag_vars["level_set"][0],
                cls.diag_params["domain_dim_x"]
                * cls.diag_params["material_interface_y"],
                cls.diag_params["entrainment_height"],
            )
        )

        if MPI.COMM_WORLD.rank == 0:
            np.savez(
                f"{cls.name.lower()}/output_{Simulation.restart_from_checkpoint}_check",
                diag_fields=cls.diag_fields,
            )

    @classmethod
    def plot_diagnostics(cls):
        if MPI.COMM_WORLD.rank == 0:
            fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

            ax[0].grid()
            ax[1].grid()

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["rms_velocity"],
                label="Conservative level set",
            )
            ax[1].plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["entrainment"],
                label="Conservative level set",
            )

            ax[0].legend(fontsize=12, fancybox=True, shadow=True)
            ax[1].legend(fontsize=12, fancybox=True, shadow=True)

            fig.savefig(
                f"{cls.name}/rms_velocity_and_entrainment.pdf".lower(),
                dpi=300,
                bbox_inches="tight",
            )
