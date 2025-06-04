import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


class Simulation:
    """Compositional benchmark.
    Van Keken, P. E., King, S. D., Schmeling, H., Christensen, U. R., Neumeister, D.,
    & Doin, M. P. (1997).
    A comparison of methods for the modeling of thermochemical convection.
    Journal of Geophysical Research: Solid Earth, 102(B10), 22477-22495.
    """

    name = "van_Keken_1997_isothermal"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (0.9142, 1)
    domain_origin = (0, 0)
    mesh_elements = (128, 128)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level set
    interface_coords_x = np.linspace(0, domain_dims[0], 1000)
    callable_args = (
        interface_deflection := 0.02,
        perturbation_wavelength := 2 * domain_dims[0],
        initial_interface_y := 0.2,
    )
    signed_distance_kwargs = {
        "interface_geometry": "curve",
        "interface_callable": "cosine",
        "interface_args": (interface_coords_x, *callable_args),
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
    dense_material = ga.Material(RaB=0)
    buoyant_material = ga.Material(RaB=-1)
    materials = [buoyant_material, dense_material]
    reference_material = None

    # Physical parameters
    Ra, g = 0, 1

    # Boundary conditions
    temp_bcs = {}
    stokes_bcs = {
        1: {"ux": 0},
        2: {"ux": 0},
        3: {"ux": 0, "uy": 0},
        4: {"ux": 0, "uy": 0},
    }

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1
    subcycles = 1
    dump_period = 10
    checkpoint_period = 5
    time_end = 2000

    # Diagnostic objects
    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.2
    diag_params = {
        "domain_dim_x": domain_dims[0],
        "material_interface_y": initial_interface_y,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

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
            rms_vel_van_keken = np.loadtxt("data/pvk80_001.vrms.dat")
            entr_van_keken = np.loadtxt("data/pvk80_001.entr.dat")

            fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

            ax[0].grid()
            ax[1].grid()

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(
                rms_vel_van_keken[:, 0],
                rms_vel_van_keken[:, 1],
                linestyle="dotted",
                label="PvK (van Keken et al., 1997)",
            )
            ax[1].plot(
                entr_van_keken[:, 0],
                entr_van_keken[:, 1],
                linestyle="dotted",
                label="PvK (van Keken et al., 1997)",
            )

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
