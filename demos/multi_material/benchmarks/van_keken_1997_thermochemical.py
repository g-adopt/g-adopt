from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


class Simulation:
    """Thermochemical benchmark.
    Van Keken, P. E., King, S. D., Schmeling, H., Christensen, U. R., Neumeister, D.,
    & Doin, M. P. (1997).
    A comparison of methods for the modeling of thermochemical convection.
    Journal of Geophysical Research: Solid Earth, 102(B10), 22477-22495.
    """

    name = "van_Keken_1997_thermochemical"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (2, 1)
    domain_origin = (0, 0)
    mesh_elements = (128, 64)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level sets
    material_interface_y = 0.025
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
    reference_material = ga.Material(RaB=0)
    dense_material = ga.Material(RaB=4.5e5)
    materials = [dense_material, reference_material]
    reference_material = None

    # Physical parameters
    Ra, g = 3e5, 1

    # Boundary conditions
    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e-6
    subcycles = 1
    dump_period = 1e-4
    checkpoint_period = 5
    time_end = 0.05

    # Diagnostic objects
    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.2
    diag_params = {
        "domain_dim_x": domain_dims[0],
        "material_interface_y": material_interface_y,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        u0 = (
            cls.domain_dims[0] ** (7 / 3)
            / (1 + cls.domain_dims[0] ** 4) ** (2 / 3)
            * (cls.Ra / 2 / fd.sqrt(fd.pi)) ** (2 / 3)
        )
        v0 = u0
        Q = 2 * fd.sqrt(cls.domain_dims[0] / fd.pi / u0)
        Tu = fd.erf((1 - mesh_coords[1]) / 2 * fd.sqrt(u0 / mesh_coords[0])) / 2
        Tl = 1 - 1 / 2 * fd.erf(
            mesh_coords[1] / 2 * fd.sqrt(u0 / (cls.domain_dims[0] - mesh_coords[0]))
        )
        Tr = 1 / 2 + Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(
            v0 / (mesh_coords[1] + 1)
        ) * fd.exp(-(mesh_coords[0] ** 2) * v0 / (4 * mesh_coords[1] + 4))
        Ts = 1 / 2 - Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(
            v0 / (2 - mesh_coords[1])
        ) * fd.exp(
            -((cls.domain_dims[0] - mesh_coords[0]) ** 2)
            * v0
            / (8 - 4 * mesh_coords[1])
        )

        temperature.interpolate(
            fd.max_value(fd.min_value(Tu + Tl + Tr + Ts - 3 / 2, 1), 0)
        )
        fd.DirichletBC(temperature.function_space(), 1, 3).apply(temperature)
        fd.DirichletBC(temperature.function_space(), 0, 4).apply(temperature)

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
            rms_vel_schmeling = np.loadtxt("data/HSvrms4800.dat")
            entr_schmeling = np.loadtxt("data/HSentr4800.dat")
            rms_vel_van_keken = np.loadtxt("data/pvk120_003.vrms.dat")
            entr_van_keken = np.loadtxt("data/pvk120_003.entr.dat")
            rms_vel_christensen = np.loadtxt("data/URC125x40_240k.vrms.dat")
            entr_christensen = np.loadtxt("data/URC125x40_240k.entr.dat")
            rms_vel_christensen_chain = np.loadtxt("data/URCchain.vrms.dat")
            entr_christensen_chain = np.loadtxt("data/URCchain.entr.dat")

            fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(
                rms_vel_schmeling[:, 0],
                rms_vel_schmeling[:, 1],
                linestyle="dotted",
                label="HS (van Keken et al., 1997)",
            )
            ax[1].plot(
                entr_schmeling[:, 0],
                entr_schmeling[:, 1],
                linestyle="dotted",
                label="HS (van Keken et al., 1997)",
            )
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
                rms_vel_christensen[:, 0],
                rms_vel_christensen[:, 1],
                linestyle="dotted",
                label="CND (van Keken et al., 1997)",
            )
            ax[1].plot(
                entr_christensen[:, 0],
                entr_christensen[:, 1],
                linestyle="dotted",
                label="CND (van Keken et al., 1997)",
            )
            ax[0].plot(
                rms_vel_christensen_chain[:, 0],
                rms_vel_christensen_chain[:, 1],
                linestyle="dotted",
                label="CND Markerchain (van Keken et al., 1997)",
            )
            ax[1].plot(
                entr_christensen_chain[:, 0],
                entr_christensen_chain[:, 1],
                linestyle="dotted",
                label="CND Markerchain (van Keken et al., 1997)",
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
