from functools import partial

import firedrake as fd
import flib
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


class Simulation:
    """Thermochemical benchmark.
    Trim, S. J., Butler, S. L., McAdam, S. S., & Spiteri, R. J. (2023).
    Manufacturing an exact solution for 2D thermochemical mantle convection models.
    Geochemistry, Geophysics, Geosystems, 24(4), e2022GC010807.
    """

    name = "Trim_2023"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (1, 1)
    domain_origin = (0, 0)
    mesh_elements = (128, 128)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level sets
    slope = 0
    intercept = 0.5
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(slope, intercept)]
    initialise_signed_distance = [
        partial(
            isd.isd_simple_curve, domain_origin[0], domain_dims[0], isd.straight_line
        )
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    top_material = ga.Material(B=0)
    bottom_material = ga.Material(B=0.5)
    materials = [bottom_material, top_material]
    reference_material = None

    # Physical parameters
    Ra, g = 1e5, 1
    RaB = Ra * bottom_material.B

    # Parameters to initialise temperature
    a = 100
    b = 100
    k = 35

    # Boundary conditions
    C0_0 = 1 / (1 + fd.exp(-2 * k * (intercept - 0)))
    C0_1 = 1 / (1 + fd.exp(-2 * k * (intercept - 1)))
    temp_bc_bot = RaB / Ra * (C0_0 - 1) + 1
    temp_bc_top = RaB / Ra * C0_1
    temp_bcs = {3: {"T": temp_bc_bot}, 4: {"T": temp_bc_top}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e-6
    subcycles = 1
    dump_period = 1e-4
    checkpoint_period = 5
    time_end = 0.01

    # Diagnostic objects
    diag_fields = {
        "output_time": [],
        "rms_velocity": [],
        "rms_velocity_analytical": [],
        "entrainment": [],
    }
    material_area = domain_dims[0] * intercept
    entrainment_height = 0.5

    @classmethod
    def C0(cls, mesh_coord_y):
        return 1 / (1 + fd.exp(-2 * cls.k * (cls.intercept - mesh_coord_y)))

    @classmethod
    def f(cls, t):
        return cls.a * fd.sin(fd.pi * cls.b * t)

    @classmethod
    def initialise_temperature(cls, temperature):
        λ = cls.domain_dims[0]
        x, y = fd.SpatialCoordinate(temperature.function_space().mesh())

        temperature.interpolate(
            (
                -(fd.pi**3)
                * (λ**2 + 1) ** 2
                / λ**3
                * fd.cos(fd.pi * x / λ)
                * fd.sin(fd.pi * y)
                * cls.f(0)
                + cls.RaB * cls.C0(y)
                + (cls.Ra - cls.RaB) * (1 - y)
            )
            / cls.Ra
        )

    @classmethod
    def internal_heating_rate(cls, int_heat_rate, simu_time):
        node_coords_x, node_coords_y = ga.node_coordinates(int_heat_rate)

        # flib can be obtained from
        # https://github.com/seantrim/exact-thermochem-solution
        analytical_values = []
        for coord_x, coord_y in zip(node_coords_x, node_coords_y):
            analytical_values.append(
                flib.h_python(
                    coord_x,
                    coord_y,
                    float(simu_time),
                    cls.domain_dims[0],
                    cls.k,
                    cls.intercept,
                    cls.Ra,
                    cls.RaB,
                )
            )

        int_heat_rate.dat.data[:] = analytical_values

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        λ = cls.domain_dims[0]
        rms_velocity_analytical = (
            fd.pi * fd.sqrt(λ**2 + 1) / 2 / λ * abs(cls.f(simu_time))
        )

        cls.diag_fields["output_time"].append(simu_time)
        cls.diag_fields["rms_velocity"].append(geo_diag.u_rms())
        cls.diag_fields["rms_velocity_analytical"].append(rms_velocity_analytical)
        cls.diag_fields["entrainment"].append(
            ga.entrainment(
                diag_vars["level_set"][0], cls.material_area, cls.entrainment_height
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

            ax[0].set_ylim(0, 250)
            ax[1].set_ylim(0, 0.8)

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["rms_velocity"],
                label="Simulation",
            )
            ax[0].plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["rms_velocity_analytical"],
                label="Analytical",
            )
            ax[1].plot(cls.diag_fields["output_time"], cls.diag_fields["entrainment"])

            ax[0].legend()

            fig.savefig(
                f"{cls.name}/rms_velocity_and_entrainment.pdf".lower(),
                dpi=300,
                bbox_inches="tight",
            )
