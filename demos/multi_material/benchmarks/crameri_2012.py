from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


class Mantle(ga.AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return None

    @classmethod
    def density(cls):
        return 3300

    @classmethod
    def viscosity(cls, velocity):
        return 1e21

    @classmethod
    def thermal_expansion(cls):
        return 1

    @classmethod
    def thermal_conductivity(cls):
        return 1

    @classmethod
    def specific_heat_capacity(cls):
        return 1

    @classmethod
    def internal_heating_rate(cls):
        return 0


class Lithosphere(ga.AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return None

    @classmethod
    def density(cls):
        return 3300

    @classmethod
    def viscosity(cls, velocity):
        return 1e23

    @classmethod
    def thermal_expansion(cls):
        return 1

    @classmethod
    def thermal_conductivity(cls):
        return 1

    @classmethod
    def specific_heat_capacity(cls):
        return 1

    @classmethod
    def internal_heating_rate(cls):
        return 0


class Air(ga.AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return None

    @classmethod
    def density(cls):
        return 0

    @classmethod
    def viscosity(cls, velocity):
        return 1e18

    @classmethod
    def thermal_expansion(cls):
        return 1

    @classmethod
    def thermal_conductivity(cls):
        return 1

    @classmethod
    def specific_heat_capacity(cls):
        return 1

    @classmethod
    def internal_heating_rate(cls):
        return 0


class Simulation:
    name = "Crameri_2012"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (2.8e6, 8e5)
    mesh_elements = (512, 1024)

    # Parameters to initialise level sets
    slope = 0
    intercept = 6e5
    material_interface_y = 7e5
    interface_deflection = 7e3
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [
        (slope, intercept),
        (interface_deflection, domain_dimensions[0], material_interface_y),
    ]
    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.cosine_curve),
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    materials = [Mantle, Lithosphere, Air]
    reference_material = Mantle

    # Physical parameters
    Ra, g = 1, 10

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e10
    subcycles = 1
    time_end = 1e5 * 365.25 * 8.64e4
    dump_period = 2e3 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {
        "output_time": [],
        "max_topography": [],
        "max_topography_analytical": [],
    }
    relaxation_rate = -0.2139e-11

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        max_topography_analytical = (
            cls.interface_deflection / 1e3 * np.exp(cls.relaxation_rate * simu_time)
        )
        max_topo_per_core = (
            fd.Function(variables["level_set"][1].function_space())
            .interpolate(
                fd.conditional(
                    variables["level_set"][1] <= 0.5,
                    variables["level_set"][1].function_space().mesh().coordinates[1],
                    0,
                )
            )
            .dat.data_ro_with_halos.max(initial=0)
        )
        max_topography = variables["level_set"][1].comm.allreduce(
            max_topo_per_core, MPI.MAX
        )

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
        cls.diag_fields["max_topography"].append(
            (max_topography - cls.material_interface_y) / 1e3
        )
        cls.diag_fields["max_topography_analytical"].append(max_topography_analytical)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 7)

            ax.set_xlabel("Time (kyr)")
            ax.set_ylabel("Maximum topography (km)")

            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["max_topography"],
                label="Simulation",
            )
            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["max_topography_analytical"],
                label="Analytical",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/maximum_topography.pdf".lower(),
                dpi=300,
                bbox_inches="tight",
            )
