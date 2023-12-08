from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.special import erf

import gadopt as ga


class ReferenceMaterial(ga.AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return 0

    @classmethod
    def density(cls):
        return None

    @classmethod
    def viscosity(cls, velocity):
        return 1

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


class DenseMaterial(ga.AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return 4.5e5

    @classmethod
    def density(cls):
        return None

    @classmethod
    def viscosity(cls, velocity):
        return 1

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
    name = "van_Keken_1997_thermochemical"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (2, 1)
    mesh_elements = (512, 256)

    # Parameters to initialise level sets
    slope = 0
    intercept = 0.025
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(slope, intercept)]
    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    materials = [DenseMaterial, ReferenceMaterial]
    reference_material = ReferenceMaterial

    # Physical parameters
    Ra, g = 3e5, 1

    # Boundary conditions
    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e-6
    subcycles = 1
    time_end = 0.05
    dump_period = 1e-4

    # Diagnostic objects
    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.2
    diag_params = {
        "domain_dim_x": domain_dimensions[0],
        "material_interface_y": intercept,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        u0 = (
            cls.domain_dimensions[0] ** (7 / 3)
            / (1 + cls.domain_dimensions[0] ** 4) ** (2 / 3)
            * (cls.Ra / 2 / fd.sqrt(fd.pi)) ** (2 / 3)
        )
        v0 = u0
        Q = 2 * fd.sqrt(cls.domain_dimensions[0] / fd.pi / u0)
        Tu = erf((1 - mesh_coords[1]) / 2 * fd.sqrt(u0 / mesh_coords[0])) / 2
        Tl = 1 - 1 / 2 * erf(
            mesh_coords[1]
            / 2
            * fd.sqrt(u0 / (cls.domain_dimensions[0] - mesh_coords[0]))
        )
        Tr = 1 / 2 + Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(
            v0 / (mesh_coords[1] + 1)
        ) * fd.exp(-(mesh_coords[0] ** 2) * v0 / (4 * mesh_coords[1] + 4))
        Ts = 1 / 2 - Q / 2 / fd.sqrt(fd.pi) * fd.sqrt(
            v0 / (2 - mesh_coords[1])
        ) * fd.exp(
            -((cls.domain_dimensions[0] - mesh_coords[0]) ** 2)
            * v0
            / (8 - 4 * mesh_coords[1])
        )

        temperature.interpolate(Tu + Tl + Tr + Ts - 3 / 2)
        fd.DirichletBC(temperature.function_space(), 1, 3).apply(temperature)
        fd.DirichletBC(temperature.function_space(), 0, 4).apply(temperature)
        temperature.interpolate(fd.max_value(fd.min_value(temperature, 1), 0))

    @classmethod
    def diagnostics(cls, simu_time, variables):
        cls.diag_fields["output_time"].append(simu_time)
        cls.diag_fields["rms_velocity"].append(ga.rms_velocity(variables["velocity"]))
        cls.diag_fields["entrainment"].append(
            ga.entrainment(
                variables["level_set"][0],
                cls.diag_params["domain_dim_x"],
                cls.diag_params["material_interface_y"],
                cls.diag_params["entrainment_height"],
            )
        )

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(cls.diag_fields["output_time"], cls.diag_fields["rms_velocity"])
            ax[1].plot(cls.diag_fields["output_time"], cls.diag_fields["entrainment"])

            fig.savefig(
                f"{cls.name}/rms_velocity_and_entrainment.pdf".lower(),
                dpi=300,
                bbox_inches="tight",
            )
