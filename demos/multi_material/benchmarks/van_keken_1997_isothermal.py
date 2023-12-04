from functools import partial

import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np

from gadopt.diagnostics import entrainment, rms_velocity
from gadopt.level_set_tools import AbstractMaterial


class ReferenceMaterial(AbstractMaterial):
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


class BuoyantMaterial(AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return -1

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
    name = "van_Keken_1997_isothermal"

    # List simulation materials such that, starting from the end, each material corresponds
    # to the negative side of the signed distance function associated to each level set.
    materials = [BuoyantMaterial, ReferenceMaterial]
    reference_material = ReferenceMaterial

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (0.9142, 1)
    mesh_elements = (64, 64)

    material_interface_y = 0.2
    interface_deflection = 0.02
    isd_params = [
        (interface_deflection, 2 * domain_dimensions[0], material_interface_y)
    ]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.cosine_curve)
    ]

    Ra, g = 0, 1

    temp_bcs = None
    stokes_bcs = {
        1: {"ux": 0},
        2: {"ux": 0},
        3: {"ux": 0, "uy": 0},
        4: {"ux": 0, "uy": 0},
    }

    dt = 1
    subcycles = 1
    time_end = 2000
    dump_period = 10

    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.2
    diag_params = {
        "domain_dim_x": domain_dimensions[0],
        "material_interface_y": material_interface_y,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        cls.diag_fields["output_time"].append(simu_time)
        cls.diag_fields["rms_velocity"].append(rms_velocity(variables["velocity"]))
        cls.diag_fields["entrainment"].append(
            entrainment(
                variables["level_set"][0],
                cls.diag_params["domain_dim_x"],
                cls.diag_params["material_interface_y"],
                cls.diag_params["entrainment_height"],
            )
        )

    @classmethod
    def save_and_plot(cls):
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
