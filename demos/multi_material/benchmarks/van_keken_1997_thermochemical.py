from functools import partial

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from gadopt.diagnostics import entrainment, rms_velocity
from gadopt.level_set_tools import AbstractMaterial
from gadopt.utility import node_coordinates
from scipy.special import erf

import initial_signed_distance as isd


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
    def B():
        return None

    def RaB():
        return 4.5e5

    def density():
        return None

    @classmethod
    def viscosity(cls, velocity):
        return 1

    def thermal_expansion():
        return 1

    def thermal_conductivity():
        return 1

    def specific_heat_capacity():
        return 1

    def internal_heating_rate():
        return 0


class Simulation:
    name = "van_Keken_1997_isothermal"

    # In material_interfaces, for each sub-list, the first material corresponds to the
    # negative side of the signed distance function
    materials = {"ref_mat": ReferenceMaterial, "buoy_mat": BuoyantMaterial}
    material_interfaces = [[materials["buoy_mat"], materials["ref_mat"]]]

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (2, 1)
    mesh_elements = (256, 32)

    slope = 0
    intercept = 0.025
    isd_params = [(slope, intercept)]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    Ra, g = 3e5, 1

    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    dt = 1e-6
    subcycles = 1
    time_end = 0.05
    dump_period = 1e-4

    diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
    entrainment_height = 0.2
    diag_params = {
        "domain_dim_x": domain_dimensions[0],
        "material_interface_y": intercept,
        "entrainment_height": entrainment_height,
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        node_coords_x, node_coords_y = node_coordinates(temperature)

        u0 = (
            cls.domain_dimensions[0] ** (7 / 3)
            / (1 + cls.domain_dimensions[0]**4) ** (2 / 3)
            * (cls.Ra / 2 / np.sqrt(np.pi)) ** (2 / 3)
        )
        v0 = u0
        Q = 2 * np.sqrt(cls.domain_dimensions[0] / np.pi / u0)
        Tu = erf((1 - node_coords_y) / 2 * np.sqrt(u0 / node_coords_x)) / 2
        Tl = 1 - 1 / 2 * erf(
            node_coords_y / 2 * np.sqrt(u0 / (cls.domain_dimensions[0] - node_coords_x))
        )
        Tr = 1 / 2 + Q / 2 / np.sqrt(np.pi) * np.sqrt(
            v0 / (node_coords_y + 1)
        ) * np.exp(-(node_coords_x**2) * v0 / (4 * node_coords_y + 4))
        Ts = 1 / 2 - Q / 2 / np.sqrt(np.pi) * np.sqrt(
            v0 / (2 - node_coords_y)
        ) * np.exp(
            -((cls.domain_dimensions[0] - node_coords_x) ** 2) * v0 / (8 - 4 * node_coords_y)
        )

        temperature.dat.data[:] = Tu + Tl + Tr + Ts - 3 / 2
        fd.DirichletBC(temperature.function_space(), 1, 3).apply(temperature)
        fd.DirichletBC(temperature.function_space(), 0, 4).apply(temperature)
        temperature.interpolate(fd.max_value(fd.min_value(temperature, 1), 0))

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
        np.savez(f"{cls.name}/output", diag_fields=cls.diag_fields)

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