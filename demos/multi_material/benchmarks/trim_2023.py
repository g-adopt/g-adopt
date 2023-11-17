from functools import partial

import initial_signed_distance as isd
import numpy as np

from gadopt.level_set_tools import AbstractMaterial
from gadopt.utility import node_coordinates


class ReferenceMaterial(AbstractMaterial):
    def B():
        return 0

    def RaB():
        return None

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


class BottomMaterial(AbstractMaterial):
    def B():
        return -0.5

    def RaB():
        return None

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
    name = "Trim_2023"

    # In material_interfaces, for each sub-list, the first material corresponds to the
    # negative side of the signed distance function
    materials = {"ref_mat": ReferenceMaterial, "bottom_mat": BottomMaterial}
    material_interfaces = [[materials["bottom_mat"], materials["ref_mat"]]]

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (1, 1)
    mesh_elements = (32, 32)

    slope = 0
    intercept = 0.5
    isd_params = [(slope, intercept)]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    Ra, g = 1e5, 1

    a = 100
    b = 100
    t = 0
    f = a * np.sin(np.pi * b * t)
    k = 35

    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    dt = 1e-6
    subcycles = 1
    time_end = 0.01
    dump_period = 1e-4

    @classmethod
    def initialise_temperature(cls, temperature):
        node_coords_x, node_coords_y = node_coordinates(temperature)

        C0 = 1 / (1 + np.exp(-2 * cls.k * (cls.intercept - node_coords_y)))

        temperature.dat.data[:] = (
            -np.pi**3
            * (cls.domain_dimensions[0] ** 2 + 1) ** 2
            / cls.domain_dimensions[0] ** 3
            * np.cos(np.pi * node_coords_x / cls.domain_dimensions[0])
            * np.sin(np.pi * node_coords_y)
            * cls.f
            + RaB * C0
            + (cls.Ra - RaB) * (1 - node_coords_y)
        ) / cls.Ra

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
