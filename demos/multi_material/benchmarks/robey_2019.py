from functools import partial

import initial_signed_distance as isd
import numpy as np

from gadopt.level_set_tools import AbstractMaterial
from gadopt.utility import node_coordinates


class ReferenceMaterial(AbstractMaterial):
    @classmethod
    def B(cls):
        return 0

    @classmethod
    def RaB(cls):
        return None

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


class BottomMaterial(AbstractMaterial):
    @classmethod
    def B(cls):
        return 0.2

    @classmethod
    def RaB(cls):
        return None

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
    name = "Robey_2019"

    # In material_interfaces, for each sub-list, the first material corresponds to the
    # negative side of the signed distance function
    materials = {"ref_mat": ReferenceMaterial, "bottom_mat": BottomMaterial}
    material_interfaces = [[materials["bottom_mat"], materials["ref_mat"]]]

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (3, 1)
    mesh_elements = (196, 64)

    slope = 0
    intercept = 0.5
    isd_params = [(slope, intercept)]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    Ra, g = 1e5, 1

    A = 0.05
    k = 1.5

    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    dt = 1e-6
    subcycles = 1
    time_end = 0.0236
    dump_period = 4e-4

    @classmethod
    def initialise_temperature(cls, temperature):
        node_coords_x, node_coords_y = node_coordinates(temperature)

        mask_bottom = node_coords_y <= 1 / 10
        mask_top = node_coords_y >= 9 / 10

        temperature.dat.data[:] = 0.5
        temperature.dat.data[mask_bottom] = (
            1
            - 5 * node_coords_y[mask_bottom]
            + cls.A
            * np.sin(10 * np.pi * node_coords_y[mask_bottom])
            * (1 - np.cos(2 / 3 * cls.k * np.pi * node_coords_x[mask_bottom]))
        )
        temperature.dat.data[mask_top] = (
            5
            - 5 * node_coords_y[mask_top]
            + cls.A
            * np.sin(10 * np.pi * node_coords_y[mask_top])
            * (1 - np.cos(2 / 3 * cls.k * np.pi * node_coords_x[mask_top] + np.pi))
        )

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
