from functools import partial

import flib
import initial_signed_distance as isd
import numpy as np

from gadopt.level_set_tools import AbstractMaterial
from gadopt.utility import node_coordinates


class TopMaterial(AbstractMaterial):
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
        return None


class BottomMaterial(AbstractMaterial):
    @classmethod
    def B(cls):
        return 0.5

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
        return None


class Simulation:
    name = "Trim_2023"

    # List simulation materials such that, starting from the end, each material
    # corresponds to the negative side of the signed distance function associated with
    # each level set.
    materials = [BottomMaterial, TopMaterial]
    reference_material = None

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (1, 1)
    mesh_elements = (64, 64)

    slope = 0
    intercept = 0.5
    isd_params = [(slope, intercept)]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    Ra, g = 1e5, 1
    RaB = Ra * BottomMaterial.B()

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
            + cls.RaB * C0
            + (cls.Ra - cls.RaB) * (1 - node_coords_y)
        ) / cls.Ra

    @classmethod
    def internal_heating_rate(cls, int_heat_rate, simu_time):
        node_coords_x, node_coords_y = node_coordinates(int_heat_rate)

        analytical_values = []
        for coord_x, coord_y in zip(node_coords_x, node_coords_y):
            analytical_values.append(
                flib.h_python(
                    coord_x,
                    coord_y,
                    simu_time,
                    cls.domain_dimensions[0],
                    cls.k,
                    cls.intercept,
                    cls.Ra,
                    cls.RaB,
                )
            )

        int_heat_rate.dat.data[:] = analytical_values

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
