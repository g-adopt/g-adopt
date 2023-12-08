from functools import partial

import firedrake as fd
import flib
import initial_signed_distance as isd
import numpy as np

import gadopt as ga


class TopMaterial(ga.AbstractMaterial):
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


class BottomMaterial(ga.AbstractMaterial):
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

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (1, 1)
    mesh_elements = (64, 64)

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
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line)
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    materials = [BottomMaterial, TopMaterial]
    reference_material = None

    # Physical parameters
    Ra, g = 1e5, 1
    RaB = Ra * BottomMaterial.B()

    # Parameters to initialise temperature
    a = 100
    b = 100
    t = 0
    f = a * np.sin(np.pi * b * t)
    k = 35

    # Boundary conditions
    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e-6
    subcycles = 1
    time_end = 0.01
    dump_period = 1e-4

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        C0 = 1 / (1 + fd.exp(-2 * cls.k * (cls.intercept - mesh_coords[1])))

        temperature.interpolate(
            -fd.pi**3
            * (cls.domain_dimensions[0] ** 2 + 1) ** 2
            / cls.domain_dimensions[0] ** 3
            * fd.cos(fd.pi * mesh_coords[0] / cls.domain_dimensions[0])
            * fd.sin(fd.pi * mesh_coords[1])
            * cls.f
            + cls.RaB * C0
            + (cls.Ra - cls.RaB) * (1 - mesh_coords[1])
        ) / cls.Ra

    @classmethod
    def internal_heating_rate(cls, int_heat_rate, simu_time):
        node_coords_x, node_coords_y = ga.node_coordinates(int_heat_rate)

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
