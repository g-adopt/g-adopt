from functools import partial

import initial_signed_distance as isd

from gadopt.level_set_tools import AbstractMaterial


class Mantle(AbstractMaterial):
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


class Lithosphere(AbstractMaterial):
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


class Air(AbstractMaterial):
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

    # List simulation materials such that, starting from the end, each material
    # corresponds to the negative side of the signed distance function associated with
    # each level set.
    materials = [Mantle, Lithosphere, Air]
    reference_material = Mantle

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (2.8e6, 8e5)
    mesh_elements = (128, 64)

    slope = 0
    intercept = 6e5
    material_interface_y = 7e5
    interface_deflection = 7e3
    isd_params = [
        (slope, intercept),
        (interface_deflection, domain_dimensions[0], material_interface_y),
    ]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.cosine_curve),
    ]

    Ra, g = 1, 10

    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"uy": 0}}

    dt = 1e10
    subcycles = 1
    time_end = 1e5 * 365.25 * 8.64e4
    dump_period = 2e3 * 365.25 * 8.64e4

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
