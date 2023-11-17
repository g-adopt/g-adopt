from functools import partial

import initial_signed_distance as isd

from gadopt.level_set_tools import AbstractMaterial


class Mantle(AbstractMaterial):
    def B():
        return None

    def RaB():
        return None

    def density():
        return 3300

    @classmethod
    def viscosity(cls, velocity):
        return 1e21

    def thermal_expansion():
        return 1

    def thermal_conductivity():
        return 1

    def specific_heat_capacity():
        return 1

    def internal_heating_rate():
        return 0


class Lithosphere(AbstractMaterial):
    def B():
        return None

    def RaB():
        return None

    def density():
        return 3300

    @classmethod
    def viscosity(cls, velocity):
        return 1e23

    def thermal_expansion():
        return 1

    def thermal_conductivity():
        return 1

    def specific_heat_capacity():
        return 1

    def internal_heating_rate():
        return 0


class Air(AbstractMaterial):
    def B():
        return None

    def RaB():
        return None

    def density():
        return 0

    @classmethod
    def viscosity(cls, velocity):
        return 1e18

    def thermal_expansion():
        return 1

    def thermal_conductivity():
        return 1

    def specific_heat_capacity():
        return 1

    def internal_heating_rate():
        return 0


class Simulation:
    name = "Crameri_2012"

    # In material_interfaces, for each sub-list, the first material corresponds to the
    # negative side of the signed distance function
    materials = {"ref_mat": Mantle, "lithosphere": Lithosphere, "air": Air}
    material_interfaces = [
        [materials["lithosphere"], materials["air"]],
        [materials["ref_mat"], materials["lithosphere"]],
    ]

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (2.8e6, 8e5)
    mesh_elements = (128, 128)

    slope = 0
    intercept = 6e5
    material_interface_y = 7e5
    interface_deflection = 7e3
    isd_params = [
        (interface_deflection, domain_dimensions[0], material_interface_y),
        (slope, intercept),
    ]

    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.cosine_curve),
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
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
