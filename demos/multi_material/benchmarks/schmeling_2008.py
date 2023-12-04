from functools import partial

import initial_signed_distance as isd

from gadopt.level_set_tools import AbstractMaterial


class Mantle(AbstractMaterial):
    def B():
        return None

    def RaB():
        return None

    def density():
        return 3200

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
        return 1e19

    def thermal_expansion():
        return 1

    def thermal_conductivity():
        return 1

    def specific_heat_capacity():
        return 1

    def internal_heating_rate():
        return 0


class Simulation:
    name = "Schmeling_2008"

    # List simulation materials such that, starting from the end, each material
    # corresponds to the negative side of the signed distance function associated with
    # each level set.
    materials = [Mantle, Air, Lithosphere]
    reference_material = Mantle

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics
    # tracked by the level-set approach. Insufficient mesh refinement leads to the
    # vanishing of the material interface during advection and to unwanted motion of
    # the material interface during reinitialisation.
    domain_dimensions = (3e6, 7.5e5)
    mesh_elements = (256, 64)

    slope = 0
    intercept = 7e5
    # Ordering of the following two lists must match the logic implemented above in
    # materials. For example, the last level set should be able to uniquely describe
    # the last material.
    isd_params = [(slope, intercept), None]
    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
        isd.isd_schmeling,
    ]

    Ra, g = 1, 9.81

    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    dt = 1e11
    subcycles = 1
    time_end = 4e7 * 365.25 * 8.64e4
    dump_period = 8e5 * 365.25 * 8.64e4

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
