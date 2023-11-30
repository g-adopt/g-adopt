import initial_signed_distance as isd

from gadopt.level_set_tools import AbstractMaterial


class ReferenceMaterial(AbstractMaterial):
    @classmethod
    def B(cls):
        return None

    @classmethod
    def RaB(cls):
        return None

    @classmethod
    def density(cls):
        return 3200

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


class DenseMaterial(AbstractMaterial):
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


class Simulation:
    name = "Gerya_2003"

    # List simulation materials such that, starting from the end, each material
    # corresponds to the negative side of the signed distance function associated with
    # each level set.
    materials = [ReferenceMaterial, DenseMaterial]
    reference_material = ReferenceMaterial

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (5e5, 5e5)
    mesh_elements = (128, 128)

    ref_vertex_x = 2e5
    ref_vertex_y = 3.5e5
    edge_length = 1e5
    isd_params = [(ref_vertex_x, ref_vertex_y, edge_length)]

    initialise_signed_distance = [isd.isd_rectangle]

    Ra, g = 1, 9.8

    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    dt = 1e11
    subcycles = 1
    time_end = 9.886e6 * 365.25 * 8.64e4
    dump_period = 1e5 * 365.25 * 8.64e4

    @staticmethod
    def initialise_temperature(temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
