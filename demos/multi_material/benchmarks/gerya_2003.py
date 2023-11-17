import initial_signed_distance as isd

from gadopt.level_set_tools import AbstractMaterial


class ReferenceMaterial(AbstractMaterial):
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


class DenseMaterial(AbstractMaterial):
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


class Simulation:
    name = "Gerya_2003"

    # In material_interfaces, for each sub-list, the first material corresponds to the
    # negative side of the signed distance function
    materials = {"ref_mat": ReferenceMaterial, "dens_mat": DenseMaterial}
    material_interfaces = [[materials["dens_mat"], materials["ref_mat"]]]

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
