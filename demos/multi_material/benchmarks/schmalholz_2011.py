import firedrake as fd
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
        return 3150

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
    visc_coeff = 4.75e11
    stress_exponent = 4
    visc_bounds = (1e21, 1e25)

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
        strain_rate = fd.sym(fd.grad(velocity))
        strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2)

        visc = cls.visc_coeff * strain_rate_sec_inv ** (1 / cls.stress_exponent - 1)

        return fd.conditional(
            visc > cls.visc_bounds[1],
            cls.visc_bounds[1],
            fd.conditional(visc < cls.visc_bounds[0], cls.visc_bounds[0], visc),
        )

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
    name = "Schmalholz_2011"

    # List simulation materials such that, starting from the end, each material
    # corresponds to the negative side of the signed distance function associated with
    # each level set.
    materials = [Mantle, Lithosphere]
    reference_material = Mantle

    # Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
    # the level-set approach. Insufficient mesh refinement leads to the vanishing of the
    # material interface during advection and to unwanted motion of the material interface
    # during reinitialisation.
    domain_dimensions = (1e6, 6.6e5)
    mesh_elements = (96, 64)

    isd_params = [None]

    initialise_signed_distance = [isd.isd_schmalholz]

    Ra, g = 1, 9.81

    temp_bcs = None
    stokes_bcs = {
        1: {"ux": 0, "uy": 0},
        2: {"ux": 0, "uy": 0},
        3: {"uy": 0},
        4: {"uy": 0},
    }

    dt = 1e11
    subcycles = 1
    time_end = 25e6 * 365.25 * 8.64e4
    dump_period = 5e5 * 365.25 * 8.64e4

    slab_length = 2.5e5
    characteristic_time = (
        4
        * Lithosphere.visc_coeff
        / (Lithosphere.density() - Mantle.density())
        / g
        / slab_length
    ) ** Lithosphere.stress_exponent

    @staticmethod
    def initialise_temperature(temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
