import firedrake as fd
import initial_signed_distance as isd

import gadopt as ga


class Mantle(ga.AbstractMaterial):
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


class Lithosphere(ga.AbstractMaterial):
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

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 1

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (1e6, 6.6e5)
    mesh_elements = (96, 64)

    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [None]
    initialise_signed_distance = [isd.isd_schmalholz]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    materials = [Mantle, Lithosphere]
    reference_material = Mantle

    # Physical parameters
    Ra, g = 1, 9.81

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {
        1: {"ux": 0, "uy": 0},
        2: {"ux": 0, "uy": 0},
        3: {"uy": 0},
        4: {"uy": 0},
    }

    # Timestepping objects
    dt = 1e11
    subcycles = 1
    time_end = 25e6 * 365.25 * 8.64e4
    dump_period = 5e5 * 365.25 * 8.64e4

    # Diagnostic objects
    slab_length = 2.5e5
    characteristic_time = (
        4
        * Lithosphere.visc_coeff
        / (Lithosphere.density() - Mantle.density())
        / g
        / slab_length
    ) ** Lithosphere.stress_exponent

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
