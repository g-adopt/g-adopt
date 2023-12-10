from functools import partial

import firedrake as fd
import initial_signed_distance as isd

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
        return 0


class BottomMaterial(ga.AbstractMaterial):
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

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (3, 1)
    mesh_elements = (192, 64)

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
    reference_material = TopMaterial

    # Physical parameters
    Ra, g = 1e5, 1

    # Parameters to initialise temperature
    A = 0.05
    k = 1.5

    # Boundary conditions
    temp_bcs = {1: {"flux": 0}, 2: {"flux": 0}, 3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
    dt = 1e-6
    subcycles = 1
    time_end = 0.0236
    dump_period = 2e-4

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        bottom_tbl = (
            1
            - 5 * mesh_coords[1]
            + cls.A
            * fd.sin(10 * fd.pi * mesh_coords[1])
            * (1 - fd.cos(2 / 3 * cls.k * fd.pi * mesh_coords[0]))
        )
        top_tbl = (
            5
            - 5 * mesh_coords[1]
            + cls.A
            * fd.sin(10 * fd.pi * mesh_coords[1])
            * (1 - fd.cos(2 / 3 * cls.k * fd.pi * mesh_coords[0] + fd.pi))
        )

        initial_temperature = fd.conditional(
            mesh_coords[1] <= 0.1,
            bottom_tbl,
            fd.conditional(mesh_coords[1] >= 0.9, top_tbl, 0.5),
        )
        temperature.interpolate(initial_temperature)

    @classmethod
    def diagnostics(cls, simu_time, variables):
        pass

    @classmethod
    def save_and_plot(cls):
        pass
