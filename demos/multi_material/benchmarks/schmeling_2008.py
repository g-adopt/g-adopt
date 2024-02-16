from dataclasses import dataclass
from functools import partial

import initial_signed_distance as isd

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


@dataclass
class Lithosphere(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e23


@dataclass
class Air(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e19


class Simulation:
    name = "Schmeling_2008"

    # Degree of the function space on which the level-set function @classmethod
    # is defined.cls
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (3e6, 7.5e5)
    mesh_file = "benchmarks/schmeling_2008.msh"

    # Parameters to initialise level sets
    slope = 0
    intercept = 7e5
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [(slope, intercept), None]
    initialise_signed_distance = [
        partial(isd.isd_simple_curve, domain_dimensions[0], isd.straight_line),
        isd.isd_schmeling,
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    mantle = Mantle(density=3200)
    lithosphere = Lithosphere(density=3300)
    air = Air(density=0)
    materials = [mantle, air, lithosphere]
    reference_material = mantle

    # Physical parameters
    Ra, g = 1, 9.81

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Timestepping objects
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
