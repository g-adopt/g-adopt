from dataclasses import dataclass
from typing import ClassVar, Tuple

import firedrake as fd
import initial_signed_distance as isd

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    def viscosity(self, *args, **kwargs):
        return 1e21


@dataclass
class Lithosphere(ga.Material):
    visc_coeff: ClassVar[float] = 4.75e11
    stress_exponent: ClassVar[float] = 4.0
    visc_bounds: ClassVar[Tuple[float, float]] = (1e21, 1e25)

    @classmethod
    def viscosity(cls, *args, **kwargs):
        strain_rate = fd.sym(fd.grad(kwargs["velocity"]))
        strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2 + 1e-99)

        return fd.min_value(
            fd.max_value(
                cls.visc_coeff * strain_rate_sec_inv ** (1 / cls.stress_exponent - 1),
                cls.visc_bounds[0],
            ),
            cls.visc_bounds[1],
        )


class Simulation:
    name = "Schmalholz_2011"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (1e6, 6.6e5)
    mesh_elements = (192, 128)

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
    mantle = Mantle(density=3150.0)
    lithosphere = Lithosphere(density=3300.0)
    materials = [mantle, lithosphere]
    reference_material = mantle

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
        / (lithosphere.density - mantle.density)
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
