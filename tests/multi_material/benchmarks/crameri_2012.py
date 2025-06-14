from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga
from gadopt.level_set_tools import min_max_height


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
        return 1e18


class Simulation:
    """Compositional benchmark.
    Crameri, F., Schmeling, H., Golabek, G. J., Duretz, T., Orendt, R.,
    Buiter, S. J. H., ... & Tackley, P. J. (2012).
    A comparison of numerical surface topography calculations in geodynamic modelling:
    an evaluation of the 'sticky air' method.
    Geophysical Journal International, 189(1), 38-54.
    """

    name = "Crameri_2012"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (2.8e6, 8e5)
    domain_origin = (0, 0)
    mesh_file = "benchmarks/crameri_2012.msh"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise surface level set
    interface_coords_x = np.linspace(0.0, domain_dims[0], int(domain_dims[0] / 1e3) + 1)
    callable_args = (
        surface_deflection := 7e3,
        surface_perturbation_wavelength := domain_dims[0],
        surface_coord_y := 7e5,
    )
    surface_signed_distance_kwargs = {
        "interface_geometry": "curve",
        "interface_callable": "cosine",
        "interface_args": (interface_coords_x, *callable_args),
    }
    # Parameters to initialise LAB level set
    interface_coords_x = np.array([0.0, domain_dims[0]])
    callable_args = (lab_slope := 0, lab_coord_y := 6e5)
    lab_signed_distance_kwargs = {
        "interface_geometry": "curve",
        "interface_callable": "line",
        "interface_args": (interface_coords_x, *callable_args),
    }
    # The following list must be ordered such that, unpacking from the end, each dictionary
    # contains the keyword arguments required to initialise the signed-distance array
    # corresponding to the interface between a given material and the remainder of the
    # numerical domain (all previous materials excluded). By convention, the material thus
    # isolated occupies the positive side of the signed-distance array.
    signed_distance_kwargs_list = [
        lab_signed_distance_kwargs,
        surface_signed_distance_kwargs,
    ]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    mantle = Mantle(density=3300)
    lithosphere = Lithosphere(density=3300)
    air = Air(density=0)
    materials = [mantle, lithosphere, air]
    reference_material = mantle

    # Approximation parameters
    dimensional = True
    Ra, g = 1, 10

    # Boundary conditions
    temp_bcs = {}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e10
    subcycles = 1
    dump_period = 2e3 * 365.25 * 8.64e4
    checkpoint_period = 5
    time_end = 1e5 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {
        "output_time": [],
        "max_topography": [],
        "max_topography_analytical": [],
    }
    relaxation_rate = -0.2139e-11

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        max_topo = min_max_height(
            diag_vars["level_set"][1], diag_vars["epsilon"], side=0, mode="max"
        )
        max_topography_analytical = (
            cls.surface_deflection / 1e3 * np.exp(cls.relaxation_rate * simu_time)
        )

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
        cls.diag_fields["max_topography"].append((max_topo - cls.surface_coord_y) / 1e3)
        cls.diag_fields["max_topography_analytical"].append(max_topography_analytical)

        if MPI.COMM_WORLD.rank == 0:
            np.savez(
                f"{cls.name.lower()}/output_{Simulation.restart_from_checkpoint}_check",
                diag_fields=cls.diag_fields,
            )

    @classmethod
    def plot_diagnostics(cls):
        if MPI.COMM_WORLD.rank == 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.grid()

            ax.set_xlabel("Time (kyr)")
            ax.set_ylabel("Maximum topography (km)")

            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["max_topography_analytical"],
                label="Analytical (Crameri et al., 2012)",
            )
            ax.plot(
                cls.diag_fields["output_time"],
                cls.diag_fields["max_topography"],
                label="Conservative level set",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/maximum_topography.pdf".lower(),
                dpi=300,
                bbox_inches="tight",
            )
