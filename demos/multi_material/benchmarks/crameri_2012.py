from dataclasses import dataclass
from functools import partial

import firedrake as fd
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

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
        return 1e18


class Simulation:
    name = "Crameri_2012"

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dimensions = (2.8e6, 8e5)
    domain_origin = (0, 0)
    mesh_file = "benchmarks/crameri_2012.msh"

    # Parameters to initialise level sets
    bottom_material_interface_y = 6e5
    bottom_interface_slope = 0
    top_material_interface_y = 7e5
    top_interface_deflection = 7e3
    # The following two lists must be ordered such that, unpacking from the end, each
    # pair of arguments enables initialising a level set whose 0-contour corresponds to
    # the entire interface between a given material and the remainder of the numerical
    # domain. By convention, the material thereby isolated occupies the positive side
    # of the signed-distance level set.
    isd_params = [
        (bottom_interface_slope, bottom_material_interface_y),
        (top_interface_deflection, domain_dimensions[0], top_material_interface_y),
    ]
    initialise_signed_distance = [
        partial(
            isd.isd_simple_curve,
            domain_origin[0],
            domain_dimensions[0],
            isd.straight_line,
        ),
        partial(
            isd.isd_simple_curve,
            domain_origin[0],
            domain_dimensions[0],
            isd.cosine_curve,
        ),
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

    # Physical parameters
    Ra, g = 1, 10

    # Boundary conditions
    temp_bcs = None
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"ux": 0, "uy": 0}, 4: {"uy": 0}}

    # Stokes nullspace
    stokes_nullspace_args = {}

    # Timestepping objects
    dt = 1e10
    subcycles = 1
    time_end = 1e5 * 365.25 * 8.64e4
    dump_period = 2e3 * 365.25 * 8.64e4

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
    def diagnostics(cls, simu_time, variables):
        max_topography_analytical = (
            cls.top_interface_deflection / 1e3 * np.exp(cls.relaxation_rate * simu_time)
        )

        epsilon = float(variables["epsilon"])
        level_set = variables["level_set"][1]
        level_set_data = level_set.dat.data_ro_with_halos
        coords_data = (
            fd.Function(
                fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
            )
            .interpolate(fd.SpatialCoordinate(level_set))
            .dat.data_ro_with_halos
        )

        mask_ls = level_set_data <= 0.5
        if mask_ls.any():
            ind_lower_bound = coords_data[mask_ls, 1].argmax()
            max_topo_lower_bound = coords_data[mask_ls, 1][ind_lower_bound]
            if not mask_ls.all():
                hor_coord_lower_bound = coords_data[mask_ls, 0][ind_lower_bound]
                mask_hor_coord = (
                    abs(coords_data[~mask_ls, 0] - hor_coord_lower_bound) < 1e3
                )
                if mask_hor_coord.any():
                    ind_upper_bound = coords_data[~mask_ls, 1][mask_hor_coord].argmin()
                    max_topo_upper_bound = coords_data[~mask_ls, 1][mask_hor_coord][
                        ind_upper_bound
                    ]

                    ls_lower_bound = level_set_data[mask_ls][ind_lower_bound]
                    sdls_lower_bound = epsilon * np.log(
                        ls_lower_bound / (1 - ls_lower_bound)
                    )

                    ls_upper_bound = level_set_data[~mask_ls][mask_hor_coord][
                        ind_upper_bound
                    ]
                    sdls_upper_bound = epsilon * np.log(
                        ls_upper_bound / (1 - ls_upper_bound)
                    )

                    ls_dist = sdls_upper_bound / (sdls_upper_bound - sdls_lower_bound)
                    max_topo = (
                        ls_dist * max_topo_lower_bound
                        + (1 - ls_dist) * max_topo_upper_bound
                    )
                else:
                    max_topo = max_topo_lower_bound
            else:
                max_topo = max_topo_lower_bound
        else:
            max_topo = -np.inf

        max_topo_global = level_set.comm.allreduce(max_topo, MPI.MAX)

        cls.diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
        cls.diag_fields["max_topography"].append(
            (max_topo_global - cls.top_material_interface_y) / 1e3
        )
        cls.diag_fields["max_topography_analytical"].append(max_topography_analytical)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

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
