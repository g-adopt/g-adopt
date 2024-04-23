from dataclasses import dataclass
from typing import ClassVar, Tuple

import firedrake as fd
import gmsh
import initial_signed_distance as isd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.io import loadmat

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    @staticmethod
    def viscosity(*args, **kwargs):
        return 1e21


@dataclass
class Lithosphere(ga.Material):
    visc_coeff: ClassVar[float] = 4.75e11
    stress_exponent: ClassVar[float] = 4.0
    visc_bounds: ClassVar[Tuple[float, float]] = (1e21, 1e25)

    @classmethod
    def viscosity(cls, velocity, *args, **kwargs):
        strain_rate = fd.sym(fd.grad(velocity))
        strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2 + 1e-99)

        return fd.min_value(
            fd.max_value(
                cls.visc_coeff * strain_rate_sec_inv ** (1 / cls.stress_exponent - 1),
                cls.visc_bounds[0],
            ),
            cls.visc_bounds[1],
        )


class Simulation:
    """Compositional benchmark.
    Schmalholz, S. M. (2011).
    A simple analytical solution for slab detachment.
    Earth and Planetary Science Letters, 304(1-2), 45-54.
    """

    name = "Schmalholz_2011"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (1e6, 6.6e5)
    domain_origin = (0, 0)
    mesh_file = "benchmarks/schmalholz_2011.msh"
    mesh_vert_res = 1.5e4
    mesh_fine_layer_min_x = 4.2e5
    mesh_fine_layer_width = 1.6e5
    mesh_fine_layer_hor_res = 8e3

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

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
    mantle = Mantle(density=3150)
    lithosphere = Lithosphere(density=3300)
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

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e11
    subcycles = 1
    dump_period = 5e5 * 365.25 * 8.64e4
    checkpoint_period = 5
    time_end = 25e6 * 365.25 * 8.64e4

    # Diagnostic objects
    diag_fields = {"normalised_time": [], "slab_necking": []}
    lithosphere_thickness = 8e4
    slab_length = 2.5e5
    slab_width = 8e4
    characteristic_time = (
        4
        * Lithosphere.visc_coeff
        / (lithosphere.density - mantle.density)
        / g
        / slab_length
    ) ** Lithosphere.stress_exponent

    @classmethod
    def generate_mesh(cls):
        gmsh.initialize()
        gmsh.model.add("mesh")

        point_1 = gmsh.model.geo.addPoint(*cls.domain_origin, 0, cls.mesh_vert_res)
        point_2 = gmsh.model.geo.addPoint(
            cls.domain_origin[0], cls.domain_dims[1], 0, cls.mesh_vert_res
        )

        line_1 = gmsh.model.geo.addLine(point_1, point_2)

        gmsh.model.geo.extrude(
            [(1, line_1)],
            cls.mesh_fine_layer_min_x,
            0,
            0,
            numElements=[21],
            recombine=True,
        )  # Horizontal resolution: 20 km

        num_layers = int(cls.mesh_fine_layer_width / cls.mesh_fine_layer_hor_res)
        gmsh.model.geo.extrude(
            [(1, line_1 + 1)],
            cls.mesh_fine_layer_width,
            0,
            0,
            numElements=[num_layers],
            recombine=True,
        )

        gmsh.model.geo.extrude(
            [(1, line_1 + 5)],
            cls.domain_dims[0] - cls.mesh_fine_layer_min_x - cls.mesh_fine_layer_width,
            0,
            0,
            numElements=[21],
            recombine=True,
        )  # Horizontal resolution: 20 km

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [line_1], tag=1)
        gmsh.model.addPhysicalGroup(1, [line_1 + 9], tag=2)
        gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [2, 6, 10]], tag=3)
        gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [3, 7, 11]], tag=4)

        gmsh.model.addPhysicalGroup(2, [5, 9, 13], tag=1)

        gmsh.model.mesh.generate(2)

        gmsh.write(cls.mesh_file)
        gmsh.finalize()

    @classmethod
    def initialise_temperature(cls, temperature):
        pass

    @classmethod
    def steady_state_condition(cls, velocity, velocity_old):
        pass

    @classmethod
    def diagnostics(cls, simu_time, geo_diag):
        epsilon = float(geo_diag.diag_vars["epsilon"])
        level_set = geo_diag.diag_vars["level_set"][0]
        level_set_data = level_set.dat.data_ro_with_halos
        coords_data = (
            fd.Function(
                fd.VectorFunctionSpace(level_set.ufl_domain(), level_set.ufl_element())
            )
            .interpolate(fd.SpatialCoordinate(level_set))
            .dat.data_ro_with_halos
        )

        mask_ls_outside = (
            (coords_data[:, 0] <= cls.domain_dims[0] / 2)
            & (coords_data[:, 1] < cls.domain_dims[1] - cls.lithosphere_thickness - 2e4)
            & (
                coords_data[:, 1]
                > cls.domain_dims[1] - cls.lithosphere_thickness - cls.slab_length
            )
            & (level_set_data < 0.5)
        )
        mask_ls_inside = (
            (coords_data[:, 0] <= cls.domain_dims[0] / 2)
            & (coords_data[:, 1] < cls.domain_dims[1] - cls.lithosphere_thickness - 2e4)
            & (
                coords_data[:, 1]
                > cls.domain_dims[1] - cls.lithosphere_thickness - cls.slab_length
            )
            & (level_set_data >= 0.5)
        )
        if mask_ls_outside.any():
            ind_outside = coords_data[mask_ls_outside, 0].argmax()
            hor_coord_outside = coords_data[mask_ls_outside, 0][ind_outside]
            if not mask_ls_outside.all():
                ver_coord_outside = coords_data[mask_ls_outside, 1][ind_outside]
                mask_ver_coord = (
                    abs(coords_data[mask_ls_inside, 1] - ver_coord_outside) < 1e3
                )
                if mask_ver_coord.any():
                    ind_inside = coords_data[mask_ls_inside, 0][mask_ver_coord].argmin()
                    hor_coord_inside = coords_data[mask_ls_inside, 0][mask_ver_coord][
                        ind_inside
                    ]

                    ls_outside = max(
                        1e-6,
                        min(1 - 1e-6, level_set_data[mask_ls_outside][ind_outside]),
                    )
                    sdls_outside = epsilon * np.log(ls_outside / (1 - ls_outside))

                    ls_inside = max(
                        1e-6,
                        min(
                            1 - 1e-6,
                            level_set_data[mask_ls_inside][mask_ver_coord][ind_inside],
                        ),
                    )
                    sdls_inside = epsilon * np.log(ls_inside / (1 - ls_inside))

                    ls_dist = sdls_inside / (sdls_inside - sdls_outside)
                    hor_coord_interface = (
                        ls_dist * hor_coord_outside + (1 - ls_dist) * hor_coord_inside
                    )
                    min_width = cls.domain_dims[0] - 2 * hor_coord_interface
                else:
                    min_width = cls.domain_dims[0] - 2 * hor_coord_outside
            else:
                min_width = cls.domain_dims[0] - 2 * hor_coord_outside
        else:
            min_width = np.inf

        min_width_global = level_set.comm.allreduce(min_width, MPI.MIN)

        cls.diag_fields["normalised_time"].append(simu_time / cls.characteristic_time)
        cls.diag_fields["slab_necking"].append(min_width_global / cls.slab_width)

    @classmethod
    def save_and_plot(cls):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(f"{cls.name.lower()}/output", diag_fields=cls.diag_fields)

            slab_necking_schmalholz = loadmat("data/DET_FREE_NEW_TOP_R100.mat")

            fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

            ax.grid()

            ax.set_xlabel("Normalised time")
            ax.set_ylabel("Slab necking")

            ax.plot(
                slab_necking_schmalholz["Time"][0] / cls.characteristic_time,
                slab_necking_schmalholz["Thickness"][0] / cls.slab_width * 1e3,
                label="Schmalholz (2011)",
            )

            ax.plot(
                cls.diag_fields["normalised_time"],
                cls.diag_fields["slab_necking"],
                label="Conservative level set",
            )

            ax.legend()

            fig.savefig(
                f"{cls.name}/slab_necking.pdf".lower(), dpi=300, bbox_inches="tight"
            )
