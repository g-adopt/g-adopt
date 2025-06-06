from dataclasses import dataclass
from typing import ClassVar

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import gadopt as ga


@dataclass
class Mantle(ga.Material):
    visc_contrast_temp: ClassVar[float] = 1e5
    visc_contrast_pres: ClassVar[float] = 1e1
    visc_eff_high_stress: ClassVar[float] = 1e-3
    yield_stress: ClassVar[float] = 1

    @classmethod
    def viscosity(cls, velocity, temperature, *args, **kwargs):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        strain_rate = fd.sym(fd.grad(velocity))

        visc_lin = fd.exp(
            -fd.ln(cls.visc_contrast_temp) * temperature
            + fd.ln(cls.visc_contrast_pres) * (1 - mesh_coords[1])
        )
        visc_plast = cls.visc_eff_high_stress + cls.yield_stress / fd.sqrt(
            fd.inner(strain_rate, strain_rate) + 1e-99
        )

        return 2 / (1 / visc_lin + 1 / visc_plast)


class Simulation:
    """Thermal benchmark.
    Tosi, N., Stein, C., Noack, L., Hüttig, C., Maierova, P., Samuel, H., ...
    & Tackley, P. J. (2015).
    A community benchmark for viscoplastic thermal convection in a 2‐D square box.
    Geochemistry, Geophysics, Geosystems, 16(7), 2175-2196.
    """

    name = "Tosi_2015"

    restart_from_checkpoint = 0

    # Mesh resolution should be sufficient to capture eventual small-scale dynamics
    # in the neighbourhood of material interfaces tracked by the level-set approach.
    # Insufficient mesh refinement can lead to unwanted motion of material interfaces.
    domain_dims = (1, 1)
    domain_origin = (0, 0)
    mesh_elements = (64, 64)

    # Degree of the function space on which the level-set function is defined.
    level_set_func_space_deg = 2

    # Parameters to initialise level set
    interface_coords_x = np.array([0.0, domain_dims[0]])
    callable_args = (interface_slope := 0, interface_coord_y := 0.5)
    signed_distance_kwargs = {
        "interface_geometry": "curve",
        "interface_callable": "line",
        "interface_args": (interface_coords_x, *callable_args),
    }
    # The following list must be ordered such that, unpacking from the end, each dictionary
    # contains the keyword arguments required to initialise the signed-distance array
    # corresponding to the interface between a given material and the remainder of the
    # numerical domain (all previous materials excluded). By convention, the material thus
    # isolated occupies the positive side of the signed-distance array.
    signed_distance_kwargs_list = [signed_distance_kwargs]

    # Material ordering must follow the logic implemented in the above two lists. In
    # other words, the last material in the below list corresponds to the portion of
    # the numerical domain entirely isolated by the level set initialised using the
    # first pair of arguments (unpacking from the end) in the above two lists.
    # Consequently, the first material in the below list occupies the negative side of
    # the level set resulting from the last pair of arguments above.
    top_material = Mantle(B=0)
    bottom_material = Mantle(B=0)
    materials = [bottom_material, top_material]
    reference_material = None

    # Approximation parameters
    dimensional = False
    Ra, g = 1e2, 1

    # Parameters to initialise temperature
    A = 0.01

    # Boundary conditions
    temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
    stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

    # Stokes solver options
    stokes_nullspace_args = {}
    stokes_solver_params = None

    # Timestepping objects
    initial_timestep = 1e-6
    subcycles = 1
    dump_period = 1e-3
    checkpoint_period = 5
    steady_state_threshold = 1e-6
    time_end = None

    # Diagnostic objects
    diag_fields = {
        "output_time": [],
        "avg_temperature": [],
        "nusselt_top": [],
        "nusselt_bottom": [],
        "rms_velocity": [],
        "min_visc": [],
        "max_visc": [],
    }

    @classmethod
    def initialise_temperature(cls, temperature):
        mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

        temperature.interpolate(
            1
            - mesh_coords[1]
            + cls.A * fd.cos(fd.pi * mesh_coords[0]) * fd.sin(fd.pi * mesh_coords[1])
        )

    @classmethod
    def steady_state_condition(cls, stokes_solver):
        velocity = stokes_solver.solution.subfunctions[0]

        return fd.norm(velocity - cls.velocity_old) < cls.steady_state_threshold

    @classmethod
    def diagnostics(cls, simu_time, geo_diag, diag_vars):
        cls.diag_fields["output_time"].append(simu_time)
        cls.diag_fields["avg_temperature"].append(geo_diag.T_avg())
        cls.diag_fields["nusselt_top"].append(geo_diag.Nu_top())
        cls.diag_fields["nusselt_bottom"].append(geo_diag.Nu_bottom())
        cls.diag_fields["rms_velocity"].append(geo_diag.u_rms())
        cls.diag_fields["min_visc"].append(
            geo_diag.T.comm.allreduce(diag_vars["viscosity"].dat.data.min(), MPI.MIN)
        )
        cls.diag_fields["max_visc"].append(
            geo_diag.T.comm.allreduce(diag_vars["viscosity"].dat.data.max(), MPI.MAX)
        )

        if MPI.COMM_WORLD.rank == 0:
            np.savez(
                f"{cls.name.lower()}/output_{Simulation.restart_from_checkpoint}_check",
                diag_fields=cls.diag_fields,
            )

    @classmethod
    def plot_diagnostics(cls):
        if MPI.COMM_WORLD.rank == 0:
            fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

            for axis in ax.flatten():
                axis.grid()
                axis.set_xlabel("Time (non-dimensional)")

            ax[0, 0].set_ylabel("Average temperature (non-dimensional)")
            ax[0, 1].set_ylabel("Top Nusselt number (non-dimensional)")
            ax[0, 2].set_ylabel("Bottom Nusselt number (non-dimensional)")
            ax[1, 0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1, 1].set_ylabel("Minimum viscosity (non-dimensional)")
            ax[1, 2].set_ylabel("Maximum viscosity (non-dimensional)")

            ax[0, 0].plot(
                cls.diag_fields["output_time"], cls.diag_fields["avg_temperature"]
            )
            ax[0, 1].plot(
                cls.diag_fields["output_time"], cls.diag_fields["nusselt_top"]
            )
            ax[0, 2].plot(
                cls.diag_fields["output_time"], cls.diag_fields["nusselt_bottom"]
            )
            ax[1, 0].plot(
                cls.diag_fields["output_time"], cls.diag_fields["rms_velocity"]
            )
            ax[1, 1].plot(cls.diag_fields["output_time"], cls.diag_fields["min_visc"])
            ax[1, 2].plot(cls.diag_fields["output_time"], cls.diag_fields["max_visc"])

            fig.savefig(
                f"{cls.name}/diagnostics.pdf".lower(), dpi=300, bbox_inches="tight"
            )
