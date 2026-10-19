from mpi4py import MPI

from gadopt import *

# isort: off
import parameters as prms
from field_initialisation import initial_level_set, initial_temperature
from rheology import material_viscosity
from utility import generate_mesh
# isort: on


class AdaptiveSubduction(AdaptiveSimulation):
    def initialise(self) -> None:
        # Set up solvers
        self.set_up(initial=True)
        # Initialise transported fields (temperature and level set)
        initial_temperature(
            self.T,
            smoothing_params={
                "wavelength": prms.smoothing_wavelength,
                "bcs": self.temp_bcs,
            },
        )
        initial_level_set(self.psi)
        # Initialise Stokes fields (velocity, pressure, and free-surface height)
        log("Initial Stokes solve")
        self.stokes_solver.solve()

    def set_up(self, initial: bool = False) -> None:
        # Mesh
        self.mesh.cartesian = True
        self.boundary = get_boundary_ids(self.mesh)

        # Function spaces
        V = VectorFunctionSpace(self.mesh, "CG", 2)
        W = FunctionSpace(self.mesh, "CG", 1)
        if prms.free_surface:
            Z = MixedFunctionSpace([V, W, W])
        else:
            Z = MixedFunctionSpace([V, W])
        Q = FunctionSpace(self.mesh, "DG", 2)
        K = FunctionSpace(self.mesh, "DG", 2)

        # Functions
        self.stokes = Function(Z, name="Stokes")
        self.stokes.subfunctions[0].rename("Velocity")
        self.stokes.subfunctions[1].rename("Pressure")
        if prms.free_surface:
            self.stokes.subfunctions[2].rename("Free surface")
        u = split(self.stokes)[0]
        self.T = Function(Q, name="Temperature")
        self.psi = Function(K, name="Level set")

        # The Stokes function is checkpointed as a single mixed entry
        self.solutions = [self.stokes, self.T, self.psi]
        # The state carried across mesh adaptation and written as output
        state_fields = [*self.stokes.subfunctions, self.T, self.psi]

        if not initial:
            for field in state_fields:
                field.assign(self.fields[field.name()]["field"])

            self.fields.clear()

        self.fields = {
            field.name(): {"field": field, "output": Function(field, name=field.name())}
            for field in state_fields
        }

        # Rheology
        mu_material, rheol_expr = material_viscosity(self.mesh, u, self.T, self.psi)
        self.fields |= {
            field_name: {
                "field": Function(W, name=field_name),
                "output": Function(W, name=field_name),
                "expr": expr,
            }
            for field_name, expr in rheol_expr.items()
        }

        # Boundary conditions
        self.temp_bcs = {
            self.boundary.bottom: {"T": prms.temperature_scaling(prms.T_pot)},
            self.boundary.top: {"T": prms.temperature_scaling(prms.T_surf)},
        }
        stokes_bcs = {
            self.boundary.bottom: {"uy": 0.0},
            self.boundary.left: {"ux": 0.0},
            self.boundary.right: {"ux": 0.0},
        }
        if prms.free_surface:
            self.volume_multiplier = Constant(0.0)

        if prms.dimensionless:
            RaB_material = material_field(
                self.psi, [0.0, prms.Ra * prms.B], interface="arithmetic"
            )
            RaB_name = "Rayleigh number (compositional)"
            self.fields[RaB_name] = {
                "field": Function(W, name=RaB_name),
                "output": Function(W, name=RaB_name),
                "expr": RaB_material,
            }

            approximation_params = {
                "alpha": 1.0,
                "g": 1.0,
                "H": 0.0,
                "kappa": 1.0,
                "mu": mu_material,
                "Ra": prms.Ra,
                "RaB": RaB_material,
                "rho": 1.0,
                "delta_rho": 1.0,
                "T0": 0.0,
            }

            if prms.free_surface:
                RaFS = prms.Ra * prms.BFS
                stokes_bcs[self.boundary.top] = {
                    "free_surface": {
                        "RaFS": RaFS,
                        "exact_normal": True,
                        "volume_multiplier": self.volume_multiplier,
                    }
                }
            else:
                stokes_bcs[self.boundary.top] = {"uy": 0.0}
        else:
            delta_rho_material = material_field(
                self.psi,
                [0.0, prms.rho_weak_layer - prms.rho_mantle],
                interface="arithmetic",
            )
            rho_material = (prms.rho_mantle + delta_rho_material) * (
                1.0 - prms.alpha * (self.T - prms.T_surf)
            )
            rho_name = "Density"
            self.fields[rho_name] = {
                "field": Function(W, name=rho_name),
                "output": Function(W, name=rho_name),
                "expr": rho_material,
            }

            approximation_params = {
                "alpha": prms.alpha,
                "g": prms.g,
                "H": 0.0,
                "kappa": prms.kappa,
                "mu": mu_material,
                "Ra": 1.0,
                "RaB": 1.0,
                "rho": prms.rho_mantle,
                "delta_rho": delta_rho_material,
                "T0": prms.T_surf,
            }

            if prms.free_surface:
                stokes_bcs[self.boundary.top] = {
                    "free_surface": {
                        "delta_rho_fs": prms.rho_mantle - prms.rho_water,
                        "exact_normal": True,
                        "volume_multiplier": self.volume_multiplier,
                    }
                }
            else:
                stokes_bcs[self.boundary.top] = {"uy": 0.0}

        for field_name, field_specs in self.fields.items():
            field_specs["include_in_metric"] = field_name in prms.metric_fields
            field_specs["metric_mods"] = prms.metric_fields.get(field_name)
            field_specs["scale"] = prms.scales[field_name]

        if prms.free_surface:
            nullspace_args = {}
        else:
            stokes_nullspace = create_stokes_nullspace(Z)
            nullspace_args = {
                "nullspace": stokes_nullspace,
                "transpose_nullspace": stokes_nullspace,
            }

        # Adaptive time stepping
        self.tstep_adapt = TimestepAdaptor(
            self.time_step,
            u,
            V,
            target_cfl=0.6 * prms.subcycles,
            maximum_timestep=prms.output_frequency,
        )

        # Solvers
        approximation = BoussinesqApproximation(**approximation_params)

        self.stokes_solver = StokesSolver(
            self.stokes,
            approximation,
            self.T,
            dt=self.time_step,
            bcs=stokes_bcs,
            solver_parameters="direct",
            solver_parameters_extra={
                "snes_linesearch_type": "bisection",
                "snes_max_it": 300,
                "snes_rtol": 1e-5,
            },
            **nullspace_args,
        )

        self.energy_solver = EnergySolver(
            self.T,
            u,
            approximation,
            self.time_step,
            ImplicitMidpoint,
            bcs=self.temp_bcs,
        )

        epsilon = interface_thickness(K, min_cell_edge_length=True)
        adv_kwargs = {"u": u, "timestep": self.time_step, "subcycles": prms.subcycles}
        reini_kwargs = {"epsilon": epsilon, "frequency": 3}
        self.level_set_solver = LevelSetSolver(
            self.psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs
        )

        self.solvers = [self.level_set_solver, self.energy_solver, self.stokes_solver]

    def time_loop(self) -> None:
        for _ in range(prms.iterations):  # Time loop
            log(f"Time loop iteration #{self.step}")

            self.tstep_adapt.update_timestep()  # Update time step

            if prms.free_surface:
                u, _, eta = split(self.stokes)
                self.volume_multiplier.assign(
                    free_surface_volume_multiplier(u, eta, self.boundary.top)
                )

            for solver in self.solvers:  # Perform solves
                solver.solve()

            # Increment iteration count and time
            self.step += 1
            self.time.assign(self.time + self.time_step)

            # Write checkpoint and output
            if float(self.time) >= self.checkpoint_counter * prms.checkpoint_frequency:
                self.write_checkpoint()
            if float(self.time) >= self.output_counter * prms.output_frequency:
                self.write_output()

            # Check if simulation has completed
            if float(self.time) >= prms.time_end:
                log("Reached end of simulation")

                break


if __name__ == "__main__":
    match prms.mesh_generation:  # Mesh generation
        case "firedrake":
            mesh = RectangleMesh(
                *prms.mesh_elements, *prms.domain_dims, quadrilateral=False
            )
        case "gmsh":
            if MPI.COMM_WORLD.rank == 0:
                generate_mesh(prms.domain_dims, prms.mesh_layers)

            MPI.COMM_WORLD.barrier()

            mesh = Mesh("mesh.msh")

    simulation = AdaptiveSubduction(
        mesh=mesh,
        metric_parameters=prms.metric_parameters,
        time_end=prms.time_end,
        time_normalisation=prms.myr_to_seconds,
        time_step=prms.time_step,
        adapt_calls=prms.adapt_calls,
        initial_adapt_loops=prms.initial_adapt_loops,
    )
    simulation.run()
