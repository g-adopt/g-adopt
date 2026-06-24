from collections.abc import Callable
from pathlib import Path
from shutil import rmtree

from animate.adapt import adapt
from animate.metric import RiemannianMetric
from mpi4py import MPI

from gadopt import *

import parameters as prms
from field_initialisation import initial_level_set, initial_temperature
from rheology import material_viscosity
from utility import function_name, generate_mesh


class AdaptiveSimulation:
    def __init__(self):
        match prms.mesh_generation:  # Mesh generation
            case "firedrake":
                self.mesh = RectangleMesh(
                    *prms.mesh_elements, *prms.domain_dims, quadrilateral=False
                )
            case "gmsh":
                if MPI.COMM_WORLD.rank == 0:
                    generate_mesh(prms.domain_dims, prms.mesh_layers)

                self.mesh = Mesh("mesh.msh")

        # Time-stepping objects
        self.time = Constant(0.0)  # Initial time
        self.time_step = Constant(prms.time_step)  # Initial time step
        self.step = 0  # A counter to keep track of the simulation time-loop iterations
        self.checkpoint_counter = 0  # A counter to keep track of checkpoint files
        self.output_counter = 0  # A counter to keep track of output files

        self.initialise()  # Initialise solutions on the initial mesh

        for iteration in range(prms.initial_adapt_loops):
            # Adapt original mesh based on initial fields
            log(f"Initial adapt iteration #{iteration}")
            self.adapt_mesh(initial=True)

            self.initialise()  # Initialise solutions on the adapted mesh

        # Write initial checkpoint and output
        self.write_checkpoint(initial=True)
        self.write_output(initial=True)

    def adapt_mesh(self, initial: bool = False) -> None:
        def add_metric(field: Function):
            # Firedrake function for a metric over a mesh where a field lives
            metric = RiemannianMetric(M, name=f"Metric ({function_name(field)})")
            metric.set_parameters(prms.metric_parameters)  # Set metric parameters
            metric.compute_hessian(field)  # Field Hessian
            metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)

            metric_magnitude.interpolate(sqrt(inner(metric, metric)))  # Frobenius norm
            metric_magnitude_max = MPI.COMM_WORLD.allreduce(
                metric_magnitude.dat.data_ro.max(), MPI.MAX
            )
            # Scale metric by the maximum value of its Frobenius norm across the mesh
            metric.assign(metric / metric_magnitude_max)

            metrics.append(metric)

        for field_specs in self.fields.values():
            if field_specs["include_in_metric"] and "expr" in field_specs:
                field_specs["field"].interpolate(field_specs["expr"])

        for call_count in range(prms.adapt_calls):
            log(f"Adapt call #{call_count}")

            M = TensorFunctionSpace(self.mesh, "CG", 1)
            N = FunctionSpace(self.mesh, "CG", 1)
            metric_magnitude = Function(N)

            metrics = []
            for field_specs in self.fields.values():
                if not field_specs["include_in_metric"]:
                    continue

                if isinstance(field_specs["field"].ufl_element(), VectorElement):
                    for dim in range(field_specs["field"].ufl_shape[0]):
                        add_metric(field_specs["field"][dim])
                else:
                    add_metric(field_specs["field"])

            overall_metric = metrics[0].copy(deepcopy=True)  # Overall metric
            overall_metric.rename("Metric (overall)")
            # Minimum in all directions across all metrics
            overall_metric.intersect(*metrics[1:])
            # Ensure boundedness (symmetric positive-definite)
            overall_metric.enforce_spd()
            # Rescale metric to achieve the desired target complexity
            overall_metric.normalise()

            # Generate new mesh based on overall metric
            self.mesh = adapt(self.mesh, overall_metric)
            # Ensure boundary coordinates are not exceeded
            for i in range(self.mesh.geometric_dimension):
                self.mesh.coordinates.dat.data[:, i].clip(
                    0.0, prms.domain_dims[i], out=self.mesh.coordinates.dat.data[:, i]
                )

            if initial:
                break

            self.interpolate_fields()

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

    def interpolate_fields(self) -> None:
        for field_specs in self.fields.values():
            if not field_specs["include_in_metric"] and "expr" in field_specs:
                continue

            field = field_specs.pop("field")
            space = FunctionSpace(self.mesh, field.ufl_element())
            new_field = Function(space, name=field.name())
            new_field.interpolate(field)
            field_specs["field"] = new_field

    def run(self) -> None:
        while True:  # Mesh adaptivity loop
            self.time_loop()  # Run the time loop

            # Exit loop after reaching target time
            if float(self.time) >= prms.time_end:
                break

            self.adapt_mesh()  # Adapt mesh based on current fields
            self.set_up()  # Set simulation objects on the new mesh

    def set_up(self, initial: bool = False) -> None:
        # Mesh
        self.mesh.cartesian = True
        boundary = get_boundary_ids(self.mesh)

        # Function spaces
        V = VectorFunctionSpace(self.mesh, "CG", 2)
        W = FunctionSpace(self.mesh, "CG", 1)
        W_equispaced = FunctionSpace(self.mesh, "CG", 1, variant="equispaced")
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

        self.solutions = [*self.stokes.subfunctions, self.T, self.psi]

        if not initial:
            for solution in self.solutions:
                solution.assign(self.fields[solution.name()]["field"])

            self.fields.clear()

        self.fields = {
            field.name(): {
                "field": field,
                "output": Function(
                    FunctionSpace(self.mesh, field.ufl_element(), variant="equispaced"),
                    name=field.name(),
                ),
            }
            for field in self.solutions
        }

        # Rheology
        mu_material, rheol_expr = material_viscosity(self.mesh, u, self.T, self.psi)
        self.fields |= {
            field_name: {
                "field": Function(W, name=f"{field_name}"),
                "output": Function(W_equispaced, name=field_name),
                "expr": expr,
            }
            for field_name, expr in rheol_expr.items()
        }

        # Boundary conditions
        self.temp_bcs = {
            boundary.bottom: {"T": prms.temperature_scaling(prms.T_pot)},
            boundary.top: {"T": prms.temperature_scaling(prms.T_surf)},
        }
        stokes_bcs = {
            boundary.bottom: {"uy": 0.0},
            boundary.left: {"ux": 0.0},
            boundary.right: {"ux": 0.0},
        }

        if prms.dimensionless:
            RaB_material = material_field(
                self.psi, [0.0, prms.Ra * prms.B], interface="arithmetic"
            )
            RaB_name = "Rayleigh number (compositional)"
            self.fields[RaB_name] = {
                "field": Function(W, name=RaB_name),
                "output": Function(W_equispaced, name=RaB_name),
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
                RaFS_material = material_field(
                    self.psi, [prms.Ra * B for B in prms.BFS], interface="arithmetic"
                )
                stokes_bcs[boundary.top] = {"free_surface": {"RaFS": RaFS_material}}
            else:
                stokes_bcs[boundary.top] = {"uy": 0.0}
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
                "output": Function(W_equispaced, name=rho_name),
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
                delta_rho_fs = prms.rho_mantle + delta_rho_material - prms.rho_water
                stokes_bcs[boundary.top] = {
                    "free_surface": {"delta_rho_fs": delta_rho_fs}
                }
            else:
                stokes_bcs[boundary.top] = {"uy": 0.0}

        for field_name, field_specs in self.fields.items():
            field_specs["include_in_metric"] = field_name in prms.metric_fields
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
                "snes_max_it": 200,
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

    def write_checkpoint(self, initial: bool = False) -> None:
        if initial and MPI.COMM_WORLD.rank == 0:
            check_dir = Path("checkpoints").resolve()
            if check_dir.is_dir():
                rmtree(check_dir)
            check_dir.mkdir()

        with CheckpointFile(
            f"checkpoints/checkpoint_{self.checkpoint_counter}.h5", "w"
        ) as checkpoint_file:
            checkpoint_file.save_mesh(self.mesh)
            checkpoint_file.save_function(self.stokes, name="Stokes")
            checkpoint_file.save_function(self.T, name="Temperature")
            checkpoint_file.save_function(self.psi, name="Level set")

        self.checkpoint_counter += 1

    def write_output(self, initial: bool = False) -> None:
        if initial:
            self.output_file = VTKFile("output.pvd", adaptive=True)

        output_fields = []
        for field_specs in self.fields.values():
            scale = field_specs["scale"] if prms.dimensionless else 1.0
            field_data = field_specs.get("expr", field_specs["field"])

            if isinstance(scale, Callable):
                field_specs["output"].interpolate(scale(field_data))
            else:
                field_specs["output"].interpolate(scale * field_data)

            output_fields.append(field_specs["output"])

        self.output_file.write(
            *output_fields, time=float(self.time) / prms.myr_to_seconds
        )

        self.output_counter += 1


simulation = AdaptiveSimulation()
simulation.run()
