from animate.adapt import adapt
from animate.metric import RiemannianMetric
from gadopt import *
from mpi4py import MPI

import parameters as prms
from field_initialisation import initial_level_set, initial_temperature
from rheology import material_viscosity
from utility import function_name, generate_mesh, write_output


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
        self.time_now = Constant(0.0)  # Initial time
        self.time_step = Constant(prms.time_step)  # Initial time step
        self.step = 0  # A counter to keep track of the simulation time-loop iterations

        self.mesh_fields = prms.mesh_fields  # Fields involved in mesh adaptivity

        self.initialise()  # Initialise solutions on the initial mesh

        for iteration in range(prms.initial_adapt_loops):
            # Adapt original mesh based on initial fields
            PETSc.Sys.Print(f"Initial adapt iteration #{iteration}")
            self.adapt_mesh(initial=True)

            self.initialise()  # Initialise solutions on the adapted mesh

        # Write initial output
        self.output_file = VTKFile("output_adaptive.pvd", adaptive=True)
        write_output(
            self.output_file,
            float(self.time_now) * prms.time_scale / prms.myr_to_seconds,
            *self.stokes.subfunctions,
            self.T,
            self.psi,
            field_expressions=self.field_expr,
        )

    def adapt_mesh(self, initial: bool = False) -> None:
        def add_metric_field(field: Function, scaling: float) -> None:
            nonlocal metric_fields

            if isinstance(field.ufl_element(), VectorElement):
                for dim in range(field.ufl_shape[0]):
                    metric_fields.append([field[dim], scaling])
            else:
                metric_fields.append([field, scaling])

        for call_count in range(prms.adapt_calls):
            PETSc.Sys.Print(f"Adapt call #{call_count}")

            M = TensorFunctionSpace(self.mesh, "CG", 1)

            metric_fields = []
            for field_specs in self.mesh_fields.values():
                field = field_specs["field"]

                if isinstance(field_specs["add_to_metric"], list):
                    for dim, (add_to_metric, scaling) in enumerate(
                        zip(field_specs["add_to_metric"], field_specs["scaling"])
                    ):
                        if add_to_metric:
                            add_metric_field(field.subfunctions[dim], scaling)
                elif field_specs["add_to_metric"]:
                    add_metric_field(field, field_specs["scaling"])

            metrics = []
            for field, scaling in metric_fields:
                # Firedrake function for a metric over a mesh where a field lives
                metric = RiemannianMetric(M, name=f"Metric ({function_name(field)})")
                metric.set_parameters(prms.metric_parameters)  # Set metric parameters
                metric.compute_hessian(field)  # Field Hessian
                metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)
                metric.assign(metric * scaling)
                metrics.append(metric)

            overall_metric = metrics[0].copy(deepcopy=True)  # Overall metric
            overall_metric.rename("Metric (overall)")
            # Minimum in all directions across all metrics
            overall_metric.intersect(*metrics[1:])
            overall_metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)
            overall_metric.normalise()  # Rescale metric to achieve the desired target complexity

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
        initial_temperature(self.T)
        initial_level_set(self.psi)
        # Initialise Stokes fields (velocity, pressure, and free-surface height)
        PETSc.Sys.Print("Initial Stokes solve")
        self.stokes_solver.solve()

    def interpolate_fields(self) -> None:
        new_fields = []
        for field_specs in self.mesh_fields.values():
            field = field_specs["field"]
            field_element = field.ufl_element()

            if isinstance(field_element, MixedElement):
                spaces = [
                    FunctionSpace(self.mesh, element)
                    for element in field_element.sub_elements
                ]
                mixed_space = MixedFunctionSpace(spaces)

                new_field = Function(mixed_space, name=field.name())
                for sub_field, new_sub_field in zip(
                    field.subfunctions, new_field.subfunctions
                ):
                    new_sub_field.rename(sub_field.name())
                new_fields.append(new_field)
            else:
                space = FunctionSpace(self.mesh, field_element)
                new_fields.append(Function(space, name=field.name()))

        for (field_name, field_specs), new_field in zip(
            self.mesh_fields.items(), new_fields
        ):
            field = field_specs.pop("field")

            if isinstance(field.ufl_element(), MixedElement):
                for sub_field, new_sub_field in zip(
                    field.subfunctions, new_field.subfunctions
                ):
                    new_sub_field.interpolate(sub_field)
            else:
                new_field.interpolate(field)

            field_specs["field"] = new_field
            self.mesh_fields[field_name] = field_specs

    def run(self) -> None:
        while True:  # Mesh adaptivity loop
            self.time_loop()  # Run the time loop

            # Exit loop after reaching target time
            if float(self.time_now) >= prms.time_end:
                break

            self.adapt_mesh()  # Adapt mesh based on current fields
            self.set_up()  # Set simulation objects on the new mesh

    def set_up(self, initial: bool = False) -> None:
        # Mesh
        self.mesh.cartesian = True
        boundary = get_boundary_ids(self.mesh)

        if initial:
            # Function spaces
            V = VectorFunctionSpace(self.mesh, "CG", 2)
            W = FunctionSpace(self.mesh, "CG", 1)
            if prms.free_surface:
                Z = MixedFunctionSpace([V, W, W])
            else:
                Z = MixedFunctionSpace([V, W])
            Q = FunctionSpace(self.mesh, "DG", 2, variant="equispaced")
            K = FunctionSpace(self.mesh, "DG", 2, variant="equispaced")

            # Functions
            self.stokes = Function(Z, name="Stokes")
            u_func, p_func = self.stokes.subfunctions[:2]
            u_func.rename("Velocity")
            p_func.rename("Pressure")
            if prms.free_surface:
                eta_func = self.stokes.subfunctions[2]
                eta_func.rename("Free surface")
            self.T = Function(Q, name="Temperature")
            self.psi = Function(K, name="Level set")
        else:
            self.stokes = self.mesh_fields["Stokes"]["field"]
            self.T = self.mesh_fields["Temperature"]["field"]
            self.psi = self.mesh_fields["Level set"]["field"]
            Z = self.stokes.function_space()
            V, W = Z.subspaces[:2]
            K = self.psi.function_space()

        u = split(self.stokes)[0]

        # Rheology
        mu_material, rheol_expr = material_viscosity(self.mesh, u, self.T, self.psi)

        self.field_expr = {}
        for name, expr in rheol_expr.items():
            self.field_expr[Function(W, name=name)] = expr

        # Boundary conditions
        temp_bcs = {
            boundary.bottom: {"T": prms.temperature_scaling(prms.T_pot)},
            boundary.top: {"T": prms.temperature_scaling(prms.T_surf)},
        }
        stokes_bcs = {
            boundary.bottom: {"uy": 0.0},
            boundary.left: {"ux": 0.0},
            boundary.right: {"ux": 0.0},
        }

        if prms.dimensionless:
            compositional_rayleigh = Function(W, name="Rayleigh number (compositional)")
            RaB_material = material_field(
                self.psi, [0.0, prms.Ra * prms.B], interface="sharp"
            )
            self.field_expr[compositional_rayleigh] = RaB_material

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
                    self.psi, [prms.Ra * B for B in prms.BFS], interface="sharp"
                )
                stokes_bcs[boundary.top] = {"free_surface": {"RaFS": RaFS_material}}
            else:
                stokes_bcs[boundary.top] = {"uy": 0.0}
        else:
            density = Function(W, name="Density")
            delta_rho_material = material_field(
                self.psi,
                [0.0, prms.rho_weak_layer - prms.rho_mantle],
                interface="sharp",
            )
            rho_material = (prms.rho_mantle + delta_rho_material) * (
                1.0 - prms.alpha * (self.T - prms.T_surf)
            )
            self.field_expr[density] = rho_material

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
            target_cfl=0.55 * prms.subcycles,
            maximum_timestep=prms.myr_to_seconds / prms.time_scale,
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
            self.T, u, approximation, self.time_step, ImplicitMidpoint, bcs=temp_bcs
        )

        epsilon = interface_thickness(K, min_cell_edge_length=True)
        epsilon = MPI.COMM_WORLD.allreduce(epsilon.dat.data_ro.min(), MPI.MIN)
        adv_kwargs = {"u": u, "timestep": self.time_step, "subcycles": prms.subcycles}
        reini_kwargs = {"epsilon": epsilon}
        self.level_set_solver = LevelSetSolver(
            self.psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs
        )

        self.solvers = [self.level_set_solver, self.energy_solver, self.stokes_solver]

        for field in [self.stokes, self.T, self.psi]:
            self.mesh_fields[field.name()]["field"] = field

    def time_loop(self) -> None:
        for _ in range(prms.iterations):  # Time loop
            PETSc.Sys.Print(f"Time loop iteration #{self.step}")

            self.tstep_adapt.update_timestep()  # Update time step

            for solver in self.solvers:  # Perform solves
                solver.solve()

            # Increment iteration count and time
            self.step += 1
            self.time_now.assign(self.time_now + self.time_step)

            # Write output
            if self.step % prms.output_frequency == 0:
                write_output(
                    self.output_file,
                    float(self.time_now) * prms.time_scale / prms.myr_to_seconds,
                    *self.stokes.subfunctions,
                    self.T,
                    self.psi,
                    field_expressions=self.field_expr,
                )

            # Check if simulation has completed
            if float(self.time_now) >= prms.time_end:
                # Checkpoint solution fields to disk
                with CheckpointFile("final_state.h5", "w") as final_checkpoint:
                    final_checkpoint.save_mesh(self.mesh)
                    final_checkpoint.save_function(self.stokes, name="self.stokes")
                    final_checkpoint.save_function(self.T, name="Temperature")
                    final_checkpoint.save_function(self.psi, name="Level set")

                log("Reached end of simulation -- exiting time-self.step loop")

                break


simulation = AdaptiveSimulation()
simulation.run()
