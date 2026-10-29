from numpy import inf

from gadopt import *


class AdaptiveBaseCase(AdaptiveSimulation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.checkpoint_frequency = 1
        self.output_frequency = 1

        self.plog = ParameterLog("params.log", self.mesh)
        self.plog.log_str(
            "timestep time dt maxchange "
            "u_rms u_rms_surf ux_max nu_top nu_base energy avg_t"
        )

        self.steady_state_tolerance = 1e-9

    def initialise(self) -> None:
        # Set up solvers
        self.set_up(initial=True)
        # Initialise transported fields (temperature)
        x, y = SpatialCoordinate(mesh)
        self.T.interpolate(1.0 - y + 0.05 * cos(pi * x) * sin(pi * y))
        # Initialise Stokes fields (velocity and pressure)
        log("Initial Stokes solve")
        self.stokes_solver.solve()

    def set_up(self, initial: bool = False) -> None:
        # Mesh
        self.mesh.cartesian = True
        self.boundary = get_boundary_ids(self.mesh)

        # Function spaces
        V = VectorFunctionSpace(self.mesh, "CG", 2)
        W = FunctionSpace(self.mesh, "CG", 1)
        Q = FunctionSpace(self.mesh, "CG", 1)
        Z = MixedFunctionSpace([V, W])

        log(f"Number of Velocity DOF: {V.dim()}")
        log(f"Number of Pressure DOF: {W.dim()}")
        log(f"Number of Temperature DOF: {Q.dim()}")

        # Functions
        self.stokes = Function(Z, name="Stokes")
        self.stokes.subfunctions[0].rename("Velocity")
        self.stokes.subfunctions[1].rename("Pressure")
        u = split(self.stokes)[0]
        self.T = Function(Q, name="Temperature")

        # The Stokes function is checkpointed as a single mixed entry
        self.solutions = [self.stokes, self.T]
        # The state carried across mesh adaptation and written as output
        state_fields = [*self.stokes.subfunctions, self.T]

        if not initial:
            for field in state_fields:
                field.assign(self.fields[field.name()]["field"])

            self.fields.clear()

        self.fields = {
            field.name(): {"field": field, "output": Function(field, name=field.name())}
            for field in state_fields
        }

        # Boundary conditions
        self.temp_bcs = {
            self.boundary.bottom: {"T": 1.0},
            self.boundary.top: {"T": 0.0},
        }
        stokes_bcs = {
            self.boundary.left: {"ux": 0.0},
            self.boundary.right: {"ux": 0.0},
            self.boundary.bottom: {"uy": 0.0},
            self.boundary.top: {"uy": 0.0},
        }

        for field_name, field_specs in self.fields.items():
            field_specs["include_in_metric"] = field_name in self.metric_fields
            field_specs["metric_mods"] = self.metric_fields.get(field_name)

        stokes_nullspace = create_stokes_nullspace(Z)
        nullspace_args = {
            "nullspace": stokes_nullspace,
            "transpose_nullspace": stokes_nullspace,
        }

        # Adaptive time stepping
        self.tstep_adapt = TimestepAdaptor(self.time_step, u, V, maximum_timestep=0.1)

        # Solvers
        approximation = BoussinesqApproximation(Ra=1e5)

        self.stokes_solver = StokesSolver(
            self.stokes,
            approximation,
            self.T,
            bcs=stokes_bcs,
            solver_parameters="iterative",
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

        self.solvers = [self.energy_solver, self.stokes_solver]

        self.gd = GeodynamicalDiagnostics(
            self.stokes, self.T, self.boundary.bottom, self.boundary.top
        )

    def time_loop(self) -> None:
        for _ in range(self.iterations):  # Time loop
            log(f"Time loop iteration #{self.step}")

            self.tstep_adapt.update_timestep()  # Update time step

            for solver in self.solvers:  # Perform solves
                solver.solve()

            # Increment iteration count and time
            self.step += 1
            self.time.assign(self.time + self.time_step)

            # Write checkpoint and output
            # if float(self.time) >= self.checkpoint_counter * self.checkpoint_frequency:
            self.write_checkpoint()
            # if float(self.time) >= self.output_counter * self.output_frequency:
            self.write_output()

            # Compute diagnostics
            maxchange = sqrt(assemble((self.T - self.energy_solver.T_old) ** 2 * dx))
            energy_conservation = abs(abs(self.gd.Nu_top()) - abs(self.gd.Nu_bottom()))

            # Log diagnostics
            self.plog.log_str(
                f"{self.step} {float(self.time)} {float(self.time_step)} {maxchange} "
                f"{self.gd.u_rms()} {self.gd.u_rms_top()} "
                f"{self.gd.ux_max(self.boundary.top)} {self.gd.Nu_top()} "
                f"{self.gd.Nu_bottom()} {energy_conservation} {self.gd.T_avg()}"
            )

            # Check if simulation has completed
            if maxchange < self.steady_state_tolerance:
                log("Steady-state achieved")
                break


if __name__ == "__main__":
    mesh = UnitSquareMesh(nx=10, ny=10, quadrilateral=False)

    metric_fields = {
        "Temperature": {"scale": 1.0, "log": False},
        "Velocity": {"scale": [1.0, 1.0], "log": [False, False]},
    }
    metric_parameters = {  # For further information: `set_parameters` in animate/metric.py
        "dm_plex_metric": {
            "target_complexity": 3_000,  # Metric complexity, analogous to cell count
            "h_min": 1e-3,  # Minimum metric magnitude (i.e. cell size)
            "h_max": 1e-1,  # Maximum metric magnitude (i.e. cell size)
            "a_max": 10.0,  # Maximum metric anisotropy (cell aspect ratio)
            "p": inf,  # Metric normalisation order
            "gradation_factor": 1.5,  # Maximum variation in length between adjacent edges
        }
    }

    simulation = AdaptiveBaseCase(
        mesh=mesh,
        metric_fields=metric_fields,
        metric_parameters=metric_parameters,
        time_end=inf,
        time_normalisation=1.0,
        time_step=1e-6,
    )
    simulation.run()
    simulation.plog.close()
