from numpy import inf

from gadopt import *


class AdaptiveBaseCase(AdaptiveSimulation):
    """Mantle-convection base case on an adaptively refined mesh.

    The temperature and velocity fields drive the mesh adaptation; the simulation
    objects are rebuilt on each adapted mesh by `set_up`, and the state is carried
    across adaptations by interpolation.
    """

    def diagnostics(self, initial: bool = False) -> None:
        """Computes and logs simulation diagnostics.

        Args:
          initial:
            Whether the diagnostics are the initial ones, in which case the parameter
            log file is created and its header written.

        """
        # Compute diagnostics
        self.maxchange = sqrt(assemble((self.T - self.energy_solver.T_old) ** 2 * dx))
        energy_conservation = abs(abs(self.gd.Nu_top()) - abs(self.gd.Nu_bottom()))

        # Log diagnostics
        if initial:
            self.plog = ParameterLog("params.log", self.mesh)
            self.plog.log_str(
                "timestep time dt maxchange u_rms u_rms_surf ux_max "
                "nu_top nu_base energy avg_t"
            )
        self.plog.log_str(
            f"{self.step} {float(self.time)} {float(self.time_step)} {self.maxchange} "
            f"{self.gd.u_rms()} {self.gd.u_rms_top()} "
            f"{self.gd.ux_max(self.boundary.top)} {self.gd.Nu_top()} "
            f"{self.gd.Nu_bottom()} {energy_conservation} {self.gd.T_avg()}"
        )

    def initialise(self) -> None:
        """Initialises the simulation on the current mesh.

        Sets up the simulation objects, interpolates the initial temperature field, and
        drives an initial Stokes solve.

        """
        # Set up solvers
        self.set_up(initial=True)
        # Initialise the temperature field with the base-case perturbation
        x, y = SpatialCoordinate(self.mesh)
        self.T.interpolate(1.0 - y + 0.05 * cos(pi * x) * sin(pi * y))
        # Initialise Stokes fields (velocity and pressure)
        log("Initial Stokes solve")
        self.stokes_solver.solve()

    def set_up(self, initial: bool = False) -> None:
        """Sets up the simulation objects on the current mesh.

        Creates the function spaces, functions, boundary conditions, and solvers of the
        mantle-convection base case, as well as the instance attributes required by
        `AdaptiveSimulation`: `solutions`, `fields`, `solvers`, and `tstep_adapt`.

        Args:
          initial:
            Whether the set-up is the initial one, in which case the fields are left at
            their default, zero state; otherwise, the current state is assigned from the
            fields interpolated onto the current mesh.

        """
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
            # Assign the current state from the interpolated fields
            for field in state_fields:
                field.assign(self.fields[field.name()]["field"])

            self.fields.clear()

        self.fields = {
            field.name(): {"field": field, "output": Function(field, name=field.name())}
            for field in state_fields
        }

        # Boundary conditions
        temp_bcs = {
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

        # Null space
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
            self.T, u, approximation, self.time_step, ImplicitMidpoint, bcs=temp_bcs
        )

        self.solvers = [self.energy_solver, self.stokes_solver]

        # Diagnostics
        self.gd = GeodynamicalDiagnostics(
            self.stokes, self.T, self.boundary.bottom, self.boundary.top
        )


if __name__ == "__main__":
    mesh = UnitSquareMesh(nx=10, ny=10, quadrilateral=False)

    metric_fields = {
        "Temperature": {"scale": 1.0, "log": False},
        "Velocity": {"scale": [1.0, 1.0], "log": [False, False]},
    }
    metric_parameters = {  # For further information, refer to animate/metric.py
        "dm_plex_metric": {
            "target_complexity": 5_000,  # Metric complexity, analogous to cell count
            "h_min": 5e-3,  # Minimum metric magnitude (i.e. cell size)
            "h_max": 1e-1,  # Maximum metric magnitude (i.e. cell size)
            "a_max": 10.0,  # Maximum metric anisotropy (cell aspect ratio)
            "p": inf,  # Metric normalisation order
            "gradation_factor": 1.5,  # Maximum length variation between adjacent edges
        }
    }

    simulation = AdaptiveBaseCase(
        mesh=mesh,
        metric_fields=metric_fields,
        metric_parameters=metric_parameters,
        time_step=1e-6,
    )
    simulation.run(steady_state=lambda: simulation.maxchange < 1e-5)
    simulation.plog.close()
