from gadopt import *
from test_viscous_surface import run_benchmark


class FreeSurfaceModel:

    name = "implicit"
    bottom_free_surface = False

    def __init__(self, dt_factor, nx=80, do_write=False, iterative_2d=False):

        self.do_write = do_write
        mesh = self.setup_mesh(nx)

        # Set up function spaces - currently using the bilinear Q2Q1 element pair:
        V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
        W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
        Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
        Z = self.setup_function_space(V, W)

        # Output function space information:
        log("Number of Velocity DOF:", V.dim())
        log("Number of Pressure DOF:", W.dim())
        log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

        # Function to store the solutions:
        z = Function(Z)  # a field over the mixed function space Z.
        self.stokes_vars = z.subfunctions

        # Next rename for output:
        self.stokes_vars[0].rename("Velocity")
        self.stokes_vars[1].rename("Pressure")
        self.stokes_vars[2].rename("eta")

        self.initialise_wavenumber()

        self.X = SpatialCoordinate(mesh)

        T = self.initialise_temperature(Q)

        self.initialise_approximation()
        self.rho0 = self.approximation.rho  # This defaults to rho0 = 1 (dimensionless)
        g = self.approximation.g  # This defaults to g = 1 (dimensionless)

        self.initialise_free_surfaces()

        # timestepping
        mu = Constant(1)  # Shear modulus (dimensionless)
        self.tau0 = Constant(2 * self.kk * mu / (self.rho0 * g))  # Characteristic time scale (dimensionless)
        log("tau0", self.tau0)

        self.dt = Constant(dt_factor*self.tau0)  # timestep (dimensionless)
        log("dt (dimensionless)", self.dt)

        self.time = Constant(0.0)
        self.max_timesteps = round(10*self.tau0/self.dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
        log("max_timesteps", self.max_timesteps)

        self.setup_bcs()

        self.stokes_solver = StokesSolver(z, T, self.approximation, bcs=self.stokes_bcs, mu=mu, cartesian=True, free_surface_dt=self.dt, iterative_2d=iterative_2d)

        self.error = 0

        # analytical function
        self.update_analytical_free_surfaces()
        # Create output file and select output_frequency:
        if do_write:
            # Write output files in VTK format:
            self.dump_period = 1
            log("dump_period ", self.dump_period)
            self.output_file = File(f"{self.name}_freesurface_D{float(self.D/self.L0)}_mu{float(mu)}_nx{self.nx}_dt{float(self.dt/self.tau0)}tau.pvd")
            self.write_file()

    def setup_mesh(self, nx):
        # Set up geometry:
        self.D = 3e6  # Depth of domain in m
        self.L = self.D  # Length of the domain in m
        self.L0 = self.D  # characteristic length scale for scaling the equations
        self.nx = nx
        ny = self.nx
        self.left_id, self.right_id, self.bottom_id, self.top_id = 1, 2, 3, 4  # Boundary IDs
        return RectangleMesh(self.nx, ny, self.L/self.L0, self.D/self.L0)  # Rectangle mesh generated via firedrake

    def setup_function_space(self, V, W):
        return MixedFunctionSpace([V, W, W])  # Mixed function space.

    def initialise_wavenumber(self):
        lam_dimensional = self.D/2  # wavelength of load in m
        self.lam = lam_dimensional/self.L0  # dimensionless lambda
        self.kk = Constant(2 * pi / self.lam)  # wavenumber (dimensionless)

    def initialise_temperature(self, Q):
        return Function(Q, name="Temperature").assign(0)  # Setup a dummy function for temperature

    def initialise_approximation(self):
        Ra = Constant(0)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
        self.approximation = BoussinesqApproximation(Ra)

    def initialise_free_surfaces(self):
        self.F0 = Constant(1000 / self.L0)  # initial free surface amplitude (dimensionless)
        self.stokes_vars[2].interpolate(self.F0 * cos(self.kk * self.X[0]))  # Initial free surface condition
        self.eta_analytical = Function(self.stokes_vars[2], name="eta analytical")

    def setup_bcs(self):
        # No normal flow except on the free surface
        self.stokes_bcs = {
            self.top_id: {'free_surface': {}},  # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
            self.bottom_id: {'un': 0},
            self.left_id: {'un': 0},
            self.right_id: {'un': 0},
        }

    def update_analytical_free_surfaces(self):
        self.eta_analytical.interpolate(exp(-self.time/self.tau0)*self.F0 * cos(self.kk * self.X[0]))

    def calculate_error(self):
        self.update_analytical_free_surfaces()
        local_error = assemble(pow(self.stokes_vars[2]-self.eta_analytical, 2)*ds(self.top_id))
        self.error += local_error*self.dt

    def calculate_final_error(self):
        self.final_error = pow(self.error, 0.5)/self.L

    def write_file(self):
        self.output_file.write(self.stokes_vars[0], self.stokes_vars[1], self.stokes_vars[2], self.eta_analytical)

    def run_simulation(self):
        # Now perform the time loop:
        for timestep in range(1, self.max_timesteps+1):

            # Solve Stokes sytem:
            self.stokes_solver.solve()
            self.time.assign(self.time + self.dt)

            self.calculate_error()

            # Write output:
            if self.do_write:
                if timestep % self.dump_period == 0:
                    log("timestep", timestep)
                    log("time", float(self.time))
                    self.write_file()
        self.calculate_final_error()


if __name__ == "__main__":
    run_benchmark(FreeSurfaceModel)
