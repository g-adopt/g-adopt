from gadopt import *
from implicit_free_surface import ImplicitFreeSurfaceModel
from test_viscous_surface import run_benchmark


class TopBottomImplicitFreeSurfaceModel(ImplicitFreeSurfaceModel):
    # Test case from Section 3.1.2 of `An implicit free surface algorithm
    # for geodynamical simulations', Kramer et al 2012.

    name = "implicit-both"
    bottom_free_surface = True

    def __init__(self, dt_factor, nx=80, do_write=True, iterative_2d=False, cartesian=True):
        self.dt_factor = dt_factor
        self.rho_bottom = 2
        self.zeta_error = 0
        super().__init__(dt_factor, nx=nx, do_write=do_write, iterative_2d=iterative_2d, cartesian=cartesian)

        if iterative_2d:
            # Schur complement splitting leads to a nullspace in the velocity block.
            # Adding a small absorption term bringing the vertical velocity to zero removes this nullspace
            # and does not effect convergence provided that this term is small compared with the overall numerical error.
            self.absorption_penalty()
            self.stokes_solver.F += self.penalty * self.stokes_solver.test[0][1] * (self.stokes_solver.stokes_vars[0][1] - 0)*dx

    def setup_function_space(self):
        self.Z = MixedFunctionSpace([self.V, self.W, self.W, self.W])  # Mixed function space with bottom free surface.

    def setup_variables(self):
        super().setup_variables()
        # Rename zeta for output:
        self.stokes_vars[3].rename("zeta")

    def initialise_free_surfaces(self):
        super().initialise_free_surfaces()
        self.stokes_vars[3].interpolate(self.F0 * cos(self.kk * self.X[0]))  # Initial Bottom free surface condition
        self.zeta_analytical = Function(self.stokes_vars[3], name="zeta analytical")

    def setup_bcs(self):
        super().setup_bcs()
        # N.b. stokes_integrators assumes that the order of the bcs matches the order of the free surfaces defined in the
        # mixed space, i.e. in this case top_id comes before bottom_id in the dictionary.
        self.stokes_bcs[self.bottom_id] = {'free_surface': {'exterior_density': self.rho_bottom}}

    def update_analytical_free_surfaces(self):
        super().update_analytical_free_surfaces()
        self.zeta_analytical.interpolate(exp(-self.time/self.tau0)*self.F0 * cos(self.kk * self.X[0]))

    def calculate_error(self):
        super().calculate_error()
        zeta_local_error = assemble(pow(self.stokes_vars[3]-self.zeta_analytical, 2)*self.ds(self.bottom_id))
        self.zeta_error += zeta_local_error*self.dt

    def calculate_final_error(self):
        super().calculate_final_error()
        self.final_zeta_error = pow(self.zeta_error, 0.5)/self.L0

    def write_file(self):
        self.output_file.write(self.stokes_vars[0], self.stokes_vars[1], self.stokes_vars[2], self.stokes_vars[3], self.eta_analytical, self.zeta_analytical)

    def absorption_penalty(self):
        self.penalty = self.dt_factor / 2


if __name__ == "__main__":
    run_benchmark(TopBottomImplicitFreeSurfaceModel)
