from implicit_free_surface import ImplicitFreeSurfaceModel
from test_free_surface import run_benchmark

from gadopt import *


class TopBottomImplicitFreeSurfaceModel(ImplicitFreeSurfaceModel):
    # Test case from Section 3.1.2 of `An implicit free surface algorithm
    # for geodynamical simulations', Kramer et al 2012.

    name = "implicit-both"
    bottom_free_surface = True
    direct = True
    iterative = True

    def __init__(self, dt_factor, **kwargs):
        self.rho_bottom = 2
        self.zeta_error = 0
        super().__init__(dt_factor, **kwargs)

        if self.solver_parameters == "iterative":
            # Schur complement splitting leads to a nullspace in the velocity block.
            # Adding a small absorption term bringing the vertical velocity to zero removes this nullspace
            # and does not affect convergence provided that this term is small compared with the overall numerical error.
            self.absorption_penalty(dt_factor)
            self.stokes_solver.F += (
                self.penalty
                * self.stokes_solver.tests[0][1]
                * (self.stokes_solver.solution[1] - 0)
                * dx
            )

    def setup_function_space(self):
        # Mixed function space with bottom free surface.
        self.Z = MixedFunctionSpace([self.V, self.W, self.W, self.W])

    def setup_variables(self):
        super().setup_variables()
        # Rename zeta for output:
        self.stokes_vars[3].rename("zeta")

    def initialise_free_surfaces(self):
        super().initialise_free_surfaces()
        # Initial Bottom free surface condition
        self.stokes_vars[3].interpolate(self.F0 * cos(self.kk * self.X[0]))
        self.zeta_analytical = Function(self.stokes_vars[3], name="zeta analytical")

    def setup_bcs(self):
        super().setup_bcs()
        self.stokes_bcs[self.bottom_id] = {
            "free_surface": {"eta_index": 1, "Ra_fs": -1}
        }

    def update_analytical_free_surfaces(self):
        super().update_analytical_free_surfaces()
        # Equation A.9 from Kramer et al., 2012. Here we have a simplified form assuming
        # that the relaxation time scale, tau = tau0 (see Equation A.11) which is valid
        # for wavelengths << depth (e.g. see Table 2 from Kramer et al 2012).
        self.zeta_analytical.interpolate(
            exp(-self.time / self.tau0) * self.F0 * cos(self.kk * self.X[0])
        )

    def calculate_error(self):
        super().calculate_error()
        zeta_local_error = assemble(
            pow(self.stokes_vars[3] - self.zeta_analytical, 2) * self.ds(self.bottom_id)
        )
        self.zeta_error += zeta_local_error * self.dt

    def calculate_final_error(self):
        super().calculate_final_error()
        self.final_zeta_error = pow(self.zeta_error, 0.5) / self.L0

    def write_file(self):
        self.output_file.write(
            self.stokes_vars[0],
            self.stokes_vars[1],
            self.stokes_vars[2],
            self.stokes_vars[3],
            self.eta_analytical,
            self.zeta_analytical,
        )

    def absorption_penalty(self, dt_factor):
        self.penalty = dt_factor / 2


if __name__ == "__main__":
    run_benchmark(TopBottomImplicitFreeSurfaceModel)
