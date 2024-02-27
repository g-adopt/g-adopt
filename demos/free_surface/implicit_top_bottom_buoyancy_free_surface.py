from gadopt import *
from implicit_top_bottom_free_surface import TopBottomImplicitFreeSurfaceModel
from test_viscous_surface import run_benchmark


class BuoyancyTopBottomImplicitFreeSurfaceModel(TopBottomImplicitFreeSurfaceModel):

    name = "implicit-buoyancy"
    bottom_free_surface = True

    def __init__(self, dt_factor, nx=320, do_write=False, iterative_2d=False):

        super().__init__(dt_factor, nx=nx, do_write=do_write, iterative_2d=iterative_2d)

    def initialise_wavenumber(self):
        lam_dimensional = self.D  # wavelength of load in m
        self.lam = lam_dimensional/self.L0  # dimensionless lambda
        self.kk = Constant(2 * pi / self.lam)  # wavenumber (dimensionless)

    def initialise_temperature(self):
        self.alpha = 2e-5
        self.Q_temp_scale = 1
        self.forcing_depth = 0.5
        width = (1/self.nx)
        delta = exp(-pow((self.X[1]-self.forcing_depth)/width, 2)) / (width * sqrt(pi))
        self.T = Function(self.Q, name="Temperature").interpolate(delta * self.Q_temp_scale * cos(self.kk * self.X[0]))  # Initialise temperature field

    def initialise_approximation(self):
        Ra = Constant(1)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
        self.approximation = BoussinesqApproximation(Ra, alpha=self.alpha)

    def initialise_free_surfaces(self):
        self.F0 = 0
        self.G0 = 0
        self.stokes_vars[2].assign(self.F0)  # initial top free surface condition
        self.stokes_vars[3].assign(self.G0)  # initial bottom free surface condition
        self.eta_analytical = Function(self.stokes_vars[2], name="eta analytical")
        self.zeta_analytical = Function(self.stokes_vars[3], name="zeta analytical")

    def update_analytical_free_surfaces(self):
        # analytical function
        self.delta_rho = self.rho_bottom - self.rho0
        self.M = self.alpha * self.Q_temp_scale * (-self.kk * 1 * sinh(self.kk*self.forcing_depth) + self.kk * self.forcing_depth * cosh(self.kk*(1-self.forcing_depth))*sinh(self.kk * 1) + sinh(self.kk*(1-self.forcing_depth))*sinh(self.kk * 1))/(sinh(self.kk*1)*sinh(self.kk*1))
        self.N = self.alpha * (self.rho0 / self.delta_rho) * self.Q_temp_scale * (self.kk * 1 * sinh(self.kk*self.forcing_depth)*cosh(self.kk*1) - self.kk * self.forcing_depth * cosh(self.kk*self.forcing_depth)*sinh(self.kk * 1) + sinh(self.kk*self.forcing_depth)*sinh(self.kk * 1))/(sinh(self.kk*1)*sinh(self.kk*1))

        self.tau_eta = self.tau0 * (1 * self.kk + sinh(1*self.kk)*cosh(1*self.kk)) / (sinh(1*self.kk)*sinh(1*self.kk))
        self.tau_zeta = self.tau_eta

        self.gamma = (1*self.kk*cosh(1*self.kk)+sinh(1*self.kk))/(1*self.kk + sinh(1*self.kk)*cosh(1*self.kk))

        self.tau_plus = 0.5 * (self.tau_eta+self.tau_zeta) + 0.5 * pow(pow(self.tau_eta+self.tau_zeta, 2) - 4*(1-pow(self.gamma, 2))*self.tau_eta*self.tau_zeta, 0.5)
        self.tau_minus = 0.5 * (self.tau_eta+self.tau_zeta) - 0.5 * pow(pow(self.tau_eta+self.tau_zeta, 2) - 4*(1-pow(self.gamma, 2))*self.tau_eta*self.tau_zeta, 0.5)

        self.eta_analytical.interpolate(exp(-self.time/self.tau_plus) * cos(self.kk * self.X[0]) * ((self.F0-self.M)*(self.tau_eta - self.tau_minus)-self.gamma * (self.G0-self.N)*self.tau_eta)/(self.tau_plus-self.tau_minus)-exp(-self.time/self.tau_minus) * cos(self.kk * self.X[0]) * ((self.F0-self.M)*(self.tau_eta - self.tau_plus)-self.gamma * (self.G0-self.N)*self.tau_eta)/(self.tau_plus-self.tau_minus) + self.M * cos(self.kk*self.X[0]))

        self.zeta_analytical.interpolate(exp(-self.time/self.tau_plus) * cos(self.kk * self.X[0]) * ((self.G0-self.N)*(self.tau_zeta - self.tau_minus)-self.gamma * (self.F0-self.M)*self.tau_zeta)/(self.tau_plus-self.tau_minus)-exp(-self.time/self.tau_minus) * cos(self.kk * self.X[0]) * ((self.G0-self.N)*(self.tau_zeta - self.tau_plus)-self.gamma * (self.F0-self.M)*self.tau_zeta)/(self.tau_plus-self.tau_minus) + self.N * cos(self.kk*self.X[0]))


if __name__ == "__main__":
    run_benchmark(BuoyancyTopBottomImplicitFreeSurfaceModel)
