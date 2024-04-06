# 2d forward model with two disc ice sheets

from gadopt import *
from spada_cylindrical_2d import SpadaCylindrical2d
from gadopt.utility import bivariate_gaussian

class ForwardCylindrical2d(SpadaCylindrical2d):
    name = "forward-cylindrical-heterogenousviscosity-minvisc1e-3"
    vertical_component = 1

    def __init__(self, heterogenous_viscosity=True, **kwargs):
        self.heterogenous_viscosity = heterogenous_viscosity
        super().__init__(**kwargs)


    def setup_ice_load(self):
        self.Hice1 = 1000
        self.Hice2 = 2000

        # Disc ice load but with a smooth transition given by a tanh profile
        disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
        disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
        surface_resolution_radians = 2*pi / self.ncells
        colatitude = self.initialise_colatitude()
        disc1_centre = (2*pi/360) * 25  # centre of disc1
        disc2_centre = pi  # centre of disc2
        self.disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
        self.disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))
        self.ramp = Constant(1)
        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * (self.Hice1 * self.disc1 + self.Hice2 * self.disc2))
    
    def setup_heterogenous_viscosity(self):
        self.heterogenous_viscosity_field = Function(self.viscosity.function_space())
        antarctica_x, antarctica_y = -2e6, -5.5e6

        low_viscosity_antarctica = bivariate_gaussian(self.X[0], self.X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4)
        self.heterogenous_viscosity_field.interpolate(0.001*low_viscosity_antarctica + self.viscosity * (1-low_viscosity_antarctica))
        
        llsvp1_x, llsvp1_y = 3.5e6, 0
        llsvp1 = bivariate_gaussian(self.X[0], self.X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
        self.heterogenous_viscosity_field.interpolate(0.001*llsvp1 + self.heterogenous_viscosity_field * (1-llsvp1))
        
        llsvp2_x, llsvp2_y = -3.5e6, 0
        llsvp2 = bivariate_gaussian(self.X[0], self.X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
        self.heterogenous_viscosity_field.interpolate(0.001*llsvp2 + self.heterogenous_viscosity_field * (1-llsvp2))

        slab_x, slab_y = 3e6, 4.5e6
        slab = bivariate_gaussian(self.X[0], self.X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
        self.heterogenous_viscosity_field.interpolate(0.1*slab + self.heterogenous_viscosity_field * (1-slab))

        high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6
        high_viscosity_craton = bivariate_gaussian(self.X[0], self.X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6, 0.5e6, 0.2)
        self.heterogenous_viscosity_field.interpolate(0.1*high_viscosity_craton + self.heterogenous_viscosity_field * (1-high_viscosity_craton))
        
        if self.heterogenous_viscosity:
            self.viscosity.interpolate(self.heterogenous_viscosity_field)



    def update_ramp(self):
        # already initialised with 1 for instantaneous loading
        pass

    def update_ice_load(self):
        # interpolating ice load at each timestep breaks adjoint (and is probably not 'adjointable')
        pass
    
    def setup_bcs(self):
        self.stokes_bcs = {
            self.top_id: {'normal_stress': self.ice_load, 'free_surface': {'exterior_density': self.rho_ice*(self.disc1+self.disc2)}},
            self.bottom_id: {'un': 0}
        }


if __name__ == "__main__":
    simulation = ForwardCylindrical2d(dx=250*1e3, nz=80, Tend_years=10000, dt_out_years=1000, vertical_tanh_width=40e3, do_write=True)
    simulation.run_simulation()
