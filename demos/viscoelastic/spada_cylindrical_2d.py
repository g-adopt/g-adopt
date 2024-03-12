# 2d cylindrical model based on Spada et al 2011

from gadopt import *
from weerdesteijn_2d import Weerdesteijn2d

class SpadaCylindrical2d(Weerdesteijn2d):
    name = f"spada-cylindrical-2d"
    vertical_component = 1

    def __init__(self, cartesian=False, **kwargs):
        super().__init__(cartesian=cartesian, **kwargs)

    def setup_mesh(self):
        # Set up geometry:
        self.rmin = self.radius_values[-1] #/self.D
        log(self.rmin)
        self.rmax = self.radius_values[0] #/self.D
        log(self.rmax)
        # Construct a circle mesh and then extrude into a cylinder:
        radius_earth = 6371e3
        self.ncells = round(2 * pi * radius_earth / self.dx)
        surface_dx = 2 * pi * radius_earth / self.ncells
        log("target surface resolution = ", self.dx)
        log("actual surface resolution = ", surface_dx)
        self.setup_surface_mesh()
        dz = self.D / self.nz 
        self.mesh = ExtrudedMesh(self.surface_mesh, layers=self.nz, layer_height=dz, extrusion_type='radial')
        self.bottom_id, self.top_id = "bottom", "top"
    
    def initialise_background_field(self, field, background_values):
        print("hello background field")
        print(self.D)
        r = sqrt(self.X[0]**2 + self.X[1]**2)-self.radius_values[0]
        rfunc = Function(self.M.sub(1)).interpolate(r)
        File("rfunc.pvd").write(rfunc)
        for i in range(0, len(background_values)-1):
            field.interpolate(conditional(r >= self.radius_values[i+1] - self.radius_values[0],
                              conditional(r <= self.radius_values[i] - self.radius_values[0],
                              background_values[i], field), field))

        # maybe a bug in the meshing/machine tolerance above z=0 surface by 1e-9 and below depth of mantle... 
        field.interpolate(conditional(r >0, background_values[0], field))
        field.interpolate(conditional(r <-self.D, background_values[-2], field))
    
    def setup_surface_mesh(self):
        self.surface_mesh = CircleManifoldMesh(self.ncells, radius=self.rmin, degree=2, name='surface_mesh')

    def setup_ice_load(self):
        self.Hice = 1000   

        # Disc ice load but with a smooth transition given by a tanh profile
        disc_halfwidth = (2*pi/360) * 10  # Disk half width in radians
        surface_resolution_radians = 2*pi / self.ncells
        colatitude =  self.initialise_colatitude()
        self.disc = 0.5*(1-tanh( (abs(colatitude) - disc_halfwidth) / (2*surface_resolution_radians)))
        self.ramp = Constant(1)  # Spada starts from a Heaviside loading, so spin up...
    
    def initialise_colatitude(self):
        # This gives 0 at the north pole and pi (-pi) near the South pole
        return atan2(self.X[0], self.X[1])

    def update_ice_load(self):
        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * self.Hice * self.disc)
    
    def setup_bcs(self):
        self.stokes_bcs = {
            self.top_id: {'normal_stress': self.ice_load, 'free_surface': {'exterior_density': self.rho_ice*self.disc}},
            self.bottom_id: {'un': 0}
        }
    
    def setup_nullspaces(self):
        # Nullspaces and near-nullspaces:
        self.Z_nullspace = create_stokes_nullspace(self.M, closed=False, rotational=True)
        self.Z_near_nullspace = create_stokes_nullspace(self.M, closed=False, rotational=True, translations=[0, 1])

    def checkpoint_filename(self):
        return f"{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-chk.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years.dat"


if __name__ == "__main__":
    simulation = SpadaCylindrical2d(dx=500*1e3, nz=80,cartesian=False,do_write=True)
    simulation.run_simulation()
