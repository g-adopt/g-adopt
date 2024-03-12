# 2d box model based on weerdestejin et al 2023

from gadopt import *
from mpi4py import MPI
import numpy as np
from gadopt.utility import vertical_component as vc

class Weerdesteijn2d:
    name = "weerdesteijn-2d"
    vertical_component = 1

    def __init__(self, dx=10e3, nz=80, dt_years=50, Tend_years=110e3, short_simulation=False, do_write=False, LOAD_CHECKPOINT=False, checkpoint_file=None, Tstart=0, cartesian=True,**kwargs):
        # Set up geometry:
        self.dx = dx  # horizontal grid resolution in m
        self.nz = nz
        self.L = 1500e3  # length of the domain in m
        self.dt_years = dt_years
        self.short_simulation = short_simulation
        self.do_write = do_write
        self.LOAD_CHECKPOINT = LOAD_CHECKPOINT
        self.cartesian=cartesian

        # layer properties from spada et al 2011
        self.radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]

        density_values = [3037, 3438, 3871, 4978, 10750]

        shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0]

        viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

        self.D = self.radius_values[0]-self.radius_values[-1]

        if LOAD_CHECKPOINT:
            if checkpoint_file is None:
                raise TypeError("Please provide a checkpoint .h5 file for loading simulation data.")
            self.checkpoint_file = checkpoint_file

        self.setup_mesh()
        self.X = SpatialCoordinate(self.mesh)

        # Set up function spaces - currently using the bilinear Q2Q1 element pair:
        V = VectorFunctionSpace(self.mesh, "CG", 2)  # Displacement function space (vector)
        W = FunctionSpace(self.mesh, "CG", 1)  # Pressure function space (scalar)
        M = MixedFunctionSpace([V, W])  # Mixed function space.
        self.M = M
        TP1 = TensorFunctionSpace(self.mesh, "DG", 2)

        m = Function(M)  # a field over the mixed function space M.
        # Function to store the solutions:
        if LOAD_CHECKPOINT:
            with CheckpointFile(checkpoint_file, 'r') as afile:
                u_dump = afile.load_function(self.mesh, name="Incremental Displacement")
                p_dump = afile.load_function(self.mesh, name="Pressure")
                self.u_, self.p_ = m.subfunctions
                self.u_.assign(u_dump)
                self.p_.assign(p_dump)
                self.displacement = afile.load_function(self.mesh, name="Displacement")
                self.deviatoric_stress = afile.load_function(self.mesh, name="Deviatoric stress")
        else:
            self.u_, self.p_ = m.subfunctions
            self.displacement = Function(V, name="displacement").assign(0)
            self.deviatoric_stress = Function(TP1, name='deviatoric_stress')

        self.u, self.p = split(m)  # Returns symbolic UFL expression for u and p

        self.u_old = Function(V, name="u old")
        self.u_old.assign(self.u_)
        self.vertical_displacement = Function(V.sub(0), name="vertical displacement")  # Function to store vertical displacement for output 

        # Output function space information:
        log("Number of Velocity DOF:", V.dim())
        log("Number of Pressure DOF:", W.dim())
        log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

        # Timing info:
        self.stokes_stage = PETSc.Log.Stage("stokes_solve")

        self.rho_ice = 931
        self.g = 9.8125  # there is also a list but Aspect doesnt use...

        self.viscosity = Function(W, name="viscosity")
        self.initialise_background_field(self.viscosity, viscosity_values)

        self.shear_modulus = Function(W, name="shear modulus")
        self.initialise_background_field(self.shear_modulus, shear_modulus_values)

        self.density = Function(W, name="density")
        self.initialise_background_field(self.density, density_values)
        
        # Timestepping parameters
        self.year_in_seconds = Constant(3600 * 24 * 365.25)

        if self.LOAD_CHECKPOINT and Tstart == 0:
            raise ValueError("If loading from checkpoint please provide a start time")
        self.time = Constant(Tstart * self.year_in_seconds)

        if self.short_simulation:
            self.dt = Constant(2.5 * self.year_in_seconds)  # Initial time-step
            self.Tend = Constant(200 * self.year_in_seconds)
            self.dt_out = Constant(10 * self.year_in_seconds)
        else:
            self.dt = Constant(self.dt_years * self.year_in_seconds)
            self.Tend = Constant(Tend_years * self.year_in_seconds)
            self.dt_out = Constant(10e3 * self.year_in_seconds)

        self.max_timesteps = round((self.Tend - Tstart*self.year_in_seconds)/self.dt)
        log("max timesteps", self.max_timesteps)

        self.dump_period = round(self.dt_out / self.dt)
        log("dump_period", self.dump_period)
        log("dt", self.dt.values()[0])
        log(f"Simulation start time {Tstart} years")
        
        # Initialise ice loading
        self.ice_load = Function(W)
        self.setup_ice_load()
        self.update_ice_load()

        approximation = SmallDisplacementViscoelasticApproximation(self.density, self.displacement, g=self.g)

        self.setup_bcs()
        
        self.setup_nullspaces()

        self.stokes_solver = ViscoelasticStokesSolver(m, self.viscosity, self.shear_modulus, self.density,
                                                      self.deviatoric_stress, self.displacement, approximation,
                                                      self.dt, bcs=self.stokes_bcs, cartesian=self.cartesian,
                                                      nullspace=self.Z_nullspace, transpose_nullspace=self.Z_nullspace, 
                                                      near_nullspace=self.Z_near_nullspace)

        self.prefactor_prestress = Function(W, name='prefactor prestress').interpolate(self.stokes_solver.prefactor_prestress)
        self.effective_viscosity = Function(W, name='effective viscosity').interpolate(self.stokes_solver.effective_viscosity)

        if self.do_write:
            # Rename for output
            self.u_.rename("Incremental Displacement")
            self.p_.rename("Pressure")
            # Create output file
            self.output_file = File(f"{self.name}/out.pvd")
            self.output_file.write(self.u_, self.u_old, self.displacement, self.p_, self.stokes_solver.previous_stress, self.shear_modulus, self.viscosity, self.density, self.prefactor_prestress, self.effective_viscosity, self.vertical_displacement)

        # Now perform the time loop:
        self.displacement_min_array = []

    def setup_mesh(self):
        self.bottom_id, self.top_id = "bottom", "top"  # Boundary IDs for extruded meshes
        if self.LOAD_CHECKPOINT:
            with CheckpointFile(self.checkpoint_file, 'r') as afile:
                self.mesh = afile.load_mesh("surface_mesh_extruded")
        else:
            self.nx = round(self.L/self.dx)
            self.dz = self.D / self.nz  # because of extrusion need to define dz after
            surface_mesh = self.setup_surface_mesh()
            self.mesh = ExtrudedMesh(surface_mesh, self.nz, layer_height=self.dz)
            self.mesh.coordinates.dat.data[:, self.vertical_component] -= self.D
            X = SpatialCoordinate(self.mesh)

            # rescale vertical resolution
            a = Constant(4)
            b = Constant(0)
            depth_c = 500.0
            z_scaled = X[self.vertical_component] / self.D
            Cs = (1.-b) * sinh(a*z_scaled) / sinh(a) + b*(tanh(a*(z_scaled + 0.5))/(2*tanh(0.5*a)) - 0.5)
            Vc = self.mesh.coordinates.function_space()

            scaled_z_coordinates = [X[i] for i in range(self.vertical_component)]
            scaled_z_coordinates.append(depth_c*z_scaled + (self.D - depth_c)*Cs)
            f = Function(Vc).interpolate(as_vector(scaled_z_coordinates))
            self.mesh.coordinates.assign(f)

    def setup_surface_mesh(self):
        return IntervalMesh(self.nx, self.L, name="surface_mesh")

    def initialise_background_field(self, field, background_values):
        for i in range(0, len(background_values)-1):
            field.interpolate(conditional(self.X[self.vertical_component] >= self.radius_values[i+1] - self.radius_values[0],
                              conditional(self.X[self.vertical_component] <= self.radius_values[i] - self.radius_values[0],
                              background_values[i], field), field))
    
    def setup_ice_load(self):
        if self.short_simulation:
            self.T1_load = 100 * self.year_in_seconds
            self.Hice = 100
        else:
            self.T1_load = 90e3 * self.year_in_seconds
            self.Hice = 1000

        self.T2_load = 100e3 * self.year_in_seconds
        
        # Disc ice load but with a smooth transition given by a tanh profile
        disc_radius = 100e3
        k_disc = 2*pi/(8*self.dx)  # wavenumber for disk 2pi / lambda
        r = self.initialise_r()
        self.disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
        self.ramp = Constant(0)
    
    def update_ice_load(self):
        self.update_ramp()
        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * self.Hice * self.disc)

    def update_ramp(self):
        if self.short_simulation:
            self.ramp.assign(conditional(self.time < self.T1_load, self.time / self.T1_load, 1))
        else:
            self.ramp.assign(conditional(self.time < self.T1_load, self.time / self.T1_load,
                             conditional(self.time < self.T2_load, 1 - (self.time - self.T1_load) / (self.T2_load - self.T1_load),
                                         0)
                                         )
                             )
    
    def initialise_r(self):
        return self.X[0]

    def setup_bcs(self):
        # Setup boundary conditions
        self.stokes_bcs = {
            self.bottom_id: {'uy': 0},
            self.top_id: {'normal_stress': self.ice_load, 'free_surface': {'exterior_density': conditional(self.time < self.T2_load, self.rho_ice*self.disc, 0)}},
            1: {'ux': 0},
            2: {'ux': 0},
        }
    
    def setup_nullspaces(self):
        # Nullspaces and near-nullspaces:
        self.Z_nullspace = None  # Default: don't add nullspace for now
        self.Z_near_nullspace = None  # Default: don't add nullspace for now

    def checkpoint_filename(self):
        return f"{self.name}-testcyldinderchanges-chk.h5"

    def displacement_filename(self):
        return f"displacement-testcylinderchanges-{self.name}.dat"
    

    def run_simulation(self):
        checkpoint_filename = self.checkpoint_filename()
        displacement_filename = self.displacement_filename()

        for timestep in range(1, self.max_timesteps+1):
            self.update_ice_load()

            with self.stokes_stage: self.stokes_solver.solve()

            self.time.assign(self.time+self.dt)

  #          bc_displacement = DirichletBC(self.displacement.function_space(), 0, self.top_id)
  #          displacement_z_min = self.displacement.dat.data_ro_with_halos[bc_displacement.nodes, self.vertical_component].min(initial=0)
  #          displacement_min = self.displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading

            # Compute diagnostics:
            self.vertical_displacement.interpolate(vc(self.displacement, cartesian=self.cartesian))
            bc_displacement = DirichletBC(self.vertical_displacement.function_space(), 0, self.top_id)
            displacement_z_min = self.vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
            displacement_min = self.vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
            log("Greatest (-ve) displacement", displacement_min)
            self.displacement_min_array.append([float(self.time/self.year_in_seconds), displacement_min])

            # Write output:
            if timestep % self.dump_period == 0:
                log("timestep", timestep)
                log("time", self.time.values()[0])
                if self.do_write:
                    self.output_file.write(self.u_, self.u_old, self.displacement, self.p_, self.stokes_solver.previous_stress, self.shear_modulus, self.viscosity, self.density, self.prefactor_prestress, self.effective_viscosity, self.vertical_displacement)

                with CheckpointFile(checkpoint_filename, "w") as checkpoint:
                    checkpoint.save_function(self.u_, name="Incremental Displacement")
                    checkpoint.save_function(self.p_, name="Pressure")
                    checkpoint.save_function(self.displacement, name="Displacement")
                    checkpoint.save_function(self.deviatoric_stress, name="Deviatoric stress")

                if MPI.COMM_WORLD.rank == 0:
                    np.savetxt(displacement_filename, self.displacement_min_array)
    

if __name__ == "__main__":
    simulation = Weerdesteijn2d()
    simulation.run_simulation()
