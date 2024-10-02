# 2d box model based on weerdestejin et al 2023

from gadopt import *
from mpi4py import MPI
import numpy as np
from gadopt.utility import vertical_component as vc
from gadopt.utility import CombinedSurfaceMeasure, step_func
import pandas as pd
from decimal import Decimal

class Weerdesteijn2d:
    name = "weerdesteijn-2d"
    vertical_component = 1

    def __init__(self, dx=10e3, nz=80, dt_years=50, Tend_years=110e3, dt_out_years=10e3, short_simulation=False, do_write=True, LOAD_CHECKPOINT=False, LOAD_MESH=False,
                 LOAD_VISCOSITY=False, checkpoint_file=None, Tstart=0, cartesian=True, vertical_squashing=True, vertical_tanh_width=None, low_viscosity_region=False, WRITE_OBJ_CHECKPOINT=False, **kwargs):
        # Set up geometry:
        self.dx = dx  # horizontal grid resolution in m
        self.nz = nz
        self.L = 1500e3  # length of the domain in m
        self.dt_years = dt_years
        self.short_simulation = short_simulation
        self.do_write = do_write
        self.LOAD_CHECKPOINT = LOAD_CHECKPOINT
        self.LOAD_MESH = LOAD_MESH
        self.LOAD_VISCOSITY = LOAD_VISCOSITY
        self.vertical_squashing = vertical_squashing
        self.vertical_tanh_width = vertical_tanh_width
        self.low_viscosity_region = low_viscosity_region
        self.WRITE_OBJ_CHECKPOINT = WRITE_OBJ_CHECKPOINT

        # layer properties from spada et al 2011
        self.radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]

        density_values = [3037, 3438, 3871, 4978, 10750]

        shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0]

        viscosity_values = self.viscosity_values()

        self.D = self.radius_values[0]-self.radius_values[-1]

        if LOAD_CHECKPOINT or LOAD_MESH or LOAD_VISCOSITY:
            if checkpoint_file is None:
                raise TypeError("Please provide a checkpoint .h5 file for loading simulation data.")
            self.checkpoint_file = checkpoint_file

        self.setup_mesh()
        self.mesh.cartesian = cartesian
        self.disk_checkpointing()
        self.X = SpatialCoordinate(self.mesh)

        # Set up function spaces - currently using the bilinear Q2Q1 element pair:
        V = VectorFunctionSpace(self.mesh, "CG", 2)  # Displacement function space (vector)
        W = FunctionSpace(self.mesh, "CG", 1)  # Pressure function space (scalar)
        M = MixedFunctionSpace([V, W])  # Mixed function space.
        self.M = M
        TP1 = TensorFunctionSpace(self.mesh, "DG", 2)
        self.R = FunctionSpace(self.mesh, "R", 0)

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
        self.vertical_displacement = Function(V.sub(1), name="vertical displacement")  # Function to store vertical displacement for output

        # Output function space information:
        log("Number of Velocity DOF:", V.dim())
        log("Number of Pressure DOF:", W.dim())
        log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

        # Timing info:
        self.stokes_stage = PETSc.Log.Stage("stokes_solve")

        self.rho_ice = 931
        self.g = 9.8125  # there is also a list but Aspect doesnt use...

        if self.LOAD_VISCOSITY:
            with CheckpointFile(checkpoint_file, 'r') as afile:
                self.viscosity = afile.load_function(self.mesh, name="Updated viscosity")
        else:
            self.viscosity = Function(W, name="viscosity")
            self.initialise_background_field(self.viscosity, viscosity_values)
            self.setup_heterogenous_viscosity()

        self.shear_modulus = Function(W, name="shear modulus")
        self.initialise_background_field(self.shear_modulus, shear_modulus_values)

        self.density = Function(W, name="density")
        self.initialise_background_field(self.density, density_values)

        # Timestepping parameters
        self.year_in_seconds = Constant(3600 * 24 * 365.25)

        if self.LOAD_CHECKPOINT and Tstart == 0:
            raise ValueError("If loading from checkpoint please provide a start time")
        self.time = Function(self.R).assign(Tstart * self.year_in_seconds)

        if self.short_simulation:
            self.dt = Constant(2.5 * self.year_in_seconds)  # Initial time-step
            self.Tend = Constant(200 * self.year_in_seconds)
            self.dt_out = Constant(10 * self.year_in_seconds)
        else:
            self.dt = Constant(self.dt_years * self.year_in_seconds)
            self.Tend = Constant(Tend_years * self.year_in_seconds)
            self.dt_out = Constant(dt_out_years * self.year_in_seconds)

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
        if do_write:
            File("ice.pvd").write(self.ice_load)
        # Add low viscosity region
        if low_viscosity_region:
            r = self.initialise_r()
            self.viscosity.interpolate(conditional(r < 100e3,
                                                   conditional(self.X[self.vertical_component] < -70e3,
                                                               conditional(self.X[self.vertical_component] > -170e3, 1e-4, self.viscosity),
                                                               self.viscosity),
                                                   self.viscosity)
                                       )

        self.setup_control()

        approximation = SmallDisplacementViscoelasticApproximation(self.density, self.displacement, g=self.g)

        self.setup_bcs()

        self.setup_nullspaces()

        self.stokes_solver = ViscoelasticStokesSolver(m, 1e23*self.viscosity, self.shear_modulus, self.density,
                                                      self.deviatoric_stress, self.displacement, approximation,
                                                      self.dt, bcs=self.stokes_bcs, 
                                                      nullspace=self.Z_nullspace, transpose_nullspace=self.Z_nullspace,
                                                      near_nullspace=self.Z_near_nullspace)

        self.prefactor_prestress = Function(W, name='prefactor prestress').interpolate(self.stokes_solver.prefactor_prestress)
        self.effective_viscosity = Function(W, name='effective viscosity').interpolate(self.stokes_solver.effective_viscosity)

        if self.do_write:
            # Rename for output
            self.u_.rename("Incremental Displacement")
            self.p_.rename("Pressure")
            # Create output file
            self.output_file = File(f"{self.name}/out_dtout{dt_out_years}a.pvd")
            self.output_file.write(self.u_, self.u_old, self.displacement, self.p_, self.stokes_solver.previous_stress, self.shear_modulus, self.viscosity, self.density, self.prefactor_prestress, self.effective_viscosity, self.vertical_displacement)

#            self.setup_displacement_vom_output()
        # Now perform the time loop:
        self.displacement_min_array = []

    def setup_mesh(self):
        self.bottom_id, self.top_id = "bottom", "top"  # Boundary IDs for extruded meshes
        if self.LOAD_CHECKPOINT or self.LOAD_MESH or self.LOAD_VISCOSITY:
            with CheckpointFile(self.checkpoint_file, 'r') as afile:
                self.mesh = afile.load_mesh("surface_mesh_extruded")
        else:
            self.nx = round(self.L/self.dx)
            self.dz = self.D / self.nz  # because of extrusion need to define dz after
            surface_mesh = self.setup_surface_mesh()
            self.mesh = ExtrudedMesh(surface_mesh, self.nz, layer_height=self.dz)

            self.mesh.coordinates.dat.data[:, self.vertical_component] -= self.D

            if self.vertical_squashing:
                # rescale vertical resolution
                X = SpatialCoordinate(self.mesh)
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

        self.ds = CombinedSurfaceMeasure(self.mesh, degree=6)

    def setup_surface_mesh(self):
        return IntervalMesh(self.nx, self.L, name="surface_mesh")

    def disk_checkpointing(self):
        # For non adjoint runs ignore disk checkpointing
        pass

    def viscosity_values(self):
        # Log10(viscosity) using math log10 function
     #   return [17, -2, -2, -1.6989700043360187, 0]
        return [1e17, 1e-2, 1e-2, 2e-2, 0]

    def initialise_background_field(self, field, background_values):
        if self.vertical_tanh_width is None:
            for i in range(0, len(background_values)-1):
                field.interpolate(conditional(self.X[self.vertical_component] >= self.radius_values[i+1] - self.radius_values[0],
                                  conditional(self.X[self.vertical_component] <= self.radius_values[i] - self.radius_values[0],
                                  background_values[i], field), field))
        else:
            profile = background_values[0]
            sharpness = 1 / self.vertical_tanh_width
            depth = self.initialise_depth()
            for i in range(1, len(background_values)-1):
                centre = self.radius_values[i] - self.radius_values[0]
                mag = background_values[i] - background_values[i-1]
                profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)
        
            field.interpolate(profile)


    def setup_heterogenous_viscosity(self):
        pass

    def initialise_depth(self):
        return self.X[self.vertical_component]

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
        disc_dx = 5e3
        k_disc = 2*pi/(8*disc_dx)  # wavenumber for disk 2pi / lambda
        r = self.initialise_r()
        self.disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
        self.setup_ramp()

    def setup_ramp(self):
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

    def setup_control(self):
        # For non adjoint runs ignore controls
        pass

    def setup_bcs(self):
        # Setup boundary conditions
        exterior_density = conditional(self.time < self.T2_load, self.rho_ice*self.disc, 0)
        self.stokes_bcs = {
            self.bottom_id: {'uy': 0},
            self.top_id: {'normal_stress': self.ice_load, 'free_surface': {'delta_rho_fs': self.density - exterior_density}},
            1: {'ux': 0},
            2: {'ux': 0},
        }

    def setup_nullspaces(self):
        # Nullspaces and near-nullspaces:
        self.Z_nullspace = None  # Default: don't add nullspace for now
        self.Z_near_nullspace = None  # Default: don't add nullspace for now

    def checkpoint_filename(self):
        return f"{self.name}-chk.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}.dat"

    def setup_displacement_vom_output(self):
        self.displacement_vom_matplotlib_df = pd.DataFrame()
        surface_nodes = []
        surface_nx = round(self.L / (0.5*self.dx))

        for i in range(surface_nx):
            surface_nodes.append([i*0.5*self.dx, 0])

        if self.mesh.comm.rank == 0:
            self.displacement_vom_matplotlib_df['surface_points'] = surface_nodes
        surface_VOM = VertexOnlyMesh(self.mesh, surface_nodes, missing_points_behaviour='warn')
        DG0_vom = VectorFunctionSpace(surface_VOM, "DG", 0)
        self.displacement_vom = Function(DG0_vom)

        DG0_vom_input_ordering = VectorFunctionSpace(surface_VOM.input_ordering, "DG", 0)
        self.displacement_vom_input = Function(DG0_vom_input_ordering)

    def displacement_vom_out(self):
        self.displacement_vom.interpolate(self.displacement)
        self.displacement_vom_input.interpolate(self.displacement_vom)
        if self.mesh.comm.rank == 0:
            log("check min displacement", self.displacement_vom_input.dat.data[:, 1].min(initial=0))
            log("check arg min displacement", self.displacement_vom_input.dat.data[:, 1].argmin())
            for i in range(self.mesh.geometric_dimension()):
                self.displacement_vom_matplotlib_df[f'displacement{i}_vom_array_{float(self.time/self.year_in_seconds):.0f}years'] = self.displacement_vom_input.dat.data[:, i]
            self.displacement_vom_matplotlib_df.to_csv(f"{self.name}/surface_displacement_arrays.csv")
    
    def integrated_time_misfit(self, timestep):
        pass

    def run_simulation(self):
        checkpoint_filename = self.checkpoint_filename()
        displacement_filename = self.displacement_filename()
        objective_checkpoint_file = CheckpointFile(f"{self.name}-disp-incdisp.h5", "w")
        objective_checkpoint_file.save_mesh(self.mesh)
        for timestep in range(1, self.max_timesteps+1):
            self.update_ice_load()

            with self.stokes_stage: self.stokes_solver.solve()

            # Storing displacement and incremental displacement to be used in the objective 
            if self.WRITE_OBJ_CHECKPOINT:
                objective_checkpoint_file.save_function(self.u_, name="Incremental Displacement", idx=timestep)
                objective_checkpoint_file.save_function(self.displacement, name="Displacement", idx=timestep)
            
            self.integrated_time_misfit(timestep)
            self.time.assign(self.time+self.dt)

            # Compute diagnostics:
            self.vertical_displacement.interpolate(vc(self.displacement))
            bc_displacement = DirichletBC(self.vertical_displacement.function_space(), 0, self.top_id)
            displacement_z_min = self.vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
            displacement_min = self.vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
            log("Greatest (-ve) displacement", displacement_min)
            self.displacement_min_array.append([float(self.time/self.year_in_seconds), displacement_min])

#            if self.do_write and timestep == 1:
                # Write out the elastic displacement
#                self.displacement_vom_out()
            # Write output:
            if timestep % self.dump_period == 0:
                log("timestep", timestep)
            #    log("time", self.time.values()[0])
                if self.do_write:
                    self.output_file.write(self.u_, self.u_old, self.displacement, self.p_, self.stokes_solver.previous_stress, self.shear_modulus, self.viscosity, self.density, self.prefactor_prestress, self.effective_viscosity, self.vertical_displacement)

    #                self.displacement_vom_out()

                with CheckpointFile(checkpoint_filename, "w") as checkpoint:
                    checkpoint.save_function(self.u_, name="Incremental Displacement")
                    checkpoint.save_function(self.p_, name="Pressure")
                    checkpoint.save_function(self.displacement, name="Displacement")
                    checkpoint.save_function(self.deviatoric_stress, name="Deviatoric stress")

                if MPI.COMM_WORLD.rank == 0:
                    np.savetxt(displacement_filename, self.displacement_min_array)

        objective_checkpoint_file.close()

if __name__ == "__main__":
    simulation = Weerdesteijn2d()
    simulation.run_simulation()
