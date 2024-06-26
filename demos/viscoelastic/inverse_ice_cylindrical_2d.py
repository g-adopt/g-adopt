# 2d adjoint model based on Weerdesteijn et al 2023

from gadopt import *
# from weerdesteijn_instanticeload_2d import InstantIceLoadWeerdesteijn2d
from inverse_cylindrical_2d import InverseCylindrical2d
from gadopt.inverse import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alpha_smoothing", default=0.1, type=float, help="smoothing factor", required=False)
parser.add_argument("--alpha_damping", default=0.1, type=float, help="damping factor", required=False)
args = parser.parse_args()



class InverseIceCylindrical2d(InverseCylindrical2d):
    name = "inverse-ice-cylindrical-Jdispveltimeint-scipy"
    vertical_component = 1

    def __init__(self, alpha_smoothing=0.1, alpha_damping=0.1, LOAD_ice=False, **kwargs):
        self.tape = get_working_tape()
        self.tape.clear_tape()
        self.alpha_smoothing = alpha_smoothing
        self.alpha_damping = alpha_damping
        self.name = f"{self.name}-smooth{alpha_smoothing}-damping{alpha_damping}"
        checkpoint_file = "./forward-cylindrical-heterogenousviscosity-minvisc1e-3-nz80dx250km-disp-incdisp.h5"
        self.velocity_misfit = 0
        self.displacement_misfit = 0
        super().__init__(LOAD_MESH=True, checkpoint_file=checkpoint_file, **kwargs)

#    def disk_checkpointing(self):
#        enable_disk_checkpointing()
#        self.mesh = checkpointable_mesh(self.mesh)
    def setup_ice_load(self):
        super().setup_ice_load()
        # Redefine ice load in terms of a normalised ice thickness
        self.normalised_ice_thickness = Function(self.ice_load.function_space(), name="normalised ice thickness")
        # Only update the ice load at initial times
        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * self.normalised_ice_thickness * self.Hice1)
        self.target_normalised_ice_thickness = Function(self.ice_load.function_space(), name="target normalised ice thickness")
        self.target_normalised_ice_thickness.interpolate(self.disc1 + (self.Hice2/self.Hice1)*self.disc2)

    def setup_control(self):
        self.control = Control(self.normalised_ice_thickness)

    def checkpoint_filename(self):
        return f"{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-chk-ice-inversion.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-ice-inversion.dat"

    def integrated_time_misfit(self, timestep):
        with CheckpointFile(self.checkpoint_file, 'r') as afile:
            self.target_incremental_displacement = afile.load_function(self.mesh, name="Incremental Displacement", idx=timestep)
            self.target_displacement = afile.load_function(self.mesh, name="Displacement", idx=timestep)

        circumference = 2 * pi * self.rmax
        target_velocity = self.target_incremental_displacement/self.dt_years
        velocity = self.u_/self.dt_years
        velocity_error = velocity - target_velocity
        velocity_scale = 0.1/self.dt_years
        self.velocity_misfit += assemble(dot(velocity_error, velocity_error) / (circumference * velocity_scale**2) * self.ds(self.top_id))
        
        displacement_error = self.displacement - self.target_displacement
        displacement_scale = 50
        self.displacement_misfit += assemble(dot(displacement_error, displacement_error) / (circumference * displacement_scale**2) * self.ds(self.top_id))

    def setup_objective_function(self):
#        with CheckpointFile(self.checkpoint_file, 'r') as afile:
 #           self.target_incremental_displacement = afile.load_function(self.mesh, name="Incremental Displacement")
        circumference = 2 * pi * self.rmax
        max_inc_disp_normal = 0.2/self.dt_years
        max_inc_disp_tangential = 0.1/self.dt_years
        n = FacetNormal(self.mesh)
        # Define the component terms of the overall objective functional
        damping = assemble((self.normalised_ice_thickness) ** 2 /circumference * self.ds)
        smoothing = assemble(dot(grad(self.normalised_ice_thickness), grad(self.normalised_ice_thickness)) / circumference * self.ds)

#        misfit = assemble(dot(n, (self.u_ - self.target_incremental_displacement)/self.dt_years)**2 / (circumference * max_inc_disp**2) * self.ds(self.top_id))
        target_velocity = self.target_incremental_displacement/self.dt_years
        target_normal_velocity = dot(n, target_velocity)*n
        target_tangential_velocity = target_velocity - target_normal_velocity 
        
        velocity = self.u_/self.dt_years
        normal_velocity = dot(n, velocity)*n
        tangential_velocity = velocity - normal_velocity

        normal_velocity_error = normal_velocity - target_normal_velocity
        tangential_velocity_error = tangential_velocity - target_tangential_velocity
       
       #velocity_error = (self.u_-self.target_incremental_displacement)/self.dt_years
        #misfit = assemble(dot(velocity_error, velocity_error) / (circumference * max_inc_disp**2) * self.ds(self.top_id))
        normal_misfit = assemble(dot(normal_velocity_error, normal_velocity_error) / (circumference * max_inc_disp_normal**2) * self.ds(self.top_id))
        tangential_misfit = assemble(dot(tangential_velocity_error, tangential_velocity_error) / (circumference * max_inc_disp_tangential**2) * self.ds(self.top_id))
       # alpha_damping = 0.1
       # alpha_smoothing = 0.1
        #self.J = normal_misfit + tangential_misfit #+ alpha_damping * damping + alpha_smoothing * smoothing
        self.J = (self.displacement_misfit + self.velocity_misfit)/self.max_timesteps + self.alpha_damping * damping + self.alpha_smoothing * smoothing #alpha_damping * damping
        log(self.J)

    def run_rf_check(self):
        pass
        #log("J", self.J)
        #log("new J", self.reduced_functional([self.normalised_ice_thickness]))

    def calculate_derivative(self):
        pass

    def run_taylor_test(self, control):
        pass
#        h = Function(self.normalised_ice_thickness)
#        h.dat.data[:] = np.random.random(h.dat.data_ro.shape)
#        taylor_test(self.reduced_functional, self.normalised_ice_thickness, h)

    def run_optimisation(self):
        # Perform a bounded nonlinear optimisation for the viscosity
        # is only permitted to lie in the range [1e19, 1e40...]
        ice_thickness_lb = Function(self.normalised_ice_thickness.function_space(), name="Lower bound ice thickness")
        ice_thickness_ub = Function(self.normalised_ice_thickness.function_space(), name="Upper bound ice thickness")
        ice_thickness_lb.assign(0.0)
        ice_thickness_ub.assign(5)

        self.updated_ice_thickness = Function(self.normalised_ice_thickness)
        self.updated_ice_thickness_file = File(f"{self.name}/update_ice_thickness.pvd")
        self.updated_displacement = Function(self.displacement, name="updated displacement")
        self.updated_incremental_displacement = Function(self.displacement, name="updated incremental displacement")
        self.updated_out_file = File(f"{self.name}/updated_out.pvd")
        bounds = [ice_thickness_lb, ice_thickness_ub]
        self.c = 0
        minimize(self.reduced_functional, bounds=bounds, options={"disp": True})
        ''' 
        def callback():
            circumference = 2 * pi * self.rmax
            initial_misfit = assemble(
                (self.normalised_ice_thickness.block_variable.checkpoint - self.target_normalised_ice_thickness) ** 2/circumference * self.ds
            )
            
            max_inc_disp = 0.2/self.dt_years
            n = FacetNormal(self.mesh)
            final_misfit = assemble(dot(n, (self.u_.block_variable.checkpoint - self.target_incremental_displacement)/self.dt_years)**2 / (circumference * max_inc_disp**2) * self.ds(self.top_id))

            log(f"Initial misfit; {initial_misfit}; final misfit: {final_misfit}")

        minimisation_problem = MinimizationProblem(self.reduced_functional, bounds=bounds)

        optimiser = LinMoreOptimiser(
            minimisation_problem,
            minimisation_parameters,
            checkpoint_dir=f"{self.name}/optimisation_checkpoint",
        )
        optimiser.add_callback(callback)
        optimiser.run()'''

    def eval_cb(self, J, m):
        log("Control", m.dat.data[:])
        log("minimum Control", m.dat.data[:].min())
        log("J", J)
        circumference = 2 * pi * self.rmax
        # Define the component terms of the overall objective functional
        damping = assemble((self.normalised_ice_thickness.block_variable.checkpoint) ** 2 /circumference * self.ds)
        smoothing = assemble(dot(grad(self.normalised_ice_thickness.block_variable.checkpoint), grad(self.normalised_ice_thickness.block_variable.checkpoint)) / circumference * self.ds)
        log("damping", damping)
        log("smoothing", smoothing)
        
        self.updated_ice_thickness.assign(m)
        self.updated_ice_thickness_file.write(self.updated_ice_thickness, self.target_normalised_ice_thickness)
        self.updated_displacement.interpolate(self.displacement.block_variable.checkpoint) 
        self.updated_incremental_displacement.interpolate(self.u_.block_variable.checkpoint)
        self.updated_out_file.write(self.updated_displacement, self.updated_incremental_displacement, self.target_displacement, self.target_incremental_displacement)
        ice_thickness_checkpoint_file = f"{self.name}/updated-ice-thickness-iteration{self.c}.h5"
        with CheckpointFile(ice_thickness_checkpoint_file, "w") as checkpoint:
            checkpoint.save_function(self.updated_ice_thickness, name="Updated ice thickness")
        self.c += 1

if __name__ == "__main__":
    simulation = InverseIceCylindrical2d(dx=100*1e3, nz=80, Tend_years=10000,dt_out_years=1000,vertical_tanh_width=40e3, do_write=True,heterogenousviscosity=True, alpha_smoothing=args.alpha_smoothing, alpha_damping=args.alpha_damping)
    simulation.run_inverse()

#    optimised_simulation = InstantIceLoadWeerdesteijn2d(dx=5e3, nz=160, Tend_years=5000, LOAD_VISCOSITY=True, checkpoint_file=f"{simulation.name}/updated-viscosity-iteration{simulation.c}.h5")
#    optimised_simulation.name = f"{simulation.name}-optimised-viscosity-iteration{simulation.c}"
#    optimised_simulation.run_simulation()
