# 2d adjoint model based on Weerdesteijn et al 2023

from gadopt import *
from weerdesteijn_instanticeload_2d import InstantIceLoadWeerdesteijn2d
from inverse_weerdesteijn_2d import InverseWeerdesteijn2d
from gadopt.inverse import *


class InverseViscosityWeerdesteijn2d(InverseWeerdesteijn2d):
    name = "inverse-viscosity-weerdesteijn-2d_5ka_squashed_lb1e-5"
    vertical_component = 1

    def __init__(self, **kwargs):
        self.tape = get_working_tape()
        self.tape.clear_tape()
        checkpoint_file = "./weerdesteijn-low-viscosity-2d-squashed-dx5km-nz160-dt50years-low-viscosityTrue-chk.h5"
        super().__init__(LOAD_MESH=True, checkpoint_file=checkpoint_file, **kwargs)

#    def disk_checkpointing(self):
#        enable_disk_checkpointing()
#        self.mesh = checkpointable_mesh(self.mesh)

    def setup_control(self):
        print("hi control")
        self.control = Control(self.viscosity)

    def checkpoint_filename(self):
        return f"{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-chk-viscosity-inversion.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-viscosity-inversion.dat"

    def setup_objective_function(self):
        with CheckpointFile(self.checkpoint_file, 'r') as afile:
            self.target_displacement = afile.load_function(self.mesh, name="Displacement")
        self.J = assemble((self.displacement[1] - self.target_displacement[1])**2 / (self.L*1000) * self.ds(self.top_id))
        print(self.J)

    def run_rf_check(self):
        pass

    def calculate_derivative(self):
        pass

    def run_taylor_test(self):
        pass

    def run_optimisation(self):
        # Perform a bounded nonlinear optimisation for the viscosity
        # is only permitted to lie in the range [1e19, 1e40...]
        viscosity_lb = Function(self.viscosity.function_space(), name="Lower bound viscosity")
        viscosity_ub = Function(self.viscosity.function_space(), name="Upper bound viscosity")
        viscosity_lb.assign(0.00001)
        viscosity_ub.assign(1)

        self.updated_viscosity = Function(self.viscosity)
        self.updated_viscosity_file = File(f"{self.name}/update_viscosity.pvd")
        bounds = [viscosity_lb, viscosity_ub]
        self.c = 0
        minimize(self.reduced_functional, bounds=bounds, options={"disp": True})

    def callback(self):
        final_misfit = assemble((self.displacement[1] - self.target_displacement[1]) ** 2 / (self.L*1000) * self.ds(self.top_id))
        log(f"final misfit: {final_misfit}")

    def eval_cb(self, J, m):
        log("Control", m.dat.data[:])
        log("minimum Control", m.dat.data[:].min())
        log("J", J)
        self.updated_viscosity.assign(m)
        self.updated_viscosity_file.write(self.updated_viscosity)
        viscosity_checkpoint_file = f"{self.name}/updated-viscosity-iteration{self.c}.h5"
        with CheckpointFile(viscosity_checkpoint_file, "w") as checkpoint:
            checkpoint.save_function(self.updated_viscosity, name="Updated viscosity")
        self.c += 1


if __name__ == "__main__":
    simulation = InverseViscosityWeerdesteijn2d(dx=5e3, nz=160, Tend_years=5000, do_write=True)
    simulation.run_inverse()

    optimised_simulation = InstantIceLoadWeerdesteijn2d(dx=5e3, nz=160, Tend_years=5000, LOAD_VISCOSITY=True, checkpoint_file=f"{simulation.name}/updated-viscosity-iteration{simulation.c}.h5")
    optimised_simulation.name = f"{simulation.name}-optimised-viscosity-iteration{simulation.c}"
    optimised_simulation.run_simulation()
