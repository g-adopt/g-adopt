# 2d adjoint model based on Weerdesteijn et al 2023

from gadopt import *
# from weerdesteijn_instanticeload_2d import InstantIceLoadWeerdesteijn2d
from inverse_weerdesteijn_2d import InverseWeerdesteijn2d
from gadopt.inverse import *


class InverseIceWeerdesteijn2d(InverseWeerdesteijn2d):
    name = "inverse-ice-weerdesteijn-2d_5ka"
    vertical_component = 1

    def __init__(self, **kwargs):
        self.tape = get_working_tape()
        self.tape.clear_tape()
        checkpoint_file = "./weerdesteijn-instant-ice-dx5km-nz160-dt50years-low-viscosityFalse-chk.h5"
        super().__init__(LOAD_MESH=True, checkpoint_file=checkpoint_file, **kwargs)

#    def disk_checkpointing(self):
#        enable_disk_checkpointing()
#        self.mesh = checkpointable_mesh(self.mesh)
    def setup_ice_load(self):
        super().setup_ice_load()
        # Redefine ice load in terms of a normalised ice thickness
        self.normalised_ice_thickness = Function(self.ice_load.function_space(), name="normalised ice thickness")
        # Only update the ice load at initial times
        self.ice_load.interpolate(self.normalised_ice_thickness * self.rho_ice * self.g * self.Hice * self.disc)

    def setup_control(self):
        self.control = Control(self.normalised_ice_thickness)

    def checkpoint_filename(self):
        return f"{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-chk-ice-inversion.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-ice-inversion.dat"

    def setup_objective_function(self):
        with CheckpointFile(self.checkpoint_file, 'r') as afile:
            self.target_displacement = afile.load_function(self.mesh, name="Displacement")
        self.J = assemble((self.displacement[1] - self.target_displacement[1])**2 / (self.L*1000) * self.ds(self.top_id))
        log(self.J)

    def run_rf_check(self):
        pass

    def calculate_derivative(self):
        pass

    def run_taylor_test(self):
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
        ice_thickness_ub.assign(1)

        self.updated_ice_thickness = Function(self.normalised_ice_thickness)
        self.updated_ice_thickness_file = File(f"{self.name}/update_ice_thickness.pvd")
        bounds = [ice_thickness_lb, ice_thickness_ub]
        self.c = 0
        minimize(self.reduced_functional, bounds=bounds, options={"disp": True})

    def eval_cb(self, J, m):
        log("Control", m.dat.data[:])
        log("minimum Control", m.dat.data[:].min())
        log("J", J)
        self.updated_ice_thickness.assign(m)
        self.updated_ice_thickness_file.write(self.updated_ice_thickness)
        ice_thickness_checkpoint_file = f"{self.name}/updated-ice-thickness-iteration{self.c}.h5"
        with CheckpointFile(ice_thickness_checkpoint_file, "w") as checkpoint:
            checkpoint.save_function(self.updated_ice_thickness, name="Updated ice thickness")
        self.c += 1


if __name__ == "__main__":
    simulation = InverseIceWeerdesteijn2d(dx=5e3, nz=160, Tend_years=500, do_write=True)
    simulation.run_inverse()

#    optimised_simulation = InstantIceLoadWeerdesteijn2d(dx=5e3, nz=160, Tend_years=5000, LOAD_VISCOSITY=True, checkpoint_file=f"{simulation.name}/updated-viscosity-iteration{simulation.c}.h5")
#    optimised_simulation.name = f"{simulation.name}-optimised-viscosity-iteration{simulation.c}"
#    optimised_simulation.run_simulation()
