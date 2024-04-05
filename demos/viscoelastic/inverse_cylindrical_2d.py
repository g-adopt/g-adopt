# 2d adjoint model based on Weerdesteijn et al 2023

from gadopt import *
from forward_cylindrical_2d import ForwardCylindrical2d
from gadopt.inverse import *
import numpy as np


class InverseCylindrical2d(ForwardCylindrical2d):
    name = "inverse-cylindrical-2d"
    vertical_component = 1

    def __init__(self, **kwargs):
        self.tape = get_working_tape()
        self.tape.clear_tape()
        super().__init__(**kwargs)

#    def disk_checkpointing(self):
#        enable_disk_checkpointing()
#        self.mesh = checkpointable_mesh(self.mesh)

    def setup_control(self):
        print("hi control")
        self.control = Control(self.ice_load)
#        self.control = Control(self.viscosity)

    def setup_ramp(self):
        # Seems like you need domain for constant but I thought this had
        # depreciated?
        self.ramp = Constant(1, domain=self.mesh)

#    def setup_ice_load(self):
        super().setup_ice_load()
        # Redefine ice load in terms of a normalised ice thickness
#        self.normalised_ice_thickness = Function(self.ice_load.function_space(), name="normalised ice thickness")
#        # Only update the ice load at initial times
#        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * self.normalised_ice_thickness * self.Hice1)

    def run_inverse(self):
        # Add outputs for initial sensitivities
        adj_visc_file = File(f"{self.name}/adj_viscosity.pvd")
        self.tape.add_block(DiagnosticBlock(adj_visc_file, self.viscosity))

        adj_iceload_file = File(f"{self.name}/adj_iceload.pvd")
        converter = RieszL2BoundaryRepresentation(self.ice_load.function_space(), self.top_id)  # convert to surface L2 representation
        self.tape.add_block(DiagnosticBlock(adj_iceload_file, self.ice_load, riesz_options={'riesz_representation': converter}))
        adj_icethickness_file = File(f"{self.name}/adj_icethickness.pvd")
#        self.tape.add_block(DiagnosticBlock(adj_icethickness_file, self.normalised_ice_thickness, riesz_options={'riesz_representation': converter}))

        self.run_simulation()
        self.setup_objective_function()

        # All done with the forward run, stop annotating anything else to the tape
        pause_annotation()

        self.reduced_functional = ReducedFunctional(self.J, self.control, eval_cb_post=self.eval_cb)

        self.run_rf_check()
        self.run_taylor_test(self.ice_load)
        self.calculate_derivative()

        self.run_optimisation()

        # If we're performing mulitple successive optimisations, we want
        # to ensure the annotations are switched back on for the next code
        # to use them
        continue_annotation()

    def setup_objective_function(self):
        disc_radius = 500e3
        k_disc = 2*pi/(8*self.dx)  # wavenumber for disk 2pi / lambda
        r = self.initialise_r()
        disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
        self.J = assemble(dot(self.displacement[1], self.displacement[1])/ (2*pi*self.rmax) * self.ds(self.top_id))
        print(self.J)
   # def setup_objective_function(self):
   #     with CheckpointFile(self.checkpoint_file, 'r') as afile:
   #         self.target_displacement = afile.load_function(self.mesh, name="Displacement")
   #     self.J = assemble((self.displacement[1] - self.target_displacement[1])**2 / (2*pi*self.rmax) * self.ds(self.top_id))
   #     log(self.J)

    def run_rf_check(self):
        log("J", self.J)
        log("new J", self.reduced_functional([self.ice_load]))

    def calculate_derivative(self):
        self.reduced_functional.derivative()

    def run_taylor_test(self, control):
        h = Function(control)
        h.dat.data[:] = 2*9.81*1000*921*np.random.random(h.dat.data_ro.shape)
        taylor_test(self.reduced_functional, control, h)

    def eval_cb(self, J, m):
        pass

    def run_optimisation(self):
        pass


if __name__ == "__main__":
    simulation = InverseCylindrical2d(dx=100*1e3, nz=80, Tend_years=100, dt_out_years=1000, vertical_tanh_width=20e3, do_write=True, heterogenous_viscosity=False)
    simulation.run_inverse()
