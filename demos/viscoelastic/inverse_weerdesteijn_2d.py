# 2d adjoint model based on Weerdesteijn et al 2023

from gadopt import *
from weerdesteijn_2d import Weerdesteijn2d
from gadopt.inverse import *
import numpy as np


class InverseWeerdesteijn2d(Weerdesteijn2d):
    name = "inverse-weerdesteijn-2d"
    vertical_component = 1

    def __init__(self, **kwargs):
        self.tape = get_working_tape()
        self.tape.clear_tape()
        super().__init__(vertical_squashing=False, **kwargs)

#    def disk_checkpointing(self):
#        enable_disk_checkpointing()
#        self.mesh = checkpointable_mesh(self.mesh)

    def setup_control(self):
        print("hi control")
        self.control = Control(self.ice_load)

    def setup_ramp(self):
        # Seems like you need domain for constant but I thought this had
        # depreciated?
        self.ramp = Constant(1, domain=self.mesh)

    def setup_ice_load(self):
        super().setup_ice_load()
        # Only update the ice load at initial times
        self.ice_load.interpolate(self.ramp * self.rho_ice * self.g * self.Hice * self.disc)

    def update_ramp(self):
        # already initialised with 1 for instantaneous loading
        pass

    def update_ice_load(self):
        # interpolating ice load at each timestep breaks adjoint (and is probably not 'adjointable')
        pass

    def checkpoint_filename(self):
        return f"{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years-chk.h5"

    def displacement_filename(self):
        return f"displacement-{self.name}-dx{round(self.dx/1000)}km-nz{self.nz}-dt{self.dt_years}years.dat"

    def run_inverse(self):
        # Add outputs for initial sensitivities
        adj_visc_file = File(f"{self.name}/adj_viscosity.pvd")
        self.tape.add_block(DiagnosticBlock(adj_visc_file, self.viscosity))

        adj_iceload_file = File(f"{self.name}/adj_iceload.pvd")
        converter = RieszL2BoundaryRepresentation(self.ice_load.function_space(), self.top_id)  # convert to surface L2 representation
        self.tape.add_block(DiagnosticBlock(adj_iceload_file, self.ice_load, riesz_options={'riesz_representation': converter}))

        adj_iceload_file_woutconversion = File(f"{self.name}/adj_iceload_woutL2surf.pvd")
        self.tape.add_block(DiagnosticBlock(adj_iceload_file_woutconversion, self.ice_load))

        self.run_simulation()
        self.setup_objective_function()

        # All done with the forward run, stop annotating anything else to the tape
        pause_annotation()

        self.reduced_functional = ReducedFunctional(self.J, self.control, eval_cb_post=self.eval_cb)

        self.run_rf_check()
        self.run_taylor_test()
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
        self.J = assemble(disc*dot(self.displacement[1], self.displacement[1]) * self.ds(self.top_id))
        print(self.J)

    def run_rf_check(self):
        log("J", self.J)
        log("new J", reduced_functional([self.viscosity]))

    def calculate_derivative(self):
        self.reduced_functional.derivative()

    def run_taylor_test(self):
        h = Function(self.viscosity)
        h.dat.data[:] = 1e21*np.random.random(h.dat.data_ro.shape)
        taylor_test(self.reduced_functional, self.viscosity, h)

    def eval_cb(self, J, m):
        pass

    def run_optimisation(self):
        pass


if __name__ == "__main__":
    simulation = InverseWeerdesteijn2d(dx=10e3, nz=80, Tend_years=500, do_write=True)
    simulation.run_inverse()
