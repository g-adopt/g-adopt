"""Assembled mass moments for the Spada frame study (Firedrake required)."""

import numpy as np
from firedrake import Constant, assemble

import selfgrav_common as common


class FrameDiagnostics:
    """Whole-body moments from the actual source forms, never profile fits.

    D/M uses the background body mass, giving the linearised displacement
    of the centre of mass. Multipliers retain the library's nondimensional
    convention. Unconstrained runs have no multiplier columns.
    """

    def __init__(self, solver):
        self.solver = solver
        self.mass = float(assemble(solver.mass_dipole_form(
            0, u=Constant((1., 0., 0.)), include_load=False)))
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Frame diagnostics require a positive background body mass")
        self.moments = [solver.mass_dipole_form(i) for i in range(3)]
        zero = Constant((0., 0., 0.))
        load_dipole = np.array([float(assemble(solver.mass_dipole_form(i, u=zero)))
                                for i in range(3)])
        load_mass = sum(float(assemble(solver.form.sheet_integral(sigma, tag, kind)))
                        for tag, sigma, kind in solver.form.sigma_bcs)
        if not np.all(np.isfinite([load_mass, *load_dipole])):
            raise ValueError("Nonfinite assembled load moments")
        mass_scale = common.RHO_BAR * common.D_SCALE**3
        self.metadata = {
            "background_mass_kg": self.mass * mass_scale,
            "assembled_load_mass_kg": load_mass * mass_scale,
            "assembled_load_dipole_kg_m": (load_dipole * mass_scale * common.D_SCALE).tolist(),
            "load_dipole_over_background_mass_m": (load_dipole / self.mass * common.D_SCALE).tolist(),
            "centre_of_mass_definition": "total mantle + fluid-core + prescribed-load dipole / background body mass",
            "multiplier_units": "library nondimensional convention; absent when unconstrained",
        }
        self.columns = ["com_x_m", "com_y_m", "com_z_m"]
        if solver.layout.centre_of_mass:
            self.columns += ["com_lambda_x", "com_lambda_y", "com_lambda_z"]
        self.latest = None

    def sample(self):
        dipole = np.array([float(assemble(form)) for form in self.moments])
        centre = dipole / self.mass * common.D_SCALE
        multipliers = self.solver.centre_of_mass_multipliers()
        if not np.all(np.isfinite([*centre, *multipliers])):
            raise ValueError("Nonfinite centre-of-mass diagnostics")
        self.latest = {
            "mass_dipole_kg_m": (dipole * common.RHO_BAR * common.D_SCALE**4).tolist(),
            "centre_of_mass_m": centre.tolist(),
            "centre_of_mass_multipliers": multipliers,
        }
        return [*centre.tolist(), *multipliers]
