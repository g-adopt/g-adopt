from firedrake import DirichletBC
from .scalar_equation import EnergyEquation
from .utility import is_continuous, ensure_constant
from .utility import log_level, INFO, DEBUG

iterative_energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}

direct_energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}

class EnergySolver:
    def __init__(self, T, u, approximation,
            delta_t, timestepper, bcs=None, solver_parameters=None):
        self.Q = T.function_space()
        self.mesh = self.Q.mesh()
        rhocp = approximation.rhocp()
        self.eq = EnergyEquation(self.Q, self.Q, rhocp=rhocp)
        self.fields = {
            'diffusivity': ensure_constant(approximation.kappa()),
            'reference_for_diffusion': approximation.Tbar,
            'source': approximation.energy_source(u),
            'velocity': u,
            'advective_velocity_scaling': rhocp
        }
        sink = approximation.linearized_energy_sink(u)
        if sink:
            self.fields['absorption_coefficient'] = sink

        if solver_parameters is None:
            if self.mesh.topological_dimension() == 2:
                self.solver_parameters = direct_energy_solver_parameters.copy()
                if INFO >= log_level:
                    # not really "informative", but at least we get a 1-line message saying we've passed the energy solve
                    self.solver_parameters['ksp_converged_reason'] = None
            else:
                self.solver_parameters = iterative_energy_solver_parameters.copy()
                if DEBUG >= log_level:
                    self.solver_parameters['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['ksp_converged_reason'] = None
        else:
            self.solver_parameters = solver_parameters
        apply_strongly = is_continuous(T)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == 'T':
                    if apply_strongly:
                        self.strong_bcs.append(DirichletBC(self.Q, value, id))
                    else:
                        weak_bc['q'] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.ts = timestepper(self.eq, T, self.fields, delta_t, self.weak_bcs, strong_bcs=self.strong_bcs,
                solver_parameters=self.solver_parameters)
        self.T_old = self.ts.solution_old

    def solve(self):
        t = 0  # not used atm
        self.ts.advance(t)
