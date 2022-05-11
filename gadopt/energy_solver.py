from firedrake import DirichletBC
from .scalar_equation import EnergyEquation
from .utility import is_continuous, ensure_constant


class EnergySolver:
    def __init__(self, T, u, delta_t, timestepper, kappa=1, bcs=None, solver_parameters=None):
        self.Q = T.function_space()
        self.eq = EnergyEquation(self.Q, self.Q)
        self.fields = {
            'diffusivity': ensure_constant(kappa),
            'velocity': u,
        }

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
                solver_parameters=solver_parameters)
        self.T_old = self.ts.solution_old

    def solve(self):
        t = 0  # not used atm
        self.ts.advance(t)
