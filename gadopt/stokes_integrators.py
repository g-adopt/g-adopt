from .time_stepper import TimeIntegratorBase
from .momentum_equation import StokesEquations
from .utility import upward_normal, ensure_constant
import firedrake


class StokesSolver:
    name = 'Stokes'

    def __init__(self, z, T, delta_t, bcs=None, Ra=1, g=1, mu=1,
                 quad_degree=6, cartesian=True, solver_parameters={}, **kwargs):
        self.Z = z.function_space()
        self.test = firedrake.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree)
        self.solution = z
        self.delta_t = ensure_constant(delta_t)
        self.Ra = ensure_constant(Ra)
        self.g = ensure_constant(g)
        self.mu = ensure_constant(mu)
        self.solver_parameters = solver_parameters

        self.kwargs = kwargs
        u, p = firedrake.split(self.solution)
        self.k = upward_normal(self.Z.mesh(), cartesian)
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'source': self.Ra * self.g * T * self.k,
        }

        self.weak_bcs = {}
        self.strong_bcs = []
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == 'u':
                    self.strong_bcs.append(firedrake.DirichletBC(self.Z.sub(0), value, id))
                elif type == 'ux':
                    self.strong_bcs.append(firedrake.DirichletBC(self.Z.sub(0).sub(0), value, id))
                elif type == 'uy':
                    self.strong_bcs.append(firedrake.DirichletBC(self.Z.sub(0).sub(1), value, id))
                elif type == 'uz':
                    self.strong_bcs.append(firedrake.DirichletBC(self.Z.sub(0).sub(2), value, id))
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.F = 0
        for test, eq, u in zip(self.test, self.equations, firedrake.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        self.problem = firedrake.NonlinearVariationalProblem(self.F, self.solution, bcs=self.strong_bcs, **self.kwargs)
        self.solver = firedrake.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           options_prefix=self.name)
        self._initialized = True

    def solve(self):
        self.solver.solve()
