from firedrake import DirichletBC, Function

from .scalar_equation import LevelSetEquation
from .utility import is_continuous


class LevelSetSolver:
    def __init__(self, level_set, u, dt, timestepper, bcs=None, solver_parameters=None):
        self.level_set = level_set
        self.Q = level_set.function_space()
        self.mesh = self.Q.mesh()
        self.dt = dt
        self.eq = LevelSetEquation(self.Q, self.Q)
        self.fields = {"velocity": u}

        if solver_parameters is None:
            self.solver_parameters = {
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        else:
            self.solver_parameters = solver_parameters
        apply_strongly = is_continuous(level_set)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == "T":
                    if apply_strongly:
                        self.strong_bcs.append(DirichletBC(self.Q, value, id))
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.timestepper = timestepper
        self.level_set_old = Function(self.Q)
        # solver is set up only last minute to enable overwriting of the parameters we
        # have set up here
        self._solver_setup = False

    def setup_solver(self):
        """Setup timestepper and associated solver"""
        self.ts = self.timestepper(
            self.eq,
            self.level_set,
            self.fields,
            self.dt,
            bnd_conditions=self.weak_bcs,
            solution_old=self.level_set_old,
            strong_bcs=self.strong_bcs,
            solver_parameters=self.solver_parameters,
        )
        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        t = 0  # not used atm
        self.ts.advance(t)
