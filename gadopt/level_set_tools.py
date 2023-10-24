from firedrake import (
    DirichletBC,
    Function,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TestFunction,
    TrialFunction,
    div,
    dot,
    sqrt,
)

from .equations import BaseEquation, BaseTerm
from .scalar_equation import ScalarAdvectionEquation
from .utility import is_continuous


class ProjectionTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        target = fields["target"]
        n = self.n

        ufl_element = target * div(test) * self.dx
        ufl_bc = target * dot(test, n) * self.ds

        return -ufl_element + ufl_bc


class ReinitialisationTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        level_set_grad = fields["level_set_grad"]
        epsilon = fields["epsilon"]

        sharpen_term = -trial * (1 - trial) * (1 - 2 * trial) * test * self.dx
        balance_term = (
            epsilon
            * (1 - 2 * trial)
            * sqrt(level_set_grad[0] ** 2 + level_set_grad[1] ** 2)
            * test
            * self.dx
        )

        return sharpen_term + balance_term


class LevelSetEquation(ScalarAdvectionEquation):
    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class ProjectionEquation(BaseEquation):
    terms = [ProjectionTerm]

    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class ReinitialisationEquation(BaseEquation):
    terms = [ReinitialisationTerm]

    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class TimeStepperSolver:
    def __init__(
        self,
        function,
        fields,
        dt,
        timestepper,
        equation,
        coupled_solver=None,
        bcs=None,
        solver_parameters=None,
    ):
        self.func_space = function.function_space()
        self.mesh = self.func_space.mesh()

        self.function = function
        self.function_old = Function(self.func_space)

        self.fields = fields
        self.dt = dt
        self.timestepper = timestepper
        self.coupled_solver = coupled_solver

        self.eq = equation(self.func_space, self.func_space)

        apply_strongly = is_continuous(function)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == "T":
                    if apply_strongly:
                        self.strong_bcs.append(DirichletBC(self.func_space, value, id))
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        if solver_parameters is None:
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        else:
            self.solver_parameters = solver_parameters

        self._solver_setup = False

    def setup_solver(self):
        """Setup timestepper and associated solver"""
        self.ts = self.timestepper(
            self.eq,
            self.function,
            self.fields,
            self.dt,
            bnd_conditions=self.weak_bcs,
            solution_old=self.function_old,
            strong_bcs=self.strong_bcs,
            solver_parameters=self.solver_parameters,
            coupled_solver=self.coupled_solver,
        )
        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        t = 0  # not used atm
        self.ts.advance(t)


class ProjectionSolver:
    def __init__(self, function, fields, bcs=None, solver_parameters=None):
        self.func_space = function.function_space()
        self.mesh = self.func_space.mesh()

        self.test = TestFunction(self.func_space)
        self.trial = TrialFunction(self.func_space)

        self.function = function

        self.fields = fields

        self.eq = ProjectionEquation(self.func_space, self.func_space)

        self.bilinear = self.eq.mass_term(self.test, self.trial)
        self.linear = self.eq.residual(self.test, None, None, self.fields, None)

        self.bcs = bcs
        if solver_parameters is None:
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            self.solver_parameters = solver_parameters

        self._solver_setup = False

    def setup_solver(self):
        self.problem = LinearVariationalProblem(
            self.bilinear, self.linear, self.function, bcs=self.bcs
        )
        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=self.solver_parameters
        )
        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        self.solver.solve()
