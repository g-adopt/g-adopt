from abc import ABC, abstractmethod, abstractproperty
import firedrake
import operator
import numpy as np
from .utility import ensure_constant

"""
Timestepper code, this is mostly copied from Thetis. At the moment explicit RK methods only.
"""


class TimeIntegratorBase(ABC):
    """
    Abstract class that defines the API for all time integrators

    Both :class:`TimeIntegrator` and :class:`CoupledTimeIntegrator` inherit
    from this class.
    """

    @abstractmethod
    def advance(self, t, update_forcings=None):
        """
        Advances equations for one time step

        :arg t: simulation time
        :type t: float
        :arg update_forcings: user-defined function that takes the simulation
            time and updates any time-dependent boundary conditions
        """
        pass

    @abstractmethod
    def initialize(self, init_solution):
        """
        Initialize the time integrator

        :arg init_solution: initial solution
        """
        pass


class TimeIntegrator(TimeIntegratorBase):
    """
    Base class for all time integrator objects that march a single equation
    """
    def __init__(self, equation, solution, fields, dt, solution_old=None,
                 solver_parameters=None, strong_bcs=None):
        """
        :arg equation: the equation to solve
        :type equation: :class:`BaseEquation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg solution_old: :class:`Function` where solution at previous timestep
                             is stored. New one will be created if not provided.
        :kwarg dict solver_parameters: PETSc solver options
        :kwarg list strong_bcs: list of DirichletsBCs
        """
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        self.test = firedrake.TestFunction(solution.function_space())
        self.solution = solution
        self.fields = fields
        self.dt = dt
        self.dt_const = ensure_constant(dt)
        self.solution_old = solution_old or firedrake.Function(solution, name='Old'+solution.name())

        # unique identifier used in solver
        self.name = '-'.join([self.__class__.__name__,
                              self.equation.__class__.__name__])

        self.solver_parameters = {}
        if solver_parameters:
            self.solver_parameters.update(solver_parameters)

        self.strong_bcs = strong_bcs or []
        self.hom_bcs = [firedrake.DirichletBC(bci.function_space(), 0, bci.sub_domain) for bci in self.strong_bcs]


class RungeKuttaTimeIntegrator(TimeIntegrator):
    """Abstract base class for all Runge-Kutta time integrators"""

    @abstractmethod
    def get_final_solution(self):
        """
        Evaluates the final solution
        """
        pass

    @abstractmethod
    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        pass

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if not self._initialized:
            self.initialize(self.solution)
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()


class ERKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form. All terms in the equation are treated explicitly.
    """
    def __init__(self, equation, solution, fields, dt,
                 solution_old=None, bnd_conditions=None,
                 solver_parameters={}, strong_bcs=None):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg solution_old: :class:`Function` where solution at previous timestep
                             is stored. New one will be created if not provided.
        :kwarg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg dict solver_parameters: PETSc solver options
        :kwarg list strong_bcs: list of DirichletsBCs
        """
        super(ERKGeneric, self).__init__(equation, solution, fields, dt,
                                         solution_old, solver_parameters, strong_bcs)
        self._initialized = False
        V = solution.function_space()
        assert V == equation.trial_space

        self.tendency = []
        for i in range(self.n_stages):
            k = firedrake.Function(V, name='tendency{:}'.format(i))
            self.tendency.append(k)

        # fully explicit evaluation
        trial = firedrake.TrialFunction(V)
        self.a_rk = self.equation.mass_term(self.test, trial)
        self.l_rk = self.dt_const*self.equation.residual(self.test, self.solution, self.solution, self.fields, bnd_conditions)

        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        if self._nontrivial:
            self.sol_expressions = []
            for i_stage in range(self.n_stages):
                sol_expr = sum(map(operator.mul, self.tendency[:i_stage], self.a[i_stage][:i_stage]))
                self.sol_expressions.append(sol_expr)
            self.final_sol_expr = sum(map(operator.mul, self.tendency, self.b))

        self.update_solver()

    def update_solver(self):
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = firedrake.LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency[i], bcs=self.hom_bcs)
                solver = firedrake.LinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                                           solver_parameters=self.solver_parameters)
                self.solver.append(solver)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        self._initialized = True

    def update_solution(self, i_stage):
        """
        Computes the solution of the i-th stage

        Tendencies must have been evaluated first.

        """
        self.solution.assign(self.solution_old)
        if self._nontrivial and i_stage > 0:
            self.solution += self.sol_expressions[i_stage]

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            self.solver[i_stage].solve()

    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution`
        """
        self.solution.assign(self.solution_old)
        if self._nontrivial:
            self.solution += self.final_sol_expr
        self.solution_old.assign(self.solution)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)


class DIRKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic implementation of Diagonally Implicit Runge Kutta schemes.

    All derived classes must define the Butcher tableau coefficients :attr:`a`,
    :attr:`b`, :attr:`c`.
    """
    def __init__(self, equation, solution, fields, dt,
                 solution_old=None, bnd_conditions=None,
                 solver_parameters={}, strong_bcs=None, terms_to_add='all'):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg solution_old: :class:`Function` where solution at previous timestep
                             is stored. New one will be created if not provided.
        :kwarg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg dict solver_parameters: PETSc solver options
        :kwarg list strong_bcs: list of DirichletsBCs
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(DIRKGeneric, self).__init__(equation, solution, fields, dt,
                                          solution_old, solver_parameters, strong_bcs)
        self.solver_parameters.setdefault('snes_type', 'newtonls')
        self._initialized = False

        fs = solution.function_space()
        assert fs == equation.trial_space

        mixed_space = len(fs) > 1

        # Allocate tendency fields
        self.k = []
        for i in range(self.n_stages):
            fname = '{:}_k{:}'.format(self.name, i)
            self.k.append(firedrake.Function(fs, name=fname))

        # construct variational problems
        self.F = []
        if not mixed_space:
            for i in range(self.n_stages):
                for j in range(i+1):
                    if j == 0:
                        u = self.solution_old + self.a[i][j]*self.dt_const*self.k[j]
                    else:
                        u += self.a[i][j]*self.dt_const*self.k[j]
                self.F.append(self.equation.mass_term(self.test, self.k[i]) -
                              self.equation.residual(self.test, u, self.solution_old, fields, bnd_conditions))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in range(self.n_stages):
                for j in range(i+1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(firedrake.split(self.solution_old), firedrake.split(self.k[j])):
                            u.append(s + self.a[i][j]*self.dt_const*k)
                    else:
                        for l, k in enumerate(firedrake.split(self.k[j])):
                            u[l] += self.a[i][j]*self.dt_const*k
                self.F.append(self.equation.mass_term(self.test, self.k[i]) -
                              self.equation.residual(self.test, u, self.solution_old, fields, bnd_conditions))
        self.update_solver()

        # construct expressions for stage solutions
        self.sol_expressions = []
        for i_stage in range(self.n_stages):
            sol_expr = sum(map(operator.mul, self.k[:i_stage+1], self.dt_const*self.a[i_stage][:i_stage+1]))
            self.sol_expressions.append(sol_expr)
        self.final_sol_expr = self.solution_old + sum(map(operator.mul, self.k, self.dt_const*self.b))

    def update_solver(self):
        """Create solver objects"""
        self.solver = []
        for i in range(self.n_stages):
            p = firedrake.NonlinearVariationalProblem(self.F[i], self.k[i], bcs=self.hom_bcs)
            sname = '{:}_stage{:}_'.format(self.name, i)
            self.solver.append(
                firedrake.NonlinearVariationalSolver(
                    p, solver_parameters=self.solver_parameters,
                    options_prefix=sname))

    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)
        self._initialized = True

    def update_solution(self, i_stage):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        """
        self.solution.assign(self.solution_old + self.sol_expressions[i_stage])

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage.
        """
        if i_stage == 0:
            # NOTE solution may have changed in coupled system
            for bci in self.strong_bcs:
                bci.apply(self.solution)
            self.solution_old.assign(self.solution)
        if not self._initialized:
            raise ValueError('Time integrator {:} is not initialized'.format(self.name))
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()

    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution`"""
        self.solution.assign(self.final_sol_expr)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.solve_tendency(i_stage, t, update_forcings)
        self.update_solution(i_stage)


CFL_UNCONDITIONALLY_STABLE = -1


class AbstractRKScheme(ABC):
    """
    Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """

    @abstractproperty
    def a(self):
        """Runge-Kutta matrix :math:`a_{i,j}` of the Butcher tableau"""
        pass

    @abstractproperty
    def b(self):
        """weights :math:`b_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def c(self):
        """nodes :math:`c_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def cfl_coeff(self):
        """
        CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.
        """
        pass

    def __init__(self):
        super(AbstractRKScheme, self).__init__()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert np.allclose(np.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()


class ForwardEulerAbstract(AbstractRKScheme):
    """
    Forward Euler method
    """
    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ERKLSPUM2Abstract(AbstractRKScheme):
    """
    ERKLSPUM2, 3-stage, 2nd order Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [5.0/6.0, 0, 0],
         [11.0/24.0, 11.0/24.0, 0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [0, 5.0/6.0, 11.0/12.0]
    cfl_coeff = 1.2


class ERKLPUM2Abstract(AbstractRKScheme):
    """
    ERKLPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [1.0/2.0, 0, 0],
         [1.0/2.0, 1.0/2.0, 0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [0, 1.0/2.0, 1.0]
    cfl_coeff = 2.0


class ERKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33Abstract(AbstractRKScheme):
    r"""
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau

    .. math::
        \begin{array}{c|ccc}
            0 &                 \\
            1 & 1               \\
          1/2 & 1/4 & 1/4 &     \\ \hline
              & 1/6 & 1/6 & 2/3
        \end{array}

    CFL coefficient is 1.0
    """
    a = [[0, 0, 0],
         [1.0, 0, 0],
         [0.25, 0.25, 0]]
    b = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class BackwardEulerAbstract(AbstractRKScheme):
    """
    Backward Euler method
    """
    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class ImplicitMidpointAbstract(AbstractRKScheme):
    r"""
    Implicit midpoint method, second order.

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|c}
        0.5 & 0.5 \\ \hline
            & 1.0
        \end{array}

    """
    a = [[0.5]]
    b = [1.0]
    c = [0.5]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class CrankNicolsonAbstract(AbstractRKScheme):
    """
    Crack-Nicolson scheme
    """
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22Abstract(AbstractRKScheme):
    r"""
    2-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
        \gamma &   \gamma &       0 \\
              1 & 1-\gamma & \gamma \\ \hline
                &       1/2 &     1/2
        \end{array}

    with :math:`\gamma = (2 + \sqrt{2})/2`.

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = (2.0 + np.sqrt(2.0))/2.0
    a = [[gamma, 0],
         [1-gamma, gamma]]
    b = [1-gamma, gamma]
    c = [gamma, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK23Abstract(AbstractRKScheme):
    r"""
    2-stage, 3rd order Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
          \gamma &    \gamma &       0 \\
        1-\gamma & 1-2\gamma & \gamma \\ \hline
                  &        1/2 &     1/2
        \end{array}

    with :math:`\gamma = (3 + \sqrt{3})/6`.

    From DIRK(2,3,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = (3 + np.sqrt(3))/6
    a = [[gamma, 0],
         [1-2*gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1-gamma]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK33Abstract(AbstractRKScheme):
    """
    3-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = 0.4358665215
    b1 = -3.0/2.0*gamma**2 + 4*gamma - 1.0/4.0
    b2 = 3.0/2.0*gamma**2 - 5*gamma + 5.0/4.0
    a = [[gamma, 0, 0],
         [(1-gamma)/2, gamma, 0],
         [b1, b2, gamma]]
    b = [b1, b2, gamma]
    c = [gamma, (1+gamma)/2, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK43Abstract(AbstractRKScheme):
    """
    4-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    a = [[0.5, 0, 0, 0],
         [1.0/6.0, 0.5, 0, 0],
         [-0.5, 0.5, 0.5, 0],
         [3.0/2.0, -3.0/2.0, 0.5, 0.5]]
    b = [3.0/2.0, -3.0/2.0, 0.5, 0.5]
    c = [0.5, 2.0/3.0, 0.5, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRKLSPUM2Abstract(AbstractRKScheme):
    """
    DIRKLSPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[2.0/11.0, 0, 0],
         [205.0/462.0, 2.0/11.0, 0],
         [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.82


class DIRKLPUM2Abstract(AbstractRKScheme):
    """
    DIRKLPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[2.0/11.0, 0, 0],
         [41.0/154.0, 2.0/11.0, 0],
         [289.0/847.0, 42.0/121.0, 2.0/11.0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [2.0/11.0, 69.0/154.0, 67.0/77.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.09


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ERKEuler(ERKGeneric, ForwardEulerAbstract):
    pass


class SSPRK33(ERKGeneric, SSPRK33Abstract):
    pass


class BackwardEuler(DIRKGeneric, BackwardEulerAbstract):
    pass


class ImplicitMidpoint(DIRKGeneric, ImplicitMidpointAbstract):
    pass


class CrankNicolsonRK(DIRKGeneric, CrankNicolsonAbstract):
    pass


class DIRK22(DIRKGeneric, DIRK22Abstract):
    pass


class DIRK23(DIRKGeneric, DIRK23Abstract):
    pass


class DIRK33(DIRKGeneric, DIRK33Abstract):
    pass


class DIRK43(DIRKGeneric, DIRK43Abstract):
    pass


class DIRKLSPUM2(DIRKGeneric, DIRKLSPUM2Abstract):
    pass


class DIRKLPUM2(DIRKGeneric, DIRKLPUM2Abstract):
    pass
