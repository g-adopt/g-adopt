r"""This module provides several classes to perform integration of time-dependent
equations. Users choose if they require an explicit or implicit time integrator, and
they instantiate one of the implemented algorithm class, for example, `ERKEuler`, by
providing relevant parameters defined in the parent class (i.e. `ERKGeneric` or
`DIRKGeneric`). Then, they call the `advance` method to request a solver update.

This module includes Irksome integration.
"""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Optional

import firedrake
import numpy as np

from .equations import Equation
from .utility import ensure_constant

# Irksome imports
from irksome import (
    MeshConstant, TimeStepper as IrksomeTimeStepper,
    RadauIIA, GaussLegendre, LobattoIIIA, LobattoIIIC,
    BackwardEuler as IrksomeBackwardEuler, Alexander, QinZhang, PareschiRusso)
from irksome.ButcherTableaux import ButcherTableau


class IrksomeIntegrator:
    """Time integrator using Irksome as the backend.

    This class wraps Irksome's TimeStepper while maintaining G-ADOPT's API
    for compatibility with our existing code.

    Args:
        equation: G-ADOPT equation to integrate
        solution: Firedrake function representing the equation's solution
        dt: Integration time step (Firedrake Constant or float)
        butcher: Irksome Butcher tableau (e.g., GaussLegendre, RadauIIA)
        stage_type: Type of stage formulation (e.g., "deriv", "dirk", "explicit")
        solution_old: Firedrake function representing the equation's solution
                      at the previous timestep
        strong_bcs: List of Firedrake boundary conditions (DirichletBC or EquationBC).
                    Note: EquationBC is only compatible with bc_type="DAE".
        bc_type: Boundary condition type for Irksome ("DAE" or "ODE").
                 Only applies when stage_type="deriv".

                 - "DAE" (default): Differential-Algebraic Equation style BCs.
                   Enforces BCs as constraints, handling incompatible BC + IC gracefully.
                   Supports both DirichletBC and EquationBC.

                 - "ODE": Ordinary Differential Equation style BCs.
                   Takes time derivative of boundary data. Requires compatible BC + IC.
                   Only supports DirichletBC (EquationBC raises NotImplementedError).
                   Only works with splitting=AI (additive implicit), where AI splits the
                   Butcher matrix A as (A, I) with I being the identity matrix. This is
                   the default splitting strategy and reformulates the RK method to have
                   a denser mass matrix with block-diagonal stiffness.
        solver_parameters: Dictionary of solver parameters provided to PETSc
        initial_time: Initial time value (default: 0.0). This initialises the internal
                      time variable that Irksome uses in time-dependent forms.
        adaptive_parameters: Optional dict for adaptive time-stepping (stage_type="deriv" only).
                            Keys: tol, dtmin, dtmax, KI, KP, max_reject, onscale_factor,
                            safety_factor, gamma0_params. See Irksome documentation.
        **irksome_kwargs: Additional keyword arguments passed directly to Irksome's TimeStepper.
                         Examples: splitting, nullspace, transpose_nullspace, near_nullspace,
                         appctx, form_compiler_parameters, etc.

    Note:
        The internal time variable (self.t) is shared with Irksome's TimeStepper.
        Users should manage time externally and pass it to advance(t=...).

        For adaptive time-stepping, the advance() method returns (error, dt_used) tuple
        when adaptive_parameters is provided.

    Example:
        # Basic usage
        integrator = IrksomeIntegrator(eq, T, dt, GaussLegendre(2))

        # With adaptive time-stepping
        integrator = IrksomeIntegrator(
            eq, T, dt, RadauIIA(3),
            adaptive_parameters={"tol": 1e-3, "dtmin": 1e-6, "dtmax": 0.1}
        )

        # With additional Irksome parameters
        integrator = IrksomeIntegrator(
            eq, T, dt, butcher,
            splitting=AI,  # Passed to Irksome
            nullspace=my_nullspace  # Passed to Irksome
        )
    """

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        butcher: ButcherTableau,
        stage_type: str = "deriv",
        solution_old: Optional[firedrake.Function] = None,
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        bc_type: str = "DAE",
        solver_parameters: Optional[dict[str, Any]] = None,
        initial_time: Number = 0.0,
        adaptive_parameters: Optional[dict[str, Any]] = None,
        **irksome_kwargs,
    ):
        self.equation = equation
        self.solution = solution
        self.solution_old = solution_old or firedrake.Function(solution, name=solution.name() + " (old)")

        # Unique identifier used in solver
        self.name = '-'.join([self.__class__.__name__, self.equation.__class__.__name__])

        # Time management: maintain two dt objects for syncing between user and Irksome
        # dt_reference: User's dt (e.g., Firedrake Constant) - what users control via .dt property
        # dt_irksome: Irksome's dt (MeshConstant) - synced from dt_reference before each step
        self.dt_reference = ensure_constant(dt)

        # Create MeshConstant objects for time variables (what Irksome expects)
        # These are shared with Irksome's TimeStepper (ensures synchronisation)
        mc = MeshConstant(equation.mesh)
        self.t = mc.Constant(initial_time)  # Shared time variable with Irksome
        self.dt_irksome = mc.Constant(float(dt))  # Irksome's dt, synced from dt_reference

        # Build the Irksome form directly
        # This constructs the residual form: Dt(u) - residual(u) = 0
        # The mass term applies the time derivative operator Dt internally,
        # allowing each equation to decide how to construct the time derivative
        # (e.g., Dt(solution) or Dt(coeff * solution) for time-dependent coefficients).
        # The negative sign accounts for G-ADOPT's RHS convention.
        # Build the Irksome form
        F = equation.mass(solution) - equation.residual(solution)

        # Store strong_bcs for applying at initialization
        # This ensures BC-consistency like the original G-ADOPT DIRKGeneric
        self.strong_bcs = strong_bcs or []

        # Build kwargs for Irksome TimeStepper
        # Start with g-adopt's standard parameters
        stepper_kwargs = {
            "stage_type": stage_type,
            "bcs": strong_bcs,
            "solver_parameters": solver_parameters,
        }

        # Add bc_type only for stage formulations that support it
        if stage_type == "deriv":
            stepper_kwargs["bc_type"] = bc_type

        # Add adaptive_parameters if provided
        self.is_adaptive = adaptive_parameters is not None
        if self.is_adaptive:
            stepper_kwargs["adaptive_parameters"] = adaptive_parameters

        # Merge in any additional Irksome-specific kwargs
        # This allows users to pass splitting, nullspace, etc.
        stepper_kwargs.update(irksome_kwargs)

        self.stepper = IrksomeTimeStepper(
            F,
            butcher,
            self.t,  # Shared time variable (MeshConstant)
            self.dt_irksome,  # Irksome's dt (MeshConstant, synced from dt_reference)
            solution,
            **stepper_kwargs
        )

    def advance(self, t: float | None = None) -> tuple[float, float] | None:
        """Advance the solution by one time step.

        Args:
            t: Optional current simulation time. If provided, updates the internal time variable
               before advancing. If not provided, uses the current value of self.t.

        Returns:
            When adaptive_parameters are provided: tuple (error, dt_used) where:
                - error: Error estimate from the adaptive stepper
                - dt_used: Actual time step used (may differ from initial dt)
            When adaptive_parameters are not provided: None

        Note:
            Following Irksome's design, this method does NOT automatically update the time variable
            after advancing. Users should manually update time after calling advance():
                # Non-adaptive case:
                integrator.advance(t=current_time)
                current_time += dt

                # Adaptive case:
                result = integrator.advance(t=current_time)
                if result is not None:
                    error, dt_used = result
                    current_time += dt_used
            This ensures time synchronisation between g-adopt and Irksome's internal state.

            When adaptive timestepping is enabled, Irksome updates dt_irksome internally.
            This method syncs dt_irksome back to dt_reference so the .dt property returns
            the actual dt used.

            For time-dependent forcings, include time-dependent expressions directly in your
            UFL form using the time variable `t` (e.g., `sin(t)`, `exp(-t)`, etc.), or use
            Firedrake's `ExternalOperator` for complex dependencies.
        """
        # Apply boundary conditions
        for bci in self.strong_bcs:
            bci.apply(self.solution)

        # Save current solution to solution_old before advancing
        self.solution_old.assign(self.solution)

        # Sync dt from user to Irksome before advancing
        # This ensures Irksome uses the current dt value (in case user modified dt_reference)
        self.dt_irksome.assign(self.dt_reference)

        # Update internal time if provided by user
        # This ensures Irksome uses the correct time during this advance() call
        if t is not None:
            self.t.assign(ensure_constant(t))

        # Advance using Irksome
        # Note: Irksome uses self.t internally but does not modify it
        # The time used during stages is: t + c[i] * dt
        result = self.stepper.advance()

        # Handle adaptive timestepping return value
        if self.is_adaptive:
            # Irksome returns (error, dt_used) tuple when adaptive is enabled
            adapt_error, adapt_dt = result

            # Sync dt from Irksome back to user
            # (Irksome updated dt_irksome internally during adaptive step)
            self.dt_reference.assign(float(adapt_dt))

            # Return tuple so users can track the actual dt used
            return (adapt_error, float(adapt_dt))

        # Non-adaptive: return None for consistency
        return

    @property
    def time(self) -> float:
        """Get the current value of the internal time variable.

        Returns:
            The current time.
        """
        return self.t

    @property
    def dt(self) -> float:
        """Get the current value of the time step from dt_reference.

        Returns:
            The current time step.
        """
        return self.dt_reference


def create_custom_tableau(a, b, c):
    """Create a custom Irksome ButcherTableau from arrays.

    Args:
        a: Butcher matrix
        b: weights
        c: nodes

    Returns:
        An Irksome ButcherTableau instance
    """
    return ButcherTableau(
        A=np.array(a),
        b=np.array(b),
        btilde=None,
        c=np.array(c),
        order=len(b),  # Estimate order from number of stages
        embedded_order=None,
        gamma0=None
    )


class RKGeneric(IrksomeIntegrator):
    """Generic Runge-Kutta time integrator using Irksome.

    This base class constructs the Butcher tableau from the a, b, c class attributes
    inherited from AbstractRKScheme subclasses. Subclasses should set the stage_type
    class attribute to specify the formulation ("explicit", "dirk", or "deriv").
    """

    stage_type = "deriv"  # Default stage type, can be overridden in subclasses

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs,
    ):
        # Create Butcher tableau from instance attributes (inherited from Abstract class)
        butcher = create_custom_tableau(self.a, self.b, self.c)

        # Initialise with Irksome backend
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type=self.stage_type,
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs
        )


class ERKGeneric(RKGeneric):
    """Generic explicit Runge-Kutta time integrator using Irksome."""
    stage_type = "explicit"


class DIRKGeneric(RKGeneric):
    """Generic diagonally implicit Runge-Kutta time integrator using Irksome."""
    stage_type = "dirk"


class StaticButcherTableauIntegrator(IrksomeIntegrator):
    """Time integrator using a pre-built Irksome Butcher tableau.

    This base class is for schemes that have direct Irksome equivalents.
    Subclasses should set:
    - butcher_tableau: An Irksome ButcherTableau instance (e.g., IrksomeBackwardEuler())
    - stage_type: The stage formulation type (e.g., "dirk")
    """

    butcher_tableau = None  # Must be set in subclasses
    stage_type = "dirk"  # Default stage type

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs,
    ):
        if self.butcher_tableau is None:
            raise ValueError(
                f"{self.__class__.__name__} must define a butcher_tableau class attribute"
            )

        # Initialise with Irksome backend using the pre-built tableau
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=self.butcher_tableau,
            stage_type=self.stage_type,
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs
        )


CFL_UNCONDITIONALLY_STABLE = -1


class AbstractRKScheme(ABC):
    """Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """

    @property
    @abstractmethod
    def a(self):
        """Runge-Kutta matrix :math:`a_{i,j}` of the Butcher tableau"""
        pass

    @property
    @abstractmethod
    def b(self):
        """weights :math:`b_{i}` of the Butcher tableau"""
        pass

    @property
    @abstractmethod
    def c(self):
        """nodes :math:`c_{i}` of the Butcher tableau"""
        pass

    @property
    @abstractmethod
    def cfl_coeff(self):
        """CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.

        """
        pass

    def __init__(self, *args, **kwargs):
        super(AbstractRKScheme, self).__init__(*args, **kwargs)
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert np.allclose(np.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()


def shu_osher_butcher(alpha_or_lambda, beta_or_mu):
    """
    Generate arrays composing the Butcher tableau of a Runge-Kutta method from the
    coefficient arrays of the equivalent, original or modified, Shu-Osher form.
    Code adapted from RK-Opt written in MATLAB by David Ketcheson.
    See also Ketcheson, Macdonald, and Gottlieb (2009).

    Function arguments:
    alpha_or_lambda : array_like, shape (n + 1, n)
    beta_or_mu : array_like, shape (n + 1, n)
    """

    X = np.identity(alpha_or_lambda.shape[1]) - alpha_or_lambda[:-1]
    A = np.linalg.solve(X, beta_or_mu[:-1])
    b = np.transpose(beta_or_mu[-1] + np.dot(alpha_or_lambda[-1], A))
    c = np.sum(A, axis=1)
    return A, b, c


class ERKEuler(AbstractRKScheme, ERKGeneric):
    """Forward Euler method"""
    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ERKLSPUM2(AbstractRKScheme, ERKGeneric):
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


class ERKLPUM2(AbstractRKScheme, ERKGeneric):
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


class ERKMidpoint(AbstractRKScheme, ERKGeneric):
    a = [[0.0, 0.0],
         [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33(AbstractRKScheme, ERKGeneric):
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


class eSSPRKs3p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [[0.0, 0.0, 0.0], [2 / 3, 0.0, 0.0], [2 / 9, 4 / 9, 0.0]]
    b = [0.25, 0.1875, 0.5625]
    c = [0, 2 / 3, 2 / 3]
    cfl_coeff = 3 / 4


class eSSPRKs4p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0],
        [11 / 20, 0.0, 0.0, 0.0],
        [11 / 32, 11 / 32, 0.0, 0.0],
        [55 / 288, 55 / 288, 11 / 36, 0.0],
    ]
    b = [0.24517906, 0.13774105, 0.22038567, 0.39669421]
    c = [0, 11 / 20, 11 / 16, 11 / 16]
    cfl_coeff = 20 / 11


class eSSPRKs5p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.37949799, 0.0, 0.0, 0.0, 0.0],
        [0.35866028, 0.35866028, 0.0, 0.0, 0.0],
        [0.23456423, 0.23456423, 0.24819211, 0.0, 0.0],
        [0.15340527, 0.15340527, 0.16231792, 0.24819211, 0.0],
    ]
    b = [0.20992362, 0.1975535, 0.1217419, 0.18614938, 0.28463159]
    c = [0.0, 0.37949799, 0.71732056, 0.71732057, 0.71732057]
    cfl_coeff = 2.63506005


class eSSPRKs6p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.28422072, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.28422072, 0.28422072, 0.0, 0.0, 0.0, 0.0],
        [0.23301578, 0.23301578, 0.23301578, 0.0, 0.0, 0.0],
        [0.16684082, 0.16532461, 0.16532461, 0.20165449, 0.0, 0.0],
        [0.21178186, 0.102324, 0.10202706, 0.12444738, 0.17540162, 0.0],
    ]
    b = [0.21181784, 0.10241434, 0.10198818, 0.12438557, 0.17531451, 0.28407956]
    c = [0.0, 0.28422072, 0.56844144, 0.69904734, 0.69914453, 0.71598192]
    cfl_coeff = 3.51839231


class eSSPRKs7p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23333473, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23333473, 0.23333473, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23144338, 0.23144338, 0.23144338, 0.0, 0.0, 0.0, 0.0],
        [0.17322863, 0.17322863, 0.17322863, 0.17464425, 0.0, 0.0, 0.0],
        [0.13071968, 0.12941249, 0.12941249, 0.13047004, 0.17431545, 0.0, 0.0],
        [0.16655731, 0.16570664, 0.08421603, 0.08490424, 0.11343693, 0.15184412, 0.0],
    ]
    b = [
        0.16655731,
        0.16570664,
        0.08421603,
        0.08490424,
        0.11343693,
        0.15184412,
        0.23333473,
    ]
    c = [0.0, 0.23333473, 0.46666946, 0.69433014, 0.69433014, 0.69433015, 0.76666527]
    cfl_coeff = 4.28568865


class eSSPRKs8p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.19580402, 0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.15369244, 0.15369244, 0.15369244, 0.15369244, 0.0, 0.0, 0.0, 0.0],
        [0.11656615, 0.11656615, 0.11656615, 0.11656615, 0.14850516, 0.0, 0.0, 0.0],
        [
            0.12960593,
            0.09738344,
            0.09738344,
            0.09738344,
            0.12406641,
            0.16358153,
            0.0,
            0.0,
        ],
        [
            0.12970594,
            0.09753214,
            0.09723632,
            0.09723632,
            0.12387897,
            0.16333439,
            0.1955082,
            0.0,
        ],
    ]
    b = [
        0.1462899,
        0.12218849,
        0.12196689,
        0.10127077,
        0.09279782,
        0.1223539,
        0.14645532,
        0.14667691,
    ]
    c = [
        0.0,
        0.19580402,
        0.39160804,
        0.58741206,
        0.61476976,
        0.61476976,
        0.70940419,
        0.90443228,
    ]
    cfl_coeff = 5.10714756


class eSSPRKs9p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.13333333,
            0.13333333,
            0.13333333,
            0.13333333,
            0.13333333,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [0.14166667, 0.1, 0.1, 0.1, 0.1, 0.125, 0.0, 0.0, 0.0],
        [
            0.15,
            0.12222222,
            0.06666667,
            0.06666667,
            0.06666667,
            0.08333333,
            0.11111111,
            0.0,
            0.0,
        ],
        [
            0.15,
            0.12222222,
            0.06666667,
            0.06666667,
            0.06666667,
            0.08333333,
            0.11111111,
            0.16666667,
            0.0,
        ],
    ]
    b = [
        0.15,
        0.12222222,
        0.06666667,
        0.06666667,
        0.06666667,
        0.08333333,
        0.11111111,
        0.16666667,
        0.16666667,
    ]
    c = [
        0.0,
        0.16666667,
        0.33333334,
        0.50000001,
        0.66666668,
        0.66666665,
        0.66666667,
        0.66666667,
        0.83333334,
    ]
    cfl_coeff = 6.0


class eSSPRKs10p3(AbstractRKScheme, ERKGeneric):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.11790205,
            0.11790205,
            0.11790205,
            0.11790205,
            0.11790205,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.10906732,
            0.10906703,
            0.10906703,
            0.10906703,
            0.10906703,
            0.13633378,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.11862231,
            0.11856848,
            0.08186453,
            0.08186453,
            0.08186453,
            0.10233067,
            0.11062,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.12708168,
            0.12704369,
            0.08137218,
            0.0577812,
            0.0577812,
            0.07222649,
            0.07807723,
            0.10402125,
            0.0,
            0.0,
        ],
        [
            0.1270886,
            0.12705062,
            0.08139469,
            0.05776149,
            0.05776149,
            0.07220186,
            0.07805061,
            0.10398578,
            0.1473273,
            0.0,
        ],
    ]
    b = [
        0.1270886,
        0.12705062,
        0.08139469,
        0.05776149,
        0.05776149,
        0.07220186,
        0.07805061,
        0.10398578,
        0.1473273,
        0.14737756,
    ]
    c = [
        0.0,
        0.14737756,
        0.29475512,
        0.44213268,
        0.58951024,
        0.58951025,
        0.68166922,
        0.69573505,
        0.70538492,
        0.85262244,
    ]
    cfl_coeff = 6.78529356


class BackwardEuler(StaticButcherTableauIntegrator):
    """Backward Euler scheme using Irksome's built-in implementation."""
    butcher_tableau = IrksomeBackwardEuler()
    stage_type = "dirk"
    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class ImplicitMidpoint(StaticButcherTableauIntegrator):
    """Implicit midpoint scheme using Irksome's GaussLegendre(1) implementation."""
    butcher_tableau = GaussLegendre(1)
    stage_type = "dirk"
    a = [[0.5]]
    b = [1.0]
    c = [0.5]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class CrankNicolsonRK(AbstractRKScheme, DIRKGeneric):
    """
    Crank-Nicolson scheme
    """
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22(AbstractRKScheme, DIRKGeneric):
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


class DIRK23(AbstractRKScheme, DIRKGeneric):
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


class DIRK33(AbstractRKScheme, DIRKGeneric):
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


class DIRK43(AbstractRKScheme, DIRKGeneric):
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


class DIRKLSPUM2(AbstractRKScheme, DIRKGeneric):
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


class DIRKLPUM2(AbstractRKScheme, DIRKGeneric):
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


# Alias for backwards compatibility
ForwardEuler = ERKEuler


class IrksomeRadauIIA(IrksomeIntegrator):
    """Direct access to Irksome's RadauIIA scheme."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        order: int = 3,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs  # Accept additional kwargs
    ):
        # Create Irksome RadauIIA tableau
        butcher = RadauIIA(order)

        # Initialise with Irksome backend (RadauIIA is fully implicit)
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type="deriv",  # "deriv" for fully implicit schemes
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs  # Pass adaptive_parameters and other kwargs to parent
        )


class IrksomeGaussLegendre(IrksomeIntegrator):
    """Direct access to Irksome's GaussLegendre scheme."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        order: int = 2,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs  # Accept additional kwargs for compatibility
    ):
        # Create Irksome GaussLegendre tableau
        butcher = GaussLegendre(order)

        # Initialise with Irksome backend (GaussLegendre is fully implicit )
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type="deriv",  # "deriv" for fully implicit schemes
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs  # Pass adaptive_parameters and other kwargs to parent
        )


class IrksomeLobattoIIIA(IrksomeIntegrator):
    """Direct access to Irksome's LobattoIIIA scheme.

    Note: LobattoIIIA requires ODE-style boundary conditions because its
    Butcher matrix cannot be inverted (required for DAE-style BCs).
    """

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        order: int = 2,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs,
    ):
        # Create Irksome LobattoIIIA tableau
        butcher = LobattoIIIA(order)

        # Initialise with Irksome backend
        # LobattoIIIA is a collocation method - use "deriv" for adaptive support
        # Must use ODE-style BCs because the Butcher matrix is not invertible
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type="deriv",
            bc_type="ODE",  # LobattoIIIA requires ODE-style BCs
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs,
        )


class IrksomeLobattoIIIC(IrksomeIntegrator):
    """Direct access to Irksome's LobattoIIIC scheme."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        order: int = 2,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs,
    ):
        # Create Irksome LobattoIIIC tableau
        butcher = LobattoIIIC(order)

        # Initialise with Irksome backend (LobattoIIIC is fully implicit)
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type="deriv",  # for fully implicit schemes
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs  # Pass adaptive_parameters and other kwargs to parent
        )


class IrksomeAlexander(StaticButcherTableauIntegrator):
    """Direct access to Irksome's Alexander scheme."""
    butcher_tableau = Alexander()
    stage_type = "dirk"


class IrksomeQinZhang(StaticButcherTableauIntegrator):
    """Direct access to Irksome's QinZhang scheme."""
    butcher_tableau = QinZhang()
    stage_type = "dirk"


class IrksomePareschiRusso(IrksomeIntegrator):
    """Direct access to Irksome's PareschiRusso scheme."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        x: float = 0.5,  # Default value for PareschiRusso parameter
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = {},
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
        **kwargs  # Accept additional kwargs compatibility
    ):
        # Create Irksome PareschiRusso tableau
        butcher = PareschiRusso(x)

        # Initialise with Irksome backend
        super().__init__(
            equation=equation,
            solution=solution,
            dt=dt,
            butcher=butcher,
            stage_type="dirk",
            solution_old=solution_old,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
            **kwargs  # Pass additional kwargs to parent
        )
