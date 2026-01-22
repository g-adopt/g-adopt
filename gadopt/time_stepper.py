"""This module provides several classes to perform integration of time-dependent
equations via Irksome. Users choose if they require an explicit or diagonally implicit
time integrator, and they instantiate one of the implemented algorithm classes, for
example, `ForwardEuler`, by providing relevant parameters defined in `RKGeneric`. Then,
they call the `advance` method to request a solver update.
"""

from abc import ABC, abstractmethod
from typing import Any

import firedrake as fd
import numpy as np
from irksome import MeshConstant, TimeStepper
from irksome.ButcherTableaux import (
    Alexander,
    BackwardEuler,
    ButcherTableau,
    GaussLegendre,
    LobattoIIIA,
    LobattoIIIC,
    PareschiRusso,
    QinZhang,
    RadauIIA,
)

from .equations import Equation
from .utility import ensure_constant


class IrksomeIntegrator:
    """Time integrator using Irksome as the backend.

    Args:
        equation:
            G-ADOPT equation or list thereof or UFL form to integrate
        solution:
            Firedrake function representing the equation's solution
        dt:
            Integration time step
        butcher_tableau:
            Irksome Butcher tableau (e.g. `GaussLegendre`, `RadauIIA`)
        stage_type:
            Type of stage formulation (e.g. `"deriv"`, `"dirk"`, `"explicit"`)
        solution_old:
            Firedrake function for the equation's solution at the previous timestep
        strong_bcs:
            List of Firedrake boundary conditions (`DirichletBC` or `EquationBC`).
            `EquationBC` is only compatible with `bc_type="DAE"`.
        bc_type:
            Boundary condition type for Irksome ("DAE" or "ODE"). Only applies when
            `stage_type="deriv"`.
            - `"DAE"` (default): Differential-Algebraic Equation style BCs.
                Enforces BCs as constraints, handling incompatible BC + IC gracefully.
                Supports both `DirichletBC` and `EquationBC`.
            - `"ODE"`: Ordinary Differential Equation style BCs.
                Takes time derivative of boundary data. Requires compatible BC + IC.
                Only supports `DirichletBC` (`EquationBC` raises `NotImplementedError`).
                Only works with `splitting=AI` (additive implicit), where `AI` splits
                the Butcher matrix A as (A, I), with I the identity matrix. This is the
                default splitting strategy; it reformulates the RK method to have a
                denser mass matrix with block-diagonal stiffness.
        solver_parameters:
            Dictionary of solver parameters provided to PETSc
        adaptive_parameters:
            Optional dict for adaptive time-stepping (`stage_type="deriv"` only).
            - `tol`
            - `dtmin`
            - `dtmax`
            - `KI`
            - `KP`
            - `max_reject`
            - `onscale_factor`
            - `safety_factor`
            - `gamma0_params`
        **irksome_kwargs:
            Additional keyword arguments passed directly to Irksome's `TimeStepper`.
            - `splitting`
            - `nullspace`
            - `transpose_nullspace`
            - `near_nullspace`
            - `appctx`
            - `form_compiler_parameters`

    **Note**:
        Irksome's `TimeStepper` does not update time, nor does `IrksomeIntegrator`.
        Users should manage time externally.

        For adaptive time-stepping (i.e. when `adaptive_parameters` is provided), the
        `advance` method returns `(adapt_error, adapt_dt)` tuple. Otherwise, it returns
        `None`.

    Examples:
        ```
        # Basic usage (here, GaussLegendre comes from Irksome, not G-ADOPT)
        integrator = IrksomeIntegrator(eq, T, dt, GaussLegendre(2))

        # With adaptive time-stepping
        integrator = IrksomeIntegrator(
            eq,
            T,
            dt,
            RadauIIA(3),  # Irksome class, not G-ADOPT equivalent
            adaptive_parameters={"tol": 1e-3, "dtmin": 1e-6, "dtmax": 0.1},
        )

        # With additional Irksome parameters
        integrator = IrksomeIntegrator(
            eq, T, dt, butcher_tableau, splitting=AI, nullspace=my_nullspace
        )
        ```
    """

    def __init__(
        self,
        equation: Equation,
        solution: fd.Function,
        dt: fd.Function,
        butcher_tableau: ButcherTableau,
        *,
        stage_type: str = "deriv",
        solution_old: fd.Function | None = None,
        strong_bcs: list[fd.DirichletBC] | None = None,
        bc_type: str = "DAE",
        solver_parameters: dict[str, Any] | None = None,
        initial_time: float = 0.0,
        adaptive_parameters: dict[str, Any] | None = None,
        **irksome_kwargs,
    ):
        # Unique solver identifier
        self.name = "-".join([self.__class__.__name__, equation.__class__.__name__])

        self.solution = solution
        self.solution_old = solution_old or fd.Function(
            solution, name=solution.name() + " (old)"
        )
        # Time management: maintain two dt objects for syncing between user and Irksome
        # dt_reference: User's dt (e.g., Firedrake Constant) - what users control via .dt property
        # dt_irksome: Irksome's dt (MeshConstant) - synced from dt_reference before each step
        self.dt_reference = ensure_constant(dt)
        # Create MeshConstant objects for time variables (what Irksome expects)
        # These are shared with Irksome's TimeStepper (ensures synchronisation)
        mc = MeshConstant(equation.mesh)
        self.t = mc.Constant(initial_time)  # Shared time variable with Irksome
        self.dt_irksome = mc.Constant(dt)  # Irksome's dt, synced from dt_reference
        # Store Dirichlet conditions for application before advancing the integrator
        self.strong_bcs = strong_bcs or []

        # Irksome form; the negative residual sign is conventional in G-ADOPT.
        # The mass term provided to `Equation` must employ the time derivative operator
        # `Dt`, allowing an equation-specific implementation of the time derivative
        # term.
        if isinstance(equation, fd.Form):
            F = equation
        elif isinstance(equation, list):
            F = 0.0
            for eq, sol in zip(equation, fd.split(solution)):
                if eq.mass_term is not None:
                    F += eq.mass(sol)
                F -= eq.residual(sol)
        else:
            F = equation.mass(solution) - equation.residual(solution)

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
            if stage_type != "deriv":
                raise ValueError(
                    "A method with stage_type=deriv is required for adaptive_parameters"
                )
            stepper_kwargs["adaptive_parameters"] = adaptive_parameters
        # Merge in any additional Irksome-specific kwargs, such as splitting
        stepper_kwargs.update(irksome_kwargs)

        self.stepper = TimeStepper(
            F, butcher_tableau, self.t, self.dt_irksome, solution, **stepper_kwargs
        )

    def advance(self, t: float | None = None) -> tuple[float, float] | None:
        """Advance the solution by one time step.

        When adaptive timestepping is enabled, Irksome updates the time step internally.

        Args:
            t:
                Optional current simulation time. If provided, updates the internal time
                variable before advancing. If not provided, uses the current value of
                self.t.

        Returns:
            Either `None` if `adaptive_parameters` is omitted or
            `(adapt_error, adapt_dt)` if it is provided, in which case we have
            - `adapt_error`: Error estimate from the adaptive stepper
            - `adapt_dt`: Actual time step used (may differ from expected time step)

        **Note**:
            Following Irksome's design, this method does NOT automatically update the
            time variable after advancing. This strategy allows multiple Irksome
            integrators to share the same time and time-step variable. Therefore, users
            should manually update time after calling `advance`:

            ```
            # Non-adaptive case:
            integrator.advance()  # Advance all integrators that share the same time
            t.assign(t + dt)  # t and dt are generated by the user

            # Adaptive case:
            adapt_error, adapt_dt = integrator.advance()  # Advance single integrator
            t.assign(t + adapt_dt)  # Use actual adapted time step to increment time
            ```

            Updating time is essential when the UFL Form contains time-dependent
            forcings whose expressions directly involve the time variable `t` (e.g.
            `sin(t)`, `exp(-t)`, etc...). To include complex dependencies beyond an
            explicit time-variable dependency, Firedrake's `ExternalOperator` should be
            used.
        """
        # Apply boundary conditions
        for bci in self.strong_bcs:
            bci.apply(self.solution)

        # Save current solution to solution_old before advancing
        self.solution_old.assign(self.solution)

        # Sync dt from user to Irksome before advancing
        # This ensures Irksome uses the current dt value (in case user modified
        # dt_reference)
        self.dt_irksome.assign(self.dt_reference)

        # Update internal time if provided by user
        # This ensures Irksome uses the correct time during this advance() call
        if t is not None:
            self.t.assign(t)

        # Advance using Irksome
        # Note: Irksome uses self.t internally but does not modify it
        # The time used during stages is: t + c[i] * dt
        result = self.stepper.advance()

        # Handle adaptive timestepping return value
        if self.is_adaptive:
            # Irksome returns (error, dt_used) tuple when adaptive is enabled
            # Note: adapt_dt is the dt that was USED in this step
            # self.dt_irksome now contains the NEW recommended dt for next step
            adapt_error, adapt_dt = result

            # Sync dt from Irksome back to user
            # Use dt_irksome (new recommendation), not adapt_dt (old used value)
            self.dt_reference.assign(self.dt_irksome)

            # Return the new recommended dt for the G-ADOPT usage pattern:
            # error, dt = solver.solve(); delta_t.assign(dt)
            return adapt_error, float(self.dt_irksome)

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


class RKGeneric(IrksomeIntegrator):
    """Generic Runge-Kutta time integrator using Irksome.

    Args:
        equation:
            G-ADOPT equation to integrate
        solution:
            Firedrake function representing the equation's solution
        t:
            Integration time (Firedrake Function)
        dt:
            Integration time step (Firedrake Function)
        tableau_parameter:
            Parameter to initialise an Irksome Butcher tableau (e.g. `GaussLegendre`)
        **kwargs:
            Additional keyword arguments (see `IrksomeIntegrator`)

    Subclasses must set the `butcher_tableau` class attribute either directly from
    Irksome or via a subclass of `AbstractRKScheme` that defines the `a`, `b`, and `c`
    class attributes. Note that Irksome tableaux may accept a parameter, such as the
    number of stages used, which can be provided here via the `tableau_parameter`
    argument. Subclasses should also set the `stage_type` class attribute to specify the
    formulation (e.g. "explicit", "dirk", "deriv", etc...).
    """

    butcher_tableau = None  # Must be set in subclasses
    stage_type = "deriv"  # Default stage type, can be overridden in subclasses
    bc_type = "DAE"  # Default boundary condition type, can be overridden in subclasses

    def __init__(
        self,
        equation: Equation,
        solution: fd.Function,
        dt: fd.Function,
        tableau_parameter: int | float | None = None,
        **kwargs,
    ):
        if tableau_parameter is None:
            tableau_parameter = getattr(self, "tableau_parameter", None)
        if tableau_parameter is not None:
            self.butcher_tableau = self.butcher_tableau(tableau_parameter)

        if self.butcher_tableau is None:
            raise ValueError(
                f"{self.__class__.__name__} must define a butcher_tableau attribute"
            )

        super().__init__(
            equation,
            solution,
            dt,
            self.butcher_tableau,
            stage_type=self.stage_type,
            bc_type=self.bc_type,
            **kwargs,
        )


class ERKGeneric(RKGeneric):
    """Generic explicit Runge-Kutta time integrator."""

    stage_type = "explicit"


class DIRKGeneric(RKGeneric):
    """Generic diagonally implicit Runge-Kutta time integrator."""

    stage_type = "dirk"


class CRKGeneric(RKGeneric):
    """Generic collocation Runge-Kutta time integrator."""

    stage_type = "deriv"


CFL_UNCONDITIONALLY_STABLE = -1


def create_custom_tableau(
    a: list[list[float]], b: list[float], c: list[float]
) -> ButcherTableau:
    """Create a custom Irksome ButcherTableau from relevant arrays.

    Args:
        a: Butcher matrix
        b: weights
        c: nodes

    Returns:
        An Irksome ButcherTableau instance
    """
    if not np.allclose(np.sum(a, axis=1), c):
        raise ValueError("Inconsistent Butcher tableau: Row sum of 'a' is not 'c'")

    return ButcherTableau(
        A=a, b=b, btilde=None, c=c, order=len(b), embedded_order=None, gamma0=None
    )


def shu_osher_butcher(
    alpha_or_lambda: np.ndarray, beta_or_mu: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the Butcher tableau of a Runge-Kutta method from the Shu-Osher form.

    Arrays composing the Butcher tableau of a Runge-Kutta method are derived from the
    coefficient arrays of the equivalent, original or modified, Shu-Osher form.

    Code adapted from RK-Opt written in MATLAB by David Ketcheson. See also Ketcheson,
    Macdonald, and Gottlieb (2009, https://doi.org/10.1016/j.apnum.2008.03.034).

    Args:
      alpha_or_lambda: array_like, shape (n + 1, n)
      beta_or_mu: array_like, shape (n + 1, n)
    """

    X = np.identity(alpha_or_lambda.shape[1]) - alpha_or_lambda[:-1]
    A = np.linalg.solve(X, beta_or_mu[:-1])
    b = np.transpose(beta_or_mu[-1] + np.dot(alpha_or_lambda[-1], A))
    c = np.sum(A, axis=1)

    return A, b, c


class AbstractRKScheme(ABC):
    """Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays `a`, `b`, and `c`) and the
    CFL number (`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """

    def __init_subclass__(cls):
        if cls.__name__ == "eSSPRK":
            return

        cls.butcher_tableau = create_custom_tableau(cls.a, cls.b, cls.c)

        if cls.butcher_tableau.is_fully_implicit:
            raise ValueError(
                "Butcher tableau is neither explicit nor diagonally implicit"
            )

    @property
    @abstractmethod
    def a(cls):
        """Runge-Kutta matrix `a_{i,j}` of the Butcher tableau"""

    @property
    @abstractmethod
    def b(cls):
        """weights `b_{i}` of the Butcher tableau"""

    @property
    @abstractmethod
    def c(cls):
        """nodes `c_{i}` of the Butcher tableau"""

    @property
    @abstractmethod
    def cfl_coeff(cls):
        """CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.
        """


class ForwardEuler(AbstractRKScheme, ERKGeneric):
    """Forward Euler method"""

    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ERKLSPUM2(AbstractRKScheme, ERKGeneric):
    """ERKLSPUM2, 3-stage, 2nd order, explicit Runge-Kutta method

    From IMEX RK scheme (17) in Higueras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [[0, 0, 0], [5.0 / 6.0, 0, 0], [11.0 / 24.0, 11.0 / 24.0, 0]]
    b = [24.0 / 55.0, 1.0 / 5.0, 4.0 / 11.0]
    c = [0, 5.0 / 6.0, 11.0 / 12.0]
    cfl_coeff = 1.2


class ERKLPUM2(AbstractRKScheme, ERKGeneric):
    """ERKLPUM2, 3-stage, 2nd order, explicit Runge-Kutta method

    From IMEX RK scheme (20) in Higueras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [[0, 0, 0], [1.0 / 2.0, 0, 0], [1.0 / 2.0, 1.0 / 2.0, 0]]
    b = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    c = [0, 1.0 / 2.0, 1.0]
    cfl_coeff = 2.0


class Midpoint(AbstractRKScheme, ERKGeneric):
    a = [[0.0, 0.0], [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33(AbstractRKScheme, ERKGeneric):
    r"""3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau

    $$
        \begin{array}{c|ccc}
            0 &                 \\
            1 & 1               \\
          1/2 & 1/4 & 1/4 &     \\ \hline
              & 1/6 & 1/6 & 2/3
        \end{array}
    $$

    CFL coefficient is 1.0
    """

    a = [[0, 0, 0], [1.0, 0, 0], [0.25, 0.25, 0]]
    b = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class eSSPRK(AbstractRKScheme, ERKGeneric):
    """Assemble explicit Runge-Kutta matrix based on non-zero entries."""

    def __init_subclass__(cls):
        cls.a.insert(0, [0.0] * (len(cls.a) + 1))
        for row in cls.a:
            row += [0.0] * (len(cls.a) - len(row))

        super().__init_subclass__()


class eSSPRKs3p3(eSSPRK):
    """3rd order, 3-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [[2 / 3], [2 / 9, 4 / 9]]
    b = [0.25, 0.1875, 0.5625]
    c = [0, 2 / 3, 2 / 3]
    cfl_coeff = 3 / 4


class eSSPRKs4p3(eSSPRK):
    """3rd order, 4-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [[11 / 20], [11 / 32, 11 / 32], [55 / 288, 55 / 288, 11 / 36]]
    b = [0.24517906, 0.13774105, 0.22038567, 0.39669421]
    c = [0, 11 / 20, 11 / 16, 11 / 16]
    cfl_coeff = 20 / 11


class eSSPRKs5p3(eSSPRK):
    """3rd order, 5-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.37949799],
        [0.35866028, 0.35866028],
        [0.23456423, 0.23456423, 0.24819211],
        [0.15340527, 0.15340527, 0.16231792, 0.24819211],
    ]
    b = [0.20992362, 0.1975535, 0.1217419, 0.18614938, 0.28463159]
    c = [0.0, 0.37949799, 0.71732056, 0.71732057, 0.71732057]
    cfl_coeff = 2.63506005


class eSSPRKs6p3(eSSPRK):
    """3rd order, 6-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.28422072],
        [0.28422072, 0.28422072],
        [0.23301578, 0.23301578, 0.23301578],
        [0.16684082, 0.16532461, 0.16532461, 0.20165449],
        [0.21178186, 0.102324, 0.10202706, 0.12444738, 0.17540162],
    ]
    b = [0.21181784, 0.10241434, 0.10198818, 0.12438557, 0.17531451, 0.28407956]
    c = [0.0, 0.28422072, 0.56844144, 0.69904734, 0.69914453, 0.71598192]
    cfl_coeff = 3.51839231


class eSSPRKs7p3(eSSPRK):
    """3rd order, 7-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.23333473],
        [0.23333473, 0.23333473],
        [0.23144338, 0.23144338, 0.23144338],
        [0.17322863, 0.17322863, 0.17322863, 0.17464425],
        [0.13071968, 0.12941249, 0.12941249, 0.13047004, 0.17431545],
        [0.16655731, 0.16570664, 0.08421603, 0.08490424, 0.11343693, 0.15184412],
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


class eSSPRKs8p3(eSSPRK):
    """3rd order, 8-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.19580402],
        [0.19580402, 0.19580402],
        [0.19580402, 0.19580402, 0.19580402],
        [0.15369244, 0.15369244, 0.15369244, 0.15369244],
        [0.11656615, 0.11656615, 0.11656615, 0.11656615, 0.14850516],
        [0.12960593, 0.09738344, 0.09738344, 0.09738344, 0.12406641, 0.16358153],
        [
            0.12970594,
            0.09753214,
            0.09723632,
            0.09723632,
            0.12387897,
            0.16333439,
            0.1955082,
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


class eSSPRKs9p3(eSSPRK):
    """3rd order, 9-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.16666667],
        [0.16666667, 0.16666667],
        [0.16666667, 0.16666667, 0.16666667],
        [0.16666667, 0.16666667, 0.16666667, 0.16666667],
        [0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333],
        [0.14166667, 0.1, 0.1, 0.1, 0.1, 0.125],
        [0.15, 0.12222222, 0.06666667, 0.06666667, 0.06666667, 0.08333333, 0.11111111],
        [
            0.15,
            0.12222222,
            0.06666667,
            0.06666667,
            0.06666667,
            0.08333333,
            0.11111111,
            0.16666667,
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


class eSSPRKs10p3(eSSPRK):
    """3rd order, 10-stage, explicit, strong-stability-preserving Runge-Kutta method.

    This method has a nondecreasing-abscissa condition.
    See Isherwood, Grant, and Gottlieb (2018, https://doi.org/10.1137/17M1143290).
    """

    a = [
        [0.14737756],
        [0.14737756, 0.14737756],
        [0.14737756, 0.14737756, 0.14737756],
        [0.14737756, 0.14737756, 0.14737756, 0.14737756],
        [0.11790205, 0.11790205, 0.11790205, 0.11790205, 0.11790205],
        [0.10906732, 0.10906703, 0.10906703, 0.10906703, 0.10906703, 0.13633378],
        [
            0.11862231,
            0.11856848,
            0.08186453,
            0.08186453,
            0.08186453,
            0.10233067,
            0.11062,
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


class BackwardEuler(DIRKGeneric):
    """Backward Euler scheme using Irksome's built-in implementation."""

    butcher_tableau = BackwardEuler()


class ImplicitMidpoint(DIRKGeneric):
    """Implicit midpoint scheme using Irksome's GaussLegendre(1) implementation."""

    butcher_tableau = GaussLegendre(1)


class CrankNicolson(AbstractRKScheme, DIRKGeneric):
    """2-stage, 2nd order, implicit Runge-Kutta method."""

    a = [[0.0, 0.0], [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22(AbstractRKScheme, DIRKGeneric):
    r"""2-stage, 2nd order, L-stable Diagonally Implicit Runge-Kutta method.

    This method has the Butcher tableau

    $$
        \begin{array}{c|cc}
        \gamma &   \gamma &       0 \\
              1 & 1-\gamma & \gamma \\ \hline
                &       1/2 &     1/2
        \end{array}
    $$

    with $`\gamma = (2 + \sqrt{2})/2$.

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = (2.0 + np.sqrt(2.0)) / 2.0
    a = [[gamma, 0], [1 - gamma, gamma]]
    b = [1 - gamma, gamma]
    c = [gamma, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK23(AbstractRKScheme, DIRKGeneric):
    r"""2-stage, 3rd order Diagonally Implicit Runge-Kutta method.

    This method has the Butcher tableau

    $$
        \begin{array}{c|cc}
          \gamma &    \gamma &       0 \\
        1-\gamma & 1-2\gamma & \gamma \\ \hline
                  &        1/2 &     1/2
        \end{array}
    $$

    with $\gamma = (3 + \sqrt{3})/6$.

    From DIRK(2,3,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = (3 + np.sqrt(3)) / 6
    a = [[gamma, 0], [1 - 2 * gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1 - gamma]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK33(AbstractRKScheme, DIRKGeneric):
    """3-stage, 3rd order, L-stable Diagonally Implicit Runge-Kutta method.

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = 0.4358665215
    b1 = -3.0 / 2.0 * gamma**2 + 4 * gamma - 1.0 / 4.0
    b2 = 3.0 / 2.0 * gamma**2 - 5 * gamma + 5.0 / 4.0
    a = [[gamma, 0, 0], [(1 - gamma) / 2, gamma, 0], [b1, b2, gamma]]
    b = [b1, b2, gamma]
    c = [gamma, (1 + gamma) / 2, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK43(AbstractRKScheme, DIRKGeneric):
    """4-stage, 3rd order, L-stable Diagonally Implicit Runge-Kutta method.

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    a = [
        [0.5, 0, 0, 0],
        [1.0 / 6.0, 0.5, 0, 0],
        [-0.5, 0.5, 0.5, 0],
        [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5],
    ]
    b = [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5]
    c = [0.5, 2.0 / 3.0, 0.5, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRKLSPUM2(AbstractRKScheme, DIRKGeneric):
    """DIRKLSPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge-Kutta method.

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [
        [2.0 / 11.0, 0, 0],
        [205.0 / 462.0, 2.0 / 11.0, 0],
        [2033.0 / 4620.0, 21.0 / 110.0, 2.0 / 11.0],
    ]
    b = [24.0 / 55.0, 1.0 / 5.0, 4.0 / 11.0]
    c = [2.0 / 11.0, 289.0 / 462.0, 751.0 / 924.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.82


class DIRKLPUM2(AbstractRKScheme, DIRKGeneric):
    """DIRKLPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge-Kutta method.

    From IMEX RK scheme (20) in Higueras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [
        [2.0 / 11.0, 0, 0],
        [41.0 / 154.0, 2.0 / 11.0, 0],
        [289.0 / 847.0, 42.0 / 121.0, 2.0 / 11.0],
    ]
    b = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    c = [2.0 / 11.0, 69.0 / 154.0, 67.0 / 77.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.09


class GaussLegendre(CRKGeneric):
    """Direct access to Irksome's GaussLegendre scheme."""

    butcher_tableau = GaussLegendre
    tableau_parameter = 2


class LobattoIIIA(CRKGeneric):
    """Direct access to Irksome's LobattoIIIA scheme."""

    butcher_tableau = LobattoIIIA
    tableau_parameter = 2
    bc_type = "ODE"


class RadauIIA(CRKGeneric):
    """Direct access to Irksome's RadauIIA scheme."""

    butcher_tableau = RadauIIA
    tableau_parameter = 3


class LobattoIIIC(CRKGeneric):
    """Direct access to Irksome's LobattoIIIC scheme."""

    butcher_tableau = LobattoIIIC
    tableau_parameter = 2


class PareschiRusso(DIRKGeneric):
    """Direct access to Irksome's PareschiRusso scheme."""

    butcher_tableau = PareschiRusso
    tableau_parameter = 0.5


class QinZhang(DIRKGeneric):
    """Direct access to Irksome's QinZhang scheme."""

    butcher_tableau = QinZhang()


class Alexander(DIRKGeneric):
    """Direct access to Irksome's Alexander scheme."""

    butcher_tableau = Alexander()


rk_schemes_gadopt = [
    ForwardEuler,
    ERKLSPUM2,
    ERKLPUM2,
    Midpoint,
    SSPRK33,
    eSSPRKs3p3,
    eSSPRKs4p3,
    eSSPRKs5p3,
    eSSPRKs6p3,
    eSSPRKs7p3,
    eSSPRKs8p3,
    eSSPRKs9p3,
    eSSPRKs10p3,
    BackwardEuler,  # Implemented using Irksome's BackwardEuler()
    ImplicitMidpoint,  # Implemented using Irksome's GaussLegendre(1)
    CrankNicolson,
    DIRK22,
    DIRK23,
    DIRK33,
    DIRK43,
    DIRKLSPUM2,
    DIRKLPUM2,
]

rk_schemes_irksome = [
    GaussLegendre,
    LobattoIIIA,
    RadauIIA,
    LobattoIIIC,
    PareschiRusso,
    QinZhang,
    Alexander,
]

__all__ = (
    ["IrksomeIntegrator"]
    + [scheme.__name__ for scheme in rk_schemes_gadopt]
    + [scheme.__name__ for scheme in rk_schemes_irksome]
)
