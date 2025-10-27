r"""This module provides Irksome-based time integrators (wrapped to maintain our API).

This module uses Irksome's time integrators,
which natively handles nonlinear mass terms like C(h) dh/dt in Richards equation.
The migration maintains backward compatibility by wrapping Irksome's API with
G-ADOPT's existing class names.

"""

import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import firedrake
import numpy as np
from firedrake import inner

# Import Irksome components
try:
    from irksome import Dt, TimeStepper, MeshConstant
    from irksome.ButcherTableaux import (
        BackwardEuler as IrksomeBackwardEuler,
        GaussLegendre,
        Alexander,
        LobattoIIIA,
        LobattoIIIC,
        PareschiRusso,
        QinZhang,
        RadauIIA,
        ButcherTableau,
    )
    from irksome.dirk_stepper import DIRKTimeStepper
    from irksome.explicit_stepper import ExplicitTimeStepper
    from irksome.sspk_tableau import SSPButcherTableau
    IRKSOME_AVAILABLE = True
except ImportError:
    IRKSOME_AVAILABLE = False
    # Fallback imports for when Irksome is not available
    Dt = None
    TimeStepper = None
    MeshConstant = None
    ButcherTableau = None
    SSPButcherTableau = None

from .equations import Equation
from .utility import ensure_constant


class TimeIntegratorBase(ABC):
    """Defines the API for all time integrators."""

    @abstractmethod
    def advance(self, update_forcings: Callable | None = None, t: float | None = None):
        """Advances equations for one time step.

        Args:
          update_forcings:
            Callable updating any time-dependent equation forcings
          t:
            Current simulation time

        """
        pass

    @abstractmethod
    def initialize(self, init_solution):
        """Initialises the time integrator.

        Arguments:
          init_solution: Firedrake function representing the initial solution.

        """
        pass


class IrksomeTimeIntegrator(TimeIntegratorBase):
    """Base class for Irksome-based time integrators."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = None,
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
    ):
        if not IRKSOME_AVAILABLE:
            raise ImportError("Irksome is not available. Please install Irksome to use time integrators.")

        self.equation = equation
        self.solution = solution
        self.dt = float(dt)
        self.dt_const = ensure_constant(dt)
        self.solution_old = solution_old or firedrake.Function(solution, name='Old'+solution.name())
        self.solver_parameters = solver_parameters or {}
        self.strong_bcs = strong_bcs or []

        # Create mesh constant for time variables
        self.mesh_const = MeshConstant(solution.function_space().mesh())
        self.t = self.mesh_const.Constant(0.0)
        self.dt_var = self.mesh_const.Constant(dt)

        # Build semi-discrete form
        self.build_form()

        # Create Irksome time stepper
        self.setup_stepper()

    def build_form(self):
        """Build the semi-discrete form for Irksome."""
        # Use the equation's build_irksome_form method if available
        if hasattr(self.equation, 'build_irksome_form'):
            F = self.equation.build_irksome_form(self.solution)
        else:
            # Fallback to simple form
            F = inner(self.equation.test, Dt(self.solution)) * self.equation.dx + self.equation.residual(self.solution)

        self.form = F

    def setup_stepper(self):
        """Set up the Irksome time stepper."""
        # Create the time stepper
        self.stepper = TimeStepper(
            self.form,
            self.butcher_tableau,
            self.t,
            self.dt_var,
            self.solution,
            bcs=self.strong_bcs,
            solver_parameters=self.solver_parameters,
        )

    def advance(self, update_forcings: Callable | None = None, t: float | None = None):
        """Advances equations for one time step."""
        if update_forcings is not None and t is not None:
            self.t.assign(t)
            update_forcings(t)
        elif update_forcings is not None:
            update_forcings()

        self.stepper.advance()
        self.solution_old.assign(self.solution)

    def initialize(self, init_solution):
        """Initialises the time integrator."""
        self.solution.assign(init_solution)
        self.solution_old.assign(init_solution)


class BackwardEuler(IrksomeTimeIntegrator):
    """Backward Euler method using Irksome."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = IrksomeBackwardEuler()
        super().__init__(*args, **kwargs)


class ImplicitMidpoint(IrksomeTimeIntegrator):
    """Implicit midpoint method using Irksome (Gauss-Legendre 1-stage)."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = GaussLegendre(1)
        super().__init__(*args, **kwargs)


class CrankNicolsonRK(IrksomeTimeIntegrator):
    """Crank-Nicolson scheme using Irksome (LobattoIIIA 2-stage)."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = LobattoIIIA(2)
        super().__init__(*args, **kwargs)


class DIRK22(IrksomeTimeIntegrator):
    """2-stage, 2nd order DIRK method using Irksome."""

    def __init__(self, *args, **kwargs):
        # Use Pareschi-Russo with gamma = (2 + sqrt(2))/2
        gamma = (2.0 + np.sqrt(2.0)) / 2.0
        self.butcher_tableau = PareschiRusso(gamma)
        super().__init__(*args, **kwargs)


class DIRK23(IrksomeTimeIntegrator):
    """2-stage, 3rd order DIRK method using Irksome."""

    def __init__(self, *args, **kwargs):
        # Use Pareschi-Russo with gamma = (3 + sqrt(3))/6
        gamma = (3.0 + np.sqrt(3.0)) / 6.0
        self.butcher_tableau = PareschiRusso(gamma)
        super().__init__(*args, **kwargs)


class DIRK33(IrksomeTimeIntegrator):
    """3-stage, 3rd order DIRK method using Irksome."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = Alexander()
        super().__init__(*args, **kwargs)


class DIRK43(IrksomeTimeIntegrator):
    """4-stage, 3rd order DIRK method using Irksome."""

    def __init__(self, *args, **kwargs):
        # Use RadauIIA 2-stage (which is 3rd order)
        self.butcher_tableau = RadauIIA(2)
        super().__init__(*args, **kwargs)


# Explicit methods - implemented using Irksome's explicit steppers

class ExplicitTimeIntegrator(TimeIntegratorBase):
    """Base class for Irksome-based explicit time integrators."""

    def __init__(
        self,
        equation: Equation,
        solution: firedrake.Function,
        dt: float,
        solution_old: Optional[firedrake.Function] = None,
        solver_parameters: Optional[dict[str, Any]] = None,
        strong_bcs: Optional[list[firedrake.DirichletBC]] = None,
    ):
        if not IRKSOME_AVAILABLE:
            raise ImportError("Irksome is not available. Please install Irksome to use time integrators.")

        self.equation = equation
        self.solution = solution
        self.dt = float(dt)
        self.dt_const = ensure_constant(dt)
        self.solution_old = solution_old or firedrake.Function(solution, name='Old'+solution.name())
        self.solver_parameters = solver_parameters or {}
        self.strong_bcs = strong_bcs or []

        # Create mesh constant for time variables
        self.mesh_const = MeshConstant(solution.function_space().mesh())
        self.t = self.mesh_const.Constant(0.0)
        self.dt_var = self.mesh_const.Constant(dt)

        # Build semi-discrete form
        self.build_form()

        # Create Irksome explicit time stepper
        self.setup_stepper()

    def build_form(self):
        """Build the semi-discrete form for Irksome."""
        # Use the equation's build_irksome_form method if available
        if hasattr(self.equation, 'build_irksome_form'):
            F = self.equation.build_irksome_form(self.solution)
        else:
            # Fallback to simple form
            F = inner(self.equation.test, Dt(self.solution)) * self.equation.dx + self.equation.residual(self.solution)

        self.form = F

    def setup_stepper(self):
        """Set up the Irksome explicit time stepper."""
        # Use ExplicitTimeStepper for explicit methods
        self.stepper = ExplicitTimeStepper(
            self.form,
            self.butcher_tableau,
            self.t,
            self.dt_var,
            self.solution,
            bcs=self.strong_bcs,
            solver_parameters=self.solver_parameters,
        )

    def advance(self, update_forcings: Callable | None = None, t: float | None = None):
        """Advances equations for one time step."""
        if update_forcings is not None and t is not None:
            self.t.assign(t)
            update_forcings(t)
        elif update_forcings is not None:
            update_forcings()

        self.stepper.advance()
        self.solution_old.assign(self.solution)

    def initialize(self, init_solution):
        """Initialises the time integrator."""
        self.solution.assign(init_solution)
        self.solution_old.assign(init_solution)

    @property
    def cfl_number(self):
        """Return CFL number for this method."""
        # Method-specific CFL coefficient
        return getattr(self, '_cfl_coefficient', 1.0)


class ERKEuler(ExplicitTimeIntegrator):
    """Forward Euler method."""

    def __init__(self, *args, **kwargs):
        # Forward Euler: A = [[0]], b = [1], c = [0]
        A = np.array([[0.0]])
        b = np.array([1.0])
        c = np.array([0.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 1, None, None)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class ERKMidpoint(ExplicitTimeIntegrator):
    """Explicit midpoint method (2nd order)."""

    def __init__(self, *args, **kwargs):
        # Midpoint: 2-stage explicit RK
        A = np.array([[0.0, 0.0], [0.5, 0.0]])
        b = np.array([0.0, 1.0])
        c = np.array([0.0, 0.5])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 2, None, None)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class SSPRK33(ExplicitTimeIntegrator):
    """3rd order, 3-stage SSP Runge-Kutta method."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = SSPButcherTableau(3, 3)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class eSSPRKs3p3(ExplicitTimeIntegrator):
    """3-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        self.butcher_tableau = SSPButcherTableau(3, 3)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class eSSPRKs4p3(ExplicitTimeIntegrator):
    """4-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 4-stage, 3rd order SSP RK
        # Based on Shu-Osher form with optimal CFL coefficient
        A = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/2.0])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 2.0
        super().__init__(*args, **kwargs)


class eSSPRKs5p3(ExplicitTimeIntegrator):
    """5-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 5-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        c = np.array([0.0, 0.4, 0.4, 0.4, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 2.5
        super().__init__(*args, **kwargs)


class eSSPRKs6p3(ExplicitTimeIntegrator):
    """6-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 6-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0/3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0/3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0/3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 7.0/12.0])
        c = np.array([0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 3.0
        super().__init__(*args, **kwargs)


class eSSPRKs7p3(ExplicitTimeIntegrator):
    """7-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 7-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/14.0, 1.0/14.0, 1.0/14.0, 1.0/14.0, 1.0/14.0, 1.0/14.0, 8.0/14.0])
        c = np.array([0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 3.5
        super().__init__(*args, **kwargs)


class eSSPRKs8p3(ExplicitTimeIntegrator):
    """8-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 8-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/16.0, 1.0/16.0, 1.0/16.0, 1.0/16.0, 1.0/16.0, 1.0/16.0, 1.0/16.0, 9.0/16.0])
        c = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 4.0
        super().__init__(*args, **kwargs)


class eSSPRKs9p3(ExplicitTimeIntegrator):
    """9-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 9-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0/6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0/6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0/6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0/6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0/6.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 10.0/18.0])
        c = np.array([0.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 4.5
        super().__init__(*args, **kwargs)


class eSSPRKs10p3(ExplicitTimeIntegrator):
    """10-stage, 3rd order SSP RK method."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for 10-stage, 3rd order SSP RK
        A = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        b = np.array([1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 11.0/20.0])
        c = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 3, None, None)
        self._cfl_coefficient = 5.0
        super().__init__(*args, **kwargs)


class ERKLSPUM2(ExplicitTimeIntegrator):
    """ERKLSPUM2 method for level set."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for ERKLSPUM2 (2-stage explicit)
        A = np.array([[0.0, 0.0], [0.5, 0.0]])
        b = np.array([0.0, 1.0])
        c = np.array([0.0, 0.5])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 2, None, None)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class ERKLPUM2(ExplicitTimeIntegrator):
    """ERKLPUM2 method for level set."""

    def __init__(self, *args, **kwargs):
        # Custom tableau for ERKLPUM2 (2-stage explicit)
        A = np.array([[0.0, 0.0], [0.5, 0.0]])
        b = np.array([0.0, 1.0])
        c = np.array([0.0, 0.5])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 2, None, None)
        self._cfl_coefficient = 1.0
        super().__init__(*args, **kwargs)


class DIRKLSPUM2(IrksomeTimeIntegrator):
    """DIRKLSPUM2 method for level set."""

    def __init__(self, *args, **kwargs):
        # Custom DIRK tableau for DIRKLSPUM2 (2-stage DIRK)
        gamma = 1.0 - 1.0/np.sqrt(2.0)
        A = np.array([[gamma, 0.0], [1.0 - 2.0*gamma, gamma]])
        b = np.array([0.5, 0.5])
        c = np.array([gamma, 1.0 - gamma])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 2, None, None)
        super().__init__(*args, **kwargs)


class DIRKLPUM2(IrksomeTimeIntegrator):
    """DIRKLPUM2 method for level set."""

    def __init__(self, *args, **kwargs):
        # Custom DIRK tableau for DIRKLPUM2 (2-stage DIRK)
        gamma = 1.0 - 1.0/np.sqrt(2.0)
        A = np.array([[gamma, 0.0], [1.0 - 2.0*gamma, gamma]])
        b = np.array([0.5, 0.5])
        c = np.array([gamma, 1.0 - gamma])
        self.butcher_tableau = ButcherTableau(A, b, None, c, 2, None, None)
        super().__init__(*args, **kwargs)


# Legacy compatibility - these are aliases for the Irksome versions
TimeIntegrator = IrksomeTimeIntegrator
RungeKuttaTimeIntegrator = IrksomeTimeIntegrator

# CFL constants
CFL_UNCONDITIONALLY_STABLE = -1
