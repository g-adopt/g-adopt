from firedrake import *

from .approximations import (
    BoussinesqApproximation,
    ExtendedBoussinesqApproximation,
    AnelasticLiquidApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from .diagnostics import GeodynamicalDiagnostics
from .energy_solver import EnergySolver
from .limiter import VertexBasedP1DGLimiter
from .momentum_equation import StokesEquations
from .preconditioners import SPDAssembledPC, VariableMassInvPC
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint
from .utility import log, ParameterLog, TimestepAdaptor, LayerAveraging, timer_decorator

PETSc.Sys.popErrorHandler()

