from firedrake import *
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint, BackwardEuler
from .limiter import VertexBasedP1DGLimiter
from .utility import log, ParameterLog, TimestepAdaptor, LayerAveraging, timer_decorator, InteriorBC
from .diagnostics import GeodynamicalDiagnostics
from .momentum_equation import StokesEquations, FreeSurfaceStokesEquations
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .energy_solver import EnergySolver
from .approximations import BoussinesqApproximation, ExtendedBoussinesqApproximation, AnelasticLiquidApproximation, TruncatedAnelasticLiquidApproximation
from .preconditioners import SPDAssembledPC, VariableMassInvPC
from .free_surface_equation import FreeSurfaceEquation

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
