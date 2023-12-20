from firedrake import *
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint, eSSPRKs3p3, eSSPRKs10p3
from .limiter import VertexBasedP1DGLimiter
from .utility import log, ParameterLog, TimestepAdaptor, LayerAveraging, timer_decorator, node_coordinates
from .diagnostics import GeodynamicalDiagnostics, rms_velocity, entrainment
from .momentum_equation import StokesEquations
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .energy_solver import EnergySolver
from .approximations import BoussinesqApproximation, ExtendedBoussinesqApproximation, AnelasticLiquidApproximation, TruncatedAnelasticLiquidApproximation
from .preconditioners import SPDAssembledPC, VariableMassInvPC
from .level_set_tools import Material, LevelSetSolver, diffuse_interface, sharp_interface, density_RaB

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
