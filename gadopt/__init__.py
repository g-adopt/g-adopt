from firedrake import *
from firedrake.output import VTKFile
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint
from .utility import log, ParameterLog, TimestepAdaptor, timer_decorator, collect_garbage, DiffusiveSmoothingSolver

from .approximations import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    ExtendedBoussinesqApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from .diagnostics import GeodynamicalDiagnostics
from .energy_solver import EnergySolver
from .level_set_tools import (
    LevelSetSolver,
    Material,
    density_RaB,
    entrainment,
    field_interface,
)
from .limiter import VertexBasedP1DGLimiter
from .momentum_equation import StokesEquations
from .preconditioners import SPDAssembledPC
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .time_stepper import eSSPRKs3p3, eSSPRKs10p3
from .utility import (
    LayerAveraging,
    node_coordinates,
)

PETSc.Sys.popErrorHandler()
