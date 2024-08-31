from firedrake import *

from .approximations import EquationSystem
from .diagnostics import GeodynamicalDiagnostics
from .energy_solver import EnergySolver
from .free_surface_equation import FreeSurfaceEquation
from .level_set_tools import LevelSetSolver, entrainment, material_field
from .limiter import VertexBasedP1DGLimiter
from .momentum_equation import StokesEquations
from .preconditioners import FreeSurfaceMassInvPC, SPDAssembledPC
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .time_stepper import (
    BackwardEuler,
    CrankNicolsonRK,
    ImplicitMidpoint,
    eSSPRKs3p3,
    eSSPRKs10p3,
)
from .utility import (
    InteriorBC,
    LayerAveraging,
    ParameterLog,
    TimestepAdaptor,
    interpolate_1d_profile,
    log,
    node_coordinates,
    timer_decorator,
)

PETSc.Sys.popErrorHandler()
