from firedrake import *
from firedrake.output import VTKFile

from .approximations import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    ExtendedBoussinesqApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from .diagnostics import GeodynamicalDiagnostics
from .energy_solver import EnergySolver
from .free_surface_equation import FreeSurfaceEquation
from .level_set_tools import (
    LevelSetSolver,
    Material,
    density_RaB,
    entrainment,
    field_interface,
)
from .limiter import VertexBasedP1DGLimiter
from .momentum_equation import StokesEquations
from .preconditioners import SPDAssembledPC, VariableMassInvPC
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .time_stepper import BackwardEuler, CrankNicolsonRK, ImplicitMidpoint, eSSPRKs3p3, eSSPRKs10p3
from .utility import (
    InteriorBC,
    LayerAveraging,
    ParameterLog,
    TimestepAdaptor,
    log,
    node_coordinates,
    timer_decorator,
)

PETSc.Sys.popErrorHandler()
