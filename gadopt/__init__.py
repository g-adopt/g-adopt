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
from .level_set_tools import (
    LevelSetSolver,
    Material,
    density_RaB,
    entrainment,
    field_interface,
)
from .limiter import VertexBasedP1DGLimiter
from .preconditioners import FreeSurfaceMassInvPC, SPDAssembledPC
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .time_stepper import BackwardEuler, CrankNicolsonRK, ImplicitMidpoint, eSSPRKs3p3, eSSPRKs10p3
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
