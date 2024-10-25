from logging import CRITICAL as CRITICAL
from logging import DEBUG as DEBUG
from logging import ERROR as ERROR
from logging import INFO as INFO
from logging import WARNING as WARNING

from firedrake import *

from .approximations import Approximation as Approximation
from .diagnostics import GeodynamicalDiagnostics as GeodynamicalDiagnostics
from .energy_solver import EnergySolver as EnergySolver
from .energy_solver import GenericTransportSolver as GenericTransportSolver
from .level_set_tools import LevelSetSolver as LevelSetSolver
from .level_set_tools import entrainment as entrainment
from .level_set_tools import material_field as material_field
from .limiter import VertexBasedP1DGLimiter as VertexBasedP1DGLimiter
from .preconditioners import FreeSurfaceMassInvPC as FreeSurfaceMassInvPC
from .preconditioners import SPDAssembledPC as SPDAssembledPC
from .stokes_integrators import StokesSolver as StokesSolver
from .stokes_integrators import ViscoelasticSolver as ViscoelasticSolver
from .stokes_integrators import create_stokes_nullspace as create_stokes_nullspace
from .time_stepper import BackwardEuler as BackwardEuler
from .time_stepper import CrankNicolsonRK as CrankNicolsonRK
from .time_stepper import ImplicitMidpoint as ImplicitMidpoint
from .time_stepper import eSSPRKs3p3 as eSSPRKs3p3
from .time_stepper import eSSPRKs10p3 as eSSPRKs10p3
from .utility import InteriorBC as InteriorBC
from .utility import LayerAveraging as LayerAveraging
from .utility import ParameterLog as ParameterLog
from .utility import TimestepAdaptor as TimestepAdaptor
from .utility import interpolate_1d_profile as interpolate_1d_profile
from .utility import log as log
from .utility import node_coordinates as node_coordinates

PETSc.Sys.popErrorHandler()
