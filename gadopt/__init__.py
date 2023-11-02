from firedrake import *
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint
from .limiter import VertexBasedP1DGLimiter
from .utility import log, ParameterLog, TimestepAdaptor, LayerAveraging, timer_decorator, LabeledMeshHierarchy
from .diagnostics import GeodynamicalDiagnostics
from .momentum_equation import StokesEquations
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .energy_solver import EnergySolver
from .approximations import BoussinesqApproximation, ExtendedBoussinesqApproximation, AnelasticLiquidApproximation, TruncatedAnelasticLiquidApproximation
<<<<<<< HEAD
from .preconditioners import SPDAssembledPC, VariableMassInvPC
=======
from .preconditioners import SPDAssembledPC, VariableMassInvPC, P0MassInvPC, AugmentedAssembledPC
>>>>>>> refs/rewritten/Add-AugmentedLagrangianPC-approach-for-Q2P1dg-and-merge-branch-p2p0-al-into-augmented-lagrangian

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
