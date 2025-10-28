from firedrake import *
from firedrake.output import VTKFile

from .approximations import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    ExtendedBoussinesqApproximation,
    SmallDisplacementViscoelasticApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from .diagnostics import GeodynamicalDiagnostics
from .level_set_tools import (
    LevelSetSolver,
    assign_level_set_values,
    interface_thickness,
    material_entrainment,
    material_field,
    min_max_height,
)
from .limiter import VertexBasedP1DGLimiter
from .nullspaces import create_stokes_nullspace, rigid_body_modes
from .preconditioners import FreeSurfaceMassInvPC, SPDAssembledPC
from .solver_options_manager import DeleteParam
from .stokes_integrators import (
    StokesSolver,
    ViscoelasticStokesSolver,
    BoundaryNormalStressSolver,
)
from .time_stepper import (
    BackwardEuler,
    CrankNicolsonRK,
    ImplicitMidpoint,
    eSSPRKs3p3,
    eSSPRKs10p3,
    # Direct Irksome scheme access
    IrksomeRadauIIA,
    IrksomeGaussLegendre,
    IrksomeLobattoIIIA,
    IrksomeLobattoIIIC,
    IrksomeAlexander,
    IrksomeQinZhang,
    IrksomePareschiRusso,
)
from .transport_solver import EnergySolver, GenericTransportSolver
from .utility import (
    InteriorBC,
    LayerAveraging,
    ParameterLog,
    TimestepAdaptor,
    interpolate_1d_profile,
    log,
    node_coordinates,
    timer_decorator,
    get_boundary_ids,
)

PETSc.Sys.popErrorHandler()
