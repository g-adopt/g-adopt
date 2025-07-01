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
from .preconditioners import FreeSurfaceMassInvPC, SPDAssembledPC
from .stokes_integrators import (
    StokesSolver,
    ViscoelasticStokesSolver,
    BoundaryNormalStressSolver,
    create_stokes_nullspace,
)
from .time_stepper import (
    BackwardEuler,
    CrankNicolsonRK,
    ImplicitMidpoint,
    eSSPRKs3p3,
    eSSPRKs10p3,
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
