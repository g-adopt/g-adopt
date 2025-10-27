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
from .richards_solver import RichardsSolver
from .soil_curves import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)
from .solver_options_manager import DeleteParam
from .stokes_integrators import (
    StokesSolver,
    ViscoelasticStokesSolver,
    BoundaryNormalStressSolver,
)
from .time_stepper import (
    # Implicit methods
    BackwardEuler,
    CrankNicolsonRK,
    DIRK22,
    DIRK23,
    DIRK33,
    DIRK43,
    ImplicitMidpoint,
    # Explicit methods
    ERKEuler,
    ERKMidpoint,
    SSPRK33,
    eSSPRKs3p3,
    eSSPRKs4p3,
    eSSPRKs5p3,
    eSSPRKs6p3,
    eSSPRKs7p3,
    eSSPRKs8p3,
    eSSPRKs9p3,
    eSSPRKs10p3,
    # Level set methods
    ERKLSPUM2,
    ERKLPUM2,
    DIRKLSPUM2,
    DIRKLPUM2,
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
