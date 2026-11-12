from firedrake import *

from .adaptive_simulation import AdaptiveSimulation
from .approximations import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    CompressibleInternalVariableApproximation,
    ExtendedBoussinesqApproximation,
    IncompressibleMaxwellApproximation,
    MaxwellApproximation,
    QuasiCompressibleInternalVariableApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from .diagnostics import *
from .free_surface_equation import (
    free_surface_normal,
    free_surface_volume_multiplier,
)
from .level_set_tools import (
    LevelSetSolver,
    assign_level_set_values,
    interface_thickness,
    material_entrainment,
    material_field,
    min_max_height,
)

# from .limiter import VertexBasedP1DGLimiter
from .nullspaces import create_stokes_nullspace, rigid_body_modes
from .preconditioners import (
    FreeSurfaceMassInvPC,
    SPDAssembledPC,
    StokesMassInvPC,
)
from .solver_options_manager import DeleteParam
from .stokes_integrators import (
    BoundaryNormalStressSolver,
    CoupledInternalVariableSolver,
    InternalVariableSolver,
    StokesSolver,
    ViscoelasticStokesSolver,
)
from .time_stepper import *
from .transport_solver import (
    DiffusiveSmoothingSolver,
    EnergySolver,
    GenericTransportSolver,
)
from .utility import (
    InteriorBC,
    LayerAveraging,
    ParameterLog,
    TimestepAdaptor,
    function_name,
    get_boundary_ids,
    interpolate_1d_profile,
    log,
    node_coordinates,
    timer_decorator,
    vertical_component,
)

PETSc.Sys.popErrorHandler()
