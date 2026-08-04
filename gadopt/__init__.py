from firedrake import *
from firedrake.output import VTKFile

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
from .dtn_form import DtNGravityForm
from .gia_gravity import (
    FluidCore,
    GIASpaceLayout,
    RigidBodyAssembledPC,
    SelfGravitatingGIASolver,
    rigid_rotation_nullspace,
    self_gravitating_gia_space,
    selfgrav_dtn_iterative_solver_parameters,
)
from .gravity_solver import CylindricalDtN, GravitySolver, SphericalDtN
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
from .preconditioners import (DtNMultiplierDiagPC,  # noqa: F401
                              DtNTwoBlockSchurPC, FreeSurfaceMassInvPC,
                              SPDAssembledPC)
from .solver_options_manager import (GAMG_PARAMETERS, DeleteParam,
                                     gamg_parameters)
from .stokes_integrators import (
    BoundaryNormalStressSolver,
    CoupledInternalVariableSolver,
    InternalVariableSolver,
    StokesSolver,
    ViscoelasticStokesSolver,
)
from .time_stepper import *
from .transport_solver import DiffusiveSmoothingSolver, EnergySolver, GenericTransportSolver
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
