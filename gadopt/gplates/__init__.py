from .connectors import ScalarFieldConnector
from .interpolation import InterpolationConfig, SphericalKNNInterpolator
from .gplates import (
    GplatesScalarFunction,
    GplatesVelocityFunction,
    ensure_reconstruction,
    pyGplatesConnector,
    PlateModelFiles,
)
from .outputs import (
    InterpolatedBaseDepth,
    MembershipCorrectedBaseDepth,
    FixedBaseDepth,
    HalfSpaceCoolingGeotherm,
    LinearGeotherm,
    LayerIndicator,
    LateralWeight,
    MembershipField,
    BoundedLinearGeotherm,
    MembershipLateralWeight,
    MeshConfig,
    UniformLateralWeight,
    OutputStrategy,
    BoundedLayerIndicator,
    GlobalLayerIndicator,
    RadialQuinticTransition,
    SourceLateralWeight,
    MappedMembershipWeight,
    continental_linear,
    ocean_erf_normalized,
    radial_quintic_step,
)
from .sources import (
    CloudDataType,
    PointCloudSource,
    Source,
)
from .factories import (
    ConnectorFactory,
    LithosphereConnectorFactory,
    PolygonConnectorFactory,
)

__all__ = [
    # Firedrake function wrappers
    "GplatesVelocityFunction",
    "GplatesScalarFunction",
    # Plate-reconstruction backbone
    "pyGplatesConnector",
    "PlateModelFiles",
    "ensure_reconstruction",
    # Connector + config
    "ScalarFieldConnector",
    "InterpolationConfig",
    "SphericalKNNInterpolator",
    "MeshConfig",
    # Sources
    "Source",
    "PointCloudSource",
    "CloudDataType",
    # Outputs
    "OutputStrategy",
    "LayerIndicator",
    "LateralWeight",
    "BoundedLayerIndicator",
    "GlobalLayerIndicator",
    "HalfSpaceCoolingGeotherm",
    "LinearGeotherm",
    "BoundedLinearGeotherm",
    "MembershipField",
    # Composable indicator parts
    "RadialQuinticTransition",
    "FixedBaseDepth",
    "InterpolatedBaseDepth",
    "MembershipCorrectedBaseDepth",
    "UniformLateralWeight",
    "MembershipLateralWeight",
    "SourceLateralWeight",
    "MappedMembershipWeight",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_quintic_step",
    # Factories
    "ConnectorFactory",
    "LithosphereConnectorFactory",
    "PolygonConnectorFactory",
]
