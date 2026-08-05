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
    DataBase,
    DeblendedBase,
    FixedBase,
    GeothermERFOutput,
    GeothermLinearOutput,
    IndicatorOutput,
    LateralFractionOutput,
    MaskedGeothermLinearOutput,
    MembershipAmplitude,
    MeshConfig,
    NoAmplitude,
    OutputStrategy,
    MaskedQuinticOutput,
    QuinticOutput,
    QuinticStep,
    TaperedAmplitude,
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
    "IndicatorOutput",
    "MaskedQuinticOutput",
    "QuinticOutput",
    "GeothermERFOutput",
    "GeothermLinearOutput",
    "MaskedGeothermLinearOutput",
    "LateralFractionOutput",
    # Composable indicator parts
    "QuinticStep",
    "FixedBase",
    "DataBase",
    "DeblendedBase",
    "NoAmplitude",
    "MembershipAmplitude",
    "TaperedAmplitude",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_quintic_step",
    # Factories
    "ConnectorFactory",
    "LithosphereConnectorFactory",
    "PolygonConnectorFactory",
]
