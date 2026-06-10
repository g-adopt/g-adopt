from .connectors import ScalarFieldConnector, InterpolationConfig
from .gplates import (
    GplatesScalarFunction,
    GplatesVelocityFunction,
    ensure_reconstruction,
    pyGplatesConnector,
    PlateModelFiles,
)
from .outputs import (
    FadedRadialStepOutput,
    FadedTanhOutput,
    GeothermERFOutput,
    GeothermLinearOutput,
    LateralFractionOutput,
    MeshConfig,
    OutputStrategy,
    TanhOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_tanh_step,
)
from .sources import (
    CloudDataType,
    LithosphereSource,
    LithosphereSourceConfig,
    PolygonSource,
    PolygonSourceConfig,
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
    "MeshConfig",
    # Sources
    "Source",
    "LithosphereSource",
    "LithosphereSourceConfig",
    "PolygonSource",
    "PolygonSourceConfig",
    "CloudDataType",
    # Outputs
    "OutputStrategy",
    "TanhOutput",
    "GeothermERFOutput",
    "GeothermLinearOutput",
    "LateralFractionOutput",
    "FadedRadialStepOutput",
    "FadedTanhOutput",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_tanh_step",
    # Factories
    "ConnectorFactory",
    "LithosphereConnectorFactory",
    "PolygonConnectorFactory",
]
