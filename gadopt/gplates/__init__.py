from .connectors import ScalarFieldConnector, InterpolationConfig
from .gplates import (
    GplatesScalarFunction,
    GplatesVelocityFunction,
    ensure_reconstruction,
    pyGplatesConnector,
    PlateModelFiles,
)
from .outputs import (
    GeothermERFOutput,
    GeothermLinearOutput,
    LateralFractionOutput,
    MeshConfig,
    OutputStrategy,
    QuinticOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_quintic_step,
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
    "QuinticOutput",
    "GeothermERFOutput",
    "GeothermLinearOutput",
    "LateralFractionOutput",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_quintic_step",
    # Factories
    "ConnectorFactory",
    "LithosphereConnectorFactory",
    "PolygonConnectorFactory",
]
