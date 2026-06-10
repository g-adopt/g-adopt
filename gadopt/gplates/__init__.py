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
    MeshConfig,
    OutputStrategy,
    TanhOutput,
    ocean_erf_normalized,
    radial_tanh_step,
)
from .sources import (
    CloudDataType,
    LithosphereSource,
    LithosphereSourceConfig,
    Source,
)
from .factories import (
    ConnectorFactory,
    LithosphereConnectorFactory,
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
    "CloudDataType",
    # Outputs
    "OutputStrategy",
    "TanhOutput",
    "GeothermERFOutput",
    "ocean_erf_normalized",
    "radial_tanh_step",
    # Factories
    "ConnectorFactory",
    "LithosphereConnectorFactory",
]
