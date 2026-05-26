from .connectors import ScalarFieldConnector, InterpolationConfig
from .gplates import (
    GplatesScalarFunction,
    GplatesVelocityFunction,
    ensure_reconstruction,
    lithosphere_geotherm,
    lithosphere_indicator,
    polygon_geotherm,
    polygon_indicator,
    pyGplatesConnector,
    PlateModelFiles,
)
from .outputs import (
    GeothermERFOutput,
    GeothermLinearOutput,
    MeshConfig,
    OutputStrategy,
    TanhOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_tanh_step,
)
from .sources import (
    LithosphereSource,
    LithosphereSourceConfig,
    PolygonSource,
    PolygonSourceConfig,
    Source,
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
    # Outputs
    "OutputStrategy",
    "TanhOutput",
    "GeothermERFOutput",
    "GeothermLinearOutput",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_tanh_step",
    # Factories
    "lithosphere_indicator",
    "lithosphere_geotherm",
    "polygon_indicator",
    "polygon_geotherm",
]
