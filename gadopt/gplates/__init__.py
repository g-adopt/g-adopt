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
    LateralFractionOutput,
    MeshConfig,
    OutputStrategy,
    QuinticOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_quintic_step,
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
    "QuinticOutput",
    "GeothermERFOutput",
    "GeothermLinearOutput",
    "LateralFractionOutput",
    "ocean_erf_normalized",
    "continental_linear",
    "radial_quintic_step",
    # Factories
    "lithosphere_indicator",
    "lithosphere_geotherm",
    "polygon_indicator",
    "polygon_geotherm",
]
