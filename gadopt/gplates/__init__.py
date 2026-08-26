"""Plate-reconstruction fields for G-ADOPT.

Two things live here. ``GplatesVelocityFunction`` reconstructs surface plate
velocities for a Stokes boundary condition, and ``GplatesScalarFunction``
carries a time-dependent scalar field, such as a lithosphere indicator or a
geotherm, that follows the same reconstruction.

The scalar side is built from three pieces that stay separate on purpose. A
Source says where the reconstructed points are at a geological age and what
they carry (``sources``); an OutputStrategy turns those interpolated values
into the field the model wants (``outputs``); and a ScalarFieldConnector pairs
one of each and manages caching across MPI ranks (``connectors``). The kNN
machinery between them is in ``interpolation``, and ``factories`` assembles the
common combinations, which is where most users should start.
"""

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
