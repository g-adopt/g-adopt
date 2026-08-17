"""Create source, indicator, and geotherm connectors."""

from mpi4py import MPI
from typing import Callable, TYPE_CHECKING
from functools import cached_property

from .connectors import ScalarFieldConnector
from .interpolation import InterpolationConfig
from .outputs import (
    HalfSpaceCoolingGeotherm,
    LateralWeight,
    BoundedLinearGeotherm,
    MeshConfig,
    OutputStrategy,
    BoundedLayerIndicator,
    GlobalLayerIndicator,
    UniformLateralWeight,
)
from .sources import PointCloudSource, Source

if TYPE_CHECKING:
    from gtrack.age_sources import AgeCloudSource
    from .gplates import pyGplatesConnector

__all__ = [
    "ConnectorFactory",
    "LithosphereConnectorFactory",
    "PolygonConnectorFactory",
]


class ConnectorFactory:
    """Create linked source, indicator, and geotherm connectors.

    The indicator and geotherm share one source.
    A caller can assign each object or create it with a factory method.
    Each factory slot accepts one object.

    Args:
        source_class: The source class that `create_source` uses.
        output_class: The output class that `create_indicator` uses.
        geotherm_output_class: The output class that `create_geotherm` uses.
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: Forwarded to every `ScalarFieldConnector` this
            factory creates; see `ScalarFieldConnector` for the semantics.

    Examples:
        >>> factory = ConnectorFactory()
        >>> factory.source = source
        >>> factory.output = output
        >>> indicator = factory.indicator

        >>> factory = ConnectorFactory(
        ...     source_class=PointCloudSource,
        ...     output_class=GlobalLayerIndicator,
        ... )
        >>> factory.create_source(lithosphere_cloud_source, plate_model)
        >>> factory.create_indicator()
        >>> indicator = factory.indicator
    """

    def __init__(
        self,
        source_class: type[Source] | None = None,
        output_class: type[OutputStrategy] | None = None,
        geotherm_output_class: type[OutputStrategy] | None = None,
        *,
        mesh: MeshConfig | None = None,
        interpolation: InterpolationConfig | None = None,
        gc_collect_frequency: int | None = 10,
    ):
        self._source_class = source_class
        self._output_class = output_class
        self._geotherm_output_class = geotherm_output_class
        self._source: Source | None = None
        self._output: OutputStrategy | None = None
        self._geotherm_output: OutputStrategy | None = None
        self._mesh = mesh
        self._interpolation = interpolation
        self._gc_collect_frequency = gc_collect_frequency

    @property
    def source(self):
        """Return the source that the connectors share."""
        return self._source

    @source.setter
    def source(self, source: Source | None):
        if self._source is not None:
            raise RuntimeError("This factory already has a source.")
        self._source = source

    def create_source(self, *source_args, **source_kwargs):
        """Create the source with the configured source class."""
        if self._source is not None:
            raise RuntimeError("This factory already has a source.")
        if self._source_class is None:
            raise TypeError("The source class is not configured.")
        self._source = self._source_class(*source_args, **source_kwargs)

    @property
    def output(self):
        """Return the output that creates the indicator field."""
        return self._output

    @output.setter
    def output(self, output: OutputStrategy | None):
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator output.")
        self._output = output

    def create_indicator(self, **output_kwargs):
        """Create the indicator with the configured output class."""
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator output.")
        if self._output_class is None:
            raise TypeError("The indicator output class is not configured.")
        self._output = self._output_class(**output_kwargs)

    @property
    def geotherm_output(self):
        """Return the output that creates the geotherm field."""
        return self._geotherm_output

    @geotherm_output.setter
    def geotherm_output(self, geotherm_output: OutputStrategy | None):
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm output.")
        self._geotherm_output = geotherm_output

    def create_geotherm(self, **output_kwargs):
        """Create the geotherm with the configured output class."""
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm output.")
        if self._geotherm_output_class is None:
            raise TypeError("The geotherm output class is not configured.")
        self._geotherm_output = self._geotherm_output_class(**output_kwargs)

    @cached_property
    def indicator(self):
        """Return the indicator connector.

        The source and output must exist before the first access.
        """
        if self._source is None:
            raise RuntimeError(
                "A source must be created or assigned before you access the indicator"
            )
        if self._output is None:
            raise RuntimeError(
                "An output must be created or assigned before you access the indicator"
            )
        # The connector validates the pairing because callers can bypass this
        # factory and create a connector directly.
        return ScalarFieldConnector(
            self._source,
            self._output,
            mesh=self._mesh,
            interpolation=self._interpolation,
            gc_collect_frequency=self._gc_collect_frequency,
        )

    @cached_property
    def geotherm(self):
        """Return the geotherm connector.

        The source and geotherm output must exist before the first access.
        """
        if self._source is None:
            raise RuntimeError(
                "A source must be created or assigned before you access the geotherm"
            )
        if self._geotherm_output is None:
            raise RuntimeError(
                "A geotherm output must be created or assigned before you access the geotherm"
            )
        return ScalarFieldConnector(
            self._source,
            self._geotherm_output,
            mesh=self._mesh,
            interpolation=self._interpolation,
            gc_collect_frequency=self._gc_collect_frequency,
        )


class LithosphereConnectorFactory(ConnectorFactory):
    """Create connectors for a global lithosphere source.

    Args:
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: Forwarded to every `ScalarFieldConnector` this
            factory creates; see `ScalarFieldConnector` for the semantics.
    """

    def __init__(
        self,
        *,
        mesh: MeshConfig | None = None,
        interpolation: InterpolationConfig | None = None,
        gc_collect_frequency: int | None = 10,
    ):
        super().__init__(
            PointCloudSource,
            GlobalLayerIndicator,
            HalfSpaceCoolingGeotherm,
            mesh=mesh,
            interpolation=interpolation,
            gc_collect_frequency=gc_collect_frequency,
        )

    def create_source(
        self,
        producer: "AgeCloudSource",
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Create a source from a gtrack lithosphere producer."""
        super().create_source(producer, gplates_connector, comm=comm)

    def create_indicator(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
        fallback_thickness_km: float = 100.0,
        lateral_weight: LateralWeight | None = None,
    ):
        """Create a global layer indicator.

        `lateral_weight` selects the lateral-weight strategy.
        The default strategy returns one at every target node.

        Example:
            `factory.create_indicator(lateral_weight=SourceLateralWeight())`
        """
        if lateral_weight is None:
            lateral_weight = UniformLateralWeight()
        super().create_indicator(
            base_transition_width_km=base_transition_width_km,
            fixed_base_depth_km=fixed_base_depth_km,
            fallback_thickness_km=fallback_thickness_km,
            lateral_weight=lateral_weight,
        )

    def create_geotherm(
        self,
        thermal_diffusivity_m2_per_s: float = 1e-6,
        fallback_thickness_km: float = 100.0,
        fallback_age_myr: float = 500.0,
        geotherm: Callable | None = None,
    ):
        """Create a half-space cooling geotherm."""
        super().create_geotherm(
            thermal_diffusivity_m2_per_s=thermal_diffusivity_m2_per_s,
            fallback_thickness_km=fallback_thickness_km,
            fallback_age_myr=fallback_age_myr,
            geotherm=geotherm,
        )


class PolygonConnectorFactory(ConnectorFactory):
    """Create connectors for a bounded polygon source.

    Args:
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: Forwarded to every `ScalarFieldConnector` this
            factory creates; see `ScalarFieldConnector` for the semantics.
    """

    def __init__(
        self,
        *,
        mesh: MeshConfig | None = None,
        interpolation: InterpolationConfig | None = None,
        gc_collect_frequency: int | None = 10,
    ):
        super().__init__(
            PointCloudSource,
            BoundedLayerIndicator,
            BoundedLinearGeotherm,
            mesh=mesh,
            interpolation=interpolation,
            gc_collect_frequency=gc_collect_frequency,
        )

    def create_source(
        self,
        producer: "AgeCloudSource",
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Create a source from a gtrack polygon producer."""
        super().create_source(producer, gplates_connector, comm=comm)

    def create_indicator(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
    ):
        """Create a bounded layer indicator."""
        super().create_indicator(
            base_transition_width_km=base_transition_width_km,
            fixed_base_depth_km=fixed_base_depth_km,
        )

    def create_geotherm(
        self,
        geotherm: Callable | None = None,
    ):
        """Create a bounded linear geotherm."""
        super().create_geotherm(
            geotherm=geotherm,
        )
