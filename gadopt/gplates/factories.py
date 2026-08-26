"""Create source, indicator, and geotherm connectors.

A lithosphere indicator and its geotherm read the same reconstructed points,
and they must read them at the same age. Building the two connectors by hand
means constructing the Source once and remembering to pass it to both; forget,
and the model quietly advances the gtrack producer twice per timestep and pays
for two point clouds. A factory holds one source and hands it to every
connector it creates, so the sharing is the default rather than a thing to
remember.

``ConnectorFactory`` is the general form: give it the classes to use, or assign
the objects directly. ``LithosphereConnectorFactory`` and
``PolygonConnectorFactory`` are presets that fix those classes for the global
and bounded cases and narrow the signatures to the arguments that apply.

Every slot is write-once, and the connectors are cached properties. Reassigning
a source after a connector exists would leave the two disagreeing about which
points they read, so the factory refuses instead.
"""

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

    The indicator and the geotherm share one source. Each slot takes exactly
    one object, which a caller either assigns directly or builds through the
    matching ``create_*`` method, and a second attempt at the same slot raises.

    Args:
        source_class: Source class that ``create_source`` instantiates.
        output_class: Output class that ``create_indicator`` instantiates.
        geotherm_output_class: Output class that ``create_geotherm``
            instantiates.
        mesh: ``MeshConfig`` forwarded to every ``ScalarFieldConnector`` this
            factory creates.
        interpolation: ``InterpolationConfig`` forwarded to every
            ``ScalarFieldConnector`` this factory creates.
        gc_collect_frequency: Forwarded to every ``ScalarFieldConnector`` this
            factory creates; see ``ScalarFieldConnector`` for the semantics.

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
        """Return the source that every connector from this factory shares."""
        return self._source

    @source.setter
    def source(self, source: Source | None):
        if self._source is not None:
            raise RuntimeError("This factory already has a source.")
        self._source = source

    def create_source(self, *source_args, **source_kwargs):
        """Create the shared source with the configured source class.

        Args:
            *source_args: Positional arguments for the source class.
            **source_kwargs: Keyword arguments for the source class.

        Raises:
            RuntimeError: If this factory already has a source.
            TypeError: If no source class was configured.
        """
        if self._source is not None:
            raise RuntimeError("This factory already has a source.")
        if self._source_class is None:
            raise TypeError("The source class is not configured.")
        self._source = self._source_class(*source_args, **source_kwargs)

    @property
    def output(self):
        """Return the output strategy that creates the indicator field."""
        return self._output

    @output.setter
    def output(self, output: OutputStrategy | None):
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator output.")
        self._output = output

    def create_indicator(self, **output_kwargs):
        """Create the indicator output with the configured output class.

        Args:
            **output_kwargs: Keyword arguments for the output class.

        Raises:
            RuntimeError: If this factory already has an indicator output.
            TypeError: If no indicator output class was configured.
        """
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator output.")
        if self._output_class is None:
            raise TypeError("The indicator output class is not configured.")
        self._output = self._output_class(**output_kwargs)

    @property
    def geotherm_output(self):
        """Return the output strategy that creates the geotherm field."""
        return self._geotherm_output

    @geotherm_output.setter
    def geotherm_output(self, geotherm_output: OutputStrategy | None):
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm output.")
        self._geotherm_output = geotherm_output

    def create_geotherm(self, **output_kwargs):
        """Create the geotherm output with the configured output class.

        Args:
            **output_kwargs: Keyword arguments for the output class.

        Raises:
            RuntimeError: If this factory already has a geotherm output.
            TypeError: If no geotherm output class was configured.
        """
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm output.")
        if self._geotherm_output_class is None:
            raise TypeError("The geotherm output class is not configured.")
        self._geotherm_output = self._geotherm_output_class(**output_kwargs)

    @cached_property
    def indicator(self):
        """Return the indicator connector, building it on first access.

        The connector revalidates the source against the output, because a
        caller can build a ``ScalarFieldConnector`` directly and bypass this
        factory altogether.

        Returns:
            The ``ScalarFieldConnector`` pairing the shared source with the
            indicator output.

        Raises:
            RuntimeError: If the source or the indicator output is missing.
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
        """Return the geotherm connector, building it on first access.

        Returns:
            The ``ScalarFieldConnector`` pairing the shared source with the
            geotherm output.

        Raises:
            RuntimeError: If the source or the geotherm output is missing.
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

    Fixes the classes to ``PointCloudSource``, ``GlobalLayerIndicator`` and
    ``HalfSpaceCoolingGeotherm``, which is the combination for oceanic
    lithosphere: present everywhere, with plate age driving the thermal
    structure.

    Args:
        mesh: ``MeshConfig`` forwarded to every ``ScalarFieldConnector`` this
            factory creates.
        interpolation: ``InterpolationConfig`` forwarded to every
            ``ScalarFieldConnector`` this factory creates.
        gc_collect_frequency: Forwarded to every ``ScalarFieldConnector`` this
            factory creates; see ``ScalarFieldConnector`` for the semantics.
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
        """Create the shared source from a gtrack lithosphere producer.

        Args:
            producer: gtrack source satisfying the ``AgeCloudSource`` protocol.
            gplates_connector: Plate-model time mapping and maximum age.
            comm: MPI communicator for the source broadcast.
        """
        super().create_source(producer, gplates_connector, comm=comm)

    def create_indicator(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
        fallback_thickness_km: float = 100.0,
        lateral_weight: LateralWeight | None = None,
    ):
        """Create the indicator output as a global layer indicator.

        Args:
            base_transition_width_km: Width of the radial transition, in
                kilometres.
            fixed_base_depth_km: One base depth for all target nodes, in
                kilometres. If None, the ``thickness`` channel sets each base
                depth.
            lateral_weight: Lateral-weight strategy. Defaults to
                ``UniformLateralWeight``, which returns one at every target
                node.
            fallback_thickness_km: Thickness used outside the source range, in
                kilometres.

        Examples:
            >>> factory.create_indicator(lateral_weight=SourceLateralWeight())
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
        """Create the geotherm output as a half-space cooling geotherm.

        Args:
            thermal_diffusivity_m2_per_s: Thermal diffusivity, in square metres
                per second.
            fallback_thickness_km: Thickness used outside the source range, in
                kilometres.
            fallback_age_myr: Material age used outside the source range, in
                millions of years.
            geotherm: Profile function. Defaults to ``ocean_erf_normalized``.
        """
        super().create_geotherm(
            thermal_diffusivity_m2_per_s=thermal_diffusivity_m2_per_s,
            fallback_thickness_km=fallback_thickness_km,
            fallback_age_myr=fallback_age_myr,
            geotherm=geotherm,
        )


class PolygonConnectorFactory(ConnectorFactory):
    """Create connectors for a bounded polygon source.

    Fixes the classes to ``PointCloudSource``, ``BoundedLayerIndicator`` and
    ``BoundedLinearGeotherm``, which is the combination for a layer confined to
    polygons, such as continental lithosphere.

    Args:
        mesh: ``MeshConfig`` forwarded to every ``ScalarFieldConnector`` this
            factory creates.
        interpolation: ``InterpolationConfig`` forwarded to every
            ``ScalarFieldConnector`` this factory creates.
        gc_collect_frequency: Forwarded to every ``ScalarFieldConnector`` this
            factory creates; see ``ScalarFieldConnector`` for the semantics.
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
        """Create the shared source from a gtrack polygon producer.

        Args:
            producer: gtrack source satisfying the ``AgeCloudSource`` protocol.
            gplates_connector: Plate-model time mapping and maximum age.
            comm: MPI communicator for the source broadcast.
        """
        super().create_source(producer, gplates_connector, comm=comm)

    def create_indicator(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
    ):
        """Create the indicator output as a bounded layer indicator.

        Args:
            base_transition_width_km: Width of the radial transition, in
                kilometres.
            fixed_base_depth_km: One base depth for all target nodes, in
                kilometres. If None, the membership-corrected thickness sets
                each base depth.
        """
        super().create_indicator(
            base_transition_width_km=base_transition_width_km,
            fixed_base_depth_km=fixed_base_depth_km,
        )

    def create_geotherm(
        self,
        geotherm: Callable | None = None,
    ):
        """Create the geotherm output as a bounded linear geotherm.

        Args:
            geotherm: Profile function. Defaults to ``continental_linear``.
        """
        super().create_geotherm(
            geotherm=geotherm,
        )
