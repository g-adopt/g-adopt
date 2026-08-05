"""---------------------------------------------------------------------------
Factory class
---------------------------------------------------------------------------
The ConnectorFactory class controls the ways in which users can combine
various Source, Output and ScalarFieldConnector objects (e.g. disallow
the construction of a Source if the factory has already been assigned a
Source). Construction of objects in this class happens on an as-needed
basis. Common combinations of Source + Output + Geotherm can be created by
subclassing ConnectorFactory with a call to super().__init__() with the
source_class, output_class and geotherm_output_class specified.
"""

from mpi4py import MPI
from typing import Callable, TYPE_CHECKING
from functools import cached_property

from .connectors import ScalarFieldConnector
from .interpolation import InterpolationConfig
from .outputs import (
    GeothermERFOutput,
    MaskedGeothermLinearOutput,
    MeshConfig,
    OutputStrategy,
    MaskedQuinticOutput,
    QuinticOutput,
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
    """Control the creation of linked Source, Output and Geotherm objects.

    `ConnectorFactory` manages the creation of `ScalarFieldConnector` objects
    based on the Source and Output objects it manages. Because the factory
    holds a single Source, the indicator and geotherm connectors it creates
    share that Source by construction — the underlying (possibly stateful)
    machinery advances once per geological age no matter how many connectors
    it feeds.

    The factory can take existing Source and Output objects (via the
    `source`, `output` and `geotherm_output` setters) or construct them from
    classes provided at construction time (via the `construct_<object>`
    methods). The two routes are mutually exclusive per slot: a factory
    refuses a second Source or a second indicator/geotherm Output, however
    they were made. Accessing `indicator` (or `geotherm`) before both the
    Source and the corresponding Output exist raises a `RuntimeError`;
    nothing is defaulted silently.

    Args:
        source_class: The type of Source object to construct.
        output_class: The type of Output object to construct for the indicator.
        geotherm_output_class: The type of Output object to construct for the
            associated geotherm.
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: forwarded to every `ScalarFieldConnector` this
            factory creates; see `ScalarFieldConnector` for the semantics.

    Examples:
        >>> factory = ConnectorFactory()
        >>> factory.source = source
        >>> factory.output = output
        >>> indicator = factory.indicator

        >>> factory = ConnectorFactory(source_class=PointCloudSource, output_class=QuinticOutput)
        >>> factory.construct_source(producer=lithosphere_cloud_source, gplates_connector=plate_model)
        >>> factory.construct_output()
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
        """The Source object used to construct the `ScalarFieldConnector`

        The initial value is None, and can be set to an initialised Source object
        by the setter.

        Returns:
            The Source object

        Raises:
            RuntimeError: This ConnectorFactory is already managing a `Source` object
        """
        return self._source

    @source.setter
    def source(self, source: Source | None):
        if self._source is not None:
            raise RuntimeError("This factory already has a Source!")
        self._source = source

    def construct_source(self, *source_args, **source_kwargs):
        """Have this ConnectorFactory construct a Source object

        All input arguments are passed directly to the `self._source_class`
        constructor, which must be set on initialisation of this ConnectorFactory
        object. Note that calling this function is mutually exclusive to calling
        the `source` setter.

        Raises:
            RuntimeError: This ConnectorFactory is already managing a `Source` object
            TypeError: Attempted to construct a Source without setting `_source_class`
        """
        if self._source is not None:
            raise RuntimeError("This factory already has a Source!")
        if self._source_class is None:
            raise TypeError("Do not know what kind of Source to construct!")
        self._source = self._source_class(*source_args, **source_kwargs)

    @property
    def output(self):
        """The `OutputStrategy` object used to construct the `ScalarFieldConnector`
        belonging to the indicator created by this `ConnectorFactory` object

        The initial value is None, and can be set to an initialised `OutputStrategy`
        object by the setter.

        Returns:
            The `OutputStrategy` object

        Raises:
            RuntimeError: This ConnectorFactory is already managing an indicator
                          `OutputStrategy` object
        """
        return self._output

    @output.setter
    def output(self, output: OutputStrategy | None):
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator Output!")
        self._output = output

    def construct_output(self, **output_kwargs):
        """Have this ConnectorFactory construct an `OutputStrategy` object for an
        indicator

        All input arguments are passed directly to the `self._output_class`
        constructor, which must be set on initialisation of this ConnectorFactory
        object. Note that calling this function is mutually exclusive to calling
        the `output` setter.

        Raises:
            RuntimeError: This ConnectorFactory is already managing an indicator
                          `OutputStrategy` object
            TypeError: Attempted to construct `output` without setting `_output_class`
        """
        if self._output is not None:
            raise RuntimeError("This factory already has an indicator Output!")
        if self._output_class is None:
            raise TypeError("Do not know what kind of Output to construct!")
        self._output = self._output_class(**output_kwargs)

    @property
    def geotherm_output(self):
        """The `OutputStrategy` object used to construct the `ScalarFieldConnector`
        belonging to the geotherm created by this `ConnectorFactory` object

        The initial value is None, and can be set to an initialised Output object
        by the setter.

        Returns:
            The `OutputStrategy` object

        Raises:
            RuntimeError: This ConnectorFactory is already managing a geotherm
                          `OutputStrategy` object

        """
        return self._geotherm_output

    @geotherm_output.setter
    def geotherm_output(self, geotherm_output: OutputStrategy | None):
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm Output!")
        self._geotherm_output = geotherm_output

    def construct_geotherm(self, **output_kwargs):
        """Have this ConnectorFactory construct an `OutputStrategy` object for a
        geotherm

        All input arguments are passed directly to the `self._geotherm_output_class`
        constructor, which must be set on initialisation of this ConnectorFactory
        object. Note that calling this function is mutually exclusive to calling
        the `geotherm_output` setter.

        Raises:
            RuntimeError: This ConnectorFactory is already managing a geotherm
                          `OutputStrategy` object
            TypeError: Attempted to construct `geotherm_output` without setting
                       `geotherm_output_class`
        """
        if self._geotherm_output is not None:
            raise RuntimeError("This factory already has a geotherm Output!")
        if self._geotherm_output_class is None:
            raise TypeError("Do not know what kind of geotherm Output to construct!")
        self._geotherm_output = self._geotherm_output_class(**output_kwargs)

    @cached_property
    def indicator(self):
        """Construct and retrieve the indicator `ScalarFieldConnector`.

        This function creates the `ScalarFieldConnector` for the indicator. If
        the Source and/or the indicator `OutputStrategy` object have not been
        created (or assigned), this function raises a RuntimeError — nothing is
        defaulted silently. `cached_property` is used to ensure sanity checks
        and object creation only run once.

        Returns:
            `ScalarFieldConnector` for the indicator

        Raises:
            RuntimeError: Attempted to construct the indicator while no source
                          or output is present.
        """
        if self._source is None:
            raise RuntimeError(
                "A source must be either constructed or connected in order to construct the indicator"
            )
        if self._output is None:
            raise RuntimeError(
                "An output must be either constructed or connected in order to construct the indicator"
            )
        # The source/output pairing check lives in ScalarFieldConnector.__init__,
        # not here: this factory is only one route to a connector, and a caller
        # constructing one directly would bypass a check placed at this level.
        return ScalarFieldConnector(
            self._source,
            self._output,
            mesh=self._mesh,
            interpolation=self._interpolation,
            gc_collect_frequency=self._gc_collect_frequency,
        )

    @cached_property
    def geotherm(self):
        """Construct and retrieve the geotherm `ScalarFieldConnector`.

        This function creates the `ScalarFieldConnector` for the geotherm. If
        the Source and/or the geotherm `OutputStrategy` object have not been
        created (or assigned), this function raises a RuntimeError — nothing is
        defaulted silently. `cached_property` is used to ensure sanity checks
        and object creation only run once.

        Returns:
            `ScalarFieldConnector` for the geotherm

        Raises:
            RuntimeError: Attempted to construct the geotherm while no source
                          or geotherm_output is present.
        """
        if self._source is None:
            raise RuntimeError(
                "A source must be either constructed or connected in order to construct the geotherm"
            )
        if self._geotherm_output is None:
            raise RuntimeError(
                "A geotherm_output must be either constructed or connected in order to construct the geotherm"
            )
        return ScalarFieldConnector(
            self._source,
            self._geotherm_output,
            mesh=self._mesh,
            interpolation=self._interpolation,
            gc_collect_frequency=self._gc_collect_frequency,
        )


class LithosphereConnectorFactory(ConnectorFactory):
    """A subclass of ConnectorFactory used for constructing Lithosphere objects

    `LithosphereConnectorFactory` ties together a `PointCloudSource` (wrapping
    a gtrack `LithosphereCloudSource` producer), `QuinticOutput` and
    `GeothermERFOutput` to create a convenience class for a common combination
    of Sources and Outputs.

    Args:
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: forwarded to every `ScalarFieldConnector` this
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
            QuinticOutput,
            GeothermERFOutput,
            mesh=mesh,
            interpolation=interpolation,
            gc_collect_frequency=gc_collect_frequency,
        )

    def construct_source(
        self,
        producer: "AgeCloudSource",
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Overloaded construct_source

        Wrap a gtrack producer — build a `LithosphereCloudSource` (its plate
        files, tracer config, checkpointing and walk_start_age all live on it
        now) and pass it here. Match the argument list to `PointCloudSource` to
        allow static argument checking and IDE introspection.
        """
        super().construct_source(producer, gplates_connector, comm=comm)

    def construct_output(
        self,
        width_km: float = 10.0,
        *,
        base_depth_km: float | None = None,
        default_thickness_km: float = 100.0,
    ):
        """Overloaded construct_output

        Match argument list to `QuinticOutput` to allow static argument
        checking and IDE introspection; see `QuinticOutput` for the meaning
        of each argument.

        The lithosphere covers the whole sphere and its thickness channel
        never vanishes laterally (``default_thickness_km`` fills uncovered
        nodes), so the plain one-sided step is the right indicator here: every
        column is inside the region and only its base depth varies.
        """
        super().construct_output(
            width_km=width_km,
            base_depth_km=base_depth_km,
            default_thickness_km=default_thickness_km,
        )

    def construct_geotherm(
        self,
        kappa: float = 1e-6,
        default_thickness_km: float = 100.0,
        too_far_age_myr: float = 500.0,
        geotherm: Callable | None = None,
    ):
        """Overloaded construct_geotherm

        Match argument list to `GeothermERFOutput` to allow static argument
        checking and IDE introspection; see `GeothermERFOutput` for the
        meaning of each argument.
        """
        super().construct_geotherm(
            kappa=kappa,
            default_thickness_km=default_thickness_km,
            too_far_age_myr=too_far_age_myr,
            geotherm=geotherm,
        )


class PolygonConnectorFactory(ConnectorFactory):
    """A subclass of ConnectorFactory used for constructing polygon-bounded objects

    `PolygonConnectorFactory` ties together a `PointCloudSource` (wrapping a
    gtrack `PolygonIndicatorSource` producer), `MaskedQuinticOutput` and
    `MaskedGeothermLinearOutput` to create a convenience class for a common
    combination of Sources and Outputs.

    Args:
        mesh: `MeshConfig` forwarded to every `ScalarFieldConnector` this
            factory creates.
        interpolation: `InterpolationConfig` forwarded to every
            `ScalarFieldConnector` this factory creates.
        gc_collect_frequency: forwarded to every `ScalarFieldConnector` this
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
            MaskedQuinticOutput,
            MaskedGeothermLinearOutput,
            mesh=mesh,
            interpolation=interpolation,
            gc_collect_frequency=gc_collect_frequency,
        )

    def construct_source(
        self,
        producer: "AgeCloudSource",
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Overloaded construct_source

        Wrap a gtrack producer — build a `PolygonIndicatorSource` (its polygon
        files, static polygons, thickness data and background/exclusion config
        all live on it now) and pass it here. Match the argument list to
        `PointCloudSource` to allow static argument checking and IDE
        introspection.
        """
        super().construct_source(producer, gplates_connector, comm=comm)

    def construct_output(
        self,
        width_km: float = 10.0,
        *,
        base_depth_km: float | None = None,
    ):
        """Overloaded construct_output

        Match argument list to `MaskedQuinticOutput` to allow static argument
        checking and IDE introspection; see `MaskedQuinticOutput` for the
        meaning of each argument.

        There is no ``default_thickness_km``: a node with no source point
        nearby is a statement about membership, not about depth, and
        `MaskedQuinticOutput` treats it as outside the region rather than
        filling in a fallback thickness. Where the region lies is answered by
        ``membership``; how deep it goes is answered by ``thickness``, or by
        ``base_depth_km`` when the region has no meaningful depth data.
        """
        super().construct_output(
            width_km=width_km,
            base_depth_km=base_depth_km,
        )

    def construct_geotherm(
        self,
        geotherm: Callable | None = None,
    ):
        """Overloaded construct_geotherm

        Match argument list to `MaskedGeothermLinearOutput` to allow static
        argument checking and IDE introspection; see `MaskedGeothermLinearOutput`
        for the meaning of each argument.
        """
        super().construct_geotherm(
            geotherm=geotherm,
        )
