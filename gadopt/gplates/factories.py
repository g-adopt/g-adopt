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

import numpy as np
from mpi4py import MPI
from typing import Callable, TYPE_CHECKING
from functools import cached_property

from .connectors import ScalarFieldConnector, InterpolationConfig
from .outputs import MeshConfig, OutputStrategy, TanhOutput, GeothermERFOutput
from .sources import CloudDataType, Source, LithosphereSource, LithosphereSourceConfig

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector, PlateModelFiles

__all__ = [
    "ConnectorFactory",
    "LithosphereConnectorFactory",
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

        >>> factory = ConnectorFactory(source_class=LithosphereSource, output_class=TanhOutput)
        >>> factory.construct_source(
        ...    gplates_connector=plate_model_with_polygons,
        ...    continental_data=synthetic_data,
        ...    age_to_property=half_space_cooling,
        ...    plate_files=plate_files,
        ... )
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

    `LithosphereConnectorFactory` ties together a `LithosphereSource`,
    `TanhOutput` and `GeothermERFOutput` to create a convenience class for a
    common combination of Sources and Outputs.

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
            LithosphereSource,
            TanhOutput,
            GeothermERFOutput,
            mesh=mesh,
            interpolation=interpolation,
            gc_collect_frequency=gc_collect_frequency,
        )

    def construct_source(
        self,
        gplates_connector: "pyGplatesConnector",
        continental_data: CloudDataType,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        plate_files: "PlateModelFiles",
        config: LithosphereSourceConfig | None = None,
        *,
        default_continental_age_myr: float = 500.0,
        walk_start_age: float | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Overloaded construct_source

        Match argument list to `LithosphereSource` to allow static argument
        checking and IDE introspection; see `LithosphereSource` for the
        meaning of each argument.
        """
        super().construct_source(
            gplates_connector,
            continental_data,
            age_to_property,
            plate_files,
            config,
            default_continental_age_myr=default_continental_age_myr,
            walk_start_age=walk_start_age,
            comm=comm,
        )

    def construct_output(
        self,
        transition_width_km: float = 10.0,
        default_thickness_km: float = 100.0,
    ):
        """Overloaded construct_output

        Match argument list to `TanhOutput` to allow static argument checking
        and IDE introspection; see `TanhOutput` for the meaning of each
        argument.
        """
        super().construct_output(
            transition_width_km=transition_width_km,
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
