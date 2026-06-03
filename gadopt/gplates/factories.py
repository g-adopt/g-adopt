"""---------------------------------------------------------------------------
Factory class
---------------------------------------------------------------------------
The ConnectorFactory class controls the ways in which users can combine
various Source, Output and ScalarFieldConnector objects. (e.g. disallow
the construction of a Source if the factory has already been assigned a
Source). Constructions for objects in this class is done so on an as-needed
basis. Common combinations of Source + Output + Geotherm can be created by
subclassing ConnectorFactory with a call to super().__init__() with the
source_class, output_class and geotherm_output_class specified.
"""

import numpy as np
from mpi4py import MPI
from typing import Type, Any, Callable, TYPE_CHECKING
from functools import cached_property

from .connectors import ScalarFieldConnector
from .outputs import OutputStrategy, TanhOutput, GeothermERFOutput
from .sources import Source, LithosphereSource, LithosphereSourceConfig

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector, PlateModelFiles

__all__ = [
    "ConnectorFactory",
    "LithosphereIndicator",
]


class ConnectorFactory:
    """Control the creation of linked Source, Output and Geotherm objects

    `ConnectorFactory` manages the creation of `ScalarFieldConnectors` based on
    Source and Output objects it manages. The `ConnectorFactory` can take existing
    Sources and Outputs and use those to construct `ScalarFieldConnectors` or can
    initialise Source and Output objects based on classes provided at construction
    time and parameters passed to the `construct_<object>` functions. All arguments
    are optional, however there are some mandatory combinations. If a `source_class`
    is not provided, the `source` attribute must be set before the indicator is
    constructed. If `source_class` is provided, but `construct_source` is not called,
    `self.indicator` will attempt to construct a Source with no arguments, which is
    very likely to fail. If `output_class` is not provided and `output` is not set,
    a TanhOutput with no input parameters is constructed by `self.indicator` and the
    user is warned. If `geotherm_output_class` is not provided and `geotherm_output`
    is not set, a `GeothermERFOutput` is constructed and the user is warned

    Args:
        source_class: The type of Source object to construct
        output_class: The type of Output object to construct
        geotherm_output_class: The type of Output to construct for the associated geotherm.

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
        >>> indicator = factory.indicator
    """

    def __init__(
        self,
        source_class: Type[Source] | None = None,
        output_class: Type[OutputStrategy] | None = None,
        geotherm_output_class: Type[OutputStrategy] | None = None,
    ):
        self._source_class = source_class
        self._output_class = output_class
        self._geotherm_output_class = geotherm_output_class
        self._indicator: ScalarFieldConnector | None = None
        self._geotherm: ScalarFieldConnector | None = None
        self._source: Source | None = None
        self._output: OutputStrategy | None = None
        self._geotherm_output: OutputStrategy | None = None
        self._conn_params: dict[str, Any] = {
            "mesh": None,
            "interpolation": None,
            "gc_collect_frequency": 10,
        }

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
            raise RuntimeError("This indicator already has a Source!")
        self._source = source

    def construct_source(self, *source_args, **source_kwargs):
        """Have this ConnectorFactory construct a Source object

        All input arguments are passed directory to the `self._source_class`
        constructor, which must be set on initialisation of this ConnectorFactory
        object. Note that calling this function is mutually exclusive to calling
        the `source` setter.

        Raises:
            RuntimeError: This ConnectorFactory is already managing a `Source` object
            TypeError: Attempted to construct a Source without setting `_source_class`
        """
        if self._source is not None:
            raise RuntimeError("This indicator already has a Source!")
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
            raise RuntimeError("This indicator already has an Output!")
        self._output = output

    def construct_output(self, **output_kwargs):
        """Have this ConnectorFactory construct an `OutputStrategy` object for an
        indicator

        All input arguments are passed directory to the `self._output_class`
        constructor, which must be set on initialisation of this ConnectorFactory
        object. Note that calling this function is mutually exclusive to calling
        the `output` setter.

        Raises:
            RuntimeError: This ConnectorFactory is already managing an indicator
                          `OutputStrategy` object
            TypeError: Attempted to construct `output` without setting `_output_class`
        """
        if self._output is not None:
            raise RuntimeError("This indicator already has an Output!")
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
            raise RuntimeError("This indicator already has a Geotherm!")
        self._geotherm_output = geotherm_output

    def construct_geotherm(self, **output_kwargs):
        """Have this ConnectorFactory construct an `OutputStrategy` object for a
        geotherm

        All input arguments are passed directory to the `self._geotherm_output_class`
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
            raise RuntimeError("This indicator already has a Geotherm!")
        if self._geotherm_output_class is None:
            raise TypeError("Do not know what kind of Geotherm to construct!")
        self._geotherm_output = self._geotherm_output_class(**output_kwargs)

    def update_connector_params(self, **kwargs):
        """Update `ScalarFieldConnector` parameters

        A default set of parameters to be passed to the `ScalarFieldConnector`
        constructor is initialised in `self._conn_params`. This function allows
        the user to update those parameters before initialising the
        `ScalarFieldConnector`
        """
        self._conn_params |= kwargs

    @cached_property
    def indicator(self):
        """Construct and retrieve the indicator `ScalarFieldConnector`.

        This function creates the `ScalarFieldConnector` for the indicator. If
        Source and/or OutputStrategy objects have not been created, this function
        will raise a RuntimeError. `cached_property` is used to ensure sanity checks
        and object creation only run once.

        Returns:
            `ScalarFieldConnector` for indicator

        Raises:
            RuntimeError: Attempted to construct the isotherm without a source or
                          output is present.
        """
        if self._source is None:
            raise RuntimeError(
                "A source must be either constructed or connected in order to construct the indicator"
            )
        if self._output is None:
            raise RuntimeError(
                "An output must be either constructed or connected in order to construct the indicator"
            )
        return ScalarFieldConnector(self._source, self._output, **self._conn_params)

    @cached_property
    def geotherm(self):
        """Construct and retrieve the indicator `ScalarFieldConnector`.

        This function creates the `ScalarFieldConnector` for the geotherm. If
        Source and/or `OutputStrategy` objects have not been created , this function
        will attempt to create them using default classes and parameters, and warn the
        user that it is doing so. `cached_property` is used to ensure sanity checks
        and object creation only run once.

        Returns:
            `ScalarFieldConnector` for indicator

        Raises:
            RuntimeError: Attempted to construct the isotherm without a source or
                          geotherm_output is present.
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
            self._source, self._geotherm_output, **self._conn_params
        )


class LithosphereIndicator(ConnectorFactory):
    """A subclass of ConnectorFactory used for constructing Lithosphere objects

    `LithosphereIndicator` ties together a `LithosphereSource`, `TanhOutput` and
    `GeothermERFOutput` to create a convencience class for a common combination
    of Sources and Outputs.
    """

    def __init__(self):
        super().__init__(LithosphereSource, TanhOutput, GeothermERFOutput)

    def construct_source(
        self,
        gplates_connector: "pyGplatesConnector",
        continental_data,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        plate_files: "PlateModelFiles",
        config: LithosphereSourceConfig | None = None,
        *,
        default_continental_age_myr: float = 500.0,
        walk_start_age: float | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        """Overloaded construct_source

        Match argument list to `LithosphereSource` to allow static argument checking
        and IDE introspection
        """

        return super().construct_source(
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
        **output_kwargs,
    ):
        """Overloaded construct_output

        Match argument list to `TanhOutput` to allow static argument checking
        and IDE introspection
        """
        return super().construct_output(
            transition_width_km=transition_width_km,
            default_thickness_km=default_thickness_km,
            **output_kwargs,
        )

    def construct_geotherm(
        self,
        kappa: float = 1e-6,
        default_thickness_km: float = 100.0,
        too_far_age_myr: float = 500.0,
        geotherm: Callable | None = None,
        **output_kwargs,
    ):
        """Overloaded construct_geotherm

        Match argument list to `GeothermERFOutput` to allow static argument checking
        and IDE introspection
        """
        return super().construct_geotherm(
            kappa=kappa,
            default_thickness_km=default_thickness_km,
            too_far_age_myr=too_far_age_myr,
            geotherm=geotherm,
            **output_kwargs,
        )
