"""Abstract base classes for GPlates indicator connectors.

This module defines the interface that all scalar indicator connectors must
implement. Indicator connectors produce time-dependent 3D scalar fields
(values in [0, 1]) that can be used to modify material properties such as
viscosity in geodynamic simulations.

Examples of indicator connectors:
- LithosphereConnector: Lithosphere indicator from ocean ages + continental data
- CratonConnector: Craton indicator from polygon shapefiles

All indicator connectors share a common interface:
- get_indicator(target_coords, ndtime) -> np.ndarray
- Time conversion via gplates_connector (ndtime2age, age2ndtime)
- MPI support via comm attribute
- Configuration via config + config_extra pattern
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector


@dataclass
class IndicatorConfigBase:
    """Base configuration pattern for indicator connectors.

    This is not meant to be instantiated directly. Subclasses should define
    their own config dataclass with specific parameters.

    All indicator configs should implement:
    - to_dict() -> dict
    - from_dict(dict) -> Config
    - with_overrides(dict) -> Config

    Common parameters that most indicator configs will have:
    - n_points: Number of sample points for spatial coverage
    - k_neighbors: Neighbors for interpolation
    - distance_threshold: Max distance for valid interpolation
    - r_outer: Outer mesh radius (non-dimensional)
    """

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary.

        Unknown keys are ignored, allowing forward compatibility.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)

    def with_overrides(self, overrides: dict):
        """Create a new config with specified overrides.

        Parameters
        ----------
        overrides : dict
            Dictionary of parameter overrides.

        Returns
        -------
        Config
            New configuration with overrides applied.
        """
        current = self.to_dict()
        current.update(overrides)
        return self.from_dict(current)


class IndicatorConnector(ABC):
    """Abstract base class for scalar indicator field connectors.

    Indicator connectors produce time-dependent scalar fields that are ~1 in
    regions of interest (e.g., lithosphere, cratons) and ~0 elsewhere, with
    smooth transitions at boundaries. These fields can be used to modify
    material properties in geodynamic simulations.

    Subclasses must implement:
    - get_indicator(target_coords, ndtime) -> np.ndarray

    Subclasses must have these attributes:
    - gplates_connector: pyGplatesConnector for plate model and time conversion
    - config: Configuration dataclass with tunable parameters
    - comm: MPI communicator (None for serial execution)
    - reconstruction_age: Last computed geological age (for caching)

    The config + config_extra pattern:
        Subclasses should accept both `config` and `config_extra` in __init__.
        If config is None, use a module-level default. If config_extra is
        provided, apply overrides via config.with_overrides(config_extra).

    MPI parallelization:
        When comm is provided, rank 0 should perform I/O and heavy computation,
        then broadcast results to other ranks. Each rank interpolates to its
        local mesh points.

    Examples
    --------
    >>> # Creating a custom indicator connector
    >>> class MyConnector(IndicatorConnector):
    ...     def __init__(self, gplates_connector, config=None, config_extra=None, comm=None):
    ...         self.gplates_connector = gplates_connector
    ...         self.comm = comm
    ...         self.reconstruction_age = None
    ...         # Build config with defaults and overrides
    ...         if config is None:
    ...             config = MyConfigDefault
    ...         if config_extra is not None:
    ...             config = config.with_overrides(config_extra)
    ...         self.config = config
    ...
    ...     def get_indicator(self, target_coords, ndtime):
    ...         # Implementation here
    ...         pass
    """

    # Required attributes (set by subclasses)
    gplates_connector: "pyGplatesConnector"
    config: Any  # Subclass-specific config dataclass
    comm: Any  # MPI communicator or None
    reconstruction_age: float  # Last computed age (Ma)

    @abstractmethod
    def get_indicator(
        self,
        target_coords: np.ndarray,
        ndtime: float
    ) -> np.ndarray:
        """Get indicator values at target coordinates for given time.

        Returns a scalar field that is ~1 in regions of interest and ~0
        elsewhere, with smooth transitions at boundaries.

        When running with MPI (comm is set), rank 0 performs heavy computation
        and broadcasts results. All ranks then interpolate to local mesh points.

        Parameters
        ----------
        target_coords : np.ndarray
            (M, 3) array of mesh coordinates in mesh units.
            Each MPI rank provides its local mesh coordinates.
        ndtime : float
            Non-dimensional time.

        Returns
        -------
        np.ndarray
            (M,) array of indicator values in [0, 1].
        """
        ...

    def ndtime2age(self, ndtime: float) -> float:
        """Convert non-dimensional time to geological age (Ma).

        Delegates to gplates_connector.

        Parameters
        ----------
        ndtime : float
            Non-dimensional time.

        Returns
        -------
        float
            Geological age in millions of years before present.
        """
        return self.gplates_connector.ndtime2age(ndtime)

    def age2ndtime(self, age: float) -> float:
        """Convert geological age (Ma) to non-dimensional time.

        Delegates to gplates_connector.

        Parameters
        ----------
        age : float
            Geological age in millions of years before present.

        Returns
        -------
        float
            Non-dimensional time.
        """
        return self.gplates_connector.age2ndtime(age)

    @property
    def delta_t(self) -> float:
        """Time window for caching decisions.

        If the time change is less than delta_t, cached results may be reused.
        Delegates to gplates_connector.delta_t.

        Returns
        -------
        float
            Time window in Myr.
        """
        return self.gplates_connector.delta_t
