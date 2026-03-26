"""This module provides base classes and shared infrastructure for GPlates indicator
connectors. Indicator connectors produce time-dependent 3D scalar fields (values in
[0, 1]) that modify material properties such as viscosity in geodynamic simulations.

IndicatorConfigBase is a dataclass base with serialisation helpers (to_dict, from_dict,
with_overrides). IndicatorConnector is a template-method abstract base class whose
get_indicator method orchestrates validation, caching, MPI broadcast, IDW interpolation,
and indicator computation. Subclasses only need to implement _prepare_sources(age).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
from pathlib import Path
# TYPE_CHECKING: False at runtime, True for static type checkers (avoids runtime overhead)
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from ..utility import log, DEBUG

if TYPE_CHECKING:
    from gtrack import PointCloud
    from .gplates import pyGplatesConnector


@dataclass
class IndicatorConfigBase:
    """Base configuration for indicator connectors.

    Not instantiated directly. Subclass with domain-specific fields.
    Provides serialisation helpers and shared field validation.

    Shared fields validated here:
        n_points, k_neighbors, distance_threshold, default_thickness,
        r_outer, depth_scale, transition_width.
    """

    def __post_init__(self):
        """Validate fields common to all indicator configs."""
        if hasattr(self, "n_points") and self.n_points < 100:
            raise ValueError(f"n_points must be at least 100, got {self.n_points}")
        if hasattr(self, "k_neighbors") and self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {self.k_neighbors}")
        if hasattr(self, "distance_threshold") and self.distance_threshold <= 0:
            raise ValueError(
                f"distance_threshold must be positive, got {self.distance_threshold}"
            )
        if hasattr(self, "default_thickness") and self.default_thickness < 0:
            raise ValueError(
                f"default_thickness must be non-negative, got {self.default_thickness}"
            )
        if hasattr(self, "r_outer") and self.r_outer <= 0:
            raise ValueError(f"r_outer must be positive, got {self.r_outer}")
        if hasattr(self, "depth_scale") and self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")
        if hasattr(self, "transition_width") and self.transition_width <= 0:
            raise ValueError(
                f"transition_width must be positive, got {self.transition_width}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)

    def with_overrides(self, overrides: dict):
        """Return a new config with *overrides* applied on top of current values."""
        current = self.to_dict()
        current.update(overrides)
        return self.from_dict(current)


class IndicatorConnector(ABC):
    """Template-method base class for indicator field connectors.

    Subclasses must:
      1. Set gplates_connector, config, comm, _is_root, reconstruction_age,
         _cached_result, _cached_coords_hash, _transition_width_nondim,
         _polygon_filter, _rotator, _PointCloud in __init__.
      2. Implement _prepare_sources(age) -> dict[str, np.ndarray].

    Optionally override:
      - _validate_age_extra(age): additional age checks (e.g. ocean tracker).
      - _compute_indicator(sources, target_coords): default uses tanh.
      - _apply_indicator(r_target, thickness_km, too_far): post-IDW step.
      - _empty_indicator(n): what to return when source data is empty.
    """

    # Required attributes (set by subclasses in __init__)
    gplates_connector: "pyGplatesConnector"
    config: Any
    comm: MPI.Comm
    reconstruction_age: float | None
    _is_root: bool
    _cached_result: np.ndarray | None
    _cached_coords_hash: tuple | None
    _transition_width_nondim: float
    _initialized: bool

    # Template method

    def get_indicator(
        self,
        target_coords: np.ndarray,
        ndtime: float,
    ) -> np.ndarray:
        """Get indicator values at target coordinates for a given time.

        Orchestrates validation, caching, source preparation (rank 0 only),
        MPI broadcast, and indicator computation.

        Args:
            target_coords: (M, 3) array of mesh coordinates in mesh units.
            ndtime: Non-dimensional time.

        Returns:
            (M,) array of indicator values in [0, 1].
        """
        # Verify subclass set all required attributes in __init__.
        # Checked once per instance to give a clear TypeError rather than
        # a confusing AttributeError deep in the template method.
        if not getattr(self, "_init_checked", False):
            for attr in ("gplates_connector", "config", "comm", "_is_root",
                         "reconstruction_age", "_cached_result",
                         "_cached_coords_hash", "_transition_width_nondim",
                         "_initialized"):
                if not hasattr(self, attr):
                    raise TypeError(
                        f"{type(self).__name__} must set '{attr}' in __init__"
                    )
            self._init_checked = True

        age = self.ndtime2age(ndtime)

        self._validate_age(age)

        use_cache = self._check_cache(age, target_coords) is not None

        # Ensure all MPI ranks agree on the cache decision. If any rank
        # sees changed coordinates, all ranks must recompute to avoid
        # hanging on the collective broadcast that follows.
        use_cache = self.comm.allreduce(use_cache, op=MPI.MIN)

        if use_cache:
            return self._cached_result

        if self._is_root:
            sources = self._prepare_sources(age)
        else:
            sources = None

        sources = self._broadcast_sources(sources)
        result = self._compute_indicator(sources, target_coords)
        self._update_cache(age, target_coords, result)
        # Mark initialised on all ranks so that _validate_age_extra
        # checks (e.g. backward-time guard) run consistently everywhere.
        self._initialized = True
        return result

    # Hooks for subclasses

    @abstractmethod
    def _prepare_sources(self, age: float) -> dict[str, np.ndarray]:
        """Prepare source data arrays on rank 0.

        Returns a dict of numpy arrays (e.g. 'xyz', 'thickness') that will
        be broadcast to all ranks.
        """
        ...

    def _validate_age_extra(self, age: float):
        """Hook for connector-specific age validation (e.g. ocean tracker)."""

    def _empty_indicator(self, n: int) -> np.ndarray:
        """Indicator returned when source data is empty."""
        return np.zeros(n)

    def _apply_indicator(
        self,
        r_target: np.ndarray,
        thickness_km: np.ndarray,
        too_far: np.ndarray,
    ) -> np.ndarray:
        """Convert interpolated thickness to indicator values.

        Default: use default_thickness for far points and apply tanh.
        PolygonConnector overrides to zero out far points instead.
        """
        thickness_km[too_far] = self.config.default_thickness
        thickness_nondim = thickness_km / self.config.depth_scale
        base_r = self.config.r_outer - thickness_nondim
        indicator = 0.5 * (1.0 + np.tanh(
            (r_target - base_r) / self._transition_width_nondim
        ))
        return indicator

    # Shared concrete methods

    def _validate_age(self, age: float):
        """Validate age is within plate model bounds, then call subclass hook."""
        oldest = self.gplates_connector.oldest_age
        if age > oldest:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's "
                f"oldest age ({oldest:.2f} Ma)."
            )
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future). "
                f"Ages must be >= 0 (present day)."
            )
        self._validate_age_extra(age)

    def _check_cache(self, age: float, target_coords: np.ndarray):
        """Return cached result if age and coordinates haven't changed."""
        if self.reconstruction_age is None:
            return None
        if abs(age - self.reconstruction_age) >= self.delta_t:
            return None
        coords_hash = hash(target_coords.tobytes())
        if self._cached_result is not None and coords_hash == self._cached_coords_hash:
            log(f"{type(self).__name__}: Age {age:.2f} Ma unchanged "
                f"(within delta_t={self.delta_t}), using cached result.")
            return self._cached_result
        return None

    def _update_cache(self, age: float, target_coords: np.ndarray, result: np.ndarray):
        """Store computed result in cache."""
        self.reconstruction_age = age
        self._cached_result = result
        self._cached_coords_hash = hash(target_coords.tobytes())

    def _broadcast_sources(self, sources: dict | None) -> dict:
        """Broadcast source arrays from rank 0 to all MPI ranks."""
        keys = self.comm.bcast(
            list(sources.keys()) if self._is_root else None, root=0
        )
        return {
            k: self.comm.bcast(sources[k] if self._is_root else None, root=0)
            for k in keys
        }

    # IDW interpolation

    def _interpolate_idw(
        self,
        source_xyz: np.ndarray,
        target_coords: np.ndarray,
        *source_properties: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Inverse-distance weighted interpolation on the unit sphere.

        Normalises source and target to the unit sphere, builds a KDTree,
        and computes IDW-weighted values for each property array.

        Args:
            source_xyz: (N, 3) source coordinates (gtrack metres or mesh units).
            target_coords: (M, 3) target mesh coordinates.
            *source_properties: (N,) arrays to interpolate.

        Returns:
            (interpolated, too_far) where *interpolated* is a list of (M,)
            arrays and *too_far* is a boolean mask of shape (M,).
        """
        epsilon = 1e-10

        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], epsilon)

        r_target = np.linalg.norm(target_coords, axis=1)
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], epsilon)

        tree = cKDTree(unit_source)
        k = min(self.config.k_neighbors, len(source_xyz))
        dists, idx = tree.query(unit_target, k=k)

        if k == 1:
            results = [prop[idx].copy() for prop in source_properties]
            too_far = dists > self.config.distance_threshold
        else:
            exact_match = dists[:, 0] < epsilon
            too_far = dists[:, 0] > self.config.distance_threshold

            safe_dists = np.maximum(dists, epsilon)
            weights = 1.0 / safe_dists
            weights /= weights.sum(axis=1, keepdims=True)

            results = []
            for prop in source_properties:
                interpolated = np.sum(weights * prop[idx], axis=1)
                interpolated[exact_match] = prop[idx[exact_match, 0]]
                results.append(interpolated)

        return results, too_far

    # Default indicator computation (tanh)

    def _compute_indicator(
        self,
        sources: dict[str, np.ndarray],
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Compute tanh-based indicator from source thickness data.

        Override in subclasses for alternative computations (e.g. geotherm).
        """
        source_xyz = sources["xyz"]
        source_thickness = sources["thickness"]

        if source_xyz is None or len(source_xyz) == 0:
            return self._empty_indicator(len(target_coords))

        r_target = np.linalg.norm(target_coords, axis=1)
        (thickness_km,), too_far = self._interpolate_idw(
            source_xyz, target_coords, source_thickness
        )
        return self._apply_indicator(r_target, thickness_km, too_far)

    # Data loading (shared by LithosphereConnector & PolygonConnector)

    def _load_data(self, data) -> "PointCloud":
        """Load thickness data, filter to polygon region, assign plate IDs.

        Accepts a PointCloud, file path, (latlon, values) tuple, or scalar.
        Requires self._PointCloud, self._polygon_filter, and self._rotator
        to be set by the subclass before calling.

        Args:
            data: Thickness data in any supported format.

        Returns:
            PointCloud filtered to region with plate IDs assigned.
        """
        PointCloud = self._PointCloud

        if hasattr(data, "xyz") and hasattr(data, "properties"):
            cloud = data
        elif isinstance(data, (str, Path)):
            cloud = self._load_from_hdf5(data)
        elif isinstance(data, tuple) and len(data) == 2:
            latlon, values = data
            cloud = PointCloud.from_latlon(np.asarray(latlon))
            cloud.add_property(self.config.property_name, np.asarray(values))
        elif isinstance(data, (int, float)):
            from gtrack.mesh import create_sphere_mesh_latlon
            lats, lons = create_sphere_mesh_latlon(self.config.n_points)
            latlon = np.column_stack([lats, lons])
            cloud = PointCloud.from_latlon(latlon)
            cloud.add_property(
                self.config.property_name, np.full(len(latlon), float(data))
            )
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Expected PointCloud, file path, (latlon, values) tuple, or scalar."
            )

        n_before = cloud.n_points
        log(f"{type(self).__name__}: Filtering {n_before} points to region...",
            level=DEBUG)
        cloud = self._polygon_filter.filter_inside(cloud, at_age=0.0)
        # Guard against zero-point input; the f-string is evaluated eagerly
        # so it would raise ZeroDivisionError even if the log level is off.
        if n_before > 0:
            log(f"{type(self).__name__}: After filtering: {cloud.n_points} points "
                f"({100 * cloud.n_points / n_before:.1f}% retained)", level=DEBUG)
        else:
            log(f"{type(self).__name__}: Input had 0 points, nothing to filter.",
                level=DEBUG)

        cloud = self._rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=True
        )
        log(f"{type(self).__name__}: After plate ID assignment: "
            f"{cloud.n_points} points.", level=DEBUG)

        return cloud

    def _load_from_hdf5(self, filepath) -> "PointCloud":
        """Load lat/lon/values from an HDF5 or NetCDF file.

        Expected datasets: 'lon', 'lat', and either config.property_name or 'z'.
        """
        PointCloud = self._PointCloud

        log(f"{type(self).__name__}: Loading data from {filepath}.", level=DEBUG)

        with h5py.File(filepath, "r") as f:
            lon = f["lon"][:]
            lat = f["lat"][:]

            if self.config.property_name in f:
                values = f[self.config.property_name][:]
            elif "z" in f:
                values = f["z"][:]
            else:
                raise KeyError(
                    f"File must contain '{self.config.property_name}' or 'z'. "
                    f"Available datasets: {list(f.keys())}"
                )

        lon = np.where(lon > 180, lon - 360, lon)

        if lon.ndim == 1 and lat.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            values = values.ravel()
        else:
            latlon = np.column_stack([lat.ravel(), lon.ravel()])
            values = values.ravel()

        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property(self.config.property_name, values)

        log(f"{type(self).__name__}: Loaded {cloud.n_points} points from file.",
            level=DEBUG)

        return cloud

    # Time conversion delegates

    def ndtime2age(self, ndtime: float) -> float:
        """Convert non-dimensional time to geological age (Ma)."""
        return self.gplates_connector.ndtime2age(ndtime)

    def age2ndtime(self, age: float) -> float:
        """Convert geological age (Ma) to non-dimensional time."""
        return self.gplates_connector.age2ndtime(age)

    @property
    def delta_t(self) -> float:
        """Time window below which cached results are reused."""
        return self.gplates_connector.delta_t
