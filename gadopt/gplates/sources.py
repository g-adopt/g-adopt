"""Data sources for plate-reconstruction indicator/geotherm fields.

A Source owns the stateful gtrack machinery (SeafloorAgeTracker, PointRotator,
PolygonFilter) and exposes a single collective method ``prepare(age)`` returning
a dict of numpy arrays. Two consumers (e.g. an indicator and a geotherm) can
share one Source instance; the per-age cache guarantees the underlying tracker
advances at most once per geological age, regardless of how many connectors
hold a reference.

The Source/Output split is the central design point: Sources answer "where do
source points live and what properties do they carry at age X?", Outputs answer
"given interpolated arrays at target points, what scalar field do we want?".
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
from mpi4py import MPI

from gtrack import PointCloud, PointRotator, PolygonFilter, SeafloorAgeTracker
from gtrack.config import TracerConfig
from gtrack.mesh import create_sphere_mesh_latlon

from ..utility import log, DEBUG, INFO

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector, PlateModelFiles


# Minimum points to make any sphere-coverage source meaningful. Anything
# below this and the kNN interpolator gives nonsense.
SOURCE_MIN_POINTS = 100


# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------

@dataclass
class LithosphereSourceConfig:
    """Knobs for LithosphereSource — ocean tracker, checkpointing, data loading.

    Mesh geometry (r_outer, depth_scale) lives on MeshConfig.
    Interpolation (k_neighbors, kernel, ...) lives on InterpolationConfig.
    Output knobs (transition_width, kappa, ...) live on the output strategy.
    """

    time_step: float = 1.0
    n_points: int = 10000
    reinit_interval_myr: float = 50.0
    property_name: str = "thickness"
    gtrack_config: dict | None = None
    checkpoint_interval_myr: float | None = None
    checkpoint_dir: str | None = None

    def __post_init__(self):
        if self.n_points < SOURCE_MIN_POINTS:
            raise ValueError(
                f"n_points must be at least {SOURCE_MIN_POINTS}, got {self.n_points}"
            )
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.reinit_interval_myr <= 0:
            raise ValueError(
                f"reinit_interval_myr must be positive, "
                f"got {self.reinit_interval_myr}"
            )
        if (self.checkpoint_interval_myr is not None
                and self.checkpoint_interval_myr <= 0):
            raise ValueError(
                f"checkpoint_interval_myr must be positive or None, "
                f"got {self.checkpoint_interval_myr}"
            )


# ---------------------------------------------------------------------------
# Data-loading helpers (module-level — both Sources need them)
# ---------------------------------------------------------------------------

def _load_from_hdf5(filepath, property_name: str) -> PointCloud:
    """Read lat/lon/values from an HDF5 (or NetCDF) file into a PointCloud."""
    with h5py.File(filepath, "r") as f:
        lon = f["lon"][:]
        lat = f["lat"][:]
        if property_name in f:
            values = f[property_name][:]
        elif "z" in f:
            values = f["z"][:]
        else:
            raise KeyError(
                f"File must contain '{property_name}' or 'z'. "
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
    cloud.add_property(property_name, values)
    return cloud


def _build_cloud(data, property_name: str, n_points_fallback: int) -> PointCloud:
    """Dispatch the input ``data`` to the right PointCloud constructor.

    Accepts: an existing PointCloud, an HDF5 path, a (latlon, values) tuple,
    or a scalar (broadcast onto a Fibonacci sphere mesh).
    """
    if hasattr(data, "xyz") and hasattr(data, "properties"):
        return data
    if isinstance(data, (str, Path)):
        return _load_from_hdf5(data, property_name)
    if isinstance(data, tuple) and len(data) == 2:
        latlon, values = data
        cloud = PointCloud.from_latlon(np.asarray(latlon))
        cloud.add_property(property_name, np.asarray(values))
        return cloud
    if isinstance(data, (int, float)):
        lats, lons = create_sphere_mesh_latlon(n_points_fallback)
        latlon = np.column_stack([lats, lons])
        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property(property_name, np.full(len(latlon), float(data)))
        return cloud
    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        "Expected PointCloud, file path, (latlon, values) tuple, or scalar."
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_RE = re.compile(r"^ocean_checkpoint_(\d+)Ma\.npz$")


def _find_best_checkpoint(checkpoint_dir: str | None, target_age: float):
    """Return the checkpoint path whose age is the smallest value >= target_age.

    Smallest-age-not-younger-than-target minimises forward stepping. Returns
    None if the directory is missing or contains no matching files.
    """
    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        return None
    best_path = None
    best_age = None
    for fname in os.listdir(checkpoint_dir):
        m = _CHECKPOINT_RE.match(fname)
        if m is None:
            continue
        ckpt_age = int(m.group(1))
        if ckpt_age < target_age:
            continue
        if best_age is None or ckpt_age < best_age:
            best_age = ckpt_age
            best_path = os.path.join(checkpoint_dir, fname)
    return best_path


# ---------------------------------------------------------------------------
# Source ABC
# ---------------------------------------------------------------------------

class Source(ABC):
    """Time-dependent source of (xyz, properties) on a sphere.

    Subclasses expose a ``provides`` set listing the property keys (excluding
    "xyz") that ``prepare(age)`` returns. ``provides`` is declared here as an
    abstract property, which a subclass may satisfy either with a plain class
    constant (the concrete sources do this, keeping class-level access like
    ``LithosphereSource.provides``) or with an instance property (handy for
    test doubles and dynamically-configured sources). The ABC implements a
    per-age cache so that consumers sharing this Source can call ``prepare``
    in any order and the underlying gtrack state advances at most once per
    age.

    ``prepare`` is collective across ``self.comm``. Subclasses implement
    ``_compute_sources(age)``, which runs on rank 0 only; the ABC handles
    broadcast and caching.
    """

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """The set of property keys (excluding 'xyz') that prepare(age) returns."""
        ...

    gplates_connector: "pyGplatesConnector"
    comm: MPI.Comm
    _is_root: bool

    # Per-age cache (populated by prepare)
    _cached_age: float | None = None
    _cached_dict: dict[str, np.ndarray] | None = None

    # kNN interpolation-geometry cache. Keyed on
    # (target_coords content hash, id(interpolation_config)); the value is the
    # geometry bundle produced by the connector for that (source cloud, mesh,
    # cfg) triple. Siblings sharing this source at the same age reuse one
    # cKDTree build + query. Lazily created (a class-level dict default would
    # be shared across instances). Cleared whenever the source dict advances
    # to a new age, so geometry never disagrees with _cached_dict on age.
    _interp_geometry_cache: dict | None = None

    # Lazy-load flag: gtrack/pyGplates handles and the present-day cloud are
    # built on the first prepare(), not in __init__, so a Source is cheap to
    # construct and mockable without touching reconstruction data.
    _loaded: bool = False

    @abstractmethod
    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        """Build the source-arrays dict on rank 0. Must include 'xyz' plus
        every key in ``self.provides``."""

    def _load(self) -> None:
        """Rank-0 I/O: build the gtrack machinery and the present-day cloud.

        Default is a no-op; subclasses that own gtrack handles override this.
        Runs at most once, on root, driven by ``_ensure_loaded``.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Trigger the one-off rank-0 load before the first compute.

        Called on every rank so the ``_loaded`` flag stays coherent, but the
        actual I/O in ``_load`` only runs on root.
        """
        if not self._loaded:
            if self._is_root:
                self._load()
            self._loaded = True

    def prepare(self, age: float) -> dict[str, np.ndarray]:
        """Return source arrays at ``age`` (collective across self.comm)."""
        # Cache hit: deterministic on age alone, so the decision is identical
        # on every rank — no allreduce needed.
        if (self._cached_age is not None
                and abs(self._cached_age - age) < self.delta_t):
            return self._cached_dict

        self._ensure_loaded()
        sources = self._compute_sources(age) if self._is_root else None
        sources = self.comm.bcast(sources, root=0)

        self._cached_age = age
        self._cached_dict = sources
        # The source cloud just changed, so any cached interpolation geometry
        # (cKDTree indices/weights over the previous cloud) is stale. Clear it
        # in lockstep with _cached_dict so the two caches never disagree on age.
        if self._interp_geometry_cache is not None:
            self._interp_geometry_cache.clear()
        return sources

    def get_or_build_geometry(self, key, build_fn):
        """Return the cached interpolation-geometry bundle for ``key``, building
        it via ``build_fn()`` on a miss.

        The cache is rank-local and keyed on
        ``(target_coords content hash, id(interpolation_config))``; the
        connector owns the geometry math (``build_fn``) while the source owns
        the cache so siblings sharing this source reuse one build. Cleared by
        ``prepare`` whenever the source cloud advances to a new age.
        """
        if self._interp_geometry_cache is None:
            self._interp_geometry_cache = {}
        bundle = self._interp_geometry_cache.get(key)
        if bundle is None:
            bundle = build_fn()
            self._interp_geometry_cache[key] = bundle
        return bundle

    # Time delegates

    def ndtime2age(self, ndtime: float) -> float:
        return self.gplates_connector.ndtime2age(ndtime)

    def age2ndtime(self, age: float) -> float:
        return self.gplates_connector.age2ndtime(age)

    @property
    def delta_t(self) -> float:
        return self.gplates_connector.delta_t

    @property
    def oldest_age(self) -> float:
        return self.gplates_connector.oldest_age

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` is outside the plate model's range.

        Subclasses with extra state (e.g. forward-only ocean trackers) may
        override to add further checks. Called by ScalarFieldConnector before
        prepare().
        """
        if age > self.oldest_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's "
                f"oldest age ({self.oldest_age:.2f} Ma)."
            )
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future). "
                f"Ages must be >= 0 (present day)."
            )


# ---------------------------------------------------------------------------
# LithosphereSource
# ---------------------------------------------------------------------------

class LithosphereSource(Source):
    """Combined oceanic + continental lithosphere source.

    Oceanic seeds come from a forward-stepped SeafloorAgeTracker; their
    thickness is derived from age via the user-supplied ``age_to_property``
    (typically a half-space cooling rule). Continental seeds are present-day
    thickness data back-rotated to the target age by a PointRotator.

    The two sets are concatenated and returned together. Each seed carries
    a thickness (km) and an age (Myr); the age key is needed by erf
    geotherms and ignored by the tanh indicator.

    The ocean tracker is the only stateful piece: it advances forward in
    geological time (decreasing age) and cannot rewind. Per-age caching
    on the base class guarantees one ``step_to`` per age regardless of how
    many connectors share this source.

    Forward-only contract: the first ``prepare`` across every connector that
    shares this source sets the oldest reachable age, and every subsequent
    request must be at that age or younger. When several connectors share one
    source and update at different cadences, whichever one fires first would
    otherwise silently fix the walk's ceiling. Declare ``walk_start_age`` to
    pin the oldest age you intend to request up front, so an out-of-range
    request fails immediately at wiring time rather than locking the walk at
    the first touch. Declaring ``walk_start_age`` does NOT let you revisit
    ages already stepped past — it only declares the oldest age you intend to
    request.
    """

    provides = frozenset({"xyz", "thickness", "age"})

    def __init__(
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
        """Build a combined oceanic + continental lithosphere source.

        Args:
            walk_start_age: the oldest age you will ever request; requests
                older than this fail immediately rather than silently locking
                the forward-only walk at the first touch. Does not enable
                revisiting older ages once stepped past. Default None keeps the
                pre-declaration behaviour (the first prepare fixes the ceiling).
        """
        if plate_files.continental_polygons is None:
            raise ValueError(
                "plate_files.continental_polygons must be set. "
                "Pass continental_polygons to PlateModelFiles."
            )
        if plate_files.static_polygons is None:
            raise ValueError(
                "plate_files.static_polygons must be set. "
                "Pass static_polygons to PlateModelFiles."
            )

        self.gplates_connector = gplates_connector
        self.age_to_property = age_to_property
        self.config = config or LithosphereSourceConfig()
        self._default_continental_age_myr = default_continental_age_myr
        self.comm = comm
        self._is_root = (comm.rank == 0)

        if walk_start_age is not None and not (
            0 <= walk_start_age <= self.oldest_age
        ):
            raise ValueError(
                f"walk_start_age ({walk_start_age}) must be between 0 and the "
                f"plate model's oldest age ({self.oldest_age:.2f} Ma)."
            )
        self._walk_start_age = walk_start_age

        # Stashed for the lazy _load; not touched between here and prepare.
        self._continental_data = continental_data
        self._plate_files = plate_files

        self._initialized = False
        self._last_reinit_age: float | None = None
        self._last_checkpoint_age: float | None = None
        self._checkpoint_dir = (
            self.config.checkpoint_dir or "./gtrack_checkpoints"
            if self.config.checkpoint_interval_myr is not None
            else None
        )

        # gtrack handles + present-day cloud are built lazily in _load (rank 0
        # only). On non-root these stay None for the source's lifetime and are
        # never read — all gtrack work happens on root.
        self._ocean_tracker = None
        self._rotator = None
        self._continental_filter = None
        self._continental_present = None

    # Public properties (some tests / consumers want them)

    @property
    def is_root(self) -> bool:
        return self._is_root

    # Loading

    def _load(self) -> None:
        """Build the ocean tracker, rotators and present-day continental
        cloud. Runs once on rank 0, driven by Source._ensure_loaded."""
        gplates_connector = self.gplates_connector
        plate_files = self._plate_files
        tracer_kwargs = {
            "time_step": self.config.time_step,
            "default_mesh_points": self.config.n_points,
        }
        if self.config.gtrack_config is not None:
            tracer_kwargs.update(self.config.gtrack_config)
        tracer_config = TracerConfig(**tracer_kwargs)

        self._ocean_tracker = SeafloorAgeTracker(
            rotation_files=gplates_connector.rotation_filenames,
            topology_files=gplates_connector.topology_filenames,
            continental_polygons=plate_files.continental_polygons,
            config=tracer_config,
        )
        self._rotator = PointRotator(
            rotation_files=gplates_connector.rotation_filenames,
            static_polygons=plate_files.static_polygons,
        )
        self._continental_filter = PolygonFilter(
            polygon_files=plate_files.continental_polygons,
            rotation_files=gplates_connector.rotation_filenames,
        )
        self._continental_present = self._load_continental(self._continental_data)

    def _load_continental(self, data) -> PointCloud:
        cloud = _build_cloud(data, self.config.property_name, self.config.n_points)
        n_before = cloud.n_points
        cloud = self._continental_filter.filter_inside(cloud, at_age=0.0)
        if n_before > 0:
            log(f"LithosphereSource: continental data filtered "
                f"{n_before} -> {cloud.n_points} points "
                f"({100 * cloud.n_points / n_before:.1f}% retained).", level=DEBUG)
        cloud = self._rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=True
        )
        return cloud

    # Age validation extension

    def validate_age(self, age: float) -> None:
        """Validate a requested age against the forward-only contract.

        Two independent checks run after the base range / non-negative check
        (which fires first). An age must pass both:

        (a) Promise check: if ``walk_start_age`` was declared, the age must be
            no older than it. This fires even before the first ``prepare``, so
            an out-of-range request is caught at wiring time.
        (b) Physical-floor check: once the tracker has actually stepped to some
            age, an older age is unreachable (the walk is forward-only and
            cannot rewind), regardless of any declaration.

        Declaring ``walk_start_age`` does NOT let you revisit ages already
        stepped past — it only declares the oldest age you intend to request.
        """
        super().validate_age(age)
        # (a) Promise check — fires even before the first prepare.
        if self._walk_start_age is not None and age > self._walk_start_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the declared "
                f"walk_start_age ({self._walk_start_age:.2f} Ma). Raise "
                f"walk_start_age to the oldest age you will request, or request a "
                f"younger age. The ocean tracker evolves forward only (decreasing age)."
            )
        # (b) Physical-floor check: once we've stepped to some age, we cannot
        # go back to an older age. Check on all ranks (not just root) using
        # the per-age cache, which is coherent.
        if self._initialized and self._cached_age is not None:
            if age > self._cached_age:
                raise ValueError(
                    f"Requested age {age:.2f} Ma is older than the last "
                    f"computed age ({self._cached_age:.2f} Ma). The ocean "
                    f"tracker can only evolve forward (decreasing age)."
                )

    # Computation (rank-0 only)

    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        ocean_cloud = self._step_ocean_to(age)
        ocean_ages = ocean_cloud.get_property("age")
        ocean_cloud.add_property(
            self.config.property_name, self.age_to_property(ocean_ages)
        )

        cont_cloud = self._rotator.rotate(
            self._continental_present, from_age=0.0, to_age=age
        )
        cont_cloud.add_property(
            "age", np.full(cont_cloud.n_points, self._default_continental_age_myr)
        )

        combined = PointCloud.concatenate([ocean_cloud, cont_cloud], warn=False)
        # PointCloud.xyz / get_property return gtrack's internal arrays by
        # reference (the accessors do not copy). The two-cloud concatenate
        # vstacks into fresh arrays today, but we copy here regardless: the
        # returned dict becomes the source's per-age _cached_dict, which is
        # shared across sibling connectors and must survive the next step_to /
        # rotate. The copy is a deliberate ownership boundary — it is cheap
        # (~n_points elements) and does not rely on concatenate's freshness,
        # which is an uncontracted gtrack implementation detail.
        return {
            "xyz": combined.xyz.copy(),
            "thickness": combined.get_property(self.config.property_name).copy(),
            "age": combined.get_property("age").copy(),
        }

    def _initialize_walk(self, age: float) -> None:
        """First-call walk init: load the nearest checkpoint if one exists, else
        initialize the tracker at the model's oldest age. Idempotent via the
        _initialized latch. Distinct from Source._load (which builds the gtrack
        objects); this initialises the forward age-walk state."""
        if self._initialized:
            return
        loaded = False
        best = _find_best_checkpoint(self._checkpoint_dir, age)
        if best is not None:
            try:
                self._ocean_tracker.load_checkpoint(best)
                loaded_age = self._ocean_tracker.current_age
                log(f"LithosphereSource: loaded ocean checkpoint at "
                    f"{loaded_age} Ma from {best}.", level=DEBUG)
                self._last_reinit_age = loaded_age
                self._last_checkpoint_age = loaded_age
                loaded = True
            except Exception as exc:
                log(f"LithosphereSource: failed to load checkpoint "
                    f"{best}: {exc}. Falling back to full init.", level=INFO)
        if not loaded:
            # pyGplates uses integer ages (Ma).
            starting_age = int(self.gplates_connector.oldest_age)
            log(f"LithosphereSource: initialising ocean tracker at "
                f"{starting_age} Ma.", level=DEBUG)
            self._ocean_tracker.initialize(starting_age=starting_age)
            self._last_reinit_age = starting_age
        self._initialized = True

    def _step_ocean_to(self, age: float) -> PointCloud:
        """Initialise (from checkpoint if available) and step the tracker to ``age``."""
        self._initialize_walk(age)

        if self._last_reinit_age is not None:
            if abs(self._last_reinit_age - age) >= self.config.reinit_interval_myr:
                log(f"LithosphereSource: reinitialising ocean tracker at "
                    f"{age:.2f} Ma.", level=DEBUG)
                self._ocean_tracker.reinitialize(n_points=self.config.n_points)
                self._last_reinit_age = age

        cloud = self._ocean_tracker.step_to(int(round(age)))
        self._save_checkpoint_if_due(age)
        return cloud

    def _save_checkpoint_if_due(self, age: float) -> None:
        if self._checkpoint_dir is None:
            return
        interval = self.config.checkpoint_interval_myr
        if (self._last_checkpoint_age is not None
                and abs(self._last_checkpoint_age - age) < interval):
            return
        rounded_age = int(round(age))
        filepath = os.path.join(
            self._checkpoint_dir, f"ocean_checkpoint_{rounded_age}Ma.npz"
        )
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            self._ocean_tracker.save_checkpoint(filepath)
            self._last_checkpoint_age = rounded_age
            log(f"LithosphereSource: saved ocean checkpoint at "
                f"{rounded_age} Ma -> {filepath}.", level=DEBUG)
        except Exception as exc:
            log(f"LithosphereSource: failed to save checkpoint at "
                f"{rounded_age} Ma: {exc}", level=INFO)
