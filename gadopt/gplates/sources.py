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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector


# Minimum points to make any sphere-coverage source meaningful. Anything
# below this and the kNN interpolator gives nonsense.
SOURCE_MIN_POINTS = 100


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
