"""Data sources for plate-reconstruction indicator/geotherm fields.

A Source owns nothing about plate reconstruction itself. It wraps a gtrack
``AgeCloudSource`` *producer* — the object that answers "what labelled cloud
exists at age X" — and adapts it to gadopt's collective, per-age-cached
``prepare(age)`` interface. Two consumers (e.g. an indicator and a geotherm)
can share one Source instance; the per-age cache guarantees the producer is
advanced at most once per geological age, regardless of how many connectors
hold a reference.

The Source/Output split is the central design point on the gadopt side:
Sources answer "where do source points live and what properties do they carry
at age X?" — by delegating to a gtrack producer — while Outputs answer "given
interpolated arrays at target points, what scalar field do we want?". The seam
is deliberately narrow: gadopt imports one gtrack protocol and one gtrack
concrete type, and knows the walk contract only through the single boolean
``producer.monotonic_backward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gtrack import PointCloud
from gtrack.age_sources import AgeCloudSource

if TYPE_CHECKING:
    from .gplates import pyGplatesConnector


# What a gtrack producer accepts as its thickness / continental data. Kept as a
# convenience alias for callers annotating the data they hand to a producer;
# gtrack's ``PointCloud.from_data`` accepts exactly this union.
CloudDataType = (
    PointCloud | tuple[npt.ArrayLike, npt.ArrayLike] | str | Path | int | float
)


# ---------------------------------------------------------------------------
# Source ABC
# ---------------------------------------------------------------------------

class Source(ABC):
    """Time-dependent source of (xyz, properties) on a sphere.

    Subclasses expose a ``provides`` set listing the property keys —
    EXCLUDING "xyz" — that ``prepare(age)`` returns. Coordinates are always
    present in the returned dict under ``"xyz"`` and are never listed in
    ``provides`` (F2); it names only the interpolable channels, so a consumer
    can iterate it to know which ``get_property`` calls will succeed. This
    matches gtrack's ``AgeCloudSource.provides``. ``provides`` is declared here
    as an abstract property, which a subclass may satisfy either with a read-only
    property (the concrete sources do this) or with a plain class constant
    (handy for test doubles and dynamically-configured sources, and keeps
    class-level access like ``SomeSource.provides`` working). The ABC
    implements a per-age cache so that consumers sharing this Source can call
    ``prepare`` in any order and the underlying gtrack state advances at most
    once per age.

    ``prepare`` is collective across ``self.comm``. Subclasses implement
    ``_compute_sources(age)``, which runs on rank 0 only; the ABC handles
    broadcast and caching.
    """

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """The property keys (EXCLUDING 'xyz') that prepare(age) returns.

        'xyz' is always in the returned dict but is never listed here (F2):
        ``provides`` names the interpolable channels only, matching gtrack's
        ``AgeCloudSource.provides``.
        """
        ...

    gplates_connector: "pyGplatesConnector"
    comm: MPI.Comm
    _is_root: bool

    # Per-age cache (populated by prepare)
    _cached_age: float | None = None
    _cached_dict: dict[str, np.ndarray] | None = None

    # kNN interpolation-geometry cache. Keyed on
    # (target_coords content hash, interpolation config by value); the value is the
    # geometry bundle produced by the connector for that (source cloud, mesh,
    # cfg) triple. Siblings sharing this source at the same age reuse one
    # cKDTree build + query. Lazily created (a class-level dict default would
    # be shared across instances). Cleared whenever the source dict advances
    # to a new age, so geometry never disagrees with _cached_dict on age.
    _interp_geometry_cache: dict | None = None

    # Lazy-load flag: gtrack/pyGplates handles are built on the first prepare(),
    # not in __init__, so a Source is cheap to construct and mockable without
    # touching reconstruction data.
    _loaded: bool = False

    @abstractmethod
    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        """Build the source-arrays dict on rank 0. Must include 'xyz' plus
        every key in ``self.provides`` (which does not itself list 'xyz')."""

    def _load(self) -> None:
        """Rank-0 I/O hook, run at most once on root by ``_ensure_loaded``.

        Default is a no-op. ``PointCloudSource`` needs nothing here — its gtrack
        producer builds its own machinery lazily on the first ``at_age`` — but
        the hook stays on the ABC for subclasses that own resources directly.
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
        ``(target_coords content hash, interpolation config by value)``; the
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

        Subclasses with extra state (e.g. forward-only producers) may override
        to add further checks. Called by ScalarFieldConnector before prepare(),
        on every rank, so any raise is coherent across the collective.
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
# PointCloudSource
# ---------------------------------------------------------------------------

class PointCloudSource(Source):
    """Adapt a gtrack ``AgeCloudSource`` producer to the gadopt Source interface.

    The producer answers "what labelled cloud exists at age X"; this class adds
    the collective, per-age-cached ``prepare(age)`` that gadopt's connectors
    drive. It holds no plate-reconstruction machinery of its own — the producer
    owns all of it and builds it lazily on its first ``at_age``, which this
    class only ever calls on rank 0.

    ``provides`` is delegated verbatim to the producer (it already excludes
    'xyz', which this class adds to the prepared dict). Age validation is
    F19-safe on every rank in two parts. The forward-only floor for a
    ``monotonic_backward`` producer is enforced directly, using the
    rank-coherent per-age cache — gadopt owns that state, the producer's floor
    reads rank-0-only ``_last_walked_age``. Every other limit the producer
    enforces (a declared ``walk_start_age``, its own ``oldest_age``) is
    evaluated by ``producer.validate_age`` on rank 0 and its VERDICT broadcast,
    so all ranks raise together: a producer that rejects an in-range age must
    not raise on rank 0 alone inside the collective in ``prepare``, which per
    F19 is a hang, not a traceback.

    Args:
        producer: any object satisfying gtrack's ``AgeCloudSource`` protocol
            (``provides``, ``monotonic_backward``, ``at_age``, ``validate_age``)
            — e.g. ``LithosphereCloudSource`` or ``PolygonIndicatorSource``.
        gplates_connector: the ``pyGplatesConnector`` providing the
            age<->non-dimensional-time mapping and the plate model's oldest age.
        comm: MPI communicator; the producer is driven on rank 0 and results
            are broadcast.
    """

    def __init__(
        self,
        producer: AgeCloudSource,
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        # Reject anything that is not a producer at construction, on every rank
        # (no I/O), so a wiring mistake fails identically everywhere rather than
        # deadlocking later inside a collective (F19). runtime_checkable checks
        # for the presence of the protocol members.
        if not isinstance(producer, AgeCloudSource):
            raise TypeError(
                "producer must satisfy the gtrack AgeCloudSource protocol "
                "(provides, monotonic_backward, at_age, validate_age); got "
                f"{type(producer).__name__}."
            )
        self.producer = producer
        self.gplates_connector = gplates_connector
        self.comm = comm
        self._is_root = (comm.rank == 0)

    @property
    def provides(self) -> frozenset[str]:
        return frozenset(self.producer.provides)

    def validate_age(self, age: float) -> None:
        # Base range / non-negative check first (coherent: oldest_age comes
        # from the connector, which every rank holds).
        super().validate_age(age)
        # Forward-only floor for a monotonic-backward producer, on every rank,
        # via the rank-coherent _cached_age (bcast in prepare). This one check
        # is done in gadopt because only gadopt holds rank-coherent walk state;
        # the producer's own floor reads rank-0-only walk state.
        if (self.producer.monotonic_backward
                and self._cached_age is not None
                and age > self._cached_age):
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the last computed "
                f"age ({self._cached_age:.2f} Ma). This producer is "
                f"monotonic-backward — it walks forward in geological time, "
                f"towards decreasing age — and cannot rewind."
            )
        # Everything else the producer would reject — a declared walk_start_age,
        # its own oldest_age (which may be tighter than the connector's), any
        # future limit — is evaluated by producer.validate_age. That runs on
        # rank 0 only, because it reads rank-0 walk state, so we make its OUTCOME
        # collective here: evaluate on root, broadcast the verdict, and raise on
        # every rank together. Without this, a producer that rejects an in-range
        # age would raise on rank 0 alone inside the bcast in prepare — an MPI
        # deadlock, not a traceback (F19). Broadcasting a plain message rather
        # than the exception keeps the reduction picklable and the raise
        # coherent regardless of the producer's exception type.
        message = None
        if self._is_root:
            try:
                self.producer.validate_age(age)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
        message = self.comm.bcast(message, root=0)
        if message is not None:
            raise ValueError(message)

    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        cloud = self.producer.at_age(age)
        # F23: PointCloud.xyz / get_property return gtrack's internal arrays by
        # reference. The returned dict becomes the shared per-age _cached_dict,
        # held across the next at_age, so these copies are load-bearing
        # ownership boundaries, not defensive duplication.
        return {
            "xyz": cloud.xyz.copy(),
            **{k: cloud.get_property(k).copy() for k in self.producer.provides},
        }
