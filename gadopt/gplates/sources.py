"""Adapt gtrack source points and channels for G-ADOPT connectors.

A Source owns the stateful gtrack machinery and answers one question: where do
the source points live at geological age X, and what properties do they carry
there? What to do with those arrays is the other half of the split, and belongs
to ``gadopt.gplates.outputs``.

The per-age cache is the reason this class exists rather than a plain function.
Several connectors -- an indicator and a geotherm, say -- commonly share one
Source, and the gtrack producer behind it is stateful and often walks forward
in geological time without being able to rewind. The cache guarantees the
producer advances at most once per age, whatever order the consumers ask in.
It also holds the interpolation geometry those consumers share, and drops that
geometry whenever the points move to a new age.

``prepare`` is collective across ``comm``. The producer runs on rank zero only
and the result is broadcast, which keeps one copy of the gtrack state in the
job and keeps every rank in step. Age validation follows the same rule for the
same reason: a rank-zero exception on its own leaves the other ranks blocked in
the next collective call, so failures are broadcast before they are raised.
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


# This alias matches the inputs that ``gtrack.PointCloud.from_data`` accepts.
CloudDataType = (
    PointCloud | tuple[npt.ArrayLike, npt.ArrayLike] | str | Path | int | float
)


# ---------------------------------------------------------------------------
# Source ABC
# ---------------------------------------------------------------------------

class Source(ABC):
    """Provide source coordinates and channels for one geological age.

    A subclass declares its channels in ``provides`` and implements
    ``_compute_sources(age)``, which runs on rank zero only; this class handles
    the broadcast, the per-age cache, and the geometry cache. ``provides`` is
    an abstract property, which a subclass can satisfy either with a plain
    class constant, keeping class-level access, or with an instance property,
    which suits a test double or a source configured at run time.

    Attributes:
        provides: Channel names ``prepare`` returns, excluding ``xyz``.
        gplates_connector: Plate-model time mapping and maximum age.
        comm: MPI communicator the broadcast uses.
    """

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """Return the channel names, excluding ``xyz``."""
        ...

    gplates_connector: "pyGplatesConnector"
    comm: MPI.Comm
    _is_root: bool

    # Per-age cache, populated by ``prepare``.
    _cached_age: float | None = None
    _cached_dict: dict[str, np.ndarray] | None = None

    # The geometry cache is local to each rank and geological age.
    _interp_geometry_cache: dict | None = None

    # gtrack and pyGPlates load their resources on the first prepare call.
    _loaded: bool = False

    @abstractmethod
    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        """Return ``xyz`` and every channel in ``provides``, on rank zero.

        Args:
            age: Geological age, in millions of years before present.

        Returns:
            Arrays keyed by channel name, including ``xyz``.
        """

    def _load(self) -> None:
        """Load rank-zero resources once before the first calculation.

        Loading is deferred to the first ``prepare`` rather than done in
        ``__init__``, so that constructing a Source is cheap and needs no
        reconstruction data on disk.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Load resources on rank zero and update the flag on all ranks."""
        if not self._loaded:
            if self._is_root:
                self._load()
            self._loaded = True

    def prepare(self, age: float) -> dict[str, np.ndarray]:
        """Return the source arrays at one age, on every rank in ``comm``.

        Ages within ``delta_t`` of the cached one reuse the cache. That
        decision depends on age alone, so every rank takes the same branch and
        the collective broadcast below stays matched.

        Args:
            age: Geological age, in millions of years before present.

        Returns:
            Arrays keyed by channel name, including ``xyz``. The same object is
            returned to every consumer, so treat it as read-only.
        """
        # The cache decision depends only on age and is identical on all ranks.
        if (self._cached_age is not None
                and abs(self._cached_age - age) < self.delta_t):
            return self._cached_dict

        self._ensure_loaded()
        sources = self._compute_sources(age) if self._is_root else None
        sources = self.comm.bcast(sources, root=0)

        self._cached_age = age
        self._cached_dict = sources
        # New source points invalidate geometry that refers to the previous age.
        if self._interp_geometry_cache is not None:
            self._interp_geometry_cache.clear()
        return sources

    def get_or_build_geometry(self, key, build_fn):
        """Return the rank-local interpolation geometry for one key.

        Consumers sharing this source at the same age reuse a single
        ``cKDTree`` build and query through this cache. It is rank-local, since
        the target nodes differ from rank to rank, and ``prepare`` clears it
        when the age changes, so the geometry can never disagree with the
        points it was built from.

        Args:
            key: Cache key, identifying the target nodes and the interpolation
                settings.
            build_fn: Callable that builds the geometry when the key is absent.

        Returns:
            The geometry dict for that key. Treat it as read-only.
        """
        if self._interp_geometry_cache is None:
            self._interp_geometry_cache = {}
        geometry = self._interp_geometry_cache.get(key)
        if geometry is None:
            geometry = build_fn()
            self._interp_geometry_cache[key] = geometry
        return geometry

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
        """Check one age against the plate model range.

        The connector calls this on every rank before collective source
        preparation, so that a bad age fails everywhere at once. A subclass can
        add checks of its own for a stateful producer.

        Args:
            age: Geological age, in millions of years before present.

        Raises:
            ValueError: If the age is negative or older than the plate model.
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
    """Adapt a gtrack ``AgeCloudSource`` to the G-ADOPT source interface.

    This is the only place gtrack is spoken to. The producer runs on rank zero
    and every rank receives the same coordinates, channels, and validation
    errors. Arrays are copied out of the returned cloud, because a producer is
    free to reuse its internal storage between calls.

    A producer marked ``monotonic_backward`` walks forward in geological time,
    towards decreasing age, and cannot rewind; asking for an older age than the
    last one is rejected here, against the age G-ADOPT knows every rank
    received, rather than against the producer's rank-zero state.

    Args:
        producer: gtrack source satisfying the ``AgeCloudSource`` protocol.
        gplates_connector: Plate-model time mapping and maximum age.
        comm: MPI communicator for the source broadcast.

    Raises:
        TypeError: If the producer does not satisfy the protocol.
    """

    def __init__(
        self,
        producer: AgeCloudSource,
        gplates_connector: "pyGplatesConnector",
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        # Check the protocol on all ranks before collective work starts.
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
        # Every rank holds the same plate model range.
        super().validate_age(age)
        # G-ADOPT owns the last age that all ranks received. The gtrack producer
        # stores its corresponding state on rank zero only.
        if (self.producer.monotonic_backward
                and self._cached_age is not None
                and age > self._cached_age):
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the last computed "
                f"age ({self._cached_age:.2f} Ma). This producer is "
                f"monotonic-backward — it walks forward in geological time, "
                f"towards decreasing age — and cannot rewind."
            )
        # The producer validates its rank-zero state. Broadcast its error text so
        # that all ranks raise before the next collective operation. A rank-zero
        # exception alone can leave the other ranks blocked in ``prepare``.
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
        # Copy each array because the producer can reuse its internal storage.
        return {
            "xyz": cloud.xyz.copy(),
            **{k: cloud.get_property(k).copy() for k in self.producer.provides},
        }
