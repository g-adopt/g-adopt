"""Adapt gtrack source points and channels for G-ADOPT connectors.

A source caches one point cloud for each geological age.
Source preparation is collective across its MPI communicator.
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


# This alias matches the inputs that `gtrack.PointCloud.from_data` accepts.
CloudDataType = (
    PointCloud | tuple[npt.ArrayLike, npt.ArrayLike] | str | Path | int | float
)


# ---------------------------------------------------------------------------
# Source ABC
# ---------------------------------------------------------------------------

class Source(ABC):
    """Provide source coordinates and channels for one geological age.

    `provides` lists each channel that `prepare` returns.
    The list excludes the `xyz` coordinate array.
    `prepare` computes on rank zero and broadcasts across `comm`.
    """

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """Return the channel names, excluding `xyz`."""
        ...

    gplates_connector: "pyGplatesConnector"
    comm: MPI.Comm
    _is_root: bool

    # `prepare` populates this cache.
    _cached_age: float | None = None
    _cached_dict: dict[str, np.ndarray] | None = None

    # The geometry cache is local to each rank and geological age.
    _interp_geometry_cache: dict | None = None

    # gtrack and pyGPlates load their resources on the first prepare call.
    _loaded: bool = False

    @abstractmethod
    def _compute_sources(self, age: float) -> dict[str, np.ndarray]:
        """Return `xyz` and all channels in `provides` on rank zero."""

    def _load(self) -> None:
        """Load rank-zero resources once before the first calculation."""
        pass

    def _ensure_loaded(self) -> None:
        """Load resources on rank zero and update the flag on all ranks."""
        if not self._loaded:
            if self._is_root:
                self._load()
            self._loaded = True

    def prepare(self, age: float) -> dict[str, np.ndarray]:
        """Return source arrays at `age` on all ranks in `comm`."""
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
        """Return rank-local interpolation geometry for `key`.

        If the key is absent, `build_fn` creates the geometry.
        `prepare` clears this cache when the geological age changes.
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
        """Raise if `age` is outside the plate model range.

        Subclasses can add checks for stateful producers. The connector calls
        this method on all ranks before collective source preparation.
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
    """Adapt a gtrack `AgeCloudSource` to the G-ADOPT source interface.

    The producer runs on rank zero.
    All ranks receive the same point coordinates, channels, and age errors.
    A monotonic producer cannot return to an older requested age.

    Args:
        producer: A gtrack source that satisfies the `AgeCloudSource` protocol.
        gplates_connector: The plate-model time mapping and maximum age.
        comm: The MPI communicator for source broadcasts.
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
        # exception alone can leave the other ranks blocked in `prepare`.
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
