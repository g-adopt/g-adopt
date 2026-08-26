"""Connect a source to an output strategy on a target mesh.

A connector is the object a model actually holds. It pairs one Source with one
OutputStrategy, checks at construction time that the pairing is coherent, and
turns a non-dimensional model time into a field on the target nodes.

There are two caches here, at different levels, and they are easy to confuse.
This module owns the result cache: the finished field, keyed on the age and on
the identity of the target coordinate array, which is what makes repeated calls
inside one timestep cheap. The geometry cache lives on the Source, because its
whole point is that several connectors sharing that source reuse one
``cKDTree`` build; see ``gadopt.gplates.sources``.

Every cache decision has to be unanimous across the MPI ranks, since a miss
leads into the collective ``Source.prepare``. The connector therefore reduces
its local hit across the communicator before acting on it: if any rank must
recompute, all of them do.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
from mpi4py import MPI

from ..utility import log, DEBUG
from .interpolation import InterpolationConfig, SphericalKNNInterpolator
from .outputs import MeshConfig, OutputStrategy
from .sources import Source

__all__ = ["ScalarFieldConnector"]


# ---------------------------------------------------------------------------
# ScalarFieldConnector
# ---------------------------------------------------------------------------

class ScalarFieldConnector:
    """Evaluate source channels as one time-dependent target field.

    The output's ``requires`` is checked against the source's ``provides`` here,
    so a mismatched pair fails when the connector is built rather than with a
    ``KeyError`` on the first timestep.

    Each evaluation drops a point cloud and a set of interpolation arrays that
    are large and largely cyclic, and on a long reconstruction the automatic
    collector lets them accumulate far enough to matter; collecting on a fixed
    count keeps the high water mark down at a cost that is negligible next to
    the kNN query.

    Args:
        source: Source providing coordinates and channels.
        output: Output strategy turning interpolated channels into the field.
        mesh: Mesh geometry. Defaults to ``MeshConfig()``.
        interpolation: Interpolation settings. Defaults to
            ``InterpolationConfig()``. Equal settings share cached geometry.
        gc_collect_frequency: Run ``gc.collect()`` after this many evaluations.
            None disables collection at the connector level.

    Raises:
        ValueError: If the output requires channels the source does not
            provide, or the collection frequency is less than one.
    """

    def __init__(
        self,
        source: Source,
        output: OutputStrategy,
        *,
        mesh: MeshConfig | None = None,
        interpolation: InterpolationConfig | None = None,
        gc_collect_frequency: int | None = 10,
    ):
        if not output.requires <= source.provides:
            missing = output.requires - source.provides
            raise ValueError(
                f"{type(output).__name__} requires {sorted(missing)} which "
                f"{type(source).__name__} does not provide "
                f"(provides={sorted(source.provides)})."
            )
        if gc_collect_frequency is not None and gc_collect_frequency < 1:
            raise ValueError(
                f"gc_collect_frequency must be >= 1 or None, "
                f"got {gc_collect_frequency}"
            )

        self.source = source
        self.output = output
        self.mesh = mesh or MeshConfig()
        self.interpolation = interpolation or InterpolationConfig()
        # Equal interpolation configurations share cached geometry.
        self._interpolator = SphericalKNNInterpolator(self.interpolation)
        self.gc_collect_frequency = gc_collect_frequency

        # The result cache uses age and target-array identity. A weak reference
        # avoids retaining an array after its caller releases it.
        self.reconstruction_age: float | None = None
        self._cached_result: np.ndarray | None = None
        self._cached_coords_ref: weakref.ref | None = None
        self._gc_call_counter = 0

    # Time delegates (most callers reach through the connector)

    def ndtime2age(self, ndtime: float) -> float:
        return self.source.ndtime2age(ndtime)

    def age2ndtime(self, age: float) -> float:
        return self.source.age2ndtime(age)

    @property
    def delta_t(self) -> float:
        return self.source.delta_t

    @property
    def comm(self) -> MPI.Comm:
        return self.source.comm

    # Main entry point
    def get_indicator(
        self,
        target_coords: np.ndarray,
        ndtime: float,
    ) -> np.ndarray:
        """Evaluate the scalar field at the target nodes for one model time.

        Args:
            target_coords: Target node coordinates, shape ``(n_target, 3)``.
            ndtime: Non-dimensional model time.

        Returns:
            The scalar field, one value per target node. A cache hit returns
            the cached array itself, so treat the result as read-only.

        Raises:
            ValueError: If the age is outside the range the source accepts.
        """
        age = self.source.ndtime2age(ndtime)
        self.source.validate_age(age)

        use_cache = self._check_cache(age, target_coords)
        # All ranks must take the same branch. A mixed cache result can deadlock
        # when only the ranks with a cache miss enter ``source.prepare``.
        use_cache = self.comm.allreduce(use_cache, op=MPI.MIN)
        if use_cache:
            return self._cached_result

        sources_dict = self.source.prepare(age)
        result = self._compute(sources_dict, target_coords)
        self._update_cache(age, target_coords, result)

        self._gc_call_counter += 1
        if (self.gc_collect_frequency is not None
                and self._gc_call_counter % self.gc_collect_frequency == 0):
            gc.collect()

        return result

    # Cache
    def _check_cache(self, age: float, target_coords: np.ndarray) -> bool:
        if self.reconstruction_age is None:
            return False
        if abs(age - self.reconstruction_age) >= self.delta_t:
            return False
        # Treat an incomplete cache as a miss.
        if self._cached_result is None or self._cached_coords_ref is None:
            return False
        # A released coordinate array gives a dead weak reference and a miss.
        if self._cached_coords_ref() is not target_coords:
            return False
        log(f"{type(self).__name__}: age {age:.2f} Ma unchanged "
            f"(within delta_t={self.delta_t}); reusing cached result.",
            level=DEBUG)
        return True

    def _update_cache(
        self, age: float, target_coords: np.ndarray, result: np.ndarray
    ) -> None:
        self.reconstruction_age = age
        self._cached_result = result
        self._cached_coords_ref = weakref.ref(target_coords)

    def _construct_cache_key(self, target_coords: np.ndarray):
        return (hash(target_coords.tobytes()), self.interpolation)

    # Computation
    def _compute(
        self,
        sources_dict: dict[str, np.ndarray],
        target_coords: np.ndarray,
    ) -> np.ndarray:
        source_xyz = sources_dict["xyz"]
        if source_xyz is None or len(source_xyz) == 0:
            return np.zeros(len(target_coords))

        r_target = np.linalg.norm(target_coords, axis=1)

        # Geometry depends on target coordinates and interpolation settings, but
        # not on the channel values that the geometry later gathers.
        key = self._construct_cache_key(target_coords)
        geometry = self.source.get_or_build_geometry(
            key, lambda: self._interpolator.geometry(source_xyz, target_coords)
        )

        channel_keys = sorted(self.output.requires)
        interpolated = {
            k: SphericalKNNInterpolator.gather(geometry, sources_dict[k])
            for k in channel_keys
        }
        return self.output.compute(
            interpolated, r_target, geometry["outside_source_range"], self.mesh
        )
