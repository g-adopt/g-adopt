"""Connect a source to an output strategy on a target mesh.

The source caches one point cloud for each geological age.
The connector caches one result for each age and target coordinate array.
All MPI ranks agree on a connector cache hit before collective source work.
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

# Re-export `InterpolationConfig` to preserve its existing import path.
__all__ = ["ScalarFieldConnector", "InterpolationConfig"]


# ---------------------------------------------------------------------------
# ScalarFieldConnector
# ---------------------------------------------------------------------------

class ScalarFieldConnector:
    """Evaluate source channels as one time-dependent target field.

    The connector rejects an output that requires unavailable source channels.
    Shared sources prepare one point cloud for each geological age.
    `gc_collect_frequency` runs `gc.collect()` after that number of evaluations.
    A value of `None` disables connector-level collection.
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
        """Evaluate the scalar field at ``target_coords`` for time ``ndtime``."""
        age = self.source.ndtime2age(ndtime)
        self.source.validate_age(age)

        use_cache = self._check_cache(age, target_coords)
        # All ranks must take the same branch. A mixed cache result can deadlock
        # when only the ranks with a cache miss enter `source.prepare`.
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
        bundle = self.source.get_or_build_geometry(
            key, lambda: self._interpolator.geometry(source_xyz, target_coords)
        )

        channel_keys = sorted(self.output.requires)
        interpolated = {
            k: SphericalKNNInterpolator.gather(bundle, sources_dict[k])
            for k in channel_keys
        }
        return self.output.compute(
            interpolated, r_target, bundle["outside_source_range"], self.mesh
        )
