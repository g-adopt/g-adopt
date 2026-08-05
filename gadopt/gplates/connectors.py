"""ScalarFieldConnector — composes a Source with an OutputStrategy.

The connector orchestrates one timestep:

  age = source.ndtime2age(ndtime)
  source.validate_age(age)
  if cached (age, identity of the target_coords buffer) match: return cached
  sources_dict = source.prepare(age)             # collective; per-age cached
  interp = self._interpolate(sources_dict, target_coords, output.requires)
  result = output.compute(interp, r_target, too_far, mesh)
  cache + return

The two cache layers (source.prepare's per-age cache and the connector's
(age, identity of the target_coords buffer) cache) are independent. The
source cache decision is deterministic on ``age`` alone, so it is identical
on every rank — no collective is needed there. The connector cache decision
differs per rank (the target buffer differs with partitioning) and is
allreduced before any collective work runs.

Source/Output pairing is validated at construction: ``output.requires`` must be
a subset of ``source.provides``; mismatches fail immediately with a clear error
rather than silently dropping the missing key. This also rules out the bad
polygon pairing structurally — a polygon source provides ``masked_thickness``
(depth times membership), so a `QuinticOutput` requiring plain ``thickness``
fails the subset check outright, as would any user output reading ``thickness``
as a depth off such a source.

The check lives here rather than on `ConnectorFactory` because the factory is
only one route to a connector; constructing a `ScalarFieldConnector` directly
(as demos and user scripts do) would bypass a check placed there.
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

# ``InterpolationConfig`` lives in ``interpolation.py`` alongside the
# interpolator that consumes it; re-exported here so existing imports of
# ``gadopt.gplates.connectors.InterpolationConfig`` keep resolving.
__all__ = ["ScalarFieldConnector", "InterpolationConfig"]


# ---------------------------------------------------------------------------
# ScalarFieldConnector
# ---------------------------------------------------------------------------

class ScalarFieldConnector:
    """Composition of a Source and an OutputStrategy into a single time-
    varying scalar field on a mesh.

    Construction validates ``output.requires <= source.provides``; this
    catches the obvious mis-pairings (e.g. a polygon source, which has no
    ``"age"`` channel, paired with a GeothermERFOutput that needs it) at the
    point of wiring rather than at the first ``get_indicator`` call.

    Two consumers (e.g. an indicator and a geotherm) that share the same
    Source instance see a single, coherent advance of the source's
    underlying state per geological age — the source's per-age cache
    enforces that, so the order of ``get_indicator`` calls between
    consumers is immaterial.

    ``gc_collect_frequency`` controls how often a full ``gc.collect()`` runs
    in the update loop (every Nth ``get_indicator``). The default of ``10``
    matches gtrack's own internal GC cadence and periodically breaks the
    pygplates C++ reference cycles without paying a collection on every call.
    Set it to ``None`` to disable the connector-level collect entirely (relying
    on gtrack's internal collect plus Python's automatic generational GC) when
    GC is documented as hot and confirmed memory stays bounded; set it to
    ``1`` for a lithosphere spin-up or very-long adjoint run where the
    connector is driven for thousands of ages and the tightest bound on C++
    cycle accumulation is wanted (the per-call cost is negligible there against
    the per-age ``step_to``).
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
        # Validate the source/output pairing. A polygon source provides
        # ``masked_thickness``, not ``thickness``, so a ``QuinticOutput`` (which
        # requires ``thickness``) is rejected here, structurally — see the
        # module docstring.
        if not output.requires <= source.provides:
            missing = output.requires - source.provides
            raise ValueError(
                f"{type(output).__name__} requires {sorted(missing)} which "
                f"{type(source).__name__} does not provide "
                f"(provides={sorted(source.provides)})."
            )
        # Validate the GC collect frequency
        if gc_collect_frequency is not None and gc_collect_frequency < 1:
            raise ValueError(
                f"gc_collect_frequency must be >= 1 or None, "
                f"got {gc_collect_frequency}"
            )

        self.source = source
        self.output = output
        self.mesh = mesh or MeshConfig()
        self.interpolation = interpolation or InterpolationConfig()
        # The interpolator owns the geometry math; the config is the by-value
        # half of the geometry cache key (see _compute).
        self._interpolator = SphericalKNNInterpolator(self.interpolation)
        self.gc_collect_frequency = gc_collect_frequency

        # Result cache: keyed on (age, identity of the target_coords buffer).
        # Distinct from the source's per-age cache, which only sees the age
        # axis. GplatesScalarFunction.mesh_coords is allocated once and held
        # for the SF lifetime, so a weakref to that buffer is a sound O(1)
        # key — no need to hash ~24 MB of coordinates on every call.
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
        # All ranks must agree, else a rank that misses will enter the
        # collective broadcast inside source.prepare while a rank that hits
        # returns early and hangs the collective.
        use_cache = self.comm.allreduce(use_cache, op=MPI.MIN)
        if use_cache:
            return self._cached_result

        # If the cache is not suitable, prepare the source and compute the result
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
        # Case when everything is fresh
        if self.reconstruction_age is None:
            return False
        # Case where we have gone over the delta_t
        if abs(age - self.reconstruction_age) >= self.delta_t:
            return False
        # Case where we do not even have a cached result (Not sure how this happens)
        # But just for safety!
        if self._cached_result is None or self._cached_coords_ref is None:
            return False
        # A dead referent dereferences to None and is never ``is`` the live
        # target buffer, so a freed coords array correctly misses.
        if self._cached_coords_ref() is not target_coords:
            return False
        log(f"{type(self).__name__}: age {age:.2f} Ma unchanged "
            f"(within delta_t={self.delta_t}); reusing cached result.",
            level=DEBUG)
        return True

    # Update the cache
    def _update_cache(
        self, age: float, target_coords: np.ndarray, result: np.ndarray
    ) -> None:
        # Here we are just weak referencing the target_coords array
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

        # Interpolation geometry (cKDTree indices + weights) depends only on
        # (source cloud, target coords, cfg) — not on the gathered property —
        # so it is identical across every output sharing this source at a given
        # age. Build it once and cache it on the source; siblings reuse it.
        # The key is (coords content hash, config by value); the config is
        # frozen and hashable, so two configs holding the same numbers share one
        # build. The coords hash is collidable in principle (2^-64), unlike the
        # by-value config half.
        key = self._construct_cache_key(target_coords)
        bundle = self.source.get_or_build_geometry(
            key, lambda: self._interpolator.geometry(source_xyz, target_coords)
        )

        prop_keys = sorted(self.output.requires)
        interpolated = {
            k: SphericalKNNInterpolator.gather(bundle, sources_dict[k])
            for k in prop_keys
        }
        return self.output.compute(
            interpolated, r_target, bundle["too_far"], self.mesh
        )
