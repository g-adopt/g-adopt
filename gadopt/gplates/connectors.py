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

Source/Output pairing is validated at construction: ``output.requires`` must
be a subset of ``source.provides``. Mismatches (e.g. PolygonSource paired
with GeothermERFOutput, which needs ``"age"``) fail immediately with a clear
error rather than silently dropping the missing key.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from ..utility import log, DEBUG
from .outputs import MeshConfig, OutputStrategy
from .sources import Source


# ---------------------------------------------------------------------------
# Interpolation config
# ---------------------------------------------------------------------------

@dataclass
class InterpolationConfig:
    """Spherical kNN-interpolation knobs.

    The interpolation step normalises both source and target points to the
    unit sphere, builds a cKDTree over sources, queries ``k_neighbors``
    nearest neighbours per target node, and computes a weighted average
    using either inverse-distance (``"idw"``) or Gaussian (``"gaussian"``)
    weights.

    Target nodes whose nearest source seed is farther than
    ``distance_threshold`` (in radians on the unit sphere) are flagged as
    ``too_far`` and handled by the OutputStrategy (which decides whether
    to fill with a default thickness, switch to mantle temperature, etc).

    Defaults: 0.1 rad ≈ 640 km — generous; tighten to e.g. 0.02 rad
    (~127 km) for sharper boundaries. For polygon sources where the
    boundary is encoded by zero-thickness halo seeds, the threshold
    degenerates into a pathological-query guard and the actual roll-off
    length is controlled by ``gaussian_sigma`` instead.
    """

    kernel: str = "idw"
    k_neighbors: int = 50
    distance_threshold: float = 0.1
    gaussian_sigma: float = 0.04

    def __post_init__(self):
        valid_kernels = ("idw", "gaussian")
        if self.kernel not in valid_kernels:
            raise ValueError(
                f"kernel must be one of {valid_kernels}, got '{self.kernel}'"
            )
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {self.k_neighbors}")
        if self.distance_threshold <= 0:
            raise ValueError(
                f"distance_threshold must be positive, got {self.distance_threshold}"
            )
        if self.gaussian_sigma <= 0:
            raise ValueError(
                f"gaussian_sigma must be positive, got {self.gaussian_sigma}"
            )


# ---------------------------------------------------------------------------
# ScalarFieldConnector
# ---------------------------------------------------------------------------

class ScalarFieldConnector:
    """Composition of a Source and an OutputStrategy into a single time-
    varying scalar field on a mesh.

    Construction validates ``output.requires <= source.provides``; this
    catches the obvious mis-pairings (e.g. PolygonSource + GeothermERFOutput,
    which needs ``"age"``) at the point of wiring rather than at the first
    ``get_indicator`` call.

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
        # Validate the pairing of source and output
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
        key = (hash(target_coords.tobytes()), id(self.interpolation))
        bundle = self.source.get_or_build_geometry(
            key, lambda: self._interp_geometry(source_xyz, target_coords)
        )

        prop_keys = sorted(self.output.requires)
        interpolated = {
            k: self._interp_gather(bundle, sources_dict[k]) for k in prop_keys
        }
        return self.output.compute(
            interpolated, r_target, bundle["too_far"], self.mesh
        )

    def _interp_geometry(
        self,
        source_xyz: np.ndarray,
        target_coords: np.ndarray,
    ) -> dict:
        """Build the spherical kNN interpolation geometry over a source cloud.

        Normalises both clouds to the unit sphere, builds a cKDTree over the
        source points and queries the ``k`` nearest neighbours per target node,
        then precomputes the (normalised) blend weights. The returned bundle is
        the part of the interpolation that is independent of the gathered
        property, so it can be shared across every output sharing this source.

        The bundle must be treated read-only by ``_interp_gather`` — its arrays
        are shared across sibling outputs.
        """
        cfg = self.interpolation
        epsilon = 1e-10

        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], epsilon)

        r_target = np.linalg.norm(target_coords, axis=1)
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], epsilon)

        tree = cKDTree(unit_source)
        k = min(cfg.k_neighbors, len(source_xyz))
        dists, idx = tree.query(unit_target, k=k)

        if k == 1:
            too_far = dists > cfg.distance_threshold
            return {"k1": True, "idx": idx, "too_far": too_far}

        exact_match = dists[:, 0] < epsilon
        too_far = dists[:, 0] > cfg.distance_threshold

        if cfg.kernel == "gaussian":
            weights = np.exp(-dists**2 / (2 * cfg.gaussian_sigma**2))
        else:
            weights = 1.0 / np.maximum(dists, epsilon)

        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, epsilon)

        return {
            "k1": False,
            "idx": idx,
            "too_far": too_far,
            "exact_match": exact_match,
            "weights": weights,
        }

    @staticmethod
    def _interp_gather(bundle: dict, prop: np.ndarray) -> np.ndarray:
        """Gather one property through a prebuilt geometry bundle.

        Reads the bundle strictly read-only (its arrays are shared across
        sibling outputs); only the freshly-allocated result array is written.
        """
        idx = bundle["idx"]
        if bundle["k1"]:
            return prop[idx].copy()

        weights = bundle["weights"]
        exact_match = bundle["exact_match"]
        interpolated = np.sum(weights * prop[idx], axis=1)
        interpolated[exact_match] = prop[idx[exact_match, 0]]
        return interpolated
