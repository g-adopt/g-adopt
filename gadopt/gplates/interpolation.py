"""Spherical kNN interpolation from a source cloud onto mesh nodes.

`SphericalKNNInterpolator` is the geometry half of a `ScalarFieldConnector`:
given a source point cloud and a set of target coordinates it builds the
cKDTree, queries the nearest neighbours and precomputes the blend weights,
returning a *bundle* that is independent of any gathered property. The same
bundle then gathers each required channel through `gather`, so every output
sharing one source at one age reuses a single tree build and query.

The split matters for two reasons. The bundle is the expensive, property-
independent part, so caching it on the source (keyed by target coords and the
config by value) lets sibling outputs share it. And pulling the math out of the
connector gives it a name and a test surface of its own: the interpolation can
be exercised without constructing a connector, a source, or any reconstruction
machinery.

The bundle carries ``coverage`` alongside ``too_far``. ``coverage`` is the
fraction of a node's queried neighbours that fall within the distance
threshold, in [0, 1]; ``too_far`` is its zero case — a node with no neighbour
inside the threshold, i.e. the nearest source seed is farther than the
threshold. ``too_far`` is what the built-in outputs read; ``coverage`` is the
finer-grained companion (D3), added to the same read-only bundle for outputs
and diagnostics that want a graded "how well is this node covered" rather than
the boolean. Both are defined in the ``k == 1`` and ``k > 1`` branches alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Interpolation config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpolationConfig:
    """Spherical kNN-interpolation knobs.

    The interpolation step normalises both source and target points to the
    unit sphere, builds a cKDTree over sources, queries ``k_neighbors``
    nearest neighbours per target node, and computes a weighted average
    using either inverse-distance (``"idw"``) or Gaussian (``"gaussian"``)
    weights.

    Target nodes whose nearest source seed is farther than
    ``distance_threshold`` are flagged as ``too_far`` and handled by the
    OutputStrategy (which decides whether to fill with a default thickness,
    switch to mantle temperature, etc). The threshold is a great-circle angle
    in radians; the cKDTree query returns straight-line chord lengths, so the
    threshold is converted to a chord, ``2·sin(θ/2)``, before the comparison.
    It must lie in ``(0, π]``, and ``π`` — antipodal, the largest separation
    on a sphere — disables the test: nothing is ever ``too_far``.

    Defaults: 0.1 rad ≈ 640 km — generous; tighten to e.g. 0.02 rad
    (~127 km) for sharper boundaries. For polygon sources where the
    boundary is encoded by zero-thickness halo seeds, the threshold
    degenerates into a pathological-query guard and the actual roll-off
    length is controlled by ``gaussian_sigma`` instead.

    Frozen, and EVERY FIELD MUST STAY HASHABLE. ``ScalarFieldConnector`` keys
    the shared interpolation-geometry cache on this object by value, so a
    field holding a list, dict or set would make the config unhashable and the
    cache insertion would raise. Frozen also means ``__post_init__``
    validation cannot be bypassed by assigning to a field after construction.
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
        # Upper bound, inclusive. The threshold is a great-circle angle, so pi
        # is antipodal and already means "never too far"; beyond pi the chord
        # conversion 2*sin(theta/2) turns back down and a larger angle would
        # silently flag MORE nodes than a smaller one. Callers pass exactly
        # np.pi to disable the test, so the bound must admit it.
        if self.distance_threshold > np.pi:
            raise ValueError(
                f"distance_threshold must be at most pi ({np.pi:.6f} rad, "
                f"antipodal), got {self.distance_threshold}. It is a "
                f"great-circle angle in radians; pi already disables the "
                f"too_far test."
            )
        if self.gaussian_sigma <= 0:
            raise ValueError(
                f"gaussian_sigma must be positive, got {self.gaussian_sigma}"
            )


def _angle_to_chord(angle: float) -> float:
    """Convert a great-circle angle to the chord length subtending it.

    Distances on the unit sphere come out of cKDTree as straight lines through
    the sphere, not as arcs along its surface, so an angular threshold has to
    be converted before it can be compared against them. gtrack does the same
    conversion for its collision-removal radius.

    Computed where it is used and stored nowhere, so it cannot fall out of step
    with the angle it was derived from.

    Args:
        angle: great-circle angle in radians, in ``(0, π]``.

    Returns:
        The chord length, ``2·sin(angle/2)`` — except at ``π``, which returns
        infinity. ``2·sin(π/2)`` is exactly 2.0, but a float64 chord between
        two near-antipodal unit vectors comes out marginally *above* 2.0
        (measured 2.0000000000000004 through this code path), so a strict
        ``>`` against 2.0 would flag a node whose nearest source happens to sit
        almost opposite it. Callers pass ``π`` to mean "disable this test";
        infinity is that intent, exactly.
    """
    if angle >= np.pi:
        return np.inf
    return 2.0 * np.sin(angle / 2.0)


# ---------------------------------------------------------------------------
# SphericalKNNInterpolator
# ---------------------------------------------------------------------------

class SphericalKNNInterpolator:
    """Build spherical kNN geometry over a source cloud and gather through it.

    Holds an `InterpolationConfig`. ``geometry`` produces the property-
    independent bundle (cKDTree indices, blend weights, the ``too_far`` and
    ``coverage`` masks); ``gather`` is a stateless static method that applies a
    prebuilt bundle to one property array. The bundle is treated read-only by
    ``gather`` — its arrays are shared across sibling outputs, so only the
    freshly-allocated result array is ever written.
    """

    def __init__(self, config: InterpolationConfig | None = None):
        self.config = config or InterpolationConfig()

    def geometry(
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

        The bundle must be treated read-only by ``gather`` — its arrays are
        shared across sibling outputs.

        ``cfg.distance_threshold`` is a great-circle angle while the queried
        distances are chords, so the threshold is converted here, at the point
        of comparison, rather than being derived once and stored.

        The bundle carries ``coverage`` (the fraction of queried neighbours
        within the chord threshold) alongside ``too_far`` (the coverage-is-zero
        case, computed from the nearest neighbour). Both are defined in the
        ``k == 1`` branch too, so a consumer never has to special-case the
        single-neighbour bundle (F16).
        """
        cfg = self.config
        epsilon = 1e-10
        chord_threshold = _angle_to_chord(cfg.distance_threshold)

        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], epsilon)

        r_target = np.linalg.norm(target_coords, axis=1)
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], epsilon)

        tree = cKDTree(unit_source)
        k = min(cfg.k_neighbors, len(source_xyz))
        dists, idx = tree.query(unit_target, k=k)

        if k == 1:
            within = dists <= chord_threshold
            too_far = dists > chord_threshold
            coverage = within.astype(float)
            return {
                "k1": True,
                "idx": idx,
                "too_far": too_far,
                "coverage": coverage,
            }

        exact_match = dists[:, 0] < epsilon
        too_far = dists[:, 0] > chord_threshold
        # coverage: fraction of the k queried neighbours inside the threshold.
        # too_far is its zero case — the nearest (dists[:, 0]) is beyond the
        # threshold iff every neighbour is, iff coverage is 0 — so the two are
        # consistent by construction and too_far is computed exactly as before.
        coverage = np.mean(dists <= chord_threshold, axis=1)

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
            "coverage": coverage,
            "exact_match": exact_match,
            "weights": weights,
        }

    @staticmethod
    def gather(bundle: dict, prop: np.ndarray) -> np.ndarray:
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
