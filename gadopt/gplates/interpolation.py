"""Interpolate source channels onto target nodes on a sphere.

`SphericalKNNInterpolator` builds one geometry bundle for each source cloud.
All outputs for that source can use the same bundle.

The bundle contains `outside_source_range` and `neighbor_coverage`.
`outside_source_range` identifies target nodes with no source point in range.
`neighbor_coverage` gives the fraction of queried neighbors in range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# This angle gives the previous default unit-sphere chord width of 0.04.
DEFAULT_GAUSSIAN_WIDTH_RAD = 2.0 * np.arcsin(0.04 / 2.0)


# ---------------------------------------------------------------------------
# Interpolation config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpolationConfig:
    """Configure spherical nearest-neighbor interpolation.

    `neighbor_count` sets the maximum source-point count for each target node.
    `max_source_separation_rad` sets the maximum source separation in radians.
    `gaussian_width_rad` sets the Gaussian width as a surface angle in radians.

    The interpolator converts both angles to unit-sphere chord lengths.
    The default Gaussian angle preserves the previous chord width of 0.04.

    This frozen object is a key in the interpolation cache.
    Therefore, all fields must remain hashable.
    """

    kernel: str = "idw"
    neighbor_count: int = 50
    max_source_separation_rad: float = 0.1
    gaussian_width_rad: float = DEFAULT_GAUSSIAN_WIDTH_RAD

    def __post_init__(self):
        valid_kernels = ("idw", "gaussian")
        if self.kernel not in valid_kernels:
            raise ValueError(
                f"kernel must be one of {valid_kernels}, got '{self.kernel}'"
            )
        if self.neighbor_count < 1:
            raise ValueError(f"neighbor_count must be at least 1, got {self.neighbor_count}")
        if self.max_source_separation_rad <= 0:
            raise ValueError(
                f"max_source_separation_rad must be positive, got {self.max_source_separation_rad}"
            )
        # An angle of pi includes the complete sphere. Larger angles make the
        # chord threshold decrease and reverse the intended comparison.
        if self.max_source_separation_rad > np.pi:
            raise ValueError(
                f"max_source_separation_rad must be at most pi ({np.pi:.6f} rad, "
                f"antipodal), got {self.max_source_separation_rad}. It is a "
                f"great-circle angle in radians; pi already disables the "
                f"outside_source_range test."
            )
        if self.gaussian_width_rad <= 0:
            raise ValueError(
                f"gaussian_width_rad must be positive, got {self.gaussian_width_rad}"
            )
        if self.gaussian_width_rad > np.pi:
            raise ValueError(
                "gaussian_width_rad must be at most pi radians, "
                f"got {self.gaussian_width_rad}"
            )


def _angle_to_chord(angle: float) -> float:
    """Return `2 * sin(angle / 2)` for a unit sphere.

    An angle of pi returns infinity and disables the separation limit.
    """
    if angle >= np.pi:
        return np.inf
    return 2.0 * np.sin(angle / 2.0)


# ---------------------------------------------------------------------------
# SphericalKNNInterpolator
# ---------------------------------------------------------------------------

class SphericalKNNInterpolator:
    """Build interpolation geometry and apply it to source channels.

    `geometry` creates a read-only bundle for one source cloud.
    `gather` applies that bundle to one source channel.
    """

    def __init__(self, config: InterpolationConfig | None = None):
        self.config = config or InterpolationConfig()

    def geometry(
        self,
        source_xyz: np.ndarray,
        target_coords: np.ndarray,
    ) -> dict:
        """Build the interpolation geometry for a source cloud.

        The method normalizes source points and target nodes to the unit sphere.
        It converts each configured surface angle to a unit-sphere chord length.
        The returned bundle is independent of the source channels.
        """
        cfg = self.config
        epsilon = 1e-10
        chord_threshold = _angle_to_chord(cfg.max_source_separation_rad)

        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], epsilon)

        r_target = np.linalg.norm(target_coords, axis=1)
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], epsilon)

        tree = cKDTree(unit_source)
        k = min(cfg.neighbor_count, len(source_xyz))
        source_chord_distances, idx = tree.query(unit_target, k=k)

        if k == 1:
            within = source_chord_distances <= chord_threshold
            outside_source_range = source_chord_distances > chord_threshold
            neighbor_coverage = within.astype(float)
            return {
                "k1": True,
                "idx": idx,
                "outside_source_range": outside_source_range,
                "neighbor_coverage": neighbor_coverage,
            }

        exact_match = source_chord_distances[:, 0] < epsilon
        outside_source_range = source_chord_distances[:, 0] > chord_threshold
        # Neighbour coverage is zero when the nearest source point is out of range.
        neighbor_coverage = np.mean(source_chord_distances <= chord_threshold, axis=1)

        if cfg.kernel == "gaussian":
            gaussian_chord_width = 2.0 * np.sin(cfg.gaussian_width_rad / 2.0)
            weights = np.exp(
                -source_chord_distances**2 / (2 * gaussian_chord_width**2)
            )
        else:
            weights = 1.0 / np.maximum(source_chord_distances, epsilon)

        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, epsilon)

        return {
            "k1": False,
            "idx": idx,
            "outside_source_range": outside_source_range,
            "neighbor_coverage": neighbor_coverage,
            "exact_match": exact_match,
            "weights": weights,
        }

    @staticmethod
    def gather(bundle: dict, prop: np.ndarray) -> np.ndarray:
        """Gather one channel through a geometry bundle.

        The method does not modify the bundle because outputs share its arrays.
        It writes only to the new result array.
        """
        idx = bundle["idx"]
        if bundle["k1"]:
            return prop[idx].copy()

        weights = bundle["weights"]
        exact_match = bundle["exact_match"]
        interpolated = np.sum(weights * prop[idx], axis=1)
        interpolated[exact_match] = prop[idx[exact_match, 0]]
        return interpolated
