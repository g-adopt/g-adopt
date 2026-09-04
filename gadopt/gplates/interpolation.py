"""Interpolate source channels onto target nodes on a sphere.

The split between geometry and gathering is the design point here. Building a
``cKDTree`` over the source cloud and querying it for every target node costs
far more than the weighted sum that follows, and that cost depends only on the
1) source cloud, 2) target nodes, 3) config.
``SphericalKNNInterpolator.geometry`` therefore returns a dict of
neighbor indices and weights that every output reading the same source at the
same age can reuse, and ``gather`` applies that geometry to one channel at a
time. The source-side cache in ``Source.get_or_build_geometry`` is what actually
holds the geometry between siblings; see ``gadopt.gplates.sources``.

The geometry has two diagnostic masks so outputs can decide what to do
where the interpolation is untrustworthy. ``outside_source_range`` for target
nodes whose nearest source point is further away than
``InterpolationConfig.max_source_separation_rad``, which is how a bounded
source (a continental polygon, say) tells an output to use the "fallback" value.
``neighbor_coverage`` decides the fraction of the
queried neighbors that fell inside that range, which fades smoothly to zero
across the edge of a source cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# This angle gives the previous default unit-sphere chord width of 0.04.
DEFAULT_GAUSSIAN_WIDTH_RAD = 2.0 * np.arcsin(0.04 / 2.0)

# Numerical safety margin. Used to floor a radius before normalising,
# to detect a chord distance close enough to zero to count as an exact
# match, and to floor a divisor against division by zero.
_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Interpolation config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpolationConfig:
    """Configure spherical nearest-neighbor interpolation.

    Both angles are given as great-circle angles on the sphere, in radians,
    because that is the unit a user can reason about: it is a distance along
    the surface, independent of the mesh radius. The interpolator converts them
    to unit-sphere chord lengths internally, since that is what ``cKDTree``
    measures in three dimensions. The default ``gaussian_width_rad`` is the
    angle whose chord is 0.04, which is the width this code used before the
    angles became configurable.

    Args:
        kernel: Weighting kernel, either ``"idw"`` (inverse distance) or
            ``"gaussian"``.
        neighbor_count: Maximum number of source points to query for each
            target node.
        max_source_separation_rad: Great-circle angle beyond which a source
            point counts as out of range. A value of pi disables the test.
        gaussian_width_rad: Great-circle angle used as the Gaussian width.
            Ignored when ``kernel`` is ``"idw"``.

    Raises:
        ValueError: If the kernel is unknown, or any angle or count is outside
            its valid range.
    """

    VALID_KERNELS = ("idw", "gaussian")

    kernel: str = "gaussian"
    neighbor_count: int = 50
    max_source_separation_rad: float = 0.1
    gaussian_width_rad: float = DEFAULT_GAUSSIAN_WIDTH_RAD

    def __post_init__(self):
        if self.kernel not in self.VALID_KERNELS:
            raise ValueError(
                f"kernel must be one of {self.VALID_KERNELS}, got '{self.kernel}'"
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
        # Forcing an unhashable field (a list, a numpy array) to fail.
        # This is to avoid silent breaking of the geometry cache
        # the first time this config is used as a key.
        hash(self)


def _angle_to_chord(angle: float) -> float:
    """Convert a great-circle angle to a unit-sphere chord length.

    An angle of pi (antipodal) maps to infinity rather than to the true chord
    of 2, so that a caller asking for the whole sphere gets a threshold nothing
    can exceed. The exact value 2 would still reject points at machine
    precision on the far side.

    Args:
        angle: Great-circle angle in radians.

    Returns:
        The chord length ``2 * sin(angle / 2)``, or infinity for an angle of pi
        or more.
    """
    if angle >= np.pi:
        return np.inf
    return 2.0 * np.sin(angle / 2.0)


# ---------------------------------------------------------------------------
# SphericalKNNInterpolator
# ---------------------------------------------------------------------------

class SphericalKNNInterpolator:
    """Build interpolation geometry and apply it to source channels.

    ``geometry`` does the expensive part once for a source cloud and returns a
    dict that callers treat as read-only, because siblings share it. Each
    call to ``gather`` reads one channel through that geometry and allocates its
    own result.

    Args:
        config: Interpolation settings. Defaults to ``InterpolationConfig()``.
    """

    def __init__(self, config: InterpolationConfig | None = None):
        self.config = config or InterpolationConfig()

    def geometry(
        self,
        source_xyz: np.ndarray,
        target_coords: np.ndarray,
    ) -> dict:
        """Build the interpolation geometry for a source cloud.

        Both point-sets are projected onto the unit sphere before the query, so
        that source points reconstructed at one radius and mesh nodes sitting
        at another still compare by angular distance alone.

        The returned dict carries no channel data, which is what makes it shareable
        between every output reading this source.

        Args:
            source_xyz: Source point coordinates, shape ``(n_source, 3)``.
            target_coords: Target node coordinates, shape ``(n_target, 3)``.

        Returns:
            A dictionary with the neighbor indices ``idx``, the masks
            ``outside_source_range`` and ``neighbor_coverage``, and a ``k1``
            flag. When ``k1`` is False the geometry also carries the normalised
            ``weights`` and the ``exact_match`` mask.
            Should be treated as read-only.
        """
        cfg = self.config
        chord_threshold = _angle_to_chord(cfg.max_source_separation_rad)

        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], _EPSILON)

        r_target = np.linalg.norm(target_coords, axis=1)
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], _EPSILON)

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

        exact_match = source_chord_distances[:, 0] < _EPSILON
        outside_source_range = source_chord_distances[:, 0] > chord_threshold
        # Neighbor coverage is zero when the nearest source point is out of range.
        neighbor_coverage = np.mean(source_chord_distances <= chord_threshold, axis=1)

        if cfg.kernel == "gaussian":
            gaussian_chord_width = 2.0 * np.sin(cfg.gaussian_width_rad / 2.0)
            weights = np.exp(
                -source_chord_distances**2 / (2 * gaussian_chord_width**2)
            )
        else:
            weights = 1.0 / np.maximum(source_chord_distances, _EPSILON)

        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, _EPSILON)

        return {
            "k1": False,
            "idx": idx,
            "outside_source_range": outside_source_range,
            "neighbor_coverage": neighbor_coverage,
            "exact_match": exact_match,
            "weights": weights,
        }

    @staticmethod
    def gather(geometry: dict, prop: np.ndarray) -> np.ndarray:
        """Gather one channel through a geometry dict.

        Nothing in the geometry is modified, since several outputs hold the
        same arrays; the only writes go to the freshly allocated result.
        Target nodes that coincide with a source point take that point's
        value directly, which avoids the division by a near-zero distance
        that the inverse-distance kernel would otherwise hit.

        Args:
            geometry: The dict returned by ``geometry``.
            prop: Source channel values, shape ``(n_source,)``.

        Returns:
            The interpolated values at the target nodes, shape
            ``(n_target,)``.
        """
        idx = geometry["idx"]
        if geometry["k1"]:
            return prop[idx].copy()

        weights = geometry["weights"]
        exact_match = geometry["exact_match"]
        interpolated = np.sum(weights * prop[idx], axis=1)
        interpolated[exact_match] = prop[idx[exact_match, 0]]
        return interpolated
