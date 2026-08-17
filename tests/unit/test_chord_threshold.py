"""Test the angular-to-chord conversion for `max_source_separation_rad`.

`max_source_separation_rad` is a great-circle angle, but `cKDTree` returns chord
distances. The difference scales with the cube of the angle and can change the
`outside_source_range` value near the configured limit.

The tests use synthetic geometry with known angular separations.
"""

import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import (
    InterpolationConfig,
    OutputStrategy,
    ScalarFieldConnector,
    Source,
    SphericalKNNInterpolator,
)
from gadopt.gplates.interpolation import (
    DEFAULT_GAUSSIAN_WIDTH_RAD,
    _angle_to_chord,
)


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class FakeGplatesConnector:
    """Time conversions with no plate model behind them.

    Args:
        oldest_age: oldest age the fake model accepts, in Ma.
        delta_t: age window within which a cached result is reused, in Ma.
    """

    def __init__(self, oldest_age=200.0, delta_t=1.0):
        self.oldest_age = oldest_age
        self.delta_t = delta_t

    def ndtime2age(self, ndtime):
        return float(ndtime)

    def age2ndtime(self, age):
        return float(age)


class ExplicitCloudSource(Source):
    """A source whose cloud is given verbatim and never changes with age.

    Args:
        xyz: ``(n, 3)`` array of source points. Not normalised here; the
            connector normalises both clouds itself.
    """

    provides = frozenset({"thickness"})

    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)
        self.thickness = np.linspace(50.0, 200.0, len(self.xyz))
        self.gplates_connector = FakeGplatesConnector()
        self.comm = MPI.COMM_SELF
        self._is_root = True

    def _compute_sources(self, age):
        return {"xyz": self.xyz, "thickness": self.thickness}


class SourceRangeProbeOutput(OutputStrategy):
    """Return and retain the `outside_source_range` mask.

    The floating-point return value passes through the normal connector path.
    The retained mask lets tests inspect its Boolean type and shape.
    """

    requires = frozenset({"thickness"})

    def __init__(self):
        self.last_outside_source_range = None

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        self.last_outside_source_range = outside_source_range
        return outside_source_range.astype(float)


NORTH_POLE = np.array([0.0, 0.0, 1.0])

# These distant source points exercise the multi-neighbour branch. They are not
# the nearest source point for any target in these tests.
SOUTHERN_DECOYS = np.array([
    [0.10, 0.00, -0.99],
    [0.00, 0.10, -0.99],
    [-0.10, 0.00, -0.99],
])

# This fixed pair is nearly antipodal. Floating-point normalisation can place
# its measured chord on either side of the theoretical limit of 2.0.
ANTIPODAL_SEED = np.array([-2.013482844627813, -0.48239538641784946, 1.5224601960339834])
ANTIPODAL_TARGET = np.array([3.1728399750360494, 0.7601571425767154, -2.399088019680025])


def point_at_angle(theta):
    """Return the unit point at great-circle angle ``theta`` from the north pole.

    Args:
        theta: great-circle angle in radians.

    Returns:
        A length-3 array on the unit sphere, in the x-z plane.
    """
    return np.array([np.sin(theta), 0.0, np.cos(theta)])


def outside_source_range_mask(source_xyz, thetas, max_source_separation_rad, neighbor_count=50):
    """Run the connector once and return its `outside_source_range` mask.

    Args:
        source_xyz: the source cloud.
        thetas: great-circle angles from the north pole, one per target node.
        max_source_separation_rad: the configured threshold, in radians.
        neighbor_count: Source-point count for each target node.

    Returns:
        One Boolean `outside_source_range` value per target node.
    """
    source = ExplicitCloudSource(source_xyz)
    output = SourceRangeProbeOutput()
    connector = ScalarFieldConnector(
        source,
        output,
        interpolation=InterpolationConfig(
            neighbor_count=neighbor_count, max_source_separation_rad=max_source_separation_rad
        ),
    )
    targets = np.array([point_at_angle(t) for t in thetas])
    connector.get_indicator(targets, ndtime=0.0)
    return output.last_outside_source_range


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------

class TestAngleToChord:
    def test_matches_the_closed_form(self):
        assert _angle_to_chord(0.5) == 2.0 * np.sin(0.25)
        assert _angle_to_chord(0.02) == 2.0 * np.sin(0.01)

    def test_pi_disables_the_test(self):
        # Infinity avoids a floating-point comparison at the antipodal limit.
        assert _angle_to_chord(np.pi) == np.inf


class TestGaussianWidthConversion:
    def test_default_preserves_the_previous_chord_width(self):
        chord_width = 2.0 * np.sin(DEFAULT_GAUSSIAN_WIDTH_RAD / 2.0)
        assert chord_width == pytest.approx(0.04)

    def test_geometry_converts_the_angular_width(self):
        width_rad = 0.5
        interpolator = SphericalKNNInterpolator(
            InterpolationConfig(
                kernel="gaussian",
                neighbor_count=2,
                gaussian_width_rad=width_rad,
            )
        )
        source_xyz = np.array([NORTH_POLE, point_at_angle(0.5)])
        bundle = interpolator.geometry(source_xyz, NORTH_POLE[None, :])
        chord_width = 2.0 * np.sin(width_rad / 2.0)
        source_distance = 2.0 * np.sin(0.5 / 2.0)
        unnormalized = np.array(
            [1.0, np.exp(-source_distance**2 / (2.0 * chord_width**2))]
        )
        expected = unnormalized / unnormalized.sum()
        np.testing.assert_allclose(bundle["weights"][0], expected)

    def test_width_more_than_pi_is_rejected(self):
        with pytest.raises(ValueError, match="at most pi radians"):
            InterpolationConfig(gaussian_width_rad=np.pi + 1e-9)


# ---------------------------------------------------------------------------
# The upper bound
# ---------------------------------------------------------------------------

class TestUpperBound:
    def test_pi_is_admitted(self):
        # The inclusive limit lets callers disable the source-range test.
        cfg = InterpolationConfig(max_source_separation_rad=np.pi)
        assert cfg.max_source_separation_rad == np.pi

    def test_above_pi_is_rejected(self):
        # Beyond pi, a larger angle gives a smaller chord threshold.
        with pytest.raises(ValueError, match="at most pi"):
            InterpolationConfig(max_source_separation_rad=np.pi + 1e-9)
        with pytest.raises(ValueError, match="at most pi"):
            InterpolationConfig(max_source_separation_rad=4.0)

    def test_existing_lower_bound_survives(self):
        with pytest.raises(ValueError, match="must be positive"):
            InterpolationConfig(max_source_separation_rad=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            InterpolationConfig(max_source_separation_rad=-0.1)


# ---------------------------------------------------------------------------
# The straddle: nodes in the window where the two conventions disagree
# ---------------------------------------------------------------------------

class TestStraddle:
    @pytest.mark.parametrize(
        "source_xyz",
        [
            NORTH_POLE[None, :],
            np.vstack([NORTH_POLE[None, :], SOUTHERN_DECOYS]),
        ],
        ids=["single-neighbour", "multiple-neighbours"],
    )
    def test_threshold_and_mask_contract(self, source_xyz):
        # The cases exercise the single-neighbour and multi-neighbour branches.
        thetas = [0.497, 0.503, 0.51]
        mask = outside_source_range_mask(source_xyz, thetas, 0.5)
        np.testing.assert_array_equal(mask, [False, True, True])
        assert mask.dtype == np.bool_
        assert mask.shape == (len(thetas),)


# ---------------------------------------------------------------------------
# theta = pi means disabled, including for near-antipodal nodes
# ---------------------------------------------------------------------------

class TestAntipodal:
    def measured_chord(self):
        """The chord the connector's own normalisation produces for the pair."""
        seed = ANTIPODAL_SEED / np.linalg.norm(ANTIPODAL_SEED)
        target = ANTIPODAL_TARGET / np.linalg.norm(ANTIPODAL_TARGET)
        return float(np.linalg.norm(seed - target))

    def test_the_pair_really_is_pathological(self):
        # The selected pair must remain at the floating-point antipodal limit.
        assert self.measured_chord() == pytest.approx(2.0, abs=1e-12)

    def run(self, cloud, max_source_separation_rad):
        """Query the antipodal target against ``cloud`` and return ``outside_source_range``.

        Args:
            cloud: the source cloud.
            max_source_separation_rad: the configured threshold, in radians.

        Returns:
            The boolean ``outside_source_range`` array for the single antipodal target.
        """
        source = ExplicitCloudSource(cloud)
        output = SourceRangeProbeOutput()
        connector = ScalarFieldConnector(
            source,
            output,
            interpolation=InterpolationConfig(
                max_source_separation_rad=max_source_separation_rad
            ),
        )
        connector.get_indicator(ANTIPODAL_TARGET[None, :], ndtime=0.0)
        return output.last_outside_source_range

    @pytest.mark.parametrize(
        "cloud",
        [
            ANTIPODAL_SEED[None, :],
            np.vstack([
                ANTIPODAL_SEED[None, :],
                (ANTIPODAL_SEED + np.array([1e-3, -1e-3, 1e-3]))[None, :],
            ]),
        ],
        ids=["single-neighbour", "multiple-neighbours"],
    )
    def test_nothing_is_outside_source_range_at_pi(self, cloud):
        assert not self.run(cloud, np.pi).any()

    def test_just_below_pi_still_flags_it(self):
        # A smaller angle restores the finite source-range limit.
        assert self.run(ANTIPODAL_SEED[None, :], 3.0).all()
