"""Tests for the angular-to-chord conversion of ``distance_threshold``.

``distance_threshold`` is documented as a great-circle angle in radians, but
the distances it is compared against come out of cKDTree as chords — straight
lines through the sphere. The two agree to O(θ³/24), which is why the mismatch
survived unnoticed, and disagree by a full ``too_far`` flip for any node whose
nearest seed lands in the window between the two conventions.

Everything here constructs its own geometry: one seed at the north pole and
target nodes at chosen polar angles from it, so the great-circle angle between
a target and its nearest seed is exactly the number written in the test. No
reconstruction data, no downloaded files, no Firedrake solve.
"""

import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import InterpolationConfig, OutputStrategy, ScalarFieldConnector, Source
from gadopt.gplates.interpolation import _angle_to_chord


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


class TooFarProbeOutput(OutputStrategy):
    """Hand back the ``too_far`` mask itself, and keep a reference to it.

    Returning the mask as floats routes it through the real connector path
    (``compute`` is called exactly as any output would be), while the stored
    reference lets a test check the mask's dtype and shape directly.
    """

    requires = frozenset({"thickness"})

    def __init__(self):
        self.last_too_far = None

    def compute(self, interpolated, r_target, too_far, mesh):
        self.last_too_far = too_far
        return too_far.astype(float)


NORTH_POLE = np.array([0.0, 0.0, 1.0])

# Far-away decoys, clustered near the south pole. Their only job is to push the
# neighbour count above one so the k > 1 branch of _interp_geometry runs; they
# are never the nearest seed for any target used here.
SOUTHERN_DECOYS = np.array([
    [0.10, 0.00, -0.99],
    [0.00, 0.10, -0.99],
    [-0.10, 0.00, -0.99],
])

# A deliberately near-antipodal pair, found by search and hard-coded so the
# arithmetic is fixed. Through the connector's own normalisation the chord
# between them comes out just ABOVE 2.0 (2.0000000000000004 measured here),
# which is what makes a literal 2.0 threshold unsafe at theta = pi.
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


def too_far_mask(source_xyz, thetas, distance_threshold, k_neighbors=50):
    """Run the connector once and return the ``too_far`` mask it produced.

    Args:
        source_xyz: the source cloud.
        thetas: great-circle angles from the north pole, one per target node.
        distance_threshold: the configured threshold, in radians.
        k_neighbors: neighbour count handed to the config.

    Returns:
        The boolean ``too_far`` array, one entry per target node.
    """
    source = ExplicitCloudSource(source_xyz)
    output = TooFarProbeOutput()
    connector = ScalarFieldConnector(
        source,
        output,
        interpolation=InterpolationConfig(
            k_neighbors=k_neighbors, distance_threshold=distance_threshold
        ),
    )
    targets = np.array([point_at_angle(t) for t in thetas])
    connector.get_indicator(targets, ndtime=0.0)
    return output.last_too_far


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------

class TestAngleToChord:
    def test_matches_the_closed_form(self):
        assert _angle_to_chord(0.5) == 2.0 * np.sin(0.25)
        assert _angle_to_chord(0.02) == 2.0 * np.sin(0.01)

    def test_small_angles_barely_move(self):
        # chord = theta - theta**3/24 + ..., so the correction is 0.042% at the
        # 0.1 default. This is why the mismatch went unnoticed for so long.
        theta = 0.1
        assert abs(_angle_to_chord(theta) - theta) / theta < 1e-3

    def test_pi_disables_the_test(self):
        # NOT 2.0, which is what the closed form gives. See the near-antipodal
        # case below for why 2.0 is the wrong number to compare against.
        assert _angle_to_chord(np.pi) == np.inf
        assert 2.0 * np.sin(np.pi / 2.0) == 2.0


# ---------------------------------------------------------------------------
# The upper bound
# ---------------------------------------------------------------------------

class TestUpperBound:
    def test_pi_is_admitted(self):
        # Inclusive on purpose: all four of the production drivers pass
        # exactly np.pi to mean "never flag anything". An exclusive bound
        # would break every one of them at construction.
        cfg = InterpolationConfig(distance_threshold=np.pi)
        assert cfg.distance_threshold == np.pi

    def test_above_pi_is_rejected(self):
        # Beyond pi, 2*sin(theta/2) turns back down, so a larger angle would
        # flag MORE nodes than a smaller one — a silent behaviour reversal.
        with pytest.raises(ValueError, match="at most pi"):
            InterpolationConfig(distance_threshold=np.pi + 1e-9)
        with pytest.raises(ValueError, match="at most pi"):
            InterpolationConfig(distance_threshold=4.0)

    def test_existing_lower_bound_survives(self):
        with pytest.raises(ValueError, match="must be positive"):
            InterpolationConfig(distance_threshold=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            InterpolationConfig(distance_threshold=-0.1)


# ---------------------------------------------------------------------------
# The straddle: nodes in the window where the two conventions disagree
# ---------------------------------------------------------------------------

class TestStraddle:
    # At distance_threshold = 0.5 the old code compared the chord against 0.5
    # directly, so it flagged a node only above 2*asin(0.25) = 0.505361 rad.
    # The new code compares against 2*sin(0.25) = 0.497 chord, i.e. exactly
    # 0.5 rad. Everything between those two angles flips.
    OLD_BOUNDARY_RAD = 2.0 * np.arcsin(0.25)
    NEW_BOUNDARY_RAD = 0.5

    def test_the_window_is_where_it_is_claimed_to_be(self):
        assert self.NEW_BOUNDARY_RAD < 0.503 < self.OLD_BOUNDARY_RAD
        assert 0.497 < self.NEW_BOUNDARY_RAD
        assert self.OLD_BOUNDARY_RAD < 0.51
        assert np.isclose(self.OLD_BOUNDARY_RAD, 0.505361, atol=1e-6)

    def test_single_seed_branch(self):
        # One seed, so k collapses to 1 and the k == 1 branch runs.
        mask = too_far_mask(NORTH_POLE[None, :], [0.497, 0.503, 0.51], 0.5)
        # 0.497: inside under both conventions.
        # 0.503: in the window — too_far under the new convention only.
        # 0.51:  outside under both.
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_multi_neighbour_branch(self):
        # Same geometry plus distant decoys, so k > 1 and the other branch of
        # _interp_geometry runs. The nearest seed is still the pole.
        cloud = np.vstack([NORTH_POLE[None, :], SOUTHERN_DECOYS])
        mask = too_far_mask(cloud, [0.497, 0.503, 0.51], 0.5)
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_mask_shape_and_dtype_unchanged(self):
        # too_far's values move; its presence and shape must not. Both
        # branches produce one boolean per target node.
        thetas = [0.1, 0.503, 1.0]
        single = too_far_mask(NORTH_POLE[None, :], thetas, 0.5)
        multi = too_far_mask(
            np.vstack([NORTH_POLE[None, :], SOUTHERN_DECOYS]), thetas, 0.5
        )
        for mask in (single, multi):
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
        # If this ever fails, the antipodal pair above stopped being the edge
        # case it was chosen for and the test below stops proving anything. The
        # chord sits right at the antipodal limit of 2.0, where float64 rounding
        # lands it at or a couple of ulp either side of 2.0 depending on the
        # platform (exactly 2.0 on Linux x86, 2.0000000000000004 on macOS/arm64).
        # A literal 2.0 threshold is a knife-edge there, which is why
        # _angle_to_chord(pi) returns inf rather than 2.0.
        assert self.measured_chord() == pytest.approx(2.0, abs=1e-12)

    def run(self, cloud, distance_threshold):
        """Query the antipodal target against ``cloud`` and return ``too_far``.

        Args:
            cloud: the source cloud.
            distance_threshold: the configured threshold, in radians.

        Returns:
            The boolean ``too_far`` array for the single antipodal target.
        """
        source = ExplicitCloudSource(cloud)
        output = TooFarProbeOutput()
        connector = ScalarFieldConnector(
            source,
            output,
            interpolation=InterpolationConfig(
                distance_threshold=distance_threshold
            ),
        )
        connector.get_indicator(ANTIPODAL_TARGET[None, :], ndtime=0.0)
        return output.last_too_far

    def test_nothing_is_too_far_at_pi(self):
        # The cloud is the antipodal seed alone. Adding any other seed would
        # defeat the test: nothing on a sphere can be farther from the target
        # than its antipode, so any companion seed becomes the nearest one and
        # the near-antipodal distance never reaches the comparison.
        assert not self.run(ANTIPODAL_SEED[None, :], np.pi).any()

    def test_nothing_is_too_far_at_pi_multi_neighbour(self):
        # Same at k > 1, with a second seed perturbed off the first so both
        # stay near-antipodal to the target and the k > 1 branch runs.
        companion = ANTIPODAL_SEED + np.array([1e-3, -1e-3, 1e-3])
        cloud = np.vstack([ANTIPODAL_SEED[None, :], companion[None, :]])
        assert not self.run(cloud, np.pi).any()

    def test_just_below_pi_still_flags_it(self):
        # The counterpart, and the proof that pi is a genuine special case
        # rather than the threshold having been disabled outright: at 3.0 rad
        # the chord bound is 1.99499 and the same node IS too_far.
        assert self.run(ANTIPODAL_SEED[None, :], 3.0).all()
