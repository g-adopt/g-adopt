"""Tests for the gadopt Source layer — PointCloudSource and the Source ABC.

After the gtrack/gadopt split the source layer is thin: PointCloudSource adapts
a gtrack ``AgeCloudSource`` producer to gadopt's collective, per-age-cached
``prepare(age)`` interface, and owns none of the plate-reconstruction machinery
itself. All the cloud production, checkpointing and walk logic lives in — and is
tested by — gtrack. So these tests exercise the adapter and the ABC contracts
against a fake producer, with no reconstruction data:

  * provides delegates to the producer;
  * the prepared dict carries xyz plus every provided channel, each copied out
    of the producer's cloud (F23);
  * validate_age enforces the forward-only floor for a monotonic-backward
    producer, on every rank, via the rank-coherent per-age cache (F19);
  * the per-age cache advances the producer at most once per age and returns
    the same dict by identity, and clears the interpolation geometry in lockstep.

The end-to-end path — PointCloudSource wrapping a REAL gtrack producer, driven
through GplatesScalarFunction — is covered by the reconstruction-backed
regression in test_connectors.py.
"""

import numpy as np
import pytest

from gadopt.gplates import PointCloudSource
from gtrack.age_sources import AgeCloudSource


# ---------------------------------------------------------------------------
# Test doubles — a fake producer and a fake gplates connector, no I/O
# ---------------------------------------------------------------------------

class _FakeCloud:
    """A cloud exposing exactly what PointCloudSource reads: ``xyz`` and
    ``get_property``. Both return the stored arrays BY REFERENCE, so a test can
    prove PointCloudSource copies them out (F23) by mutating afterwards."""

    def __init__(self, xyz, props):
        self.xyz = xyz
        self._props = props

    def get_property(self, name):
        return self._props[name]


class _FakeProducer:
    """Minimal gtrack ``AgeCloudSource``: returns a fixed cloud, counts at_age.

    Bound to the protocol (F27): ``provides`` and ``monotonic_backward`` are set
    as instance attributes, and ``at_age(age)`` / ``validate_age(age)`` match the
    protocol signatures, so ``isinstance`` against ``AgeCloudSource`` holds and a
    real producer could stand in without a signature drift going unnoticed. The
    single returned cloud is reused, so its arrays are shared between calls —
    which is exactly the aliasing PointCloudSource must copy away from.
    """

    def __init__(self, provides=("thickness", "age"),
                 monotonic_backward=False, n=12, reject_above=None):
        self.provides = frozenset(provides)
        self.monotonic_backward = monotonic_backward
        # Stands in for a producer-side limit only rank 0 can see (a declared
        # walk_start_age, or a producer oldest_age tighter than the connector's).
        self.reject_above = reject_above
        self.at_age_calls = 0
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(n, 3))
        xyz = 6.371e6 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        props = {k: rng.uniform(1.0, 2.0, size=n) for k in self.provides}
        self.cloud = _FakeCloud(xyz, props)

    def at_age(self, age):
        self.at_age_calls += 1
        return self.cloud

    def validate_age(self, age):
        if self.reject_above is not None and age > self.reject_above:
            raise ValueError(
                f"producer rejects {age}: older than its declared limit "
                f"{self.reject_above}"
            )


class _DummyGplates:
    """Stand-in for pyGplatesConnector: only the time-conversion helpers,
    delta_t and oldest_age are touched by the source path."""

    oldest_age = 100.0
    delta_t = 1.0

    def ndtime2age(self, ndtime):
        return float(ndtime) * 100.0

    def age2ndtime(self, age):
        return age / 100.0


def _source(monotonic_backward=False, provides=("thickness", "age")):
    return PointCloudSource(
        _FakeProducer(provides=provides, monotonic_backward=monotonic_backward),
        _DummyGplates(),
    )


# ---------------------------------------------------------------------------
# The producer protocol and construction guard
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_fake_producer_satisfies_the_protocol(self):
        # If this fails, every test below is testing the wrong thing.
        assert isinstance(_FakeProducer(), AgeCloudSource)

    def test_rejects_a_non_producer(self):
        with pytest.raises(TypeError, match="AgeCloudSource"):
            PointCloudSource(object(), _DummyGplates())

    def test_construction_does_not_drive_the_producer(self):
        producer = _FakeProducer()
        PointCloudSource(producer, _DummyGplates())
        assert producer.at_age_calls == 0


# ---------------------------------------------------------------------------
# provides delegation and the prepared dict
# ---------------------------------------------------------------------------

class TestProvidesAndPreparedDict:
    def test_provides_delegates_to_the_producer(self):
        src = _source(provides=("thickness", "membership"))
        assert src.provides == frozenset({"thickness", "membership"})
        # xyz is never listed in provides (F2), though it is in the dict.
        assert "xyz" not in src.provides

    def test_prepared_dict_has_xyz_plus_every_channel(self):
        src = _source(provides=("thickness", "age"))
        d = src.prepare(0.0)
        assert set(d) == {"xyz", "thickness", "age"}
        assert d["xyz"].shape == (12, 3)

    def test_channels_are_copied_out_of_the_producer_cloud(self):
        # F23: the accessors alias the producer's internal arrays. Mutating them
        # after prepare must NOT change the source's cached dict.
        producer = _FakeProducer(provides=("thickness",))
        src = PointCloudSource(producer, _DummyGplates())
        d = src.prepare(0.0)
        before_xyz = d["xyz"].copy()
        before_thk = d["thickness"].copy()
        producer.cloud.xyz[:] = -999.0
        producer.cloud.get_property("thickness")[:] = -999.0
        np.testing.assert_array_equal(d["xyz"], before_xyz)
        np.testing.assert_array_equal(d["thickness"], before_thk)


# ---------------------------------------------------------------------------
# validate_age: base range + forward-only floor (F19)
# ---------------------------------------------------------------------------

class TestValidateAge:
    def test_rejects_out_of_range(self):
        src = _source()
        with pytest.raises(ValueError, match="oldest age"):
            src.validate_age(150.0)  # > oldest_age 100
        with pytest.raises(ValueError, match="negative"):
            src.validate_age(-1.0)

    def test_forward_only_floor_for_monotonic_producer(self):
        src = _source(monotonic_backward=True)
        # Before stepping, any in-range age is allowed.
        src.validate_age(90.0)
        # Step to 50 Ma.
        src.prepare(50.0)
        # An older age is now unreachable; a younger one is fine.
        with pytest.raises(ValueError, match="last computed age"):
            src.validate_age(70.0)
        src.validate_age(40.0)

    def test_no_floor_for_non_monotonic_producer(self):
        src = _source(monotonic_backward=False)
        src.prepare(50.0)
        # A recompute-from-scratch producer may be asked for any age, in any
        # order — no floor.
        src.validate_age(70.0)

    def test_producer_limit_is_surfaced_before_the_collective(self):
        # A producer-side limit (a declared walk_start_age, or a producer
        # oldest_age tighter than the connector's) is enforced only inside the
        # producer, which runs on rank 0. PointCloudSource.validate_age must
        # surface it — evaluate on root, broadcast the verdict — so every rank
        # raises together rather than rank 0 raising alone inside the bcast in
        # prepare (F19: that is a hang, not a traceback). The age is within the
        # connector's range (oldest_age 100), so nothing but the producer's own
        # check can reject it.
        producer = _FakeProducer(reject_above=80.0)
        src = PointCloudSource(producer, _DummyGplates())
        src.validate_age(70.0)  # within the producer limit: fine
        with pytest.raises(ValueError, match="declared limit"):
            src.validate_age(90.0)  # in connector range, past the producer limit
        # And crucially the producer was never stepped — the rejection happened
        # in validate_age, before any collective / at_age.
        assert producer.at_age_calls == 0


# ---------------------------------------------------------------------------
# Per-age cache contracts (the ABC, driven through PointCloudSource)
# ---------------------------------------------------------------------------

class TestPerAgeCache:
    def test_producer_advances_at_most_once_per_age(self):
        producer = _FakeProducer()
        src = PointCloudSource(producer, _DummyGplates())
        src.prepare(60.0)
        src.prepare(60.0)  # within delta_t -> cache hit, no second at_age
        assert producer.at_age_calls == 1
        src.prepare(50.0)  # delta_t away -> miss
        assert producer.at_age_calls == 2

    def test_cache_hit_returns_the_same_dict_by_identity(self):
        # This identity is what lets two connectors sharing the source see a
        # single advance per age.
        src = _source()
        d1 = src.prepare(60.0)
        d2 = src.prepare(60.0)
        assert d1 is d2

    def test_geometry_cache_cleared_on_age_advance(self):
        src = _source()
        src.prepare(60.0)
        # Simulate a connector having cached geometry against this age.
        src.get_or_build_geometry("sentinel", lambda: {"built": True})
        assert src._interp_geometry_cache == {"sentinel": {"built": True}}
        # Advancing to a new age must clear it in lockstep with the source dict.
        src.prepare(50.0)
        assert src._interp_geometry_cache == {}
