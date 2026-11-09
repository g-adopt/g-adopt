"""Tests for PointCloudSource, the adapter between a gtrack producer and gadopt.

PointCloudSource wraps a gtrack ``AgeCloudSource`` producer and gives it the
cached, collective ``prepare(age)`` interface of ``Source``. These tests drive
it with a Mock producer and a Mock plate model connector, so they need no
reconstruction data. The path with a real gtrack producer is covered by the
reconstruction-backed tests in test_connectors.py.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import PointCloudSource, pyGplatesConnector
from gtrack import PointCloud
from gtrack.age_sources import AgeCloudSource

# Oldest age that the Mock plate model accepts, in Ma.
OLDEST_AGE = 100.0
# Age window within which Source.prepare reuses its cached result, in Ma.
DELTA_T = 1.0


def _source(provides=("thickness", "age"), monotonic_backward=False,
            comm=MPI.COMM_WORLD):
    """Build a PointCloudSource around a Mock producer.

    The producer returns the same cloud at every age. The cloud's ``xyz`` and
    ``get_property`` return the stored arrays themselves, not copies, so a test
    can change them after ``prepare`` and check that PointCloudSource copied
    them out.

    Args:
        provides: Channel names that the producer declares.
        monotonic_backward: Whether the producer can only step to younger ages.
        comm: MPI communicator that PointCloudSource broadcasts on.

    Returns:
        Tuple ``(source, producer)``. A test uses the producer to inspect call
        counts and to reach the cloud through ``producer.at_age.return_value``.
    """
    # A few random points on a sphere of Earth radius, in m, each with random
    # channel values. The values are irrelevant; only their identity matters.
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(12, 3))
    xyz *= 6.371e6 / np.linalg.norm(xyz, axis=1, keepdims=True)
    channels = {name: rng.uniform(1.0, 2.0, size=len(xyz)) for name in provides}

    # spec=AgeCloudSource lets the Mock pass the isinstance check in
    # PointCloudSource. Both protocol attributes must be set explicitly: with a
    # spec, an unset ``provides`` raises AttributeError, and an unset
    # ``monotonic_backward`` returns a child Mock, which is truthy and would
    # switch on the forward-only age check.
    producer = Mock(spec=AgeCloudSource)
    producer.provides = frozenset(provides)
    producer.monotonic_backward = monotonic_backward
    producer.at_age.return_value = Mock(
        spec=PointCloud, xyz=xyz, get_property=Mock(side_effect=channels.__getitem__)
    )

    gplates_connector = Mock(
        spec=pyGplatesConnector, oldest_age=OLDEST_AGE, delta_t=DELTA_T
    )
    return PointCloudSource(producer, gplates_connector, comm=comm), producer


def test_construction_does_not_drive_the_producer():
    _, producer = _source()
    producer.at_age.assert_not_called()


class TestProvidesAndPreparedDict:
    def test_provides_delegates_to_the_producer(self):
        src, _ = _source(provides=("thickness", "membership"))
        assert src.provides == frozenset({"thickness", "membership"})
        # The prepared dict always carries xyz, but provides never lists it.
        assert "xyz" not in src.provides

    def test_prepared_dict_has_xyz_plus_every_channel(self):
        src, producer = _source()
        prepared = src.prepare(0.0)
        cloud = producer.at_age.return_value

        assert set(prepared) == {"xyz", "thickness", "age"}
        np.testing.assert_array_equal(prepared["xyz"], cloud.xyz)
        for name in producer.provides:
            np.testing.assert_array_equal(prepared[name], cloud.get_property(name))

    def test_channels_are_copied_out_of_the_producer_cloud(self):
        # A gtrack producer can reuse its internal arrays between calls, so a
        # later change to the cloud must not reach the cached dict.
        #
        # mpi4py's bcast pickles the dict, so it returns a copy even on rank
        # zero of a one-process run, and that copy would hide a missing copy
        # in PointCloudSource. This Mock communicator returns its argument
        # unchanged, so the only copy is the one PointCloudSource makes.
        comm = Mock(spec=MPI.Comm, rank=0)
        comm.bcast.side_effect = lambda obj, root: obj
        src, producer = _source(provides=("thickness",), comm=comm)
        prepared = src.prepare(0.0)
        xyz_before = prepared["xyz"].copy()
        thickness_before = prepared["thickness"].copy()

        cloud = producer.at_age.return_value
        cloud.xyz[:] = -999.0
        cloud.get_property("thickness")[:] = -999.0

        np.testing.assert_array_equal(prepared["xyz"], xyz_before)
        np.testing.assert_array_equal(prepared["thickness"], thickness_before)


class TestValidateAge:
    @pytest.mark.parametrize(
        "age, message",
        [
            (OLDEST_AGE + 50.0, "oldest age"),
            (-1.0, "negative"),
        ],
    )
    def test_rejects_out_of_range(self, age, message):
        src, _ = _source()
        with pytest.raises(ValueError, match=message):
            src.validate_age(age)

    def test_forward_only_floor_for_monotonic_producer(self):
        src, _ = _source(monotonic_backward=True)
        # Before the first prepare, any age inside the plate model is allowed.
        src.validate_age(90.0)
        src.prepare(50.0)
        # After a step to 50 Ma, an older age is unreachable and a younger one
        # is allowed.
        with pytest.raises(ValueError, match="last computed age"):
            src.validate_age(70.0)
        src.validate_age(40.0)

    def test_no_floor_for_non_monotonic_producer(self):
        # A producer that recomputes from scratch accepts ages in any order.
        src, _ = _source(monotonic_backward=False)
        src.prepare(50.0)
        src.validate_age(70.0)

    def test_producer_limit_is_checked_before_the_collective(self):
        # A limit that only the producer knows, for example a declared oldest
        # requested age, is checked on rank zero. PointCloudSource.validate_age
        # must call that check before any collective work, so the error comes
        # from validate_age and not from inside the broadcast in prepare. The
        # ages here are inside the plate model range, so only the producer's
        # own check can reject them.
        src, producer = _source()
        producer_limit = 80.0

        def reject_older_than_limit(age):
            if age > producer_limit:
                raise ValueError(
                    f"Age {age} Ma is older than the producer limit "
                    f"{producer_limit} Ma"
                )

        producer.validate_age.side_effect = reject_older_than_limit

        src.validate_age(70.0)
        with pytest.raises(ValueError, match="older than the producer limit"):
            src.validate_age(90.0)
        # The error came from validate_age, so the producer never stepped.
        producer.at_age.assert_not_called()


class TestPerAgeCache:
    def test_producer_advances_at_most_once_per_age(self):
        src, producer = _source()
        src.prepare(60.0)
        src.prepare(60.0)
        assert producer.at_age.call_count == 1
        # A difference of exactly DELTA_T is outside the reuse window.
        src.prepare(60.0 - DELTA_T)
        assert producer.at_age.call_count == 2

    def test_cache_hit_returns_the_same_dict(self):
        # Two connectors that share one source see a single producer step per
        # age because both receive this same dict.
        src, _ = _source()
        assert src.prepare(60.0) is src.prepare(60.0)

    def test_geometry_cache_cleared_on_age_advance(self):
        src, _ = _source()
        build_geometry = Mock(return_value={"tree": "built"})

        # Two requests at one age share a single build.
        src.prepare(60.0)
        first = src.get_or_build_geometry("key", build_geometry)
        second = src.get_or_build_geometry("key", build_geometry)
        assert build_geometry.call_count == 1
        assert first is build_geometry.return_value
        assert second is first

        # A new age clears the geometry, so the next request builds again.
        src.prepare(50.0)
        src.get_or_build_geometry("key", build_geometry)
        assert build_geometry.call_count == 2
