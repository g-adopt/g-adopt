"""Tests for InterpolationConfig and the interpolation-geometry cache key.

The tests mock the classes, so they do not depend on the interpolation
mathematics. Each class is minimal and only passes the internal API checks.
Some fixtures override a method with a Mock. The call count of that Mock is a
proxy for cache misses.

The property under test is that the geometry cache is keyed on the
interpolation config *by value*. Two connectors that hold equal-valued but
distinct configs must share one cKDTree build. Two connectors whose configs
differ must not, and each must get a separate cache entry.
"""

import numpy as np
from mpi4py import MPI

from gadopt.gplates import (
    InterpolationConfig,
    OutputStrategy,
    ScalarFieldConnector,
    Source,
)

import pytest
from unittest.mock import Mock


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


class MockSource(Source):
    """A minimal Source subclass that satisfies the API of a Connector.

    `_compute_sources` is an abstract method, so the subclass must define it. A
    fixture replaces `_compute_sources` with a Mock to record the call count.
    """

    provides = frozenset({"thickness"})

    def __init__(self):
        # Fake Gplates connector
        self.gplates_connector = FakeGplatesConnector()
        self.comm = MPI.COMM_SELF
        self._is_root = True

    def _compute_sources(self, age):
        # To be overridden by a Mock() object
        pass


class PassThroughOutput(OutputStrategy):
    """Return the interpolated thickness unchanged."""

    requires = frozenset({"thickness"})

    def compute(self, interpolated, r_target, too_far, mesh):
        return interpolated["thickness"]


@pytest.fixture
def mockedSource():
    """Provides a MockSource object with a Mock() `_compute_sources` method

    Returns:
        A MockSource object with a Mock `_compute_sources` function
    """
    source = MockSource()
    source._compute_sources = Mock(
        return_value={"xyz": np.array([[1, 2, 3]]), "thickness": np.array([1.0])}
    )
    return source


@pytest.fixture
def mockedFieldConnector(mockedSource):
    """Provides a ScalarFieldConnector whose `_interp_geometry` method is a Mock().

    The connector's `_interp_geometry` method builds the interpolation geometry on a
    cache miss. The Mock replaces this method and counts the calls. The call count is
    a proxy for the number of cache misses.

    Args:
        mockedSource: The `mockedSource` pytest fixture

    Returns:
        A ScalarFieldConnector object with a Mock `_interp_geometry` method
    """
    cfg_idw = InterpolationConfig(kernel="idw", k_neighbors=4)
    conn_idw = ScalarFieldConnector(
        mockedSource, PassThroughOutput(), interpolation=cfg_idw
    )
    conn_idw._interp_geometry = Mock(
        return_value={"idx": 0, "k1": True, "too_far": None}
    )
    return conn_idw


def unit_targets(n_points=2, seed=1):
    """Build target coordinates on the unit sphere.

    The tests mock the geometry, so the code does not interpolate these targets.
    The targets only need to be a reproducible and valid coordinate array. The
    cache key is a hash of the array bytes. This key must be stable across a run.
    When the targets change, the key must change too.

    Args:
        n_points: number of target nodes.
        seed: RNG seed, for reproducibility.

    Returns:
        An ``(n_points, 3)`` array of points on the unit sphere.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_points, 3))
    return xyz / np.linalg.norm(xyz, axis=1)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Geometry cache keyed by value
# ---------------------------------------------------------------------------


class TestGeometryCacheKey:
    def test_cache_used(self, mockedSource, mockedFieldConnector):
        # Test that the geometry cache is used when the same `ScalarFieldConnector`
        # makes two `get_indicator` calls on the same target at the same `ndtime`.
        targets = unit_targets()
        _ = mockedFieldConnector.get_indicator(targets, ndtime=0.0)
        _ = mockedFieldConnector.get_indicator(targets, ndtime=0.0)

        # Source computed once
        mockedSource._compute_sources.assert_called_once()

        # Cache contains a single entry
        assert len(mockedSource._interp_geometry_cache) == 1

    def test_equal_configs_share_one_geometry_and_one_source(self, mockedSource):
        # Test that the geometry cache is used when the two different
        # `ScalarFieldConnector` objects with an identical (by value) configuration
        # each make a `get_indicator` call on the same target at the same `ndtime`.
        cfg_a = InterpolationConfig(kernel="idw", k_neighbors=4)
        cfg_b = InterpolationConfig(kernel="idw", k_neighbors=4)
        assert cfg_a is not cfg_b
        assert cfg_a == cfg_b

        targets = unit_targets()

        conn_a = ScalarFieldConnector(
            mockedSource, PassThroughOutput(), interpolation=cfg_a
        )
        conn_b = ScalarFieldConnector(
            mockedSource, PassThroughOutput(), interpolation=cfg_b
        )

        _ = conn_a.get_indicator(targets, ndtime=0.0)
        _ = conn_b.get_indicator(targets, ndtime=0.0)

        # Source computed once
        mockedSource._compute_sources.assert_called_once()

        # Cache contains a single entry
        assert len(mockedSource._interp_geometry_cache) == 1

    def test_differing_configs_get_their_own_geometry_and_same_source(
        self, mockedSource
    ):
        # Test that the geometry cache is not reused and is populated correctly when
        # two different `ScalarFieldConnector` objects with differing configurations
        # each make a `get_indicator` call on the same target at the same `ndtime`.
        cfg_idw = InterpolationConfig(kernel="idw", k_neighbors=4)
        cfg_gauss = InterpolationConfig(
            kernel="gaussian", k_neighbors=4, gaussian_sigma=0.5
        )

        targets = unit_targets()
        conn_idw = ScalarFieldConnector(
            mockedSource, PassThroughOutput(), interpolation=cfg_idw
        )
        conn_gauss = ScalarFieldConnector(
            mockedSource, PassThroughOutput(), interpolation=cfg_gauss
        )

        expected_keys = [
            conn_idw._construct_cache_key(targets),
            conn_gauss._construct_cache_key(targets),
        ]

        # No key collisions
        assert expected_keys[0] != expected_keys[1]

        _ = conn_idw.get_indicator(targets, ndtime=0.0)
        # First correct key placed in the cache
        assert expected_keys[0] in mockedSource._interp_geometry_cache

        _ = conn_gauss.get_indicator(targets, ndtime=0.0)
        # Second correct key placed in the cache
        assert expected_keys[1] in mockedSource._interp_geometry_cache

        # Previous key not erased
        assert len(mockedSource._interp_geometry_cache) == 2

        # Source computed once
        mockedSource._compute_sources.assert_called_once()

    def test_advancing_time_invalidates_cache(self, mockedSource, mockedFieldConnector):
        # Test that advancing time beyond `delta_t` invalidates and repopulates the
        # correct geometry cache and source cache entries.
        targets = unit_targets()
        _ = mockedFieldConnector.get_indicator(targets, ndtime=0.0)
        _ = mockedFieldConnector.get_indicator(targets, ndtime=10.0)

        # Same cache entry re-used
        assert len(mockedSource._interp_geometry_cache) == 1

        # Age change triggers geometry rebuild
        assert mockedFieldConnector._interp_geometry.call_count == 2

        # Age change triggers source recompute
        assert mockedSource._compute_sources.call_count == 2

    def test_different_targets_invalidates_cache(
        self, mockedSource, mockedFieldConnector
    ):
        # Test that calling `get_indicator` from a `ScalarFieldConnector` object after
        # changing the target creates a new cache key and correctly populates the
        # geometry cache without recomputing the corresponding source cache entry.
        expected_keys = []
        targets = unit_targets()
        expected_keys.append(mockedFieldConnector._construct_cache_key(targets))
        _ = mockedFieldConnector.get_indicator(targets, ndtime=0.0)

        # First correct key placed in the cache
        assert expected_keys[0] in mockedSource._interp_geometry_cache

        targets = -unit_targets()
        expected_keys.append(mockedFieldConnector._construct_cache_key(targets))
        _ = mockedFieldConnector.get_indicator(targets, ndtime=0.0)
        # Second correct key placed in the cache
        assert expected_keys[1] in mockedSource._interp_geometry_cache

        # No key collisions
        assert expected_keys[0] != expected_keys[1]

        # Previous key not erased
        assert len(mockedSource._interp_geometry_cache) == 2

        # Target change does not trigger source recompute
        mockedSource._compute_sources.assert_called_once()

        # Target change triggers geometry rebuild
        assert mockedFieldConnector._interp_geometry.call_count == 2
