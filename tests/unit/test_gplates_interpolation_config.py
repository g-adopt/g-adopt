"""Tests for InterpolationConfig and the interpolation-geometry cache key.

Everything here constructs its own inputs: no reconstruction data, no
downloaded files, no Firedrake solve. The source is a fixed point cloud on the
unit sphere carrying one property, and the output strategy hands that property
straight back, so the numbers the connector returns are exactly what the kNN
interpolation produced.

The property under test is that the geometry cache is keyed on the
interpolation config *by value*. Two connectors holding equal-valued but
distinct configs must share one cKDTree build; two connectors whose configs
differ must not, and must return different fields.
"""

import dataclasses

import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import InterpolationConfig, OutputStrategy, ScalarFieldConnector, Source


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


class FixedCloudSource(Source):
    """A source whose cloud and property are the same at every age.

    The cloud is deterministic (seeded RNG) and lies on the unit sphere; the
    property is a per-point scalar. Age-independence is deliberate — it keeps
    the per-age source cache out of the way so the geometry cache is the only
    thing under test.

    Args:
        n_points: number of source points.
        seed: RNG seed for the cloud and the property values.
    """

    provides = frozenset({"thickness"})

    def __init__(self, n_points=60, seed=0):
        rng = np.random.default_rng(seed)
        xyz = rng.normal(size=(n_points, 3))
        self.xyz = xyz / np.linalg.norm(xyz, axis=1)[:, np.newaxis]
        self.thickness = rng.uniform(50.0, 200.0, size=n_points)
        self.gplates_connector = FakeGplatesConnector()
        self.comm = MPI.COMM_SELF
        self._is_root = True
        self.compute_calls = 0

    def _compute_sources(self, age):
        self.compute_calls += 1
        return {"xyz": self.xyz, "thickness": self.thickness}


class PassThroughOutput(OutputStrategy):
    """Return the interpolated thickness unchanged."""

    requires = frozenset({"thickness"})

    def compute(self, interpolated, r_target, too_far, mesh):
        return interpolated["thickness"]


def unit_targets(n_points=40, seed=1):
    """Build target coordinates that avoid the source seeds.

    A target sitting exactly on a seed is snapped to that seed's raw value by
    the exact-match branch, which reports a perfect answer independent of the
    kernel, so such points would hide any kernel difference.

    Args:
        n_points: number of target nodes.
        seed: RNG seed, chosen distinct from the source cloud's.

    Returns:
        An ``(n_points, 3)`` array of points on the unit sphere.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_points, 3))
    return xyz / np.linalg.norm(xyz, axis=1)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Frozen dataclass behaviour
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_rejects_field_assignment(self):
        cfg = InterpolationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.kernel = "gaussian"
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.distance_threshold = 0.5

    def test_is_hashable(self):
        # Guards the "every field must stay hashable" clause in the docstring:
        # a field holding a list, dict or set would raise here, and would raise
        # again on every geometry-cache insertion.
        hash(InterpolationConfig())
        hash(InterpolationConfig(kernel="gaussian", k_neighbors=4))

    def test_equal_values_hash_equal(self):
        assert hash(InterpolationConfig()) == hash(InterpolationConfig())
        assert InterpolationConfig() == InterpolationConfig()


# ---------------------------------------------------------------------------
# Geometry cache keyed by value
# ---------------------------------------------------------------------------

class TestGeometryCacheKey:
    def test_equal_configs_share_one_geometry(self):
        # Two distinct config objects holding identical values. Under the old
        # id()-based key this produces two cache entries and two cKDTree
        # builds; under a by-value key, one.
        cfg_a = InterpolationConfig(kernel="idw", k_neighbors=4)
        cfg_b = InterpolationConfig(kernel="idw", k_neighbors=4)
        assert cfg_a is not cfg_b
        assert cfg_a == cfg_b

        source = FixedCloudSource()
        targets = unit_targets()
        conn_a = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=cfg_a
        )
        conn_b = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=cfg_b
        )

        field_a = conn_a.get_indicator(targets, ndtime=0.0)
        field_b = conn_b.get_indicator(targets, ndtime=0.0)

        assert len(source._interp_geometry_cache) == 1
        np.testing.assert_array_equal(field_a, field_b)

    def test_differing_configs_get_their_own_geometry(self):
        # The half that catches the original symptom: two configs that differ
        # must not share a build, and the fields they produce must differ.
        cfg_idw = InterpolationConfig(kernel="idw", k_neighbors=4)
        cfg_gauss = InterpolationConfig(
            kernel="gaussian", k_neighbors=4, gaussian_sigma=0.5
        )

        source = FixedCloudSource()
        targets = unit_targets()
        conn_idw = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=cfg_idw
        )
        conn_gauss = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=cfg_gauss
        )

        field_idw = conn_idw.get_indicator(targets, ndtime=0.0)
        field_gauss = conn_gauss.get_indicator(targets, ndtime=0.0)

        assert len(source._interp_geometry_cache) == 2
        assert np.max(np.abs(field_idw - field_gauss)) > 1e-8

    def test_one_source_advance_for_both_consumers(self):
        # The geometry cache sits on the source, so sharing it must not
        # disturb the per-age source cache: one advance, two consumers.
        source = FixedCloudSource()
        targets = unit_targets()
        conn_a = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=InterpolationConfig()
        )
        conn_b = ScalarFieldConnector(
            source, PassThroughOutput(), interpolation=InterpolationConfig()
        )

        conn_a.get_indicator(targets, ndtime=0.0)
        conn_b.get_indicator(targets, ndtime=0.0)

        assert source.compute_calls == 1
