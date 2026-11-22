"""Tests for ScalarFieldConnector and the connector factories.

These tests need no plate reconstruction. Small in-memory sources stand in
for gtrack producers. The tests cover:

  * the channel contract between a source and an output, checked when a
    connector is created;
  * the connector-level garbage collection interval;
  * the result cache, keyed on the identity of the target coordinate array;
  * the separation of membership from thickness in bounded outputs;
  * the interpolation geometry and source step that sibling connectors share;
  * the function space check in GplatesScalarFunction.

The regression test against a real reconstruction is in
tests/gplates_connectors/test_gplates_connectors.py.
"""

import gc
from unittest.mock import Mock

import firedrake as fd
import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import (
    HalfSpaceCoolingGeotherm,
    LinearGeotherm,
    GplatesScalarFunction,
    ScalarFieldConnector,
    InterpolationConfig,
    BoundedLinearGeotherm,
    BoundedLayerIndicator,
    MeshConfig,
    Source,
    GlobalLayerIndicator,
    ConnectorFactory,
    LithosphereConnectorFactory,
    PolygonConnectorFactory,
)
# Not part of the public gadopt.gplates API, so imported from the modules
# that define them.
from gadopt.gplates.outputs import MEMBERSHIP_FLOOR
from gadopt.gplates.interpolation import gather


# ---------------------------------------------------------------------------
# Lightweight mock source for validation contract tests (no I/O)
# ---------------------------------------------------------------------------

class _DummySource(Source):
    """Source that does no work — exists only so the validation contract
    can be exercised without paying for a real reconstruction load."""

    def __init__(self, provides, comm=MPI.COMM_WORLD):
        self._provides = frozenset(provides)
        self.comm = comm
        self._is_root = (comm.rank == 0)
        self.gplates_connector = _DummyGplates()

    @property
    def provides(self) -> frozenset[str]:
        return self._provides

    def _compute_sources(self, age):
        raise RuntimeError("The test must not call _compute_sources")


class _DummyGplates:
    """Minimal stand-in for pyGplatesConnector — only the time-conversion
    helpers and oldest_age are touched by the validation path."""
    oldest_age = 100.0
    delta_t = 1.0

    def ndtime2age(self, ndtime):
        return float(ndtime) * 100.0

    def age2ndtime(self, age):
        return age / 100.0


# ---------------------------------------------------------------------------
# Validation contracts (no reconstruction data)
# ---------------------------------------------------------------------------

class TestRequiresProvidesContract:
    """Validate required output channels against available source channels."""

    @pytest.mark.parametrize(
        "provides, output",
        [
            # A lithosphere source provides {"thickness","age"} (xyz is implicit
            # and never listed); GlobalLayerIndicator requires {"thickness"}.
            (frozenset({"thickness", "age"}), GlobalLayerIndicator()),
            # LinearGeotherm requires only {"thickness"}.
            (frozenset({"thickness", "age"}), LinearGeotherm()),
            # An unmasked thickness channel can supply a global indicator.
            (frozenset({"thickness", "membership"}),
             GlobalLayerIndicator(fallback_thickness_km=0.0)),
            (frozenset({"thickness", "membership"}), LinearGeotherm()),
        ],
    )
    def test_pairing_allowed(self, provides, output):
        ScalarFieldConnector(_DummySource(provides), output)  # must not raise

    def test_missing_channel_raises(self):
        # HalfSpaceCoolingGeotherm needs "age"; this source provides only
        # {"thickness","membership"}, so requires<=provides rejects the pairing.
        src = _DummySource({"thickness", "membership"})
        with pytest.raises(ValueError, match="age"):
            ScalarFieldConnector(src, HalfSpaceCoolingGeotherm())


class TestConnectorConstruction:
    def test_gc_collect_frequency_validated(self):
        src = _DummySource({"thickness"})
        with pytest.raises(ValueError, match="gc_collect_frequency"):
            ScalarFieldConnector(src, GlobalLayerIndicator(), gc_collect_frequency=0)

    def test_defaults_use_module_level_configs(self):
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, GlobalLayerIndicator())
        assert conn.mesh == MeshConfig()
        assert conn.interpolation == InterpolationConfig()


class TestGcCollectDefault:
    """Use the same garbage-collection interval on every construction path.

    The default interval is 10 calls, because a collection on every call is
    slow in a long time loop. Gtrack also collects the pyGPlates wrapper
    cycles internally.
    """

    def test_default_is_ten_direct(self):
        conn = ScalarFieldConnector(_DummySource({"thickness"}), GlobalLayerIndicator())
        assert conn.gc_collect_frequency == 10

    @pytest.mark.parametrize(
        "factory_class, source_factory",
        [
            (LithosphereConnectorFactory, lambda: _DataSource()),
            (PolygonConnectorFactory, lambda: _CapSource()),
        ],
        ids=["lithosphere", "polygon"],
    )
    def test_default_is_ten_factory(self, factory_class, source_factory):
        factory = factory_class()
        factory.source = source_factory()
        factory.create_indicator_output()
        assert factory.indicator.gc_collect_frequency == 10

    @pytest.mark.parametrize(
        "frequency, n_calls, expected_collects",
        [
            (3, 9, 3),      # collect every Nth call
            (None, 5, 0),   # None disables the connector-level collect entirely
            (1, 4, 4),      # collect on every call
        ],
    )
    def test_collect_cadence(self, monkeypatch, frequency, n_calls, expected_collects):
        mock_gc = Mock()
        with monkeypatch.context() as m:
            m.setattr("gadopt.gplates.connectors.gc.collect", mock_gc)
            conn = ScalarFieldConnector(
                _DataSource(), GlobalLayerIndicator(), gc_collect_frequency=frequency
            )
            target = _target_coords()
            # Distinct ages (spaced > delta_t=1.0, all <= oldest_age=100) so every
            # call is a cache miss and runs _compute (where the gc counter lives).
            for age in range(90, 90 - 10 * n_calls, -10):
                conn.get_indicator(target, conn.source.age2ndtime(float(age)))
        assert mock_gc.call_count == expected_collects


class TestResultCacheKey:
    """The result cache is keyed on the *identity* of the target_coords
    buffer (via a weakref), not on its contents. GplatesScalarFunction holds
    one mesh_coords array for its lifetime, so identity is a sound O(1) key
    and avoids hashing the whole coordinate buffer on every call.
    """

    @staticmethod
    def _conn():
        return ScalarFieldConnector(_DummySource({"thickness"}), GlobalLayerIndicator())

    def test_same_array_same_age_hits(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        conn._update_cache(10.0, arr, result)
        assert conn._check_cache(10.0, arr) is True

    def test_byte_equal_distinct_array_misses(self):
        # arr2 has the same bytes but a different identity. The cache must miss.
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        conn._update_cache(10.0, arr, result)
        arr2 = arr.copy()
        assert np.array_equal(arr, arr2)
        assert conn._check_cache(10.0, arr2) is False

    def test_dead_referent_misses_without_raising(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        conn._update_cache(10.0, arr, np.zeros(4))
        # Drop every binding to the cached array, then collect.
        del arr
        gc.collect()
        assert conn._cached_coords_ref() is None
        new_array = np.arange(12, dtype=float).reshape(4, 3)
        assert conn._check_cache(10.0, new_array) is False

    def test_age_guard_independent_of_identity(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        # delta_t is 1.0 (from _DummyGplates).
        conn._update_cache(10.0, arr, result)
        # An age change of `delta_t` produces a cache miss.
        assert conn._check_cache(11.0, arr) is False
        # Re-cache at the new age; a sub-delta_t age with the same buffer hits.
        conn._update_cache(11.0, arr, result)
        assert conn._check_cache(11.5, arr) is True


# ---------------------------------------------------------------------------
# Small in-memory sources for the interpolation tests
# ---------------------------------------------------------------------------

class _DataSource(Source):
    """Return a small fixed cloud for the real kNN interpolation path.

    Provides ``thickness`` (a genuine depth everywhere) plus ``membership``, so
    it represents a global source for ``GlobalLayerIndicator`` and geotherms.
    A separate subclass provides bounded ``masked_thickness`` data.
    """

    provides = frozenset({"thickness", "membership"})

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self._is_root = (comm.rank == 0)
        self.gplates_connector = _DummyGplates()
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(20, 3))
        xyz = 6.371e6 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        self._fixed = {
            "xyz": xyz,
            "thickness": rng.uniform(50.0, 200.0, size=20),
            "membership": rng.uniform(0.0, 1.0, size=20),
        }

    def _compute_sources(self, age):
        # Same cloud at every age (good enough for the geometry-sharing tests).
        return {k: v.copy() for k, v in self._fixed.items()}


def _fibonacci_directions(n, offset=0.0):
    """``n`` quasi-uniform unit vectors, offset so two clouds never coincide."""
    i = np.arange(n, dtype=float) + 0.5 + offset
    polar = np.arccos(np.clip(1.0 - 2.0 * i / n, -1.0, 1.0))
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar),
        np.sin(azimuth) * np.sin(polar),
        np.cos(polar),
    ])


class _CapSource(_DataSource):
    """A constant-depth polar cap on a quasi-uniform source cloud.

    The source provides `masked_thickness` and `membership` for bounded outputs.
    Inside the cap, membership is one and thickness is `DEPTH_KM`.
    Outside the cap, both channels are zero.

    Linear interpolation preserves `masked_thickness = DEPTH_KM * membership`.
    Therefore, division by nonzero membership must recover the constant depth
    for every interpolation width.
    """

    provides = frozenset({"masked_thickness", "membership"})
    CAP_RADIUS_RAD = 0.5
    DEPTH_KM = 200.0

    def __init__(self, n_points=20000, comm=MPI.COMM_WORLD):
        super().__init__(comm=comm)
        directions = _fibonacci_directions(n_points)
        inside = np.arccos(np.clip(directions[:, 2], -1.0, 1.0)) < self.CAP_RADIUS_RAD
        self._fixed = {
            "xyz": 6.371e6 * directions,
            "membership": inside.astype(float),
            "masked_thickness": self.DEPTH_KM * inside.astype(float),
        }


def _target_coords():
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(15, 3))
    return 2.0 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


class TestChannelNameGuard:
    """Reject outputs that interpret weighted thickness as physical thickness.

    A bounded source provides `masked_thickness`, not `thickness`.
    The connector checks this channel contract for factory and direct creation.
    """

    def test_global_indicator_on_bounded_source_raises_via_factory(self):
        factory = ConnectorFactory()
        factory.source = _CapSource()
        factory.output = GlobalLayerIndicator()
        with pytest.raises(ValueError, match="thickness"):
            _ = factory.indicator

    def test_bounded_indicator_on_bounded_source_passes(self):
        factory = ConnectorFactory()
        factory.source = _CapSource()
        factory.output = BoundedLayerIndicator(base_transition_width_km=50.0)
        assert factory.indicator is not None

    def test_bounded_geotherm_on_bounded_source_passes(self):
        """Pair the bounded geotherm with its required source channels."""
        assert ScalarFieldConnector(
            _CapSource(), BoundedLinearGeotherm()
        ) is not None

    def test_unmasked_geotherm_on_bounded_source_raises(self):
        with pytest.raises(ValueError, match="thickness"):
            ScalarFieldConnector(_CapSource(), LinearGeotherm())


class TestMembershipDoesNotContaminateDepth:
    """Keep regional membership separate from physical thickness.

    If one channel contains their product, interpolation changes both the
    recovered depth and the lateral weight. Separate channels allow the kernel
    width to control spatial smoothing without changing a constant depth.

    A variable depth field still depends on kernel width because interpolation
    smooths that field. These tests isolate membership contamination by using a
    constant physical depth.

    The target nodes differ from the source points. Exact source-point matches
    bypass interpolation and cannot expose this error.
    """

    # 2.5x apart, both comfortably above the ~0.025 rad seed spacing so
    # neither degenerates into nearest-neighbour lookup.
    SIGMAS = (0.03, 0.075)
    # Large enough that the Gaussian is not truncated by the kNN cutoff:
    # sqrt(400/pi) * 0.025 ~ 0.28 rad, about 3.7 sigma at the wider setting.
    K = 400

    @staticmethod
    def _targets(polar_angles, mesh, depth_km):
        """Directions at the given polar angles, at one depth below surface."""
        azimuth = np.linspace(0.0, 2.0 * np.pi, len(polar_angles), endpoint=False)
        directions = np.column_stack([
            np.cos(azimuth) * np.sin(polar_angles),
            np.sin(azimuth) * np.sin(polar_angles),
            np.cos(polar_angles),
        ])
        return directions * (mesh.r_outer - depth_km / mesh.depth_scale)

    def _field(self, src, sigma, polar_angles, depth_km, mesh):
        cfg = InterpolationConfig(
            kernel="gaussian",
            gaussian_width_rad=sigma,
            neighbor_count=self.K,
            max_source_separation_rad=1.0,
        )
        conn = ScalarFieldConnector(
            src, BoundedLayerIndicator(base_transition_width_km=20.0), mesh=mesh, interpolation=cfg
        )
        target = self._targets(polar_angles, mesh, depth_km)
        return conn.get_indicator(target, src.age2ndtime(0.0))

    def test_recovered_depth_is_uncontaminated_across_the_edge(self):
        """Compare the surface and deep fields across the cap boundary.

        At 190 km depth, each covered node remains above the 200 km base.
        Its value must equal its surface membership for every kernel width.
        A base computed from `membership * thickness` fails this comparison.
        """
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.30, 0.70, 40)  # cap edge is at 0.5 rad

        for sigma in self.SIGMAS:
            surface = self._field(src, sigma, polar, 0.0, mesh)
            above_base = self._field(src, sigma, polar, 190.0, mesh)

            # Where the region is present at all, the two agree to machine
            # precision — measured max difference 7e-16, not a loose bound.
            covered = surface > MEMBERSHIP_FLOOR
            # The compared nodes must lie on both sides of the cap edge, so
            # the comparison includes the transition and not only the interior.
            edge = _CapSource.CAP_RADIUS_RAD
            assert covered[polar < edge].any() and covered[polar > edge].any(), (
                "the compared nodes must include both sides of the cap edge"
            )
            np.testing.assert_allclose(
                above_base[covered], surface[covered], rtol=1e-12, atol=1e-14,
                err_msg=f"base depth contaminated by membership at sigma={sigma}",
            )

            # Below the floor, the correction rejects a depth recovered from
            # division by vanishing membership.
            np.testing.assert_array_equal(above_base[~covered], 0.0)
            assert np.all(surface[~covered] <= MEMBERSHIP_FLOOR)

    def test_interior_weight_is_uncontaminated(self):
        """Keep the interior weight independent of depth and kernel width."""
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.0, 0.15, 12)  # >= 4.7 sigma clear of the edge

        for sigma in self.SIGMAS:
            surface = self._field(src, sigma, polar, 0.0, mesh)
            np.testing.assert_allclose(surface, 1.0, atol=1e-6)

    def test_bandwidth_still_controls_the_transition_width(self):
        """Confirm that kernel width still controls the membership transition."""
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.30, 0.70, 40)

        narrow = self._field(src, self.SIGMAS[0], polar, 0.0, mesh)
        wide = self._field(src, self.SIGMAS[1], polar, 0.0, mesh)

        assert not np.allclose(narrow, wide, atol=1e-3)

        # Measure the angular span between membership values of 0.9 and 0.1.
        def span(field):
            inside = polar[field >= 0.9]
            outside = polar[field <= 0.1]
            return outside.min() - inside.max()

        assert span(wide) > span(narrow)


class TestGeometrySharing:
    """Share one source step and one geometry build between sibling connectors.

    Connectors on the same source, target array and interpolation config
    reuse the source arrays and the interpolation geometry at one age.
    """

    @pytest.fixture(autouse=True)
    def _patch_tree(self, monkeypatch):
        # These tests count geometry builds, so the tree itself is a Mock:
        # its query returns fixed neighbour distances and indices of the
        # right shape and the interpolation never runs for real.
        self.mock_tree = Mock()
        self.mock_tree.return_value.query.return_value = (
            np.ones((15, 2)), np.ones((15, 2), dtype=np.int32)
        )
        monkeypatch.setattr("gadopt.gplates.interpolation.cKDTree", self.mock_tree)

    def test_siblings_share_one_build_and_agree(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        # Two indicator connectors share one source, target array, and config.
        conn_a = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)
        conn_b = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)

        ndtime = src.age2ndtime(50.0)
        out_a = conn_a.get_indicator(target, ndtime)
        out_b = conn_b.get_indicator(target, ndtime)

        # Geometry built exactly once, shared by both.
        assert self.mock_tree.call_count == 1
        # Both GlobalLayerIndicators on the same source/geometry must agree byte-for-byte.
        np.testing.assert_array_equal(out_a, out_b)

    def test_indicator_and_geotherm_share_one_source_step(self):
        # An indicator and a geotherm on one source compute different fields
        # from the same source arrays. The second connector at the same age
        # must take those arrays from the source cache, so the source
        # computes them once.
        src = _DataSource()
        src._compute_sources = Mock(wraps=src._compute_sources)
        target = _target_coords()
        indicator = ScalarFieldConnector(src, GlobalLayerIndicator())
        geotherm = ScalarFieldConnector(src, LinearGeotherm())

        ndtime = src.age2ndtime(50.0)
        indicator.get_indicator(target, ndtime)
        geotherm.get_indicator(target, ndtime)

        assert src._compute_sources.call_count == 1

    def test_distinct_config_values_build_separately(self):
        src = _DataSource()
        target = _target_coords()
        conn_a = ScalarFieldConnector(
            src, GlobalLayerIndicator(), interpolation=InterpolationConfig(neighbor_count=10)
        )
        conn_b = ScalarFieldConnector(
            src, GlobalLayerIndicator(), interpolation=InterpolationConfig(neighbor_count=15)
        )

        ndtime = src.age2ndtime(50.0)
        conn_a.get_indicator(target, ndtime)
        conn_b.get_indicator(target, ndtime)

        # Different config values produce different cache keys.
        assert self.mock_tree.call_count == 2

    def test_age_advance_rebuilds_geometry(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)

        conn.get_indicator(target, src.age2ndtime(50.0))
        assert self.mock_tree.call_count == 1
        # Advancing by `delta_t` invalidates the source and geometry caches.
        conn.get_indicator(target, src.age2ndtime(20.0))
        assert self.mock_tree.call_count == 2

    def test_gather_matches_hand_computed(self):
        # The gathered channel must equal the explicit weighted sum.
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)

        source_dict = src.prepare(50.0)
        bundle = conn._interpolator.geometry(source_dict["xyz"], target)
        prop = source_dict["thickness"]
        gathered = gather(bundle, prop)

        idx = bundle["idx"]
        weights = bundle["weights"]
        expected = np.sum(weights * prop[idx], axis=1)
        expected[bundle["exact_match"]] = prop[idx[bundle["exact_match"], 0]]
        np.testing.assert_array_equal(gathered, expected)


# ---------------------------------------------------------------------------
# GplatesScalarFunction rejects non-scalar spaces
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_mesh():
    """Create a small mesh for the Firedrake connector tests."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=1, degree=1)
    return fd.ExtrudedMesh(mesh2d, layers=2, layer_height=0.5, extrusion_type="radial")


class TestGplatesScalarFunctionSpaceCheck:
    def test_vector_space_rejected(self, tiny_mesh):
        V = fd.VectorFunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, GlobalLayerIndicator())
        with pytest.raises(TypeError, match="scalar function space"):
            GplatesScalarFunction(V, indicator_connector=conn)

    def test_scalar_space_accepted(self, tiny_mesh):
        Q = fd.FunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, GlobalLayerIndicator())
        # Construction does not request data from the mock source.
        GplatesScalarFunction(Q, indicator_connector=conn)
