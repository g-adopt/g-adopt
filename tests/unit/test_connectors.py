"""Tests for ScalarFieldConnector and its factory functions.

Three groups:

  * **Validation contracts** — no reconstruction data; verify that bad
    Source/Output pairings raise at construction, that bad function spaces
    are caught up front by GplatesScalarFunction, and that
    InterpolationConfig rejects invalid kernels.

  * **Connector regression** — for each of the four factory pairings
    (lithosphere indicator, lithosphere geotherm, polygon indicator, polygon
    geotherm) plus the GplatesScalarFunction wrapper, evaluate on a small
    extruded cubed-sphere mesh at a fixed series of geological ages, record
    volume/surface integrals, and compare against a pickled reference. Built
    once against the Muller 2022 SE v1.2 reconstruction.

  * **Shared-source consistency** — confirms that a single LithosphereSource
    can drive both an indicator and a geotherm connector simultaneously
    without producing duplicate state.
"""

import gc
import sys
from pathlib import Path
import pickle

import firedrake as fd
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import (
    HalfSpaceCoolingGeotherm,
    LinearGeotherm,
    GplatesScalarFunction,
    ScalarFieldConnector,
    InterpolationConfig,
    SphericalKNNInterpolator,
    BoundedLinearGeotherm,
    BoundedLayerIndicator,
    MeshConfig,
    PlateModelFiles,
    PointCloudSource,
    Source,
    GlobalLayerIndicator,
    ensure_reconstruction,
    ConnectorFactory,
    LithosphereConnectorFactory,
    PolygonConnectorFactory,
    pyGplatesConnector,
)
from gtrack import (
    LithosphereCloudConfig,
    LithosphereCloudSource,
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
)
from gtrack.config import TracerConfig
# Internal to the deblend rather than public API, so imported from the module
# it lives in rather than widening gadopt.gplates.
from gadopt.gplates.outputs import MEMBERSHIP_FLOOR


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
DATA_DIR = Path(__file__).resolve().parent / "data"

# The regression fixture's configuration lives in one place — data/
# regression_config.py — imported here and by the generator that builds the
# pickle, so the two cannot silently disagree on the numbers the reference
# values are computed from.
sys.path.insert(0, str(DATA_DIR))
from regression_config import (  # noqa: E402
    OLDEST_AGE,
    LITH_TRACKER_POINT_COUNT,
    POLYGON_BACKGROUND_POINT_COUNT,
    POLYGON_SCALAR_INPUT_POINT_COUNT,
    TEST_AGES,
    REGRESSION_LAYER_HEIGHTS,
)


def _require_data():
    if not (GPLATES_GLOBAL / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
        pytest.skip("Muller 2022 SE reconstruction not downloaded; run `make data`.")
    if not CONTINENTAL_DATA.exists():
        pytest.skip(f"Continental thickness data missing at {CONTINENTAL_DATA}.")


def _require_craton():
    if not CRATON_SHAPEFILE.exists():
        pytest.skip(f"Craton shapefile missing at {CRATON_SHAPEFILE}.")


def half_space_cooling(age_myr):
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def _load_continental_data():
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]]), values


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
        assert isinstance(conn.mesh, MeshConfig)
        assert isinstance(conn.interpolation, InterpolationConfig)


class TestGcCollectDefault:
    """Use the same garbage-collection interval on every construction path.

    Collection on every call added approximately 130 seconds to a 5000-step
    loop. Gtrack also collects the pyGPlates wrapper cycles internally.
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
        factory.create_indicator()
        assert factory.indicator.gc_collect_frequency == 10

    def _drive(self, monkeypatch, frequency, n_calls):
        calls = {"n": 0}
        monkeypatch.setattr(
            "gadopt.gplates.connectors.gc.collect",
            lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
        )
        conn = ScalarFieldConnector(
            _DataSource(), GlobalLayerIndicator(), gc_collect_frequency=frequency
        )
        target = _target_coords()
        # Distinct ages (spaced > delta_t=1.0, all <= oldest_age=100) so every
        # call is a cache miss and runs _compute (where the gc counter lives).
        for age in range(90, 90 - 10 * n_calls, -10):
            conn.get_indicator(target, conn.source.age2ndtime(float(age)))
        return calls["n"]

    @pytest.mark.parametrize(
        "frequency, n_calls, expected_collects",
        [
            (3, 9, 3),      # collect every Nth call
            (None, 5, 0),   # None disables the connector-level collect entirely
            (1, 4, 4),      # collect on every call
        ],
    )
    def test_collect_cadence(self, monkeypatch, frequency, n_calls, expected_collects):
        assert self._drive(monkeypatch, frequency, n_calls) == expected_collects


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
# Shared interpolation geometry (P10)
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

    The proportional channels cannot detect confusion between membership and
    thickness. `test_recovers_a_laterally_varying_depth_exactly` uses
    non-proportional channels to test that distinction.
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


class _CountingCKDTree:
    """Wraps cKDTree, counting constructions so a test can assert how many
    interpolation geometries were built."""

    count = 0

    def __init__(self, *args, **kwargs):
        type(self).count += 1
        from scipy.spatial import cKDTree as _real
        self._tree = _real(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self._tree.query(*args, **kwargs)


def _target_coords():
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(15, 3))
    return 2.0 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


class _MaskedSource(_CapSource):
    """Stands in for a polygon-style source: provides ``masked_thickness`` and
    ``membership`` rather than a plain ``thickness`` channel."""


class TestChannelNameGuard:
    """Reject outputs that interpret weighted thickness as physical thickness.

    A bounded source provides `masked_thickness`, not `thickness`.
    The connector checks this channel contract for factory and direct creation.
    """

    def test_global_indicator_on_bounded_source_raises_via_factory(self):
        factory = ConnectorFactory()
        factory.source = _MaskedSource()
        factory.output = GlobalLayerIndicator()
        with pytest.raises(ValueError, match="thickness"):
            _ = factory.indicator

    def test_bounded_indicator_on_bounded_source_passes(self):
        factory = ConnectorFactory()
        factory.source = _MaskedSource()
        factory.output = BoundedLayerIndicator(base_transition_width_km=50.0)
        assert factory.indicator is not None

    def test_bounded_geotherm_on_bounded_source_passes(self):
        """Pair the bounded geotherm with its required source channels."""
        assert ScalarFieldConnector(
            _MaskedSource(), BoundedLinearGeotherm()
        ) is not None

    def test_unmasked_geotherm_on_bounded_source_raises(self):
        with pytest.raises(ValueError, match="thickness"):
            ScalarFieldConnector(_MaskedSource(), LinearGeotherm())


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
            assert covered.sum() > 10, "sweep must straddle the boundary"
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
    """Reuse interpolation geometry for one source, target array, and config."""

    @pytest.fixture(autouse=True)
    def _patch_tree(self, monkeypatch):
        _CountingCKDTree.count = 0
        monkeypatch.setattr(
            "gadopt.gplates.interpolation.cKDTree", _CountingCKDTree
        )

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
        assert _CountingCKDTree.count == 1
        # Both GlobalLayerIndicators on the same source/geometry must agree byte-for-byte.
        np.testing.assert_array_equal(out_a, out_b)

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
        assert _CountingCKDTree.count == 2

    def test_age_advance_rebuilds_geometry(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)

        conn.get_indicator(target, src.age2ndtime(50.0))
        assert _CountingCKDTree.count == 1
        # Advancing by `delta_t` invalidates the source and geometry caches.
        conn.get_indicator(target, src.age2ndtime(20.0))
        assert _CountingCKDTree.count == 2

    def test_gather_matches_hand_computed(self):
        # The gathered channel must equal the explicit weighted sum.
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, GlobalLayerIndicator(), interpolation=cfg)

        source_dict = src.prepare(50.0)
        bundle = conn._interpolator.geometry(source_dict["xyz"], target)
        prop = source_dict["thickness"]
        gathered = SphericalKNNInterpolator.gather(bundle, prop)

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


# ---------------------------------------------------------------------------
# Reconstruction-backed fixtures for regression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plate_model():
    _require_data()
    files = ensure_reconstruction("Muller 2022 SE v1.2", GPLATES_GLOBAL)
    return pyGplatesConnector(
        rotation_filenames=files["rotation_filenames"],
        topology_filenames=files["topology_filenames"],
        oldest_age=OLDEST_AGE,
    )


@pytest.fixture(scope="module")
def plate_files():
    _require_data()
    files = ensure_reconstruction("Muller 2022 SE v1.2", GPLATES_GLOBAL)
    return PlateModelFiles(
        continental_polygons=files.get("continental_polygons"),
        static_polygons=files.get("static_polygons"),
    )


def _make_lith_source(plate_model, plate_files):
    """Wrap a gtrack LithosphereCloudSource in a gadopt PointCloudSource."""
    producer = LithosphereCloudSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        continental_polygons=plate_files.continental_polygons,
        static_polygons=plate_files.static_polygons,
        continental_data=_load_continental_data(),
        oceanic_thickness_from_age=half_space_cooling,
        plate_model_max_age_ma=OLDEST_AGE,
        config=LithosphereCloudConfig(
            tracer=TracerConfig(tracker_point_count=LITH_TRACKER_POINT_COUNT),
        ),
    )
    return PointCloudSource(producer, plate_model)


@pytest.fixture(scope="module")
def lith_source(plate_model, plate_files):
    return _make_lith_source(plate_model, plate_files)


@pytest.fixture(scope="module")
def poly_source(plate_model, plate_files):
    _require_craton()
    producer = PolygonIndicatorSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        polygons=str(CRATON_SHAPEFILE),
        static_polygons=plate_files.static_polygons,
        thickness_data=200.0,
        # Scalar thickness -> scalar_input_point_count sizes the seeds; pin both so the
        # regression matches the old single-n_points seeding.
        config=PolygonIndicatorConfig(
            background_point_count=POLYGON_BACKGROUND_POINT_COUNT,
            scalar_input_point_count=POLYGON_SCALAR_INPUT_POINT_COUNT,
        ),
    )
    return PointCloudSource(producer, plate_model)


@pytest.fixture(scope="module")
def regression_mesh():
    """Bigger than `tiny_mesh` to give the indicator field somewhere to
    show structure, but still cheap. Refinement level 2 + 6 graded layers ~
    2700 DoFs at CG1."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=2, degree=1)
    mesh = fd.ExtrudedMesh(
        mesh2d,
        layers=len(REGRESSION_LAYER_HEIGHTS),
        layer_height=np.array(REGRESSION_LAYER_HEIGHTS, dtype=float),
        extrusion_type="radial",
    )
    mesh.cartesian = False
    return mesh


@pytest.fixture(scope="module")
def Q(regression_mesh):
    return fd.FunctionSpace(regression_mesh, "CG", 1)


# ---------------------------------------------------------------------------
# Regression: four factory pairings + GplatesScalarFunction wrapper
# ---------------------------------------------------------------------------

def _reduced_quantities(values, mesh):
    """Volume and surface integrals of a scalar field on the regression mesh,
    plus mean / std / min / max of the DoF values."""
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    f.dat.data_with_halos[:] = values
    return {
        "volume": float(fd.assemble(f * fd.dx)),
        "surface": float(fd.assemble(f * fd.ds_t)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _evaluate_connectors_lockstep(connectors_by_name, mesh, Q, ages):
    """Drive all connectors in lockstep through the same age series.

    Connectors sharing a forward-only source (LithosphereSource) must
    advance together: once the tracker passes an age, it can't be revisited
    by any sibling connector. The lockstep walk mirrors a real time loop.
    """
    sfs = {
        name: GplatesScalarFunction(Q, indicator_connector=c, name=name)
        for name, c in connectors_by_name.items()
    }
    out = {name: {} for name in connectors_by_name}
    sample = next(iter(connectors_by_name.values()))
    for age in ages:
        ndtime = sample.source.age2ndtime(float(age))
        for name in connectors_by_name:
            sfs[name].update_plate_reconstruction(ndtime)
            out[name][age] = _reduced_quantities(
                sfs[name].dat.data_ro_with_halos.copy(), mesh
            )
    return out


def _load_reference():
    ref_path = DATA_DIR / "test_connectors.pkl"
    if not ref_path.exists():
        pytest.skip(
            f"Reference fixture missing: {ref_path}. "
            "Generate via tests/unit/data/generate_expected_connectors.py."
        )
    with open(ref_path, "rb") as f:
        return pickle.load(f)


def _check_reduced(observed, expected, label):
    # atol floors the round-off on quantities that are analytically zero
    # (for example a geotherm's surface integral) so they aren't held to rtol.
    for key, expected_value in expected.items():
        np.testing.assert_allclose(
            observed[key], expected_value, rtol=1e-3, atol=1e-9,
            err_msg=f"{label}: {key}",
        )


class TestConnectorRegression:
    """Drive the four factory connectors in lockstep (siblings sharing a
    Source advance together) and assert volume/surface integrals match the
    pickled reference."""

    def test_lithosphere_pair(self, lith_source, regression_mesh, Q):
        ref = _load_reference()
        factory = LithosphereConnectorFactory()
        factory.source = lith_source
        # Must match generate_expected_connectors.py. Age sensitivity comes
        # from the radial grading of `regression_mesh`: with the plain
        # one-sided step, nodes have to sit inside the lithosphere for a
        # moving base depth to change the reduced integrals at all.
        factory.create_indicator()
        factory.create_geotherm()
        observed = _evaluate_connectors_lockstep({
            "lith_indicator": factory.indicator,
            "lith_geotherm": factory.geotherm,
        }, regression_mesh, Q, TEST_AGES)
        for name in ("lith_indicator", "lith_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_polygon_pair(self, poly_source, regression_mesh, Q):
        ref = _load_reference()
        factory = PolygonConnectorFactory()
        factory.source = poly_source
        factory.create_indicator()
        factory.create_geotherm()
        observed = _evaluate_connectors_lockstep({
            "polygon_indicator": factory.indicator,
            "polygon_geotherm": factory.geotherm,
        }, regression_mesh, Q, TEST_AGES)
        for name in ("polygon_indicator", "polygon_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_reference_is_age_sensitive(self):
        """Require the fixture to vary with reconstruction age.

        On a uniform four-layer mesh, no node samples the lithosphere interior.
        The indicator is then one at the surface and zero at the next node.
        Its volume remains 6.856154 at all ages. The graded
        ``REGRESSION_LAYER_HEIGHTS`` restores sensitivity to base-depth changes.

        Only volume, mean, and standard deviation contain the base-depth signal.
        The surface integral, minimum, and maximum are constant by construction.
        The constant surface contribution cancels from each age difference.
        """
        ref = _load_reference()
        for name in ("lith_indicator", "polygon_indicator"):
            volumes = [ref[name][age]["volume"] for age in TEST_AGES]
            assert len(set(volumes)) == len(TEST_AGES), (
                f"{name} has repeated volumes across {TEST_AGES}: {volumes}. "
                "The reference does not depend on the reconstruction, so the "
                "regression cannot detect a change in it."
            )

    def test_geotherm_surface_is_zero(self):
        """Keep the global lithosphere geotherm at zero on the surface.

        The background source points cover the complete sphere. Therefore, a
        nonzero surface integral indicates a gap in the source range.

        The bounded geotherm is excluded because it assigns mantle temperature
        outside its region, including the exterior surface.
        """
        ref = _load_reference()
        for age in TEST_AGES:
            name = "lith_geotherm"
            assert abs(ref[name][age]["surface"]) < 1e-9, (
                f"{name} at {age} Ma has surface integral "
                f"{ref[name][age]['surface']:.3e}, not ~0 — some surface "
                "node was flagged outside_source_range."
            )


class TestSharedSourceConsistency:
    """Two connectors holding the same lithosphere source must produce the
    same field as a connector built standalone (same source instance, only a
    different consumer).

    Uses a fresh source (class-scoped) so the forward-only tracker hasn't
    been walked past the test age by sibling regression tests.
    """

    @pytest.fixture(scope="class")
    def fresh_lith_source(self, plate_model, plate_files):
        return _make_lith_source(plate_model, plate_files)

    def test_two_indicators_sharing_source_agree(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Two factories create independent indicators that share one source.
        # The shared source and target nodes must produce identical values.
        factory_a = LithosphereConnectorFactory()
        factory_a.source = fresh_lith_source
        factory_a.create_indicator()
        factory_b = LithosphereConnectorFactory()
        factory_b.source = fresh_lith_source
        factory_b.create_indicator()
        ind_a = factory_a.indicator
        ind_b = factory_b.indicator

        sf_a = GplatesScalarFunction(Q, indicator_connector=ind_a, name="ind_a")
        sf_b = GplatesScalarFunction(Q, indicator_connector=ind_b, name="ind_b")

        ndtime = fresh_lith_source.age2ndtime(50.0)
        sf_a.update_plate_reconstruction(ndtime)
        sf_b.update_plate_reconstruction(ndtime)

        # DoF arrays must match to machine precision — the kNN interpolation
        # is deterministic given identical inputs.
        np.testing.assert_array_equal(
            sf_a.dat.data_ro_with_halos,
            sf_b.dat.data_ro_with_halos,
        )

    def test_indicator_and_geotherm_share_one_tracker_step(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Pair an indicator with a geotherm on the same source. Both produce
        # different scalar fields (quintic step vs. erf geotherm), but they must
        # use the same source arrays. The second call must use the age cache.
        factory = LithosphereConnectorFactory()
        factory.source = fresh_lith_source
        factory.create_indicator()
        factory.create_geotherm()
        ind = factory.indicator
        geo = factory.geotherm

        sf_ind = GplatesScalarFunction(Q, indicator_connector=ind, name="ind")
        sf_geo = GplatesScalarFunction(Q, indicator_connector=geo, name="geo")

        # Use a fresh age that lies forward of the previous test's stop
        # (we share the class-scoped source, which is forward-only).
        ndtime = fresh_lith_source.age2ndtime(40.0)
        sf_ind.update_plate_reconstruction(ndtime)
        d_first = fresh_lith_source._cached_dict
        sf_geo.update_plate_reconstruction(ndtime)
        d_second = fresh_lith_source._cached_dict

        # The geotherm call must use the source cache.
        assert d_first is d_second
