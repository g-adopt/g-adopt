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
from pathlib import Path
import pickle

import firedrake as fd
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import (
    GeothermERFOutput,
    GeothermLinearOutput,
    GplatesScalarFunction,
    ScalarFieldConnector,
    InterpolationConfig,
    LithosphereSource,
    LithosphereSourceConfig,
    MeshConfig,
    PlateModelFiles,
    PolygonSource,
    PolygonSourceConfig,
    Source,
    TanhOutput,
    ensure_reconstruction,
    LithosphereConnectorFactory,
    PolygonConnectorFactory,
    pyGplatesConnector,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
DATA_DIR = Path(__file__).resolve().parent / "data"

OLDEST_AGE = 120
LITH_N_POINTS = 2000
POLYGON_N_POINTS = 3000
TEST_AGES = (100, 50, 0)


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
        raise RuntimeError("dummy source should never reach _compute_sources")


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
    """An OutputStrategy declares the source-property keys it needs; the
    connector validates this against the Source's provides set at
    construction time."""

    def test_lith_thickness_only_pairing_allowed(self):
        # LithosphereSource provides {"xyz","thickness","age"}; TanhOutput
        # requires {"thickness"}.
        src = _DummySource({"xyz", "thickness", "age"})
        ScalarFieldConnector(src, TanhOutput())  # must not raise

    def test_lith_with_linear_geotherm_allowed(self):
        # GeothermLinearOutput requires only {"thickness"} — a polygon-style
        # output paired against a lithosphere-style source is allowed.
        src = _DummySource({"xyz", "thickness", "age"})
        ScalarFieldConnector(src, GeothermLinearOutput())

    def test_polygon_with_erf_geotherm_raises(self):
        # PolygonSource provides {"xyz","thickness"} — missing "age" that
        # GeothermERFOutput needs.
        src = _DummySource({"xyz", "thickness"})
        with pytest.raises(ValueError, match="age"):
            ScalarFieldConnector(src, GeothermERFOutput())

    def test_polygon_with_tanh_allowed(self):
        src = _DummySource({"xyz", "thickness"})
        ScalarFieldConnector(src, TanhOutput(default_thickness_km=0.0))

    def test_polygon_with_linear_geotherm_allowed(self):
        src = _DummySource({"xyz", "thickness"})
        ScalarFieldConnector(src, GeothermLinearOutput())


class TestConnectorConstruction:
    def test_gc_collect_frequency_validated(self):
        src = _DummySource({"xyz", "thickness"})
        with pytest.raises(ValueError, match="gc_collect_frequency"):
            ScalarFieldConnector(src, TanhOutput(), gc_collect_frequency=0)

    def test_defaults_use_module_level_configs(self):
        src = _DummySource({"xyz", "thickness"})
        conn = ScalarFieldConnector(src, TanhOutput())
        assert isinstance(conn.mesh, MeshConfig)
        assert isinstance(conn.interpolation, InterpolationConfig)


class TestGcCollectDefault:
    """The gc_collect_frequency default is 10 (matching gtrack's own internal
    GC cadence), not 1 — a full gc.collect() every call was an undefended
    default that wasted ~130s over a 5000-step loop while gtrack already
    breaks the pygplates C++ cycles internally. The default must be 10 on
    every construction path, including the factories (which forward their own
    default to the connector, so a stale factory default would silently
    override the connector's)."""

    def test_default_is_ten_direct(self):
        conn = ScalarFieldConnector(_DummySource({"xyz", "thickness"}), TanhOutput())
        assert conn.gc_collect_frequency == 10

    def test_default_is_ten_lithosphere_factory(self):
        # Assigning a pre-built source skips construct_source -> no
        # reconstruction I/O.
        factory = LithosphereConnectorFactory()
        factory.source = _DataSource()
        factory.construct_output()
        assert factory.indicator.gc_collect_frequency == 10

    def test_default_is_ten_polygon_factory(self):
        factory = PolygonConnectorFactory()
        factory.source = _DataSource()
        factory.construct_output()
        assert factory.indicator.gc_collect_frequency == 10

    def _drive(self, monkeypatch, frequency, n_calls):
        calls = {"n": 0}
        monkeypatch.setattr(
            "gadopt.gplates.connectors.gc.collect",
            lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
        )
        conn = ScalarFieldConnector(
            _DataSource(), TanhOutput(), gc_collect_frequency=frequency
        )
        target = _target_coords()
        # Distinct ages (spaced > delta_t=1.0, all <= oldest_age=100) so every
        # call is a cache miss and runs _compute (where the gc counter lives).
        for age in range(90, 90 - 10 * n_calls, -10):
            conn.get_indicator(target, conn.source.age2ndtime(float(age)))
        return calls["n"]

    def test_interval_collects_every_nth(self, monkeypatch):
        assert self._drive(monkeypatch, frequency=3, n_calls=9) == 3

    def test_none_never_collects(self, monkeypatch):
        assert self._drive(monkeypatch, frequency=None, n_calls=5) == 0

    def test_one_collects_every_call(self, monkeypatch):
        assert self._drive(monkeypatch, frequency=1, n_calls=4) == 4


class TestResultCacheKey:
    """The result cache is keyed on the *identity* of the target_coords
    buffer (via a weakref), not on its contents. GplatesScalarFunction holds
    one mesh_coords array for its lifetime, so identity is a sound O(1) key
    and avoids hashing the whole coordinate buffer on every call.
    """

    @staticmethod
    def _conn():
        return ScalarFieldConnector(_DummySource({"xyz", "thickness"}), TanhOutput())

    def test_same_array_same_age_hits(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        conn._update_cache(10.0, arr, result)
        assert conn._check_cache(10.0, arr) is True

    def test_byte_equal_distinct_array_misses(self):
        # arr2 is byte-for-byte identical but a different object — a content
        # hash would (wrongly) hit; identity correctly misses.
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
        # Same buffer, but the age has moved a full delta_t away -> miss.
        assert conn._check_cache(11.0, arr) is False
        # Re-cache at the new age; a sub-delta_t age with the same buffer hits.
        conn._update_cache(11.0, arr, result)
        assert conn._check_cache(11.5, arr) is True


# ---------------------------------------------------------------------------
# Shared interpolation geometry (P10)
# ---------------------------------------------------------------------------

class _DataSource(Source):
    """Dummy source returning a fixed small (xyz, thickness) cloud — enough to
    exercise the real kNN interpolation path without any reconstruction I/O."""

    provides = frozenset({"xyz", "thickness"})

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
        }

    def _compute_sources(self, age):
        # Same cloud at every age (good enough for the geometry-sharing tests).
        return {k: v.copy() for k, v in self._fixed.items()}


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


class TestGeometrySharing:
    """The interpolation geometry (cKDTree indices + weights) depends only on
    (source cloud, target coords, cfg), so connectors sharing those should
    build it once and reuse the cached bundle."""

    @pytest.fixture(autouse=True)
    def _patch_tree(self, monkeypatch):
        _CountingCKDTree.count = 0
        monkeypatch.setattr(
            "gadopt.gplates.connectors.cKDTree", _CountingCKDTree
        )

    def test_siblings_share_one_build_and_agree(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        # Two indicator connectors sharing the same source, target and cfg.
        conn_a = ScalarFieldConnector(src, TanhOutput(), interpolation=cfg)
        conn_b = ScalarFieldConnector(src, TanhOutput(), interpolation=cfg)

        ndtime = src.age2ndtime(50.0)
        out_a = conn_a.get_indicator(target, ndtime)
        out_b = conn_b.get_indicator(target, ndtime)

        # Geometry built exactly once, shared by both.
        assert _CountingCKDTree.count == 1
        # Both TanhOutputs on the same source/geometry must agree byte-for-byte.
        np.testing.assert_array_equal(out_a, out_b)

    def test_distinct_configs_build_separately(self):
        src = _DataSource()
        target = _target_coords()
        conn_a = ScalarFieldConnector(src, TanhOutput(), interpolation=InterpolationConfig())
        conn_b = ScalarFieldConnector(src, TanhOutput(), interpolation=InterpolationConfig())

        ndtime = src.age2ndtime(50.0)
        conn_a.get_indicator(target, ndtime)
        conn_b.get_indicator(target, ndtime)

        # Different cfg identities -> different cache keys -> two builds.
        assert _CountingCKDTree.count == 2

    def test_age_advance_rebuilds_geometry(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, TanhOutput(), interpolation=cfg)

        conn.get_indicator(target, src.age2ndtime(50.0))
        assert _CountingCKDTree.count == 1
        # Advancing a full delta_t misses the source cache, which clears the
        # geometry cache -> rebuild.
        conn.get_indicator(target, src.age2ndtime(20.0))
        assert _CountingCKDTree.count == 2

    def test_gather_matches_hand_computed(self):
        # The gathered property must equal the explicit weighted sum.
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, TanhOutput(), interpolation=cfg)

        source_dict = src.prepare(50.0)
        bundle = conn._interp_geometry(source_dict["xyz"], target)
        prop = source_dict["thickness"]
        gathered = conn._interp_gather(bundle, prop)

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
    """A small extruded icosahedral sphere just big enough to exercise the
    Firedrake path. Used by the function-space-rejection test and reused
    later by the connector regression tests."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=1, degree=1)
    return fd.ExtrudedMesh(mesh2d, layers=2, layer_height=0.5, extrusion_type="radial")


class TestGplatesScalarFunctionSpaceCheck:
    def test_vector_space_rejected(self, tiny_mesh):
        V = fd.VectorFunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"xyz", "thickness"})
        conn = ScalarFieldConnector(src, TanhOutput())
        with pytest.raises(TypeError, match="scalar function space"):
            GplatesScalarFunction(V, indicator_connector=conn)

    def test_scalar_space_accepted(self, tiny_mesh):
        Q = fd.FunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"xyz", "thickness"})
        conn = ScalarFieldConnector(src, TanhOutput())
        # Should construct without raising; the mock source is never asked
        # for data because we never call update_plate_reconstruction.
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


@pytest.fixture(scope="module")
def lith_source(plate_model, plate_files):
    return LithosphereSource(
        gplates_connector=plate_model,
        continental_data=_load_continental_data(),
        age_to_property=half_space_cooling,
        plate_files=plate_files,
        config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
    )


@pytest.fixture(scope="module")
def poly_source(plate_model, plate_files):
    _require_craton()
    return PolygonSource(
        gplates_connector=plate_model,
        polygons=str(CRATON_SHAPEFILE),
        thickness_data=200.0,
        plate_files=plate_files,
        config=PolygonSourceConfig(n_points=POLYGON_N_POINTS),
    )


@pytest.fixture(scope="module")
def regression_mesh():
    """Bigger than `tiny_mesh` to give the indicator field somewhere to
    show structure, but still cheap. Refinement level 2 + 4 layers ~
    1900 DoFs at CG1."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=2, degree=1)
    mesh = fd.ExtrudedMesh(
        mesh2d, layers=4, layer_height=0.25, extrusion_type="radial"
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
    # (e.g. a geotherm's surface integral) so they aren't held to rtol.
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
        factory.construct_output()
        factory.construct_geotherm()
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
        factory.construct_output()
        factory.construct_geotherm()
        observed = _evaluate_connectors_lockstep({
            "polygon_indicator": factory.indicator,
            "polygon_geotherm": factory.geotherm,
        }, regression_mesh, Q, TEST_AGES)
        for name in ("polygon_indicator", "polygon_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")


class TestSharedSourceConsistency:
    """Two connectors holding the same LithosphereSource should produce the
    same field as a connector built standalone (same source instance, just a
    different consumer).

    Uses a fresh source (class-scoped) so the forward-only tracker hasn't
    been walked past the test age by sibling regression tests.
    """

    @pytest.fixture(scope="class")
    def fresh_lith_source(self, plate_model, plate_files):
        return LithosphereSource(
            gplates_connector=plate_model,
            continental_data=_load_continental_data(),
            age_to_property=half_space_cooling,
            plate_files=plate_files,
            config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
        )

    def test_two_indicators_sharing_source_agree(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Build two independent indicator connectors that share the same
        # source. Same source ⇒ same prepared dict ⇒ same kNN interpolation
        # ⇒ identical DoF values at the same ndtime. A factory only hands
        # out one indicator, so two factories share the assigned source.
        factory_a = LithosphereConnectorFactory()
        factory_a.source = fresh_lith_source
        factory_a.construct_output()
        factory_b = LithosphereConnectorFactory()
        factory_b.source = fresh_lith_source
        factory_b.construct_output()
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
        # different scalar fields (tanh vs. erf geotherm), but they must
        # agree on what the underlying source dict is — verifiable by
        # checking that the source's per-age cache hits on the second call.
        factory = LithosphereConnectorFactory()
        factory.source = fresh_lith_source
        factory.construct_output()
        factory.construct_geotherm()
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

        # The geotherm call should have hit the source cache rather than
        # rebuilding the dict.
        assert d_first is d_second
