"""Tests for Source subclasses.

Three groups:
  * config validation (no reconstruction data)
  * shape / reduced-quantity regression against pickled references
  * cache contracts: per-age cache identity, single-step guarantee for the
    forward-only ocean tracker when two consumers share a LithosphereSource.

The regression pickles live in ``tests/unit/data/`` and are produced by the
generation scripts also under that directory. Reference values were captured
once against the Muller 2022 SE v1.2 reconstruction.
"""

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pickle
import pytest

from gadopt.gplates import (
    InterpolationConfig,
    ScalarFieldConnector,
    LithosphereSource,
    LithosphereSourceConfig,
    PlateModelFiles,
    PolygonSource,
    PolygonSourceConfig,
    QuinticOutput,
    GeothermERFOutput,
    ensure_reconstruction,
    pyGplatesConnector,
)


# ---------------------------------------------------------------------------
# Paths and fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Fixed reconstruction parameters used across all regression tests in this
# file. Bumping any of these invalidates the pickled fixtures.
OLDEST_AGE = 120
LITH_N_POINTS = 2000
POLYGON_N_POINTS = 3000
TEST_AGES = (100, 50, 0)


def _require_data():
    """Skip the test if reconstruction or thickness data isn't present."""
    if not (GPLATES_GLOBAL / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
        pytest.skip("Muller 2022 SE reconstruction not downloaded; run `make data`.")
    if not CONTINENTAL_DATA.exists():
        pytest.skip(f"Continental thickness data missing at {CONTINENTAL_DATA}.")


def _require_craton():
    if not CRATON_SHAPEFILE.exists():
        pytest.skip(f"Craton shapefile missing at {CRATON_SHAPEFILE}.")


def half_space_cooling(age_myr):
    """Standard half-space cooling thickness from seafloor age."""
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def _load_continental_data():
    """Load (latlon, thickness) from the demo HDF5 in the format the source
    expects. The file stores (lon, lat); we swap to (lat, lon)."""
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]]), values


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
def polygon_source(plate_model, plate_files):
    _require_craton()
    return PolygonSource(
        gplates_connector=plate_model,
        polygons=str(CRATON_SHAPEFILE),
        thickness_data=200.0,
        plate_files=plate_files,
        config=PolygonSourceConfig(n_points=POLYGON_N_POINTS),
    )


# ---------------------------------------------------------------------------
# Config validation (no reconstruction data needed)
# ---------------------------------------------------------------------------

class TestLithosphereSourceConfig:
    def test_defaults(self):
        cfg = LithosphereSourceConfig()
        assert cfg.n_points == 10000
        assert cfg.time_step == 1.0
        assert cfg.reinit_interval_myr == 50.0
        assert cfg.checkpoint_interval_myr is None

    def test_rejects_small_n_points(self):
        with pytest.raises(ValueError, match="n_points must be at least"):
            LithosphereSourceConfig(n_points=10)

    def test_rejects_nonpositive_time_step(self):
        with pytest.raises(ValueError, match="time_step must be positive"):
            LithosphereSourceConfig(time_step=0)

    def test_rejects_nonpositive_reinit_interval(self):
        with pytest.raises(ValueError, match="reinit_interval_myr must be positive"):
            LithosphereSourceConfig(reinit_interval_myr=-1.0)

    def test_rejects_nonpositive_checkpoint_interval(self):
        with pytest.raises(ValueError, match="checkpoint_interval_myr must be positive"):
            LithosphereSourceConfig(checkpoint_interval_myr=0)


class TestPolygonSourceConfig:
    def test_defaults(self):
        cfg = PolygonSourceConfig()
        assert cfg.n_points == 20000

    def test_rejects_small_n_points(self):
        with pytest.raises(ValueError, match="n_points must be at least"):
            PolygonSourceConfig(n_points=10)


class TestInterpolationConfig:
    def test_defaults(self):
        cfg = InterpolationConfig()
        assert cfg.kernel == "idw"
        assert cfg.k_neighbors == 50

    def test_rejects_bad_kernel(self):
        with pytest.raises(ValueError, match="kernel must be one of"):
            InterpolationConfig(kernel="rbf")

    def test_rejects_nonpositive_k_neighbors(self):
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            InterpolationConfig(k_neighbors=0)

    def test_rejects_nonpositive_distance_threshold(self):
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            InterpolationConfig(distance_threshold=0)


# ---------------------------------------------------------------------------
# Source contract: provides set, plate-model validation
# ---------------------------------------------------------------------------

class TestSourceContracts:
    # provides is a read-only instance property on the concrete sources, so
    # assert through instances; construction is I/O-free (lazy load).
    def test_lithosphere_provides(self):
        src = LithosphereSource(
            gplates_connector=_StubConnector(),
            continental_data=100.0,
            age_to_property=half_space_cooling,
            plate_files=PlateModelFiles(
                continental_polygons="continental.gpml",
                static_polygons="static.gpml",
            ),
        )
        assert src.provides == frozenset({"xyz", "thickness", "age"})

    def test_polygon_provides(self):
        src = PolygonSource(
            gplates_connector=_StubConnector(),
            polygons="craton.shp",
            thickness_data=200.0,
            plate_files=PlateModelFiles(static_polygons="static.gpml"),
        )
        assert src.provides == frozenset({"xyz", "thickness"})

    def test_lithosphere_requires_continental_polygons(self):
        # The check is against PlateModelFiles, not the connector, and fires
        # AT the constructor — no reconstruction data needed.
        plate_files = PlateModelFiles(static_polygons="static.gpml")
        with pytest.raises(ValueError, match="continental_polygons"):
            LithosphereSource(
                _StubConnector(), 100.0, half_space_cooling, plate_files
            )

    def test_polygon_requires_static_polygons(self):
        plate_files = PlateModelFiles(continental_polygons="continental.gpml")
        with pytest.raises(ValueError, match="static_polygons"):
            PolygonSource(_StubConnector(), "craton.shp", 200.0, plate_files)


# ---------------------------------------------------------------------------
# Lazy load: construction is I/O-free, _load fires once
# ---------------------------------------------------------------------------

class _StubConnector:
    """Cheap stand-in for pyGplatesConnector.

    Carries only the velocity-relevant attributes: the rotation/topology
    filename lists that _load reaches for, plus delta_t / oldest_age. The
    polygon paths now live on PlateModelFiles, not here. Building one of these
    touches no reconstruction data, so it isolates the construction path from
    the gtrack I/O in _load.
    """

    rotation_filenames = ["rot.rot"]
    topology_filenames = ["topo.gpml"]
    delta_t = 1.0
    oldest_age = 100.0


class TestLazyLoad:
    """Source construction must not perform gtrack/pyGplates I/O — that is
    deferred to the first prepare() via _load. These tests run without any
    reconstruction data."""

    def test_lithosphere_construction_is_io_free(self):
        src = LithosphereSource(
            gplates_connector=_StubConnector(),
            continental_data=100.0,
            age_to_property=half_space_cooling,
            plate_files=PlateModelFiles(
                continental_polygons="continental.gpml",
                static_polygons="static.gpml",
            ),
        )
        # Nothing loaded, no gtrack handles built yet.
        assert src._loaded is False
        assert src._ocean_tracker is None
        assert src._rotator is None
        assert src._continental_filter is None
        assert src._continental_present is None

    def test_polygon_construction_is_io_free(self):
        src = PolygonSource(
            gplates_connector=_StubConnector(),
            polygons="craton.shp",
            thickness_data=200.0,
            plate_files=PlateModelFiles(static_polygons="static.gpml"),
        )
        assert src._loaded is False
        assert src._polygon_filter is None
        assert src._rotator is None
        assert src._region_present is None

    def test_load_fires_exactly_once_across_prepares(self):
        # Drive prepare() twice and assert _load ran once. We stub _load (so no
        # gtrack) and _compute_sources (so prepare returns without real work),
        # which isolates the _ensure_loaded latch.
        src = PolygonSource(
            gplates_connector=_StubConnector(),
            polygons="craton.shp",
            thickness_data=200.0,
            plate_files=PlateModelFiles(static_polygons="static.gpml"),
        )
        load_calls = {"n": 0}

        def fake_load():
            load_calls["n"] += 1

        src._load = fake_load
        src._compute_sources = lambda age: {
            "xyz": np.zeros((1, 3)), "thickness": np.zeros(1)
        }

        # Two prepares at distinct ages (second is far enough to miss the cache).
        src.prepare(10.0)
        src.prepare(50.0)
        assert load_calls["n"] == 1
        assert src._loaded is True


# ---------------------------------------------------------------------------
# walk_start_age: declared forward-only ceiling
# ---------------------------------------------------------------------------

class TestWalkStartAge:
    """The forward-only ocean tracker would otherwise let whichever connector
    updates first silently fix the walk's oldest reachable age. Declaring
    walk_start_age pins that ceiling up front so an out-of-range request fails
    immediately. These tests drive validate_age directly — no real prepare,
    no gtrack — using the cheap stub connector (oldest_age=100.0)."""

    @staticmethod
    def _lith(walk_start_age):
        return LithosphereSource(
            gplates_connector=_StubConnector(),
            continental_data=100.0,
            age_to_property=half_space_cooling,
            plate_files=PlateModelFiles(
                continental_polygons="continental.gpml",
                static_polygons="static.gpml",
            ),
            walk_start_age=walk_start_age,
        )

    def test_declared_promise_check(self):
        src = self._lith(walk_start_age=80.0)
        # Pre-prepare: anything older than the declared start fails the promise.
        with pytest.raises(ValueError, match="walk_start_age"):
            src.validate_age(95.0)
        # The declared start itself and younger ages pass.
        src.validate_age(80.0)
        src.validate_age(50.0)

    def test_physical_floor_overrides_declaration(self):
        # Declaring an old start does NOT let you revisit ages stepped past.
        src = self._lith(walk_start_age=80.0)
        src._initialized = True
        src._cached_age = 50.0
        # 70 <= declared 80 (passes promise) but > last computed 50 -> floor.
        with pytest.raises(ValueError, match="last .*computed age"):
            src.validate_age(70.0)
        # Younger than the floor is fine.
        src.validate_age(40.0)

    def test_default_none_preprepare_allows_old_age(self):
        src = self._lith(walk_start_age=None)
        # No declaration and not yet stepped: any in-range age passes.
        src.validate_age(95.0)
        # After stepping, the physical floor still applies.
        src._initialized = True
        src._cached_age = 50.0
        with pytest.raises(ValueError, match="last .*computed age"):
            src.validate_age(95.0)

    def test_init_time_range_validation(self):
        # walk_start_age above the model's oldest age (stub: 100.0) is rejected
        # at construction, before any I/O.
        with pytest.raises(ValueError, match="walk_start_age"):
            self._lith(walk_start_age=150.0)
        # Negative is rejected too.
        with pytest.raises(ValueError, match="walk_start_age"):
            self._lith(walk_start_age=-5.0)

    def test_message_variants_distinct(self):
        # Promise-check message names walk_start_age; physical-floor names the
        # last computed age. They must be distinguishable.
        src = self._lith(walk_start_age=80.0)
        with pytest.raises(ValueError) as promise:
            src.validate_age(95.0)
        assert "walk_start_age" in str(promise.value)
        assert "last computed age" not in str(promise.value)

        src._initialized = True
        src._cached_age = 50.0
        with pytest.raises(ValueError) as floor:
            src.validate_age(70.0)
        assert "last computed age" in str(floor.value)


class _FakeTracker:
    """Minimal stand-in for SeafloorAgeTracker — counts initialize() calls
    and no-ops the rest, so the walk-init path can be exercised without
    touching gtrack."""

    def __init__(self):
        self.initialize_calls = 0
        self.current_age = 0.0

    def initialize(self, starting_age):
        self.initialize_calls += 1

    def reinitialize(self, n_points):
        pass

    def load_checkpoint(self, path):
        pass

    def step_to(self, age):
        return object()


class TestInitializeWalk:
    """The first-call walk init was pulled out of _step_ocean_to into
    _initialize_walk with an inverted early-return guard. This pins the
    idempotency at the mockable layer — a botched guard inversion (e.g.
    dropping the latch) would re-initialise on every call and trip here,
    without needing the slow data-backed regression."""

    @staticmethod
    def _lith():
        src = LithosphereSource(
            gplates_connector=_StubConnector(),
            continental_data=100.0,
            age_to_property=half_space_cooling,
            plate_files=PlateModelFiles(
                continental_polygons="continental.gpml",
                static_polygons="static.gpml",
            ),
        )
        src._ocean_tracker = _FakeTracker()
        # No checkpoint dir -> _find_best_checkpoint returns None -> the
        # initialize() path is taken, not load_checkpoint.
        src._checkpoint_dir = None
        return src

    def test_initialize_runs_exactly_once_across_steps(self):
        src = self._lith()
        src._step_ocean_to(100)
        assert src._initialized is True
        assert src._ocean_tracker.initialize_calls == 1
        # A second (younger) step must not re-initialise the walk.
        src._step_ocean_to(90)
        assert src._ocean_tracker.initialize_calls == 1

    def test_initialize_walk_is_idempotent(self):
        src = self._lith()
        src._initialize_walk(100)
        src._initialize_walk(100)
        assert src._ocean_tracker.initialize_calls == 1


# ---------------------------------------------------------------------------
# prepare(age) regression
# ---------------------------------------------------------------------------

def _reduce_lithosphere(d):
    """Reduced quantities for a LithosphereSource prepare() dict."""
    return {
        "n_points": int(len(d["xyz"])),
        "thickness_mean": float(d["thickness"].mean()),
        "thickness_min": float(d["thickness"].min()),
        "thickness_max": float(d["thickness"].max()),
        "age_mean": float(d["age"].mean()),
        "age_min": float(d["age"].min()),
        "age_max": float(d["age"].max()),
    }


def _reduce_polygon(d):
    """Reduced quantities for a PolygonSource prepare() dict."""
    thick = d["thickness"]
    return {
        "n_points": int(len(d["xyz"])),
        "thickness_mean": float(thick.mean()),
        "thickness_sum": float(thick.sum()),
        "thickness_nonzero": int(np.count_nonzero(thick)),
    }


class TestLithosphereSourceRegression:
    """Pickled regression against a fixed Muller 2022 SE reconstruction."""

    def test_prepare_matches_reference(self, lith_source):
        ref_path = DATA_DIR / "test_lithosphere_source.pkl"
        if not ref_path.exists():
            pytest.skip(
                f"Reference fixture missing: {ref_path}. "
                "Generate via tests/unit/data/generate_expected_sources.py."
            )
        with open(ref_path, "rb") as f:
            reference = pickle.load(f)

        for age in TEST_AGES:
            d = lith_source.prepare(float(age))
            reduced = _reduce_lithosphere(d)
            ref = reference[age]
            for key, expected in ref.items():
                if key == "n_points":
                    # Ocean-tracker seed count can drift by a handful of
                    # points between runs (ridge-spawning under fp noise);
                    # allow ±1% rather than strict equality.
                    np.testing.assert_allclose(
                        reduced[key], expected, rtol=0.01,
                        err_msg=f"age {age}: {key}",
                    )
                else:
                    np.testing.assert_allclose(
                        reduced[key], expected, rtol=1e-3,
                        err_msg=f"age {age}: {key}",
                    )


class TestPolygonSourceRegression:
    def test_prepare_matches_reference(self, polygon_source):
        ref_path = DATA_DIR / "test_polygon_source.pkl"
        if not ref_path.exists():
            pytest.skip(
                f"Reference fixture missing: {ref_path}. "
                "Generate via tests/unit/data/generate_expected_sources.py."
            )
        with open(ref_path, "rb") as f:
            reference = pickle.load(f)

        for age in TEST_AGES:
            d = polygon_source.prepare(float(age))
            reduced = _reduce_polygon(d)
            ref = reference[age]
            for key, expected in ref.items():
                if isinstance(expected, int):
                    assert reduced[key] == expected, (
                        f"age {age}: {key} mismatch (got {reduced[key]}, "
                        f"expected {expected})"
                    )
                else:
                    np.testing.assert_allclose(
                        reduced[key], expected, rtol=1e-3,
                        err_msg=f"age {age}: {key}",
                    )


# ---------------------------------------------------------------------------
# Cache contracts
# ---------------------------------------------------------------------------

class TestSourceCacheIdentity:
    """A second prepare() call at the same age must return the cached dict
    by identity, not just a numerically-equal copy. This is the contract
    that lets two connectors sharing the source see a single advance per
    age."""

    def test_polygon_source_cache_hit_returns_identical_dict(self, polygon_source):
        # Polygon source is stateless w.r.t. time, so cache identity is the
        # cleanest demonstration without touching the ocean tracker.
        d1 = polygon_source.prepare(60.0)
        d2 = polygon_source.prepare(60.0)
        assert d1 is d2

    def test_lithosphere_source_cache_hit_returns_identical_dict(self, lith_source):
        # Use age=0 so this test is robust to whatever the tracker's current
        # state is at the time pytest runs it (the regression test in the
        # same module walks ages down to 0; the forward-only tracker can't
        # revisit older ages from there).
        d1 = lith_source.prepare(0.0)
        d2 = lith_source.prepare(0.0)
        assert d1 is d2


class TestSingleStepGuarantee:
    """The forward-only ocean tracker must step at most once per geological
    age, regardless of how many connectors share the source. This is the
    central correctness property of the Source/Output split."""

    def test_shared_source_steps_once_per_age(self, plate_model, plate_files):
        # Build a fresh source so the call counter starts clean.
        source = LithosphereSource(
            gplates_connector=plate_model,
            continental_data=_load_continental_data(),
            age_to_property=half_space_cooling,
            plate_files=plate_files,
            config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
        )

        ind_connector = ScalarFieldConnector(
            source, QuinticOutput(width_km=10.0),
        )
        geo_connector = ScalarFieldConnector(
            source, GeothermERFOutput(kappa=1e-6),
        )

        # A tiny target coordinate cloud; the connectors only need this to
        # complete the kNN interpolation step.
        n_targets = 10
        np.random.seed(0)
        target_coords = 1.5 * np.random.randn(n_targets, 3)

        with patch.object(
            source, "_step_ocean_to",
            wraps=source._step_ocean_to,
        ) as spy:
            # First age: each connector independently asks the source. The
            # second call must hit the cache; total step count == 1.
            ndtime_100 = source.age2ndtime(100.0)
            ind_connector.get_indicator(target_coords, ndtime_100)
            geo_connector.get_indicator(target_coords, ndtime_100)
            assert spy.call_count == 1, (
                f"Two connectors at the same age must trigger exactly one "
                f"tracker step; got {spy.call_count}."
            )

            # Second age: cache invalidates, expect exactly one more step.
            ndtime_50 = source.age2ndtime(50.0)
            ind_connector.get_indicator(target_coords, ndtime_50)
            geo_connector.get_indicator(target_coords, ndtime_50)
            assert spy.call_count == 2, (
                f"Advancing to a new age should step the tracker exactly "
                f"once more; got total {spy.call_count}."
            )
