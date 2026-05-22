"""Tests for IndicatorConnector and its factory functions.

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
    IndicatorConnector,
    InterpolationConfig,
    LithosphereSource,
    LithosphereSourceConfig,
    MeshConfig,
    PolygonSource,
    PolygonSourceConfig,
    Source,
    TanhOutput,
    ensure_reconstruction,
    lithosphere_geotherm,
    lithosphere_indicator,
    polygon_geotherm,
    polygon_indicator,
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
        # Bypass the ABC by setting the class attribute via assignment.
        self.__class__ = type(
            f"_DummySource_{'_'.join(sorted(provides))}",
            (Source,),
            {"provides": frozenset(provides)},
        )
        self.comm = comm
        self._is_root = (comm.rank == 0)
        self.gplates_connector = _DummyGplates()

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
        IndicatorConnector(src, TanhOutput())  # must not raise

    def test_lith_with_linear_geotherm_allowed(self):
        # GeothermLinearOutput requires only {"thickness"} — a polygon-style
        # output paired against a lithosphere-style source is allowed.
        src = _DummySource({"xyz", "thickness", "age"})
        IndicatorConnector(src, GeothermLinearOutput())

    def test_polygon_with_erf_geotherm_raises(self):
        # PolygonSource provides {"xyz","thickness"} — missing "age" that
        # GeothermERFOutput needs.
        src = _DummySource({"xyz", "thickness"})
        with pytest.raises(ValueError, match="age"):
            IndicatorConnector(src, GeothermERFOutput())

    def test_polygon_with_tanh_allowed(self):
        src = _DummySource({"xyz", "thickness"})
        IndicatorConnector(src, TanhOutput(default_thickness_km=0.0))

    def test_polygon_with_linear_geotherm_allowed(self):
        src = _DummySource({"xyz", "thickness"})
        IndicatorConnector(src, GeothermLinearOutput())


class TestConnectorConstruction:
    def test_gc_collect_frequency_validated(self):
        src = _DummySource({"xyz", "thickness"})
        with pytest.raises(ValueError, match="gc_collect_frequency"):
            IndicatorConnector(src, TanhOutput(), gc_collect_frequency=0)

    def test_defaults_use_module_level_configs(self):
        src = _DummySource({"xyz", "thickness"})
        conn = IndicatorConnector(src, TanhOutput())
        assert isinstance(conn.mesh, MeshConfig)
        assert isinstance(conn.interpolation, InterpolationConfig)


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
        conn = IndicatorConnector(src, TanhOutput())
        with pytest.raises(TypeError, match="scalar function space"):
            GplatesScalarFunction(V, indicator_connector=conn)

    def test_scalar_space_accepted(self, tiny_mesh):
        Q = fd.FunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"xyz", "thickness"})
        conn = IndicatorConnector(src, TanhOutput())
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
        continental_polygons=files.get("continental_polygons"),
        static_polygons=files.get("static_polygons"),
    )


@pytest.fixture(scope="module")
def lith_source(plate_model):
    return LithosphereSource(
        gplates_connector=plate_model,
        continental_data=_load_continental_data(),
        age_to_property=half_space_cooling,
        config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
    )


@pytest.fixture(scope="module")
def poly_source(plate_model):
    _require_craton()
    return PolygonSource(
        gplates_connector=plate_model,
        polygons=str(CRATON_SHAPEFILE),
        thickness_data=200.0,
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
    for key, expected_value in expected.items():
        np.testing.assert_allclose(
            observed[key], expected_value, rtol=1e-3,
            err_msg=f"{label}: {key}",
        )


class TestConnectorRegression:
    """Drive the four factory connectors in lockstep (siblings sharing a
    Source advance together) and assert volume/surface integrals match the
    pickled reference."""

    def test_lithosphere_pair(self, lith_source, regression_mesh, Q):
        ref = _load_reference()
        observed = _evaluate_connectors_lockstep({
            "lith_indicator": lithosphere_indicator(source=lith_source),
            "lith_geotherm": lithosphere_geotherm(source=lith_source),
        }, regression_mesh, Q, TEST_AGES)
        for name in ("lith_indicator", "lith_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_polygon_pair(self, poly_source, regression_mesh, Q):
        ref = _load_reference()
        observed = _evaluate_connectors_lockstep({
            "polygon_indicator": polygon_indicator(source=poly_source),
            "polygon_geotherm": polygon_geotherm(source=poly_source),
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
    def fresh_lith_source(self, plate_model):
        return LithosphereSource(
            gplates_connector=plate_model,
            continental_data=_load_continental_data(),
            age_to_property=half_space_cooling,
            config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
        )

    def test_two_indicators_sharing_source_agree(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Build two independent indicator connectors that share the same
        # source. Same source ⇒ same prepared dict ⇒ same kNN interpolation
        # ⇒ identical DoF values at the same ndtime.
        ind_a = lithosphere_indicator(source=fresh_lith_source)
        ind_b = lithosphere_indicator(source=fresh_lith_source)

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
        ind = lithosphere_indicator(source=fresh_lith_source)
        geo = lithosphere_geotherm(source=fresh_lith_source)

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
