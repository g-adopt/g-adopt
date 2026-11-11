"""Regression test for the gplates connectors on a real plate reconstruction.

The lithosphere and polygon factories each create an indicator and a geotherm
connector that share one source. Each pair is evaluated through
GplatesScalarFunction on a small radially graded extruded mesh at the ages in
``TEST_AGES``, driven by gtrack producers on the Muller 2022 SE v1.2
reconstruction. Volume and surface integrals and DoF statistics of each field
are compared with the reference in ``expected.pkl``. Run
``generate_expected.py`` in this directory to rebuild that reference.

The connector tests that need no reconstruction data are in
tests/unit/test_connectors.py.
"""

from pathlib import Path
import pickle

import firedrake as fd
import h5py
import numpy as np
import pytest

from gadopt.gplates import (
    GplatesScalarFunction,
    LithosphereConnectorFactory,
    PlateModelFiles,
    PointCloudSource,
    PolygonConnectorFactory,
    ensure_reconstruction,
    pyGplatesConnector,
)
from gtrack import (
    LithosphereCloudConfig,
    LithosphereCloudSource,
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
)
from gtrack.config import TracerConfig

# The reference values are computed from these numbers. The generator script
# imports the same module, so the test and the reference cannot use different
# settings. pytest puts this directory on sys.path, so the import works as is.
from regression_config import (
    OLDEST_AGE,
    LITH_TRACKER_POINT_COUNT,
    POLYGON_BACKGROUND_POINT_COUNT,
    POLYGON_SCALAR_INPUT_POINT_COUNT,
    TEST_AGES,
    REGRESSION_LAYER_HEIGHTS,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
REFERENCE_FILE = Path(__file__).resolve().parent / "expected.pkl"


def _require_data():
    """Fail with a download hint if the continental thickness data is absent.

    The plate model itself is fetched by ``ensure_reconstruction`` in the
    fixtures that need it, matching ``tests/unit/test_gplates.py``.
    """
    assert CONTINENTAL_DATA.exists(), (
        f"Continental thickness data missing at {CONTINENTAL_DATA}; download "
        "https://data.gadopt.org/demos/continental_lithospheric_thickness_mesh.h5 there."
    )


def _require_craton():
    """Fail with a download hint if the craton shapefile is absent."""
    assert CRATON_SHAPEFILE.exists(), (
        f"Craton shapefile missing at {CRATON_SHAPEFILE}; download the .shp, "
        ".shx, .dbf and .prj parts of Craton_Boundaries_Inferred from "
        "https://data.gadopt.org/demos/ there."
    )


def half_space_cooling(age_myr):
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def _load_continental_data():
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]]), values


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
def poly_source(plate_model, plate_files):
    _require_craton()
    producer = PolygonIndicatorSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        polygons=str(CRATON_SHAPEFILE),
        static_polygons=plate_files.static_polygons,
        thickness_data=200.0,
        # thickness_data is a scalar, so scalar_input_point_count sets the
        # number of seed points. Both counts come from regression_config so
        # that the cloud matches the one the reference was built from.
        config=PolygonIndicatorConfig(
            background_point_count=POLYGON_BACKGROUND_POINT_COUNT,
            scalar_input_point_count=POLYGON_SCALAR_INPUT_POINT_COUNT,
        ),
    )
    return PointCloudSource(producer, plate_model)


@pytest.fixture(scope="module")
def regression_mesh():
    """Create the regression mesh: about 2700 CG1 nodes in six graded layers.

    The layers get thinner towards the surface, so several nodes lie inside
    the lithosphere. A change in base depth with age then changes the
    reduced integrals.
    """
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
# Regression: the four factory connectors through GplatesScalarFunction
# ---------------------------------------------------------------------------

def _reduced_quantities(f):
    """Volume and surface integrals of a scalar Function on the regression
    mesh, plus mean / std / min / max of its DoF values."""
    values = f.dat.data_ro_with_halos
    return {
        "volume": float(fd.assemble(f * fd.dx)),
        "surface": float(fd.assemble(f * fd.ds_t)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _evaluate_connectors_lockstep(connectors_by_name, Q, ages):
    """Drive all connectors in lockstep through the same age series.

    Connectors sharing a forward-only source (the lithosphere source) must
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
            out[name][age] = _reduced_quantities(sfs[name])
    return out


def _load_reference():
    assert REFERENCE_FILE.exists(), (
        f"Reference file missing: {REFERENCE_FILE}. "
        "Generate it with generate_expected.py in this directory."
    )
    with open(REFERENCE_FILE, "rb") as f:
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

    def test_lithosphere_pair(self, lith_source, Q):
        ref = _load_reference()
        factory = LithosphereConnectorFactory()
        factory.source = lith_source
        # Must match generate_expected.py. Age sensitivity comes
        # from the radial grading of `regression_mesh`: with the plain
        # one-sided step, nodes have to sit inside the lithosphere for a
        # moving base depth to change the reduced integrals at all.
        factory.create_indicator_output()
        factory.create_geotherm_output()
        observed = _evaluate_connectors_lockstep({
            "lith_indicator": factory.indicator,
            "lith_geotherm": factory.geotherm,
        }, Q, TEST_AGES)
        for name in ("lith_indicator", "lith_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_polygon_pair(self, poly_source, Q):
        ref = _load_reference()
        factory = PolygonConnectorFactory()
        factory.source = poly_source
        factory.create_indicator_output()
        factory.create_geotherm_output()
        observed = _evaluate_connectors_lockstep({
            "polygon_indicator": factory.indicator,
            "polygon_geotherm": factory.geotherm,
        }, Q, TEST_AGES)
        for name in ("polygon_indicator", "polygon_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")
