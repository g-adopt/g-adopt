#!/usr/bin/env python3
"""Generate the pickled reference fixture used by tests/unit/test_connectors.py.

Runs the four factory connectors (lith indicator/geotherm, polygon
indicator/geotherm) against the Muller 2022 SE v1.2 reconstruction on a
small extruded mesh, collects reduced quantities per age, and pickles them
into `test_connectors.pkl`.

Usage:
    python tests/unit/data/generate_expected_connectors.py
"""

from pathlib import Path
import pickle

import firedrake as fd
import h5py
import numpy as np

from gadopt.gplates import (
    GplatesScalarFunction,
    PlateModelFiles,
    PointCloudSource,
    ensure_reconstruction,
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

# Shared with tests/unit/test_connectors.py so the reference and the test that
# loads it cannot disagree on the fixture configuration (this script's own
# directory is on sys.path when it is run directly).
from regression_config import (
    OLDEST_AGE,
    LITH_TRACKER_POINT_COUNT,
    POLYGON_BACKGROUND_POINT_COUNT,
    POLYGON_SCALAR_INPUT_POINT_COUNT,
    TEST_AGES,
    REGRESSION_LAYER_HEIGHTS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
OUT = Path(__file__).resolve().parent / "test_connectors.pkl"


def half_space_cooling(age_myr):
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def load_continental_data():
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]]), values


def make_mesh():
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=2, degree=1)
    mesh = fd.ExtrudedMesh(
        mesh2d,
        layers=len(REGRESSION_LAYER_HEIGHTS),
        layer_height=np.array(REGRESSION_LAYER_HEIGHTS, dtype=float),
        extrusion_type="radial",
    )
    mesh.cartesian = False
    return mesh


def reduced(values):
    f_space = fd.FunctionSpace(_mesh, "CG", 1)
    f = fd.Function(f_space)
    f.dat.data_with_halos[:] = values
    return {
        "volume": float(fd.assemble(f * fd.dx)),
        "surface": float(fd.assemble(f * fd.ds_t)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def walk_connectors(connectors_by_name, Q, ages):
    """Drive all connectors in lockstep through the same age series.

    Connectors that share a forward-only source (LithosphereSource) must
    advance together — once we step past an age, no sibling connector can
    revisit it. The lockstep walk matches the real simulation loop: at
    each timestep we update every plate-reconstruction field before moving
    on.
    """
    sfs = {
        name: GplatesScalarFunction(Q, indicator_connector=c, name=name)
        for name, c in connectors_by_name.items()
    }
    out = {name: {} for name in connectors_by_name}
    sample_connector = next(iter(connectors_by_name.values()))
    for age in ages:
        ndtime = sample_connector.source.age2ndtime(float(age))
        for name in connectors_by_name:
            sfs[name].update_plate_reconstruction(ndtime)
            out[name][age] = reduced(sfs[name].dat.data_ro_with_halos.copy())
            print(f"    {name} age={age} Ma  ->  "
                  f"vol={out[name][age]['volume']:.6e}, "
                  f"surf={out[name][age]['surface']:.6e}")
    return out


def main():
    global _mesh
    files = ensure_reconstruction("Muller 2022 SE v1.2", GPLATES_GLOBAL)
    plate_model = pyGplatesConnector(
        rotation_filenames=files["rotation_filenames"],
        topology_filenames=files["topology_filenames"],
        oldest_age=OLDEST_AGE,
    )
    plate_files = PlateModelFiles(
        continental_polygons=files.get("continental_polygons"),
        static_polygons=files.get("static_polygons"),
    )

    _mesh = make_mesh()
    Q = fd.FunctionSpace(_mesh, "CG", 1)

    reference = {}

    print("Lithosphere producer (shared between indicator and geotherm)...")
    lith_producer = LithosphereCloudSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        continental_polygons=plate_files.continental_polygons,
        static_polygons=plate_files.static_polygons,
        continental_data=load_continental_data(),
        oceanic_thickness_from_age=half_space_cooling,
        plate_model_max_age_ma=OLDEST_AGE,
        config=LithosphereCloudConfig(
            tracer=TracerConfig(tracker_point_count=LITH_TRACKER_POINT_COUNT),
        ),
    )
    lith_src = PointCloudSource(lith_producer, plate_model)

    lith_factory = LithosphereConnectorFactory()
    lith_factory.source = lith_src
    # Age sensitivity comes from REGRESSION_LAYER_HEIGHTS, not from an
    # lateral-weight term: the graded mesh puts nodes inside the lithosphere so a
    # moving base depth actually moves the reduced integrals.
    lith_factory.create_indicator()
    lith_factory.create_geotherm()
    lith_result = walk_connectors({
        "lith_indicator": lith_factory.indicator,
        "lith_geotherm": lith_factory.geotherm,
    }, Q, TEST_AGES)
    reference.update(lith_result)

    print("Polygon producer (shared between indicator and geotherm)...")
    poly_producer = PolygonIndicatorSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        polygons=str(CRATON_SHAPEFILE),
        static_polygons=plate_files.static_polygons,
        thickness_data=200.0,
        # thickness_data=200.0 is a SCALAR, so scalar_input_point_count sizes the seed
        # Fibonacci mesh; the old single n_points did both jobs, so pin both.
        config=PolygonIndicatorConfig(
            background_point_count=POLYGON_BACKGROUND_POINT_COUNT,
            scalar_input_point_count=POLYGON_SCALAR_INPUT_POINT_COUNT,
        ),
    )
    poly_src = PointCloudSource(poly_producer, plate_model)
    poly_factory = PolygonConnectorFactory()
    poly_factory.source = poly_src
    poly_factory.create_indicator()
    poly_factory.create_geotherm()
    poly_result = walk_connectors({
        "polygon_indicator": poly_factory.indicator,
        "polygon_geotherm": poly_factory.geotherm,
    }, Q, TEST_AGES)
    reference.update(poly_result)

    with open(OUT, "wb") as f:
        pickle.dump(reference, f)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
