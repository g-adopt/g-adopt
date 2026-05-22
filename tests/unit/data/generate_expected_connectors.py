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
    LithosphereSource,
    LithosphereSourceConfig,
    PolygonSource,
    PolygonSourceConfig,
    ensure_reconstruction,
    lithosphere_geotherm,
    lithosphere_indicator,
    polygon_geotherm,
    polygon_indicator,
    pyGplatesConnector,
)


# Mirrored in tests/unit/test_connectors.py.
OLDEST_AGE = 120
LITH_N_POINTS = 2000
POLYGON_N_POINTS = 3000
TEST_AGES = (100, 50, 0)

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
        mesh2d, layers=4, layer_height=0.25, extrusion_type="radial"
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
        continental_polygons=files.get("continental_polygons"),
        static_polygons=files.get("static_polygons"),
    )

    _mesh = make_mesh()
    Q = fd.FunctionSpace(_mesh, "CG", 1)

    reference = {}

    print("LithosphereSource (shared between indicator and geotherm)...")
    lith_src = LithosphereSource(
        gplates_connector=plate_model,
        continental_data=load_continental_data(),
        age_to_property=half_space_cooling,
        config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
    )

    lith_result = walk_connectors({
        "lith_indicator": lithosphere_indicator(source=lith_src),
        "lith_geotherm": lithosphere_geotherm(source=lith_src),
    }, Q, TEST_AGES)
    reference.update(lith_result)

    print("PolygonSource (shared between indicator and geotherm)...")
    poly_src = PolygonSource(
        gplates_connector=plate_model,
        polygons=str(CRATON_SHAPEFILE),
        thickness_data=200.0,
        config=PolygonSourceConfig(n_points=POLYGON_N_POINTS),
    )
    poly_result = walk_connectors({
        "polygon_indicator": polygon_indicator(source=poly_src),
        "polygon_geotherm": polygon_geotherm(source=poly_src),
    }, Q, TEST_AGES)
    reference.update(poly_result)

    with open(OUT, "wb") as f:
        pickle.dump(reference, f)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
