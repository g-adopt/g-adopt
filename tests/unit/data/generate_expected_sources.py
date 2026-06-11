#!/usr/bin/env python3
"""Generate the pickled reference fixtures used by tests/unit/test_sources.py.

Run once on a workstation that has the Muller 2022 SE reconstruction and the
continental thickness data downloaded. Commit the resulting pickles. Do not
re-run unless the underlying algorithm intentionally changes (e.g. a gtrack
version bump or a source-config parameter change).

Usage:
    python tests/unit/data/generate_expected_sources.py
"""

from pathlib import Path
import pickle

import h5py
import numpy as np

from gadopt.gplates import (
    LithosphereSource,
    LithosphereSourceConfig,
    PlateModelFiles,
    PolygonSource,
    PolygonSourceConfig,
    ensure_reconstruction,
    pyGplatesConnector,
)


# These constants are mirrored in tests/unit/test_sources.py. If you change
# one, change the other.
OLDEST_AGE = 120
LITH_N_POINTS = 2000
POLYGON_N_POINTS = 3000
TEST_AGES = (100, 50, 0)

REPO_ROOT = Path(__file__).resolve().parents[3]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
OUT_DIR = Path(__file__).resolve().parent


def half_space_cooling(age_myr):
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def load_continental_data():
    """Load (latlon, thickness) from the demo HDF5 file.

    Matches the demo's load idiom in gplates_fields.py — the file stores
    points as (lon, lat) but our sources expect (lat, lon), hence the swap.
    """
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
    return latlon, values


def make_plate_model():
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
    return plate_model, plate_files


def reduce_lithosphere(d):
    return {
        "n_points": int(len(d["xyz"])),
        "thickness_mean": float(d["thickness"].mean()),
        "thickness_min": float(d["thickness"].min()),
        "thickness_max": float(d["thickness"].max()),
        "age_mean": float(d["age"].mean()),
        "age_min": float(d["age"].min()),
        "age_max": float(d["age"].max()),
    }


def reduce_polygon(d):
    thick = d["thickness"]
    return {
        "n_points": int(len(d["xyz"])),
        "thickness_mean": float(thick.mean()),
        "thickness_sum": float(thick.sum()),
        "thickness_nonzero": int(np.count_nonzero(thick)),
    }


def main():
    plate_model, plate_files = make_plate_model()

    print("Building LithosphereSource...")
    lith = LithosphereSource(
        gplates_connector=plate_model,
        continental_data=load_continental_data(),
        age_to_property=half_space_cooling,
        plate_files=plate_files,
        config=LithosphereSourceConfig(n_points=LITH_N_POINTS),
    )
    lith_ref = {}
    for age in TEST_AGES:
        d = lith.prepare(float(age))
        lith_ref[age] = reduce_lithosphere(d)
        print(f"  age={age} Ma  ->  {lith_ref[age]}")
    out_lith = OUT_DIR / "test_lithosphere_source.pkl"
    with open(out_lith, "wb") as f:
        pickle.dump(lith_ref, f)
    print(f"Wrote {out_lith}")

    print("\nBuilding PolygonSource...")
    poly = PolygonSource(
        gplates_connector=plate_model,
        polygons=str(CRATON_SHAPEFILE),
        thickness_data=200.0,
        plate_files=plate_files,
        config=PolygonSourceConfig(n_points=POLYGON_N_POINTS),
    )
    poly_ref = {}
    for age in TEST_AGES:
        d = poly.prepare(float(age))
        poly_ref[age] = reduce_polygon(d)
        print(f"  age={age} Ma  ->  {poly_ref[age]}")
    out_poly = OUT_DIR / "test_polygon_source.pkl"
    with open(out_poly, "wb") as f:
        pickle.dump(poly_ref, f)
    print(f"Wrote {out_poly}")


if __name__ == "__main__":
    main()
