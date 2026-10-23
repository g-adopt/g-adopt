r"""GIAMIP Exp05: the surface fields of a forward run as a ParaView time series.

WHAT THIS SCRIPT DOES
    It reads `forward_exp05_surface.h5` of `forward_exp05.py` and writes the
    surface fields of every step as ParaView files (`VTKFile`), with `time`
    = the model year. It writes nothing else.

    The file holds the surface submesh `giamip_surface` and its fields.
    `CheckpointFile` loads that mesh on another number of ranks than the
    writer's by partitioning it again, so this script runs on 1 rank as well
    as on a few. It must not run on the rank count of the writer: there the
    file loads with its saved distribution, and for this codim-1 submesh
    that load is broken (the cells are counted more than once). The script
    checks the area of the loaded surface and stops if it is wrong. At L0 the surface has about 130 000
    CG2 nodes, which one rank reads in a few seconds per step.

THE GEOMETRY
    The surface submesh has P1 geometry (flat triangles, see
    `exp05_common.surface_mesh`). The file also holds `coordinates_m`: the
    curved P2 positions of the CG2 nodes in metres. This script builds a new
    mesh with those coordinates (`Mesh(coordinates_m)`, same topology), so
    ParaView shows the curved surface at its true size in metres. A field
    moves to the new mesh as a copy of its data: the new mesh has the
    topology of the old one, so every space numbers its nodes the same way.

THE TWO SERIES
    `VTKFile` writes all the fields of one call on one set of points, the
    points of the largest element of the call. The DG0 facet fractions and
    the CG2 fields in one call would put every field on discontinuous P2
    points (six points per triangle). They are therefore written as two
    series:

      forward_exp05_fields.pvd     the CG2 and CG1 fields, continuous, on the
                                   P2 points (`--degree 2`) or on the
                                   vertices only (`--degree 1`, smaller)
      forward_exp05_fractions.pvd  the DG0 facet fractions as DG1 point
                                   data on flat triangles: `VTKFile` writes
                                   point data only, so each triangle has
                                   three points with the same value

    Open both in ParaView; they share the time values (model years).

RUN
    python3 paraview_exp05.py --input <dir>/forward_exp05_surface.h5
        --output_path <dir>/paraview [--degree 2] [--every 1]
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

# The shared module imports gadopt before firedrake. Only its names are used.
import exp05_common as common  # noqa: E402
import numpy as np  # noqa: E402
from gadopt import (COMM_WORLD, CheckpointFile, Constant,  # noqa: E402
                    Function, FunctionSpace, Mesh, SpatialCoordinate,
                    VTKFile, assemble, dot, dx, log, sqrt)

#: The continuous fields of `forward_exp05.SurfaceOutput`, in the order of
#: the ParaView file.
CONTINUOUS_FIELDS = ("displacement_m", "delta_bed_m", "displacement_east_m",
                     "displacement_north_m", "geoid_m", "sea_surface_m",
                     "delta_g_m", "sea_level_m", "bed_m", "ice_thickness_m")
#: The DG0 facet fractions.
FRACTION_FIELDS = ("ocean_function", "ocean_without_grounded_ice",
                   "grounded_ice_fraction", "land_ice_fraction")


def on_mesh(f, mesh):
    """`f` as a Function on `mesh`, which has the topology of f's mesh.

    The spaces of two meshes on one topology number their nodes alike, so
    the data is copied as it is. The copy is named as `f`.

    Args:
      f: a Function on the loaded surface mesh.
      mesh: the mesh with the curved coordinates.

    Returns:
      The new Function.
    """
    space = FunctionSpace(mesh, f.function_space().ufl_element())
    out = Function(space, name=f.name())
    out.dat.data_wo_with_halos[:] = f.dat.data_ro_with_halos
    return out


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="forward_exp05_surface.h5 of forward_exp05.py")
    p.add_argument("--output_path", default="./paraview")
    p.add_argument("--degree", type=int, choices=[1, 2], default=2,
                   help="the points of the continuous series: 2 for the P2 "
                        "nodes (all the data), 1 for the vertices only")
    p.add_argument("--every", type=int, default=1,
                   help="write every n-th step (the first and the last are "
                        "always written)")
    p.add_argument("--skip_area_check", action="store_true",
                   help="do not compare the surface area with a sphere (for "
                        "a test file on a mesh that is not a sphere)")
    return p.parse_args()


def main():
    """Read the surface file and write the two ParaView series."""
    args = parse_args()
    tic_run = time.time()
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()

    with CheckpointFile(args.input, "r") as chk:
        flat = chk.load_mesh(common.SURFACE_NAME)
        coordinates = chk.load_function(flat, "coordinates_m")
        # The steps and their years, from the timestepping values that
        # `forward_exp05.py` saved with every entry.
        history = chk.get_timestepping_history(flat, "delta_bed_m")
        indices = [int(i) for i in history["index"]]
        years = [float(y) for y in history["year"]]
        # The curved surface in metres, on the topology of the flat one.
        curved = Mesh(coordinates, name="giamip_surface_curved_m")
        # A guard against a broken load. On the rank count of the writer,
        # CheckpointFile loads the surface submesh with its saved
        # distribution, and that load counts cells more than once (job
        # 51182555: area 4.07 times the sphere on 384 ranks). The area of
        # the curved mesh must be the area of the sphere of its mean radius
        # to the accuracy of the P2 surface.
        area = float(assemble(Constant(1.0) * dx(domain=curved)))
        radius = float(assemble(sqrt(dot(SpatialCoordinate(curved),
                                         SpatialCoordinate(curved)))
                                * dx(domain=curved))) / area
        ratio = area / (4.0 * np.pi * radius ** 2)
        log(f"  area of the loaded surface / 4 pi r^2 = {ratio:.6f}")
        if abs(ratio - 1.0) > 1e-3 and not args.skip_area_check:
            raise RuntimeError(
                f"the surface mesh did not load correctly (area ratio "
                f"{ratio}); run on another number of ranks than the writer")
        log(f"{args.input}: {len(indices)} steps, years {years[0]:.0f} .. "
            f"{years[-1]:.0f}, {COMM_WORLD.size} ranks")

        fields_file = VTKFile(
            os.path.join(args.output_path, "forward_exp05_fields.pvd"),
            target_degree=args.degree)
        fractions_file = VTKFile(
            os.path.join(args.output_path, "forward_exp05_fractions.pvd"),
            target_degree=1)
        # `target_degree=1` with DG0 inputs gives DG1 output points: each
        # facet fraction is written as discontinuous P1 point data with the
        # same value at the three points of a triangle, which ParaView shows
        # as one constant colour per triangle.
        last = len(indices) - 1
        for position, (index, year) in enumerate(zip(indices, years)):
            if position not in (0, last) and position % args.every:
                continue
            tic = time.time()
            continuous = [on_mesh(chk.load_function(flat, name, idx=index),
                                  curved) for name in CONTINUOUS_FIELDS]
            fractions = [on_mesh(chk.load_function(flat, name, idx=index),
                                 curved) for name in FRACTION_FIELDS]
            fields_file.write(*continuous, time=year)
            fractions_file.write(*fractions, time=year)
            log(f"  step {index:3d}  year {year:8.0f}  "
                f"({time.time() - tic:.1f} s)")
    log(f"wrote {args.output_path} in {time.time() - tic_run:.1f} s")


if __name__ == "__main__":
    main()
