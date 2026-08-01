"""Generate A2 meshes with the anisotropy knobs varied, for the B2 probe.

A2's generator holds the lithosphere at 35 km radial spacing at every rung,
because `litho_layers` and `min_cells_per_great_circle` are configuration-
independent defaults while only lateral `h` changes. The cell aspect ratio in
the lithosphere is therefore roughly `h_lateral / h_radial`, which *falls* as
the mesh refines:

    coarse  500 km / 35 km = 14.3        measured max aspect ratio 24.6
    medium  250 km / 35 km =  7.1                                 11.2
    fine    120 km / 35 km =  3.4                                  9.8

This script calls `generate_selfgrav_sphere.generate` at the *call site* with
`litho_layers` and `min_cells_per_great_circle` scaled alongside the
configuration, so a matched-anisotropy pair can be built and the mesh-quality
hypothesis separated from problem size. It edits nothing: refdata owns the
generator.

    python b2_genmesh.py --tag medium_ar14 --configuration medium \
        --litho-layers 4 --min-cells 64
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_selfgrav_sphere as gen  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True,
                   help="written to b2_<tag>.msh next to this script")
    p.add_argument("--configuration", default="coarse")
    p.add_argument("--litho-layers", type=int, default=2)
    p.add_argument("--min-cells", type=int, default=32)
    p.add_argument("--grade", type=float, default=2.0)
    args = p.parse_args()

    path = os.path.join(HERE, f"b2_{args.tag}.msh")
    t0 = time.perf_counter()
    gen.generate(path, configuration=args.configuration,
                 litho_layers=args.litho_layers,
                 min_cells_per_great_circle=args.min_cells,
                 grade=args.grade)
    dt = time.perf_counter() - t0
    h_lat = gen.lateral_spacing(args.configuration)
    # `litho_layers` layers are placed across `RE - R_LITHO`, so the radial
    # spacing and hence the design aspect ratio are both known before the mesh
    # is read. Taken from the generator's own constants rather than the 70 km
    # of the prose, so the two cannot drift.
    h_rad = (gen.RE - gen.R_LITHO) / args.litho_layers
    print(f"wrote {path} in {dt:.1f}s: {args.configuration}, "
          f"litho_layers={args.litho_layers}, min_cells={args.min_cells}, "
          f"h_lat={h_lat:.4f}, h_rad={h_rad:.4f}, "
          f"design aspect ratio {h_lat / h_rad:.1f}", flush=True)


if __name__ == "__main__":
    main()
