"""Lower Murrumbidgee floodplain -- scaling driver.

Real-world basin-scale benchmark (~300 x 120 km) on a terrain-following
extruded mesh with three geological layers (depth-dependent Haverkamp
soil) and rainfall-driven top BC. Used for both vertical weak scaling
(horizontal resolution fixed, layers scale with nodes) and horizontal
weak scaling (layers fixed, resolution halves with nodes).

Requires:
    - omega package for mesh generation and the terrain-following
      extruded MeshHierarchy.
    - CSV data bundle in ``--data-dir`` with columns (x, y, z) for
      ``elevation_data``, ``bedrock_data``, ``shallow_layer``,
      ``lower_layer``, ``water_table``, and ``rainfall_data``.

Usage:
    mpiexec -n 104 python murrumbidgee_3d.py \
        --horiz-res 1775 --layers 300 --solver vlumping_hmg \
        --data-dir ./murrumbidgee_data
"""


def _parse_args() -> "argparse.Namespace":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horiz-res", type=float, required=True,
                        help="Horizontal mesh resolution in metres.")
    parser.add_argument("--layers", type=int, required=True,
                        help="Number of vertical extruded layers.")
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--solver", type=str, required=True,
                        choices=("iterative", "vlumping", "vlumping_hmg"),
                        help="RichardsSolver preset name.")
    parser.add_argument("--hmg-levels", type=int, default=1,
                        help="Base MeshHierarchy depth. Only meaningful "
                             "for vlumping_hmg; other presets use the "
                             "fine level alone.")
    parser.add_argument("--dt-init", type=float, default=60.0,
                        help="Initial dt for adaptive ramp (seconds).")
    parser.add_argument("--dt-max", type=float, default=43200.0,
                        help="Maximum dt after ramp-up (seconds).")
    parser.add_argument("--dt-growth", type=float, default=1.5)
    parser.add_argument("--dt-shrink", type=float, default=0.5)
    parser.add_argument("--t-final", type=float, default=2_592_000.0,
                        help="Simulation end time (seconds, default 30 d).")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Hard cap on number of time steps.")
    parser.add_argument("--data-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    _ARGS = _parse_args()
    sys.argv = sys.argv[:1]

import time as time_mod  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

from gadopt import *  # noqa: E402, F401


# Paper domain polygon (Murrumbidgee floodplain outline).
_DOMAIN_VERTICES = [
    (0, 35000), (140000, 0), (280000, 0), (280000, 68000),
    (201000, 130000), (121000, 130000), (0, 100000),
]

# omega tags the polygon side boundary with physical group id 1.
_SIDE_BC_ID = 1


def _load_csv(path: Path):
    df = pd.read_csv(path)
    return df[["x", "y"]].values, df["z"].values


def _load_spatial_field(V, V_cg, mesh_xy, csv_path, name):
    """Project a (x, y, z) CSV into the DG solution space.

    CSV -> CG1 via ``griddata`` -> DG via Firedrake interpolation.
    Living in the same DG space as the solution avoids function-space
    mismatches in the nonlinear forms.
    """
    src_coords, src_values = _load_csv(csv_path)
    interp = griddata(src_coords, src_values, mesh_xy, method="linear")
    nan_mask = np.isnan(interp)
    if np.any(nan_mask):
        interp[nan_mask] = griddata(
            src_coords, src_values, mesh_xy[nan_mask], method="nearest"
        )
    cg = Function(V_cg)
    cg.dat.data[:] = interp
    return Function(V, name=name).interpolate(cg)


def build_mesh(horiz_res, n_layers, hmg_levels, data_dir):
    """Return the fine-level terrain-following mesh.

    ``hmg_levels`` controls the depth of the base ``MeshHierarchy``. For
    ``vlumping_hmg`` the coarse PCMG descends this hierarchy; for the
    other presets the hierarchy is harmless overhead (we still use the
    top level), but we keep it minimal (``hmg_levels=0``) in those cases
    to keep mesh generation fast.
    """
    from omega import SurfaceMesh, Polygon
    from omega.mesh.builder import build_mesh_hierarchy

    poly = Polygon(_DOMAIN_VERTICES)
    if hmg_levels > 0:
        # Generate the coarsest mesh at a coarsened resolution; omega
        # handles the horizontal refinement through the hierarchy.
        coarse_res = horiz_res * (2 ** hmg_levels)
        sm = SurfaceMesh(poly, resolution=coarse_res)
    else:
        sm = SurfaceMesh(poly, resolution=horiz_res)
    sm.generate()
    mesh2d = sm.to_firedrake_mesh()

    elev_c, elev_v = _load_csv(data_dir / "elevation_data.csv")
    bed_c, bed_v = _load_csv(data_dir / "bedrock_data.csv")

    mh3d = build_mesh_hierarchy(
        mesh2d,
        elevation_coords=elev_c, elevation_values=elev_v,
        depth_coords=bed_c, depth_values=bed_v,
        n_layers=n_layers,
        refinement_levels=hmg_levels,
        refinement_ratio=1,
    )
    return mh3d[-1]


def model(horiz_res, n_layers, solver, *, degree=1, hmg_levels=1,
          dt_init=60.0, dt_max=43200.0, dt_growth=1.5, dt_shrink=0.5,
          t_final=2_592_000.0, max_steps=200, data_dir="./murrumbidgee_data"):
    data_dir = Path(data_dir)
    levels_for_mesh = hmg_levels if solver == "vlumping_hmg" else 0
    mesh = build_mesh(horiz_res, n_layers, levels_for_mesh, data_dir)

    # Tensor-product DG on triangular prisms.
    horiz_elt = FiniteElement("DG", triangle, degree)
    vert_elt = FiniteElement("DG", interval, degree)
    V = FunctionSpace(mesh, TensorProductElement(horiz_elt, vert_elt))
    log(f"DOFs: {V.dim()}  (dx={horiz_res}m, layers={n_layers}, "
        f"DG{degree}, preset={solver})")

    # Build a CG1 coordinate field for griddata interpolation targets.
    V_cg = FunctionSpace(mesh, "CG", 1)
    coords_cg = Function(VectorFunctionSpace(mesh, "CG", 1))
    coords_cg.interpolate(SpatialCoordinate(mesh))
    mesh_xy = coords_cg.dat.data_ro[:, :2]

    spatial = {
        name: _load_spatial_field(V, V_cg, mesh_xy, data_dir / fname, name)
        for name, fname in [
            ("elevation", "elevation_data.csv"),
            ("bedrock", "bedrock_data.csv"),
            ("shallow_layer", "shallow_layer.csv"),
            ("lower_layer", "lower_layer.csv"),
            ("water_table", "water_table.csv"),
            ("rainfall", "rainfall_data.csv"),
        ]
    }

    x = SpatialCoordinate(mesh)
    elevation = spatial["elevation"]
    depth = Function(V, name="depth")
    depth.interpolate(conditional(elevation - x[2] < 0, 0.0, elevation - x[2]))

    # Three depth-dependent geological layers smoothed with tanh.
    shallow, lower = spatial["shallow_layer"], spatial["lower_layer"]
    delta = 0.2
    I1 = 0.5 * (1 + tanh(delta * (shallow - depth)))
    I2 = 0.5 * (1 + tanh(delta * (lower - depth)))

    S_depth = max_value(1 / ((1 + 0.000071 * depth) ** 5.989), 0)
    K_depth = max_value((1 - depth / (58 + 1.02 * depth)) ** 3, 0)
    Ks_shapperton, Ks_calivil, Ks_renmark = 2.5e-05, 1e-03, 5e-04
    Ks = Function(V, name="SaturatedConductivity")
    Ks.interpolate(
        K_depth * (Ks_shapperton * I1
                   + Ks_calivil * (1 - I1) * I2
                   + Ks_renmark * (1 - I2))
    )

    soil_curves = HaverkampCurve(
        theta_r=0.025, theta_s=0.40 * S_depth, Ks=Ks,
        alpha=0.44, beta=1.2924, A=0.0104, gamma=1.5722, Ss=0,
    )

    # Hydrostatic initial condition from the water-table field.
    water_table = spatial["water_table"]
    h = Function(V, name="PressureHead")
    h.interpolate(depth - water_table)

    # Rainfall flux at the top boundary; fraction-entering-ground * mm/yr -> m/s.
    rainfall = spatial["rainfall"]
    rain_scale = 0.14 * 3.171e-11
    richards_bcs = {
        "bottom": {"flux": 0},
        "top": {"flux": rain_scale * rainfall},
        _SIDE_BC_ID: {"flux": -(h - (depth - water_table))},
    }

    # Adaptive dt ramp: grow on success, shrink on solver failure.
    dt_current = dt_init
    dt = Constant(dt_current)
    diagnostics = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_converged_reason": None,
    }

    richards_solver = RichardsSolver(
        h, soil_curves, dt,
        timestepper=BackwardEuler,
        bcs=richards_bcs,
        solver_parameters=solver,
        solver_parameters_extra=diagnostics,
    )

    h_backup = Function(V, name="PressureHead_backup")
    sim_time = 0.0
    total_nl = 0
    total_l = 0
    wall_times: list[float] = []
    step = 0
    failed = 0
    while sim_time < t_final and step < max_steps:
        h_backup.assign(h)
        t0 = time_mod.perf_counter()
        try:
            richards_solver.solve()
        except Exception as exc:
            # Roll back and shrink dt; nonlinear Richards on terrain data
            # will occasionally fail at the first few ramp steps.
            failed += 1
            dt_current *= dt_shrink
            if dt_current < 1.0:
                log(f"dt shrunk below 1 s ({dt_current:.2e}), aborting")
                break
            dt.assign(dt_current)
            h.assign(h_backup)
            log(f"step {step + 1} FAILED ({exc.__class__.__name__}), "
                f"shrinking dt to {dt_current:.1f}s")
            continue
        wall_times.append(time_mod.perf_counter() - t0)
        sim_time += float(dt)
        step += 1

        snes = richards_solver.solver.snes
        nl = snes.getIterationNumber()
        lit = snes.getLinearSolveIterations()
        total_nl += nl
        total_l += lit
        log(f"step {step} | t={sim_time/86400:.2f}d | dt={dt_current:.1f}s | "
            f"wall={wall_times[-1]:.2f}s | NL={nl} | L={lit}")

        dt_current = min(dt_current * dt_growth, dt_max)
        dt.assign(dt_current)

    if wall_times:
        mean_wall = sum(wall_times) / len(wall_times)
        log(f"done | total NL={total_nl} | total L={total_l} | "
            f"mean wall/step={mean_wall:.2f}s | failed={failed} | "
            f"sim_time={sim_time/86400:.2f}d")
    else:
        log(f"FAILED - no successful steps | failed={failed}")


if __name__ == "__main__":
    model(
        _ARGS.horiz_res, _ARGS.layers, _ARGS.solver,
        degree=_ARGS.degree, hmg_levels=_ARGS.hmg_levels,
        dt_init=_ARGS.dt_init, dt_max=_ARGS.dt_max,
        dt_growth=_ARGS.dt_growth, dt_shrink=_ARGS.dt_shrink,
        t_final=_ARGS.t_final, max_steps=_ARGS.max_steps,
        data_dir=_ARGS.data_dir,
    )
