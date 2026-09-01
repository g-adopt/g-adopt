"""Lower Murrumbidgee floodplain -- scaling driver.

Basin-scale benchmark (~280 x 130 km) on a terrain-following extruded
mesh with three geological layers (depth-dependent Haverkamp soil) and a
rainfall-driven top boundary condition. The mesh carries the extreme
horizontal-to-vertical aspect ratio (500:1 up to 4000:1) that the
vertically lumped preconditioners exist to handle.

Two weak-scaling families use this driver. ``murr_vertical`` holds the
horizontal resolution fixed and grows the layer count with the node
count. ``murr_seasonal`` holds the layer count fixed, halves the
horizontal resolution with the node count, and runs the three-month time
step on a near-saturated basin.

Terrain and spatial fields come from the observational bundle at
``DATA_URL``, wrapped as ``omega`` ``Surface`` objects -- the same
primitive the mesh extrusion consumes. The bundle is fetched by the doit
task that runs these cases, not by this driver: it must already be on
disk before the job starts, because compute nodes have no outbound
network and every rank would otherwise race on one download.

Requires the ``omega`` package at a revision that provides the Surface
API: ``build_mesh_hierarchy`` taking ``top_surface`` and
``thickness_surface``. Older revisions took four coordinate/value arrays
and will fail on the call below. omega is needed only to run these
cases, so it is not a dependency of g-adopt itself.

Usage:
    curl -fsSLO https://data.gadopt.org/github-actions/murrumbidgee_data.npz
    mpiexec -n 104 python murrumbidgee_3d.py \
        --horiz-res 1775 --layers 300 --solver vlumping
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
                        choices=("iterative", "vlumping",
                                 "vlumping_linesmooth", "vlumping_hmg"),
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

    # Soil-regime levers. The defaults reproduce the ordinary (daily)
    # regime; the seasonal cases raise all three to drive the
    # column-integrated diffusion number into the range where only the
    # vertically lumped presets stay cheap.
    parser.add_argument("--watertable-offset", type=float, default=0.0,
                        help="Raise the initial water table by this many "
                             "metres. Saturates more of each column.")
    parser.add_argument("--retention-flatten", type=float, default=1.0,
                        help="Divide the retention curve's specific "
                             "moisture capacity by this factor. 1.0 "
                             "leaves the Haverkamp curve unchanged.")
    parser.add_argument("--ss", type=float, default=0.0,
                        help="Specific storage in 1/m.")
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    _ARGS = _parse_args()
    sys.argv = sys.argv[:1]

import os  # noqa: E402
import time as time_mod  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from gadopt import *  # noqa: E402, F401


# Paper domain polygon (Murrumbidgee floodplain outline), metres.
_DOMAIN_VERTICES = [
    (0, 35000), (140000, 0), (280000, 0), (280000, 68000),
    (201000, 130000), (121000, 130000), (0, 100000),
]

# omega tags the polygon side boundary with physical group id 1.
_SIDE_BC_ID = 1

#: Basin terrain and field data, published alongside the other g-adopt test
#: fixtures. A compressed .npz holding five float32 fields on a shared regular
#: lattice, plus the six scalars that describe it. Storing the lattice rather
#: than a coordinate pair per point, and float32 rather than text, takes the
#: bundle from 31 MB of CSV to 3.3 MB; the round-trip error is 1.5e-05 m
#: against a 400 m elevation, which is far below anything the mesh resolves.
DATA_URL = "https://data.gadopt.org/github-actions/murrumbidgee_data.npz"
#: Resolved against this file's directory rather than the working directory.
#: Under doit the case runs with its cwd set to the case directory and the
#: bundle symlinked in, so the two coincide; anchoring to __file__ keeps a
#: manual run from any other directory working too.
DATA_FILE = Path(__file__).parent / "murrumbidgee_data.npz"


def _load_surfaces(data_file=DATA_FILE):
    """Load the basin terrain and field data as omega Surfaces.

    Every value is wrapped in a ``GridSurface``: a ``Surface`` is any pure
    callable mapping horizontal points ``(m, 2)`` to scalars ``(m,)``, and
    that is the one contract both ``build_mesh_hierarchy`` and the field
    sampling below consume. Using the same primitive for the extrusion and
    for the solution fields means the terrain is interpolated once, one way.

    ``GridSurface`` is the right primitive here rather than omega's
    Gaussian-kernel fitter: the source is a dense, regular 500 m lattice,
    finer than any mesh in the scaling ladder. There are no co-located
    conflicting picks to decluster, and a kernel mean is bounded by its
    inputs, so smoothing such a source would only flatten real relief.

    The file stores field values in ``meshgrid(indexing="ij")`` order
    against a lattice described by ``x0``, ``y0``, ``dx``, ``dy``, ``nx``
    and ``ny``. Rebuilding the coordinates from those six scalars is exact
    in float64 and is what lets the file drop two thirds of its bulk.

    Args:
        data_file: Path to the ``.npz`` bundle.

    Returns:
        A dict with keys ``elevation``, ``thickness``, ``shallow_layer``,
        ``lower_layer``, ``water_table`` and ``rainfall``.

    Raises:
        FileNotFoundError: If the bundle is absent, with the URL to fetch.
    """
    from omega import GridSurface

    path = Path(data_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The basin cases need the terrain bundle; "
            f"fetch it with: curl -fsSLO {DATA_URL}"
        )

    data = np.load(path)
    # Reconstruct the source lattice. meshgrid indexing="ij" matches the
    # order the field arrays were flattened in; getting this wrong would
    # transpose the basin rather than fail loudly.
    xs = data["x0"] + data["dx"] * np.arange(int(data["nx"]))
    ys = data["y0"] + data["dy"] * np.arange(int(data["ny"]))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    surfaces = {
        name: GridSurface(coords, data[name])
        for name in ("elevation", "thickness", "shallow_layer",
                     "lower_layer", "water_table")
    }
    # Rainfall is an orographic proxy: the source bundle's rainfall grid is
    # byte-identical to its elevation grid, so it is not stored twice.
    surfaces["rainfall"] = surfaces["elevation"]
    return surfaces


def _sample(V, V_cg, mesh_xy, surface, name):
    """Sample an omega Surface into the DG solution space.

    Surface -> CG1 nodal values -> DG by Firedrake interpolation. The
    field ends up in the same DG space as the solution, which avoids
    function-space mismatches in the nonlinear residual.

    Args:
        V: The DG space the solution lives in.
        V_cg: A CG1 space on the same mesh, used as the sampling target.
        mesh_xy: The ``(n, 2)`` horizontal coordinates of the CG1 nodes.
        surface: An ``omega.Surface``.
        name: Name given to the returned Function.

    Returns:
        A ``Function`` in ``V`` holding the sampled field.
    """
    cg = Function(V_cg)
    cg.dat.data[:] = surface(mesh_xy)
    return Function(V, name=name).interpolate(cg)


def build_mesh(horiz_res, n_layers, hmg_levels, surfaces):
    """Return the fine-level terrain-following mesh.

    Args:
        horiz_res: Horizontal resolution in metres.
        n_layers: Number of extruded layers.
        hmg_levels: Base ``MeshHierarchy`` depth. ``vlumping_hmg``'s
            coarse PCMG descends this hierarchy; the other presets use
            the fine level alone, so we pass 0 for them to keep mesh
            generation fast.
        surfaces: The dict returned by ``_load_surfaces``.

    Returns:
        The finest mesh of the hierarchy, tagged Cartesian.
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

    mh3d = build_mesh_hierarchy(
        mesh2d,
        top_surface=surfaces["elevation"],
        thickness_surface=surfaces["thickness"],
        n_layers=n_layers,
        refinement_levels=hmg_levels,
        # Horizontal-only coarsening: the vertical layer count is
        # identical on every level, which is what makes the coarse
        # problem 2D-like.
        refinement_ratio=1,
    )
    mesh = mh3d[-1]
    # Tag every level, not just the fine one: the gravity term's
    # upward_normal/is_cartesian lookup runs on the coarse grids too when
    # vlumping_hmg descends the hierarchy.
    for m in mh3d:
        m.cartesian = True
    return mesh


def model(horiz_res, n_layers, solver, *, degree=1, hmg_levels=1,
          dt_init=60.0, dt_max=43200.0, dt_growth=1.5, dt_shrink=0.5,
          t_final=2_592_000.0, max_steps=200,
          watertable_offset=0.0, retention_flatten=1.0, ss=0.0):
    """Run the basin benchmark and log per-step iteration counts.

    Args:
        horiz_res: Horizontal resolution in metres.
        n_layers: Number of extruded layers.
        solver: A ``RichardsSolver`` preset name.
        degree: Polynomial degree of the DG solution space.
        hmg_levels: Base ``MeshHierarchy`` depth for ``vlumping_hmg``.
        dt_init: Initial time step in seconds.
        dt_max: Time-step ceiling in seconds.
        dt_growth: Multiplier applied to dt after a successful step.
        dt_shrink: Multiplier applied to dt after a failed step.
        t_final: Simulation end time in seconds.
        max_steps: Hard cap on the number of accepted steps.
        watertable_offset: Metres by which to raise the initial water
            table. Saturates more of each column.
        retention_flatten: Factor dividing the specific moisture
            capacity. 1.0 leaves the Haverkamp curve unchanged.
        ss: Specific storage in 1/m.
    """
    surfaces = _load_surfaces()
    levels_for_mesh = hmg_levels if solver == "vlumping_hmg" else 0
    mesh = build_mesh(horiz_res, n_layers, levels_for_mesh, surfaces)

    # Tensor-product DG on triangular prisms.
    horiz_elt = FiniteElement("DG", triangle, degree)
    vert_elt = FiniteElement("DG", interval, degree)
    V = FunctionSpace(mesh, TensorProductElement(horiz_elt, vert_elt))
    log(f"DOFs: {V.dim()}  (dx={horiz_res}m, layers={n_layers}, "
        f"DG{degree}, preset={solver})")

    # CG1 coordinates are the sampling target for every surface.
    V_cg = FunctionSpace(mesh, "CG", 1)
    coords_cg = Function(VectorFunctionSpace(mesh, "CG", 1))
    coords_cg.interpolate(SpatialCoordinate(mesh))
    mesh_xy = coords_cg.dat.data_ro[:, :2]

    spatial = {
        name: _sample(V, V_cg, mesh_xy, surfaces[name], name)
        for name in ("elevation", "shallow_layer", "lower_layer",
                     "water_table", "rainfall")
    }

    x = SpatialCoordinate(mesh)
    elevation = spatial["elevation"]
    # Depth below ground, clamped at zero so cells poking above the
    # sampled surface do not produce a negative depth.
    depth = Function(V, name="depth")
    depth.interpolate(conditional(elevation - x[2] < 0, 0.0, elevation - x[2]))

    # Three depth-dependent geological layers, smoothed with tanh so the
    # conductivity contrast is differentiable for Newton. delta sets the
    # transition width: 0.2 /m gives a ~10 m blend across each interface.
    shallow, lower = spatial["shallow_layer"], spatial["lower_layer"]
    delta = 0.2
    I1 = 0.5 * (1 + tanh(delta * (shallow - depth)))
    I2 = 0.5 * (1 + tanh(delta * (lower - depth)))

    # Compaction with depth: porosity and conductivity both fall as the
    # sediment column is loaded. Both fits are from the basin study.
    S_depth = max_value(1 / ((1 + 0.000071 * depth) ** 5.989), 0)
    K_depth = max_value((1 - depth / (58 + 1.02 * depth)) ** 3, 0)
    Ks_shapperton, Ks_calivil, Ks_renmark = 2.5e-05, 1e-03, 5e-04
    Ks = Function(V, name="SaturatedConductivity")
    Ks.interpolate(
        K_depth * (Ks_shapperton * I1
                   + Ks_calivil * (1 - I1) * I2
                   + Ks_renmark * (1 - I2))
    )

    theta_s_expr = 0.40 * S_depth
    theta_r_base = 0.025
    # Retention flattening. The Haverkamp specific moisture capacity
    # C = dtheta/dh is proportional to (theta_s - theta_r), so raising
    # theta_r toward theta_s by the same factor scales C by exactly
    # 1/retention_flatten while leaving theta_s and K(h) untouched. A
    # large value is a soil that stays near saturation (low specific
    # yield), which shrinks the column storativity and drives the
    # column-integrated diffusion number up.
    theta_r_eff = theta_s_expr - (theta_s_expr - theta_r_base) / retention_flatten

    soil_curves = HaverkampCurve(
        theta_r=theta_r_eff, theta_s=theta_s_expr, Ks=Ks,
        alpha=0.44, beta=1.2924, A=0.0104, gamma=1.5722, Ss=ss,
    )

    # Hydrostatic initial condition from the water-table field. The
    # field is a depth below ground, so subtracting the offset raises
    # the water table and saturates more of each column.
    water_table = spatial["water_table"] - watertable_offset
    h = Function(V, name="PressureHead")
    h.interpolate(depth - water_table)

    # Rainfall flux at the top boundary. rain_scale converts the
    # elevation-proxy rainfall (mm/yr) to m/s and applies the fraction
    # that enters the ground rather than running off.
    rainfall = spatial["rainfall"]
    rain_scale = 0.14 * 3.171e-11
    richards_bcs = {
        "bottom": {"flux": 0},
        "top": {"flux": rain_scale * rainfall},
        # Robin-type side condition relaxing to the hydrostatic profile,
        # which anchors the saturated cells when Ss is zero.
        _SIDE_BC_ID: {"flux": -(h - (depth - water_table))},
    }

    log(f"Soil regime: watertable_offset = {watertable_offset} m, "
        f"retention_flatten = {retention_flatten}, Ss = {ss} /m")

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

    # Per-step diagnostics, written by rank 0 to a file named after the job
    # tag so the 40 steps of the suite can share one directory. Two physical
    # scalars are recorded alongside the solver counts:
    #
    #   theta_total  total water content, int theta(h) dx, in m^3. Sensitive
    #                to the solution everywhere the front is moving, unlike
    #                int h dx, which the deep saturated column dominates.
    #   flux_total   net boundary flux, the sum of the prescribed and
    #                solution-dependent boundary integrals, in m^3/s.
    #
    # They are logged separately rather than combined into a mass residual.
    # The Richards residual is written with the divergence terms positive,
    # so the sign a positive `flux` boundary value carries is not obvious
    # from the equation; keeping the halves apart lets the balance be formed
    # afterwards, once the convention is confirmed, and shows which side
    # moved when it fails.
    theta_expr = soil_curves.moisture_content(h)
    # Boundary flux integrals. The side condition is solution-dependent, so
    # it has to be re-assembled each step rather than scaled from a constant.
    flux_forms = [
        rain_scale * rainfall * ds_t,
        -(h - (depth - water_table)) * ds_v(_SIDE_BC_ID),
    ]
    tag = os.environ.get("TAG", "run")
    plog = ParameterLog(f"params_{tag}.log", mesh)
    plog.log_str("step time dt wall nl lin failed theta_total flux_total")

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
            # Roll back and shrink dt. Nonlinear Richards on a
            # terrain-following mesh will occasionally fail during the
            # first few ramp steps, which is expected, not a defect.
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
        # Assembled after the step is accepted, so a rolled-back failure
        # never contributes a row. Both are collective over the communicator.
        theta_total = assemble(theta_expr * dx)
        flux_total = sum(assemble(form) for form in flux_forms)
        log(f"step {step} | t={sim_time/86400:.2f}d | dt={dt_current:.1f}s | "
            f"wall={wall_times[-1]:.2f}s | NL={nl} | L={lit} | "
            f"theta={theta_total:.10e}")
        plog.log_str(
            f"{step} {sim_time:.10e} {dt_current:.10e} {wall_times[-1]:.6e} "
            f"{nl} {lit} {failed} {theta_total:.10e} {flux_total:.10e}"
        )

        dt_current = min(dt_current * dt_growth, dt_max)
        dt.assign(dt_current)

    if wall_times:
        mean_wall = sum(wall_times) / len(wall_times)
        log(f"done | total NL={total_nl} | total L={total_l} | "
            f"mean wall/step={mean_wall:.2f}s | failed={failed} | "
            f"sim_time={sim_time/86400:.2f}d")
    else:
        log(f"FAILED - no successful steps | failed={failed}")
    plog.close()


if __name__ == "__main__":
    model(
        _ARGS.horiz_res, _ARGS.layers, _ARGS.solver,
        degree=_ARGS.degree, hmg_levels=_ARGS.hmg_levels,
        dt_init=_ARGS.dt_init, dt_max=_ARGS.dt_max,
        dt_growth=_ARGS.dt_growth, dt_shrink=_ARGS.dt_shrink,
        t_final=_ARGS.t_final, max_steps=_ARGS.max_steps,
        watertable_offset=_ARGS.watertable_offset,
        retention_flatten=_ARGS.retention_flatten, ss=_ARGS.ss,
    )
