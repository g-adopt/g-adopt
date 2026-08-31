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

The terrain and the spatial fields are analytic, not observational. They
are built as ``omega`` ``Surface`` objects, the same primitive the mesh
extrusion consumes, so the basin has the correct shape, stratigraphy and
aspect ratio with no data bundle in the repository. See ``_SURFACES``
below for the values they reproduce.

CAUTION: because the fields are analytic, the iteration counts and
timings this driver produces are NOT the observational basin of Morrow
et al. (2026). They are a self-consistent regression baseline for the
solver presets on a basin-shaped problem. Do not compare them against
the numbers in the manuscript.

Requires the ``omega`` package at a revision that provides the Surface
API: ``build_mesh_hierarchy`` taking ``top_surface`` and
``thickness_surface``. Older revisions took four coordinate/value arrays
and will fail on the call below.

Usage:
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

import time as time_mod  # noqa: E402

import numpy as np  # noqa: E402

from gadopt import *  # noqa: E402, F401


# Paper domain polygon (Murrumbidgee floodplain outline), metres.
_DOMAIN_VERTICES = [
    (0, 35000), (140000, 0), (280000, 0), (280000, 68000),
    (201000, 130000), (121000, 130000), (0, 100000),
]

# omega tags the polygon side boundary with physical group id 1.
_SIDE_BC_ID = 1

# Bounding box of the polygon above, used to normalise the analytic
# surfaces onto the unit square. Both are in metres.
_DOMAIN_LENGTH = 280000.0
_DOMAIN_WIDTH = 130000.0


def _analytic_surfaces():
    """Return the analytic terrain and field surfaces for the basin.

    Every surface is an ``omega.Surface``: a pure callable mapping
    horizontal points ``(m, 2)`` to scalar values ``(m,)``. That is the
    same contract ``build_mesh_hierarchy`` consumes for the extrusion, so
    the terrain and the solution fields are sampled through one
    primitive.

    The expressions are chosen to reproduce the range and the spatial
    character of the observational basin without carrying its data:

    ==================  ===========  ================
    surface             this driver  observed basin
    ==================  ===========  ================
    ground elevation    60-414 m     64-416 m
    sediment thickness  115-374 m    124-424 m
    shallow interface   ~46 m deep   15-83 m
    lower interface     ~104 m deep  56-153 m
    water-table depth   13-31 m      9-47 m
    ==================  ===========  ================

    Ground elevation falls steeply from the upstream (west) edge and
    flattens across the floodplain, which is the feature that sets the
    terrain-following mesh's vertical grading. Sediment thickness runs
    the other way, deepening downstream as the basin opens out.

    The two layer interfaces are defined as fixed fractions of the
    sediment thickness rather than as independent surfaces. That
    guarantees ``0 < shallow < lower < thickness`` everywhere by
    construction, so the tanh layer indicators below can never invert and
    no monotonicity repair is needed.

    Returns:
        A dict with keys ``elevation``, ``thickness``, ``shallow_layer``,
        ``lower_layer``, ``water_table`` and ``rainfall``.
    """
    from omega import Surface

    class _AnalyticSurface(Surface):
        """A Surface backed by a closed-form expression in (x, y).

        Satisfies omega's requirement that a surface be pure,
        deterministic and replicated on every MPI rank: there is no
        state beyond the function itself, so every rank evaluating the
        same nodes gets bit-identical geometry.
        """

        def __init__(self, fn, label):
            self._fn = fn
            self._label = label

        def __call__(self, xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=float)
            # Normalise onto the unit square so the expressions below
            # read as shape functions rather than as metre-scaled magic.
            u = xy[:, 0] / _DOMAIN_LENGTH
            v = xy[:, 1] / _DOMAIN_WIDTH
            return self._fn(u, v)

        def __repr__(self) -> str:
            return f"_AnalyticSurface({self._label})"

    # Ground elevation: a steep upstream rise decaying eastward, plus a
    # low-amplitude relief mode so the terrain is not a pure ramp.
    elevation = _AnalyticSurface(
        lambda u, v: 72.0 + 330.0 * np.exp(-6.0 * u)
        + 12.0 * np.sin(3.0 * np.pi * u) * np.cos(2.0 * np.pi * v),
        "elevation",
    )

    # Depth to bedrock, positive metres below ground. omega's extrusion
    # maps normalised z in [0, 1] to thickness * z + top - thickness, so
    # z = 0 lands on bedrock and z = 1 on the ground surface.
    thickness = _AnalyticSurface(
        lambda u, v: 140.0 + 220.0 * (1.0 - np.exp(-3.0 * u))
        + 25.0 * np.cos(2.0 * np.pi * v),
        "thickness",
    )

    # Layer interfaces as fractions of the sediment column. The
    # fractions are set so the mean interface depths land on the
    # observed basin's (~46 m and ~104 m).
    shallow_layer = thickness * 0.16
    lower_layer = thickness * 0.36

    # Depth to the water table, positive metres below ground. Shallow
    # everywhere, which is what keeps the basin near saturation.
    water_table = _AnalyticSurface(
        lambda u, v: 22.0 + 9.0 * np.sin(np.pi * u) * np.cos(np.pi * v),
        "water_table",
    )

    return {
        "elevation": elevation,
        "thickness": thickness,
        "shallow_layer": shallow_layer,
        "lower_layer": lower_layer,
        "water_table": water_table,
        # Rainfall is taken proportional to ground elevation, the same
        # orographic proxy the observational bundle uses (its rainfall
        # grid is the elevation grid).
        "rainfall": elevation,
    }


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
        surfaces: The dict returned by ``_analytic_surfaces``.

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
    surfaces = _analytic_surfaces()
    levels_for_mesh = hmg_levels if solver == "vlumping_hmg" else 0
    mesh = build_mesh(horiz_res, n_layers, levels_for_mesh, surfaces)

    # Tensor-product DG on triangular prisms.
    horiz_elt = FiniteElement("DG", triangle, degree)
    vert_elt = FiniteElement("DG", interval, degree)
    V = FunctionSpace(mesh, TensorProductElement(horiz_elt, vert_elt))
    log(f"DOFs: {V.dim()}  (dx={horiz_res}m, layers={n_layers}, "
        f"DG{degree}, preset={solver})")

    # CG1 coordinates are the sampling target for every analytic surface.
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
        watertable_offset=_ARGS.watertable_offset,
        retention_flatten=_ARGS.retention_flatten, ss=_ARGS.ss,
    )
