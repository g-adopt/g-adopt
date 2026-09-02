"""Cockett et al. (2018) 3D heterogeneous infiltration -- scaling driver.

Weak-scaling test exercising the g-adopt default iterative presets for
``RichardsSolver`` on a synthetic 2 x 2 x 2.6 m box. Soil heterogeneity
follows a tanh band of sand/loamy-sand contrasts; BCs are Dirichlet top
and bottom, no-flux on the sides.

Reference:
    Cockett, R., Heagy, L. J., & Haber, E. (2018). Efficient 3D inversions
    using the Richards equation. Computers & Geosciences, 116, 91-102.

Usage:
    mpiexec -n 104 python cockett_3d.py --nx 120 --nz 156 \
        --solver vlumping --steps 30
"""


def _parse_args() -> "argparse.Namespace":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True,
                        help="Horizontal cells per dimension (fine level).")
    parser.add_argument("--nz", type=int, required=True,
                        help="Number of vertical layers.")
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--dt", type=float, default=300.0,
                        help="Time step in seconds.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--solver", type=str, required=True,
                        choices=("iterative", "vlumping", "vlumping_hmg"),
                        help="RichardsSolver preset name.")
    parser.add_argument("--hmg-levels", type=int, default=2,
                        help="MeshHierarchy depth used by the vlumping_hmg "
                             "preset. Ignored for other presets.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse before importing Firedrake; PETSc consumes sys.argv at import
    # and emits "unused options" warnings for anything it doesn't recognise.
    import sys
    _ARGS = _parse_args()
    sys.argv = sys.argv[:1]

import time as time_mod  # noqa: E402

from gadopt import *  # noqa: E402, F401


def build_mesh(nx: int, nz: int, solver: str, hmg_levels: int):
    """Return the fine-level extruded mesh for a given preset.

    ``vlumping_hmg`` needs the 2D base mesh to live in a ``MeshHierarchy``
    so the coarse PCMG can descend a geometric hierarchy on the base.
    ``nx`` is the *fine-level* horizontal cell count; the coarsest level
    has ``nx // 2**hmg_levels`` cells per direction and must divide evenly.
    """
    Lx, Ly, Lz = 2.0, 2.0, 2.6

    if solver == "vlumping_hmg":
        divisor = 2 ** hmg_levels
        if nx % divisor:
            raise ValueError(
                f"--nx={nx} must be divisible by 2**hmg_levels={divisor} "
                f"so the coarse 2D mesh has a whole number of cells."
            )
        nx_coarse = nx // divisor
        coarse = RectangleMesh(nx_coarse, nx_coarse, Lx, Ly, quadrilateral=True)
        base_hierarchy = MeshHierarchy(coarse, hmg_levels)
        mh3d = ExtrudedMeshHierarchy(
            base_hierarchy, Lz, base_layer=nz,
            refinement_ratio=1,  # horizontal-only coarsening
            extrusion_type="uniform",
        )
        mesh = mh3d[-1]
    else:
        base = RectangleMesh(nx, nx, Lx, Ly, quadrilateral=True)
        mesh = ExtrudedMesh(base, nz, layer_height=Lz / nz)

    mesh.cartesian = True
    return mesh


def model(nx, nz, solver, *, degree=1, dt_value=300.0, steps=30, hmg_levels=2):
    mesh = build_mesh(nx, nz, solver, hmg_levels)
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", degree)
    log(f"DOFs: {V.dim()}  (mesh {nx}x{nx}x{nz}, DQ{degree}, preset={solver})")

    # Heterogeneous soil indicator (sharp tanh transition).
    r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837,
         0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
    epsilon = 1 / 500
    I = (sin(3 * (X[0] - r[0])) + sin(3 * (X[1] - r[1]))
         + sin(3 * (X[2] - r[2])) + sin(3 * (X[0] - r[3]))
         + sin(3 * (X[1] - r[4])) + sin(3 * (X[2] - r[5]))
         + sin(3 * (X[0] - r[6])) + sin(3 * (X[1] - r[7]))
         + sin(3 * (X[2] - r[8])))
    I = 0.5 * (1 + tanh(I / epsilon))

    soil_curves = VanGenuchtenCurve(
        theta_r=0.02 * I + 0.035 * (1 - I),
        theta_s=0.417 * I + 0.401 * (1 - I),
        Ks=5.82e-05 * I + 1.69e-05 * (1 - I),
        alpha=13.8 * I + 11.5 * (1 - I),
        n=1.592 * I + 1.474 * (1 - I),
        Ss=0,
    )

    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        boundary_ids.left: {"flux": 0},
        boundary_ids.right: {"flux": 0},
        boundary_ids.back: {"flux": 0},
        boundary_ids.front: {"flux": 0},
        boundary_ids.bottom: {"h": -0.3},
        boundary_ids.top: {"h": -0.1},
    }

    h = Function(V, name="PressureHead")
    h.interpolate(0.2 * exp(5 * (X[2] - 2.6)) - 0.3)

    dt = Constant(dt_value)
    # Diagnostics needed by the scaling-test parser (`scaling.get_data`).
    # -log_view is enabled by run.template via PETSC_OPTIONS, so the
    # driver itself only needs the per-step iteration counts.
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

    sim_time = 0.0
    total_nl = 0
    total_l = 0
    wall_times: list[float] = []
    for step in range(steps):
        t0 = time_mod.perf_counter()
        richards_solver.solve()
        wall_times.append(time_mod.perf_counter() - t0)
        sim_time += float(dt)

        snes = richards_solver.solver.snes
        nl = snes.getIterationNumber()
        lit = snes.getLinearSolveIterations()
        total_nl += nl
        total_l += lit
        log(f"step {step + 1}/{steps} | t={sim_time:.1f}s | "
            f"wall={wall_times[-1]:.2f}s | NL={nl} | L={lit}")

    mean_wall = sum(wall_times) / len(wall_times)
    log(f"done | total NL={total_nl} | total L={total_l} | "
        f"mean wall/step={mean_wall:.2f}s")


if __name__ == "__main__":
    model(
        _ARGS.nx, _ARGS.nz, _ARGS.solver,
        degree=_ARGS.degree, dt_value=_ARGS.dt, steps=_ARGS.steps,
        hmg_levels=_ARGS.hmg_levels,
    )
