"""Preconditioner sweep for the near-incompressibility conditioning of the
segregated GIA mechanics.

Same b1 Spada mantle fixture as `b1_elastic --mechanics-only`, but run as a
clean solver benchmark rather than a physics check: the segregated
`InternalVariableSolver` on the mantle alone, `un = 0` at the CMB, self-gravity
off. What is swept is the bulk/shear ratio -- from nu = 0.28 (where the
self-gravity tests run) up into the near-incompressible band nu -> 0.5 where the
coupled solver diverges today -- against three u-block preconditioners:

  * firedrake.AssembledPC          -- GAMG, no near-nullspace (the b1 baseline)
  * gadopt.RigidBodyAssembledPC    -- GAMG + the six rigid-body modes
  * gadopt.NearlyIncompressibleAssembledPC
                                   -- GAMG + rigid + divergence-free modes (A3)

Because the internal variable is eliminated by the backward-Euler substitution,
the operator's ratio is bulk_shear_ratio*(1 + dt/tau); at the elastic snapshot
dt/tau ~ 1e-4 that is the nominal ratio to four figures, so sweeping
`bulk_shear_ratio` sweeps the operator directly. This is the incompressibility
axis only -- the preconditioner is designed for the (u,u) block, and the
self-gravity coupling is deliberately absent so the effect is not confounded.

Metric: outer KSP iterations. On this single-field CG/FGMRES + GAMG solve one
GAMG V-cycle is applied per Krylov iteration, so the iteration count *is* the
AMG-application count here (unlike the coupled solver, where it is not). Wall
time is reported alongside. Not a physics check; the solution is discarded.

Usage (Firedrake python, PYTHONPATH at this worktree):
  python b1_pc_sweep.py [coarse|medium] [nmax] [max_it]
"""
import sys
import time

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import b1_elastic as b1  # noqa: E402  reuse its mesh/field helpers verbatim
from firedrake import (COMM_WORLD, Function, SpatialCoordinate,  # noqa: E402
                       TensorFunctionSpace, VectorFunctionSpace, dot, sqrt)
from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                    InternalVariableSolver, gamg_parameters, rigid_body_modes)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "coarse"
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else b1.NMAX_OF[CONFIG]
MAX_IT = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
# Optional explicit mesh path (a b2_genmesh low-anisotropy mesh); when given,
# CONFIG is used only for the load degree, not for choosing the mesh.
MESH = sys.argv[4] if len(sys.argv) > 4 else None

RATIOS = [1.9394, 100.0, 1000.0]
PCS = {
    "AssembledPC": "firedrake.AssembledPC",
    "RigidBody": "gadopt.RigidBodyAssembledPC",
    "NearlyIncompressible": "gadopt.NearlyIncompressibleAssembledPC",
}


def log(msg):
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


def build_meshes_from(path):
    """B1's build_meshes against an explicit .msh -- the same read, curve,
    Submesh, curve steps in the same order, so the fixture matches b2_probe."""
    from firedrake import Mesh, Submesh
    parent = b1.curve(Mesh(path))
    parent.cartesian = False
    sub = b1.curve(Submesh(parent, 3, b1.gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


if MESH is not None:
    parent, sub = build_meshes_from(MESH)
else:
    parent, sub, _, _ = b1.build_meshes(CONFIG, curve_enabled=True)
sigma_sub = b1.load_field(sub, NMAX, b1.cap_sigma_hat(NMAX))
rho = b1.layered(sub, 2, "density")
mu = b1.layered(sub, 3, "shear_modulus")
eta = b1.layered(sub, 4, "viscosity")
rm = sqrt(dot(SpatialCoordinate(sub), SpatialCoordinate(sub)))

V = VectorFunctionSpace(sub, "CG", 2)
S = TensorFunctionSpace(sub, "DG", 1)

log(f"# b1_pc_sweep  config={CONFIG}  nmax={NMAX}  "
    f"mesh={MESH or CONFIG}  u-dofs={V.dim()}  "
    f"ranks={COMM_WORLD.size}  max_it={MAX_IT}")
log(f"# {'ratio':>9} {'nu':>7} | " + " | ".join(f"{k:>24}" for k in PCS))


def run(ratio, pc_python):
    log(f"\n{'#' * 78}\n# SOLVE  ratio={ratio}  pc={pc_python}\n{'#' * 78}")
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=ratio, g=b1.refstate.gravity_exact_ufl(rm), B_mu=b1.B_MU)
    u = Function(V)
    m = Function(S)
    params = {
        "mat_type": "matfree", "snes_type": "ksponly",
        "ksp_type": "fgmres", "ksp_rtol": 1e-8, "ksp_max_it": MAX_IT,
        # PETSc view/monitor: the full solver tree (GAMG levels, smoothers, and
        # -- for the two gadopt PCs -- the near-nullspace attached to the block),
        # the per-iteration true residual, and why the solve stopped. Printed for
        # every (ratio, PC) so the three preconditioners can be told apart in the
        # log and the near-nullspace enrichment confirmed present.
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_view": None,
        "pc_type": "python", "pc_python_type": pc_python,
        **gamg_parameters("assembled_"),
    }
    solver = InternalVariableSolver(
        u, approx, dt=b1.DT_ELASTIC, internal_variables=[m],
        bcs={b1.gen.SURF_RE: {"normal_stress": b1.B_MU * sigma_sub},
             b1.gen.SURF_RC: {"un": 0.0}},
        solver_parameters=params,
        nullspace=rigid_body_modes(V, rotational=True),
        transpose_nullspace=rigid_body_modes(V, rotational=True))
    t0 = time.perf_counter()
    try:
        solver.solve()
        ksp = solver.solver.snes.ksp
        its, reason = ksp.getIterationNumber(), ksp.getConvergedReason()
        dt = time.perf_counter() - t0
        if reason > 0:
            return f"{its:>4} it {dt:6.0f}s"
        return f"DIVERGED({reason}) {its}it"
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        return f"FAIL:{type(exc).__name__}"


for ratio in RATIOS:
    nu = (3 * ratio - 2) / (2 * (3 * ratio + 1))
    cells = {name: run(ratio, pc) for name, pc in PCS.items()}
    log(f"  {ratio:>9.3f} {nu:>7.4f} | "
        + " | ".join(f"{cells[k]:>24}" for k in PCS))

log("DONE")
