"""B2 solver probe: instrument block 0 of the coupled self-gravitating system.

Reuses B1's geometry, load and solver construction verbatim and varies only the
block-0 treatment inside `DtNTwoBlockSchurPC`. Nothing here is a physics test;
every number it prints is about the linear algebra.

    mpiexec -np 64 python b2_probe.py --config fs --configuration coarse

Configurations are in CONFIGS below. Each is a `solver_parameters_extra`
dictionary handed to `SelfGravitatingGIASolver`, so it lands on top of
`selfgrav_dtn_schur_solver_parameters` and overrides whatever it names.
"""
import argparse
import os
import sys
import time

import gadopt  # noqa: F401  BEFORE firedrake
from firedrake import COMM_WORLD  # noqa: E402
from firedrake.petsc import PETSc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_elastic as b1  # noqa: E402
import b2_pc  # noqa: E402


def log(*a):
    PETSc.Sys.Print(*a)


def rss_gb():
    """Peak resident set summed over ranks, in GB.

    `ru_maxrss` is a high-water mark per process, so the sum over ranks is the
    job's high-water mark only if the ranks peak together - they do here,
    because every phase is collective. Reported alongside PBS's own number as
    a cross-check rather than instead of it.
    """
    import resource
    from mpi4py import MPI
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2
    return COMM_WORLD.allreduce(peak, op=MPI.SUM)


GAMG = {
    "pc_type": "gamg",
    "mg_levels_pc_type": "sor",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_square_graph": 100,
    "pc_gamg_coarse_eq_limit": 1000,
    "pc_gamg_mis_k_minimum_degree_ordering": True,
}


def build_meshes_from(path):
    """B1's `build_meshes` against an explicit file rather than a named rung.

    Same two steps in the same order - read, curve, `Submesh`, curve - so a
    mesh built with different anisotropy knobs is treated identically to one
    off the ladder.
    """
    import time as _time
    from firedrake import Mesh, Submesh
    t0 = _time.perf_counter()
    parent = b1.curve(Mesh(path))
    parent.cartesian = False
    t_parent = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    sub = b1.curve(Submesh(parent, 3, b1.gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub, t_parent, _time.perf_counter() - t0


def prefixed(prefix, d):
    return {prefix + k: v for k, v in d.items()}


def block(prefix, kind, ksp="preonly", rtol=None, maxit=None):
    """Options for one diagonal block of the inner [u, m, psi] fieldsplit."""
    out = {prefix + "ksp_type": ksp}
    if rtol is not None:
        out[prefix + "ksp_rtol"] = rtol
    if maxit is not None:
        out[prefix + "ksp_max_it"] = maxit
    if ksp != "preonly":
        out[prefix + "ksp_converged_reason"] = None
    out[prefix + "pc_type"] = "python"
    if kind == "u":
        out[prefix + "pc_python_type"] = "b2_pc.RigidBodyAssembledPC"
        out.update(prefixed(prefix + "assembled_", GAMG))
    elif kind == "psi":
        out[prefix + "pc_python_type"] = "gadopt.SPDAssembledPC"
        out.update(prefixed(prefix + "assembled_", GAMG))
    elif kind == "m":
        # (m,m) is block-diagonal per cell: bjacobi/ilu is exact.
        out[prefix + "pc_python_type"] = "firedrake.AssembledPC"
        out[prefix + "assembled_pc_type"] = "bjacobi"
        out[prefix + "assembled_sub_pc_type"] = "ilu"
    else:
        raise ValueError(kind)
    return out


def inner_fieldsplit(order, split_type, inner_ksp="fgmres", inner_rtol=1e-2,
                     inner_maxit=100, block_ksp="preonly", block_rtol=None,
                     block_maxit=None):
    """Block 0 = Krylov over a fieldsplit of [u, m, psi]."""
    B = "dtn_fieldsplit_0_"
    idx = {"u": 0, "m": 1, "psi": 2}
    p = {
        B + "ksp_type": inner_ksp,
        B + "ksp_rtol": inner_rtol,
        B + "ksp_max_it": inner_maxit,
        B + "ksp_converged_reason": None,
        B + "pc_type": "fieldsplit",
        B + "pc_fieldsplit_type": split_type,
    }
    for pos, name in enumerate(order):
        p[B + f"pc_fieldsplit_{pos}_fields"] = str(idx[name])
        p.update(block(B + f"fieldsplit_{pos}_", name, ksp=block_ksp,
                       rtol=block_rtol, maxit=block_maxit))
    return p


LU = {
    "dtn_fieldsplit_0_ksp_type": "preonly",
    "dtn_fieldsplit_0_pc_type": "python",
    "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "dtn_fieldsplit_0_assembled_pc_type": "lu",
    "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
}

ILU = {
    "dtn_fieldsplit_0_ksp_type": "gmres",
    "dtn_fieldsplit_0_ksp_rtol": 1e-8,
    "dtn_fieldsplit_0_ksp_max_it": 300,
    "dtn_fieldsplit_0_ksp_converged_reason": None,
    "dtn_fieldsplit_0_ksp_monitor_singular_value": None,
    "dtn_fieldsplit_0_pc_type": "python",
    "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "dtn_fieldsplit_0_assembled_pc_type": "bjacobi",
    "dtn_fieldsplit_0_assembled_sub_pc_type": "ilu",
}

def condensed_schur(inner_rtol=1e-2, inner_maxit=300, schur_fact="full"):
    """Block 0 as an exact Schur complement onto [u, psi], eliminating m.

    `(m,m)` is block-diagonal per cell, so bjacobi/ILU(0) inverts it exactly -
    ILU has no fill to drop on a block-diagonal matrix - and the Schur
    complement onto `[u, psi]` is therefore *exactly* the statically condensed
    operator, with no form-level rewrite. See the report: the Nitsche
    consistency half is built from `eq.stress` and so contributes to the (u,m)
    block, which means it is eliminated here exactly as it would be by
    substituting the backward-Euler update into the stress; the adjoint and
    penalty halves are pure (u,u) and are eliminated by neither.
    """
    B = "dtn_fieldsplit_0_"
    p = {
        B + "ksp_type": "fgmres",
        B + "ksp_rtol": inner_rtol,
        B + "ksp_max_it": inner_maxit,
        B + "ksp_converged_reason": None,
        B + "pc_type": "fieldsplit",
        B + "pc_fieldsplit_type": "schur",
        B + "pc_fieldsplit_schur_fact_type": schur_fact,
        B + "pc_fieldsplit_schur_precondition": "a11",
        B + "pc_fieldsplit_0_fields": "1",       # m, eliminated exactly
        B + "pc_fieldsplit_1_fields": "0,2",     # u and psi, the condensed pair
    }
    p.update(block(B + "fieldsplit_0_", "m", ksp="preonly"))
    # The condensed [u, psi] pair, preconditioned by its own diagonal block.
    C = B + "fieldsplit_1_"
    p.update({
        C + "ksp_type": "fgmres",
        C + "ksp_rtol": 1e-2,
        C + "ksp_max_it": 200,
        C + "ksp_converged_reason": None,
        C + "pc_type": "fieldsplit",
        C + "pc_fieldsplit_type": "multiplicative",
        C + "pc_fieldsplit_0_fields": "0",
        C + "pc_fieldsplit_1_fields": "1",
    })
    p.update(block(C + "fieldsplit_0_", "u"))
    p.update(block(C + "fieldsplit_1_", "psi"))
    return p


CONFIGS = {
    # the shipped 2-D default, the working baseline
    "lu": LU,
    # the recorded failure, with instruments on
    "ilu": ILU,
    # the proposed design: multiplicative sweep m -> u -> psi, one V-cycle each
    "fs": inner_fieldsplit(["m", "u", "psi"], "multiplicative"),
    # orderings and split types
    "fs_upsi": inner_fieldsplit(["m", "psi", "u"], "multiplicative"),
    "fs_add": inner_fieldsplit(["m", "u", "psi"], "additive"),
    "fs_sym": inner_fieldsplit(["m", "u", "psi"], "symmetric_multiplicative"),
    # converged inner blocks: separates "the blocks are badly solved" from
    # "the coupling is not captured"
    "fs_exact": inner_fieldsplit(["m", "u", "psi"], "multiplicative",
                                 block_ksp="fgmres", block_rtol=1e-10,
                                 block_maxit=500),
    # diagonal blocks only, no sweep: measures the coupling's strength
    "fs_diag": inner_fieldsplit(["m", "u", "psi"], "additive",
                                block_ksp="fgmres", block_rtol=1e-10,
                                block_maxit=500),
    # same, but with CG: legal only if the blocks really are SPD
    "fs_diag_cg": inner_fieldsplit(["m", "u", "psi"], "additive",
                                   block_ksp="cg", block_rtol=1e-10,
                                   block_maxit=500),
    # m eliminated exactly by a Schur complement rather than by a form rewrite
    "cond": condensed_schur(),
    "cond_upper": condensed_schur(schur_fact="upper"),
    # GAMG straight onto the assembled coupled block 0, no field structure
    "gamg0": {
        "dtn_fieldsplit_0_ksp_type": "fgmres",
        "dtn_fieldsplit_0_ksp_rtol": 1e-2,
        "dtn_fieldsplit_0_ksp_max_it": 300,
        "dtn_fieldsplit_0_ksp_converged_reason": None,
        "dtn_fieldsplit_0_pc_type": "python",
        "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        **prefixed("dtn_fieldsplit_0_assembled_", GAMG),
    },
}


MULT_PC = {
    "none": {"dtn_fieldsplit_1_pc_type": "none"},
    # rung A: the exact (c,c) inverse, read off the form
    "diag": {"dtn_fieldsplit_1_pc_type": "python",
             "dtn_fieldsplit_1_pc_python_type": "b2_pc.DtNMultiplierDiagPC"},
    # rung B: the whole Schur complement, formed once at setup
    "dense": {"dtn_fieldsplit_1_pc_type": "python",
              "dtn_fieldsplit_1_pc_python_type":
                  "b2_pc.DtNMultiplierDenseSchurPC"},
}


def multiplier_pc(name):
    if name in MULT_PC:
        return dict(MULT_PC[name])
    return {"dtn_fieldsplit_1_pc_type": name}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="fs", choices=sorted(CONFIGS))
    p.add_argument("--configuration", default="coarse")
    p.add_argument("--h", type=float, default=None)
    p.add_argument("--mesh", default=None,
                   help="explicit .msh path, for meshes b2_genmesh.py built "
                        "with the anisotropy knobs varied")
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--ivdeg", type=int, default=1)
    p.add_argument("--outer-rtol", type=float, default=1e-8)
    p.add_argument("--outer-maxit", type=int, default=100)
    p.add_argument("--mult-pc", default="jacobi")
    p.add_argument("--mult-maxit", type=int, default=200)
    p.add_argument("--eig", action="store_true",
                   help="ask block 0's GMRES for its Ritz values")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--nullspace", action="store_true",
                   help="declare the rigid-rotation kernel on the coupled system")
    p.add_argument("--exact-block", choices=["u", "psi", "both"], default=None,
                   help="converge this block of the inner [u, psi] sweep "
                        "instead of applying one V-cycle; the remaining "
                        "iteration count then belongs to the other block")
    p.add_argument("--block0-only", action="store_true",
                   help="one outer iteration, one multiplier iteration: cuts "
                        "the number of block-0 applications to ~3 so a block-0 "
                        "diagnostic does not print 75 times over")
    p.add_argument("--extra", action="append", default=[],
                   help="key=value appended to solver_parameters_extra")
    args = p.parse_args()

    if args.block0_only:
        args.outer_maxit = 1
        args.mult_maxit = 1

    nmax = b1.NMAX_OF[args.configuration] if args.nmax is None else args.nmax

    log(f"=== b2_probe  config={args.config}  mesh={args.configuration}  "
        f"L={args.dtn_degree}  ranks={COMM_WORLD.size}")

    t0 = time.perf_counter()
    if args.mesh:
        parent, sub, t_parent, t_sub = build_meshes_from(args.mesh)
    else:
        parent, sub, t_parent, t_sub = b1.build_meshes(args.configuration,
                                                       h=args.h)
    log(f"  meshes: parent {parent.num_cells()} cells ({t_parent:.1f}s), "
        f"mantle {sub.num_cells()} cells ({t_sub:.1f}s)")

    extra = dict(CONFIGS[args.config])
    extra.update({
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_rtol": args.outer_rtol,
        "ksp_max_it": args.outer_maxit,
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "dtn_fieldsplit_1_ksp_type": "gmres",
        "dtn_fieldsplit_1_ksp_max_it": args.mult_maxit,
        "dtn_fieldsplit_1_ksp_converged_reason": None,
        **multiplier_pc(args.mult_pc),
    })
    if args.exact_block:
        # cond puts [u, psi] at dtn_fieldsplit_0_fieldsplit_1_, u at position 0.
        C = "dtn_fieldsplit_0_fieldsplit_1_fieldsplit_"
        for name, pos in (("u", 0), ("psi", 1)):
            if args.exact_block in (name, "both"):
                extra[f"{C}{pos}_ksp_type"] = "fgmres"
                extra[f"{C}{pos}_ksp_rtol"] = 1e-10
                extra[f"{C}{pos}_ksp_max_it"] = 800
                extra[f"{C}{pos}_ksp_converged_reason"] = None

    if args.eig:
        extra["dtn_fieldsplit_0_ksp_view_eigenvalues"] = None
    for kv in args.extra:
        k, _, v = kv.partition("=")
        if v in ("", "none", "None"):
            extra[k] = None
        else:
            try:
                extra[k] = int(v)
            except ValueError:
                try:
                    extra[k] = float(v)
                except ValueError:
                    extra[k] = v

    t0 = time.perf_counter()
    solver, z, layout, *_ = b1.build_solver(
        parent, sub, nmax, dtn_degree=args.dtn_degree, ivdeg=args.ivdeg,
        block0="lu", declare_nullspace=args.nullspace,
        solver_parameters_extra=extra)
    t_build = time.perf_counter() - t0
    dims = {"u": z.subfunctions[layout.displacement].function_space().dim(),
            "m": sum(z.subfunctions[i].function_space().dim()
                     for i in layout.internal_variables),
            "psi": z.subfunctions[layout.potential].function_space().dim()}
    log(f"  solver built in {t_build:.1f}s; {z.function_space().dim()} dofs in "
        f"{len(z.subfunctions)} fields")
    log(f"    block 0 = u {dims['u']} + m {dims['m']} + psi {dims['psi']} "
        f"= {sum(dims.values())}, plus {len(layout.multipliers)} Real")

    if args.mult_pc == "diag":
        b2_pc.set_dtn_diagonal(b2_pc.dtn_diagonal_from_solver(solver, layout))

    if args.setup_only:
        return

    t0 = time.perf_counter()
    failed = None
    try:
        solver.solve()
    except Exception as exc:                       # noqa: BLE001
        failed = repr(exc)
    t_solve = time.perf_counter() - t0
    snes = solver.solver.snes

    log(f"  RSS after solve {rss_gb():.2f} GB summed over ranks")
    log(f"  RESULT config={args.config} mesh={args.configuration} "
        f"t_solve={t_solve:.1f}s snes_its={snes.getIterationNumber()} "
        f"snes_reason={snes.getConvergedReason()} "
        f"ksp_its={snes.ksp.getIterationNumber()} "
        f"ksp_reason={snes.ksp.getConvergedReason()}")
    if failed:
        log(f"  RAISED {failed}")

    if failed is None:
        u = solver.displacement
        with u.dat.vec_ro as v:
            log(f"  |u|_2 = {v.norm():.8e}")
        psi = z.subfunctions[layout.potential]
        with psi.dat.vec_ro as v:
            log(f"  |psi|_2 = {v.norm():.8e}")


if __name__ == "__main__":
    main()
