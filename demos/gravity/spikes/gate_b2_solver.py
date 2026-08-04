"""B2: a composed iterative solver for the coupled 3-D system, and its counts.

`NOTES/REVIEW-SPADA-B2-PRECOMMIT.md` designs a solver for the self-gravitating
GIA system out of the two presets this codebase already runs iteratively, and
commits its iteration counts in advance.  This script builds it and measures
them.  **MUMPS is not a candidate at any 3-D resolution here**, so nothing below
falls back on a direct solve; where a block is inverted exactly it is because it
is block-diagonal per cell.

## The structure

    outer FGMRES
      +- DtNTwoBlockSchurPC                        [ (u, m, psi) | (c, r) ]
      |    +- block 0: FGMRES + fieldsplit multiplicative over [m, u, psi]
      |    |     +- m   : bjacobi/ilu     (block-diagonal per cell -> exact)
      |    |     +- u   : AssembledPC/GAMG + rigid-body near-nullspace
      |    |     +- psi : SPDAssembledPC/GAMG   (GravitySolver's preset verbatim)
      |    +- block 1: GMRES on the 75 Real, preconditioned per --multiplier-pc
      +- (the Real block is where DtNTwoBlockSchurPC takes its Schur complement)

`m` is swept **first** because it is exact and nearly free, so it should never
be the last thing updated; `u` before `psi` because the load enters through `u`.

## What is measured, and the falsification test

Committed in `REVIEW-SPADA-B2-PRECOMMIT.md` §9, before any of this ran:

| level | expected | condemns the design if |
|---|---|---|
| psi standalone | 15-30 | > 60, or growing with mesh |
| u standalone at nu = 0.49 | 40-80 | growing faster than sqrt(ratio) |
| m (uncondensed) | 1 | anything else |
| **block-0 FGMRES** | **5-20** | **grows between --coarse and --medium** |
| multiplier GMRES, jacobi | 5-15 | > 30 |
| outer FGMRES | 2-5 | > 15 |

> **The falsification test is mesh-independence of the block-0 FGMRES count.**
> Flat between the two rungs, the design works and scales.  Growing with
> resolution, the coupling is not being captured, there is no fix short of a
> genuinely coupled multigrid, and "no plausible iterative form at Lambda ~ 1"
> is the honest outcome — which redirects the project to Track 1.

## Two departures from the design, both recorded rather than worked around

**The internal variable is NOT statically condensed.**  §1 calls condensation a
precondition rather than an optimisation: `TensorFunctionSpace(mesh, "DG", 1)`
is 36 dofs per cell, 8.8x the displacement, and uncondensed it is 83 % of all
Krylov traffic.  But condensation is a change under `gadopt/`, which this agent
does not own, so the design's own stated fallback is used instead: `m` as a
third field in the block-0 sweep with `bjacobi`/`ilu`, which on a block-diagonal
operator is exact.  **Every count below is therefore for the uncondensed system**,
and the condensed one would have the same iteration counts on `u` and `psi` with
6x less vector traffic.  Iteration counts are what B2 is for, so the departure
costs time and memory but not the answer.

**`rigid_body_modes` is absent from `coupled_gia_solver_parameters` entirely**
(§3), and it cannot be supplied as an options entry — it is a `near_nullspace`
argument on the displacement sub-space.  It is built here.  It cannot fix the
penalty conditioning and nothing in a near-nullspace can (§6), but it is
necessary for the deviatoric part regardless and rigid modes are divergence
free, so they lie in the penalty's near-null space too: incomplete, not wrong,
and it can only help.

## Counting

Every level is given `ksp_converged_reason`, and PETSc prints one line per
*application* naming its own options prefix.  So the parse yields both the
iteration count and the number of applications — and the second is what sets
the cost, since block-0 applications per outer iteration is the product that
decides the node-hours.  Reading `getIterationNumber()` after the fact would
give only the last application of each level and would silently miss the
multiplier loop, which is exactly where §5 expects the 15x lever to be.
"""

import argparse
import os
import re
import resource
import sys
import time

import numpy as np

import gadopt  # noqa: F401  (before firedrake; Irksome's import-order guard)
from gadopt import (
    CompressibleInternalVariableApproximation,
    SelfGravitatingGIASolver,
    SphericalDtN,
    rigid_body_modes,
    self_gravitating_gia_space,
)
from firedrake import (
    COMM_WORLD,
    Constant,
    Function,
    FunctionSpace,
    Measure,
    Mesh,
    MixedVectorSpaceBasis,
    SpatialCoordinate,
    Submesh,
    VectorSpaceBasis,
    assemble,
    dot,
    sqrt,
)
from firedrake.petsc import PETSc
from gadopt import RigidBodyAssembledPC as _RigidBodyAssembledPC
from gadopt.solver_options_manager import GAMG_PARAMETERS


#: This spike is where `RigidBodyAssembledPC` was found and first written. It
#: has since been promoted to `gadopt.preconditioners`, and the local copy is
#: gone rather than kept in sync: two identical definitions of a preconditioner
#: reached by *string* from an options dictionary is precisely the arrangement
#: in which one of them quietly stops matching the other. The name stays bound
#: here so that `u_pc="__main__.RigidBodyAssembledPC"` - this module's own
#: default, and what every recorded B2 measurement was taken with - still
#: resolves when the spike is run as a script.
RigidBodyAssembledPC = _RigidBodyAssembledPC

HERE = os.path.dirname(os.path.abspath(__file__))
SPADA = os.path.normpath(os.path.join(
    HERE, "..", "..", "glacial_isostatic_adjustment", "3d_spada_selfgrav"))
sys.path.insert(0, SPADA)

import generate_selfgrav_sphere as gen  # noqa: E402

# Non-dimensional constants, from the 2-D driver; Lambda ~ 1 is the point.
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3
LOAD_DEGREE = 2

#: K/mu = 2(1+nu)/(3(1-2nu)) at nu = 0.28 and 0.49, the ladder A5 also sweeps.
NU_TO_RATIO = {0.28: 1.94, 0.45: 9.67, 0.49: 49.7, 0.499: 499.7}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
class Phases:
    """Wall time per named phase, kept apart from the solve.

    A2 measured `Mesh()` at 49 s and `Submesh` at 28 s at 113 653 cells, which
    linear extrapolation puts at ~50 min at `--fine` and is largely serial.
    Folding that into a solver cost would flatter the solver.
    """

    def __init__(self):
        self.t = {}
        self._order = []

    def __call__(self, name):
        return _Phase(self, name)

    def report(self):
        print("\n  setup, by phase")
        for name in self._order:
            print(f"    {name:<34} {self.t[name]:8.2f} s")
        print(f"    {'TOTAL SETUP':<34} {sum(self.t.values()):8.2f} s")


class _Phase:
    def __init__(self, phases, name):
        self.phases, self.name = phases, name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        self.phases.t[self.name] = self.phases.t.get(self.name, 0.0) + dt
        if self.name not in self.phases._order:
            self.phases._order.append(self.name)
        print(f"    [{self.name}] {dt:.2f} s", flush=True)
        return False


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


# ---------------------------------------------------------------------------
# the options dictionary
# ---------------------------------------------------------------------------
#: `gadopt.solver_options_manager.GAMG_PARAMETERS`, imported rather than
#: copied. This spike's recorded measurements were taken with these exact six
#: values, and they are the shipped ones; if the library's ever change, this
#: spike's numbers stop being reproducible, and that is a fact worth having
#: surface as a changed result rather than hiding behind a stale local copy.
GAMG = dict(GAMG_PARAMETERS)


def _prefixed(d, prefix):
    return {prefix + k: v for k, v in d.items()}


def solver_parameters(multiplier_pc="jacobi", block0_type="multiplicative",
                      block0_rtol=1e-2, inner_ksp="preonly", inner_its=1,
                      outer_rtol=1e-6, block0_max_it=60,
                      u_pc="__main__.RigidBodyAssembledPC"):
    """§8's dictionary, with the uncondensed three-field block-0 sweep.

    **`fieldsplit_N_` indexes the SPLIT, not the field.**  This cost a run.
    The sweep order is `m, u, psi` (§4), while the mixed space's own field
    order is `u = 0, m = 1, psi = 2`, so
    `pc_fieldsplit_0_fields: "1"` makes split *0* the *m* field — and the
    options under `fieldsplit_0_` then go to `m`, not to field 0.  Writing the
    two out by hand put GAMG on the 2.9M-dof DG1 tensor block and `bjacobi/ilu`
    on the displacement, which does not raise, does not warn, and shows up only
    as block-0 FGMRES hitting its iteration cap.  The sweep is therefore built
    from one list below and every prefix is derived from its position, so the
    two can no longer disagree.
    """
    p = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-15,
        "snes_rtol": 1e-4,
        "snes_converged_reason": None,

        "ksp_type": "fgmres",
        "ksp_rtol": outer_rtol,
        "ksp_max_it": 200,
        "ksp_converged_reason": None,

        "pc_type": "python",
        "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        "dtn_pc_fieldsplit_schur_fact_type": "full",

        # ---- block 0 : the physical fields ----
        "dtn_fieldsplit_0_ksp_type": "fgmres",
        "dtn_fieldsplit_0_ksp_rtol": block0_rtol,
        "dtn_fieldsplit_0_ksp_max_it": block0_max_it,
        "dtn_fieldsplit_0_ksp_converged_reason": None,
        "dtn_fieldsplit_0_pc_type": "fieldsplit",
        "dtn_fieldsplit_0_pc_fieldsplit_type": block0_type,

        # ---- block 1 : the 75 Real ----
        "dtn_fieldsplit_1_ksp_type": "gmres",
        "dtn_fieldsplit_1_ksp_rtol": 1e-4,
        "dtn_fieldsplit_1_ksp_max_it": 200,
        "dtn_fieldsplit_1_ksp_converged_reason": None,
    }
    # §5(a) asked for `jacobi`; that needs MatGetDiagonal, which on a matfree
    # block goes through TSFC's `diagonal=True` path and cannot handle `Real`
    # arguments. But the diagnosis points at the ROUTE, not the object:
    # ordinary assembly of Real-Real blocks works (the DtN constraint rows
    # assemble every iteration), so `AssembledPC` + `lu` forms the 75x75
    # through the normal path and inverts it EXACTLY -- strictly better than
    # the jacobi that was asked for, which only approximated the same diagonal.
    if multiplier_pc == "assembled_lu":
        p.update({
            "dtn_fieldsplit_1_pc_type": "python",
            "dtn_fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
            "dtn_fieldsplit_1_assembled_pc_type": "lu",
        })
    else:
        p["dtn_fieldsplit_1_pc_type"] = multiplier_pc

    def inner(pc_python, extra=None):
        d = {"ksp_type": inner_ksp, "pc_type": "python",
             "pc_python_type": pc_python, "ksp_converged_reason": None}
        if inner_ksp != "preonly":
            d["ksp_max_it"] = inner_its   # the cap is the real control
            d["ksp_rtol"] = 1e-12
        d.update(_prefixed(extra or GAMG, "assembled_"))
        return d

    # (field index in the mixed space, name, options).  Order IS the sweep
    # order: `m` first because it is exact and nearly free, so it should never
    # be the last thing updated; `u` before `psi` because the load enters
    # through `u` (§4).
    sweep = [
        # m : block-diagonal per cell, so ilu on the cell blocks is exact
        (1, "m", inner("firedrake.AssembledPC",
                       {"pc_type": "bjacobi", "sub_pc_type": "ilu"})),
        # u : AssembledPC + GAMG, near-nullspace supplied in code (§6)
        (0, "u", inner(u_pc)),
        # psi : GravitySolver's own preset, unaltered (§6)
        (2, "psi", inner("gadopt.SPDAssembledPC")),
    ]
    for split, (field, _name, opts) in enumerate(sweep):
        p[f"dtn_fieldsplit_0_pc_fieldsplit_{split}_fields"] = str(field)
        p.update(_prefixed(opts, f"dtn_fieldsplit_0_fieldsplit_{split}_"))
    return p


#: split index -> what actually lives there, for reading the counts back.
SPLIT_NAMES = {0: "m   (DG1 tensor)", 1: "u   (CG2 vector)", 2: "psi (CG2)"}


# ---------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------
def build(mesh_path, phases, *, nu=0.49, truncation=5, dt=1.0, rotation=True,
          near_nullspace=True, **solver_kwargs):
    with phases("Mesh() parent"):
        parent = Mesh(mesh_path)
        parent.cartesian = False
    with phases("Submesh(mantle)"):
        sub = Submesh(parent, 3, gen.CELL_MANTLE)
        sub.cartesian = False

    X = SpatialCoordinate(parent)
    r = sqrt(dot(X, X))
    # A degree-2 zonal load, the same object as the mechanical surface load.
    sigma = SIGMA_HAT * (3.0 * (X[2] / r) ** 2 - 1.0) / 2.0

    with phases("spaces + DtNGravityForm"):
        gravity_bcs = {
            gen.SURF_OUTER: {"dtn": SphericalDtN(truncation)},
            gen.SURF_INNER: {"dtn": SphericalDtN(truncation)},
            gen.SURF_RE: {"interior_sigma": sigma},
        }
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
            self_gravity_number=LAMBDA)
        z = Function(Z)
        z.subfunctions[layout.displacement].rename("displacement")
        z.subfunctions[layout.potential].rename("potential")

    with phases("approximation + moments"):
        approx = CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
            bulk_shear_ratio=NU_TO_RATIO[nu], g=1.0, B_mu=B_MU,
            self_gravity_number=LAMBDA)
        Xm = SpatialCoordinate(sub)
        dx_m = Measure("dx", domain=sub,
                       intersect_measures=(Measure("dx", domain=parent),))
        rm = sqrt(dot(Xm, Xm))
        # `C` is the polar moment and is assemblable; `C - A` is the dynamical
        # ellipticity of the hydrostatic figure and is **not** computable from a
        # spherically symmetric reference density, so it is an input. Earth's
        # `H = (C - A)/C = 3.2737e-3`. B2 measures iteration counts, and this
        # constant sets only the scale of two Jacobian rows out of ~3.3e6 — but
        # it must be present and physical or the polar-wander rows are nonsense.
        C = assemble(approx.density * (Xm[0] ** 2 + Xm[1] ** 2) * dx_m)
        C_minus_A = 3.2737e-3 * C
        sigma_m = SIGMA_HAT * (3.0 * (Xm[2] / rm) ** 2 - 1.0) / 2.0
        bcs = {gen.SURF_RE: {"normal_stress": B_MU * sigma_m},
               gen.SURF_RC: {"un": 0.0}}

    if near_nullspace:
        with phases("rigid-body near-nullspace"):
            # §6: necessary for the deviatoric part, a subset of what the
            # volumetric part wants, and it cannot appear in the options dict.
            #
            # **All six modes, translations included — and that is not a
            # contradiction of road-map §7's "translations are not a kernel
            # mode".** The two are different objects and want different sets.
            # A `nullspace` must be the *exact* kernel of the operator, and
            # there translation is not in it: the prescribed reference
            # potential is anchored to the origin, so translating the body
            # changes the residual (measured at 1.0e-02, against rotation's
            # 8.3e-13). A `near_nullspace` describes what the operator GAMG
            # coarsens *nearly* annihilates, and GAMG coarsens the (u,u)
            # elasticity block, where all six rigid modes have `sym(grad) = 0`
            # exactly. Dropping the translations here would hand GAMG an
            # incomplete picture of its own operator's low-energy space and
            # make the V-cycle worse. Do not "fix" this to match the
            # nullspace.
            #
            # **And attaching it here is not enough.** This basis is composed
            # onto the OUTER mixed space, and `PCFIELDSPLIT` recovers a
            # near-nullspace by querying the field index sets it was given -
            # which `DtNTwoBlockSchurPC` replaces with merged sets of its own.
            # The query matches nothing and the modes are silently dropped, so
            # this argument alone buys exactly zero. `RigidBodyAssembledPC`
            # (top of this file) is what actually delivers them, by building
            # the modes from the displacement block's own function space. Both
            # are kept: the outer basis is harmless and is what a reader
            # expects to see, and the PC is what works.
            V = Z.sub(layout.displacement)
            rbm = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
            bases = [rbm if i == layout.displacement
                     else Z.sub(i) for i in range(len(Z))]
            solver_kwargs["near_nullspace"] = MixedVectorSpaceBasis(Z, bases)

    with phases("SelfGravitatingGIASolver.__init__"):
        solver = SelfGravitatingGIASolver(
            z, approx, layout=layout, dt=dt, bcs=bcs,
            rotation_moments={"C": C, "C_minus_A": C_minus_A},
            **solver_kwargs)
    return solver, z, layout, parent, sub


def dof_report(Z, layout, parent, sub):
    print("\n  degrees of freedom, per block")
    names = {layout.displacement: "u   (CG2 vector, mantle)",
             layout.potential: "psi (CG2, parent)"}
    for i in layout.internal_variables:
        names[i] = "m   (DG1 tensor, mantle)"
    total = 0
    per = {}
    for i in range(len(Z)):
        n = Z.sub(i).dim()
        total += n
        label = names.get(i, "Real")
        per[label] = per.get(label, 0) + n
    for label, n in per.items():
        print(f"    {label:<28} {n:>12,}")
    print(f"    {'TOTAL':<28} {total:>12,}")
    nm = FunctionSpace(sub, "DG", 0).dim()
    npar = FunctionSpace(parent, "DG", 0).dim()
    print(f"    parent cells {npar:,}, mantle cells {nm:,} "
          f"({nm / npar:.3f} of parent)")
    m_dofs = per.get("m   (DG1 tensor, mantle)", 0)
    print(f"    internal variable is {m_dofs / max(total - m_dofs, 1):.1f}x "
          f"everything else; condensation would remove "
          f"{m_dofs / total:.0%} of the Krylov traffic")
    return total, per


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------
LINE = re.compile(
    r"Linear\s+(\S*?)\s*solve converged due to (\S+) iterations (\d+)")
DIVERGED = re.compile(r"Linear\s+(\S*?)\s*solve did not converge due to (\S+)")


def parse_counts(text):
    """Per options prefix: applications, total and per-application iterations."""
    out = {}
    for prefix, _reason, its in LINE.findall(text):
        d = out.setdefault(prefix or "(outer)", {"n": 0, "its": [], "fail": 0})
        d["n"] += 1
        d["its"].append(int(its))
    for prefix, _reason in DIVERGED.findall(text):
        out.setdefault(prefix or "(outer)",
                       {"n": 0, "its": [], "fail": 0})["fail"] += 1
    return out


def report_counts(counts):
    print("\n  iteration counts, by options prefix")
    print(f"    {'prefix':<44} {'applied':>8} {'min':>5} {'med':>6} {'max':>5} "
          f"{'total':>8} {'diverged':>9}")
    for prefix in sorted(counts, key=lambda p: (len(p), p)):
        d = counts[prefix]
        its = d["its"] or [0]
        label = prefix
        m = re.search(r"fieldsplit_0_fieldsplit_(\d)_$", prefix)
        if m:
            label = f"{prefix}  <- {SPLIT_NAMES[int(m.group(1))]}"
        print(f"    {label:<62} {d['n']:>8} {min(its):>5} "
              f"{int(np.median(its)):>6} {max(its):>5} {sum(its):>8} "
              f"{d['fail']:>9}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--parse":
        # PETSc writes `ksp_converged_reason` from C, so it lands in the
        # redirected file rather than in any Python-level capture; parsing the
        # log afterwards is both simpler and the only thing that works.
        report_counts(parse_counts(open(sys.argv[2]).read()))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=os.path.join(HERE, "b2_sphere_coarse.msh"))
    ap.add_argument("--label", default="coarse")
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--truncation", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--multiplier-pc", default="jacobi",
                    choices=["none", "jacobi", "assembled_lu"])
    ap.add_argument("--block0-type", default="multiplicative",
                    choices=["multiplicative", "symmetric_multiplicative",
                             "additive"])
    ap.add_argument("--block0-rtol", type=float, default=1e-2)
    ap.add_argument("--inner-ksp", default="preonly",
                    choices=["preonly", "cg", "richardson"])
    ap.add_argument("--inner-its", type=int, default=1)
    ap.add_argument("--no-rotation", action="store_true")
    ap.add_argument("--no-near-nullspace", action="store_true",
                    help="use plain firedrake.AssembledPC on u, i.e. no "
                         "near-nullspace reaching GAMG at all")
    ap.add_argument("--dofs-only", action="store_true")
    args = ap.parse_args()

    print(__doc__)
    print(f"\n=== B2, {args.label}, nu = {args.nu} "
          f"(K/mu = {NU_TO_RATIO[args.nu]}), L = {args.truncation}, "
          f"multiplier pc = {args.multiplier_pc} ===", flush=True)

    phases = Phases()
    params = solver_parameters(
        multiplier_pc=args.multiplier_pc, block0_type=args.block0_type,
        block0_rtol=args.block0_rtol, inner_ksp=args.inner_ksp,
        inner_its=args.inner_its,
        u_pc=("firedrake.AssembledPC" if args.no_near_nullspace
              else "__main__.RigidBodyAssembledPC"))

    solver, z, layout, parent, sub = build(
        args.mesh, phases, nu=args.nu, truncation=args.truncation, dt=args.dt,
        rotation=not args.no_rotation,
        near_nullspace=not args.no_near_nullspace,
        solver_parameters=params)

    total, per = dof_report(z.function_space(), layout, parent, sub)
    phases.report()
    print(f"\n  resident after setup: {rss_gb():.2f} GB", flush=True)
    if args.dofs_only:
        return

    print("\n  --- one linear solve ---", flush=True)
    t0 = time.time()
    with PETSc.Log.Stage("b2_solve"):
        solver.solve()
    dt_solve = time.time() - t0
    print(f"  solve wall time {dt_solve:.1f} s, "
          f"peak resident {rss_gb():.2f} GB", flush=True)
    return solver, dt_solve


if __name__ == "__main__":
    main()
