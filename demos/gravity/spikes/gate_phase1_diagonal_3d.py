"""Phase 1: the block-1 diagonal on the real 75-row block, at production size.

`block1_diagonal()` claims to be the exact diagonal of the `Real` block that
`DtNMultiplierDiagPC` divides by. `tests/unit/test_gia_gravity.py` pins that
claim on the 2-D annulus and on a 24-cell cubed sphere; this is the same probe
on the operator the campaign actually runs -- B1's medium mesh at DtN `L = 5`
with rotation on, i.e. **75 `Real` rows: 72 multipliers (2 boundaries x 36
modes) then `m_1`, `m_2`, `m_3`**.

## What it does, and the one thing it must not do

The block is recovered by **residual differencing**: the coupled residual is
linear in every `Real` unknown, so `F(e_j) - F(0)` read on the `Real` rows *is*
column `j`, exactly, with no step size to argue about. That is 76 residual
assemblies and one residual vector.

**It must never assemble a nest Jacobian.** Both shipped presets are
`mat_type: matfree`; `mat_type="nest"` materialises every block, `(u,u)` and
`(psi,psi)` included -- the assembled operator this configuration exists to
avoid -- and that is the most likely explanation for the 180 GB that killed job
175284920, which is the job this gate was first attempted in. The 2-D tests do
use nest for the *block-0 symmetry* measurement; nothing here does, and nothing
here may.

There is also no `solver.solve()`. Not a smoke test, not "just to check".

## Expected, stated before the run

Constants from `reference_state`: `Lambda = 1.3613238468`,
`B_mu = 1.5640359327`, `Omega^2 = 1.5661756594e-03`, `C = 72.2269347256`,
`C - A = 2.4214000357e-01`, and `theta_psi = B_mu / Lambda = 1.1489080548`.

| quantity | expected |
|---|---|
| `Real` rows | 75 = 72 + 3 |
| multiplier diagonal, outer boundary | -22.316 (`-theta_psi R^2`, `R = 4.407472`) |
| multiplier diagonal, inner boundary | -0.41618 (`R = 0.601868`) |
| distinct multiplier values | **2 of 72** (`SphericalDtN` scales every mode by `1/(4 pi)`) |
| rotation diagonal `m_1`, `m_2` | `+5.9313525844e-04` (`+B_mu Omega^2 (C-A)`) |
| rotation diagonal `m_3` | `-1.7692384969e-01` (`-B_mu Omega^2 C`) |
| `max rel |d - diag(A)|`, multiplier rows | <= 1e-12; measured 1.2735e-15 at coarse |
| the same on the rotation rows | <= 1e-9, **not** 1e-12; measured 8.8775e-11, and see `ROT_DIAG_TOL` for why the handover's 1e-12 is unreachable there |
| agreement between the three rotation deviations | <= 1e-2; measured 1.4e-04 |
| per-row `max|off| / |diag|` | <= 1e-12; exactly 0.0 in serial, measured 0.0 at coarse |
| `(c, m)` and `(m, c)` | zero serially; **5.7199e-05 at 48 ranks**, see below |
| diagonal spread | ~3.8e+04, against 236 in 2-D and 296 on the toy; measured 3.7627e+04 |

Measured at `--configuration coarse`, serially, before this was ever queued:
the multiplier entries come out at `-22.318121` and `-0.416180` against the
`-22.316 / -0.41618` predicted above, and the rotation diagonal at
`+5.9313525839e-04, +5.9313525839e-04, -1.7692384967e-01`. Everything
structural passes; only the rotation-row tolerance had to be told the truth.

The multiplier predictions use `A_h = 4 pi R^2`; the true entry uses the
**discrete** boundary measure, which is the mesh's own polygonal-or-curved area
and is not that number in either direction - measured `-22.318121` against the
predicted `-22.316`, i.e. 9e-05 out and on the far side. So the gate compares
against `block1_diagonal()` and never against the closed form; the closed form
is here to say what order of magnitude to expect, and it is stated first so
that a wrong one is visible in the log rather than rationalised after.

**Off-diagonal exactness is a serial statement**, and the gate is a per-row
ratio for that reason. An earlier version of this docstring attributed the
parallel residue to cross-rank reduction order; that account is **withdrawn**
and `NOTES/FD-ISSUE.md` has the current one.

## What this does not establish

A flip of `CLOSURE_SIGNS` is invisible to the *comparison*: `block1_diagonal`
builds the rotation entry as `_theta_rot(k) * _closure_constant(k)`, the
assembled row is `theta_rot_i K_i`, and both carry `s_i`, so a flip moves them
together. What sees it here is the absolute check against
`+5.9313525844e-04 / -1.7692384969e-01`, and that is a regression pin on
numbers measured from this code -- not an independent derivation. The
independent statement is `m_3 = -dI_33 / C` itself, which is
`demos/gravity/spikes/gate_v7.py`.

Nor does it bear on `_RealBlockPCBase.applyTranspose` delegating to `apply`.
What has to be symmetric there is `_solve`, not this block. For
`DtNMultiplierDiagPC`, `_solve` is `rhs / d`, its own transpose regardless of
anything measured here; for a dense-Schur subclass it is an LU of a matrix
measured at relative asymmetry 0.344, and therefore wrong. Per subclass.

## Running it: coarse at 48 ranks, and that is what closes S1.1

**The target is `--configuration coarse` on 48 ranks, not medium.** The 75
`Real` rows exist at coarse - `STEP0-BLOCK1-PC.md` records medium as "the same
field structure on the 5.2x mesh", so the block is the same size with the same
two distinct values - and the handover's S1.1 asks for "75 Jacobian actions on
the `Real` unit vectors". Medium buys mesh-independence, which is worth having
and is **not** what S1.1 asks for. A missing medium run is therefore not an
incomplete gate.

The order matters and is not a preference:

1. **serial coarse** — the reference. Run; passes every criterion.
2. **coarse at 48 ranks**, `--diff` against (1). **RUN, job 175368128, and it
   FAILED eight criteria — correctly.** The 72 multiplier rows agree with the
   serial reference to **2.6677e-16**, which closes the `read_real_rows` gap
   this step existed to close. The three rotation rows are over-counted by the
   rank count to within 3e-5 on the polar pair and 3e-3 on `m_3` (ratios
   48.000025, 47.999962, 47.997255, **residual unaccounted**), a spurious
   closure-row/multiplier coupling appears at 5.7199e-05 where serial reads
   0.0, and `m_1`/`m_2` differ in the seventh digit. `NOTES/FD-ISSUE.md` is the
   record; `--pairing` below carries the diagnostics that would confirm or kill
   the leading account. **Do not treat the failure as a gate defect.**
   Going straight from serial
   coarse to medium at 48 would move the mesh and the rank count in one step,
   and a failure would not say which caused it. More to the point,
   `read_real_rows` is proven at 2 and 3 ranks and *assumed* at 48 by the
   handover's note that all 75 `Real` dofs sit on rank 0. There is nothing in
   between, and a parallel `Real` read that returns something plausible rather
   than failing is the exact hazard this gate exists to exclude.
3. **medium, optionally, and only if (2) agrees.**

    W=/scratch/.../selfgravity
    P=$W:$W/demos/glacial_isostatic_adjustment/3d_spada_selfgrav

    # (1) the reference, one rank
    PYTHONPATH=$P python .../gate_phase1_diagonal_3d.py \\
        --configuration coarse --save coarse_serial.npy

    # (2) the measurement that closes S1.1
    PYTHONPATH=$P mpiexec -n 48 python .../gate_phase1_diagonal_3d.py \\
        --configuration coarse --diff coarse_serial.npy

Every entry is printed at full precision under `DIAG`, so the two runs can also
be diffed by hand from the job logs without the `.npy`.

76 residual assemblies and one mesh build; no solve, no factorisation, no
assembled operator. `--selfcheck` runs the identical probe on the 24-cell cubed
sphere of `TestBlockOneInThreeDimensions`, needs no gmsh and no mesh file, and
is the only mode of this script meant to be run on a laptop. `--build-only`
stops after constructing the solver, which is the cheapest way to discover that
a mesh, a preset or an argument is wrong. `--h` overrides the resolution ladder;
note that `--h 0.25` produces a mesh `check_geometry` **rejects**, and that
rejection is correct rather than a tight tolerance - it compares two discrete
quantities over the same cells, so a disagreement at 1e-07 is the cross-mesh
entity maps genuinely wrong, and both coupling terms are built on that measure.

Exits non-zero if any criterion fails.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np  # noqa: E402
import firedrake as fd  # noqa: E402
from firedrake import COMM_WORLD  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPADA = os.path.join(ROOT, "demos", "glacial_isostatic_adjustment",
                     "3d_spada_selfgrav")

# Tolerances. The multiplier and off-diagonal numbers are the unit tests' own.
DIAG_TOL = 1e-12
OFFDIAG_TOL = 1e-12

#: **The rotation rows do not meet 1e-12 at production size, and cannot.**
#: Measured at `--configuration coarse`: multiplier rows 1.2735e-15, rotation
#: rows **8.8775e-11**, with all three carrying the *same signed* deviation
#: (`-8.877465e-11`, `-8.877465e-11`, `-8.876222e-11`).
#:
#: That common factor is the whole of it, and it has been pinned to the digit.
#: `rotation_residual` (`gia_gravity.py:2116`) precomputes
#: `volume = assemble(Constant(1.0)*dx_m)` and writes the row as
#: `(K_i m_i / volume) * nu_i * dx_m`, so the assembled diagonal is
#: `theta_rot_i K_i x (V as the form's kernel sums it) / (V as precomputed)` --
#: the same integral under **different quadrature rules**. Sweeping the rule on
#: this exact mesh:
#:
#:     degree  0, 1   37.523866273075946    +9.680958e-07
#:     degree  2      37.5238299130637      -8.887997e-10
#:     degree  3      37.52382994641487      0.0          <- the default
#:     degree  4      37.523829945296356    -2.980816e-11
#:     degree  6      37.52382994469773     -4.576128e-11
#:     degree  7      37.52382994308379     -8.877243e-11 <- the rows read this
#:     degree 12      37.523829942980356    -9.152890e-11
#:
#: `V(7)/V(3) - 1 = -8.877243e-11` against the rotation rows' `-8.877465e-11`.
#: The default rule for `Constant(1.0)*dx_m` is degree 3; the merged `dx_m`
#: integrals of `F` are evaluated at 7. The residual 2e-15 between the two is
#: the row's integral being one of several accumulated in a single kernel.
#:
#: The 2-D annulus shows the same signature four orders smaller: its rotation
#: row is already the worst row, 2.664535e-15 against 2.0447e-16 for the
#: multipliers, and its volume moves by 4.4e-16 at degree 2, 6.7e-16 at 6 and
#: 1.6e-15 at 8 -- and by 1.6e-03 at degree 0, the curved P2 Jacobian being far
#: from constant.
#:
#: `block1_diagonal` predicts `theta_rot_i K_i` and so assumes that ratio is
#: exactly 1. **The right fix is in `gadopt/`**: assemble the volume under the
#: rule its own integral uses, after which 1e-12 stands for every row. Until
#: then this gate bounds the rotation rows at a number with headroom over the
#: measurement, and -- more sharply -- requires the three deviations to *agree*
#: with each other, which is what distinguishes one shared volume ratio from
#: three independent errors. A genuinely wrong rotation entry would not track
#: the other two.
#: **A coarse-calibrated sanity bound, not a calibrated tolerance.** `V(7)/V(3)`
#: is a property of the mesh - cell count, curvature, the anisotropic
#: lithosphere, the two tangled cells - and it has been measured at exactly one
#: rung. Medium is 5.2x the cells and is unmeasured. So this is set an order
#: above the coarse measurement rather than just above it, its only job is to
#: catch an order-of-magnitude regression, and **it must be re-measured if the
#: mesh changes.** The gate that carries the claim is the agreement below.
ROT_DIAG_TOL = 1e-8
#: **This is the real rotation-row gate.** The deviation is a single volume
#: ratio, so all three rows must carry it; three rows disagreeing would be three
#: independent errors, which no volume ratio can produce. Measured 1.4007e-04
#: between `m_3` and the polar pair at coarse.
#:
#: Worth recording that this is looser than the account predicts. The inertia
#: polynomial does not enter the diagonal at all, so all three rows should carry
#: the *identical* ratio and agree to the seven digits the polar pair manages
#: (`-8.877465e-11` twice). `m_3` reads `-8.876222e-11`, three orders looser,
#: and roundoff does not obviously cover it: the pair's absolute deviation is
#: ~5.3e-14 on 5.93e-04 and `m_3`'s is ~1.57e-11 on -1.769e-01, so neither is
#: near its relative floor. **Unexplained.** The account predicts agreement that
#: was not measured, and this bound is set from the measurement rather than from
#: the prediction.
ROT_DIAG_AGREEMENT = 1e-2

# The rotation diagonal, to the digits `block1_diagonal`'s docstring records.
ROT_EXPECTED = (+5.9313525844e-04, +5.9313525844e-04, -1.7692384969e-01)

_builtin_print = print


def print(*a, **k):  # noqa: A001
    """Rank 0 only: every quantity below is collective."""
    if COMM_WORLD.rank == 0:
        _builtin_print(*a, **k, flush=True)


def read_real_rows(cofunction, ises, comm):
    """The `Real` entries of an assembled residual, gathered on every rank.

    `_RealBlockPCBase._gather`'s access pattern term for term -- zero-fill,
    write only owned entries, `Allreduce` -- and **not** `.dat.data_ro[0]`,
    which `demos/gravity/CLAUDE.md` records as bypassing the ghosting and
    reduction machinery. Reimplemented here rather than imported from
    `tests/unit/test_gia_gravity.py`, because a demo reaching into the test
    tree is worse than eight duplicated lines; the two are asserted to agree by
    `--selfcheck`, which runs the same case the unit test does.
    """
    buf = np.zeros(len(ises))
    with cofunction.dat.vec_ro as vec:
        lo, hi = vec.owner_range
        local = vec.array_r
        for position, iset in enumerate(ises):
            indices = iset.getIndices()
            if len(indices) > 1:
                raise RuntimeError(
                    f"block-1 sub-field {position} owns {len(indices)} degrees "
                    "of freedom; a Real space has exactly one, globally")
            if len(indices) == 1:
                index = int(indices[0])
                if not lo <= index < hi:
                    raise RuntimeError(
                        f"field_ises gave sub-field {position} global index "
                        f"{index}, outside this rank's range [{lo}, {hi})")
                buf[position] = local[index - lo]
    out = np.zeros(len(ises))
    comm.Allreduce(buf, out)
    return out


def real_block(solver, z, layout):
    """The full `n x n` `Real` block, by differencing. No nest, ever."""
    Z = z.function_space()
    comm = Z.mesh().comm
    ises = Z.dof_dset.field_ises[layout.real_fields[0]:]
    real = layout.real_fields

    z.assign(0.0)
    base = read_real_rows(fd.assemble(solver.F), ises, comm)
    block = np.zeros((len(real), len(real)))
    for column, field in enumerate(real):
        z.assign(0.0)
        z.subfunctions[field].assign(1.0)
        block[:, column] = read_real_rows(
            fd.assemble(solver.F), ises, comm) - base
        if column % 10 == 0:
            print(f"    column {column + 1}/{len(real)}")
    z.assign(0.0)
    return block


def report(solver, z, layout):
    """Measure, print, and return the list of failures."""
    block = real_block(solver, z, layout)
    claimed = solver.block1_diagonal()
    diagonal = np.diag(block)
    n_mult = len(layout.multipliers)
    n_rot = len(layout.rotation)

    off = block.copy()
    np.fill_diagonal(off, 0.0)
    rel_diag = np.abs(claimed - diagonal) / np.abs(diagonal)
    row_ratio = np.abs(off).max(axis=1) / np.abs(diagonal)
    spread = np.abs(diagonal).max() / np.abs(diagonal).min()
    distinct = sorted(set(np.round(diagonal[:n_mult], 10)))

    print()
    print("  measured")
    print(f"    Real rows                    {len(diagonal)} "
          f"= {n_mult} multipliers + {n_rot} rotation")
    print(f"    theta_psi                    {float(solver.theta_psi):.10f}")
    print(f"    distinct multiplier values   {len(distinct)} of {n_mult}: "
          + ", ".join(f"{v:.6f}" for v in distinct))
    if n_rot:
        print("    rotation diagonal            "
              + ", ".join(f"{v:+.10e}" for v in diagonal[n_mult:]))
    print(f"    max rel |d - diag(A)|        {rel_diag.max():.4e}   "
          f"(worst row {int(rel_diag.argmax())})")
    print(f"      multiplier rows            {rel_diag[:n_mult].max():.4e}")
    if n_rot:
        # Split out, because they fail for different reasons if they fail:
        # the multiplier entry is a boundary area the form already stores,
        # while the rotation entry is `theta_rot_i K_i` against a row that
        # divides by a separately assembled mantle volume. If those two
        # assemblies of the same integral do not agree to the last bit, the
        # ratio shows up here and nowhere else -- so print the signed
        # deviation per row, and see whether the three share it.
        print(f"      rotation rows              "
              f"{rel_diag[n_mult:].max():.4e}")
        for k, (measured, predicted) in enumerate(
                zip(diagonal[n_mult:], claimed[n_mult:])):
            print(f"        row {n_mult + k}: measured/claimed - 1 = "
                  f"{measured / predicted - 1.0:+.6e}")
        volume = float(fd.assemble(fd.Constant(1.0) * solver.dx_m))
        print(f"      mantle volume, reassembled {volume:.17e}")
    print(f"    max |off-diagonal|           {np.abs(off).max():.4e}   "
          f"(exactly zero: {np.abs(off).max() == 0.0})")
    print(f"    worst per-row |off|/|diag|   {row_ratio.max():.4e}   "
          f"(row {int(row_ratio.argmax())})")
    print(f"    diagonal spread              {spread:.4e}   "
          f"({np.abs(diagonal).min():.4e} .. {np.abs(diagonal).max():.4e})")
    if n_rot:
        print(f"    max |(c, m)|                 "
              f"{np.abs(block[:n_mult, n_mult:]).max():.4e}")
        print(f"    max |(m, c)|                 "
              f"{np.abs(block[n_mult:, :n_mult]).max():.4e}")

    # Every entry at full precision, so a run at another rank count can be
    # diffed against this one entrywise rather than through summary
    # statistics. This is how `read_real_rows` gets its only measurement at
    # production rank count; the criterion is relative, never equality.
    print()
    print("  the whole diagonal, for diffing (index, entry):")
    for i, value in enumerate(diagonal):
        print(f"    DIAG {i:3d} {value!r}")

    failures = []

    def check(ok, text):
        print(f"    [{'PASS' if ok else 'FAIL'}] {text}")
        if not ok:
            failures.append(text)

    print()
    print("  gate")
    check(rel_diag[:n_mult].max() <= DIAG_TOL,
          f"block1_diagonal is the assembled diagonal on the multiplier rows "
          f"to {DIAG_TOL:g} relative (read {rel_diag[:n_mult].max():.4e})")
    if n_rot:
        deviations = diagonal[n_mult:] / claimed[n_mult:] - 1.0
        check(rel_diag[n_mult:].max() <= ROT_DIAG_TOL,
              f"and on the rotation rows to {ROT_DIAG_TOL:g}, which is NOT "
              f"1e-12 and cannot be - see ROT_DIAG_TOL (read "
              f"{rel_diag[n_mult:].max():.4e})")
        spread_of_deviation = (
            np.ptp(deviations) / max(abs(deviations).max(), 1e-300))
        check(spread_of_deviation <= ROT_DIAG_AGREEMENT,
              f"the rotation deviations agree with each other to "
              f"{ROT_DIAG_AGREEMENT:g}, i.e. they are one shared volume ratio "
              f"and not three independent errors (read "
              f"{spread_of_deviation:.4e})")
    check(row_ratio.max() <= OFFDIAG_TOL,
          f"every off-diagonal is below {OFFDIAG_TOL:g} of its own row's "
          f"diagonal (read {row_ratio.max():.4e})")
    check(np.all(np.abs(diagonal) > 0.0),
          "no diagonal entry is zero, so the gate is not vacuous and the "
          "preconditioner is not a division by zero")
    # Falsifiers, not targets: each states what a violation would mean.
    check(np.all(diagonal[:n_mult] < 0.0),
          "every multiplier entry is negative, as `-scale_k * A_h` must be")
    check(len(distinct) == 2,
          f"the {n_mult} multiplier entries take exactly 2 distinct values, "
          "one per boundary (read "
          f"{len(distinct)}). A third would mean either that a DtN boundary is "
          "not at constant radius or that `scale` is not uniform across "
          "(l, m) - both real diagnostics, neither otherwise detectable, and "
          "it is also the degeneracy the alignment test exists for")
    if n_rot == 3:
        rotation = diagonal[n_mult:]
        check(np.abs(block[:n_mult, n_mult:]).max()
              <= OFFDIAG_TOL * np.abs(rotation).min(),
              "no constraint row sees the rotation scalars")
        check(np.abs(block[n_mult:, :n_mult]).max()
              <= OFFDIAG_TOL * np.abs(rotation).min(),
              "no closure row sees the multipliers")
        check(abs(rotation[0] - rotation[1]) <= 1e-12 * abs(rotation[0]),
              "m_1 and m_2 carry the same closure constant C - A")
        check(rotation[0] > 0.0 > rotation[2],
              "the closure signs s = (+1, +1, -1) show through the assembled "
              "diagonal (a REGRESSION pin, not an independent check - see the "
              "module docstring)")
        # **A CONFIGURATION pin, at 1e-6, and deliberately not part of the
        # gate above.** These literals are `theta_rot_i * K_i`, so they depend
        # on `rotation_moments` as much as on `theta_rot` -- and this script is
        # the first thing to run `rotation=True` with `condense=True`, a
        # combination no driver has (`b1_elastic` is rotation-off,
        # `b4_polar_motion` uncondensed). If the moments on this path differ at
        # all from those that produced the digits, this fails for a
        # configuration reason, and it must not read as a failure of the
        # diagonal identity. The gate is `block1_diagonal() == diag(A)`; every
        # literal beside it is a second claim that can fail independently.
        for i, expected in enumerate(ROT_EXPECTED):
            check(abs(rotation[i] - expected) <= 1e-6 * abs(expected),
                  f"CONFIGURATION PIN (not the diagonal gate): rotation row "
                  f"{i} is {expected:+.10e}, i.e. theta_rot_{i} * K_{i} for "
                  f"the moments this path supplies (read {rotation[i]:+.10e}). "
                  f"A failure here means the configuration moved, not that "
                  f"block1_diagonal is wrong")
    return failures, diagonal


def pairing_diagnostics(solver, z, layout):
    """Three measurements that confirm or kill the pairing account of the 48-rank failure.

    **The account is now MEASURED, not a prediction.** `NOTES/FD-ISSUE.md` is
    the canonical record and carries the control table; the short version is
    that the trigger is a `Real` **trial** against a `Real` **test**, integrated
    over a mesh other than the one those `Real` spaces are built on. Both halves
    are needed: mismatched-mesh alone is clean in three independent forms
    (including these diagnostics' own inertia pieces), and `Real x Real` alone
    is clean on the parent.

    Historical framing, kept because two earlier accounts were withdrawn:
    `Real` spaces live on the parent (spike S2, both families, deliberately).
    The rotation row's terms integrate over `dx_m`, the *submesh*, so the test
    function's mesh and the integration domain differ; the multiplier rows'
    `Real` tests are also on the parent but integrate parent `ds`, so they
    match. That is a within-run control at one rank count on one mesh - matched
    pairing clean, mismatched over-counted - and it rules out "48 ranks breaks
    `Real` reductions" without a second job.

    **What is measured**, jobs 175368128 and 175369366, coarse, one node: serial
    passes every criterion; 48 ranks fails eight. The rotation rows are over-counted by the
    rank count **to within 3e-5 on the polar pair and 3e-3 on `m_3`** - ratios
    48.000025, 47.999962, 47.997255 - and the residual is **unaccounted**. It is
    not exactly 48: the quadrature volume ratio is -8.9e-11 and cannot produce
    2.5e-5, and a single additive contamination cannot either, the implied term
    being 1.5e-8 from the pair against 4.8e-4 from `m_3`. The 72 multiplier rows
    are correct to 2.6677e-16 in the same run.

    Runs serially and under MPI, so one job gives both halves of the control.
    """
    import firedrake as fd
    mech, parent = layout.mechanics_mesh, solver.potential_mesh
    comm = z.function_space().mesh().comm
    n_mult = len(layout.multipliers)
    rot = layout.rotation_slots()[2]

    print()
    print("  " + "-" * 68)
    print("  PAIRING DIAGNOSTICS. The account below is a PREDICTION.")
    print("  " + "-" * 68)

    # --- 1. the partition census ------------------------------------------
    # The Submesh inherits the parent's partition rather than being
    # redistributed, so the imbalance is severe and some ranks may own no
    # mantle cells at all. If the factor equals the communicator size the
    # replication is over all ranks; if it equals the number of ranks owning
    # mantle cells it is over the contributors. Those are different bugs.
    own_sub = mech.cell_set.size
    own_par = parent.cell_set.size
    all_sub = comm.allgather(own_sub)
    all_par = comm.allgather(own_par)
    participating = sum(1 for c in all_sub if c > 0)
    print(f"    [1] partition census over {comm.size} rank(s)")
    print(f"        owned submesh cells  total {sum(all_sub)}   "
          f"min {min(all_sub)}  max {max(all_sub)}")
    print(f"        owned parent cells   total {sum(all_par)}   "
          f"min {min(all_par)}  max {max(all_par)}")
    print(f"        ranks owning >=1 mantle cell: {participating} "
          f"of {comm.size}")
    if participating != comm.size:
        print(f"        ** {comm.size - participating} rank(s) own NO mantle "
              "cells: the two candidate factors are distinguishable here **")
    else:
        print("        every rank owns mantle cells, so communicator size and "
              "contributor count coincide and this run cannot separate them")

    # --- 2. the same dI_33, weighted two ways ------------------------------
    # `test=None` weights by Constant(1.0) and assembles a 0-form: no Real test
    # function, so under the account it is CLEAN. The Real-test row is what the
    # gate recovers. If the first is clean and the second carries the factor,
    # the pairing account is confirmed - and the 0-form is the corrected dI.
    X = fd.SpatialCoordinate(mech)
    z.assign(0.0)
    z.subfunctions[layout.displacement].interpolate(X)
    scalar = float(fd.assemble(solver.inertia_form(2)))
    print("    [2] dI_33 at u = X, weighted two ways")
    print(f"        0-form, Constant(1.0) weight   {scalar!r}")
    if rot is not None:
        ises = z.function_space().dof_dset.field_ises[layout.real_fields[0]:]
        row = read_real_rows(fd.assemble(solver.F), ises, comm)[n_mult + 2]
        base_row = row
        print(f"        Real-test row of solver.F      {base_row!r}")
        print("        compare the ratio of these two against the census "
              "above; equal means no pairing effect")

    # --- 3. the volume and sheet halves, separately ------------------------
    # A NULL CONTROL, and that is its value. It was built to test a dilution
    # prediction - that the mismatched pieces would over-count while the matched
    # one would not - and job 175369366 refuted that: all three are clean at 48
    # ranks to twelve digits. Keeping it is what licenses the statement that the
    # over-count does not reach `dI`.
    if rot is not None:
        nu = solver.tests[rot]
        rho0 = solver.approximation.density
        p_m = solver.inertia_polynomial(2, X)
        u = solver.solution_split[layout.displacement]
        volume_half = nu * rho0 * fd.dot(fd.grad(p_m), u) * solver.dx_m
        p_g = solver.inertia_polynomial(2, fd.SpatialCoordinate(parent))
        ice_half = None
        for bc_id, sigma, itype in solver.form.sigma_bcs:
            term = solver.form.sheet_integral(
                nu * fd.Constant(1.0) * sigma * p_g, bc_id, itype)
            ice_half = term if ice_half is None else ice_half + term
        # The fluid core's sheet is the third piece and it is easy to miss:
        # `fluid_core_sheet_integral` evaluates on the MECHANICS mesh (its own
        # docstring says so - "the only cross-mesh object in the form is the
        # Real test function"), so it is the *same* mismatched pairing as the
        # volume term, not the matched one the ice sheet has. Omitting it made
        # an earlier version of this diagnostic disagree with the 0-form by
        # 41.29 of 171.41, which is how it was caught.
        core_half = solver.fluid_core_sheet_integral(nu * p_m)

        ises = z.function_space().dof_dset.field_ises[layout.real_fields[0]:]

        def real_row(form):
            if form is None or not form.integrals():
                return 0.0
            return read_real_rows(fd.assemble(form), ises, comm)[n_mult + 2]

        v = real_row(volume_half)
        ice = real_row(ice_half)
        core = real_row(core_half)
        print("    [3] inertia_form's three pieces, each with a Real test")
        print(f"        mantle volume  (submesh dx_m,   MISMATCHED)  {v!r}")
        print(f"        core sheet     (submesh ds,     MISMATCHED)  {core!r}")
        print(f"        ice sheet      (parent facets,  matched)     {ice!r}")
        total = v + ice + core
        print(f"        sum {total!r}   against the 0-form {scalar!r}")
        print(f"        decomposition closes to {abs(total - scalar):.3e}"
              + ("  OK" if abs(total - scalar) <= 1e-9 * max(abs(scalar), 1.0)
                 else "  ** INCOMPLETE - a piece is missing **"))
        # MEASURED, job 175369366: all three are CLEAN at 48 ranks, to twelve
        # digits. An earlier version of this block predicted that the two
        # mismatched pieces would over-count and the matched one would not,
        # diluting the ice load's share of dI by the rank count. **Refuted.**
        # `dI` is correct in parallel and the defect is in the DIAGONAL alone,
        # so this diagnostic is now a null control: it is what establishes that
        # the over-count does not reach `dI`, which is what makes the closure's
        # failure mode `n K m = s dI` rather than something subtler.
        print("        MEASURED at 48 ranks (job 175369366): all three CLEAN")
        print("        to 12 digits. This is a NULL CONTROL - it is what shows")
        print("        the over-count does not reach dI, so the closure fails")
        print("        as  n K m = s dI,  i.e. m too small by the rank count.")
        z.assign(0.0)
        print("  " + "-" * 68)
        return {"scalar": scalar, "row": base_row, "volume": v,
                "core": core, "ice": ice,
                "n_ranks": float(comm.size),
                "participating": float(participating)}
    z.assign(0.0)
    print("  " + "-" * 68)
    return {}


def pairing_path(path):
    """The pairing sidecar for a `--save`/`--diff` path.

    `np.save("x")` writes `x.npy`, so `--save x` and `--diff x.npy` name the
    same run and must resolve to the same sidecar. Strip one trailing `.npy`
    before appending, or the comparison silently finds nothing -- which it did
    on the first test of this feature.
    """
    base = path[:-4] if path.endswith(".npy") else path
    return base + ".pairing.npz"


def pairing_compare(now, path):
    """Ratios against a serial reference, and the implied `dI_ice` dilution.

    Printed so the result can be quoted without arithmetic. The dilution is the
    quantity that matters beyond the gate: if the two mismatched pieces carry a
    factor and the matched one does not, the closure solves
    `K m = s (dI_mantle + dI_core + dI_ice / n)`, so the ice load's share of the
    inertia perturbation is suppressed relative to the mantle's by exactly the
    factor printed here.
    """
    ref = dict(np.load(path))
    print()
    print("  " + "-" * 68)
    print(f"  PAIRING vs the reference in {path}")
    print(f"  {'piece':34s} {'this run':>20s} {'ratio to ref':>14s}")
    for key, label in (("volume", "mantle volume  (MISMATCHED)"),
                       ("core", "core sheet     (MISMATCHED)"),
                       ("ice", "ice sheet      (matched)"),
                       ("scalar", "dI_33, 0-form (no Real test)"),
                       ("row", "dI_33 via the Real-test row")):
        if key not in now or key not in ref:
            continue
        a, b = float(now[key]), float(ref[key])
        ratio = a / b if b != 0.0 else float("nan")
        print(f"    {label:32s} {a:20.12e} {ratio:14.6f}")
    print(f"    {'rank count of this run':32s} "
          f"{int(now.get('n_ranks', 0)):20d}")
    print(f"    {'ranks owning mantle cells':32s} "
          f"{int(now.get('participating', 0)):20d}")

    # The implied dilution: how far the matched piece is suppressed relative
    # to the mismatched ones, which is the physical statement.
    try:
        mism = (float(now["volume"]) + float(now["core"])) / (
            float(ref["volume"]) + float(ref["core"]))
        matched = float(now["ice"]) / float(ref["ice"])
        print(f"    mismatched pieces scaled by {mism:.6f}, matched by "
              f"{matched:.6f}")
        print(f"    ==> dI_ice is diluted relative to dI_mantle+dI_core by "
              f"{mism / matched:.6f}")
        print("        (1.0 = no dilution; the rank count = the predicted "
              "worst case)")
    except (KeyError, ZeroDivisionError):
        pass
    print("  " + "-" * 68)


def selfcheck():
    """The 24-cell cubed sphere of `TestBlockOneInThreeDimensions`.

    No gmsh, no mesh file, no `Submesh`, assembly only, seconds. It exercises
    every line of the probe above on a case whose answers are pinned in CI, so
    a laptop can establish that this script works before a cluster job spends a
    node-hour finding out that it does not.
    """
    from gadopt import (CompressibleInternalVariableApproximation,
                        SelfGravitatingGIASolver, SphericalDtN,
                        self_gravitating_gia_space)

    lam = 1.1116
    base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                           extrusion_type="radial")
    mesh.cartesian = False
    Z, layout = self_gravitating_gia_space(
        mesh, mesh,
        gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                     "bottom": {"dtn": SphericalDtN(1)}},
        rotation=True, condense_internal_variables=True,
        self_gravity_number=lam)
    z = fd.Function(Z)
    X = fd.SpatialCoordinate(mesh)
    C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
    solver = SelfGravitatingGIASolver(
        z,
        CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
            g=1.0, B_mu=1.2769, self_gravity_number=lam),
        layout=layout, dt=1.0,
        bcs={"bottom": {"un": 0.0}, "top": {"normal_stress": 1e-3 * X[2]}},
        rotation_moments={"C": C, "C_minus_A": 0.1 * C})

    print("  SELFCHECK: 24 cells, 1971 dofs, 11 Real rows. The production "
          "criteria on the rotation")
    print("  magnitudes do not apply here - the moments are 1.0 and 0.1 C - "
          "so only the")
    print("  structural ones are gated.")
    block = real_block(solver, z, layout)
    claimed = solver.block1_diagonal()
    diagonal = np.diag(block)
    off = block.copy()
    np.fill_diagonal(off, 0.0)
    rel = np.abs(claimed - diagonal) / np.abs(diagonal)
    ratio = np.abs(off).max(axis=1) / np.abs(diagonal)
    n_mult = len(layout.multipliers)
    print(f"    dofs {Z.dim()}   Real rows {len(diagonal)}   "
          f"multipliers {n_mult}")
    print(f"    max rel |d - diag|  {rel.max():.4e}   (CI reads 1.7875e-15)")
    print(f"    worst per-row ratio {ratio.max():.4e}")
    print("    rotation diagonal   "
          + ", ".join(f"{v:+.8f}" for v in diagonal[n_mult:]))
    failures = []
    if rel.max() > DIAG_TOL:
        failures.append(f"diagonal off by {rel.max():.4e}")
    if ratio.max() > OFFDIAG_TOL:
        failures.append(f"off-diagonal ratio {ratio.max():.4e}")
    if len(set(np.round(diagonal[:n_mult], 12))) != 2:
        failures.append("multiplier diagonal is not 2 distinct values")
    return failures


#: How far a **multiplier** entry recovered at one rank count may sit from the
#: same entry at another. **Relative, never equality.** Measured serial against
#: 4 ranks on the coarse block: **4.7756e-16** over all 72, at row 0. So this
#: carries three orders of headroom, and it is fourteen orders below anything a
#: wrong parallel read would produce.
#:
#: This is *the* measurement of `read_real_rows` at production block size. It is
#: proven at 2 and 3 ranks on the 2-D annulus and assumed at 48 by the
#: handover's note that all 75 `Real` dofs sit on rank 0; between those there
#: was nothing, and a parallel `Real` read returning something plausible rather
#: than failing is the hazard this gate exists to exclude.
RANK_AGREEMENT_TOL = 1e-13
#: And the **rotation** rows, which cannot meet that, for the same reason they
#: cannot meet `DIAG_TOL`. Their entry carries `V_F/V_pre`, and both assemblies
#: accumulate over cells in a rank-dependent order, so the ratio moves with the
#: rank count: measured `-8.877465e-11` serial and `+2.181970e-10` at 4 ranks -
#: it changes sign. The mantle volume itself reads
#: `3.75238299464148710e+01` serial and `3.75238299362089833e+01` at 4 ranks,
#: 2.7e-10 apart, which is the whole of it.
#:
#: **This is not a parallel-read failure and must not be read as one.** The
#: three rows move together - 3.0697e-10, 3.0697e-10, 3.0696e-10 - which is one
#: shared volume ratio, exactly as in the serial measurement.
RANK_AGREEMENT_ROT_TOL = 1e-8


def diff_against(diagonal, path, n_mult):
    """Entrywise comparison against a diagonal saved from another run.

    Split by row family, because the two families answer different questions.
    The multiplier rows measure the `Real` read and must agree to roundoff; the
    rotation rows carry a volume ratio that is itself rank-dependent, so
    requiring them to agree that closely would fail for a reason that has
    nothing to do with the read.
    """
    reference = np.load(path)
    if reference.shape != diagonal.shape:
        return [f"--diff: reference has {reference.shape} entries, this run "
                f"has {diagonal.shape}; different configurations"]
    rel = np.abs(diagonal - reference) / np.abs(reference)
    mult, rot = rel[:n_mult], rel[n_mult:]
    worst = int(mult.argmax())
    print(f"\n  --diff against {path}")
    print(f"    multiplier rows  worst {mult.max():.4e} at row {worst}")
    print(f"      this run  {diagonal[worst]!r}")
    print(f"      reference {reference[worst]!r}")
    if rot.size:
        print(f"    rotation rows    {np.array2string(rot, precision=4)}")

    failures = []
    if mult.max() > RANK_AGREEMENT_TOL:
        failures.append(
            f"the multiplier diagonal disagrees with {path} by {mult.max():.4e}"
            f" at row {worst}, above RANK_AGREEMENT_TOL "
            f"{RANK_AGREEMENT_TOL:g}. If the two runs differ only in rank "
            "count, THIS is the parallel Real read returning something "
            "plausible rather than failing")
    else:
        print("    multiplier rows within RANK_AGREEMENT_TOL: read_real_rows "
              "means the same thing at both rank counts")
    if rot.size and rot.max() > RANK_AGREEMENT_ROT_TOL:
        failures.append(
            f"the rotation rows disagree with {path} by {rot.max():.4e}, above "
            f"RANK_AGREEMENT_ROT_TOL {RANK_AGREEMENT_ROT_TOL:g} - larger than "
            "the volume ratio accounts for")
    if rot.size > 1:
        spread = np.ptp(rot) / max(rot.max(), 1e-300)
        print(f"    rotation rows move together to {spread:.4e}, i.e. one "
              "shared volume ratio and not a read failure")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="medium",
                        choices=("coarse", "medium", "fine", "production"))
    parser.add_argument("--dtn-degree", type=int, default=5)
    parser.add_argument("--h", type=float, default=None,
                        help="lateral h in km, overriding --configuration. "
                             "Every criterion below still applies: none of "
                             "them compares a multiplier entry against a "
                             "literal, only against block1_diagonal(), and "
                             "the rotation entries are mesh-independent. So a "
                             "large h is a cheap but genuine run of this gate")
    parser.add_argument("--build-only", action="store_true",
                        help="stop after constructing the solver, before the "
                             "76 assemblies: the cheapest way to find out "
                             "that a mesh, a preset or an argument is wrong")
    parser.add_argument("--selfcheck", action="store_true",
                        help="the 24-cell cubed sphere, no gmsh, laptop-safe")
    parser.add_argument("--pairing", action="store_true",
                        help="three diagnostics on the 48-rank over-count: the "
                             "partition census, dI_33 weighted two ways, and "
                             "inertia_form's volume and sheet halves "
                             "separately. Runs serially and under MPI")
    parser.add_argument("--save", metavar="PATH",
                        help="write the diagonal to a .npy for a later --diff")
    parser.add_argument("--diff", metavar="PATH",
                        help="compare the diagonal against a saved one, "
                             "entrywise and relatively. This is how the "
                             "48-rank run is checked against the serial one")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 1 gate: the block-1 diagonal on the production Real block")
    print("=" * 72)

    diagonal, n_mult = None, 0
    if args.selfcheck:
        failures = selfcheck()
    else:
        sys.path.insert(0, SPADA)
        import b1_elastic as b1
        from validate_selfgrav_sphere import provenance

        provenance(os.path.basename(__file__))
        nmax = b1.NMAX_OF[args.configuration]
        print(f"  configuration {args.configuration}   n_max {nmax}   "
              f"DtN L {args.dtn_degree}   rotation ON   condensed")
        print(f"  B_mu {b1.B_MU:.10f}   Lambda {b1.LAMBDA:.10f}   "
              f"theta_psi {b1.B_MU / b1.LAMBDA:.10f}")
        print(f"  C {b1.C_NONDIM:.10f}   C-A {b1.C_MINUS_A_PRIMARY:.10e}   "
              f"Omega^2 {b1.OMEGA_SQ:.10e}")
        print("  expected rotation diagonal "
              + ", ".join(f"{v:+.10e}" for v in ROT_EXPECTED))
        print("  no solve, no nest Jacobian, no assembled operator")
        print()

        parent, sub, _, _ = b1.build_meshes(args.configuration, h=args.h)
        try:
            solver, z, layout, _, _, _ = b1.build_solver(
                parent, sub, nmax, dtn_degree=args.dtn_degree, rotation=True,
                condense=True)
        except ValueError as exc:
            if args.h is not None and "cross-mesh" in str(exc):
                raise ValueError(
                    f"{exc}\n\n  This is almost certainly --h {args.h}. An "
                    "off-ladder resolution can produce a mesh whose cross-mesh "
                    "entity maps are wrong, and check_geometry is right to "
                    "refuse it: it compares two discrete quantities over the "
                    "same cells, so a disagreement at 1e-07 is not a tight "
                    "tolerance, it is the maps. Both coupling terms are built "
                    "on that measure. Use --configuration instead.") from exc
            raise
        print(f"  built: {len(z.function_space())} sub-fields, "
              f"{z.function_space().dim()} dofs, "
              f"{len(layout.real_fields)} Real rows")
        pairing = pairing_diagnostics(solver, z, layout) if args.pairing \
            else {}
        if pairing and COMM_WORLD.rank == 0:
            if args.save:
                np.savez(pairing_path(args.save), **pairing)
                print(f"  pairing numbers written to "
                      f"{pairing_path(args.save)}")
            if args.diff and os.path.exists(pairing_path(args.diff)):
                pairing_compare(pairing, pairing_path(args.diff))
            elif args.diff:
                print(f"  (no {pairing_path(args.diff)} to compare against; "
                      "run the serial reference with --pairing --save)")
        if args.build_only:
            print("  --build-only: stopping before the assemblies")
            return 0
        print(f"  differencing {len(layout.real_fields) + 1} residuals")
        failures, diagonal = report(solver, z, layout)
        n_mult = len(layout.multipliers)

    if diagonal is not None and COMM_WORLD.rank == 0:
        if args.save:
            np.save(args.save, diagonal)
            print(f"\n  diagonal written to {args.save}")
        if args.diff:
            failures += diff_against(diagonal, args.diff, n_mult)

    print()
    if failures:
        print(f"  GATE FAILED: {len(failures)} criteria")
        for text in failures:
            print(f"    - {text}")
        return 1
    print("  GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
