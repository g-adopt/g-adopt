"""Check the recording of BNK's Hessian shift in one or more run records.

Used by ``smoke_s2b.slurm`` before the long runs of step S2b. For each record it
checks:

- every Krylov solve holds ``hessian_shift``, and BNK made at most one ``MatShift``
  call per solve (``hessian_shift_calls`` is 0 or 1);
- every recorded value equals the value that the update rules of PETSc v3.25.0
  give from the previous solve (``summarise_shift.replay_shift``), to 1e-12
  relative. The rules use the parameters of the run, so a perturbation option
  that did not reach BNK makes this check fail at the first change of the shift;
- with ``--expect-first VALUE``: the first positive shift of the record equals
  ``VALUE`` to 1e-12 relative. For a run whose shift starts from zero, this is the
  value that BNK sets after the first negative-curvature solve, so it tests the
  options that clamp the initial perturbation (``-tao_bnk_imin``, ``-tao_bnk_imax``)
  or, for the default options, ``imfac * gnorm``.

Usage:
    python check_shift.py RECORD.json [RECORD.json ...] [--expect-first VALUE]
    python check_shift.py RECORD.json --expect-first-from-gnorm

``--expect-first-from-gnorm`` takes the expected first value from the default rule
``min(imax, max(imin, imfac * gnorm))`` with the run's parameters and ``gnorm`` from
the record, which is the check for a run with the default options.

Exit status 0 when every check holds in every record, 1 otherwise. Only the standard
library is used.
"""

import argparse
import json
import sys

from summarise_shift import _initialise, _same, replay_shift, shift_parameters


def check(path, expect_first=None, expect_from_gnorm=False):
    """Print the checks of one record and return True when all hold."""
    record = json.load(open(path))
    solves = [(row["iteration"], s) for row in record["iterations"]
              for s in row.get("krylov_solves", [])]
    ok = bool(solves)
    print(f"{path}: {len(solves)} Krylov solves, shift parameters "
          f"{shift_parameters(record)}")
    for iteration, s in solves:
        if "hessian_shift" not in s or s.get("hessian_shift_calls", 0) > 1:
            print(f"   iteration {iteration}: shift missing or more than one call: "
                  f"{s.get('hessian_shift')!r}, {s.get('hessian_shift_calls')!r}")
            ok = False
    replay = replay_shift(record)
    for entry in replay:
        good = entry["match"] is not False
        ok &= good
        print(f"   iteration {entry['iteration']:3d} {entry['reason']:>22s} recorded "
              f"{entry['recorded']!r:>24s} replayed {entry['replayed']!r:>24s} "
              f"{entry['rule']}{'' if good else '  MISMATCH'}")
    if expect_from_gnorm:
        # The first positive shift follows the first solve with an indefinite
        # reason. Its gnorm is that of the iteration row before that solve's row.
        gnorm = {row["iteration"]: row.get("gradient_norm_metric")
                 for row in record["iterations"]}
        first_indefinite = next((e for e in replay if e["reason"] == "CONVERGED_NEG_CURVE"),
                                None)
        if first_indefinite is None:
            print("   no negative-curvature solve, so no expected first shift")
            ok = False
        else:
            expect_first = _initialise(0.0, gnorm[first_indefinite["iteration"] - 1],
                                       shift_parameters(record))
    if expect_first is not None:
        first = next((e["recorded"] for e in replay if e["recorded"]), None)
        good = first is not None and _same(first, expect_first)
        ok &= good
        print(f"   first positive shift {first!r}, expected {expect_first!r}: "
              f"{'ok' if good else 'WRONG'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="+")
    parser.add_argument("--expect-first", type=float, default=None)
    parser.add_argument("--expect-first-from-gnorm", action="store_true")
    args = parser.parse_args()
    results = [check(p, args.expect_first, args.expect_first_from_gnorm)
               for p in args.records]
    print("all checks hold" if all(results) else "A CHECK FAILED")
    sys.exit(0 if all(results) else 1)
