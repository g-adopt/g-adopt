"""Check the S2c records: the metric shift and the negative-curvature record.

Used by ``smoke_s2c.slurm`` before the long runs of step S2c. For each record:

- every recorded shift follows the update rules of BNK (``summarise_shift.replay_shift``),
  which are the same in both shift modes, because only the operator changes;
- every solve that stops on negative curvature holds ``negative_curvature``, its
  curvature ``kappa`` (rebuilt from the last Hessian action) is not positive
  beyond roundoff (``kappa <= 1e-10 * (|pHp| + pert * |pAp or pp|)``), and its
  ``pAp`` and ``pp`` are positive;
- with ``--mode metric``: the record states ``bnk_shift = metric``, every solve
  holds ``operator_shift`` equal to ``hessian_shift`` (the Hessian operator applied
  the shift that BNK set), and at least one solve has a positive shift;
- with ``--mode identity``: the record states ``bnk_shift = identity`` and no
  solve holds ``operator_shift``;
- with ``--need-negative-curvature``: at least one negative-curvature solve exists.

Usage:
    python check_s2c.py RECORD.json --mode metric|identity [--need-negative-curvature]

Exit status 0 when every check holds, 1 otherwise. Only the standard library is used.
"""

import argparse
import json
import sys

from summarise_shift import replay_shift


def check(path, mode, need_negative_curvature):
    """Print the checks of one record and return True when all hold."""
    record = json.load(open(path))
    ok = record.get("bnk_shift") == mode
    print(f"{path}: bnk_shift {record.get('bnk_shift')!r}, expected {mode!r}")
    solves = [(row["iteration"], s) for row in record["iterations"]
              for s in row.get("krylov_solves", [])]
    mismatches = [e for e in replay_shift(record) if e["match"] is False]
    print(f"   {len(solves)} solves, shift replay mismatches {len(mismatches)}")
    ok &= not mismatches
    negative = 0
    positive_shift = False
    for iteration, s in solves:
        shift = s.get("hessian_shift")
        positive_shift |= bool(shift)
        if mode == "metric":
            good = s.get("operator_shift") == shift
            if not good:
                print(f"   iteration {iteration}: operator_shift {s.get('operator_shift')!r} "
                      f"differs from hessian_shift {shift!r}")
            ok &= good
        elif "operator_shift" in s:
            print(f"   iteration {iteration}: operator_shift present in identity mode")
            ok = False
        if s["reason_name"] != "CONVERGED_NEG_CURVE":
            continue
        negative += 1
        nc = s.get("negative_curvature")
        if not nc:
            print(f"   iteration {iteration}: negative-curvature solve without a record")
            ok = False
            continue
        size = abs(nc["pHp"]) + nc["pert"] * (nc["pAp"] if nc["shift_mode"] == "metric" else nc["pp"])
        good = nc["kappa"] <= 1e-10 * size and nc["pAp"] > 0 and nc["pp"] > 0
        ok &= good
        print(f"   iteration {iteration}: kappa {nc['kappa']:.4e} pHp {nc['pHp']:.4e} "
              f"pAp {nc['pAp']:.4e} pp {nc['pp']:.4e} pert {nc['pert']:.4e} "
              f"({nc['shift_mode']}) pHp/pAp {nc['pHp_over_pAp']:.4e} "
              f"{'ok' if good else 'WRONG'}")
    if need_negative_curvature and negative == 0:
        print("   no negative-curvature solve")
        ok = False
    if mode == "metric" and not positive_shift:
        print("   no solve with a positive shift, so the metric shift was never applied")
        ok = False
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="+")
    parser.add_argument("--mode", required=True, choices=("identity", "metric"))
    parser.add_argument("--need-negative-curvature", action="store_true")
    args = parser.parse_args()
    results = [check(p, args.mode, args.need_negative_curvature) for p in args.records]
    print("all checks hold" if all(results) else "A CHECK FAILED")
    sys.exit(0 if all(results) else 1)
