"""Compare two JSON run records of the box benchmark row by row.

The prior-covariance campaign accepts a code change only if it leaves a run where it
was before, so it needs one plain answer to the question "do these two runs take the
same path?". This script gives it. It reads two records written by ``inverse.py``,
pairs their rows, and prints for each pair the relative difference of every compared
field, then the largest relative difference per field and the first row where a
field is not bitwise equal.

By default it compares the fields named in the work order of step S0: ``J``,
``initial_misfit``, ``gradient_norm`` and the three sweep counts ``forward``,
``adjoint`` and ``hessian``. ``--all-fields`` compares every numeric field that both
rows hold, except the wall-clock fields, which differ between any two runs.

Rows are paired by position. Iteration rows carry their iteration number, and the
script checks that the paired numbers agree. A run stopped by the work budget stops
at a point that depends on wall time, because the budget prices a Hessian-vector
product at its measured cost, so two otherwise identical runs can hold a different
number of rows. Only the rows that both records hold are compared, and the counts
are printed.

Usage:
    python compare_records.py A.json B.json [--rows iterations|gradients|both]
                              [--all-fields] [--fields J gradient_norm ...]
                              [--solves] [--max-rows N] [--quiet]

``--solves`` also compares the Krylov solve records that the iteration rows hold
(``krylov_solves``), paired by position over the iteration rows that both records
hold: every numeric field that both solve records hold, and the lists of residual
norms and Ritz values entry by entry. A field that only one record holds, for
example a field that a newer recording adds, is not compared.

``--max-rows N`` compares only the first N rows of each table (and, with
``--solves``, the solves of the first N iteration rows). Use it when two runs are
meant to agree up to a known iteration and then part, for example a run that
changes the Hessian shift, which acts only after the first negative-curvature
solve.

Exit status 0 when every compared field is bitwise equal in every paired row, 1
otherwise. Only the standard library is used, so the script runs on any machine.
"""

import argparse
import json
import math
import sys

# The fields of step S0: the objective, the twin-experiment measure, the gradient
# norm of the recorder, and the three sweep counts.
DEFAULT_FIELDS = ("J", "initial_misfit", "gradient_norm", "forward", "adjoint", "hessian")

# Fields that measure wall time, or are computed from it. They differ between any two
# runs and say nothing about the path of the optimiser. ``work`` prices a
# Hessian-vector product at its measured cost, so it belongs here too.
TIMING_FIELDS = {"time", "work", "wall_time"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", help="first record (the reference)")
    parser.add_argument("b", help="second record")
    parser.add_argument("--rows", default="iterations",
                        choices=("iterations", "gradients", "both"),
                        help="which rows to pair: one per optimiser iteration, one "
                             "per gradient evaluation, or both tables in turn")
    parser.add_argument("--fields", nargs="+", default=None,
                        help="fields to compare; default: " + " ".join(DEFAULT_FIELDS))
    parser.add_argument("--all-fields", action="store_true",
                        help="compare every numeric field that both rows hold, "
                             "except the wall-clock fields")
    parser.add_argument("--solves", action="store_true",
                        help="also compare the Krylov solve records of the paired "
                             "iteration rows")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="compare only the first N rows of each table")
    parser.add_argument("--quiet", action="store_true",
                        help="print the summary table only, not one line per row")
    return parser.parse_args(argv)


def is_number(value):
    """True for an int or a float, and False for a bool, None, a list or a string."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def relative_difference(a, b):
    """``|a - b| / max(|a|, |b|)``, zero when both are zero, inf for a NaN mismatch."""
    if a == b:
        return 0.0
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return 0.0
    scale = max(abs(a), abs(b))
    if scale == 0.0 or not math.isfinite(scale):
        return math.inf
    return abs(a - b) / scale


def fields_to_compare(rows_a, rows_b, args):
    """The ordered list of field names to compare in one table.

    Args:
        rows_a, rows_b: the paired row lists.
        args: the parsed arguments.

    Returns:
        list of str.
    """
    if args.fields:
        return list(args.fields)
    if not args.all_fields:
        return list(DEFAULT_FIELDS)
    names = []
    for row_a, row_b in zip(rows_a, rows_b):
        for key, value in row_a.items():
            if (key not in names and key not in TIMING_FIELDS and key in row_b
                    and is_number(value) and is_number(row_b[key])):
                names.append(key)
    return names


def compare_table(name, rows_a, rows_b, args):
    """Compare one table of rows and print the result.

    Args:
        name: ``"iterations"`` or ``"gradients"``, for the printout.
        rows_a, rows_b: the two row lists.
        args: the parsed arguments.

    Returns:
        True when every compared field is bitwise equal in every paired row.
    """
    n = min(len(rows_a), len(rows_b))
    print(f"== {name}: {len(rows_a)} rows in A, {len(rows_b)} rows in B, "
          f"{n} rows compared")
    if n == 0:
        return True
    paired_a, paired_b = rows_a[:n], rows_b[:n]
    fields = fields_to_compare(paired_a, paired_b, args)

    # Iteration rows must pair the same iteration; a mismatch means one record
    # skipped or repeated an iteration, and every later pair is then meaningless.
    if name == "iterations":
        for k, (row_a, row_b) in enumerate(zip(paired_a, paired_b)):
            if row_a.get("iteration") != row_b.get("iteration"):
                print(f"   row {k}: iteration {row_a.get('iteration')} in A against "
                      f"{row_b.get('iteration')} in B; pairing stops here")
                paired_a, paired_b = paired_a[:k], paired_b[:k]
                break

    worst = {field: (0.0, None) for field in fields}
    first = {field: None for field in fields}
    missing = {field: 0 for field in fields}
    label = "iteration" if name == "iterations" else "row"
    if not args.quiet:
        print("   " + f"{label:>9s} " + " ".join(f"{f[:14]:>14s}" for f in fields))
    for k, (row_a, row_b) in enumerate(zip(paired_a, paired_b)):
        cells = []
        for field in fields:
            a, b = row_a.get(field), row_b.get(field)
            if not (is_number(a) and is_number(b)):
                missing[field] += 1
                cells.append(f"{'-':>14s}")
                continue
            rel = relative_difference(a, b)
            if rel > worst[field][0]:
                worst[field] = (rel, k)
            if a != b and first[field] is None:
                first[field] = k
            cells.append(f"{rel:14.2e}")
        if not args.quiet:
            index = row_a.get("iteration", k) if name == "iterations" else k
            print("   " + f"{index:>9} " + " ".join(cells))

    print("   " + f"{'field':<34s} {'max rel diff':>12s} {'at row':>7s} "
          f"{'first unequal row':>18s}")
    identical = True
    for field in fields:
        rel, where = worst[field]
        first_row = first[field]
        if first_row is not None:
            identical = False
        note = f"  ({missing[field]} rows without the field)" if missing[field] else ""
        print("   " + f"{field:<34s} {rel:12.3e} "
              f"{'-' if where is None else where:>7} "
              f"{'none' if first_row is None else first_row:>18}{note}")
    print(f"   bitwise equal in every compared field: {'yes' if identical else 'no'}")
    return identical


def solve_values(solve):
    """The comparable entries of one Krylov solve record, as ``{name: number}``.

    Scalars are kept under their own name; the lists ``residual_norms`` and
    ``ritz_values`` are split into one entry per position, so that a difference
    in one Krylov iteration shows with its index. Wall-clock fields are left out.
    """
    out = {}
    for key, value in solve.items():
        if key in TIMING_FIELDS:
            continue
        if is_number(value):
            out[key] = value
        elif isinstance(value, list) and all(is_number(v) for v in value):
            out[f"{key}#len"] = len(value)
            for k, v in enumerate(value):
                out[f"{key}[{k}]"] = v
    return out


def compare_solves(rows_a, rows_b):
    """Compare the Krylov solve records of the iteration rows that both records hold.

    Args:
        rows_a, rows_b: the iteration rows of the two records.

    Returns:
        True when every compared entry is bitwise equal.
    """
    solves_a, solves_b = [], []
    for row_a, row_b in zip(rows_a, rows_b):
        if row_a.get("iteration") != row_b.get("iteration"):
            break
        solves_a.extend(row_a.get("krylov_solves", []))
        solves_b.extend(row_b.get("krylov_solves", []))
    n = min(len(solves_a), len(solves_b))
    print(f"== Krylov solves: {len(solves_a)} in A, {len(solves_b)} in B, {n} compared")
    identical = True
    worst = {}
    compared = 0
    for k in range(n):
        a, b = solve_values(solves_a[k]), solve_values(solves_b[k])
        for key in a.keys() & b.keys():
            compared += 1
            rel = relative_difference(a[key], b[key])
            base = key.split("[")[0]
            if rel > worst.get(base, (0.0, None))[0]:
                worst[base] = (rel, k)
            if a[key] != b[key]:
                identical = False
    for base, (rel, k) in sorted(worst.items()):
        print(f"   {base:<34s} max rel diff {rel:12.3e} at solve {k}")
    if len(solves_a) != len(solves_b):
        identical = False
        print("   the numbers of solves differ")
    print(f"   {compared} entries compared; bitwise equal: {'yes' if identical else 'no'}")
    return identical


def main(argv=None):
    args = parse_args(argv)
    with open(args.a) as f:
        record_a = json.load(f)
    with open(args.b) as f:
        record_b = json.load(f)
    print(f"A: {args.a}\nB: {args.b}")
    for key in ("optimiser", "riesz", "window", "initial_radius", "work_budget",
                "stop_reason"):
        print(f"   {key:<16s} A: {record_a.get(key)!s:<40s} B: {record_b.get(key)!s}")
    for key in ("metric_c_d", "metric_c_s"):
        a = record_a.get("weights", {}).get(key)
        b = record_b.get("weights", {}).get(key)
        print(f"   {key:<16s} A: {a!r:<40s} B: {b!r}  "
              f"{'equal' if a == b else 'DIFFERENT'}")

    tables = ("iterations", "gradients") if args.rows == "both" else (args.rows,)
    identical = True
    limit = slice(None, args.max_rows)
    for table in tables:
        identical &= compare_table(table, record_a.get(table, [])[limit],
                                   record_b.get(table, [])[limit], args)
    if args.solves:
        identical &= compare_solves(record_a.get("iterations", [])[limit],
                                    record_b.get("iterations", [])[limit])
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
