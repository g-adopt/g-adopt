"""Check that a run with pinned walls keeps its wall nodes in BNK's fixed set.

Used by ``smoke_pinned.slurm`` before the long runs of step S2. For each record it
checks:

- every Krylov solve has ``active_fixed`` equal to the number of wall nodes that
  the record states (``prior.wall_treatment``), so BNK put exactly the pinned nodes
  in its fixed set, from the first solve on;
- every gradient row has ``wall_max_dev_bottom`` and ``wall_max_dev_top`` exactly
  zero, so no pinned node moved at any evaluated point, the first included.

Usage: python check_pinned.py RECORD.json [RECORD.json ...]

Exit status 0 when every check holds in every record, 1 otherwise. Only the standard
library is used.
"""

import json
import sys


def check(path):
    """Print the checks of one record and return True when all hold."""
    record = json.load(open(path))
    walls = record["prior"]["wall_treatment"]
    expected = walls["wall_nodes_bottom"] + walls["wall_nodes_top"]
    ok = walls["bounds"] == "pinned"
    print(f"{path}: bounds {walls['bounds']}, operator {walls['operator_bc']}, "
          f"wall nodes {walls['wall_nodes_bottom']} + {walls['wall_nodes_top']} = {expected}")
    solves = [s for row in record["iterations"] for s in row.get("krylov_solves", [])]
    if not solves:
        print("   no Krylov solve recorded")
        ok = False
    for k, s in enumerate(solves):
        good = s["active_fixed"] == expected
        ok &= good
        print(f"   solve {k}: fixed {s['active_fixed']} lower {s['active_lower']} "
              f"upper {s['active_upper']} inactive {s['inactive']} "
              f"{'ok' if good else 'WRONG'}")
    deviation = max(max(row["wall_max_dev_bottom"], row["wall_max_dev_top"])
                    for row in record["gradients"])
    ok &= deviation == 0.0
    print(f"   largest wall deviation over {len(record['gradients'])} gradient rows: "
          f"{deviation!r} {'ok' if deviation == 0.0 else 'WRONG'}")
    return ok


if __name__ == "__main__":
    results = [check(path) for path in sys.argv[1:]]
    print("all checks hold" if all(results) else "A CHECK FAILED")
    sys.exit(0 if all(results) else 1)
