"""Driver of the `forward` case: one forward march of the coupled GIA solver.

Usage: `python gia_forward.py --entry <name>`, with the entry names of
`gia_entries.FORWARD`. The driver builds the configuration of the entry,
marches it over `gia_march.FORWARD_STEPS` with the ice history
`gia_march.ice_at`, and writes one row of `gia_march.diagnostics` per step
to `out/forward-<entry>.log`. An entry with `save_state` also writes the
parent mesh and the solver state after that step to `out/forward-<entry>.h5`
(`gadopt.gia_state.write_state`), for the restart entries.

The test `test_gia_forward.py` reads the table; this script asserts nothing.
"""

import argparse
from dataclasses import replace

import gadopt  # noqa: F401  (gadopt is imported before firedrake)
import firedrake as fd
from gadopt.gia_state import write_state

import gia_cases as C
import gia_march as M
from gia_entries import CASE_DIR, FORWARD, forward_log, forward_state

BASES = {"SEA": M.SEA, "LOAD": M.LOAD}


def configuration(entry):
    """The `Case` of a forward entry: its base with the entry's changes."""
    return replace(BASES[entry["base"]], **entry.get("changes", {}))


def run_entry(name):
    """March the forward entry `name` and write its table (and its state)."""
    entry = FORWARD[name]
    meshes = C.annulus_meshes()
    run = M.build(meshes, configuration(entry), preset=entry["preset"],
                  default_options=entry.get("default_options", True))
    M.enable_residual_history(run.solver)
    rows = []

    def on_step(k, run):
        rows.append(M.diagnostics(run, k))
        if entry.get("save_state") == k + 1:
            save(run, CASE_DIR / forward_state(name))

    M.advance(run, M.FORWARD_STEPS, on_step=on_step)
    M.write_log(CASE_DIR / forward_log(name), rows)


def save(run, path):
    """Write the parent mesh and the solver state to `path`, collectively.

    The restart reads the parent from the file (`CheckpointFile.load_mesh`)
    and builds its solver on it, as `gadopt.gia_state.read_state` requires.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = run.psi.function_space().mesh()
    with fd.CheckpointFile(str(path), "w") as chk:
        chk.save_mesh(parent)
        write_state(chk, run.solver, idx=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(FORWARD))
    run_entry(parser.parse_args().entry)
