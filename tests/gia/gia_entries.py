"""The entries of the GIA cases, their tolerances and the lists the tests use.

One entry is one driver process for one configuration. `meta.py` turns each
entry of `ENTRIES` into one step of `doit run_case`, and the test files
parametrize over the same lists. A name of an entry, a tolerance or an output
file exists in this file only.

This module is plain Python with no Firedrake import, because the test files
import it and the weekly long-test run collects them on a login node.
"""

from pathlib import Path

#: The directory of the cases; every output path is relative to it.
CASE_DIR = Path(__file__).resolve().parent
OUT = "out"
EXPECTED = "expected"

# -- forward ------------------------------------------------------------------------

#: The forward entries, by name. Each value says how the entry differs from
#: the base configuration `base` (`SEA` or `LOAD` of `gia_march.py`):
#: `changes` are `Case` fields, `preset` the solver preset, and
#: `default_options` whether the tight tolerances of `gia_cases.build_new`
#: are added (True) or the preset keeps its own (False).
#: `save_state` is the 1-based step after which the direct reference writes
#: its state for the restart entries.
FORWARD = {
    "sea-direct": dict(base="SEA", preset="direct", save_state=4),
}

#: Serial forward entries with recorded values in `expected/`.
FORWARD_RECORDED = ["sea-direct"]

#: Entries whose solve is direct (tight tolerances); the others are iterative.
FORWARD_DIRECT = [name for name, e in FORWARD.items() if e["preset"] == "direct"]

#: Relative tolerance of a recorded column: the direct entries solve to
#: round-off (Newton `snes_rtol` 1e-11), the iterative entries to the
#: tolerances of the iterative preset.
RECORDED_RTOL_DIRECT = 1e-8
RECORDED_RTOL_ITERATIVE = 1e-5

#: A column whose expected value is below this fraction of the displacement
#: norm `u_l2` is round-off; it is compared absolutely, with this bound
#: times `u_l2`.
ROUNDOFF_FRACTION = 1e-9

#: Columns that are counts, compared by their own rules and not as values.
COUNT_COLUMNS = ("newton", "linear", "operator_version")

#: The largest Newton count of a step with a moving coastline. The scenario
#: was measured at 3 to 5 iterations per step.
MAX_NEWTON_MOVING = 5

#: Each coupling must do something: at the last step the named column is
#: above this fraction of `u_l2`, so a stored value of a quantity that sits
#: at round-off cannot pass for a check.
COUPLING_FLOOR = 1e-4
COUPLING_COLUMNS = {"sea-direct": ("shift", "com_0")}

#: `|D_i| <= DIPOLE_RTOL * sheet_moment_i`: the centre of mass holds to this
#: fraction of the first moment of the sheets that the displacement must
#: cancel (the bound of `test_sea_level.py` for the frame).
DIPOLE_RTOL = 1e-8

#: `|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2`: the fluid core keeps the
#: volume of the core.
CMB_FLUX_RTOL = 1e-9


def forward_log(entry):
    """The path of the per-step table of a forward entry, relative to `CASE_DIR`."""
    return f"{OUT}/forward-{entry}.log"


def forward_state(entry):
    """The path of the state file a forward entry writes for a restart."""
    return f"{OUT}/forward-{entry}.h5"


def forward_expected(entry):
    """The path of the recorded values of a forward entry."""
    return f"{EXPECTED}/forward-{entry}.txt"


def _forward_entries():
    """The `ENTRIES` items of the forward case."""
    out = []
    for name, entry in FORWARD.items():
        outputs = [forward_log(name)]
        if "save_state" in entry:
            outputs.append(forward_state(name))
        item = {"case": "forward", "entry": name, "driver": "gia_forward.py",
                "cores": entry.get("cores", 1), "outputs": outputs}
        if "deps" in entry:
            item["deps"] = entry["deps"]
        out.append(item)
    return out


#: The files that `gia_record_expected.py` copies, by case: pairs of the
#: driver output and the expected file, both relative to `CASE_DIR`.
RECORDED = {
    "forward": {name: (forward_log(name), forward_expected(name))
                for name in FORWARD_RECORDED},
}

#: Every step of `meta.py`: `case`, `entry`, `driver`, `cores`, `outputs`,
#: and optionally `deps`, `hpc` and `launcher_args`.
ENTRIES = _forward_entries()
