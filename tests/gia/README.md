# GIA cases

This directory holds the case runs of the self-gravitating GIA solver
(`gadopt.gia_solver`) and the helper module that the cases and the unit
tests share.

`gia_cases.py` defines the configurations of the coupled GIA system (`Case`)
and the builders for them (`build_new`, `annulus_meshes`). It also holds the
functions that compare a state with the stored references of the earlier
solver (`coordinate_state`, `fingerprint`, `probe_pairing`). The stored mesh
and the reference values are in `tests/unit/data`.

The unit tests in `tests/unit` import the helper as `gia_cases`.
`tests/unit/conftest.py` puts this directory on `sys.path` for them.

CAUTION: Do not change the labels of `Case` or the functions
`fingerprint`, `probe_pairing` and `coordinate_state`. The stored references
depend on them, and they cannot be recorded again on the present code.

## The cases

Each case is a driver that runs one entry in one process and writes its
numbers to `out/`, and a `test_gia_*.py` file that reads them and asserts.
`gia_entries.py` lists every entry, its driver, its core count, its outputs
and the tolerances. `meta.py` makes one `doit run_case` step of each entry.
`gia_march.py` holds the scenarios (`SEA`, `LOAD`, the ice history), the
march loop and the diagnostics of one step. A test file imports only
`numpy`, `pandas`, `pytest` and `gia_entries`, so it never builds a solver.

The unit tests in `tests/unit` run on one rank. A check that needs more
than one rank is an entry with `cores` above 1: an entry of the `parallel`
case (`gia_parallel.py`), or a multi-rank entry of another case. A check of
a file read on another number of ranks is a write entry and a read entry,
with `deps` from the read entry to the files of the write entry.

The `gravity` case (`gia_gravity.py`, tests in `test_gia_gravity_case.py`)
runs `GravitySolver` alone against closed forms. Its test file has another
basename than `test_gia_gravity.py`, because `tests/unit/test_gia_gravity.py`
exists and pytest does not collect two test modules of one name. An entry
with `local` set to False runs a 3-D solve. The development machine does not
run it, and its tests skip where its output is missing.

To run one entry and its tests:

```
batch_mode=NO doit run run_case:tests/gia:forward-sea-direct
python -m pytest tests/gia
```

A driver can also run by hand from this directory, for example
`python gia_forward.py --entry sea-direct`.

## The weekly entry

The entry `sphere-submesh` (`gia_sphere_submesh.py`) runs the `SEA`
configuration on a tetrahedral sphere with a mantle submesh, on 48 cores.
gmsh writes the mesh in the job. The entry has `"hpc": True`, so it is a
step of `doit run_case_hpc` and not of `doit run_case`. The weekly long-test
workflow runs it on Gadi. `test_gia_hpc.py` reads its outputs. That file is
marked `longtest`, and its tests skip where the outputs do not exist.
`doit check_hpc` runs only `test_gia_hpc.py` (`pytest_hpc` in `meta.py`).

To run the entry and its tests on Gadi, from the repository root:

```
doit run run_case_hpc:tests/gia:weekly-sphere-submesh
python3 -m pytest tests/gia/test_gia_hpc.py
```

The recorded values of this entry come from a Gadi run, not from the
development machine. After a review of the run, record them from this
directory with `python3 gia_record_expected.py --case weekly`. Until
`expected/sphere-submesh.txt` exists, the recorded-value test skips and
prints the measured table.

## Recorded values

Some assertions compare with values recorded from this code in
`expected/`. A recorded value detects a change of the results. It does not
detect a wrong result, so every property that a closed form, a limit, an
order, an exact identity or an independent system can check is checked
against that source instead.

To record the values of a case after a review of the measured values, run
the entries, then:

```
python gia_record_expected.py --case forward [--entry sea-direct]
```

The script adds a header with the commits, the PETSc version, the host and
the date. Record an entry once. If an expected file changes later, give the
cause in the commit message. Do not record on the CI runner to make a test
pass.
