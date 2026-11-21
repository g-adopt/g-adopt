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

The package `gia_models` holds the builders, meshes, sources, forward maps
and constants that the case drivers and the unit tests share. Each module
holds the helpers of one file of `tests/unit` (for example
`gia_models.sea_level` for `test_sea_level.py`). A driver imports them from
`gia_models` and never from a test module, so that a change to a unit test
file cannot break a driver. When a unit test needs a helper that a driver
also uses, put the helper in `gia_models` and import it from there in both.

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
run it.

## Missing outputs and `GADOPT_GIA_LOCAL`

Some entries run on CI only: the entries with `local` set to False (3-D
solves and runs on more than 2 ranks) and the weekly entry. If the output
of such an entry is missing, its tests fail. The CI case stage ignores the
exit status of the drivers, so a missing output means that a driver crashed
or did not start, and the failed tests show this.

The development machine does not run these entries. On that machine, set
the environment variable `GADOPT_GIA_LOCAL` to `1`. The tests of these
entries then skip when their outputs are missing, and the skip message
names the entry. Do not set the variable on CI or in the weekly run. The
variable has no effect on the other entries: a missing output of an entry
that runs everywhere is always a failure.

To run one entry and its tests on the development machine:

```
batch_mode=NO doit run run_case:tests/gia:forward-sea-direct
GADOPT_GIA_LOCAL=1 python -m pytest tests/gia
```

A driver can also run by hand from this directory, for example
`python gia_forward.py --entry sea-direct`.

## The weekly entry

The entry `sphere-submesh` (`gia_sphere_submesh.py`) runs the `SEA`
configuration on a tetrahedral sphere with a mantle submesh, on 48 cores.
gmsh writes the mesh in the job. The entry has `"hpc": True`, so it is a
step of `doit run_case_hpc` and not of `doit run_case`. The weekly long-test
workflow runs it on Gadi. `test_gia_hpc.py` reads its outputs. That file is
marked `longtest`. If the outputs do not exist, its tests fail, or skip
where `GADOPT_GIA_LOCAL` is `1`.
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
