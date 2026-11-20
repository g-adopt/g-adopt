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
