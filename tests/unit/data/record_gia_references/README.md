# Recording the stored GIA references

The scripts in this directory made the stored reference data of the GIA
unit tests in `tests/unit/data`. They need the earlier self-gravitating GIA
solver, `gadopt.gia_gravity`, which exists at commit `2f4b1397` and not
later. They do not run on the present code.

| Stored file | Script | Read by |
|---|---|---|
| `gia_gate_reference.json` | `record_all.py` | `test_gia_solver_gate.py`, `test_gia_solver_gate_3d.py` |
| `gia_forward_reference.json` | `record_all.py` | `test_gia_solver.py::test_three_steps_agree_with_the_reference` |
| `gia_old_checkpoint_*.h5` | `record_checkpoints.py`, then `repack.py` | the conversion tests of `test_gia_solver.py` |

All scripts use the stored mesh `gia_triangle_annulus.msh`, so the values do
not depend on the gmsh version of the machine. The JSON files hold the three
numbers of `probe_pairing` for each field, so they do not depend on the
numbering of the degrees of freedom.

## Procedure

1. Create a temporary worktree at the earlier solver:
   `git worktree add /tmp/gia-2f4b1397 2f4b1397`.
2. Copy `tests/unit/gia_solver_cases.py` of the present code into the
   temporary worktree as `tests/unit/cases_coord.py`. The scripts use only
   its stored mesh and its coordinate helpers, and they use the
   `gia_solver_cases.py` of `2f4b1397` for the earlier solver.
3. Copy `tests/unit/data/gia_triangle_annulus.msh` into
   `tests/unit/data/` of the temporary worktree.
4. Copy the three scripts of this directory into the root of the temporary
   worktree.
5. In the root of the temporary worktree, set `PYTHONPATH` to that root and
   run the scripts in serial:

   ```
   python record_all.py OUT_DIR
   python record_checkpoints.py RAW_DIR
   python repack.py RAW_DIR OUT_DIR
   ```

6. Copy the files of `OUT_DIR` into `tests/unit/data` of the present code.
7. Remove the temporary worktree: `git worktree remove /tmp/gia-2f4b1397`.

`record_all.py` also compares the new solver at `2f4b1397` with the earlier
one in the same run and stops if they disagree. The Firedrake of the run
must be one that the code at `2f4b1397` supports.

The checkpoint files are in git LFS (`.gitattributes`).
