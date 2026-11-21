"""Write the stored checkpoint files of the earlier GIA solver (serial).

This script needs the earlier solver (`gadopt.gia_gravity`), which exists at
commit 2f4b1397 and not later. Run it from the root of a checkout of
2f4b1397, as the README of this directory describes, with the output
directory as the first argument. It writes one file per case after two steps
of the earlier solver, in the layouts that the drivers of that solver wrote:
`selfgrav_common` (every sub-function of the mixed solution as
`solution_{i}` in the broken encoding, the BDF2 levels as `time_*`, the
Quadrature history also as `internal_variables_0`) and Exp01 (own names,
unbroken, `real_{i}` in the metadata). `repack.py` then compresses the files
into the ones stored in `tests/unit/data`.

Usage (from the root of the 2f4b1397 checkout):
    python record_checkpoints.py RAW_DIR
"""
import sys
sys.path.insert(0, "tests/unit")
import gadopt  # noqa: E402,F401  (gadopt before firedrake)
import firedrake as fd  # noqa: E402
import gia_solver_cases as C  # noqa: E402  (2f4b1397 helpers: build_old)
import cases_coord as H  # noqa: E402  (present helpers: stored mesh)
from gadopt.viscoelastic_checkpointing import CheckpointField  # noqa: E402
from gadopt.viscoelastic_timestepping import time_history_fields, time_history_metadata  # noqa: E402

out_dir = sys.argv[1]
Case = C.Case
FOUR = (.08, .13, .07, .12)
CASES = {
    "gia_old_checkpoint_quadrature_bdf2.h5": (
        Case(family="Quadrature", scheme="bdf2", representation="lowrank",
             model="maxwell2", core="fluid", centre=True), False),
    "gia_old_checkpoint_dg_be.h5": (Case(rotation=True), False),
    "gia_old_checkpoint_dg_bdf2.h5": (
        Case(scheme="bdf2", representation="lowrank", core="fluid", rotation=True,
             centre=True), False),
    "gia_old_checkpoint_exp01.h5": (
        Case(scheme="bdf2", representation="lowrank", core="fluid", rotation=True,
             centre=True), True),
}

meshes = H.annulus_meshes()
parent, sub = meshes
print("parent mesh name", parent.name)
for filename, (case, exp01) in CASES.items():
    old, z, layout = C.build_old(meshes, case, dt=C.dt_function(sub),
                                 declare_nullspace=case.core == "fluid")
    for h in FOUR[:2]:
        old.dt.assign(h)
        old.solve()
    zs = z.subfunctions
    info = dict(time_history_metadata(old)) if old.time_integrator else {}
    if exp01:
        fields = [("checkpoint_displacement", zs[layout.displacement]),
                  ("checkpoint_internal_variables", zs[layout.internal_variable_field])]
        fields += time_history_fields(old)
        encoded = [CheckpointField(n, f, parent, broken=False) for n, f in fields]
        potential = fd.Function(fd.FunctionSpace(parent, zs[layout.potential].ufl_element()),
                                name="checkpoint_potential").assign(zs[layout.potential])
        for i, f in enumerate(zs):
            if f.ufl_element().family() == "Real":
                info[f"real_{i}"] = float(f.dat.data_ro_with_halos[0])
    else:
        fields = [(f"solution_{i}", f) for i, f in enumerate(zs)]
        if case.family == "Quadrature":
            fields.append(("internal_variables_0", old.internal_variables))
        fields += time_history_fields(old)
        encoded = [CheckpointField(n, f, parent) for n, f in fields]
    info = {k: float(v) for k, v in info.items()} | {"step": 2.0}
    with fd.CheckpointFile(f"{out_dir}/{filename}", "w") as chk:
        chk.save_mesh(parent)
        for field in encoded:
            chk.save_function(field.encode(), idx=0, timestepping_info=info)
        if exp01:
            chk.save_function(potential, idx=0, timestepping_info=info)
    print(filename, sorted(info), [f.stored.name() for f in encoded], flush=True)
