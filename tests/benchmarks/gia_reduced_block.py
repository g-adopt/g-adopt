"""Compare u/psi inner coupling strengths; run from the repository root.

source /data/will/codex_sbx/firedrake-2026.10/activate.sh
python tests/benchmarks/gia_reduced_block.py

This small 2D experiment reports work counts, not production scaling.
"""
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'unit'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'gia'))
# The imports below need the three paths above, so they follow them. The unit
# test modules import their GIA helpers (`gia_cases`, `gia_reference_mixed`,
# `gia_models`) from `tests/gia`.
import gadopt  # noqa: E402  (imported before firedrake, as G-ADOPT requires)
import firedrake as fd  # noqa: E402
import test_gia_direct_bdf2 as test  # noqa: E402
from test_gia_eliminated_lowrank import load  # noqa: E402

meshes = test.meshes.__wrapped__()
_, sub = meshes
control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
original_apply = gadopt.ReducedGIABlockPC.apply
stats = {}


def counted_apply(self, pc, x, y):
    original_apply(self, pc, x, y)
    stats['block_applications'] += 1
    stats['inner_iterations'] += self.condensed_ksp.getIterationNumber()


gadopt.ReducedGIABlockPC.apply = counted_apply
reference = None
for name, extra in (
    ('coupled_1e-4', {}),
    ('coupled_1e-2', {'dtn_fieldsplit_0_condensed_ksp_rtol': 1e-2}),
    ('one_u_psi_sweep', {'dtn_fieldsplit_0_condensed_ksp_type': 'preonly'}),
):
    stats = dict(block_applications=0, inner_iterations=0)
    solver, dt, ice = test.build(meshes, control, direct=True, representation='lowrank',
                                 iterative=True, options_extra=extra)
    outer = newton = 0
    times = []
    try:
        for k, h in enumerate(test.STEPS):
            dt.assign(h)
            load(ice, k)
            start = perf_counter()
            solver.solve()
            times.append(perf_counter()-start)
            outer += solver.solver.snes.getLinearSolveIterations()
            newton += solver.solver.snes.getIterationNumber()
        if reference is None:
            reference = solver.solution.copy(deepcopy=True)
        with solver.solution.dat.vec_ro as x, reference.dat.vec_ro as y:
            error = x.duplicate()
            x.copy(error)
            error.axpy(-1., y)
            relative_error = error.norm()/y.norm()
            error.destroy()
        block = solver.solver.snes.getKSP().getPC().getPythonContext()
        reduced = block.pc.getFieldSplitSchurGetSubKSP()[0].getPC().getPythonContext()
        potential = reduced.condensed_ksp.getPC().getFieldSplitSubKSP()[1].getPC().getPythonContext()
        result = dict(case=name, outer_iterations=outer, newton_iterations=newton,
                      relative_state_error=relative_error, seconds_per_step=times,
                      potential_column_builds=potential.column_builds, **stats)
    except Exception as error:
        result = dict(case=name, error=str(error), **stats)
    if sub.comm.rank == 0:
        print('BLOCK_COMPARISON '+json.dumps(result), flush=True)
