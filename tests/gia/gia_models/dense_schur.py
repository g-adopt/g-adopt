"""Helpers of `tests/unit/test_dtn_multiplier_dense_schur.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_dtn_multiplier_dense_schur.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import numpy as np
from gadopt.gia_presets import selfgrav_dtn_schur_solver_parameters
from gadopt.stokes_integrators import newton_stokes_solver_parameters


# ---------------------------------------------------------------------------
# Geometry and constants: the coarse 2-D annulus of the adjoint suite
# ---------------------------------------------------------------------------
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3
G0 = 1.0
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5
TRUNCATION = 3


def _dense_schur_params():
    """The 2-D `direct` Schur preset, with block 1 on the dense-Schur PC.

    Block 0 is `preonly` + assembled LU, so its inverse is exact and the Schur
    complement the PC builds is exact and linear -- which is what lets the
    `schur` entry of the `operators` case assert parity to roundoff rather
    than to a preconditioner tolerance.
    """
    params = dict(newton_stokes_solver_parameters)
    params.update(selfgrav_dtn_schur_solver_parameters)
    params["dtn_fieldsplit_1_pc_type"] = "python"
    params["dtn_fieldsplit_1_pc_python_type"] = (
        "gadopt.DtNMultiplierDenseSchurPC")
    # The true-residual monitor is pure noise in a test log; drop it.
    params.pop("ksp_monitor_true_residual", None)
    return params


def _dense_ctx(solver):
    """The `DtNMultiplierDenseSchurPC` instance PETSc built inside the solve.

    Outer python PC -> its inner Schur fieldsplit -> the block-1 sub-KSP's PC ->
    its python context. Reaching it this way (rather than constructing one) is
    what makes the test check the object the real solve used.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    _, ksp_real = outer.pc.getFieldSplitSchurGetSubKSP()
    return ksp_real, ksp_real.getPC().getPythonContext()


def _apply_pc(petsc_pc, ctx, rhs, transpose=False):
    """Run the PC's `apply`/`applyTranspose` through real PETSc vectors."""
    A = petsc_pc.getOperators()[0]
    x, y = A.createVecRight(), A.createVecLeft()
    lo, hi = x.owner_range
    x.array_w[:] = np.asarray(rhs)[lo:hi]
    x.assemble()
    if transpose:
        ctx.applyTranspose(petsc_pc, x, y)
    else:
        ctx.apply(petsc_pc, x, y)
    out = ctx._gather(petsc_pc.comm.tompi4py(), y, ctx._n)
    x.destroy()
    y.destroy()
    return out
