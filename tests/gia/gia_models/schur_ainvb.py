"""Helpers of `tests/unit/test_dtn_schur_ainvb.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_dtn_schur_ainvb.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

from gadopt.gia_presets import selfgrav_dtn_schur_solver_parameters
from gadopt.stokes_integrators import newton_stokes_solver_parameters


def ainvb_cache(pc):
    """The `Z`/`S` cache the option installs on a `DtNTwoBlockSchurPC`.

    Reached through the PETSc object the solve actually ran, never constructed
    by hand, so that a test checks the object PETSc built.

    Args:
      pc: the outer PETSc `PC` (python type `gadopt.DtNTwoBlockSchurPC`).

    Returns:
      The cache object. The design (section 7.1) names it `ainvb` on the
      python context and requires it to expose `n`, `S`, `Z`, `build_count`,
      `apply_count` and `apply_transpose_count`.
    """
    context = pc.getPythonContext()
    cache = getattr(context, "ainvb", None)
    assert cache is not None, (
        "the outer DtNTwoBlockSchurPC carries no `ainvb` cache, so the option "
        "dtn_schur_ainvb did not select the cached apply path")
    return cache


# ---------------------------------------------------------------------------
# The `direct` fixture: exact block 0, so the algebra can be pinned exactly
# ---------------------------------------------------------------------------
def direct_params(*, ainvb=True, **overrides):
    """The 2-D `direct` Schur preset, with block 1 left to the cached factors.

    Block 0 is `preonly` + `firedrake.AssembledPC` + MUMPS LU, so `M0` is an
    exact linear operator; block 1 is `preonly` + `none`, because under the
    option nothing solves it -- the cache applies the factors of the exact
    complement itself. The option is written as the raw PETSc key rather than
    through the preset, so that a change to the preset's signature cannot stop
    this fixture from building.

    Args:
      ainvb: whether to write the `dtn_schur_ainvb` key.
      overrides: extra solver-parameter entries, for the refusal tests.

    Returns:
      The solver-parameter dictionary.
    """
    params = dict(newton_stokes_solver_parameters)
    params.update(selfgrav_dtn_schur_solver_parameters)
    # The true-residual monitor is pure noise in a test log; drop it.
    params.pop("ksp_monitor_true_residual", None)
    # Block 1 is never solved on the cached path, and the design refuses any
    # other setting there (section 5.2), so the fixture writes what it asks
    # for and the refusal tests write what it refuses.
    params["dtn_fieldsplit_1_ksp_type"] = "preonly"
    params["dtn_fieldsplit_1_pc_type"] = "none"
    params.pop("dtn_fieldsplit_1_ksp_rtol", None)
    params.pop("dtn_fieldsplit_1_ksp_max_it", None)
    if ainvb:
        params["dtn_schur_ainvb"] = True
    params.update(overrides)
    return params
