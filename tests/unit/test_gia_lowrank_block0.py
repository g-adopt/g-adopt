r"""The low-rank DtN update inside block 0 of the self-gravity solver.

Stage 1 of `NOTES/HANDOVER-LOWRANK-2026-09-16.md`. What is under test is
`gadopt.LowRankPotentialPC` and the Python potential block that
`gadopt.CondensedBlockPC` builds on the low-rank representation, selected by
`selfgrav_dtn_iterative_solver_parameters(dtn_representation="lowrank")`.

## The defect these tests pin

`AugmentedImplicitMatrixContext` adds `B = theta_psi * sum_b C_b^T W_b C_b` to
the OUTER Jacobian's action only. Firedrake's `createSubMatrix` builds every
fieldsplit sub-block as a plain `ImplicitMatrixContext`, so block 0 never saw
`B`, and `CondensedBlockPC` assembles `A_psipsi` from the bilinear form, which
has no `B` in it either. Every solve converged and the outer FGMRES simply
cost 8 or 9 iterations where the multiplier representation cost 3
(`NOTES/FINDING-LOWRANK-ITERATIVE-2026-09-16.md` section 1). A cost with no
error message is what these tests exist to catch, so most of them assert on
iteration counts and not only on the answer.

## Why the answer is not enough on its own

Both representations solve the same system to the same outer tolerance, so a
test that compares states passes whether or not the update reached block 0.
Every test here that can assert a count does.

Most runs are on the 2-D annulus. `TestThreeDimensions` at the end is the
24-cell cubed sphere: it proves that the construction works on extruded
hexahedra with a fluid core and the rotational closure, and nothing about
speed. Every 3-D performance number comes from Gadi.

The solver is `gadopt.gia_solver.SelfGravitatingGIASolver`. `build` here is
also the annulus builder of `test_gia_nullspace_projection_adjoint.py`.
"""
import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from pyadjoint.tape import annotate_tape

from gadopt import CylindricalDtN, SphericalDtN
from gadopt.approximations import CompressibleInternalVariableApproximation
from gadopt.gia_couplings import FluidCore, Rotation, SurfaceLoad
from gadopt.gia_presets import (
    OMEGA_SQ_EARTH,
    selfgrav_dtn_iterative_solver_parameters,
)
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.internal_variable_equation import history_slices
from gadopt.preconditioners import (
    LowRankPotentialPC, _LowRankPotentialOperator)

# The annulus of the GIA suite (dr 0.2, 32 cells around) and its constants.
import test_gia_gravity  # noqa: E402
from test_gia_gravity import (
    B_MU,
    CURVE_INNER,
    CURVE_OUTER,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    SIGMA_HAT,
)
from parallel_ranks import assert_marked_rank_count
# The module fixture of `test_gia_gravity`, bound by assignment so that pytest finds
# it here and flake8 does not read the test arguments of the same name as
# redefinitions of an import.
meshes = test_gia_gravity.meshes


# The outer tolerance is tight so that the iteration counts compared below are
# counts of the same work and not of two different stopping points.
OUTER_RTOL = 1e-10
BLOCK0_RTOL = 1e-4

#: The reference gravity of the annulus, units of g_bar.
G0 = 1.0
#: The DtN truncation of the gradient tests.
TRUNCATION = 3
#: The central-difference step and the bar of the fresh-solve gradient check.
#: With an outer tolerance of 1e-11 the quotient's floor is about 1e-11
#: (round-off 1e-16/1e-5, truncation eps^2 ~ 1e-10), five orders under the
#: bar.
FD_EPS = 1e-5
FD_GATE = 1e-6
#: The low-rank term must carry at least this share of the residual norm
#: before a gradient through it is believed (`b_liveness`).
B_LIVENESS_FLOOR = 1e-3


@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


def preset(representation, **kwargs):
    """The two-block iterative preset of the production configuration."""
    options = dict(condensed=False, block0="condensed",
                   block0_rtol=BLOCK0_RTOL, outer_rtol=OUTER_RTOL,
                   block0_max_it=200, snes_type="ksponly",
                   multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
                   dtn_representation=representation)
    options.update(kwargs)
    return selfgrav_dtn_iterative_solver_parameters(**options)


def build(meshes, representation, *, truncation=3, fluid_core=True,
          rotation=True, solver_parameters=None, dt=1.0,
          approximation_kwargs=None, nullspace=None,
          initial_displacement=None):
    """One coupled solver on the annulus, in either representation.

    Mirrors `NOTES/measurements/lowrank-iterative/lowrank_block0.py`, which is
    where the measurements in the finding come from, so that a count asserted
    here is the count recorded there.

    The surface load is `SIGMA_HAT (cos 2 theta + m)` on Re, with the mean
    `m = 0.25` on a fluid core and 0 on a rigid one. With a fluid core the
    mean is the part of the load that the core pressure balances, so the
    core-pressure row carries a nonzero value. The load is one `SurfaceLoad`:
    the traction `B_mu g sigma` on the mechanics, the mass sheet in the
    potential row, the rotation row and the 2-D monopole datum.

    Args:
      meshes: the `(parent, sub)` pair.
      representation: `"multiplier"` or `"lowrank"`.
      truncation: the DtN truncation of both boundaries.
      fluid_core: a `FluidCore` of density 2 at the CMB, or a rigid core
        (`un = 0`).
      rotation: the rotational closure with the polar moment of the mantle.
      solver_parameters: a complete dictionary or a preset name; `None` is
        `preset(representation)`.
      dt: the time step, held in a `Constant`.
      approximation_kwargs: approximation settings on top of the defaults;
        a control (`shear_modulus`, `viscosity`) goes in here.
      nullspace: the `nullspace` argument of the solver.
      initial_displacement: UFL interpolated into the displacement before
        construction, so it is the initial state of the first solve.

    Returns:
      `(solver, u, psi)`.
    """
    parent, sub = meshes
    Xm = fd.SpatialCoordinate(sub)
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    if initial_displacement is not None:
        u.interpolate(initial_displacement)
    gravity_bcs = {
        CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    mean = fd.Constant(0.25) if fluid_core else fd.Constant(0.0)
    couplings = [SurfaceLoad(
        boundary=CURVE_RE, g_surface=G0,
        sigma=SIGMA_HAT * (fd.cos(2 * fd.atan2(Xm[1], Xm[0])) + mean))]
    mechanics_bcs = {}
    if fluid_core:
        couplings.append(FluidCore(boundary=CURVE_RC, rho_core=2.0))
    else:
        mechanics_bcs[CURVE_RC] = {"un": 0.0}
    if rotation:
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        couplings.append(Rotation(C=fd.assemble(fd.dot(Xm, Xm) * dx_m),
                                  Omega_sq=OMEGA_SQ_EARTH))

    settings = dict(
        bulk_modulus=1.0, density=fd.Constant(1.0),
        shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
        g=fd.Constant(G0), B_mu=fd.Constant(B_MU),
        self_gravity_number=fd.Constant(LAMBDA))
    settings.update(approximation_kwargs or {})
    approx = CompressibleInternalVariableApproximation(**settings)
    if solver_parameters is None:
        solver_parameters = preset(representation)
    solver = SelfGravitatingGIASolver(
        u, psi, approx, dt=fd.Constant(dt), gravity_bcs=gravity_bcs,
        bcs=mechanics_bcs, couplings=couplings,
        dtn_representation=representation, nullspace=nullspace,
        solver_parameters=solver_parameters)
    return solver, u, psi


def block0_context(solver):
    """The `CondensedBlockPC` instance block 0 runs, after a solve."""
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    block0, _ = outer.pc.getFieldSplitSchurGetSubKSP()
    return block0.getPC().getPythonContext()


def potential_pc(condensed_block):
    """The preconditioner of the potential split of the block-0 nest."""
    split = condensed_block.condensed_ksp.getPC().getFieldSplitSubKSP()[1]
    return split.getPC().getPythonContext()


def outer_iterations(solver):
    return solver.solver.snes.ksp.getIterationNumber()


# ---------------------------------------------------------------------------
# 1. The identity, on its own terms
# ---------------------------------------------------------------------------
def test_the_capacitance_form_survives_zero_weights():
    """`Cap = I + theta W (U^T Z)` reproduces the dense inverse exactly.

    The form of the Woodbury identity in every textbook carries `W^-1/theta`
    in the capacitance, and it does not exist for this problem: a weight is
    `(lam_k - alpha/R) / (scale_k * A_h)`, and `gadopt.dtn_form` builds
    `lam = (l+1)/R` on an exterior boundary and `l/R` on an interior one with
    `alpha = 1`, so the weight is EXACTLY zero for the exterior `l = 0` mode
    and the interior `l = 1` modes. The class carries `theta W` instead.

    This is the algebra alone, on a dense random system with two exact zeros
    and two negatives in `W`, so that a failure here is a failure of the
    identity and not of any solver around it.
    """
    rng = np.random.default_rng(0)
    n, k = 40, 6
    M = rng.normal(size=(n, n))
    A = M @ M.T + n * np.eye(n)
    U = rng.normal(size=(n, k))
    w = np.array([0.0, -0.7, 1.3, 0.0, 2.1, -0.2])
    theta = 3.7

    exact = np.linalg.inv(A + theta * U @ np.diag(w) @ U.T)
    A_inv = np.linalg.inv(A)
    Z = A_inv @ U
    cap = np.eye(k) + (theta * w)[:, np.newaxis] * (U.T @ Z)
    applied = A_inv - Z @ np.linalg.solve(
        cap, (theta * w)[:, np.newaxis] * (U.T @ A_inv))

    assert np.abs(exact - applied).max() / np.abs(exact).max() < 1e-13


def test_zero_weights_are_actually_present(meshes):
    """The zero weights above are this configuration's, not a hypothetical.

    If this ever returns zero, the test above is guarding an unreachable case
    and the simpler `W^-1` form would do. It returns four.
    """
    solver, _, _ = build(meshes, "lowrank")
    weights = np.concatenate(
        [rows.weights for rows in solver.dtn_operator.mode_rows])
    assert int((weights == 0.0).sum()) == 4


# ---------------------------------------------------------------------------
# 2. Block 0 is no longer blind to `B`
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truncation", [3, 6, 10])
def test_the_outer_count_matches_the_multiplier_path(meshes, truncation):
    """The whole point: the outer FGMRES stops paying for the update.

    Before this change the low-rank arm cost 8, 8 and 9 outer iterations at
    these three truncations against the multiplier path's 3
    (`NOTES/FINDING-LOWRANK-ITERATIVE-2026-09-16.md` section 1). The bar is
    the multiplier path's own count plus one, measured in the same test rather
    than written down here, because the count is a property of the mesh and
    the tolerances and would otherwise drift into a number nobody can justify.
    """
    reference, *fields_ref = build(meshes, "multiplier", truncation=truncation)
    reference.solve()
    multiplier_its = outer_iterations(reference)

    solver, *fields = build(meshes, "lowrank", truncation=truncation)
    solver.solve()

    assert outer_iterations(solver) <= multiplier_its + 1

    # The same state, to the outer tolerance. This is the check that the
    # iteration count above was bought with the right operator and not by
    # preconditioning a different problem.
    for field, field_ref in zip(fields, fields_ref):
        reference_norm = fd.norm(field_ref)
        assert abs(fd.norm(field) - reference_norm) < 1e-8 * reference_norm


def test_the_update_is_applied_inside_block_zero(meshes):
    """The potential block applies the update, many times per solve.

    A state comparison cannot see the difference between an update that is
    inside block 0 and one that is only in the outer operator, because both
    converge to the same answer. This counts the applications of the Python
    potential block, which exist only on the route under test.
    """
    solver, _, _ = build(meshes, "lowrank")
    solver.solve()
    condensed = block0_context(solver)
    assert isinstance(condensed.potential_operator, _LowRankPotentialOperator)
    assert condensed.potential_operator.applications > 0
    assert isinstance(potential_pc(condensed), LowRankPotentialPC)


def test_the_correction_inverts_the_potential_block(meshes):
    """One application of the preconditioner is `(A + theta U W U^T)^-1`.

    The outer and block-0 counts above cannot see the Woodbury correction:
    with `apply` reduced to the plain multigrid solve they stay at 3 and 11,
    because the update is in the OPERATOR of the potential split and the
    inner Krylov method absorbs the missing correction (measured, inner count
    15 against 19). This test isolates the correction. The inner solve is made
    exact for the duration of one application, so the only approximation
    left is the correction itself, and the relative residual against the
    Python block must be at the inner tolerance. Measured 1.2e-10; with any
    part of the correction disconnected, 1.5e-1.
    """
    solver, _, _ = build(meshes, "lowrank")
    solver.solve()
    condensed = block0_context(solver)
    pc = potential_pc(condensed)
    block = condensed.condensed_operator.getNestSubMatrix(1, 1)
    x, y, r = block.createVecRight(), block.createVecRight(), block.createVecLeft()
    x.setRandom()
    x.assemble()
    ksp = pc.ksp
    saved_type = ksp.getType()
    saved_rtol, saved_atol, saved_dtol, saved_max = ksp.getTolerances()
    ksp.setType("cg")
    ksp.setTolerances(rtol=1e-13, max_it=1000)
    try:
        pc.apply(None, x, y)
    finally:
        ksp.setType(saved_type)
        ksp.setTolerances(rtol=saved_rtol, atol=saved_atol, divtol=saved_dtol,
                          max_it=saved_max)
    block.mult(y, r)
    r.axpy(-1.0, x)
    assert r.norm() / x.norm() < 1e-8


def test_the_multiplier_path_keeps_plain_multigrid(meshes):
    """No `dtn_operator` in the context leaves the potential split alone.

    This is the guard on the production path. The key is published only by
    `build_dtn_operator`, which runs only on the low-rank representation, and
    its absence must leave the nest and the split byte for byte as they were.
    """
    solver, _, _ = build(meshes, "multiplier")
    solver.solve()
    condensed = block0_context(solver)
    assert condensed.dtn_operator is None
    assert condensed.potential_operator is None
    split = condensed.condensed_ksp.getPC().getFieldSplitSubKSP()[1]
    assert split.getPC().getType() == "gamg"


# ---------------------------------------------------------------------------
# 3. Reuse: one build for the life of a run
# ---------------------------------------------------------------------------
def test_the_columns_are_built_once_across_steps_and_a_dt_change(meshes):
    """`Z`, the capacitance and the hierarchy survive every step.

    The potential block is `theta * (grad psi . grad v dx + Robin)` and the
    update is `theta * B0`, so neither depends on `dt`, on the stress or on the
    rheology. Three solves at one `dt` and one more after `dt.assign` must
    leave the build counter at 1. The `dt` change is the half that matters:
    it moves `operator_version`, so `CondensedBlockPC` reassembles all four of
    its blocks, and a preconditioner that keyed its reuse on that version
    instead of on the values would rebuild here for nothing.
    """
    solver, _, _ = build(meshes, "lowrank", dt=1.0)
    for _ in range(3):
        solver.solve()
    condensed = block0_context(solver)
    assert potential_pc(condensed).column_builds == 1

    solver.dt.assign(2.0)
    solver.solve()
    assert condensed.assembly_count > 1, (
        "the test is vacuous unless the dt change did reassemble the blocks")
    assert potential_pc(condensed).column_builds == 1


def test_the_columns_survive_newton_on_a_power_law(meshes):
    """Under Newton the potential split is built once; the blocks per linear solve.

    With `exponent = 3` the Jacobian depends on the state, the solver
    publishes `operator_version = None`, and `CondensedBlockPC` reassembles
    its blocks at every linear solve, that is at every Newton iteration
    (measured: `assembly_count` 1 to 3 inside one three-iteration solve). The
    potential block `A_psipsi + B` does not depend on the rheology or on the
    state, so the Woodbury columns must survive every Newton iteration and
    every solve: the build counter stays at 1 while the assembly counter moves.
    """
    approx_kwargs = {"exponent": 3.0, "transition_stress": 1e-3}
    # `always` reassembles the blocks at every Newton iteration, the most
    # rebuilds any setting of the shared rule makes, which is the case this
    # test needs: the columns must survive every one of them.
    parameters = preset("lowrank", snes_type="newtonls")
    parameters["gia_rebuild"] = "always"
    solver, _, _ = build(
        meshes, "lowrank", solver_parameters=parameters,
        approximation_kwargs=approx_kwargs)
    solver.solve()
    assert solver.solver.snes.getIterationNumber() >= 2, (
        "the test is vacuous unless Newton took more than one iteration")
    condensed = block0_context(solver)
    assert potential_pc(condensed).column_builds == 1
    first = condensed.assembly_count
    solver.solve()
    assert condensed.assembly_count > first
    assert potential_pc(condensed).column_builds == 1


@pytest.mark.parallel(nprocs=2)
def test_two_ranks_match_the_multiplier_path(request, meshes):
    """Item 6 of the stage-1 test list: the `Allreduce` paths of the new PC.

    The boundary dofs are partitioned and the modes are global, so every
    application of `LowRankPotentialPC` reduces a `k`-vector across ranks,
    and the column build reduces `U^T Z`. A rank with no boundary dofs must
    contribute a zero and not skip the reduction. Run on two ranks, the
    low-rank arm must reproduce the multiplier arm's state to the outer
    tolerance and its outer count within one, as the serial tests assert.
    """
    assert_marked_rank_count(request)
    comm = fd.COMM_WORLD
    assert comm.size == 2, "this test must run on exactly two ranks"
    reference, *fields_ref = build(meshes, "multiplier")
    reference.solve()
    solver, *fields = build(meshes, "lowrank")
    solver.solve()
    assert solver.solver.snes.ksp.getConvergedReason() > 0
    assert outer_iterations(solver) <= outer_iterations(reference) + 1
    condensed = block0_context(solver)
    assert isinstance(potential_pc(condensed), LowRankPotentialPC)
    assert potential_pc(condensed).column_builds == 1
    for field, field_ref in zip(fields, fields_ref):
        reference_norm = fd.norm(field_ref)
        assert abs(fd.norm(field) - reference_norm) < 1e-8 * reference_norm


# ---------------------------------------------------------------------------
# 3b. The stopping rule of the outer FGMRES
# ---------------------------------------------------------------------------
def test_the_outer_stopping_options_reach_the_dictionary():
    """`outer_atol`, `outer_max_it` and `outer_accept_capped` set the outer KSP.

    The defaults leave the dictionary as callers had it before: no
    `ksp_atol` (PETSc's own, effectively off), a cap of 200 and no
    `ksp_converged_maxits`.

    CATCHES: an option that never reaches the outer KSP, or a default that
    changes the dictionary of every existing caller.
    """
    default = preset("lowrank")
    assert "ksp_atol" not in default
    assert "ksp_converged_maxits" not in default
    assert default["ksp_max_it"] == 200
    chosen = preset("lowrank", outer_atol=1e-11, outer_max_it=30,
                    outer_accept_capped=True)
    assert chosen["ksp_atol"] == 1e-11
    assert chosen["ksp_max_it"] == 30
    assert "ksp_converged_maxits" in chosen


@pytest.mark.parametrize("kwargs", [{"outer_atol": -1.0},
                                    {"outer_max_it": 0}])
def test_the_preset_refuses_a_wrong_outer_stopping_option(kwargs):
    with pytest.raises(ValueError, match="outer_atol must be >= 0"):
        preset("lowrank", **kwargs)


def test_a_capped_outer_solve_lets_newton_finish(meshes):
    """With `outer_accept_capped`, Newton converges through capped outer solves.

    An outer cap of 2 iterations stops every linear solve short of
    `outer_rtol`. Without the option SNES ends with `DIVERGED_LINEAR_SOLVE`;
    with it the capped solve counts as converged, Newton takes its update
    and repeats until `snes_rtol`, and the state agrees with the one of the
    uncapped preset. This is what lets a late Exp01 step, whose relative
    target lies below the residual floor, finish instead of aborting.

    FLOOR: the state agrees to 1e-3 relative, because Newton stops at
    `snes_rtol = 1e-4` on the residual and not at the outer tolerance.
    """
    reference, *fields_ref = build(
        meshes, "lowrank",
        solver_parameters=preset("lowrank", snes_type="newtonls"))
    reference.solve()

    capped = preset("lowrank", snes_type="newtonls", outer_max_it=2)
    solver, _, _ = build(meshes, "lowrank", solver_parameters=capped)
    with pytest.raises(fd.ConvergenceError, match="DIVERGED_LINEAR_SOLVE"):
        solver.solve()

    accepted = preset("lowrank", snes_type="newtonls", outer_max_it=2,
                      outer_accept_capped=True)
    solver, *fields = build(meshes, "lowrank", solver_parameters=accepted)
    solver.solve()
    snes = solver.solver.snes
    assert snes.getConvergedReason() > 0
    # Vacuity guard: the cap really bound, so the option did something.
    assert snes.getLinearSolveIterations() >= 2 * snes.getIterationNumber()
    for field, field_ref in zip(fields, fields_ref):
        reference_norm = fd.norm(field_ref)
        difference = fd.norm(field - field_ref)
        assert difference < 1e-3 * reference_norm


# ---------------------------------------------------------------------------
# 4. What the configuration refuses
# ---------------------------------------------------------------------------
def test_the_preset_refuses_the_nested_block_zero_on_the_low_rank_path():
    """`block0="pair"` has nowhere to put the update."""
    with pytest.raises(ValueError, match="no low-rank route"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, block0="pair", dtn_representation="lowrank")


def test_the_preset_supports_the_eliminated_layout_on_the_low_rank_path():
    """The retained u/psi nest carries the low-rank potential correction."""
    options = selfgrav_dtn_iterative_solver_parameters(condensed=True, dtn_representation="lowrank")
    assert options["dtn_fieldsplit_0_pc_python_type"] == "gadopt.ReducedGIABlockPC"
    assert options["dtn_fieldsplit_0_condensed_fieldsplit_1_pc_python_type"] == "gadopt.LowRankPotentialPC"


def test_the_preset_refuses_an_unknown_representation():
    with pytest.raises(ValueError, match="must be 'multiplier' or 'lowrank'"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, dtn_representation="low_rank")


def test_a_low_rank_preset_on_a_multiplier_solver_is_refused(meshes):
    """This direction raises late and unhelpfully, so it is caught early.

    Without the check the failure is inside `PCSetUp`, reported as an
    unhandled Python exception under a PETSc error code.
    """
    with pytest.raises(ValueError,
                       match="LowRankPotentialPC needs dtn_representation='lowrank'"):
        build(meshes, "multiplier", solver_parameters=preset("lowrank"))


def test_a_multiplier_preset_on_a_low_rank_solver_is_refused(meshes):
    """This direction also raises late: plain GAMG refuses the Python block.

    `CondensedBlockPC.initialize` installs the Python potential block whenever
    `dtn_operator` is in the context, and `PCSetUp_GAMG` then fails with `No
    method getinfo for Mat of type python`, an unhandled exception under a
    PETSc error code. The early check gives a message that names the fix.
    """
    with pytest.raises(ValueError,
                       match="low-rank solver needs gadopt.LowRankPotentialPC"):
        build(meshes, "lowrank", solver_parameters=preset("multiplier"))


def test_the_nested_block_zero_on_a_low_rank_solver_is_refused(meshes):
    """The combination that converges quietly at three times the cost.

    `block0="pair"` is legal on a multiplier preset, and a low-rank solver
    given that dictionary solves block 0 through the outer operator's block,
    which carries the update, with no potential split to precondition it on.
    Every solve converges; the outer FGMRES costs 8 outer iterations against
    3 (measured, annulus, truncation 3), and nothing in any log says so. The
    representation check cannot see it, because the dictionary names no
    potential split to be wrong about, so the block-0 route is refused on its
    own.
    """
    nested = selfgrav_dtn_iterative_solver_parameters(
        condensed=False, block0="pair", block0_rtol=BLOCK0_RTOL,
        outer_rtol=OUTER_RTOL, block0_max_it=200, snes_type="ksponly",
        multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
        dtn_representation="multiplier")
    with pytest.raises(ValueError, match="block0='pair'"):
        build(meshes, "lowrank", solver_parameters=nested)


@pytest.mark.parametrize("kind", ["assembled", "wrong_context"])
def test_the_class_refuses_a_preconditioning_matrix_it_cannot_read(kind):
    """Selected anywhere but on the Python potential block, it says so.

    Two ways to get there: on the multiplier path the potential block is an
    assembled matrix, and outside `CondensedBlockPC` altogether it may be a
    Python matrix carrying something else. The assembled case is why the class
    tests the matrix type before asking for a context: `getPythonContext` on an
    assembled matrix raises inside PETSc, and this runs in a PETSc callback,
    where that surfaces as an error code far from its cause.
    """
    from firedrake.petsc import PETSc

    if kind == "assembled":
        matrix = PETSc.Mat().createAIJ([2, 2], comm=fd.COMM_WORLD)
        matrix.setUp()
        matrix.assemble()
    else:
        matrix = PETSc.Mat().createPython([2, 2], object(),
                                          comm=fd.COMM_WORLD)
        matrix.setUp()
    pc = PETSc.PC().create(comm=fd.COMM_WORLD)
    pc.setOperators(matrix, matrix)
    with pytest.raises(ValueError, match="_LowRankPotentialOperator"):
        LowRankPotentialPC().initialize(pc)


def test_the_transpose_application_is_refused(meshes):
    """An adjoint solve reaches the forward `apply`; a transpose is a new path.

    The applied map is also genuinely non-symmetric, because the leading term
    comes from one multigrid V-cycle and the columns from an accurate solve.
    """
    solver, _, _ = build(meshes, "lowrank")
    solver.solve()
    instance = potential_pc(block0_context(solver))
    with pytest.raises(NotImplementedError, match="transpose application"):
        instance.applyTranspose(None, None, None)


# ---------------------------------------------------------------------------
# 5. Stage 2: the adjoint runs through this preset, on both representations
# ---------------------------------------------------------------------------
# Two separate defects kept the adjoint off the iterative preset, one per
# representation, and neither was about the low-rank update itself.
#
# On the MULTIPLIER path `CondensedBlockPC.initialize` failed in the Slate
# compiler with `too many values to unpack (expected 1)`. The `(u, M)`
# sub-block of an adjoint operator keeps one cell integral on the PARENT mesh
# that the forward sub-block does not have, and Slate compiles one mesh at a
# time. That integral is identically zero — it is the potential Laplacian with
# the potential slot already zeroed, which `ExtractSubBlock` leaves
# unsimplified and `expand_derivatives` reduces to nothing —
# so `_restrict_to_mesh` drops it and no number changes.
#
# On the LOW-RANK path the solve never reached a preconditioner at all:
# `dmhooks.create_subdm` raised `Action right argument must be either
# Coefficient or BaseForm`, because the adjoint solved a PRE-ASSEMBLED operator
# and the boundary-lifting term of such a problem is an `Action` that UFL
# cannot split. The adjoint and tangent now run as variational solves on the
# adjoint and tangent FORMS, through Firedrake's own adjoint solver for the
# block, which `LowRankVariationalSolver.solve` builds with both augmentation
# callbacks, the forward application context and `snes_type: ksponly`.
#
# Both are checked here by running the gradient, because both failures were
# raised errors and a test that only builds the solver would not see them.
def shear_modulus_forward(meshes, representation, control, solver_parameters):
    """One solve with `shear_modulus = control`; returns `(J, solver)`.

    The fluid-core annulus without rotation, the configuration of the
    low-rank adjoint suite. `J = ||u||^2` over the mantle reads the solution
    through the UFL component of the mixed solution, a reference the tape
    owns. The solver is built without annotation; the caller decides whether
    the solve and the functional are taped.
    """
    _, sub = meshes
    with stop_annotating():
        solver, _, _ = build(
            meshes, representation, truncation=TRUNCATION, fluid_core=True,
            rotation=False, solver_parameters=solver_parameters,
            approximation_kwargs={"shear_modulus": control})
    solver.solve()
    u = solver.field("displacement")
    return fd.assemble(fd.inner(u, u) * fd.dx(domain=sub)), solver


def b_liveness(solver, representation):
    """Assert that the DtN coupling carries a share of the residual.

    On the low-rank arm, `||theta B z|| / ||F(z)||` at the converged state
    must exceed `B_LIVENESS_FLOOR`; on the multiplier arm the norm of the DtN
    multipliers takes the place of `theta B z`. A gradient test on an inert
    coupling passes while measuring nothing.
    """
    residual = fd.assemble(solver.F)
    with residual.dat.vec_ro as rv:
        fnorm = rv.norm()
    assert fnorm > 0.0, "residual is identically zero: nothing to measure"
    z = solver.solution
    if representation == "multiplier":
        num = float(np.sqrt(sum(
            float(np.sum(z.subfunctions[i].dat.data_ro ** 2))
            for i in solver.multiplier_indices)))
    else:
        # `mult(x, y, theta=None)` gives `theta B0 z` on the monolithic mixed
        # vector, the vector `F` is assembled on.
        Bz = fd.Function(z.function_space())
        with z.dat.vec_ro as xv, Bz.dat.vec_rw as yv:
            solver.dtn_operator.mult(xv, yv, theta=None)
            num = yv.norm()
    assert num / fnorm > B_LIVENESS_FLOOR, (
        f"the DtN coupling is inert: ratio {num / fnorm:.3e}")


def shear_modulus_reduced_functional(meshes, representation, solver_parameters):
    """Tape one solve with a `Real` shear-modulus control of value 1.

    Returns:
      `(Jhat, control, direction, J, solver)`; the direction is 0.1.
    """
    _, sub = meshes
    R = fd.FunctionSpace(sub, "R", 0)
    with stop_annotating():
        control = fd.Function(R).assign(1.0)
        direction = fd.Function(R).assign(0.1)
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        m = Control(control)
        J, solver = shear_modulus_forward(meshes, representation, control,
                                          solver_parameters)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    b_liveness(solver, representation)
    return Jhat, control, direction, J, solver


@pytest.mark.parametrize("representation", ["multiplier", "lowrank"])
def test_the_gradient_is_right_through_this_preset(meshes, representation):
    """Gradient and Taylor rate, with the iterative preset in place of LU.

    The bars are the low-rank suite's own: the adjoint gradient within 1e-6
    relative of a central difference of fresh solves, and a Taylor rate of
    at least 1.99. The fresh solves rebuild the solver at each control, so
    a channel the tape misses shows here; a Taylor test replays the tape and
    cannot see it.
    """
    from pyadjoint import taylor_test

    _, sub = meshes
    iterative = preset(representation, outer_rtol=1e-11)
    Jhat, control, direction, _, taped = shear_modulus_reduced_functional(
        meshes, representation, iterative)
    gradient = float(Jhat.derivative().dat.data_ro[0])
    get_working_tape().clear_tape()

    def at(value):
        with stop_annotating():
            m = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(value)
            J, fresh = shear_modulus_forward(meshes, representation, m, iterative)
            assert fresh is not taped
            return float(J)

    difference = (at(1.0 + FD_EPS) - at(1.0 - FD_EPS)) / (2 * FD_EPS)
    relative = abs(gradient - difference) / abs(difference)
    print(f"    [fresh-FD/{representation}] adjoint {gradient:.10e} "
          f"fresh {difference:.10e} rel {relative:.3e}")
    assert relative < FD_GATE

    Jhat, control, direction, _, _ = shear_modulus_reduced_functional(
        meshes, representation, iterative)
    assert taylor_test(Jhat, control, direction) > 1.99


# ---------------------------------------------------------------------------
# 6. The production geometry, on 24 cells
# ---------------------------------------------------------------------------
class TestThreeDimensions:
    """The 24-cell extruded cubed sphere, low-rank against multiplier.

    Item 7 of the stage-1 test list in `NOTES/HANDOVER-LOWRANK-2026-09-16.md`
    section 3.2. The sphere has a fluid core and the rotational closure, so the
    Real block carries three rotation rows and the core pressure on both
    representations, and the multiplier representation carries its DtN
    multipliers on top. The two solvers share one mesh, because `fd.norm` on a
    difference of functions from two mesh objects refuses with "Multiple
    domains found".

    The pattern is `test_gia_condensed_block0.TestThreeDimensions`, with the
    reference being the multiplier representation under the same iterative
    preset instead of the direct route: the claim under test is that the
    low-rank arm reproduces the multiplier arm's state and its outer count in
    3-D, which is what the annulus tests above establish in 2-D.
    """

    @staticmethod
    def solver(mesh, C, representation):
        """A Newtonian fluid-core solver on the sphere, under the iterative preset.

        The CMB of a radially extruded sphere is its bottom surface, so the
        fluid core is on `"bottom"`, and there is no `un` there because a
        fluid core and `un` on one boundary are refused. The surface load is
        a traction with no mass sheet (`normal_stress` in `bcs`), which the
        solver accepts with a warning.

        Args:
          mesh: the extruded cubed sphere, both the mechanics and the
            potential mesh.
          C: the polar moment.
          representation: `"multiplier"` or `"lowrank"`.

        Returns:
          `(solver, u, psi)`.
        """
        X = fd.SpatialCoordinate(mesh)
        u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
        approx = CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=fd.Constant(1.0),
            shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
            g=fd.Constant(G0), B_mu=fd.Constant(B_MU),
            self_gravity_number=fd.Constant(LAMBDA))
        solver = SelfGravitatingGIASolver(
            u, psi, approx, dt=fd.Constant(1.0),
            gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                         "bottom": {"dtn": SphericalDtN(1)}},
            bcs={"top": {"normal_stress": SIGMA_HAT * X[2]}},
            couplings=[FluidCore(boundary="bottom", rho_core=2.0),
                       Rotation(C=C, C_minus_A=0.1 * C, Omega_sq=OMEGA_SQ_EARTH)],
            solver_parameters=preset(representation),
            dtn_representation=representation)
        return solver, u, psi

    @pytest.fixture(scope="class")
    def solved(self):
        """Both representations solved once on one 24-cell sphere."""
        base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        out = {}
        for representation in ("multiplier", "lowrank"):
            with pytest.warns(UserWarning, match="normal_stress in bcs"):
                solver, u, psi = self.solver(mesh, C, representation)
            solver.solve()
            out[representation] = (solver, u, psi)
        return out

    def test_it_matches_the_multiplier_representation(self, solved):
        """The same state on the sphere, to the outer tolerance.

        The annulus agreement between representations is 1e-12 relative
        (`NOTES/FINDING-LOWRANK-ITERATIVE-2026-09-16.md` section 6); the bar
        of 1e-8 is the one the multiplier 3-D test uses against its direct
        route, three to four orders above what is measured.
        """
        reference, *fields_ref = solved["multiplier"]
        solver, *fields = solved["lowrank"]
        for s in (reference, solver):
            assert s.solver.snes.getConvergedReason() > 0
            assert s.solver.snes.ksp.getConvergedReason() > 0
        for field, field_ref in zip(fields, fields_ref):
            ref = fd.norm(field_ref)
            assert ref > 0.0
            assert fd.norm(field - field_ref) < 1e-8 * ref

        def history(s):
            # The solved DG history unknown of the mixed solution.
            return s.solution.subfunctions[s.field_index["history"]]

        m_ref, = history_slices(history(reference))
        m, = history_slices(history(solver))
        assert fd.norm(m - m_ref) < 1e-8 * fd.norm(m_ref)

    def test_the_outer_count_matches_the_multiplier_representation(self, solved):
        """Outer FGMRES within one of the multiplier arm's, as on the annulus."""
        reference = solved["multiplier"][0]
        solver = solved["lowrank"][0]
        assert outer_iterations(solver) <= outer_iterations(reference) + 1

    def test_the_update_is_applied_inside_block_zero(self, solved):
        """The Python potential block and its preconditioner ran, in 3-D.

        The state and count tests cannot tell an update inside block 0 from
        one on the outer operator alone; the application counter can, and the
        column-build counter says the Woodbury data was built once for the
        solve on the extruded trace, which is the `supports_trace_build` case
        the handover asked to be checked.
        """
        condensed = block0_context(solved["lowrank"][0])
        assert isinstance(condensed.potential_operator, _LowRankPotentialOperator)
        assert condensed.potential_operator.applications > 0
        pc = potential_pc(condensed)
        assert isinstance(pc, LowRankPotentialPC)
        assert pc.column_builds == 1

    def test_the_multiplier_arm_keeps_plain_multigrid(self, solved):
        """The guard on the production path, on the sphere as on the annulus."""
        condensed = block0_context(solved["multiplier"][0])
        assert condensed.dtn_operator is None
        assert condensed.potential_operator is None
        split = condensed.condensed_ksp.getPC().getFieldSplitSubKSP()[1]
        assert split.getPC().getType() == "gamg"


def test_the_adjoint_solve_carries_both_augmentations(meshes):
    """The Jacobian and the residual of the adjoint solve agree, under Newton.

    The rule from `augment_jacobian`'s docstring, both augmentations or
    neither, pinned on the derivative solve where it was broken once: an
    adjoint solver with the Jacobian callback alone under `newtonls` takes one
    correct step and then walks to the root of a residual with no `B` in it
    (Taylor rate 1.08, gradient 1.9e-2 out, direct preset). The structural
    half checks what the solver was built with; the behavioural half forces
    Newton for one adjoint solve and asserts that it stops after one
    iteration, which it can only do if the residual at the Krylov solution is
    zero, that is if both operators carry `B`. The type is asserted after the
    solve so that a solver that quietly reverted to `ksponly` cannot pass.
    """
    functional, _, _, _, solver = shear_modulus_reduced_functional(
        meshes, "lowrank", "direct")
    # The adjoint solver of the adopted block: the stock block's
    # `_ad_solvers` entry, or the cached adjoint solver with Firedrake PR 4638.
    from gadopt.dtn_coupled_adjoint import adjoint_solver as find_adjoint_solver
    adjoint_solver = find_adjoint_solver(solver.adjoint_block)
    context = adjoint_solver._ctx
    assert context._post_function_callback == solver.augment_residual
    assert context._post_jacobian_callback == solver.augment_jacobian
    assert "dtn_operator" in context.appctx
    assert adjoint_solver.snes.getType() == "ksponly"

    adjoint_solver.snes.setType("newtonls")
    try:
        functional.derivative()
        assert adjoint_solver.snes.getType() == "newtonls"
        assert adjoint_solver.snes.getIterationNumber() == 1
    finally:
        adjoint_solver.snes.setType("ksponly")


def test_install_refuses_a_jacobian_that_is_not_matrix_free():
    """A non-Python Jacobian is refused with a message, not a crash.

    `install_augmented_context` swaps the Python context of a matrix-free
    Jacobian. On any other matrix type `getPythonContext` reads invalid
    memory, and the process dies. The pyop3 Firedrake produces such a matrix
    (a nest matrix in place of the requested matrix-free one, for every mixed
    form with a Real subspace), so the guard must turn it into an error that
    names the cause.
    """
    from firedrake.petsc import PETSc
    from gadopt.dtn_coupled import install_augmented_context

    # A small assembled matrix stands for the wrong Jacobian. The DtN operator
    # is never reached, so None is enough.
    matrix = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
    matrix.setUp()
    matrix.assemble()
    try:
        with pytest.raises(RuntimeError, match="matrix-free Jacobian"):
            install_augmented_context(matrix, None)
    finally:
        matrix.destroy()
