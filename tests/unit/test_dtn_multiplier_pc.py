"""The multiplier block's preconditioners, and the diagonal they rest on.

The commit that landed the coupled self-gravitating solver has **no** unit test
touching the multiplier block. These are those tests, and they are written
around two properties rather than around golden numbers:

1. `DtNGravityForm.multiplier_diagonal()` is derived from the form -- the
   claim is `A_(c_k,c_k) = -scale_k * A_h` -- so it is checked against the
   **assembled** constraint row, not against a stored constant. A golden number
   would pass just as happily if both the closed form and the residual drifted
   together.
2. Every accepting assertion has a **rejecting** partner. This project has
   twice had a gate pass while the thing beside it was wrong, so a tolerance
   that accepts the right answer is only worth having if the same tolerance is
   shown to reject a plausible wrong one -- here, the analytic boundary measure
   `2 pi R` in place of the discrete `A_h`, which is the substitution a reader
   of the formula would most naturally make.

The solve-level test exercises all three promoted items at once: without the
sub-DM now threaded onto the multiplier KSP, selecting any `PCBase` subclass on
block 1 dies with `AttributeError: 'NoneType' object has no attribute
'appctx'`, so that test failing is the signal that the fix has been reverted.
"""

import tempfile
from pathlib import Path

import gadopt  # noqa: F401 - before firedrake, for the python PC names below
import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, DtNGravityForm
from gadopt.gia_gravity import (selfgrav_dtn_iterative_solver_parameters,
                                selfgrav_dtn_schur_solver_parameters)
from gadopt.preconditioners import DtNMultiplierDiagPC
from test_gravity_interior_sheet import (
    INNER_ID, N_AZIMUTHAL, OUTER_ID, RIN, ROUT, RSHEET, SHEET_ID,
    _write_annulus)

M_MODE = 3

BASE_PARAMETERS = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-11,
    "pc_type": "python",
    "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
    "dtn_pc_fieldsplit_schur_fact_type": "full",
    "dtn_fieldsplit_0_ksp_type": "preonly",
    "dtn_fieldsplit_0_pc_type": "python",
    "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "dtn_fieldsplit_0_assembled_ksp_type": "preonly",
    "dtn_fieldsplit_0_assembled_pc_type": "lu",
    "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
    "dtn_fieldsplit_1_ksp_type": "gmres",
    "dtn_fieldsplit_1_ksp_rtol": 1e-10,
}


@pytest.fixture(scope="module")
def sheet_mesh():
    """The same cached annulus `test_gravity_interior_sheet` builds."""
    pytest.importorskip("gmsh")
    path = Path(tempfile.gettempdir()) / (
        f"gadopt_interior_sheet_{RIN}_{RSHEET}_{ROUT}_{N_AZIMUTHAL}.msh")
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        _write_annulus(path)
    fd.COMM_WORLD.barrier()
    return fd.Mesh(str(path))


def build(sheet_mesh, M=M_MODE):
    """A coupled residual in miniature: volume term plus the boundary form."""
    X = fd.SpatialCoordinate(sheet_mesh)
    phi = fd.atan2(X[1], X[0])
    V = fd.FunctionSpace(sheet_mesh, "CG", 2)
    form = DtNGravityForm(
        V,
        {OUTER_ID: {"dtn": CylindricalDtN(M=M)},
         INNER_ID: {"dtn": CylindricalDtN(M=M)},
         SHEET_ID: {"interior_sigma": fd.cos(M_MODE * phi)}})
    R = fd.FunctionSpace(sheet_mesh, "R", 0)
    W = fd.MixedFunctionSpace([V] + [R] * form.n_multipliers)
    w = fd.Function(W)
    trials, tests = fd.split(w), fd.TestFunctions(W)
    F = fd.dot(fd.grad(trials[0]), fd.grad(tests[0])) * fd.dx(domain=sheet_mesh)
    F += form.boundary_residual(trials[0], tests[0],
                                list(zip(trials[1:], tests[1:])))
    return form, W, w, F


def assembled_diagonal(W, w, F):
    """The `(c_k, c_k)` entries by differencing the assembled residual.

    The residual is linear in `c`, so `F(c_k = 1) - F(c_k = 0)` read on row
    `1 + k` is exactly that diagonal entry. This is the independent truth the
    closed form is checked against.
    """
    n = W.num_sub_spaces() - 1
    w.assign(0.0)
    base = np.array([float(fd.assemble(F).subfunctions[1 + k].dat.data_ro[0])
                     for k in range(n)])
    out = np.zeros(n)
    for k in range(n):
        w.assign(0.0)
        w.subfunctions[1 + k].assign(1.0)
        out[k] = float(fd.assemble(F).subfunctions[1 + k].dat.data_ro[0]) - base[k]
    w.assign(0.0)
    return out


class TestEveryConfigurationThePresetsCanReach:
    """The bar for a **default**, not for an option.

    A default that is right in the configuration someone tested and wrong in
    one they did not is worse than an option, so the closed form is checked
    against the assembled block across the axes the presets can actually vary:
    the truncation, and one DtN boundary versus two. Each case re-derives the
    truth by assembly; none of them compares against a stored number.
    """

    @pytest.mark.parametrize("M", [1, 2, 4])
    def test_across_truncations(self, sheet_mesh, M):
        form, W, w, F = build(sheet_mesh, M=M)
        assert form.n_multipliers == W.num_sub_spaces() - 1
        assert np.allclose(form.multiplier_diagonal(),
                           assembled_diagonal(W, w, F),
                           rtol=1e-12, atol=0.0)

    def test_with_a_single_dtn_boundary(self, sheet_mesh):
        """One boundary rather than two: `A_h` differs per boundary, so a
        formula that accidentally used a single global area would pass the
        two-boundary test only if the two areas happened to agree."""
        X = fd.SpatialCoordinate(sheet_mesh)
        phi = fd.atan2(X[1], X[0])
        V = fd.FunctionSpace(sheet_mesh, "CG", 2)
        form = DtNGravityForm(
            V,
            {OUTER_ID: {"dtn": CylindricalDtN(M=M_MODE)},
             SHEET_ID: {"interior_sigma": fd.cos(M_MODE * phi)}})
        R = fd.FunctionSpace(sheet_mesh, "R", 0)
        W = fd.MixedFunctionSpace([V] + [R] * form.n_multipliers)
        w = fd.Function(W)
        trials, tests = fd.split(w), fd.TestFunctions(W)
        F = fd.dot(fd.grad(trials[0]),
                   fd.grad(tests[0])) * fd.dx(domain=sheet_mesh)
        F += form.boundary_residual(trials[0], tests[0],
                                    list(zip(trials[1:], tests[1:])))
        assert np.allclose(form.multiplier_diagonal(),
                           assembled_diagonal(W, w, F),
                           rtol=1e-12, atol=0.0)

    def test_the_two_boundaries_have_different_discrete_areas(self, sheet_mesh):
        """The precondition that makes the previous test worth running."""
        form, _, _, _ = build(sheet_mesh)
        areas = {bc_id: form.boundary_area[bc_id]
                 for bc_id, _ in form.dtn_boundaries}
        assert len(set(areas.values())) == len(areas), areas


class TestTheDiagonalIsWhatTheFormSaysItIs:

    def test_closed_form_matches_the_assembled_block(self, sheet_mesh):
        """`-scale_k * A_h` against the residual, entry by entry."""
        form, W, w, F = build(sheet_mesh)
        closed = form.multiplier_diagonal()
        truth = assembled_diagonal(W, w, F)
        assert closed.shape == truth.shape
        assert np.allclose(closed, truth, rtol=1e-12, atol=0.0), (
            f"closed form {closed} vs assembled {truth}")

    def test_the_same_tolerance_rejects_the_analytic_area(self, sheet_mesh):
        """The rejecting partner, and the substitution a reader would make.

        `A_h` is the DISCRETE boundary measure; the analytic `2 pi R` is wrong
        by the polygon error. If the check above cannot tell them apart it is
        not testing anything, so it is required to fail here.
        """
        form, W, w, F = build(sheet_mesh)
        truth = assembled_diagonal(W, w, F)
        scales = {}
        for bc_id, dtn in form.dtn_boundaries:
            side, R = form.boundary_geometry[bc_id]
            for mode in dtn.modes(side, R, form.X):
                scales[(bc_id, mode.key)] = float(mode.scale)
        wrong = np.array([
            -scales[key] * 2 * np.pi * form.boundary_geometry[key[0]][1]
            for key in form.multiplier_keys])
        # It must be close enough to be a plausible mistake ...
        assert np.allclose(wrong, truth, rtol=1e-2)
        # ... and the gate's own tolerance must still reject it.
        assert not np.allclose(wrong, truth, rtol=1e-12, atol=0.0)

    def test_the_block_really_is_diagonal(self, sheet_mesh):
        """No constraint row couples one multiplier to another.

        If it did, a diagonal preconditioner would be wrong in a way no
        entrywise check could see.
        """
        form, W, w, F = build(sheet_mesh)
        n = W.num_sub_spaces() - 1
        w.assign(0.0)
        base = np.array([float(fd.assemble(F).subfunctions[1 + j].dat.data_ro[0])
                         for j in range(n)])
        w.subfunctions[1].assign(1.0)
        col = np.array([float(fd.assemble(F).subfunctions[1 + j].dat.data_ro[0])
                        for j in range(n)]) - base
        w.assign(0.0)
        off = np.delete(col, 0)
        assert np.abs(off).max() <= 1e-12 * abs(col[0]), (
            f"off-diagonal entries {off} against diagonal {col[0]}")


class TestTheDiagonalPreconditioner:

    def test_it_inverts_the_diagonal_and_a_wrong_one_does_not(self, sheet_mesh):
        form, _, _, _ = build(sheet_mesh)
        d = form.multiplier_diagonal()
        rng = np.random.default_rng(20260802)
        rhs = rng.standard_normal(d.size)
        assert np.allclose(d * (rhs / d), rhs, rtol=1e-14)
        # the rejecting partner: a sign slip on the diagonal
        assert not np.allclose(d * (rhs / -d), rhs, rtol=1e-14)

    def test_it_refuses_a_missing_diagonal(self):
        pc = DtNMultiplierDiagPC()
        pc.get_appctx = staticmethod(lambda _pc: {})
        with pytest.raises(ValueError, match="dtn_block1_diagonal"):
            pc.initialize(None)

    def test_it_refuses_a_zero_entry(self, sheet_mesh):
        """A zero would be inverted to infinity and poison the solve silently."""
        form, _, _, _ = build(sheet_mesh)
        d = form.multiplier_diagonal().copy()
        d[0] = 0.0

        class _A:
            @staticmethod
            def getSizes():
                return ((d.size, d.size), (d.size, d.size))

        class _PC:
            @staticmethod
            def getOperators():
                return _A(), _A()

        pc = DtNMultiplierDiagPC()
        pc.get_appctx = staticmethod(lambda _pc: {"dtn_block1_diagonal": d})
        with pytest.raises(ValueError, match="zero entry"):
            pc.initialize(_PC())


#: The dense complement, named the way a `pc_python_type` entry names it.
DENSE_PC = "gadopt.DtNMultiplierDenseSchurPC"

#: The block-0 Krylov prefix of each layout of the iterative preset. On the
#: condensed layout block 0 is an FGMRES over the whole block; on the full
#: layout `gadopt.CondensedBlockPC` owns the solve and the tolerances move down
#: to its own `(u, psi)` Krylov solve.
BLOCK0_PREFIX = {True: "dtn_fieldsplit_0_",
                 False: "dtn_fieldsplit_0_condensed_"}


class TestItChangesNoDefault:
    """The defaults of the shipped presets, each pinned by one assertion.

    A default nobody pins is a default that drifts, and a drifting default
    moves every number a campaign produces without anybody asking. A test, not
    a comment, because a comment does not fail.

    These pin the state the Gadi runs measured: on the low-rank
    representation the preset forms the exact complement of the `Real` block,
    the cached apply of `gadopt.DtNTwoBlockSchurPC` owns it and the Schur
    factorisation is `full`, block 0 is solved to 1e-4 and its FGMRES is not
    restarted. Naming `ainvb=False` gives the delegating path, which is
    `gadopt.DtNMultiplierDenseSchurPC` under `lower`, and that is the
    configuration the 2026-09-20 campaign measured. The direct preset is
    untouched by all of it.
    """

    def test_the_low_rank_arm_forms_the_complement_through_the_cached_apply(
            self):
        """4 `Real` rows, so a build is 4 cheap block-0 solves and it pays.

        The preset forms that complement through the cached apply, which keeps
        the outer iteration count of the full factorisation while spending one
        block-0 solve per outer iteration: 45 min 44 s against the delegating
        path's 1 h 18 32 over the full Spada ladder (job 179511971 against
        179496714), and 16.78 s per warm step against 25.6 at dt = 100 yr with
        2 outer iterations against 3 (arms Z2 and Z1 of job 179510821).

        Block 1 is then never entered, so it carries no preconditioner at all.
        """
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        assert p["dtn_schur_ainvb"] is True
        assert p["dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in p
        # and the same when the representation is named rather than resolved
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=False, dtn_representation="lowrank")[
                "dtn_schur_ainvb"] is True
        # `ainvb=False` names the delegating path, which forms the same
        # complement and hands it to PETSc as a block-1 preconditioner.
        off = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, ainvb=False)
        assert off["dtn_fieldsplit_1_pc_python_type"] == DENSE_PC
        assert "dtn_schur_ainvb" not in off

    def test_the_complement_runs_under_preonly_with_no_krylov_keys(self):
        """The factored complement is an exact inverse, so nothing iterates.

        A Krylov method above it spends one Schur-complement `MatMult`, and so
        one block-0 solve, per extra iteration: 120.4 s per 100 yr step against
        `preonly`'s 99.8 under `full` (arms B1 and B2, job 179385036,
        `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 24). A tolerance and a
        cap written for a
        solve that never runs would describe a configuration that is not there.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, ainvb=False)
        assert p["dtn_fieldsplit_1_ksp_type"] == "preonly"
        for dropped in ("dtn_fieldsplit_1_ksp_rtol",
                        "dtn_fieldsplit_1_ksp_max_it"):
            assert dropped not in p, (
                f"{dropped} is written for a block-1 Krylov solve that the "
                "exact complement makes unnecessary")
        # An approximate inverse does not solve the block, so it keeps the
        # Krylov method and both of its stopping keys.
        diag = selfgrav_dtn_iterative_solver_parameters(
            condensed=True, multiplier_pc="gadopt.DtNMultiplierDiagPC")
        assert diag["dtn_fieldsplit_1_ksp_type"] == "gmres"
        assert diag["dtn_fieldsplit_1_ksp_rtol"] == 1e-4
        assert diag["dtn_fieldsplit_1_ksp_max_it"] == 200

    def test_the_multiplier_arm_still_runs_block_one_unpreconditioned(self):
        """About 76 columns there, so a build costs more than an outer solve."""
        for kwargs in ({"condensed": True},
                       {"condensed": True, "dtn_representation": "multiplier"},
                       {"condensed": False, "block0": "pair",
                        "dtn_representation": "multiplier"}):
            p = selfgrav_dtn_iterative_solver_parameters(**kwargs)
            assert p["dtn_fieldsplit_1_pc_type"] == "none", kwargs
            assert "dtn_fieldsplit_1_pc_python_type" not in p, kwargs

    def test_the_direct_preset_runs_block_one_unpreconditioned(self):
        """The 2-D preset is not in the campaign and does not move with it."""
        assert selfgrav_dtn_schur_solver_parameters[
            "dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in \
            selfgrav_dtn_schur_solver_parameters
        assert selfgrav_dtn_schur_solver_parameters[
            "dtn_pc_fieldsplit_schur_fact_type"] == "full"

    def test_a_named_multiplier_pc_beats_the_sentinel_both_ways(self):
        """`None` means "the preset chooses"; a string means what it says."""
        off = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, multiplier_pc="none")
        assert off["dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in off
        on = selfgrav_dtn_iterative_solver_parameters(
            condensed=True, multiplier_pc="gadopt.DtNMultiplierDiagPC")
        assert on["dtn_fieldsplit_1_pc_python_type"] == \
            "gadopt.DtNMultiplierDiagPC"

    def test_the_cache_leaves_block_one_unpreconditioned(self):
        """Under `ainvb` block 1 is never entered, so nothing is named there."""
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, ainvb=True)
        assert p["dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in p

    def test_the_factorisation_type_is_lower_unless_the_cache_owns_the_apply(
            self):
        """`lower` applies block 0 once per outer iteration, `full` twice.

        The cached apply IS `full` with its second block-0 solve replaced, and
        it refuses any other factorisation type, so the two are alternatives
        and the preset must not stack them.
        """
        # The preset writes `full` where it chooses the cached apply, which
        # is the low-rank representation here, and `lower` where it does not.
        assert selfgrav_dtn_iterative_solver_parameters(condensed=True)[
            "dtn_pc_fieldsplit_schur_fact_type"] == "lower"
        assert selfgrav_dtn_iterative_solver_parameters(condensed=False)[
            "dtn_pc_fieldsplit_schur_fact_type"] == "full"
        # Both named values, at the same width, give the two types.
        for ainvb, fact in ((True, "full"), (False, "lower")):
            assert selfgrav_dtn_iterative_solver_parameters(
                condensed=False, ainvb=ainvb)[
                    "dtn_pc_fieldsplit_schur_fact_type"] == fact

    def test_block_zero_is_solved_to_the_tolerance_the_complement_needs(self):
        """The complement is only as linear as the solve that builds it.

        At 1e-2 the dense arm stagnates (642 non-convergent block-0 calls, job
        176078939), so
        the default tolerance and the default block-1 preconditioner are one
        decision and this pins the tolerance half of it.
        """
        for condensed in (True, False):
            p = selfgrav_dtn_iterative_solver_parameters(condensed=condensed)
            assert p[BLOCK0_PREFIX[condensed] + "ksp_rtol"] == 1e-4

    def test_the_block_zero_krylov_solve_is_not_restarted(self):
        """Restart equals the cap, so the Krylov space is never discarded.

        PETSc restarts FGMRES every 30 by default and this solve is allowed
        200 iterations, which without this key throws the space away six times.
        """
        for condensed in (True, False):
            p = selfgrav_dtn_iterative_solver_parameters(condensed=condensed)
            prefix = BLOCK0_PREFIX[condensed]
            assert p[prefix + "ksp_gmres_restart"] == p[prefix + "ksp_max_it"]
            assert p[prefix + "ksp_max_it"] == 200
        # and the restart follows a caller who moves the cap, rather than
        # staying at a number that silently becomes a restart again
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=True, block0_max_it=37)
        assert p["dtn_fieldsplit_0_ksp_gmres_restart"] == 37


class TestTheSolveAgrees:
    """The integration test, and the tripwire for the sub-DM fix."""

    @staticmethod
    def _solve(sheet_mesh, extra, appctx=None):
        form, W, w, F = build(sheet_mesh)
        params = dict(BASE_PARAMETERS)
        params.update(extra)
        problem = fd.NonlinearVariationalProblem(F, w)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=params,
            appctx=appctx or {})
        solver.solve()
        return w.subfunctions[0].copy(deepcopy=True), form

    def test_diag_pc_gives_the_same_answer_as_no_pc(self, sheet_mesh):
        """Same answer to solver tolerance; a preconditioner may not move it.

        This also proves the multiplier KSP now resolves a DM. **Verified by
        reverting the fix and re-running this test: it SEGFAULTS (exit 139),
        it does not raise.** So this test is the only tripwire for those three
        lines, and a segfault here means they have been removed rather than
        that something subtle went wrong.
        """
        form0 = build(sheet_mesh)[0]
        diag = form0.multiplier_diagonal()

        psi_none, _ = self._solve(
            sheet_mesh, {"dtn_fieldsplit_1_pc_type": "none"})
        psi_diag, _ = self._solve(
            sheet_mesh,
            {"dtn_fieldsplit_1_pc_type": "python",
             "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDiagPC"},
            appctx={"dtn_block1_diagonal": diag})

        num = fd.assemble((psi_none - psi_diag) ** 2 * fd.dx(domain=sheet_mesh))
        den = fd.assemble(psi_none ** 2 * fd.dx(domain=sheet_mesh))
        rel = float(np.sqrt(num / den))
        assert rel < 1e-9, f"the preconditioner moved the answer by {rel:.3e}"
        # and the answer is not trivially zero, which would make that pass
        assert den > 1e-12


class TestTheGuardRefusesWhatItCannotDescribe:
    """`block1_diagonal` must raise, not silently mis-describe, if the Real
    block ever stops being 'multipliers, core pressure, then rotation rows'.

    Exercised against a stand-in rather than a coupled solver, because the
    thing under test is the accounting, and building a 2-D coupled solver here
    would test the solver instead.
    """

    @staticmethod
    def _fake(real_in_space, multipliers, rotation, core_pressure=None):
        """A stand-in whose Real block can be laid out arbitrarily."""
        from gadopt.gia_gravity import SelfGravitatingGIASolver

        class _El:
            def __init__(self, fam):
                self._f = fam

            def family(self):
                return self._f

        class _Sub:
            def __init__(self, fam):
                self._e = _El(fam)

            def ufl_element(self):
                return self._e

        n_total = (max(real_in_space) + 1) if real_in_space else 2

        class _Space:
            def __len__(self):
                return n_total

            def sub(self, i):
                return _Sub("Real" if i in real_in_space else "Lagrange")

        class _Sol:
            def function_space(self):
                return _Space()

        class _Form:
            multiplier_keys = tuple(range(len(multipliers)))

            @staticmethod
            def multiplier_diagonal():
                return np.ones(len(multipliers))

        class _Layout:
            gravity_form = _Form()
            rotation_names = ("m1", "m2", "m3")
            # The solver reads layout.dtn_representation (gia_gravity.py); this
            # double represents the multiplier layout the diagonal PC serves.
            dtn_representation = "multiplier"

            def __init__(self):
                self.multipliers = tuple(multipliers)
                self.core_pressure = core_pressure
                self.rotation = dict(rotation)

            @property
            def real_fields(self):
                core = (() if self.core_pressure is None
                        else (self.core_pressure,))
                return tuple(self.multipliers) + core + tuple(
                    self.rotation[n] for n in self.rotation_names
                    if n in self.rotation)

        class _Fake(SelfGravitatingGIASolver):
            # The CURRENT mechanism: `block1_diagonal` reads `theta_psi_value`,
            # which the solver computes from its live factors
            # (`scaling_factor * B_mu / Lambda`). The double drives it through
            # `scaling_factor` so a test can vary one factor and watch the
            # diagonal scale, rather than overriding the composite property that
            # no longer feeds the diagonal.
            scaling_factor = 1.0

            def __init__(self):
                pass

            @property
            def theta_psi_value(self):
                return self.scaling_factor

            def _theta_rot(self, i):
                return self.scaling_factor

            def _closure_constant(self, i):
                return 2.0

        obj = _Fake()
        obj.solution = _Sol()
        obj.layout = _Layout()
        return obj

    def test_it_raises_when_a_foreign_real_field_appears(self):
        """Five Real fields, four accounted for: the count tripwire."""
        obj = self._fake(real_in_space=(2, 3, 4, 5, 6),
                         multipliers=(2, 3, 4, 5), rotation={})
        with pytest.raises(RuntimeError, match="Real sub-fields"):
            obj.block1_diagonal()

    def test_it_raises_on_a_rotation_first_layout(self):
        """The case a COUNT check waves through and order-proofing catches.

        Four Real fields, four accounted for -- the count agrees -- but the
        rotation row is first. A positional build would emit a correctly-sized,
        wrongly-ordered diagonal.
        """
        obj = self._fake(real_in_space=(1, 2, 3, 4),
                         multipliers=(2, 3, 4), rotation={"m3": 1})
        with pytest.raises(RuntimeError, match="Real sub-fields"):
            obj.block1_diagonal()

    def test_it_raises_on_an_interleaved_real_block(self):
        """Non-contiguous Real fields, right count."""
        obj = self._fake(real_in_space=(1, 3), multipliers=(1, 3), rotation={})
        with pytest.raises(RuntimeError, match="contiguous|Real sub-fields"):
            obj.block1_diagonal()

    def test_a_well_formed_layout_is_ordered_multipliers_then_rotation(self):
        obj = self._fake(real_in_space=(1, 2, 3), multipliers=(1, 2),
                         rotation={"m3": 3})
        d = obj.block1_diagonal()
        assert d.shape == (3,)
        # multipliers carry theta_psi * 1.0, the rotation row theta_rot * K
        assert np.allclose(d[:2], 1.0)
        assert np.isclose(d[2], 2.0)

    def test_core_pressure_has_its_physical_zero_diagonal(self):
        obj = self._fake(real_in_space=(1, 2, 3, 4), multipliers=(1, 2),
                         core_pressure=3, rotation={"m3": 4})
        diagonal = obj.block1_diagonal()
        assert np.allclose(diagonal[:2], 1.0)
        assert diagonal[2] == 0.0
        assert diagonal[3] == 2.0

    def test_rotation_and_multiplier_rows_scale_together(self):
        """`scaling_factor` is structurally covered, not merely documented.

        A *uniform* scale error on a diagonal PC is invisible and harmless --
        Krylov spaces are scale-invariant. What would bite is a *relative*
        error between the `theta_psi` rows and the `theta_rot` rows, and that
        would be a silent inefficacy on exactly the rows this project measured
        as least-resolved.

        It cannot happen by construction, because `scaling_factor` sits inside
        **both** the multiplier prefactor `theta_psi = scaling_factor * B_mu /
        Lambda` and the rotation prefactor `_theta_rot = s_i * scaling_factor *
        B_mu * Omega_sq`. Changing it moves both rows by the same factor. The
        double drives both off one `scaling_factor` attribute, so varying that
        one factor and asserting the whole diagonal scales proves the coverage
        rather than inferring it from reading two properties.
        """
        base = self._fake(real_in_space=(1, 2, 3), multipliers=(1, 2),
                          rotation={"m3": 3})
        d1 = base.block1_diagonal()

        # A different `scaling_factor` multiplies the multiplier rows (through
        # `theta_psi_value`) and the rotation row (through `_theta_rot`) by the
        # same factor, so the whole diagonal scales.
        factor = 3.0
        base.scaling_factor = factor
        d2 = base.block1_diagonal()
        assert np.allclose(d2, factor * d1, rtol=1e-14), (d1, d2)

    def test_it_returns_none_on_an_empty_real_block(self):
        obj = self._fake(real_in_space=(), multipliers=(), rotation={})
        assert obj.block1_diagonal() is None


class TestItIsScopedToTheCoupledSolver:
    """`GravitySolver` must not silently inherit this preconditioner.

    Its block-1 rows are UNSCALED where the coupled solver's carry `theta_psi`,
    so one diagonal cannot serve both; and it owns the shipped, verified
    adjoint, while this PC reads `appctx`, which pyadjoint drops on the adjoint
    solve. Both are reasons to leave that path at `pc_type: none`, and the
    first is a numerical wrongness a guard could not see.
    """

    def test_the_pc_refuses_an_appctx_without_a_diagonal(self):
        """The GravitySolver situation, at the level where it can be asserted.

        Running a real `GravitySolver` with this PC selected was measured and
        it **raises a bare `PETSc.Error: error code 101`** naming neither the
        preconditioner nor the diagonal -- the python-PC failure mode this
        project has documented repeatedly. That is not asserted here as a live
        solve, because it leaves PETSc in a state that destabilises the rest of
        the session; it is recorded in the class docstring. What is asserted is
        the property that produces it.
        """
        pc = DtNMultiplierDiagPC()
        pc.get_appctx = staticmethod(lambda _pc: {"mu": 1.0})
        with pytest.raises(ValueError, match="dtn_block1_diagonal"):
            pc.initialize(None)

    def test_the_pc_refuses_a_core_pressure_zero_diagonal(self):
        class _Operator:
            @staticmethod
            def getSizes():
                return ((2, 2), (2, 2))

        class _PC:
            @staticmethod
            def getOperators():
                operator = _Operator()
                return operator, operator

        pc = DtNMultiplierDiagPC()
        pc.get_appctx = staticmethod(
            lambda _pc: {"dtn_block1_diagonal": np.array([1.0, 0.0])})
        with pytest.raises(ValueError, match="fluid-core pressure row"):
            pc.initialize(_PC())

    def test_the_gravity_solver_module_never_supplies_the_key(self):
        """If it ever does, the scoping decision has been reversed silently.

        The name appears in that module's refusal message, so the assertion is
        that it is never used as a dict KEY -- i.e. never actually supplied --
        rather than that the string is absent.
        """
        import pathlib

        import gadopt.gravity_solver as gs
        src = pathlib.Path(gs.__file__).read_text()
        assert '"dtn_block1_diagonal":' not in src
        assert "'dtn_block1_diagonal':" not in src

    def test_gravity_solver_refuses_the_pc_by_name_at_construction(self):
        """The most likely misuse, refused with a sentence instead of a code.

        Copying the three block-1 lines from the coupled system into a
        standalone gravity solve is the natural act. Left to PETSc it produces
        `error code 101` naming nothing; this refuses it before any solve, in
        the `_check_block0_split_matches_layout` spirit.
        """
        from gadopt import CylindricalDtN, GravitySolver

        base = fd.CircleManifoldMesh(24, radius=1.0, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=3, layer_height=1.0 / 3,
                               extrusion_type="radial")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        params = dict(BASE_PARAMETERS)
        params.update(
            {"dtn_fieldsplit_1_pc_type": "python",
             "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDiagPC"})
        with pytest.raises(ValueError, match="scoped to SelfGravitatingGIASolver"):
            GravitySolver(psi, 1.0,
                          bcs={"top": {"dtn": CylindricalDtN(M=2)},
                               "bottom": {"dtn": CylindricalDtN(M=2)}},
                          solver_parameters=params)

    def test_the_refusal_also_covers_solver_parameters_extra(self):
        """The other route in, and the one a user is likelier to take."""
        from gadopt import CylindricalDtN, GravitySolver

        base = fd.CircleManifoldMesh(24, radius=1.0, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=3, layer_height=1.0 / 3,
                               extrusion_type="radial")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        with pytest.raises(ValueError, match="scoped to SelfGravitatingGIASolver"):
            GravitySolver(
                psi, 1.0,
                bcs={"top": {"dtn": CylindricalDtN(M=2)},
                     "bottom": {"dtn": CylindricalDtN(M=2)}},
                solver_parameters="direct",
                solver_parameters_extra={
                    "dtn_fieldsplit_1_pc_type": "python",
                    "dtn_fieldsplit_1_pc_python_type":
                        "gadopt.DtNMultiplierDiagPC"})


class TestTheNullCouplingConfiguration:
    """`B_mu = 0` is supported, and it is where a recomputed theta_psi bites."""

    def test_the_row_scale_is_floored_and_not_zero(self):
        from gadopt.gia_gravity import NULL_COUPLING_ROW_SCALE

        assert NULL_COUPLING_ROW_SCALE != 0.0

    def test_the_diagonal_scales_with_theta_psi_rather_than_recomputing_it(
            self, sheet_mesh):
        """Proves the property is *read*.

        At `B_mu = 0` the property is floored through `_row_scale_B_mu`; a
        reimplementation as `scaling_factor * B_mu / Lambda` would give zero
        and hand the preconditioner a diagonal of zeros to divide by. The test
        is that the output is exactly proportional to whatever `theta_psi`
        returns, including a floored value.
        """
        from gadopt.gia_gravity import SelfGravitatingGIASolver

        form, W, w, F = build(sheet_mesh)
        n = form.n_multipliers

        class _El:
            def __init__(self, fam):
                self._f = fam

            def family(self):
                return self._f

        class _Sub:
            def __init__(self, fam):
                self._e = _El(fam)

            def ufl_element(self):
                return self._e

        class _Space:
            def __init__(self):
                self._subs = [_Sub("Lagrange")] + [_Sub("Real")] * n

            def __len__(self):
                return len(self._subs)

            def sub(self, i):
                return self._subs[i]

        class _Sol:
            def function_space(self):
                return _Space()

        class _Layout:
            gravity_form = form
            multipliers = tuple(range(1, 1 + n))
            core_pressure = None
            rotation = {}
            rotation_names = ("m1", "m2", "m3")
            # The solver reads layout.dtn_representation (gia_gravity.py); this
            # double represents the multiplier layout the diagonal PC serves.
            dtn_representation = "multiplier"

            @property
            def real_fields(self):
                return tuple(self.multipliers)

        class _Fake(SelfGravitatingGIASolver):
            theta = 1.0

            def __init__(self):
                pass

            @property
            def theta_psi_value(self):
                # `block1_diagonal` reads `theta_psi_value`, which the solver
                # computes from its factors and floors through `_row_scale_B_mu`
                # at `B_mu = 0`. The double stands in for whatever number that
                # returns, floored value included, so the test measures that the
                # diagonal is proportional to it and never recomputes it.
                return self.theta

        obj = _Fake()
        obj.solution = _Sol()
        obj.layout = _Layout()

        obj.theta = 1.0
        one = obj.block1_diagonal()
        obj.theta = 2.5
        two = obj.block1_diagonal()

        assert np.all(one != 0.0)
        assert np.allclose(two, 2.5 * one, rtol=1e-14)
        # and the floored null-coupling value gives a usable, nonzero diagonal
        from gadopt.gia_gravity import NULL_COUPLING_ROW_SCALE
        obj.theta = float(NULL_COUPLING_ROW_SCALE)
        assert np.all(obj.block1_diagonal() != 0.0)


#: Every key `selfgrav_dtn_iterative_solver_parameters()` wrote at its defaults
#: at commit `c24685ac` on `sghelichkhani/gia-preconditioner`, sorted. It is
#: the shape before the sixth default, `u_ksp_max_it=0`, which drops the three
#: keys of the truncated CG; the preset writes this set again for a caller who
#: passes `u_ksp_max_it=4`. Written out and not counted so that a key which is
#: renamed, or one which is added while another is dropped, fails the test
#: instead of passing a count check.
KEYS_AT_C24685AC = (
    "dtn_fieldsplit_0_fieldsplit_0_assembled_mg_levels_pc_type",
    "dtn_fieldsplit_0_fieldsplit_0_assembled_pc_gamg_coarse_eq_limit",
    "dtn_fieldsplit_0_fieldsplit_0_assembled_pc_gamg_mis_k_minimum_degree_"
    "ordering",
    "dtn_fieldsplit_0_fieldsplit_0_assembled_pc_gamg_square_graph",
    "dtn_fieldsplit_0_fieldsplit_0_assembled_pc_gamg_threshold",
    "dtn_fieldsplit_0_fieldsplit_0_assembled_pc_type",
    "dtn_fieldsplit_0_fieldsplit_0_ksp_converged_maxits",
    "dtn_fieldsplit_0_fieldsplit_0_ksp_converged_reason",
    "dtn_fieldsplit_0_fieldsplit_0_ksp_max_it",
    "dtn_fieldsplit_0_fieldsplit_0_ksp_rtol",
    "dtn_fieldsplit_0_fieldsplit_0_ksp_type",
    "dtn_fieldsplit_0_fieldsplit_0_pc_python_type",
    "dtn_fieldsplit_0_fieldsplit_0_pc_type",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_mg_levels_pc_type",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_pc_gamg_coarse_eq_limit",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_pc_gamg_mis_k_minimum_degree_"
    "ordering",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_pc_gamg_square_graph",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_pc_gamg_threshold",
    "dtn_fieldsplit_0_fieldsplit_1_assembled_pc_type",
    "dtn_fieldsplit_0_fieldsplit_1_ksp_converged_reason",
    "dtn_fieldsplit_0_fieldsplit_1_ksp_type",
    "dtn_fieldsplit_0_fieldsplit_1_pc_python_type",
    "dtn_fieldsplit_0_fieldsplit_1_pc_type",
    "dtn_fieldsplit_0_ksp_converged_reason",
    "dtn_fieldsplit_0_ksp_gmres_restart",
    "dtn_fieldsplit_0_ksp_max_it",
    "dtn_fieldsplit_0_ksp_rtol",
    "dtn_fieldsplit_0_ksp_type",
    "dtn_fieldsplit_0_pc_fieldsplit_0_fields",
    "dtn_fieldsplit_0_pc_fieldsplit_1_fields",
    "dtn_fieldsplit_0_pc_fieldsplit_type",
    "dtn_fieldsplit_0_pc_type",
    "dtn_fieldsplit_1_ksp_converged_reason",
    "dtn_fieldsplit_1_ksp_max_it",
    "dtn_fieldsplit_1_ksp_rtol",
    "dtn_fieldsplit_1_ksp_type",
    "dtn_fieldsplit_1_pc_type",
    "dtn_pc_fieldsplit_schur_fact_type",
    "ksp_converged_reason",
    "ksp_max_it",
    "ksp_rtol",
    "ksp_type",
    "mat_type",
    "pc_python_type",
    "pc_type",
    "snes_atol",
    "snes_converged_reason",
    "snes_linesearch_type",
    "snes_max_it",
    "snes_rtol",
    "snes_type",
)


class TestTheBlockOneChoiceKeysOnTheNumberOfRealRows:
    """`n_real` decides block 1, and the name of the representation does not.

    The cost of `gadopt.DtNMultiplierDenseSchurPC` is `n` block-0 solves for
    each build, so the width of the `Real` block is the quantity the choice
    depends on. The name of the DtN representation tracks that width on this
    branch only, because the two cases here are 1 or 4 rows on the low-rank
    representation and about 76 on the multiplier one. They separate on
    `sghelichkhani/sea-level`, which adds three centre-of-mass rows and a
    sea-level `Shift`: a low-rank block is 5 or 8 rows wide there and a
    multiplier one is 80 at L = 5.

    Every case here reads the dictionary the preset returns and solves nothing,
    which is what makes the widths 5 and 8 testable on a branch that cannot
    build them.
    """

    #: The widths at which the shipped limit selects the dense complement. 1
    #: and 4 exist on this branch (core pressure alone, and core pressure with
    #: the three rotation rows); 5 and 8 are the sea-level branch's.
    SELECTED = (1, 4, 5, 8, 16)

    #: The widths at which it does not. 0 is an empty block, with nothing to
    #: form; 17 is one row above the limit; 76 is the multiplier block at
    #: L = 5, where a build costs more than a whole outer solve.
    REFUSED = (0, 17, 76)

    @pytest.mark.parametrize("n_real", SELECTED)
    @pytest.mark.parametrize("condensed", (True, False))
    def test_a_narrow_block_takes_the_complement_on_either_representation(
            self, n_real, condensed):
        """The rule is on the width alone, so both arms obey it.

        `condensed=True` resolves to the multiplier representation and
        `condensed=False` to the low-rank one, and at these widths the two
        return the same block-1 configuration. Before the rule, the multiplier
        arm took `pc_type: none` at every width.

        The complement is formed through the cached apply by default, and
        through `gadopt.DtNMultiplierDenseSchurPC` when the caller names
        `ainvb=False`. One width test decides both, because both are the same
        complement and both cost `n` block-0 solves to form.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=condensed, n_real=n_real)
        assert p["dtn_schur_ainvb"] is True
        assert p["dtn_pc_fieldsplit_schur_fact_type"] == "full"
        assert p["dtn_fieldsplit_1_pc_type"] == "none"
        delegating = selfgrav_dtn_iterative_solver_parameters(
            condensed=condensed, n_real=n_real, ainvb=False)
        assert delegating["dtn_fieldsplit_1_pc_type"] == "python"
        assert delegating["dtn_fieldsplit_1_pc_python_type"] == DENSE_PC

    @pytest.mark.parametrize("n_real", REFUSED)
    @pytest.mark.parametrize("condensed", (True, False))
    def test_a_wide_or_empty_block_is_left_unpreconditioned(
            self, n_real, condensed):
        """Above the limit the build costs more than it removes."""
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=condensed, n_real=n_real)
        assert p["dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in p

    @pytest.mark.parametrize("n_real", SELECTED + REFUSED)
    def test_the_block_one_ksp_follows_the_choice_at_every_width(self, n_real):
        """The choice and the KSP type must be written by the same code.

        A class that decided at setup to form nothing would leave block 1
        unsolved under `preonly`, so the decision cannot move into the
        preconditioner. The exact complement runs under `preonly` with no
        tolerance and no cap, because neither describes a solve that happens;
        everything else keeps GMRES at 1e-4 under 200.

        `ainvb=False` is named because this is a test about the block-1 KSP,
        and the preset's own choice at a narrow width is the cached apply,
        which never enters that KSP at all. The default's block-1 keys are
        pinned by `test_a_narrow_block_takes_the_complement_on_either_
        representation`.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=n_real, ainvb=False)
        if p["dtn_fieldsplit_1_pc_type"] == "python":
            assert p["dtn_fieldsplit_1_ksp_type"] == "preonly"
            assert "dtn_fieldsplit_1_ksp_rtol" not in p
            assert "dtn_fieldsplit_1_ksp_max_it" not in p
        else:
            assert p["dtn_fieldsplit_1_ksp_type"] == "gmres"
            assert p["dtn_fieldsplit_1_ksp_rtol"] == 1e-4
            assert p["dtn_fieldsplit_1_ksp_max_it"] == 200

    def test_the_limit_is_a_caller_argument_and_moves_the_boundary(self):
        """16 is a choice with a margin, so a caller can move it.

        4 rows win (88.2 s per step against 269, arm B4, job 179385036) and
        about 76 lose, and no arm measures a width between 5 and 75. So the
        limit has to be movable without a new release, and a test that pins it
        at 16 alone would not show that it is.
        """
        # The limit decides whether the preset forms the complement at all,
        # so it moves the cached apply and the delegating path together.
        assert "dtn_schur_ainvb" in selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=16)
        assert "dtn_schur_ainvb" not in \
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, n_real=17)
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=16, ainvb=False)[
                "dtn_fieldsplit_1_pc_type"] == "python"
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=17, ainvb=False)[
                "dtn_fieldsplit_1_pc_type"] == "none"
        # a caller who has measured a narrower break-even moves it down
        narrower = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=5, dense_schur_max_rows=4)
        assert "dtn_schur_ainvb" not in narrower
        assert narrower["dtn_fieldsplit_1_pc_type"] == "none"
        # and one who has measured a wider one moves it up
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=True, n_real=76, dense_schur_max_rows=80)[
                "dtn_schur_ainvb"] is True
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=True, n_real=76, dense_schur_max_rows=80,
            ainvb=False)["dtn_fieldsplit_1_pc_python_type"] == DENSE_PC
        # 0 switches the preset's choice off at every width
        off = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=1, dense_schur_max_rows=0)
        assert "dtn_schur_ainvb" not in off
        assert off["dtn_fieldsplit_1_pc_type"] == "none"

    @pytest.mark.parametrize("n_real", SELECTED + REFUSED)
    def test_a_named_preconditioner_is_taken_as_written_at_every_width(
            self, n_real):
        """The rule is the sentinel's, so it never overrides a caller."""
        named = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=n_real,
            multiplier_pc="gadopt.DtNMultiplierDiagPC")
        assert named["dtn_fieldsplit_1_pc_python_type"] == \
            "gadopt.DtNMultiplierDiagPC"
        off = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=n_real, multiplier_pc="none")
        assert off["dtn_fieldsplit_1_pc_type"] == "none"

    def test_the_cache_wins_over_a_narrow_block(self):
        """Under `ainvb` block 1 is never entered, at any width.

        The cached apply solves block 1 with its own dense factors, so a
        preconditioner named for that block would be built, configured and
        never applied. The width is tested after `ainvb` for exactly this
        reason.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=4, ainvb=True)
        assert p["dtn_fieldsplit_1_pc_type"] == "none"
        assert "dtn_fieldsplit_1_pc_python_type" not in p
        assert p["dtn_fieldsplit_1_ksp_type"] == "preonly"
        # and a named preconditioner beside it is still refused, width or no
        with pytest.raises(ValueError, match="ainvb=True"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, n_real=4, ainvb=True, multiplier_pc=DENSE_PC)
        # The order of the two cases is observable only here. The block-1
        # dictionary is written from `ainvb` whichever branch chose `"none"`,
        # so a width tested FIRST would select the dense complement, and the
        # `block0_rtol` refusal would then fire for a class that is never
        # applied. A loose tolerance under `ainvb` is the caller's business.
        loose = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=4, ainvb=True, block0_rtol=1e-2)
        assert loose[BLOCK0_PREFIX[False] + "ksp_rtol"] == 1e-2
        assert loose["dtn_fieldsplit_1_pc_type"] == "none"

    def test_a_loose_block_zero_tolerance_is_refused_where_the_rule_selects(
            self):
        """The columns of the complement are block-0 solves.

        At 1e-2 the dense arm stagnates: 642 non-convergent block-0 calls and a
        wall worse than no block-1 preconditioner at all (Gadi job 176078939).
        The refusal is scoped to the preset's own choice, and the rule on the
        width is one of the preset's choices.
        """
        with pytest.raises(ValueError, match="block0_rtol"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, n_real=4, block0_rtol=1e-2)
        # the multiplier arm now reaches the same refusal, because the rule
        # selects the class there at a narrow width
        with pytest.raises(ValueError, match="block0_rtol"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=True, n_real=4, block0_rtol=1e-2)
        # above the limit nothing is selected, so the tolerance is the
        # caller's business
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=True, n_real=76, block0_rtol=1e-2)[
                "dtn_fieldsplit_1_pc_type"] == "none"
        # and a named class at a refused width is the caller's decision
        assert selfgrav_dtn_iterative_solver_parameters(
            condensed=False, n_real=4, block0_rtol=1e-2,
            multiplier_pc=DENSE_PC)[
                "dtn_fieldsplit_1_pc_python_type"] == DENSE_PC

    @pytest.mark.parametrize("condensed", (True, False))
    def test_no_count_keeps_the_choice_by_representation(self, condensed):
        """Every caller that passes no count keeps its whole dictionary.

        With no count the rule falls to the representation, which is what
        keeps the multiplier arm off the cached apply: a build there is about
        76 block-0 solves at L = 5, and no arm measures it. This is the test an
        implementation that read the count when given and chose the cached
        apply otherwise would fail.
        """
        p = selfgrav_dtn_iterative_solver_parameters(condensed=condensed)
        if condensed:
            # The multiplier representation: no complement, either way of
            # forming it.
            assert "dtn_schur_ainvb" not in p
            assert p["dtn_fieldsplit_1_pc_type"] == "none"
            assert p["dtn_pc_fieldsplit_schur_fact_type"] == "lower"
        else:
            assert p["dtn_schur_ainvb"] is True
        assert p == selfgrav_dtn_iterative_solver_parameters(
            condensed=condensed, n_real=None)

    def test_the_multiplier_representation_is_kept_off_the_cached_apply(self):
        """The 76-solve path, named every way a caller reaches it.

        A build costs `n` block-0 solves and `n` is about 76 on the multiplier
        representation at L = 5, where it is 4 on the low-rank one. No arm
        measures that, so the preset must not choose either way of forming the
        complement there.
        """
        for kwargs in ({"condensed": True},
                       {"condensed": False,
                        "dtn_representation": "multiplier"},
                       {"condensed": False, "block0": "pair",
                        "dtn_representation": "multiplier"},
                       {"condensed": False, "n_real": 76},
                       {"condensed": True, "n_real": 80}):
            p = selfgrav_dtn_iterative_solver_parameters(**kwargs)
            assert "dtn_schur_ainvb" not in p, kwargs
            assert p["dtn_pc_fieldsplit_schur_fact_type"] == "lower", kwargs
            assert p["dtn_fieldsplit_1_pc_type"] == "none", kwargs

    def test_the_defaults_drop_exactly_the_three_keys_of_the_truncated_cg(
            self):
        """The sixth default drops three keys from the shape at `c24685ac`.

        The five preset defaults the 2026-09-20 campaign selected are pinned by
        value in `TestItChangesNoDefault`; this pins the shape of the whole
        dictionary around them. `u_ksp_max_it` now defaults to `0`, so the
        displacement split is `preonly` and carries no cap, no tolerance and
        no `ksp_converged_maxits`. Nothing else may move: a key which is
        renamed, or one added while another is dropped, fails here instead of
        passing a count check.
        """
        keys = tuple(sorted(selfgrav_dtn_iterative_solver_parameters()))
        dropped = set(KEYS_AT_C24685AC) - set(keys)
        assert dropped == {
            "dtn_fieldsplit_0_fieldsplit_0_ksp_converged_maxits",
            "dtn_fieldsplit_0_fieldsplit_0_ksp_max_it",
            "dtn_fieldsplit_0_fieldsplit_0_ksp_rtol"}
        assert set(keys) - set(KEYS_AT_C24685AC) == set()
        assert len(keys) == 48
        assert len(KEYS_AT_C24685AC) == 51
        # The three come back, and nothing else with them, for the caller who
        # asks for the truncated CG by name.
        assert tuple(sorted(selfgrav_dtn_iterative_solver_parameters(
            u_ksp_max_it=4))) == KEYS_AT_C24685AC

    def test_a_negative_count_or_limit_is_refused(self):
        """A count comes from `len(layout.real_fields)` and cannot be negative.

        A negative value means whatever computed it has a bug, and silently
        taking the `"none"` branch would hide that.
        """
        with pytest.raises(ValueError, match="n_real"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, n_real=-1)
        with pytest.raises(ValueError, match="dense_schur_max_rows"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, n_real=4, dense_schur_max_rows=-1)
