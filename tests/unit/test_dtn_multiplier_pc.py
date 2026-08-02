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


class TestItChangesNoDefault:
    """The constraint that matters most: these are opt-in.

    Flipping a default would move every number the running campaign produces
    and would make two arms of it incomparable. A test, not a comment, because
    a comment does not fail.
    """

    def test_both_shipped_presets_still_run_block_one_unpreconditioned(self):
        assert selfgrav_dtn_schur_solver_parameters[
            "dtn_fieldsplit_1_pc_type"] == "none"
        for condensed in (True, False):
            assert selfgrav_dtn_iterative_solver_parameters(
                condensed=condensed)["dtn_fieldsplit_1_pc_type"] == "none"

    def test_neither_preset_names_a_python_pc_on_block_one(self):
        assert "dtn_fieldsplit_1_pc_python_type" not in \
            selfgrav_dtn_schur_solver_parameters
        assert "dtn_fieldsplit_1_pc_python_type" not in \
            selfgrav_dtn_iterative_solver_parameters()


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
    block ever stops being 'multipliers then rotation rows'.

    Exercised against a stand-in rather than a coupled solver, because the
    thing under test is the accounting, and building a 2-D coupled solver here
    would test the solver instead.
    """

    @staticmethod
    def _fake(real_in_space, multipliers, rotation):
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

            def __init__(self):
                self.multipliers = tuple(multipliers)
                self.rotation = dict(rotation)

            @property
            def real_fields(self):
                return tuple(self.multipliers) + tuple(
                    self.rotation[n] for n in self.rotation_names
                    if n in self.rotation)

        class _Fake(SelfGravitatingGIASolver):
            def __init__(self):
                pass

            @property
            def theta_psi(self):
                return 1.0

            def _theta_rot(self, i):
                return 1.0

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

    def test_rotation_and_multiplier_rows_scale_together(self):
        """`scaling_factor` is structurally covered, not merely documented.

        A *uniform* scale error on a diagonal PC is invisible and harmless --
        Krylov spaces are scale-invariant. What would bite is a *relative*
        error between the `theta_psi` rows and the `theta_rot` rows, and that
        would be a silent inefficacy on exactly the rows this project measured
        as least-resolved.

        It cannot happen by construction, because `scaling_factor` sits inside
        **both** properties: `theta_psi` is `scaling_factor * B_mu / Lambda`
        and `_theta_rot` is `s_i * scaling_factor * B_mu * Omega_sq`. Changing
        it moves both rows by the same factor. This asserts that explicitly, so
        the coverage is visible rather than inferred from reading two
        properties.
        """
        class _Scaled(type(self._fake(real_in_space=(1, 2, 3),
                                      multipliers=(1, 2),
                                      rotation={"m3": 3}))):
            pass

        base = self._fake(real_in_space=(1, 2, 3), multipliers=(1, 2),
                          rotation={"m3": 3})
        d1 = base.block1_diagonal()

        # the same object with BOTH scalings multiplied by one factor, which is
        # what a different `scaling_factor` does
        factor = 3.0

        class _F2(type(base)):
            @property
            def theta_psi(self):
                return 1.0 * factor

            def _theta_rot(self, i):
                return 1.0 * factor

            def _closure_constant(self, i):
                return 2.0

        obj2 = _F2()
        obj2.solution = base.solution
        obj2.layout = base.layout
        d2 = obj2.block1_diagonal()
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
            rotation = {}
            rotation_names = ("m1", "m2", "m3")

            @property
            def real_fields(self):
                return tuple(self.multipliers)

        class _Fake(SelfGravitatingGIASolver):
            theta = 1.0

            def __init__(self):
                pass

            @property
            def theta_psi(self):
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
