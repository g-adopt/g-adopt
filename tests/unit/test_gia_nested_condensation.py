"""Static condensation of the internal variables inside block 0 of the
self-gravity preconditioner.

On the DG family the mixed space is `(u, M, psi, Real...)`, with one
combined internal-variable field `M` of shape `(n, d, d)`. The nested
iterative preset hands the pair `(u, M)` to `gadopt.InternalVariableSCPC` as
split 0 of the block-0 sweep, so the displacement operator the Krylov solver
sees is the exact condensed matrix and the internal variables never appear as
a split of their own.

These tests check that the nesting reaches the condensation class, that the
solve it produces is the direct solve's answer, that the condensed operator is
reused across fixed-`dt` steps and rebuilt when `dt` changes, and that the
Krylov method on it is CG where the operator is symmetric.

The 2-D cases reuse the annulus of `test_gia_gravity.py`: a rigid core
(`un = 0` on Rc), the `cos 2 phi` load on Re as a `SurfaceLoad`, and the
rotation row `m3`. The 3-D case builds the level-1 radially extruded cubed
sphere with the displacement and the potential on one mesh, and runs one
solve on its 24 cells. `dt` is a `Constant` throughout, because the
operator-reuse fingerprint reads its value and a plain number cannot change.
"""

import gadopt  # noqa: F401  (imported before firedrake, as G-ADOPT requires)
import firedrake as fd
import numpy as np
import pytest

from gadopt.gia_couplings import FluidCore, Rotation, SurfaceLoad
from gadopt.gia_presets import (
    OMEGA_SQ_EARTH,
    selfgrav_dtn_iterative_solver_parameters,
    selfgrav_dtn_schur_solver_parameters,
)
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.internal_variable_equation import history_slices
from gadopt.preconditioners import DtNMultiplierDenseSchurPC, InternalVariableSCPC

import test_gia_gravity  # noqa: E402
from test_gia_gravity import (  # noqa: E402  (module-level fixtures and helpers)
    CURVE_RC,
    CURVE_RE,
    SIGMA_HAT,
    approximation,
    gravity_bcs,
    load_density,
    new_fields,
    polar_moment,
)
# The module fixture of `test_gia_gravity`, bound by assignment so that pytest finds
# it here and flake8 does not read the test arguments of the same name as
# redefinitions of an import.
meshes = test_gia_gravity.meshes


def build(meshes, *, approximation_kwargs=None, dt=1.0,
          dtn_representation="multiplier", **kwargs):
    """The annulus solver of `test_gia_gravity.build`, with a `Constant` dt.

    The configuration is the rigid core (`un = 0` on Rc), the `cos 2 phi`
    load on Re as a `SurfaceLoad` with `g_surface = 1` (its traction is
    `B_mu SIGMA_HAT cos 2 phi` and the same density enters the potential
    row), and the rotation row `m3` with the mantle's polar moment.
    `test_gia_gravity.build` passes `dt=1.0` as a float, which the
    operator-reuse fingerprint cannot see change; the reuse tests here need a
    `Constant` they can assign to. The number of Maxwell elements comes from
    the approximation (a list of two viscosities gives two).

    Args:
      meshes: the `(parent, sub)` pair of the annulus fixture.
      approximation_kwargs: settings merged into `approximation`.
      dt: the initial value of the `Constant` time step.
      dtn_representation: `"multiplier"` by default, because the nested
        route `block0="pair"` exists on the multiplier representation only.
      kwargs: passed to `SelfGravitatingGIASolver` (`solver_parameters`,
        `solver_parameters_extra`, `internal_variable_family`, ...).

    Returns:
      `(solver, u, psi)`, with `u` and `psi` the Functions the solver writes
      after every solve. The history is `solver.internal_variables`.
    """
    parent, sub = meshes
    u, psi = new_fields(meshes)
    couplings = [
        SurfaceLoad(boundary=CURVE_RE, sigma=load_density(sub), g_surface=1.0),
        # `Omega_sq` is the Earth's non-dimensional squared rotation rate.
        Rotation(C=polar_moment(meshes), Omega_sq=OMEGA_SQ_EARTH),
    ]
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(**(approximation_kwargs or {})),
        dt=fd.Constant(dt), gravity_bcs=gravity_bcs(parent, sheet=False),
        bcs={CURVE_RC: {"un": 0.0}}, couplings=couplings,
        dtn_representation=dtn_representation, **kwargs)
    return solver, u, psi


def history(solver):
    """The history tensor of the single Maxwell element, accepted state.

    Args:
      solver: a `SelfGravitatingGIASolver` with one Maxwell element.

    Returns:
      The `(d, d)` slice of `solver.internal_variables`.
    """
    m, = history_slices(solver.internal_variables)
    return m


def nested_preset(**kwargs):
    """The DG-family iterative preset, tightened for a direct comparison.

    `block0="pair"` is named explicitly, and every test in this file depends on
    it. This file is about the nested route: block 0 is a two-split sweep whose
    split 0 is the pair `(u, M)` under `gadopt.InternalVariableSCPC`. The
    preset's default block-0 route is `gadopt.CondensedBlockPC`, which
    eliminates `M` once per block-0 application instead and has no such split;
    `tests/unit/test_gia_condensed_block0.py` is its file. Keeping this arm
    exercised is also what keeps the T2 Gadi measurements reproducible.

    The result is a complete dictionary, which the solver uses as given.

    Args:
      **kwargs: overrides passed to
        `selfgrav_dtn_iterative_solver_parameters`.

    Returns:
      The solver-parameter dictionary.
    """
    # The nested route exists on the multiplier representation alone, and
    # the solver's default is low-rank, so the representation is named here
    # and on every solver in this file.
    settings = dict(condensed=False, block0="pair", snes_type="ksponly",
                    outer_rtol=1e-10, block0_rtol=1e-4, block0_max_it=200,
                    dtn_representation="multiplier")
    settings.update(kwargs)
    return selfgrav_dtn_iterative_solver_parameters(**settings)


def condensation_context(solver) -> InternalVariableSCPC:
    """Reach the nested condensation class through the PETSc objects.

    Outer PC (`DtNTwoBlockSchurPC`) -> its Schur fieldsplit -> block-0 KSP
    -> its multiplicative fieldsplit -> split 0 -> python context.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    block0_ksp, _ = outer.pc.getFieldSplitSchurGetSubKSP()
    split0_ksp = block0_ksp.getPC().getFieldSplitSubKSP()[0]
    context = split0_ksp.getPC().getPythonContext()
    assert isinstance(context, InternalVariableSCPC)
    return context


def multiplier_context(solver):
    """Reach the block-1 (`Real`) preconditioner's python context.

    Outer PC (`DtNTwoBlockSchurPC`) -> its Schur fieldsplit -> the block-1 KSP
    -> its PC -> python context. The block-1 PC is `pc_type: none` in both
    shipped presets, so this only resolves when a test names a python PC
    through `multiplier_pc`.

    Args:
      solver: a solved `SelfGravitatingGIASolver` on a nested preset.

    Returns:
      The `gadopt.DtNMultiplierDenseSchurPC` (or other python PC) instance.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    _, ksp_real = outer.pc.getFieldSplitSchurGetSubKSP()
    return ksp_real.getPC().getPythonContext()


def relative_difference(a, b):
    """`|a - b| / |b|` in the L2 norm, for two fields on the same space."""
    return fd.norm(a - b) / fd.norm(b)


def symmetry_defect(petsc_matrix):
    """`|S - S^T| / |S|` in the Frobenius norm of the assembled operator.

    The transpose is taken on a **copy**, and that is the whole content of this
    helper. `Mat.transpose()` with no `out` argument transposes in place and
    returns the same object, so `petsc_matrix.transpose()` gives back
    `petsc_matrix` itself, already transposed: the subtraction is then
    `S^T - S^T` and the answer is 0.0 for every matrix, including
    `[[1, 2], [0, 1]]`. It also leaves the solver's condensed operator
    transposed for whatever reads it next. `duplicate(copy=True)` first, then
    transpose the duplicate, and the argument is untouched. Same pattern as
    `asymmetry` in `tests/unit/test_internal_variable_history.py`.

    Args:
      petsc_matrix: the assembled operator, left unchanged.

    Returns:
      The relative asymmetry as a float.
    """
    transpose = petsc_matrix.duplicate(copy=True)
    transpose.transpose()
    difference = petsc_matrix.copy()
    difference.axpy(-1.0, transpose)
    return difference.norm() / petsc_matrix.norm()


class TestNestedSolve:
    """The nested preset solves the annulus to the direct preset's answer."""

    @pytest.fixture(scope="class")
    def direct(self, meshes):
        """The direct preset (LU on block 0), solved once; no later solve."""
        solver, _, _ = build(meshes, solver_parameters="direct",
                             solver_parameters_extra={"snes_type": "ksponly"})
        solver.solve()
        return solver

    @pytest.fixture(scope="class")
    def nested(self, meshes):
        """The nested preset, solved once."""
        solver, _, _ = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        return solver

    def test_the_condensation_class_sits_in_split_zero(self, nested):
        solver = nested
        context = condensation_context(solver)
        W = context.cxt.a.arguments()[0].function_space()
        assert len(W) == 2
        assert W[1].value_shape == (1, 2, 2)
        assert context.cxt.appctx is solver.appctx

    def test_every_krylov_level_converged(self, nested):
        solver = nested
        ksp = solver.solver.snes.ksp
        assert ksp.getConvergedReason() > 0
        block0_ksp, _ = ksp.pc.getPythonContext().pc.getFieldSplitSchurGetSubKSP()
        assert block0_ksp.getConvergedReason() > 0
        assert condensation_context(solver).condensed_ksp.getConvergedReason() > 0

    def test_it_matches_the_direct_solve(self, nested, direct):
        """One discrete system, two routes to it.

        The nested preset preconditions; it changes no residual. So the answer
        must be the direct preset's to the accuracy of the Krylov tolerances
        `nested_preset` sets, `outer_rtol` 1e-10 and `block0_rtol` 1e-4. The
        displacement split runs the preset's truncated CG, which reaches its
        own tolerance on almost every application and never stops the block-0
        FGMRES short of `block0_rtol`, so the margin is set by `block0_rtol`
        and not by the displacement split.
        """
        u, u_direct = nested.u, direct.u
        psi, psi_direct = nested.psi, direct.psi
        assert fd.norm(u_direct) > 0.0
        # Measured margins on this annulus: 7.1e-10 in the displacement,
        # 1.8e-11 in the potential, 1.1e-11 in the internal variable. The
        # threshold is an order and a half above the largest of them, so a
        # formulation difference fails it and Krylov noise does not.
        assert relative_difference(u, u_direct) < 1e-8
        assert relative_difference(psi, psi_direct) < 1e-8
        assert relative_difference(history(nested), history(direct)) < 1e-8

    def test_the_condensed_matrix_gets_the_near_incompressible_modes(
            self, nested):
        """GAMG on the condensed operator sees more than the rigid modes.

        `near_nullspace` defaults to `"incompressible"`, so
        `gadopt.near_nullspace_basis` builds the rigid-body modes and the
        low-degree divergence-free fields and `gadopt.InternalVariableSCPC`
        sets them on the assembled condensed matrix. In 2-D the rigid set is 3
        vectors (two translations and one rotation) and the degree-1
        divergence-free space adds 2 more after the drop-tolerant
        orthogonalisation, so a count of 5 is what separates the default from
        the rigid one. Counted on the matrix rather than on the provider,
        because a basis that never reaches `MatSetNearNullSpace` is the
        failure this whole line of work exists to catch.
        """
        context = condensation_context(nested)
        near_nullspace = context.S.petscmat.getNearNullSpace()
        assert len(near_nullspace.getVecs()) == 5


class TestOperatorReuse:
    """The condensed operator is built once per `dt`, not once per solve."""

    def test_assembly_count_follows_dt(self, meshes):
        solver, _, _ = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        context = condensation_context(solver)
        assert context.assembly_count == 1
        assert solver.appctx["operator_version"] == 0
        solver.solve()
        assert context.assembly_count == 1
        assert solver.appctx["operator_version"] == 0
        solver.dt.assign(0.5)
        solver.solve()
        assert context.assembly_count == 2
        assert solver.appctx["operator_version"] == 1
        solver.solve()
        assert context.assembly_count == 2

    def test_assembly_follows_the_tape_sweep(self, meshes):
        """A replay of the tape reassembles the condensed operator at its start.

        A replay does not run `SelfGravitatingGIASolver.solve`, so
        `operator_version` does not move, while the control can have
        changed since the last build. `TapeSweepMarkerBlock` moves
        `gia_tape_sweep` at the start of every replay, and the class reads it:
        the second replay reassembles once, at its only solve. The replay
        solver is the cached forward solver of the solve block
        (Firedrake PR 4638), which shares the solver's context.
        """
        from types import SimpleNamespace

        from firedrake.adjoint import (Control, ReducedFunctional,
                                       continue_annotation, get_working_tape,
                                       pause_annotation, stop_annotating)
        from gadopt.dtn_coupled_adjoint import CACHED_SOLVER_BLOCKS, SOLVE_BLOCK_TYPE

        if not CACHED_SOLVER_BLOCKS:
            pytest.skip("the replay solver is a cached solver of Firedrake PR 4638")
        tape = get_working_tape()
        tape.clear_tape()
        sub = meshes[1]
        with stop_annotating():
            factor = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0)
            solver, u, _ = build(meshes, approximation_kwargs=dict(viscosity=factor),
                                 solver_parameters=nested_preset())
        continue_annotation()
        try:
            solver.solve()
            J = fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))
            rf = ReducedFunctional(J, Control(factor))
        finally:
            pause_annotation()
        block, = [b for b in tape.get_blocks() if isinstance(b, SOLVE_BLOCK_TYPE)
                  and any(o.output is solver.solution for o in b.get_outputs())]
        replay = SimpleNamespace(solver=block.forward_cache.solver)
        with stop_annotating():
            other = fd.Function(factor.function_space()).assign(1.3)
        rf(factor)
        count = condensation_context(replay).assembly_count
        rf(other)
        assert condensation_context(replay).assembly_count == count + 1
        tape.clear_tape()


class TestPowerLaw:
    """Newton on the nested preset, with a state-dependent Jacobian.

    Every other test in this file runs a Newtonian residual, where the Jacobian
    is constant and `DtNTwoBlockSchurPC.update` being a no-op costs nothing.
    These tests pin the two facts that make the same nesting serve a power law,
    whose Jacobian moves at every Newton iteration.

    First, the preconditioner follows the state without being told to. The
    inner fieldsplit is a separate PETSc object with its own setup state: every
    Newton iteration reassembles the matrix-free Jacobian, which moves the
    outer `Mat`'s object state, and `PCApply` on the inner fieldsplit calls
    `PCSetUp` first, which re-extracts the sub-matrices and so runs every
    sub-preconditioner's `update`. `gadopt.InternalVariableSCPC.assembly_count`
    is the readable consequence.

    Second, the answer is the direct route's answer. The nesting preconditions
    and changes no residual, so Newton on it must take the same path and land
    in the same place as Newton on an LU block 0.

    Exponent 3 at transition stress 1e-3 is the operating point the plan
    measured: Newton converges in a few iterations on both routes. At 1e-4 and
    below Newton fails on both routes alike, which is a time-stepping question
    (parent plan S2) and not a statement about this nesting.
    """

    POWER_LAW = {"exponent": 3.0, "transition_stress": 1e-3}
    #: Both routes run the same Newton tolerance, so that an iteration-count
    #: comparison between them is a comparison of the paths and not of two
    #: different stopping tests.
    NEWTON = {"snes_rtol": 1e-8}

    @staticmethod
    def _newton_count(solver):
        """Newton iterations taken by the most recent solve."""
        return solver.solver.snes.getIterationNumber()

    def test_newton_matches_the_direct_route_and_rebuilds_each_iteration(
            self, meshes):
        """Two routes, one Newton path, and the condensation rebuilt each step.

        Asserted together because they are one statement: the nested route can
        only track the direct route's Newton iterates if the operator inside
        the preconditioner tracks the Jacobian, and `assembly_count` is the
        evidence that it does. A condensation frozen at the first iterate is
        still a valid preconditioner, so it moves the iteration counts and not
        the answer, which is why both are checked.

        `operator_version` is `None` for a power law: the fingerprint that lets
        a Newtonian solve reuse its condensed operator across time steps cannot
        express "the operator moves within the solve", so the solver publishes
        `None` and `InternalVariableSCPC` rebuilds on every update.

        The condensed field runs GMRES with a restart equal to the preset's
        iteration cap, because a power law makes that operator nonsymmetric in
        2-D. The restart is read from the options dictionary: this petsc4py
        exposes `setGMRESRestart` but no getter.
        """
        nested, _, _ = build(
            meshes, approximation_kwargs=self.POWER_LAW,
            solver_parameters=nested_preset(snes_type="newtonls"),
            solver_parameters_extra=dict(self.NEWTON))
        direct, _, _ = build(
            meshes, approximation_kwargs=self.POWER_LAW,
            solver_parameters=dict(selfgrav_dtn_schur_solver_parameters,
                                   snes_type="newtonls",
                                   snes_linesearch_type="l2"),
            solver_parameters_extra=dict(self.NEWTON))

        newton_counts = []
        context = None
        for _ in range(2):
            nested.solve()
            direct.solve()
            # The PETSc objects the condensation sits under exist only after a
            # solve has set the preconditioner up, so the context is reached
            # here and not at construction.
            context = context or condensation_context(nested)
            # Published by `_refresh_operator_version` at the start of every
            # solve: a Jacobian that depends on the solution has no version
            # that a cache could key on.
            assert nested.appctx["operator_version"] is None
            assert nested.solver.snes.getConvergedReason() > 0
            assert direct.solver.snes.getConvergedReason() > 0
            newton_counts.append(self._newton_count(nested))
            # The same residual and the same Newton tolerance on both routes,
            # so a difference in iteration count means the two are not solving
            # the same problem.
            assert self._newton_count(direct) == newton_counts[-1]
            # A power law needs more than one Newton step; a test that passes
            # on a linear residual proves nothing.
            assert newton_counts[-1] >= 2
            # One condensed-operator assembly per Newton iteration, cumulative
            # over solves.
            assert context.assembly_count == sum(newton_counts)
        print(f"\n    [power law] Newton iterations per solve {newton_counts}"
              f"  assembly_count {context.assembly_count}")

        # The preset's default on this split is one GAMG V-cycle, so there is
        # no Krylov method for the power-law rule to replace and no restart
        # key is written. `test_power_law_gets_gmres` covers the caller who
        # asks for the truncated CG instead.
        assert context.condensed_ksp.getType() == "preonly"
        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert prefix + "ksp_gmres_restart" not in nested.solver_parameters

        u, u_direct = nested.u, direct.u
        psi, psi_direct = nested.psi, direct.psi
        m, m_direct = history(nested), history(direct)
        assert fd.norm(u_direct) > 0.0
        # The margin is set by the nested preset's `block0_rtol` of 1e-4, as it
        # is for the Newtonian comparison in `TestNestedSolve`; the measured
        # values go in the handoff.
        differences = {
            "u": relative_difference(u, u_direct),
            "m": relative_difference(m, m_direct),
            "psi": relative_difference(psi, psi_direct)}
        summary = "  ".join(f"{k}={v:.2e}" for k, v in differences.items())
        print(f"    [power law] relative difference against the direct route "
              f"{summary}")
        for key, value in differences.items():
            assert value < 1e-8, f"{key} differs by {value:.3e}"

    @pytest.mark.parametrize("rule,expected", [
        ("per_solve", [1, 2, 3]), (None, [1, 1, 2])])
    def test_dense_schur_complement_rebuilds_once_per_solve_for_a_power_law(
            self, meshes, rule, expected):
        """One dense complement per time step, not one per Newton iteration.

        `gadopt.DtNMultiplierDenseSchurPC` forms the whole multiplier Schur
        complement by one block-0 solve per column. For a power law the block-0
        operator moves with the state, so the complement built at one Newton
        iteration describes a Jacobian the later iterations no longer have.
        A rebuild at every iteration roughly triples the block-0 work of a
        three-iteration step, and a lagged complement changes no residual -
        it is a preconditioner - so the rule is one build per solve, from the
        state each solve starts from.

        The rule is driven by `gia_solve_index`, which
        `SelfGravitatingGIASolver.solve` increments before each nonlinear
        solve, and gated by `gia_jacobian_depends_on_solution`. A time-step
        change rebuilds under either rheology, which the last step checks.

        `per_solve` builds at each of the three solves. The default
        (`gia_rebuild = lagged`, every 5 solves) keeps the first complement
        for the second solve and rebuilds at the third because `dt` changed.
        """
        extra = dict(self.NEWTON)
        if rule is not None:
            extra["gia_rebuild"] = rule
        solver, _, _ = build(
            meshes, approximation_kwargs=self.POWER_LAW,
            solver_parameters=nested_preset(
                snes_type="newtonls",
                multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"),
            solver_parameters_extra=extra)
        # The markers the rule reads must be in the context before the first
        # build, which happens inside the first solve.
        assert "operator_version" in solver.appctx
        assert "gia_solve_index" in solver.appctx

        solver.solve()
        context = multiplier_context(solver)
        assert isinstance(context, DtNMultiplierDenseSchurPC)
        # More than one Newton iteration, so "once per solve" and "once per
        # Newton iteration" are distinguishable at all.
        assert solver.solver.snes.getIterationNumber() >= 2
        counts = [context.build_count]

        solver.solve()
        counts.append(context.build_count)
        solver.dt.assign(0.5)
        solver.solve()
        counts.append(context.build_count)
        print(f"\n    [dense, power law, {rule}] build_count {counts}")
        assert counts == expected

    def test_dense_schur_complement_rebuilds_when_a_coefficient_changes(
            self, meshes):
        """The Newtonian control: one build per state of the mechanics block.

        `gia_solve_index` is published on every path, so a rule that reads it
        unconditionally rebuilds the complement on every Newtonian solve as
        well and costs one block-0 solve per column per time step for nothing.
        The solve-index test is reached only when `operator_version` is `None`,
        which a Newtonian rheology never publishes, so a fixed-coefficient
        march keeps the complement it has.

        What the version buys over the time step alone is the middle step here.
        A march that writes a new viscosity between steps, or assigns a shear
        modulus or `B_mu`, changes the mechanics block without touching `dt`,
        and a rule keyed on the time-step value alone leaves the complement
        describing the old block with no diagnostic. `operator_version` is the
        solver's own statement that a Jacobian coefficient moved - it
        fingerprints every one of them at the start of each `solve` - and
        `gadopt.InternalVariableSCPC` already trusts it for the condensed
        operator, so the complement follows the same signal.

        The viscosity is asked for as a `Constant` here because the shared
        `approximation` helper stores a plain float otherwise, and a float
        cannot be assigned to. With `shear_modulus` at its default 1.0 the
        Maxwell time is that same `Constant`, so the assignment reaches the
        history equation and the fingerprint sees it.
        """
        solver, _, _ = build(
            meshes,
            approximation_kwargs={"viscosity": fd.Constant(1.0)},
            solver_parameters=nested_preset(
                multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"))
        assert "operator_version" in solver.appctx

        solver.solve()
        context = multiplier_context(solver)
        counts = [context.build_count]
        versions = [solver.appctx["operator_version"]]
        solver.solve()
        counts.append(context.build_count)
        versions.append(solver.appctx["operator_version"])

        # A coefficient of the Jacobian moves, `dt` untouched.
        solver.approximation.viscosity[0].assign(2.0)
        solver.solve()
        counts.append(context.build_count)
        versions.append(solver.appctx["operator_version"])
        # The fingerprint saw the assignment. Without this check the build
        # count below pins the wrong mechanism.
        assert versions[2] != versions[1]

        solver.dt.assign(0.5)
        solver.solve()
        counts.append(context.build_count)
        versions.append(solver.appctx["operator_version"])

        solver.solve()
        counts.append(context.build_count)
        versions.append(solver.appctx["operator_version"])

        print(f"\n    [dense, Newtonian] build_count {counts} "
              f"operator_version {versions}")
        assert counts == [1, 1, 2, 3, 3]

    def test_fluid_core_matches_the_direct_route(self, meshes):
        """A power law with a fluid core, nested against direct.

        The fluid core puts a `Real` core-pressure unknown next to the DtN
        multipliers, and its diagonal entry is a physical zero: the row states
        that the core's boundary is a level surface, not that the pressure is
        proportional to anything. So `pc_type: none` and the exact diagonal are
        both unavailable on that block and the dense Schur complement is the
        multiplier preconditioner this configuration needs, which is why this
        test names it.

        Nothing about the power law is special to the core, and that is the
        claim: the same two routes that agree without a core agree with one,
        at the same Newton counts.

        Three tolerances, and they differ because the quantities do. `M` and
        `psi` agree at 1e-15. The displacement agrees at 4e-9 (measured), which
        is four orders worse than the 3e-12 of the core-free comparison and
        does not improve when `block0_rtol` is tightened from 1e-6 to 1e-8 - it
        moves to 1.7e-8 - so 1e-8 sits on the noise of this saddle system and
        the gate is set at 1e-6, an order and a half above the measurement.
        The core pressure is compared **absolutely**: a `cos 2phi` load drives
        no net radial flux through the CMB, so the row solves to roundoff on
        both routes (4e-19) and a relative comparison of two roundoff numbers
        measures nothing.
        """
        nested, _, _ = self._build_fluid_core(
            meshes, nested_preset(
                snes_type="newtonls", block0_rtol=1e-6,
                multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"))
        direct, _, _ = self._build_fluid_core(
            meshes, dict(selfgrav_dtn_schur_solver_parameters,
                         snes_type="newtonls", snes_linesearch_type="l2"))

        # 2-D, so the condensed operator is treated as nonsymmetric whatever
        # the core treatment is. The preset's default on this split is one
        # GAMG V-cycle, which assumes no symmetry, so the power-law rule has
        # nothing to replace and writes no restart key.
        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert nested.solver_parameters[prefix + "ksp_type"] == "preonly"
        assert prefix + "ksp_gmres_restart" not in nested.solver_parameters

        newton_counts = []
        for _ in range(2):
            nested.solve()
            direct.solve()
            assert nested.solver.snes.getConvergedReason() > 0
            assert direct.solver.snes.getConvergedReason() > 0
            newton_counts.append(nested.solver.snes.getIterationNumber())
            assert direct.solver.snes.getIterationNumber() == newton_counts[-1]
            assert newton_counts[-1] >= 2

        context = condensation_context(nested)
        # The split PETSc built, and not only the dictionary: the preset's
        # default is one GAMG V-cycle here too.
        assert context.condensed_ksp.getType() == "preonly"
        # The core-pressure row is inside the block the dense complement
        # inverts, which is what makes this a test of that saddle and not of
        # the multiplier rows alone.
        assert "core_pressure" in nested.field_index
        assert multiplier_context(nested)._n == len(nested.real_indices)

        u, u_direct = nested.u, direct.u
        psi, psi_direct = nested.psi, direct.psi
        m, m_direct = history(nested), history(direct)
        pressure, pressure_direct = (
            s.real_field("core_pressure") for s in (nested, direct))
        assert fd.norm(u_direct) > 0.0

        differences = {"u": relative_difference(u, u_direct),
                       "m": relative_difference(m, m_direct),
                       "psi": relative_difference(psi, psi_direct)}
        # The mean absolute gap of the two `Real` values over the parent mesh,
        # which is the absolute difference of the two numbers.
        parent = nested.potential_mesh
        core = fd.assemble(abs(pressure - pressure_direct) * fd.dx(
            domain=parent)) / fd.assemble(fd.Constant(1.0) * fd.dx(domain=parent))
        summary = "  ".join(f"{k}={v:.2e}" for k, v in differences.items())
        print(f"\n    [fluid core] Newton iterations per solve "
              f"{newton_counts}  {summary}  core_pressure_gap={core:.2e}  "
              f"asymmetry={symmetry_defect(context.S.petscmat):.2e}")
        assert differences["u"] < 1e-6
        assert differences["m"] < 1e-8
        assert differences["psi"] < 1e-8
        assert core < 1e-12
        assert np.isfinite(core)

    @staticmethod
    def _build_fluid_core(meshes, solver_parameters):
        """The annulus of `build`, with a `FluidCore` in place of `un = 0`.

        Mirrors `test_gia_gravity.TestFluidCore.build_fluid`, with the
        `Constant` time step this module needs and the power-law rheology.
        No rotation, so the only `Real` fields are the DtN multipliers and
        the core pressure.

        Args:
          meshes: the `(parent, sub)` pair of the annulus fixture.
          solver_parameters: the preset or dictionary under test.

        Returns:
          `(solver, u, psi)`.
        """
        parent, sub = meshes
        u, psi = new_fields(meshes)
        couplings = [
            # No `un` at the CMB: the fluid core replaces it, and the
            # solver refuses both on one boundary.
            FluidCore(boundary=CURVE_RC, rho_core=2.0),
            # The `cos 2 phi` load, its traction `B_mu SIGMA_HAT cos 2 phi`
            # and its density in the potential row.
            SurfaceLoad(boundary=CURVE_RE, sigma=load_density(sub),
                        g_surface=1.0),
        ]
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(**TestPowerLaw.POWER_LAW),
            dt=fd.Constant(1.0), gravity_bcs=gravity_bcs(parent, sheet=False),
            bcs={}, couplings=couplings, dtn_representation="multiplier",
            solver_parameters=solver_parameters,
            solver_parameters_extra=dict(TestPowerLaw.NEWTON))
        return solver, u, psi


class TestKrylovSelection:
    """CG where the condensed operator is symmetric, GMRES where it is not."""

    def test_newtonian_two_elements_is_symmetric_and_gets_cg(self, meshes):
        # Two viscosities and two shear moduli give two Maxwell elements.
        solver, _, _ = build(
            meshes,
            approximation_kwargs={"viscosity": [1.0, 3.0],
                                  "shear_modulus": [1.0, 0.5]},
            solver_parameters=nested_preset(u_ksp_max_it=4))
        solver.solve()
        context = condensation_context(solver)
        assert context.condensed_ksp.getType() == "cg"
        # `getTolerances` is (rtol, atol, divtol, max_it). The cap is the
        # preset's `u_ksp_max_it`: the condensed solve is a preconditioner
        # inside the block-0 FGMRES, so it is truncated on purpose and a KSP
        # that ran to convergence here would be the oversolve this nesting was
        # measured to suffer from.
        assert context.condensed_ksp.getTolerances()[3] == 4
        # A Newtonian rheology is symmetric on every cell shape, and this
        # annulus is triangles. Measured 1.0e-16.
        assert symmetry_defect(context.S.petscmat) < 1e-12

    def test_power_law_gets_gmres(self, meshes):
        """The switch keeps the length of the solve it replaces.

        A GMRES restart longer than the iteration cap would allocate Krylov
        vectors the solve never reaches, so the restart the switch writes is
        the cap the preset wrote. The CG is asked for by name, because the
        preset's default on this split is one GAMG V-cycle and there is then
        no Krylov method to replace.
        """
        solver, _, _ = build(
            meshes, approximation_kwargs={"exponent": 3.0,
                                          "transition_stress": 1.0},
            solver_parameters=nested_preset(snes_type="newtonls",
                                            u_ksp_max_it=4))
        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert solver.solver_parameters[prefix + "ksp_type"] == "gmres"
        assert solver.rheology_nonlinear
        assert (solver.solver_parameters[prefix + "ksp_gmres_restart"]
                == solver.solver_parameters[prefix + "ksp_max_it"] == 4)

    def test_the_default_split_is_left_alone_by_the_power_law_rule(
            self, meshes):
        """The pair route takes the sixth default like the other route.

        `u_ksp_max_it` writes one displacement split for both block-0 routes,
        so this route is `preonly` by default as well, and a second default
        here would break the comparison the route exists for. One GAMG V-cycle
        is not a Krylov method, so the power-law rule leaves it alone and
        writes no restart key at a prefix that would never read it.
        """
        solver, _, _ = build(
            meshes, approximation_kwargs={"exponent": 3.0,
                                          "transition_stress": 1.0},
            solver_parameters=nested_preset(snes_type="newtonls"))
        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert solver.rheology_nonlinear
        assert solver.solver_parameters[prefix + "ksp_type"] == "preonly"
        assert prefix + "ksp_gmres_restart" not in solver.solver_parameters
        assert prefix + "ksp_max_it" not in solver.solver_parameters

    def test_a_caller_named_krylov_type_wins(self, meshes):
        key = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_ksp_type"
        solver, _, _ = build(
            meshes, approximation_kwargs={"exponent": 3.0,
                                          "transition_stress": 1.0},
            solver_parameters=nested_preset(snes_type="newtonls"),
            solver_parameters_extra={key: "fgmres"})
        assert solver.solver_parameters[key] == "fgmres"


class TestThreeDimensions:
    """The production path: 3-D, weak `un` at the CMB, Newtonian, CG."""

    @staticmethod
    def shell():
        """The level-1 radially extruded cubed sphere, 24 cells, inner radius 1.

        The base sphere has degree-2 coordinates, so the mesh is curved; two
        layers of height 0.5 put the free surface (`"top"`) at radius 2 and
        the CMB (`"bottom"`) at radius 1.

        Returns:
          `(mesh, C)`: the mesh, which carries both the displacement and the
          potential, and its polar second moment `C = int r^2 dx` for the
          rotational closure.
        """
        base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        return mesh, C

    @staticmethod
    def solver_on(mesh, C, solver_parameters, *, bcs, couplings=(), L=1,
                  approximation_kwargs=None, solver_parameters_extra=None):
        """A solver on the shell, with DtN on both surfaces and rotation.

        The load is the traction `SIGMA_HAT z` on the top, a load with no
        mass (the solver warns), as in the tests this class pins.

        Args:
          mesh, C: the return of `shell`.
          solver_parameters: the preset or dictionary under test.
          bcs: the displacement conditions; the top traction is added here.
          couplings: couplings besides the rotation (a `FluidCore`).
          L: the `SphericalDtN` truncation degree on both surfaces.
          approximation_kwargs: rheology settings, empty for the Newtonian case.
          solver_parameters_extra: extra PETSc options.

        Returns:
          `(solver, u, psi)`.
        """
        from gadopt import SphericalDtN

        X = fd.SpatialCoordinate(mesh)
        u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2),
                        name="displacement")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
        # `C - A = 0.1 C` closes the polar-wander pair `m1`, `m2`; `Omega_sq`
        # is the Earth's non-dimensional squared rotation rate.
        rotation = Rotation(C=C, C_minus_A=0.1 * C, Omega_sq=OMEGA_SQ_EARTH)
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(**(approximation_kwargs or {})),
            dt=fd.Constant(1.0),
            gravity_bcs={"top": {"dtn": SphericalDtN(L)},
                         "bottom": {"dtn": SphericalDtN(L)}},
            bcs={**bcs, "top": {"normal_stress": SIGMA_HAT * X[2]}},
            couplings=[rotation, *couplings],
            dtn_representation="multiplier",
            solver_parameters=solver_parameters,
            solver_parameters_extra=solver_parameters_extra)
        return solver, u, psi

    @classmethod
    def build(cls, L=1):
        """The rigid-core shell (`un = 0` at the bottom) on the nested preset.

        Args:
          L: the `SphericalDtN` truncation degree.

        Returns:
          `(solver, u, psi)`.
        """
        mesh, C = cls.shell()
        return cls.solver_on(mesh, C, nested_preset(),
                             bcs={"bottom": {"un": 0.0}}, L=L)

    def test_condensed_operator_is_symmetric_and_solved_by_one_v_cycle(self):
        """One solve on 24 cells, then read the operator the split saw.

        A bare `pc.setUp()` outside a solve has no solver context on the DM
        (PETSc error 101 from `DMCreateSubDM`), and the setup-only recipe
        needs Firedrake's private solver context, so the cheapest reliable
        route to the assembled condensed operator is one solve.

        The fixture names no `u_ksp_max_it`, so the split is the preset's
        default: one GAMG V-cycle, `ksp_type preonly`. The symmetry assertion
        below stands on its own, because a caller who asks for the truncated
        CG with `u_ksp_max_it=4` needs a symmetric operator for it.
        """
        solver, _, _ = self.build()
        assert solver.field_index["history"] == 1
        solver.solve()
        assert solver.solver.snes.ksp.getConvergedReason() > 0
        context = condensation_context(solver)
        assert context.condensed_ksp.getType() == "preonly"
        W = context.cxt.a.arguments()[0].function_space()
        assert W[1].value_shape == (1, 3, 3)
        # Newtonian, so symmetric on the extruded hexahedra of this sphere as
        # well: the cell shape limits the power-law case only. Measured
        # 7.9e-17 with the rigid `un` core here, and 1.7e-16 with a fluid core
        # in the test below.
        assert symmetry_defect(context.S.petscmat) < 1e-12

    @staticmethod
    def fluid_core_space():
        """The 24-cell extruded sphere for the fluid-core solvers.

        Built once and shared by the two solvers of the comparison below,
        because `fd.norm(a - b)` refuses two functions on two different mesh
        objects with "Multiple domains found" - each `CubedSphereMesh` call
        makes a new one.

        Returns:
          `(mesh, C)`, the mesh and the polar second moment the rotational
          closure needs.
        """
        return TestThreeDimensions.shell()

    @staticmethod
    def fluid_core_solver(mesh, C, solver_parameters, *,
                          approximation_kwargs=None,
                          solver_parameters_extra=None, boundary="bottom"):
        """A solver on that mesh with a `FluidCore` in place of the CMB `un`.

        The CMB of a radially extruded sphere is its bottom surface, so the
        boundary is named `"bottom"` and the fluid-core measure resolves it
        through `gadopt.utility.CombinedSurfaceMeasure`.

        Args:
          mesh, C: the return of `fluid_core_space`.
          solver_parameters: the preset or dictionary under test.
          approximation_kwargs: rheology settings, empty for the Newtonian case.
          solver_parameters_extra: extra PETSc options.
          boundary: the fluid core's boundary, so a test can pass a wrong one.

        Returns:
          `(solver, u, psi)`.
        """
        # No `un` at the bottom: the fluid core replaces it, and the solver
        # refuses both on one boundary.
        return TestThreeDimensions.solver_on(
            mesh, C, solver_parameters, bcs={},
            couplings=[FluidCore(boundary=boundary, rho_core=2.0)],
            approximation_kwargs=approximation_kwargs,
            solver_parameters_extra=solver_parameters_extra)

    def test_fluid_core_newtonian_matches_the_direct_route(self):
        """The control for the extruded CMB measure, with no rheology in it.

        The fluid-core measure resolves `"bottom"` through
        `gadopt.utility.CombinedSurfaceMeasure`, which maps it to `ds_b`; a
        plain `ds("bottom")` refuses the name and makes the fluid core
        unreachable on the 3-D production geometry. This test says the mapped
        measure gives the right system, and it says it on a Newtonian rheology
        so that the statement is about the measure and nothing else.

        `FluidCore.check` runs inside the constructor, so both builds
        succeeding is the assertion that it accepts `"bottom"`; the wrong-tag
        case below is the assertion that it still refuses what it must.
        """
        mesh, C = self.fluid_core_space()
        nested, _, _ = self.fluid_core_solver(
            mesh, C,
            nested_preset(block0_rtol=1e-6,
                          multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"))
        direct, _, _ = self.fluid_core_solver(
            mesh, C,
            dict(selfgrav_dtn_schur_solver_parameters, snes_type="ksponly"))

        nested.solve()
        direct.solve()
        assert nested.solver.snes.getConvergedReason() > 0
        assert direct.solver.snes.getConvergedReason() > 0
        assert nested.solver.snes.ksp.getConvergedReason() > 0

        u, u_direct = nested.u, direct.u
        psi, psi_direct = nested.psi, direct.psi
        m, m_direct = history(nested), history(direct)
        assert fd.norm(u_direct) > 0.0

        differences = {"u": relative_difference(u, u_direct),
                       "m": relative_difference(m, m_direct),
                       "psi": relative_difference(psi, psi_direct)}
        # The core pressure is compared absolutely and not relatively. The
        # load `SIGMA_HAT * X[2]` is odd in z and drives no net radial flux
        # through the CMB, so the row solves to roundoff on both routes and a
        # relative comparison of two roundoff numbers measures nothing.
        pressure, pressure_direct = (
            s.real_field("core_pressure") for s in (nested, direct))
        volume = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        core = fd.assemble(
            abs(pressure - pressure_direct) * fd.dx(domain=mesh)) / volume
        value = fd.assemble(pressure * fd.dx(domain=mesh)) / volume
        summary = "  ".join(f"{k}={v:.2e}" for k, v in differences.items())
        print(f"\n    [3-D fluid core, Newtonian] {summary}  "
              f"core_pressure={value:.2e}  gap={core:.2e}")
        # Measured: 3.6e-11 in `u`, 1.5e-12 in `m`, 1.9e-12 in `psi`, and a
        # core-pressure gap of 7.7e-16.
        for key, difference in differences.items():
            assert difference < 1e-8, f"{key} differs by {difference:.3e}"
        assert core < 1e-12
        assert np.isfinite(value)
        assert "core_pressure" in nested.field_index
        assert multiplier_context(nested)._n == len(nested.real_indices)

    def test_a_fluid_core_tag_with_no_facets_is_still_refused(self):
        """The extruded measure resolves more names, and refuses no fewer.

        `CombinedSurfaceMeasure` sends every integer tag to `ds_v`, the
        vertical (side) facets, and a radially extruded closed sphere has none.
        So an integer tag names an empty facet set here and must reach the
        fluid core's empty-measure refusal, exactly as a wrong tag does on the
        annulus. Without this check, a branch that resolves every name it is
        given leaves the geometry check dead on extruded meshes.
        """
        mesh, C = self.fluid_core_space()
        with pytest.raises(ValueError, match="empty measure"):
            self.fluid_core_solver(mesh, C, nested_preset(), boundary=4)

    def test_fluid_core_power_law_is_nonsymmetric_and_gets_gmres(self):
        """The measurement the 2-D fluid-core test cannot make, and its answer.

        `rheology_nonlinear` sends every power law to GMRES, so the method
        here follows from the rheology alone. What this test adds is the
        measurement that the rule is also right on the facts for this
        configuration, which is not obvious: a fluid core carries no stress, no
        `mu` and no Nitsche term, so the guess that it leaves the `(u, M)`
        block symmetric is a reasonable one to want checked.

        **Measured on this 24-cell sphere: asymmetry 9.4e-2**, nine percent,
        with a fluid core; 1.4e-2 with the rigid `un` core; 3.4e-3 on an affine
        hexahedral box with the standalone coupled solver, against 1.1e-16 on
        tetrahedra. The cell shape is what does it: a radially extruded cubed
        sphere reports a `TensorProductCell(quadrilateral, interval)`, and the
        symmetry argument needs the deviatoric strain of a displacement
        increment to lie in the DG history space, which holds for P2 on a
        tetrahedron and fails for Q2 on a hexahedron or a prism.

        The fluid core is not the cause and the Newtonian control above is what
        says so: the same mesh, the same core, asymmetry 1.7e-16. The fluid
        core's energy carries `B_mu`, `rho_core`, `g0`, `u.n`, `psi` and
        `p_core` only, so it adds nothing to the block the condensation builds.

        This is a 24-cell toy, so it establishes that the construction works
        and not that it scales; the cost of a fluid-core power law at
        production rank counts is the parent plan's S4.
        """
        mesh, C = self.fluid_core_space()
        solver, _, _ = self.fluid_core_solver(
            mesh, C,
            nested_preset(snes_type="newtonls", block0_rtol=1e-6,
                          u_ksp_max_it=4,
                          multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"),
            approximation_kwargs={"exponent": 3.0, "transition_stress": 1e-3},
            solver_parameters_extra={"snes_rtol": 1e-8})

        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert solver.rheology_nonlinear
        assert solver.solver_parameters[prefix + "ksp_type"] == "gmres"
        assert solver.solver_parameters[prefix + "ksp_gmres_restart"] == 4

        solver.solve()
        assert solver.solver.snes.getConvergedReason() > 0
        newton = solver.solver.snes.getIterationNumber()
        assert newton >= 2
        assert solver.solver.snes.ksp.getConvergedReason() > 0

        context = condensation_context(solver)
        assert context.condensed_ksp.getType() == "gmres"
        assert context.condensed_ksp.getConvergedReason() > 0
        defect = symmetry_defect(context.S.petscmat)
        pressure = solver.real_field("core_pressure")
        volume = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        value = fd.assemble(pressure * fd.dx(domain=mesh)) / volume
        print(f"\n    [3-D fluid core, power law] Newton {newton}  "
              f"outer_ksp {solver.solver.snes.ksp.getIterationNumber()}  "
              f"ksp {context.condensed_ksp.getType()}  "
              f"asymmetry {defect:.3e}  core_pressure {value:.2e}")
        # Nine percent asymmetry: far outside the band where CG on this
        # operator has a defence. Measured 9.4e-2, gate 1e-3.
        assert defect > 1e-3
        assert np.isfinite(value)
