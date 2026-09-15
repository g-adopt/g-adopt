"""One elimination of the internal variables per block-0 application.

`gadopt.CondensedBlockPC` is the block-0 preconditioner of
`SelfGravitatingGIASolver` on the full layout (mixed space `(u, M, psi,
Real...)`). It does the whole block-0 solve itself: it eliminates the combined
internal-variable field `M` once with Slate (cell-local), runs a Krylov solve
on the assembled `(u, psi)` condensed system

    [[S_uu, A_upsi], [A_psiu, A_psipsi]] (u, psi) = (r_u', r_psi)

with `S_uu = A_uu - A_uM A_MM^-1 A_Mu` and
`r_u' = r_u - A_uM A_MM^-1 r_M`, and back-substitutes `M` once. The route it
replaces, `gadopt.InternalVariableSCPC` inside a two-split sweep, carries `M`
in every block-0 Krylov vector (about 85 percent of the block-0 unknowns) and
eliminates it on every block-0 inner iteration, about 900 times per production
time step.

Everything here is preconditioner-side: the residual, the mixed space, the
tape and the adjoint are the same on both routes. So the governing test is that
the answer is the direct preset's answer, and the rest of the file is about
the machinery that must be exact for that to hold (the elimination, the
back-substitution, the assembled blocks, the near-nullspace) and about the
options that select the route.

The interface these tests read, which the implementation must provide:

- `gadopt.CondensedBlockPC`, selected by `pc_type python` on block 0 with
  `dtn_fieldsplit_0_ksp_type preonly`.
- On the instance: `cxt` (the block-0 `ImplicitMatrixContext`, as
  `InternalVariableSCPC` has), `condensed_ksp` (the `(u, psi)` Krylov solve,
  whose operator is the nest of the four blocks and whose PC is the
  multiplicative `u`/`psi` fieldsplit), the four assembled blocks `S_uu`,
  `A_upsi`, `A_psiu` and `A_psipsi`, the reuse counter `assembly_count` and
  the elimination counter `elimination_count`.
- Options under the block-0 prefix: `condensed_*` for the `(u, psi)` solve,
  `condensed_fieldsplit_0_*` for the displacement split and
  `condensed_fieldsplit_1_*` for the potential split.
- `selfgrav_dtn_iterative_solver_parameters(condensed=False,
  block0="condensed"|"pair")`, `"condensed"` being the default.

The 2-D cases reuse the annulus of `test_gia_gravity.py` and the helpers of
`test_gia_nested_condensation.py`; the 3-D case is the 24-cell cubed sphere of
that file's `TestThreeDimensions`. `dt` is a `Constant` throughout, because the
operator-reuse fingerprint reads its value and a plain number cannot change.
"""

import firedrake as fd
import numpy as np
import pytest
from firedrake.petsc import PETSc

import gadopt
from gadopt.gia_gravity import (
    FluidCore,
    SelfGravitatingGIASolver,
    selfgrav_dtn_iterative_solver_parameters,
    selfgrav_dtn_schur_solver_parameters,
    self_gravitating_gia_space,
)
from gadopt.internal_variable_equation import history_slices

from test_gia_gravity import (  # noqa: E402  (module-level fixtures and helpers)
    B_MU,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    SIGMA_HAT,
    approximation,
    gravity_bcs,
    mechanics_bcs,
    meshes,  # noqa: F401  (pytest fixture)
)
from test_gia_nested_condensation import (  # noqa: E402
    build,
    condensation_context,
    nested_preset,
    relative_difference,
)
# Imported under private names so that pytest does not collect these classes a
# second time in this module; only their static helpers are used.
from test_gia_nested_condensation import TestPowerLaw as _PowerLawSettings
from test_gia_nested_condensation import TestThreeDimensions as _ThreeD

#: The block-0 prefix of the preset, and the prefixes the new route nests
#: under it. Written out once: attaching an option at the wrong depth is a
#: silent no-op rather than an error, so the prefix is part of every assertion
#: that names a key.
BLOCK0 = "dtn_fieldsplit_0_"
CONDENSED = BLOCK0 + "condensed_"
U_SPLIT = CONDENSED + "fieldsplit_0_"
PSI_SPLIT = CONDENSED + "fieldsplit_1_"


def condensed_preset(**kwargs):
    """The uncondensed iterative preset on the new block-0 route.

    The same tightening as `test_gia_nested_condensation.nested_preset`, so
    that the two routes are compared at one set of tolerances and a count
    difference between them is a property of the preconditioner and not of two
    different stopping tests. `block0` is left at its default on purpose: these
    tests are also the statement that the default is the new class.

    Args:
      **kwargs: overrides passed to
        `selfgrav_dtn_iterative_solver_parameters`.

    Returns:
      The solver-parameter dictionary.
    """
    settings = dict(condensed=False, snes_type="ksponly",
                    outer_rtol=1e-10, block0_rtol=1e-4, block0_max_it=200)
    settings.update(kwargs)
    return selfgrav_dtn_iterative_solver_parameters(**settings)


def block0_ksp(solver):
    """The block-0 KSP of the outer two-block Schur preconditioner.

    Outer PC (`DtNTwoBlockSchurPC`) -> its Schur fieldsplit -> block-0 KSP.
    On the new route this KSP is `preonly`, so one `solve` on it is exactly one
    application of `gadopt.CondensedBlockPC`.

    Args:
      solver: a solved `SelfGravitatingGIASolver`.

    Returns:
      The `PETSc.KSP` of block 0.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    ksp, _ = outer.pc.getFieldSplitSchurGetSubKSP()
    return ksp


def block0_context(solver):
    """The `gadopt.CondensedBlockPC` instance sitting on block 0.

    Args:
      solver: a solved `SelfGravitatingGIASolver` on `condensed_preset`.

    Returns:
      The preconditioner instance.
    """
    pc = block0_ksp(solver).getPC()
    # The type is checked before the context is asked for: `getPythonContext`
    # on a PC that is not of type python reads a pointer that is not a Python
    # object and takes the whole process down with a bus error, which would
    # hide every other failure in the run behind a crash.
    assert pc.getType() == "python", \
        f"block 0 runs pc_type {pc.getType()}, not a python preconditioner"
    context = pc.getPythonContext()
    assert isinstance(context, gadopt.CondensedBlockPC)
    return context


def assert_new_route(solver):
    """Fail unless block 0 of `solver` really ran `gadopt.CondensedBlockPC`.

    Every comparison against the direct route in this file would also pass on
    the `"pair"` route, because both routes solve the same system; they are
    comparisons of answers, not of preconditioners. This one line is what makes
    each of them a statement about the new class: it identifies the class
    through the PETSc objects the solve actually built, not through the
    options dictionary that asked for it.
    """
    block0_context(solver)


def displacement_ksp(context):
    """Split 0 of the `(u, psi)` fieldsplit: the displacement Krylov solve."""
    return context.condensed_ksp.getPC().getFieldSplitSubKSP()[0]


def petsc_matrix(matrix):
    """The `PETSc.Mat` behind an assembled block.

    Accepts either a bare `PETSc.Mat` or a Firedrake `Matrix`, because which
    of the two the implementation keeps is an implementation choice and no
    test here depends on it.
    """
    return getattr(matrix, "petscmat", matrix)


def function_space_of(W, field):
    """A standalone copy of one field of a mixed space.

    The assembled sub-blocks act on the individual displacement and potential
    spaces, so a test that multiplies by them needs `Function`s that are not
    sub-functions of the mixed space. `W.mesh()[field]` is used rather than
    `W[field].mesh()` because the self-gravity space spans two meshes: `psi`
    lives on the parent and `u` on the mantle submesh.

    Args:
      W: the block-0 mixed function space `(u, M, psi)`.
      field: the field index.

    Returns:
      The standalone `FunctionSpace`.
    """
    return fd.FunctionSpace(W.mesh()[field], W[field].ufl_element())


def random_function(V, seed):
    """A reproducible random field on `V`, for an operator-action comparison.

    The entries are `N(0, 1)` on the owned degrees of freedom. The values carry
    no physical meaning: these comparisons are linear-algebra identities that
    hold for any vector, and a random one exercises every mode of the operator
    instead of the smooth ones a physical load excites.
    """
    f = fd.Function(V)
    rng = np.random.default_rng(seed)
    f.dat.data_wo[...] = rng.normal(size=f.dat.data_ro.shape)
    return f


def relative_vector_gap(a, b):
    """`|a - b| / |b|` for two `PETSc.Vec`s of the same layout."""
    difference = a.copy()
    difference.axpy(-1.0, b)
    return difference.norm() / b.norm()


def matrix_free_action(solver, u=None, m=None, psi=None):
    """The block-0 operator applied to a `(u, M, psi)` state, matrix-free.

    The block-0 operator handed to the preconditioner is Firedrake's
    `ImplicitMatrixContext` of the block-0 sub-matrix, i.e. the exact Jacobian
    rows and columns of `(u, M, psi)` with no preconditioning in it. It is the
    reference every assembled block in this file is checked against.

    Args:
      solver: a solved `SelfGravitatingGIASolver` on `condensed_preset`.
      u, m, psi: the fields to place in the input state; `None` means zero.

    Returns:
      A `Function` on the block-0 mixed space holding the action.
    """
    A = block0_operator(solver)
    W = block0_context(solver).cxt.a.arguments()[0].function_space()
    state, action = fd.Function(W), fd.Function(W)
    for field, value in ((0, u), (1, m), (2, psi)):
        if value is not None:
            state.subfunctions[field].dat.data_wo[...] = value.dat.data_ro
    with state.dat.vec_ro as x, action.dat.vec_wo as y:
        A.mult(x, y)
    return action


def block0_operator(solver):
    """The matrix-free block-0 operator: the `(u, M, psi)` Jacobian rows.

    This is the `A` of the block-0 KSP, Firedrake's `ImplicitMatrixContext` of
    the block-0 sub-matrix. It carries no preconditioning, so it is the
    reference every assembled block in this file is compared against.
    """
    return block0_ksp(solver).getOperators()[0]


class TestThePresetSelectsTheRoute:
    """Which class the preset puts on block 0, and under which keys.

    Dictionary-only: no mesh, no solve. These are the statements a driver reads
    when it reproduces a run, and the keys the Gadi log counters key on.
    """

    def test_the_default_uncondensed_preset_puts_the_class_on_block_zero(self):
        """`preonly` is half the point: the class does the whole block-0 solve.

        Left at `fgmres`, block 0 would still carry `M` in every Krylov vector
        and the elimination would run once per inner iteration again, which is
        the cost this route exists to remove. A `preonly` KSP around the class
        means exactly one elimination and one back-substitution per block-0
        application.
        """
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        assert p[BLOCK0 + "pc_type"] == "python"
        assert p[BLOCK0 + "pc_python_type"] == "gadopt.CondensedBlockPC"
        assert p[BLOCK0 + "ksp_type"] == "preonly"

    def test_the_default_preset_names_no_block_zero_fieldsplit_fields(self):
        """There is no block-0 fieldsplit on this route, so no `_fields` keys.

        `SelfGravitatingGIASolver._check_block0_split_matches_layout` counts
        the fields named across those keys to catch a preset that disagrees
        with the space. A leftover `_fields` key here would make that check
        compare a stale claim against the space.
        """
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        named = [k for k in p
                 if k.startswith(BLOCK0 + "pc_fieldsplit_")
                 and k.endswith("_fields")]
        assert named == []

    def test_block0_pair_restores_the_internal_variable_scpc(self):
        """The `"pair"` arm, kept so the T2 gate's measurements reproduce.

        It is the route measured at 472 s per 500 yr step on 96 ranks, and a
        Gadi job that compares the two arms selects it by this flag alone.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, block0="pair")
        assert p[BLOCK0 + "pc_fieldsplit_0_fields"] == "0,1"
        assert p[BLOCK0 + "pc_fieldsplit_1_fields"] == "2"
        assert p[BLOCK0 + "fieldsplit_0_pc_python_type"] \
            == "gadopt.InternalVariableSCPC"
        assert p[BLOCK0 + "ksp_type"] == "fgmres"

    def test_the_condensed_layout_refuses_a_block0_choice(self):
        """On the condensed layout there is no `M` to eliminate.

        The argument has no meaning there, and silently ignoring it would let
        a driver believe it selected an arm it did not select.
        """
        with pytest.raises(ValueError, match="block0"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=True, block0="pair")

    def test_the_inner_krylov_solve_carries_the_block_zero_tolerances(self):
        """`block0_rtol` and `block0_max_it` are the `(u, psi)` solve's now.

        On the `"pair"` route they were the block-0 FGMRES's. Block 0 is
        `preonly` here, so the same two numbers must land on the Krylov solve
        that actually does the work; dropping them would leave the `(u, psi)`
        solve at PETSc's default 1e-5 and 10000 iterations.
        """
        p = condensed_preset(block0_rtol=1e-7, block0_max_it=123)
        assert p[CONDENSED + "ksp_type"] == "fgmres"
        assert p[CONDENSED + "ksp_rtol"] == 1e-7
        assert p[CONDENSED + "ksp_max_it"] == 123
        assert CONDENSED + "ksp_converged_reason" in p

    def test_the_default_route_drops_the_block_zero_converged_reason(self):
        """One reason line per block-0 application, not two.

        `bench_dtn_baseline.parse_counts` counts a block-0 application from
        every reason line whose prefix ends in `dtn_fieldsplit_0_`, and on
        this route the line that reports the work is the inner `(u, psi)`
        solve's at `dtn_fieldsplit_0_condensed_`. Block 0 itself is `preonly`,
        and a `preonly` KSP asked for `ksp_converged_reason` still prints one
        (`... converged due to CONVERGED_ITS iterations 0`) - the preset
        already relies on that for the `preonly` splits of the condensed
        layout. So keeping `dtn_fieldsplit_0_ksp_converged_reason` on this
        route puts a second line in every Gadi log for the same application
        and `block0_applies` reads twice the truth, which is the number the B2
        cost model is built on. The key must go, and the inner solve's key
        (the assertion above) takes over its job.
        """
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        assert BLOCK0 + "ksp_converged_reason" not in p

    def test_the_other_routes_keep_the_block_zero_converged_reason(self):
        """A guard, not a change: this key holds today and must keep holding.

        On the condensed layout and on the `"pair"` arm block 0 is a real
        FGMRES, its reason line is the one that counts the application, and
        removing it with the new route's key would leave both of those routes
        reporting zero block-0 applications in every Gadi log. The key lives
        in the dictionary both layouts share, so the removal the test above
        asks for is easy to make one line too wide.
        """
        for parameters in (
            selfgrav_dtn_iterative_solver_parameters(condensed=True),
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, block0="pair"),
        ):
            assert BLOCK0 + "ksp_converged_reason" in parameters

    def test_the_inner_solve_sweeps_u_and_psi_multiplicatively(self):
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        assert p[CONDENSED + "pc_type"] == "fieldsplit"
        assert p[CONDENSED + "pc_fieldsplit_type"] == "multiplicative"

    def test_the_displacement_split_is_the_truncated_cg_with_gamg(self):
        """The same short CG both layouts run, at the new prefix.

        Four iterations at 1e-2 with `ksp_converged_maxits`: the split is a
        preconditioner inside the flexible `(u, psi)` FGMRES, so it is
        truncated on purpose and PETSc must count the truncation as a
        convergence, otherwise every Gadi log line from it reads
        `DIVERGED_ITS`.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, u_ksp_max_it=4, u_ksp_rtol=1e-2)
        assert p[U_SPLIT + "ksp_type"] == "cg"
        assert p[U_SPLIT + "ksp_max_it"] == 4
        assert p[U_SPLIT + "ksp_rtol"] == 1e-2
        assert U_SPLIT + "ksp_converged_maxits" in p
        assert p[U_SPLIT + "pc_type"] == "gamg"
        # The Gadi counter reads the multigrid sweeps of this split from its
        # `ksp_converged_reason` lines (PLAN section 2 line 108): one line per
        # application of the split. Without the key PETSc prints nothing, the
        # split still runs, every unit test here stays green, and
        # `mg_sweeps["split_0"]` reads 0 in every 96-rank log, which is the
        # measurement this task exists to make. The option table of the plan
        # (line 78) does not list the key, so it has to be pinned here.
        assert U_SPLIT + "ksp_converged_reason" in p

    def test_the_potential_split_is_one_gamg_v_cycle(self):
        """The potential block is a Laplacian; GAMG handles it in one sweep."""
        p = selfgrav_dtn_iterative_solver_parameters(condensed=False)
        assert p[PSI_SPLIT + "ksp_type"] == "preonly"
        assert p[PSI_SPLIT + "pc_type"] == "gamg"
        # Same reason as on the displacement split above: the sweep counter in
        # the 3-D drivers has nothing to count unless this split prints a
        # converged-reason line per application, and the plan's option table
        # for `condensed_fieldsplit_1_*` (line 79) omits the key.
        assert PSI_SPLIT + "ksp_converged_reason" in p

    def test_one_v_cycle_is_selectable_on_the_displacement_split(self):
        """`u_ksp_max_it=0` is the route the P3 march ran.

        A cap left behind on a `preonly` KSP is inert and would read as a
        truncated solve to anyone auditing the dictionary.
        """
        p = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, u_ksp_max_it=0)
        assert p[U_SPLIT + "ksp_type"] == "preonly"
        assert U_SPLIT + "ksp_max_it" not in p


class TestTheLayoutGuards:
    """The new class must not reach a space that has no internal variables."""

    def test_the_class_refuses_a_two_field_block_zero(self):
        """The class's own guard, tested away from the solver.

        A two-field mixed space is exactly what
        `condense_internal_variables=True` produces, so the message must name
        that flag: a caller who reaches this error has set the space and the
        preset independently, and has to be told which of the two to change.
        The message must also name the space it was handed, because the same
        error is what a hand-written dictionary on some other three-field
        block would produce.

        The toy system is a unit square with a vector and a tensor field and a
        block-diagonal form. Nothing about it is a GIA problem: the guard fires
        on the field count of the block-0 bilinear form before any Slate
        expression, any sub-DM and any KSP is built, and building a whole
        coupled solver to reach one `raise` would cost a minute.

        `initialize` is called directly on a `PETSc.PC` that is never set up.
        Going through `PCSetUp` instead loses the message: PETSc catches the
        Python exception inside the callback and re-raises it as
        `PETSc.Error` code 101, so the text this test is about never reaches
        the caller. This also keeps the guard's placement honest -- it has to
        fire before the DM work that a bare `setUp` outside a solve cannot do.
        """
        mesh = fd.UnitSquareMesh(2, 2)
        V = fd.VectorFunctionSpace(mesh, "CG", 1)
        S = fd.TensorFunctionSpace(mesh, "DG", 0)
        W = V * S
        trials, tests = fd.TrialFunctions(W), fd.TestFunctions(W)
        a = (fd.inner(fd.grad(trials[0]), fd.grad(tests[0]))
             + fd.inner(trials[0], tests[0])
             + fd.inner(trials[1], tests[1])) * fd.dx
        A = fd.assemble(a, mat_type="matfree").petscmat

        pc = PETSc.PC().create(comm=mesh.comm)
        pc.setType("python")
        pc.setOperators(A, A)
        pc.setOptionsPrefix("two_field_block0_")
        with pytest.raises(ValueError, match="condense_internal_variables"):
            gadopt.CondensedBlockPC().initialize(pc)


def build_condensed_layout_solver(meshes, solver_parameters):
    """A condensed-layout solver, built to be refused.

    The space is built with `condense_internal_variables=True`, which the
    uncondensed preset's block 0 must not be applied to.
    """
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
        n_internal_variables=1, condense_internal_variables=True,
        self_gravity_number=LAMBDA)
    z = fd.Function(Z)
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    C = fd.assemble(fd.dot(Xm, Xm) * dx_m)
    return SelfGravitatingGIASolver(
        z, approximation(), layout=layout, dt=fd.Constant(1.0),
        bcs=mechanics_bcs(sub), rotation_moments={"C": C},
        solver_parameters=solver_parameters)


def build_with_nullspaces(meshes, solver_parameters):
    """An annulus solver that declares the kernel and the transpose kernel.

    `test_gia_gravity.build(declare_nullspace=True)` declares `nullspace`
    alone, and the basis has to be built on the mixed space before the solver
    exists, so the space is built here instead of going through that helper.
    The same `MixedVectorSpaceBasis` serves as both the kernel and the left
    kernel: for this system they coincide in the continuum (the rigid rotation
    is annihilated from both sides), and what the test reads is whether the
    declared left kernel reaches the assembled displacement block at all.

    Args:
      meshes: the `(parent, sub)` fixture of `test_gia_gravity`.
      solver_parameters: the preset to run.

    Returns:
      The solved `SelfGravitatingGIASolver`.
    """
    from gadopt.gia_gravity import rigid_rotation_nullspace

    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
        n_internal_variables=1, self_gravity_number=LAMBDA)
    z = fd.Function(Z)
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    # The polar moment of inertia the rotational closure needs; the value is
    # the annulus's own, so the closure constants are the physical ones.
    C = fd.assemble(fd.dot(Xm, Xm) * dx_m)
    basis = rigid_rotation_nullspace(Z, layout)
    solver = SelfGravitatingGIASolver(
        z, approximation(), layout=layout, dt=fd.Constant(1.0),
        bcs=mechanics_bcs(sub), rotation_moments={"C": C},
        nullspace=basis, transpose_nullspace=basis,
        solver_parameters=solver_parameters)
    solver.solve()
    return solver


def test_the_condensed_layout_refuses_the_new_block_zero(meshes):
    """`_check_block0_split_matches_layout` must catch this preset too.

    The `"pair"` route announces its field count through
    `dtn_fieldsplit_0_pc_fieldsplit_N_fields`, and the existing check compares
    that count with the space's. The new route names no such key, so the check
    has to recognise the class by name instead. Without that, an uncondensed
    preset on a condensed space runs a preconditioner that looks for a history
    field the space does not have.
    """
    with pytest.raises(ValueError, match="CondensedBlockPC"):
        build_condensed_layout_solver(meshes, condensed_preset())


class TestTheSolveMatchesTheDirectRoute:
    """The governing test: one discrete system, two routes to it."""

    @pytest.fixture(scope="class")
    def direct(self, meshes):
        solver, z, layout = build(
            meshes, solver_parameters="direct",
            solver_parameters_extra={"snes_type": "ksponly"})
        solver.solve()
        return z.copy(deepcopy=True), layout

    @pytest.fixture(scope="class")
    def condensed(self, meshes):
        solver, z, layout = build(meshes, solver_parameters=condensed_preset())
        solver.solve()
        return solver, z, layout

    @pytest.fixture(scope="class")
    def pair(self, meshes):
        solver, z, layout = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        return solver, z, layout

    def test_the_class_sits_on_block_zero(self, condensed):
        """The route under test is the one the preset selected.

        Every comparison below is worthless if the solve silently ran the
        `"pair"` route, so the class is identified through the PETSc objects
        and not through the options dictionary.
        """
        solver, _, _ = condensed
        context = block0_context(solver)
        W = context.cxt.a.arguments()[0].function_space()
        assert len(W) == 3
        assert W[1].value_shape == (1, 2, 2)
        assert context.cxt.appctx is solver.appctx

    def test_every_krylov_level_converged(self, condensed):
        """Outer, inner `(u, psi)` and both splits, on their own exit status.

        Block 0 is a preconditioner application inside a flexible outer Krylov
        solve, so a degraded inner solve costs outer iterations rather than
        accuracy and shows up nowhere else.
        """
        solver, _, _ = condensed
        assert solver.solver.snes.ksp.getConvergedReason() > 0
        assert block0_context(solver).condensed_ksp.getConvergedReason() > 0

    def test_it_matches_the_direct_solve(self, condensed, direct):
        """The preconditioner changes no residual, so the answer is the same.

        The margin is set by the preset's Krylov tolerances, `outer_rtol`
        1e-10 and `block0_rtol` 1e-4. On the `"pair"` route the measured gaps
        on this annulus are 7.1e-10 in the displacement, 1.8e-11 in the
        potential and 1.1e-11 in the internal variable, and the same system
        solved a different way cannot do better than its own outer tolerance.
        The gate at 1e-8 is an order and a half above the largest of them, so
        a formulation difference fails it and Krylov noise does not.
        """
        solver, z, layout = condensed
        z_direct, _ = direct
        assert_new_route(solver)
        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z, z_direct))
        psi, psi_direct = (f.subfunctions[layout.potential]
                           for f in (z, z_direct))
        m, = history_slices(z.subfunctions[layout.internal_variable_field])
        m_direct, = history_slices(
            z_direct.subfunctions[layout.internal_variable_field])
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-8
        assert relative_difference(psi, psi_direct) < 1e-8
        assert relative_difference(m, m_direct) < 1e-8

    def test_the_real_fields_match_the_direct_solve(self, condensed, direct):
        """The DtN multipliers and the rotational closure, one number each.

        These are the `Real` unknowns of block 1. They are compared separately
        because they are global scalars: an L2 norm over the mesh cannot see
        them, and every one of them sits on a single rank, which is what makes
        them the fragile part of a parallel run.

        Absolute gaps against the largest coefficient of the direct solve, so
        that a multiplier whose true value is roundoff is not compared
        relatively against roundoff. Gate 1e-8, the outer tolerance's scale.
        """
        solver, z, layout = condensed
        z_direct, _ = direct
        assert_new_route(solver)
        values = np.array([float(z.subfunctions[i].dat.data_ro[0])
                           for i in layout.real_fields])
        reference = np.array([float(z_direct.subfunctions[i].dat.data_ro[0])
                              for i in layout.real_fields])
        scale = max(abs(reference).max(), 1e-300)
        assert len(values) > 0
        assert abs(values - reference).max() / scale < 1e-8

    def test_the_outer_count_is_no_worse_than_the_pair_route(
            self, condensed, pair):
        """Eliminating `M` once must not cost outer iterations.

        Both routes precondition the same outer FGMRES with the same
        tolerances, and the new one applies an exact cell-local elimination
        where the old one applied it repeatedly inside its own Krylov solve.
        The preconditioner is not identical, so the counts need not be equal;
        what would condemn the route is needing MORE outer iterations, because
        the whole saving is per-iteration cost.
        """
        condensed_solver, _, _ = condensed
        pair_solver, _, _ = pair
        new = condensed_solver.solver.snes.ksp.getIterationNumber()
        old = pair_solver.solver.snes.ksp.getIterationNumber()
        print(f"\n    [outer FGMRES] condensed {new}  pair {old}")
        assert new > 0
        assert new <= old


class TestOneBlockZeroApplication:
    """What one application of the class does, measured on its own.

    Block 0 is `preonly`, so one `solve` on the block-0 KSP is one application
    of the preconditioner and nothing else. That makes the elimination, the
    inner solve and the back-substitution measurable without going through the
    outer Krylov solve at all.
    """

    @pytest.fixture(scope="class")
    def exact(self, meshes):
        """A solve whose inner `(u, psi)` solve is tightened to 1e-12.

        At that tolerance the application is an (almost) exact solve of the
        block-0 system, so the residual it leaves measures the elimination and
        the back-substitution rather than the Krylov truncation. The cap is
        raised with it: a solve that stops at its cap would leave a residual
        set by the cap.
        """
        solver, z, layout = build(
            meshes,
            solver_parameters=condensed_preset(block0_rtol=1e-12,
                                               block0_max_it=400))
        solver.solve()
        return solver

    def test_one_application_solves_the_block_zero_system(self, exact):
        """The check `PLAN-COUPLED-TRANSITION.md` 13.2 item 6 asked for.

        A random right-hand side goes in, the application comes out, and the
        residual is measured against the matrix-free block-0 operator itself -
        the exact `(u, M, psi)` Jacobian rows, with no preconditioning in it.
        The `M` rows of that residual are the ones that catch a wrong
        elimination or a wrong back-substitution: they are solved cell-locally
        and exactly, so any error there is visible well above the Krylov floor
        of the `u` and `psi` rows.

        Gate 1e-8 relative against an inner tolerance of 1e-12: four orders of
        head-room for the GMRES stopping test, which measures the
        preconditioned residual and not this one.
        """
        ksp = block0_ksp(exact)
        assert ksp.getType() == "preonly"
        A = block0_operator(exact)
        x, b = A.createVecs()
        b.setRandom()
        ksp.solve(b, x)
        residual = b.duplicate()
        A.mult(x, residual)
        residual.axpy(-1.0, b)
        gap = residual.norm() / b.norm()
        print(f"\n    [one application] relative residual {gap:.3e}")
        assert gap < 1e-8

    def test_the_elimination_runs_once_per_application(self, exact):
        """Once per `apply`, not once per inner iteration.

        This is the whole point of the route: on the `"pair"` arm the
        elimination and the back-substitution run on every block-0 inner
        iteration, about 900 times per production step. Three applications are
        driven by hand so the count is compared with a number this test knows
        exactly, instead of with a count inferred from the outer solve.
        """
        context = block0_context(exact)
        ksp = block0_ksp(exact)
        A = block0_operator(exact)
        x, b = A.createVecs()
        before = context.elimination_count
        for seed in range(3):
            b.setRandom()
            ksp.solve(b, x)
        assert context.elimination_count - before == 3

    def test_the_transpose_application_is_refused(self, exact):
        """`applyTranspose` must raise, because it is not implemented.

        Firedrake's `PCBase.applyTranspose` is abstract with a `pass` body, so
        a subclass that leaves it alone - or defines it as `pass` - is a
        perfectly instantiable class whose transpose application writes
        nothing into the output vector. PETSc then hands the caller whatever
        was already in that memory and the solve around it converges to a
        wrong answer with no error anywhere. Nothing in the quick set drives
        this path by accident (`tests/unit/test_gia_gravity_adjoint.py` builds
        every case with `solver_parameters="direct"`), so the only thing that
        can catch a silent no-op is an explicit call.

        The behaviour the plan asks for is `NotImplementedError`: pyadjoint
        solves `adjoint(J)` with the forward options, so the forward `apply`
        is what an adjoint solve of this system reaches, and any caller that
        does arrive at `applyTranspose` has reached a path this class does not
        support and must be told so rather than handed a silent no-op.

        `x` and `y` are only there to fill the signature; the refusal has to
        come before either is read.
        """
        context = block0_context(exact)
        pc = block0_ksp(exact).getPC()
        x, y = block0_operator(exact).createVecs()
        with pytest.raises(NotImplementedError):
            context.applyTranspose(pc, x, y)

    def test_the_operator_is_rebuilt_only_when_dt_changes(self, meshes):
        """A Newtonian march at fixed `dt` has one condensed operator.

        The four assembled blocks and the GAMG hierarchies on `S_uu` and
        `A_psipsi` can serve the whole march, and rebuilding them per solve is
        the cost this reuse rule removes. The counter starts at 1 (the build
        inside `initialize`), stays at 1 across repeated solves, and goes to 2
        exactly when `dt` moves, because `dt` is in the Jacobian fingerprint.
        """
        solver, _, _ = build(meshes, solver_parameters=condensed_preset())
        solver.solve()
        context = block0_context(solver)
        assert context.assembly_count == 1
        solver.solve()
        assert context.assembly_count == 1
        solver.dt.assign(0.5)
        solver.solve()
        assert context.assembly_count == 2
        solver.solve()
        assert context.assembly_count == 2


class TestTheAssembledBlocks:
    """The four sparse blocks are the block-0 operator, minus `M`.

    `S_uu` is the exact Schur complement of the cell-local `M` block, and the
    other three are sub-blocks of the block-0 bilinear form. If any of them is
    assembled from the wrong form, the solve still converges - to the wrong
    preconditioner's fixed point, i.e. more slowly - which is why they are
    compared against the matrix-free operator directly.
    """

    @pytest.fixture(scope="class")
    def solved(self, meshes):
        solver, z, layout = build(meshes, solver_parameters=condensed_preset())
        solver.solve()
        return solver

    @pytest.fixture(scope="class")
    def pair_solved(self, meshes):
        solver, z, layout = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        return solver

    def test_the_displacement_block_is_the_condensation_class_matrix(
            self, solved, pair_solved):
        """`S_uu` here must be the matrix `InternalVariableSCPC` assembles.

        Both are `A_uu - A_uM A_MM^-1 A_Mu` on the same problem, written from
        the same Slate expression (the plan asks for one helper both classes
        call). Entrywise, therefore, and not merely spectrally: the Frobenius
        gate is 1e-12 relative, which is roundoff-scale for an assembly that
        walks the same cells in the same order, and any difference of form -
        a missing cross-coupling, a different sign - is orders above it.
        """
        S_new = petsc_matrix(block0_context(solved).S_uu)
        S_pair = petsc_matrix(condensation_context(pair_solved).S)
        assert S_new.getSize() == S_pair.getSize()
        difference = S_new.copy()
        difference.axpy(-1.0, S_pair)
        gap = difference.norm() / S_pair.norm()
        print(f"\n    [S_uu] relative Frobenius gap {gap:.3e}")
        assert gap < 1e-12

    def test_the_potential_rows_reproduce_the_matrix_free_action(self, solved):
        """`A_psiu u + A_psipsi psi` against the operator's `psi` rows.

        The potential rows of the block-0 Jacobian carry no `M` at all - the
        history field enters the mechanics rows only - so with `M = 0` the
        assembled pair must reproduce those rows exactly, with no Schur
        correction anywhere.

        Gate 1e-12 relative. The scouted value on this annulus is 2e-16, so
        the gate is four orders above roundoff and still far below any
        difference a wrong block would make.
        """
        context = block0_context(solved)
        W = context.cxt.a.arguments()[0].function_space()
        u = random_function(function_space_of(W, 0), seed=11)
        psi = random_function(function_space_of(W, 2), seed=12)

        action = matrix_free_action(solved, u=u, psi=psi)
        assembled = fd.Function(function_space_of(W, 2))
        scratch = fd.Function(function_space_of(W, 2))
        with u.dat.vec_ro as uv, assembled.dat.vec_wo as out:
            petsc_matrix(context.A_psiu).mult(uv, out)
        with psi.dat.vec_ro as pv, scratch.dat.vec_wo as out:
            petsc_matrix(context.A_psipsi).mult(pv, out)
        with assembled.dat.vec as a, scratch.dat.vec_ro as s:
            a.axpy(1.0, s)

        reference = fd.Function(function_space_of(W, 2))
        reference.dat.data_wo[...] = action.subfunctions[2].dat.data_ro
        with assembled.dat.vec_ro as a, reference.dat.vec_ro as r:
            gap = relative_vector_gap(a, r)
        print(f"\n    [psi rows] relative gap {gap:.3e}")
        assert gap < 1e-12

    def test_the_displacement_potential_coupling_reproduces_the_action(
            self, solved):
        """`A_upsi psi` against the operator's `u` rows at `u = 0, M = 0`.

        With both `u` and `M` zero the mechanics rows of the block-0 operator
        are the coupling term alone, so this isolates `A_upsi` from the Schur
        complement `S_uu` that the previous test's route cannot separate.

        Same 1e-12 gate and the same reason as above.
        """
        context = block0_context(solved)
        W = context.cxt.a.arguments()[0].function_space()
        psi = random_function(function_space_of(W, 2), seed=13)

        action = matrix_free_action(solved, psi=psi)
        assembled = fd.Function(function_space_of(W, 0))
        with psi.dat.vec_ro as pv, assembled.dat.vec_wo as out:
            petsc_matrix(context.A_upsi).mult(pv, out)

        reference = fd.Function(function_space_of(W, 0))
        reference.dat.data_wo[...] = action.subfunctions[0].dat.data_ro
        with assembled.dat.vec_ro as a, reference.dat.vec_ro as r:
            gap = relative_vector_gap(a, r)
        print(f"\n    [u-psi coupling] relative gap {gap:.3e}")
        assert gap < 1e-12

    def test_the_near_incompressible_modes_reach_the_displacement_block(
            self, solved):
        """GAMG on `S_uu` must see more than the rigid modes.

        `condensed_near_nullspace` defaults to `"incompressible"`, so
        `gadopt.near_nullspace_basis` builds the rigid-body modes and the
        low-degree divergence-free fields. In 2-D the rigid set is 3 vectors
        (two translations and one rotation) and the degree-1 divergence-free
        space adds 2 more after the drop-tolerant orthogonalisation, so 5 is
        what separates the default from the rigid set. Counted on the matrix,
        because a basis that never reaches `MatSetNearNullSpace` is exactly
        the failure this line of work exists to catch: on Gadi the modes are
        worth 33 699 GAMG V-cycles against 16 853 for a 500 yr step.
        """
        near = petsc_matrix(block0_context(solved).S_uu).getNearNullSpace()
        assert len(near.getVecs()) == 5

    @pytest.fixture(scope="class")
    def with_nullspaces(self, meshes):
        """One solve that declares both the kernel and the transpose kernel.

        Two of the four application-context keys the class must read are
        published only when the caller declares the matching basis on the
        solver, so they need a solve of their own. Both are declared here and
        one solve serves both tests, because the annulus solve is the dear
        part and the two claims are read off the same assembled `S_uu`.
        """
        return build_with_nullspaces(meshes, condensed_preset())

    def test_a_declared_nullspace_reaches_the_displacement_block(
            self, with_nullspaces):
        """`condensed_field_nullspace` is the same appctx key both classes read.

        The annulus with `un = 0` at the CMB and traction at the surface has
        the rigid rotation as a genuine kernel, and the solver publishes its
        displacement part as a provider. Left unset on `S_uu`, GAMG and the
        Krylov solve on the displacement split work on a singular operator
        with an inconsistent right-hand side.
        """
        nullspace = petsc_matrix(
            block0_context(with_nullspaces).S_uu).getNullSpace()
        assert nullspace.handle != 0
        assert len(nullspace.getVecs()) == 1

    def test_a_declared_transpose_nullspace_reaches_the_displacement_block(
            self, with_nullspaces):
        """`condensed_field_transpose_nullspace`, the fourth appctx key.

        The solver publishes it beside `condensed_field_nullspace` whenever
        the caller passes `transpose_nullspace=`, and
        `gadopt.InternalVariableSCPC` sets it on its condensed matrix. The new
        class must do the same: PETSc projects the *left* kernel out of the
        right-hand side, and on a nonsymmetric condensed operator - every
        power law, and the Newtonian annulus to 3.6e-3 through the Nitsche
        terms - the left kernel is not the right one, so a matrix carrying
        only `setNullSpace` leaves an inconsistent right-hand side behind.

        The vector is checked and not only counted, because an empty or a
        wrong basis of size one would pass a count. The reference is the same
        rigid rotation `gadopt.rigid_rotation_nullspace` builds in 2-D,
        `u = (-y, x)`, interpolated into the same displacement space. Both are
        the same interpolant of the same expression and PETSc orthonormalises
        in the l2 vector norm, so the two differ by a positive scalar and
        nothing else: the cosine between them is 1 to roundoff and the gate
        1e-10 is several orders above that while still failing any other mode.
        """
        solver = with_nullspaces
        matrix = petsc_matrix(block0_context(solver).S_uu)
        transpose = matrix.getTransposeNullSpace()
        assert transpose.handle != 0
        vectors = transpose.getVecs()
        assert len(vectors) == 1

        # The expected mode, built on the displacement space of the block-0
        # form so that its degree-of-freedom layout is the matrix's.
        W = block0_context(solver).cxt.a.arguments()[0].function_space()
        V = function_space_of(W, 0)
        X = fd.SpatialCoordinate(V.mesh())
        expected = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))
        with expected.dat.vec_ro as e:
            cosine = abs(vectors[0].dot(e)) / (vectors[0].norm() * e.norm())
        print(f"\n    [transpose nullspace] cosine {cosine:.12f}")
        assert abs(cosine - 1.0) < 1e-10


class TestPowerLaw:
    """A Jacobian that moves inside one Newton solve.

    Exponent 3 at transition stress 1e-3 is the operating point the plan
    measured: Newton converges in a few iterations on both routes. The two
    facts that must hold are that the preconditioner follows the state without
    being told to, and that the answer is still the direct route's.
    """

    def test_newton_matches_the_direct_route(self, meshes):
        """Two routes, one Newton path.

        The nesting preconditions and changes no residual, so Newton on it
        takes the same number of steps and lands in the same place as Newton
        on an LU block 0. Tolerances as in the Newtonian comparison; the
        inner tolerance is tightened to 1e-6 because a power law's Newton step
        is only as good as the linear solve under it.
        """
        nested, z_nested, layout = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=condensed_preset(snes_type="newtonls",
                                               block0_rtol=1e-6),
            solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
        direct, z_direct, _ = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=dict(selfgrav_dtn_schur_solver_parameters,
                                   snes_type="newtonls",
                                   snes_linesearch_type="l2"),
            solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
        nested.solve()
        direct.solve()
        assert_new_route(nested)
        assert nested.solver.snes.getConvergedReason() > 0
        assert direct.solver.snes.getConvergedReason() > 0
        assert nested.solver.snes.getIterationNumber() \
            == direct.solver.snes.getIterationNumber() >= 2

        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z_nested, z_direct))
        psi, psi_direct = (f.subfunctions[layout.potential]
                           for f in (z_nested, z_direct))
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-6
        assert relative_difference(psi, psi_direct) < 1e-6

    def test_the_operator_is_rebuilt_at_every_newton_iteration(self, meshes):
        """`operator_version` is `None` for a power law, so nothing is reused.

        The fingerprint that lets a Newtonian march reuse its blocks across
        time steps cannot express "the operator moves inside the solve", so
        the solver publishes `None` and the class rebuilds on every update.
        `assembly_count` is the readable consequence: it must equal the number
        of Newton iterations, the first build being the one in `initialize`.
        """
        solver, _, _ = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=condensed_preset(snes_type="newtonls"),
            solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
        solver.solve()
        context = block0_context(solver)
        assert solver.appctx["operator_version"] is None
        assert context.assembly_count == solver.solver.snes.getIterationNumber()

    def test_the_displacement_split_switches_to_gmres(self, meshes):
        """A power law makes the condensed operator nonsymmetric, so no CG.

        The preset writes a short CG on the displacement split, which is a
        valid preconditioner inside the flexible `(u, psi)` FGMRES only while
        `S_uu` is symmetric. `_attach_condensation_context` must make that
        switch at the new prefix as well; reading only the `"pair"` route's
        key leaves CG on a nonsymmetric operator, which does not raise and
        converges to the wrong thing or not at all.

        The restart is the split's own cap: a longer restart would allocate
        Krylov vectors the truncated solve never reaches.
        """
        solver, _, _ = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=condensed_preset(snes_type="newtonls"),
            solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
        assert not solver.condensed_operator_symmetric()
        p = solver.solver_parameters
        assert p[U_SPLIT + "ksp_type"] == "gmres"
        assert p[U_SPLIT + "ksp_gmres_restart"] == p[U_SPLIT + "ksp_max_it"] == 4

    def test_a_caller_named_krylov_type_wins(self, meshes):
        """The one way to keep a method the symmetry rule would replace."""
        key = U_SPLIT + "ksp_type"
        solver, _, _ = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=condensed_preset(snes_type="newtonls"),
            solver_parameters_extra={key: "fgmres",
                                     **dict(_PowerLawSettings.NEWTON)})
        assert solver.solver_parameters[key] == "fgmres"
        # The dictionary is one statement and what PETSc built is another: an
        # option attached at a prefix nothing reads is a silent no-op.
        solver.solve()
        assert displacement_ksp(block0_context(solver)).getType() == "fgmres"

    def test_the_displacement_split_actually_runs_the_named_method(
            self, meshes):
        """The switch is only worth something if it reaches the KSP.

        The options dictionary is one statement; what PETSc built is another,
        and a key attached at the wrong depth is a silent no-op.
        """
        solver, _, _ = build(
            meshes, approximation_kwargs=_PowerLawSettings.POWER_LAW,
            solver_parameters=condensed_preset(snes_type="newtonls"),
            solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
        solver.solve()
        assert displacement_ksp(block0_context(solver)).getType() == "gmres"


class TestOtherConfigurations:
    """The configurations the production runs actually use."""

    def test_a_fluid_core_matches_the_direct_route(self, meshes):
        """The core-pressure `Real` row sits inside the block-1 saddle.

        The fluid core replaces the legacy `un = 0` at the CMB with an
        eliminated inviscid core, and its uniform-pressure multiplier has a
        physical zero on the diagonal. Nothing about it is special to block 0,
        and that is the claim: the same two routes that agree without a core
        agree with one, at the same Newton counts. The dense multiplier Schur
        complement is named because `pc_type none` is unavailable on a block
        with that zero row.

        Tolerances follow the measured `"pair"` comparison on this system:
        4e-9 in the displacement, which does not improve when the inner
        tolerance is tightened (it moves to 1.7e-8), so 1e-6 is the gate for
        `u` and 1e-8 for `M` and `psi`, which agree at 1e-15 there.
        """
        parent, sub = meshes
        Xm = fd.SpatialCoordinate(sub)

        def make(solver_parameters):
            Z, layout = self_gravitating_gia_space(
                sub, parent, gravity_bcs=gravity_bcs(parent), rotation=False,
                fluid_core=True, n_internal_variables=1,
                condense_internal_variables=False, self_gravity_number=LAMBDA)
            z = fd.Function(Z)
            solver = SelfGravitatingGIASolver(
                z, approximation(**_PowerLawSettings.POWER_LAW), layout=layout,
                dt=fd.Constant(1.0),
                bcs={CURVE_RE: {"normal_stress": B_MU * SIGMA_HAT * fd.cos(
                    2 * fd.atan2(Xm[1], Xm[0]))}},
                fluid_core=FluidCore(boundary=CURVE_RC, rho_core=2.0),
                solver_parameters=solver_parameters,
                solver_parameters_extra=dict(_PowerLawSettings.NEWTON))
            return solver, z, layout

        nested, z_nested, layout = make(condensed_preset(
            snes_type="newtonls", block0_rtol=1e-6,
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"))
        direct, z_direct, _ = make(
            dict(selfgrav_dtn_schur_solver_parameters, snes_type="newtonls",
                 snes_linesearch_type="l2"))
        nested.solve()
        direct.solve()
        assert_new_route(nested)
        assert nested.solver.snes.getConvergedReason() > 0
        assert nested.solver.snes.getIterationNumber() \
            == direct.solver.snes.getIterationNumber() >= 2
        assert layout.core_pressure is not None

        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z_nested, z_direct))
        psi, psi_direct = (f.subfunctions[layout.potential]
                           for f in (z_nested, z_direct))
        m, = history_slices(
            z_nested.subfunctions[layout.internal_variable_field])
        m_direct, = history_slices(
            z_direct.subfunctions[layout.internal_variable_field])
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-6
        assert relative_difference(m, m_direct) < 1e-8
        assert relative_difference(psi, psi_direct) < 1e-8

    def test_two_maxwell_elements_match_the_direct_route(self, meshes):
        """`n = 2`: the combined `(n, d, d)` history field, not one `(d, d)`.

        The elimination is of the whole combined field whatever `n` is, so the
        number of Maxwell elements must not appear anywhere in the class. Two
        elements with different viscosities and shear moduli is the cheapest
        configuration that would catch an `n = 1` assumption.
        """
        common = dict(
            n_internal_variables=2,
            approximation_kwargs={"viscosity": [1.0, 3.0],
                                  "shear_modulus": [1.0, 0.5]})
        nested, z_nested, layout = build(
            meshes, solver_parameters=condensed_preset(), **common)
        direct, z_direct, _ = build(
            meshes, solver_parameters="direct",
            solver_parameters_extra={"snes_type": "ksponly"}, **common)
        nested.solve()
        direct.solve()

        context = block0_context(nested)
        W = context.cxt.a.arguments()[0].function_space()
        assert W[1].value_shape == (2, 2, 2)
        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z_nested, z_direct))
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-8
        for m, m_direct in zip(
                history_slices(
                    z_nested.subfunctions[layout.internal_variable_field]),
                history_slices(
                    z_direct.subfunctions[layout.internal_variable_field])):
            assert relative_difference(m, m_direct) < 1e-8

    def test_a_strong_displacement_condition_is_handled_or_refused(
            self, meshes):
        """A `DirichletBC` on `u` must not be silently dropped.

        Production 3-D and every other case in this file use weak boundaries
        (`un`, `normal_stress`), so this configuration is not on the critical
        path. What it must never do is apply the condition to the residual and
        not to the assembled blocks the preconditioner solves on: the solve
        then either converges slowly for no visible reason or, with a zero
        row left unconstrained in `S_uu`, not at all.

        Either behaviour is acceptable and the design chooses which: carry the
        condition into the blocks and match the direct route (gate 1e-8, the
        outer tolerance's scale; the `"pair"` route measures 9.8e-13 on this
        problem), or refuse it with a message that names it. What fails this
        test is a solve that neither raises nor matches.
        """
        parent, sub = meshes
        Xm = fd.SpatialCoordinate(sub)
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))

        def make(solver_parameters, extra=None):
            Z, layout = self_gravitating_gia_space(
                sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
                n_internal_variables=1, condense_internal_variables=False,
                self_gravity_number=LAMBDA)
            z = fd.Function(Z)
            C = fd.assemble(fd.dot(Xm, Xm) * dx_m)
            solver = SelfGravitatingGIASolver(
                z, approximation(), layout=layout, dt=fd.Constant(1.0),
                bcs={CURVE_RC: {"u": fd.as_vector([0.0, 0.0])},
                     CURVE_RE: {"normal_stress": B_MU * SIGMA_HAT * fd.cos(
                         2 * fd.atan2(Xm[1], Xm[0]))}},
                rotation_moments={"C": C},
                solver_parameters=solver_parameters,
                solver_parameters_extra=extra)
            return solver, z, layout

        try:
            nested, z_nested, layout = make(condensed_preset())
            nested.solve()
        except Exception as error:  # noqa: BLE001 (the refusal is the outcome)
            message = str(error).lower()
            assert any(word in message for word in
                       ("strong", "dirichlet", "boundary condition")), \
                f"a refusal must name the condition it refuses: {error}"
            return

        assert_new_route(nested)
        direct, z_direct, _ = make("direct", {"snes_type": "ksponly"})
        direct.solve()
        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z_nested, z_direct))
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-8


class TestThreeDimensions:
    """The production geometry, on 24 cells.

    A 24-cell cubed sphere proves that the construction works in 3-D, on
    extruded hexahedra, with a fluid core and with the `Real` rows of the
    rotational closure present. It proves nothing about whether the route is
    faster: every 3-D performance number comes from Gadi.
    """

    @pytest.fixture(scope="class")
    def solved(self):
        Z, layout, mesh, C = _ThreeD.fluid_core_space()
        z_nested, z_direct = fd.Function(Z), fd.Function(Z)
        nested = _ThreeD.fluid_core_solver(
            z_nested, layout, mesh, C,
            condensed_preset(block0_rtol=1e-6,
                             multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"))
        direct = _ThreeD.fluid_core_solver(
            z_direct, layout, mesh, C,
            dict(selfgrav_dtn_schur_solver_parameters, snes_type="ksponly"))
        nested.solve()
        direct.solve()
        return nested, z_nested, z_direct, layout

    def test_it_matches_the_direct_route(self, solved):
        """Newtonian with a fluid core, 3-D, against LU on block 0.

        Measured on the `"pair"` route on this sphere: 3.6e-11 in `u`, 1.5e-12
        in `m`, 1.9e-12 in `psi`. The gate at 1e-8 is three orders above the
        largest of them.
        """
        nested, z_nested, z_direct, layout = solved
        assert_new_route(nested)
        assert nested.solver.snes.getConvergedReason() > 0
        assert nested.solver.snes.ksp.getConvergedReason() > 0
        u, u_direct = (f.subfunctions[layout.displacement]
                       for f in (z_nested, z_direct))
        psi, psi_direct = (f.subfunctions[layout.potential]
                           for f in (z_nested, z_direct))
        m, = history_slices(
            z_nested.subfunctions[layout.internal_variable_field])
        m_direct, = history_slices(
            z_direct.subfunctions[layout.internal_variable_field])
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-8
        assert relative_difference(m, m_direct) < 1e-8
        assert relative_difference(psi, psi_direct) < 1e-8

    def test_the_displacement_split_runs_the_truncated_cg(self, solved):
        """Newtonian in 3-D: `S_uu` is symmetric, so CG is the right method.

        The 2-D annulus is sent to GMRES by the power-law rule, so this is the
        only place in the file where the symmetric branch is exercised through
        a real 3-D solve. The test also pins which block split 0 of the
        `(u, psi)` fieldsplit acts on, because "the truncated CG" is a claim
        about the displacement block and nothing else in the file says that
        split 0 is the displacement one.
        """
        nested = solved[0]
        context = block0_context(nested)
        ksp = displacement_ksp(context)
        # Split 0 of the `(u, psi)` fieldsplit must carry the displacement
        # unknowns. Nothing else in this file pins the order of the nest: the
        # near-nullspace tests read `S_uu` directly, and every `ksp_type` and
        # cap assertion here reads `getFieldSplitSubKSP()[0]` whatever sits
        # there. A nest built as `(psi, u)` would therefore pass all of them
        # while applying the truncated CG with the near-incompressible modes
        # to the potential Laplacian and one V-cycle to the displacement
        # block: no answer changes, nothing raises, and the only symptom is a
        # slow 96-rank job. On this sphere the two blocks have different sizes
        # (displacement 1470 rows, potential 490, measured on
        # `_ThreeD.fluid_core_space`), so the size is enough to tell them
        # apart and the second assertion makes sure the first one cannot pass
        # by the two blocks happening to have equal size.
        assert (ksp.getOperators()[0].getSize()
                == petsc_matrix(context.S_uu).getSize())
        assert (petsc_matrix(context.A_psipsi).getSize()
                != petsc_matrix(context.S_uu).getSize())
        assert ksp.getType() == "cg"
        # `getTolerances` is (rtol, atol, divtol, max_it); the cap is the
        # preset's `u_ksp_max_it`, and a split that ran to convergence here
        # would be the oversolve the truncation exists to avoid.
        assert ksp.getTolerances()[3] == 4

    def test_the_history_field_is_three_dimensional(self, solved):
        """One combined `(n, d, d)` field with `d = 3`, eliminated as one."""
        nested = solved[0]
        W = block0_context(nested).cxt.a.arguments()[0].function_space()
        assert W[1].value_shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# Two ranks. Module-level function, not a method: mpi-pytest cannot relaunch a
# test that lives inside a class.
#
# This exists because the class is going to 96 ranks and every `Real` dof - the
# DtN multipliers, the rotational closure, the core pressure - sits on exactly
# one rank. The serial suite structurally cannot see a rank-local decision made
# inside a collective: the elimination is cell-local and rank-local, the
# assembled blocks are distributed, and the `(u, psi)` nest's index sets come
# from the nest rather than from the mixed space's own field ordering.
# ---------------------------------------------------------------------------
@pytest.mark.parallel(nprocs=2)
def test_two_ranks_match_the_direct_route(meshes):  # noqa: F811
    """The 2-rank annulus solve against LU on block 0, on the same 2 ranks.

    The reference is the direct preset on the same communicator rather than a
    serial answer, because a serial answer is not reachable from inside an
    mpi-pytest run and because the comparison that matters is "the same system,
    preconditioned differently" on one decomposition.

    Gate 1e-8 in every field, the preset's outer tolerance's scale, the same
    gate as the serial comparison.
    """
    assert fd.COMM_WORLD.size == 2, "this test must run on exactly 2 ranks"
    nested, z_nested, layout = build(
        meshes, solver_parameters=condensed_preset())
    direct, z_direct, _ = build(
        meshes, solver_parameters="direct",
        solver_parameters_extra={"snes_type": "ksponly"})
    nested.solve()
    direct.solve()
    assert_new_route(nested)
    assert nested.solver.snes.ksp.getConvergedReason() > 0

    u, u_direct = (f.subfunctions[layout.displacement]
                   for f in (z_nested, z_direct))
    psi, psi_direct = (f.subfunctions[layout.potential]
                       for f in (z_nested, z_direct))
    m, = history_slices(z_nested.subfunctions[layout.internal_variable_field])
    m_direct, = history_slices(
        z_direct.subfunctions[layout.internal_variable_field])
    assert fd.norm(u_direct) > 0.0
    assert relative_difference(u, u_direct) < 1e-8
    assert relative_difference(psi, psi_direct) < 1e-8
    assert relative_difference(m, m_direct) < 1e-8


class TestTheGadiLogCounters:
    """The 3-D drivers count block-0 applications from the reason lines.

    `bench_dtn_baseline.parse_counts` classifies every
    `ksp_converged_reason` line by its options prefix, and the B2 cost model is
    read from its output and from nowhere else. On the new route block 0 is
    `preonly`, so the line that counts one block-0 application is the inner
    `(u, psi)` solve's, at `dtn_fieldsplit_0_condensed_`, and the multigrid
    sweeps are at `dtn_fieldsplit_0_condensed_fieldsplit_N_`. Left unchanged,
    the counter reports zero block-0 applications and files every sweep under
    `unclassified`, which reads as a configuration that never ran.

    The lines below are PETSc's own format, copied from a run of the `"pair"`
    route with the prefixes rewritten.
    """

    @staticmethod
    def parse(text):
        """`parse_counts` of the 3-D driver, imported on demand.

        The driver hides the command line from PETSc at import by replacing
        `sys.argv`, so it is saved and restored around the import.
        """
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        driver = root / "demos" / "glacial_isostatic_adjustment" \
            / "3d_spada_selfgrav"
        saved_argv, saved_path = list(sys.argv), list(sys.path)
        sys.path.insert(0, str(driver))
        try:
            import bench_dtn_baseline
        finally:
            sys.argv[:] = saved_argv
            sys.path[:] = saved_path
        return bench_dtn_baseline.parse_counts(text)

    #: One outer solve, two block-0 applications, two sweeps of each split.
    #:
    #: There is deliberately no bare `..._dtn_fieldsplit_0_ ` line here. The
    #: preset must not ask the `preonly` block-0 KSP for its converged reason
    #: on this route, because that line would be a second one for the same
    #: application and `parse_counts` would count it
    #: (`prefix.endswith("dtn_fieldsplit_0_")`), so `block0_applies` would read
    #: twice the truth. That requirement is pinned on the dictionary by
    #: `TestThePresetSelectsTheRoute
    #: ::test_the_default_route_drops_the_block_zero_converged_reason`; this
    #: log is what the counter sees once it holds.
    NEW_ROUTE_LOG = """\
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_fieldsplit_0_ solve converged due to CONVERGED_ITS iterations 4
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_fieldsplit_1_ solve converged due to CONVERGED_ITS iterations 1
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_ solve converged due to CONVERGED_RTOL iterations 9
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_fieldsplit_0_ solve converged due to CONVERGED_ITS iterations 4
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_fieldsplit_1_ solve converged due to CONVERGED_ITS iterations 1
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_condensed_ solve did not converge due to DIVERGED_ITS iterations 200
  Linear SelfGravitatingGIA_dtn_fieldsplit_1_ solve converged due to CONVERGED_RTOL iterations 3
  Linear SelfGravitatingGIA_ solve converged due to CONVERGED_RTOL iterations 5
"""

    #: The `"pair"` route's lines, which must keep classifying as they do now.
    OLD_ROUTE_LOG = """\
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_fieldsplit_0_condensed_field_ solve converged due to CONVERGED_RTOL iterations 2
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_fieldsplit_1_ solve converged due to CONVERGED_ITS iterations 1
  Linear SelfGravitatingGIA_dtn_fieldsplit_0_ solve converged due to CONVERGED_RTOL iterations 13
  Linear SelfGravitatingGIA_dtn_fieldsplit_1_ solve converged due to CONVERGED_RTOL iterations 3
  Linear SelfGravitatingGIA_ solve converged due to CONVERGED_RTOL iterations 5
"""

    def test_the_inner_solve_counts_as_a_block_zero_application(self):
        """Two `dtn_fieldsplit_0_condensed_` lines are two applications.

        The iteration total and the capped count come with it: a solve that
        hit its cap reports `DIVERGED_ITS` and still did the work, and dropping
        those lines undercounts block 0 exactly in the runs where block 0 is
        dearest.
        """
        counts = self.parse(self.NEW_ROUTE_LOG)
        assert counts["block0_applies"] == 2
        assert counts["block0_its"] == 209
        assert counts["block0_diverged"] == 1

    def test_the_inner_splits_count_as_multigrid_sweeps(self):
        """The two splits of the `(u, psi)` solve, counted separately.

        The order of the prefix tests inside `parse_counts` decides this: a
        split's prefix CONTAINS the inner solve's, so testing for the inner
        solve first would count every sweep as a block-0 application and
        inflate the headline cost by the sweep count.
        """
        counts = self.parse(self.NEW_ROUTE_LOG)
        assert counts["mg_sweeps"] == {"split_0": 2, "split_1": 2}

    def test_nothing_from_the_new_route_is_unclassified(self):
        counts = self.parse(self.NEW_ROUTE_LOG)
        assert counts["unclassified"] == []
        assert counts["outer"] == 5
        assert counts["block1_applies"] == 1

    def test_the_pair_route_still_counts_as_it_does_today(self):
        """A guard: the `"pair"` arm's numbers must not move.

        The plan says to add the new patterns and not to remove the old ones,
        and the reason is that the T2 gate's `"pair"` measurements have to stay
        reproducible: `bench_dtn_baseline` reduces `mg_sweeps` to one total
        (`mg_sweeps_total`) that the B2 cost model reads, so any new pattern
        that also matches an old line silently rewrites a published number for
        a route this task does not touch.

        The values below are what `parse_counts` returns for this text today,
        run against the driver as it stands and not derived from the regexes
        by reading. They include one line the parser does not classify: the
        `"pair"` arm's displacement solve prints
        `..._dtn_fieldsplit_0_fieldsplit_0_condensed_field_`, and the sweep
        pattern anchors on `dtn_fieldsplit_0_fieldsplit_(\\d+)_$`, so that line
        ends up in `unclassified` and only the `psi` split is counted as a
        sweep. That is today's behaviour and this test pins it; whether it is
        the behaviour anyone wants is a separate question from this task, and
        changing it here would change the `"pair"` arm's `mg_sweeps_total`.
        """
        counts = self.parse(self.OLD_ROUTE_LOG)
        assert counts["block0_applies"] == 1
        assert counts["block0_its"] == 13
        assert counts["mg_sweeps"] == {"split_1": 1}
        assert counts["unclassified"] == [
            "SelfGravitatingGIA_dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        ]
