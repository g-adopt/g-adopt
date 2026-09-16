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
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_gia_gravity_adjoint_lowrank as tl  # noqa: E402
from gadopt.approximations import (  # noqa: E402
    CompressibleInternalVariableApproximation)
from gadopt import SphericalDtN  # noqa: E402
from gadopt.gia_gravity import (  # noqa: E402
    FluidCore, SelfGravitatingGIASolver, self_gravitating_gia_space,
    selfgrav_dtn_iterative_solver_parameters)
from gadopt.internal_variable_equation import history_slices  # noqa: E402
from gadopt.preconditioners import (  # noqa: E402
    LowRankPotentialPC, _LowRankPotentialOperator)


# The outer tolerance is tight so that the iteration counts compared below are
# counts of the same work and not of two different stopping points.
OUTER_RTOL = 1e-10
BLOCK0_RTOL = 1e-4


@pytest.fixture(scope="module")
def meshes():
    """The annulus pair of the low-rank suite, built once for this module."""
    return tl.meshes.__wrapped__()


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
          rotation=True, solver_parameters=None, dt=1.0):
    """One coupled solver on the annulus, in either representation.

    Mirrors `NOTES/measurements/lowrank-iterative/lowrank_block0.py`, which is
    where the measurements in the finding come from, so that a count asserted
    here is the count recorded there.
    """
    parent, sub = meshes
    Xp = fd.SpatialCoordinate(parent)
    Xm = fd.SpatialCoordinate(sub)
    gravity_bcs = {
        tl.CURVE_OUTER: {"dtn": tl.CylindricalDtN(truncation)},
        tl.CURVE_INNER: {"dtn": tl.CylindricalDtN(truncation)},
        tl.CURVE_RE: {"interior_sigma":
                      tl.SIGMA_HAT * fd.cos(2 * fd.atan2(Xp[1], Xp[0]))},
    }
    surface_load = fd.Constant(tl.B_MU) * tl.SIGMA_HAT * (
        fd.cos(2 * fd.atan2(Xm[1], Xm[0]))
        + (fd.Constant(0.25) if fluid_core else fd.Constant(0.0)))
    mechanics_bcs = {tl.CURVE_RE: {"normal_stress": surface_load}}
    if not fluid_core:
        mechanics_bcs[tl.CURVE_RC] = {"un": 0.0}

    lam = fd.Constant(tl.LAMBDA)
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        fluid_core=fluid_core, n_internal_variables=1,
        condense_internal_variables=False, self_gravity_number=lam,
        dtn_representation=representation)
    z = fd.Function(Z)
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=fd.Constant(1.0),
        shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
        g=fd.Constant(tl.G0), B_mu=fd.Constant(tl.B_MU),
        self_gravity_number=lam)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    if solver_parameters is None:
        solver_parameters = preset(representation)
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=fd.Constant(dt), bcs=mechanics_bcs,
        rotation_moments={"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)},
        fluid_core=(FluidCore(boundary=tl.CURVE_RC, rho_core=2.0)
                    if fluid_core else None),
        solver_parameters=solver_parameters,
        dtn_representation=representation)
    return solver, z, layout


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
    reference, z_ref, layout = build(meshes, "multiplier",
                                     truncation=truncation)
    reference.solve()
    multiplier_its = outer_iterations(reference)

    solver, z, _ = build(meshes, "lowrank", truncation=truncation)
    solver.solve()

    assert outer_iterations(solver) <= multiplier_its + 1

    # The same state, to the outer tolerance. This is the check that the
    # iteration count above was bought with the right operator and not by
    # preconditioning a different problem.
    for field in (layout.displacement, layout.potential):
        reference_norm = fd.norm(z_ref.subfunctions[field])
        assert abs(fd.norm(z.subfunctions[field]) - reference_norm) < (
            1e-8 * reference_norm)


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


# ---------------------------------------------------------------------------
# 4. What the configuration refuses
# ---------------------------------------------------------------------------
def test_the_preset_refuses_the_nested_block_zero_on_the_low_rank_path():
    """`block0="pair"` has nowhere to put the update."""
    with pytest.raises(ValueError, match="no low-rank route"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, block0="pair", dtn_representation="lowrank")


def test_the_preset_refuses_the_condensed_layout_on_the_low_rank_path():
    """The condensed layout's block 0 has no potential split to put it in."""
    with pytest.raises(ValueError, match="needs the full layout"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=True, dtn_representation="lowrank")


def test_the_preset_refuses_an_unknown_representation():
    with pytest.raises(ValueError, match="must be 'multiplier' or 'lowrank'"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, dtn_representation="low_rank")


def test_a_low_rank_preset_on_a_multiplier_solver_is_refused(meshes):
    """This direction raises late and unhelpfully, so it is caught early.

    Without the check the failure is inside `PCSetUp`, reported as an
    unhandled Python exception under a PETSc error code.
    """
    with pytest.raises(ValueError, match="carries no such update"):
        build(meshes, "multiplier", solver_parameters=preset("lowrank"))


def test_a_multiplier_preset_on_a_low_rank_solver_is_refused(meshes):
    """This direction also raises late: plain GAMG refuses the Python block.

    `CondensedBlockPC.initialize` installs the Python potential block whenever
    `dtn_operator` is in the context, and `PCSetUp_GAMG` then fails with `No
    method getinfo for Mat of type python`, an unhandled exception under a
    PETSc error code. The early check gives a message that names the fix.
    """
    with pytest.raises(ValueError, match="plain GAMG"):
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
@pytest.mark.parametrize("representation", ["multiplier", "lowrank"])
def test_the_gradient_is_right_through_this_preset(meshes, representation):
    """Gradient and Taylor rate, with the iterative preset in place of LU.

    The bars are the low-rank suite's own: the adjoint gradient within 1e-6
    relative of a finite difference of fresh solves, and a Taylor rate of at
    least 1.99.
    """
    import test_gia_gravity_adjoint_lowrank as suite
    from pyadjoint import taylor_test

    iterative = preset(representation, outer_rtol=1e-11)
    original = suite.SelfGravitatingGIASolver

    class Iterative(original):
        """The suite builds with `solver_parameters="direct"`; swap in ours."""

        def __init__(self, *args, **kwargs):
            if kwargs.get("solver_parameters") == "direct":
                kwargs["solver_parameters"] = iterative
            super().__init__(*args, **kwargs)

    suite.SelfGravitatingGIASolver = Iterative
    try:
        value = suite.CONTROL_VALUES["shear_modulus"]
        _, _, relative = suite.gradient_against_fresh_solves(
            meshes, "shear_modulus", value, representation, fluid_core=True)
        assert relative < 1e-6

        functional, control, direction, _, _ = suite.reduced_functional(
            meshes, "shear_modulus", value, representation, fluid_core=True)
        assert taylor_test(functional, control, direction) > 1.99
    finally:
        suite.SelfGravitatingGIASolver = original


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
    def space(mesh, representation):
        """The mixed space of `test_gia_nested_condensation.TestThreeDimensions.fluid_core_space`.

        Rebuilt here with the representation as an argument and the mesh
        passed in, so that both representations sit on one mesh object.

        Args:
          mesh: the extruded cubed sphere.
          representation: `"multiplier"` or `"lowrank"`.

        Returns:
          `(Z, layout)`.
        """
        return self_gravitating_gia_space(
            mesh, mesh,
            gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                         "bottom": {"dtn": SphericalDtN(1)}},
            rotation=True, fluid_core=True, condense_internal_variables=False,
            n_internal_variables=1, self_gravity_number=tl.LAMBDA,
            dtn_representation=representation)

    @staticmethod
    def solver(z, layout, mesh, C, representation):
        """A Newtonian fluid-core solver on that space, under the iterative preset.

        The CMB of a radially extruded sphere is its bottom surface, so the
        fluid core is on `"bottom"`, and there is no `un` there because the
        constructor refuses both on one boundary.

        Args:
          z: the mixed solution `Function` to solve into.
          layout, mesh, C: the space's layout, the mesh and the polar moment.
          representation: `"multiplier"` or `"lowrank"`.

        Returns:
          The `SelfGravitatingGIASolver`.
        """
        X = fd.SpatialCoordinate(mesh)
        approx = CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=fd.Constant(1.0),
            shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
            g=fd.Constant(tl.G0), B_mu=fd.Constant(tl.B_MU),
            self_gravity_number=fd.Constant(tl.LAMBDA))
        return SelfGravitatingGIASolver(
            z, approx, layout=layout, dt=fd.Constant(1.0),
            bcs={"top": {"normal_stress": tl.SIGMA_HAT * X[2]}},
            rotation_moments={"C": C, "C_minus_A": 0.1 * C},
            fluid_core=FluidCore(boundary="bottom", rho_core=2.0),
            solver_parameters=preset(representation),
            dtn_representation=representation)

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
            Z, layout = self.space(mesh, representation)
            z = fd.Function(Z)
            solver = self.solver(z, layout, mesh, C, representation)
            solver.solve()
            out[representation] = (solver, z, layout)
        return out

    def test_it_matches_the_multiplier_representation(self, solved):
        """The same state on the sphere, to the outer tolerance.

        The annulus agreement between representations is 1e-12 relative
        (`NOTES/FINDING-LOWRANK-ITERATIVE-2026-09-16.md` section 6); the bar
        of 1e-8 is the one the multiplier 3-D test uses against its direct
        route, three to four orders above what is measured.
        """
        reference, z_ref, layout_ref = solved["multiplier"]
        solver, z, layout = solved["lowrank"]
        for s in (reference, solver):
            assert s.solver.snes.getConvergedReason() > 0
            assert s.solver.snes.ksp.getConvergedReason() > 0
        for field_ref, field in ((layout_ref.displacement, layout.displacement),
                                 (layout_ref.potential, layout.potential)):
            ref = fd.norm(z_ref.subfunctions[field_ref])
            assert ref > 0.0
            assert fd.norm(z.subfunctions[field]
                           - z_ref.subfunctions[field_ref]) < 1e-8 * ref
        m_ref, = history_slices(
            z_ref.subfunctions[layout_ref.internal_variable_field])
        m, = history_slices(z.subfunctions[layout.internal_variable_field])
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
    import test_gia_gravity_adjoint_lowrank as suite

    value = suite.CONTROL_VALUES["shear_modulus"]
    functional, _, _, _, solver = suite.reduced_functional(
        meshes, "shear_modulus", value, "lowrank", fluid_core=True)
    adjoint_solver = solver.adjoint_block._ad_solvers["adjoint_lvs"]
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
