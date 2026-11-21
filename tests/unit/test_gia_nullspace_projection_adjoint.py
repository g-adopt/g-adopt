"""The kernel projection after a coupled GIA solve is on the tape.

`gadopt.gia_solver.SelfGravitatingGIASolver.solve` removes the declared
rigid-rotation kernel (`nullspace="rigid_rotation"`) from the solution after
the solve. The removal goes through the PETSc vector of the solution, so the
solver records it with `gadopt.nullspaces.project_out_kernel` as a
`KernelProjectionBlock`. If it were not recorded, a replay of the tape would
skip it, and every recorded operation that reads the solution afterwards (the
history update, the copy into `u` and `psi`, the next solve, the functional)
would see a value that the replay does not reproduce.

What each test checks:

- The replay of a small mixed problem without a solver, where the projection
  removes a large component that depends on the control, at the recorded
  control and at a shifted control against a fresh forward run: the
  `projection-np2` entry of the `parallel` case in `tests/gia`
  (`gia_parallel.py`, which imports `_small_case`, `_small_taped`,
  `_small_forward` and `_shifted` from here).
- `test_small_derivatives_to_second_order`: on the same problem, the Taylor
  remainders converge at rates 1, 2 and 3 (gradient and Hessian action), and
  the tangent linear model equals the adjoint directional derivative. Fails if
  the adjoint, the Hessian or the tangent of the block is wrong.
- `test_no_block_without_basis_or_annotation`,
  `test_non_orthonormal_basis_is_refused`: the bookkeeping and the refusal of
  a basis that would make the projection non-symmetric.
- `test_gia_projection_is_recorded`: the coupled solver on the 2-D annulus
  (`test_gia_lowrank_block0.build`: low-rank DtN, DG internal variables, a
  fluid core, the rigid rotation declared as nullspace and transpose
  nullspace, three time steps, a spatially varying viscosity as control). Two functionals: a weighted integral of the
  tangential surface displacement, which a rigid rotation changes, and the
  strain energy, which it does not. Checks replay against the recording and
  against a fresh run, the guarded Taylor test, and the tangent linear model
  against the adjoint. The replay check fails if the projection is not
  recorded: with the `direct` preset the solve leaves a rotation of about
  5e-7 of the norm of `u`, and a start with a rotation in `u` (as after a
  restart) makes it of order one. The tangent check fails if the tangent of
  the block is wrong.
- `test_gia_block_adjoint_on_kernel_seed`: the adjoint of the block on the
  coupled mixed space, on a seed with a large kernel component. With the
  low-rank DtN representation, a wrong adjoint of the block cannot change a
  gradient: every adjoint value that reaches the block goes on to the
  adjoint of the solve, and that adjoint solve removes the forward nullspace
  from its right-hand side (its transpose nullspace is the forward
  nullspace, whether or not a forward transpose nullspace is declared). The
  multiplier representation keeps the forward slots in its adjoint solve
  (see `SelfGravitatingGIASolver.adjoint_solve_keywords`), so there the
  block adjoint is what removes the kernel when a solve declares only a
  forward nullspace. The coupled replay tests therefore cannot see the block
  adjoint on the low-rank path, and this test applies the
  block's adjoint directly and checks it against its tangent with a
  dot-product test.

The replay and gradient checks of the small problem on two ranks, where the
products with the basis are global, are the entry `projection-np2` of the
`parallel` case in `tests/gia`. `_small_case`, `_small_forward`,
`_small_taped` and `_shifted` are in
`tests/gia/gia_models/projection_adjoint.py`, which that driver and this
module share.
"""

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
    taylor_to_dict,
)

from gadopt import KernelProjectionBlock, project_out_kernel

# The annulus, its constants, the coupled solver builder and the tape
# fixture. Importing the fixtures makes pytest use them here.
from test_gia_gravity import meshes  # noqa: F401
from gia_models.annulus import CURVE_RE  # noqa: F401
from test_gia_lowrank_block0 import (  # noqa: F401
    build,
    clean_tape,
)
from test_gravity_adjoint import assert_taylor_with_guards
from gia_models.projection_adjoint import (
    _small_case,
    _small_taped,
)

#: Number of coupled time steps. Three steps give two steps in which the
#: replayed initial guess and `solution_old` come from a recorded projection.
N_STEPS = 3

#: Tolerance for "replay equals the recorded value" and "replay equals a fresh
#: forward run". The replay solves the same systems with the same solver, so
#: the agreement is at round-off. A missing projection gives 2e-6 or more.
REPLAY_RTOL = 1e-10

#: The G-ADOPT threshold for the Taylor rate of the first-order remainder.
MIN_RATE = 1.98


@pytest.mark.longtest
def test_small_derivatives_to_second_order():
    """Adjoint, tangent linear model and Hessian of the recorded projection."""
    Z, nullspace, control, direction = _small_case()
    J, rf = _small_taped(Z, nullspace, control)

    # The guarded first-order test: the gradient is not trivially small, the
    # functional moves, rate 2 with the gradient and rate 1 without it.
    assert_taylor_with_guards(rf, control, direction, J, min_rate=MIN_RATE)

    # Rates 1, 2, 3 of the zeroth, first and second-order remainders. The
    # third is the Hessian action, which goes through the second-order
    # adjoint of the block.
    rates = taylor_to_dict(rf, control, direction)
    print(f"    [small] R0 {rates['R0']['Rate']}  R1 {rates['R1']['Rate']}  "
          f"R2 {rates['R2']['Rate']}")
    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > MIN_RATE
    assert min(rates["R2"]["Rate"]) > 2.95

    # Tangent linear model against the adjoint directional derivative.
    tape = get_working_tape()
    rf(control)
    adjoint = direction._ad_dot(rf.derivative())
    tape.reset_tlm_values()
    control.block_variable.tlm_value = direction
    tape.evaluate_tlm()
    tlm = float(J.block_variable.tlm_value)
    print(f"    [small] tlm {tlm:.15e}  adjoint {adjoint:.15e}")
    assert abs(tlm - adjoint) <= 1e-12 * abs(adjoint)


def test_no_block_without_basis_or_annotation():
    """Nothing is recorded when nothing is removed or annotation is off."""
    Z, nullspace, _, _ = _small_case()
    tape = get_working_tape()
    tape.clear_tape()
    z = fd.Function(Z)
    continue_annotation()
    try:
        assert project_out_kernel(None, z) is False
        assert project_out_kernel(
            fd.MixedVectorSpaceBasis(Z, [Z.sub(0), Z.sub(1)]), z) is False
        with stop_annotating():
            assert project_out_kernel(nullspace, z) is True
    finally:
        pause_annotation()
    assert not tape.get_blocks()


def test_non_orthonormal_basis_is_refused():
    """A basis that is not orthonormal gives a non-symmetric P: refuse it.

    Without annotation too, because the forward projection is wrong with
    such a basis as well.
    """
    mesh = fd.UnitSquareMesh(4, 4)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    X = fd.SpatialCoordinate(mesh)
    basis = fd.VectorSpaceBasis(
        [fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))])
    z = fd.Function(V)
    with pytest.raises(ValueError, match="orthonormal"):
        project_out_kernel(basis, z)


# ---------------------------------------------------------------------------
# Part 2: the coupled GIA solver
# ---------------------------------------------------------------------------
def _functionals(solver, sub):
    """The tangential surface functional and the strain energy.

    `J_tan` weights the tangential displacement on the surface with
    `1 + sin(2 theta)`. The constant part integrates a rigid rotation to
    `2 pi r^2` times its amplitude, so the functional sees the rotation
    content of `u`. The `sin(2 theta)` part sees the tangential response to
    the degree-2 load, so the functional is not zero without a rotation.

    `J_str` is the integral of the squared symmetric gradient. A rigid
    rotation is linear in the coordinates and the coordinate field of the
    mesh is P2, so its symmetric gradient is zero up to round-off and
    `J_str` does not see the projection.

    Args:
      solver: the solver; the functionals read its mixed solution.
      sub: the mantle mesh.

    Returns:
      A dict with the two functionals.
    """
    u = solver.field("displacement")
    X = fd.SpatialCoordinate(sub)
    n = fd.FacetNormal(sub)
    tangent = fd.as_vector([-n[1], n[0]])
    weight = 1 + fd.sin(2 * fd.atan2(X[1], X[0]))
    J_tan = fd.assemble(fd.dot(u, tangent) * weight
                        * fd.ds(CURVE_RE, domain=sub))
    strain = fd.sym(fd.grad(u))
    J_str = fd.assemble(fd.inner(strain, strain) * fd.dx(domain=sub))
    return {"tangential": J_tan, "strain": J_str}


def _gia_forward(meshes, viscosity, seed, preset, n_steps=N_STEPS):  # noqa: F811
    """Build the solver without annotation and march `n_steps` steps.

    Annotation of the march and of the functionals follows the caller.

    Args:
      meshes: the parent and mantle meshes.
      viscosity: the control, a CG1 Function on the mantle submesh.
      seed: amplitude of a rigid rotation put into the initial
        displacement, relative to 1e-3 (the size of the response to the
        load). Zero for a start from rest.
      preset: the solver preset name, `"direct"`.
      n_steps: the number of time steps.

    Returns:
      `(functionals, solver)`.
    """
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    with stop_annotating():
        solver, _, _ = build(
            meshes, "lowrank", fluid_core=True, rotation=False,
            approximation_kwargs={"viscosity": viscosity},
            nullspace="rigid_rotation",
            solver_parameters=preset,
            initial_displacement=(seed * 1e-3 * fd.as_vector([-X[1], X[0]])
                                  if seed else None))
    for _ in range(n_steps):
        solver.solve()
    return _functionals(solver, sub), solver


def _control_and_direction(sub):
    """The viscosity control and the Taylor direction on the mantle."""
    Q = fd.FunctionSpace(sub, "CG", 1)
    X = fd.SpatialCoordinate(sub)
    theta = fd.atan2(X[1], X[0])
    with stop_annotating():
        viscosity = fd.Function(Q).interpolate(1.0 + 0.2 * X[0])
        direction = fd.Function(Q).interpolate(0.3 * fd.cos(3 * theta) + 0.1)
    return viscosity, direction


@pytest.mark.longtest
@pytest.mark.parametrize("preset,seed", [
    ("direct", 0.0), ("direct", 1.0)],
    ids=["direct-from_rest", "direct-rotated_start"])
def test_gia_projection_is_recorded(meshes, preset, seed):  # noqa: F811
    """Replay, fresh forward run, Taylor test and tangent for both functionals."""
    _, sub = meshes
    viscosity, direction = _control_and_direction(sub)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        control = Control(viscosity)
        recorded, solver = _gia_forward(meshes, viscosity, seed, preset)
    finally:
        pause_annotation()

    # One recorded projection per step.
    projections = [b for b in tape.get_blocks()
                   if isinstance(b, KernelProjectionBlock)]
    assert len(projections) == N_STEPS

    with stop_annotating():
        fresh, _ = _gia_forward(meshes, viscosity.copy(deepcopy=True),
                                seed, preset)

    for name, J in recorded.items():
        rf = ReducedFunctional(J, control)
        replay = float(rf(viscosity))
        print(f"    [{preset}, {name}, seed {seed}] recorded {float(J):.15e}  "
              f"replay {replay:.15e}  fresh {float(fresh[name]):.15e}")
        assert abs(replay - float(J)) <= REPLAY_RTOL * abs(float(J))
        assert abs(float(fresh[name]) - float(J)) <= REPLAY_RTOL * abs(float(J))
        assert_taylor_with_guards(rf, viscosity, direction, J,
                                  min_rate=MIN_RATE)

    # The tangent linear model of the march, which applies the tangent of the
    # projection block at every step, against the adjoint derivative of the
    # tangential functional. The adjoint side does not depend on the block
    # (see the module docstring), so a wrong tangent of the block shows here.
    rf = ReducedFunctional(recorded["tangential"], control)
    rf(viscosity)
    adjoint = direction._ad_dot(rf.derivative())
    tape.reset_tlm_values()
    viscosity.block_variable.tlm_value = direction
    tape.evaluate_tlm()
    tlm = float(recorded["tangential"].block_variable.tlm_value)
    print(f"    [{preset}, tangential, seed {seed}] tlm {tlm:.15e}  "
          f"adjoint {adjoint:.15e}")
    assert abs(tlm - adjoint) <= 1e-8 * abs(adjoint)


def _coefficient_dot(a, b):
    """The l2 product of the coefficient vectors of two mixed objects."""
    return sum(float(np.vdot(x.dat.data_ro, y.dat.data_ro))
               for x, y in zip(a.subfunctions, b.subfunctions))


def _random_fill(target, rng, kernel=None, index=None, amplitude=0.0):
    """Fill every sub-function with random values, plus a kernel component.

    Args:
      target: a mixed `Function` or `Cofunction`, changed in place.
      rng: a NumPy random generator.
      kernel: the coefficient vector of the kernel basis vector, or `None`.
      index: the sub-function that `kernel` belongs to.
      amplitude: the multiple of `kernel` that is added.
    """
    for i, f in enumerate(target.subfunctions):
        f.dat.data[:] = rng.standard_normal(f.dat.data_ro.shape)
        if kernel is not None and i == index:
            f.dat.data[:] += amplitude * kernel


@pytest.mark.longtest
def test_gia_block_adjoint_on_kernel_seed(meshes):  # noqa: F811
    """The block adjoint on the coupled space removes the kernel and is P^T.

    One taped step of the `direct` case. The first `KernelProjectionBlock` on
    the tape is driven by hand through pyadjoint's `evaluate_adj` and
    `evaluate_tlm`, with a random adjoint seed `s` and a random tangent `x`,
    both with a kernel component a hundred times larger than the rest.

    Checks:
      - the adjoint output has no kernel component (round-off of the seed);
      - the blocks without a basis are passed through unchanged;
      - the dot-product test `<P^T s, x> = <s, P x>` with the block's own
        adjoint and tangent.
    """
    _, sub = meshes
    viscosity, _ = _control_and_direction(sub)
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        Control(viscosity)
        _, solver = _gia_forward(meshes, viscosity, 0.0, "direct", n_steps=1)
    finally:
        pause_annotation()
    block = next(b for b in tape.get_blocks()
                 if isinstance(b, KernelProjectionBlock))
    dependency, output = block.get_dependencies()[0], block.get_outputs()[0]
    disp = solver.field_index["displacement"]
    basis = list(solver.nullspace)[disp]
    kernel = basis._vecs[0].dat.data_ro.copy()
    Z = solver.solution.function_space()

    rng = np.random.default_rng(20261118)
    seed = fd.Cofunction(Z.dual())
    _random_fill(seed, rng, kernel, disp, 100.0)
    tangent = fd.Function(Z)
    _random_fill(tangent, rng, kernel, disp, 100.0)

    # Adjoint of the block applied to `seed`.
    tape.reset_variables()
    output.adj_value = seed
    block.evaluate_adj()
    adjoint = dependency.adj_value
    # Tangent of the block applied to `tangent`.
    dependency.tlm_value = tangent
    block.evaluate_tlm()
    projected_tangent = output.tlm_value

    seed_content = abs(float(np.vdot(kernel, seed.subfunctions[disp].dat.data_ro)))
    left_content = abs(float(np.vdot(kernel, adjoint.subfunctions[disp].dat.data_ro)))
    print(f"    [block adjoint] kernel content of the seed {seed_content:.3e}, "
          f"after the adjoint {left_content:.3e}")
    assert seed_content > 10.0
    assert left_content <= 1e-12 * seed_content
    for i, (a, s) in enumerate(zip(adjoint.subfunctions, seed.subfunctions)):
        if i != disp:
            assert np.array_equal(a.dat.data_ro, s.dat.data_ro)

    lhs = _coefficient_dot(adjoint, tangent)
    rhs = _coefficient_dot(seed, projected_tangent)
    print(f"    [block adjoint] <P^T s, x> {lhs:.15e}  <s, P x> {rhs:.15e}")
    assert abs(lhs - rhs) <= 1e-12 * abs(rhs)
    # The test only discriminates if P changes the pairing.
    assert abs(_coefficient_dot(seed, tangent) - rhs) > 1e-3 * abs(rhs)
