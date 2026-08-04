"""First-order (Taylor) adjoint test for the coupled `SelfGravitatingGIASolver`.

`GravitySolver` (gravity-alone) has a verified first-order adjoint on both its
representations -- `tests/unit/test_gravity_adjoint.py`. The **coupled**
self-gravitating solver has none, and until this file no test referenced its
adjoint at all. This is road map §9.6 / handover S1.6 step 1: the one capability
that is a genuine *absence* rather than an optimisation, and the measurement that
decides whether the coupled tape is a free asset (worth protecting) or a broken
one (worth fixing) before any low-rank work is costed.

Laptop-safe by construction: 2-D coarse annulus (dr 0.2, nazim 32), a direct LU
solve, one solve. No 3-D, no cluster.

Design, mirroring `test_gravity_adjoint.py`:

  * **The control is a Real-space `Function`, never a `Constant` or a float.**
    `ensure_constant` (`gadopt/utility.py`) wraps a float as a `Constant`, and a
    `Constant` control is silently non-differentiable in this Firedrake (see the
    gravity-alone findings); a `Function` is passed through untouched and threads
    into the residual as a differentiable coefficient.
  * **The functional reads the solution through UFL `split` of the mixed
    solution**, i.e. a reference the tape owns -- never `float(...)` or
    `.coefficients()`, which detach (S1.6).
  * **Only `solve()` and the objective assemble are taped.** The solver is
    constructed under `stop_annotating()` so the geometry assembles in
    `__init__` never clutter the tape.
  * **Matfree-safe: first-order ladder only, no Hessian leg.**

`nullspace=None` is deliberate for the primary test. `project_out_nullspace`
(`gia_gravity.py`) does an in-place `basis.orthogonalize(solution.subfunctions[i])`
*after* the taped solve, which the tape cannot see; declaring no nullspace makes
that method a no-op and gives the cleanest taped path. The nullspace-on case is
a separate, diagnostic test -- S1.6 predicts it may sever, unverified.
"""

import sys
from pathlib import Path

import firedrake as fd
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from pyadjoint.tape import annotate_tape

# Siblings on the test path (pytest puts tests/unit on sys.path): the guarded
# Taylor driver and matfree-safe ladder, and the coupled-system builder.
from test_gravity_adjoint import (
    assert_taylor_with_guards,
    taylor_first_order_ladder,
)
from test_gia_gravity import build


# ---------------------------------------------------------------------------
# Fixtures: coarse 2-D annulus (annotation-free, cached) and tape hygiene
# ---------------------------------------------------------------------------
DR_MANTLE = 0.2
N_AZIMUTHAL = 32
CELL_MANTLE = 101


@pytest.fixture(autouse=True)
def clean_tape():
    """Fresh tape per test; annotation guaranteed off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


@pytest.fixture(scope="module")
def meshes():
    """The parent annulus and its mantle submesh, both P2-curved.

    Duplicated from `test_gia_gravity.meshes` (a fixture cannot be shared across
    modules) but reads the same cached `.msh`: same dr/nazim, so the file is
    generated at most once across the whole suite.
    """
    pytest.importorskip("gmsh")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "demos" / "gravity"))
    import generate_selfgrav_annulus as gen
    from validate_selfgrav_annulus import curve_mesh

    path = Path("/tmp") / f"gadopt_selfgrav_gia_{DR_MANTLE}_{N_AZIMUTHAL}.msh"
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        gen.generate(str(path), dr_mantle=DR_MANTLE, n_azimuthal=N_AZIMUTHAL)
    fd.COMM_WORLD.barrier()

    parent = curve_mesh(fd.Mesh(str(path)))
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


# ---------------------------------------------------------------------------
# Forward map and the Taylor driver
# ---------------------------------------------------------------------------
def _forward(control, meshes, *, declare_nullspace):
    """Coupled solve with `shear_modulus = control`; J = ||u||^2 over the mantle.

    `shear_modulus` is the strongest single control on the coupled operator
    (measured 0.11 relative change in `check_control_independence.py`), so the
    directional derivative is large and the Taylor signal clean -- which is what
    a first *existence* test wants. It also governs the elastic response
    directly, so J moves strongly with it.
    """
    _, sub = meshes
    with stop_annotating():
        solver, _, layout = build(
            meshes,
            rotation=False,
            condensed=False,
            declare_nullspace=declare_nullspace,
            approximation_kwargs={"shear_modulus": control},
            solver_parameters="direct",
        )
    solver.solve()  # the taped variational solve
    u = fd.split(solver.solution)[layout.displacement]
    return fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))


def _assert_replay_matches_fresh_solve(Jhat, meshes, *, declare_nullspace,
                                       factor=1.05):
    """Tape replay at a shifted control equals a fresh build-and-solve there.

    The one false-green class the four Taylor guards do NOT exclude: a control
    value baked into a coefficient at `__init__` time. That would make the tape
    self-consistent -- rate 2, nonzero dJdm, negative control would all still
    pass -- while the gradient was wrong for the *true* forward map, because the
    tape would carry a frozen control the real solve does not. Replaying the
    tape at a control 5% away and comparing against a fresh, untaped
    build-and-solve at the same value catches exactly that: if anything were
    baked, the two would diverge. Mirrors `test_gravity_adjoint.assert_replay_b2`.
    """
    _, sub = meshes
    R = fd.FunctionSpace(sub, "R", 0)
    m2 = fd.Function(R).assign(factor)
    with stop_annotating():
        J_direct = float(_forward(m2, meshes, declare_nullspace=declare_nullspace))
    J_replay = float(Jhat(m2))
    assert abs(J_replay - J_direct) <= 1e-9 * abs(J_direct), (
        f"tape replay {J_replay:.12e} != fresh solve {J_direct:.12e} at "
        f"control {factor} -- a control value is baked into the tape")
    print(f"    [replay-b2] fresh={J_direct:.10e}  replay={J_replay:.10e}"
          f"  rel={abs(J_replay - J_direct) / abs(J_direct):.2e}")
    return J_direct, J_replay


def _reduced_functional(meshes, *, declare_nullspace):
    """Tape the forward once and wrap it, with the Real-space control and h."""
    _, sub = meshes
    R = fd.FunctionSpace(sub, "R", 0)
    control = fd.Function(R).assign(1.0)  # created annotation-free
    h = fd.Function(R).assign(1.0)        # Real space: one dof, direction is +1

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        m = Control(control)
        J = _forward(control, meshes, declare_nullspace=declare_nullspace)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    return Jhat, control, h, J


def test_coupled_adjoint_is_second_order(meshes):
    """The coupled adjoint exists: Taylor rate 2 with the gradient, 1 without.

    This is the deliverable. A passing rate here means pyadjoint tapes the whole
    coupled solve -- displacement, internal variable, potential, DtN multipliers
    -- and its gradient is exact, so the coupled adjoint is a free asset and the
    S1.6 sequence proceeds (mode-row Schur PC next, low-rank operator only if it
    ever buys more than the free adjoint costs).
    """
    Jhat, control, h, J = _reduced_functional(meshes, declare_nullspace=False)

    ladder = taylor_first_order_ladder(Jhat, control, h)
    print(f"\n    [ladder] R0_rate={[f'{r:.3f}' for r in ladder['R0_rate']]}"
          f"  R1_rate={[f'{r:.3f}' for r in ladder['R1_rate']]}")

    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)

    # Close the one false-green class the guards above cannot: a control baked
    # into a coefficient at construction time (the tape would stay
    # self-consistent while the gradient was wrong for the true forward map).
    _assert_replay_matches_fresh_solve(Jhat, meshes, declare_nullspace=False)


def test_coupled_adjoint_with_nullspace_declared(meshes):
    """The declared-nullspace production path also tapes to second order.

    Declaring `rigid_rotation_nullspace` makes `solve()` call
    `project_out_nullspace`, whose `basis.orthogonalize(...)` mutates
    `solution.subfunctions[i]` in place *after* the taped solve -- the operation
    S1.6 predicts might sever the tape.

    Measured: the gradient is still exact (rate 2). **This does NOT prove the
    orthogonalize is tape-safe.** The rigid rotation this configuration projects
    out is a genuine kernel, so the solve leaves only ~1e-13 of it in `u`
    (documented in `project_out_nullspace`); removing ~1e-13 moves J below the
    finite-difference floor, so a taken-at-the-wrong-point gradient and a correct
    one are indistinguishable here. What this test pins is the weaker, still
    useful fact that the production path with a nullspace declared is
    differentiable to the same tolerance as without one. Actually stressing the
    orthogonalize needs a configuration where the projected component is O(1) --
    e.g. seeding `u` with a large rigid rotation before the objective -- which is
    left as the follow-up S1.6 names.
    """
    Jhat, control, h, J = _reduced_functional(meshes, declare_nullspace=True)
    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)
