"""The operator version of the GIA solver moves only when the Jacobian changes.

Every cached preconditioner piece of `gadopt.gia_solver.SelfGravitatingGIASolver`
(the dense Schur complement of the multiplier path, the cached block-1 apply
of `DtNTwoBlockSchurPC`, `CondensedBlockPC`) rebuilds when the integer
`appctx["operator_version"]` changes. The solver raises that integer when a
Jacobian coefficient other than the solution changes. The time integrator
assigns its `Real` coefficients before every solve, also when their value
stays the same, so a snapshot by `dat_version` alone raises the version at
every step and every cache rebuilds at a fixed time step. These tests count
the version changes directly, so they do not depend on a particular cache.
"""

import gadopt  # noqa: F401  (gadopt before firedrake)
import pytest

from gia_cases import Case, annulus_meshes, build_new

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest


@pytest.fixture(scope="module")
def meshes():
    """The small P2-curved annulus and its mantle submesh of the gate tests."""
    return annulus_meshes()


def versions_over(solver, dt, steps):
    """Run `steps` solves at time step `dt` and return the version after each.

    Args:
      solver: a `SelfGravitatingGIASolver` built with a `Real` time step.
      dt: the time step, assigned before the first of the solves.
      steps: the number of solves.

    Returns:
      The list of `appctx["operator_version"]` after each solve.
    """
    solver.dt.assign(dt)
    versions = []
    for _ in range(steps):
        solver.solve()
        versions.append(solver.appctx["operator_version"])
    return versions


@pytest.mark.parametrize("scheme", ["backward_euler", "bdf2"])
def test_the_operator_version_moves_only_when_the_time_step_does(meshes, scheme):
    """Three solves at one `dt` raise the version at most once; a new `dt` once more.

    Backward Euler has the same Jacobian at every step of one `dt`, so the
    three versions are equal to the one at construction. BDF2 starts with a
    backward-Euler step and changes its coefficient `alpha` at the second
    step, which is a real change of the Jacobian: one raise, then none. After
    `dt` changes, the first solve raises the version once (BDF2 at a new step
    ratio can change `alpha` once more at the next step), and the following
    solves at that `dt` do not.

    CATCHES: a fingerprint that keys a `Real` coefficient on `dat_version`,
    which `assign` moves at every step with the same value; a fingerprint
    that misses the time step, so a cache built for the old `dt` is reused.
    """
    # A fluid core, because the Quadrature history takes no displacement
    # condition; its rotation kernel is declared, as in the other tests.
    # The DG family only: on the Quadrature family the history is recovered
    # inside the Newton loop, the Jacobian depends on the solution, and the
    # version is `None` at every solve by design.
    case = Case(family="DG", scheme=scheme, representation="lowrank",
                core="fluid", load="surface")
    solver, _, _ = build_new(meshes, case, nullspace="rigid_rotation")
    start = solver.appctx["operator_version"]
    first = versions_over(solver, 0.1, 3)
    second = versions_over(solver, 0.2, 3)

    # A power law or a moving coastline would set the version to `None`;
    # this Maxwell case keeps an integer version.
    assert all(isinstance(v, int) for v in first + second), (first, second)
    if scheme == "backward_euler":
        assert first == [start] * 3, (start, first)
        assert second == [start + 1] * 3, (start, first, second)
    else:
        # One raise when BDF2 leaves its backward-Euler start, none after.
        assert first[-1] == first[-2] <= start + 1, (start, first)
        # The new `dt` raises once, a new step ratio at most once more, and
        # the last two solves at the new `dt` leave the version unchanged.
        assert second[-1] == second[-2], (first, second)
        assert first[-1] + 1 <= second[-1] <= first[-1] + 2, (first, second)
