"""The low-rank DtN build, checked against the machinery it reproduces.

Two independent things have to hold, and they fail differently.

The *build* has to produce the same vectors as `assemble(e_k * v * ds)`. That is
checked against the reference build directly, and the interesting case is the
one where it could be silently wrong: the trace-space nodes coinciding with the
boundary quadrature points is a variant choice inside FInAT, not a guarantee, so
these tests exercise the mismatch deliberately and confirm it is caught.

The *elimination* has to be the algebra the multiplier path solves. That is
checked by closing the loop against a live solve: `u_k . psi / (scale_k * A_h)`
must reproduce `solver.coefficients()`, which the solver obtained by an entirely
different route (a Schur complement onto Real unknowns).
"""

import numpy as np
import pytest
import firedrake as fd
from mpi4py import MPI

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.dtn_lowrank import (
    boundary_facet_cellname, build_boundary_mode_rows, supports_trace_build)
from test_gravity_solver import annulus_mesh, shell_mesh_3d


def rows_for(solver, bc_id, **kwargs):
    descriptor = dict(solver.dtn_boundaries)[bc_id]
    side, radius = solver.boundary_geometry[bc_id]
    return build_boundary_mode_rows(
        solver.solution_space, solver.ds(bc_id), descriptor, side, radius,
        solver.alpha, solver.quad_degree, **kwargs)


@pytest.fixture(scope="module")
def annulus():
    return annulus_mesh(n_azimuthal=96, dr=0.25)


@pytest.fixture(scope="module")
def shell():
    return shell_mesh_3d(refinement_level=2, dr=0.25)


def annulus_solver(mesh, M=5):
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    return GravitySolver(
        psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=M)},
                       "bottom": {"dtn": CylindricalDtN(M=M)}},
        solver_parameters="direct")


def shell_solver(mesh, L=3):
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    return GravitySolver(
        psi, 0.0, bcs={"top": {"dtn": SphericalDtN(L=L)},
                       "bottom": {"dtn": SphericalDtN(L=L)}},
        solver_parameters="iterative")


# ---------------------------------------------------------------------------
# The build reproduces the reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_trace_build_matches_the_reference(geometry, annulus, shell):
    """Bit-identical to `assemble(e_k * v * ds)`, on both boundaries.

    Both sides are taken because the interior boundary carries the `mean` mode,
    which is the one entry of the 2-D table that does not come from the
    numerical tabulation and so is the one place the two orderings could drift.
    """
    solver = (annulus_solver(annulus) if geometry == "2d"
              else shell_solver(shell))
    for bc_id in ("top", "bottom"):
        fast = rows_for(solver, bc_id)
        reference = rows_for(solver, bc_id, force_reference=True)
        assert fast.used_trace_build
        assert fast.keys == reference.keys
        assert np.array_equal(fast.dofs, reference.dofs)
        comm = solver.mesh.comm
        scale = comm.allreduce(np.max(np.abs(reference.rows)) if reference.rows.size
                               else 0.0, op=MPI.MAX)
        local = (np.max(np.abs(fast.rows - reference.rows)) if fast.rows.size
                 else 0.0)
        worst = comm.allreduce(
            local, op=MPI.MAX) / scale
        print(f"    [{geometry} {bc_id}] {len(fast.keys)} modes, "
              f"trace degree {fast.trace_degree}, rel {worst:.3e}, "
              f"build {fast.build_time:.3f}s vs reference "
              f"{reference.build_time:.3f}s")
        assert worst <= 1e-13


# There is deliberately no speed assertion here. The reference build's cost is
# almost entirely form compilation, which Firedrake caches per process, so by
# the time a second test asks for it the forms are already compiled and the
# "reference" is just n parloops - measured, the ratio inside one pytest
# session is under 10x against 140-180x in a fresh process. Asserting on it
# would be measuring the test-ordering, which is the same class of mistake as
# timing a cold kernel cache. The speed measurement lives in NOTES/bench, where
# the cold/warm protocol is controlled.


def test_mismatched_trace_degree_is_caught(shell):
    """The self-assertion must catch a node/quadrature mismatch.

    This is the failure the assertion exists for - the FInAT element's node
    variant changing under us - and it is provoked here by asking for the wrong
    trace degree, which produces the same symptom. Without the assertion the
    build is wrong by about 1e-8 and nothing else notices.
    """
    solver = shell_solver(shell)
    descriptor = dict(solver.dtn_boundaries)["top"]
    side, radius = solver.boundary_geometry["top"]
    with pytest.raises(RuntimeError, match="disagrees with the symbolic one"):
        build_boundary_mode_rows(
            solver.solution_space, solver.ds("top"), descriptor, side, radius,
            solver.alpha, solver.quad_degree + 2)


def test_the_constant_mode_cannot_detect_the_mismatch(shell):
    """Why the assertion is on the highest degree and not on Y00.

    Any nodal interpolation reproduces a constant exactly whatever its node
    placement, so `Y_00` is blind to exactly the defect being guarded against.
    This pins that: at a deliberately wrong trace degree, `Y_00` agrees to
    round-off while the degree-L modes are wrong by orders of magnitude.
    """
    solver = shell_solver(shell)
    descriptor = dict(solver.dtn_boundaries)["top"]
    side, radius = solver.boundary_geometry["top"]
    v = fd.TestFunction(solver.solution_space)
    measure = solver.ds("top")
    modes = descriptor.modes(side, radius, solver.X)

    wrong = build_boundary_mode_rows(
        solver.solution_space, measure, descriptor, side, radius, solver.alpha,
        solver.quad_degree + 2, rtol=np.inf)
    keys = wrong.keys
    scale = np.max(np.abs(wrong.rows))

    def deviation(key):
        i = keys.index(key)
        reference = np.asarray(
            assemble_row(modes[i], v, measure), dtype=float)[wrong.dofs]
        return np.max(np.abs(wrong.rows[i] - reference)) / scale

    blind = deviation("Y0,0")
    sensitive = max(deviation(f"Y{descriptor.L},{m}")
                    for m in (0, descriptor.L, -descriptor.L))
    print(f"    [blindness] Y0,0 {blind:.3e}   degree-L {sensitive:.3e}")
    # Y00 is at round-off: it cannot see the defect at all.
    assert blind <= 1e-14
    # The degree-L modes are far outside the build's own gate, so the assertion
    # as written catches this and an assertion on Y00 would not.
    assert sensitive > 1e-11
    assert sensitive > 1e3 * max(blind, 1e-300)


def assemble_row(mode, v, measure):
    return fd.assemble(mode.expr * v * measure).dat.data_ro


# ---------------------------------------------------------------------------
# The elimination reproduces the multiplier path
# ---------------------------------------------------------------------------
def annulus_source(mesh, m_mode=3):
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    phi = fd.atan2(X[1], X[0])
    return fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(
        fd.cos(m_mode * phi) * fd.exp(-(((r - 1.7) / 0.2) ** 2)))


def shell_source(mesh):
    from gadopt.spherical_harmonics import real_spherical_harmonic
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    return fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(
        real_spherical_harmonic(2, 1, X) * fd.exp(-(((r - 1.7) / 0.2) ** 2)))


@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_recovered_coefficients_match_the_solver(geometry, annulus, shell):
    """`u_k . psi / (scale_k * A_h)` reproduces `solver.coefficients()`.

    This is the check that the hand elimination is the algebra the multiplier
    path actually solves. The solver obtained those numbers through a Schur
    complement onto Real unknowns; this route never forms a multiplier at all,
    so agreement to round-off is a statement about the mathematics rather than
    about two implementations of one formula.

    The negative control is in the same test: `/norm_k`, the plausible wrong
    denominator, is asserted to MISS. It agrees with the right answer whenever
    the discrete boundary measure equals the analytic one, so on a fine curved
    mesh it is nearly right - which is what makes it dangerous.
    """
    mesh = annulus if geometry == "2d" else shell
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    rho = annulus_source(mesh) if geometry == "2d" else shell_source(mesh)
    descriptor = (CylindricalDtN(M=5) if geometry == "2d"
                  else SphericalDtN(L=3))
    solver = GravitySolver(
        psi, rho,
        bcs={"top": {"dtn": descriptor}, "bottom": {"dtn": descriptor}},
        solver_parameters="direct" if geometry == "2d" else "iterative")
    solver.mixed_solution.assign(0)
    solver.solve()
    solved = solver.coefficients()
    psi_local = np.asarray(
        solver.mixed_solution.subfunctions[0].dat.data_ro, dtype=float)

    comm = mesh.comm
    worst_correct = worst_wrong = 0.0
    worst_ratio = area_ratio = None
    for bc_id in ("top", "bottom"):
        side, radius = solver.boundary_geometry[bc_id]
        rows = rows_for(solver, bc_id)
        # `coefficients` returns the rank-local contribution: the owned entries
        # of an assembled Cofunction are already halo-reduced, so the sum over
        # ranks is the exact global dot with nothing double counted. Summing is
        # part of the contract, so the test does it rather than assuming
        # serial.
        recovered = np.array(comm.allreduce(rows.coefficients(psi_local)))
        wrong = np.array(comm.allreduce(
            rows.rows @ psi_local[rows.dofs]))
        modes = dict(solver.dtn_boundaries)[bc_id].modes(side, radius, solver.X)
        analytic = (2 * np.pi * radius if mesh.geometric_dimension == 2
                    else 4 * np.pi * radius ** 2)
        area_ratio = rows.area / analytic
        for key, value, raw, mode in zip(rows.keys, recovered, wrong, modes):
            reference = solved[bc_id][key]
            worst_correct = max(worst_correct, abs(value - reference))
            worst_wrong = max(worst_wrong, abs(raw / mode.norm - reference))
            # Ratio of the two MEASURED recoveries, not of their metadata:
            # `value` comes through `rows.recovery`, `raw / mode.norm` does
            # not, so this tests that `recovery` really is 1/(scale_k * A_h).
            # Computed from the metadata instead it would be the tautology
            # scale_k = norm_k / |boundary|_analytic, which is a different
            # test and is `test_scale_matches_the_descriptors`.
            if abs(value) > 1e-8:
                ratio = (raw / mode.norm) / value
                if (worst_ratio is None
                        or abs(ratio - area_ratio) > abs(worst_ratio - area_ratio)):
                    worst_ratio = ratio

    print(f"    [{geometry}] /(scale*A_h) {worst_correct:.3e}   "
          f"/norm_k {worst_wrong:.3e}   ratio {worst_ratio:.6f} vs "
          f"A_h/A_analytic {area_ratio:.6f}")
    assert worst_correct <= 1e-12

    # The negative control asserts the MECHANISM, not the magnitude. An
    # assertion that `/norm_k` misses by some large factor is mesh-fragile:
    # measured on this 2-D configuration at n_azimuthal 96 / 192 / 384 the
    # ratio of wrong to correct is 3971 / 58 / 0.8, so it fails at one
    # refinement and is vacuous at two - and it fails in the direction that
    # looks like an improvement, so the obvious repair is to lower the
    # threshold, which removes the guard.
    #
    # The two denominators differ by exactly `scale_k * A_h / norm_k =
    # A_h / |boundary|_analytic`, so asserting THAT identity is
    # mesh-independent and says more: it confirms the discrepancy really is
    # the discrete-versus-analytic boundary measure rather than merely that
    # it is large.
    assert worst_ratio is not None, "no coefficient large enough to form a ratio"
    # The discriminating power of this check is exactly the mesh's discrete
    # area error, `|A_h/A_analytic - 1|`, because that is the whole difference
    # between the two formulas. So the tolerance is scaled to it rather than
    # fixed: a tenth of the deviation catches the wrong denominator, and on a
    # mesh fine enough that the deviation reaches round-off the check becomes
    # honestly uninformative instead of falsely reassuring.
    deviation = abs(area_ratio - 1.0)
    print(f"    [{geometry}] A_h/A_analytic - 1 = {deviation:.3e} "
          f"(the discriminating power of the ratio check)")
    if deviation > 1e-12:
        assert abs(worst_ratio - area_ratio) <= 0.1 * deviation
    else:
        pytest.skip(
            f"discrete area error {deviation:.2e} is at round-off, so the two "
            "denominators are numerically indistinguishable on this mesh - a "
            "property of the problem, not a gap in the suite")

    # On a fine 2-D mesh no test can separate the two denominators, because
    # they genuinely converge to each other - a property of the problem, not a
    # gap in the suite. The 3-D arm is the load-bearing one: a degree-2 cubed
    # sphere's area error falls slowly, so there the two stay far apart.
    if geometry == "3d":
        assert worst_wrong >= 1e3 * max(worst_correct, 1e-16)


def test_scale_matches_the_descriptors(annulus, shell):
    """`scale_k` is `norm_k / |boundary|_analytic` for all three descriptors."""
    X2 = fd.SpatialCoordinate(annulus)
    X3 = fd.SpatialCoordinate(shell)
    R = 2.0
    for mode in CylindricalDtN(M=3).modes("interior", R, X2):
        analytic = 2 * np.pi * R
        assert abs(mode.scale - mode.norm / analytic) < 1e-15
    for mode in SphericalDtN(L=2).modes("exterior", R, X3):
        analytic = 4 * np.pi * R**2
        assert abs(mode.scale - mode.norm / analytic) < 1e-15
    # And the three values the plan names explicitly.
    assert CylindricalDtN(M=1).modes("exterior", R, X2)[0].scale == 0.5
    assert CylindricalDtN(M=1).modes("interior", R, X2)[0].scale == 1.0
    assert abs(SphericalDtN(L=0).modes("exterior", R, X3)[0].scale
               - 1.0 / (4 * np.pi)) < 1e-16


def test_facet_cell_detection(annulus, shell):
    """The trace build is selected by facet type, decided up front."""
    assert boundary_facet_cellname(annulus) == "interval"
    assert boundary_facet_cellname(shell) == "quadrilateral"
    assert supports_trace_build(annulus)
    assert supports_trace_build(shell)
    # Tetrahedra give triangular facets, where the coincidence does not hold.
    assert boundary_facet_cellname(fd.UnitCubeMesh(1, 1, 1)) == "triangle"
    assert not supports_trace_build(fd.UnitCubeMesh(1, 1, 1))
