"""The low-rank fast path against the multiplier path it must reproduce.

Both paths discretise the *same* operator, so they must agree to round-off, and
that is the acceptance test for the representation - accuracy against a closed
form would not distinguish them, since both are equally right or equally wrong
about the physics.

The gates are the plan's: potentials <= 1e-14 relative, trace coefficients
<= 1e-12. They are met by four to five orders on every configuration here, which
is the useful signal - a change that broke the elimination would not land at
1e-13, it would land at 1e-4.

Run under `mpiexec` as well as serially. The low-rank operator sums rank-local
dot products through one `Allreduce`, so a rank-ownership mistake would show up
only in parallel, and the repeated-application check exists because a wrong
operator in this code area has previously been right once and wrong afterwards.
"""

import numpy as np
import pytest
import firedrake as fd
from mpi4py import MPI

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from test_gravity_solver import (
    RMAX, RMIN, annulus_mesh, relative_l2_error, shell_mesh_3d)


@pytest.fixture(scope="module")
def annulus():
    return annulus_mesh(n_azimuthal=96, dr=0.25)


@pytest.fixture(scope="module")
def shell():
    return shell_mesh_3d(refinement_level=2, dr=0.25)


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


def solve_both(mesh, rho, bcs_factory, degree=1, **kwargs):
    """The same problem on both representations; returns (psi, solver) pairs."""
    out = {}
    for representation in ("multiplier", "lowrank"):
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", degree))
        solver = GravitySolver(psi, rho, bcs=bcs_factory(),
                               dtn_representation=representation, **kwargs)
        solver.mixed_solution.assign(0)
        solver.solve()
        out[representation] = (psi, solver)
    return out


def compare(mesh, results, psi_rtol=1e-14, coeff_atol=1e-12, label=""):
    (psi_m, solver_m), (psi_l, solver_l) = (results["multiplier"],
                                            results["lowrank"])
    psi_error = relative_l2_error(psi_l, psi_m, mesh)
    coefficients_m, coefficients_l = solver_m.coefficients(), solver_l.coefficients()
    assert set(coefficients_m) == set(coefficients_l)
    worst = 0.0
    for bc_id in coefficients_m:
        assert set(coefficients_m[bc_id]) == set(coefficients_l[bc_id])
        for key in coefficients_m[bc_id]:
            worst = max(worst,
                        abs(coefficients_m[bc_id][key] - coefficients_l[bc_id][key]))
    print(f"    [{label}] psi {psi_error:.3e}   coefficients {worst:.3e}")
    assert psi_error <= psi_rtol
    assert worst <= coeff_atol
    return psi_error, worst


# ---------------------------------------------------------------------------
# Parity across the configurations of the shipped test suite
# ---------------------------------------------------------------------------
def test_parity_annulus_two_boundaries(annulus):
    """Both boundaries, so the interior `mean` mode is exercised - the mode
    whose weight is negative and whose presence is what makes the SPD argument
    non-trivial."""
    rho = annulus_source(annulus)
    results = solve_both(
        annulus, rho,
        lambda: {"top": {"dtn": CylindricalDtN(M=5)},
                 "bottom": {"dtn": CylindricalDtN(M=5)}})
    compare(annulus, results, label="annulus M=5")


def test_parity_annulus_sheet(annulus):
    """A `sigma` sheet on a DtN boundary: the right-hand side term the fast
    path re-assembles on every solve."""
    X = fd.SpatialCoordinate(annulus)
    phi = fd.atan2(X[1], X[0])
    results = solve_both(
        annulus, 0.0,
        lambda: {"top": {"dtn": CylindricalDtN(M=4), "sigma": fd.cos(3 * phi)},
                 "bottom": {"dtn": CylindricalDtN(M=4)}}, degree=2)
    compare(annulus, results, label="annulus sheet")


def tight(representation, rtol):
    """Solver options that make both paths solve to the same tight tolerance.

    The multiplier path's outer FGMRES converges in one iteration - a full
    Schur factorisation is exact - so its accuracy is set by the potential
    block's inner tolerance, not by the outer one. Tightening only the outer
    rtol would leave it unchanged and make the comparison look tolerance-proof
    when it is not.
    """
    if representation == "multiplier":
        return {"ksp_rtol": rtol,
                "dtn": {"fieldsplit_0": {"ksp_rtol": min(rtol * 1e-2, 1e-12)}}}
    return {"ksp_rtol": rtol}


def test_parity_shell(shell):
    """3-D, two boundaries, the production geometry.

    Solved to `rtol = 1e-14` on both paths, because at the shipped `1e-11` the
    two agree only to 4.2e-12 and that is the Krylov tolerance rather than the
    representation - see `test_shell_parity_is_tolerance_limited`, which is the
    evidence for that claim rather than an assertion of it.
    """
    rho = shell_source(shell)
    out = {}
    for representation in ("multiplier", "lowrank"):
        psi = fd.Function(fd.FunctionSpace(shell, "CG", 1))
        solver = GravitySolver(
            psi, rho,
            bcs={"top": {"dtn": SphericalDtN(L=3)},
                 "bottom": {"dtn": SphericalDtN(L=3)}},
            dtn_representation=representation, solver_parameters="iterative",
            solver_parameters_extra=tight(representation, 1e-14))
        solver.mixed_solution.assign(0)
        solver.solve()
        out[representation] = (psi, solver)
    compare(shell, out, label="shell L=3 (rtol 1e-14)")


def test_shell_parity_is_tolerance_limited(shell):
    """Show the residual 3-D disagreement is the Krylov tolerance.

    The sensitivity evidence for the previous test. If the two representations
    differed as *discretisations*, the gap would sit at some floor independent
    of how hard the linear systems are solved. It does not: it tracks `rtol`
    essentially one-for-one over five orders, which is what a
    tolerance-limited comparison looks like and what a representation error
    does not.
    """
    rho = shell_source(shell)
    gaps = {}
    for rtol in (1e-9, 1e-11, 1e-13):
        out = {}
        for representation in ("multiplier", "lowrank"):
            psi = fd.Function(fd.FunctionSpace(shell, "CG", 1))
            solver = GravitySolver(
                psi, rho,
                bcs={"top": {"dtn": SphericalDtN(L=3)},
                     "bottom": {"dtn": SphericalDtN(L=3)}},
                dtn_representation=representation,
                solver_parameters="iterative",
                solver_parameters_extra=tight(representation, rtol))
            solver.mixed_solution.assign(0)
            solver.solve()
            out[representation] = psi
        gaps[rtol] = relative_l2_error(out["lowrank"], out["multiplier"], shell)
    print("    [tolerance sweep] "
          + "  ".join(f"rtol {r:.0e} -> {g:.3e}" for r, g in gaps.items()))
    # Each hundredfold tightening buys about two orders on the gap.
    assert gaps[1e-9] / gaps[1e-11] > 30
    assert gaps[1e-11] / gaps[1e-13] > 30
    # And the gap is never much larger than the tolerance that produced it.
    for rtol, gap in gaps.items():
        assert gap < 10 * rtol


def test_parity_with_strong_bc(annulus):
    """A prescribed `psi` alongside a DtN boundary.

    This is the configuration that fails if the columns of `C` are not zeroed
    at constrained degrees of freedom: `A` has those rows and columns
    eliminated, so a `B` still coupling through them is inconsistent with it.
    """
    X = fd.SpatialCoordinate(annulus)
    r = fd.sqrt(fd.dot(X, X))
    phi = fd.atan2(X[1], X[0])
    reference = (r / RMAX) ** 2 * fd.cos(2 * phi)
    results = solve_both(
        annulus, 0.0,
        lambda: {"top": {"psi": reference},
                 "bottom": {"dtn": CylindricalDtN(M=4)}}, degree=2)
    compare(annulus, results, label="strong bc")


def test_monopole_datum_reaches_the_fast_path(annulus):
    """The 2-D monopole datum runs from `solve()` on the fast path too.

    It used to be `check_net_mass` refusing here. The datum replaced the
    refusal, and the property worth guarding is the same one: the enclosed-mass
    bookkeeping is driven from `solve`, so it cannot be silently skipped by the
    representation that does not go through a variational solve.
    """
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 2))
    solver = GravitySolver(
        psi, 1.0, bcs={"top": {"dtn": CylindricalDtN(M=1)}},
        solver_parameters="direct", dtn_representation="lowrank")
    assert solver.total_enclosed_mass() == 0.0  # not yet solved
    solver.solve()
    assert solver.total_enclosed_mass() == pytest.approx(
        fd.assemble(1 * solver.dx), rel=1e-12)

    # The mass being computed is not enough - it would still be computed if the
    # datum were dropped from the form. The gauge is what says the term is in
    # the residual: without it the boundary mean sits at the spurious
    # 2 G M / alpha instead of zero.
    perimeter = fd.assemble(1 * solver.ds("top"))
    boundary_mean = fd.assemble(psi * solver.ds("top")) / perimeter
    assert abs(boundary_mean) < 1e-9 * abs(solver.total_enclosed_mass())


def test_accuracy_against_the_closed_form(annulus):
    """Both paths are equally right about the physics, not merely equal.

    Parity alone cannot detect a shared error, so one configuration is also
    checked against the closed form: a sheet `sigma cos(m phi)` at radius `a`
    gives `(2 pi G sigma a / m) (r/a)^m cos(m phi)` inside.
    """
    m_mode = 3
    X = fd.SpatialCoordinate(annulus)
    r = fd.sqrt(fd.dot(X, X))
    phi = fd.atan2(X[1], X[0])
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 2))
    solver = GravitySolver(
        psi, 0.0,
        bcs={"top": {"dtn": CylindricalDtN(M=4), "sigma": fd.cos(m_mode * phi)},
             "bottom": {"dtn": CylindricalDtN(M=4)}},
        dtn_representation="lowrank")
    solver.solve()
    amplitude = 2 * np.pi * RMAX / m_mode
    exact = amplitude * (r / RMAX) ** m_mode * fd.cos(m_mode * phi)
    error = relative_l2_error(psi, exact, annulus)
    print(f"    [closed form] {error:.3e}")
    assert error < 1e-4
    assert abs(solver.coefficients()["top"][f"cos{m_mode}"] / amplitude - 1) < 1e-4


# ---------------------------------------------------------------------------
# The operator itself
# ---------------------------------------------------------------------------
def test_repeated_application_is_stable(annulus):
    """Applying `A + B` repeatedly must give the same answer every time.

    Guards the defect this code area has produced before: an operator correct
    on its first application and wrong afterwards, with Krylov converging
    happily against the corrupted version. The low-rank apply accumulates into
    its output vector, so a missing zero would show exactly this signature.
    """
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 1))
    solver = GravitySolver(
        psi, annulus_source(annulus),
        bcs={"top": {"dtn": CylindricalDtN(M=3)},
             "bottom": {"dtn": CylindricalDtN(M=3)}},
        dtn_representation="lowrank")

    x = solver.N.createVecRight()
    low, high = x.getOwnershipRange()
    x.setValues(range(low, high),
                np.random.default_rng(42).standard_normal(x.getSize())[low:high])
    x.assemble()

    first = solver.N.createVecLeft()
    solver.N.mult(x, first)
    for application in range(3):
        y = solver.N.createVecLeft()
        solver.N.mult(x, y)
        y.axpy(-1.0, first)
        assert y.norm() <= 1e-14 * first.norm(), (
            f"application {application} differs from the first")


def test_operator_is_symmetric(annulus):
    """`<y, (A+B) x> == <x, (A+B) y>` to round-off.

    The elimination is only legitimate if what it leaves is symmetric; and the
    adjoint work in step 8 rests on `K^T = K`, so this is the property that
    makes the adjoint the same solve rather than a second one.
    """
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 1))
    solver = GravitySolver(
        psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=3)},
                       "bottom": {"dtn": CylindricalDtN(M=3)}},
        dtn_representation="lowrank")
    rng = np.random.default_rng(7)
    vectors = []
    for seed in range(2):
        v = solver.N.createVecRight()
        low, high = v.getOwnershipRange()
        v.setValues(range(low, high),
                    rng.standard_normal(v.getSize())[low:high])
        v.assemble()
        vectors.append(v)
    x, y = vectors
    Ax, Ay = solver.N.createVecLeft(), solver.N.createVecLeft()
    solver.N.mult(x, Ax)
    solver.N.mult(y, Ay)
    left, right = y.dot(Ax), x.dot(Ay)
    print(f"    [symmetry] {abs(left - right) / abs(left):.3e}")
    assert abs(left - right) <= 1e-12 * abs(left)


def test_low_rank_term_is_not_a_no_op(annulus):
    """The operator must actually differ from `A` alone.

    Without this the parity tests could pass on a build where `B` never
    contributed, provided the multiplier path were broken the same way. It is
    the negative control for the whole fast path.
    """
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 1))
    solver = GravitySolver(
        psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=3)},
                       "bottom": {"dtn": CylindricalDtN(M=3)}},
        dtn_representation="lowrank")
    x = solver.N.createVecRight()
    x.set(1.0)
    full, stiffness = solver.N.createVecLeft(), solver.N.createVecLeft()
    solver.N.mult(x, full)
    solver.A.petscmat.mult(x, stiffness)
    difference = full.copy()
    difference.axpy(-1.0, stiffness)
    print(f"    [B is live] |Bx|/|Ax| = {difference.norm() / stiffness.norm():.3e}")
    assert difference.norm() > 1e-6 * stiffness.norm()
    assert solver.operator_context.applications >= 1


#: Filled by `test_cg_iteration_counts`, read by the verdict test below. The
#: two refinements have to be compared with each other, and a single test
#: cannot render that verdict.
CG_GAPS = {}


def multiplier_potential_iterations(solver):
    """Iterations of the multiplier path's potential block on its last solve.

    Reached through `DtNTwoBlockSchurPC`'s inner fieldsplit rather than off the
    outer KSP, whose count is 1 - a full Schur factorisation is exact, so the
    outer number says nothing about how hard the potential block was.
    """
    inner = solver.solver.snes.ksp.pc.getPythonContext().pc
    potential, _ = inner.getFieldSplitSchurGetSubKSP()
    return potential.getIterationNumber()


@pytest.mark.parametrize("refinement", [2, 3])
def test_cg_iteration_counts(refinement):
    """CG counts on the fast path against the multiplier path's fieldsplit_0.

    Reported at two refinements deliberately. `B` lowers the smallest
    eigenvalue, so "multigrid on `A` alone" preconditions a spectrum `B` has
    moved, and the question is not whether the counts match but whether the gap
    GROWS with the mesh - a fixed offset is a constant, a growing one is a
    broken preconditioner, and at a single configuration those are the same
    number.

    The instrument is coarse and that is stated rather than hidden: the
    multiplier path runs about eight iterations here, so its resolution is
    roughly one part in eight. It can see the sqrt(2.2)-ish effect predicted
    from the measured eigenvalue shift; it could not see a 20% degradation. So
    the counts are integers, and a difference of one or two is not a result in
    either direction. The assertion is on the *gap not growing*, which is the
    property that distinguishes a constant from a broken preconditioner.
    """
    mesh = shell_mesh_3d(refinement_level=refinement, dr=0.25)
    rho = shell_source(mesh)

    def build(representation):
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, rho,
            bcs={"top": {"dtn": SphericalDtN(L=3)},
                 "bottom": {"dtn": SphericalDtN(L=3)}},
            dtn_representation=representation, solver_parameters="iterative")
        solver.mixed_solution.assign(0)
        solver.solve()
        return solver

    fast = build("lowrank")
    slow = build("multiplier")
    fast_iterations = fast.ksp.getIterationNumber()
    slow_iterations = multiplier_potential_iterations(slow)
    print(f"    [cg refinement={refinement}] "
          f"dofs={fast.solution_space.dim()} "
          f"lowrank CG {fast_iterations}   multiplier fieldsplit_0 "
          f"{slow_iterations}   gap {fast_iterations - slow_iterations}")
    assert 0 < fast_iterations < 200
    CG_GAPS[refinement] = fast_iterations - slow_iterations


def test_cg_gap_does_not_grow_with_refinement():
    """The verdict the two-refinement comparison exists to render.

    A fast path costing a fixed handful of extra iterations at every refinement
    is a constant and acceptable; one whose count grows with `N` is a broken
    preconditioner. Only two configurations separate those, which is why this
    is a separate assertion rather than a remark on a single number.
    """
    gaps = CG_GAPS
    if set(gaps) < {2, 3}:
        pytest.skip("needs both refinements from test_cg_iteration_counts")
    print(f"    [cg gap] refinement 2: {gaps[2]}   refinement 3: {gaps[3]}")
    assert gaps[3] <= gaps[2] + 2


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------
def test_parallel_parity(annulus):
    """Same comparison on more than one rank.

    The low-rank apply is the only place in this path where ranks have to
    agree, and it does so through a single `Allreduce` of the mode vector; an
    ownership mistake there is invisible in serial.
    """
    if fd.COMM_WORLD.size < 2:
        pytest.skip("needs >= 2 MPI ranks; run under mpiexec")
    rho = annulus_source(annulus)
    results = solve_both(
        annulus, rho,
        lambda: {"top": {"dtn": CylindricalDtN(M=5)},
                 "bottom": {"dtn": CylindricalDtN(M=5)}})
    compare(annulus, results, label=f"parallel np={fd.COMM_WORLD.size}")


def test_parallel_coefficients_are_rank_consistent(annulus):
    """Every rank must report the same trace spectrum.

    `coefficients()` sums rank-local dot products, so a rank that skipped the
    reduction would return its own partial sum and nothing would complain.
    """
    if fd.COMM_WORLD.size < 2:
        pytest.skip("needs >= 2 MPI ranks; run under mpiexec")
    psi = fd.Function(fd.FunctionSpace(annulus, "CG", 1))
    solver = GravitySolver(
        psi, annulus_source(annulus),
        bcs={"top": {"dtn": CylindricalDtN(M=4)},
             "bottom": {"dtn": CylindricalDtN(M=4)}},
        dtn_representation="lowrank")
    solver.solve()
    local = solver.coefficients()
    gathered = fd.COMM_WORLD.allgather(local)
    for other in gathered[1:]:
        for bc_id in local:
            for key in local[bc_id]:
                assert local[bc_id][key] == other[bc_id][key]


# ---------------------------------------------------------------------------
# Cross-mesh: density on a Submesh, potential on the parent
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def submesh_pair():
    mesh = fd.UnitDiskMesh(refinement_level=3)
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    indicator = fd.Function(DG0).interpolate(fd.conditional(r < 0.55, 1.0, 0.0))
    marked = fd.RelabeledMesh(mesh, [indicator], [99])
    return marked, fd.Submesh(marked, marked.topological_dimension, 99)


def test_submesh_parity_and_the_mixed_space_disappearing(submesh_pair):
    """The cross-mesh configuration, and the structural claim behind it.

    The mixed space exists on the multiplier path for two reasons: one Real
    field per mode, and a DG0 dummy whose only job is to make Firedrake set up
    the cross-mesh entity maps, needed because the *bilinear* form has
    arguments on two meshes. With the multipliers eliminated there is no
    bilinear form spanning two meshes at all - only the linear source term
    `4 pi G rho v dx` with `rho` on the submesh and `v` on the parent - so both
    reasons go away together.

    That is asserted rather than assumed, because it is the part of the plan
    that was predicted from reading rather than measured: the multiplier path
    carries 8 sub-spaces here (psi + dummy + 6 multipliers) and the fast path
    carries one.
    """
    marked, submesh = submesh_pair
    Xs = fd.SpatialCoordinate(submesh)
    phis = fd.atan2(Xs[1], Xs[0])
    rho = fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(
        fd.cos(2 * phis))

    solvers, potentials = {}, {}
    for representation in ("multiplier", "lowrank"):
        psi = fd.Function(fd.FunctionSpace(marked, "CG", 2))
        solver = GravitySolver(psi, rho, bcs={1: {"dtn": CylindricalDtN(M=3)}},
                               dtn_representation=representation)
        assert solver.cross_mesh
        solver.mixed_solution.assign(0)
        solver.solve()
        solvers[representation], potentials[representation] = solver, psi

    assert len(solvers["multiplier"].mixed_space) == 8
    assert solvers["multiplier"].n_multipliers == 6
    assert solvers["multiplier"]._multiplier_offset == 2  # psi, then the dummy
    # No mixed space, no dummy, no multipliers.
    assert solvers["lowrank"].mixed_space is solvers["lowrank"].solution_space
    assert solvers["lowrank"].n_multipliers == 0

    error = relative_l2_error(potentials["lowrank"], potentials["multiplier"],
                              marked)
    worst = max(
        abs(solvers["multiplier"].coefficients()[b][k]
            - solvers["lowrank"].coefficients()[b][k])
        for b in solvers["multiplier"].coefficients()
        for k in solvers["multiplier"].coefficients()[b])
    print(f"    [submesh] psi {error:.3e}   coefficients {worst:.3e}")
    assert error <= 1e-14
    assert worst <= 1e-12
