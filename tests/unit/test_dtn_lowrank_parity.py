"""The low-rank fast path against the multiplier path it must reproduce.

Both paths discretise the *same* operator, so they must agree to round-off, and
that is the acceptance test for the representation - accuracy against a closed
form would not distinguish them, since both are equally right or equally wrong
about the physics.

The gates are the plan's: potentials <= 1e-14 relative, trace coefficients
<= 1e-12. They are met by four to five orders on every configuration here, which
is the useful signal - a change that broke the elimination would not land at
1e-13, it would land at 1e-4.

The 2-D parity problems (two DtN boundaries, a sheet, a strong boundary
condition), the monopole datum on the fast path, the accuracy against the
closed form and the submesh parity are rows of the `closed-2d-lowrank` and
`submesh` entries of the `gravity` case in `tests/gia` (driver
`gia_gravity.py`), which imports `solve_both`, `shell_source`, `tight` and
`multiplier_potential_iterations` from here. The 3-D parity and the CG counts
stay here until the CI-only entries `closed-3d-*` have run.

The repeated-application check exists because a wrong operator in this code
area has previously been right once and wrong afterwards. The low-rank operator
sums rank-local dot products through one `Allreduce`, so a rank-ownership
mistake shows only on several ranks; the parity on two ranks is the entry
`gravity-np2` of the `parallel` case in `tests/gia`, which uses `solve_both`.
"""

import numpy as np
import pytest
import firedrake as fd

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from test_gravity_solver import annulus_mesh, relative_l2_error, shell_mesh_3d


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


@pytest.mark.longtest
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


@pytest.mark.longtest
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


@pytest.mark.longtest
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
# Cross-mesh: density on a Submesh, potential on the parent
# ---------------------------------------------------------------------------
