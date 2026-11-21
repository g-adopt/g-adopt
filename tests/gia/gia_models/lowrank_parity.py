"""Helpers of `tests/unit/test_dtn_lowrank_parity.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_dtn_lowrank_parity.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
from gadopt import GravitySolver


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


def multiplier_potential_iterations(solver):
    """Iterations of the multiplier path's potential block on its last solve.

    Reached through `DtNTwoBlockSchurPC`'s inner fieldsplit rather than off the
    outer KSP, whose count is 1 - a full Schur factorisation is exact, so the
    outer number says nothing about how hard the potential block was.
    """
    inner = solver.solver.snes.ksp.pc.getPythonContext().pc
    potential, _ = inner.getFieldSplitSchurGetSubKSP()
    return potential.getIterationNumber()
