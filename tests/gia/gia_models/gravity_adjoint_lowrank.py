"""Helpers of `tests/unit/test_gravity_adjoint_lowrank.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gravity_adjoint_lowrank.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
from firedrake.adjoint import stop_annotating
from gadopt import CylindricalDtN, GravitySolver, SphericalDtN


LOW = dict(dtn_representation="lowrank")


# ---------------------------------------------------------------------------
# Forward maps, mirroring test_gravity_adjoint's but on the fast path
# ---------------------------------------------------------------------------
def forward_rho_2d(m_rho, gravitational_constant=1.0):
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}},
            gravitational_constant=gravitational_constant, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_rho_3d(m_rho, gravitational_constant=1.0):
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": SphericalDtN(L=1)},
                 "bottom": {"dtn": SphericalDtN(L=1)}},
            gravitational_constant=gravitational_constant, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_no_multiplier(m_rho, bc_variant):
    """The two configurations with no DtN modes at all.

    `dirichlet2d` has no DtN boundary, so the low-rank path degenerates to a
    plain assembled Poisson solve with an empty mode set - worth covering
    because the `C` machinery must survive having nothing to build.
    """
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        bcs = ({"top": {"psi": 0.0}} if bc_variant == "dirichlet2d"
               else {"top": {"dtn": CylindricalDtN(M=0)}})
        solver = GravitySolver(psi, m_rho, bcs=bcs, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_sheet(sigma_fn, geometry):
    mesh = sigma_fn.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        if geometry == "2d":
            bcs = {"top": {"dtn": CylindricalDtN(M=2), "sigma": sigma_fn},
                   "bottom": {"dtn": CylindricalDtN(M=2)}}
        else:
            bcs = {"top": {"dtn": SphericalDtN(L=1), "sigma": sigma_fn},
                   "bottom": {"dtn": SphericalDtN(L=1)}}
        solver = GravitySolver(psi, 0.0, bcs=bcs, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_G(G_const, geometry, rho_fixed):
    if geometry == "2d":
        return forward_rho_2d(rho_fixed, G_const)
    return forward_rho_3d(rho_fixed, G_const)


# ---------------------------------------------------------------------------
# Transpose consistency and sabotage detection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cross-mesh
# ---------------------------------------------------------------------------
def forward_submesh(m_rho_sub, marked):
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(marked, "CG", 1))
        solver = GravitySolver(psi, m_rho_sub,
                               bcs={1: {"dtn": CylindricalDtN(M=2)}}, **LOW)
    assert solver.cross_mesh
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver
