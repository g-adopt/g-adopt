"""Regression tests for the gravity flux on weakly imposed Dirichlet head boundaries.

The total flux is ``K grad(h + z)``, but ``scalar_equation.diffusion_term`` builds
its Nitsche consistency term from ``h`` alone, so ``richards_gravity_term`` has to
contribute the ``K grad(z).n`` half on any boundary carrying a weak Dirichlet head.
Omitting it leaves a spurious ``+int K grad(z).n v ds`` in the residual -- exactly
the free-drainage flux that should be crossing that boundary -- and the scheme is
inconsistent there.

Hydrostatic equilibrium is the guard. With ``h = C - z`` the total potential is
constant, so ``q = -K grad(h + z) = 0`` identically and this is an exact steady
solution for *any* soil curve, with no reference data and no special functions
needed. A consistent scheme must return a zero residual at that field, and a
steady solve started from it must stay there.

Two controls run alongside: an unspecified (natural) boundary and ``flux = 0``.
Both are already correct without any gravity boundary term -- for them, carrying
none is precisely the right natural condition ``q.n = 0`` -- so they guard against
a "fix" that adds the term unconditionally on every boundary.

Continuous spaces are not covered here: ``RichardsSolver`` imposes a head boundary
on them with a strong ``DirichletBC``, so no weak boundary term is generated and
none is needed. The boundary rows never see the residual.
"""
import numpy as np
import pytest
from firedrake import (
    RectangleMesh, FunctionSpace, Function, SpatialCoordinate, TestFunction,
    Constant, assemble, dx,
)

from gadopt import ExponentialCurve, RichardsSolver, BackwardEuler
from gadopt import richards_equation as richards_eq
from gadopt import scalar_equation as scalar_eq
from gadopt.equations import Equation, cell_edge_integral_ratio

# h = C - z on a 1 x 2 m column. C is well into the unsaturated range so K(h)
# stays far from Ks and the exponential curve is exercised away from its cap.
C = -5.0
LX, LZ = 1.0, 2.0
ALPHA, KS = 0.25, 1.0e-5


def _soil():
    return ExponentialCurve(theta_r=0.15, theta_s=0.45, alpha=ALPHA, Ks=KS, Ss=0.0)


def _K_scale():
    """K at the driest point of the boundary, used to normalise the residual.

    The residual scales with K, so the raw number tracks Ks and tells you nothing
    on its own. Dividing by this makes the tolerance independent of the soil.
    """
    return KS * np.exp(ALPHA * (C - LZ))


def _setup(nodes, family, degree):
    mesh = RectangleMesh(nodes, nodes, LX, LZ, quadrilateral=True)
    mesh.cartesian = True
    _, z = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, family, degree)
    exact = C - z
    return mesh, V, exact, Function(V).interpolate(exact)


def _bcs(mode, exact):
    """Boundary dictionaries in `diffusion_term` vocabulary ('q', not 'h').

    RichardsSolver translates the user-facing 'h' into 'q' in
    `set_boundary_conditions`; assembling the Equation directly means using the
    internal key.
    """
    ids = (1, 2, 3, 4)
    return {
        "natural": {},
        "flux_zero": {i: {"flux": 0.0} for i in ids},
        "dirichlet": {i: {"q": exact} for i in ids},
    }[mode]


def _residual(nodes, family, degree, mode):
    """max |residual| at the exact hydrostatic field, normalised by boundary K."""
    mesh, V, exact, h = _setup(nodes, family, degree)

    # Mirrors the penalty RichardsSolver builds: scalar diffusion uses shift=-1,
    # rescaled to the shift=0 constant. See RichardsSolver.set_equation.
    penalty = 2.0 * (
        cell_edge_integral_ratio(mesh, max(degree, 1))
        / cell_edge_integral_ratio(mesh, max(degree - 1, 0))
    )
    soil = _soil()
    eq = Equation(
        TestFunction(V), V,
        residual_terms=[scalar_eq.diffusion_term, richards_eq.richards_gravity_term],
        eq_attrs={
            "soil_curve": soil,
            "diffusivity": soil.hydraulic_conductivity(h),
            "interior_penalty": penalty,
        },
        bcs=_bcs(mode, exact),
    )
    residual = assemble(eq.residual(h))
    return float(np.abs(residual.dat.data_ro).max()) / _K_scale()


@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("mode", ["natural", "flux_zero", "dirichlet"])
def test_hydrostatic_residual_vanishes(degree, mode):
    """The hydrostatic field is an exact steady solution, so its residual is zero.

    DQ0 cannot represent the linear field h = C - z, so it carries a
    representation error that is not the boundary term under test; it is checked
    only for consistency across the three boundary configurations, which is what
    isolates the Dirichlet path.
    """
    got = _residual(8, "DQ", degree, mode)

    if degree == 0:
        natural = _residual(8, "DQ", 0, "natural")
        assert got == pytest.approx(natural, rel=1e-6), (
            f"DQ0 {mode} residual {got:.3e} differs from the natural-boundary "
            f"value {natural:.3e}; all three configurations must agree, since "
            f"what remains at DQ0 is representation error, not a boundary defect"
        )
    else:
        assert got < 1e-10, (
            f"DQ{degree} {mode}: max|residual|/K = {got:.3e} at the exact "
            f"hydrostatic field. A consistent scheme returns zero here; a "
            f"non-zero value on 'dirichlet' is the missing gravity boundary "
            f"flux, and one on 'natural'/'flux_zero' means the term is being "
            f"applied where it does not belong"
        )


@pytest.mark.parametrize("degree", [1, 2])
def test_hydrostatic_steady_solve_stays_put(degree):
    """Started at the exact answer with Dirichlet data, the solve must not move.

    With the boundary term missing this drifts to a mesh-scale error that is
    independent of K -- the missing flux and the Nitsche penalty both carry K,
    so it cancels -- and converges at about 1.5 regardless of degree, capping
    the attainable order wherever a Dirichlet head is active.
    """
    _, _, exact, h = _setup(8, "DQ", degree)
    dt = Constant(1.0e4)
    solver = RichardsSolver(
        h, _soil(), delta_t=dt, timestepper=BackwardEuler,
        bcs={i: {"h": exact} for i in (1, 2, 3, 4)},
        solver_parameters="direct",
        quad_degree=2 * degree + 4,
        interior_penalty=2.0,
    )
    for _ in range(20):
        solver.solve()
        dt.assign(float(dt) * 1.5)

    error = np.sqrt(assemble((h - exact) ** 2 * dx(
        metadata={"quadrature_degree": 2 * degree + 4})))
    assert error < 1e-12, (
        f"DQ{degree}: hydrostatic solve drifted to L2 error {error:.3e} m from "
        f"an exact initial state; expected it to stay at machine precision"
    )
