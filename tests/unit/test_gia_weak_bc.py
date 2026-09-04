r"""Weak boundary conditions of the viscoelastic (GIA) momentum equation.

The glacial-isostatic-adjustment solvers impose a no-normal-displacement
condition weakly, through the Nitsche/SIPG terms in
`gadopt.momentum_equation.viscosity_term`. Three of those terms depend on the
stress the equation carries: the flux, the symmetrising term (which uses the
tangent of that stress in the direction of the test function) and the penalty
(which uses the approximation's stress-from-gradient helper). This file checks
the properties those terms must have for the viscoelastic approximations.

The two internal-variable solvers linearise differently and that is the point
of most of the tests here. `InternalVariableSolver` substitutes the
backward-Euler update of the internal variables into the stress, so the stress
seen by the momentum equation has shear coefficient
$\eta_{eff} = \sum_i \eta_i/(\tau_i + \Delta t)$.
`CoupledInternalVariableSolver` keeps the internal variables as unknowns, so at
fixed history the same stress has shear coefficient $\mu_0 = \sum_i \mu_i$, the
elastic modulus. The symmetrising term must carry whichever of the two the
stress actually carries, or the displacement block of the Jacobian is
asymmetric by $(\mu_0 - \eta_{eff})(C - C^T)$ with $C$ the boundary consistency
term. That block is preconditioned with CG in the coupled preset, so the
asymmetry is not merely cosmetic.

The penalty coefficient stays at $\eta_{eff}$ in both solvers. Raising it to
$\mu_0$ would over-penalise the problem obtained by eliminating the internal
variables by a factor $1 + \Delta t/\tau$.

Fixtures and reference helpers are shared with `test_symmetry.py`, which holds
the same checks for the mantle-convection approximations.
"""

from math import log2, sqrt
from pathlib import Path

import firedrake as fd
import gadopt
import numpy as np
import pytest

from gadopt.equations import Equation
from gadopt.momentum_equation import viscosity_term
from test_symmetry import (
    GIA_BULK_MODULUS,
    GIA_BULK_SHEAR_RATIO,
    GIA_DT,
    WEAK_UN_VALUE,
    assert_symmetric,
    dev_stress_per_mu,
    exterior_facet_form,
    generic_velocity,
    meshes,
    penalty_coefficient,
    raw_effective_viscosity,
    raw_internal_variable_stress,
    raw_internal_variables_update,
)

# Elastic shear modulus and viscosity of the single Maxwell element used
# throughout, chosen so the Maxwell time tau = viscosity / shear_modulus is 1
# and dt / tau is simply dt.
SHEAR_MODULUS = 2.0
VISCOSITY = 2.0
MAXWELL_TIME = VISCOSITY / SHEAR_MODULUS
# Directory holding the stored Jacobian fingerprint.
DATA_DIR = Path(__file__).parent.resolve() / "data"


def maxwell_approximation(mesh, *, exponent=1, B_mu=1.27):
    """A single-element compressible Maxwell approximation on `mesh`.

    The density is a DG0 field so that the buoyancy term, which differentiates
    it, is well defined. `exponent` selects Newtonian (1) or composite creep
    rheology; the power-law factor multiplies the Maxwell times in the internal
    variable equations only, and never reaches the momentum stress.
    """
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    return gadopt.MaxwellApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        SHEAR_MODULUS,
        VISCOSITY,
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        exponent=exponent,
        transition_stress=5.0,
        B_mu=B_mu,
    )


def history_state(mesh, space, factor=0.1):
    """A nonzero, anisotropic internal-variable field.

    A zero history would make every term proportional to the internal variables
    vanish, hiding a wrong coefficient in front of them.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    return fd.Function(space).interpolate(
        factor * fd.sym(fd.outer(X, fd.as_vector([float(i + 1) for i in range(dim)])))
    )


def weak_un_functional(eq, u, boundary_ids, un, *, stress, mu_penalty, bulk):
    r"""The boundary functional whose first variation is the weak "un" residual.

    $$ E = \int_\Gamma \left[ -w_n\,(n \cdot \sigma(u)\,n)
       + \sigma_{pen} \left\langle G,\ \mu A(G)
       + \kappa_r \kappa\,\mathrm{tr}(G)\,I \right\rangle \right] ds $$

    with $w_n = n \cdot u - u_n$, $G = n \otimes w_n n$ and $A$ the deviatoric
    stress per $\mu$. The penalty part is a quadratic form in $G$ built from a
    self-adjoint operator, so its variation contributes twice, which is where
    the factor 2 in the penalty residual comes from.

    Args:
      eq: the `Equation` supplying the measures and the facet normal.
      u: the displacement the functional is evaluated at.
      boundary_ids: the boundaries carrying the weak condition.
      un: the prescribed normal component.
      stress: the full stress, written out from raw approximation attributes.
      mu_penalty: the shear coefficient of the penalty.
      bulk: the bulk coefficient of the penalty.

    Returns:
      A UFL form for the boundary functional.
    """
    n = eq.n
    dim = eq.mesh.geometric_dimension
    sigma = penalty_coefficient(eq)

    normal_jump = fd.dot(n, u) - un
    G = fd.outer(n, normal_jump * n)
    penalty_stress = (
        mu_penalty * dev_stress_per_mu(G, True)
        + bulk * fd.tr(G) * fd.Identity(dim)
    )
    integrand = (
        -normal_jump * fd.dot(n, fd.dot(stress, n))
        + sigma * fd.inner(G, penalty_stress)
    )
    return sum(integrand * eq.ds(bid) for bid in boundary_ids)


def assert_first_variation(form, functional, solution, rtol=1e-12):
    """Assert `form` is the first variation of `functional` at `solution`."""
    variation = fd.derivative(functional, solution)
    residual = fd.assemble(form - variation)
    reference = fd.assemble(variation)
    assert residual.dat.norm <= rtol * reference.dat.norm


@pytest.mark.parametrize("exponent", [1, 3])
@pytest.mark.parametrize("dt_over_tau", [0.25, 25.0])
@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet"])
def test_coupled_displacement_block_symmetry(mesh_key, dt_over_tau, exponent):
    """The displacement block of the coupled Jacobian must be symmetric.

    `CoupledInternalVariableSolver` solves for the displacement and the internal
    variables together, so at fixed history the momentum stress is elastic: its
    shear coefficient is the elastic modulus $\\mu_0$, not the effective
    viscosity. A symmetrising term built with the effective viscosity leaves the
    (0,0) block asymmetric by $(\\mu_0 - \\eta_{eff})(C - C^T)$, an error that
    grows with $\\Delta t/\\tau$. The default coupled preset preconditions that
    block with CG, which assumes a symmetric operator.

    Only the displacement block is expected to be symmetric. The full coupled
    Jacobian is not: the internal-variable rows are scaled independently, and
    for composite creep the Maxwell times depend on the stress.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    Z = V * S

    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    z.subfunctions[1].assign(history_state(mesh, S))

    approximation = maxwell_approximation(mesh, exponent=exponent)
    bids = list(gadopt.get_boundary_ids(mesh))
    bcs = {bids[0]: {"un": WEAK_UN_VALUE}, bids[1]: {"free_surface": {}}}
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=dt_over_tau * MAXWELL_TIME, bcs=bcs,
        solver_parameters="direct",
    )

    jacobian = fd.assemble(fd.derivative(solver.F, z), mat_type="nest")
    displacement_block = jacobian.petscmat.getNestSubMatrix(0, 0).convert("aij")
    assert_symmetric(displacement_block, rtol=1e-13)


@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet"])
def test_pointwise_variational_structure(mesh_key):
    """The pointwise weak "un" residual is the first variation of its functional.

    Symmetry alone cannot see a mis-scaled penalty or a wrong constant in a
    term that is still the first variation of some functional. Pinning the
    residual to the functional the code documents can. Here the functional is
    written with the full internal-variable stress (bulk part and history
    included) rebuilt from raw attributes, and with the penalty coefficient
    held at the effective viscosity.

    `InternalVariableSolver` substitutes the backward-Euler update into the
    stress, so differentiating the functional also differentiates through that
    update. The result carries the effective viscosity, which is what makes the
    pointwise symmetrising term and the pointwise penalty share a coefficient.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    m = history_state(mesh, S)
    approximation = maxwell_approximation(mesh)
    bids = list(gadopt.get_boundary_ids(mesh))[:2]
    solver = gadopt.InternalVariableSolver(
        u, approximation, dt=GIA_DT, internal_variables=m,
        bcs={bid: {"un": WEAK_UN_VALUE} for bid in bids},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    updated = raw_internal_variables_update(approximation, u, [m], GIA_DT)
    functional = weak_un_functional(
        eq, u, bids, WEAK_UN_VALUE,
        stress=raw_internal_variable_stress(approximation, u, updated),
        mu_penalty=raw_effective_viscosity(approximation, GIA_DT),
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    assert_first_variation(form, functional, u)


@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet"])
def test_coupled_variational_structure(mesh_key):
    """The coupled weak "un" residual is the first variation at fixed history.

    In the coupled formulation the internal variables are unknowns of the
    system, so the displacement rows of the residual must be the first
    variation of the boundary functional taken with the history held fixed.
    The stress in that functional is elastic in the displacement, so its
    variation carries $\\mu_0$; the penalty stays at the effective viscosity.
    The reference therefore differs from the pointwise one only through the
    stress, which is exactly the difference the code has to pick up
    automatically.

    The history is supplied as separate Functions holding the same values as the
    internal-variable component of the solution, so that differentiating the
    functional with respect to the mixed solution varies the displacement only.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    Z = V * S

    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    z.subfunctions[1].assign(history_state(mesh, S))
    frozen_history = fd.Function(S).assign(z.subfunctions[1])

    approximation = maxwell_approximation(mesh)
    bids = list(gadopt.get_boundary_ids(mesh))[:2]
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=GIA_DT,
        bcs={bid: {"un": WEAK_UN_VALUE} for bid in bids},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    u = solver.solution_split[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    functional = weak_un_functional(
        eq, u, bids, WEAK_UN_VALUE,
        stress=raw_internal_variable_stress(approximation, u, [frozen_history]),
        mu_penalty=raw_effective_viscosity(approximation, GIA_DT),
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    assert_first_variation(form, functional, z)


@pytest.mark.parametrize("history", ["pointwise", "elastic"])
@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet"])
def test_weak_u_symmetry(mesh_key, history):
    """Symmetry of the weak "u" branch for the viscoelastic stress.

    Every `StokesSolverBase` subclass turns a "u" boundary condition into a
    strong `DirichletBC`, so the weak "u" branch is reachable only by driving
    `viscosity_term` at the `Equation` level, which is what this test does. The
    branch has to work for a stress with a bulk part: the tangent and the
    penalty both pick that part up, and the residual is the first variation of
    a boundary functional, so the Jacobian is a Hessian and symmetric.

    Both linearisations are covered. "pointwise" substitutes the backward-Euler
    update into the stress, as `InternalVariableSolver` does. "elastic" holds
    the history fixed, as the displacement rows of
    `CoupledInternalVariableSolver` see it.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = True
    dim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    m = history_state(mesh, S)
    approximation = maxwell_approximation(mesh)
    # The solvers set this before assembly; here the Equation is driven
    # directly, so the effective viscosity is supplied the same way.
    approximation.mu = approximation.effective_viscosity(GIA_DT)

    if history == "pointwise":
        internal_variables = raw_internal_variables_update(
            approximation, u, [m], GIA_DT
        )
    else:
        internal_variables = [m]
    stress = approximation.stress(u, internal_variables=internal_variables)

    bids = list(gadopt.get_boundary_ids(mesh))
    # Exercise both weak branches at once.
    bcs = {
        bids[0]: {"u": fd.Constant([0.1 * (i + 1) for i in range(dim)])},
        bids[1]: {"un": WEAK_UN_VALUE},
    }
    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={"stress": stress},
        approximation=approximation,
        bcs=bcs,
        quad_degree=6,
    )
    jacobian = fd.assemble(fd.derivative(eq.residual(u), u), mat_type="aij")
    assert_symmetric(jacobian.petscmat, rtol=1e-13)


def solve_surface_load(solver_kind, mesh, dt):
    """Solve one viscoelastic step under a surface load and return displacement.

    A unit square is loaded by a normal stress on the top boundary, held fixed
    at the bottom, and given a weak no-normal-displacement condition on the two
    sides. Cells are affine, so the pointwise and coupled formulations
    discretise the same continuous problem and differ only through the weak
    boundary terms.

    Args:
      solver_kind: "pointwise" or "coupled".
      mesh: the mesh both formulations are solved on.
      dt: time step.

    Returns:
      The displacement Function.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    x, _ = fd.SpatialCoordinate(mesh)
    # A smooth, mean-free surface load; mean-free so the weak sides are not
    # asked to absorb a net normal force.
    load = fd.cos(2 * fd.pi * x)
    approximation = maxwell_approximation(mesh, B_mu=0.0)
    bcs = {
        1: {"un": 0.0},
        2: {"un": 0.0},
        3: {"uy": 0.0},
        4: {"normal_stress": load},
    }

    if solver_kind == "pointwise":
        u = fd.Function(V)
        solver = gadopt.InternalVariableSolver(
            u, approximation, dt=dt, internal_variables=fd.Function(S),
            bcs=bcs, solver_parameters="direct",
        )
        solver.solve()
        return u

    Z = V * S
    z = fd.Function(Z)
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=dt, bcs=bcs, solver_parameters="direct",
    )
    solver.solve()
    return z.subfunctions[0]


@pytest.mark.parametrize("dt_over_tau", [0.25, 25.0])
def test_coupled_matches_pointwise_under_refinement(dt_over_tau):
    """Bound the solution gap the coupled weak boundary term introduces.

    The coupled displacement rows carry the elastic tangent, so their residual
    differs from the pointwise one by a term proportional to the normal jump
    $n \\cdot u$ on the weak boundary. That term is consistent: it vanishes on
    the exact solution, so the two converged discrete solutions approach each
    other under refinement. This test measures that gap and the normal jump
    itself on two resolutions and requires both to fall at least at second
    order for the P2 displacement space.

    The gap is also required to be resolvable on the coarse mesh, at least
    $10^{-8}$ of the displacement itself. Without that bound the two rates
    would be computed from round-off and would mean nothing. The measured
    relative gap is about 2e-5 at $\\Delta t/\\tau = 0.25$ and about 1e-3 at
    $\\Delta t/\\tau = 25$, so the bound has a wide margin.

    This test bounds the size of the difference between the two formulations;
    it does not detect an asymmetric displacement block.
    `test_coupled_displacement_block_symmetry` does that.
    """
    dt = dt_over_tau * MAXWELL_TIME
    resolutions = (8, 16)
    gaps = []
    normal_jumps = []
    coarse_displacement_norm = None
    for resolution in resolutions:
        mesh = fd.UnitSquareMesh(resolution, resolution)
        mesh.cartesian = True
        u_pointwise = solve_surface_load("pointwise", mesh, dt)
        u_coupled = solve_surface_load("coupled", mesh, dt)
        difference = u_coupled - u_pointwise
        gaps.append(sqrt(fd.assemble(fd.inner(difference, difference) * fd.dx)))
        n = fd.FacetNormal(mesh)
        normal_jumps.append(
            sqrt(fd.assemble(fd.dot(n, u_coupled) ** 2 * (fd.ds(1) + fd.ds(2))))
        )
        if coarse_displacement_norm is None:
            coarse_displacement_norm = sqrt(
                fd.assemble(fd.inner(u_pointwise, u_pointwise) * fd.dx)
            )

    # The coarse gap must be a real difference between the two formulations and
    # not round-off, or the rates below are computed from noise.
    assert gaps[0] >= 1e-8 * coarse_displacement_norm, (
        f"gap {gaps[0]} is at round-off relative to |u| {coarse_displacement_norm}"
    )
    gap_rate = log2(gaps[0] / gaps[1])
    jump_rate = log2(normal_jumps[0] / normal_jumps[1])
    assert gap_rate >= 2.0, f"gaps {gaps}, rate {gap_rate}"
    assert jump_rate >= 2.0, f"normal jumps {normal_jumps}, rate {jump_rate}"


def test_coupled_iterative_preset_converges():
    """The coupled iterative preset must converge with weak "un" boundaries.

    The preset preconditions the displacement block with CG inside a
    fieldsplit. CG is only defined for a symmetric operator, so an asymmetric
    displacement block is a solver-level defect, not only an aesthetic one. A
    large ratio $\\Delta t/\\tau$ makes the asymmetry largest, so that is what is
    used here. Both the inner CG and the outer Newton solve must report
    convergence.
    """
    mesh = fd.UnitSquareMesh(8, 8)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    Z = V * S
    z = fd.Function(Z)

    x, _ = fd.SpatialCoordinate(mesh)
    approximation = maxwell_approximation(mesh, B_mu=0.0)
    bcs = {
        1: {"un": 0.0},
        2: {"un": 0.0},
        3: {"uy": 0.0},
        4: {"normal_stress": fd.cos(2 * fd.pi * x)},
    }
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=25.0 * MAXWELL_TIME, bcs=bcs,
        solver_parameters="iterative",
    )
    solver.solve()

    snes = solver.solver.snes
    assert snes.getConvergedReason() > 0, (
        f"SNES diverged, reason {snes.getConvergedReason()}"
    )
    displacement_ksp = snes.getKSP().getPC().getFieldSplitSubKSP()[0]
    assert displacement_ksp.getConvergedReason() > 0, (
        "fieldsplit displacement CG diverged, reason "
        f"{displacement_ksp.getConvergedReason()}"
    )


def fingerprint_jacobian():
    """Assemble the fixed pointwise weak-"un" Jacobian stored under `data/`.

    Kept as a module-level function so that the stored matrix and the matrix
    the test compares against come from the same definition.
    """
    mesh = fd.UnitSquareMesh(3, 2)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    m = history_state(mesh, S)
    approximation = maxwell_approximation(mesh)
    solver = gadopt.InternalVariableSolver(
        u, approximation, dt=GIA_DT, internal_variables=m,
        bcs={1: {"un": WEAK_UN_VALUE}, 2: {"un": 0.0}},
        solver_parameters="direct",
    )
    jacobian = fd.assemble(fd.derivative(solver.F, u), mat_type="aij")
    return jacobian.petscmat.convert("dense").getDenseArray().copy()


def test_pointwise_maxwell_jacobian_fingerprint():
    """The pointwise weak-"un" Jacobian must match a stored reference matrix.

    `data/gia_pointwise_maxwell_weak_un_jacobian.npy` is the reference matrix
    for the pointwise weak-"un" Jacobian on this mesh. It pins entries, not
    properties, so it catches a change that no symmetry or
    variational-structure test can see. Regenerate it with
    `fingerprint_jacobian()` only for a deliberate change to the pointwise
    boundary terms.

    The reference is a dense matrix in the degree-of-freedom ordering
    Firedrake produces for this mesh and element pair, so a change of that
    ordering also breaks this test. That is a deliberate trade: the test is
    meant to be sensitive.
    """
    stored = np.load(DATA_DIR / "gia_pointwise_maxwell_weak_un_jacobian.npy")
    computed = fingerprint_jacobian()
    assert computed.shape == stored.shape
    assert np.abs(computed - stored).max() <= 1e-14 * np.abs(stored).max()
