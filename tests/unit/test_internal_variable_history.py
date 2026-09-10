"""Tests for the combined internal-variable field and its solvers.

Every Maxwell element of an internal-variable rheology lives in one
discontinuous tensor field of shape `(n, d, d)` (see
`gadopt.internal_variable_equation`). These tests pin:

- the space helper and the slicing helpers for every accepted layout;
- the history residual on the combined field against the equation written
  out by hand, boundary term included;
- the symmetry of the condensed displacement operator, which is what lets
  the shipped preset run CG, and the fact that the boundary term is what
  makes it symmetric;
- the shipped static-condensation preset against a direct solve for one,
  two and three Maxwell elements (three is the case Firedrake's `SCPC`
  refuses in the per-field layout), with weak and with component-wise
  strong boundary conditions, and with a power-law rheology under Newton
  against the direct solve;
- the Krylov selection on the condensed operator: CG where the operator is
  symmetric, GMRES where the power-law tangent is not, and the asymmetry
  itself;
- the reuse of the condensed operator across time steps and its rebuild
  when the time step or a material field changes;
- the substituted solver on every history layout;
- the assembled boundary penalty under Slate against standard assembly,
  which guards the exterior-facet coefficient of `viscosity_term`.

The meshes are tiny: these are algebraic properties and small solves.
"""

import firedrake as fd
import numpy as np
import pytest
import ufl
from firedrake.formmanipulation import split_form
from firedrake.slate import Tensor

import gadopt
from gadopt.equations import Equation
from gadopt.internal_variable_equation import (
    assign_history_slices,
    history_slices,
    internal_variable_history_terms,
    internal_variable_space,
)
from gadopt.momentum_equation import viscosity_term


# Shear moduli and viscosities of up to three Maxwell elements. The Maxwell
# times differ so that a term applied to the wrong element is visible.
SHEAR_MODULI = [1.0, 2.0, 0.5]
VISCOSITIES = [2.0, 1.0, 1.5]
DT = 0.4


def square_mesh(n=4):
    mesh = fd.UnitSquareMesh(n, n)
    mesh.cartesian = True
    return mesh


def approximation(mesh, n_elements, exponent=1, B_mu=1.27):
    """A compressible internal-variable approximation with `n_elements` elements."""
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    return gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=3.0,
        density=fd.Function(DG0).assign(1),
        shear_modulus=SHEAR_MODULI[:n_elements],
        viscosity=VISCOSITIES[:n_elements],
        bulk_shear_ratio=1.5,
        B_mu=B_mu,
        exponent=exponent,
        transition_stress=0.05,
    )


def surface_load(mesh):
    X = fd.SpatialCoordinate(mesh)
    return 0.1 * fd.exp(-((X[0] - 0.5) ** 2) / 0.02)


def weak_bcs(mesh):
    """Weak normal conditions on three sides, a loaded free surface on top."""
    bids = list(gadopt.get_boundary_ids(mesh))
    return {
        bids[0]: {"un": 0},
        bids[1]: {"un": 0},
        bids[2]: {"un": 0},
        bids[3]: {"normal_stress": surface_load(mesh), "free_surface": {}},
    }


def strong_bcs(mesh):
    """Component-wise strong conditions, as the free-surface benchmarks use."""
    bids = list(gadopt.get_boundary_ids(mesh))
    return {
        bids[0]: {"ux": 0},
        bids[1]: {"ux": 0},
        bids[2]: {"uy": 0},
        bids[3]: {"normal_stress": surface_load(mesh), "free_surface": {}},
    }


def anisotropic_history(mesh, space, n_elements):
    """A nonzero, element-dependent history so that every term is exercised."""
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    blocks = [
        0.1 * (i + 1) * fd.sym(fd.outer(X, fd.as_vector([float(k + 1) for k in range(dim)])))
        for i in range(n_elements)
    ]
    return fd.Function(space).interpolate(fd.as_tensor(blocks))


def asymmetry(petscmat):
    transpose = petscmat.duplicate(copy=True)
    transpose.transpose()
    difference = petscmat.copy()
    difference.axpy(-1.0, transpose)
    return difference.norm() / petscmat.norm()


def condensed_operator(J, history_block=None):
    """Assemble `A_uu - A_uM A_MM^-1 A_Mu` with Slate from the Jacobian form `J`.

    `history_block` replaces the `(M, u)` block when given, so that a test
    can drop part of it.
    """
    blocks = dict(split_form(J))
    A00, A01, A11 = (Tensor(blocks[key]) for key in ((0, 0), (0, 1), (1, 1)))
    A10 = Tensor(blocks[(1, 0)] if history_block is None else history_block)
    return fd.assemble(A00 - A01 * A11.inv * A10).petscmat


# --------------------------------------------------------------------------
# Space and slicing helpers


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_elements", [1, 2, 3])
def test_internal_variable_space_layout(dim, n_elements):
    mesh = fd.UnitSquareMesh(2, 2) if dim == 2 else fd.UnitCubeMesh(2, 2, 2)
    symmetric = internal_variable_space(mesh, n_elements)
    full = internal_variable_space(mesh, n_elements, symmetric=False)
    assert symmetric.value_shape == (n_elements, dim, dim)
    assert full.value_shape == (n_elements, dim, dim)
    nodes = fd.FunctionSpace(mesh, "DG", 1).dim()
    assert full.dim() == nodes * n_elements * dim * dim
    assert symmetric.dim() == nodes * n_elements * dim * (dim + 1) // 2
    # The approximation knows its own element count.
    approx = approximation(mesh, n_elements)
    assert approx.internal_variable_space(mesh).dim() == symmetric.dim()


def test_internal_variable_space_on_hexahedra():
    base = fd.UnitSquareMesh(2, 2, quadrilateral=True)
    mesh = fd.ExtrudedMesh(base, 2)
    space = internal_variable_space(mesh, 2)
    assert space.value_shape == (2, 3, 3)
    assert space.finat_element.space_dimension() == 8


def test_internal_variable_space_rejects_no_elements():
    with pytest.raises(ValueError):
        internal_variable_space(square_mesh(), 0)


def test_history_slices_accept_every_layout():
    mesh = square_mesh()
    combined = fd.Function(internal_variable_space(mesh, 2))
    single = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1))
    assert [s.ufl_shape for s in history_slices(combined)] == [(2, 2), (2, 2)]
    assert history_slices(single) == [single]
    assert len(history_slices([single, single])) == 2
    assert len(history_slices([single, combined])) == 3
    with pytest.raises(ValueError):
        history_slices(fd.Function(fd.VectorFunctionSpace(mesh, "DG", 1)))


def test_assign_history_slices_round_trip():
    mesh = square_mesh()
    X = fd.SpatialCoordinate(mesh)
    # Linear in X, so DG1 holds both blocks exactly.
    blocks = [
        fd.sym(fd.outer(X, fd.as_vector([3.0, -1.0]))),
        2 * fd.sym(fd.outer(X, fd.as_vector([1.0, 2.0]))),
    ]

    combined = fd.Function(internal_variable_space(mesh, 2))
    assign_history_slices(combined, blocks)
    for block, stored in zip(blocks, history_slices(combined)):
        assert fd.assemble(fd.inner(stored - block, stored - block) * fd.dx) < 1e-28

    singles = [fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1)) for _ in blocks]
    assign_history_slices(singles, blocks)
    for block, stored in zip(blocks, singles):
        assert fd.assemble(fd.inner(stored - block, stored - block) * fd.dx) < 1e-28

    with pytest.raises(ValueError):
        assign_history_slices(combined, blocks[:1])


# --------------------------------------------------------------------------
# The history residual on the combined field


@pytest.mark.parametrize("bc_type", ["un", "u"])
def test_combined_history_residual_matches_hand_written_form(bc_type):
    """The three history terms equal the backward-Euler equation written out.

    One equation on the `(2, d, d)` field against the sum over elements of
    `w_i : ((m_i - m_i_old)/dt + m_i/tau_i - d(u)/tau_i)` plus the boundary
    term `w_i : d_Gamma/tau_i`, both written here from scratch.
    """
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = internal_variable_space(mesh, 2)
    approx = approximation(mesh, 2)
    X = fd.SpatialCoordinate(mesh)
    u = fd.Function(V).interpolate(X + 0.3 * X[0] * X)
    M = anisotropic_history(mesh, S, 2)
    M_old = fd.Function(S).assign(0.5 * M)
    bids = list(gadopt.get_boundary_ids(mesh))
    boundary_value = 0.3 if bc_type == "un" else fd.Constant([0.1, 0.2])
    bcs = {bids[0]: {bc_type: boundary_value}, bids[1]: {bc_type: boundary_value}}

    maxwell_times = [eta / mu for eta, mu in zip(VISCOSITIES[:2], SHEAR_MODULI[:2])]
    eq = Equation(
        fd.TestFunction(S),
        S,
        internal_variable_history_terms,
        eq_attrs={
            "maxwell_times": maxwell_times,
            "displacement": u,
            "dt": DT,
            "trial_old": M_old,
        },
        approximation=approx,
        bcs=bcs,
    )
    residual = fd.assemble(eq.residual(M))

    # The same equation written out. The deviatoric strain removes the trace
    # with a factor 1/3 in every dimension, as the approximation does.
    W = fd.TestFunction(S)
    n = fd.FacetNormal(mesh)
    identity = fd.Identity(2)
    strain = fd.sym(fd.grad(u)) - fd.tr(fd.grad(u)) / 3 * identity
    if bc_type == "un":
        jump = (fd.dot(n, u) - boundary_value) * n
    else:
        jump = u - boundary_value
    G = fd.outer(n, jump)
    boundary_strain = fd.sym(G) - fd.tr(G) / 3 * identity
    expected = 0
    for i, tau in enumerate(maxwell_times):
        expected += fd.inner(W[i, :, :], (M[i, :, :] - M_old[i, :, :]) / DT) * eq.dx
        expected += fd.inner(W[i, :, :], M[i, :, :] / tau) * eq.dx
        expected -= fd.inner(W[i, :, :], strain / tau) * eq.dx
        for bid in bcs:
            expected += fd.inner(W[i, :, :], boundary_strain / tau) * eq.ds(bid)
    expected = fd.assemble(expected)

    difference = residual.dat.data_ro - expected.dat.data_ro
    assert np.abs(difference).max() <= 1e-13 * np.abs(expected.dat.data_ro).max()


def test_history_terms_reject_a_wrong_element_count():
    mesh = square_mesh()
    S = internal_variable_space(mesh, 2)
    approx = approximation(mesh, 3)
    with pytest.raises(ValueError, match="Maxwell time"):
        Equation(
            fd.TestFunction(S),
            S,
            internal_variable_history_terms,
            eq_attrs={
                "maxwell_times": approx.maxwell_times,
                "displacement": fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)),
                "dt": DT,
                "trial_old": fd.Function(S),
            },
            approximation=approx,
        ).residual(fd.Function(S))


# --------------------------------------------------------------------------
# Symmetry of the condensed operator


def test_condensed_operator_symmetric_because_of_boundary_term():
    """With the boundary term the condensed operator is symmetric; without it, not.

    `dt/tau` is large so that the asymmetry a missing boundary term leaves is
    large too.
    """
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    approx = approximation(mesh, 2)
    S = approx.internal_variable_space(mesh)
    z = fd.Function(V * S)
    X = fd.SpatialCoordinate(mesh)
    z.subfunctions[0].interpolate(X + 0.3 * X[0] * X)
    z.subfunctions[1].assign(anisotropic_history(mesh, S, 2))
    bids = list(gadopt.get_boundary_ids(mesh))
    bcs = {bid: {"un": 0.3} for bid in bids[:3]}
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=25.0, bcs=bcs, solver_parameters="direct"
    )
    J = fd.derivative(solver.F, z)

    with_term = condensed_operator(J)
    assert asymmetry(with_term) < 1e-12

    # Drop the exterior-facet part of the (M, u) block: the boundary term.
    history_block = dict(split_form(J))[(1, 0)]
    facet_part = ufl.Form(
        [i for i in history_block.integrals() if i.integral_type() == "exterior_facet"]
    )
    assert fd.assemble(facet_part).petscmat.norm() > 0
    volume_part = ufl.Form(
        [i for i in history_block.integrals() if i.integral_type() == "cell"]
    )
    without_term = condensed_operator(J, history_block=volume_part)
    assert asymmetry(without_term) > 1e-4


def test_slate_boundary_penalty_matches_standard_assembly():
    """The displacement block assembles the same under Slate and under PETSc.

    Slate evaluates `avg(CellVolume)` on an exterior facet as half the cell
    volume; `viscosity_term` therefore writes its exterior penalty with the
    plain cell volume. This guards that choice.
    """
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    approx = approximation(mesh, 1)
    approx.mu = approx.mu0
    bids = list(gadopt.get_boundary_ids(mesh))
    bcs = {bids[0]: {"un": 0.3}, bids[1]: {"u": fd.Constant([0.1, 0.2])}}
    u = fd.Function(V)
    m = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1))
    stress = approx.stress(u, internal_variables=[m])
    eq = Equation(
        fd.TestFunction(V), V, viscosity_term, eq_attrs={"stress": stress},
        approximation=approx, bcs=bcs,
    )
    J = fd.derivative(eq.residual(u), u)
    standard = fd.assemble(J).petscmat
    slate = fd.assemble(Tensor(J)).petscmat
    difference = standard.copy()
    difference.axpy(-1.0, slate)
    assert difference.norm() / standard.norm() < 1e-13


# --------------------------------------------------------------------------
# The shipped preset


def coupled_solve(mesh, n_elements, bcs, params, steps=1, exponent=1, extra=None,
                  space=None, dt=None):
    approx = approximation(mesh, n_elements, exponent=exponent)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = approx.internal_variable_space(mesh) if space is None else space
    z = fd.Function(V * S)
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=DT if dt is None else dt, bcs=bcs, solver_parameters=params,
        solver_parameters_extra=extra,
    )
    for _ in range(steps):
        solver.solve()
    return solver, z


@pytest.mark.parametrize("n_elements", [1, 2, 3])
@pytest.mark.parametrize("bc_builder", [weak_bcs, strong_bcs], ids=["weak", "strong"])
def test_static_condensation_preset_matches_direct_solve(n_elements, bc_builder):
    """The preset converges and agrees with a direct solve for 1 to 3 elements.

    Three elements is the case the per-field layout could not run: Firedrake's
    `SCPC` refuses more than three fields.
    """
    mesh = square_mesh()
    bcs = bc_builder(mesh)
    direct, z_direct = coupled_solve(mesh, n_elements, bcs, "direct", steps=2)
    iterative, z_iter = coupled_solve(
        mesh, n_elements, bcs, "iterative", steps=2,
        extra={"condensed_field_ksp_rtol": 1e-10},
    )
    assert iterative.solver_parameters["pc_python_type"] == "gadopt.InternalVariableSCPC"
    assert iterative.solver_parameters["condensed_field"]["ksp_type"] == "cg"
    assert iterative.solver.snes.getConvergedReason() > 0
    pc = iterative.solver.snes.ksp.pc.getPythonContext()
    assert pc.condensed_ksp.getConvergedReason() > 0
    for field in (0, 1):
        reference = z_direct.subfunctions[field]
        error = fd.errornorm(reference, z_iter.subfunctions[field]) / fd.norm(reference)
        assert error < 1e-8


def test_single_element_legacy_layout_matches_combined():
    """A `(d, d)` field for one Maxwell element gives the combined field's answer."""
    mesh = square_mesh()
    bcs = weak_bcs(mesh)
    legacy_space = fd.TensorFunctionSpace(mesh, "DG", 1)
    _, z_legacy = coupled_solve(mesh, 1, bcs, "direct", steps=2, space=legacy_space)
    _, z_combined = coupled_solve(mesh, 1, bcs, "direct", steps=2)
    u_legacy, u_combined = z_legacy.subfunctions[0], z_combined.subfunctions[0]
    assert fd.errornorm(u_legacy, u_combined) / fd.norm(u_legacy) < 1e-12
    m_legacy = z_legacy.subfunctions[1]
    m_combined = z_combined.subfunctions[1][0, :, :]
    assert fd.assemble(fd.inner(m_legacy - m_combined, m_legacy - m_combined) * fd.dx) < 1e-24


def test_power_law_newton_with_static_condensation():
    """A power-law rheology runs Newton on GMRES and matches the direct solve.

    Two elements sharing the total-stress creep factor have no dissipation
    potential, so the condensed operator is not symmetric and the solver
    selects GMRES. The operator is rebuilt at every Newton iteration.
    """
    mesh = square_mesh()
    bcs = weak_bcs(mesh)
    direct, z_direct = coupled_solve(mesh, 2, bcs, "direct", exponent=3)
    solver, z = coupled_solve(
        mesh, 2, bcs, "iterative", exponent=3,
        extra={"condensed_field_ksp_rtol": 1e-12},
    )
    assert solver.solver_parameters["snes_type"] == "newtonls"
    assert solver.solver_parameters["condensed_field"]["ksp_type"] == "gmres"
    assert not solver.condensed_operator_symmetric()
    snes = solver.solver.snes
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() >= 1
    assert snes.getIterationNumber() == direct.solver.snes.getIterationNumber()
    pc = snes.ksp.pc.getPythonContext()
    # Every Newton iteration linearises anew, so the operator is rebuilt for
    # every linear solve; the first build is the initialisation.
    assert pc.assembly_count == snes.getIterationNumber()
    assert solver.appctx["operator_version"] is None
    for field in (0, 1):
        reference = z_direct.subfunctions[field]
        assert fd.errornorm(reference, z.subfunctions[field]) / fd.norm(reference) < 1e-8


def test_power_law_condensed_operator_is_not_symmetric():
    """Two elements with the total-stress factor: the exact tangent is not symmetric.

    This pins the reason for GMRES. A single element with strong boundary
    conditions in 3-D is symmetric, which pins the reason CG is kept there.
    """
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    approx = approximation(mesh, 2, exponent=3)
    S = approx.internal_variable_space(mesh)
    z = fd.Function(V * S)
    X = fd.SpatialCoordinate(mesh)
    z.subfunctions[0].interpolate(0.05 * (X + 0.3 * X[0] * X))
    z.subfunctions[1].assign(0.1 * anisotropic_history(mesh, S, 2))
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=strong_bcs(mesh), solver_parameters="direct"
    )
    two_elements = condensed_operator(fd.derivative(solver.F, z))
    assert asymmetry(two_elements) > 1e-8

    cube = fd.UnitCubeMesh(2, 2, 2)
    cube.cartesian = True
    V3 = fd.VectorFunctionSpace(cube, "CG", 2)
    approx3 = approximation(cube, 1, exponent=3)
    S3 = approx3.internal_variable_space(cube)
    z3 = fd.Function(V3 * S3)
    X3 = fd.SpatialCoordinate(cube)
    z3.subfunctions[0].interpolate(0.05 * (X3 + 0.3 * X3[0] * X3))
    # The one-element symmetry needs a trace-free history, which the march
    # preserves in 3-D; the artificial state here is made trace free by hand.
    history = anisotropic_history(cube, S3, 1)
    block = history[0, :, :]
    z3.subfunctions[1].interpolate(
        fd.as_tensor([0.1 * (block - fd.tr(block) / 3 * fd.Identity(3))])
    )
    bids = list(gadopt.get_boundary_ids(cube))
    bcs3 = {bids[0]: {"ux": 0}, bids[2]: {"uy": 0}, bids[4]: {"uz": 0},
            bids[5]: {"normal_stress": 0.1, "free_surface": {}}}
    solver3 = gadopt.CoupledInternalVariableSolver(
        z3, approx3, dt=DT, bcs=bcs3, solver_parameters="direct"
    )
    assert solver3.condensed_operator_symmetric()
    one_element = condensed_operator(fd.derivative(solver3.F, z3))
    assert asymmetry(one_element) < 1e-12


@pytest.mark.parametrize(
    "n_elements, exponent, bc_builder, dim, expected",
    [
        (2, 1, weak_bcs, 2, "cg"),
        (1, 3, weak_bcs, 2, "gmres"),
        (1, 3, strong_bcs, 2, "gmres"),
        (2, 3, strong_bcs, 2, "gmres"),
    ],
    ids=["newtonian", "power-law-weak", "power-law-2d", "power-law-two-elements"],
)
def test_condensed_krylov_selection(n_elements, exponent, bc_builder, dim, expected):
    mesh = square_mesh()
    approx = approximation(mesh, n_elements, exponent=exponent)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z = fd.Function(V * approx.internal_variable_space(mesh))
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=bc_builder(mesh), solver_parameters="iterative"
    )
    assert solver.solver_parameters["condensed_field"]["ksp_type"] == expected
    # An explicit choice by the user wins over the rule.
    forced = gadopt.CoupledInternalVariableSolver(
        fd.Function(z.function_space()), approximation(mesh, n_elements, exponent=exponent),
        dt=DT, bcs=bc_builder(mesh), solver_parameters="iterative",
        solver_parameters_extra={"condensed_field": {"ksp_type": "fgmres"}},
    )
    assert forced.solver_parameters["condensed_field"]["ksp_type"] == "fgmres"


def test_constant_exponent_one_keeps_operator_reuse():
    """`exponent=Constant(1)` is Newtonian: one linear solve and one assembly per march."""
    mesh = square_mesh()
    approx = approximation(mesh, 2, exponent=fd.Constant(1.0))
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z = fd.Function(V * approx.internal_variable_space(mesh))
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    assert solver.solver_parameters["snes_type"] == "ksponly"
    for _ in range(3):
        solver.solve()
    assert solver.solver.snes.ksp.pc.getPythonContext().assembly_count == 1


def test_condensed_operator_reuse_and_rebuild():
    """The condensed operator is built once per change of the Jacobian."""
    mesh = square_mesh()
    dt = fd.Constant(DT)
    approx = approximation(mesh, 2)
    # A material field the test can change in place.
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    approx.viscosity[0] = fd.Function(DG0).assign(VISCOSITIES[0])
    approx.maxwell_times[0] = approx.viscosity[0] / approx.shear_modulus[0]
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z = fd.Function(V * approx.internal_variable_space(mesh))
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=dt, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    for _ in range(3):
        solver.solve()
    pc = solver.solver.snes.ksp.pc.getPythonContext()
    assert pc.assembly_count == 1

    dt.assign(0.5 * DT)
    solver.solve()
    assert pc.assembly_count == 2
    solver.solve()
    assert pc.assembly_count == 2

    approx.viscosity[0].assign(2 * VISCOSITIES[0])
    solver.solve()
    assert pc.assembly_count == 3

    solver.invalidate_jacobian()
    solver.solve()
    assert pc.assembly_count == 4

    # The rebuilt operator is the right one: a fresh solver at the changed
    # coefficients gives the same displacement.
    fresh_approx = approximation(mesh, 2)
    fresh_approx.viscosity[0] = fd.Function(DG0).assign(2 * VISCOSITIES[0])
    fresh_approx.maxwell_times[0] = fresh_approx.viscosity[0] / fresh_approx.shear_modulus[0]
    z_fresh = fd.Function(z.function_space())
    fresh = gadopt.CoupledInternalVariableSolver(
        z_fresh, fresh_approx, dt=dt, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    z_fresh.assign(z)
    fresh.solution_old.assign(solver.solution_old)
    fresh.solve()
    solver.solve()
    u, u_fresh = z.subfunctions[0], z_fresh.subfunctions[0]
    assert fd.errornorm(u, u_fresh) / fd.norm(u) < 1e-8


def test_singular_condensed_operator_solved_with_cg_and_nullspace():
    """A translation mode reaches the condensed operator and CG handles it.

    Weak normal conditions on the bottom and the top, free vertical walls and
    a balanced tangential load on the top: the horizontal translation
    `(1, 0)` is then a rigid mode of the operator. Buoyancy is off, because
    the hydrostatic prestress term makes a translation across a free wall
    do work, which removes the mode. The test checks that the mode is in
    the kernel of the condensed operator, that all three nullspaces are set
    on it, and that CG converges on the singular system to the same
    displacement a direct solve gives up to that mode.
    """
    mesh = square_mesh()
    approx = approximation(mesh, 2, B_mu=0.0)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    Z = V * approx.internal_variable_space(mesh)
    bids = list(gadopt.get_boundary_ids(mesh))
    X = fd.SpatialCoordinate(mesh)
    # Bottom (bids[2]) and top (bids[3]) carry `un = 0`; the vertical walls
    # are free. The horizontal translation is then unconstrained.
    bcs = {
        bids[2]: {"un": 0},
        bids[3]: {"un": 0, "stress": fd.as_vector([0.1 * fd.sin(2 * fd.pi * X[0]), 0.0])},
    }
    translation = gadopt.rigid_body_modes(V, rotational=False, translations=[0])
    exact = fd.MixedVectorSpaceBasis(Z, [translation, Z.sub(1)])
    z = fd.Function(Z)
    solver = gadopt.CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=bcs, solver_parameters="iterative",
        solver_parameters_extra={"condensed_field_ksp_rtol": 1e-10},
        nullspace=exact, transpose_nullspace=exact, near_nullspace=exact,
    )
    solver.solve()
    pc = solver.solver.snes.ksp.pc.getPythonContext()
    S = pc.S.petscmat
    assert S.getNullSpace().handle != 0
    assert S.getTransposeNullSpace().handle != 0
    assert S.getNearNullSpace().handle != 0
    mode = fd.Function(V).interpolate(fd.as_vector([1.0, 0.0]))
    with mode.dat.vec_ro as v:
        image = v.duplicate()
        S.mult(v, image)
        assert image.norm() / (S.norm() * v.norm()) < 1e-12
    assert pc.condensed_ksp.getConvergedReason() > 0
    assert pc.condensed_ksp.getType() == "cg"

    z_direct = fd.Function(Z)
    gadopt.CoupledInternalVariableSolver(
        z_direct, approximation(mesh, 2, B_mu=0.0), dt=DT, bcs=bcs,
        solver_parameters="direct", nullspace=exact, transpose_nullspace=exact,
    ).solve()
    # Compare modulo the translation: remove the mean horizontal displacement.

    def centred(u):
        mean = fd.assemble(u[0] * fd.dx) / fd.assemble(1 * fd.dx(mesh))
        return u - fd.as_vector([mean, 0.0])
    diff = centred(z.subfunctions[0]) - centred(z_direct.subfunctions[0])
    reference = centred(z_direct.subfunctions[0])
    assert fd.sqrt(fd.assemble(fd.inner(diff, diff) * fd.dx)) < 1e-7 * fd.sqrt(
        fd.assemble(fd.inner(reference, reference) * fd.dx)
    )


def test_extruded_hex_with_weak_boundary_matches_direct():
    """Static condensation on an extruded hex mesh with the boundary term.

    The sphere cases use `CombinedSurfaceMeasure` with `un` on `bottom`; this
    is the smallest such case, with two elements and the shipped preset.
    """
    base = fd.UnitSquareMesh(2, 2, quadrilateral=True)
    mesh = fd.ExtrudedMesh(base, 2)
    mesh.cartesian = True
    approx = approximation(mesh, 2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    Z = V * approx.internal_variable_space(mesh)
    X = fd.SpatialCoordinate(mesh)
    boundary = gadopt.get_boundary_ids(mesh)
    bcs = {
        boundary.bottom: {"un": 0},
        boundary.top: {
            "normal_stress": 0.1 * fd.exp(-((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2) / 0.05),
            "free_surface": {},
        },
        boundary.left: {"ux": 0}, boundary.right: {"ux": 0},
        boundary.front: {"uy": 0}, boundary.back: {"uy": 0},
    }
    z_direct = fd.Function(Z)
    gadopt.CoupledInternalVariableSolver(
        z_direct, approx, dt=DT, bcs=bcs, solver_parameters="direct"
    ).solve()
    z = fd.Function(Z)
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation(mesh, 2), dt=DT, bcs=bcs, solver_parameters="iterative",
        solver_parameters_extra={"condensed_field_ksp_rtol": 1e-10},
    )
    solver.solve()
    pc = solver.solver.snes.ksp.pc.getPythonContext()
    assert asymmetry(pc.S.petscmat) < 1e-12
    for field in (0, 1):
        reference = z_direct.subfunctions[field]
        assert fd.errornorm(reference, z.subfunctions[field]) / fd.norm(reference) < 1e-8


def test_full_storage_and_scaling_factor_under_the_preset():
    """Full `(d, d)` storage and a residual scaling factor give the same answer."""
    mesh = square_mesh()
    bcs = weak_bcs(mesh)
    approx = approximation(mesh, 2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z_ref = fd.Function(V * approx.internal_variable_space(mesh))
    gadopt.CoupledInternalVariableSolver(
        z_ref, approx, dt=DT, bcs=bcs, solver_parameters="direct"
    ).solve()

    approx_full = approximation(mesh, 2)
    z_full = fd.Function(V * approx_full.internal_variable_space(mesh, symmetric=False))
    solver = gadopt.CoupledInternalVariableSolver(
        z_full, approx_full, dt=DT, bcs=bcs, solver_parameters="iterative",
        scaling_factor=1e6,
        solver_parameters_extra={"condensed_field_ksp_rtol": 1e-10},
    )
    solver.solve()
    pc = solver.solver.snes.ksp.pc.getPythonContext()
    assert asymmetry(pc.S.petscmat) < 1e-12
    u_ref, u_full = z_ref.subfunctions[0], z_full.subfunctions[0]
    assert fd.errornorm(u_ref, u_full) / fd.norm(u_ref) < 1e-8
    m_ref, m_full = z_ref.subfunctions[1], z_full.subfunctions[1]
    assert fd.assemble(fd.inner(m_ref - m_full, m_ref - m_full) * fd.dx) < 1e-16 * fd.assemble(
        fd.inner(m_ref, m_ref) * fd.dx
    )


# --------------------------------------------------------------------------
# Layout errors


def test_coupled_solver_rejects_per_element_fields():
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    z = fd.Function(V * S * S)
    with pytest.raises(ValueError, match="internal_variable_space"):
        gadopt.CoupledInternalVariableSolver(
            z, approximation(mesh, 2), dt=DT, bcs=weak_bcs(mesh), solver_parameters="direct"
        )


def test_coupled_solver_rejects_wrong_element_count():
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z = fd.Function(V * internal_variable_space(mesh, 2))
    with pytest.raises(ValueError, match="Maxwell element"):
        gadopt.CoupledInternalVariableSolver(
            z, approximation(mesh, 3), dt=DT, bcs=weak_bcs(mesh), solver_parameters="direct"
        )


def test_pointwise_history_rejects_a_mixed_space():
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = internal_variable_space(mesh, 1)
    z = fd.Function(V * S)
    with pytest.raises(ValueError, match="displacement-only"):
        gadopt.InternalVariableSolver(
            z, approximation(mesh, 1), dt=DT, internal_variables=fd.Function(S),
            bcs=weak_bcs(mesh), solver_parameters="direct",
        )


# --------------------------------------------------------------------------
# The substituted solver on every layout


def substituted_solve(mesh, n_elements, internal_variables, steps=2):
    approx = approximation(mesh, n_elements)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    u = fd.Function(V)
    solver = gadopt.InternalVariableSolver(
        u, approx, dt=DT, internal_variables=internal_variables,
        bcs=weak_bcs(mesh), solver_parameters="direct",
    )
    for _ in range(steps):
        solver.solve()
    return solver, u


def test_substituted_solver_layouts_agree():
    """A list of `(d, d)` fields and one `(n, d, d)` field give the same march."""
    mesh = square_mesh()
    combined = fd.Function(internal_variable_space(mesh, 2))
    separate = [fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1)) for _ in range(2)]
    _, u_combined = substituted_solve(mesh, 2, combined)
    solver, u_separate = substituted_solve(mesh, 2, separate)
    assert fd.errornorm(u_combined, u_separate) / fd.norm(u_combined) < 1e-12
    for i, m in enumerate(separate):
        stored = combined[i, :, :]
        assert fd.assemble(fd.inner(stored - m, stored - m) * fd.dx) < 1e-24
    # The stored history advanced: the update is not the zero field.
    assert fd.norm(separate[0]) > 0
    assert len(solver.internal_variables_update) == 2


def test_substituted_solver_single_element_layouts_agree():
    mesh = square_mesh()
    single = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1))
    combined = fd.Function(internal_variable_space(mesh, 1))
    _, u_single = substituted_solve(mesh, 1, single)
    _, u_combined = substituted_solve(mesh, 1, combined)
    assert fd.errornorm(u_single, u_combined) / fd.norm(u_single) < 1e-12


def substituted_coupled_gap(n):
    """Relative displacement gap between the two formulations on an n x n mesh.

    With strong boundary conditions on an affine mesh the two are identical
    to roundoff (the strain of a P2 displacement lies in the DG1 history
    space, so the projection is exact). The weak normal conditions are what
    separates them: the coupled form carries the elastic penalty `mu0` and
    the boundary term, the substituted form the effective one, and the gap
    is a boundary discretisation error.
    """
    mesh = square_mesh(n)
    bcs = weak_bcs(mesh)
    approx = approximation(mesh, 2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = approx.internal_variable_space(mesh)
    u = fd.Function(V)
    substituted = gadopt.InternalVariableSolver(
        u, approx, dt=DT, internal_variables=fd.Function(S), bcs=bcs,
        solver_parameters="direct",
    )
    substituted.solve()
    _, z = coupled_solve(mesh, 2, bcs, "direct", steps=1)
    return fd.errornorm(u, z.subfunctions[0]) / fd.norm(u)


def test_substituted_and_coupled_solvers_converge_to_each_other():
    """The gap between the two formulations shrinks under refinement.

    Both are consistent discretisations of the same time step, so the gap
    is a discretisation error and must decrease with the mesh size. Measured
    4.2e-5, 3.7e-6 and 7.0e-7 at 4, 8 and 16 cells per side; a factor of
    four per refinement is required.
    """
    coarse, fine = substituted_coupled_gap(4), substituted_coupled_gap(8)
    assert coarse < 1e-3
    assert fine < coarse / 4
