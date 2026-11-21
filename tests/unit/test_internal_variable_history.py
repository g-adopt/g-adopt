"""Tests for the combined internal-variable field and its solvers.

Every Maxwell element of an internal-variable rheology lives in one
discontinuous tensor field of shape `(n, d, d)` (see
`gadopt.internal_variable_equation`). These tests pin:

- the space helper and the slicing helpers for every accepted layout;
- the history residual on the combined field against the equation written
  out by hand, boundary term included;
- the Krylov selection on the condensed operator: CG where the operator is
  symmetric, GMRES where the power-law tangent is not;
- the refusals of the coupled solver for a wrong history layout;
- the assembled boundary penalty under Slate against standard assembly,
  which guards the exterior-facet coefficient of `viscosity_term`.

The solves of the static-condensation preset (against a direct solve, with
weak and strong boundary conditions, with a power law), the symmetry of the
condensed operator, its reuse and rebuild, and the substituted solver on
every layout are checked by the `condensation-*` entries of
`tests/weak_bc_gia` (driver `gia_condensation.py`).

The meshes are tiny: these are algebraic properties and small solves.
"""

import firedrake as fd
import numpy as np
import pytest
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


def hexahedral_cube_mesh(n=2):
    """A unit cube of affine hexahedra, the shape every 3-D production run has.

    The extruded spheres the self-gravity solver runs on report a
    `TensorProductCell`, which is not a simplex either; this affine box
    isolates the cell shape from curvature and from extrusion.
    """
    mesh = fd.BoxMesh(n, n, n, 1.0, 1.0, 1.0, hexahedral=True)
    mesh.cartesian = True
    return mesh


def cube_mesh(n=2):
    """A unit cube of affine tetrahedra: the one 3-D cell that keeps CG."""
    mesh = fd.UnitCubeMesh(n, n, n)
    mesh.cartesian = True
    return mesh


def strong_bcs_3d(mesh):
    """Component-wise strong conditions on a box, no weak displacement term.

    The one-element power-law CG branch requires exactly this: no `"u"` and no
    `"un"` anywhere, so the Nitsche boundary term that needs a cell-constant
    Maxwell time is absent and only the volume relation decides.
    """
    bids = list(gadopt.get_boundary_ids(mesh))
    return {bids[0]: {"ux": 0}, bids[2]: {"uy": 0}, bids[4]: {"uz": 0},
            bids[5]: {"normal_stress": 0.1, "free_surface": {}}}


@pytest.mark.parametrize(
    "n_elements, exponent, mesh_builder, bc_builder, expected",
    [
        (2, 1, square_mesh, weak_bcs, "cg"),
        (1, 3, square_mesh, weak_bcs, "gmres"),
        (1, 3, square_mesh, strong_bcs, "gmres"),
        (2, 3, square_mesh, strong_bcs, "gmres"),
        (1, 1, cube_mesh, strong_bcs_3d, "cg"),
        (1, 3, cube_mesh, strong_bcs_3d, "gmres"),
        (1, 3, hexahedral_cube_mesh, strong_bcs_3d, "gmres"),
    ],
    ids=["newtonian", "power-law-weak", "power-law-2d", "power-law-two-elements",
         "newtonian-tetrahedra", "power-law-tetrahedra", "power-law-hexahedra"],
)
def test_condensed_krylov_selection(n_elements, exponent, mesh_builder,
                                    bc_builder, expected):
    """The rheology picks the method, and nothing else does.

    Newtonian keeps the preset's CG on every mesh; every power law takes
    GMRES, including the one-element tetrahedral 3-D case whose operator is
    genuinely symmetric (`test_power_law_condensed_operator_is_not_symmetric`
    measures it at 1.1e-16). That case is given away deliberately: it occurs
    on no mesh this project runs in 3-D, and what detecting it saves is the
    orthogonalisation of a short truncated GMRES. The tetrahedral and
    hexahedral power-law rows differ only in the cell and expect the same
    method, which is what pins that the rule no longer reads the mesh.
    """
    mesh = mesh_builder()
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
