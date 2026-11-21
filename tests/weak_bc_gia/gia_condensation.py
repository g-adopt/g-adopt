r"""The coupled internal-variable solver under its static-condensation preset.

`CoupledInternalVariableSolver` solves for the displacement `u` and the
combined internal-variable field `M` of shape `(n, d, d)` together. Its
iterative preset eliminates `M` cell by cell (`gadopt.InternalVariableSCPC`)
and solves the condensed displacement operator

$$ S = A_{uu} - A_{uM} A_{MM}^{-1} A_{Mu} $$

with CG when the rheology is Newtonian and with GMRES for a power law. This
driver measures the properties that preset depends on and writes them to
`condensation-<entry>.dat`, one line `name value` per quantity.
`test_weak_bc_gia.py` reads the file and applies the tolerances.

Each row writes the measured value next to its reference: a direct solve of
the same system, the other layout of the same history, the transpose of the
same operator, or a count that the solver logic fixes. No value is recorded
from an earlier run.

The entries:

- `preset`: the preset against a direct solve for one to three Maxwell
  elements with weak and strong boundary conditions, a power law under the
  preset, the extruded hexahedral mesh with a weak boundary, and the history
  layouts `(d, d)` and `(n, d, d)` for the coupled and the substituted solver.
- `operator`: the symmetry of the condensed operator with and without the
  boundary term of the history equation and for a power law, the reuse of the
  condensed operator across solves, CG on a singular condensed operator, and
  the gap between the substituted and the coupled solver under refinement.
- `pc`: the divergence-free near-nullspace modes, the iteration counts of the
  near-incompressible preconditioner, the patch multigrid parameters, the
  options of the derivative solvers that a taped solve builds, and the SPD flag
  with the near-nullspace on one assembled block.

Usage:
    python3 gia_condensation.py --entry preset
"""

import argparse

# G-ADOPT must be imported before Firedrake: it sets the options and the
# patches that the Firedrake solvers below read.
import gadopt
import firedrake as fd
import ufl
from firedrake.adjoint import (
    continue_annotation,
    get_working_tape,
    pause_annotation,
)
from firedrake.formmanipulation import split_form
from firedrake.slate import Tensor
from gadopt.approximations import CompressibleInternalVariableApproximation
from gadopt.internal_variable_equation import internal_variable_space
from gadopt.nullspaces import solenoidal_modes
from gadopt.preconditioners import SPDAssembledPC
from gadopt.solver_options_manager import (
    gamg_parameters,
    nearly_incompressible_mg_parameters,
)
from gadopt.stokes_integrators import CoupledInternalVariableSolver

from meta import (
    CONDENSATION_ENTRIES,
    CONDENSATION_MG_COARSE,
    CONDENSATION_PRESET_CASES,
    CONDENSATION_SOLENOIDAL_DIMS,
)

# Shear moduli and viscosities of up to three Maxwell elements. The Maxwell
# times (2, 0.5 and 3) differ, so that a term applied to the wrong element is
# visible in the solution.
SHEAR_MODULI = [1.0, 2.0, 0.5]
VISCOSITIES = [2.0, 1.0, 1.5]
# The time step of every march here, of the order of the Maxwell times.
DT = 0.4
# The relative tolerance of the condensed Krylov solve in the rows that
# compare with a direct solve. It is far below the comparison tolerance, so
# that the comparison measures the elimination and not the Krylov stop.
CONDENSED_RTOL = 1e-10


# --------------------------------------------------------------------------
# Meshes, approximation and boundary conditions


def square_mesh(n=4):
    """A unit square of `n` by `n` cells, two triangles per cell.

    Args:
      n: The number of cells along each side.

    Returns:
      The mesh, marked cartesian so that the vertical is a coordinate direction.
    """
    mesh = fd.UnitSquareMesh(n, n)
    mesh.cartesian = True
    return mesh


def cube_mesh(n=2):
    """A unit cube of affine tetrahedra.

    Args:
      n: The number of cells along each side.

    Returns:
      The mesh, marked cartesian.
    """
    mesh = fd.UnitCubeMesh(n, n, n)
    mesh.cartesian = True
    return mesh


def hexahedral_cube_mesh(n=2):
    """A unit cube of affine hexahedra, the cell shape of the 3-D production runs.

    The extruded spheres report a `TensorProductCell`, which is not a simplex
    either. This affine box separates the cell shape from the curvature and
    from the extrusion.

    Args:
      n: The number of cells along each side.

    Returns:
      The mesh, marked cartesian.
    """
    mesh = fd.BoxMesh(n, n, n, 1.0, 1.0, 1.0, hexahedral=True)
    mesh.cartesian = True
    return mesh


def approximation(mesh, n_elements, exponent=1, B_mu=1.27):
    """A compressible internal-variable approximation with `n_elements` elements.

    The bulk modulus is 3 and the bulk-to-shear ratio 1.5, so the operator is
    compressible and well away from the incompressible limit. The transition
    stress 0.05 is below the stresses of the surface load, so a power law
    with `exponent=3` is in its nonlinear range.

    Args:
      mesh: The mesh of the density field.
      n_elements: The number of Maxwell elements, at most three.
      exponent: The power-law exponent of the creep; 1 is Newtonian.
      B_mu: The buoyancy number. Zero removes the hydrostatic prestress term.

    Returns:
      The approximation.
    """
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    return CompressibleInternalVariableApproximation(
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
    """A Gaussian normal load of amplitude 0.1 centred on the top boundary."""
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


def strong_bcs_3d(mesh):
    """Component-wise strong conditions on a box, with no weak displacement term.

    One face per direction holds that displacement component, and the top
    carries a uniform load. No `"u"` or `"un"` condition appears, so the
    Nitsche boundary term is absent and only the volume terms decide the
    symmetry of the condensed operator.
    """
    bids = list(gadopt.get_boundary_ids(mesh))
    return {bids[0]: {"ux": 0}, bids[2]: {"uy": 0}, bids[4]: {"uz": 0},
            bids[5]: {"normal_stress": 0.1, "free_surface": {}}}


BC_BUILDERS = {"weak": weak_bcs, "strong": strong_bcs}


def anisotropic_history(mesh, space, n_elements):
    """A nonzero history that differs between the elements.

    Element `i` holds `0.1 (i + 1) sym(X (1, 2, ...)^T)`, so every term of the
    history equation is nonzero and no two elements carry the same field.

    Args:
      mesh: The mesh of the coordinate field.
      space: The combined internal-variable space of shape `(n, d, d)`.
      n_elements: The number of Maxwell elements `n`.

    Returns:
      The history as a Function on `space`.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    direction = fd.as_vector([float(k + 1) for k in range(dim)])
    blocks = [0.1 * (i + 1) * fd.sym(fd.outer(X, direction))
              for i in range(n_elements)]
    return fd.Function(space).interpolate(fd.as_tensor(blocks))


def trace_free_history(mesh, space):
    """One-element history made trace free, as the 3-D march keeps it.

    The symmetry of the one-element power-law tangent needs a trace-free
    history. An artificial state does not have one, so the trace is removed
    here by hand.

    Args:
      mesh: The 3-D mesh.
      space: The one-element internal-variable space of shape `(1, 3, 3)`.

    Returns:
      The expression to interpolate into `space`.
    """
    block = anisotropic_history(mesh, space, 1)[0, :, :]
    return fd.as_tensor([0.1 * (block - fd.tr(block) / 3 * fd.Identity(3))])


# --------------------------------------------------------------------------
# Measurements


def asymmetry(petscmat):
    """Return ||A - A^T||_F / ||A||_F for an assembled PETSc matrix.

    A ratio to the matrix norm, so that the value is a dimensionless distance
    from exact symmetry for any scaling of the operator.
    """
    transpose = petscmat.duplicate(copy=True)
    transpose.transpose()
    difference = petscmat.copy()
    difference.axpy(-1.0, transpose)
    return difference.norm() / petscmat.norm()


def condensed_operator(J, history_block=None):
    """Assemble `A_uu - A_uM A_MM^-1 A_Mu` with Slate from the Jacobian form `J`.

    This is the same Schur complement that `InternalVariableSCPC` builds, here
    assembled outside the solver so that the symmetry can be measured at any
    state.

    Args:
      J: The Jacobian form of the coupled `(u, M)` system.
      history_block: A form that replaces the `(M, u)` block, so that a row
        can drop part of it.

    Returns:
      The assembled condensed operator as a PETSc matrix.
    """
    blocks = dict(split_form(J))
    A00, A01, A11 = (Tensor(blocks[key]) for key in ((0, 0), (0, 1), (1, 1)))
    A10 = Tensor(blocks[(1, 0)] if history_block is None else history_block)
    return fd.assemble(A00 - A01 * A11.inv * A10).petscmat


def relative_error(reference, value):
    """Return the L2 norm of `value - reference` relative to that of `reference`."""
    return fd.errornorm(reference, value) / fd.norm(reference)


def squared_difference(a, b):
    """Return the integral of `|a - b|^2` over the mesh, an absolute measure."""
    return fd.assemble(fd.inner(a - b, a - b) * fd.dx)


def relative_divergence(field):
    """Return ||div m||_L2 / ||m||_L2, zero for a divergence-free field."""
    norm = fd.sqrt(fd.assemble(fd.inner(field, field) * fd.dx))
    div_norm = fd.sqrt(fd.assemble(fd.inner(fd.div(field), fd.div(field)) * fd.dx))
    return float(div_norm / norm)


def condensation_pc(solver):
    """The Python context of the `InternalVariableSCPC` of a coupled solver."""
    return solver.solver.snes.ksp.pc.getPythonContext()


def coupled_solve(mesh, n_elements, bcs, params, steps=1, exponent=1, extra=None,
                  space=None, dt=None):
    """March a coupled solver from rest under the given preset.

    Args:
      mesh: The mesh.
      n_elements: The number of Maxwell elements.
      bcs: The boundary condition dictionary.
      params: The solver preset, "direct" or "iterative".
      steps: The number of time steps.
      exponent: The power-law exponent.
      extra: Extra solver options, applied after the preset.
      space: The history space; the combined `(n, d, d)` space when None.
      dt: The time step; `DT` when None.

    Returns:
      The solver and the solution Function on the mixed space.
    """
    approx = approximation(mesh, n_elements, exponent=exponent)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = approx.internal_variable_space(mesh) if space is None else space
    z = fd.Function(V * S)
    solver = CoupledInternalVariableSolver(
        z, approx, dt=DT if dt is None else dt, bcs=bcs, solver_parameters=params,
        solver_parameters_extra=extra,
    )
    for _ in range(steps):
        solver.solve()
    return solver, z


def substituted_solve(mesh, n_elements, internal_variables, steps=2):
    """March the substituted `InternalVariableSolver` with weak boundaries.

    Args:
      mesh: The mesh.
      n_elements: The number of Maxwell elements.
      internal_variables: The history: one combined Function, one `(d, d)`
        Function, or a list of `(d, d)` Functions.
      steps: The number of time steps.

    Returns:
      The solver and the displacement Function.
    """
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


# --------------------------------------------------------------------------
# Entry "preset"


def preset_against_direct(record):
    """The preset converges and agrees with a direct solve for 1 to 3 elements.

    Two steps of each march, so that the second step reads a nonzero history.
    Each row writes the preset and the condensed Krylov method the solver
    chose, the two converged reasons, the relative error of each field against
    the direct solve, and the norm of the direct solution.
    """
    for n_elements, bc_kind in CONDENSATION_PRESET_CASES:
        key = f"preset-n{n_elements}-{bc_kind}"
        mesh = square_mesh()
        bcs = BC_BUILDERS[bc_kind](mesh)
        _, z_direct = coupled_solve(mesh, n_elements, bcs, "direct", steps=2)
        iterative, z_iter = coupled_solve(
            mesh, n_elements, bcs, "iterative", steps=2,
            extra={"condensed_field_ksp_rtol": CONDENSED_RTOL},
        )
        pc = condensation_pc(iterative)
        record[f"{key}-pc_python_type"] = iterative.solver_parameters["pc_python_type"]
        record[f"{key}-condensed_ksp_type"] = (
            iterative.solver_parameters["condensed_field"]["ksp_type"])
        record[f"{key}-snes_reason"] = iterative.solver.snes.getConvergedReason()
        record[f"{key}-condensed_ksp_reason"] = pc.condensed_ksp.getConvergedReason()
        record[f"{key}-condensed_ksp_its"] = pc.condensed_ksp.getIterationNumber()
        for field, name in ((0, "u"), (1, "m")):
            reference = z_direct.subfunctions[field]
            record[f"{key}-{name}_reference_norm"] = fd.norm(reference)
            record[f"{key}-{name}_error"] = relative_error(
                reference, z_iter.subfunctions[field])


def power_law_under_preset(record):
    """A power law runs Newton on GMRES under the preset and matches a direct solve.

    Two elements share the total-stress creep factor, so the relaxation rule
    has no dissipation potential, the condensed operator is not symmetric, and
    the solver selects GMRES. Every Newton iteration linearises anew, so the
    condensed operator is assembled once per Newton iteration and the solver
    publishes `operator_version = None` ("always rebuild").
    """
    mesh = square_mesh()
    bcs = weak_bcs(mesh)
    direct, z_direct = coupled_solve(mesh, 2, bcs, "direct", exponent=3)
    solver, z = coupled_solve(
        mesh, 2, bcs, "iterative", exponent=3,
        extra={"condensed_field_ksp_rtol": 1e-12},
    )
    snes = solver.solver.snes
    record["power-snes_type"] = solver.solver_parameters["snes_type"]
    record["power-condensed_ksp_type"] = (
        solver.solver_parameters["condensed_field"]["ksp_type"])
    record["power-rule_says_symmetric"] = int(solver.condensed_operator_symmetric())
    record["power-snes_reason"] = snes.getConvergedReason()
    record["power-newton_its"] = snes.getIterationNumber()
    record["power-direct_newton_its"] = direct.solver.snes.getIterationNumber()
    record["power-assembly_count"] = condensation_pc(solver).assembly_count
    record["power-operator_version"] = str(solver.appctx["operator_version"])
    for field, name in ((0, "u"), (1, "m")):
        reference = z_direct.subfunctions[field]
        record[f"power-{name}_reference_norm"] = fd.norm(reference)
        record[f"power-{name}_error"] = relative_error(reference, z.subfunctions[field])


def extruded_hexahedra(record):
    """Static condensation on an extruded hexahedral mesh with the boundary term.

    The sphere cases put `un` on the bottom of an extruded mesh, where the
    exterior facets are split into a base, a top and the side walls. This is
    the smallest such case: two Maxwell elements, `un` on the bottom, a loaded
    free surface on the top and component-wise conditions on the walls.
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
    CoupledInternalVariableSolver(
        z_direct, approx, dt=DT, bcs=bcs, solver_parameters="direct"
    ).solve()
    z = fd.Function(Z)
    solver = CoupledInternalVariableSolver(
        z, approximation(mesh, 2), dt=DT, bcs=bcs, solver_parameters="iterative",
        solver_parameters_extra={"condensed_field_ksp_rtol": CONDENSED_RTOL},
    )
    solver.solve()
    pc = condensation_pc(solver)
    record["extruded-condensed_asymmetry"] = asymmetry(pc.S.petscmat)
    record["extruded-condensed_ksp_reason"] = pc.condensed_ksp.getConvergedReason()
    for field, name in ((0, "u"), (1, "m")):
        reference = z_direct.subfunctions[field]
        record[f"extruded-{name}_reference_norm"] = fd.norm(reference)
        record[f"extruded-{name}_error"] = relative_error(
            reference, z.subfunctions[field])


def history_layouts(record):
    """The `(d, d)` and the `(n, d, d)` layouts of the history give one answer.

    Four comparisons. The coupled solver with one element on a `(d, d)` field
    and on the combined `(1, d, d)` field. The coupled solver under the preset
    with full (not symmetric) `(d, d)` storage and a residual scaling factor
    of 1e6, against the direct solve on the symmetric combined field. The
    substituted solver with a list of two `(d, d)` fields and with one
    `(2, d, d)` field. The substituted solver with one `(d, d)` field and with
    one `(1, d, d)` field.
    """
    # Coupled solver, one element: (d, d) against (1, d, d).
    mesh = square_mesh()
    bcs = weak_bcs(mesh)
    legacy_space = fd.TensorFunctionSpace(mesh, "DG", 1)
    _, z_legacy = coupled_solve(mesh, 1, bcs, "direct", steps=2, space=legacy_space)
    _, z_combined = coupled_solve(mesh, 1, bcs, "direct", steps=2)
    u_legacy, u_combined = z_legacy.subfunctions[0], z_combined.subfunctions[0]
    m_legacy = z_legacy.subfunctions[1]
    m_combined = z_combined.subfunctions[1][0, :, :]
    record["layout-coupled-u_reference_norm"] = fd.norm(u_legacy)
    record["layout-coupled-u_error"] = relative_error(u_legacy, u_combined)
    record["layout-coupled-m_reference_squared"] = fd.assemble(
        fd.inner(m_legacy, m_legacy) * fd.dx)
    record["layout-coupled-m_squared_difference"] = squared_difference(
        m_legacy, m_combined)

    # Coupled solver, full storage and scaling factor under the preset.
    approx = approximation(mesh, 2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z_ref = fd.Function(V * approx.internal_variable_space(mesh))
    CoupledInternalVariableSolver(
        z_ref, approx, dt=DT, bcs=bcs, solver_parameters="direct"
    ).solve()
    approx_full = approximation(mesh, 2)
    z_full = fd.Function(
        V * approx_full.internal_variable_space(mesh, symmetric=False))
    solver = CoupledInternalVariableSolver(
        z_full, approx_full, dt=DT, bcs=bcs, solver_parameters="iterative",
        scaling_factor=1e6,
        solver_parameters_extra={"condensed_field_ksp_rtol": CONDENSED_RTOL},
    )
    solver.solve()
    u_ref, u_full = z_ref.subfunctions[0], z_full.subfunctions[0]
    m_ref, m_full = z_ref.subfunctions[1], z_full.subfunctions[1]
    record["layout-full-condensed_asymmetry"] = asymmetry(
        condensation_pc(solver).S.petscmat)
    record["layout-full-u_reference_norm"] = fd.norm(u_ref)
    record["layout-full-u_error"] = relative_error(u_ref, u_full)
    record["layout-full-m_reference_squared"] = fd.assemble(
        fd.inner(m_ref, m_ref) * fd.dx)
    record["layout-full-m_squared_difference"] = squared_difference(m_ref, m_full)

    # Substituted solver, two elements: a list of (d, d) against (2, d, d).
    combined = fd.Function(internal_variable_space(mesh, 2))
    separate = [fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1)) for _ in range(2)]
    _, u_combined = substituted_solve(mesh, 2, combined)
    solver, u_separate = substituted_solve(mesh, 2, separate)
    record["layout-substituted-u_reference_norm"] = fd.norm(u_combined)
    record["layout-substituted-u_error"] = relative_error(u_combined, u_separate)
    record["layout-substituted-m_squared_difference"] = max(
        squared_difference(combined[i, :, :], m) for i, m in enumerate(separate))
    # The stored history advanced: the update is not the zero field.
    record["layout-substituted-m0_norm"] = fd.norm(separate[0])
    record["layout-substituted-n_updates"] = len(solver.internal_variables_update)

    # Substituted solver, one element: (d, d) against (1, d, d).
    single = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 1))
    combined_one = fd.Function(internal_variable_space(mesh, 1))
    _, u_single = substituted_solve(mesh, 1, single)
    _, u_combined_one = substituted_solve(mesh, 1, combined_one)
    record["layout-substituted_single-u_reference_norm"] = fd.norm(u_single)
    record["layout-substituted_single-u_error"] = relative_error(
        u_single, u_combined_one)


# --------------------------------------------------------------------------
# Entry "operator"


def symmetry_with_boundary_term(record):
    """The condensed operator is symmetric with the boundary term and not without it.

    The history equation carries a boundary term on the weak `un` sides. With
    it, the history rows are the transpose of the history part of the
    displacement rows, so the condensed operator is symmetric. Without the
    exterior-facet part of the `(M, u)` block it is not. `dt / tau` is 25 or
    larger, so the asymmetry that a missing term leaves is large.
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
    solver = CoupledInternalVariableSolver(
        z, approx, dt=25.0, bcs=bcs, solver_parameters="direct"
    )
    J = fd.derivative(solver.F, z)
    record["symmetry-with_term_asymmetry"] = asymmetry(condensed_operator(J))

    # Split the (M, u) block into its exterior-facet part (the boundary term)
    # and its cell part.
    history_block = dict(split_form(J))[(1, 0)]
    facet_part = ufl.Form([i for i in history_block.integrals()
                           if i.integral_type() == "exterior_facet"])
    volume_part = ufl.Form([i for i in history_block.integrals()
                            if i.integral_type() == "cell"])
    record["symmetry-boundary_term_norm"] = fd.assemble(facet_part).petscmat.norm()
    record["symmetry-without_term_asymmetry"] = asymmetry(
        condensed_operator(J, history_block=volume_part))


def power_law_asymmetry(record):
    """The power-law condensed operator: where it is symmetric and where it is not.

    Two elements sharing one stress-dependent factor have no dissipation
    potential, so the tangent is asymmetric at any state (2-D, strong
    boundary conditions). One element with strong conditions on tetrahedra in
    3-D is symmetric to round-off, and the same state on affine hexahedra is
    not, because the gradient of a Q2 displacement does not lie in the DG1
    history space. The Krylov selection reads the rheology only, so it reports
    the tetrahedral case as not symmetric.
    """
    mesh = square_mesh()
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    approx = approximation(mesh, 2, exponent=3)
    S = approx.internal_variable_space(mesh)
    z = fd.Function(V * S)
    X = fd.SpatialCoordinate(mesh)
    z.subfunctions[0].interpolate(0.05 * (X + 0.3 * X[0] * X))
    z.subfunctions[1].assign(0.1 * anisotropic_history(mesh, S, 2))
    solver = CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=strong_bcs(mesh), solver_parameters="direct"
    )
    record["power_symmetry-two_elements_asymmetry"] = asymmetry(
        condensed_operator(fd.derivative(solver.F, z)))

    for name, mesh_builder in (("tetrahedra", cube_mesh),
                               ("hexahedra", hexahedral_cube_mesh)):
        mesh = mesh_builder()
        approx = approximation(mesh, 1, exponent=3)
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        S = approx.internal_variable_space(mesh)
        z = fd.Function(V * S)
        X = fd.SpatialCoordinate(mesh)
        z.subfunctions[0].interpolate(0.05 * (X + 0.3 * X[0] * X))
        z.subfunctions[1].interpolate(trace_free_history(mesh, S))
        solver = CoupledInternalVariableSolver(
            z, approx, dt=DT, bcs=strong_bcs_3d(mesh), solver_parameters="direct"
        )
        record[f"power_symmetry-{name}_asymmetry"] = asymmetry(
            condensed_operator(fd.derivative(solver.F, z)))
        record[f"power_symmetry-{name}_rule_says_symmetric"] = int(
            solver.condensed_operator_symmetric())


def operator_reuse(record):
    """The condensed operator is built once per change of the Jacobian.

    A Newtonian march with a fixed time step has a constant Jacobian, so the
    preconditioner assembles the condensed operator once. A change of the time
    step or of a material field, or an explicit `invalidate_jacobian`, makes
    it assemble once more. A fresh solver at the changed coefficients gives
    the displacement of the reused operator. A constant exponent of one is
    Newtonian too, so it keeps `ksponly` and one assembly per march.
    """
    mesh = square_mesh()
    dt = fd.Constant(DT)
    approx = approximation(mesh, 2)
    # A material field that the row can change in place.
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    approx.viscosity[0] = fd.Function(DG0).assign(VISCOSITIES[0])
    approx.maxwell_times[0] = approx.viscosity[0] / approx.shear_modulus[0]
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    z = fd.Function(V * approx.internal_variable_space(mesh))
    solver = CoupledInternalVariableSolver(
        z, approx, dt=dt, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    for _ in range(3):
        solver.solve()
    pc = condensation_pc(solver)
    record["reuse-count_after_three_solves"] = pc.assembly_count

    dt.assign(0.5 * DT)
    solver.solve()
    record["reuse-count_after_dt_change"] = pc.assembly_count
    solver.solve()
    record["reuse-count_after_same_dt"] = pc.assembly_count

    approx.viscosity[0].assign(2 * VISCOSITIES[0])
    solver.solve()
    record["reuse-count_after_viscosity_change"] = pc.assembly_count

    solver.invalidate_jacobian()
    solver.solve()
    record["reuse-count_after_invalidate"] = pc.assembly_count

    # A fresh solver at the changed coefficients, started from the same state,
    # must give the displacement of the reused operator.
    fresh_approx = approximation(mesh, 2)
    fresh_approx.viscosity[0] = fd.Function(DG0).assign(2 * VISCOSITIES[0])
    fresh_approx.maxwell_times[0] = (
        fresh_approx.viscosity[0] / fresh_approx.shear_modulus[0])
    z_fresh = fd.Function(z.function_space())
    fresh = CoupledInternalVariableSolver(
        z_fresh, fresh_approx, dt=dt, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    z_fresh.assign(z)
    fresh.solution_old.assign(solver.solution_old)
    fresh.solve()
    solver.solve()
    u, u_fresh = z.subfunctions[0], z_fresh.subfunctions[0]
    record["reuse-u_norm"] = fd.norm(u)
    record["reuse-fresh_u_error"] = relative_error(u, u_fresh)

    # A constant exponent equal to one.
    approx = approximation(mesh, 2, exponent=fd.Constant(1.0))
    z = fd.Function(V * approx.internal_variable_space(mesh))
    solver = CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=weak_bcs(mesh), solver_parameters="iterative"
    )
    record["reuse_constant_exponent-snes_type"] = solver.solver_parameters["snes_type"]
    for _ in range(3):
        solver.solve()
    record["reuse_constant_exponent-assembly_count"] = condensation_pc(
        solver).assembly_count


def singular_condensed_operator(record):
    """A translation reaches the condensed operator, and CG solves the singular system.

    Weak normal conditions on the bottom and the top, free vertical walls and
    a balanced tangential load on the top: the horizontal translation `(1, 0)`
    is then in the kernel of the operator. Buoyancy is off, because the
    hydrostatic prestress term makes a translation across a free wall do
    work, which removes the mode. The row writes whether the three
    nullspaces are set on the condensed operator, the image of the mode, and
    the gap to a direct solve after both are centred, so that the mode does
    not enter the comparison.
    """
    mesh = square_mesh()
    approx = approximation(mesh, 2, B_mu=0.0)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    Z = V * approx.internal_variable_space(mesh)
    bids = list(gadopt.get_boundary_ids(mesh))
    X = fd.SpatialCoordinate(mesh)
    bcs = {
        bids[2]: {"un": 0},
        bids[3]: {"un": 0,
                  "stress": fd.as_vector([0.1 * fd.sin(2 * fd.pi * X[0]), 0.0])},
    }
    translation = gadopt.rigid_body_modes(V, rotational=False, translations=[0])
    exact = fd.MixedVectorSpaceBasis(Z, [translation, Z.sub(1)])
    z = fd.Function(Z)
    solver = CoupledInternalVariableSolver(
        z, approx, dt=DT, bcs=bcs, solver_parameters="iterative",
        solver_parameters_extra={"condensed_field_ksp_rtol": CONDENSED_RTOL},
        nullspace=exact, transpose_nullspace=exact, near_nullspace=exact,
    )
    solver.solve()
    pc = condensation_pc(solver)
    S = pc.S.petscmat
    record["singular-nullspace_set"] = int(S.getNullSpace().handle != 0)
    record["singular-transpose_nullspace_set"] = int(
        S.getTransposeNullSpace().handle != 0)
    record["singular-near_nullspace_set"] = int(S.getNearNullSpace().handle != 0)
    # The image of the translation, relative to ||S|| ||mode||.
    mode = fd.Function(V).interpolate(fd.as_vector([1.0, 0.0]))
    with mode.dat.vec_ro as v:
        image = v.duplicate()
        S.mult(v, image)
        record["singular-mode_image"] = image.norm() / (S.norm() * v.norm())
    record["singular-condensed_ksp_reason"] = pc.condensed_ksp.getConvergedReason()
    record["singular-condensed_ksp_type"] = pc.condensed_ksp.getType()

    z_direct = fd.Function(Z)
    CoupledInternalVariableSolver(
        z_direct, approximation(mesh, 2, B_mu=0.0), dt=DT, bcs=bcs,
        solver_parameters="direct", nullspace=exact, transpose_nullspace=exact,
    ).solve()

    def centred(u):
        """Remove the mean horizontal displacement, the component along the mode."""
        mean = fd.assemble(u[0] * fd.dx) / fd.assemble(1 * fd.dx(mesh))
        return u - fd.as_vector([mean, 0.0])

    difference = centred(z.subfunctions[0]) - centred(z_direct.subfunctions[0])
    reference = centred(z_direct.subfunctions[0])
    record["singular-u_reference_norm"] = float(
        fd.sqrt(fd.assemble(fd.inner(reference, reference) * fd.dx)))
    record["singular-u_difference_norm"] = float(
        fd.sqrt(fd.assemble(fd.inner(difference, difference) * fd.dx)))


def substituted_coupled_gap(n):
    """Relative displacement gap between the two formulations on an n x n mesh.

    With weak normal conditions the coupled form carries the elastic penalty
    and the boundary term of the history equation, the substituted form the
    effective penalty, so the gap is a boundary discretisation error.

    Args:
      n: The number of cells along each side.

    Returns:
      The relative L2 gap of the displacement after one step.
    """
    mesh = square_mesh(n)
    bcs = weak_bcs(mesh)
    approx = approximation(mesh, 2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = approx.internal_variable_space(mesh)
    u = fd.Function(V)
    gadopt.InternalVariableSolver(
        u, approx, dt=DT, internal_variables=fd.Function(S), bcs=bcs,
        solver_parameters="direct",
    ).solve()
    _, z = coupled_solve(mesh, 2, bcs, "direct", steps=1)
    return relative_error(u, z.subfunctions[0])


def substituted_against_coupled(record):
    """The gap between the substituted and the coupled solver shrinks under refinement.

    Both are consistent discretisations of the same time step, so the gap is a
    discretisation error and falls with the mesh size.
    """
    record["gap-coarse"] = substituted_coupled_gap(4)
    record["gap-fine"] = substituted_coupled_gap(8)


# --------------------------------------------------------------------------
# Entry "pc"


def box_mesh(dim, n=8):
    """A unit square or cube of affine simplices, `n` cells along each side."""
    return fd.UnitSquareMesh(n, n) if dim == 2 else fd.UnitCubeMesh(n, n, n)


def deviatoric(e, dim):
    """`e - tr(e) I / 3` in every dimension, as the internal-variable stress uses."""
    return e - (1.0 / 3.0) * fd.tr(e) * fd.Identity(dim)


def elasticity_problem(mesh, ratio):
    r"""The displacement tangent of the compressible internal-variable stress.

    $$ a(u, v) = 2 \mu\, \mathrm{dev}\,\varepsilon(u) : \mathrm{dev}\,
       \varepsilon(v) + \lambda (\nabla\cdot u)(\nabla\cdot v) $$

    with `mu = 1` and `lambda = ratio`, a unit body force in the last
    direction and one clamped face. A large ratio makes the operator nearly
    incompressible.

    Args:
      mesh: The mesh.
      ratio: The ratio of the volumetric to the shear coefficient.

    Returns:
      The bilinear form, the linear form, the boundary condition and the
      solution Function.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    dim = mesh.geometric_dimension
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    a = (2 * fd.inner(deviatoric(fd.sym(fd.grad(u)), dim),
                      deviatoric(fd.sym(fd.grad(v)), dim))
         + fd.Constant(ratio) * fd.div(u) * fd.div(v)) * fd.dx
    load = [0.0] * dim
    load[-1] = -1.0
    L = fd.inner(fd.as_vector(load), v) * fd.dx
    bc = fd.DirichletBC(V, fd.as_vector([0.0] * dim), 1)
    return a, L, bc, fd.Function(V)


def elasticity_iterations(mesh, ratio, params):
    """Solve `elasticity_problem` and return the Krylov iteration count and reason.

    Args:
      mesh: The mesh.
      ratio: The ratio of the volumetric to the shear coefficient.
      params: The solver parameters.

    Returns:
      The iteration count and the converged reason of the Krylov solve.
    """
    a, L, bc, uh = elasticity_problem(mesh, ratio)
    problem = fd.LinearVariationalProblem(a, L, uh, bcs=bc)
    solver = fd.LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    ksp = solver.snes.ksp
    return ksp.getIterationNumber(), ksp.getConvergedReason()


def assembled_pc_parameters(pc_python):
    """CG on an assembled GAMG with the near-nullspace of `pc_python`."""
    return {"ksp_type": "cg", "ksp_rtol": 1e-8, "ksp_max_it": 5000,
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "python", "pc_python_type": pc_python,
            **gamg_parameters("assembled_")}


def divergence_free_modes(record):
    """The solenoidal modes are divergence-free, and the filter drops the others.

    On an affine simplex mesh a polynomial of degree two is reproduced exactly
    in CG2, so every generated field is divergence-free to round-off. A curved
    cubed sphere reproduces only degree-one fields exactly, so the degree-two
    fields carry a few percent of divergence and the filter must drop them.
    """
    for dim in CONDENSATION_SOLENOIDAL_DIMS:
        V = fd.VectorFunctionSpace(box_mesh(dim), "CG", 2)
        modes = solenoidal_modes(V, max_degree=2)
        record[f"modes-{dim}d-count"] = len(modes)
        record[f"modes-{dim}d-max_divergence"] = max(
            relative_divergence(m) for m in modes)

    base = fd.CubedSphereMesh(radius=1.22, refinement_level=1, degree=2)
    shell = fd.ExtrudedMesh(base, layers=3, layer_height=1.0 / 3,
                            extrusion_type="radial")
    V = fd.VectorFunctionSpace(shell, "CG", 2)
    unfiltered = solenoidal_modes(V, max_degree=2)
    filtered = solenoidal_modes(V, max_degree=2, divfree_tol=1e-8)
    record["modes-curved-unfiltered_count"] = len(unfiltered)
    record["modes-curved-filtered_count"] = len(filtered)
    record["modes-curved-unfiltered_max_divergence"] = max(
        relative_divergence(m) for m in unfiltered)
    record["modes-curved-filtered_max_divergence"] = max(
        relative_divergence(m) for m in filtered)


def near_incompressible_iterations(record):
    """The divergence-free modes lower the CG iteration count on GAMG.

    The same nearly incompressible operator (ratio 1000, 32 by 32 cells)
    solved with no near-nullspace, with the rigid-body modes and with the
    rigid-body modes enriched by the divergence-free ones. A second row at
    ratio 100 on 48 by 48 cells solves with the default mode count, which
    must leave GAMG a full-rank prolongator.
    """
    mesh = fd.UnitSquareMesh(32, 32)
    for name, pc_python in (("none", "firedrake.AssembledPC"),
                            ("rigid", "gadopt.RigidBodyAssembledPC"),
                            ("incompressible", "gadopt.NearlyIncompressibleAssembledPC")):
        its, reason = elasticity_iterations(
            mesh, 1000.0, assembled_pc_parameters(pc_python))
        record[f"iterations-{name}"] = its
        record[f"iterations-{name}_reason"] = reason

    its, reason = elasticity_iterations(
        fd.UnitSquareMesh(48, 48), 100.0,
        assembled_pc_parameters("gadopt.NearlyIncompressibleAssembledPC"))
    record["iterations-default_modes"] = its
    record["iterations-default_modes_reason"] = reason


def patch_multigrid(record):
    """The patch multigrid parameters converge on a nearly incompressible operator.

    Geometric multigrid on a two-level hierarchy with vertex-star patch
    smoothers, over a GAMG coarse solve with the divergence-free
    near-nullspace or over LU, at the ratio 1000.
    """
    mesh = fd.MeshHierarchy(fd.UnitSquareMesh(6, 6), 2)[-1]
    for coarse in CONDENSATION_MG_COARSE:
        params = {"ksp_type": "fgmres", "ksp_rtol": 1e-8, "ksp_max_it": 2000,
                  "ksp_norm_type": "unpreconditioned",
                  **nearly_incompressible_mg_parameters(coarse=coarse)}
        its, reason = elasticity_iterations(mesh, 1000.0, params)
        record[f"mg-{coarse}-iterations"] = its
        record[f"mg-{coarse}-reason"] = reason


def taped_derivative_solvers(record):
    """The taped solve builds the derivative solvers with the adjoint options.

    A power-law forward runs `newtonls`, so `ksponly` on the derivative
    solvers can only come from `adjoint_solver_parameters`. The row counts,
    for each derivative solver that Firedrake builds, the forward-only SNES
    options that reached it and the flat Krylov options that differ from the
    adjoint options. Firedrake flattens nested option dictionaries, so only
    the flat Krylov options are compared.
    """
    from gadopt.utility import CACHED_SOLVER_BLOCKS

    mesh = cube_mesh(2)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0, g=1.0,
        exponent=3, transition_stress=1.0)
    solver = CoupledInternalVariableSolver(
        fd.Function(fd.MixedFunctionSpace([V, S])), approx, dt=1.0,
        bcs={1: {"ux": 0}},
    )
    record["taped-forward_snes_type"] = solver.solver_parameters["snes_type"]
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        solver.solve()
    finally:
        pause_annotation()
    if CACHED_SOLVER_BLOCKS:
        derivative_solvers = [solver.solver._ad_adjoint_cache.solver,
                              solver.solver._ad_tangent_cache.solver]
    else:
        from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
        blocks = [b for b in tape.get_blocks()
                  if isinstance(b, NonlinearVariationalSolveBlock)]
        derivative_solvers = [blocks[-1]._ad_solvers["adjoint_lvs"]]
    tape.clear_tape()

    expected = solver.adjoint_solver_parameters()
    krylov = {key: value for key, value in expected.items()
              if not key.startswith("snes_") and not isinstance(value, dict)}
    record["taped-n_derivative_solvers"] = len(derivative_solvers)
    record["taped-krylov_options_compared"] = len(krylov)
    for i, derivative_solver in enumerate(derivative_solvers):
        parameters = derivative_solver.parameters
        record[f"taped-{i}-snes_type"] = parameters["snes_type"]
        record[f"taped-{i}-forward_only_options"] = sum(
            option in parameters for option in solver.FORWARD_ONLY_SNES_OPTIONS)
        record[f"taped-{i}-krylov_mismatches"] = sum(
            parameters.get(key) != value for key, value in krylov.items())


#: What `SpyAssembledPC` saw on its assembled block at setup.
SPY_SEEN = {}


class SpyAssembledPC(SPDAssembledPC):
    """`SPDAssembledPC` that records the near-nullspace and the SPD flag it set."""

    def initialize(self, pc):
        """Set up as `SPDAssembledPC` does, then read what reached the matrix."""
        super().initialize(pc)
        nullspace = self.P.petscmat.getNearNullSpace()
        SPY_SEEN["vecs"] = len(nullspace.getVecs()) if nullspace.handle else 0
        SPY_SEEN["spd"] = self._spd_wanted


def spd_with_modes(record):
    """The SPD flag and a near-nullspace reach one assembled block together.

    The row reads the near-nullspace that is attached to the assembled matrix,
    so a solve that ignored both options gives zero vectors and no SPD flag.
    """
    mesh = cube_mesh(2)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + fd.inner(u, v) * fd.dx
    L = fd.inner(fd.Constant((1.0, 0.0, 0.0)), v) * fd.dx
    uh = fd.Function(V)
    fd.solve(a == L, uh, solver_parameters={
        "ksp_type": "cg", "pc_type": "python",
        "pc_python_type": f"{__name__}.SpyAssembledPC",
        "near_nullspace": "incompressible",
        "spd": True,
        "assembled_pc_type": "gamg",
    })
    record["spd-near_nullspace_vectors"] = SPY_SEEN.get("vecs", -1)
    record["spd-flag"] = int(SPY_SEEN.get("spd", False))
    record["spd-solution_norm"] = fd.norm(uh)


ENTRIES = {
    "preset": [preset_against_direct, power_law_under_preset,
               extruded_hexahedra, history_layouts],
    "operator": [symmetry_with_boundary_term, power_law_asymmetry,
                 operator_reuse, singular_condensed_operator,
                 substituted_against_coupled],
    "pc": [divergence_free_modes, near_incompressible_iterations,
           patch_multigrid, taped_derivative_solvers, spd_with_modes],
}
assert sorted(ENTRIES) == sorted(CONDENSATION_ENTRIES)


def format_value(value):
    """Format a value for the output file: 17 significant digits for a float."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def write_values(path, record):
    """Write one line `name value` per entry of `record`. Only rank 0 writes.

    Args:
      path: The output file.
      record: The mapping of names to numbers or single-word strings.
    """
    if fd.COMM_WORLD.rank != 0:
        return
    with open(path, "w") as f:
        for name, value in record.items():
            f.write(f"{name} {format_value(value)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, choices=CONDENSATION_ENTRIES,
                        help="which group of rows to run")
    args = parser.parse_args()

    record = {}
    for row in ENTRIES[args.entry]:
        row(record)
        # Print each row as it finishes, so that a failure later in the
        # entry leaves the earlier numbers in the log.
        print(f"{row.__name__} done", flush=True)
    write_values(f"condensation-{args.entry}.dat", record)
