"""Algebraic diagnostics of the viscoelastic weak boundary terms, one group on one mesh.

The two viscoelastic solvers linearise the momentum stress differently.
`InternalVariableSolver` substitutes the backward-Euler update of the internal
variables into the stress, so its tangent carries the effective viscosity.
`CoupledInternalVariableSolver` solves for the displacement and the internal
variables together, so at fixed history its stress is elastic and its tangent
carries the elastic modulus. Each solver therefore needs its own reference, and
a symmetrising term built with the wrong one of the two leaves the displacement
block asymmetric by an amount that grows with the time step.

Usage:
    python3 gia_symmetry.py --test <group> --mesh <key>
"""

import argparse

import firedrake as fd
import gadopt
import numpy as np
from gadopt.equations import Equation
from gadopt.momentum_equation import viscosity_term

from gia_helpers import (
    BLOCK_SYMMETRY_CASES,
    GIA_BULK_MODULUS,
    GIA_BULK_SHEAR_RATIO,
    GIA_DT,
    MAXWELL_TIME,
    WEAK_U_HISTORIES,
    WEAK_UN_VALUE,
    asymmetry,
    build_mesh,
    exterior_facet_form,
    first_variation,
    generic_velocity,
    history_state,
    maxwell_approximation,
    raw_effective_viscosity,
    raw_internal_variable_stress,
    raw_internal_variables_update,
    weak_un_functional,
)


def displacement_and_history_spaces(mesh):
    """The P2 displacement space and the DG1 internal-variable space."""
    return (fd.VectorFunctionSpace(mesh, "CG", 2),
            fd.TensorFunctionSpace(mesh, "DG", 1))


def block_symmetry(mesh):
    """Asymmetry of the displacement block of the coupled Jacobian.

    At fixed history the coupled momentum stress is elastic, so its shear
    coefficient is the elastic modulus mu_0 and not the effective viscosity. A
    symmetrising term built with the effective viscosity leaves the (0,0) block
    asymmetric by (mu_0 - eta_eff)(C - C^T), an error that grows with dt/tau.
    The default coupled preset preconditions that block with CG, which assumes
    a symmetric operator.

    Only the displacement block is expected to be symmetric. The full coupled
    Jacobian is not: the internal-variable rows are scaled independently, and
    for composite creep the Maxwell times depend on the stress.

    Returns:
      One row per case in `BLOCK_SYMMETRY_CASES`, holding the asymmetry ratio.
    """
    V, S = displacement_and_history_spaces(mesh)
    rows = []
    for dt_over_tau, exponent in BLOCK_SYMMETRY_CASES:
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
        block = jacobian.petscmat.getNestSubMatrix(0, 0).convert("aij")
        rows.append([asymmetry(block)])
    return np.array(rows)


def pointwise_structure(mesh):
    """Distance from the pointwise weak "un" residual to its boundary functional.

    The functional is written with the full internal-variable stress, bulk part
    and history included, rebuilt from raw attributes, and with the penalty held
    at the effective viscosity. `InternalVariableSolver` substitutes the
    backward-Euler update into the stress, so differentiating the functional
    differentiates through that update too. The result carries the effective
    viscosity, which is what makes the pointwise symmetrising term and the
    pointwise penalty share a coefficient.

    Returns:
      A single row holding ||F_bdy - dE|| / ||dE||.
    """
    V, S = displacement_and_history_spaces(mesh)
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
    return np.array([[first_variation(form, functional, u)]])


def coupled_structure(mesh):
    """Distance from the coupled weak "un" residual to its functional at fixed history.

    In the coupled formulation the internal variables are unknowns, so the
    displacement rows must be the first variation of the boundary functional
    taken with the history held fixed. The stress there is elastic in the
    displacement, so both its variation and the penalty carry mu_0. The
    reference therefore differs from the pointwise one through the stress and
    through the penalty coefficient.

    The history is supplied as a separate Function holding the same values as
    the internal-variable component of the solution, so that differentiating
    with respect to the mixed solution varies the displacement only. The
    internal-variable rows carry no boundary term.

    Returns:
      A single row holding ||F_bdy - dE|| / ||dE||.
    """
    V, S = displacement_and_history_spaces(mesh)
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
        mu_penalty=sum(approximation.shear_modulus),
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    return np.array([[first_variation(form, functional, z)]])


def weak_u_symmetry(mesh):
    """Asymmetry of the weak "u" branch for the viscoelastic stress.

    Every `StokesSolverBase` subclass turns a "u" boundary condition into a
    strong `DirichletBC`, so this branch is reachable only by driving
    `viscosity_term` at the `Equation` level. It has to work for a stress with a
    bulk part: the tangent and the penalty both pick that part up, and the
    residual is the first variation of a boundary functional, so its Jacobian is
    a Hessian and symmetric.

    Returns:
      One row per entry of `WEAK_U_HISTORIES`, holding the asymmetry ratio.
    """
    V, S = displacement_and_history_spaces(mesh)
    dim = mesh.geometric_dimension
    rows = []
    for history in WEAK_U_HISTORIES:
        u = fd.Function(V).interpolate(generic_velocity(mesh))
        m = history_state(mesh, S)
        approximation = maxwell_approximation(mesh)
        # Driving the Equation directly selects the effective viscosity as the
        # penalty scale, matching InternalVariableSolver.
        approximation.mu = approximation.effective_viscosity(GIA_DT)

        if history == "pointwise":
            internal_variables = raw_internal_variables_update(
                approximation, u, [m], GIA_DT)
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
            fd.TestFunction(V), V, viscosity_term,
            eq_attrs={"stress": stress},
            approximation=approximation, bcs=bcs, quad_degree=6,
        )
        jacobian = fd.assemble(fd.derivative(eq.residual(u), u), mat_type="aij")
        rows.append([asymmetry(jacobian.petscmat)])
    return np.array(rows)


GROUPS = {
    "block_symmetry": block_symmetry,
    "pointwise_structure": pointwise_structure,
    "coupled_structure": coupled_structure,
    "weak_u_symmetry": weak_u_symmetry,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, choices=sorted(GROUPS),
                        help="which group of diagnostics to run")
    parser.add_argument("--mesh", required=True,
                        help="key of the mesh to run on, see MESH_BUILDERS")
    args = parser.parse_args()

    result = GROUPS[args.test](build_mesh(args.mesh))
    np.savetxt(f"{args.test}-{args.mesh}.dat", result)
