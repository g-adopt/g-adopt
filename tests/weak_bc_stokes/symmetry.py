"""Algebraic diagnostics of the weak boundary terms, for one test group on one mesh.

Each invocation runs one group on one mesh and writes its numbers to a `.dat`
file. `test_weak_bc_stokes.py` reads those files and applies the tolerances.

The split into one process per (group, mesh) pair exists because the cost of
these checks is firedrake kernel compilation, not solving, and compilation is
serial within a process. `doit run_case` runs the steps across the machine, so
the wall clock becomes the slowest single step instead of the sum.

Usage:
    python3 symmetry.py --test <group> --mesh <key>
"""

import argparse

import firedrake as fd
import gadopt
import numpy as np
from gadopt.equations import Equation
from gadopt.momentum_equation import viscosity_term

from stokes_helpers import (
    STRUCTURE_CASES,
    SYMMETRY_APPROXIMATIONS,
    WEAK_BOUNDARY_CASES,
    asymmetry,
    build_incompressible_maxwell_weak_un_case,
    build_internal_variable_weak_u_case,
    build_internal_variable_weak_un_case,
    build_mesh,
    build_stokes_weak_un_case,
    deviatoric_tensor,
    exterior_facet_form,
    form_agreement,
    generic_velocity,
    nonlinear_mu,
    penalty_coefficient,
    taylor_hood,
)


def build_approximation(approx_class, mu):
    """Instantiate `approx_class` with viscosity `mu`.

    The compressible approximations take a dissipation number; the Boussinesq
    one does not.
    """
    if approx_class is gadopt.BoussinesqApproximation:
        return approx_class(1, mu=mu)
    return approx_class(1, Di=1, mu=mu)


def stokes_bcs(mesh):
    """Weak boundary conditions covering every kind the solver accepts.

    The first two identifiers carry a weak normal-velocity and a normal-stress
    condition. Cylinder and sphere meshes have only two boundaries; where there
    are more, the remaining two exercise the full-stress and the velocity kinds
    as well. "u" is converted to a strong DirichletBC by the solver, so it does
    not reach the weak branch here; `weak_u_symmetry` covers that branch.
    """
    bids = list(gadopt.get_boundary_ids(mesh))
    bcs = {bids[0]: {"un": 0}, bids[1]: {"normal_stress": 0}}
    if len(bids) > 2:
        zero_vec = fd.Constant([0] * mesh.geometric_dimension)
        bcs[bids[2]] = {"stress": zero_vec}
        bcs[bids[3]] = {"u": zero_vec}
    return bcs


def nonlinear_viscosity(mesh):
    """Asymmetry of the true Jacobian of the solver residual, per approximation.

    With a strain-rate-dependent viscosity the weak boundary terms use the
    tangent stress and a penalty-derivative term, so the boundary residual is
    the exact first variation of a boundary functional and `derivative(F, z)` is
    symmetric by construction. No custom Jacobian is built, so `solver.J` must
    be None and the raw derivative is what gets measured.

    Returns:
      One row per approximation: the asymmetry ratio, and 1.0 if `solver.J` is
      None as it must be, 0.0 otherwise.
    """
    rows = []
    for _, approx_class in SYMMETRY_APPROXIMATIONS:
        Z = taylor_hood(mesh)
        z = fd.Function(Z)
        z.subfunctions[0].interpolate(generic_velocity(mesh))
        u, _ = fd.split(z)

        compressible = approx_class is not gadopt.BoussinesqApproximation
        approximation = build_approximation(
            approx_class, nonlinear_mu(u, compressible))

        T = fd.Function(Z.sub(1))
        solver = gadopt.StokesSolver(
            z, approximation, T, bcs=stokes_bcs(mesh))

        M = fd.assemble(fd.derivative(solver.F, z),
                        mat_type="nest" if approximation.compressible else "aij")
        if approximation.compressible:
            # Only the velocity block is symmetric for a compressible stress.
            # That block is assembled as 'baij', whose norm and transpose behave,
            # but convert to 'aij' to match the incompressible path exactly.
            petscmat = M.petscmat.getNestSubMatrix(0, 0).convert("aij")
        else:
            petscmat = M.petscmat
        rows.append([asymmetry(petscmat), float(solver.J is None)])
    return np.array(rows)


def weak_u_symmetry(mesh):
    """Asymmetry of the weak "u" branch, per approximation.

    `StokesSolver` converts a "u" boundary condition to a strong `DirichletBC`,
    so that branch is never reached from the solver. Driving `viscosity_term`
    directly on a velocity-only space keeps "u" weak. A generic linearisation
    point is needed so the "u" penalty coefficient is exercised at all.

    Returns:
      One row per approximation, holding the asymmetry ratio.
    """
    rows = []
    for _, approx_class in SYMMETRY_APPROXIMATIONS:
        compressible = approx_class is not gadopt.BoussinesqApproximation

        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        u = fd.Function(V).interpolate(generic_velocity(mesh))
        approximation = build_approximation(
            approx_class, nonlinear_mu(u, compressible))

        zero_vec = fd.Constant([0] * mesh.geometric_dimension)
        bids = list(gadopt.get_boundary_ids(mesh))
        # Exercise both the weak "u" and the weak "un" branch.
        bcs = {bids[0]: {"u": zero_vec}, bids[1]: {"un": 0}}

        eq = Equation(
            fd.TestFunction(V), V, viscosity_term,
            eq_attrs={"stress": approximation.stress(u)},
            approximation=approximation, bcs=bcs, quad_degree=6,
        )
        M = fd.assemble(fd.derivative(eq.residual(u), u), mat_type="aij")
        rows.append([asymmetry(M.petscmat)])
    return np.array(rows)


def variational_structure(mesh):
    r"""Distance from the weak boundary residual to the first variation of its functional.

    Symmetry cannot see an error that keeps the residual symmetric: a mis-scaled
    penalty, or a wrong constant in a term that is still the first variation of
    some functional, leaves the Jacobian symmetric. Being consistent it also
    keeps the optimal convergence order, so the solver-output cases do not see
    it either. The property that pins such an error is that the residual is the
    first variation of the boundary functional the code documents,

    $$ E_{bdy} = \int_{\partial\Omega} \left[ -w \cdot \sigma(u) n
       + \sigma_{pen}\,\mu\,\langle G, A(G) \rangle \right] ds, $$

    with $G = n \otimes w$, $A$ the deviatoric stress per $\mu$, and
    $w = u - u_D$ for weak "u" or $w = (n \cdot u - u_n) n$ for weak "un".

    The flux part of E_bdy reuses `approximation.stress`, so a bug inside
    `stress` itself sits on both sides of the identity and is not caught here.
    The MMS cases pin `stress` absolutely against a manufactured field instead.

    Returns:
      One row per (bc kind, compressible) pair, holding
      ||F_bdy - dE|| / ||dE||.
    """
    dim = mesh.geometric_dimension
    rows = []
    for bc_kind, compressible in STRUCTURE_CASES:
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        u = fd.Function(V).interpolate(generic_velocity(mesh))
        mu = nonlinear_mu(u, compressible)

        if compressible:
            approximation = gadopt.TruncatedAnelasticLiquidApproximation(
                1, Di=1, mu=mu)
        else:
            approximation = gadopt.BoussinesqApproximation(1, mu=mu)

        bids = list(gadopt.get_boundary_ids(mesh))
        # Nonzero boundary data keeps the jump away from zero, so a wrong
        # coefficient in a term proportional to it is observable.
        u_D = fd.Constant([0.1 * (i + 1) for i in range(dim)])
        un = 0.2
        bcs = {bid: ({"u": u_D} if bc_kind == "u" else {"un": un})
               for bid in bids}

        eq = Equation(
            fd.TestFunction(V), V, viscosity_term,
            eq_attrs={"stress": approximation.stress(u)},
            approximation=approximation, bcs=bcs, quad_degree=8,
        )
        F_bdy = exterior_facet_form(eq.residual(u))

        sigma = penalty_coefficient(eq)
        n = eq.n
        stress_u = approximation.stress(u)
        # For "un" the jump keeps only its normal component; both branches then
        # share the same functional with G = outer(n, w).
        w = (u - u_D) if bc_kind == "u" else (fd.dot(n, u) - un) * n
        G = fd.outer(n, w)
        E_bdy = sum(
            (-fd.dot(w, fd.dot(stress_u, n))
             + sigma * mu * fd.inner(G, deviatoric_tensor(G, compressible)))
            * eq.ds(bid)
            for bid in bids
        )
        dE = fd.derivative(E_bdy, u, fd.TestFunction(V))

        residual = fd.assemble(F_bdy - dE)
        reference = fd.assemble(dE)
        rows.append([residual.dat.norm / reference.dat.norm])
    return np.array(rows)


def explicit_forms(mesh):
    """Agreement of the weak boundary residual with terms written out by hand.

    `viscosity_term` builds the symmetrising term by differentiating the stress
    the equation carries, and the penalty from the approximation's
    stress-from-gradient helper. Both are generic, so nothing in the code says
    which coefficient ends up in front of which term. The reference forms in
    `stokes_helpers` rebuild every coefficient from raw approximation attributes
    (`bulk_modulus`, `bulk_shear_ratio`, `viscosity`, `shear_modulus`, `dt`), so
    a wrong bulk coefficient, a dropped bulk penalty, a penalty raised from the
    effective viscosity to the elastic shear modulus, or a sign flip in the
    symmetrising term all show up.

    Returns:
      One row per case in `WEAK_BOUNDARY_CASES` order, holding the residual and
      the Jacobian difference relative to the reference.
    """
    bc_id = list(gadopt.get_boundary_ids(mesh))[0]
    rows = []
    for case in WEAK_BOUNDARY_CASES:
        match case:
            case "Maxwell":
                built = build_internal_variable_weak_un_case(
                    mesh, [2.0], [2.0], bc_id)
            case "Maxwell-weak-u":
                built = build_internal_variable_weak_u_case(mesh, bc_id)
            case "Burgers":
                built = build_internal_variable_weak_un_case(
                    mesh, [2.0, 0.5], [2.0, 0.1], bc_id)
            case "IncompressibleMaxwell":
                built = build_incompressible_maxwell_weak_un_case(mesh, bc_id)
            case _:
                approx_class, compressible, nonlinear = WEAK_BOUNDARY_CASES[case]
                built = build_stokes_weak_un_case(
                    mesh, approx_class, compressible, nonlinear, bc_id)
        rows.append(list(form_agreement(*built)))
    return np.array(rows)


GROUPS = {
    "nonlinear_viscosity": nonlinear_viscosity,
    "weak_u_symmetry": weak_u_symmetry,
    "variational_structure": variational_structure,
    "explicit_forms": explicit_forms,
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
