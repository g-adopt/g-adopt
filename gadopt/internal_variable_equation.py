r"""History equations of the internal-variable viscoelastic solvers.

A Maxwell element with shear modulus $\mu_i$ and Maxwell time $\tau_i$ carries
one internal variable $m_i$, a symmetric tensor (trace free in 3-D; in 2-D it
carries the `tr/3` residual of the fixed-2/3 deviatoric operator) that relaxes
towards the deviatoric strain $d(u) = \mathrm{dev}\,\varepsilon(u)$ of the
displacement:

$$ \dot m_i = \frac{d(u) - m_i}{\tau_i}. $$

The rheology of a G-ADOPT approximation is a list of such elements, one per
entry of `maxwell_times`. This module stores every element of that list in
one discontinuous tensor field of shape `(n, d, d)`, where `n` is the number
of Maxwell elements and `d` the geometric dimension, so that the number of
elements never appears in a function-space layout, a fieldsplit index, a
static-condensation option or a checkpoint. A single Maxwell element gives a
field of shape `(1, d, d)`. The approximation keeps taking a list of `n`
tensor expressions, which `history_slices` builds by slicing the combined
field. The older layout, one `(d, d)` field per element, is still accepted by
the same helpers so that existing drivers keep working.

The backward-Euler history equation of one element,

$$ \frac{m_i^{new} - m_i^{old}}{\Delta t} + \frac{m_i^{new}}{\tau_i}
   - \frac{d(u^{new})}{\tau_i} = 0, $$

is written here as three residual terms on the combined field, each of which
sums over the elements internally: `history_mass_term`,
`history_relaxation_term` and `history_strain_term`. The last one also
carries the boundary term that makes the coupled displacement Schur
complement symmetric under weak displacement boundary conditions; see its
docstring.
"""

from typing import Any

import firedrake as fd
from firedrake import TensorFunctionSpace, inner, outer, dot
from ufl.core.expr import Expr

from .equations import Equation

__all__ = [
    "internal_variable_space",
    "history_slices",
    "assign_history_slices",
    "history_mass_term",
    "history_relaxation_term",
    "history_strain_term",
    "internal_variable_history_terms",
]


def internal_variable_space(
    mesh: fd.MeshGeometry,
    n_maxwell: int,
    degree: int = 1,
    symmetric: bool = True,
) -> fd.functionspaceimpl.WithGeometry:
    r"""Build the discontinuous tensor space for all internal variables.

    The space has value shape `(n_maxwell, d, d)` with `d` the geometric
    dimension of `mesh`. The internal variables relax towards the deviatoric
    strain, which is symmetric, and the backward-Euler update keeps them
    symmetric, so the two trailing indices carry a symmetry by default. UFL
    still sees a full `(d, d)` block for every element, so every form and
    every interpolation reads as for a full tensor; only the storage, the
    tape, the checkpoints and the cell-local blocks that static condensation
    inverts shrink, from `d * d` to `d (d + 1) / 2` components per element.

    Firedrake's automatic `symmetry=True` refuses a non-square shape, so the
    symmetry is spelled out as the map from each below-diagonal component to
    its above-diagonal partner within one element.

    Args:
      mesh: the mesh the space is built on.
      n_maxwell: the number of Maxwell elements, `len(approximation.maxwell_times)`.
      degree: polynomial degree of the discontinuous element. The default, 1,
        matches the P2 displacement the GIA drivers use.
      symmetric: store only the independent components of each `(d, d)` block.

    Returns:
      A `TensorFunctionSpace` with the shape above.

    """
    if n_maxwell < 1:
        raise ValueError(f"n_maxwell must be at least 1, got {n_maxwell}.")
    dim = mesh.geometric_dimension
    shape = (n_maxwell, dim, dim)
    # "DG" selects the discontinuous Lagrange family on simplices and the
    # tensor-product ("DQ") family on quadrilaterals and hexahedra alike.
    if not symmetric:
        return TensorFunctionSpace(mesh, "DG", degree, shape=shape)
    symmetry = {
        (element, j, i): (element, i, j)
        for element in range(n_maxwell)
        for i in range(dim)
        for j in range(i + 1, dim)
    }
    return TensorFunctionSpace(mesh, "DG", degree, shape=shape, symmetry=symmetry)


def history_slices(internal_variables: Any) -> list[Expr]:
    r"""Return one `(d, d)` tensor expression per Maxwell element.

    Accepts every layout the solvers take:

    - a combined field of shape `(n, d, d)`, which is sliced as `M[i, :, :]`;
    - a single `(d, d)` field or expression, the layout of a single Maxwell
      element in existing drivers;
    - a list of the above, whose slices are concatenated in order.

    The slices are UFL expressions, so the caller can build the stress from
    them and interpolate an update into the combined field with
    `as_tensor([...])`.

    Args:
      internal_variables: a Function, an expression, or a list of them.

    Returns:
      A list of `(d, d)` tensor expressions.

    """
    if isinstance(internal_variables, (list, tuple)):
        slices = []
        for item in internal_variables:
            slices.extend(history_slices(item))
        return slices
    shape = internal_variables.ufl_shape
    if len(shape) == 2:
        return [internal_variables]
    if len(shape) == 3:
        return [internal_variables[i, :, :] for i in range(shape[0])]
    raise ValueError(
        "An internal-variable field must have shape (d, d) or (n, d, d); "
        f"got {shape}."
    )


def _element_slices(eq: Equation, trial: Any) -> tuple[list[Expr], list[Expr]]:
    """Return the per-element slices of the test function and of `trial`.

    The equation's test function and its trial share the layout of the
    history space, so both are sliced the same way. The number of slices must
    equal the number of Maxwell times the equation carries; a mismatch means
    the space was built for a different rheology.
    """
    tests = history_slices(eq.test)
    trials = history_slices(trial)
    n_times = len(eq.maxwell_times)
    if len(tests) != n_times:
        raise ValueError(
            f"The history space holds {len(tests)} internal variable(s) but the "
            f"approximation has {n_times} Maxwell time(s). Build the space with "
            "internal_variable_space(mesh, n_maxwell) for this rheology."
        )
    return tests, trials


def history_mass_term(eq: Equation, trial: Any) -> fd.Form:
    r"""Backward-Euler time derivative of every internal variable.

    $$ \sum_i \int_\Omega w_i : \frac{m_i - m_i^{old}}{\Delta t} \, dx $$

    Written on the whole field because the time derivative has the same
    coefficient for every element. `trial_old` is the field at the previous
    time level.
    """
    return inner(eq.test, (trial - eq.trial_old) / eq.dt) * eq.dx


def history_relaxation_term(eq: Equation, trial: Any) -> fd.Form:
    r"""Relaxation of every internal variable at its own Maxwell time.

    $$ \sum_i \int_\Omega w_i : \frac{m_i}{\tau_i} \, dx $$

    For a power-law rheology the Maxwell times carry the stress-dependent
    creep factor, so this term is then nonlinear in the displacement through
    the deviatoric stress. The term functions do not care; they take the
    list the solver hands them.
    """
    tests, trials = _element_slices(eq, trial)
    residual = 0
    for test, element, maxwell_time in zip(tests, trials, eq.maxwell_times):
        residual += inner(test, element / maxwell_time) * eq.dx
    return residual


def history_strain_term(eq: Equation, trial: Any) -> fd.Form:
    r"""The strain every internal variable relaxes towards, volume and boundary.

    The volume part is the source of the history equation,

    $$ -\sum_i \int_\Omega w_i : \frac{d(u)}{\tau_i} \, dx, $$

    with $d(u) = \mathrm{dev}\,\varepsilon(u)$ the deviatoric strain of the
    displacement the equation names in `displacement`.

    The boundary part exists for the weak displacement boundary conditions
    (`"u"` and `"un"` in the boundary dictionary). The Nitsche consistency
    term of the displacement equation contains the internal-variable part of
    the stress, so the displacement rows of the coupled Jacobian couple to
    the internal variables on those boundaries. Without a matching term in
    the history rows the `(u, m)` and `(m, u)` blocks are not transposes of
    each other and the displacement Schur complement, which static
    condensation hands to the Krylov solver, is not symmetric. The matching
    term is the transpose of that consistency term, scaled so that it sits
    in the same ratio to the volume coupling as the volume blocks do. It
    reads

    $$ \sum_i \int_\Gamma w_i : \frac{d_\Gamma}{\tau_i} \, ds,
       \qquad d_\Gamma = \tfrac{1}{2} A(n \otimes w), $$

    where $w$ is the boundary jump the condition penalises ($u - u_D$ for
    `"u"`, $(n \cdot u - u_n)\,n$ for `"un"`) and $A$ is the operator that
    turns a gradient-like tensor into twice its deviatoric symmetric part,
    the same operator that gives $d(u)$ from $\nabla u$. So the boundary
    term is the volume term with the lifted boundary jump in place of the
    displacement gradient. In 3-D it is trace free, which keeps the internal
    variables trace free; in 2-D it carries the same `tr/3` residual as the
    deviatoric strain, so the two stay consistent. At a converged solution
    the jump vanishes, so the term does not change the discretisation's
    consistency.

    With this term the condensed displacement operator is symmetric for a
    Newtonian rheology, and the Krylov solver on it can be CG. The transpose
    relation holds per boundary cell only when `1/tau_i` is constant within
    the cell, and the volume relation needs a symmetric relaxation tangent,
    which a power law with several elements does not have; see
    `CoupledInternalVariableSolver.condensed_operator_symmetric` and
    `NOTES/coupled-schur/FINDING-POWER-LAW-TANGENT-SYMMETRY.md`.
    """
    tests, _ = _element_slices(eq, trial)
    approximation = eq.approximation
    displacement = eq.displacement
    strain = approximation.deviatoric_strain(displacement)

    residual = 0
    for test, maxwell_time in zip(tests, eq.maxwell_times):
        residual -= inner(test, strain / maxwell_time) * eq.dx

    for bc_id, bc in eq.bcs.items():
        if "u" in bc:
            jump = displacement - bc["u"]
        elif "un" in bc:
            jump = (dot(eq.n, displacement) - bc["un"]) * eq.n
        else:
            continue
        # Half of A(G) is the deviatoric symmetric part of G, which is what
        # `deviatoric_strain` computes from grad(u).
        boundary_strain = 0.5 * approximation.deviatoric_tensor_from_grad(
            outer(eq.n, jump)
        )
        for test, maxwell_time in zip(tests, eq.maxwell_times):
            residual += inner(test, boundary_strain / maxwell_time) * eq.ds(bc_id)

    return residual


history_mass_term.required_attrs = {"dt", "trial_old"}
history_mass_term.optional_attrs = {"maxwell_times", "displacement"}
history_relaxation_term.required_attrs = {"maxwell_times"}
history_relaxation_term.optional_attrs = {"dt", "trial_old", "displacement"}
history_strain_term.required_attrs = {"maxwell_times", "displacement"}
history_strain_term.optional_attrs = {"dt", "trial_old"}

internal_variable_history_terms = [
    history_mass_term,
    history_relaxation_term,
    history_strain_term,
]
"""The three residual terms of the backward-Euler history equation."""


def _stacked_history(slices: list[Expr]) -> Expr:
    """Stack `(d, d)` expressions into one `(n, d, d)` expression.

    The inverse of `history_slices` for a combined field: the pointwise
    solver interpolates the stacked update into its combined field in one
    call.
    """
    return fd.as_tensor(slices)


def assign_history_slices(internal_variables: Any, slices: list[Expr]) -> None:
    """Interpolate one `(d, d)` expression per element into the stored history.

    The inverse of `history_slices`: `slices` holds one expression per
    Maxwell element in the order `history_slices(internal_variables)`
    returns them, and each stored field receives its own elements, stacked
    with `as_tensor` when the field is a combined `(n, d, d)` one.

    Args:
      internal_variables: the stored history, in any accepted layout.
      slices: the new value of every element, one `(d, d)` expression each.

    """
    targets = (
        list(internal_variables)
        if isinstance(internal_variables, (list, tuple))
        else [internal_variables]
    )
    expected = sum(len(history_slices(target)) for target in targets)
    if len(slices) != expected:
        raise ValueError(
            f"{len(slices)} history update(s) for a layout holding {expected} "
            "internal variable(s)."
        )
    offset = 0
    for target in targets:
        count = len(history_slices(target))
        block = slices[offset:offset + count]
        offset += count
        if len(target.ufl_shape) == 3:
            target.interpolate(_stacked_history(block))
        else:
            target.interpolate(block[0])
