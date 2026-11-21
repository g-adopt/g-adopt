"""Shared pieces for the weak boundary condition cases of the viscoelastic solvers.

The entrypoints in this directory drive the weak "un" and weak "u" branches of
`gadopt.momentum_equation.viscosity_term` through `InternalVariableSolver` and
`CoupledInternalVariableSolver`, and write their diagnostics to `.dat` files.
`test_weak_bc_gia.py` reads those files and applies the tolerances.

The functions here return numbers instead of asserting on them, because the
assertion belongs in the test file where its tolerance is visible.

The general pieces are a copy of the equivalent code in
`tests/weak_bc_stokes/stokes_helpers.py`, which `tests/unit/test_symmetry.py`
also keeps a copy of. The repository has no mechanism for sharing a module
between two case directories, so the copy follows what `tests/adjoint` does with
`cases.py`. The module basename differs from the one in the Stokes directory
because the repository-level pytest collects both in one process.
"""

import firedrake as fd
import gadopt
import ufl
from gadopt.equations import interior_penalty_factor


# Mesh resolution in each direction. These cases test algebraic properties of
# the assembled operator, which do not need a converged solution.
N = 4

# How to build each mesh from its key, on demand: each step runs in its own
# process and needs exactly one of them. Only affine simplex cells are used, so
# that the pointwise and the coupled formulation discretise the same continuous
# problem and can be compared directly.
MESH_BUILDERS = {
    "2D-tri": lambda: fd.UnitSquareMesh(N, N, quadrilateral=False),
    "3D-tet": lambda: fd.UnitCubeMesh(N, N, N, hexahedral=False),
}

# The (dt/tau, creep exponent) pairs of the displacement block symmetry group,
# in the order their rows appear in the output file. A large dt/tau is where a
# symmetrising term built with the effective viscosity instead of the elastic
# modulus leaves the largest asymmetry; exponent 3 selects composite creep.
BLOCK_SYMMETRY_CASES = [(0.25, 1), (0.25, 3), (25.0, 1), (25.0, 3)]

# The two linearisations of the weak "u" group, in row order. "pointwise"
# substitutes the backward-Euler update into the stress, as
# `InternalVariableSolver` does; "elastic" holds the history fixed, as the
# displacement rows of `CoupledInternalVariableSolver` see it.
WEAK_U_HISTORIES = ["pointwise", "elastic"]

# The two resolutions the refinement case compares. Which time steps it runs at
# is a property of the case layout, so that list lives in meta.py.
REFINEMENT_RESOLUTIONS = (8, 16)


def build_mesh(id):
    """Build the named mesh and mark it cartesian.

    Every mesh here is a unit square or cube, so the radial direction is a
    coordinate direction in all of them.
    """
    mesh = MESH_BUILDERS[id]()
    mesh.cartesian = True
    return mesh


def first_variation(form, functional, solution):
    """Return ||form - dE|| / ||dE||, the distance from `form` to a first variation.

    Symmetry alone cannot see a mis-scaled penalty, or a wrong constant in a
    term that is still the first variation of some functional. This ratio can.
    """
    variation = fd.derivative(functional, solution)
    residual = fd.assemble(form - variation)
    reference = fd.assemble(variation)
    return residual.dat.norm / reference.dat.norm


def deviatoric_tensor(gradient, compressible):
    r"""Twice the deviatoric symmetric part $A(G)$, written out in the test.

    For a gradient-like tensor $G$ this is $2\,\mathrm{sym}(G)$, minus
    $\tfrac{2}{3}\,\mathrm{tr}(G)\,I$ in the compressible case. The expression
    is spelled out here instead of calling
    `approximation.deviatoric_tensor_from_grad`, so that a bug in the operator
    under test cannot hide by appearing identically on both sides of an
    identity.
    """
    tensor = 2 * fd.sym(gradient)
    if compressible:
        dim = gradient.ufl_shape[0]
        tensor = tensor - 2 / 3 * fd.tr(gradient) * fd.Identity(dim)
    return tensor


def exterior_facet_form(form):
    """The exterior-facet (`ds`) part of a UFL form.

    The weak boundary conditions are the only source of `ds` integrals in the
    forms built here, so this isolates the terms under test from the volume
    term of `viscosity_term`.
    """
    return ufl.Form(
        [i for i in form.integrals() if "exterior_facet" in i.integral_type()]
    )


def generic_velocity(mesh):
    """A generic velocity field for symmetry tests.

    Unlike u = X (identity), this has n.u != 0 on every boundary and an
    anisotropic strain, so a symmetric-but-wrong penalty coefficient in the
    weak boundary terms becomes observable. At u = X the normal jump n.w is
    zero on every boundary face in this suite, which hides such errors.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    return X + fd.Constant([float(i + 1) for i in range(dim)]) + 0.3 * X[0] * X


def asymmetry(petscmat):
    """Return ||A - A^T|| / ||A|| for a PETSc matrix.

    A ratio relative to the matrix norm, and not PETSc's absolute
    isSymmetric(), because a generic strongly strained linearisation point gives
    entries of very different magnitudes (order 1e7 on a manifold mesh), where
    an absolute threshold means nothing.
    """
    transpose = petscmat.duplicate(copy=True)
    transpose.transpose()
    difference = petscmat.copy()
    difference.axpy(-1.0, transpose)
    return difference.norm() / petscmat.norm()


def penalty_coefficient(eq):
    r"""The SIPG penalty coefficient $\sigma_{pen}$ used by `viscosity_term`.

    `interior_penalty_factor` returns the mesh-independent safety factor; the
    weak boundary terms scale it by the facet-area-to-cell-volume ratio, which
    carries the $1/h$ of the Nitsche penalty. Written out here so that a change
    to the scaling in the code under test is visible.
    """
    sigma = interior_penalty_factor(eq)
    return sigma * fd.FacetArea(eq.mesh) / fd.avg(fd.CellVolume(eq.mesh))


def raw_deviatoric_strain(u):
    r"""$\mathrm{dev}(\mathrm{sym}(\nabla u))$, written out from `u` alone.

    The internal-variable formulation splits the strain into this deviatoric
    part and the volumetric part `div(u)`. The trace is removed with a factor
    1/3 in every dimension, matching the 3D convention the viscoelastic
    benchmarks are built on.
    """
    strain = fd.sym(fd.grad(u))
    return strain - fd.tr(strain) / 3 * fd.Identity(len(u))


def raw_maxwell_times(approximation):
    """Maxwell relaxation times, rebuilt from raw viscosity and shear modulus."""
    return [
        eta / mu for eta, mu in zip(approximation.viscosity, approximation.shear_modulus)
    ]


def raw_effective_viscosity(approximation, dt):
    r"""$\eta_{eff} = \sum_i \eta_i / (\tau_i + \Delta t)$ from raw attributes.

    `InternalVariableSolver` uses this coefficient for its weak boundary penalty.
    """
    return sum(
        eta / (tau + dt)
        for eta, tau in zip(approximation.viscosity, raw_maxwell_times(approximation))
    )


def raw_internal_variables_update(approximation, u, internal_variables, dt):
    r"""Backward-Euler update of the internal variables, from raw attributes.

    Each internal variable relaxes towards the deviatoric strain with its own
    Maxwell time, $\dot m_i = (\mathrm{dev}\,\varepsilon(u) - m_i)/\tau_i$, so
    one implicit step gives
    $m_i^{new} = (m_i + \Delta t\,\mathrm{dev}\,\varepsilon(u)/\tau_i)
    / (1 + \Delta t/\tau_i)$.
    """
    dev_strain = raw_deviatoric_strain(u)
    return [
        (m + dt / tau * dev_strain) / (1 + dt / tau)
        for m, tau in zip(internal_variables, raw_maxwell_times(approximation))
    ]


def raw_internal_variable_stress(approximation, u, internal_variables):
    r"""The internal-variable stress, written out from raw attributes.

    $$ \sigma = \kappa_r\,\kappa\,(\nabla \cdot u)\,I
       + 2\mu_0\,\mathrm{dev}\,\varepsilon(u)
       - \sum_i 2\mu_i m_i, $$

    with $\kappa$ the bulk modulus, $\kappa_r$ the bulk-to-shear ratio and
    $\mu_0 = \sum_i \mu_i$ the unrelaxed (elastic) shear modulus.
    """
    identity = fd.Identity(len(u))
    stress = (
        approximation.bulk_shear_ratio
        * approximation.bulk_modulus
        * fd.div(u)
        * identity
    )
    stress += 2 * sum(approximation.shear_modulus) * raw_deviatoric_strain(u)
    for shear_modulus, m in zip(approximation.shear_modulus, internal_variables):
        stress -= 2 * shear_modulus * m
    return stress


# Boundary data for the weak "un" reference tests. A nonzero value keeps the
# normal jump w_n = n.u - un away from zero, so a wrong coefficient in any term
# proportional to w_n is observable.
WEAK_UN_VALUE = 0.3


# Time step and material constants shared by the viscoelastic reference cases.
GIA_DT = 0.25


GIA_BULK_MODULUS = 3.0


GIA_BULK_SHEAR_RATIO = 1.5


# Elastic shear modulus and viscosity of the single Maxwell element used
# throughout, chosen so the Maxwell time tau = viscosity / shear_modulus is 1
# and dt / tau is simply dt.
SHEAR_MODULUS = 2.0


VISCOSITY = 2.0


MAXWELL_TIME = VISCOSITY / SHEAR_MODULUS


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
        mu_penalty * deviatoric_tensor(G, True)
        + bulk * fd.tr(G) * fd.Identity(dim)
    )
    integrand = (
        -normal_jump * fd.dot(n, fd.dot(stress, n))
        + sigma * fd.inner(G, penalty_stress)
    )
    return sum(integrand * eq.ds(bid) for bid in boundary_ids)
