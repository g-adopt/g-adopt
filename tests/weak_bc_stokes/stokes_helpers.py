"""Shared pieces for the weak boundary condition cases of the Stokes momentum term.

The entrypoints in this directory drive `gadopt.momentum_equation.viscosity_term`
and write their diagnostics to `.dat` files; `test_weak_bc_stokes.py` reads those
files and applies the tolerances. This module holds what both sides need, plus
the reference forms the diagnostics are measured against.

The functions here return numbers instead of asserting on them, because the
assertion belongs in the test file where its tolerance is visible. Everything
else is a copy of the equivalent code in `tests/unit/test_symmetry.py`, which
keeps its own copy for the two whole-matrix symmetry tests that stay there. The
repository has no mechanism for sharing a module between two case directories,
so the copy follows what `tests/adjoint` does with `cases.py`.
"""

import firedrake as fd
import gadopt
import ufl
from gadopt.equations import Equation, interior_penalty_factor
from gadopt.momentum_equation import viscosity_term


# Resolution of every mesh below, in each direction. Kept small: these cases
# test algebraic properties of the assembled operator, which do not need a
# converged solution, and each extra cell costs assembly time in every step.
N = 4

# How to build each mesh from its key. The meshes are built on demand, one per
# process, because each step of this case runs in its own process and needs
# exactly one of them; building all ten at import would make every step pay for
# the cubed-sphere and the icosahedral sphere it does not use.
MESH_BUILDERS = {
    "2D-tri": lambda: fd.UnitSquareMesh(N, N, quadrilateral=False),
    "2D-quad": lambda: fd.UnitSquareMesh(N, N, quadrilateral=True),
    "3D-tet": lambda: fd.UnitCubeMesh(N, N, N, hexahedral=False),
    "3D-extruded": lambda: fd.ExtrudedMesh(
        fd.UnitSquareMesh(N, N, quadrilateral=True), N),
    "2D-cylinder": lambda: fd.ExtrudedMesh(fd.CircleManifoldMesh(N), N),
}


def build_mesh(id):
    """Build the named mesh and set its coordinate system.

    `cartesian` tells the approximations whether the radial direction is a
    coordinate direction. The cylinder is an immersed manifold, so it is False
    there.
    """
    mesh = MESH_BUILDERS[id]()
    mesh.cartesian = "cylinder" not in id
    return mesh


def taylor_hood(mesh):
    """The P2-P1 velocity-pressure mixed space on `mesh`."""
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    return V * W


# The two approximations the symmetry groups run, in the order their rows appear
# in the output file. One incompressible and one compressible, which is the only
# distinction the weak boundary terms make between the shipped approximations.
SYMMETRY_APPROXIMATIONS = [
    ("Boussinesq", gadopt.BoussinesqApproximation),
    ("TALA", gadopt.TruncatedAnelasticLiquidApproximation),
]

# The (boundary condition kind, compressible) pairs of the variational structure
# group, in the order their rows appear in the output file.
STRUCTURE_CASES = [("u", False), ("u", True), ("un", False), ("un", True)]

# Refinements of the manufactured-solution cases and of the Tosi case. Three
# levels give two refinement pairs, which is the minimum for a trend rather than
# a single ratio. They live here so that the entrypoint that writes the rows and
# the test that reads them agree on how many there are.
MMS_RESOLUTIONS = (8, 16, 32)
TOSI_RESOLUTIONS = (16, 32, 64)


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


def form_agreement(form, reference, solution):
    """Return the residual and Jacobian differences of two forms, relative to the reference.

    The residual alone would miss a difference that is zero at the current
    linearisation point, and the Jacobian alone would miss a difference that is
    constant in the solution, so both are measured.

    Returns:
      (residual ratio, Jacobian ratio), each relative to the norm of the
      reference.
    """
    residual = fd.assemble(form - reference)
    residual_ref = fd.assemble(reference)

    jacobian = fd.assemble(
        fd.derivative(form - reference, solution), mat_type="aij"
    ).petscmat
    jacobian_ref = fd.assemble(
        fd.derivative(reference, solution), mat_type="aij"
    ).petscmat
    return (residual.dat.norm / residual_ref.dat.norm,
            jacobian.norm() / jacobian_ref.norm())


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


def nonlinear_mu(u, compressible):
    """A solution-dependent viscosity for symmetry tests.

    For compressible stress the viscosity must depend on the invariant that
    matches the -2/3 deviatoric stress operator; otherwise the volume Jacobian
    itself is asymmetric at a generic linearisation point and the test would be
    checking an unsatisfiable property. For incompressible stress the full
    strain invariant is the matching one.
    """
    eps = fd.sym(fd.grad(u))
    if compressible:
        dim = eps.ufl_shape[0]
        invariant = fd.inner(eps, eps - fd.tr(eps) / 3 * fd.Identity(dim))
    else:
        invariant = fd.inner(eps, eps)
    return fd.Constant(1.0) + invariant


def penalty_coefficient(eq):
    r"""The SIPG penalty coefficient $\sigma_{pen}$ used by `viscosity_term`.

    `interior_penalty_factor` returns the mesh-independent safety factor; the
    weak boundary terms scale it by the facet-area-to-cell-volume ratio, which
    carries the $1/h$ of the Nitsche penalty. Written out here so that a change
    to the scaling in the code under test is visible.
    """
    sigma = interior_penalty_factor(eq)
    return sigma * fd.FacetArea(eq.mesh) / fd.avg(fd.CellVolume(eq.mesh))


def explicit_weak_boundary_form(
    eq,
    trial,
    bc_id,
    boundary_value,
    *,
    bc_kind,
    stress,
    tangent,
    mu_penalty,
    compressible,
    bulk=None,
    dmu=None,
):
    r"""A weak velocity boundary residual, written out term by term.

    This is the reference the code under test must reproduce. The jump is
    $w = u - u_D$ for a weak "u" condition and $w = (n \cdot u - u_n) n$ for a
    weak "un" one. With $G = n \otimes w$ and $A$ the deviatoric stress per
    $\mu$, the contributions are

      * the penalty $2\sigma_{pen}\,\langle n \otimes \phi,\ \mu A(G) \rangle$,
      * the symmetrising term $-w \cdot (T n)$ with $T$ the tangent stress,
      * the flux, $-\langle n \otimes \phi,\ \sigma(u) \rangle$ for "u" and its
        normal component $-(n \cdot \phi)\,(n \cdot \sigma(u)\, n)$ for "un",
      * for a stress with a bulk part, the matching pair of bulk penalty and
        bulk symmetrising terms with coefficient
        `bulk_shear_ratio * bulk_modulus`, both driven by
        $\mathrm{tr}(G) = n \cdot w$,
      * for a solution-dependent viscosity, the derivative of the penalty
        through $\mu$.

    Every coefficient is supplied by the caller from raw approximation
    attributes, so nothing is taken from the operator under test.

    Args:
      eq: the `Equation` supplying the measures, the facet normal and the test
        function.
      trial: the field the boundary condition constrains.
      bc_id: the boundary identifier the condition applies to.
      boundary_value: the prescribed velocity for "u", or its prescribed normal
        component for "un".
      bc_kind: "u" or "un".
      stress: the full stress $\sigma(u)$ entering the flux term.
      tangent: the tangent stress $D\sigma(u)[\phi]$ entering the symmetrising
        term. For a stress with a bulk part this is the deviatoric tangent
        only; the bulk part is added through `bulk`.
      mu_penalty: the shear coefficient multiplying the deviatoric penalty.
      compressible: whether the deviatoric stress carries the $-2/3$ trace term.
      bulk: the bulk coefficient, or None for a purely deviatoric stress.
      dmu: the directional derivative of $\mu$ in the direction of the test
        function, or None for a solution-independent viscosity.

    Returns:
      A UFL form for the weak boundary residual on `bc_id`.
    """
    n = eq.n
    dim = eq.mesh.geometric_dimension
    sigma = penalty_coefficient(eq)
    ds = eq.ds(bc_id)

    if bc_kind == "u":
        w = trial - boundary_value
    else:
        w = (fd.dot(n, trial) - boundary_value) * n
    # Trace of the jump tensor, which is what a bulk modulus responds to.
    normal_jump = fd.dot(n, w)
    G = fd.outer(n, w)
    A_G = deviatoric_tensor(G, compressible)

    # Penalty on the deviatoric part of the jump.
    F = 2 * sigma * fd.inner(fd.outer(n, eq.test), mu_penalty * A_G) * ds
    # Symmetrising term: the transpose of the flux integration by parts.
    F -= fd.dot(w, fd.dot(tangent, n)) * ds
    # Flux term. The "un" condition leaves the tangential traction free, so only
    # the normal component of the traction is removed there.
    if bc_kind == "u":
        F -= fd.inner(fd.outer(n, eq.test), stress) * ds
    else:
        F -= fd.dot(n, eq.test) * fd.dot(n, fd.dot(stress, n)) * ds

    if bulk is not None:
        # Volumetric jump tensor: the bulk stress responds to tr(G) = n . w.
        bulk_jump = fd.Identity(dim) * normal_jump
        F += 2 * sigma * fd.inner(fd.outer(n, eq.test), bulk * bulk_jump) * ds
        F -= fd.inner(bulk * fd.nabla_grad(eq.test), bulk_jump) * ds

    if dmu is not None:
        # First variation of the penalty functional through mu itself.
        F += sigma * dmu * fd.inner(G, A_G) * ds

    return F


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


def raw_nonlinear_dmu(u, direction, compressible):
    r"""Directional derivative of `nonlinear_mu` in the direction `direction`.

    `nonlinear_mu` is $1 + \langle \varepsilon, \varepsilon \rangle$ for the
    incompressible invariant and
    $1 + \langle \varepsilon, \varepsilon - \tfrac{1}{3}\mathrm{tr}
    (\varepsilon) I \rangle$ for the compressible one, so its derivative is
    $2\langle \varepsilon(\phi), \varepsilon \rangle$ and
    $2\langle \varepsilon(\phi), \varepsilon - \tfrac{1}{3}\mathrm{tr}
    (\varepsilon) I \rangle$ respectively. Written out by hand so the test does
    not lean on UFL differentiation of the same expression the code
    differentiates.
    """
    eps = fd.sym(fd.grad(u))
    eps_direction = fd.sym(fd.grad(direction))
    if compressible:
        dim = eps.ufl_shape[0]
        return 2 * fd.inner(eps_direction, eps - fd.tr(eps) / 3 * fd.Identity(dim))
    return 2 * fd.inner(eps_direction, eps)


# Boundary data for the weak "un" reference tests. A nonzero value keeps the
# normal jump w_n = n.u - un away from zero, so a wrong coefficient in any term
# proportional to w_n is observable.
WEAK_UN_VALUE = 0.3


# Time step and material constants shared by the viscoelastic reference cases.
GIA_DT = 0.25


GIA_BULK_MODULUS = 3.0


GIA_BULK_SHEAR_RATIO = 1.5


def build_stokes_weak_un_case(mesh, approx_class, compressible, nonlinear, bc_id):
    """Assemble the weak "un" form and its reference for a Stokes approximation.

    Returns the form produced by `viscosity_term` restricted to the boundary,
    the reference form written out from raw quantities, and the Function the
    Jacobians are taken with respect to.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    Z = V * W
    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    u, _ = fd.split(z)

    mu = nonlinear_mu(u, compressible) if nonlinear else fd.Constant(2.0)
    kwargs = {} if approx_class is gadopt.BoussinesqApproximation else {"Di": 1}
    approximation = approx_class(1, mu=mu, **kwargs)

    T = fd.Function(W)
    solver = gadopt.StokesSolver(
        z, approximation, T, bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    # Reference stress and tangent, written from mu and the strain alone.
    stress = mu * deviatoric_tensor(fd.grad(u), compressible)
    tangent = mu * deviatoric_tensor(fd.grad(eq.test), compressible)
    dmu = None
    if nonlinear:
        # A solution-dependent mu adds Dmu[phi] sigma/mu to the tangent, and a
        # penalty-derivative term to the residual.
        dmu = raw_nonlinear_dmu(u, eq.test, compressible)
        tangent = tangent + dmu * deviatoric_tensor(fd.grad(u), compressible)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=mu,
        compressible=compressible, dmu=dmu,
    )
    return form, reference, z


def build_internal_variable_weak_un_case(mesh, shear_moduli, viscosities, bc_id):
    """Assemble the weak "un" form and its reference for `InternalVariableSolver`.

    One entry in `shear_moduli` gives Maxwell rheology, two give a Burgers
    body. The solver substitutes the backward-Euler update of the internal
    variables into the stress, so the tangent of that stress in the direction
    of the test function carries the effective viscosity, not the elastic
    shear modulus.
    """
    dim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    X = fd.SpatialCoordinate(mesh)
    internal_variables = []
    for i in range(len(shear_moduli)):
        # A nonzero, non-isotropic history state, so that a term proportional to
        # the internal variables cannot vanish by accident.
        m = fd.Function(S).interpolate(
            0.1 * (i + 1) * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
        )
        internal_variables.append(m)

    approximation = gadopt.CompressibleInternalVariableApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        list(shear_moduli),
        list(viscosities),
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        B_mu=1.27,
    )
    solver = gadopt.InternalVariableSolver(
        u, approximation, dt=GIA_DT,
        internal_variables=internal_variables,
        bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    updated = raw_internal_variables_update(
        approximation, u, internal_variables, GIA_DT
    )
    stress = raw_internal_variable_stress(approximation, u, updated)
    eta_eff = raw_effective_viscosity(approximation, GIA_DT)
    tangent = eta_eff * deviatoric_tensor(fd.grad(eq.test), True)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=eta_eff,
        compressible=True,
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    return form, reference, u


def build_internal_variable_weak_u_case(mesh, bc_id):
    """Assemble the weak "u" form and its reference for the internal-variable stress.

    Every `StokesSolverBase` subclass turns a "u" boundary condition into a
    strong `DirichletBC`, so this drives the `Equation` directly on a
    displacement-only space. The effective viscosity is put into
    `approximation.mu` here, which matches `InternalVariableSolver`, and the
    history is advanced with the same backward-Euler update
    `InternalVariableSolver` uses.

    The weak "u" branch constrains all components of the displacement, so its
    penalty and its symmetrising term act on the full jump. A stress with a
    bulk modulus therefore has to contribute a volumetric penalty here as well,
    driven by the normal part of that jump.
    """
    dim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    X = fd.SpatialCoordinate(mesh)
    m = fd.Function(S).interpolate(
        0.1 * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
    )

    approximation = gadopt.CompressibleInternalVariableApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        [2.0],
        [2.0],
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        B_mu=1.27,
    )
    # Match the effective penalty scale of `InternalVariableSolver`. Build it
    # from raw parameters and wrap it as a UFL expression.
    eta_eff = raw_effective_viscosity(approximation, GIA_DT)
    approximation.mu = ufl.as_ufl(eta_eff)

    # The equation carries the approximation's own stress at its own
    # backward-Euler update; the reference below rebuilds both from raw
    # attributes.
    strain = approximation.deviatoric_strain(u)
    updated_equation = [
        (m + GIA_DT / maxwell_time * strain) / (1 + GIA_DT / maxwell_time)
        for maxwell_time in approximation.maxwell_times
    ]
    # Inhomogeneous boundary data, so the jump stays away from zero.
    u_D = fd.Constant([0.1 * (i + 1) for i in range(dim)])
    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={
            "stress": approximation.stress(u, internal_variables=updated_equation)
        },
        approximation=approximation,
        bcs={bc_id: {"u": u_D}},
        quad_degree=6,
    )
    form = exterior_facet_form(eq.residual(u))

    updated = raw_internal_variables_update(approximation, u, [m], GIA_DT)
    reference = explicit_weak_boundary_form(
        eq, u, bc_id, u_D, bc_kind="u",
        stress=raw_internal_variable_stress(approximation, u, updated),
        tangent=eta_eff * deviatoric_tensor(fd.grad(eq.test), True),
        mu_penalty=eta_eff,
        compressible=True,
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    return form, reference, u


def build_incompressible_maxwell_weak_un_case(mesh, bc_id):
    """Assemble the weak "un" form and its reference for `ViscoelasticStokesSolver`.

    The incremental-displacement formulation carries the previous deviatoric
    stress as data, so the stress is affine in the unknown and its tangent is
    the effective viscosity times the incompressible deviatoric shape. There is
    no bulk part.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    Z = V * W

    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    u, _ = fd.split(z)
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    stress_old = fd.Function(S).interpolate(
        0.2 * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
    )
    displacement = fd.Function(V)

    shear_modulus, viscosity = 2.0, 2.0
    approximation = gadopt.IncompressibleMaxwellApproximation(
        fd.Function(DG0).assign(1), shear_modulus, viscosity
    )
    solver = gadopt.ViscoelasticStokesSolver(
        z, approximation, stress_old, displacement, dt=GIA_DT,
        bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    # Zhong et al. (2003) incremental-displacement effective viscosity, with the
    # Maxwell time rebuilt from the raw viscosity and shear modulus.
    maxwell_time = viscosity / shear_modulus
    eta_eff = viscosity / (maxwell_time + GIA_DT / 2)
    stress = 2 * eta_eff * fd.sym(fd.grad(u)) + stress_old
    tangent = eta_eff * deviatoric_tensor(fd.grad(eq.test), False)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=eta_eff,
        compressible=False,
    )
    return form, reference, z


# Mantle-convection cases as (class, compressible stress, solution-dependent mu).
# The compressibility of each approximation is stated here rather than read back
# from the instance, so that a class silently changing it fails the test.
WEAK_BOUNDARY_CASES = {
    "Boussinesq-linear": (gadopt.BoussinesqApproximation, False, False),
    "Boussinesq-nonlinear": (gadopt.BoussinesqApproximation, False, True),
    "EBA-linear": (gadopt.ExtendedBoussinesqApproximation, False, False),
    "EBA-nonlinear": (gadopt.ExtendedBoussinesqApproximation, False, True),
    "TALA-linear": (gadopt.TruncatedAnelasticLiquidApproximation, True, False),
    "TALA-nonlinear": (gadopt.TruncatedAnelasticLiquidApproximation, True, True),
    "ALA-linear": (gadopt.AnelasticLiquidApproximation, True, False),
    "ALA-nonlinear": (gadopt.AnelasticLiquidApproximation, True, True),
    "Maxwell": None,
    "Maxwell-weak-u": None,
    "Burgers": None,
    "IncompressibleMaxwell": None,
}
