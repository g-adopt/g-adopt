"""Helper functions to generate null spaces for Stokes problems

`ala_right_nullspace` computes the pressure null space for the Anelastic Liquid
Approximation. `create_stokes_nullspace`, automatically generates null spaces
for the mixed velocity-pressure Stokes system. `rigid_body_modes' returns the
translational and rotational null spaces associated with the velocity
(or displacement) field. `solenoidal_modes` and `near_incompressible_modes`
extend that basis with low-degree divergence-free fields, which is what a
nearly-incompressible (large bulk/shear) displacement operator needs GAMG to
carry as a near-nullspace -- its slow modes live in the divergence-free space,
which the six rigid-body modes do not span.
"""

import itertools

import firedrake as fd
from .approximations import AnelasticLiquidApproximation
from .utility import upward_normal


def ala_right_nullspace(
    W: fd.functionspaceimpl.WithGeometry,
    approximation: AnelasticLiquidApproximation,
    top_subdomain_id: str | int,
):
    r"""Compute pressure null space for Anelastic Liquid Approximation.

    Arguments:
      W: pressure function space
      approximation: AnelasticLiquidApproximation with equation parameters
      top_subdomain_id: boundary id of top surface

    Returns:
      pressure null space solution

    To obtain the pressure null space solution for the Stokes equation in
    Anelastic Liquid Approximation, which includes a pressure-dependent buoyancy term,
    we try to solve the equation:

    $$
      -nabla p + g "Di" rho chi c_p/(c_v gamma) hatk p = 0
    $$

    Taking the divergence:

    $$
      -nabla * nabla p + nabla * (g "Di" rho chi c_p/(c_v gamma) hatk p) = 0,
    $$

    then testing it with q:

    $$
        int_Omega -q nabla * nabla p dx + int_Omega q nabla * (g "Di" rho chi c_p/(c_v gamma) hatk p) dx = 0
    $$

    followed by integration by parts:

    $$
        int_Gamma -bb n * q nabla p ds + int_Omega nabla q cdot nabla p dx +
        int_Gamma bb n * hatk q g "Di" rho chi c_p/(c_v gamma) p dx -
        int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0
    $$

    This elliptic equation can be solved with natural boundary conditions by imposing our
    original equation above, which eliminates all boundary terms:

    $$
      int_Omega nabla q * nabla p dx - int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0.
    $$

    However, if we do so on all boundaries we end up with a system that has the same
    null space, as the one we are after (note that we ended up merely testing the
    original equation with $nabla q$). Instead we use the fact that the gradient of
    the null mode is always vertical, and thus the null mode is constant at any
    horizontal level (geoid), specifically the top surface. Choosing any nonzero
    constant for this surface fixes the arbitrary scalar multiplier of the null
    mode. We choose the value of one and apply it as a Dirichlet boundary condition.

    Note that this procedure does not necessarily compute the exact null space of the
    *discretised* Stokes system. In particular, since not every test function
    $v in V$, the velocity test space, can be written as $v=nabla q$ with $q in W$,
    the pressure test space, the two terms do not necessarily exactly cancel when
    tested with $v$ instead of $nabla q$ as in our final equation. However, in
    practice the discrete error appears to be small enough, and providing this
    null space gives an improved convergence of the iterative Stokes solver.
    """
    W = fd.FunctionSpace(mesh=W.mesh(), family=W.ufl_element())
    q = fd.TestFunction(W)
    p = fd.Function(W, name="pressure_nullspace")

    # Fix the solution at the top boundary
    bc = fd.DirichletBC(W, 1.0, top_subdomain_id)

    F = fd.inner(fd.grad(q), fd.grad(p)) * fd.dx

    k = upward_normal(W.mesh())

    F += (
        -fd.inner(fd.grad(q), k * approximation.dbuoyancydp(p, fd.Constant(1.0)) * p)
        * fd.dx
    )

    fd.solve(F == 0, p, bcs=bc)
    return p


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: list[int] | None = None,
    ala_approximation: AnelasticLiquidApproximation | None = None,
    top_subdomain_id: str | int | None = None,
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z: Firedrake mixed function space associated with the Stokes system
      closed: Whether to include a constant pressure null space
      rotational: Whether to include all rotational modes
      translations: List of translations to include i.e for all components in
                    2D: [0, 1] and 3D: [0, 1, 2]. For example, see
                    3d_cartesian.py and 3d_spherical.py mantle convection demos.
      ala_approximation: AnelasticLiquidApproximation for calculating (non-constant)
                         right null space
      top_subdomain_id: Boundary id of top surface. Required when providing
                        ala_approximation.

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    # ala_approximation and top_subdomain_id are both needed when calculating right
    # null space for ala
    if (ala_approximation is None) != (top_subdomain_id is None):
        raise ValueError(
            "Both ala_approximation and top_subdomain_id must be provided, or both must be None."
        )

    stokes_subspaces = Z.subspaces

    V_nullspace = rigid_body_modes(
        stokes_subspaces[0],
        rotational=rotational,
        translations=translations)

    if closed:
        if ala_approximation:
            p = ala_right_nullspace(
                W=stokes_subspaces[1],
                approximation=ala_approximation,
                top_subdomain_id=top_subdomain_id,
            )
            p_nullspace = fd.VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = stokes_subspaces[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface null space
    null_space += stokes_subspaces[2:]

    return fd.MixedVectorSpaceBasis(Z, null_space)


def _rigid_body_functions(
    V: fd.functionspaceimpl.WithGeometry,
    rotational: bool,
    translations: list[int] | None,
) -> list[fd.Function]:
    """Return the rigid body modes as a list of Functions on ``V``.

    The interpolated-Function construction that both `rigid_body_modes` and
    `near_incompressible_modes` share, kept in one place so the two cannot
    drift. Returns an empty list when neither rotations nor translations are
    requested; the caller decides what an empty basis means.
    """
    X = fd.SpatialCoordinate(V.mesh())
    dim = V.mesh().geometric_dimension

    if rotational:
        if dim == 2:
            basis = [fd.Function(V).interpolate(fd.as_vector((-X[1], X[0])))]
        elif dim == 3:
            basis = [
                fd.Function(V).interpolate(fd.as_vector((0, -X[2], X[1]))),
                fd.Function(V).interpolate(fd.as_vector((X[2], 0, -X[0]))),
                fd.Function(V).interpolate(fd.as_vector((-X[1], X[0], 0))),
            ]
        else:
            raise ValueError("Can only handle 2 or 3 dimensional spaces")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(V).interpolate(fd.as_vector(vec)))

    return basis


def rigid_body_modes(
    V: fd.functionspaceimpl.WithGeometry,
    rotational: bool = False,
    translations: list[int] | None = None,
) -> fd.nullspace.VectorSpaceBasis:
    """Create a null space for the rigid body modes associated with velocity
       (or displacement) in a Stokes system

    Arguments:
      V: Firedrake function space associated with the velocity or displacement
      rotational: Whether to include all rotational modes
      translations: List of translations to include i.e for all components in
                    2D: [0, 1] and 3D: [0, 1, 2]. For example, see
                    3d_cartesian.py and 3d_spherical.py mantle convection demos.

    Returns:
      A Firedrake vector space basis incorporating the null space components

    """
    basis = _rigid_body_functions(V, rotational, translations)

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=V.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    return V_nullspace


def _monomials(X, dim: int, degree: int):
    r"""Yield UFL monomials $x^a y^b (z^c)$ of exact total ``degree``.

    Built as plain products of the spatial coordinates so that UFL's own
    differentiation (`.dx`, `curl`) applies to them symbolically.
    """
    for exps in itertools.product(range(degree + 1), repeat=dim):
        if sum(exps) != degree:
            continue
        mono = fd.Constant(1.0)
        for i, e in enumerate(exps):
            for _ in range(e):
                mono = mono * X[i]
        yield mono


def solenoidal_modes(
    V: fd.functionspaceimpl.WithGeometry,
    max_degree: int = 2,
    divfree_tol: float | None = None,
) -> list[fd.Function]:
    r"""Low-degree divergence-free vector fields on ``V``.

    The near-kernel of the volumetric penalty $\lambda \int (\nabla\cdot u)
    (\nabla\cdot v)$ that stiffens a nearly-incompressible displacement
    operator is the *divergence-free* space. GAMG's smoothed aggregation
    coarsens the slow modes of an operator onto whatever near-nullspace it is
    handed, and the six rigid-body modes do not touch that space, so as
    bulk/shear grows GAMG has nothing to represent the slow modes with. This
    returns smooth, globally-defined divergence-free fields to seed it with.

    They are built as curls, which are divergence-free by construction: in 2-D
    as $\nabla^\perp\phi = (\partial_y\phi, -\partial_x\phi)$ of a scalar
    stream function $\phi$, in 3-D as $\nabla\times A$ of a vector potential
    $A$, taking $\phi$/$A$-components over the coordinate monomials. A field of
    polynomial degree $d$ comes from a potential of degree $d+1$; potentials of
    degree $2 .. max_degree+1$ are used, so ``max_degree`` is the highest
    polynomial degree of the returned fields.

    **Divergence-free by construction is not divergence-free after
    interpolation.** A degree-$d$ field is reproduced exactly in a CG$k$ space
    (and so stays exactly solenoidal) only where the mesh's geometry map is
    affine enough to represent it. On curved production meshes the crossover is
    low and *dimension-dependent*: measured, the 2-D extruded annulus holds
    degree 2 (2.7e-15) but loses degree 3 (6.6e-4); the 3-D curved cubed sphere
    holds only degree 1 (9.4e-16) and already loses degree 2 (4.7e-2). A mode
    that is a few percent non-solenoidal has volumetric energy $\sim
    \lambda(\|div\|/\|u\|)^2$, which at large bulk/shear exceeds its deviatoric
    energy -- so it is a *fast* mode of the high-ratio operator, the opposite of
    what GAMG should coarsen onto. Pass ``divfree_tol`` to drop such modes:
    each field with $\|\nabla\cdot u\|/\|u\| > $ ``divfree_tol`` (both L2 over
    the domain) is discarded, so the surviving set is genuinely solenoidal on
    *this* mesh whatever its geometry degree. With ``divfree_tol=None`` (the
    default) no filtering is done and the raw, possibly-over-complete set is
    returned -- the rigid rotation reappears among the degree-1 fields, for
    instance; `near_incompressible_modes` both filters and prunes the
    dependence.

    Arguments:
      V: vector CG space of the displacement/velocity.
      max_degree: highest polynomial degree of the divergence-free fields.
      divfree_tol: if set, drop fields whose relative L2 divergence exceeds it.

    Returns:
      A list of divergence-free Functions on ``V`` (not orthonormalised).
    """
    mesh = V.mesh()
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension

    fields = []
    for pot_degree in range(2, max_degree + 2):
        for mono in _monomials(X, dim, pot_degree):
            if dim == 2:
                # v = curl of the scalar stream function `mono`:
                #     (d phi/dy, -d phi/dx).
                fields.append(fd.as_vector((mono.dx(1), -mono.dx(0))))
            else:
                # v = curl of each single-component vector potential
                # A = mono e_i. Written out rather than via `fd.curl` on
                # `as_vector` with `Constant(0)` slots: UFL's curl lowering
                # differentiates every slot, and a bare constant carries no
                # domain, so `find_geometric_dimension` fails there.
                d0, d1, d2 = mono.dx(0), mono.dx(1), mono.dx(2)
                # A = (mono, 0, 0) -> (0,  d/dz,  -d/dy)
                fields.append(fd.as_vector((0, d2, -d1)))
                # A = (0, mono, 0) -> (-d/dz, 0,  d/dx)
                fields.append(fd.as_vector((-d2, 0, d0)))
                # A = (0, 0, mono) -> ( d/dy, -d/dx, 0)
                fields.append(fd.as_vector((d1, -d0, 0)))

    # Drop identically-zero fields (in 3-D many single-component potentials have
    # a vanishing curl, e.g. curl(x^2 e_x) = 0) and, if asked, fields whose
    # interpolation is no longer solenoidal on this mesh.
    kept = []
    for m in [fd.Function(V).interpolate(v) for v in fields]:
        m_norm = fd.sqrt(fd.assemble(fd.inner(m, m) * fd.dx))
        if m_norm == 0.0:
            continue
        if divfree_tol is not None:
            div_norm = fd.sqrt(fd.assemble(fd.inner(fd.div(m), fd.div(m)) * fd.dx))
            if div_norm / m_norm > divfree_tol:
                continue
        kept.append(m)

    return kept


def _orthonormalised_survivors(
    functions: list[fd.Function], comm, drop_tol: float = 1e-9
) -> fd.nullspace.VectorSpaceBasis:
    """Drop-tolerant modified Gram-Schmidt over a list of Functions.

    Firedrake's `VectorSpaceBasis.orthonormalize` is classical Gram-Schmidt
    followed by a strict `check_orthogonality`; a linearly dependent input
    (the rigid rotation is dependent on the degree-1 divergence-free fields)
    orthogonalises to ~0, `normalize` divides by it, and the check then raises.
    PETSc's `MatSetNearNullSpace` also *assumes* an orthonormal basis, so a
    dependent vector silently corrupts the coarse space rather than helping it.
    This orthonormalises the Functions in place and drops any whose norm after
    orthogonalisation has collapsed relative to its own starting norm, so the
    surviving set is genuinely orthonormal and independent.

    Operates through the PETSc `Vec` of each `Function` (dot/axpy/norm are
    collective), so it is correct in parallel.
    """
    survivors: list[fd.Function] = []
    ortho_vecs = []
    for f in functions:
        with f.dat.vec as v:
            start_norm = v.norm()
            if start_norm == 0.0:
                continue
            for u in ortho_vecs:
                v.axpy(-v.dot(u), u)
            norm = v.norm()
            if norm <= drop_tol * start_norm:
                continue
            v.scale(1.0 / norm)
            ortho_vecs.append(v.copy())
        survivors.append(f)

    return fd.VectorSpaceBasis(survivors, comm=comm)


def near_incompressible_modes(
    V: fd.functionspaceimpl.WithGeometry,
    max_degree: int = 1,
    divfree_tol: float = 1e-8,
) -> fd.nullspace.VectorSpaceBasis:
    """Rigid-body modes plus low-degree divergence-free modes, orthonormalised.

    The near-nullspace to hand GAMG for a nearly-incompressible displacement
    operator: the six (three in 2-D) rigid-body modes it already uses, enriched
    with the divergence-free fields from `solenoidal_modes` that represent the
    slow modes of the volumetric penalty. The solenoidal fields are filtered by
    ``divfree_tol`` so that only those still solenoidal *on this mesh* survive
    -- which auto-adapts to the mesh's geometry degree, keeping degree 2 on the
    curved annulus but only degree 1 on the curved cubed sphere, with no need
    for the caller to know which. Dependent modes are then pruned by
    `_orthonormalised_survivors`, so the result is a valid orthonormal seed even
    though the rigid and solenoidal families overlap.

    **The default ``max_degree = 1`` is deliberate, and not merely cautious.**
    The degree-1 divergence-free fields already span the *complete* space of
    linear solenoidal vector fields (dimension 5 in 2-D, 11 in 3-D, rigid modes
    included), so degree 1 is the natural closed unit rather than a truncation.
    Going higher backfires on the actual preconditioner: GAMG's smoothed
    aggregation builds its tentative prolongator by a local QR of the near-null
    block over each aggregate, and once the mode count exceeds what a fine
    aggregate can represent that QR is rank-deficient. Measured, on a 48x48 CG2
    box the degree-2 set (9 modes) drove GAMG to a `DIVERGED_NANORINF` at
    iteration 0, while the degree-1 set (5 modes) converged and roughly halved
    the iteration count against rigid modes alone. Raising ``max_degree`` is
    therefore an experiment that needs GAMG's aggregate size raised with it, not
    a free improvement.

    Arguments:
      V: vector CG space of the displacement/velocity.
      max_degree: highest polynomial degree of the divergence-free fields to
                  consider before filtering. Default 1; see above before raising.
      divfree_tol: drop solenoidal candidates whose relative L2 divergence
                   exceeds this; the default keeps only near-exact ones.

    Returns:
      An orthonormal `VectorSpaceBasis` of the combined modes.
    """
    rigid = _rigid_body_functions(
        V, rotational=True, translations=list(range(V.value_size))
    )
    solenoidal = solenoidal_modes(V, max_degree=max_degree, divfree_tol=divfree_tol)
    return _orthonormalised_survivors(rigid + solenoidal, comm=V.mesh().comm)
