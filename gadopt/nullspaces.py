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

`project_out_kernel` removes a declared kernel from a solution after a solve
and records the removal on the pyadjoint tape (`KernelProjectionBlock`), so
that replay, adjoint, tangent linear and Hessian all apply the same
projection as the forward run. `apply_kernel_projection` is the unrecorded
operation that both use, and the one definition of the projection in G-ADOPT.
`monolithic_kernel` returns the PETSc object of a declared kernel on the
whole mixed vector, the one that a solver attaches to its matrix.
"""

import itertools

import firedrake as fd
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint
from pyadjoint import Block
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from ufl.constantvalue import Zero
from ufl.tensors import ListTensor
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

    Annotation is off throughout. A mode is auxiliary -- it is handed to PETSc
    as a (near-)nullspace and never enters a residual or a functional -- so it
    carries no derivative, but Firedrake lowers `interpolate` to an assembly and
    would otherwise leave one `AssembleBlock` per mode on a live tape. Those
    blocks are dead ends that cannot corrupt a gradient; they grow the tape and
    are re-run on every replay, which is what a time loop rebuilding modes per
    step pays for. The suppression stays correct even where the modes *do* have
    a differentiable dependency -- the coordinate field under a shape control --
    because PETSc consumes them outside anything pyadjoint differentiates.

    Callers inside a solve are already safe without this: Firedrake wraps the
    solve in `stop_annotating`, and PETSc sets a PC up only from within
    `KSPSolve`. The guard is here for driver code, which builds modes to pass as
    `nullspace=`/`near_nullspace=` with the tape live.
    """
    X = fd.SpatialCoordinate(V.mesh())
    dim = V.mesh().geometric_dimension

    with stop_annotating():
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

    Note:
      Building the modes adds nothing to the pyadjoint tape, whether or not
      annotation is live -- see `_rigid_body_functions` for why that is both
      safe and wanted. The orthonormalisation below is PETSc `Vec` work and was
      never annotated.
    """
    basis = _rigid_body_functions(V, rotational, translations)

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=V.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    return V_nullspace


def _monomials(X, dim: int, degree: int):
    r"""Yield UFL monomials $x^a y^b (z^c)$ of exact total ``degree`` with their gradients.

    Each item is `(mono, grad)`: the monomial as a plain product of the spatial
    coordinates, and its partial derivatives as a list of ``dim`` expressions.
    The derivatives are formed from the exponents, `d/dx_i x^a = a x^(a-1)`,
    and not through UFL's `.dx`. Where an exponent is zero the derivative is
    the literal integer `0`, which UFL folds into `Zero`. UFL's own
    differentiation of these products leaves an index sum with the identity
    tensor that its expansion never reduces to `Zero`, so a vanishing derivative
    stays a symbol, and a candidate field that is zero in every component is
    then compiled as a kernel that reads no coordinates. On an extruded mesh
    that kernel's argument list is one the pyop2 wrapper refuses
    (`assert len(self.accesses) == len(self.dtypes)` in
    `pyop2/local_kernel.py`), which is why exactness here matters and not only
    economy.
    """
    for exps in itertools.product(range(degree + 1), repeat=dim):
        if sum(exps) != degree:
            continue
        mono = fd.Constant(1.0)
        for i, e in enumerate(exps):
            for _ in range(e):
                mono = mono * X[i]
        grad = []
        for i in range(dim):
            if exps[i] == 0:
                # The monomial does not depend on x_i: the derivative is an
                # exact zero, so the fields built from it carry a literal 0.
                grad.append(0)
                continue
            # a x^(a-1) times the other coordinates at their own exponents. A
            # plain float for `a`: a `Constant` would add one kernel argument
            # per candidate for a value that is fixed at compile time.
            d = float(exps[i])
            for j, e in enumerate(exps):
                for _ in range(e - (1 if j == i else 0)):
                    d = d * X[j]
            grad.append(d)
        yield mono, grad


def _is_identically_zero(expr) -> bool:
    """Whether a UFL vector expression is zero in every component.

    The candidates of `solenoidal_modes` carry literal zeros from `_monomials`,
    so a vanishing field arrives as `as_vector((0, 0, 0))`, which UFL folds to a
    single `Zero`, or as a `ListTensor` whose components are all `Zero`. Both
    shapes are checked; nothing is differentiated or evaluated here.

    Args:
      expr: a UFL expression, scalar or vector.

    Returns:
      `True` if the expression is `Zero` or every component of it is.
    """
    if isinstance(expr, Zero):
        return True
    if isinstance(expr, ListTensor):
        return all(isinstance(c, Zero) for c in expr.ufl_operands)
    return False


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
        for mono, grad in _monomials(X, dim, pot_degree):
            if dim == 2:
                # v = curl of the scalar stream function `mono`:
                #     (d phi/dy, -d phi/dx).
                d0, d1 = grad
                fields.append(fd.as_vector((d1, -d0)))
            else:
                # v = curl of each single-component vector potential
                # A = mono e_i. Written out rather than via `fd.curl` on
                # `as_vector` with `Constant(0)` slots: UFL's curl lowering
                # differentiates every slot, and a bare constant carries no
                # domain, so `find_geometric_dimension` fails there.
                d0, d1, d2 = grad
                # A = (mono, 0, 0) -> (0,  d/dz,  -d/dy)
                fields.append(fd.as_vector((0, d2, -d1)))
                # A = (0, mono, 0) -> (-d/dz, 0,  d/dx)
                fields.append(fd.as_vector((-d2, 0, d0)))
                # A = (0, 0, mono) -> ( d/dy, -d/dx, 0)
                fields.append(fd.as_vector((d1, -d0, 0)))

    # Drop identically-zero fields (in 3-D many single-component potentials have
    # a vanishing curl, e.g. curl(x^2 e_x) = 0) and, if asked, fields whose
    # interpolation is no longer solenoidal on this mesh.
    #
    # `stop_annotating` because these modes are auxiliary: they seed GAMG's
    # coarse space and never enter a residual, so they carry no derivative. The
    # interpolation and the two filter norms would otherwise each add an
    # `AssembleBlock` per candidate (measured: 9 blocks for the 2-D degree-1
    # set), all dead ends depending on locally-created Functions rather than on
    # any control. They cannot corrupt a gradient, but they grow the tape and
    # are re-run on every replay. Guarding here rather than at the call site
    # keeps this true however the function is reached; the path that matters in
    # production -- `NearlyIncompressibleAssembledPC.initialize` -- is already
    # covered, because Firedrake runs the whole solve inside `stop_annotating`
    # and PETSc only sets a PC up lazily from within `KSPSolve`.
    #
    # Candidates that are zero in every component (curl(x^2 e_x) and its two
    # siblings in 3-D) are dropped BEFORE interpolation, and not only by the
    # norm test below: the norm test costs an interpolation and an assembly
    # per candidate for a field that is known to be zero from its symbols.
    kept = []
    with stop_annotating():
        candidates = [v for v in fields if not _is_identically_zero(v)]
        for m in [fd.Function(V).interpolate(v) for v in candidates]:
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
        with f.dat.vec_rw as v:
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
    # No `stop_annotating` here: both halves suppress it at their own source
    # (`_rigid_body_functions`, `solenoidal_modes`), which is where it belongs
    # so that every caller of either is covered, not only this one.
    # `_orthonormalised_survivors` is PETSc `Vec` work and never annotated.
    rigid = _rigid_body_functions(
        V, rotational=True, translations=list(range(V.value_size))
    )
    solenoidal = solenoidal_modes(V, max_degree=max_degree, divfree_tol=divfree_tol)
    return _orthonormalised_survivors(rigid + solenoidal, comm=V.mesh().comm)


def _kernel_bases(nullspace) -> list[tuple[int, fd.VectorSpaceBasis]]:
    """The sub-function indices and bases of a nullspace that remove something.

    A `MixedVectorSpaceBasis` holds one entry per sub-space. An entry that is
    an indexed function space means "no kernel in this block" and is skipped.
    A plain `VectorSpaceBasis` iterates as itself, so it maps to index 0,
    which is the function itself on a non-mixed space.

    Args:
      nullspace: a `VectorSpaceBasis`, a `MixedVectorSpaceBasis` or `None`.

    Returns:
      A list of `(index, basis)` pairs, empty when nothing is removed.
    """
    if nullspace is None:
        return []
    return [(i, basis) for i, basis in enumerate(nullspace)
            if isinstance(basis, fd.VectorSpaceBasis)]


def check_kernel_basis(nullspace) -> None:
    """Refuse a nullspace basis that is not orthonormal.

    PETSc `MatNullSpaceRemove` assumes an orthonormal set. With another set it
    removes a wrong component without a message, and the projection is not
    symmetric, so `KernelProjectionBlock` would apply a wrong adjoint. Firedrake
    does not orthonormalise a `VectorSpaceBasis` at construction; the caller
    must call `orthonormalize()`.

    The check costs a few global dot products. A basis that passes is marked,
    so a second call on the same basis does no work. A solver calls this once
    where it receives the nullspace; `project_out_kernel` calls it as well,
    for callers that use it without a solver.

    Args:
      nullspace: a `VectorSpaceBasis`, a `MixedVectorSpaceBasis` or `None`.

    Raises:
      ValueError: if a basis is not orthonormal, or not orthogonal to the
        constant vector when it was built with ``constant=True``.
    """
    for i, basis in _kernel_bases(nullspace):
        if getattr(basis, "_gadopt_orthonormal", False):
            continue
        if not basis.is_orthonormal():
            raise ValueError(
                f"the nullspace basis of sub-function {i} is not orthonormal. "
                "Call `orthonormalize()` on it: PETSc removes a non-orthonormal "
                "basis incorrectly, and the recorded projection needs a "
                "symmetric projection for its adjoint.")
        # The vectors of a basis must not change after construction (Firedrake
        # documents this), so one successful check holds for its lifetime.
        basis._gadopt_orthonormal = True


def apply_kernel_projection(nullspace, target) -> bool:
    r"""Apply the kernel projection in place, without recording it.

    For each sub-function :math:`x_i` of ``target`` whose entry in
    ``nullspace`` is a `VectorSpaceBasis` with vectors :math:`e_k`, this sets

    .. math:: x_i \leftarrow P x_i, \qquad P = I - \sum_k e_k e_k^T,

    where every product is the Euclidean (l2) product of coefficient
    vectors, not an L2 integral. The operation is PETSc `MatNullSpaceRemove`
    on the nullspace that `VectorSpaceBasis.nullspace` builds, which is what
    `VectorSpaceBasis.orthogonalize` does.
    PETSc first subtracts the mean when the basis has ``constant=True``, then
    computes all the coefficients :math:`e_k^T x_i` from the input vector
    (classical Gram-Schmidt), then subtracts. All dot products are global, so
    the call is collective and correct in parallel.

    :math:`P` is an orthogonal projection, symmetric in coefficient space, only
    if the basis is orthonormal and, with ``constant=True``, orthogonal to the
    constant vector (`check_kernel_basis`).

    ``target`` can be a `Function` or a `Cofunction`. For a `Cofunction` the
    same coefficient-space operation is :math:`P^T`, the transpose that the
    adjoint of :math:`x \mapsto P x` needs, because :math:`P^T = P`.

    Args:
      nullspace: the declared nullspace, or `None`.
      target: the `Function` or `Cofunction` to change in place. Its
        sub-functions share memory with it, so the change is visible in it.

    Returns:
      Whether a basis was applied.
    """
    bases = _kernel_bases(nullspace)
    for i, basis in bases:
        # The PETSc call of `VectorSpaceBasis.orthogonalize`, made here with a
        # read-write access to the vector. The pyop3 Firedrake opens the
        # vector of `orthogonalize` read-only and then changes it, which fails
        # the state check of pyop3 at the end of the access (an
        # `AssertionError`) and, with assertions off, leaves the ghost entries
        # out of date.
        with target.subfunctions[i].dat.vec_rw as vector:
            basis.nullspace().remove(vector)
    return bool(bases)


def monolithic_kernel(nullspace):
    """Return the PETSc nullspace that Firedrake attaches to the whole matrix.

    A solver that receives ``nullspace`` (or ``transpose_nullspace``) hangs
    one PETSc nullspace on its matrix. For a `MixedVectorSpaceBasis` that
    object lives on the whole mixed vector: each basis vector of a sub-space
    is padded with zeros in the other sub-spaces, and the set is
    orthonormalised (`MixedVectorSpaceBasis._build_monolithic_basis`).
    Firedrake builds it at the first `_apply` and keeps it, so building it
    here first gives the caller the same object that the solver attaches.
    `KSPSolve` removes the transpose nullspace of its operator from the
    right-hand side with this object, so a caller that removes it from the
    nonlinear residual (`SelfGravitatingGIASolver.complete_residual`)
    applies exactly the projection of the linear solve.

    Args:
      nullspace: a `VectorSpaceBasis`, a `MixedVectorSpaceBasis` or `None`.

    Returns:
      A `PETSc.NullSpace` on the full vector, or `None` when ``nullspace``
      removes nothing.
    """
    if not _kernel_bases(nullspace):
        return None
    if isinstance(nullspace, fd.MixedVectorSpaceBasis):
        if nullspace._nullspace is None:
            nullspace._build_monolithic_basis()
        return nullspace._nullspace
    return nullspace.nullspace()


class KernelProjectionBlock(Block):
    r"""The tape record of :math:`x \leftarrow P x` (`apply_kernel_projection`).

    The block has one dependency, the function before the projection, and one
    output, the same function after it. :math:`P` is linear and does not
    depend on :math:`x`, so

    - replay applies :math:`P` to the replayed input,
    - the tangent linear model applies :math:`P` to the input tangent,
    - the adjoint applies :math:`P^T = P` to the output adjoint, a
      `Cofunction`, in coefficient space,
    - the second-order adjoint applies :math:`P^T` to the output Hessian
      value. The second derivative of a linear map is zero, so no other term
      exists.

    Memory. Without a checkpoint schedule, the tape keeps a copy of the input
    (saved when the block takes it as a dependency) and, once a later block
    reads the output, a copy of the output. A solve followed by this block
    therefore keeps two copies of the solved Function per step. The input
    copy is the converged solve output, which the adjoint of the solve block
    needs for its Jacobian, so it cannot be dropped. With a disk or
    multistage schedule the copies go to the schedule's storage.

    The basis vectors are not dependencies. They are fixed geometric data,
    so a derivative with respect to the mesh coordinates does not include
    the change of the basis with the mesh.

    Args:
      function: the `Function` that is projected in place.
      nullspace: the `VectorSpaceBasis` or `MixedVectorSpaceBasis` that
        defines :math:`P`. It must be orthonormal (`check_kernel_basis`) and
        stay unchanged for the life of the tape.
      ad_block_tag: optional tag for `Tape.get_blocks`.
    """

    def __init__(self, function, nullspace, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.nullspace = nullspace
        self.add_dependency(function)

    def _projected_copy(self, value):
        """Return a projected copy of a tangent, adjoint or Hessian value.

        A copy, because pyadjoint adds the returned object into the
        dependency's value and can keep a reference to it. The input value
        belongs to the output block variable and must not change.

        Args:
          value: a `Function` (tangent) or `Cofunction` (adjoint, Hessian),
            or `None` when no value reached the block.

        Returns:
          A new object of the same type with :math:`P` applied, or `None`.
        """
        if value is None:
            return None
        result = value.copy(deepcopy=True)
        apply_kernel_projection(self.nullspace, result)
        return result

    def recompute_component(self, inputs, block_variable, idx, prepared):
        """Replay: copy the recorded or replayed input, then project the copy.

        `Function.assign` between equal spaces copies the coefficients
        exactly, and the projection is the same PETSc call as in the forward
        run, so the replay of an unchanged input is bit-identical to it.

        Args:
          inputs: the saved value of the dependency, the unprojected Function.
          block_variable: the output block variable.
          idx: index of the output, always 0.
          prepared: unused.

        Returns:
          The projected Function, wrapped for disk checkpointing if that is on.
        """
        with stop_annotating():
            output = fd.Function(block_variable.output.function_space())
            output.assign(inputs[0])
            apply_kernel_projection(self.nullspace, output)
        return maybe_disk_checkpoint(output)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        r"""Adjoint: :math:`\lambda_{in} = P^T \lambda_{out} = P \lambda_{out}`.

        Args:
          inputs: unused, :math:`P` does not depend on the input.
          adj_inputs: the adjoint value of the output, a `Cofunction`.
          block_variable: the dependency block variable.
          idx: index of the dependency, always 0.
          prepared: unused.

        Returns:
          The projected copy of the output adjoint value.
        """
        return self._projected_copy(adj_inputs[0])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        r"""Tangent linear model: :math:`\dot x_{out} = P \dot x_{in}`.

        Args:
          inputs: unused, :math:`P` does not depend on the input.
          tlm_inputs: the tangent of the dependency, a `Function`.
          block_variable: the output block variable.
          idx: index of the output, always 0.
          prepared: unused.

        Returns:
          The projected copy of the input tangent.
        """
        return self._projected_copy(tlm_inputs[0])

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        r"""Second-order adjoint: :math:`P^T` applied to the output Hessian value.

        The term with the second derivative of the block is zero, because
        :math:`P` is linear.

        Args:
          inputs: unused.
          hessian_inputs: the Hessian value of the output, a `Cofunction`.
          adj_inputs: unused, it multiplies the zero second derivative.
          block_variable: the dependency block variable.
          idx: index of the dependency, always 0.
          relevant_dependencies: unused.
          prepared: unused.

        Returns:
          The projected copy of the output Hessian value.
        """
        return self._projected_copy(hessian_inputs[0])

    def __str__(self):
        return f"project_out_kernel({self.get_dependencies()[0]})"


def project_out_kernel(nullspace, function) -> bool:
    """Remove a declared kernel from ``function`` in place, and record it.

    This is `apply_kernel_projection` on a `Function` that the tape can use
    later. When annotation is on, a `KernelProjectionBlock` takes the
    current value of ``function`` as its dependency before the change, and
    the changed ``function`` gets a new block variable as the output of the
    block. A later recorded operation, for example ``old.assign(function)``,
    then depends on the projected value in the forward run, in replay and in
    the adjoint.

    Without annotation, or when ``nullspace`` holds no `VectorSpaceBasis`,
    no block is added. The forward values are the same in all cases.

    Args:
      nullspace: the declared nullspace, or `None`.
      function: the `Function` to project in place.

    Returns:
      Whether a basis was applied.

    Raises:
      ValueError: if a basis is not orthonormal (`check_kernel_basis`).
    """
    if not _kernel_bases(nullspace):
        return False
    # Does work only on the first call for a given basis.
    check_kernel_basis(nullspace)
    block = None
    if annotate_tape():
        # The dependency is taken here, before the change: `add_dependency`
        # saves the unprojected value as the input checkpoint of the block.
        block = KernelProjectionBlock(function, nullspace)
        get_working_tape().add_block(block)
    with stop_annotating():
        apply_kernel_projection(nullspace, function)
    if block is not None:
        block.add_output(function.create_block_variable())
    return True
