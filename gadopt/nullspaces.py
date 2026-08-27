"""Helper functions to generate null spaces for Stokes problems

`ala_right_nullspace` computes the pressure null space for the Anelastic Liquid
Approximation. `create_stokes_nullspace` automatically generates null spaces
for the mixed velocity-pressure Stokes system.
`ConformalKillingNearNullspace` describes a near null space for the mixed
Stokes system, including the additional conformal modes of the
three-dimensional deviatoric-strain operator. `rigid_body_modes` returns the
translational and rotational null spaces associated with the velocity (or
displacement) field.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ConformalKillingNearNullspace:
    r"""Describe conformal near-null modes for a mixed Stokes system.

    In three dimensions, the kernel of the trace-free symmetric gradient

    .. math::

       \varepsilon_{\mathrm{dev}}(u)
       = \operatorname{sym}(\nabla u) - \tfrac{1}{3}\nabla\!\cdot u\,I

    is the ten-dimensional space of conformal Killing fields. In addition to
    the six rigid-body translations and rotations, it contains one dilation
    and three special conformal modes. These extra modes are therefore useful
    candidates for the algebraic multigrid near null space of three-dimensional
    TALA and ALA velocity operators.

    This specification is accepted only through the ``near_nullspace``
    argument of G-ADOPT's Stokes solvers. It cannot be supplied as an exact or
    transpose null space: boundary conditions, continuity, and other terms in
    the Stokes system prevent the conformal fields from being exact null modes
    of the full saddle-point operator.

    The fields are materialised from the mesh coordinates when the Stokes
    solver is constructed. They are intended for fixed meshes; a moving-mesh
    calculation must rebuild the solver and basis after changing coordinates.

    By default the complete ten-dimensional conformal Killing space is used.
    The complete space is independent of the choice of coordinate origin up
    to a change of basis. A subset need not have that property. The polynomial
    fields are intended for non-periodic, three-dimensional Euclidean volume
    meshes.

    PETSc permits candidate modes from the operator before boundary
    conditions, so raw modes are the default. Strong velocity conditions can
    instead be applied homogeneously to the correction modes, including when
    the prescribed velocity is nonzero. That creates a boundary-layer strain
    in the constrained candidates and is therefore an empirical option rather
    than an assumed improvement. Weak normal-velocity conditions remain part
    of the operator and are not imposed on the candidate vectors.

    Arguments:
      rotational: Whether to include all rotational modes.
      translations: Coordinate directions of translations to include.
      dilation: Whether to include the dilation ``u = x``.
      special_conformal: Whether to include the three fields
                          ``u = 2 (a . x) x - |x|^2 a`` for Cartesian unit
                          vectors ``a``.
      constrain_strong_bcs: Whether to zero strongly constrained velocity
                            degrees of freedom before orthonormalisation.
    """
    rotational: bool = True
    translations: tuple[int, ...] = (0, 1, 2)
    dilation: bool = True
    special_conformal: bool = True
    constrain_strong_bcs: bool = False

    def __post_init__(self) -> None:
        if not any(
            (
                self.rotational,
                self.translations,
                self.dilation,
                self.special_conformal,
            )
        ):
            raise ValueError("At least one near-null mode must be requested.")
        if any(
            isinstance(direction, bool) or not isinstance(direction, int)
            for direction in self.translations
        ):
            raise ValueError("Translation directions must be integers.")
        if len(set(self.translations)) != len(self.translations):
            raise ValueError("Translation directions must be unique.")
        if any(direction not in range(3) for direction in self.translations):
            raise ValueError("Translation directions must be selected from 0, 1, and 2.")

    def _build(
        self,
        Z: fd.functionspaceimpl.WithGeometry,
        strong_bcs: list[fd.DirichletBC],
    ) -> fd.nullspace.MixedVectorSpaceBasis:
        """Materialise this specification after solver boundary-condition setup."""
        if len(Z) < 2:
            raise ValueError("A mixed Stokes space must contain velocity and pressure.")

        stokes_subspaces = Z.subspaces
        indexed_velocity_space = stokes_subspaces[0]
        pressure_space = stokes_subspaces[1]
        mesh = Z.mesh()
        if mesh.topological_dimension != 3 or mesh.geometric_dimension != 3:
            raise ValueError(
                "Conformal near-null modes require a three-dimensional "
                "volumetric domain."
            )
        if indexed_velocity_space.value_shape != (3,):
            raise ValueError("The velocity subspace must contain three-component vectors.")
        if pressure_space.value_shape != ():
            raise ValueError("The pressure subspace must be scalar.")

        velocity_space = indexed_velocity_space.collapse()
        mode_functions = _rigid_body_mode_functions(
            velocity_space,
            rotational=self.rotational,
            translations=list(self.translations),
        )
        X = fd.SpatialCoordinate(mesh)

        if self.dilation:
            mode_functions.append(
                fd.Function(velocity_space).interpolate(fd.as_vector(X))
            )

        if self.special_conformal:
            radius_squared = fd.dot(X, X)
            for axis in range(3):
                direction = fd.as_vector([int(i == axis) for i in range(3)])
                mode = (
                    2 * fd.dot(direction, X) * fd.as_vector(X)
                    - radius_squared * direction
                )
                mode_functions.append(fd.Function(velocity_space).interpolate(mode))

        if self.constrain_strong_bcs:
            _apply_homogeneous_velocity_bcs(
                mode_functions,
                velocity_space,
                indexed_velocity_space,
                strong_bcs,
            )

        velocity_basis = fd.VectorSpaceBasis(mode_functions, comm=mesh.comm)
        try:
            velocity_basis.orthonormalize()
        except ValueError as error:
            raise ValueError(
                "The selected conformal modes became linearly dependent after "
                "applying the strong velocity boundary conditions."
            ) from error

        near_nullspace = [velocity_basis, pressure_space]
        near_nullspace += stokes_subspaces[2:]
        return fd.MixedVectorSpaceBasis(Z, near_nullspace)


def _apply_homogeneous_velocity_bcs(
    mode_functions: list[fd.Function],
    velocity_space: fd.functionspaceimpl.WithGeometry,
    indexed_velocity_space: fd.functionspaceimpl.WithGeometry,
    strong_bcs: list[fd.DirichletBC],
) -> None:
    """Apply homogeneous strong velocity conditions to near-null modes."""
    for bc in strong_bcs:
        bc_space = bc.function_space()
        if bc_space == indexed_velocity_space:
            mode_bc = fd.DirichletBC(velocity_space, 0, bc.sub_domain)
        elif (
            bc_space.component is not None
            and bc_space.parent == indexed_velocity_space
        ):
            mode_bc = fd.DirichletBC(
                velocity_space.sub(bc_space.component),
                0,
                bc.sub_domain,
            )
        else:
            continue
        for mode in mode_functions:
            mode_bc.zero(mode)


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
    basis = _rigid_body_mode_functions(
        V,
        rotational=rotational,
        translations=translations,
    )

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=V.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    return V_nullspace


def _rigid_body_mode_functions(
    V: fd.functionspaceimpl.WithGeometry,
    *,
    rotational: bool,
    translations: list[int] | None,
) -> list[fd.Function]:
    """Construct rigid-body mode functions without orthonormalising them."""
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
        for translation_dimension in translations:
            if translation_dimension not in range(dim):
                raise ValueError(
                    f"Translation direction {translation_dimension} is invalid for "
                    f"a {dim}-dimensional geometric domain."
                )
            vector = [0] * dim
            vector[translation_dimension] = 1
            basis.append(fd.Function(V).interpolate(fd.as_vector(vector)))

    return basis
