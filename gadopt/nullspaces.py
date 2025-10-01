"""Helper functions to generate null spaces for stokes problems

`ala_right_nullspace` computes the pressure null space for the Anelastic Liquid
Approximation. `create_stokes_nullspace`, automatically generates null spaces
for the mixed velocity-pressure Stokes system. `create_u_nullspace' returns the
translational and rotational null spaces associated with the velocity
(or displacement) field.
"""

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
      translations: List of translations to include
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

    V_nullspace = create_u_nullspace(
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


def create_u_nullspace(
    V: fd.functionspaceimpl.WithGeometry,
    rotational: bool = False,
    translations: list[int] | None = None,
    ala_approximation: AnelasticLiquidApproximation | None = None,
    top_subdomain_id: str | int | None = None,
) -> fd.nullspace.VectorSpaceBasis:
    """Create a null space for the velocity (or displacement) in a Stokes system

    Arguments:
      V: Firedrake function space associated with the velocity or displacement
      rotational: Whether to include all rotational modes
      translations: List of translations to include

    Returns:
      A Firedrake vector space basis incorporating the null space components

    """
    X = fd.SpatialCoordinate(V.mesh())
    dim = len(X)

    if rotational:
        if dim == 2:
            rotV = fd.Function(V).interpolate(
                fd.as_vector((-X[1], X[0]))
            )
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(V).interpolate(
                fd.as_vector((0, -X[2], X[1]))
            )
            y_rotV = fd.Function(V).interpolate(
                fd.as_vector((X[2], 0, -X[0]))
            )
            z_rotV = fd.Function(V).interpolate(
                fd.as_vector((-X[1], X[0], 0))
            )
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(
                fd.Function(V).interpolate(fd.as_vector(vec))
            )

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=V.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    return V_nullspace
