r"""Derived terms and associated equations for the Stokes system.

All terms are considered as if they were on the left-hand side of the equation, leading
to the following UFL expression returned by `Equation`'s `residual` method:

$$
  dq / dt + F(q) = 0.
$$

"""

from firedrake import *
from ufl.core.expr import Expr
from ufl.indexed import Indexed

from .approximations import QuasiCompressibleInternalVariableApproximation
from .equations import Equation, interior_penalty_factor
from .utility import (
    is_continuous,
    normal_is_continuous,
    tensor_jump,
    upward_normal,
)


def viscosity_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    r"""Viscosity term $-nabla * (mu nabla u)$ in the momentum equation.

    Using the symmetric interior penalty method (Epshteyn & Rivière, 2007), the weak
    form becomes

    $$
    {:( -int_Omega nabla * (mu grad u) phi dx , = , int_Omega mu (grad phi) * (grad u) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(mu grad u) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(u bb n) * "avg"(mu grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(mu) "jump"(u bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter.

    Epshteyn, Y., & Rivière, B. (2007).
    Estimation of penalty parameters for symmetric interior penalty Galerkin methods.
    Journal of Computational and Applied Mathematics, 206(2), 843-872.
    """
    dim = eq.mesh.geometric_dimension
    identity = Identity(dim)
    compressible_stress = eq.approximation.compressible

    mu = eq.approximation.mu
    stress = eq.stress
    F = inner(nabla_grad(eq.test), stress) * eq.dx

    sigma = interior_penalty_factor(eq)
    sigma *= FacetArea(eq.mesh) / avg(CellVolume(eq.mesh))
    if not is_continuous(eq.trial_space):
        trial_tensor_jump = tensor_jump(eq.n, trial) + tensor_jump(trial, eq.n)
        if compressible_stress:
            trial_tensor_jump -= 2 / 3 * identity * jump(trial, eq.n)

        F += (
            sigma
            * inner(tensor_jump(eq.n, eq.test), avg(mu) * trial_tensor_jump)
            * eq.dS
        )
        F -= inner(avg(mu * nabla_grad(eq.test)), trial_tensor_jump) * eq.dS
        F -= inner(tensor_jump(eq.n, eq.test), avg(stress)) * eq.dS

    # NOTE: Unspecified boundaries result in free stress (i.e. free in all directions).
    # NOTE: "un" can be combined with "stress" provided the stress component is
    # tangential (e.g. no normal flow with wind)
    for bc_id, bc in eq.bcs.items():
        if "u" in bc and any(bc_type in bc for bc_type in ["stress", "un"]):
            raise ValueError(
                '"stress" or "un" cannot be specified if "u" is already given.'
            )
        if "normal_stress" in bc and any(bc_type in bc for bc_type in ["u", "un"]):
            raise ValueError(
                '"u" or "un" cannot be specified if "normal_stress" is already given.'
            )

        if "u" in bc:
            trial_tensor_jump = 2 * sym(outer(eq.n, trial - bc["u"]))
            if compressible_stress:
                trial_tensor_jump -= (
                    2 / 3 * identity * (dot(eq.n, trial) - dot(eq.n, bc["u"]))
                )
            # Terms below are similar to the above terms for the DG dS integrals.
            F += (
                2
                * sigma
                * inner(outer(eq.n, eq.test), mu * trial_tensor_jump)
                * eq.ds(bc_id)
            )
            F -= inner(mu * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)
            F -= inner(outer(eq.n, eq.test), stress) * eq.ds(bc_id)

        if "un" in bc:
            trial_tensor_jump = 2 * outer(eq.n, eq.n) * (dot(eq.n, trial) - bc["un"])
            if compressible_stress:
                trial_tensor_jump -= 2 / 3 * identity * (dot(eq.n, trial) - bc["un"])
            # Terms below are similar to the above terms for the DG dS integrals.
            F += (
                2
                * sigma
                * inner(outer(eq.n, eq.test), mu * trial_tensor_jump)
                * eq.ds(bc_id)
            )
            F -= inner(mu * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)
            # We only keep the normal part of stress; the tangential part is assumed to
            # be zero stress (i.e. free slip) or prescribed via "stress".
            F -= dot(eq.n, eq.test) * dot(eq.n, dot(stress, eq.n)) * eq.ds(bc_id)

            if hasattr(eq.approximation, 'bulk_modulus'):
                trial_tensor_jump = identity * (dot(eq.n, trial) - bc["un"])
                bulk = eq.approximation.bulk_modulus * eq.approximation.bulk_shear_ratio
                # Terms below are similar to the above terms for the DG dS integrals.
                F += (
                    2
                    * sigma
                    * inner(outer(eq.n, eq.test), bulk * trial_tensor_jump)
                    * eq.ds(bc_id)
                )
                F -= inner(bulk * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)

        if "stress" in bc:  # a momentum flux, a.k.a. "force"
            # Here we need only the third term because we assume jump_u = 0
            # (u_ext = trial) and stress = n . (mu . stress_tensor).
            F -= dot(eq.test, bc["stress"]) * eq.ds(bc_id)

        if "normal_stress" in bc:
            F += dot(eq.test, bc["normal_stress"] * eq.n) * eq.ds(bc_id)

    return F


def pressure_gradient_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    assert normal_is_continuous(eq.test)

    F = -dot(div(eq.test), eq.p) * eq.dx

    # Integration by parts gives a natural condition on pressure (as part of a normal
    # stress condition). For boundaries where the normal component of u is specified, we
    # remove that condition.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc or "un" in bc:
            F += dot(eq.test, eq.n) * eq.p * eq.ds(bc_id)

    return F


def divergence_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    assert normal_is_continuous(eq.u)

    rho = eq.rho_continuity
    F = -dot(eq.test, div(rho * eq.u)) * eq.dx

    # Add boundary integral for bcs that specify the normal component of u.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc:
            F -= eq.test * rho * dot(eq.n, bc["u"] - eq.u) * eq.ds(bc_id)
        elif "un" in bc:
            F -= eq.test * rho * (bc["un"] - dot(eq.n, eq.u)) * eq.ds(bc_id)

    return F


def momentum_source_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    return -dot(eq.test, eq.source) * eq.dx


def hydrostatic_prestress_advection_and_buoyancy_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    # The advection of hydrostatic prestress and buoyancy terms are combined
    # to form an explicitly symmetric term, following Eqs. B22-B29 in
    # Appendix B of Al-Attar et al. 2014 and Scott et al. 2026, the full references
    # are provided in `approximations.py`.

    # For the Cathles 2024 benchmark in `tests/viscoelastic_internal_variable`
    # we neglect these terms to be consistent with the analytical solution
    if isinstance(eq.approximation, QuasiCompressibleInternalVariableApproximation):
        return 0

    B_mu = eq.approximation.B_mu
    rho0 = eq.approximation.density
    g = eq.approximation.g
    grad_phi = g * upward_normal(eq.mesh)

    F = 0.5 * B_mu * rho0 * dot(grad(dot(trial, grad_phi)), eq.test) * eq.dx
    F += 0.5 * B_mu * rho0 * dot(trial, grad(dot(eq.test, grad_phi))) * eq.dx

    F -= 0.5 * B_mu * rho0 * dot(div(trial)*grad_phi, eq.test) * eq.dx
    F -= 0.5 * B_mu * rho0 * dot(grad_phi, trial) * div(eq.test) * eq.dx

    return F


def self_gravity_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    r"""Self-gravitational body force $-rho_0 nabla psi$ in the momentum equation.

    The perturbed gravitational potential exerts a body force $rho_0 bb g_1$ on
    the reference density, and with every term written as if on the left-hand
    side the residual contribution is

    $$
      int_Omega -B_mu rho_0 (grad psi) * bb w dx
    $$

    **The sign is a MINUS, and that is the convention rather than an oversight.**
    The coupled $psi$ is `GravitySolver`'s potential, which is *minus* the
    Newtonian one: $nabla^2 psi = -4 pi G rho$, the perturbation gravity is
    $bb g_1 = + grad psi$, and a positive point mass has $psi = + G M \/ r$, whose
    gradient points towards the mass. The attraction of a mass excess is
    therefore $+ rho_0 grad psi$ and the residual carries its negative. The same
    minus is what makes the coupled Jacobian symmetric: the $(bb u, psi)$ and
    $(psi, bb u)$ blocks must carry the *same* constant, not opposite ones, and
    with this sign both are $-B_mu$ once the potential rows are scaled by
    $theta_psi = B_mu \/ Lambda$.

    Takes the potential through `eq_attrs` as `psi`, so that the coupled solver
    can hand in a field living on a different (parent) mesh; the volume measure
    then has to be an intersected one, which `Equation` accepts through
    `intersect_measures`.
    """
    B_mu = eq.approximation.B_mu
    rho0 = eq.approximation.density

    return -B_mu * rho0 * dot(grad(eq.psi), eq.test) * eq.dx


def rotational_potential_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Rotational body force $-rho_0 nabla psi_"rot"$ in the momentum equation.

    A perturbation of the rotation vector changes the centrifugal potential, and
    that change acts on the reference density exactly as the self-gravitational
    one does:

    $$
      int_Omega -B_mu rho_0 (grad psi_"rot") * bb w dx
    $$

    **The sign is a MINUS, matching `self_gravity_term`, and it is a minus
    because $psi_"rot"$ is the *negated* centrifugal potential** -- see
    `rotational_potential`, which is the only thing that should ever build it.
    Writing the two adjacent body forces with opposite signs, as the ordinary
    Newtonian convention for the centrifugal potential would require, is how the
    pair gets "fixed" into agreement by the next reader.

    Unlike $psi$, $psi_"rot"$ is not a solved field: it is an explicit low-order
    polynomial in the coordinates times the rotation scalars, so this term is
    linear in those scalars and its transpose is the inertia-perturbation row.
    It never enters the Dirichlet-to-Neumann treatment, because it is not a
    solution of Laplace's equation ($nabla^2 psi_"rot" = +4 Omega^2 m_3$).
    """
    B_mu = eq.approximation.B_mu
    rho0 = eq.approximation.density

    return -B_mu * rho0 * dot(grad(eq.psi_rot), eq.test) * eq.dx


def rotational_potential(rotation, mesh, *, Omega_sq=1) -> Expr:
    r"""The negated centrifugal potential perturbation, as UFL.

    With the rotation vector perturbed to $bb omega = Omega (m_1, m_2, 1 + m_3)$,
    the centrifugal potential $Phi_c = -1/2 [|bb omega|^2 r^2 - (bb omega * bb x)^2]$
    changes to first order by $Omega^2 [m_1 x z + m_2 y z - m_3 (x^2 + y^2)]$.
    What is returned is its *negative*,

    $$
      psi_"rot" = -Omega^2 [m_1 x z + m_2 y z - m_3 (x^2 + y^2)]
    $$

    so that it shares the sign convention of the gravitational potential $psi$
    (minus the Newtonian potential) and both body forces read
    $-B_mu int rho_0 grad(\cdot) * bb w$. This is the single expression the
    momentum term, the rotation rows and any geoid diagnostic should share.

    In two dimensions a disc has no polar wander: $m_1$ and $m_2$ describe a tilt
    of the rotation axis out of a plane that has no third direction, and the only
    surviving mode is the rotation-rate change $m_3$, giving
    $psi_"rot" = +Omega^2 m_3 (x^2 + y^2)$. That is a genuine physical statement
    and not a degenerate case: $m_3$ is angular-momentum conservation of the
    disc, $delta Omega \/ Omega = -Delta I_(3 3) \/ C$.

    Arguments:
      rotation: sequence of rotation scalars. Length 1 in 2-D, holding $m_3$
                alone; length 3 in 3-D, holding $(m_1, m_2, m_3)$. Each entry may
                be a number, a `Constant`, or a `Real`-space function, which is
                what the coupled solver hands in.
      mesh:     mesh supplying the spatial coordinates. Note that the polynomial
                is written about the coordinate origin, so the mesh must be
                centred on the rotation axis' intersection with the equator --
                true of the annulus and sphere meshes used here.
      Omega_sq: the squared rotation rate, non-dimensionalised by $bar g \/ L$.
                For the Earth $Omega^2 L \/ bar g approx 1.6 xx 10^-3$; the
                default of 1 exists to match the module's other non-dimensional
                numbers and is not a physical value.

    Returns:
      A scalar UFL expression.
    """
    dim = mesh.geometric_dimension
    n_rot = 1 if dim == 2 else 3
    if len(rotation) != n_rot:
        raise ValueError(
            f"A {dim}-dimensional mesh takes {n_rot} rotation "
            f"scalar(s); got {len(rotation)}."
        )

    X = SpatialCoordinate(mesh)
    # Index from the end so that the one 2-D scalar is m_3, the rotation-rate
    # change, and the same line reads correctly for both dimensions.
    m_3 = rotation[-1]

    psi_rot = Omega_sq * m_3 * (X[0] ** 2 + X[1] ** 2)
    if n_rot == 3:
        m_1, m_2 = rotation[0], rotation[1]
        psi_rot -= Omega_sq * (m_1 * X[0] * X[2] + m_2 * X[1] * X[2])

    return psi_rot


viscosity_term.required_attrs = {"stress"}
viscosity_term.optional_attrs = {"interior_penalty"}
pressure_gradient_term.required_attrs = {"p"}
pressure_gradient_term.optional_attrs = set()
divergence_term.required_attrs = {"u", "rho_continuity"}
divergence_term.optional_attrs = set()
momentum_source_term.required_attrs = {"source"}
momentum_source_term.optional_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.required_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.optional_attrs = set()
self_gravity_term.required_attrs = {"psi"}
self_gravity_term.optional_attrs = set()
rotational_potential_term.required_attrs = {"psi_rot"}
rotational_potential_term.optional_attrs = set()

momentum_terms = [momentum_source_term, pressure_gradient_term, viscosity_term]
mass_terms = divergence_term
stokes_terms = [momentum_terms, mass_terms]

compressible_viscoelastic_terms = [
    hydrostatic_prestress_advection_and_buoyancy_term,
    momentum_source_term,
    viscosity_term,
]
