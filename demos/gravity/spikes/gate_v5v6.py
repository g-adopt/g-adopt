"""V5 and V6 - the segregated Picard route, gated against the monolith.

**Why these two are now load-bearing.** The verification story of this branch is
a property of the discrete operator rather than of the resolution: the 4.2e-15
transpose and the 3.1e-14 rotation closure are statements about the forms and
hold at `--coarse` exactly as they would at production size. That matters
because the monolithic solver may not be *affordable* at production size, in
which case the route to the science is: verify the monolith cheaply, gate the
segregated Picard iteration against it, and run production segregated. V5 is
that middle link and it has never been run.

## What is being compared, and what is not

Two different questions hide inside "the segregated solver", and running them
together makes a disagreement unattributable. So there are two gates.

**V5a - the same discretisation, split** (`--v5a`). The mechanics step is a
`CoupledInternalVariableSolver` on the mantle carrying the frozen potential as a
body force and, with a fluid core, the frozen CMB traction; the gravity step
solves the *monolithic residual's own* potential and multiplier rows with `u`
frozen, by a Schur complement on its `nest` Jacobian (V8's device, and it
re-derives no form). The two steps are therefore literally the monolith's own
rows, so this is block Gauss-Seidel on the identical discrete operator and its
fixed point **is** the monolithic answer. **Expected: agreement at the linear
solver's tolerance. Anything above ~1e-8 relative is a defect in one of the two,
not a property of segregation.**

**V5b - the production route** (`--v5b`). The mechanics step is the shipped
*segregated* `InternalVariableSolver`, which is what the production GIA and
sea-level drivers use. This is a genuinely different discrete operator and the
differences are known in advance:

- **The internal variable is eliminated analytically**, `update_m` substituting
  `m = (m_old + (dt/tau) e)/(1 + dt/tau)` as a UFL *expression*, where the
  coupled solver carries `m` as a DG1 *field* satisfying the same relation in
  the L2-projected sense. `e_dev(u)` of a CG2 displacement does not lie in DG1,
  so the two differ by that projection error, `O(h^2)` and a few parts in a
  thousand on this mesh (`fluid_limit_residual`'s docstring measures the same
  quantity from the other side).
- **The Nitsche free-slip terms use `mu_eff` in both solvers; the stress does
  not.** In the segregated solver the substitution makes the stress's effective
  deviatoric modulus `mu_0 tau/(tau + dt)`, matching the penalty; in the coupled
  one `m` is independent, the `(u,u)` part of the stress carries `mu_0`, and the
  mismatch is `1 + dt/tau`. That is A4's FC-4 finding, and it predicts the V5b
  gap is **larger with `un = 0` than with a fluid core**, because the fluid core
  deletes the free-slip condition that carries it.

**Expected before measuring**: V5a at solver tolerance in both CMB treatments;
V5b at the few-1e-3 level with a fluid core and larger with `un = 0`; and under
refinement the fluid-core gap falls like `h^2` while the `un = 0` gap does not
fall at the same rate, because only one of the two mechanisms is a projection
error.

**V6 - the iteration count at `Lambda ~ 1`** (`--v6`). Measured on V5a's loop,
where the iteration is a statement about the coupling strength alone and not
about two discretisations disagreeing. It will not be 2. Reported: the count to
a fixed tolerance and the factor from the unnormalised iterate differences.
V6 also solves the monolith at the production value. A divergent Picard route
fails this gate but does not invalidate a converged monolithic solve.

Serial. Rotation is off throughout: a third row with its own closure would
answer a different question. Every comparison is made **modulo a rigid
rotation**, which is a genuine kernel mode of both operators (A4's FC-NS) and
which the two solvers are free to land on different multiples of.

    PYTHONPATH=<worktree> python demos/gravity/spikes/gate_v5v6.py --all
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see the demo's note
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import FluidCore  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

#: Road map §2.2 production constants, as the 2-D prototype uses them.
B_MU = 1.2769
LAMBDA_NOMINAL = 1.1116
LAMBDA_PRODUCTION = 1.361325
SIGMA_HAT = 1.0e-3
LOAD_DEGREE = 2
RHO_CORE = 2.0            # the fluid core's density, in units of rho_0 = 1
RC = 1.2037


def build_meshes(dr, nazim):
    path = os.path.join(HERE, f"v5_{dr}_{nazim}.msh")
    if COMM_WORLD.rank == 0:
        gen.generate(path, dr_mantle=dr, n_azimuthal=nazim)
    COMM_WORLD.barrier()
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def approximation(lam):
    """Fresh every time: every solver's constructor mutates `mu` in place."""
    return CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=lam)


def mechanics_bcs(sub, fluid):
    Xm = SpatialCoordinate(sub)
    bcs = {gen.CURVE_RE: {"normal_stress": B_MU * SIGMA_HAT * cos(
        LOAD_DEGREE * atan2(Xm[1], Xm[0]))}}
    if not fluid:
        bcs[gen.CURVE_RC] = {"un": 0.0}
    return bcs


def build_monolith(parent, sub, *, lam, fluid, dt=1.0, truncation=3,
                   rho_core=RHO_CORE):
    """The reference: one Newton solve of the whole coupled system."""
    X = SpatialCoordinate(parent)
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_RE: {"interior_sigma": SIGMA_HAT * cos(
            LOAD_DEGREE * atan2(X[1], X[0]))},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=False,
        fluid_core=fluid,
        self_gravity_number=lam)
    z = Function(Z)
    solver = SelfGravitatingGIASolver(
        z, approximation(lam), layout=layout, dt=dt,
        bcs=mechanics_bcs(sub, fluid),
        fluid_core=(FluidCore(boundary=gen.CURVE_RC, rho_core=rho_core)
                    if fluid else None),
        nullspace=rigid_rotation_nullspace(Z, layout))
    return solver, z, layout


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
def rotation_mode(sub):
    """The discrete rigid rotation `(-y, x)`, normalised in L2."""
    V = VectorFunctionSpace(sub, "CG", 2)
    X = SpatialCoordinate(sub)
    r = Function(V).interpolate(as_vector([-X[1], X[0]]))
    r /= norm(r)
    return r


def deflate_rotation(u, r):
    """`u` with its rigid-rotation content removed. Both solvers are free in it.

    A rigid rotation is a kernel mode of the continuum operator (A4's FC-NS
    measured 7.4e-09 for the coupled operator and it converges under
    refinement), so two solvers - or two runs of one - land on different
    multiples of it and a comparison that did not remove it would report a
    difference that is not there. The radial quantities that carry the physics
    are a rotation's blind spot and are unaffected either way.
    """
    dxm = Measure("dx", domain=u.function_space().mesh())
    out = u.copy(deepcopy=True)
    out -= Function(u.function_space()).assign(r) * assemble(
        dot(u, r) * dxm)
    return out


def compare(u_a, u_b, r):
    """Relative L2 difference of two displacements, modulo a rigid rotation."""
    a = deflate_rotation(u_a, r)
    b = deflate_rotation(u_b, r)
    diff = a.copy(deepcopy=True)
    diff -= b
    return norm(diff) / norm(a)


def deflection_amplitude(u, sub):
    """The `cos 2 phi` amplitude of the radial displacement at Re."""
    X = SpatialCoordinate(sub)
    n = X / sqrt(dot(X, X))
    dss = Measure("ds", domain=sub)(gen.CURVE_RE)
    length = assemble(Constant(1.0) * dss)
    return 2 * assemble(dot(u, n) * cos(
        LOAD_DEGREE * atan2(X[1], X[0])) * dss) / length


def geoid_amplitude(solver, z, layout, parent):
    """The `cos 2 phi` amplitude of `psi/g_0` on the Re circle."""
    psi = z.subfunctions[layout.potential]
    X = SpatialCoordinate(parent)
    dSs = solver.form.dS(gen.CURVE_RE)
    length = assemble(avg(Constant(1.0)) * dSs)
    return 2 * assemble(avg(psi / solver.approximation.g * cos(
        LOAD_DEGREE * atan2(X[1], X[0]))) * dSs) / length


# ---------------------------------------------------------------------------
# The two halves of a Picard step
# ---------------------------------------------------------------------------
def solve_gravity_rows(solver, z, layout):
    """Solves the monolith's potential and multiplier rows with `u` frozen.

    V8's device, and it re-derives nothing: the rows are the production
    residual's own, assembled as a `nest` Jacobian and eliminated exactly by a
    Schur complement onto the handful of `Real` multipliers. The rows are linear
    in `(psi, c)`, so one Newton update from any iterate is an *exact* solve of
    that block given `u` - which is what makes this a Picard step and not an
    inner iteration whose tolerance would confound V6's count.

    Returns `(residual, rhs)` in the max-norm, both absolute; the caller
    normalises, because a converging Picard loop drives this block's own
    right-hand side to zero and a self-normalised residual would then read
    roundoff over nothing.
    """
    F = solver.F
    J = assemble(derivative(F, z), mat_type="nest").petscmat
    R = assemble(F)

    ip = layout.potential
    ics = list(layout.multipliers)
    nc = len(ics)

    Ap = J.getNestSubMatrix(ip, ip)
    indptr, indices, data = Ap.getValuesCSR()
    n = Ap.getSize()[0]
    Amat = sp.csr_matrix((data, indices, indptr), shape=(n, n))

    Bmat = np.zeros((n, nc))
    Cmat = np.zeros((nc, n))
    Dmat = np.zeros((nc, nc))
    for a, ia in enumerate(ics):
        Mb = J.getNestSubMatrix(ip, ia)
        if Mb is not None:
            Bmat[:, a] = Mb.convert("dense").getDenseArray()[:, 0]
        Mc = J.getNestSubMatrix(ia, ip)
        if Mc is not None:
            Cmat[a, :] = Mc.convert("dense").getDenseArray()[0, :]
        for b, ib in enumerate(ics):
            Md = J.getNestSubMatrix(ia, ib)
            if Md is not None:
                Dmat[a, b] = Md.convert("dense").getDenseArray()[0, 0]

    b1 = -R.subfunctions[ip].dat.data_ro.copy()
    b2 = -np.array([float(R.subfunctions[i].dat.data_ro[0]) for i in ics])
    core_rhs = ([] if layout.core_pressure is None else [
        float(R.subfunctions[layout.core_pressure].dat.data_ro[0])])

    lu = spla.splu(Amat.tocsc())
    AinvB = np.column_stack([lu.solve(Bmat[:, a]) for a in range(nc)])
    Ainvb1 = lu.solve(b1)
    S = Dmat - Cmat @ AinvB
    y = np.linalg.solve(S, b2 - Cmat @ Ainvb1)
    x = Ainvb1 - AinvB @ y

    z.subfunctions[ip].dat.data[:] += x
    for a, ia in enumerate(ics):
        z.subfunctions[ia].dat.data[:] += y[a]

    resid = assemble(F)
    core_resid = ([] if layout.core_pressure is None else [float(
        resid.subfunctions[layout.core_pressure].dat.data_ro[0])])
    rg = np.hstack([resid.subfunctions[ip].dat.data_ro,
                    [float(resid.subfunctions[i].dat.data_ro[0]) for i in ics],
                    core_resid])
    return np.abs(rg).max(), np.abs(np.hstack([b1, b2, core_rhs])).max()


def forcing_terms(sub, w, u, psi, approx, *, fluid, rho_core=RHO_CORE,
                  core_pressure=None):
    """The frozen-potential terms of the momentum residual, on the submesh.

    Residual convention throughout - every term as if on the left-hand side,
    which is `gadopt.momentum_equation`'s:

    - the self-gravitational body force `-B_mu rho_0 grad(psi).w`, the sign
      being `self_gravity_term`'s and forced by `psi` being *minus* the
      Newtonian potential;
    - with a fluid core, `+dot(w, tau n)` at Rc with
      `tau = B_mu[rho_core psi + rho_core g_0 (u.n) + p_core]`, which is
      `normal_stress`'s own convention and is the `u`-variation of
      `SelfGravitatingGIASolver.fluid_core_energy`.

    The volume prestress term supplies the mantle-side CMB stiffness. The
    explicit spring therefore uses `rho_core`, not the density contrast.

    `u` is the mechanics solver's own unknown, so the buoyancy half of the
    traction stays **implicit**: only `psi` is frozen. Freezing `u` in it as
    well would be a different and much worse iteration, and it would not be the
    segregated scheme anybody would write.

    `psi` is a submesh field, so every measure here is single-mesh. That is not
    tidiness: a cross-mesh *facet* integral whose intersection names a cell
    measure evaluates the other mesh's field at the wrong points and does not
    warn, which `SelfGravitatingGIASolver.fluid_core_measure` documents at 21 %.
    """
    rho0 = approx.density
    dxm = Measure("dx", domain=sub)
    F = -approx.B_mu * rho0 * dot(grad(psi), w) * dxm
    if fluid:
        n = FacetNormal(sub)
        tau = approx.B_mu * (
            Constant(rho_core) * psi
            + Constant(rho_core) * approx.g * dot(u, n)
            + core_pressure)
        F += dot(w, tau * n) * Measure("ds", domain=sub)(gen.CURVE_RC)
    return F


class MechanicsStep:
    """One mechanics solve with `psi` frozen, in either of the two flavours.

    Built once and re-solved, because the frozen potential is a `Function`
    updated in place: rebuilding the solver per Picard iteration would rebuild
    a factorisation and, worse, would make the iteration count depend on how
    the driver was written.

    **`solution_old` is reset before every solve, and that is not optional.**
    `StokesSolverBase.solve` ends with `solution_old.assign(self.solution)`, and
    the segregated `InternalVariableSolver.solve` additionally interpolates the
    internal variable forward. Both are time stepping. A Picard iteration is
    *within* one step, so without the reset each iteration would advance the
    state by one backward-Euler step and the loop would converge to something
    that is not the answer to the problem posed.
    """

    def __init__(self, sub, approx, psi, *, dt, fluid, segregated,
                 rho_core=RHO_CORE):
        self.sub = sub
        self.segregated = segregated
        self.fluid = fluid
        self.core_pressure = Constant(0.0) if fluid else None
        self.rotation = rotation_mode(sub)
        V = VectorFunctionSpace(sub, "CG", 2)
        S = TensorFunctionSpace(sub, "DG", 1)

        if segregated:
            self.solution = Function(V, name="u")
            self.internal = Function(S, name="m")
            forcing = forcing_terms(
                sub, TestFunction(V), self.solution, psi, approx, fluid=fluid,
                rho_core=rho_core, core_pressure=self.core_pressure)
            self.solver = InternalVariableSolver(
                self.solution, approx, internal_variables=[self.internal],
                dt=dt, bcs=mechanics_bcs(sub, fluid),
                additional_forcing_term=forcing,
                solver_parameters="direct")
        else:
            Zm = MixedFunctionSpace([V, S])
            self.solution = Function(Zm, name="z_mech")
            forcing = forcing_terms(
                sub, TestFunctions(Zm)[0], split(self.solution)[0], psi,
                approx, fluid=fluid, rho_core=rho_core,
                core_pressure=self.core_pressure)
            self.solver = CoupledInternalVariableSolver(
                self.solution, approx, dt=dt, bcs=mechanics_bcs(sub, fluid),
                additional_forcing_term=forcing, solver_parameters="direct")

    @property
    def displacement(self):
        return (self.solution if self.segregated
                else self.solution.subfunctions[0])

    def _reset_and_solve(self, core_pressure=0.0):
        self.solver.solution_old.assign(0.0)
        self.solution.assign(0.0)
        if self.segregated:
            self.internal.assign(0.0)
        if self.core_pressure is not None:
            self.core_pressure.assign(core_pressure)
        self.solver.solve()

    def _core_flux(self, u):
        n = FacetNormal(self.sub)
        dss = Measure("ds", domain=self.sub)(gen.CURVE_RC)
        return float(assemble(dot(u, n) * dss))

    def solve(self):
        self._reset_and_solve(0.0)
        if self.fluid:
            base = self.solution.copy(deepcopy=True)
            base_internal = (self.internal.copy(deepcopy=True)
                             if self.segregated else None)
            self._reset_and_solve(1.0)
            direction = self.solution.copy(deepcopy=True)
            direction -= base
            base_u = base if self.segregated else base.subfunctions[0]
            direction_u = (direction if self.segregated
                           else direction.subfunctions[0])
            response = self._core_flux(direction_u)
            if abs(response) < 1.0e-14:
                raise RuntimeError(
                    "the unit core-pressure solve produced no CMB flux response")
            pressure = -self._core_flux(base_u) / response
            self.solution.assign(base + pressure * direction)
            if self.segregated:
                unit_internal = self.internal.copy(deepcopy=True)
                unit_internal -= base_internal
                self.internal.assign(
                    base_internal + pressure * unit_internal)
            self.core_pressure.assign(pressure)

        # Both operators have the rigid rotation in their kernel to
        # facet-geometry error, so the multiple each solve lands on is set by
        # nothing physical. Remove it, exactly as the monolith's
        # `project_out_nullspace` does after every solve.
        u = self.displacement
        u.assign(deflate_rotation(u, self.rotation))
        if self.fluid:
            flux = self._core_flux(u)
            if abs(flux) > 1.0e-11:
                raise RuntimeError(
                    f"the constrained mechanics solve left CMB flux {flux:.3e}")
        return u


# ---------------------------------------------------------------------------
# The Picard loop
# ---------------------------------------------------------------------------
def picard(parent, sub, *, lam, fluid, segregated, dt=1.0, rtol=1e-12,
           max_iter=40, verbose=True, omega=1.0, rho_core=RHO_CORE,
           diverged_at=1e4):
    """Block Gauss-Seidel between the mechanics and the gravity rows.

    Returns the history and the converged fields. The gravity half always uses
    the monolith's own rows, so `segregated` selects the mechanics
    discretisation and nothing else - which is what makes a V5b gap
    attributable to the mechanics rather than to two different Poisson solves.
    """
    solver, z, layout = build_monolith(parent, sub, lam=lam, fluid=fluid, dt=dt,
                                       rho_core=rho_core)
    psi = Function(FunctionSpace(sub, "CG", 2), name="psi_frozen")
    step = MechanicsStep(sub, approximation(lam), psi, dt=dt, fluid=fluid,
                         segregated=segregated, rho_core=rho_core)
    updated = Function(psi.function_space())

    history = []
    previous = None
    previous_z = None
    error_mode = Function(z.function_space())
    gravity_residual = 0.0
    scale = None
    first_u = None
    for k in range(1, max_iter + 1):
        u = step.solve()
        if first_u is None:
            first_u = u.copy(deepcopy=True)
        # Hand the displacement to the monolith's own mixed function and solve
        # its gravity rows exactly. The parent's CG2 space restricted to the
        # mantle IS the submesh's, so both transfers move nodal values only.
        z.subfunctions[layout.displacement].assign(u)
        if layout.core_pressure is not None:
            z.subfunctions[layout.core_pressure].assign(step.core_pressure)
        residual, rhs = solve_gravity_rows(solver, z, layout)
        # Normalised by the FIRST iterate's right-hand side and not by its own.
        # A converging Picard loop drives the gravity update's own right-hand
        # side to zero, so a self-normalised residual reads roundoff-over-zero
        # and grows: the instrument would report a defect where there is none.
        scale = rhs if scale is None else scale
        gravity_residual = max(gravity_residual, residual / max(scale, 1e-300))

        updated.interpolate(z.subfunctions[layout.potential])
        # Under-relaxation on the frozen potential. It is here because the
        # obvious response to a divergent Picard loop is to damp it - and
        # measuring that it *cannot* help is worth more than assuming it can:
        # damping maps an eigenvalue `l` of the iteration to `1 + omega(l - 1)`,
        # which for a real `l > 1` is further from the unit disc for every
        # `omega` in (0, 1].
        psi.assign((1.0 - omega) * psi + omega * updated)

        # The successive difference of the FULL mixed iterate. A linear
        # fixed-point iteration `z_{k+1} = T z_k + c` has differences obeying
        # `e_{k+1} = T e_k` exactly, so this is a power iteration on the block
        # Gauss-Seidel error operator and, when the loop diverges, converges to
        # its dominant eigenvector. The iterate itself would not: it carries
        # the fixed point as well.
        if previous_z is not None:
            error_mode.assign(z - previous_z)
        previous_z = z.copy(deepcopy=True)

        zeta = deflection_amplitude(u, sub)
        # The criterion is the whole displacement and not the surface
        # deflection. Measured the other way first, and it stops too early:
        # with a fluid core the slowest mode is the CMB's own and is nearly
        # invisible at Re, so a loop stopped on `zeta` agrees with the monolith
        # at 1e-10 in the deflection and only 2.4e-06 in `u`.
        if previous is None:
            difference_norm = float("inf")
            change = float("inf")
        else:
            delta = u.copy(deepcopy=True)
            delta -= previous
            difference_norm = norm(delta)
            change = difference_norm / max(norm(u), 1e-300)
        previous_difference = (history[-1]["difference_norm"]
                               if history else float("inf"))
        iteration_factor = (difference_norm / previous_difference
                            if np.isfinite(previous_difference)
                            and previous_difference > 0.0 else float("nan"))
        history.append({"k": k, "zeta": zeta, "change": change,
                        "difference_norm": difference_norm,
                        "iteration_factor": iteration_factor})
        previous = u.copy(deepcopy=True)
        if verbose:
            print(f"    {k:3d}  zeta {zeta: .10e}  change {change:.3e}")
        if change < rtol:
            break
        # Detect divergence from the unnormalised difference. The relative
        # change approaches a constant for a divergent linear iteration.
        recent_factors = [h["iteration_factor"] for h in history[-5:]]
        if (k > 10 and len(recent_factors) == 5
                and all(np.isfinite(factor) and factor > 1.0
                        for factor in recent_factors)):
            break
        if abs(zeta) > diverged_at:
            break

    factors = [h["iteration_factor"] for h in history
               if np.isfinite(h["iteration_factor"])]
    zetas = [h["zeta"] for h in history]
    growth = (abs(zetas[-1] / zetas[-2])
              if len(zetas) > 1 and zetas[-2] != 0.0 else float("nan"))
    converged = history[-1]["change"] < rtol
    return {"solver": solver, "z": z, "layout": layout, "u": step.displacement,
            "error_mode": error_mode,
            "history": history, "iterations": len(history),
            "iteration_factor": factors[-1] if factors else float("nan"),
            "growth": growth,
            "gravity_residual": gravity_residual,
            "first_u": first_u,
            "converged": converged,
            "diverged": not converged}


def monolithic_reference(parent, sub, *, lam, fluid, dt=1.0):
    solver, z, layout = build_monolith(parent, sub, lam=lam, fluid=fluid, dt=dt)
    solver.solve()
    return solver, z, layout


# ---------------------------------------------------------------------------
# V5
# ---------------------------------------------------------------------------
#: The low-coupling pair separates the boundary treatment from the coupling
#: strength. The nominal rigid-core case retains the original production point.
V5_CASES = (("fluid core", True, 0.1),
            #: `un = 0` at the fluid core's Lambda as well, so that the
            #: attribution of a V5b gap to the free-slip condition is made at
            #: equal coupling rather than across two different ones.
            ("un = 0 matched", False, 0.1),
            ("un = 0", False, LAMBDA_NOMINAL))


def gate_v5(dr, nazim, *, segregated, dt=1.0):
    label = ("V5b  the production segregated mechanics" if segregated
             else "V5a  the same discretisation, split")
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print("Expected, before the run:")
    if segregated:
        print("  fluid core   ~1e-03 relative, the DG1 projection of e_dev")
        print("  un = 0       larger, the mu_eff-against-mu_0 Nitsche mismatch")
        print("               on top of it (A4's FC-4)")
        print("  The ORDERING held and the MAGNITUDE did not: measured 7e-05")
        print("  and 2e-04, better than predicted by more than an order, and")
        print("  falling at order ~3.8 under refinement rather than the 2 a")
        print("  pure DG1 projection error would give. The prediction is left")
        print("  standing above so the miss is visible.")
    else:
        print("  both CMB treatments   <= 1e-08 relative - the fixed point IS")
        print("                        the monolithic answer, so anything")
        print("                        larger is a defect and not segregation")
    print("The low-coupling pair separates the CMB treatment from Lambda.")
    print("V6 measures both corrected iterations through production Lambda.")

    parent, sub = build_meshes(dr, nazim)
    r = rotation_mode(sub)
    out = {}
    for name, fluid, lam in V5_CASES:
        print(f"\n  -- {name}, Lambda {lam}")
        run = picard(parent, sub, lam=lam, fluid=fluid, segregated=segregated,
                     dt=dt, verbose=False)
        mono_solver, mono_z, mono_layout = monolithic_reference(
            parent, sub, lam=lam, fluid=fluid, dt=dt)
        u_mono = mono_z.subfunctions[mono_layout.displacement]

        zeta_s = deflection_amplitude(run["u"], sub)
        zeta_m = deflection_amplitude(u_mono, sub)
        geoid_s = geoid_amplitude(run["solver"], run["z"], run["layout"], parent)
        geoid_m = geoid_amplitude(mono_solver, mono_z, mono_layout, parent)
        out[name] = {
            "u": compare(run["u"], u_mono, r),
            "zeta": abs(zeta_s - zeta_m) / abs(zeta_m),
            "geoid": abs(geoid_s - geoid_m) / abs(geoid_m),
            "iterations": run["iterations"],
            "converged": run["converged"],
            "gravity_residual": run["gravity_residual"],
        }
        d = out[name]
        print(f"     Picard converged                   {d['converged']}"
              f"   in {d['iterations']} iterations")
        print(f"     gravity rows' residual             "
              f"{d['gravity_residual']:.3e}   (relative to the first "
              "iterate's own right-hand side)")
        print(f"     ||u_seg - u_mono|| / ||u_mono||    {d['u']:.4e}")
        print(f"     deflection  segregated {zeta_s: .8e}")
        print(f"                 monolithic {zeta_m: .8e}   rel "
              f"{d['zeta']:.3e}")
        print(f"     geoid       segregated {geoid_s: .8e}")
        print(f"                 monolithic {geoid_m: .8e}   rel "
              f"{d['geoid']:.3e}")

    if segregated:
        ratio = out["un = 0 matched"]["u"] / out["fluid core"]["u"]
        print(f"\n  At the SAME Lambda, `un = 0` disagrees {ratio:.1f}x more "
              "than the fluid core,\n  which is the FC-4 prediction: the "
              "free-slip condition is what carries the\n  mu mismatch, and the "
              "fluid core deletes it.")
        ok = all(d["converged"] for d in out.values()) and ratio > 1.0
    else:
        ok = all(d["converged"] and d["u"] <= 1e-8 and d["geoid"] <= 1e-8
                 for d in out.values())
    print(f"\n{label.split()[0]} {'PASS' if ok else 'FAIL'}")
    return ok, out


def gate_v5b_refinement(dr, nazim):
    """Does the V5b gap fall like the projection error it is claimed to be?"""
    print("\n" + "=" * 78)
    print("V5b-h  the gap under one refinement")
    print("=" * 78)
    print("Expected: the fluid core's gap is the DG1 projection of e_dev, so it")
    print("falls at O(h^2); the `un = 0` gap carries the Nitsche mismatch as")
    print("well, a boundary term of size dt/(tau+dt), and should fall slower.")
    rows = {}
    for d, n in ((dr, nazim), (dr / 2, 2 * nazim)):
        parent, sub = build_meshes(d, n)
        r = rotation_mode(sub)
        for name, fluid, lam in V5_CASES:
            run = picard(parent, sub, lam=lam, fluid=fluid, segregated=True,
                         verbose=False)
            _, mz, ml = monolithic_reference(parent, sub, lam=lam, fluid=fluid)
            rows.setdefault(name, []).append(
                compare(run["u"], mz.subfunctions[ml.displacement], r))
    print(f"\n  {'':<14s}{'coarse':>14s}{'refined':>14s}{'ratio':>10s}"
          f"{'order':>9s}")
    for k, (a, b) in rows.items():
        print(f"  {k:<14s}{a:14.4e}{b:14.4e}{b / a:10.3f}"
              f"{np.log2(a / b):9.2f}")
    ok = rows["fluid core"][1] < 0.6 * rows["fluid core"][0]
    print(f"\nV5b-h {'PASS' if ok else 'FAIL'}")
    return ok, rows


# ---------------------------------------------------------------------------
# V6
# ---------------------------------------------------------------------------
def gate_v6(dr, nazim, *, dt=1.0):
    """Measure the stationary block-Picard convergence against `Lambda`.

    The constrained mechanics block is a Lagrange saddle. Picard divergence
    therefore does not imply a negative physical energy or a bad monolith.
    It means that this stationary block iteration has spectral radius above one.
    """
    print("\n" + "=" * 78)
    print("V6  the constrained Picard iteration count at Lambda ~ 1")
    print("=" * 78)
    print("The fluid-core mechanics step includes the uniform pressure and")
    print("enforces zero CMB flux. The old degree-zero instability is excluded.")
    print(f"Both Picard routes must converge at Lambda = {LAMBDA_PRODUCTION}.")
    print("A failed Picard route does not invalidate the coupled monolith.")

    parent, sub = build_meshes(dr, nazim)
    ladder = [0.1, 0.25, 0.5, 1.0, LAMBDA_NOMINAL,
              LAMBDA_PRODUCTION, 2.0]
    rows = {}
    for name, fluid in (("un = 0", False), ("fluid core", True)):
        print(f"\n  -- {name}")
        print(f"  {'Lambda':>10s}{'iterations':>12s}{'factor':>13s}"
              f"{'growth':>11s}{'converged':>11s}")
        rows[name] = {}
        for lam in ladder:
            run = picard(parent, sub, lam=lam, fluid=fluid, segregated=False,
                         dt=dt, rtol=1e-12, max_iter=60, verbose=False)
            rows[name][lam] = run
            print(f"  {lam:10.6f}{run['iterations']:12d}"
                  f"{run['iteration_factor']:13.4f}{run['growth']:11.4f}"
                  f"{str(run['converged']):>11s}")

    print("\n  -- production monolith")
    monolith = {}
    for name, fluid in (("un = 0", False), ("fluid core", True)):
        solver, z, layout = monolithic_reference(
            parent, sub, lam=LAMBDA_PRODUCTION, fluid=fluid, dt=dt)
        # `solve()` advances `solution_old` after it solves. Restore this
        # one-step gate's zero history before reassembling the solved residual.
        solver.solution_old.assign(0.0)
        residual = assemble(solver.F)
        with residual.dat.vec_ro as residual_vec:
            residual_norm = residual_vec.norm()
        with z.dat.vec_ro as solution_vec:
            finite_solution = np.isfinite(solution_vec.norm())
        flux = 0.0
        if fluid:
            n = FacetNormal(sub)
            dss = Measure("ds", domain=sub)(gen.CURVE_RC)
            flux = float(assemble(dot(solver.displacement, n) * dss))
        first_gap = compare(
            rows[name][LAMBDA_PRODUCTION]["first_u"],
            z.subfunctions[layout.displacement], rotation_mode(sub))
        monolith[name] = {"residual": residual_norm, "flux": flux,
                          "finite": finite_solution,
                          "first_gap": first_gap}
        print(f"  {name:<12s} residual {residual_norm:.3e}"
              f"   CMB flux {flux:.3e}   first gap {first_gap:.3e}")

    production = [rows[name][LAMBDA_PRODUCTION]
                  for name in ("un = 0", "fluid core")]
    picard_ok = all(run["converged"]
                    and run["iteration_factor"] < 1.0 for run in production)
    monolith_ok = all(item["finite"] and item["residual"] < 1e-10
                      and abs(item["flux"]) < 1e-11
                      and item["first_gap"] > 1e-3
                      for item in monolith.values())
    ok = picard_ok and monolith_ok
    print(f"\nV6 {'PASS' if ok else 'FAIL'}")
    if not picard_ok and monolith_ok:
        print("The coupled monolith passes. The production fluid Picard route "
              "is rejected.")
    return ok, {"lambda": rows, "monolith": monolith}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dr", type=float, default=0.2)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--v5a", action="store_true")
    p.add_argument("--v5b", action="store_true")
    p.add_argument("--v5bh", action="store_true")
    p.add_argument("--v6", action="store_true")
    p.add_argument("--all", action="store_true")
    args, _ = p.parse_known_args()

    if COMM_WORLD.size > 1:
        raise SystemExit("Serial only.")

    run = {"v5a": args.v5a, "v5b": args.v5b, "v5bh": args.v5bh, "v6": args.v6}
    if args.all or not any(run.values()):
        run = dict.fromkeys(run, True)

    results = {}
    if run["v5a"]:
        results["V5a"], _ = gate_v5(args.dr, args.nazim, segregated=False)
    if run["v5b"]:
        results["V5b"], _ = gate_v5(args.dr, args.nazim, segregated=True)
    if run["v5bh"]:
        results["V5b-h"], _ = gate_v5b_refinement(args.dr, args.nazim)
    if run["v6"]:
        results["V6"], _ = gate_v6(args.dr, args.nazim)

    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"  {k:<8s} {'PASS' if v else 'FAIL'}")
    print("=" * 78)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
