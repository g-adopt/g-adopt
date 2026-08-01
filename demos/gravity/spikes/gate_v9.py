"""V9b, V9 and the degree-one question.

**Every predicted number is fixed in `NOTES/GATES-V9.md` Part 0, written before
any of this was run.** Nothing here re-derives one and nothing here is tuned.

Three modes:

    --mode v9b    the alpha-blind Airy control: surface load, no external
                  potential, fluid limit. Predicted `zeta = -sigma_hat/rho_0`
                  at every `lam_factor`.
    --mode v9     the magnitude gate: no load, an imposed external harmonic
                  potential `Phi_ext = phi_hat (r/Re)^n cos(n phi)` added to the
                  momentum body force. Predicted
                  `zeta = -phi_hat/(g_s(1 - c_K) - g_Lambda/n)`.
    --mode deg1   the degree-one experiments: the translation nullspace, the
                  `lam_factor` threshold, and the n = 1 stiffness.

Everything goes through the shipped `demos/gravity/selfgrav_gia_annulus.py`
`build_solver`, with exactly three monkey-patches, all of them hooks the demo
already documents:

  * `SIGMA_HAT`, so V9 can run with no load;
  * `approximation`, the one-argument-compatible factory the demo's docstring
    says gates may replace, so the bulk modulus can be swept;
  * `SelfGravitatingGIASolver`, replaced by a subclass whose `set_form` adds the
    external-potential body force. It is `self_gravity_term` written out with a
    fixed field in place of the unknown, same sign, same `B_mu rho_0`, same
    `scaling_factor`, same intersected measure, and it contributes nothing to
    the Jacobian.

Serial.
"""
import argparse
import json
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see the demo's note
import numpy as np  # noqa: E402
from firedrake import (  # noqa: E402
    Constant, Function, Measure, SpatialCoordinate, VectorSpaceBasis,
    MixedVectorSpaceBasis, as_vector, assemble, atan2, cos, dot, grad, sin,
    sqrt,
)
from gadopt.gia_gravity import SelfGravitatingGIASolver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import generate_selfgrav_annulus as gen  # noqa: E402
import selfgrav_gia_annulus as demo  # noqa: E402
from gadopt import CompressibleInternalVariableApproximation  # noqa: E402

RE = gen.RE
RC = gen.RC
RHO0 = 1.0
G_S = 1.0                       # the Airy restoring stress' surface gravity
LAMBDA = demo.LAMBDA            # 1.1116
B_MU_NOMINAL = demo.B_MU        # 1.2769
LAMBDA_STAR = 2.0 * G_S / (RHO0 * RE)     # 0.90756455, Part 0 section 0.4
#: `--lam-factor` that turns the nominal Lambda into Lambda*.
STAR_FACTOR = LAMBDA_STAR / LAMBDA

_MESHES = {}
TAG = ""

NO_LOAD = 1.0e-12
"""V9's "no surface load", and it cannot be exactly zero. DEFECT, reported not
fixed: with `sigma_hat = 0` the `SIGMA_HAT * cos(2 phi)` expression folds to UFL
`Zero`, `SelfGravitatingGIASolver.enclosed_mass_forms` returns a `mass` form
with no integrals, and `update_total_mass`'s `solve(identity == mass, ...)`
raises `ValueError: Provided RHS is not a linear form` on the first step. A
guard on an empty form would fix it. `1e-12` against a `phi_hat` of `1e-3` and a
response of `2e-3` adds `-1e-12` to `zeta` by superposition, i.e. 5e-10
relative, which is nine orders below the gate's own mesh error."""


# ---------------------------------------------------------------------------
# The three patches
# ---------------------------------------------------------------------------
class ExternalPotentialSolver(SelfGravitatingGIASolver):
    """`SelfGravitatingGIASolver` plus a fixed external-potential body force.

    `EXT = (phi_hat, n)` adds

        -f B_mu int rho_0 grad(psi_ext) . w dx_m,
        psi_ext = -phi_hat (r/Re)^n cos(n phi)

    which is `gadopt.momentum_equation.self_gravity_term` with the unknown
    replaced by a fixed field. The minus sign and the `-Phi_ext` are Part 0
    section 0.4: the solver's `psi` is *minus* the Newtonian potential, so a
    physical external potential `Phi_ext` enters the solver's convention
    negated. The term is independent of the solution, so it is a right-hand
    side and the Jacobian is untouched.
    """

    EXT = None

    def set_form(self):
        super().set_form()
        if type(self).EXT is None:
            return
        phi_hat, n = type(self).EXT
        X = SpatialCoordinate(self.mesh)
        r = sqrt(dot(X, X))
        psi_ext = -phi_hat * (r / RE) ** n * cos(n * atan2(X[1], X[0]))
        w = self.tests[self.layout.displacement]
        self.F += self.scaling_factor * (
            -self.approximation.B_mu * self.approximation.density
            * dot(grad(psi_ext), w)) * self.dx_m


def approximation_factory(bulk_modulus, beta=1.0):
    """The demo's `approximation`, with the bulk modulus and `B_mu` opened up.

    `beta` scales the approximation's `B_mu` and *only* that one, so the body
    forces are mis-scaled by `beta` relative to the Airy restoring stress, which
    `build_solver` builds from the module-level `B_MU`. Part 0 section 0.4's
    rejection region: `zeta g_0/phi_hat = -beta/(1 - beta/2)`, i.e. `-2.4444` at
    `beta = 1.1` against `-2.000`.
    """
    def approximation(density=1.0, g=1.0, self_gravity_number=None):
        return CompressibleInternalVariableApproximation(
            bulk_modulus=bulk_modulus, density=density, shear_modulus=1.0,
            viscosity=1.0, g=g, B_mu=beta * demo.B_MU,
            self_gravity_number=(LAMBDA if self_gravity_number is None
                                 else self_gravity_number))
    return approximation


def degree_one_nullspace_factory(translations):
    """A replacement for `rigid_rotation_nullspace` that adds the translations.

    The team lead's reading of the `m = 1` divergence is that it is the marginal
    degree-one mode - a uniform translation of a self-gravitating body costs no
    energy - so this declares the two uniform translations alongside the rigid
    rotation. `SelfGravitatingGIASolver.solve` projects the declared basis out
    of the solution after every step and copies `solution_old` after it, so
    declaring them really does remove them from the time stepping.
    """
    from gadopt.gia_gravity import rigid_rotation_nullspace

    def factory(Z, layout):
        if not translations:
            return rigid_rotation_nullspace(Z, layout)
        V = Z.sub(layout.displacement)
        X = SpatialCoordinate(layout.mechanics_mesh)
        modes = [as_vector([-X[1], X[0]]),
                 as_vector([Constant(1.0), Constant(0.0)]),
                 as_vector([Constant(0.0), Constant(1.0)])]
        basis = VectorSpaceBasis([Function(V).interpolate(m) for m in modes])
        basis.orthonormalize()
        entries = [Z.sub(i) for i in range(len(Z))]
        entries[layout.displacement] = basis
        return MixedVectorSpaceBasis(Z, entries)
    return factory


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def mode_amplitude(solver, n, kind="cos"):
    """The `cos n phi` (or `sin n phi`) amplitude of `u_r` at Re.

    The demo's `deflection_amplitude` generalised in the degree, spelled the
    same way: a plain `ds` on the submesh, for which Re is an exterior facet.
    """
    sub = solver.mesh
    X = SpatialCoordinate(sub)
    nrm = X / sqrt(dot(X, X))
    u_r = dot(solver.displacement, nrm)
    dss = Measure("ds", domain=sub)(gen.CURVE_RE)
    length = assemble(Constant(1.0) * dss)
    basis = (cos if kind == "cos" else sin)(n * atan2(X[1], X[0]))
    return 2 * assemble(u_r * basis * dss) / length


def degree_one_content(solver):
    """`sqrt(c1^2 + s1^2)` of `u_r` at Re: the size of the m = 1 deflection."""
    return float(np.hypot(mode_amplitude(solver, 1, "cos"),
                          mode_amplitude(solver, 1, "sin")))


# ---------------------------------------------------------------------------
# Build and relax
# ---------------------------------------------------------------------------
def build(*, dr, nazim, truncation, dt, lam, sigma_hat, bulk_modulus,
          ext=None, rotation=True, translations=False, fluid_limit=True,
          b_mu=None, beta=1.0, seed_m1=0.0, load_degree=2):
    """One solver, through the demo's own `build_solver`.

    `lam` is the multiplier on the *nominal* `LAMBDA`, i.e. the demo's
    `--lam-factor`.
    """
    key = (dr, nazim)
    if key not in _MESHES:
        _MESHES[key] = demo.build_meshes(
            dr, nazim, path=os.path.join(HERE, f"v9{TAG}_{dr}_{nazim}.msh"))
    parent, sub = _MESHES[key]

    demo.B_MU = B_MU_NOMINAL if b_mu is None else b_mu
    demo.LOAD_DEGREE = load_degree
    demo.SIGMA_HAT = sigma_hat
    demo.approximation = approximation_factory(bulk_modulus, beta)
    demo.rigid_rotation_nullspace = degree_one_nullspace_factory(translations)
    ExternalPotentialSolver.EXT = ext
    demo.SelfGravitatingGIASolver = ExternalPotentialSolver

    solver, z, layout, bcs, C = demo.build_solver(
        parent, sub, dt=dt, truncation=truncation, rotation=rotation,
        fluid_limit=fluid_limit, lam_factor=lam)
    if seed_m1:
        # DEAD END, kept so the finding is not lost: seeding the *displacement*
        # does nothing at all. `u` is not a state variable of this system - it
        # is a quasi-static equilibrium determined at every step by the internal
        # variables and the load - so the first solve overwrites the seed
        # completely. The only state is the internal variables, which is why the
        # m = 1 threshold below is measured with a genuine degree-one load
        # instead.
        Xm = SpatialCoordinate(solver.mesh)
        rm = sqrt(dot(Xm, Xm))
        solver.solution.subfunctions[solver.layout.displacement].interpolate(
            seed_m1 * cos(atan2(Xm[1], Xm[0])) * Xm / rm)
        solver.solution_old.assign(solver.solution)
    return solver


def relax(solver, *, dt, degree, max_steps, rtol, label="", every=25):
    """Step to the fluid limit, watching the gate's mode and the m = 1 one.

    The convergence criterion is on the *gate's* mode, deliberately: the system
    is linear, so an m = 1 instability does not enter the `cos(n phi)`
    projection except through mesh-induced mode coupling, and the whole point of
    the degree-one finding is that the two can be watched separately.
    """
    hist, one, changes = [], [], []
    previous = None
    converged = False
    print(f"  {'step':>5s}{'t':>9s}{'zeta_n':>18s}{'rel change':>13s}"
          f"{'|m=1|':>13s}{'dev stress':>12s}   {label}")
    for step in range(1, max_steps + 1):
        solver.solve()
        zeta = mode_amplitude(solver, degree)
        d1 = degree_one_content(solver)
        hist.append(zeta)
        one.append(d1)
        change = (abs(zeta - previous) / abs(zeta)
                  if previous is not None and zeta != 0.0 else float("inf"))
        if previous is not None:
            changes.append(change)
        previous = zeta
        converged = change < rtol
        if step == 1 or converged or step == max_steps or step % every == 0:
            print(f"  {step:5d}{step * dt:9.2f}{zeta:18.9e}{change:13.3e}"
                  f"{d1:13.3e}{demo.fluid_limit_residual(solver):12.3e}")
        if converged:
            break
        if not np.isfinite(zeta) or abs(zeta) > 1e6:
            print("  ABORTED: the gate mode itself diverged")
            break

    residual, floor = demo.fluid_limit_residual(solver, with_floor=True)
    ratio = (changes[-1] / changes[-2]
             if len(changes) >= 2 and changes[-2] > 0.0 else float("nan"))
    remaining = (changes[-1] * ratio / (1.0 - ratio)
                 if changes and ratio == ratio and ratio < 1.0
                 else float("nan"))
    out = {"steps": step, "time": step * dt, "zeta": hist[-1],
           "change": changes[-1] if changes else None, "ratio": ratio,
           "remaining": remaining, "m1": one[-1], "m1_history": one,
           "history": hist, "fluid_residual": residual, "fluid_floor": floor,
           "relaxed": bool(residual < 1.5 * floor), "converged": bool(converged)}
    print(f"    steps {step}  zeta {hist[-1]:.9e}  relaxed "
          f"{'YES' if out['relaxed'] else 'NO'} ({residual:.3e} vs floor "
          f"{floor:.3e})  converged {'YES' if converged else 'NO'}"
          f"  |m=1| {one[-1]:.3e}")
    return out


# ---------------------------------------------------------------------------
# Predictions - Part 0's closed forms, evaluated. Not re-derived.
# ---------------------------------------------------------------------------
def c_K(K, g_Lambda, n=2):
    """Part 0 section 0.5's compressibility coefficient."""
    geom = RE ** (-2 * n) * (RE ** (2 * n + 2) - RC ** (2 * n + 2)) / (2 * n + 2)
    return g_Lambda * RHO0 * demo.B_MU / (n * K * RE) * geom


def predicted_ratio(*, g_Lambda, n=2, K=None):
    """`zeta g_s / phi_hat` from Part 0's (III) with the section-0.5 correction."""
    c = 0.0 if K is None else c_K(K, g_Lambda, n)
    return -1.0 / (G_S * (1.0 - c) - g_Lambda / n) * G_S


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
def run_v9b(args):
    print("\n" + "=" * 78)
    print("V9b - the alpha-blind Airy control")
    print(f"PREDICTED: zeta = -sigma_hat/rho_0 = {-args.sigma_hat:.6e} at every "
          "lam_factor")
    print("=" * 78)
    rows = []
    for alpha in args.alphas:
        print(f"\nlam_factor {alpha}  (Lambda {LAMBDA * alpha:.6f})")
        solver = build(dr=args.dr, nazim=args.nazim, truncation=args.truncation,
                       dt=args.dt, lam=alpha, sigma_hat=args.sigma_hat,
                       bulk_modulus=args.bulk_modulus, ext=None,
                       rotation=not args.no_rotation, b_mu=args.b_mu,
                       beta=args.beta, seed_m1=args.seed_m1)
        out = relax(solver, dt=args.dt, degree=2, max_steps=args.steps,
                    rtol=args.rtol, label=f"lam_factor {alpha}",
                    every=args.every)
        out["lam_factor"] = alpha
        out["predicted"] = -args.sigma_hat / RHO0
        out["relative"] = abs(out["zeta"] - out["predicted"]) / abs(out["predicted"])
        rows.append(out)
        print(f"    predicted {out['predicted']:.6e}   measured "
              f"{out['zeta']:.6e}   relative {out['relative']:.3e}")

    print("\n  lam_factor      measured     predicted     relative  converged")
    for r in rows:
        print(f"  {r['lam_factor']:10.4g}{r['zeta']:14.6e}"
              f"{r['predicted']:14.6e}{r['relative']:13.3e}"
              f"  {'YES' if r['converged'] else 'NO'}")
    return rows


def run_v9(args):
    print("\n" + "=" * 78)
    print("V9 - the magnitude gate")
    print("=" * 78)
    rows = []
    for spec in args.cases:
        kind, alpha, K = spec
        if kind == "shipped":
            lam, g_Lambda = alpha, LAMBDA * alpha * RHO0 * RE / 2.0
        else:
            lam = STAR_FACTOR * alpha
            g_Lambda = LAMBDA * lam * RHO0 * RE / 2.0
        pred_inc = predicted_ratio(g_Lambda=g_Lambda, n=args.degree)
        pred_K = predicted_ratio(g_Lambda=g_Lambda, n=args.degree, K=K)
        print(f"\n{kind}  alpha {alpha}  K {K}  ->  lam_factor {lam:.8f}, "
              f"Lambda {LAMBDA * lam:.8f}, g_Lambda {g_Lambda:.8f}")
        print(f"  PREDICTED zeta g_s/phi_hat: {pred_K:.6f} at this K, "
              f"{pred_inc:.6f} incompressible")
        solver = build(dr=args.dr, nazim=args.nazim, truncation=args.truncation,
                       dt=args.dt, lam=lam, sigma_hat=NO_LOAD, bulk_modulus=K,
                       ext=(args.phi_hat, args.degree),
                       rotation=not args.no_rotation, b_mu=args.b_mu,
                       beta=args.beta, seed_m1=args.seed_m1)
        out = relax(solver, dt=args.dt, degree=args.degree,
                    max_steps=args.steps, rtol=args.rtol,
                    label=f"{kind} alpha {alpha} K {K}", every=args.every)
        out.update(kind=kind, alpha=alpha, K=K, lam_factor=lam,
                   g_Lambda=g_Lambda, predicted=pred_K,
                   predicted_incompressible=pred_inc,
                   ratio_measured=out["zeta"] * G_S / args.phi_hat)
        out["relative"] = abs(out["ratio_measured"] - pred_K) / abs(pred_K)
        rows.append(out)
        print(f"    measured zeta g_s/phi_hat {out['ratio_measured']:.6f}   "
              f"predicted {pred_K:.6f}   relative {out['relative']:.3e}")

    print(f"\n  {'case':<12s}{'alpha':>7s}{'K':>8s}{'measured':>13s}"
          f"{'predicted':>13s}{'rel':>11s}  conv")
    for r in rows:
        print(f"  {r['kind']:<12s}{r['alpha']:7.4g}{r['K']:8.4g}"
              f"{r['ratio_measured']:13.5f}{r['predicted']:13.5f}"
              f"{r['relative']:11.3e}  {'Y' if r['converged'] else 'N'}")
    return rows


def run_deg1(args):
    """The degree-one experiments."""
    print("\n" + "=" * 78)
    print("Part 3 - the degree-one question")
    print("=" * 78)
    out = {}

    # (a) is a uniform translation a kernel of this operator at all?
    if "kernel" in args.deg1:
        print("\n(a) ||J u||/||u|| for the rigid rotation and the translations")
        solver = build(dr=args.dr, nazim=args.nazim, truncation=args.truncation,
                       dt=args.dt, lam=1.0, sigma_hat=args.sigma_hat,
                       bulk_modulus=args.bulk_modulus, ext=None,
                       rotation=not args.no_rotation)
        out["kernel"] = kernel_content(solver)

    # (b) the demo configuration with the translations projected out.
    if "project" in args.deg1:
        for translations in (False, True):
            tag = "translations projected" if translations else "as shipped"
            print(f"\n(b) demo configuration, lam_factor 1.0, {tag}")
            solver = build(dr=args.dr, nazim=args.nazim,
                           truncation=args.truncation, dt=args.dt, lam=1.0,
                           sigma_hat=args.sigma_hat,
                           bulk_modulus=args.bulk_modulus, ext=None,
                           rotation=not args.no_rotation,
                           translations=translations, seed_m1=args.seed_m1)
            r = relax(solver, dt=args.dt, degree=2, max_steps=args.steps,
                      rtol=args.rtol, label=tag, every=args.every)
            out[f"project_{translations}"] = r

    # (c) the threshold in lam_factor, predicted 0.816449.
    if "threshold" in args.deg1:
        print(f"\n(c) the m = 1 threshold. PREDICTED lam_factor* "
              f"{STAR_FACTOR:.6f}")
        rows = []
        for alpha in args.threshold:
            solver = build(dr=args.dr, nazim=args.nazim,
                           truncation=args.truncation, dt=args.dt, lam=alpha,
                           sigma_hat=args.sigma_hat,
                           bulk_modulus=args.bulk_modulus, ext=None,
                           rotation=not args.no_rotation, load_degree=1)
            r = relax(solver, dt=args.dt, degree=1, max_steps=args.steps,
                      rtol=args.rtol, label=f"lam_factor {alpha}",
                      every=args.every)
            h = [abs(x) for x in r["history"]]
            tail = [h[i + 1] / h[i] for i in range(max(0, len(h) - 21), len(h) - 1)
                    if h[i] > 0.0]
            r["m1_growth_per_step"] = float(np.median(tail)) if tail else float("nan")
            r["lam_factor"] = alpha
            rows.append(r)
            print(f"    lam_factor {alpha}: zeta_1 {r['zeta']:.6e}, growth "
                  f"per step {r['m1_growth_per_step']:.6f}, Airy "
                  f"{-args.sigma_hat:.3e}")
        out["threshold"] = rows
        print(f"\n  {'lam_factor':>11s}{'zeta_1 final':>16s}"
              f"{'growth/step':>14s}{'converged':>11s}")
        for r in rows:
            print(f"  {r['lam_factor']:11.6f}{r['zeta']:16.6e}"
                  f"{r['m1_growth_per_step']:14.6f}"
                  f"{'YES' if r['converged'] else 'NO':>11s}")

    # (d) the n = 1 stiffness, measured as a forced response.
    if "n1" in args.deg1:
        print("\n(d) the n = 1 forced response, self-consistent constants.")
        print("    PREDICTED zeta g_s/phi_hat = -1/(1 - alpha)")
        rows = []
        for alpha in args.n1:
            lam = STAR_FACTOR * alpha
            g_Lambda = LAMBDA * lam * RHO0 * RE / 2.0
            pred = predicted_ratio(g_Lambda=g_Lambda, n=1,
                                   K=args.bulk_modulus)
            pred_inc = predicted_ratio(g_Lambda=g_Lambda, n=1)
            print(f"\n  alpha {alpha}: predicted {pred:.5f} at K "
                  f"{args.bulk_modulus}, {pred_inc:.5f} incompressible")
            solver = build(dr=args.dr, nazim=args.nazim,
                           truncation=args.truncation, dt=args.dt, lam=lam,
                           sigma_hat=NO_LOAD, bulk_modulus=args.bulk_modulus,
                           ext=(args.phi_hat, 1),
                           rotation=not args.no_rotation, b_mu=args.b_mu,
                           beta=args.beta)
            r = relax(solver, dt=args.dt, degree=1, max_steps=args.steps,
                      rtol=args.rtol, label=f"n=1 alpha {alpha}",
                      every=args.every)
            r.update(alpha=alpha, predicted=pred,
                     predicted_incompressible=pred_inc,
                     ratio_measured=r["zeta"] * G_S / args.phi_hat)
            rows.append(r)
            print(f"    measured {r['ratio_measured']:.5f}  predicted "
                  f"{pred:.5f}")
        out["n1"] = rows
    return out


def kernel_content(solver):
    """`||J v||/||v||` for the rigid rotation and for the two translations.

    The rigid rotation is a declared kernel and measures ~2e-06 (facet geometry
    error). A uniform translation is *not* one in this configuration, because
    `un = 0` at Rc has `u . n != 0` for it, so the prediction is O(1).
    """
    from firedrake import derivative, TrialFunction
    Z = solver.solution.function_space()
    layout = solver.layout
    X = SpatialCoordinate(layout.mechanics_mesh)
    modes = {"rotation": as_vector([-X[1], X[0]]),
             "translation_x": as_vector([Constant(1.0), Constant(0.0)]),
             "translation_y": as_vector([Constant(0.0), Constant(1.0)])}

    J = assemble(derivative(solver.F, solver.solution,
                            TrialFunction(Z)), mat_type="nest").petscmat
    out = {}
    with solver.solution.dat.vec_ro as ref:
        for name, mode in modes.items():
            v = Function(Z)
            v.subfunctions[layout.displacement].interpolate(mode)
            with v.dat.vec_ro as vv:
                w = ref.duplicate()
                J.mult(vv, w)
                out[name] = float(w.norm() / vv.norm())
            print(f"    ||J v||/||v||  {name:<15s} {out[name]:.6e}")
    return out


# ---------------------------------------------------------------------------
def parse_cases(strings):
    """`kind:alpha:K` triples, e.g. `star:1.0:1000`."""
    cases = []
    for s in strings:
        kind, alpha, K = s.split(":")
        cases.append((kind, float(alpha), float(K)))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["v9b", "v9", "deg1"], required=True)
    p.add_argument("--dr", type=float, default=0.2)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--truncation", type=int, default=3)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--sigma-hat", type=float, default=1.0e-3)
    p.add_argument("--phi-hat", type=float, default=1.0e-3)
    p.add_argument("--bulk-modulus", type=float, default=1.0)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--no-rotation", action="store_true")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 2.0])
    p.add_argument("--cases", type=str, nargs="+", default=["star:1.0:1000"])
    p.add_argument("--deg1", type=str, nargs="+",
                   default=["kernel", "project", "threshold", "n1"])
    p.add_argument("--threshold", type=float, nargs="+",
                   default=[0.7, 0.8, 0.85, 0.9])
    p.add_argument("--n1", type=float, nargs="+", default=[0.25, 0.5, 0.9])
    p.add_argument("--b-mu", type=float, default=None,
                   help="B_mu for BOTH the body forces and the Airy restoring "
                        "stress; V9's blindness test")
    p.add_argument("--beta", type=float, default=1.0,
                   help="scale the body forces' B_mu ONLY, leaving the Airy "
                        "restoring stress at the nominal one; V9's rejection "
                        "region")
    p.add_argument("--seed-m1", type=float, default=0.0,
                   help="initialise u with a degree-one radial field of this "
                        "amplitude, so the m = 1 rate is measurable when it "
                        "decays")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the generated mesh file, so that "
                        "two runs can share a directory")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()
    args.cases = parse_cases(args.cases)
    global TAG
    TAG = args.tag

    print(f"mesh dr {args.dr} nazim {args.nazim} truncation {args.truncation} "
          f"dt {args.dt}  Re {RE} Rc {RC}")
    print(f"LAMBDA {LAMBDA}  LAMBDA* {LAMBDA_STAR:.8f}  "
          f"STAR_FACTOR {STAR_FACTOR:.8f}  B_mu {demo.B_MU}")

    if args.mode == "v9b":
        result = run_v9b(args)
    elif args.mode == "v9":
        result = run_v9(args)
    else:
        result = run_deg1(args)

    if args.json:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if k != "history"}
            if isinstance(o, list):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o
        with open(args.json, "w") as fh:
            json.dump(clean(result), fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
