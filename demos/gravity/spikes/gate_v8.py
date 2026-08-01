"""V8 - the geoid sign and magnitude, through the `sigma_bcs` sheet path.

**The number is predicted before the run, in `NOTES/GATES-V8V7.md`, and it is
not fitted.**  A frozen mass sheet `sigma = sigma_hat cos(m phi)` on the circle
Re with `u = 0`.  The solver's convention is `grad^2 psi = -4 pi G rho` with
`G = Lambda/(4 pi)`, so the 2-D sheet solution `psi_a = 2 pi G sigma_hat a / m`
becomes

    psi(Re) = Lambda sigma_hat Re / (2 m)          [cos(m phi) amplitude]

and the geoid `N = +psi/g_0` (`BaseGIAApproximation.geoid`) is

    N(Re, phi = 0) = Lambda sigma_hat Re / (2 m g_0)   > 0.

Under the homogeneous-disc normalisation `g_0 = 2 pi G rho_0 Re
= Lambda rho_0 Re / 2` this is `sigma_hat / (m rho_0)`, i.e. review D5's
`sigma_hat / (2 rho_0)` at `m = 2`.  Both normalisations are run.

The gate is the SIGN first and the magnitude second: a mass excess must pull
the geoid up beneath it, and nothing else in the suite looks.

`u` is frozen by the same device as V3': the displacement sub-function is left
at zero and the potential and multiplier rows - which are linear in `(psi, c)` -
are solved exactly by a Schur complement on the `nest` Jacobian of the
production residual.  Nothing here re-derives a form.

Serial only.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see the demo's note
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402
from ufl import replace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

B_MU = 1.2769
LAMBDA = 1.1116
RHO0 = 1.0
SIGMA_HAT = 1.0e-3


def build(dr, nazim, truncation, m_mode, g0, rotation=True):
    """Parent + submesh, the coupled solver with a `cos(m phi)` sheet at Re."""
    path = os.path.join(HERE, f"v8_{dr}_{nazim}.msh")
    gen.generate(path, dr_mantle=dr, n_azimuthal=nazim)
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False

    X = SpatialCoordinate(parent)
    sigma = SIGMA_HAT * cos(m_mode * atan2(X[1], X[0]))

    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_RE: {"interior_sigma": sigma},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        self_gravity_number=LAMBDA)
    z = Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=RHO0, shear_modulus=1.0, viscosity=1.0,
        g=g0, B_mu=B_MU, self_gravity_number=LAMBDA)

    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    C = assemble(approx.density * dot(Xm, Xm) * dx_m)

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=1.0,
        bcs={gen.CURVE_RC: {"un": 0.0}},
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH)
    solver.update_total_mass()
    return parent, sub, solver, z, layout, approx, C


def solve_gravity_block(solver, z, layout):
    """Solve the potential and multiplier rows exactly, `u` held at zero.

    Lifted verbatim from `gate_v3prime.py`, where it was validated against
    `passess.SheetPolar2D`.
    """
    if solver.strong_bcs:  # pragma: no cover - configuration guard
        raise RuntimeError(
            f"{len(solver.strong_bcs)} strong bcs present; the reduced solve "
            "below ignores them.")

    F = solver.F
    J = assemble(derivative(F, z), mat_type="nest").petscmat
    R = assemble(F)

    ip = layout.potential
    ics = list(layout.multipliers)
    nc = len(ics)

    def block(i, j):
        return J.getNestSubMatrix(i, j)

    Ap = block(ip, ip)
    indptr, indices, data = Ap.getValuesCSR()
    n = Ap.getSize()[0]
    Amat = sp.csr_matrix((data, indices, indptr), shape=(n, n))

    Bmat = np.zeros((n, nc))
    Cmat = np.zeros((nc, n))
    Dmat = np.zeros((nc, nc))
    for a, ia in enumerate(ics):
        Mb = block(ip, ia)
        if Mb is not None:
            Bmat[:, a] = Mb.convert("dense").getDenseArray()[:, 0]
        Mc = block(ia, ip)
        if Mc is not None:
            Cmat[a, :] = Mc.convert("dense").getDenseArray()[0, :]
        for b, ib in enumerate(ics):
            Md = block(ia, ib)
            if Md is not None:
                Dmat[a, b] = Md.convert("dense").getDenseArray()[0, 0]

    b1 = -R.subfunctions[ip].dat.data_ro.copy()
    b2 = -np.array([float(R.subfunctions[i].dat.data_ro[0]) for i in ics])

    lu = spla.splu(Amat.tocsc())
    AinvB = np.column_stack([lu.solve(Bmat[:, a]) for a in range(nc)])
    Ainvb1 = lu.solve(b1)
    S = Dmat - Cmat @ AinvB
    y = np.linalg.solve(S, b2 - Cmat @ Ainvb1)
    x = Ainvb1 - AinvB @ y

    z.subfunctions[ip].dat.data[:] = x
    for a, ia in enumerate(ics):
        z.subfunctions[ia].dat.data[:] = y[a]

    resid = assemble(F)
    rg = np.hstack([resid.subfunctions[ip].dat.data_ro,
                    [float(resid.subfunctions[i].dat.data_ro[0]) for i in ics]])
    return np.abs(rg).max(), np.abs(np.hstack([b1, b2])).max()


def to_parent(solver, z, expr):
    """Swap `split(z)` components for real sub-functions.

    `SelfGravitatingGIASolver.geoid` is written in terms of `self.solution_split`,
    which is `firedrake.split` of a mixed function spanning two meshes.  Those
    components are legal inside the residual and awkward outside it, so the
    accessor's own expression is taken and its coefficients replaced - the
    expression under test is still the production one, only its coefficients are
    the concrete sub-functions.
    """
    mapping = {solver.solution_split[i]: z.subfunctions[i]
               for i in range(len(z.subfunctions))}
    return replace(expr, mapping)


def circle_amplitude(solver, parent, expr, tag, m):
    """The `cos(m phi)` Fourier amplitude of `expr` on a tagged interior circle."""
    X = SpatialCoordinate(parent)
    phi = atan2(X[1], X[0])
    dSs = solver.form.dS(tag)
    length = assemble(avg(Constant(1.0)) * dSs)
    if m == 0:
        return assemble(avg(expr) * dSs) / length
    return 2.0 * assemble(avg(expr * cos(m * phi)) * dSs) / length


def point_values(parent, expr, radius, phis):
    """`expr` sampled on a circle, via a CG2 interpolant on the parent."""
    V = FunctionSpace(parent, "CG", 2)
    f = Function(V).interpolate(expr)
    out = {}
    for label, phi in phis.items():
        r = radius * (1.0 - 1.0e-9)
        try:
            out[label] = float(
                f.at([r * np.cos(phi), r * np.sin(phi)], tolerance=1e-8))
        except Exception as exc:  # pragma: no cover - geometry dependent
            out[label] = f"unavailable ({type(exc).__name__})"
    return out


def run(dr, nazim, truncation, m_mode, g0, label):
    parent, sub, solver, z, layout, approx, C = build(
        dr, nazim, truncation, m_mode, g0)
    res, rhs = solve_gravity_block(solver, z, layout)

    psi = z.subfunctions[layout.potential]

    # The three geoids: the approximation's, the solver's without rotation, and
    # the solver's with it.
    N_approx = approx.geoid(psi)
    N_solver_norot = to_parent(solver, z, solver.geoid(include_rotation=False))
    N_solver_rot = to_parent(solver, z, solver.geoid(include_rotation=True))

    amp = {
        "psi": circle_amplitude(solver, parent, psi, gen.CURVE_RE, m_mode),
        "N_approx": circle_amplitude(
            solver, parent, N_approx, gen.CURVE_RE, m_mode),
        "N_solver_norot": circle_amplitude(
            solver, parent, N_solver_norot, gen.CURVE_RE, m_mode),
        "N_solver_rot": circle_amplitude(
            solver, parent, N_solver_rot, gen.CURVE_RE, m_mode),
    }
    purity = {
        "m=0": circle_amplitude(solver, parent, psi, gen.CURVE_RE, 0),
        f"m={m_mode}": amp["psi"],
        f"m={2 * m_mode}": circle_amplitude(
            solver, parent, psi, gen.CURVE_RE, 2 * m_mode),
    }

    phis = {"max (phi=0)": 0.0,
            "zero (phi=pi/2m)": np.pi / (2 * m_mode),
            "min (phi=pi/m)": np.pi / m_mode}
    pts = point_values(parent, N_solver_rot, gen.RE, phis)

    # m_3 must be identically zero here: u = 0 kills the volume part of dI_33
    # and a cos(m phi) sheet with m >= 1 integrates to zero against r^2.
    m3 = solver.rotation_values()
    dI = solver.inertia_perturbation()

    # The rotation half of the accessor, tested on its own: assign a nonzero
    # m_3 by hand and require the difference of the two geoids to be exactly
    # Omega_sq m_3 r^2 / g_0.
    slot = layout.rotation_slots()[2]
    z.subfunctions[slot].assign(1.0)
    Xp = SpatialCoordinate(parent)
    want = float(solver.Omega_sq) * 1.0 * dot(Xp, Xp) / float(approx.g)
    got = (to_parent(solver, z, solver.geoid(include_rotation=True))
           - to_parent(solver, z, solver.geoid(include_rotation=False)))
    dx_p = Measure("dx", domain=parent)
    rot_err = (sqrt(assemble((got - want) ** 2 * dx_p))
               / sqrt(assemble(want ** 2 * dx_p)))
    z.subfunctions[slot].assign(0.0)

    return {
        "label": label, "g0": g0, "C": C,
        "parent_cells": parent.num_cells(),
        "reduced_residual": res, "reduced_rhs": rhs,
        "amp": amp, "purity": purity, "points": pts,
        "m3": m3, "dI": dI, "rot_accessor_rel": rot_err,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--mode", type=int, default=2)
    p.add_argument("--meshes", type=str, default="0.1:64")
    args = p.parse_args()

    m = args.mode
    psi_pred = LAMBDA * SIGMA_HAT * gen.RE / (2.0 * m)
    g0_disc = LAMBDA * RHO0 * gen.RE / 2.0
    N_unit = psi_pred / 1.0
    N_disc = psi_pred / g0_disc
    N_review = SIGMA_HAT / (m * RHO0)

    print("V8 - geoid sign and magnitude, sigma_hat cos(m phi) sheet at Re")
    print(f"  Lambda {LAMBDA}  sigma_hat {SIGMA_HAT}  rho_0 {RHO0}  "
          f"Re {gen.RE}  m {m}  M {args.truncation}")
    print("\n  PREDICTED BEFORE THE RUN (NOTES/GATES-V8V7.md):")
    print(f"    psi cos{m} amplitude at Re   = Lambda sigma_hat Re / (2 m) "
          f"= {psi_pred:+.6e}")
    print(f"    g_0 = 1                 -> N = {N_unit:+.6e}")
    print(f"    g_0 = Lambda rho_0 Re/2 = {g0_disc:.7f} -> N = {N_disc:+.6e}")
    print(f"    review D5's sigma_hat/(m rho_0)         = {N_review:+.6e}")
    print("    SIGN: POSITIVE under the mass maximum (phi = 0).")

    for spec in args.meshes.split(","):
        dr, nazim = spec.split(":")
        for g0, label in ((1.0, "g_0 = 1"),
                          (g0_disc, "g_0 = homogeneous disc")):
            r = run(float(dr), int(nazim), args.truncation, m, g0, label)
            pred = psi_pred / g0
            print(f"\n  dr {dr} nazim {nazim}   {label} ({g0:.7f}), "
                  f"parent cells {r['parent_cells']}")
            print(f"    reduced residual {r['reduced_residual']:.3e} "
                  f"(rhs scale {r['reduced_rhs']:.3e})")
            print("    psi/N cos-amplitudes on the circle Re:")
            for k, v in r["amp"].items():
                if k == "psi":
                    rel = abs(v - psi_pred) / abs(psi_pred)
                    print(f"      {k:<16s} {v:+.8e}   predicted "
                          f"{psi_pred:+.8e}   rel {rel:.3e}")
                else:
                    rel = abs(v - pred) / abs(pred)
                    print(f"      {k:<16s} {v:+.8e}   predicted "
                          f"{pred:+.8e}   rel {rel:.3e}")
            print("    spectral purity of psi on Re: " + "  ".join(
                f"{k} {v:+.3e}" for k, v in r["purity"].items()))
            print("    geoid sampled on the circle Re:")
            for k, v in r["points"].items():
                s = f"{v:+.8e}" if isinstance(v, float) else v
                print(f"      {k:<18s} {s}")
            print(f"    rotation: {r['m3']}  dI {r['dI']}")
            print("    geoid(include_rotation) difference against "
                  f"Omega_sq m_3 r^2/g_0: rel {r['rot_accessor_rel']:.3e}")


if __name__ == "__main__":
    main()
