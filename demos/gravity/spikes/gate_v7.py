"""V7 - the rotation closure, written from road-map §3.3 and not from the code.

Every expected quantity in this file is derived in `NOTES/GATES-V8V7.md` before
the run, from §3.3 and §3.1 of the road map:

    m_3 = - dI_33 / C
    dI_33 = 2 int_mantle rho_0 (x . u) dV + oint_load sigma r^2 dS      (2-D)
    dI_33 > 0  ==>  m_3 < 0                (adding polar moment slows the spin)
    psi_rot = + Omega^2 m_3 r^2   ==>  body force +2 rho_0 Omega^2 m_3 r rhat,
                                       OUTWARD for m_3 > 0.

Why this gate is not redundant with V1.  V1 certified that the rotation blocks
transpose at the derived `theta_rot`.  The closure constant `K_3` and the
closure sign `s_3` do not appear in that statement: `K_3` sits in the 1x1
`(m_3, m_3)` diagonal, which is symmetric for any value, and a flip of `s_3` is
absorbed exactly by a flip of `theta_rot`.  So this gate reconstructs `K_3` and
`s_3` from the assembled row and compares them against `C` and `-1` computed
here.

`theta_rot` is taken from the DERIVATION, `s_3 f B_mu Omega_sq`, never from the
solver - it is the one constant that must not be read out of the object under
test, because it is the factor the row scaling could hide a sign in.

Serial.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

B_MU = 1.2769
LAMBDA = 1.1116
RHO0 = 1.0
SIGMA_HAT = 1.0e-3
LOAD_DEGREE = 2

#: `theta_rot_3 = s_3 f B_mu Omega_sq` with `s_3 = -1` and `f = 1`.  Asserted
#: from the derivation, NOT read from the solver.
THETA_ROT_DERIVED = -1.0 * 1.0 * B_MU * OMEGA_SQ_EARTH


def meshes(dr, nazim, tagf):
    path = os.path.join(HERE, f"v7_{tagf}_{dr}_{nazim}.msh")
    gen.generate(path, dr_mantle=dr, n_azimuthal=nazim)
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def build(parent, sub, truncation, *, kappa, with_load):
    """Coupled solver.  `kappa` is the axisymmetric part of the load, which is
    what makes `dI_33` nonzero; `with_load = False` gives a bare system."""
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    X = SpatialCoordinate(parent)
    sigma_p = None
    if with_load:
        sigma_p = SIGMA_HAT * (kappa + cos(LOAD_DEGREE * atan2(X[1], X[0])))
        gravity_bcs[gen.CURVE_RE] = {"interior_sigma": sigma_p}

    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=True,
        self_gravity_number=LAMBDA)
    z = Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=RHO0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)

    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    # C, the disc's polar second moment, assembled here from the definition
    # `int rho_0 r^2 dV` and handed to the solver.  The gate later recovers the
    # constant the rotation row actually uses and compares it against this.
    C = assemble(approx.density * dot(Xm, Xm) * dx_m)

    bcs = {gen.CURVE_RC: {"un": 0.0}}
    if with_load:
        sigma_m = SIGMA_HAT * (kappa + cos(LOAD_DEGREE * atan2(Xm[1], Xm[0])))
        bcs[gen.CURVE_RE] = {"normal_stress": B_MU * sigma_m}

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=1.0, bcs=bcs,
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH,
        nullspace=rigid_rotation_nullspace(Z, layout))
    return solver, z, layout, approx, C, sigma_p


def dI33_independent(parent, sub, u, rho0, sigma_p):
    """`dI_33` by direct quadrature, written from road-map §3.1.

    `2 int rho_0 (x . u) dx_mantle + oint sigma r^2 dS_Re`.  The measures are
    built here with UFL's own quadrature estimate rather than taken from the
    solver's form, so the only thing shared with the object under test is the
    mesh.
    """
    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    volume = 2.0 * assemble(rho0 * dot(Xm, u) * dx_m)

    sheet = 0.0
    if sigma_p is not None:
        Xp = SpatialCoordinate(parent)
        dS_re = Measure("dS", domain=parent)(gen.CURVE_RE)
        sheet = assemble(avg(sigma_p * dot(Xp, Xp)) * dS_re)
    return volume + sheet, volume, sheet


def rotation_row(solver, z, layout):
    """`(R_at_m3_zero, dR/dm3)` of the assembled rotation row.

    The row is affine in `m_3`, so these two numbers are the whole of it:
    `theta_rot (K_3 m_3 - s_3 dI_33)`.  Dividing by the DERIVED `theta_rot`
    recovers `K_3` and `-s_3 dI_33` separately, which is what pins A4's
    constant and A4's sign.
    """
    slot = layout.rotation_slots()[2]
    saved = float(z.subfunctions[slot])
    z.subfunctions[slot].assign(0.0)
    R0 = float(assemble(solver.F).subfunctions[slot].dat.data_ro[0])
    z.subfunctions[slot].assign(saved)

    J = assemble(derivative(solver.F, z), mat_type="nest").petscmat
    block = J.getNestSubMatrix(slot, slot)
    dRdm = float(block.convert("dense").getDenseArray()[0, 0])
    return R0, dRdm


def gate_closure(dr, nazim, truncation, kappa):
    """(a), (b) and the K_3/s_3 reconstruction, on a real coupled solve."""
    parent, sub = meshes(dr, nazim, "load")
    solver, z, layout, approx, C, sigma_p = build(
        parent, sub, truncation, kappa=kappa, with_load=True)
    solver.solve()

    u = z.subfunctions[layout.displacement]
    m3 = solver.rotation_values()["m3"]
    dI, dI_vol, dI_sheet = dI33_independent(
        parent, sub, u, approx.density, sigma_p)

    R0, dRdm = rotation_row(solver, z, layout)
    K3 = dRdm / THETA_ROT_DERIVED
    minus_s3_dI = R0 / THETA_ROT_DERIVED

    return {
        "C_definition": C, "K3_from_row": K3,
        "K3_rel": abs(K3 - C) / abs(C),
        "dI_indep": dI, "dI_volume": dI_vol, "dI_sheet": dI_sheet,
        "dI_from_row": minus_s3_dI,
        "dI_rel": abs(minus_s3_dI - dI) / abs(dI),
        "dI_solver_form": solver.inertia_perturbation()["dI_33"],
        "m3_solved": m3, "m3_closure": -dI / C,
        "m3_rel": abs(m3 + dI / C) / abs(dI / C),
        "row_residual": float(
            assemble(solver.F).subfunctions[layout.rotation_slots()[2]]
            .dat.data_ro[0]),
        "theta_rot_derived": THETA_ROT_DERIVED,
        "u_norm": float(norm(u)),
    }


def gate_sign(dr, nazim, truncation, eps=1.0e-3):
    """(c) the controlled sign test: a purely outward frozen `u`, no load.

    `dI_33 = 2 rho_0 eps int r dV > 0` unambiguously, and the closure must
    return `m_3 < 0`.
    """
    parent, sub = meshes(dr, nazim, "sign")
    solver, z, layout, approx, C, _ = build(
        parent, sub, truncation, kappa=0.0, with_load=False)

    Xm = SpatialCoordinate(sub)
    rhat = Xm / sqrt(dot(Xm, Xm))
    u = Function(z.function_space()[layout.displacement]).interpolate(eps * rhat)
    z.subfunctions[layout.displacement].assign(u)

    dI, _, _ = dI33_independent(parent, sub, u, approx.density, None)
    R0, dRdm = rotation_row(solver, z, layout)
    m3_root = -R0 / dRdm

    # The hand prediction: 2 rho_0 eps int r dV over the mantle.
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    dI_hand = 2.0 * eps * RHO0 * assemble(sqrt(dot(Xm, Xm)) * dx_m)

    return {
        "dI_indep": dI, "dI_hand": dI_hand,
        "dI_rel": abs(dI - dI_hand) / abs(dI_hand),
        "C": C, "m3_root": m3_root, "m3_expected": -dI / C,
        "m3_rel": abs(m3_root + dI / C) / abs(dI / C),
    }


def gate_body_force(dr, nazim, truncation, m3=1.0e-3):
    """(d) the `psi_rot` body force: outward for `m_3 > 0`.

    At `u = 0`, `psi = 0`, no load, the displacement block of the production
    residual is the rotational body force alone.  Paired with an outward radial
    test field it must equal `-f B_mu 2 Omega_sq m_3 int rho_0 (x . w) dx_m`,
    which is negative - and a negative residual pairing against an outward `w`
    is an OUTWARD force, because `gadopt.momentum_equation` writes every term as
    if on the left-hand side.
    """
    parent, sub = meshes(dr, nazim, "force")
    solver, z, layout, approx, C, _ = build(
        parent, sub, truncation, kappa=0.0, with_load=False)
    slot = layout.rotation_slots()[2]
    V = z.function_space()[layout.displacement]
    Xm = SpatialCoordinate(sub)
    w = Function(V).interpolate(Xm / sqrt(dot(Xm, Xm)))

    def pairing():
        Fu = assemble(solver.F).subfunctions[layout.displacement]
        return float(np.dot(Fu.dat.data_ro.ravel(), w.dat.data_ro.ravel()))

    at_zero = pairing()
    z.subfunctions[slot].assign(m3)
    at_m3 = pairing()

    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    predicted = -1.0 * B_MU * 2.0 * OMEGA_SQ_EARTH * m3 * assemble(
        approx.density * dot(Xm, w) * dx_m)

    # psi_rot itself, against `+Omega_sq m_3 r^2`.
    psi_rot = solver.rotational_potential_expression(sub)
    want = OMEGA_SQ_EARTH * m3 * dot(Xm, Xm)
    rel = (sqrt(assemble((psi_rot - want) ** 2 * dx_m))
           / sqrt(assemble(want ** 2 * dx_m)))

    return {
        "pairing_at_zero": at_zero, "pairing_at_m3": at_m3,
        "predicted": predicted,
        "rel": abs(at_m3 - predicted) / abs(predicted),
        "psi_rot_rel": rel, "m3": m3,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--nazim", type=int, default=64)
    p.add_argument("--kappa", type=float, default=1.0)
    args = p.parse_args()

    print("V7 - the rotation closure, from road-map §3.3")
    print(f"  DERIVED, not read from the solver: theta_rot_3 = s_3 f B_mu "
          f"Omega_sq = {THETA_ROT_DERIVED:+.12e}")
    print("  EXPECTED, stated in advance:")
    print("    m_3 = -dI_33/C ;  K_3 = C ;  s_3 = -1 ;  dI_33 > 0 => m_3 < 0 ;")
    print("    psi_rot = +Omega_sq m_3 r^2, body force OUTWARD for m_3 > 0.")

    print("\n(a)(b) closure on a coupled solve, load "
          f"sigma_hat({args.kappa} + cos 2 phi) at Re")
    r = gate_closure(args.dr, args.nazim, args.truncation, args.kappa)
    print(f"    ||u||                                {r['u_norm']:.6e}")
    print(f"    C from the definition int rho r^2    {r['C_definition']:.12e}")
    print(f"    K_3 recovered from the row           {r['K3_from_row']:.12e}"
          f"   rel {r['K3_rel']:.3e}")
    print(f"    dI_33 independent quadrature         {r['dI_indep']:+.12e}")
    print(f"      volume part                        {r['dI_volume']:+.6e}")
    print(f"      sheet part                         {r['dI_sheet']:+.6e}")
    print(f"    -s_3 dI_33 recovered from the row    {r['dI_from_row']:+.12e}"
          f"   rel {r['dI_rel']:.3e}")
    print("      (equal to +dI_33 iff s_3 = -1)")
    print(f"    dI_33 from solver.inertia_form       "
          f"{r['dI_solver_form']:+.12e}")
    print(f"    m_3 solved                           {r['m3_solved']:+.12e}")
    print(f"    m_3 = -dI_33/C predicted             {r['m3_closure']:+.12e}"
          f"   rel {r['m3_rel']:.3e}")
    print(f"    rotation row residual after solve    {r['row_residual']:.3e}")

    print("\n(c) sign, controlled: frozen outward u = eps rhat, no load")
    s = gate_sign(args.dr, args.nazim, args.truncation)
    print(f"    dI_33 independent                    {s['dI_indep']:+.6e}")
    print(f"    dI_33 by hand, 2 rho eps int r dV    {s['dI_hand']:+.6e}"
          f"   rel {s['dI_rel']:.3e}")
    print(f"    m_3 the row returns                  {s['m3_root']:+.6e}")
    print(f"    -dI_33/C                             {s['m3_expected']:+.6e}"
          f"   rel {s['m3_rel']:.3e}")
    print(f"    SIGN: dI_33 > 0 and m_3 < 0 ?        "
          f"{s['dI_indep'] > 0 and s['m3_root'] < 0}")

    print("\n(d) the psi_rot body force")
    b = gate_body_force(args.dr, args.nazim, args.truncation)
    print(f"    psi_rot vs +Omega_sq m_3 r^2         rel {b['psi_rot_rel']:.3e}")
    print(f"    <F_u, rhat> at m_3 = 0               {b['pairing_at_zero']:+.6e}")
    print(f"    <F_u, rhat> at m_3 = {b['m3']:.0e}          "
          f"{b['pairing_at_m3']:+.12e}")
    print(f"    predicted -f B_mu 2 Om^2 m_3 <rho x, w>  "
          f"{b['predicted']:+.12e}   rel {b['rel']:.3e}")
    print(f"    NEGATIVE residual pairing with an outward w => force OUTWARD: "
          f"{b['pairing_at_m3'] < 0}")


if __name__ == "__main__":
    main()
