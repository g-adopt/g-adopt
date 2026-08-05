"""FC - the fluid core at the CMB, its four gates, each number stated first.

    !!! STALE AS OF 2026-08-05 - THE CENTRAL PREMISE IS INVERTED, NEEDS REWORK.
    The CMB buoyancy spring was corrected from the contrast `(rho_core - rho_0)`
    to `rho_core` alone (see `FluidCore` docstring and
    `scratchpad/cmb_prestress_check.py`): the prestress VOLUME term already
    supplies the mantle half `-0.5 rho_0 g u_r^2` at the CMB, measured to
    2.9e-15. This gate isolates the `fluid_core_energy` block (`derivative(fc)`)
    WITHOUT the volume term, so with the fix that isolated block now equals
    `B_mu rho_core g = 3.249855` (STIFFNESS_CORE_ONLY), not `1.744945`. Two
    assertions below therefore now test the OLD, wrong coefficient and FAIL:
      - FC-1 (3) magnitude, which expects 1.744945; and
      - FC-1 (1) contrast-zero "transparency" (rho_core=rho_0 => block vanishes),
        which was only true for the contrast - with `rho_core` the block is
        transparent only in COMBINATION with the volume term.
    The physical net stiffness 1.744945 is real but is `volume(-rho_0) +
    spring(rho_core)`; a correct rework must assemble the NET (u,u) CMB block
    (prestress volume term + fluid_core_energy) and assert 1.744945 and net
    transparency on THAT. The transpose/sign/nullspace gates are unaffected.
    Until reworked, this gate enforces the bug it was written to protect.

`gadopt.gia_gravity.FluidCore` replaces the legacy `un = 0` with a real core:
one energy on the CMB facet,

    E = c B_mu int_Rc [ rho_core (u.n) psi
                        + 0.5 rho_core g_0 (u.n)^2 ] ds

whose variation supplies the traction, the mass sheet and the interface
buoyancy at once. The gates below are written against the pre-implementation
review `NOTES/REVIEW-SPADA-A4-PRECOMMIT.md`, which derived the condition
independently and measured several of these numbers *before* any of this
existed. Every expectation here is that document's, not this run's.

## What each gate is for, and what it is blind to

**FC-1, the transparent interface** (`--fc1`). Set `rho_core = rho_0` and the
physics must vanish. The naive single assertion is worthless: anything
proportional to the contrast vanishes when the contrast is zero, *whatever
multiplies it*, so a flipped sign passes and a missing `B_mu` passes. Three
assertions instead:

  1. the cancellation at **1e-15** - it is algebraic, so "small" is not enough,
     and it is asserted on the assembled block and on the net CMB sheet, the
     latter being the half the divergence-form source supplies;
  2. **positive-definiteness** of `u^T A_CMB u` for radial `u`, the only cheap
     unambiguous sign test - a sign error passes FC-1's cancellation perfectly
     and shows up dynamically only as an anti-restoring CMB;
  3. the **magnitude, 1.744945**, the only thing that catches a missing `B_mu`
     (which would read 1.115653) or the core-side-only expression the design
     documents first specified (which would read 3.249855, high by x1.8624).

Dimensionally that stiffness is `(10750 - 4978) x 10.457 = 60357.8 Pa/m`,
stabilising, and 2.0249 times the surface contrast's 29808.2.

**FC-2, the transpose** (`--fc2`), and it is **not free**. A transpose test on
a term that assembles to zero passes perfectly, and an empty measure produces
exactly that with nothing but a `WARNING Subdomain is empty`. So the transpose
is paired with a magnitude check on the same block: the CMB sheet against
`u = rhat` must carry `rho_core x 2 pi Rc`. The gate also runs the known trap
as a control - the traction written with `dot(u, rhat)` instead of `dot(u, n)`.
The review predicted an asymmetry of 5.29e-05 there; **in the gravitational
pair it is 2.0**, because at an *inner* boundary `rhat = -n` and the mistake is
a sign flip rather than an angle. The 5.29e-05 is the same mistake in the (u,u)
buoyancy block, where the sign cancels between the two factors and only the
angle survives - FC-4's last row, and it is smaller still on a P2-curved mesh.

**FC-4, the asymmetry that `un = 0` costs** (`--fc4`). Measured in the review:
`un = 0` gives 6.97e-03 on the mechanics (u,u) block while no bc, a constant
`normal_stress` and `normal_stress` proportional to `dot(u,n)` all give
1.39e-16. With the fluid core the quarantined number should **vanish, not
shrink**.

**FC-NS, the nullspace** (`--ns`). Rotations only, for the whole of A4. The
fluid core removes one of the two mechanisms that keep a translation out of the
kernel; the other is `grad_phi = g*upward_normal(mesh)`, anchored to the mesh
origin, which does not translate with the body and therefore costs energy at
the percent level whatever the CMB does. Until Phi_0 is computed from the
reference density - `gravity()` calls that "a deliberate omission" - a
translation is not a kernel mode. The gate does not infer this: it assembles
the operator action on each candidate generator, on two meshes. The review's
threshold of `||A c|| / (||A|| ||c||) <= 1e-13` was measured on the *volume
term alone* and no mesh meets it for the whole discrete coupled operator, which
annihilates the rotation only to facet-geometry error - the reason
`rigid_rotation_nullspace` exists, and its docstring records ~2e-06. So the
gate is written on the statement that separates the two: the rotation's
residual is seven orders below the translation's and falls by a factor of
thirty under one refinement, while the translation's does not move.

FC-3, the A/B of the two CMB treatments on the benchmark, is a Phase D run and
is not here.

Serial only: the block comparisons need a global dense transpose.

    PYTHONPATH=<worktree> python demos/gravity/spikes/gate_fluidcore.py --all
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see the demo's note
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import FluidCore  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

# --- The Spada M3-L70-V01 numbers, non-dimensionalised as road map §2.2 does.
# rho_bar = 5511.68 kg/m^3 (the model's own mean density) and g_bar = 9.81555
# m/s^2, both as handoff §2.2 documents them. The trailing digit of `g_bar`
# matters: dropping it moves the CMB stiffness by 5e-06, which is exactly the
# size of the discrepancy this file first mis-attributed to the review's
# rounding.
RHO_BAR = 5511.68
G_BAR = 9.81555
B_MU = 1.564037           # rho_bar g_bar D / mu_bar
LAMBDA = 1.1116           # unchanged from the prototype; FC does not read it
RHO_CORE = 10750.0 / RHO_BAR       # 1.950411
RHO_MANTLE = 4978.0 / RHO_BAR      # 0.903176
G_CMB = 10.457 / G_BAR             # 1.065349

#: `B_mu (rho_core - rho_0) g_0(Rc)`, the non-dimensional CMB stiffness, and it
#: reproduces the review's **1.744945** to 3e-07 with the documented constants.
#: It did not at first: this file carried `g_bar = 9.8155`, a digit short of
#: handoff §2.2's 9.81555, and the resulting 5e-06 gap was written up here as
#: the review's own rounding. It was not - it was this file's constant. Both
#: numbers are still asserted, the arithmetic to 1e-12 and the review's printed
#: value to 1e-4, so a coefficient error and a drift in the model constants
#: remain separately detectable.
STIFFNESS = B_MU * (RHO_CORE - RHO_MANTLE) * G_CMB          # 1.744945
STIFFNESS_REVIEW = 1.744945
#: What the core-side-only expression would give, x1.8624 too large.
STIFFNESS_CORE_ONLY = B_MU * RHO_CORE * G_CMB               # 3.249855 (x1.8624)
#: What omitting B_mu would give.
STIFFNESS_NO_B_MU = (RHO_CORE - RHO_MANTLE) * G_CMB         # 1.115668

RC = 1.2037               # the CMB radius, in units of D
SIGMA_HAT = 1.0e-3


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def build_meshes(dr, nazim, tag):
    path = os.path.join(HERE, f"fc_{tag}_{dr}_{nazim}.msh")
    if COMM_WORLD.rank == 0:
        gen.generate(path, dr_mantle=dr, n_azimuthal=nazim)
    COMM_WORLD.barrier()
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def approximation(density=RHO_MANTLE, g=G_CMB, B_mu=B_MU):
    """Fresh every time: the constructor mutates `mu` with the time step."""
    return CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=density, shear_modulus=1.0, viscosity=1.0,
        g=g, B_mu=B_mu, self_gravity_number=LAMBDA)


def build_solver(parent, sub, *, rho_core, dt=1.0, truncation=3,
                 rotation=False, **kwargs):
    """The coupled solver with a fluid core at Rc and a load sheet at Re."""
    X = SpatialCoordinate(parent)
    sigma = SIGMA_HAT * cos(2 * atan2(X[1], X[0]))
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_RE: {"interior_sigma": sigma},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        self_gravity_number=LAMBDA)
    z = Function(Z)

    approx = approximation()
    Xm = SpatialCoordinate(sub)
    sigma_m = SIGMA_HAT * cos(2 * atan2(Xm[1], Xm[0]))
    bcs = {gen.CURVE_RE: {"normal_stress": B_MU * sigma_m}}

    moments = {}
    if rotation:
        dx_m = Measure("dx", domain=sub,
                       intersect_measures=(Measure("dx", domain=parent),))
        moments["C"] = assemble(approx.density * dot(Xm, Xm) * dx_m)

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=dt, bcs=bcs, rotation_moments=moments,
        fluid_core=FluidCore(boundary=gen.CURVE_RC, rho_core=rho_core),
        **kwargs)
    return solver, z, layout


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
def nest_blocks(form, z):
    """`A.getNestSubMatrix(i, j)` as dense arrays, `None` where absent.

    Serial: a global dense transpose per block. `mat_type="nest"` is the route
    that assembles at all with `Real` blocks present, and it names the block,
    which for a sign hunt is worth more than one scalar.
    """
    Z = z.function_space()
    A = assemble(derivative(form, z), mat_type="nest").petscmat
    n = len(Z)
    out = {}
    for i in range(n):
        for j in range(n):
            M = A.getNestSubMatrix(i, j)
            out[i, j] = None if M is None else M.convert(
                "dense").getDenseArray().copy()
    return out


def contract(cofunction, function):
    """`<cofunction, function>`, summed over the mixed blocks. Serial."""
    return sum(
        float(np.dot(np.asarray(c.dat.data_ro).ravel(),
                     np.asarray(f.dat.data_ro).ravel()))
        for c, f in zip(cofunction.subfunctions, function.subfunctions))


def radial_state(Z, layout, mesh):
    """`u = rhat`, everything else zero: the probe every FC gate contracts with."""
    c = Function(Z)
    X = SpatialCoordinate(mesh)
    c.subfunctions[layout.displacement].interpolate(X / sqrt(dot(X, X)))
    return c


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# FC-1
# ---------------------------------------------------------------------------
def gate_fc1(dr, nazim):
    """FC-1: the transparent interface, in three assertions.

    Built twice - once with `rho_core = rho_0`, once with the model's real
    core - because two of the three assertions need the contrast and the first
    needs it gone.
    """
    print("\n" + "=" * 78)
    print("FC-1  transparent interface: cancellation, sign, magnitude")
    print("=" * 78)
    print("Expected, before the run:")
    print("  contrast-zero CMB (u,u) block            <= 1e-15 relative")
    print("  contrast-zero net CMB sheet              <= 1e-15 relative")
    print("  u^T A_CMB u for radial u                 > 0  (stabilising)")
    print(f"  B_mu (rho_c - rho_0) g_0(Rc)             {STIFFNESS:.6f}")
    print(f"    the core-only expression would read    "
          f"{STIFFNESS_CORE_ONLY:.6f}  (x{STIFFNESS_CORE_ONLY / STIFFNESS:.4f})")
    print(f"    omitting B_mu would read               "
          f"{STIFFNESS_NO_B_MU:.6f}  (x{STIFFNESS_NO_B_MU / STIFFNESS:.4f})")

    parent, sub = build_meshes(dr, nazim, "fc1")
    results = {}

    for label, rho_core in (("transparent", RHO_MANTLE), ("model", RHO_CORE)):
        solver, z, layout = build_solver(parent, sub, rho_core=rho_core)
        Z = z.function_space()
        fc = solver.fluid_core_residual()
        blocks = nest_blocks(fc, z)
        iu, ip = layout.displacement, layout.potential

        c = radial_state(Z, layout, sub)
        quad = contract(assemble(action(derivative(fc, z), c)), c)

        n = FacetNormal(sub)
        dss = solver.fluid_core_measure()(gen.CURVE_RC)
        # The normalisation that removes the geometry entirely: for constant
        # coefficients `u^T A u / int (u.n)^2` IS `B_mu (rho_c - rho_0) g_0`,
        # exactly. It has to be the *same* discrete `u` in both, not the
        # analytic `rhat` - the CG2 interpolant of `rhat` is not radial between
        # its nodes, and using the analytic one instead leaves 3e-06 of
        # interpolation error masquerading as a coefficient error.
        u_h = c.subfunctions[layout.displacement]
        weight = assemble(dot(u_h, n) ** 2 * dss)
        area = assemble(Constant(1.0) * dss)

        results[label] = {
            "uu": np.abs(blocks[iu, iu]).max(),
            "upsi": np.abs(blocks[iu, ip]).max(),
            "quad": quad,
            "stiffness": quad / weight,
            "per_area": quad / area,
            "weight": weight, "area": area,
            "sheet": cmb_sheet_residual(solver, z, layout, c),
        }

    t, m = results["transparent"], results["model"]

    print("\nMeasured:")
    print(f"  CMB facet measure                        {m['area']:.8f}"
          f"   (2 pi Rc = {2 * np.pi * RC:.8f})")
    print(f"  int (rhat.n)^2 ds                        {m['weight']:.8f}")
    print("\n  (1) cancellation at rho_core = rho_0")
    rel_uu = t["uu"] / m["uu"]
    rel_sheet = t["sheet"]["relative"]
    print(f"      max|A_CMB(u,u)|  transparent         {t['uu']:.6e}")
    print(f"                       model               {m['uu']:.6e}")
    print(f"                       relative            {rel_uu:.3e}")
    print(f"      net CMB sheet    |auto + core|       "
          f"{t['sheet']['net']:.6e}")
    print(f"                       |core|              "
          f"{t['sheet']['core']:.6e}")
    print(f"                       relative            {rel_sheet:.3e}")
    print(f"      divergence identity, checked         "
          f"{t['sheet']['identity']:.3e}  (the auto sheet is derived from the "
          "shipped source form, not asserted)")
    ok1 = rel_uu <= 1e-15 and rel_sheet <= 1e-15
    print(f"      {verdict(ok1)}")

    print("\n  (2) sign: u^T A_CMB u for u = rhat")
    print(f"      quadratic form                       {m['quad']:.6e}")
    ok2 = m["quad"] > 0.0
    print(f"      {verdict(ok2)}  "
          f"({'stabilising' if ok2 else 'ANTI-RESTORING - the dot(u, n) sign'})")

    print("\n  (3) magnitude")
    print(f"      u^T A u / int (rhat.n)^2             {m['stiffness']:.6f}")
    print(f"      expected                             {STIFFNESS:.6f}")
    rel_mag = abs(m["stiffness"] - STIFFNESS) / STIFFNESS
    print(f"      relative                             {rel_mag:.3e}")
    rel_rev = abs(m["stiffness"] - STIFFNESS_REVIEW) / STIFFNESS_REVIEW
    print(f"      against the review's printed         {STIFFNESS_REVIEW:.6f}"
          f"   relative {rel_rev:.3e}")
    print(f"      u^T A u / area  (facet geometry in)  {m['per_area']:.6f}")
    ok3 = rel_mag <= 1e-12 and rel_rev <= 1e-4
    print(f"      {verdict(ok3)}")

    ok = ok1 and ok2 and ok3
    print(f"\nFC-1 {verdict(ok)}")
    return ok


def cmb_sheet_residual(solver, z, layout, probe):
    """The net Rc mass sheet: the fluid core's, plus the one the source makes.

    The divergence form carries its own sheet. Integrating the shipped source
    `-Lambda int_m rho_0 u.grad(v) dx` by parts leaves
    `-Lambda oint rho_0 (u.n) v ds` on the submesh boundary, so at Rc the
    potential row already holds `theta_psi (-Lambda) int_Rc rho_0 (u.n) v`,
    which is the mantle vacating the shell a rising CMB sweeps out. The fluid
    core adds the core's half. Their sum is what has to vanish when the two
    densities are equal, and it is the *physical* content of FC-1's first
    assertion - the assembled-block half of it only tests a coefficient that is
    literally zero.

    The identity is verified rather than trusted: `identity` is the relative
    difference between the shipped volume form and
    `oint rho_0 (u.n) v ds - int div(rho_0 u) v dx` over the whole submesh
    boundary.
    """
    Z = z.function_space()
    u = probe.subfunctions[layout.displacement]
    v = TestFunctions(Z)[layout.potential]
    sub = layout.mechanics_mesh
    rho0 = solver.approximation.density
    n = FacetNormal(sub)

    dss_all = solver.fluid_core_measure()
    dss = dss_all(gen.CURVE_RC)

    # The shipped volume source, and its integration by parts. `avg(v)` on the
    # facet integrals: the parent's field is two-valued on a facet interior to
    # the parent, and the measure is the mantle's own boundary intersected with
    # the parent's `dS` - the facet-to-facet pairing, without which the parent
    # field is evaluated at the wrong points and this identity reads ~1.0.
    volume = solver.Lambda * rho0 * dot(u, grad(v)) * solver.dx_m
    parts = (solver.Lambda * rho0 * dot(u, n) * avg(v) * dss_all
             - solver.Lambda * div(rho0 * u) * v * solver.dx_m)
    a = np.abs(np.asarray(
        assemble(volume).subfunctions[layout.potential].dat.data_ro))
    b = np.abs(np.asarray(
        assemble(parts).subfunctions[layout.potential].dat.data_ro))
    identity = np.abs(a - b).max() / max(a.max(), 1e-300)

    auto = assemble(
        solver.theta_psi * (-solver.Lambda) * rho0 * dot(u, n) * avg(v) * dss)
    core = assemble(action(derivative(solver.fluid_core_residual(), z), probe))

    a_v = np.asarray(auto.subfunctions[layout.potential].dat.data_ro)
    c_v = np.asarray(core.subfunctions[layout.potential].dat.data_ro)
    net = np.abs(a_v + c_v).max()
    return {"net": net, "core": np.abs(c_v).max(),
            "auto": np.abs(a_v).max(),
            "relative": net / max(np.abs(c_v).max(), 1e-300),
            "identity": identity}


# ---------------------------------------------------------------------------
# FC-2
# ---------------------------------------------------------------------------
def gate_fc2(dr, nazim):
    """FC-2: the new (u, psi) block against its partner, and its magnitude."""
    print("\n" + "=" * 78)
    print("FC-2  transpose of the gravitational pair, and its size")
    print("=" * 78)
    mass = RHO_CORE * 2 * np.pi * RC
    print("Expected, before the run:")
    print("  max|A(u,psi) - A(psi,u)^T| / max|A(u,psi)|   <= 1e-15")
    print("  the (u, m3) / (m3, u) pair, same tolerance   <= 1e-14")
    print("  the same, for the whole coupled Jacobian     <= 1e-14")
    print(f"  CMB sheet mass at u = rhat, rho_core 2 pi Rc {mass:.8f}")
    print("  control: the same term written with dot(u, rhat)")
    print("           asymmetry 2.0 - at an INNER boundary rhat = -n, so in")
    print("           the gravitational pair the mistake is a sign flip and")
    print("           not an angle. The 5.29e-05 the review measured is the")
    print("           same mistake in the (u,u) buoyancy block, where the")
    print("           sign cancels and only the angle survives: FC-4's last row.")

    parent, sub = build_meshes(dr, nazim, "fc2")
    solver, z, layout = build_solver(parent, sub, rho_core=RHO_CORE,
                                     rotation=True)
    Z = z.function_space()
    iu, ip = layout.displacement, layout.potential

    fc = solver.fluid_core_residual()
    b = nest_blocks(fc, z)
    asym = np.abs(b[iu, ip] - b[ip, iu].T).max()
    size = np.abs(b[iu, ip]).max()

    full = nest_blocks(solver.F, z)
    asym_full = np.abs(full[iu, ip] - full[ip, iu].T).max()
    size_full = np.abs(full[iu, ip]).max()

    # Magnitude: the sheet the potential row carries, against u = rhat and
    # v = 1. The residual's own constants are in it, so the prediction is
    # c B_mu rho_core int (u.n) ds = -c B_mu rho_core 2 pi Rc; the minus is
    # the mantle's outward normal at its inner boundary.
    probe = radial_state(Z, layout, sub)
    sheet = assemble(action(derivative(fc, z), probe))
    got = float(np.sum(np.asarray(
        sheet.subfunctions[ip].dat.data_ro)))
    predicted = -float(solver.scaling_factor) * B_MU * mass

    print("\nMeasured:")
    print(f"  fluid core alone: max|A(u,psi)|              {size:.6e}")
    print(f"                    max|A(u,psi) - A(psi,u)^T| {asym:.6e}")
    rel = asym / size if size else float("inf")
    print(f"                    relative                   {rel:.3e}")
    print(f"  whole Jacobian:   max|A(u,psi)|              {size_full:.6e}")
    print(f"                    relative asymmetry         "
          f"{asym_full / size_full:.3e}")
    print(f"  sheet mass, measured                         {got:.8f}")
    print(f"              predicted -c B_mu rho_c 2 pi Rc  {predicted:.8f}")
    rel_mass = abs(got - predicted) / abs(predicted)
    print(f"              relative                         {rel_mass:.3e}")

    # The rotation pair, which this gate ran with rotation OFF and therefore
    # could not see. The fluid core's sheet enters `dI` and its transpose
    # partner is the centrifugal traction on the CMB; without that traction the
    # pair was asymmetric at 97 %, measured, while every other block stayed
    # exact. Rotation is on above for exactly this measurement.
    im = layout.rotation["m3"]
    asym_rot = np.abs(full[iu, im] - full[im, iu].T).max()
    size_rot = max(np.abs(full[iu, im]).max(), np.abs(full[im, iu]).max())
    print(f"  rotation pair:    max|A(u,m3)|                 "
          f"{np.abs(full[iu, im]).max():.6e}")
    print(f"                    max|A(m3,u)|                 "
          f"{np.abs(full[im, iu]).max():.6e}")
    print(f"                    relative asymmetry           "
          f"{asym_rot / size_rot:.3e}"
          "   (97 % without the centrifugal CMB traction)")

    trap = rhat_variant_asymmetry(solver, z, layout)
    print(f"  control: dot(u, rhat) instead of dot(u, n)   {trap:.3e}"
          "   (expected 2.0, a sign flip; that this is NOT 1e-16 is what "
          "gives the gate teeth)")

    ok = (rel <= 1e-15 and asym_full / size_full <= 1e-14
          and size > 0.0 and rel_mass <= 5e-3 and trap > 1.0
          and asym_rot / size_rot <= 1e-14)
    print(f"\nFC-2 {verdict(ok)}")
    return ok


def rhat_variant_asymmetry(solver, z, layout):
    """The trap, measured: the same energy with `rhat` in place of `n`.

    Written as an energy too, so it is still symmetric *as an energy*; what
    breaks is that the traction and the sheet then contract `u` and `w` with
    different vectors, and the two agree only to the angle between the facet
    normal and the analytic radial. On an uncurved annulus that is 5.3e-05, and
    it does not reach 1e-15 on a P2-curved one either.
    """
    Z = z.function_space()
    iu, ip = layout.displacement, layout.potential
    u = split(z)[iu]
    psi = split(z)[ip]
    sub = layout.mechanics_mesh
    X = SpatialCoordinate(sub)
    rhat = X / sqrt(dot(X, X))
    n = FacetNormal(sub)
    dss = solver.fluid_core_measure()(gen.CURVE_RC)
    B_mu = solver.approximation.B_mu
    # The (u, psi) row written with `rhat` and the (psi, u) row with `n`, which
    # is what an implementer gets who reaches for the codebase's own radial
    # idiom in the traction and leaves the sheet alone.
    w = TestFunctions(Z)[iu]
    v = TestFunctions(Z)[ip]
    row_u = B_mu * Constant(RHO_CORE) * dot(w, rhat) * avg(psi) * dss
    row_psi = B_mu * Constant(RHO_CORE) * dot(u, n) * avg(v) * dss
    b = nest_blocks(row_u + row_psi, z)
    return np.abs(b[iu, ip] - b[ip, iu].T).max() / np.abs(b[iu, ip]).max()


# ---------------------------------------------------------------------------
# FC-4
# ---------------------------------------------------------------------------
def mechanics_asymmetry(sub, cmb, dt=1.0):
    """`max|A(u,u) - A(u,u)^T| / max|A(u,u)|` for one CMB condition.

    A plain `CoupledInternalVariableSolver` on the mantle alone: the coupling
    is absent, so what is left is the mechanics and the condition under test.
    `cmb` is a callable taking the displacement and returning the bcs entry, so
    that a condition depending on `u` can reference the solver's own unknown.
    """
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    X = SpatialCoordinate(sub)
    bcs = {gen.CURVE_RE: {"normal_stress": B_MU * SIGMA_HAT * cos(
        2 * atan2(X[1], X[0]))}}
    entry = cmb(split(zm)[0], sub)
    if entry is not None:
        bcs[gen.CURVE_RC] = entry
    solver = CoupledInternalVariableSolver(
        zm, approximation(), dt=dt, bcs=bcs, solver_parameters="direct")
    A = assemble(derivative(solver.F, zm), mat_type="nest").petscmat
    M = A.getNestSubMatrix(0, 0).convert("dense").getDenseArray()
    return np.abs(M - M.T).max() / np.abs(M).max()


def gate_fc4(dr, nazim):
    """FC-4: the `un = 0` asymmetry vanishes rather than shrinking."""
    print("\n" + "=" * 78)
    print("FC-4  the mechanics (u,u) asymmetry, by CMB condition")
    print("=" * 78)
    print("Expected, before the run (review, measured in advance):")
    print("  **SUPERSEDED 2026-08-01: B2 fixed the cause at source.** The")
    print("  coupled solver now hands the Nitsche pair `mu0` rather than")
    print("  `effective_viscosity(dt)`, which is what this row was measuring")
    print("  and what FC-4's own diagnosis identified, so `un = 0` is now at")
    print("  roundoff too. The prediction below is what was measured BEFORE")
    print("  that fix, kept because the gate's value is the comparison and")
    print("  because a regression would show up as the old number returning.")
    print("  un = 0                                  6.97e-03  (now ~1e-16)")
    print("  no condition                            1.39e-16")
    print("  normal_stress, constant                 1.39e-16")
    print("  normal_stress ~ dot(u, n)               1.39e-16")
    print("  normal_stress ~ dot(u, rhat)            5.29e-05  (the trap, on")
    print("                                          an UNCURVED annulus; the")
    print("                                          P2-curved one shrinks it")
    print("                                          by orders and never to 0)")
    print("The mechanism is `mu = effective_viscosity(dt)` in the Nitsche "
          "terms\nagainst `mu_0` in the coupled stress, so it scales as "
          "dt/(tau+dt) - not\nthe structural mismatch `gate_g0`'s docstring "
          "used to name.")

    parent, sub = build_meshes(dr, nazim, "fc4")
    # The straight-sided submesh, for the `rhat` row alone: the review measured
    # that trap on an uncurved annulus, where the angle between the facet
    # normal and the analytic radial is the polygon error. `curve_mesh` shrinks
    # it by three orders and does not remove it, which is the point - the
    # `dot(u, n)` row is exact on any mesh and the `rhat` row is not.
    uncurved = Submesh(parent, 2, gen.CELL_MANTLE)
    uncurved.cartesian = False
    stiff = B_MU * (RHO_CORE - RHO_MANTLE) * G_CMB

    cases = {
        "un = 0": lambda u, m: {"un": 0.0},
        "no condition": lambda u, m: None,
        "normal_stress const": lambda u, m: {"normal_stress": Constant(1e-3)},
        "normal_stress dot(u,n)": lambda u, m: {
            "normal_stress": stiff * dot(u, FacetNormal(m))},
        "normal_stress dot(u,rhat)": lambda u, m: {
            "normal_stress": stiff * dot(
                u, SpatialCoordinate(m) / sqrt(dot(
                    SpatialCoordinate(m), SpatialCoordinate(m))))},
    }
    measured = {k: mechanics_asymmetry(sub, f) for k, f in cases.items()}
    measured["normal_stress dot(u,rhat), uncurved"] = mechanics_asymmetry(
        uncurved, cases["normal_stress dot(u,rhat)"])

    print("\nMeasured:")
    for k, v in measured.items():
        print(f"  {k:<38s} {v:.4e}")

    ok = (measured["un = 0"] <= 1e-14
          and measured["no condition"] <= 1e-14
          and measured["normal_stress const"] <= 1e-14
          and measured["normal_stress dot(u,n)"] <= 1e-14
          and measured["normal_stress dot(u,rhat)"] > 1e-12
          and measured["normal_stress dot(u,rhat), uncurved"] > 1e-6)
    print(f"\nFC-4 {verdict(ok)}   (every CMB condition is now symmetric to "
          "roundoff. The\n     quarantined 7e-03 is gone at SOURCE - B2's "
          "`mu0` fix - rather than\n     merely deleted by the fluid core, so "
          "this gate no longer\n     discriminates between the two treatments "
          "and is a regression test.\n     `dot(u,rhat)` still shows what the "
          "wrong vector would have cost.)")
    return ok


# ---------------------------------------------------------------------------
# FC-NS
# ---------------------------------------------------------------------------
def generator_residuals(dr, nazim, tag):
    """`||A c|| / (||A|| ||c||)` for each candidate generator, one assembly."""
    parent, sub = build_meshes(dr, nazim, tag)
    solver, z, layout = build_solver(parent, sub, rho_core=RHO_CORE,
                                     rotation=True)
    Z = z.function_space()
    J = derivative(solver.F, z)
    scale = max(np.abs(b).max() for b in nest_blocks(solver.F, z).values()
                if b is not None and b.size)

    X = SpatialCoordinate(sub)
    generators = {
        "rigid rotation (-y, x)": as_vector([-X[1], X[0]]),
        "translation e_x": as_vector([Constant(1.0), Constant(0.0)]),
        "translation e_y": as_vector([Constant(0.0), Constant(1.0)]),
    }
    out = {}
    for name, mode in generators.items():
        c = Function(Z)
        c.subfunctions[layout.displacement].interpolate(mode)
        norm_c = max(np.abs(np.asarray(s.dat.data_ro)).max()
                     for s in c.subfunctions)
        act = assemble(action(J, c))
        norm_a = max(np.abs(np.asarray(s.dat.data_ro)).max()
                     for s in act.subfunctions)
        out[name] = norm_a / (scale * norm_c)
    return out


def gate_nullspace(dr, nazim):
    """FC-NS: which rigid modes the fluid-core operator actually annihilates.

    **The threshold is not the review's 1e-13, and the difference is not a
    defect.** The review measured `a(rot, w)/scale = 8.3e-13` on the *volume
    term alone*; the whole discrete coupled operator annihilates the rotation
    only to facet-geometry error, because `u . n_h` is not exactly zero on a
    piecewise-quadratic approximation to a circle. That is the reason
    `rigid_rotation_nullspace` exists and its docstring records the number
    (~2e-06 on the development annulus). A fixed 1e-13 on this operator is a
    threshold no mesh meets.

    So the gate is written on the statement that actually distinguishes the two
    modes: **the rotation residual falls with refinement and the translation
    residual does not.** A continuum kernel mode's discrete residual is a
    geometry error and converges; a mode that costs real energy has a residual
    that is a physical number and stays put.
    """
    print("\n" + "=" * 78)
    print("FC-NS  candidate kernel generators, tested before being declared")
    print("=" * 78)
    print("Expected, before the run:")
    print("  rigid rotation    ||A c|| / (||A|| ||c||)   small, and falling")
    print("                    fast with refinement - a geometry error")
    print("  rigid translation                           ~1e-02, seven orders")
    print("                    larger, and falling only as the norm's own O(h)")
    print("The fluid core removes the free-slip mechanism that excluded a "
          "translation,\nbut not the second one: `grad_phi = g*upward_normal"
          "(mesh)` is anchored to the\nmesh origin and does not translate "
          "with the body. Declaring translations is\nwrong for the whole of "
          "A4, until Phi_0 is computed from the reference density.")

    coarse = generator_residuals(dr, nazim, "ns")
    fine = generator_residuals(dr / 2, 2 * nazim, "nsf")

    print(f"\nMeasured:   {'coarse':>14s}{'refined':>14s}{'ratio':>12s}")
    for name in coarse:
        r = fine[name] / coarse[name]
        print(f"  {name:<24s}{coarse[name]:14.3e}{fine[name]:14.3e}{r:12.3f}")

    rot = "rigid rotation (-y, x)"
    trans = ("translation e_x", "translation e_y")
    separation = max(coarse[rot] / coarse[t] for t in trans)
    # The rotation's residual falls by a factor of thirty under one refinement
    # while the translation's falls by two - and that two is the inf-norm's own
    # O(h), not convergence. The separation is the statement that matters.
    ok = (coarse[rot] <= 1e-5
          and fine[rot] < 0.2 * coarse[rot]
          and all(coarse[t] > 1e-4 and fine[t] > 1e-4 for t in trans)
          and separation <= 1e-5)
    print(f"\n  rotation / translation                     {separation:.2e}")
    print(f"  rotation:    kernel mode, DECLARE  (converging, "
          f"x{fine[rot] / coarse[rot]:.3f} under one refinement)")
    print(f"  translation: NOT a kernel mode, DO NOT DECLARE  "
          f"({coarse['translation e_x']:.2e}, and its x"
          f"{fine['translation e_x'] / coarse['translation e_x']:.2f} is the "
          "inf-norm's\n               own O(h) rather than convergence)")
    print(f"\nFC-NS {verdict(ok)}   (rotations only, which is what "
          "`rigid_rotation_nullspace`\n      declares and all it declares)")
    return ok


# ---------------------------------------------------------------------------
# FC-5
# ---------------------------------------------------------------------------
def gate_fc5(dr, nazim):
    """FC-5: is the CMB term azimuthally isotropic on a discrete mesh?

    **The 2-D analogue of B4's phase instrument, isolated to this term.** B4
    measures the polar-motion phase against a value that is purely geometric -
    load longitude plus 180 degrees, no rheology, no compliance, no `C - A`, no
    time dependence - and therefore known exactly. A phase error that appears
    only when the fluid core is switched on has to come from something in the
    fluid core that is not azimuthally symmetric, because the *coefficients*
    certainly are: `rho_core`, `g_0(Rc)` and `B_mu` are all 1-D.

    What is not guaranteed symmetric is the **discrete CMB surface**. The term
    is a facet integral over a piecewise-quadratic approximation to a circle,
    and any azimuthal anisotropy in that surface - an uneven facet
    distribution, a tangled cell, a quadrature rule that under-resolves the
    pattern - enters here and does not enter a `un = 0` run in the same way.

    The instrument: restrict the assembled CMB `(u, u)` block to the
    two-dimensional space spanned by `u_c = cos(2 phi) rhat` and
    `u_s = sin(2 phi) rhat`, which is one degree-2 pattern and its quadrature.
    An isotropic operator is a multiple of the identity there, so

        dphi = atan2(u_s^T A u_c, u_c^T A u_c)

    is the phase the response comes out at when the input is at phase zero,
    which is the same question B4 asks. Exactly zero for a rotationally
    symmetric operator; whatever the mesh leaves, otherwise.

    **Not the principal-axis rotation**, which was the first spelling and is
    the wrong instrument: `0.5 atan2(2 cs, cc - ss)` is degenerate exactly when
    the operator is isotropic, since a multiple of the identity has no
    principal axes. It read 47 degrees on a block whose anisotropy `|cc - ss| /
    |cc|` was 8.8e-14, i.e. it was reporting the direction of roundoff.

    **This gate cannot exonerate the 3-D term** - the 3-D CMB surface is a
    different discretisation with its own anisotropy, and 2-D has no tesseral
    pair at all. What it can do is say whether the *form* carries an intrinsic
    phase error, which would show up here too.
    """
    print("\n" + "=" * 78)
    print("FC-5  azimuthal isotropy of the CMB block: the phase instrument")
    print("=" * 78)
    print("Expected, before the run:")
    print("  principal-axis rotation of the CMB (u,u) block, restricted to")
    print("  the degree-2 pair, is 0 for an azimuthally isotropic operator.")
    print("  Anything it leaves is the discrete CMB surface, not the form,")
    print("  and it should FALL under refinement if that is what it is.")

    rows = {}
    for d, n in ((dr, nazim), (dr / 2, 2 * nazim)):
        parent, sub = build_meshes(d, n, "fc5")
        solver, z, layout = build_solver(parent, sub, rho_core=RHO_CORE)
        Z = z.function_space()
        iu = layout.displacement
        J = derivative(solver.fluid_core_residual(), z)

        X = SpatialCoordinate(sub)
        rhat = X / sqrt(dot(X, X))
        phi = atan2(X[1], X[0])
        probes = {}
        for name, pattern in (("c", cos(2 * phi) * rhat),
                              ("s", sin(2 * phi) * rhat)):
            f = Function(Z)
            f.subfunctions[iu].interpolate(pattern)
            probes[name] = f
        acts = {k: assemble(action(J, v)) for k, v in probes.items()}

        def q(a, b):
            return contract(acts[a], probes[b])

        cc, ss, cs = q("c", "c"), q("s", "s"), q("c", "s")
        dphi = np.degrees(np.arctan2(cs, cc))
        rows[d] = {"cc": cc, "ss": ss, "cs": cs, "dphi": dphi,
                   "anisotropy": abs(cc - ss) / abs(cc)}

    print(f"\nMeasured:   {'coarse':>16s}{'refined':>16s}")
    keys = list(rows)
    for label, key in (("u_c^T A u_c", "cc"), ("u_s^T A u_s", "ss"),
                       ("u_c^T A u_s", "cs"),
                       ("|cc - ss| / |cc|", "anisotropy"),
                       ("response phase, deg", "dphi")):
        print(f"  {label:<22s}{rows[keys[0]][key]:16.6e}"
              f"{rows[keys[1]][key]:16.6e}")

    coarse, fine = rows[keys[0]], rows[keys[1]]
    ok = abs(coarse["dphi"]) < 1e-6 and abs(fine["dphi"]) < 1e-6
    print(f"\nFC-5 {verdict(ok)}   (a nonzero phase here would be an intrinsic")
    print("      property of the form; zero here leaves the 3-D mesh and the")
    print("      3-D surface as the remaining candidates, which this gate")
    print("      cannot reach)")
    return ok


# ---------------------------------------------------------------------------
# FC-6
# ---------------------------------------------------------------------------
def gate_fc6(dr, nazim):
    r"""FC-6: the sheet's coefficient is `rho_core`, pinned by measurement.

    **This exists because of a coincidence, not because of a bug.** The CMB
    sheet is `sigma = rho_core (u.rhat)` and carries the core's density alone -
    the mantle side arrives from the divergence-form volume source - which is
    the one genuinely counter-intuitive thing about the whole condition. B4
    measured the sheet at **17.6 % of dI**, uniform to four digits across all
    three components, where an earlier estimate had put it near 2 %.

    At 2 % a coefficient error was invisible. At 17.6 % it is not, and the
    obvious wrong version sits exactly where it will be mistaken for a lead:
    writing the sheet with the **contrast** `(rho_core - rho_0)` instead scales
    it by 0.5369 for these constants and moves `dI` by about 8 %, which is the
    same size as the residual deficit in |m| that B4 is still carrying. Someone
    chasing that residual will find this, and it is not the cause.

    So the coefficient is pinned by a check rather than by a reading of the
    source, and the gate asserts **both** that the right coefficient matches and
    that the wrong one is rejected - the second half being what gives the first
    any force.

    The reference is computed by a deliberately different route: on the
    **parent's** `dS(Rc)` with `avg`, from a spatial expression, so it shares
    neither the mesh, nor the measure, nor the restriction with the term under
    test. `u = X` is used because it lies in CG2 exactly, so the two routes
    differ only by quadrature and facet geometry rather than by interpolation.
    Then `u.rhat = Rc` and `p_3 = Rc^2` on the circle, so the closed form is
    `rho_core 2 pi Rc^4` and the sign is pinned too - positive, because
    `sigma = -rho_core dot(u, n)` and `n = -rhat` at an inner boundary.

    Rotation is **on**, per the finding that a gate which disables a coupling
    proves nothing about it.
    """
    print("\n" + "=" * 78)
    print("FC-6  the CMB sheet's coefficient: rho_core, not the contrast")
    print("=" * 78)
    closed = RHO_CORE * 2 * np.pi * RC ** 4
    contrast = (RHO_CORE - RHO_MANTLE) / RHO_CORE
    print("Expected, before the run:")
    print(f"  int sigma p_3 dS with u = X, closed form   {closed:.8f}")
    print("      = rho_core 2 pi Rc^4, positive")
    print(f"  the contrast version would give            "
          f"{closed * contrast:.8f}")
    print(f"      a factor {contrast:.4f}, i.e. dI low by "
          f"{100 * (1 - contrast) * 0.176:.1f} % at a 17.6 % sheet - the same")
    print("      size as the residual B4 is chasing, and NOT its cause")

    parent, sub = build_meshes(dr, nazim, "fc6")
    solver, z, layout = build_solver(parent, sub, rho_core=RHO_CORE,
                                     rotation=True)
    Xm = SpatialCoordinate(sub)
    z.subfunctions[layout.displacement].interpolate(Xm)

    measured = assemble(solver.fluid_core_sheet_integral(
        solver.inertia_polynomial(2, Xm)))

    # The independent route: parent mesh, parent dS, `avg`, spatial expression.
    Xp = SpatialCoordinate(parent)
    rhat = Xp / sqrt(dot(Xp, Xp))
    independent = assemble(avg(
        Constant(RHO_CORE) * dot(Xp, rhat) * (Xp[0] ** 2 + Xp[1] ** 2))
        * solver.form.dS(gen.CURVE_RC))

    print("\nMeasured:")
    print(f"  fluid_core_sheet_integral(p_3)             {measured:.8f}")
    print(f"  independent, parent dS + avg               {independent:.8f}")
    print(f"  closed form                                {closed:.8f}")
    rel_ind = abs(measured - independent) / abs(independent)
    rel_closed = abs(measured - closed) / abs(closed)
    print(f"  relative to the independent route          {rel_ind:.3e}")
    print(f"  relative to the closed form                {rel_closed:.3e}")
    ratio = measured / closed
    print(f"  measured / closed                          {ratio:.6f}"
          f"   (contrast version would give {contrast:.6f})")

    ok = (measured > 0.0 and rel_ind <= 1e-3 and rel_closed <= 1e-3
          and abs(ratio - contrast) > 0.4)
    print(f"\nFC-6 {verdict(ok)}   (and the contrast version is rejected at "
          f"{abs(ratio - contrast):.3f},\n      which is what makes the "
          "agreement above mean something)")
    return ok


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------
def gate_switch(dr, nazim):
    """The legacy `un = 0` still solves, and a stiff core reproduces it.

    Two things at once. The switch has to keep working, or FC-3 - the A/B that
    turns road map risk #5 from an inherited approximation into a number - is
    not possible. And the *limit* has to be right: a core so dense that the
    interface stiffness dwarfs everything else is a rigid core, so raising
    `rho_core` must walk the fluid-core answer back onto the `un = 0` one.
    That is the only end-to-end statement available at this stage, and unlike a
    block comparison it exercises the solve.

    **The prototype's own configuration flips the sign of the deflection when
    the CMB is freed, and that is not the fluid core's doing.** With
    `rho_core = rho_0` - a transparent interface, contributing nothing - the
    deflection already flips, because this configuration's reference state is a
    *uniform* density in a *constant* gravity field, which is not a hydrostatic
    equilibrium, and the operator linearised about it has a growing mode that
    `un = 0` was holding down. The driver's `build_solver` docstring records the
    same phenomenon in time: the production configuration has no fluid limit,
    and self-gravity is not why. Do not read the flip as a sign error in the
    new terms; FC-1's positive-definiteness assertion is what settles that, and
    the `rho_core = 10` row below is what shows the limit is continuous.
    """
    print("\n" + "=" * 78)
    print("FC-SWITCH  the rigid core still works, and a stiff core recovers it")
    print("=" * 78)
    print("Expected: `un = 0` still solves; a transparent core flips the sign "
          "(the\nprototype's non-hydrostatic reference state, not the new "
          "terms); and a core\nstiff enough returns to within ~10 % of the "
          "rigid answer.")

    sys.path.insert(0, DEMOS)
    import selfgrav_gia_annulus as drv

    parent, sub = drv.build_meshes(dr, nazim, path=os.path.join(
        HERE, f"fc_switch_{dr}_{nazim}.msh"))

    def run(**kwargs):
        solver, _, _, _, _ = drv.build_solver(
            parent, sub, dt=1.0, truncation=3, **kwargs)
        solver.solve()
        return drv.deflection_amplitude(solver)

    def core(rho_core):
        return {"fluid_core": FluidCore(boundary=gen.CURVE_RC,
                                        rho_core=rho_core)}

    zr = run()
    z1 = run(**core(1.0))
    z2 = run(**core(2.0))
    z10 = run(**core(10.0))

    print("\nMeasured (one backward-Euler step, prototype constants, "
          "rho_0 = g = 1):")
    print(f"  un = 0                deflection    {zr: .8e}")
    print(f"  fluid core rho_c = 1  deflection    {z1: .8e}   "
          "(transparent: no contrast, no net sheet)")
    print(f"  fluid core rho_c = 2  deflection    {z2: .8e}")
    print(f"  fluid core rho_c = 10 deflection    {z10: .8e}   "
          f"relative to un = 0  {abs(z10 - zr) / abs(zr):.3e}")
    ok = (np.isfinite(zr) and np.isfinite(z1) and np.isfinite(z10)
          and abs(z10 - zr) / abs(zr) < 0.1)
    print(f"\nFC-SWITCH {verdict(ok)}   (the stiff-core limit is the rigid "
          "core; the transparent\n           core's sign flip is the "
          "reference state - see the docstring)")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dr", type=float, default=0.2)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--fc1", action="store_true")
    p.add_argument("--fc2", action="store_true")
    p.add_argument("--fc4", action="store_true")
    p.add_argument("--ns", action="store_true")
    p.add_argument("--fc5", action="store_true")
    p.add_argument("--fc6", action="store_true")
    p.add_argument("--switch", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if COMM_WORLD.size > 1:
        raise SystemExit("Serial only: the block comparisons need a global "
                         "dense transpose.")

    run = {"fc1": args.fc1, "fc2": args.fc2, "fc4": args.fc4,
           "fc5": args.fc5, "fc6": args.fc6, "ns": args.ns,
           "switch": args.switch}
    if args.all or not any(run.values()):
        run = dict.fromkeys(run, True)

    results = {}
    if run["fc1"]:
        results["FC-1"] = gate_fc1(args.dr, args.nazim)
    if run["fc2"]:
        results["FC-2"] = gate_fc2(args.dr, args.nazim)
    if run["fc4"]:
        results["FC-4"] = gate_fc4(args.dr, args.nazim)
    if run["fc5"]:
        results["FC-5"] = gate_fc5(args.dr, args.nazim)
    if run["fc6"]:
        results["FC-6"] = gate_fc6(args.dr, args.nazim)
    if run["ns"]:
        results["FC-NS"] = gate_nullspace(args.dr, args.nazim)
    if run["switch"]:
        results["FC-SWITCH"] = gate_switch(args.dr, args.nazim)

    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"  {k:<10s} {verdict(v)}")
    print("=" * 78)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
