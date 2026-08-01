r"""The 3-D polar-motion phase, measured without solving anything.

B4 reports `dphi = +0.137 deg` against a reference phase of exactly
**-105.0000000000 deg**, and only with the fluid core. The reference is purely
geometric - load longitude plus 180 degrees, no rheology, no compliance, no
`C - A`, no time dependence - so a phase error is a defect somewhere, and a
spherically symmetric core cannot produce one. FC-5 measured the *form*'s own
response phase at 3.16e-11 degrees in 2-D and is blind to mesh-induced error by
construction, because the 2-D annulus is structured and its azimuthal symmetry
is exact. The 3-D CMB is unstructured tetrahedra with no rotational symmetry at
all.

## The instrument, and why it needs no solve

The phase is carried by `dI_13` and `dI_23`, and **their phase is analytic for a
prescribed displacement**. Take

    u = sin(theta) cos(theta) cos(lambda - lambda_0) rhat

which is the degree-2 tesseral pattern. With `p_1 = -xz` and `p_2 = -yz`,

    grad(p_1) . u = -2 u_r r sin(theta) cos(theta) cos(lambda)
    grad(p_2) . u = -2 u_r r sin(theta) cos(theta) sin(lambda)

and `int cos(lambda) cos(lambda - lambda_0) dlambda = pi cos(lambda_0)`,
`int sin(lambda) cos(lambda - lambda_0) dlambda = pi sin(lambda_0)`. So
`dI_13 propto cos(lambda_0)` and `dI_23 propto sin(lambda_0)` with the **same**
constant, and

    phase = atan2(dI_23, dI_13) = lambda_0   (mod 180 deg)

exactly, for any radial profile, any density stratification and any mesh. The
same is true of the CMB sheet `sigma = rho_core u_r(Rc)`, whose `p_i` carry the
identical azimuthal structure.

That removes the solve, the rheology, the load discretisation, the DtN treatment
and the Newton tolerance from the measurement in one step. **Whatever phase
error remains is quadrature and geometry**, which is exactly the quantity under
investigation.

## What it separates

Three numbers per longitude, from the shipped forms:

- **volume**, `int rho_0 grad(p_i).u dx_m` - `inertia_form`'s first line, i.e.
  the mantle's own quadrature over tetrahedra;
- **sheet**, `fluid_core_sheet_integral(p_i)` - the CMB facet integral, which is
  the only thing the fluid core adds and therefore the only candidate for an
  error that appears only when it is switched on;
- **total**, their sum.

And the team lead's two separators come for free, because both are sweeps of
this one measurement:

1. **Sweep `lambda_0`.** A defect in the *form* gives a constant offset - it
   does not know where the load is. Mesh anisotropy gives **scatter that moves
   with `lambda_0`**, because the pattern is sampled against a different set of
   facets each time.
2. **Refine.** Anisotropy falls; a form defect does not.

The load's own longitude never enters, which is the point: this measures the
mesh and the forms, not the run.

    PYTHONPATH=<worktree> python demos/gravity/spikes/spike_phase3d.py \
        [--configuration coarse] [--mesh-file ...]

Serial is fine and parallel is fine; every quantity is an assembled scalar.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np  # noqa: E402
from firedrake import *  # noqa: E402
from firedrake.petsc import PETSc  # noqa: E402
from gadopt.gia_gravity import (  # noqa: E402
    selfgrav_dtn_iterative_solver_parameters,
)

HERE = os.path.dirname(os.path.abspath(__file__))
B4DIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "glacial_isostatic_adjustment", "3d_spada_selfgrav"))
sys.path.insert(0, B4DIR)

import b4_polar_motion as b4  # noqa: E402


def say(*a):
    PETSc.Sys.Print(*a)


def tesseral(mesh, lam0_deg):
    r"""`u = sin(theta) cos(theta) cos(lambda - lambda_0) rhat`, as UFL.

    The degree-2 tesseral pattern, whose `dI_13`, `dI_23` phase is `lambda_0`
    exactly. Written in Cartesian coordinates so nothing depends on a polar
    convention: with `X = (x, y, z)` and `rho = sqrt(x^2 + y^2)`,
    `sin(theta) cos(theta) = rho z / r^2` and
    `cos(lambda - lambda_0) = (x cos(lambda_0) + y sin(lambda_0)) / rho`, so the
    product is `z (x cos + y sin) / r^2` with no `atan2` and no pole.
    """
    X = SpatialCoordinate(mesh)
    r2 = dot(X, X)
    lam0 = np.radians(lam0_deg)
    amplitude = X[2] * (X[0] * np.cos(lam0) + X[1] * np.sin(lam0)) / r2
    return amplitude * X / sqrt(r2)


def phase_of(dI13, dI23):
    """`atan2(dI_23, dI_13)` in degrees, folded onto (-90, 90]."""
    ang = np.degrees(np.arctan2(dI23, dI13))
    return (ang + 90.0) % 180.0 - 90.0


def measure(solver, layout, lam0_deg):
    """Volume, sheet and total `dI` phases for one prescribed longitude."""
    sub = layout.mechanics_mesh
    u = solver.solution.subfunctions[layout.displacement]
    u.interpolate(tesseral(sub, lam0_deg))

    rho0 = solver.approximation.density
    Xm = SpatialCoordinate(sub)
    u_split = solver.solution_split[layout.displacement]

    out = {}
    for name in ("volume", "sheet"):
        vals = []
        for i in (0, 1):
            p = solver.inertia_polynomial(i, Xm)
            if name == "volume":
                # `inertia_form`'s first line verbatim.
                form = rho0 * dot(grad(p), u_split) * solver.dx_m
            else:
                form = solver.fluid_core_sheet_integral(p)
            vals.append(assemble(form))
        out[name] = vals
    out["total"] = [out["volume"][i] + out["sheet"][i] for i in (0, 1)]

    for name in ("volume", "sheet", "total"):
        a, b = out[name]
        out[name + "_phase"] = phase_of(a, b)
        # Fold the DIFFERENCE, not the two phases separately. Folding each and
        # subtracting puts a spurious 180 at the branch: at lambda_0 = 90 the
        # target folds to -90 and a measurement a hair above folds to +90, so a
        # 4e-09 error reads as 1.8e+02. Measured, and it is the only reason the
        # first medium run showed a "180 degree" row.
        out[name + "_dphi"] = (phase_of(a, b) - lam0_deg + 90.0) % 180.0 - 90.0
    out["sheet_fraction"] = (np.hypot(*out["sheet"])
                             / max(np.hypot(*out["total"]), 1e-300))
    return out


def sectoral(mesh, degree, order, lam0_deg=0.0):
    r"""A high-degree radial pattern, as a *sectoral* harmonic `sin^m(theta)`.

    Used to measure **leakage**, which is the one gap the degree-2 sweep leaves.
    `dI_13` and `dI_23` are degree-2 moments, so by orthogonality *no* other
    degree contributes to them at all - analytically. On a discrete mesh the
    orthogonality is only as good as the quadrature, and the load carries
    degrees to n = 32, so a high degree leaking into the degree-2 moment is a
    mechanism the pure degree-2 sweep cannot see: that sweep measures how
    faithfully a degree-2 pattern is *projected*, not whether something else is
    projected onto it.

    Written as `(rho/r)^m cos(m (lambda - lambda_0)) (z/r)^k` with
    `rho^m cos(m lambda) = Re[(x + iy)^m]` expanded by the binomial theorem, so
    it stays polynomial in the coordinates and needs no `atan2`.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    lam0 = np.radians(lam0_deg)
    # Re[(x + i y)^m e^{-i m lambda_0}] by the binomial theorem.
    real, imag = Constant(1.0), Constant(0.0)
    for _ in range(order):
        real, imag = real * X[0] - imag * X[1], real * X[1] + imag * X[0]
    horiz = real * np.cos(order * lam0) + imag * np.sin(order * lam0)
    amplitude = horiz * (X[2] / r) ** (degree - order) / r ** order
    return amplitude * X / r


def run(configuration, mesh_file, longitudes, dt_yr=0.03, untangle=True):
    # `untangle` is a parameter here so the sweep can be run BOTH ways. The
    # coarse and medium numbers reported first were taken with it ON (the
    # default), which turns out to matter: untangling is the prime suspect for
    # B4's phase error, and a sweep that already includes it and is clean at
    # 4e-05 says the correction cannot rotate the phase *through dI's
    # evaluation* - so if it is the cause, it acts through `u`.
    parent, sub = b4.build_meshes(configuration, path=mesh_file,
                                  untangle=untangle)
    say(f"parent cells {parent.num_cells()}, mantle cells {sub.num_cells()}")
    # `dt` is irrelevant to every quantity below - nothing is solved and the
    # inertia forms carry no time step - but the constructor needs one.
    solver, z, layout = b4.build_solver(parent, sub, dt=dt_yr,
                                        fluid_core=True)

    say("\n  lambda_0     volume dphi      sheet dphi      total dphi"
        "    sheet/total")
    rows = []
    for lam0 in longitudes:
        m = measure(solver, layout, lam0)
        rows.append((lam0, m))
        say(f"  {lam0:8.2f}  {m['volume_dphi']:+14.6e}  "
            f"{m['sheet_dphi']:+14.6e}  {m['total_dphi']:+14.6e}"
            f"  {m['sheet_fraction']:12.4f}")

    for name in ("volume", "sheet", "total"):
        vals = np.array([m[name + "_dphi"] for _, m in rows])
        say(f"\n  {name:>6s}: mean {vals.mean():+.6e} deg, "
            f"scatter (std) {vals.std():.6e} deg, "
            f"peak-to-peak {np.ptp(vals):.6e} deg")

    # The degree-2 reference, normalised the same way, so the rotation and
    # leakage rows below are relative to a physically comparable amplitude.
    u_ref = solver.solution.subfunctions[layout.displacement]
    u_ref.interpolate(tesseral(sub, 0.0))
    reference_norm = float(sqrt(assemble(dot(u_ref, u_ref) * solver.dx_m)))
    _vals = []
    for i in (0, 1):
        _p = solver.inertia_polynomial(i, SpatialCoordinate(sub))
        _vals.append(assemble(
            solver.approximation.density
            * dot(grad(_p), solver.solution_split[layout.displacement])
            * solver.dx_m)
            + assemble(solver.fluid_core_sheet_integral(_p)))
    reference_deg2 = float(np.hypot(*_vals))

    # Rigid rotation: how much phase can a spurious kernel mode carry?
    #
    # Handoff §12.8b: the rotation kernel is annihilated only to ~2e-06 on a
    # curved tet sphere, and an O(eps) forcing on an O(eps) stiffness gives an
    # O(1) contamination of `u`. A rotation is purely tangential, so it moves
    # |m| not at all - `(omega x x).grad(p_3) = -2xy + 2xy = 0` *pointwise* -
    # but for the tesseral pair it is only zero **as an integral**:
    # `(omega_z x x).grad(p_1) = z y`, which vanishes by symmetry and not
    # pointwise. So a rigid rotation CAN rotate the phase, through quadrature
    # error alone, and this measures by how much per unit amplitude.
    #
    # In B4's configuration the mode is both declared and projected out after
    # every solve (`SelfGravitatingGIASolver.solve` calls
    # `project_out_nullspace`), so what this bounds is the residue left by an
    # imperfect projection rather than a raw contamination.
    say("\n  rigid rotation: dI per unit rotation amplitude")
    say("  (exactly zero in the continuum for every generator; whatever this")
    say("   is, is quadrature. A rotation cannot move |m|, but it can rotate")
    say("   the phase - handoff §12.8b.)")
    Xm = SpatialCoordinate(sub)
    generators = {
        "e_x": as_vector([Constant(0.0), -Xm[2], Xm[1]]),
        "e_y": as_vector([Xm[2], Constant(0.0), -Xm[0]]),
        "e_z": as_vector([-Xm[1], Xm[0], Constant(0.0)]),
    }
    u = solver.solution.subfunctions[layout.displacement]
    u_split = solver.solution_split[layout.displacement]
    say(f"  {'generator':>10s}{'|dI_13,23|':>16s}{'relative to l=2':>18s}"
        f"{'implied dphi deg':>18s}")
    for name, mode in generators.items():
        u.interpolate(mode)
        norm_u = sqrt(assemble(dot(u, u) * solver.dx_m))
        vals = []
        for i in (0, 1):
            p = solver.inertia_polynomial(i, Xm)
            vals.append(assemble(
                solver.approximation.density * dot(grad(p), u_split)
                * solver.dx_m) + assemble(solver.fluid_core_sheet_integral(p)))
        mag = float(np.hypot(*vals)) / float(norm_u)
        rel = mag / max(reference_deg2 / reference_norm, 1e-300)
        say(f"  {name:>10s}{mag:16.6e}{rel:18.6e}"
            f"{np.degrees(rel):18.6e}")

    # Leakage: a high degree must contribute NOTHING to a degree-2 moment.
    say("\n  leakage of a high degree into the degree-2 moment")
    say("  (analytically zero by orthogonality; whatever this is, is the")
    say("   mesh's own. The load carries degrees to n = 32, so this is the")
    say("   mechanism the degree-2 sweep above cannot see.)")
    reference = None
    say(f"  {'(l, m)':>10s}{'|dI_13,23|':>16s}{'relative to l=2':>18s}")
    for degree, order in ((2, 1), (6, 3), (12, 5), (20, 7), (32, 9)):
        u = solver.solution.subfunctions[layout.displacement]
        u.interpolate(sectoral(sub, degree, order))
        Xm = SpatialCoordinate(sub)
        u_split = solver.solution_split[layout.displacement]
        vals = []
        for i in (0, 1):
            p = solver.inertia_polynomial(i, Xm)
            vals.append(assemble(
                solver.approximation.density * dot(grad(p), u_split)
                * solver.dx_m) + assemble(solver.fluid_core_sheet_integral(p)))
        mag = float(np.hypot(*vals))
        if reference is None:
            reference = mag
        say(f"  {f'({degree}, {order})':>10s}{mag:16.6e}"
            f"{mag / max(reference, 1e-300):18.6e}")
    return rows


def condensed_build(enable):
    """Inject `condense_internal_variables` into B4's builder without editing it.

    The internal variable is 2.86e6 of 3.35e6 dofs at `--coarse`, so condensing
    removes 85 % of block 0 and takes its FGMRES count from the 200-iteration
    cap to a flat 3. On the uncondensed path a single `--coarse` coupled solve
    is 814 s with block 0 hitting `DIVERGED_ITS` on every outer step, and a
    tolerance ladder makes that *worse* - tightening the outer tolerance buys
    more outer iterations each paying a capped block-0 solve, which is how a
    four-row ladder ate a six-hour walltime.

    `b4_polar_motion.py` is refdata's and exposes no such keyword, so the two
    call sites are patched in place: the space factory and the solver class,
    both of which the flag has to reach. The same monkeypatch idiom `gate_v2`
    uses on the demo module.
    """
    if not enable:
        return lambda: None
    space, cls = b4.self_gravitating_gia_space, b4.SelfGravitatingGIASolver

    def condensed_space(*a, **kw):
        kw["condense_internal_variables"] = True
        return space(*a, **kw)

    class CondensedSolver(cls):
        def __init__(self, *a, **kw):
            kw["condense_internal_variables"] = True
            super().__init__(*a, **kw)

    b4.self_gravitating_gia_space = condensed_space
    b4.SelfGravitatingGIASolver = CondensedSolver

    def restore():
        b4.self_gravitating_gia_space = space
        b4.SelfGravitatingGIASolver = cls
    return restore


def solve_phase(configuration, mesh_file, tolerances, untangle=True,
                condense=False):
    """The elastic phase against the solver tolerance. One solve per entry.

    **The remaining hypothesis, tested directly.** With the forms and the mesh
    eliminated above, the phase error has to be in the displacement the solve
    produces rather than in `dI`'s evaluation of it - and the obvious candidate
    is the tolerance. B4's default is `snes_rtol = 1e-4` with `ksp_rtol = 1e-6`,
    **both relative to the norm of the whole mixed residual**, which is
    dominated by the mechanics rows. The rotation rows carry
    `theta_rot = s_i f B_mu Omega_sq` with `Omega_sq = 1.566e-03`, so their
    contribution to that norm is three orders down and they are converged to
    correspondingly fewer digits.

    The arithmetic is suggestive: a phase error of 0.137 deg is 2.4e-03 radians,
    i.e. a relative error of 2.4e-03 in the *direction* of `m`, which is what a
    globally-normed tolerance of 1e-04 on a residual whose rotation rows are
    ~1e-03 of the total would leave. If that is the mechanism, tightening the
    tolerance moves the phase towards -105 and |m| barely at all - because the
    magnitude is set by the dominant rows and the direction by the small ones.
    """
    # `untangle=False` here deliberately: untangling costs >28 min at
    # `--coarse`, and per the A3 exposure split it changes only *facet*
    # integrals at the tangled radii. `dI` is a volume moment and |m| follows
    # from it, so this ladder's quantities are untouched by it. Keep it on for
    # anything reading a facet integral at a tangled radius.
    restore = condensed_build(condense)
    try:
        parent, sub = b4.build_meshes(configuration, path=mesh_file,
                                      untangle=untangle)
    finally:
        pass
    dt = 0.03 / b4.T_BAR_YR
    say(f"\nparent cells {parent.num_cells()}, mantle cells {sub.num_cells()}")
    say(f"reference phase {b4.REFERENCE_PHASE_DEG:.10f} deg\n")
    say(f"  {'ksp_rtol':>10s}{'snes_rtol':>11s}{'block0':>9s}"
        f"{'|m| deg':>12s}{'phase deg':>12s}{'dphi deg':>11s}"
        f"{'d|m|/|m|':>12s}")
    baseline_absm = None
    out = []
    for ksp_rtol, snes_rtol, block0 in tolerances:
        # `gadopt`'s packaged iterative preset, not B4's own dictionary and
        # not the MUMPS default whose docstring says it has no 3-D successor.
        #
        # **`condensed=True` does not work through `DtNTwoBlockSchurPC` yet**:
        # measured, the nested block-0 split dies in `PCFieldSplitSetDefaults`
        # -> `DMCreateSubDM` -> `IndexError: tuple index out of range`, i.e.
        # the sub-DM does not present the fields the split addresses once the
        # internal variable is gone. That is a solver-configuration question
        # and it belongs to whoever owns the preconditioner; this ladder wants
        # a phase, so it runs the uncondensed sweep, which is the measured and
        # working one.
        params = selfgrav_dtn_iterative_solver_parameters(
            condensed=condense, block0_rtol=block0, outer_rtol=ksp_rtol,
            snes_rtol=snes_rtol)
        solver, z, layout = b4.build_solver(
            parent, sub, dt=dt, fluid_core=True, solver_parameters=params)
        solver.solve()
        mx, my, absm, phase = b4.polar_motion_deg(solver)
        dphi = phase - b4.REFERENCE_PHASE_DEG
        if baseline_absm is None:
            baseline_absm = absm
        out.append((ksp_rtol, snes_rtol, block0, absm, phase, dphi))
        # |m| against the FIRST row, because the signature that confirms the
        # mechanism is "phase moves, magnitude does not". Either half alone is
        # ambiguous, and B4's own tolerance test measured only the magnitude -
        # which is why its null result is consistent with this hypothesis
        # rather than a refutation of it.
        say(f"  {ksp_rtol:10.1e}{snes_rtol:11.1e}{block0:9.1e}"
            f"{absm:12.7f}{phase:12.4f}{dphi:+11.4f}"
            f"{(absm - baseline_absm) / max(baseline_absm, 1e-300):12.2e}")
    restore()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse")
    ap.add_argument("--mesh-file", default=None)
    ap.add_argument("--no-untangle", action="store_true",
                    help="skip curve_mesh's untangle pass: >28 min at coarse, "
                         "and it changes only facet integrals at the tangled "
                         "radii, which dI (a volume moment) is not")
    ap.add_argument("--condense", action="store_true",
                    help="static condensation of the internal variable; see "
                         "`condensed_build`")
    ap.add_argument("--rows", type=int, default=None,
                    help="run only the first N rows of the ladder, so a "
                         "walltime overrun cannot lose every row")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the prescribed-field sweep; the tolerance "
                         "ladder alone. Worth having because `curve_mesh"
                         "(untangle=True)` is intermittently fatal in "
                         "parallel - `IndexError: index 3267 is out of bounds "
                         "for axis 0 with size 1914` on the third of three "
                         "identical builds in one job - so a run should build "
                         "the meshes as few times as it can.")
    ap.add_argument("--solve", action="store_true",
                    help="also run the elastic solve at several tolerances")
    ap.add_argument("--ladder", default="together",
                    choices=["together", "separate"],
                    help="'separate' varies ksp_rtol and snes_rtol ONE AT A "
                         "TIME, which is what says which of the two binds; "
                         "`snes_rtol = 1e-4` is much the weaker of the pair "
                         "and a run that tightened only the Krylov tolerance "
                         "never touched the binding constraint")
    ap.add_argument("--longitudes", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0,
                             150.0])
    args, _ = ap.parse_known_args()

    say("=" * 78)
    say("3-D polar-motion phase, from a PRESCRIBED degree-2 field (no solve)")
    say("=" * 78)
    say("Expected, before the run:")
    say("  dphi = 0 exactly, for every longitude and for each of the three")
    say("  numbers separately, because the phase of dI for this pattern is")
    say("  lambda_0 analytically - no rheology, no solve, no load.")
    say("  A CONSTANT offset  -> a defect in a form (it cannot know lambda_0)")
    say("  SCATTER with lambda_0 -> mesh anisotropy, and its size is the")
    say("     mesh's own contribution to B4's 0.137 deg")

    if not args.no_sweep:
        run(args.configuration, args.mesh_file, args.longitudes,
            untangle=not args.no_untangle)

    if args.solve:
        say("\n" + "=" * 78)
        say("The phase against the solver tolerance (this DOES solve)")
        say("=" * 78)
        say("Expected, if the tolerance is the mechanism: the phase moves")
        say("towards -105 as the tolerance tightens, and |m| barely moves -")
        say("the magnitude is set by the dominant rows and the direction by")
        say("the small ones.")
        ladders = {
            "together": [(1e-6, 1e-4, 1e-2), (1e-10, 1e-10, 1e-4),
                         (1e-13, 1e-13, 1e-6)],
            # One at a time: row 2 moves only the SNES tolerance, row 3 only
            # the Krylov one, row 4 both. Whichever row moves the phase names
            # the binding constraint.
            "separate": [(1e-6, 1e-4, 1e-2),
                         (1e-6, 1e-10, 1e-2),
                         (1e-12, 1e-4, 1e-4),
                         (1e-12, 1e-12, 1e-4)],
        }
        rows = ladders[args.ladder]
        if args.rows:
            rows = rows[:args.rows]
        solve_phase(args.configuration, args.mesh_file, rows,
                    untangle=not args.no_untangle, condense=args.condense)


if __name__ == "__main__":
    main()
