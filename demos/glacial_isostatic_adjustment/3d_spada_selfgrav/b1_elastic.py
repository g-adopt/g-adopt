"""B1 — the elastic snapshot: the first benchmark number the project produces.

Handoff §11 B1 *as corrected*: cap load, one solve, no time loop, `--coarse`
then `--medium`, nu = 0.28, penalty incompressibility as decided by the locking
adjudication. Uses A1's `taboo_synthesis` + `reference.npz`, A2's
`generate_selfgrav_sphere`, A3's `reference_state` and A4's `FluidCore`. It
edits none of them.

## What this is and is not

**It is a smoke test on absolute magnitudes, not a gate.** Phase B runs at
nu = 0.28 against an incompressible benchmark, so the amplitude is expected
*high* by 25-45%, most likely ~35%: U(0°) ~ -35 to -40 m against the reference
-27.7739 m. That is ~50x the whole t = 0 tolerance. **A result near -27.77 m is
evidence of two errors cancelling, not of success**, and is to be reported as a
failure of the expectation rather than a success of the model.

**It does not pin B_mu and cannot.** B_mu scales the mechanics boundary
condition but not the Poisson sheet, which makes it algebraically degenerate
with compressibility in all five numbers. The amplitude factor is *recorded*
here and diagnosed only after C2 extrapolates to nu -> 0.5.

The fingerprint that confirms compressibility rather than something else is the
*asymmetry*: `R_U ~ R_V ~ R_U180 ~ 1.35` while `R_N(0) ~ 0.96` and
`R_N(180) ~ 0.86` — the U-family moving together by a third while N(0) moves 4%
the other way. N(0°) is 111% direct term (+44.4160 from the ice's own
attraction against -4.2598 from the elastic response), so only 10.6% of the
geoid is mechanics: N is a weak test of the mechanics and a strong test of the
Poisson source.

## Why everything is done by Legendre projection

The five tabulated numbers and the per-degree ratios come from **one surface
quadrature per field**, not from point evaluation:

    U_n = int_Re u_r P_n(cos theta) dS / int_Re P_n^2 dS

and then `U(0°) = sum_n U_n`, `U(180°) = sum_n (-1)^n U_n`, and V from the
`dP_n/dtheta` series. This is the same representation the reference is
synthesised in, so the comparison is coefficient against coefficient with no
interpolation error and no `VertexOnlyMesh` point-location risk on a curved
boundary. It also makes the diagnostic the handoff calls "worth more than all
five numbers" fall out for free, because each error source has its own shape:

    B_mu / compressibility  flat and != 1
    mesh error              1 at low n, departing upward
    truncation              discontinuous at n_max
    trap 1 (ice missing)    U untouched, every degree of N moved

and the axisymmetry residual is what is left of `u_r` after the m = 0
projection is removed, which is free and reference-independent.

## The reference is synthesised at the mesh's own n_max

Per §1.2. Comparing against the archive's fixed n = 128 would carry 18.6 m of
pure truncation error at `--medium`. The truncation error also has a **sign
reversal** — dU(0°) is -2.137 m at n_max = 20 and +1.463 m at n_max = 40,
because the cap's kink at alpha = 10° makes the series ring rather than
converge — which looks like a bug across two resolutions and is not one.
Matched synthesis cancels it by construction.

## Trap 1 has a sign, not a magnitude

If the ice is missing from the Poisson source, N(0°) = -4.2598 and
N(180°) = -0.5571: a **sign reversal**, with U perfect. The gate asserts the
sign and needs no tolerance.

## The stage-1 geometry run plan, and its predictions

**Written before any of these numbers existed.** The discipline that has caught
every real error on this branch is committing to the expected value first, so
this block is a record, not a summary. Three arms, in order:

    A   curved, unrepaired          `--configuration coarse`
    D   curved, fixed-point clean   `--untangle`
    E   mechanics-only on D         `--untangle --mechanics-only`

Arm B (curved, one-pass "repair") is **dropped**: that path only ever existed
to isolate the deep folded cell when no clean arm was available, and A-vs-D
supersedes it. The one-pass code is gone, so B is not runnable in any case.
Arm C (`--no-curve`, uncurved) is **demoted to a follow-up**, to be run only if
A-vs-D comes back ambiguous, where it separates the folds from the curving map
itself.

**A, the baseline.** `U_2 = 2.66 +- 0.1`; the per-degree ratio monotone,
falling to ~1.70 by n = 20; `N_2 = 0.76 +- 0.02`.
*Falsifier: anything outside those bands and everything stops.* A does not
merely fail; it reopens the mesh-provenance question, and no comparison
proceeds until that is settled.

**D, the experiment.** `U_2` in **1.25-1.55**; per-degree spread <= 15%; and
the compliance factor inferred from `U_2` agreeing with the one inferred from
`N_2` to <= 15%. On the sideways channel: if V from the **n >= 2**
reconstruction, after L2 orthogonalisation against the rotation generators,
stays near **0.06** on D, then contamination, truncation *and* folds are all
excluded at once, and the deficit is structural or resolution.

**E on D.** 1.25-1.5 and flat against a no-gravity reference; 1.3-1.9
shape-only against the self-gravitating one. The second is a **band, never
scored pass/fail** - switching self-gravity off removes the stiffening of
exactly the low degrees that dominate U(0), so E is an upper bound on B1 and
not an estimate of it.

**The branch nobody will want to face, written now so it cannot be reframed
later.** If A reproduces its band and D *also* returns `U_2 > 2`, then geometry
is dead as a class - folds, curving and conditioning together - and the next
suspects are the coupling terms that carry the `B_mu` scale and are pinned by
nothing in 3-D, **starting with the CMB spring magnitude**. That is the one
physics difference consistent with the observed pattern: a geoid that is
approximately right sitting beside a displacement that is badly wrong.

Usage:

    PYTHONPATH=$(pwd):$(pwd)/demos/glacial_isostatic_adjustment/3d_spada_selfgrav \\
        python demos/glacial_isostatic_adjustment/3d_spada_selfgrav/b1_elastic.py \\
        --configuration coarse
"""
import argparse
import os
import time

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np  # noqa: E402
from firedrake import (Constant, Function, FunctionSpace, Mesh,  # noqa: E402
                       SpatialCoordinate, Submesh, TensorFunctionSpace,
                       VectorFunctionSpace, as_vector, assemble, avg,
                       conditional, dot, ds, dx, sqrt)
from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                    SphericalDtN, gamg_parameters)
from gadopt.gia_gravity import (FluidCore, SelfGravitatingGIASolver,  # noqa: E402
                                rigid_rotation_nullspace,
                                selfgrav_dtn_iterative_solver_parameters,
                                self_gravitating_gia_space)

# B2's solver, imported read-only rather than pasted so the two cannot drift.
import sys  # noqa: E402
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "gravity", "spikes"))
from gate_b2_solver import solver_parameters as b2_solver_parameters  # noqa: E402,E501
from gate_b2_solver import SPLIT_NAMES  # noqa: E402
from firedrake import MixedVectorSpaceBasis  # noqa: E402
from gadopt import rigid_body_modes  # noqa: E402

import generate_selfgrav_sphere as gen  # noqa: E402
import reference_state as refstate  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402
from validate_selfgrav_sphere import (curve_mesh,  # noqa: E402
                                      provenance, tangle_census)

# **The moments come from A3's leaf module, not from a literal and not from the
# rotation driver.** `C_minus_A` had been hard-copied here as 2.362822e-01,
# which is the *secondary* (prescribed) value; the primary, k_s-consistent one
# is 2.4214e-01. Harmless in this driver because B1 runs with rotation off, but
# a live copy of a constant whose mis-copying is a documented trap has already
# fired once, and a second copy is a second chance to fire.
#
# **Importing the dict and picking a key would recreate that trap one
# indirection deeper**, so the value is asserted below rather than trusted --
# and the assertion is written to REJECT the secondary, not merely to accept
# the primary. A test that only accepts the right answer is satisfied by any
# implementation that happens to produce it.
C_MINUS_A = refstate.C_MINUS_A
C_NONDIM = refstate.C_NONDIM
OMEGA_SQ = refstate.OMEGA_SQ
C_MINUS_A_PRIMARY = refstate.C_MINUS_A_PRIMARY
assert abs(C_MINUS_A_PRIMARY - 2.4214e-01) < 1e-6, (
    f"C-A primary is {C_MINUS_A_PRIMARY!r}, expected the k_s-consistent "
    "2.4214e-01")
assert abs(C_MINUS_A_PRIMARY - C_MINUS_A["prescribed"]) > 1e-3, (
    f"C-A primary {C_MINUS_A_PRIMARY!r} is indistinguishable from the "
    f"SECONDARY prescribed value {C_MINUS_A['prescribed']!r}; the trap this "
    "assertion exists to catch has fired")
assert C_MINUS_A_PRIMARY is C_MINUS_A["ks"], "primary must be the ks entry"

HERE = os.path.dirname(os.path.abspath(__file__))

# 64 ranks printing the same table is unreadable and slow. Every print below is
# of collective quantities, so rank 0 is the whole story.
from firedrake import COMM_WORLD  # noqa: E402
from mpi4py import MPI  # noqa: E402

_builtin_print = print


def print(*a, **k):  # noqa: A001
    if COMM_WORLD.rank == 0:
        _builtin_print(*a, **k, flush=True)

# §2.2. D, rho_bar and g_bar come from A3 so there is one copy.
D_M = refstate.D_SCALE if hasattr(refstate, "D_SCALE") else 2.891e6
LAMBDA = refstate.LAMBDA
RHO_BAR = refstate.RHO_BAR
G_BAR = refstate.G_BAR
MU_BAR = 1.0e11
B_MU = RHO_BAR * G_BAR * D_M / MU_BAR

# Layered material properties, §2.2. Radii descending from Re.
# (r_outer, r_inner, rho/rho_bar, mu/mu_bar, eta/eta_bar)
LAYERS = [
    (gen.RE, 2.179523, 3037.0 / RHO_BAR, 0.50605, 1.0e19),
    (2.179523, 2.058457, 3438.0 / RHO_BAR, 0.70363, 1.0),
    (2.058457, 1.971982, 3871.0 / RHO_BAR, 1.05490, 1.0),
    (1.971982, gen.RC, 4978.0 / RHO_BAR, 2.28340, 2.0),
]
RHO_CORE = 10750.0 / RHO_BAR

# The mesh's honest spectral reach, §4.  Lateral h resolves degree n when
# 2 pi a / (4 n) ~ h_lat, i.e. n ~ 2 pi Re / (4 h).
NMAX_OF = {"coarse": 20, "medium": 40, "fine": 80, "production": 128}

BULK_SHEAR_RATIO = 1.9394        # nu = 0.28, §8
DT_ELASTIC = 1.0e-4              # Maxwell times; §9's elastic snapshot


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def curve(mesh, untangle=False, enabled=True):
    """Radial P2 remap. `Submesh` does not inherit the parent's coordinates.

    **This is A2's `validate_selfgrav_sphere.curve_mesh` at its default**, line
    for line - `sqrt(dot(X, X))` and `sqrt(X[0]**2 + X[1]**2 + X[2]**2)` are the
    same expression. So every geometric number A2 measured (volumes 6.09e-07,
    Re 4.11e-07, Rc 4.60e-06, the `dS(2)`/`ds(2)` identity, cross-mesh
    5.27e-07) was measured on exactly this operation, and B1 reproduces the Re
    figure independently: 6.10279551e+01 against 4 pi Re^2 = 6.10279802e+01,
    **4.11e-07**.

    `untangle=True` delegates to A2's version, which resets the P2 edge nodes
    of tangled cells to the straight midpoints - 2 cells of 113 653 at
    `--coarse`. Kept as a switch so that "geometry" is eliminated by
    measurement rather than by the code being identical.

    **Read A2's docstring before believing the switch does what it says.**
    Measured there: the reset straightens the cells it targets and tangles two
    of their neighbours instead, so the count does not fall, and at 4 ranks it
    rises. Which is exactly why the value below is printed where `curve`
    *receives* it and the tangle census is run on the result: neither the flag
    nor the help text is evidence.
    """
    # Printed here rather than at parse time, because "the flag was parsed" and
    # "the flag reached the mesh" are different statements and this project has
    # already shipped a run where only the first was true.
    print(f"    curve(): enabled={enabled}  untangle={untangle}")
    if not enabled:
        # **Arm C, and it is the decisive arm of the geometry A/B.** The census
        # measures ZERO tangled cells on the mesh as gmsh writes it, so this is
        # a clean control -- not "less curved", but no folds at all, by
        # construction rather than by repair. An A/B between curved-and-
        # repaired and curved-and-not compares two geometries that were both
        # made by this function; this one was not.
        return mesh
    if untangle:
        return curve_mesh(mesh, untangle=True)
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    r_p1 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(mesh, "CG", 2)).interpolate((r_p1 / r) * X)
    return Mesh(X_p2)


def build_meshes(configuration, reuse=True, h=None, untangle=False,
                 curve_enabled=True):
    tag = configuration if h is None else f"h{h:g}"
    path = os.path.join(HERE, f"b1_{tag}.msh")
    if not (reuse and os.path.exists(path)):
        gen.generate(path, configuration=configuration, h=h)
    t0 = time.perf_counter()
    parent = curve(Mesh(path), untangle=untangle, enabled=curve_enabled)
    parent.cartesian = False
    t_parent = time.perf_counter() - t0
    t0 = time.perf_counter()
    sub = curve(Submesh(parent, 3, gen.CELL_MANTLE), untangle=untangle,
                enabled=curve_enabled)
    sub.cartesian = False
    t_sub = time.perf_counter() - t0
    # **The census, not the flag.** An A/B on the tangling repair is only an
    # A/B if the two arms differ, and the only way to know that from a log is
    # to count the tangled cells in the arm that ran. The repaired arm must
    # show zero and the unrepaired arm at least one; anything else means the
    # comparison is void and this line is where that becomes visible.
    tangle_census(parent, f"parent, curve={curve_enabled} untangle={untangle}")
    tangle_census(sub, f"mantle submesh, curve={curve_enabled} "
                       f"untangle={untangle}")
    return parent, sub, t_parent, t_sub


def layered(mesh, index, name):
    """A DG0 field taking `LAYERS[*][index]` between the layer radii.

    The innermost layer's value is the default and each shell overrides it
    above that shell's inner radius, so the shells are applied outward and the
    result is exact on a mesh whose cells conform to the interfaces - which
    A2's generator guarantees by making every interface a geometric entity.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    expr = Constant(LAYERS[-1][index])
    for row in reversed(LAYERS):
        expr = conditional(r >= Constant(row[1]), Constant(row[index]), expr)
    return Function(FunctionSpace(mesh, "DG", 0), name=name).interpolate(expr)


# --------------------------------------------------------------------------
# Legendre machinery, shared by the load and the projection
# --------------------------------------------------------------------------

def legendre_ufl(nmax, x):
    """P_0..P_nmax as UFL expressions in `x = cos theta`."""
    P = [Constant(1.0), x]
    for n in range(1, nmax):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return P[:nmax + 1]


def dlegendre_ufl(nmax, x, P):
    """d P_n / d theta = -sin(theta) P_n'(x), via (1-x^2) P_n' = n(P_{n-1}-x P_n).

    `d_theta P_n = -n (P_{n-1} - x P_n) / sin(theta)`, and `sin theta` is
    bounded away from zero everywhere the tangential displacement is read.
    """
    s = sqrt(conditional(1.0 - x * x > 1e-14, 1.0 - x * x, 1e-14))
    return [Constant(0.0)] + [
        -n * (P[n - 1] - x * P[n]) / s for n in range(1, nmax + 1)]


def cap_sigma_hat(nmax):
    """Non-dimensional cap coefficients, degree 0..nmax, sigma/(rho_bar D)."""
    return taboo.cap_load(nmax) / (RHO_BAR * D_M)


def load_field(mesh, nmax, sigma_n):
    """The truncated cap series as a CG2 field, §3.

    Interpolated rather than left as UFL so that the load the solver sees is
    the load the mesh can carry; the aliasing failure §11 warns about is the
    nominal n_max exceeding what the mesh actually resolves, and that is a
    property of this interpolation.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    P = legendre_ufl(nmax, X[2] / r)
    expr = Constant(0.0)
    for n in range(2, nmax + 1):
        expr = expr + Constant(sigma_n[n]) * P[n]
    return Function(FunctionSpace(mesh, "CG", 2), name="sigma_load").interpolate(expr)


# --------------------------------------------------------------------------
# projection of the answer onto Legendre modes
# --------------------------------------------------------------------------

def project_surface(field_expr, mesh, nmax, measure, interior, basis="P",
                    quad_degree=None):
    """Coefficients `f_n = int f P_n dS / int P_n^2 dS` on a sphere.

    `int P_n^2 dS = 4 pi R^2 / (2n+1)` analytically, but the *measured* value
    is used instead so that the surface's own discretisation error cancels
    between numerator and denominator.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    P = legendre_ufl(nmax, X[2] / r)
    if basis == "dP":
        P = dlegendre_ufl(nmax, X[2] / r, P)
    # **An explicit quadrature degree is mandatory here, not an optimisation.**
    # `P_n(z/r)` is a degree-n rational expression, and TSFC's estimator walks
    # the expression tree and asks for a rule of degree ~500 at n_max = 20. A
    # degree-500 rule on curved P2 facets OOM-killed a rank at 322 GB after the
    # physics had already converged. The integrand is genuinely degree ~2n after
    # the geometry map; `--quad-sweep` measures where the coefficients stop
    # moving rather than guessing, because an UNDER-integrated projection would
    # silently corrupt exactly the per-degree ratios this script exists to
    # produce.
    if quad_degree is not None:
        measure = measure(metadata={"quadrature_degree": quad_degree})
    out = np.zeros(nmax + 1)
    for n in range(nmax + 1):
        if interior:
            num = assemble(avg(field_expr * P[n]) * measure)
            den = assemble(avg(P[n] * P[n]) * measure)
        else:
            num = assemble(field_expr * P[n] * measure)
            den = assemble(P[n] * P[n] * measure)
        out[n] = num / den if abs(den) > 0.0 else 0.0
    return out


def series_at(coeffs, theta):
    """Sum_n f_n P_n(cos theta)."""
    x = np.cos(theta)
    nmax = len(coeffs) - 1
    P = np.zeros((nmax + 1, np.size(x)))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return coeffs @ P


def dseries_at(coeffs, theta):
    """Sum_n f_n dP_n/dtheta."""
    x = np.cos(theta)
    s = np.sin(theta)
    nmax = len(coeffs) - 1
    P = np.zeros((nmax + 1, np.size(x)))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    dP = np.zeros_like(P)
    ok = s > 1e-13
    for n in range(1, nmax + 1):
        dP[n, ok] = -n * (P[n - 1, ok] - x[ok] * P[n, ok]) / s[ok]
    return coeffs @ dP


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def condensed_solver_parameters(outer_rtol=1e-8, block0_rtol=1e-2,
                                block0_max_it=200,
                                u_pc="gadopt.RigidBodyAssembledPC"):
    """`gadopt.selfgrav_dtn_iterative_solver_parameters`, condensed.

    **This used to be a hand-copy of that dictionary and the comment above the
    import claimed it was imported "so the two cannot drift". It was pasted,
    and it had drifted** - `block0_max_it` 200 against the library's 60, and a
    `u_pc` naming `gate_b2_solver`'s copy of `RigidBodyAssembledPC` rather than
    the shipped one. Four copies of this dictionary existed across the library
    and three drivers; this is now a thin wrapper that names only the two
    values B1 genuinely wants different from the library defaults, so a reader
    can see the whole of the difference in one place.

    Both remaining differences are B1's own, deliberately:

    * `outer_rtol` 1e-8 rather than 1e-6, so that solver tolerance is excluded
      as an explanation for any deficit in the displacement comparison - the
      handover records six orders of `ksp_rtol` changing nothing beyond the
      last digit, and this is what makes that statement checkable;
    * `block0_max_it` 200 rather than 60, because A2's anisotropic lithosphere
      puts the condensed `[u, psi]` sweep in the 174-388 band and a cap of 60
      would bind on every application.

    Condensation removes `m` from the mixed space entirely
    (`self_gravitating_gia_space`: `spaces = [V] + [] + [Psi]`), so the fields
    are `u = 0, psi = 1` and there is no field 2; `condensed=True` is what
    selects that sweep. Getting it backwards puts GAMG on the wrong block and
    silently mis-splits, which is the failure B2's docstring records costing it
    a run - and `SelfGravitatingGIASolver._check_block0_split_matches_layout`
    now refuses the mismatch rather than letting it run to the cap.
    """
    return selfgrav_dtn_iterative_solver_parameters(
        condensed=True, outer_rtol=outer_rtol, block0_rtol=block0_rtol,
        block0_max_it=block0_max_it, u_pc=u_pc)


BLOCK0 = {
    # The 2-D development default. Its own docstring says it "has no 3-D
    # successor and is not a fallback there", and B1 is where that bites.
    "lu": {"dtn_fieldsplit_0_ksp_type": "preonly",
           "dtn_fieldsplit_0_pc_type": "python",
           "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
           "dtn_fieldsplit_0_assembled_pc_type": "lu",
           "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps"},
    "ilu": {"dtn_fieldsplit_0_ksp_type": "gmres",
            "dtn_fieldsplit_0_ksp_rtol": 1e-8,
            "dtn_fieldsplit_0_ksp_max_it": 2000,
            "dtn_fieldsplit_0_ksp_converged_reason": None,
            "dtn_fieldsplit_0_pc_type": "python",
            "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
            "dtn_fieldsplit_0_assembled_pc_type": "bjacobi",
            "dtn_fieldsplit_0_assembled_sub_pc_type": "ilu"},
}


def build_solver(parent, sub, nmax, dtn_degree=5, rotation=False,
                 drop_ice_from_poisson=False, solver_parameters=None,
                 solver_parameters_extra=None, ivdeg=1, block0=None,
                 near_nullspace=True, declare_nullspace=True,
                 outer_rtol=1e-10, block0_max_it=200, condense=False,
                 cmb_buoyancy="core", rigid_core=False,
                 bulk_shear_ratio=BULK_SHEAR_RATIO):
    sigma_n = cap_sigma_hat(nmax)
    sigma_parent = load_field(parent, nmax, sigma_n)
    sigma_sub = load_field(sub, nmax, sigma_n)

    gravity_bcs = {
        gen.SURF_OUTER: {"dtn": SphericalDtN(L=dtn_degree)},
        gen.SURF_INNER: {"dtn": SphericalDtN(L=dtn_degree)},
    }
    if not drop_ice_from_poisson:
        # Trap 1: omitting this leaves U perfect and reverses the sign of N.
        gravity_bcs[gen.SURF_RE] = {"interior_sigma": sigma_parent}

    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        self_gravity_number=LAMBDA, internal_variable_degree=ivdeg,
        condense_internal_variables=condense)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    rho = layered(sub, 2, "density")
    mu = layered(sub, 3, "shear_modulus")
    eta = layered(sub, 4, "viscosity")
    Xm = SpatialCoordinate(sub)
    rm = sqrt(dot(Xm, Xm))
    g_of_r = refstate.gravity_exact_ufl(rm)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=bulk_shear_ratio, g=g_of_r, B_mu=B_MU,
        self_gravity_number=LAMBDA)

    bcs = {gen.SURF_RE: {"normal_stress": B_MU * sigma_sub}}
    if rigid_core:
        # The rigid-core discriminator: pin the radial displacement at the CMB
        # instead of the fluid-core traction. `un = 0` and a FluidCore cannot
        # both live on Rc, so the solver takes `fluid_core=None` below.
        bcs[gen.SURF_RC] = {"un": 0.0}
    solver_kwargs = {}
    if near_nullspace:
        # **This is currently INERT, and deliberately left in place.**
        # Firedrake composes a `MixedVectorSpaceBasis` onto the *outer* mixed
        # space's field index sets, and `PCFIELDSPLIT` reads it back by
        # querying those index sets. `DtNTwoBlockSchurPC` registers merged
        # index sets of its own, and the nested split inside block 0 builds
        # fresh ones from a sub-DM, so the query finds nothing and the
        # near-nullspace is dropped with no error and no warning. Making it
        # work requires building it from the *block's* own function space
        # inside an `AssembledPC` subclass, which B2's author is implementing;
        # duplicating that here would collide. So: do not tune this, do not
        # treat its presence as a variable, and note that B2's measured
        # mesh-independence was obtained under exactly this condition - i.e.
        # GAMG on smoothed aggregation alone - so the configuration is sound
        # as measured, with an unmeasured improvement still available.
        #
        # Distinct from `nullspace` below, which is NOT inert: that one is
        # attached to the KSP and projects the rigid-rotation kernel out of
        # the residual at the outer level, which is where the kernel lives.
        V = Z.sub(layout.displacement)
        rbm = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
        solver_kwargs["near_nullspace"] = MixedVectorSpaceBasis(
            Z, [rbm if i == layout.displacement else Z.sub(i)
                for i in range(len(Z))])
    # `rho_mantle=None` takes `approximation.density`, which is right for a
    # layered rho_0: the mechanics mesh has only mantle cells, so the facet
    # trace at Rc is already the mantle value. `cmb_buoyancy` selects the spring
    # coefficient: "core" (default, correct) or "contrast" (old, 7.3x too soft,
    # for the baseline arm of the discriminator). `rigid_core` replaces the core
    # entirely with `un = 0`, so there is no FluidCore in that arm.
    core = None if rigid_core else FluidCore(
        boundary=gen.SURF_RC, rho_core=RHO_CORE,
        g=refstate.gravity_exact_ufl(Constant(gen.RC)),
        buoyancy_density=cmb_buoyancy)

    # The rigid rotation is a kernel of the whole coupled operator here: the
    # core is a fluid one, the surface carries a traction, and nothing fixes a
    # rotation of the mantle about the centre. It is annihilated only to ~2e-06
    # discretely, so MUMPS survives it and a Krylov method does not.
    nullspace = (rigid_rotation_nullspace(Z, layout) if declare_nullspace
                 else None)

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=DT_ELASTIC, bcs=bcs, fluid_core=core,
        # A3's constants, imported. The literals that used to stand here were
        # `C = 72.226893` (a rounded copy of 72.2269347) and
        # `C_minus_A = 2.362822e-01`, which is the SECONDARY prescribed value;
        # the primary is the k_s-consistent 2.4214e-01, i.e. `C_MINUS_A["ks"]`.
        rotation_moments={"C": C_NONDIM,
                          "C_minus_A": C_MINUS_A_PRIMARY},
        Omega_sq=OMEGA_SQ,
        nullspace=nullspace, transpose_nullspace=nullspace, **solver_kwargs,
        # B2's coupled iterative solver, imported rather than pasted so the two
        # cannot drift. `--block0` still selects one of the two local
        # single-block fallbacks, both of which failed in 3-D and are kept only
        # so that failure stays reproducible.
        solver_parameters=(solver_parameters
                           or (None if block0 else
                               condensed_solver_parameters(
                                   outer_rtol=outer_rtol,
                                   block0_max_it=block0_max_it)
                               if condense else b2_solver_parameters(
                               outer_rtol=outer_rtol,
                               # `jacobi` on the Real block raises
                               # MatGetDiagonal in PCSetUp_Jacobi: the 75 Real
                               # sub-fields have no assembled diagonal. B2's
                               # write-up specifies `none` here and the default
                               # in its factory does not match it.
                               multiplier_pc="none",
                               # B2's factory defaults `u_pc` to
                               # "__main__.RigidBodyAssembledPC". That string
                               # is resolved by petsc4py against the *entry
                               # point's* __main__, so it only works when
                               # gate_b2_solver.py is the script being run.
                               # Imported into any other driver it raises
                               #   ModuleNotFoundError: No module named
                               #   '__main__.RigidBodyAssembledPC'
                               # from PCPythonSetType_PYTHON, surfacing as a
                               # bare `error code 101` out of PCApply. The
                               # class now ships in gadopt, so name it there:
                               # one implementation, and a string that resolves
                               # from any entry point. It is also what makes
                               # the near-nullspace actually reach the
                               # displacement block, which a
                               # MixedVectorSpaceBasis on the outer space does
                               # not (see the near_nullspace comment above).
                               u_pc="gadopt.RigidBodyAssembledPC",
                               block0_max_it=block0_max_it))),
        solver_parameters_extra=({**BLOCK0[block0],
                                  **(solver_parameters_extra or {})}
                                 if block0 else solver_parameters_extra))
    return solver, z, layout, sigma_n, sigma_parent, sigma_sub


# --------------------------------------------------------------------------
# rigid-rotation content, shared by both paths
# --------------------------------------------------------------------------

def rotation_content(u_, tag):
    """|<u, e_i x x>| / (|e_i x x| |u|) for the three rotation generators.

    `demos/gravity/CLAUDE.md`: declaring the nullspace is NOT enough on this
    project's solvers. FGMRES is right-preconditioned, PETSc removes the kernel
    from the right-hand side but not from the preconditioner's output, and when
    the preconditioner is nearly an exact inverse the answer IS the
    preconditioner's output, kernel and all. So the kernel has to be projected
    out after the solve, and the projection's effect has to be *measured* --
    hence this before/after pair rather than a claim that it worked.

    It matters for V and CANNOT matter for U: a rigid rotation `omega x x` has
    `u_r = 0` identically, so it is purely tangential, and it peaks 90 deg from
    its axis. Nothing else on the candidate list has that asymmetry.

    **The AFTER number is not expected to be zero, and reading it as "kernel
    remaining" is wrong.** This is an *L2* overlap, taken through the mass
    matrix; `VectorSpaceBasis.orthogonalize` (and hence
    `project_out_nullspace`) calls PETSc's `MatNullSpace.remove`, which
    projects in the **dof (l2)** inner product. Different inner products, so
    the two do not agree and the residual is a measure of that disagreement
    rather than of any surviving rotation. Measured on a P2 unit cube: adding
    0.3, 3 and 30 times a rotation generator moves the BEFORE content from
    0.157 to 1.13 and leaves the AFTER content at 1.7366e-02 in every case,
    identical to 2.4e-14. So the sharp test is that AFTER does not move with
    the amount of rotation injected, not that it is small.

    Wrapped by the caller, not here: this is instrumentation and it does not
    get to vote on whether the run succeeded.
    """
    mesh_ = u_.function_space().mesh()
    Xd = SpatialCoordinate(mesh_)
    dm = dx(domain=mesh_)
    norm_u = sqrt(assemble(dot(u_, u_) * dm))
    out = []
    for gen_vec in (as_vector((0.0, -Xd[2], Xd[1])),
                    as_vector((Xd[2], 0.0, -Xd[0])),
                    as_vector((-Xd[1], Xd[0], 0.0))):
        ng = sqrt(assemble(dot(gen_vec, gen_vec) * dm))
        out.append(assemble(dot(u_, gen_vec) * dm) / (ng * norm_u))
    print(f"    rigid-rotation content {tag}: "
          + "  ".join(f"{v:+.3e}" for v in out)
          + f"   (|u| = {norm_u:.6e})")
    return out


def l2_orthogonalise_rotations(u_):
    """Remove the L2 projection of `u` onto the three rotation generators.

    `assemble(dot(u, m) * dx) / assemble(dot(m, m) * dx)` per generator, then
    subtract.  Returns a NEW `Function`; `u` is untouched, so the raw and the
    corrected answer can both be reported and the effect measured rather than
    assumed.

    **Why this and not `project_out_nullspace`.** They are different inner
    products. `VectorSpaceBasis.orthogonalize` calls PETSc `MatNullSpace.remove`,
    which projects in the **dof (l2)** inner product; every quantity this
    driver reports is an **L2** integral. Measured on a P2 unit cube: injecting
    0.3, 3 and 30 times a rotation generator moves the L2 content from 0.157 to
    1.13 and leaves the post-`orthogonalize` content at 1.7366e-02 in all four
    cases, identical to 2.4e-14. So the l2 projection removes the kernel
    exactly and the L2 residue never reaches zero, and the sharp test on it is
    amplitude-independence, not smallness.

    **Expected to change almost nothing here, and that is not a null result to
    be surprised by.** A reviewer proposed this residue as the explanation for
    the tangential deficit; it cannot be. The measured content is 3.937e-04 at
    |u| = 6.053776e-06 (`NOTES/IMPL-LOG-SPADA-B1-ELASTIC.md:552-555`), so the
    L2 rotational component is 2.38e-09 against a non-dimensional reference max
    V of 7.16/2.891e6 = 2.48e-06 -- **three orders too small**. This is an
    instrument correction. It is worth making because it is cheap and because
    "we removed it and nothing moved" is a measurement, while "it is too small
    to matter" is an argument.
    """
    V_ = u_.function_space()
    mesh_ = V_.mesh()
    dm = dx(domain=mesh_)
    Xd = SpatialCoordinate(mesh_)
    out = Function(V_).assign(u_)
    for gen_vec in (as_vector((0.0, -Xd[2], Xd[1])),
                    as_vector((Xd[2], 0.0, -Xd[0])),
                    as_vector((-Xd[1], Xd[0], 0.0))):
        m_ = Function(V_).interpolate(gen_vec)
        denom = assemble(dot(m_, m_) * dm)
        if denom > 0.0:
            coeff = assemble(dot(out, m_) * dm) / denom
            out.assign(out - coeff * m_)
    return out


def series_from(coeffs, theta, nmin=0, kind="P"):
    """`series_at`/`dseries_at` restricted to degrees >= `nmin`.

    **The model's series and the reference's do not span the same degrees, and
    that is a defect, not a convention.** `series_at` sums from n = 0 and
    `dseries_at` from n = 1 (`P[0] = 1`, `dP[1] = -sin theta`), while the load
    is built over `range(2, nmax + 1)` and the reference over
    `np.arange(2, nmax + 1)`. So any degree-0 or degree-1 content in the model
    goes straight into U(0), U(180) and max V with **nothing on the reference
    side to match it**.

    ## Why this is the first thing to look at for the tangential anomaly

    The recorded anomaly has two features, and only one of them has ever been
    discussed: max V at 0.057 of reference, *and its peak at 71.5 deg instead
    of 8.78*. A structural mis-partition of the radial/tangential response --
    locking, the Nitsche pair -- scales V; it does not move the peak to
    mid-latitudes. A peak in the 70s means the tangential field is dominated by
    very low degree content.

    The mechanism is a **soft mode, not a kernel**. With a fluid core instead
    of `un = 0`, a rigid translation of the mantle costs no elastic energy and
    is resisted only by the CMB buoyancy spring, stiffness ~ B_mu drho g. Only
    the rigid *rotation* nullspace is declared. That is the eps/eps case the
    trap list warns about: near-null because of the physics rather than by
    construction, with forcing near-orthogonal to it for the same reason, so a
    small stiffness does not imply small contamination. Consistent with the
    earlier measurement that translation is not a kernel here
    (`a(translation, w)/scale = 1.0e-02` against rotation's 8.3e-13) -- that IS
    what a soft mode looks like.

    And the earlier ruling that translation "contaminates U, not V" was half
    right. `u = c zhat` has `u_r = c cos theta` **and**
    `u_theta = -c sin theta`, and `-sin theta` is exactly `dP_1/dtheta`, so it
    lands entirely in V_1 and peaks at 90 deg. Mixed with the genuine
    degree-two response the peak lands in the 70s.

    ## Pre-commitment, so that either outcome is informative

    If the translation soft mode is the story: `|V_1|` dominates the series and
    max V recomputed over n >= 2 moves from 0.057 toward **0.7-1.3**, with the
    peak returning toward **9 deg**.

    **Falsifier:** `V_1` negligible. Then the degree-one story is dead, both
    the amplitude and the peak location survive, and a structural mis-partition
    becomes the leading candidate -- while still not explaining the peak, which
    would then need its own hypothesis.

    Both the full-series and the n >= 2 values are reported. Neither replaces
    the other: the full series is what the model actually predicts, the n >= 2
    restriction is what the reference is comparable to.
    """
    c = np.array(coeffs, dtype=float, copy=True)
    c[:nmin] = 0.0
    return (series_at(c, theta) if kind == "P" else dseries_at(c, theta))


def report_tangential(U_n, V_n, theta, theta_fine, Vf, imax, label="",
                      N_n=None):
    """max V raw and over n >= 2, the peak locations, and the degree-0/1 terms.

    Wrapped by the caller; see `series_from` for what the numbers mean and what
    was predicted before they were taken.

    ## What the degree-0 and degree-1 terms mean, so a null is not a surprise

    **`N_1` is the geocentre and is frame-dependent.** The reference is defined
    without degrees 0 and 1 precisely to dispose of the frame question, so an
    unmatched `N_1` here is a statement about which frame the discrete problem
    settled into, not an error. It is printed because it is free and because a
    large one would say the frame is not the one anybody assumed.

    **`U_0` is the breathing mode, and it is expected to be small: `|U_0|/|U_2|`
    below ~0.1.** A uniform radial expansion is *not* soft the way a
    translation is -- it costs bulk energy at O(K), the CMB spring resists it
    at O(B_mu drho g), and the load carries no degree-0 content at all. What
    keeps it from being identically zero is that the eliminated fluid core has
    no pressure degree of freedom, so degree-0 motion of the CMB is a genuine
    degree of freedom of the reduced problem, and the exterior monopole term
    closes a weak loop on it.

    **Falsifier: `|U_0|/|U_2|` of order 1 or more.** That would not be noise.
    It would be a defect in the monopole / core-mass budget, and it would be
    the first 3-D measurement of that channel.
    """
    print(f"\n    TANGENTIAL / LOW-DEGREE DIAGNOSTIC {label}")
    print("    Model series span n >= 0 (U) and n >= 1 (V); the reference and")
    print("    the load span n >= 2. Degree 0 and 1 therefore enter the model's")
    print("    numbers unmatched. Predicted, if the translation soft mode is")
    print("    the story: |V_1| dominates, max V over n >= 2 rises toward")
    print("    0.7-1.3 of reference and its peak returns toward 9 deg.")
    print(f"      U_0 {U_n[0]:+.6e}   U_1 {U_n[1]:+.6e}")
    print(f"      V_1 {V_n[1]:+.6e}   "
          f"|V_1| / max|V_n, n>=2| = "
          f"{abs(V_n[1]) / max(np.abs(V_n[2:]).max(), 1e-300):.4f}")
    if N_n is not None:
        print(f"      N_0 {N_n[0]:+.6e}   N_1 {N_n[1]:+.6e}"
              f"   (N_1 is the geocentre: frame-dependent, not an error)")
    # The breathing-mode ratio, with its falsifier stated in the docstring.
    ratio_u0 = abs(U_n[0]) / max(abs(U_n[2]), 1e-300)
    print(f"      |U_0| / |U_2| = {ratio_u0:.4f}   "
          + ("expected < 0.1" if ratio_u0 < 0.1 else
             "*** >= 0.1: see the falsifier in report_tangential's docstring; "
             "of order 1 means the monopole/core-mass budget, not noise ***"))
    for nmin, tag in ((1, "full series (n >= 1)"), (2, "n >= 2 only")):
        vv = series_from(V_n, theta_fine, nmin=nmin, kind="dP")
        j = int(np.argmax(vv))
        print(f"      max V, {tag:<22s} {vv[j]:+10.4f} m   "
              f"ratio {vv[j] / Vf[imax]:+7.4f}   at "
              f"{np.rad2deg(theta_fine[j]):6.2f} deg  (ref "
              f"{np.rad2deg(theta_fine[imax]):.2f})")
    for nmin, tag in ((0, "full series (n >= 0)"), (2, "n >= 2 only")):
        u0, u180 = series_from(U_n, theta, nmin=nmin, kind="P")
        print(f"      U(0), U(180), {tag:<22s} {u0:+10.4f}  {u180:+10.4f} m")


def diagnostic(fn, *args):
    """Run a diagnostic and return a string on failure instead of raising.

    `b4_polar_motion.diagnostic`, same contract and for the same reason: **a
    diagnostic must never be able to kill the measurement it decorates.**
    Three production runs were lost to instrumentation raising *after* the
    physics had been computed.
    """
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001 - the point is to swallow anything
        return f"UNAVAILABLE ({type(exc).__name__}: {exc})"
    return out


# --------------------------------------------------------------------------
# The units-and-magnitude check.  NOT B1.
# --------------------------------------------------------------------------

def run_mechanics_only(nmax, nproj, sig_dim, ref, U_ref, Vf, imax,
                       theta, theta_fine, args):
    """Mechanics alone, self-gravity OFF, rigid core.  **This is not B1.**

    B1 is blocked on a 3-D solver for the coupled block 0 (see the log). What
    can still be checked without one is everything B1 shares with an ordinary
    GIA run: that the cap load is wired to the right boundary with the right
    sign, that the non-dimensionalisation and B_mu carry it to metres, and that
    the resulting U-family is the right *size*. It is deliberately not the B1
    number: switching self-gravity off removes the stiffening of exactly the
    low degrees that dominate U(0) (108% low-degree per §11), so the amplitude
    here is expected LARGER than the coupled one, and the ratio to the
    reference is an upper bound on B1's rather than an estimate of it.

    The solver is the segregated `InternalVariableSolver` on the mantle alone,
    with `un = 0` at Rc: 335 751 dofs of CG2 displacement, CG + GAMG, which is
    the configuration `tests/3d_spada` already runs and which the locking probe
    measured as well behaved.
    """
    from gadopt import InternalVariableSolver, rigid_body_modes

    print("\n  MECHANICS-ONLY CHECK - NOT B1.  Self-gravity off, rigid core.")
    print("  Confirms the load wiring, the units and B_mu against the")
    print("  benchmark's magnitude.  Expect the U-family LARGER than the")
    print("  coupled answer, because self-gravity stiffens the low degrees")
    print("  that dominate U(0).  This is an upper bound, not B1.")

    parent, sub, _, _ = build_meshes(args.configuration, h=args.h,
                                     untangle=args.untangle,
                                     curve_enabled=not args.no_curve)
    sigma_n = cap_sigma_hat(nmax)
    sigma_sub = load_field(sub, nmax, sigma_n)

    rho = layered(sub, 2, "density")
    mu = layered(sub, 3, "shear_modulus")
    eta = layered(sub, 4, "viscosity")
    Xm = SpatialCoordinate(sub)
    rm = sqrt(dot(Xm, Xm))
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=BULK_SHEAR_RATIO,
        g=refstate.gravity_exact_ufl(rm), B_mu=B_MU)

    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    u = Function(V, name="displacement")
    m = Function(S, name="internal variable")
    # FGMRES, not CG, and `AssembledPC`, not `SPDAssembledPC`: the `un = 0`
    # branch of `viscosity_term` adds a Nitsche pair that is *not* symmetric
    # (NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md), so CG fails here with
    # DIVERGED_INDEFINITE_PC at 277 iterations - measured, not anticipated.
    # **This dictionary had drifted, and it is the only place in the tree that
    # had.** It wrote five of the six shipped GAMG settings out by hand and
    # omitted `pc_gamg_mis_k_minimum_degree_ordering`, so arm E aggregated in a
    # different order from every other path in the project - the coupled solve
    # it is meant to be compared against included. Nothing said so, because a
    # missing GAMG option is a different preconditioner and not an error. It
    # now takes the shared settings, which is the whole point of their having
    # one definition.
    params = {"mat_type": "matfree", "snes_type": "ksponly",
              "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_max_it": 5000,
              "ksp_converged_reason": None,
              "pc_type": "python", "pc_python_type": "firedrake.AssembledPC",
              **gamg_parameters("assembled_")}
    # One basis object, kept, because it is needed AFTER the solve as well as
    # during it. `InternalVariableSolver` has no `project_out_nullspace()` of
    # its own -- only `SelfGravitatingGIASolver` does -- so the projection is
    # done here, explicitly, with the same `VectorSpaceBasis.orthogonalize`
    # that `gadopt.gia_gravity` uses.
    rbm = rigid_body_modes(V, rotational=True)
    solver = InternalVariableSolver(
        u, approx, dt=DT_ELASTIC, internal_variables=[m],
        bcs={gen.SURF_RE: {"normal_stress": B_MU * sigma_sub},
             gen.SURF_RC: {"un": 0.0}},
        solver_parameters=params,
        nullspace=rbm,
        transpose_nullspace=rigid_body_modes(V, rotational=True))
    print(f"    displacement dofs {V.dim()}")
    t0 = time.perf_counter()
    solver.solve()
    print(f"    solved in {time.perf_counter() - t0:.1f}s, "
          f"KSP its {solver.solver.snes.ksp.getIterationNumber()}")

    # **Declaring the kernel is not projecting it out**, and this path declared
    # and never projected -- so the tangential column below carried an
    # arbitrary rotational component and nothing said so. Measured before and
    # after rather than asserted: in the coupled run the content was 3.9e-04
    # and the projection changed nothing, so a small number here is a result,
    # not a reason to skip the step.
    print("    rigid rotation is a kernel here: the load is a traction and "
          "un = 0 at Rc pins")
    print("    no rotation about the centre. FGMRES is right-preconditioned, "
          "so it survives the solve.")
    diagnostic(rotation_content, u, "BEFORE orthogonalize")
    rbm.orthogonalize(u)
    diagnostic(rotation_content, u, "AFTER  orthogonalize")

    ds_sub = ds(gen.SURF_RE, domain=sub)

    def project_both(u_field):
        """(U_n, V_n) in metres for a given displacement field."""
        e_th = as_vector((Xm[2] * Xm[0], Xm[2] * Xm[1],
                          -(Xm[0]**2 + Xm[1]**2)))
        rho_c = sqrt(Xm[0]**2 + Xm[1]**2)
        u_th = dot(u_field, e_th) / (rm * conditional(
            rho_c > 1e-12, rho_c, Constant(1e-12)))
        return (project_surface(dot(u_field, Xm / rm), sub, nproj, ds_sub,
                                interior=False,
                                quad_degree=args.quad_degree) * D_M,
                project_surface(u_th, sub, nproj, ds_sub, interior=False,
                                basis="dP",
                                quad_degree=args.quad_degree) * D_M)

    U_n, V_n = project_both(u)
    # Both, always: the effect of the correction is measured, not assumed.
    # See `l2_orthogonalise_rotations` -- expected to move almost nothing, for
    # a reason that is written down there and is worth checking rather than
    # trusting.
    u_orth = diagnostic(l2_orthogonalise_rotations, u)
    if isinstance(u_orth, Function):
        U_o, V_o = project_both(u_orth)
        vr = dseries_at(V_n, theta_fine)
        vo = dseries_at(V_o, theta_fine)
        print(f"    max V raw           {vr.max():+10.4f} m at "
              f"{np.rad2deg(theta_fine[int(np.argmax(vr))]):6.2f} deg")
        print(f"    max V L2-orthog.    {vo.max():+10.4f} m at "
              f"{np.rad2deg(theta_fine[int(np.argmax(vo))]):6.2f} deg   "
              f"(change {abs(vo.max() - vr.max()) / max(abs(vr.max()), 1e-300):.3e})")
    diagnostic(report_tangential, U_n, V_n, theta, theta_fine, Vf, imax,
               "(mechanics only)")

    U0, U180 = series_at(U_n, theta)
    Vmod = dseries_at(V_n, theta_fine)
    jmax = int(np.argmax(Vmod))
    print(f"    U(0)   {U0:+10.4f} m   ref {U_ref[0]:+10.4f}   "
          f"ratio {U0 / U_ref[0]:+7.4f}")
    print(f"    U(180) {U180:+10.4f} m   ref {U_ref[1]:+10.4f}   "
          f"ratio {U180 / U_ref[1]:+7.4f}")
    print(f"    max V  {Vmod[jmax]:+10.4f} m   ref {Vf[imax]:+10.4f}   "
          f"ratio {Vmod[jmax] / Vf[imax]:+7.4f}   at "
          f"{np.rad2deg(theta_fine[jmax]):.2f} deg")

    total = assemble(u_r * u_r * ds_sub)
    Ps = legendre_ufl(nproj, Xm[2] / rm)
    axi = sum((U_n[nn] / D_M) ** 2 * assemble(Ps[nn] * Ps[nn] * ds_sub)
              for nn in range(nproj + 1))
    print(f"    axisymmetry residual {np.sqrt(max(total - axi, 0.0) / total):.4e}")
    print("\n    per-degree ratio (mechanics only, no self-gravity)")
    # **The per-degree V ratio has never been printed.** Without it there is no
    # way to tell a wrong radial/tangential split (V_n uniformly off, U_n fine)
    # from cancellation between individually sane degrees (both fine per degree,
    # the sum wrong), and those two have different causes. `lbar` is the
    # tangential Love number and V_n's basis is dP_n/dtheta, so the reference
    # coefficient is the same `c` as for U with `lbar` in place of `hbar`.
    hbar, lbar, _ = ref.love_time(0.0, nmax)
    n = np.arange(2, nmax + 1)
    c = 3.0 / taboo.RHO_BAR * sig_dim[2:nmax + 1] / (2 * n + 1)
    print("        n      U_n model       U_n ref    ratio  "
          "     V_n model       V_n ref    ratio")
    for i, nn in enumerate(n):
        ur, vr_ = c[i] * hbar[i], c[i] * lbar[i]
        print(f"      {nn:3d}  {U_n[nn]:+.6e}  {ur:+.6e} {U_n[nn] / ur:+7.4f}"
              f"   {V_n[nn]:+.6e}  {vr_:+.6e} "
              f"{V_n[nn] / vr_ if vr_ else float('nan'):+7.4f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", default="coarse",
                   choices=list(gen.CONFIGURATIONS))
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--h", type=float, default=None,
                   help="override the lateral spacing, for sub-coarse meshes")
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--setup-only", action="store_true",
                   help="build everything and check the load, but do not solve")
    p.add_argument("--drop-ice-from-poisson", action="store_true",
                   help="trap 1 on purpose, to confirm the sign reversal")
    p.add_argument("--ivdeg", type=int, default=1,
                   help="internal-variable DG degree; 0 shrinks the dominant block")
    p.add_argument("--block0", choices=list(BLOCK0), default=None,
                   help="use a local single-block fallback instead of B2's "
                        "solver; both failed in 3-D and are kept only so that "
                        "failure stays reproducible")
    p.add_argument("--outer-rtol", type=float, default=1e-10,
                   help="outer FGMRES rtol; tight so tolerance is excluded as "
                        "an explanation for any deficit (B4 comparison)")
    p.add_argument("--no-near-nullspace", action="store_true")
    p.add_argument("--no-declare-nullspace", action="store_true",
                   help="rigid rotation IS a kernel with a fluid core and a "
                        "traction surface; not declaring it stalls block 0")
    p.add_argument("--block0-max-it", type=int, default=200)
    p.add_argument("--quad-degree", type=int, default=60,
                   help="quadrature degree for the Legendre projections; "
                        "MUST be set, see project_surface. Justified by "
                        "--quad-sweep, not chosen.")
    p.add_argument("--quad-degrees", type=str,
                   default="20,30,40,50,60,80,100",
                   help="degrees tried by --quad-sweep")
    p.add_argument("--no-curve", action="store_true",
                   help="ARM C: skip the P2 isoparametric remap entirely, on "
                        "parent and submesh. The mesh as gmsh writes it has a "
                        "ZERO tangle census, so this is the clean geometric "
                        "control -- the only arm whose folds were not made by "
                        "curve(). Costs O(h^2) surface error; check_geometry "
                        "refuses above 1% and the straight 0.5Rc DtN sphere "
                        "sits at 7.7e-03, inside it.")
    p.add_argument("--untangle", action="store_true",
                   help="use A2's curve_mesh(untangle=True); resets the P2 "
                        "edge nodes of the 2 tangled cells of 113653. It is "
                        "now actually wired to build_meshes -- it was parsed "
                        "and never read -- but MEASURED IT DOES NOT UNTANGLE: "
                        "it straightens those two cells and tangles two "
                        "neighbours instead. Read the tangle census, not this "
                        "text.")
    p.add_argument("--load-check", action="store_true",
                   help="the two assembles of handoff section 3; no solve")
    p.add_argument("--quad-sweep", action="store_true",
                   help="measure where the projected coefficients stop moving "
                        "in the quadrature degree, against the KNOWN cap "
                        "coefficients. No solve.")
    p.add_argument("--condense", action="store_true",
                   help="statically condense the internal variables out of "
                        "the mixed space; removes 85%% of block 0")
    p.add_argument("--mechanics-only", action="store_true",
                   help="NOT B1: mechanics with self-gravity OFF, to confirm "
                        "the load, the units and B_mu against the benchmark's "
                        "own magnitude while the coupled solver is blocked")
    p.add_argument("--nproj", type=int, default=None,
                   help="degrees projected out of the answer; default 2 n_max")
    p.add_argument("--cmb-buoyancy", choices=["core", "contrast"],
                   default="core",
                   help="CMB buoyancy spring density: 'core' (default, correct "
                        "rho_core) or 'contrast' (old rho_core-rho_0, 7.3x too "
                        "soft) - the baseline arm of the CMB discriminator")
    p.add_argument("--rigid-core", action="store_true",
                   help="replace the fluid core with un=0 at Rc (rigid-core "
                        "discriminator arm; no FluidCore, no buoyancy spring)")
    p.add_argument("--bulk-shear-ratio", type=float, default=BULK_SHEAR_RATIO,
                   help="K/mu everywhere; default 1.9394 (nu=0.28). Large values "
                        "(100->nu=0.495, 1000->nu=0.4998) approach the "
                        "incompressible benchmark, but P2 displacement-only "
                        "mechanics VOLUMETRICALLY LOCKS as nu->0.5")
    args = p.parse_args()

    nmax = NMAX_OF[args.configuration] if args.nmax is None else args.nmax
    nproj = 2 * nmax if args.nproj is None else args.nproj

    provenance(os.path.basename(__file__))
    print("B1 - the elastic snapshot, Spada et al. (2011) cap load at t = 0")
    print("  A SMOKE TEST ON ABSOLUTE MAGNITUDES, NOT A GATE.  nu = 0.28")
    print("  against an incompressible benchmark, so expect the U-family HIGH")
    print("  by 25-45%, most likely ~35%.  U(0) near -27.77 m would be two")
    print("  errors cancelling, not success.  B1 cannot pin B_mu: B_mu scales")
    print("  the mechanics BC but not the Poisson sheet, so it is degenerate")
    print("  with compressibility in all five numbers.")
    print()
    _r = args.bulk_shear_ratio
    _nu = (3 * _r - 2) / (2 * (3 * _r + 1))
    print(f"  bulk_shear_ratio {_r}  =>  nu = {_nu:.4f}"
          + ("  (near-incompressible; watch for volumetric locking)"
             if _r > 10 else ""))
    print(f"  configuration {args.configuration}   n_max {nmax}   "
          f"projection to n {nproj}   DtN L {args.dtn_degree}")
    print(f"  B_mu {B_MU:.6f}   Lambda {LAMBDA:.6f}   "
          f"bulk_shear_ratio {BULK_SHEAR_RATIO} (nu = 0.28)")
    print(f"  dt {DT_ELASTIC} Maxwell times")
    print()

    # --- the load, checked before anything is solved --------------------
    sigma_n_full = taboo.cap_load(512)
    mass = taboo.load_mass(sigma_n_full)
    print("  load")
    print(f"    total cap mass  {mass:.7e} kg  against TABOO's "
          f"3.6071713409e+18  rel {abs(mass - 3.607171340900778e18) / 3.607171340900778e18:.2e}")
    sig0 = float(series_at(taboo.cap_load(nmax), np.array([0.0]))[0])
    print(f"    sigma(0) at n_max = {nmax}: {sig0:.4f} kg/m^2 "
          f"(must be > 0: the ice presses the surface down)")
    assert sig0 > 0, "cap load has the wrong sign at the pole"

    # --- the reference, synthesised at the mesh's own n_max --------------
    ref = taboo.TabooReference()
    theta = np.array([0.0, np.pi])
    sig_dim = taboo.cap_load(nmax)
    U_ref, V_ref, N_ref = ref.synthesise(0.0, theta, sig_dim, nmax=nmax)
    theta_fine = np.linspace(0.0, np.pi, 4001)
    Uf, Vf, Nf = ref.synthesise(0.0, theta_fine, sig_dim, nmax=nmax)
    imax = int(np.argmax(Vf))
    print(f"  reference at n_max = {nmax} (archive n=128 values in brackets)")
    print(f"    U(0)   {U_ref[0]:+9.4f} m   [-27.7739]")
    print(f"    N(0)   {N_ref[0]:+9.4f} m   [+40.1563]")
    print(f"    U(180) {U_ref[1]:+9.4f} m   [ -0.85  ]")
    print(f"    N(180) {N_ref[1]:+9.4f} m   [ +1.36  ]")
    print(f"    max V  {Vf[imax]:+9.4f} m at {np.rad2deg(theta_fine[imax]):.2f} deg"
          f"   [+7.57 at 9.00]")
    print(f"    truncation shift in U(0) vs n=128: "
          f"{U_ref[0] - (-27.7739):+.3f} m  (sign reverses between n_max 20 and 40)")
    print()

    # --- meshes ----------------------------------------------------------
    t0 = time.perf_counter()
    parent, sub, t_parent, t_sub = build_meshes(args.configuration, h=args.h,
                                                untangle=args.untangle,
                                     curve_enabled=not args.no_curve)
    # `num_cells()` is RANK-LOCAL. At 64 ranks the first run of this printed
    # "parent 2242 cells" for a 113 653-cell mesh and looked like the wrong
    # mesh had been read.
    ncp = COMM_WORLD.allreduce(parent.cell_set.size, op=MPI.SUM)
    ncm = COMM_WORLD.allreduce(sub.cell_set.size, op=MPI.SUM)
    print(f"  meshes: parent {ncp} cells ({t_parent:.1f}s), "
          f"mantle {ncm} cells ({t_sub:.1f}s), "
          f"total {time.perf_counter() - t0:.1f}s  [{COMM_WORLD.size} ranks]")

    t0 = time.perf_counter()
    solver, z, layout, sigma_n, sigma_parent, sigma_sub = build_solver(
        parent, sub, nmax, dtn_degree=args.dtn_degree,
        drop_ice_from_poisson=args.drop_ice_from_poisson,
        ivdeg=args.ivdeg, block0=args.block0,
        outer_rtol=args.outer_rtol, block0_max_it=args.block0_max_it,
        near_nullspace=not args.no_near_nullspace, condense=args.condense,
        declare_nullspace=not args.no_declare_nullspace,
        cmb_buoyancy=args.cmb_buoyancy, rigid_core=args.rigid_core,
        bulk_shear_ratio=args.bulk_shear_ratio)
    print(f"  solver built in {time.perf_counter() - t0:.1f}s; "
          f"mixed space {z.function_space().dim()} dofs in "
          f"{len(z.subfunctions)} fields")
    dims = {"u": z.subfunctions[layout.displacement].function_space().dim(),
            "m": sum(z.subfunctions[i].function_space().dim()
                     for i in layout.internal_variables),
            "psi": z.subfunctions[layout.potential].function_space().dim()}
    print(f"    block 0 = u {dims['u']} + m {dims['m']} + psi {dims['psi']} "
          f"= {sum(dims.values())} dofs, plus {len(layout.multipliers)} Real")

    # The load must be the same object on both meshes to the surface's own
    # accuracy; the docstring's three-uses warning is about exactly this.
    ds_sub = ds(gen.SURF_RE, domain=sub)
    dS_par = solver.form.dS(gen.SURF_RE)
    m_sub = assemble(sigma_sub * ds_sub)
    m_par = assemble(avg(sigma_parent) * dS_par)
    print(f"    load surface integral: mantle {m_sub:.6e}  parent {m_par:.6e}  "
          f"rel {abs(m_sub - m_par) / abs(m_sub):.2e}")

    if args.load_check:
        # **The two assembles, no solve.** Handoff §3: the load appears in
        # three places; the sheet and the inertia row both read
        # `layout.gravity_form.sigma_bcs` and cannot drift, but the mechanics
        # `normal_stress` is the caller's and is NOT checked. The prediction
        # under test is that they differ by ~2.17.
        #
        # The sheet side is read back OUT of `sigma_bcs` rather than from the
        # object passed in, so this verifies what the solver actually stored.
        # Degree-2 weighted, because a difference in shape rather than scale
        # would not show in the totals.
        parent, sub, _, _ = build_meshes(args.configuration, h=args.h,
                                         untangle=args.untangle,
                                     curve_enabled=not args.no_curve)
        solver, z, layout, sigma_n, sigma_parent, sigma_sub = build_solver(
            parent, sub, nmax, dtn_degree=args.dtn_degree, ivdeg=args.ivdeg,
            block0=args.block0, condense=args.condense)
        qd = args.quad_degree
        Xs = SpatialCoordinate(sub)
        P_sub = legendre_ufl(2, Xs[2] / sqrt(dot(Xs, Xs)))[2]
        Xp = SpatialCoordinate(parent)
        P_par = legendre_ufl(2, Xp[2] / sqrt(dot(Xp, Xp)))[2]
        ds_s = ds(gen.SURF_RE, domain=sub,
                  metadata={"quadrature_degree": qd})

        mech = assemble(B_MU * sigma_sub * P_sub * ds_s)
        print("\n  THE TWO ASSEMBLES (no solve)")
        print(f"    mechanics  int B_mu sigma P_2 dS = {mech:+.10e}")
        print(f"    (the raw traction, before B_mu: {mech / B_MU:+.10e})")
        for bc_id, expr, integral_type in layout.gravity_form.sigma_bcs:
            meas = (layout.gravity_form.dS if integral_type == "interior_facet"
                    else layout.gravity_form.ds)
            m = meas(bc_id, metadata={"quadrature_degree": qd})
            val = assemble((avg(expr * P_par) if integral_type == "interior_facet"
                            else expr * P_par) * m)
            print(f"    sheet      int sigma P_2 dS = {val:+.10e}   "
                  f"(tag {bc_id}, {integral_type}, read from sigma_bcs)")
            print(f"      ratio mechanics/sheet          {mech / val:.8f}")
            print(f"      ratio (mechanics/B_mu)/sheet   {mech / B_MU / val:.8f}")
        print(f"    B_mu = {B_MU:.8f}    prediction under test: 2.17")
        print("    The two roles are MEANT to differ by exactly B_mu: the")
        print("    mechanics sees a traction sigma g_0/mu_bar = B_mu sigma_hat,")
        print("    the Poisson source a surface density sigma/(rho_bar D).")
        return

    if args.quad_sweep:
        # The projection is validated against a field whose Legendre
        # coefficients are KNOWN exactly: the truncated cap series itself.
        # Interpolating it into CG2 and projecting it back must return sigma_n.
        # Two things are reported per degree: the change from the previous
        # degree (the "stops moving" criterion A3 §12.3a asks for, and the one
        # that decides), and the deviation from the exact sigma_n (which also
        # contains the CG2 interpolation error and therefore plateaus at a
        # nonzero floor rather than at zero).
        parent, sub, _, _ = build_meshes(args.configuration, h=args.h,
                                         untangle=args.untangle,
                                     curve_enabled=not args.no_curve)
        sigma_n = cap_sigma_hat(nmax)
        f = load_field(sub, nmax, sigma_n)
        ds_sub = ds(gen.SURF_RE, domain=sub)
        print("\n  QUADRATURE SWEEP on the Legendre projection")
        print("  Projecting the interpolated cap load back onto P_n and")
        print("  comparing with its exact coefficients. TSFC's own estimate")
        print("  for these integrands is ~500, which OOM-killed a rank at")
        print("  322 GB; the integrand is really degree ~2n.")
        print(f"  n_max {nmax}, projecting to n {nproj}")
        print("    qdeg   max|dcoeff| vs prev   max rel err vs exact sigma_n"
              "   time")
        prev = None
        for qd in [int(x) for x in args.quad_degrees.split(",")]:
            t0 = time.perf_counter()
            c = project_surface(f, sub, nproj, ds_sub, interior=False,
                                quad_degree=qd)
            dt = time.perf_counter() - t0
            scale = np.abs(sigma_n[2:nmax + 1]).max()
            rel = np.abs(c[2:nmax + 1] - sigma_n[2:nmax + 1]).max() / scale
            move = "-" if prev is None else f"{np.abs(c - prev).max() / scale:.3e}"
            print(f"    {qd:4d}   {move:>19s}   {rel:>22.3e}   {dt:6.1f}s")
            prev = c
        print("  Choose the smallest degree at which column 2 has stopped")
        print("  moving; column 3 plateaus at the CG2 interpolation floor.")
        return

    if args.mechanics_only:
        run_mechanics_only(nmax, nproj, sig_dim, ref, U_ref, Vf, imax,
                           theta, theta_fine, args)
        return

    if args.setup_only:
        print("\n  --setup-only: stopping before the solve")
        return

    print()
    t0 = time.perf_counter()
    solver.solve()
    t_solve = time.perf_counter() - t0
    snes = solver.solver.snes
    print(f"  solved in {t_solve:.1f}s   SNES its {snes.getIterationNumber()}  "
          f"reason {snes.getConvergedReason()}  "
          f"KSP its {snes.ksp.getIterationNumber()}")

    # ---- rigid-rotation content, before and after projecting it out -------
    #
    # `demos/gravity/CLAUDE.md`: declaring the nullspace is NOT enough on this
    # solver. FGMRES is right-preconditioned, PETSc removes the kernel from the
    # right-hand side but not from the preconditioner's output, and the
    # preconditioner is nearly an exact inverse, so the answer IS the
    # preconditioner's output, kernel and all. `project_out_nullspace()` after
    # the solve is what removes it.
    #
    # This matters for V and CANNOT matter for U: a rigid rotation `omega x x`
    # has `u_r = 0` identically, so it is purely tangential. It peaks 90 deg
    # from its axis, which is why a contaminated V projects at 0.057 and peaks
    # at 71.5 deg instead of 8.78. Nothing else on the candidate list has that
    # asymmetry - compressibility, resolution, aliasing and a wrong e_theta all
    # move U and V together.
    # `rotation_content` is module level so the mechanics-only path uses the
    # same instrument; it used to be defined here and only here, which is part
    # of why that path was never measured.
    diagnostic(rotation_content, solver.displacement,
               "BEFORE project_out_nullspace")
    if solver.project_out_nullspace():
        print("    project_out_nullspace() removed a declared kernel")
    diagnostic(rotation_content, solver.displacement,
               "AFTER  project_out_nullspace")

    # --- project the answer ---------------------------------------------
    u = solver.displacement
    Xm = SpatialCoordinate(sub)
    rm = sqrt(dot(Xm, Xm))
    u_r = dot(u, Xm / rm)
    U_n = project_surface(u_r, sub, nproj, ds_sub, interior=False,
                          quad_degree=args.quad_degree)

    # V is the *tangential* displacement, not dU/dtheta.  Its natural basis is
    # dP_n/dtheta, whose norm on the sphere is 4 pi R^2 n(n+1)/(2n+1); the
    # measured denominator is used so the surface discretisation cancels.
    e_theta = as_vector((Xm[2] * Xm[0], Xm[2] * Xm[1], -(Xm[0]**2 + Xm[1]**2)))
    rho_cyl = sqrt(Xm[0]**2 + Xm[1]**2)
    u_theta = dot(u, e_theta) / (rm * conditional(rho_cyl > 1e-12, rho_cyl,
                                                  Constant(1e-12)))
    V_n = project_surface(u_theta, sub, nproj, ds_sub, interior=False,
                          basis="dP", quad_degree=args.quad_degree)

    # **Do not use `solver.geoid()` here.** It forms `psi / g_0` from
    # `approximation.g`, and this driver builds that from the *submesh*
    # coordinates because the momentum equation lives on the submesh. `psi`
    # lives on the parent, so the quotient mixes meshes and TSFC rejects it
    # with `MismatchingDomainError: ... requires measure intersection`. The
    # geoid is a parent-side quantity; build it from parent coordinates. The
    # `+` is the sign convention of `gadopt.gravity_solver` (psi is minus the
    # Newtonian potential), and rotation is off in B1 so there is no
    # rotational term to add.
    Xp = SpatialCoordinate(parent)
    geoid = solver.potential / refstate.gravity_exact_ufl(sqrt(dot(Xp, Xp)))
    N_n = project_surface(geoid, parent, nproj, dS_par, interior=True,
                          quad_degree=args.quad_degree)

    U_n *= D_M
    V_n *= D_M
    N_n *= D_M

    # The same two instruments on the coupled answer. `u` here has already had
    # `project_out_nullspace()` applied, which acts in l2; this removes the L2
    # rotational residue, which is a different and smaller thing.
    u_orth = diagnostic(l2_orthogonalise_rotations, u)
    if isinstance(u_orth, Function):
        u_th_o = dot(u_orth, e_theta) / (rm * conditional(
            rho_cyl > 1e-12, rho_cyl, Constant(1e-12)))
        V_o = project_surface(u_th_o, sub, nproj, ds_sub, interior=False,
                              basis="dP", quad_degree=args.quad_degree) * D_M
        vr = dseries_at(V_n, theta_fine)
        vo = dseries_at(V_o, theta_fine)
        print(f"  max V raw        {vr.max():+10.4f} m at "
              f"{np.rad2deg(theta_fine[int(np.argmax(vr))]):6.2f} deg")
        print(f"  max V L2-orthog. {vo.max():+10.4f} m at "
              f"{np.rad2deg(theta_fine[int(np.argmax(vo))]):6.2f} deg   "
              f"(change {abs(vo.max() - vr.max()) / max(abs(vr.max()), 1e-300):.3e})")
    diagnostic(report_tangential, U_n, V_n, theta, theta_fine, Vf, imax,
               "(coupled)", N_n)

    U0, U180 = series_at(U_n, theta)
    N0, N180 = series_at(N_n, theta)
    Vmod = dseries_at(V_n, theta_fine)
    jmax = int(np.argmax(Vmod))

    print()
    print("  the five numbers (model, reference at matched n_max, ratio)")
    print("  LEGACY reconstruction: the model series span n >= 0 (U, N) and")
    print("  n >= 1 (V) while the load and the reference span n >= 2, so these")
    print("  carry unmatched degree-0 and degree-1 content. Printed for")
    print("  continuity with the recorded -61.79 / +39.97 and so on. **The")
    print("  n >= 2 block below is the one commensurate with the reference's")
    print("  own definition; the legacy form is being retired.**")
    rows = [("U(0)", U0, U_ref[0]), ("N(0)", N0, N_ref[0]),
            ("U(180)", U180, U_ref[1]), ("N(180)", N180, N_ref[1])]
    for name, mod, refv in rows:
        print(f"    {name:<7s} {mod:+10.4f} m   {refv:+10.4f} m   "
              f"ratio {mod / refv:+7.4f}")
    print(f"    max V   {Vmod[jmax]:+10.4f} m   {Vf[imax]:+10.4f} m   "
          f"ratio {Vmod[jmax] / Vf[imax]:+7.4f}   "
          f"at {np.rad2deg(theta_fine[jmax]):.2f} vs "
          f"{np.rad2deg(theta_fine[imax]):.2f} deg")

    print()
    print("  THE FIVE NUMBERS FROM n >= 2 (commensurate with the reference)")
    U0b, U180b = series_from(U_n, theta, nmin=2, kind="P")
    N0b, N180b = series_from(N_n, theta, nmin=2, kind="P")
    Vb = series_from(V_n, theta_fine, nmin=2, kind="dP")
    jb = int(np.argmax(Vb))
    for name, mod, refv in [("U(0)", U0b, U_ref[0]), ("N(0)", N0b, N_ref[0]),
                            ("U(180)", U180b, U_ref[1]),
                            ("N(180)", N180b, N_ref[1])]:
        print(f"    {name:<7s} {mod:+10.4f} m   {refv:+10.4f} m   "
              f"ratio {mod / refv:+7.4f}")
    print(f"    max V   {Vb[jb]:+10.4f} m   {Vf[imax]:+10.4f} m   "
          f"ratio {Vb[jb] / Vf[imax]:+7.4f}   "
          f"at {np.rad2deg(theta_fine[jb]):.2f} vs "
          f"{np.rad2deg(theta_fine[imax]):.2f} deg")

    # Separates a wrong load from a wrong response: if the load's own degree-2
    # content is right, a wrong U_2 is the mechanics.
    sig_proj = project_surface(sigma_sub, sub, 4, ds_sub, interior=False,
                               quad_degree=args.quad_degree)
    # The two objects AS THEY ENTER THEIR FORMS: the mechanics sees
    # `B_mu * sigma_hat` as a traction, the Poisson source sees `sigma_hat` as
    # a mass sheet. They are meant to differ by exactly B_mu, which is physics
    # and not a bug: the traction is `sigma g_0 / mu_bar = B_mu sigma_hat`
    # while the sheet is the non-dimensional surface density `sigma/(rho_bar
    # D)`. Reported so the ratio is a measured number rather than an argument.
    mech = assemble(B_MU * sigma_sub * ds_sub)
    sheet = assemble(avg(sigma_parent) * dS_par)
    print()
    print("  the load in its two roles (mechanics traction vs Poisson sheet)")
    print(f"    int B_mu sigma dS (mechanics) {mech:+.8e}")
    print(f"    int sigma dS      (sheet)     {sheet:+.8e}")
    print(f"    ratio {mech / sheet:.6f}   B_mu = {B_MU:.6f}   "
          f"(equal => the two roles differ by exactly B_mu, as intended)")

    print()
    print("  load self-check: the projected load against its own coefficients")
    for nn in (2, 3, 4):
        print(f"    n={nn}  int sigma P_n dS/|P_n|^2 = {sig_proj[nn]:+.8e}"
              f"   sigma_{nn} = {sigma_n[nn]:+.8e}"
              f"   ratio {sig_proj[nn] / sigma_n[nn]:+.6f}")

    print()
    print("  DEGREE-2 AMPLITUDE, reported explicitly for the B4 comparison.")
    print("  B4 has a 23% deficit in |m| whose leading suspect is a solver")
    print("  tolerance on the rotation rows (theta_rot carries Omega^2 =")
    print("  1.566e-03, three orders below the dominant rows).  This is the")
    print("  same degree-2 compliance measured independently, at a tight outer")
    print("  rtol, so tolerance is excluded on this side: if the fractional")
    print("  deficit matches B4's, the two are one finding and it is the")
    print("  compliance; if this is clean, B4's is specific to the rotation rows.")
    hbar2, lbar2, kbar2 = ref.love_time(0.0, 2)
    c2 = 3.0 / taboo.RHO_BAR * sig_dim[2] / 5.0
    print(f"    U_2 model {U_n[2]:+.8e} m   ref {c2 * hbar2[0]:+.8e} m   "
          f"RATIO {U_n[2] / (c2 * hbar2[0]):+.6f}")
    print(f"    N_2 model {N_n[2]:+.8e} m   ref {c2 * kbar2[0]:+.8e} m   "
          f"RATIO {N_n[2] / (c2 * kbar2[0]):+.6f}")
    print(f"    deficit in U_2 vs reference: "
          f"{U_n[2] / (c2 * hbar2[0]) - 1.0:+.4%}")

    print()
    print("  the fingerprint: compressibility moves the U-family together by")
    print("  ~1.35 while N(0) moves ~0.96 the other way")
    print(f"    R_U(0)   {U0 / U_ref[0]:+7.4f}   expected ~1.35")
    print(f"    R_V      {Vmod[jmax] / Vf[imax]:+7.4f}   expected ~1.35")
    print(f"    R_U(180) {U180 / U_ref[1]:+7.4f}   expected ~1.35")
    print(f"    R_N(0)   {N0 / N_ref[0]:+7.4f}   expected ~0.96")
    print(f"    R_N(180) {N180 / N_ref[1]:+7.4f}   expected ~0.86")

    print()
    print("  trap 1: the ice in the Poisson source.  N(0) and N(180) must be")
    print("  POSITIVE; without the sheet they are -4.2598 and -0.5571.")
    print(f"    sign(N(0)) = {np.sign(N0):+.0f}   sign(N(180)) = {np.sign(N180):+.0f}")

    # --- the per-degree ratio, worth more than the five numbers ----------
    hbar, lbar, kbar = ref.love_time(0.0, nmax)
    n = np.arange(2, nmax + 1)
    c = 3.0 / taboo.RHO_BAR * sig_dim[2:nmax + 1] / (2 * n + 1)
    U_ref_n = c * hbar
    N_ref_n = c * kbar
    print()
    print("  per-degree ratio to the reference coefficients")
    print("  (flat and != 1 = compressibility/B_mu;  1 then rising = mesh;")
    print("   discontinuity at n_max = truncation;  U flat & N moved = trap 1)")
    print("     n     U_n model      U_n ref     ratio      N_n model      "
          "N_n ref     ratio")
    for i, nn in enumerate(n):
        print(f"    {nn:3d}  {U_n[nn]:+.6e}  {U_ref_n[i]:+.6e}  "
              f"{U_n[nn] / U_ref_n[i]:+7.4f}   {N_n[nn]:+.6e}  "
              f"{N_ref_n[i]:+.6e}  {N_n[nn] / N_ref_n[i]:+7.4f}")

    # --- axisymmetry residual, free and reference-independent ------------
    ds_q = ds_sub(metadata={"quadrature_degree": args.quad_degree})
    total = assemble(u_r * u_r * ds_q)
    area = assemble(Constant(1.0) * ds_q)
    axi = 0.0
    Xs = SpatialCoordinate(sub)
    Ps = legendre_ufl(nproj, Xs[2] / sqrt(dot(Xs, Xs)))
    for nn in range(nproj + 1):
        den = assemble(Ps[nn] * Ps[nn] * ds_q)
        axi += (U_n[nn] / D_M) ** 2 * den
    resid = np.sqrt(max(total - axi, 0.0) / total)
    print()
    print(f"  axisymmetry residual  {resid:.4e}   "
          f"(m != 0 content of u_r at Re; the cap is axisymmetric so this is")
    print(f"   pure error, and it must not dominate the {abs(U0 / U_ref[0] - 1):.1%} "
          "amplitude discrepancy)")
    print(f"  surface area at Re    {area:.8e}  vs 4 pi Re^2 "
          f"{4 * np.pi * gen.RE ** 2:.8e}")

    print()
    print(f"  cost: solve {t_solve:.1f}s, outer FGMRES "
          f"{snes.ksp.getIterationNumber()} its, SNES "
          f"{snes.getIterationNumber()} its.  Block-0 and per-split counts are"
          " in the ksp_converged_reason lines above (B2's cost model).")
    print(f"  split map: {SPLIT_NAMES}")


if __name__ == "__main__":
    main()
