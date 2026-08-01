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
                    SphericalDtN)
from gadopt.gia_gravity import (FluidCore, SelfGravitatingGIASolver,  # noqa: E402
                                rigid_rotation_nullspace,
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

def curve(mesh, untangle=False):
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
    """
    if untangle:
        from validate_selfgrav_sphere import curve_mesh  # noqa: PLC0415
        return curve_mesh(mesh, untangle=True)
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    r_p1 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(mesh, "CG", 2)).interpolate((r_p1 / r) * X)
    return Mesh(X_p2)


def build_meshes(configuration, reuse=True, h=None, untangle=False):
    tag = configuration if h is None else f"h{h:g}"
    path = os.path.join(HERE, f"b1_{tag}.msh")
    if not (reuse and os.path.exists(path)):
        gen.generate(path, configuration=configuration, h=h)
    t0 = time.perf_counter()
    parent = curve(Mesh(path), untangle=untangle)
    parent.cartesian = False
    t_parent = time.perf_counter() - t0
    t0 = time.perf_counter()
    sub = curve(Submesh(parent, 3, gen.CELL_MANTLE), untangle=untangle)
    sub.cartesian = False
    t_sub = time.perf_counter() - t0
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
                                u_pc="gate_b2_solver.RigidBodyAssembledPC"):
    """B2's dictionary with the block-0 sweep rewritten for `[u, psi]`.

    `gate_b2_solver.solver_parameters` is documented as "the uncondensed
    three-field block-0 sweep" and hard-codes `fields "1","0","2"` for
    `m, u, psi`. **Condensation removes `m` from the mixed space entirely**
    (`self_gravitating_gia_space`: `spaces = [V] + [] + [Psi]`), so the fields
    become `u = 0, psi = 1` and there is no field 2. Reusing the three-field
    sweep under condensation would put GAMG on the wrong block and silently
    mis-split, which is the exact failure mode B2's own docstring records
    costing it a run.

    Only the sweep is rewritten; the inner preconditioners are B2's, unchanged
    - `RigidBodyAssembledPC` on `u` (fully qualified, see below) and
    `SPDAssembledPC` on `psi`. Tuning the `[u, psi]` diagonal blocks is the
    PETSc round-three task's, not this one's.
    """
    from gate_b2_solver import GAMG, _prefixed
    p = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-15,
        "snes_rtol": 1e-4,
        "snes_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_rtol": outer_rtol,
        "ksp_max_it": 200,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        "dtn_pc_fieldsplit_schur_fact_type": "full",
        "dtn_fieldsplit_0_ksp_type": "fgmres",
        "dtn_fieldsplit_0_ksp_rtol": block0_rtol,
        "dtn_fieldsplit_0_ksp_max_it": block0_max_it,
        "dtn_fieldsplit_0_ksp_converged_reason": None,
        "dtn_fieldsplit_0_pc_type": "fieldsplit",
        "dtn_fieldsplit_0_pc_fieldsplit_type": "multiplicative",
        "dtn_fieldsplit_1_ksp_type": "gmres",
        "dtn_fieldsplit_1_ksp_rtol": 1e-4,
        "dtn_fieldsplit_1_ksp_max_it": 200,
        "dtn_fieldsplit_1_ksp_converged_reason": None,
        "dtn_fieldsplit_1_pc_type": "none",
    }

    def inner(pc_python):
        d = {"ksp_type": "preonly", "pc_type": "python",
             "pc_python_type": pc_python, "ksp_converged_reason": None}
        d.update(_prefixed(GAMG, "assembled_"))
        return d

    for split, (field, opts) in enumerate([(0, inner(u_pc)),
                                           (1, inner("gadopt.SPDAssembledPC"))]):
        p[f"dtn_fieldsplit_0_pc_fieldsplit_{split}_fields"] = str(field)
        p.update(_prefixed(opts, f"dtn_fieldsplit_0_fieldsplit_{split}_"))
    return p


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
                 outer_rtol=1e-10, block0_max_it=200, condense=False):
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
        bulk_shear_ratio=BULK_SHEAR_RATIO, g=g_of_r, B_mu=B_MU,
        self_gravity_number=LAMBDA)

    bcs = {gen.SURF_RE: {"normal_stress": B_MU * sigma_sub}}
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
    # trace at Rc is already the mantle value.
    core = FluidCore(boundary=gen.SURF_RC, rho_core=RHO_CORE,
                     g=refstate.gravity_exact_ufl(Constant(gen.RC)))

    # The rigid rotation is a kernel of the whole coupled operator here: the
    # core is a fluid one, the surface carries a traction, and nothing fixes a
    # rotation of the mantle about the centre. It is annihilated only to ~2e-06
    # discretely, so MUMPS survives it and a Krylov method does not.
    nullspace = (rigid_rotation_nullspace(Z, layout) if declare_nullspace
                 else None)

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=DT_ELASTIC, bcs=bcs, fluid_core=core,
        rotation_moments={"C": 72.226893, "C_minus_A": 2.362822e-01},
        Omega_sq=1.566176e-03,
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
                               # bare `error code 101` out of PCApply. Naming
                               # the module explicitly is the fix, and it also
                               # makes the near-nullspace actually reach the
                               # displacement block, which a
                               # MixedVectorSpaceBasis on the outer space does
                               # not (see the near_nullspace comment above).
                               u_pc="gate_b2_solver.RigidBodyAssembledPC",
                               block0_max_it=block0_max_it))),
        solver_parameters_extra=({**BLOCK0[block0],
                                  **(solver_parameters_extra or {})}
                                 if block0 else solver_parameters_extra))
    return solver, z, layout, sigma_n, sigma_parent, sigma_sub


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

    parent, sub, _, _ = build_meshes(args.configuration, h=args.h)
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
    params = {"mat_type": "matfree", "snes_type": "ksponly",
              "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_max_it": 5000,
              "ksp_converged_reason": None,
              "pc_type": "python", "pc_python_type": "firedrake.AssembledPC",
              "assembled_pc_type": "gamg",
              "assembled_mg_levels_pc_type": "sor",
              "assembled_pc_gamg_threshold": 0.01,
              "assembled_pc_gamg_square_graph": 100,
              "assembled_pc_gamg_coarse_eq_limit": 1000}
    solver = InternalVariableSolver(
        u, approx, dt=DT_ELASTIC, internal_variables=[m],
        bcs={gen.SURF_RE: {"normal_stress": B_MU * sigma_sub},
             gen.SURF_RC: {"un": 0.0}},
        solver_parameters=params,
        nullspace=rigid_body_modes(V, rotational=True),
        transpose_nullspace=rigid_body_modes(V, rotational=True))
    print(f"    displacement dofs {V.dim()}")
    t0 = time.perf_counter()
    solver.solve()
    print(f"    solved in {time.perf_counter() - t0:.1f}s, "
          f"KSP its {solver.solver.snes.ksp.getIterationNumber()}")

    ds_sub = ds(gen.SURF_RE, domain=sub)
    u_r = dot(u, Xm / rm)
    U_n = project_surface(u_r, sub, nproj, ds_sub, interior=False,
                          quad_degree=args.quad_degree) * D_M
    e_theta = as_vector((Xm[2] * Xm[0], Xm[2] * Xm[1], -(Xm[0]**2 + Xm[1]**2)))
    rho_cyl = sqrt(Xm[0]**2 + Xm[1]**2)
    u_theta = dot(u, e_theta) / (rm * conditional(rho_cyl > 1e-12, rho_cyl,
                                                  Constant(1e-12)))
    V_n = project_surface(u_theta, sub, nproj, ds_sub, interior=False,
                          basis="dP", quad_degree=args.quad_degree) * D_M

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
    hbar, _, _ = ref.love_time(0.0, nmax)
    n = np.arange(2, nmax + 1)
    c = 3.0 / taboo.RHO_BAR * sig_dim[2:nmax + 1] / (2 * n + 1)
    for i, nn in enumerate(n):
        print(f"      n={nn:3d}  U_n {U_n[nn]:+.6e}  ref {c[i] * hbar[i]:+.6e}"
              f"  ratio {U_n[nn] / (c[i] * hbar[i]):+7.4f}")


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
    p.add_argument("--untangle", action="store_true",
                   help="use A2's curve_mesh(untangle=True); resets the P2 "
                        "edge nodes of the 2 tangled cells of 113653")
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
    args = p.parse_args()

    nmax = NMAX_OF[args.configuration] if args.nmax is None else args.nmax
    nproj = 2 * nmax if args.nproj is None else args.nproj

    print("B1 - the elastic snapshot, Spada et al. (2011) cap load at t = 0")
    print("  A SMOKE TEST ON ABSOLUTE MAGNITUDES, NOT A GATE.  nu = 0.28")
    print("  against an incompressible benchmark, so expect the U-family HIGH")
    print("  by 25-45%, most likely ~35%.  U(0) near -27.77 m would be two")
    print("  errors cancelling, not success.  B1 cannot pin B_mu: B_mu scales")
    print("  the mechanics BC but not the Poisson sheet, so it is degenerate")
    print("  with compressibility in all five numbers.")
    print()
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
    parent, sub, t_parent, t_sub = build_meshes(args.configuration, h=args.h)
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
        declare_nullspace=not args.no_declare_nullspace)
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
        parent, sub, _, _ = build_meshes(args.configuration, h=args.h)
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
        parent, sub, _, _ = build_meshes(args.configuration, h=args.h)
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
    def rotation_content(tag):
        u_ = solver.displacement
        Vd = u_.function_space()
        Xd = SpatialCoordinate(Vd.mesh())
        norm_u = sqrt(assemble(dot(u_, u_) * dx(domain=Vd.mesh())))
        out = []
        for gen_vec in (as_vector((0.0, -Xd[2], Xd[1])),
                        as_vector((Xd[2], 0.0, -Xd[0])),
                        as_vector((-Xd[1], Xd[0], 0.0))):
            dm = dx(domain=Vd.mesh())
            ng = sqrt(assemble(dot(gen_vec, gen_vec) * dm))
            out.append(assemble(dot(u_, gen_vec) * dm) / (ng * norm_u))
        print(f"    rigid-rotation content {tag}: "
              + "  ".join(f"{v:+.3e}" for v in out)
              + f"   (|u| = {norm_u:.6e})")
        return out

    rotation_content("BEFORE project_out_nullspace")
    if solver.project_out_nullspace():
        print("    project_out_nullspace() removed a declared kernel")
    rotation_content("AFTER  project_out_nullspace")

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

    U0, U180 = series_at(U_n, theta)
    N0, N180 = series_at(N_n, theta)
    Vmod = dseries_at(V_n, theta_fine)
    jmax = int(np.argmax(Vmod))

    print()
    print("  the five numbers (model, reference at matched n_max, ratio)")
    rows = [("U(0)", U0, U_ref[0]), ("N(0)", N0, N_ref[0]),
            ("U(180)", U180, U_ref[1]), ("N(180)", N180, N_ref[1])]
    for name, mod, refv in rows:
        print(f"    {name:<7s} {mod:+10.4f} m   {refv:+10.4f} m   "
              f"ratio {mod / refv:+7.4f}")
    print(f"    max V   {Vmod[jmax]:+10.4f} m   {Vf[imax]:+10.4f} m   "
          f"ratio {Vmod[jmax] / Vf[imax]:+7.4f}   "
          f"at {np.rad2deg(theta_fine[jmax]):.2f} vs "
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
