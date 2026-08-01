r"""B4: polar motion for the Spada benchmark, on the A2 parent sphere.

Handoff §11 B4 as corrected, and `NOTES/REVIEW-SPADA-B4-PRECOMMIT.md`, which is
the specification.  Load centroid at theta_c = 25 deg, lambda_c = 75 deg,
Chandler wobble excluded.  Reference series come from `reference.npz`, which A1
extracted from `T02-03/Rot_cwoff_GRID_cap_m.txt`.

Run::

    python3 b4_polar_motion.py --configuration coarse [--epochs 0 1 2 5 10 20]

## What is being tested, and in what order

The gates are ordered by how much they cost and how sharply they fail, not by
how interesting they are.

1. **Handedness**, which is the silent failure.  A sign flip in `CLOSURE_SIGNS`,
   in the psi_rot convention or in dI gives `dphi = +-180 deg`, and **|m|,
   |mdot| and even the ratio my/mx are all blind to it**.  The only thing that
   catches it is the individual signs, so the first assertion is
   `mx < 0 and my < 0` at t = 0, against mx(0) = -0.0034211,
   my(0) = -0.0127677.
2. **The phase**, which is exact and free.  The reference's phase is
   **-105.0000 deg at all 35 epochs to 14 digits**, equal to the phase of the
   archive's own geometrical factor (Im/Re = 3.73205 = tan 75 deg to 1.1e-08).
   It is `lambda_c + 180 deg` and nothing else: purely geometric, carrying no
   rheology, no compliance, no C-A and no time dependence.  Nothing else in
   this project has a target known to 14 digits, and it gates load placement,
   axis convention, handedness and mesh anisotropy at once.  A drift in it with
   epoch is *ours*, since the reference has none.
3. **|m|**, expected within **<= 15% at `--medium`** with the C-A choice
   declared -- not "a few percent".  C-A alone contributes +3.6% to +12.1%.
4. **|mdot|, at t >= 2 kyr only.**  mdot(0) is set by the fastest degree-2
   Maxwell modes (tau = 0.298 to 0.749 kyr) and needs dt <~ 3 yr for 1%, so it
   is a time-stepping statement rather than a physical one and gets a
   dt-halving check instead of a tolerance.

## The two things this run measures that no gate before it could

**The loop.**  Eliminating u analytically gives the classical secular Liouville
relation with k_T *computed* rather than input, and switching the rotational
body force off removes the k_T term:

    |m| with feedback / |m| without = 1 / (1 - k_T/k_s) = 1.466 at t = 0

So `--no-feedback` drops `rotational_potential_term` while keeping the closure
row, and the ratio is the loop measured rather than argued.  Every rotation
gate the project has until now tests one half of the loop.

**C-A.**  The prescribed C - A = 2.63e35 implies k_s = 0.94334, against the
0.96672389 the benchmark's own rotation calculation demonstrably used -- a
2.42% inconsistency, amplified by `1/(1 - k_T/k_s)` to +3.6% at t = 0 and
+7-12% by 20 kyr.  The primary value here is therefore the **k_s-consistent**
2.6952e35, with the prescribed one available as `--c-minus-a prescribed`.
**Note §2.2's constants block carries the secondary value**; it is not copied
blind.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np
from firedrake import (COMM_WORLD, Constant, Function, FunctionSpace, Mesh,
                       SpatialCoordinate, Submesh, VectorFunctionSpace,
                       as_vector, assemble, conditional, dot, sqrt)
from gadopt import (CompressibleInternalVariableApproximation, SphericalDtN)
from firedrake import MixedVectorSpaceBasis
from gadopt import rigid_body_modes
from gadopt.gia_gravity import (FluidCore, SelfGravitatingGIASolver,
                                rigid_rotation_nullspace,
                                self_gravitating_gia_space)

import generate_selfgrav_sphere as gen
import reference_state as rs
from taboo_synthesis import cap_load
from validate_selfgrav_sphere import curve_mesh, provenance, tangle_census

HERE = os.path.dirname(os.path.abspath(__file__))

# -- Scales and numbers, handoff §2.2 ---------------------------------------
MU_BAR = 1e11                      # Pa
T_BAR_YR = 316.8809                # yr, = eta_bar/mu_bar
LAMBDA = rs.LAMBDA                 # 1.361324, measured in A3
B_MU = rs.RHO_BAR * rs.G_BAR * rs.D_SCALE / MU_BAR      # 1.564037
OMEGA_SQ = 7.292115e-5**2 * rs.D_SCALE / rs.G_BAR       # 1.566176e-03

#: Non-dimensionalising constant for the moments of inertia, rho_bar D^5.
RHO_D5 = rs.RHO_BAR * rs.D_SCALE**5

#: C is common to both choices; only C-A is in dispute.  C matters to `m_3`
#: (the rotation-rate change) and not at all to polar motion.
C_NONDIM = 8.0394e37 / RHO_D5

#: **The C-A choice, declared in advance.**  `ks` is primary.  See the module
#: docstring and `NOTES/REVIEW-SPADA-B4-PRECOMMIT.md` §4.
C_MINUS_A = {"ks": 2.6952e35 / RHO_D5, "prescribed": 2.63e35 / RHO_D5}

#: Mantle layers of M3-L70-V01, outermost first: (r_out, r_in, rho, mu, eta).
#: Non-dimensional, handoff §2.2's table.  The lithosphere's infinite viscosity
#: becomes 1e19 t_bar, as `3d_spada` uses, so the two runs differ in nothing
#: incidental.
MANTLE_LAYERS = [
    (2.203736, 2.179523, 0.551011, 0.50605, 1e19),
    (2.179523, 2.058457, 0.623766, 0.70363, 1.0),
    (2.058457, 1.971982, 0.702326, 1.05490, 1.0),
    (1.971982, 1.203736, 0.903172, 2.28340, 2.0),
]
RHO_CORE = 10750.0 / rs.RHO_BAR    # 1.950402

LOAD_COLAT_DEG, LOAD_LON_DEG = 25.0, 75.0
LOAD_THICKNESS, LOAD_ALPHA_DEG = 1500.0, 10.0
#: The reference phase, exact: lambda_c + 180 deg.
REFERENCE_PHASE_DEG = LOAD_LON_DEG + 180.0 - 360.0      # -105.0


def say(*a):
    if COMM_WORLD.rank == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------------------
# B2's coupled iterative solver
# ---------------------------------------------------------------------------
#: GAMG settings, verbatim from `demos/gravity/spikes/gate_b2_solver.py`.
GAMG = {"pc_type": "gamg", "mg_levels_pc_type": "sor",
        "pc_gamg_threshold": 0.01, "pc_gamg_square_graph": 100,
        "pc_gamg_coarse_eq_limit": 1000,
        "pc_gamg_mis_k_minimum_degree_ordering": True}


def coupled_solver_parameters(block0_rtol=1e-2, outer_rtol=1e-6,
                              block0_max_it=60):
    """B2's dictionary, reproduced here so a queue job is self-contained.

    Source of truth is `demos/gravity/spikes/gate_b2_solver.py` and
    `NOTES/IMPL-LOG-SPADA-B2-SOLVER.md`; measured mesh-independent at nu = 0.49,
    outer FGMRES 5 at both rungs with block-0 median 9 (`--coarse`) and 10
    (`--medium`) across a 5.2x increase in cells.

    **`fieldsplit_N_` indexes the SPLIT, not the field**, and B2 records that
    getting it backwards cost a run: the sweep order is `m, u, psi` while the
    mixed space orders `u = 0, m = 1, psi = 2`, so the prefixes are derived from
    the sweep's position rather than written by hand.  Doing it by hand puts
    GAMG on the DG1 tensor block and `bjacobi/ilu` on the displacement, which
    does not raise, does not warn, and surfaces only as block-0 hitting its cap.
    """
    def prefixed(d, pre):
        return {pre + k: v for k, v in d.items()}

    def inner(pc_python, extra=None):
        d = {"ksp_type": "preonly", "pc_type": "python",
             "pc_python_type": pc_python, "ksp_converged_reason": None}
        d.update(prefixed(extra or GAMG, "assembled_"))
        return d

    p = {
        "mat_type": "matfree",
        "snes_type": "newtonls", "snes_linesearch_type": "l2",
        "snes_max_it": 100, "snes_atol": 1e-15, "snes_rtol": 1e-4,
        "snes_converged_reason": None,
        "ksp_type": "fgmres", "ksp_rtol": outer_rtol, "ksp_max_it": 200,
        "ksp_converged_reason": None,
        "pc_type": "python", "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        "dtn_pc_fieldsplit_schur_fact_type": "full",
        "dtn_fieldsplit_0_ksp_type": "fgmres",
        "dtn_fieldsplit_0_ksp_rtol": block0_rtol,
        "dtn_fieldsplit_0_ksp_max_it": block0_max_it,
        "dtn_fieldsplit_0_ksp_converged_reason": None,
        "dtn_fieldsplit_0_pc_type": "fieldsplit",
        "dtn_fieldsplit_0_pc_fieldsplit_type": "multiplicative",
        "dtn_fieldsplit_1_ksp_type": "gmres", "dtn_fieldsplit_1_ksp_rtol": 1e-4,
        "dtn_fieldsplit_1_ksp_max_it": 200,
        "dtn_fieldsplit_1_ksp_converged_reason": None,
        # **`none`, and both alternatives fail in a way that names neither
        # itself nor `Real`.**  B2's measured results are all at multiplier PC
        # `none` -- unpreconditioned GMRES on the 75 Real -- and the two
        # obvious improvements are both blocked in `DtNTwoBlockSchurPC`:
        #
        # * `jacobi` needs `MatGetDiagonal`, which on a matfree block goes
        #   through TSFC's `diagonal=True` path and cannot handle `Real`
        #   arguments.  Presents as `error code 101` out of `SNES.solve`
        #   wrapping `error code 63` from `MatCreateSubMatrix_Python`.
        # * `AssembledPC` (+ `lu`) needs an appctx, which it resolves through
        #   the sub-KSP's DM -- and `DtNTwoBlockSchurPC` threads a sub-DM onto
        #   **block 0 only** (`preconditioners.py`: `ksp_potential, _ =
        #   inner.getFieldSplitSchurGetSubKSP()`).  Block 1 has no DM, so
        #   `get_appctx` returns `None` and it dies with
        #   `AttributeError: 'NoneType' object has no attribute 'appctx'`
        #   buried under another `error code 101`.
        #
        # Both are `gadopt/preconditioners.py` limitations rather than
        # configuration mistakes; see the PETSc round-2 task.
        "dtn_fieldsplit_1_pc_type": "none",
    }
    sweep = [(1, inner("firedrake.AssembledPC",
                       {"pc_type": "bjacobi", "sub_pc_type": "ilu"})),
             (0, inner("firedrake.AssembledPC")),
             (2, inner("gadopt.SPDAssembledPC"))]
    for split, (field, opts) in enumerate(sweep):
        p[f"dtn_fieldsplit_0_pc_fieldsplit_{split}_fields"] = str(field)
        p.update(prefixed(opts, f"dtn_fieldsplit_0_fieldsplit_{split}_"))
    return p


def near_nullspace_basis(Z, layout):
    """Rigid-body modes on the displacement block, for GAMG.

    B2 §3/§6: `rigid_body_modes` is absent from
    `coupled_gia_solver_parameters` entirely and **cannot** be supplied as an
    options entry, because it is a `near_nullspace` argument.  It is necessary
    for the deviatoric part and is a subset of what the volumetric penalty
    wants -- nothing in a near-nullspace can fix penalty conditioning.
    """
    rbm = rigid_body_modes(Z.sub(layout.displacement), rotational=True,
                           translations=[0, 1, 2])
    return MixedVectorSpaceBasis(
        Z, [rbm if i == layout.displacement else Z.sub(i)
            for i in range(len(Z))])


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def layered_field(mesh, column, name):
    """A DG0 field from `MANTLE_LAYERS`, exact because the mesh conforms."""
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    expr = Constant(MANTLE_LAYERS[-1][column])
    for r_out, _, *vals in MANTLE_LAYERS:
        expr = conditional(r < r_out, Constant(vals[column - 2]), expr)
    return Function(FunctionSpace(mesh, "DG", 0), name=name).interpolate(expr)


def load_axis():
    """Unit vector to the load centroid (theta_c, lambda_c)."""
    th, la = np.deg2rad(LOAD_COLAT_DEG), np.deg2rad(LOAD_LON_DEG)
    return np.array([np.sin(th) * np.cos(la), np.sin(th) * np.sin(la),
                     np.cos(th)])


def load_field(mesh, nmax, degree=2, name="sigma"):
    """The truncated cap series about the load axis, interpolated numerically.

    Handoff §3: the load is applied as `sum_{n=2}^{nmax} sigma_n P_n(cos gamma)`
    interpolated into a CG field, **not built in UFL and not the exact analytic
    cap**.  That is what the reference is, it is unambiguous, and it lets nmax
    track the mesh (A1 §1.2).

    Numerically rather than symbolically because a UFL Legendre recurrence to
    n = 32 compiles to a very deep expression for no benefit: the field is
    evaluated at CG dof coordinates with numpy and assigned.  The field is
    defined at every radius (it depends only on direction) and is correct
    wherever it is used, which is the Re facets.

    Surface density is non-dimensionalised by `rho_bar * D`.
    """
    sigma_n = cap_load(nmax, thickness=LOAD_THICKNESS,
                       alpha_deg=LOAD_ALPHA_DEG) / (rs.RHO_BAR * rs.D_SCALE)
    f = Function(FunctionSpace(mesh, "CG", degree), name=name)
    xyz = Function(VectorFunctionSpace(mesh, "CG", degree)).interpolate(
        SpatialCoordinate(mesh)).dat.data_ro
    r = np.linalg.norm(xyz, axis=1)
    mu = np.clip((xyz @ load_axis()) / np.maximum(r, 1e-300), -1.0, 1.0)
    # Legendre by recurrence, accumulating from n = 2.
    p_prev, p_cur = np.ones_like(mu), mu.copy()
    total = np.zeros_like(mu)
    for n in range(1, nmax):
        p_next = ((2 * n + 1) * mu * p_cur - n * p_prev) / (n + 1)
        if n + 1 >= 2:
            total += sigma_n[n + 1] * p_next
        p_prev, p_cur = p_cur, p_next
    f.dat.data[:] = total
    return f


def build_meshes(configuration, path=None, untangle=True, curve=True):
    """Parent and mantle submesh, both P2-curved.

    `untangle=True` here and not in A2's validator: curving tangles two cells of
    113 653 at `--coarse` (A2 addendum), which moves no integral but puts a
    negative-volume contribution into a stiffness matrix, and this is the first
    deliverable that assembles one.

    **That default has never done what it says, and until now it could not run
    in parallel at all.** `curve_mesh`'s repair loop called the collective
    accessor `Dat.data_with_halos` once per *rank-local* tangled cell, so any
    partition on which two ranks held different numbers of them hung rather
    than failed: measured, `mpiexec -n 4` on `b4_sphere.msh` never returned.
    That is fixed. What is not fixed is the repair itself -- it straightens the
    cells it targets and tangles two neighbours instead, 2 -> 2 in serial and
    2 -> 3 at 4 ranks. The census below is printed for exactly this reason.
    """
    path = path or os.path.join(HERE, "b4_sphere.msh")
    # Reuse an existing file rather than regenerating: gmsh is not collective,
    # a queue job should not spend node-hours on it, and re-running the
    # generator would give a *different* mesh (tetrahedralisation is not
    # deterministic across builds), which would silently break comparability
    # between the runs of a single campaign.
    if COMM_WORLD.rank == 0 and not os.path.exists(path):
        gen.generate(path, configuration=configuration)
    COMM_WORLD.barrier()
    say(f"  build_meshes: curve={curve}  untangle={untangle}  file={path}")
    parent = Mesh(path)
    if curve:
        parent = curve_mesh(parent, untangle=untangle)
    parent.cartesian = False
    sub = Submesh(parent, 3, gen.CELL_MANTLE)
    if curve:
        sub = curve_mesh(sub, untangle=untangle)
    sub.cartesian = False
    # The count and the radii, in whichever arm actually ran. A repaired arm
    # must read zero and an unrepaired arm at least one; equal counts mean the
    # A/B is void, and this is where that becomes visible without anyone having
    # to trust a flag.
    tangle_census(parent, f"parent, curve={curve} untangle={untangle}")
    tangle_census(sub, f"mantle submesh, curve={curve} untangle={untangle}")
    return parent, sub


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------
class FeedbackOffSolver(SelfGravitatingGIASolver):
    r"""The loop A/B, cutting only the momentum arrow.

    ## Why not the obvious switches

    `Omega_sq = 0` is wrong: the closure row's scaling is
    `theta_rot_i = s_i f B_mu Omega_sq`, so zeroing it scales
    `K_i m_i = s_i dI_i3` by zero and **deletes the row**, leaving m
    undetermined rather than uncoupled.  The body force and the row share the
    one constant.

    Zeroing `psi_rot` at source -- which is what this class used to do -- is
    wrong for two separate reasons, and both were found by measurement:

    1. **It makes the `(u, m)` block exactly zero.**  The system is then
       block-triangular in `m`, the `Real` block's Schur complement degenerates,
       and the iterative solver dies at iteration zero with
       `DIVERGED_LINEAR_SOLVE`.  It survives only on the LU-on-block-0 path,
       which does not scale -- so the loop A/B, the measurement that closes road
       map §9.6, was stranded on a solver that cannot reach `--medium`.
    2. **Since A4's fix it cuts two arrows, not one.**  The fluid core's
       traction is `B_mu[rho_core (psi + psi_rot) + ...]`, so `psi_rot` now
       reaches the momentum problem by a second route.  Zeroing the shared
       object removes both, and the measured amplification is then not the
       quantity `1/(1 - k_T/k_s)` was derived for.

    ## What this does instead

    Drops **`rotational_potential_term` alone** from the momentum equation, by
    swapping the name in `gadopt.momentum_equation`'s namespace for the duration
    of `set_equations`.  `psi_rot` itself is untouched, so A4's core traction
    still couples `m` into the momentum problem, `(u, m)` stays populated, and
    the Schur complement stays non-degenerate.

    It is also the better experiment: it cuts exactly one arrow of the loop
    rather than removing an object that appears on both sides of it.

    **It deliberately breaks the transpose symmetry**, as the old version did --
    the `(m, u)` row still reads dI while the `(u, m)` block has lost its volume
    piece.  The coupled symmetry gates will flag it.  That is the point of the
    experiment and must not be "fixed".
    """

    def set_equations(self) -> None:
        import gadopt.gia_gravity as gg  # noqa: PLC0415

        import functools  # noqa: PLC0415

        original = gg.rotational_potential_term

        # `functools.wraps` is load-bearing, not decoration: `Equation` reads
        # `term.required_attrs` off each term function, and a bare wrapper has
        # no such attribute -- `AttributeError: 'function' object has no
        # attribute 'required_attrs'`, raised from `set_equations` rather than
        # from anything naming the wrapper.  `wraps` copies `__dict__`, which
        # is where that attribute lives.
        @functools.wraps(original)
        def dropped(eq, trial):
            # A `Constant(0.0)` factor rather than a literal: a literal folds to
            # UFL `Zero`, and a form with no integrals raises somewhere
            # unhelpful (`demos/gravity/CLAUDE.md`).
            return Constant(0.0) * original(eq, trial)

        gg.rotational_potential_term = dropped
        try:
            super().set_equations()
        finally:
            gg.rotational_potential_term = original


def build_solver(parent, sub, *, dt, nmax=32, truncation=5,
                 c_minus_a="ks", feedback=True, fluid_core=True,
                 bulk_shear_ratio=1.94, solver_parameters=None,
                 solver_parameters_extra=None, near_nullspace=True,
                 quad_degree=None):
    """The coupled solver for the off-axis cap load.

    `feedback=False` drops `rotational_potential_term` while keeping the
    closure row, which is the loop A/B: it leaves arrows (A), (B) and (C) of the
    review's diagram intact and cuts (D), so |m| should fall by
    `1/(1 - k_T/k_s)`.

    `bulk_shear_ratio` stays at 1.94 (nu = 0.28), Phase B's value.  **It must
    not cross 10**: handoff §12.0, `3d_spada` switches approximation class
    above that and the replacement returns 0 from the Al-Attar volume prestress
    term by design, silently deleting the reference-state physics this whole
    demo rests on.
    """
    if bulk_shear_ratio > 10:
        raise ValueError(
            f"bulk_shear_ratio = {bulk_shear_ratio} > 10 would, in a driver "
            "copied from 3d_spada, switch approximation class and disable the "
            "volume prestress term. Handoff §12.0. Refusing rather than "
            "silently changing the physics.")

    sigma_p = load_field(parent, nmax, name="sigma_parent")
    sigma_m = load_field(sub, nmax, name="sigma_mantle")

    gravity_bcs = {
        gen.SURF_OUTER: {"dtn": SphericalDtN(truncation)},
        gen.SURF_INNER: {"dtn": SphericalDtN(truncation)},
        gen.SURF_RE: {"interior_sigma": sigma_p},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=True,
        self_gravity_number=LAMBDA, quad_degree=quad_degree)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    # The layered bulk modulus tracks the layered shear modulus, as
    # `3d_spada` does below the threshold, so nu is uniform and the ratio is
    # one number (handoff §8).
    mu = layered_field(sub, 3, "mu")
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=layered_field(sub, 2, "rho"),
        shear_modulus=mu, viscosity=layered_field(sub, 4, "eta"),
        bulk_shear_ratio=bulk_shear_ratio, g=1.0, B_mu=B_MU,
        self_gravity_number=LAMBDA)

    bcs = {gen.SURF_RE: {"normal_stress": B_MU * sigma_m}}
    kwargs = {}
    if fluid_core:
        kwargs["fluid_core"] = FluidCore(boundary=gen.SURF_RC,
                                         rho_core=RHO_CORE)
    else:
        bcs[gen.SURF_RC] = {"un": 0.0}

    if near_nullspace:
        kwargs["near_nullspace"] = near_nullspace_basis(Z, layout)
    cls = SelfGravitatingGIASolver if feedback else FeedbackOffSolver
    solver = cls(
        z, approx, layout=layout, dt=dt, bcs=bcs,
        rotation_moments={"C": C_NONDIM, "C_minus_A": C_MINUS_A[c_minus_a]},
        Omega_sq=OMEGA_SQ,
        nullspace=rigid_rotation_nullspace(Z, layout),
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra, **kwargs)
    return solver, z, layout


#: Piecewise-constant dt ladder, in years, as (t_end_yr, dt_yr).  Handoff §9
#: **as corrected**: its original ladder is struck out there, and the two
#: defects it names are independent.  The **late cap** (dt <= 10 kyr) is the one
#: that matters and is free here -- B4 stops at 20 kyr, so the coarsest segment
#: is 500 yr, twenty times inside the cap, and the 16 kyr singular step size the
#: correction warns about is never approached.  The **early refinement** is the
#: accuracy fix, and the segments below follow §9's rule `dt <~ 2 e t`: at each
#: epoch the binding mode is the one with tau ~ t, so the tolerable step grows
#: with t.  Resulting time-stepping error by that rule: 2.5% at 1 kyr, 2.5% at
#: 2 kyr, 1.0% at 5 kyr, 0.5% at 10 kyr, 1.25% at 20 kyr.
#:
#: **This matters more to B4 than to a displacement gate**, and the reason is
#: worth stating: a time-stepping error that *shrinks* with t would masquerade
#: as exactly the epoch trend in R_m that the review's rows 6 and 7 use to tell
#: a compliance error from a C-A error.  The ladder is therefore chosen so its
#: own error is below the 4-8% the trend has to resolve, at every epoch.
DT_LADDER_YR = ((100.0, 10.0), (1000.0, 50.0), (10000.0, 100.0),
                (20000.0, 500.0))


def time_ladder(epochs_kyr, ladder=DT_LADDER_YR):
    """(t_end_yr, dt_yr, [epochs landing on it]) segments covering the epochs.

    Epoch boundaries are made to land exactly on a step, so no interpolation is
    needed and every reported number is a solved state.
    """
    want = sorted(e * 1000.0 for e in epochs_kyr if e > 0)
    out, t0 = [], 0.0
    for t_end, dt in ladder:
        if t_end <= t0:
            continue
        marks = [w for w in want if t0 < w <= t_end]
        cuts = sorted(set(marks + [t_end]))
        for c in cuts:
            n = max(1, int(round((c - t0) / dt)))
            out.append((t0, c, (c - t0) / n, n, c in marks))
            t0 = c
    return out


def sheet_share_of_inertia(solver):
    """How much of dI the fluid core's CMB sheet actually carries.

    **This, and not the rigid/fluid A/B, is the test of the review's warning
    that the sheet must reach the inertia row.**  The mechanics are held fixed
    -- one solved state, two assemblies -- so nothing here is a re-solve and
    the comparison isolates the term instead of the CMB condition.

    A sheet enters every component through the same integral at the same radius
    with the same (Rc/Re)^4 weight, so its share should be *similar* across the
    three.  A rigid/fluid A/B does not have that property, because releasing
    `un = 0` changes the mechanics: p_3 = x^2 + y^2 reads the radial
    deformation the constraint was pinning and p_1 = -xz barely feels it.
    Expect ~2%, not the ~13% the A/B showed.
    """
    if solver.fluid_core is None:
        return None
    out = {}
    for i, name in enumerate(("dI_13", "dI_23", "dI_33")):
        total = float(assemble(solver.inertia_form(i)))
        sheet = float(assemble(solver.fluid_core_sheet_integral(
            Constant(1.0) * solver.inertia_polynomial(
                i, SpatialCoordinate(solver.mesh)))))
        out[name] = (total, sheet, sheet / total if total else float("nan"))
    return out


def diagnostic(fn, *args):
    """Run a diagnostic and return a string on failure instead of raising.

    **A diagnostic must never be able to kill the measurement it decorates.**
    `kernel_component` took down three runs on three distinct errors -- a
    `ListTensor` with no function space, then a `VectorSpaceBasis` that is not
    a UFL type -- and each time the physics had already been computed and was
    thrown away at the print. The gates are the deliverable; the instrumentation
    is not, and it does not get to vote on whether the run succeeds.
    """
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001 - the point is to swallow anything
        return f"UNAVAILABLE ({type(exc).__name__}: {exc})"
    return f"{out:.3e}" if isinstance(out, float) else out


def kernel_component(solver, layout):
    r"""|P_rot u| / |u|: how much rigid rotation the solution carries.

    `demos/gravity/CLAUDE.md` records that *declaring* the rigid-rotation
    nullspace is not enough on this solver -- FGMRES is right-preconditioned,
    PETSc removes the kernel from the right-hand side but not from the
    preconditioner's output -- and that `project_out_nullspace()` after each
    solve is what actually removes it.

    It is worth measuring rather than reasoning about.  The usual argument is
    that a rigid rotation contributes nothing to dI, and in exact arithmetic it
    does not.  But **dI_13 and dI_23 are degree-2 moments of the displacement,
    a rigid rotation is a degree-1 field, and their product against
    `grad(p_i)` is not orthogonal under quadrature** -- so a large kernel
    component can leak into exactly the two components polar motion reads.
    """
    # `solution_split` is the UFL split -- `ListTensor` objects, symbolic, with
    # no function space.  Right for building forms, wrong for anything needing
    # a space, a `Function` or a projection: `AttributeError: 'ListTensor'
    # object has no attribute 'function_space'`.  The actual `Function` is on
    # `solution.subfunctions`.
    u = solver.solution.subfunctions[layout.displacement]
    V = u.function_space()
    # Built explicitly rather than via `rigid_body_modes`, which returns a
    # `VectorSpaceBasis` -- a PETSc-side object, not a list of `Function`s, so
    # feeding it to UFL gives `ValueError: Invalid type conversion:
    # VectorSpaceBasis can not be converted to any UFL type`.  The three
    # generators are `e_i x x` and are two lines to write.
    Xv = SpatialCoordinate(V.mesh())
    modes = [Function(V).interpolate(as_vector([0.0, -Xv[2], Xv[1]])),
             Function(V).interpolate(as_vector([Xv[2], 0.0, -Xv[0]])),
             Function(V).interpolate(as_vector([-Xv[1], Xv[0], 0.0]))]
    nrm = np.sqrt(assemble(dot(u, u) * solver.dx_m))
    if nrm == 0.0:
        return 0.0
    comp = 0.0
    for m in modes:
        mn = np.sqrt(assemble(dot(m, m) * solver.dx_m))
        if mn > 0:
            comp += (assemble(dot(u, m) * solver.dx_m) / mn) ** 2
    return float(np.sqrt(comp) / nrm)


def polar_motion_deg(solver):
    """(mx, my, |m|, phase) in degrees, from the solved rotation scalars.

    `m_1` and `m_2` are direction cosines of the perturbed rotation axis, so
    they are radians to the accuracy that matters at 1e-04.
    """
    v = solver.rotation_values()
    mx = np.rad2deg(v.get("m_1", v.get("m1", 0.0)))
    my = np.rad2deg(v.get("m_2", v.get("m2", 0.0)))
    return mx, my, float(np.hypot(mx, my)), float(np.degrees(
        np.arctan2(my, mx)))


def run_time_loop(parent, sub, args, epochs_kyr):
    """March the ladder, rebuilding the solver at each dt change.

    `CoupledInternalVariableSolver.__init__` does
    `approximation.mu = approximation.effective_viscosity(dt)`, so dt is baked
    into the operator at construction and cannot be changed on a live solver.
    A piecewise-constant ladder therefore means one solver per segment, with the
    state carried across by copying the mixed function -- which carries the
    internal variables too, since they are subfunctions of it.
    """
    import time  # noqa: PLC0415

    results, z_prev = [], None
    for t0, t1, dt_yr, nsteps, is_epoch in time_ladder(epochs_kyr):
        solver, z, layout = build_solver(
            parent, sub, dt=dt_yr / T_BAR_YR, nmax=args.nmax,
            truncation=args.truncation, c_minus_a=args.c_minus_a,
            feedback=not args.no_feedback,
            fluid_core=not args.no_fluid_core,
            solver_parameters=coupled_solver_parameters())
        if z_prev is not None:
            z.assign(z_prev)
        say(f"  {t0 / 1000:7.3f} -> {t1 / 1000:7.3f} kyr   dt {dt_yr:7.2f} yr"
            f"   {nsteps:4d} steps")
        for k in range(nsteps):
            tic = time.time()
            solver.solve()
            if k == 0:
                say(f"      first step {time.time() - tic:.1f}s "
                    f"(includes compilation)")
            elif k == 1:
                say(f"      per step   {time.time() - tic:.1f}s")
        z_prev = z
        if is_epoch:
            mx, my, absm, phase = polar_motion_deg(solver)
            results.append((t1 / 1000.0, mx, my, absm, phase,
                            solver.inertia_perturbation()))
            say(f"      t = {t1 / 1000:6.2f} kyr  |m| {absm:.7f}  "
                f"phase {phase:+.4f}")
    return results, z_prev


def reference_series(npz_path=None):
    """The T02-03 cap polar motion, and its phase."""
    d = np.load(npz_path or os.path.join(HERE, "reference.npz"))
    t = d["pm_cap_t_kyr"]
    mx, my = d["pm_cap_mx_deg"], d["pm_cap_my_deg"]
    return t, mx, my, d["pm_cap_absm_deg"], np.degrees(np.arctan2(my, mx))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(gen.CONFIGURATIONS))
    ap.add_argument("--epochs", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--dt-yr", type=float, default=50.0)
    ap.add_argument("--elastic-dt-yr", type=float, default=0.03)
    ap.add_argument("--nmax", type=int, default=32)
    ap.add_argument("--truncation", type=int, default=5)
    ap.add_argument("--c-minus-a", default="ks", choices=list(C_MINUS_A))
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-fluid-core", action="store_true")
    ap.add_argument("--elastic-only", action="store_true",
                    help="one elastic solve at t = 0; gates 1 and 2 only")
    ap.add_argument("--mesh-file", default=None)
    ap.add_argument("--solver", default="b2", choices=["b2", "auto"],
                    help="'b2' = the coupled iterative parameters; 'auto' = "
                         "gadopt's own default, which is what every laptop "
                         "number before 2026-08-01 was measured with")
    ap.add_argument("--ksp-rtol", type=float, default=1e-6,
                    help="outer FGMRES tolerance; theta_rot carries "
                         "Omega_sq = 1.57e-03, so the rotation rows sit three "
                         "orders below the dominant ones and a tolerance set "
                         "by those can under-resolve them without failing")
    ap.add_argument("--no-curve", action="store_true",
                    help="skip the P2 isoparametric correction")
    ap.add_argument("--quad-degree", type=int, default=None)
    ap.add_argument("--no-untangle", action="store_true",
                    help="skip the curve_mesh tangling correction")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--monitor-multipliers", action="store_true",
                    help="print the Real block's TRUE residual, not just its "
                         "iteration count: it runs at pc_type none, and a "
                         "block that stops on iteration count rather than "
                         "tolerance leaves the DtN constraint unsatisfied")
    args = ap.parse_args()

    provenance(os.path.basename(__file__))
    say(f"B4 polar motion: {args.configuration}, C-A = "
        f"{C_MINUS_A[args.c_minus_a]:.6e} ({args.c_minus_a}), "
        f"feedback {not args.no_feedback}, fluid core "
        f"{not args.no_fluid_core}, n_max {args.nmax}")
    say(f"  Lambda {LAMBDA:.6f}  B_mu {B_MU:.6f}  Omega_sq {OMEGA_SQ:.6e}  "
        f"C {C_NONDIM:.6f}")

    parent, sub = build_meshes(args.configuration, args.mesh_file,
                               untangle=not args.no_untangle,
                               curve=not args.no_curve)
    say(f"  parent {parent.comm.allreduce(parent.cell_set.size)} cells, "
        f"mantle {sub.comm.allreduce(sub.cell_set.size)} cells")

    dt = args.elastic_dt_yr / T_BAR_YR
    extra = {"snes_monitor": None} if args.monitor else {}
    if args.monitor_multipliers:
        extra["dtn_fieldsplit_1_ksp_monitor_true_residual"] = None
    if args.solver == "auto" and args.ksp_rtol != 1e-6:
        # The rotation scalars are `Real`, so they sit in the DtN Schur split's
        # **block 1**, not in the LU'd block 0.  Their rows carry
        # theta_rot ~ Omega_sq = 1.57e-03, three orders below the dominant
        # rows, so a tolerance set by those can leave them under-resolved
        # without the solve ever reporting a failure.
        extra["ksp_rtol"] = args.ksp_rtol
        extra["dtn_fieldsplit_1_ksp_rtol"] = args.ksp_rtol
    extra = extra or None
    solver, z, layout = build_solver(
        parent, sub, dt=dt, nmax=args.nmax, truncation=args.truncation,
        c_minus_a=args.c_minus_a, feedback=not args.no_feedback,
        fluid_core=not args.no_fluid_core, quad_degree=args.quad_degree,
        solver_parameters=(coupled_solver_parameters(outer_rtol=args.ksp_rtol)
                           if args.solver == "b2" else None),
        solver_parameters_extra=extra)

    say(f"\n  fluid-core sheet reaches inertia_form: "
        f"{'yes' if solver.fluid_core is not None else 'n/a (rigid core)'}")
    say(f"  dt = {args.elastic_dt_yr} yr = {dt:.3e} t_bar (elastic step)")
    solver.solve()

    t_ref, mxr, myr, absr, phr = reference_series()
    mx, my, absm, phase = polar_motion_deg(solver)
    dI = solver.inertia_perturbation()

    say("\n" + "=" * 72)
    say("GATE 1  handedness: mx < 0 AND my < 0 at t = 0")
    say(f"        reference mx(0) = {mxr[0]:+.7f}, my(0) = {myr[0]:+.7f}")
    say(f"        model     mx(0) = {mx:+.7f}, my(0) = {my:+.7f}")
    g1 = bool(mx < 0 and my < 0)
    say("        PASS" if g1 else
        "        FAIL - a flip is invisible to |m|, |mdot| and my/mx")

    say("\nGATE 2  phase: exactly -105.0000 deg (lambda_c + 180), all epochs")
    say(f"        reference phase(0) = {phr[0]:.10f} deg")
    say(f"        model     phase(0) = {phase:.6f} deg   "
        f"dphi = {phase - REFERENCE_PHASE_DEG:+.6f} deg")
    say(f"        dI_13 = {dI['dI_13']:+.6e}  dI_23 = {dI['dI_23']:+.6e}  "
        f"dI_33 = {dI['dI_33']:+.6e}")

    say(f"\nGATE 3  |m| at t = 0: reference {absr[0]:.7f} deg")
    say(f"        model {absm:.7f} deg   ratio {absm / absr[0]:.4f}")

    say(f"\nKERNEL  |P_rot u| / |u| = {diagnostic(kernel_component, solver, layout)}")
    say("        (declared nullspaces are not projected out of this solver's")
    say("         output; dI_13 and dI_23 are degree-2 moments and a degree-1")
    say("         rigid rotation is not orthogonal to them under quadrature)")

    share = diagnostic(sheet_share_of_inertia, solver)
    if isinstance(share, dict):
        say("\nSHEET   the CMB sheet's share of dI, mechanics held fixed.")
        say("        Expect ~2% and SIMILAR across the three components; the")
        say("        rigid/fluid A/B is not this test.")
        for name, (total, sheet, frac) in share.items():
            say(f"        {name}: total {total:+.6e}  sheet {sheet:+.6e}  "
                f"{frac * 100:+7.3f}%")
    if args.elastic_only:
        say("=" * 72)
        return 0 if g1 else 1

    # -- the time loop -----------------------------------------------------
    say("\nTime ladder (handoff §9 as corrected; late cap dt <= 10 kyr is")
    say("free here since B4 stops at 20 kyr and the coarsest segment is 500 yr):")
    epochs = [e for e in args.epochs if e > 0]
    results, _ = run_time_loop(parent, sub, args, epochs)

    say("\n" + "=" * 78)
    say("GATE 3  |m(t)| against T02-03, and GATE 2 phase at every epoch")
    say(f"  {'t (kyr)':>8} {'|m| ref':>10} {'|m| model':>10} {'R_m':>7} "
        f"{'phase':>10} {'dphi':>9}")
    say(f"  {0.0:8.2f} {absr[0]:10.7f} {absm:10.7f} {absm / absr[0]:7.4f} "
        f"{phase:10.4f} {phase - REFERENCE_PHASE_DEG:+9.4f}")
    rows = [(0.0, absm / absr[0])]
    for tk, mxi, myi, ab, ph, _dI in results:
        j = int(np.argmin(np.abs(t_ref - tk)))
        rows.append((tk, ab / absr[j]))
        say(f"  {tk:8.2f} {absr[j]:10.7f} {ab:10.7f} {ab / absr[j]:7.4f} "
            f"{ph:10.4f} {ph - REFERENCE_PHASE_DEG:+9.4f}")

    # The epoch trend is the discriminator: review rows 6 vs 7.  A compliance
    # error is flat in t; a C-A error grows from ~1.04 toward ~1.12.
    say("\n  epoch trend in R_m (review rows 6 vs 7: compliance is FLAT in t,")
    say("  C-A GROWS from ~1.04 toward ~1.12):")
    say(f"    R_m(0) = {rows[0][1]:.4f}   R_m(20 kyr) = {rows[-1][1]:.4f}   "
        f"ratio {rows[-1][1] / rows[0][1]:.4f}")

    # -- |mdot|, t >= 2 kyr only -------------------------------------------
    d = np.load(os.path.join(HERE, "reference.npz"))
    td, mdx, mdy = (d["pm_cap_mdot_t_kyr"], d["pm_cap_mdotx_deg_per_myr"],
                    d["pm_cap_mdoty_deg_per_myr"])
    absmd = np.hypot(mdx, mdy)
    say("\nGATE 4  |mdot|, gated at t >= 2 kyr only (mdot(0) is a")
    say("        time-stepping statement: it needs dt <~ 3 yr for 1%)")
    say(f"  {'t (kyr)':>8} {'ref':>12} {'model':>12} {'ratio':>8}")
    series = [(0.0, absm)] + [(r[0], r[3]) for r in results]
    for (t0, a0), (t1, a1) in zip(series[:-1], series[1:]):
        mid = 0.5 * (t0 + t1)
        rate = (a1 - a0) / (t1 - t0) * 1000.0          # deg per Myr
        j = int(np.argmin(np.abs(td - mid)))
        flag = "gated" if mid >= 2.0 else "diagnostic"
        say(f"  {mid:8.2f} {absmd[j]:12.4f} {rate:12.4f} "
            f"{rate / absmd[j]:8.4f}   {flag}")
    say("=" * 78)
    return 0 if g1 else 1


if __name__ == "__main__":
    sys.exit(main())
