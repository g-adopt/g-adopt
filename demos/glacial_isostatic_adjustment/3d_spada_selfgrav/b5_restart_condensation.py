"""Compare condensed and uncondensed GIA updates from the 1 kyr B5 state.

Run one arm per PBS job. Each arm reads the same displacement, potential, and
DG1 Maxwell history. It then advances two 100-year steps on the AR-7 mesh.

The uncondensed arm is the weak finite-element reference. The condensed arm
must match its spectra, field norms, amplification, and DG weak history
residual. A difference above the linear-solve error rejects exact condensation.

The same restart runs a power-law rheology on the uncondensed arm:
`--exponent`, `--transition-stress-mpa`, `--background-stress-mpa` and
`--power-law-layers` build it through `b1.power_law_rheology`, and the outer
method becomes `newtonls`. Each step then prints a NEWTON line and the stress
table of the solved state. `--stress-report --stress-indices I [I ...]` prints
the per-shell deviatoric stress of the listed checkpoint states and exits
before a solver is built; use it to choose `--dt-yr` for a power law.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import (CellVolume, CheckpointFile, COMM_WORLD,  # noqa: E402
                       Constant, FacetNormal, Function, FunctionSpace,
                       SpatialCoordinate, TensorFunctionSpace, assemble, avg,
                       conditional, dot, ds, dx, grad, norm, sqrt)
from gadopt.gia_gravity import (  # noqa: E402
    resolve_dtn_representation, selfgrav_dtn_iterative_solver_parameters)
from gadopt.internal_variable_equation import (  # noqa: E402
    assign_history_slices, history_slices)
from mpi4py import MPI  # noqa: E402

import b1_elastic as b1  # noqa: E402
import b5_viscoelastic as b5  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402
from b4_polar_motion import T_BAR_YR  # noqa: E402


SUB_TOPOLOGY = "firedrake_default_submesh_topology"
PARENT_TOPOLOGY = "firedrake_default_topology"


class GravityFeedbackAblatedSolver(b1.SelfGravitatingGIASolver):
    """Remove the perturbed-potential traction from the mechanics row.

    The potential equation still reads displacement. The load still acts as a
    Poisson sheet. The hydrostatic volume term and fluid-core spring remain.
    Thus, this arm removes only the return path from potential to mechanics.
    """

    cancel_volume_feedback = True
    cancel_core_feedback = True

    def set_form(self):
        super().set_form()
        u_field = self.layout.displacement
        w = self.tests[u_field]
        psi = self.solution_split[self.layout.potential]
        B_mu = self.approximation.B_mu
        rho0 = self.approximation.density

        # Cancel the volume term from self_gravity_term:
        #   -scaling B_mu int rho0 grad(psi).w dx.
        if self.cancel_volume_feedback:
            self.F += (self.scaling_factor * B_mu * rho0
                       * dot(grad(psi), w) * self.dx_m)

        # Cancel only the mechanics variation of the fluid-core cross-term.
        # Keep its Poisson sheet and keep the fluid-core spring.
        if self.cancel_core_feedback and self.fluid_core is not None:
            rho_core = self.fluid_core.rho_core
            psi_face = avg(psi) if self.layout.cross_mesh else psi
            normal = FacetNormal(self.mesh)
            measure = self.fluid_core_measure()(self.fluid_core.boundary)
            self.F -= (self.scaling_factor * B_mu * rho_core
                       * dot(w, normal) * psi_face * measure)


class GravityVolumeFeedbackAblatedSolver(GravityFeedbackAblatedSolver):
    """Remove the self-gravity volume force but retain CMB feedback."""

    cancel_core_feedback = False


class GravityCoreFeedbackAblatedSolver(GravityFeedbackAblatedSolver):
    """Remove potential-dependent CMB traction but retain the volume force."""

    cancel_volume_feedback = False


def say(message):
    if COMM_WORLD.rank == 0:
        print(message, flush=True)


def load_restart(checkpoint, parent, sub, idx):
    """Load both mesh fields despite the old checkpoint's duplicate mesh name.

    B5 saved the parent mesh and mantle submesh with the same default name.
    The final mesh-name map therefore contains only the parent topology. The
    data for both topologies remains present. Override the read-only map for
    each load instead of changing the checkpoint.
    """
    with CheckpointFile(checkpoint, "r") as chk:
        chk._get_mesh_name_topology_name_map = lambda: {
            "firedrake_default": SUB_TOPOLOGY}
        saved_sub = chk.load_mesh()
        displacement = chk.load_function(
            saved_sub, "displacement", idx=idx)
        internal_variable = chk.load_function(
            saved_sub, "internal_variable_0", idx=idx)

    with CheckpointFile(checkpoint, "r") as chk:
        chk._get_mesh_name_topology_name_map = lambda: {
            "firedrake_default": PARENT_TOPOLOGY}
        saved_parent = chk.load_mesh()
        potential = chk.load_function(
            saved_parent, "potential", idx=idx)

    def transfer(source, target_mesh, label):
        target = Function(source.function_space().reconstruct(mesh=target_mesh),
                          name=source.name())
        source_coordinates = source.function_space().mesh().coordinates
        target_coordinates = target_mesh.coordinates
        if source_coordinates.dat.data_ro.shape != target_coordinates.dat.data_ro.shape:
            raise RuntimeError(
                f"{label} coordinate layout differs between the checkpoint "
                "and the reconstructed mesh.")
        local_coordinate_error = np.max(np.abs(
            source_coordinates.dat.data_ro - target_coordinates.dat.data_ro),
            initial=0.0)
        coordinate_error = COMM_WORLD.allreduce(
            local_coordinate_error, op=MPI.MAX)
        say(f"CHECKPOINT_MAP field={label} "
            f"coordinate_error={coordinate_error:.16e}")
        if coordinate_error > 1.0e-12:
            raise RuntimeError(
                f"{label} checkpoint ordering does not match the "
                f"reconstructed mesh: coordinate error {coordinate_error:.3e}.")
        if source.dat.data_ro.shape != target.dat.data.shape:
            raise RuntimeError(
                f"{label} data layout differs between the checkpoint and "
                "the reconstructed mesh.")
        target.dat.data[:] = source.dat.data_ro
        return target

    return (transfer(displacement, sub, "displacement"),
            transfer(potential, parent, "potential"),
            transfer(internal_variable, sub, "internal_variable_0"))


def assign_restart(solver, z, layout, displacement, potential,
                   internal_variable):
    """Put one saved physical state in a new solver and its old state."""
    z.subfunctions[layout.displacement].assign(displacement)
    z.subfunctions[layout.potential].assign(potential)
    if layout.condensed:
        # `b1.build_solver` hands the solver a list of one (d, d) field on
        # the condensed layout, the layout the checkpoint holds, so the
        # stored history is that list.
        solver.internal_variables[0].assign(internal_variable)
    else:
        # The mixed space carries one combined (n, d, d) field; the
        # checkpoint's (d, d) field is its single slice.
        assign_history_slices(
            z.subfunctions[layout.internal_variable_field], [internal_variable])
    solver.solution_old.assign(z)


def internal_variable(solver, z, layout):
    """The stored history as a Function: (d, d) condensed, (1, d, d) otherwise."""
    if layout.condensed:
        return solver.internal_variables[0]
    return z.subfunctions[layout.internal_variable_field]


def history_slice(m):
    """The single (d, d) Maxwell element of `m`, whatever its storage."""
    return history_slices(m)[0]


def state_norms(solver, z, layout):
    u = z.subfunctions[layout.displacement]
    psi = z.subfunctions[layout.potential]
    m = internal_variable(solver, z, layout)
    return norm(u), norm(psi), norm(m)


def core_flux(solver, z, layout):
    """Return mean radial CMB motion and its share of the CMB radial norm."""
    u = z.subfunctions[layout.displacement]
    mesh = solver.mesh
    X = SpatialCoordinate(mesh)
    rhat = X / sqrt(dot(X, X))
    measure = ds(domain=mesh)(b1.gen.SURF_RC)
    area = assemble(Constant(1.0) * measure)
    mean = assemble(dot(u, rhat) * measure) / area
    rms = sqrt(assemble(dot(u, rhat) ** 2 * measure) / area)
    return float(mean), float(rms), abs(float(mean)) / max(float(rms), 1.0e-300)


def weak_history_residual(solver, z, layout, m_old):
    """Return the relative DG Riesz norm of the Maxwell weak residual."""
    stored = internal_variable(solver, z, layout)
    m = history_slice(stored)
    m_old = history_slice(m_old)
    u = z.subfunctions[layout.displacement]
    strain = solver.approximation.deviatoric_strain(u)
    maxwell_time = solver.approximation.maxwell_times[0]
    residual = ((m - m_old) / solver.dt
                + (m - strain) / maxwell_time)
    # On the uncondensed layout `m` is a (d, d) slice of the combined
    # (1, d, d) field, a UFL expression without a function space, so the
    # Riesz representative is projected into a plain (d, d) DG space of the
    # stored field's degree; on the condensed layout that is the stored
    # field's own space.
    degree = stored.function_space().ufl_element().embedded_superdegree
    riesz_space = TensorFunctionSpace(solver.mesh, "DG", degree)
    riesz = Function(riesz_space).project(residual)
    scale = Function(riesz_space).project(strain / maxwell_time)
    denominator = max(norm(scale), 1.0e-300)
    return norm(riesz), norm(riesz) / denominator


def gather_owned(field):
    """Concatenate the owned values of a scalar field from every rank on rank 0.

    Halo entries are excluded (`dat.data_ro` holds the owned part), so each
    cell or node is counted once. Returns None on the other ranks.
    """
    parts = COMM_WORLD.gather(np.array(field.dat.data_ro, copy=True), root=0)
    return np.concatenate(parts) if COMM_WORLD.rank == 0 else None


def weighted_percentile(values, weights, q):
    """The `q`-th percentile of `values` with each entry weighted by `weights`.

    Used with cell values and cell volumes, so the percentile is a fraction of
    the shell volume and not of the cell count, which the mesh grading would
    otherwise bias towards the fine surface cells.
    """
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order][np.searchsorted(cumulative, q / 100.0)])


def power_law_factor_of(stress_ratio, exponent):
    """f = 1 / (1 + (sigma/sigma*)^(n-1)) for zero background stress.

    The Maxwell time of a power-law shell is `f` times its Newtonian value, so
    `f` below 1 is the drop in effective viscosity. Matches
    `CompressibleInternalVariableApproximation.power_law_factor` with
    `background_stress = 0`; for n = 1 that method returns 1, and so does this.
    """
    if exponent == 1.0:
        return np.ones_like(np.asarray(stress_ratio, dtype=float))
    return 1.0 / (1.0 + np.asarray(stress_ratio, dtype=float) ** (exponent - 1.0))


def stress_report(label, sub, displacement, history, *, exponent,
                  transition_stress_mpa, layers, dt_yr, degree):
    """Print the deviatoric stress of one state per Spada shell, and what a
    power law would make of it.

    The stress is `deviatoric_stress(u, [m])` of the Newtonian approximation,
    `2 mu0 dev(eps(u)) - 2 mu m` in units of `MU_BAR` (1e11 Pa); the rheology
    does not enter it, so the same numbers describe the Newtonian checkpoint
    and the state a power-law step starts from. Its invariant is
    `second_stress_invariant`, sqrt(J2), reported in MPa in the convention of
    `--transition-stress-mpa` (sqrt(2 J2), see `b1.stress_mpa_to_sqrt_j2`),
    so the two numbers compare directly.

    Two samplings, because a maximum and a distribution need different ones:
    the cell mean (projection onto DG0), weighted by cell volume, for the
    percentiles; and the nodal values of the interpolant in DG `degree` for
    the maximum, which a cell mean would smooth away near the load edge.

    For each shell the report gives the Newtonian Maxwell time `tau`, the
    factor `f` at each sampled stress for `exponent` and
    `transition_stress_mpa` (1 in shells outside `layers`), and the step
    ratio `dt / (tau f)` at `dt_yr`. On the 2-D annulus Newton converged with
    that ratio up to about 10 and failed from about 25
    (`NOTES/PLAN-POWER-LAW-SELFGRAVITY.md` section 2.1). The last column is
    the largest step that keeps the ratio at the shell maximum below 5.
    """
    approx = b1.spada_approximation(sub)
    dev = approx.deviatoric_stress(displacement, [history])
    invariant = approx.second_stress_invariant(dev)
    # sqrt(J2) in units of MU_BAR -> sqrt(2 J2) in MPa.
    to_mpa = b1.MU_BAR / 1.0e6 * 2.0 ** 0.5

    dg0 = FunctionSpace(sub, "DG", 0)
    cell_mean = Function(dg0).project(invariant)
    volume = Function(dg0).interpolate(CellVolume(sub))
    index = b1.layer_index(sub)
    # The nodal maximum per shell: interpolate the invariant times the shell
    # indicator, which is exact at the nodes because the indicator is DG0.
    dgk = FunctionSpace(sub, "DG", degree)
    shell_nodal_max = []
    for number in range(1, len(b1.LAYERS) + 1):
        indicator = conditional(abs(index - float(number)) < 0.5, 1.0, 0.0)
        nodal = Function(dgk).interpolate(invariant * indicator)
        with nodal.dat.vec_ro as vec:
            shell_nodal_max.append(vec.max()[1] * to_mpa)

    means = gather_owned(cell_mean)
    volumes = gather_owned(volume)
    shells = gather_owned(index)
    if COMM_WORLD.rank != 0:
        return
    means = means * to_mpa
    say(f"STRESS state={label} transition_stress_mpa={transition_stress_mpa:g} "
        f"exponent={exponent:g} layers={','.join(map(str, layers))} "
        f"dt_yr={dt_yr:g}")
    for number, row in enumerate(b1.LAYERS, start=1):
        mask = np.abs(shells - number) < 0.5
        values, weights = means[mask], volumes[mask]
        tau_yr = row[4] / row[3] * T_BAR_YR
        n = exponent if number in layers else 1.0
        p50, p90, p99 = (weighted_percentile(values, weights, q)
                         for q in (50, 90, 99))
        peak = shell_nodal_max[number - 1]
        above = float(weights[values > transition_stress_mpa].sum()
                      / weights.sum())
        f = power_law_factor_of(
            np.array([p50, p90, p99, peak]) / transition_stress_mpa, n)
        ratio = dt_yr / (tau_yr * f)
        dt_limit = 5.0 * tau_yr * f[-1]
        say(f"STRESS state={label} shell={number} n={n:g} tau_yr={tau_yr:.4g} "
            f"sqrtJ2_mpa p50={p50:.4g} p90={p90:.4g} p99={p99:.4g} "
            f"nodal_max={peak:.4g} volume_above_transition={above:.4g} "
            f"f p50={f[0]:.3g} p90={f[1]:.3g} p99={f[2]:.3g} max={f[3]:.3g} "
            f"dt_over_tau_f p50={ratio[0]:.3g} p90={ratio[1]:.3g} "
            f"p99={ratio[2]:.3g} max={ratio[3]:.3g} "
            f"dt_yr_for_ratio_5={dt_limit:.4g}")


def report_state(tag, t_kyr, solver, z, layout, ref, sigma_dim, nmax,
                 nproj, theta_fine):
    u_norm, psi_norm, m_norm = state_norms(solver, z, layout)
    cmb_mean, cmb_rms, cmb_monopole = core_flux(solver, z, layout)
    p_core = (float(solver.core_pressure)
              if solver.core_pressure is not None else float("nan"))
    say(f"STATE arm={tag} t_kyr={t_kyr:.1f} "
        f"u_norm={u_norm:.16e} psi_norm={psi_norm:.16e} "
        f"m_norm={m_norm:.16e} core_pressure={p_core:.16e} "
        f"cmb_ur_mean={cmb_mean:.16e} "
        f"cmb_ur_rms={cmb_rms:.16e} cmb_monopole={cmb_monopole:.16e}")
    U_n, V_n, N_n = b1.surface_spectra(
        solver, solver.potential_mesh, solver.mesh, nproj,
        quad_degree=40, project_out_nullspace=False)
    row = b5.compare_epoch(t_kyr, U_n, V_n, N_n, ref, sigma_dim,
                           nmax, theta_fine)
    say(f"SPECTRUM arm={tag} t_kyr={t_kyr:.1f} "
        f"U0={row['U0']:.16e} U0_ratio={row['U0'] / row['U0_ref']:.16e} "
        f"N0={row['N0']:.16e} N0_ratio={row['N0'] / row['N0_ref']:.16e}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        choices=("condensed", "uncondensed"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-index", type=int, default=1)
    # The model time of the checkpoint index, used only to label the STEP,
    # STATE and SPECTRUM lines and to pick the reference epoch they compare
    # against. The P3 20 kyr march wrote indices 0 to 6 at 0, 0.1, 1, 2, 5,
    # 10 and 20 kyr; index 1 of the P2 gate files is 1 kyr.
    parser.add_argument("--checkpoint-time-kyr", type=float, default=1.0)
    parser.add_argument("--mesh", default="b2_coarse_ar7.msh")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt-yr", type=float, default=100.0)
    parser.add_argument("--nmax", type=int, default=10)
    parser.add_argument("--dtn-degree", type=int, default=5)
    # The P3 20 kyr march (`b5-p3-full-20260817.h5`) ran CG3 displacement and
    # DG2 internal variables; the checkpoint transfer refuses any other
    # layout. The defaults are that march's, so a restart from it needs no
    # degree arguments; the P2 files need 2 and 1.
    parser.add_argument("--displacement-degree", type=int, default=3)
    parser.add_argument("--internal-variable-degree", type=int, default=2)
    parser.add_argument("--bulk-shear-ratio", type=float, default=100.0)
    parser.add_argument("--outer-rtol", type=float, default=1.0e-6)
    parser.add_argument("--block0-rtol", type=float, default=1.0e-4)
    parser.add_argument("--block0-max-it", type=int, default=200)
    # The displacement preconditioner inside block 0, on every arm. The
    # modes GAMG is seeded with: `incompressible` (rigid plus the low-degree
    # divergence-free fields) or `rigid` (the modes the P3 march ran). And the
    # Krylov method on the displacement split: 0 is one GAMG V-cycle per
    # block-0 iteration (`preonly`, the P3 march); N > 0 is CG with at most N
    # iterations to the given relative tolerance, counted as converged at
    # the cap so the block-0 FGMRES keeps going.
    #
    # Both defaults are the library preset's, so a run with no displacement
    # flag measures what `gadopt` gives a caller who states nothing. Measured
    # at 500 yr steps on `b2_coarse_ar7.msh` at 96 ranks, both jobs at block-0
    # cap 400 so that the only difference is the two flags: that pair took 111
    # block-0 applications at 46 inner iterations each, none at the cap, and
    # 271 s for the warm step (Gadi job `178765560`); `rigid` with one V-cycle
    # took 124 applications at 272 inner iterations, 64 at the cap, and 565 s
    # (job `178765557`).
    parser.add_argument("--near-nullspace",
                        choices=("rigid", "incompressible"),
                        default="incompressible")
    parser.add_argument("--displacement-ksp-max-it", type=int, default=4)
    parser.add_argument("--displacement-ksp-rtol", type=float, default=1.0e-2)
    parser.add_argument("--ablation",
                        choices=("none", "gravity-feedback",
                                 "gravity-volume-feedback",
                                 "gravity-core-feedback"),
                        default="none")
    parser.add_argument("--load-only", action="store_true")
    # Power-law rheology (`b1.power_law_rheology`): exponent n in the shells
    # of --power-law-layers (1-based rows of b1.LAYERS; 2 3 is the upper
    # mantle from 70 to 670 km, the layering of 3d_weerdesteijn_coupled), 1
    # elsewhere. The transition stress is in MPa in the convention of that
    # test, sqrt(2 J2); 0.2 MPa is its production value. Exponent 1 builds the
    # Newtonian approximation unchanged.
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--transition-stress-mpa", type=float, default=0.2)
    parser.add_argument("--background-stress-mpa", type=float, default=0.0)
    parser.add_argument("--power-law-layers", default="2 3",
                        help="space-separated 1-based rows of b1.LAYERS")
    # The outer method. Unset: newtonls for a power law, ksponly for a
    # Newtonian rheology (the solver refuses ksponly on a power law).
    parser.add_argument("--snes-type", choices=("ksponly", "newtonls"),
                        default=None)
    parser.add_argument("--snes-rtol", type=float, default=None,
                        help="Newton relative tolerance on the whole mixed "
                             "residual; unset keeps the preset's 1e-4.")
    # The stress check: load each listed checkpoint index, print the
    # per-shell deviatoric stress table of `stress_report` for the power-law
    # arguments above and --dt-yr, and exit before any solver is built.
    parser.add_argument("--stress-report", action="store_true")
    # Space-separated, because `qsub -v` splits its argument at commas.
    parser.add_argument("--stress-indices", type=int, nargs="+", default=None,
                        help="checkpoint indices for --stress-report; unset "
                             "reads --checkpoint-index")
    parser.add_argument(
        "--dtn-representation", choices=("multiplier", "lowrank"),
        default=None,
        help="how the exterior DtN condition enters the coupled system: one "
             "Real unknown per harmonic mode (multiplier) or a rank-k update "
             "on the potential rows (lowrank). Unset follows the library: "
             "lowrank on --arm uncondensed, multiplier on --arm condensed. "
             "lowrank needs --arm uncondensed and --block0 condensed; the "
             "condensed-pair and sweep arms need multiplier, named.")
    parser.add_argument(
        "--block0", choices=("condensed", "condensed-pair", "sweep"),
        default="condensed",
        help="Uncondensed arm only. 'condensed' (default) is "
             "gadopt.CondensedBlockPC on block 0: the internal-variable "
             "field is eliminated once per block-0 application and the "
             "block-0 Krylov solve runs on the assembled (u, psi) condensed "
             "system, the preset's default. 'condensed-pair' eliminates the "
             "field inside block 0 with gadopt.InternalVariableSCPC on the "
             "(u, M) split of a two-split sweep, so the elimination runs on "
             "every block-0 inner iteration; the T2 route, kept for "
             "comparison. 'sweep' is the three-way multiplicative sweep over "
             "m, u, psi that the P3 march used, kept for comparison.")
    args = parser.parse_args()

    condense = args.arm == "condensed"
    power_law_layers = tuple(int(x) for x in args.power_law_layers.split())
    if args.snes_type is None:
        args.snes_type = "newtonls" if args.exponent != 1.0 else "ksponly"
    if args.exponent != 1.0 and condense:
        # The library refuses pointwise substitution under a power law too,
        # but its check calls float() on the exponent, which is a DG0 field
        # here, so it would fail with a TypeError and no explanation.
        raise SystemExit(
            "--exponent != 1 needs --arm uncondensed: pointwise substitution "
            "of the internal variables changes the Newton linearisation of a "
            "power law, so the condensed layout refuses it.")
    named = args.dtn_representation is not None
    # The library's rule, not a copy of it: unset resolves to low-rank on the
    # full layout and multiplier on the condensed one. The nested and sweep
    # block-0 routes exist on the multiplier representation alone and are
    # not part of that rule, so an unset flag on those arms is refused below
    # with the flag to pass.
    args.dtn_representation = resolve_dtn_representation(
        args.dtn_representation, condensed=condense)
    if args.dtn_representation == "lowrank" and (
            condense or args.block0 != "condensed"):
        # The library refuses both combinations as well, with messages that
        # name the fix; refusing here saves the mesh read and the checkpoint
        # load before the same message.
        raise SystemExit(
            ("--dtn-representation lowrank" if named else
             "the default representation on --arm uncondensed is lowrank, which")
            + " needs --arm uncondensed and --block0 condensed: the low-rank "
            "update lives on the potential rows of the full layout and is "
            "preconditioned on the potential split of gadopt.CondensedBlockPC. "
            "The condensed-pair and sweep arms run on the multiplier "
            "representation: pass --dtn-representation multiplier.")
    tag = (args.arm if args.ablation == "none"
           else f"{args.arm}+{args.ablation}")
    dt = Constant(args.dt_yr / T_BAR_YR)
    theta_fine = np.linspace(0.0, np.pi, 4001)
    ref = taboo.TabooReference(os.path.join(HERE, "reference.npz"))
    sigma_dim = taboo.cap_load(args.nmax)

    say("=" * 78)
    say("B5 condensation restart discriminator")
    say("=" * 78)
    say(f"arm={args.arm} checkpoint={args.checkpoint} "
        f"index={args.checkpoint_index}")
    say(f"mesh={args.mesh} dt_yr={args.dt_yr:g} steps={args.steps} "
        f"nmax={args.nmax} dtn_degree={args.dtn_degree}")
    say(f"displacement_degree={args.displacement_degree} "
        f"internal_variable_degree={args.internal_variable_degree} "
        f"block0={args.block0 if not condense else 'condensed-layout'} "
        f"dtn_representation={args.dtn_representation}")
    say(f"ablation={args.ablation}")
    say(f"outer_rtol={args.outer_rtol:g} "
        f"block0_rtol={args.block0_rtol:g} block0_max_it={args.block0_max_it}")
    say(f"near_nullspace={args.near_nullspace} "
        f"displacement_ksp_max_it={args.displacement_ksp_max_it} "
        f"displacement_ksp_rtol={args.displacement_ksp_rtol:g}")

    parent, sub, _, _ = b1.build_meshes(
        "coarse", path=args.mesh)
    displacement, potential, history = load_restart(
        args.checkpoint, parent, sub, args.checkpoint_index)
    say(f"RESTART u_norm={norm(displacement):.16e} "
        f"psi_norm={norm(potential):.16e} m_norm={norm(history):.16e}")
    if args.stress_report:
        indices = args.stress_indices or [args.checkpoint_index]
        for idx in indices:
            # Separate names, so the restart state loaded above is never
            # replaced by another index's fields.
            if idx == args.checkpoint_index:
                u_idx, m_idx = displacement, history
            else:
                u_idx, _, m_idx = load_restart(
                    args.checkpoint, parent, sub, idx)
            stress_report(
                f"index{idx}", sub, u_idx, m_idx,
                exponent=args.exponent,
                transition_stress_mpa=args.transition_stress_mpa,
                layers=power_law_layers, dt_yr=args.dt_yr,
                degree=args.internal_variable_degree)
        say("RESULT stress_report=pass")
        return
    if args.load_only:
        say("RESULT load_only=pass")
        return

    solver_kwargs_extra = {}
    if condense:
        # The condensed layout's own preset, built here so that its block-0
        # cap and displacement options can be set below.
        solver_parameters = b1.condensed_solver_parameters(
            outer_rtol=args.outer_rtol, block0_rtol=args.block0_rtol,
            block0_max_it=args.block0_max_it, snes_type=args.snes_type,
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC")
        displacement_prefix = "dtn_fieldsplit_0_fieldsplit_0_"
    elif args.block0 == "sweep":
        solver_parameters = b1.b2_solver_parameters(
            multiplier_pc="none",
            block0_rtol=args.block0_rtol,
            outer_rtol=args.outer_rtol,
            block0_max_it=args.block0_max_it,
            u_pc="gadopt.RigidBodyAssembledPC")
        solver_parameters.update({
            "dtn_fieldsplit_1_pc_type": "python",
            "dtn_fieldsplit_1_pc_python_type":
                "gadopt.DtNMultiplierDenseSchurPC",
        })
        # In the three-way sweep `u` is split 1 (`m` is split 0).
        displacement_prefix = "dtn_fieldsplit_0_fieldsplit_1_"
    elif args.block0 == "condensed-pair":
        # Static condensation of the (u, M) pair inside block 0, the dense
        # Schur complement on the Real block, one linear solve per step.
        solver_parameters = selfgrav_dtn_iterative_solver_parameters(
            condensed=False,
            # This arm is the nested `InternalVariableSCPC` route, whose
            # displacement split sits at
            # `dtn_fieldsplit_0_fieldsplit_0_condensed_field_`; the preset's
            # default block 0 writes a different prefix and would leave the
            # options below unread.
            block0="pair",
            block0_rtol=args.block0_rtol,
            outer_rtol=args.outer_rtol,
            block0_max_it=args.block0_max_it,
            snes_type=args.snes_type,
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
            dtn_representation=args.dtn_representation)
        displacement_prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
    else:
        # `gadopt.CondensedBlockPC` on block 0: the internal-variable field is
        # eliminated once per block-0 application and the block-0 Krylov
        # solve runs on the assembled (u, psi) condensed system, so neither
        # the block-0 Krylov vectors nor the per-iteration work carry the
        # DG history field. The displacement split of that solve is split 0
        # of the nest, under `dtn_fieldsplit_0_condensed_`. The dense Schur
        # complement on the Real block and one linear solve per step, as on
        # the other arms.
        solver_parameters = selfgrav_dtn_iterative_solver_parameters(
            condensed=False,
            block0="condensed",
            block0_rtol=args.block0_rtol,
            outer_rtol=args.outer_rtol,
            block0_max_it=args.block0_max_it,
            snes_type=args.snes_type,
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
            dtn_representation=args.dtn_representation)
        displacement_prefix = "dtn_fieldsplit_0_condensed_fieldsplit_0_"

    if args.snes_rtol is not None:
        solver_parameters["snes_rtol"] = args.snes_rtol

    # The displacement preconditioner, on whichever split carries it. For
    # the assembled-block classes the modes are a PETSc option they read at
    # setup; for the condensation class they come from the solver keyword,
    # which acts only when no outer near-nullspace is declared, so the
    # builder's rigid-body basis is switched off in that case.
    if condense or args.block0 == "sweep":
        solver_parameters[displacement_prefix + "near_nullspace"] = \
            args.near_nullspace
        near_nullspace_kw = {}
    else:
        solver_kwargs_extra["condensed_near_nullspace"] = args.near_nullspace
        near_nullspace_kw = {"near_nullspace": args.near_nullspace == "rigid"}
    if args.displacement_ksp_max_it > 0:
        solver_parameters.update({
            displacement_prefix + "ksp_type": "cg",
            displacement_prefix + "ksp_max_it": args.displacement_ksp_max_it,
            displacement_prefix + "ksp_rtol": args.displacement_ksp_rtol,
            displacement_prefix + "ksp_converged_maxits": None,
            displacement_prefix + "ksp_converged_reason": None,
        })
    else:
        solver_parameters[displacement_prefix + "ksp_type"] = "preonly"

    original_solver_class = b1.SelfGravitatingGIASolver
    ablated_solvers = {
        "gravity-feedback": GravityFeedbackAblatedSolver,
        "gravity-volume-feedback": GravityVolumeFeedbackAblatedSolver,
        "gravity-core-feedback": GravityCoreFeedbackAblatedSolver,
    }
    if args.ablation != "none":
        b1.SelfGravitatingGIASolver = ablated_solvers[args.ablation]
    solver, z, layout, _, _, _ = b1.build_solver(
        parent, sub, args.nmax, dtn_degree=args.dtn_degree,
        udeg=args.displacement_degree,
        ivdeg=args.internal_variable_degree,
        condense=condense, outer_rtol=args.outer_rtol,
        bulk_shear_ratio=args.bulk_shear_ratio,
        u_pc="gadopt.RigidBodyAssembledPC", snes_type=args.snes_type,
        block0_rtol=args.block0_rtol, block0_max_it=args.block0_max_it,
        multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
        solver_parameters=solver_parameters, dt=dt,
        solver_kwargs_extra=solver_kwargs_extra,
        dtn_representation=args.dtn_representation,
        power_law=b1.power_law_rheology(
            sub, exponent=args.exponent,
            transition_stress_mpa=args.transition_stress_mpa,
            background_stress_mpa=args.background_stress_mpa,
            layers=power_law_layers),
        **near_nullspace_kw)
    b1.SelfGravitatingGIASolver = original_solver_class
    assign_restart(solver, z, layout, displacement, potential, history)

    report_state(tag, args.checkpoint_time_kyr, solver, z, layout, ref, sigma_dim,
                 args.nmax, args.nmax, theta_fine)
    previous_u_norm = state_norms(solver, z, layout)[0]
    for step in range(1, args.steps + 1):
        m_old = internal_variable(solver, z, layout).copy(deepcopy=True)
        start = time.time()
        solver.solve()
        elapsed = time.time() - start
        u_norm = state_norms(solver, z, layout)[0]
        snes = solver.solver.snes
        # One NEWTON line per step: nonlinear and total outer Krylov
        # iterations, the converged reason, and the final residual norm, so a
        # power-law step's cost splits into Newton iterations without parsing
        # the monitor.
        say(f"NEWTON arm={tag} step={step} "
            f"snes_its={snes.getIterationNumber()} "
            f"outer_ksp_its={snes.getLinearSolveIterations()} "
            f"reason={snes.getConvergedReason()} "
            f"fnorm={snes.getFunctionNorm():.6e}")
        if args.exponent != 1.0:
            # The stress table of the solved state, for the shells' actual
            # viscosity drop within this step and the step size the next
            # one could take.
            stress_report(
                f"step{step}", sub, z.subfunctions[layout.displacement],
                history_slice(internal_variable(solver, z, layout)),
                exponent=args.exponent,
                transition_stress_mpa=args.transition_stress_mpa,
                layers=power_law_layers, dt_yr=args.dt_yr,
                degree=args.internal_variable_degree)
        weak_abs, weak_rel = weak_history_residual(
            solver, z, layout, m_old)
        t_kyr = args.checkpoint_time_kyr + step * args.dt_yr / 1000.0
        say(f"STEP arm={tag} step={step} t_kyr={t_kyr:.1f} "
            f"wall_s={elapsed:.6f} u_amplification={u_norm / previous_u_norm:.16e} "
            f"weak_history_abs={weak_abs:.16e} "
            f"weak_history_rel={weak_rel:.16e}")
        report_state(tag, t_kyr, solver, z, layout, ref, sigma_dim,
                     args.nmax, args.nmax, theta_fine)
        previous_u_norm = u_norm

    say(f"RESULT arm={tag} completed_steps={args.steps}")


if __name__ == "__main__":
    main()
