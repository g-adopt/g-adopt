r"""GIAMIP Exp01: the bed at 122 ka from the observed present-day bed, by adjoint.

THE PROBLEM
    The GIAMIP topography file is the observed bed of the present day (model
    year 0). The run of `exp01.py` must start at 122 ka from a bed that is
    not known: the bed that the run carries to the observed bed at year 0.
    This script finds that bed by minimising, over the bed at 122 ka,

        J(b) = 1/2 int_Re ((SL(year 0; b) - SL_obs) D)^2 dS

    where `SL = SL_init + (N - N_init) - (u_r - ur_init) + Shift` is the sea
    level of the solver (`SelfGravitatingGIASolver.sea_level`), `SL_init =
    -b / D` is the initial sea level made from the bed `b`, `SL_obs = -b_obs /
    D` is the sea level of the observed bed, and `dS` is the non-dimensional
    area of the surface Re (area A = 4 pi Re^2 = 61.0). Since `bed = -SL D`,
    the integrand is the square of the misfit of the predicted present-day
    bed in metres, and `sqrt(2 J / A)` is its RMS value in metres. The misfit
    has the same weight everywhere. There is no smoothing and no damping term
    (Sia, 2026-10-15, handover section 18).

    The scale makes the curvature of J about 1 in the L2 norm of the control.
    The predicted bed is `b` plus the sea-level change of the run, which
    depends only weakly on `b` (through the ocean load and the coastline), so
    the Hessian of J is close to the mass matrix of the surface for any
    length of the march. ROL Lin-More starts from an identity model of the
    Hessian, and a curvature far from 1 makes its first steps useless: with
    the mean square in units of 100 m the curvature was 3e-6 and the first
    step moved the bed by less than a millimetre (job 50903390).

THE CONTROL
    `b`, the bed at 122 ka in metres, a CG2 field on the surface Re: a
    codim-1 `Submesh` of the mantle (`exp01_surface`, the mesh of the
    output files of `exp01.py`). The solver reads `SL_init`, a CG2 field on
    the mantle, only on the surface facets. One annotated interpolation maps
    the control to `SL_init` (`allow_missing_dofs=True`: the interior nodes
    of the mantle are not on the surface and are not used). The optimisation
    starts from the observed bed (cold start).

    The control carries an L2 Riesz map (`RieszMap` on the surface space,
    built once, the solver of Firedrake's `project`, `RIESZ_SOLVER_PARAMETERS`). ROL gets
    the gradient through it, and the Taylor direction uses it too.

WHAT IS ON THE TAPE
    Only the operations from the control to J:

      1. the interpolation of the control into `SL_init`;
      2. per time step: the assignment of `dt` when it changes, the
         assignment of the ice of the step, and the coupled solve with what
         the solver does around it (the null-space projection and the
         recovery of the internal variables);
      3. the misfit assemble.

    Everything else runs with annotation paused: the meshes, the earth
    model, the solver construction, the frozen surface slope of the ocean
    mask (`surface_slope` of the observed bed, never of the control), the
    observed sea level, the initial ice, the output files and every logged
    number. The ice of each step is read into a NEW Function with annotation
    paused and then assigned to the solver's ice field on the tape: a reused
    source Function would be saved once, and the replay would use the first
    ice field at every step.

CHECKPOINTING
    The time loop runs in `tape.timestepper` with
    `SingleDiskStorageSchedule`. The checkpoint data goes to node-local
    storage, one checkpoint communicator per node
    (`MPI.COMM_TYPE_SHARED`), in `$TMPDIR` (or `--checkpoint_dir`). On a
    Setonix compute node `/tmp` is a RAM file system of 126 GB, so the
    checkpoints use node memory.

FIREDRAKE
    The script runs on Firedrake 2026.10.0 and on the PR 4638 overlay
    (`env-exp01-pr4638.sh`); G-ADOPT finds out which one it runs on
    (`gadopt.dtn_coupled_adjoint.CACHED_SOLVER_BLOCKS`). The overlay is the
    one that clones the application context for the replay solvers.

MODES
    --mode taylor   tape one forward run, check that a replay at the taped
                    control gives the taped J, and run a first-order Taylor
                    test in the direction of the gradient (or a random
                    direction), of largest value `--taylor_h_m`.
    --mode invert   minimise J with ROL Lin-More (L2 Riesz map), for
                    `--iterations` iterations.

    `--steps n` stops the march after n steps (debugging); the misfit is
    then taken at the last step against the observed bed, which tests the
    machinery and has no physical meaning. Without `--steps` the march ends
    at year 0.

RUN
    srun python3 exp01_topography.py --mesh <file.msh> --output_path <dir>
        --mode taylor --steps 3
    srun python3 exp01_topography.py --mesh <file.msh> --output_path <dir>
        --mode invert --steps 10 --iterations 10

OUTPUT (in --output_path)
    topography_exp01.log    the log lines of rank 0 also go to stdout
    summary_topography.json the settings, the forward record, the Taylor
                            residuals or the ROL iterations
    controls.h5             the control (bed at 122 ka, m) of every ROL
                            iteration, with the observed bed (CheckpointFile
                            on the surface submesh)
    optimisation_checkpoint/  ROL state of every iteration (gadopt.inverse)
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

# The driver of the forward run: its setup functions build the meshes, the
# forcing and the solver exactly as the forward run does. It imports gadopt
# before firedrake.
import exp01 as driver  # noqa: E402

import numpy as np  # noqa: E402
from checkpoint_schedules import SingleDiskStorageSchedule  # noqa: E402
from firedrake import CheckpointFile, RieszMap, Submesh, dx  # noqa: E402
from firedrake.adjoint import (Control, ReducedFunctional,  # noqa: E402
                               continue_annotation, enable_disk_checkpointing,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from gadopt import COMM_WORLD, Constant, Function, FunctionSpace, assemble, log  # noqa: E402
# `gadopt.inverse` starts the tape when it is imported; the annotation is
# paused again at once, before anything is built.
from gadopt.inverse import LinMoreOptimiser, minimisation_parameters  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from pyadjoint import MinimizationProblem  # noqa: E402

pause_annotation()

import selfgrav_common as common  # noqa: E402
from giamip.mip.inputs import GiamipExperiment  # noqa: E402

D_SCALE = driver.D_SCALE

#: The solver of the L2 Riesz map of the control: the mass matrix of the CG2
#: surface space. The L2 Riesz map is the same solve as an L2 projection, so
#: it uses the default solver of Firedrake's `project` for an assembled
#: matrix (`firedrake/projection.py`): CG, block Jacobi with ICC on each
#: block, relative tolerance 1e-8. No factorisation: the default of a
#: `RieszMap` is a MUMPS LU built again at every gradient, and that solve
#: failed on Setonix with the PyOP2 cache-key fix (jobs 51055806 and
#: 51057023, DIVERGED_LINEAR_SOLVE in the Riesz solve of a ROL gradient).
RIESZ_SOLVER_PARAMETERS = {"ksp_type": "cg", "ksp_rtol": 1e-8,
                           "pc_type": "bjacobi", "sub_pc_type": "icc"}


# --------------------------------------------------------------------------
# Log
# --------------------------------------------------------------------------


class RunLog:
    """Log lines to stdout and to a file, on rank 0.

    The file keeps the record of the run next to its output, so a run can be
    read later without the Slurm output file.
    """

    def __init__(self, path):
        self.handle = open(path, "a") if COMM_WORLD.rank == 0 else None

    def __call__(self, text):
        log(text)
        if self.handle is not None:
            self.handle.write(text + "\n")
            self.handle.flush()


# --------------------------------------------------------------------------
# The forward problem on the tape
# --------------------------------------------------------------------------


class Exp01Topography:
    """The taped map from the bed at 122 ka to the misfit at the last step.

    Built in two parts. `__init__` builds everything that is not on the tape
    (annotation paused). `tape_forward` records the forward run once and
    returns J; pyadjoint replays it for every later evaluation.

    Args:
      args: the command line.
      say: the log function.
    """

    def __init__(self, args, say):
        self.args, self.say = args, say
        self.timings = {}
        tic = time.time()
        experiment = GiamipExperiment("Exp01")
        earth = experiment.earth()
        self.scales = driver.Scales(earth.elastic)
        sidecar = os.path.splitext(args.mesh)[0] + ".json"
        radial, _ = driver.read_radial(sidecar)
        self.parent, self.mantle, self.mesh_info = driver.build_meshes(
            args.mesh, radial, args.refine)
        say(f"  mesh: {self.mesh_info} ({time.time() - tic:.1f} s)")

        # The forcing: the observed bed in `SL_init` and the ice history.
        # The coarse step of the topography iteration keeps a subset of the
        # protocol years (handover section 17.2).
        self.forcing = driver.SurfaceForcing(self.mantle, experiment,
                                             step_years=args.step_years)
        # The observed sea level, frozen: the target of the misfit. A copy,
        # because `SL_init` itself is written from the control.
        self.SL_obs = Function(self.forcing.SL_init.function_space(),
                               name="SL_obs").assign(self.forcing.SL_init)

        # The solver. `build_solver` freezes the surface slope of the ocean
        # mask from the observed bed (`surface_slope` returns a Function made
        # with annotation paused), so the mask width does not depend on the
        # control.
        self.dt = Constant(1.0)
        tic = time.time()
        self.solver, self.layout, _, self.solver_record = driver.build_solver(
            self.parent, self.mantle, earth, self.scales, self.forcing,
            self.dt, args)
        say(f"  solver: {self.solver_record['unknowns']} unknowns "
            f"({time.time() - tic:.1f} s)")

        # The surface mesh of the control: the same codim-1 Submesh as the
        # output files of `exp01.py`, with the P2 geometry of the mantle.
        self.surface = Submesh(self.mantle, 2, driver.SURF_RE,
                               name="exp01_surface")
        self.V_bed = FunctionSpace(self.surface, "CG", 2)
        # The L2 Riesz map of the control, built once and given to the
        # Control. Without it, every gradient that ROL asks for
        # (`derivative(apply_riesz=True)`) builds a new RieszMap from the
        # string "L2", and with it a new solver, a new mass-matrix assembly
        # and a new factorisation. `constant_jacobian=True`: the mass matrix
        # does not depend on the control, so it is assembled once.
        self.riesz_map = RieszMap(self.V_bed, "L2", constant_jacobian=True,
                                  solver_parameters=RIESZ_SOLVER_PARAMETERS)
        # The observed bed on the surface, m. Mantle to surface follows the
        # submesh map (no point search).
        self.bed_obs = Function(self.V_bed, name="bed_observed_m")
        self.bed_obs.interpolate(-self.SL_obs * D_SCALE)

        # The steps of the march: 1 .. last; the last is year 0 unless
        # `--steps` stops earlier.
        years = np.asarray(self.forcing.years, dtype=float)
        year0 = int(np.flatnonzero(np.isclose(years, 0.0))[0])
        self.last = year0 if args.steps is None else min(year0, args.steps)
        self.years = years

        # The surface measure and area of the misfit (the solver's own sea-
        # level measure, so the misfit integrates SL as the solver does).
        self.dss = self.solver.sea_level_measure()(driver.SURF_RE)
        self.area = float(assemble(Constant(1.0) * self.dss))
        self.forward_rows = []

    def misfit(self):
        """The misfit J of the current state, as a taped assemble.

        `1/2 int ((SL - SL_obs) D)^2 dS`: half the integral of the square of
        the bed misfit in metres over the non-dimensional surface.
        """
        SL = self.solver.sea_level()
        weight = 0.5 * D_SCALE ** 2
        return assemble(Constant(weight) * (SL - self.SL_obs) ** 2 * self.dss)

    def rms_misfit_m(self, J):
        """The RMS misfit of the predicted present-day bed in metres."""
        return float(np.sqrt(2.0 * float(J) / self.area))

    def tape_forward(self, bed):
        """Record the forward run from the bed `bed` and return J.

        Annotation must be on, and disk checkpointing enabled, before this
        is called. The march runs in `tape.timestepper`, one tape step per
        model step.

        Args:
          bed: the control, the bed at 122 ka on the surface, m.

        Returns:
          J, an `AdjFloat`.
        """
        args, say, forcing, solver = self.args, self.say, self.forcing, self.solver
        tape = get_working_tape()

        # 1. The control to the initial sea level. Only the surface nodes of
        # `SL_init` are located on the surface mesh; the solver reads no
        # other node.
        forcing.SL_init.interpolate(-bed / D_SCALE, allow_missing_dofs=True)

        previous_dt = None
        for index in tape.timestepper(iter(range(1, self.last + 1))):
            dt_yr = float(self.years[index] - self.years[index - 1])
            # `dt` on the tape when it changes: a change made with annotation
            # paused would leave the replay with the first value.
            if dt_yr != previous_dt:
                self.dt.assign(dt_yr / driver.T_BAR_YR)
                previous_dt = dt_yr
            # The ice of the step: a new Function filled with annotation
            # paused, then one assignment on the tape (backward Euler: the
            # load at the end of the step).
            with stop_annotating():
                ice = Function(forcing.I.function_space(), name="ice_step")
                forcing.set_ice(ice, index)
            forcing.I.assign(ice)

            tic = time.time()
            solver.solve()
            wall = time.time() - tic
            with stop_annotating():
                newton, outer = common.iteration_counts(solver)
                shift_m = float(solver.solution.subfunctions[
                    self.layout.sea_level]) * D_SCALE
            self.forward_rows.append({"step": index, "year": self.years[index],
                                      "newton": newton, "outer": outer,
                                      "shift_m": shift_m, "solve_s": wall})
            say(f"  tape step {index:3d}  year {self.years[index]:9.0f}  dt "
                f"{dt_yr:5.0f} yr  newton {newton}  outer {outer:3d}  shift "
                f"{shift_m:9.4f} m  solve {wall:7.1f} s  peak memory "
                f"{driver.peak_memory_gb():.2f} GB")
            if args.log_memory:
                driver.memory_now(f"tape step {index}")
            # The block of this solve knows its step, for the block log.
            block = getattr(solver, "adjoint_block", None)
            if block is not None:
                block.exp01_step = index

        return self.misfit()


# --------------------------------------------------------------------------
# The block log (debug)
# --------------------------------------------------------------------------


def install_block_log(say):
    """Log time and memory around every replay and adjoint of a coupled solve.

    Wraps the methods of the coupled low-rank solve block (stock or cached
    Firedrake) at class level. Nothing goes on the tape. For debugging only
    (`--log_blocks`).
    """
    from gadopt import dtn_coupled_adjoint as dca

    if dca.CACHED_SOLVER_BLOCKS:
        cls = dca.CachedCoupledLowRankDtNSolveBlock
        names = ("recompute_component", "prepare_evaluate_adj")
    else:
        cls = dca.CoupledLowRankDtNSolveBlock
        names = ("_forward_solve", "_adjoint_solve")

    def wrap(name):
        method = getattr(cls, name)

        def logged(self, *a, **k):
            tic = time.time()
            out = method(self, *a, **k)
            label = "replay" if name in ("recompute_component",
                                         "_forward_solve") else "adjoint"
            say(f"  {label} step {getattr(self, 'exp01_step', '?')}: "
                f"{time.time() - tic:.1f} s")
            driver.memory_now(f"{label} step {getattr(self, 'exp01_step', '?')}")
            return out
        setattr(cls, name, logged)

    for name in names:
        wrap(name)


class LoggedRieszMap:
    """The Riesz map of the control, with a log line around every solve.

    For debugging only (`--log_blocks`). Before the solve it logs the l2
    norm of the gradient cofunction (a NaN or Inf there means the adjoint
    produced it), and after it the iteration count and the converged reason
    of the Riesz solver.

    Args:
      riesz_map: the `RieszMap` of the control.
      say: the log function.
    """

    def __init__(self, riesz_map, say):
        self._map, self._say = riesz_map, say

    def __call__(self, value):
        """Return the Riesz representer of `value`, as the wrapped map does."""
        with value.dat.vec_ro as v:
            gradient_norm = v.norm()
        tic = time.time()
        out = self._map(value)
        # `RieszMap._solver` holds `(solver.solve, rhs, solution)`, built at
        # the first call; the bound method gives the solver and its KSP.
        ksp = self._map._solver[0].__self__.snes.getKSP()
        self._say(f"  riesz: |gradient| {gradient_norm:.6e}, "
                  f"{ksp.getIterationNumber()} iterations, reason "
                  f"{ksp.getConvergedReason()}, {time.time() - tic:.2f} s")
        return out


# --------------------------------------------------------------------------
# Reduced functional with the taped first value
# --------------------------------------------------------------------------


class FirstValueReducedFunctional(ReducedFunctional):
    """A reduced functional that returns the taped J at the taped control.

    ROL evaluates J at the initial control first. That value is already
    known from the taped forward run, so a replay there costs one forward
    run for nothing. The first call returns the taped value only if the
    control equals the taped control; every other call replays.
    """

    def __init__(self, functional, control, **kwargs):
        super().__init__(functional, control, **kwargs)
        self._first = (float(functional),
                       control.control.copy(deepcopy=True))

    def __call__(self, values):
        if self._first is not None:
            value, taped = self._first
            self._first = None
            given = values[0] if isinstance(values, (list, tuple)) else values
            with taped.dat.vec_ro as a, given.dat.vec_ro as b:
                same = (a - b).norm() == 0.0
            if same:
                return value
        return super().__call__(values)


# --------------------------------------------------------------------------
# Taylor test
# --------------------------------------------------------------------------


def taylor_direction(gradient, problem, args):
    """The direction of the Taylor test, with a largest value of `--taylor_h_m`.

    `random`: uniform random values on the surface nodes. J is quadratic in
    the bed, and a random direction is not correlated with the misfit, so
    the second-order term of J dominates the first-order term at every
    `eps` of the test. R1 then converges at rate 2 whether the gradient is
    right or not, and the test cannot tell them apart (job 50899530).

    `gradient`: the L2 Riesz representer of the gradient. The first-order
    term `eps dJ/dm . h` is then as large as it can be for a direction of
    this size, and a wrong gradient leaves an R1 that converges at rate 1.

    Args:
      gradient: dJ/dm, a cofunction on the space of the control.
      problem: the `Exp01Topography`.
      args: the command line.

    Returns:
      The direction, a Function on the space of the control, m.
    """
    h = Function(problem.V_bed, name="taylor_direction_m")
    if args.taylor_direction == "random":
        rng = np.random.default_rng(args.seed + COMM_WORLD.rank)
        h.dat.data_wo[:] = rng.uniform(-1.0, 1.0, h.dat.data_ro.shape)
    else:
        # The Riesz map of the control, the one ROL uses (built once).
        h.assign(problem.riesz_map(gradient))
    with h.dat.vec_ro as v:
        largest = v.norm(PETSc.NormType.INFINITY)
    h.assign(h * (args.taylor_h_m / largest))
    return h


def first_order_taylor_test(rf, m, h, J0, gradient, say,
                            epsilons=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)):
    """The first-order Taylor test of `rf` at `m` in the direction `h`.

    For each `eps`:

        R0(eps) = |J(m + eps h) - J(m)|                    rate 1
        R1(eps) = |J(m + eps h) - J(m) - eps dJ/dm . h|    rate 2

    A rate of 2 for R1 means the gradient is right. The second-order term
    is not tested: it needs the Hessian, which the coupled low-rank block
    refuses. Each `eps` costs one replay.

    Args:
      rf: the reduced functional.
      m: the control at which to test (the taped control).
      h: the direction.
      J0: J at `m`.
      gradient: dJ/dm at `m`, a cofunction.
      say: the log function.
      epsilons: the step sizes, largest first.

    Returns:
      A dict with the step sizes, J values, residuals and rates.
    """
    # The cofunction applied to the direction: the dot product of the two
    # PETSc vectors, a global sum over the owned degrees of freedom.
    with gradient.dat.vec_ro as g, h.dat.vec_ro as v:
        dJdh = float(g.dot(v))
    say(f"  gradient: dJ/dm . h = {dJdh:.10e}")
    J_values, R0, R1 = [], [], []
    for eps in epsilons:
        tic = time.time()
        with stop_annotating():
            perturbed = Function(m.function_space()).assign(m + eps * h)
        Jeps = float(rf(perturbed))
        J_values.append(Jeps)
        R0.append(abs(Jeps - J0))
        R1.append(abs(Jeps - J0 - eps * dJdh))
        say(f"    eps {eps:7.4f}: J {Jeps:.10e}  R0 {R0[-1]:.3e}  R1 "
            f"{R1[-1]:.3e} ({time.time() - tic:.1f} s)")

    def rates(residuals):
        return [float(np.log(residuals[i - 1] / residuals[i])
                      / np.log(epsilons[i - 1] / epsilons[i]))
                for i in range(1, len(residuals))]

    say(f"    R0 rates {rates(R0)} (expect 1)")
    say(f"    R1 rates {rates(R1)} (expect 2)")
    return {"epsilons": list(epsilons), "dJdh": dJdh, "J": J_values,
            "R0": R0, "R1": R1, "R0_rates": rates(R0), "R1_rates": rates(R1)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line. The solver options are those of `exp01.py`."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True)
    p.add_argument("--refine", type=int, default=0)
    p.add_argument("--output_path", default="./")
    p.add_argument("--mode", choices=["taylor", "invert"], required=True)
    p.add_argument("--step_years", type=int, default=2000)
    p.add_argument("--steps", type=int, default=None,
                   help="stop the march after this many steps (debugging)")
    # Taylor test.
    p.add_argument("--taylor_h_m", type=float, default=100.0,
                   help="amplitude of the random Taylor direction, m")
    p.add_argument("--taylor_direction", choices=["gradient", "random"],
                   default="gradient",
                   help="the direction of the Taylor test (see "
                        "taylor_direction)")
    p.add_argument("--seed", type=int, default=12345)
    # Optimisation.
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--tr_initial_radius", type=float, default=1000.0,
                   help="initial trust-region radius, in the L2 norm of the "
                        "bed on the surface (a uniform 1 m change has norm "
                        "sqrt(A) = 7.8, so 1000 allows an RMS change of "
                        "about 130 m)")
    p.add_argument("--bed_bound_m", type=float, default=15000.0,
                   help="the bed stays in [-bound, bound] m; ROL Lin-More "
                        "needs bounds, and these are far outside any bed")
    # Checkpointing.
    p.add_argument("--checkpoint_dir", default=None,
                   help="node-local directory of the tape checkpoints "
                        "(default $TMPDIR)")
    # Debugging.
    p.add_argument("--log_memory", action="store_true",
                   help="log the memory after every taped step")
    p.add_argument("--log_blocks", action="store_true",
                   help="log time and memory around every replay and "
                        "adjoint solve")
    # The solver options of the forward run (`exp01.py`).
    p.add_argument("--k_s", type=float, default=0.9422)
    p.add_argument("--core_density", choices=["mean", "cmb"], default="mean")
    p.add_argument("--gia_rebuild", choices=["per_solve", "lagged", "always"],
                   default=None)
    p.add_argument("--gia_rebuild_every", type=int, default=None)
    p.add_argument("--gia_rebuild_max_outer", type=float, default=None)
    p.add_argument("--block0_max_it", type=int, default=50)
    p.add_argument("--outer_atol", type=float, default=1.0e-11)
    p.add_argument("--outer_max_it", type=int, default=30)
    return p.parse_args()


def enable_node_checkpointing(args, mesh, say):
    """Disk checkpointing of the tape on node-local storage.

    One checkpoint communicator per node: the ranks of a node share one
    directory in `$TMPDIR` (or `--checkpoint_dir`), and the data is written
    with PETSc Vec I/O, which needs no mesh file. Must run before the first
    block is on the tape.
    """
    base = args.checkpoint_dir or os.environ.get("TMPDIR", "/tmp")
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    directory = os.path.join(base, f"exp01-adjoint-{job}")
    node_comm = mesh.comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node_comm.rank == 0:
        os.makedirs(directory, exist_ok=True)
    node_comm.Barrier()
    enable_disk_checkpointing(checkpoint_comm=node_comm,
                              checkpoint_dir=directory)
    tape = get_working_tape()
    tape.enable_checkpointing(SingleDiskStorageSchedule(move_data=False),
                              gc_timestep_frequency=1)
    say(f"  tape checkpoints: {directory} (one communicator per node, "
        f"{node_comm.size} ranks)")


def main():
    """Run the Taylor test or the inversion."""
    args = parse_args()
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()
    say = RunLog(os.path.join(args.output_path, "topography_exp01.log"))
    tic_run = time.time()
    from gadopt.dtn_coupled_adjoint import CACHED_SOLVER_BLOCKS
    say("=" * 78)
    say(f"GIAMIP Exp01, bed at 122 ka by adjoint, mode {args.mode}")
    say(f"  cached solver blocks (Firedrake PR 4638): {CACHED_SOLVER_BLOCKS}")
    say("=" * 78)

    # Everything that is not on the tape, with annotation paused.
    problem = Exp01Topography(args, say)
    driver.memory_now("after the setup")
    say(f"  march: steps 1..{problem.last}, years {problem.years[0]:.0f} .. "
        f"{problem.years[problem.last]:.0f}")

    # The control, cold start: the observed bed.
    bed = Function(problem.V_bed, name="bed_122ka_m").assign(problem.bed_obs)

    tape = get_working_tape()
    tape.clear_tape()
    enable_node_checkpointing(args, problem.mantle, say)
    if args.log_blocks:
        install_block_log(say)
        # The control below takes `problem.riesz_map`, so the wrap goes first.
        problem.riesz_map = LoggedRieszMap(problem.riesz_map, say)

    # The one taped forward run.
    continue_annotation()
    tic = time.time()
    J = problem.tape_forward(bed)
    pause_annotation()
    taped_s = time.time() - tic
    say(f"  taped forward: J {float(J):.10e} (RMS bed misfit "
        f"{problem.rms_misfit_m(J):.3f} m), {taped_s:.1f} s, "
        f"{len(tape.get_blocks())} blocks")
    driver.memory_now("after the taped forward run")

    # The control carries its L2 Riesz map, so the gradient representer of
    # every ROL iteration uses the same solver.
    control = Control(bed, riesz_map=problem.riesz_map)
    record = {"args": vars(args), "mesh": problem.mesh_info,
              "solver": problem.solver_record, "J_taped": float(J),
              "taped_forward_s": taped_s, "forward_rows": problem.forward_rows,
              "cached_solver_blocks": CACHED_SOLVER_BLOCKS}
    summary_path = os.path.join(args.output_path, "summary_topography.json")

    if args.mode == "taylor":
        rf = ReducedFunctional(J, control)
        # The replay at the taped control must give the taped J.
        tic = time.time()
        replayed = float(rf(bed))
        say(f"  replay at the taped control: J {replayed:.10e}, relative "
            f"difference {abs(replayed - float(J)) / abs(float(J)):.3e} "
            f"({time.time() - tic:.1f} s)")
        record["J_replay"] = replayed
        # One adjoint gives the gradient for the direction and for the test.
        tic = time.time()
        gradient = rf.derivative()
        say(f"  gradient ({time.time() - tic:.1f} s)")
        h = taylor_direction(gradient, problem, args)
        record["taylor"] = first_order_taylor_test(rf, bed, h, float(J),
                                                   gradient, say)
    else:
        rf = FirstValueReducedFunctional(J, control)
        lower = Function(problem.V_bed, name="bed_lower_m").assign(-args.bed_bound_m)
        upper = Function(problem.V_bed, name="bed_upper_m").assign(args.bed_bound_m)
        minimisation = MinimizationProblem(rf, bounds=(lower, upper))
        parameters = minimisation_parameters
        parameters["Status Test"]["Iteration Limit"] = args.iterations
        parameters["Step"]["Trust Region"]["Initial Radius"] = args.tr_initial_radius
        optimiser = LinMoreOptimiser(
            minimisation, parameters,
            checkpoint_dir=os.path.join(args.output_path,
                                        "optimisation_checkpoint"))

        # Every iteration: the control to `controls.h5` and one log line.
        controls_path = os.path.join(args.output_path, "controls.h5")
        with CheckpointFile(controls_path, "w") as chk:
            chk.save_mesh(problem.surface)
            chk.save_function(problem.bed_obs)
        iterations = []
        # The surface measure and area of the control, for the RMS change.
        problem.surface_dx = dx(domain=problem.surface)
        problem.surface_area = float(assemble(Constant(1.0) * problem.surface_dx))

        def after_iteration():
            with stop_annotating():
                current = optimiser.rol_solver.rolvector.dat[0]
                change = Function(problem.V_bed).assign(current - problem.bed_obs)
                rms_change = np.sqrt(float(assemble(change ** 2 * problem.surface_dx))
                                     / problem.surface_area)
                with CheckpointFile(controls_path, "a") as chk:
                    chk.save_function(Function(problem.V_bed, name="bed_122ka_m")
                                      .assign(current), idx=len(iterations))
            iterations.append({"iteration": len(iterations),
                               "rms_change_from_observed_m": rms_change,
                               "wall_s": time.time() - tic_run})
            say(f"  ROL iteration {len(iterations) - 1}: RMS change of the bed "
                f"from the observed bed {rms_change:.3f} m, peak memory "
                f"{driver.peak_memory_gb():.2f} GB")
            common.write_json(summary_path, {**record, "iterations": iterations,
                                             "state": "running"})

        optimiser.add_callback(after_iteration)
        tic = time.time()
        optimiser.run()
        say(f"  optimisation: {len(iterations)} status checks, "
            f"{time.time() - tic:.1f} s")
        record["iterations"] = iterations

    record["total_s"] = time.time() - tic_run
    record["peak_memory_gb"] = driver.peak_memory_gb()
    common.write_json(summary_path, {**record, "state": "finished"})
    say(f"  total {record['total_s']:.1f} s, peak memory "
        f"{record['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
