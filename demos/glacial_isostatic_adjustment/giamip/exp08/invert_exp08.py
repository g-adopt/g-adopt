r"""GIAMIP Exp08: the bed at the start of the run from the present-day bed, by adjoint.

THE PROBLEM
    The GIAMIP topography file is the observed bed of the present day (model
    year 0). The run must start from a bed that is not known: the bed that
    the run carries to the observed bed at year 0. This script finds that
    bed by minimising, over the bed `b` at the start of the run,

        J(b) = 1/2 int_Re ((SL(year 0; b) - SL_obs) D)^2 dS

    where `SL = SL_init + (N - N_init) - (u_r - ur_init) + Shift` is the sea
    level of the solver (`SelfGravitatingGIASolver.sea_level`), `SL_init =
    -b / D` is the initial sea level made from the bed `b`, `SL_obs = -b_obs /
    D` is the sea level of the observed bed, and `dS` is the non-dimensional
    area of the surface Re (area A = 4 pi Re^2 = 61.0). Since `bed = -SL D`,
    the integrand is the square of the misfit of the predicted present-day
    bed in metres, and `sqrt(2 J / A)` is its RMS value in metres. The misfit
    has the same weight everywhere. There is no smoothing and no damping term.

    The scale makes the curvature of J about 1 in the L2 norm of the control.
    The predicted bed is `b` plus the sea-level change of the run, which
    depends only weakly on `b` (through the ocean load and the coastline), so
    the Hessian of J is close to the mass matrix of the surface for any
    length of the march. ROL Lin-More starts from an identity model of the
    Hessian, and a curvature far from 1 makes its first steps useless.

THE INPUTS
    Everything that comes from data is in the input file of
    `prepare_exp08.py`: the meshes, the earth fields, the observed sea level,
    the frozen slope of the ocean mask, the ice of every step and the record
    of the experiment. This script imports no data reader (`giamip`,
    `gdrift`, `h5netcdf`, `netCDF4`). It must run on the number of ranks of
    the prepare job; it stops with an error otherwise. After the load it
    computes the min, max and integral of every field again and stops if
    they differ from the numbers of the prepare run.

THE CONTROL
    `b`, the bed at the start of the run in metres, a CG2 field on the
    surface Re: a codim-1 `Submesh` of the mantle (`giamip_surface`). The
    surface mesh has P1 geometry (flat triangles, see
    `exp08_common.surface_mesh`). The solver reads `SL_init`, a CG2 field
    on the curved mantle, only on the facets of Re.

    The CG2 nodes of the surface are the CG2 nodes of the mantle on Re.
    `exp08_common.SurfaceNodeMap` pairs them once and moves values between
    the two spaces as exact copies: `extend` writes `SL_init = -b / D` onto
    the mantle nodes on Re (zero at the other nodes, which the solver does
    not read), and `restrict` reads the mantle nodes on Re back onto the
    surface. `restrict(extend(b)) == b` bit for bit. Firedrake's
    `interpolate` between the two meshes is not used: for a codim-1 submesh
    it locates the target nodes by a point search, and with the P1 surface
    and the P2 mantle it does not return the nodal values. A new run starts
    from the observed bed, the nodal values of `-SL_obs D` on Re (cold
    start).

    The control carries an L2 Riesz map (`RieszMap` on the surface space,
    built once, with the solver of Firedrake's `project`,
    `RIESZ_SOLVER_PARAMETERS`). ROL gets the gradient through it, and the
    Taylor direction uses it too.

WHAT IS ON THE TAPE
    Only the operations from the control to J:

      1. the copy of `-b / D` onto the mantle nodes of `SL_init` on Re
         (`SurfaceExtendBlock`, a linear block whose adjoint is the
         transpose of the node copy);
      2. per time step: the assignment of `dt` when it changes, the
         assignment of the ice of the step, and the coupled solve with what
         the solver does around it (the null-space projection and the
         recovery of the internal variables);
      3. the misfit assemble.

    Everything else runs with annotation paused: the load of the input file,
    the solver construction, the output files and every logged number. The
    ice of each step is its own Function, loaded from the file with
    annotation paused, and is assigned to the solver's ice field on the
    tape. A single source Function that is refilled at every step does not
    work: pyadjoint saves its value once, and the replay would use the first
    ice field at every step.

CHECKPOINTING
    The time loop runs in `tape.timestepper` with
    `SingleDiskStorageSchedule`. The checkpoint data goes to one directory
    per node, one checkpoint communicator per node (`MPI.COMM_TYPE_SHARED`),
    in `$TMPDIR` (or `--checkpoint_dir`). On a Setonix compute node `/tmp`
    is a RAM file system, so the checkpoints there use node memory; put them
    on `/scratch` when the node memory is short.

STOPPING
    ROL stops after `--iterations` iterations, or earlier when the RMS bed
    misfit `sqrt(2 J / A)` falls below `--stop_rms_m` (1 mm).

RESTART
    `gadopt.inverse.LinMoreOptimiser.restore` cannot restore this control:
    Firedrake loads a function only onto a mesh that was itself loaded from
    the same file, and the control lives on a `Submesh`. An interpolation
    between the surface mesh of an earlier run and the surface mesh of a new
    run gave wrong values without an error (job 51124687). The restart
    therefore never moves a field between two meshes:

    - An inversion copies the input file into its run directory as
      `state.h5` and loads the meshes and fields from that copy.
    - After each ROL iteration it appends the bed to `state.h5`: the node
      map copies the bed onto the mantle nodes on Re, and
      `exp08_common.append_control` copies that field onto a CG2 field of
      the parent mesh with `Function.assign` (`bed_start_parent_cg2_m`).
      Both steps copy nodal values, so the stored bed is the bed of the
      iteration bit for bit.
    - A run that stops at the time limit continues with
      `--restart_state <dir>/state.h5`. The new run copies that file, loads
      the meshes and the last bed from it (`assign` from the parent to the
      mantle, then `restrict` to the surface), and starts ROL from that
      bed. The first J of the new run is the J of the last iteration of the
      earlier run, up to the reproducibility of the forward solve. The
      L-BFGS memory of the earlier run is lost, so the first iterations of
      the new run take smaller steps than an uninterrupted run would.
    - A state file of an earlier version of this script holds the bed as a
      broken copy named `bed_start_m`, written from the cross-mesh
      interpolation of the bed onto the mantle. Those mantle values are
      what that version wrote into `SL_init`, so a restart from such a file
      continues from the bed that its solver used (`read_inputs`).

MODES
    --mode taylor   tape one forward run, check that a replay at the taped
                    control gives the taped J, and run a first-order Taylor
                    test in the direction of the gradient (or a random
                    direction), of largest value `--taylor_h_m`.
    --mode invert   minimise J with ROL Lin-More (L2 Riesz map).

    `--steps n` stops the march after n steps (debugging); the misfit is
    then taken at the last step against the observed bed, which tests the
    machinery and has no physical meaning. Without `--steps` the march ends
    at year 0.

RUN
    srun python3 invert_exp08.py --inputs <dir>/inputs_exp08.h5
        --output_path <dir> --mode taylor --steps 3
    srun python3 invert_exp08.py --inputs <dir>/inputs_exp08.h5
        --output_path <dir> --mode invert

OUTPUT (in --output_path)
    invert_exp08.log        the log lines of rank 0 also go to stdout
    summary_invert.json     the settings, the forward record, the Taylor
                            residuals or the ROL iterations
    controls.h5             the control (bed at the start, m) of every ROL
                            iteration, with the observed bed (CheckpointFile
                            on the surface submesh); `idx` = iteration, with
                            the J and RMS misfit of each iterate
    state.h5                (invert mode) the copy of the input file with
                            the bed of every iteration appended; the file of
                            a restart
"""

import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# `gadopt` imports `firedrake.pyplot`, which imports matplotlib's font
# manager. Without a font cache every rank scans the fonts; the job
# environment names a cache built once on a login node.
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# The shared module imports gadopt before firedrake.
import exp08_common as common  # noqa: E402
import ROL  # noqa: E402
from checkpoint_schedules import SingleDiskStorageSchedule  # noqa: E402
from firedrake import CheckpointFile, RieszMap, dx  # noqa: E402
from firedrake.adjoint import (Control, ReducedFunctional,  # noqa: E402
                               continue_annotation, enable_disk_checkpointing,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from gadopt import (COMM_WORLD, Constant, Function, FunctionSpace,  # noqa: E402
                    assemble)
# `gadopt.inverse` starts the tape when it is imported; the annotation is
# paused again at once, before anything is built.
from gadopt.inverse import LinMoreOptimiser, minimisation_parameters  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from pyadjoint import MinimizationProblem  # noqa: E402

pause_annotation()

D_SCALE = common.D_SCALE

#: The solver of the L2 Riesz map of the control: the mass matrix of the CG2
#: surface space. The L2 Riesz map is the same solve as an L2 projection, so
#: it uses the default solver of Firedrake's `project` for an assembled
#: matrix (`firedrake/projection.py`): CG, block Jacobi with ICC on each
#: block, relative tolerance 1e-8. No factorisation: the default of a
#: `RieszMap` is a MUMPS LU built again at every gradient, and that solve
#: failed on Setonix (DIVERGED_LINEAR_SOLVE in the Riesz solve of a ROL
#: gradient).
RIESZ_SOLVER_PARAMETERS = {"ksp_type": "cg", "ksp_rtol": 1e-8,
                           "pc_type": "bjacobi", "sub_pc_type": "icc"}

#: The name of the control in `controls.h5`.
CONTROL_NAME = common.CONTROL_NAME


# --------------------------------------------------------------------------
# The forward problem on the tape
# --------------------------------------------------------------------------


class BedInversion:
    """The taped map from the bed at the start of the run to the misfit.

    Built in two parts. `__init__` builds everything that is not on the tape
    (annotation paused). `tape_forward` records the forward run once and
    returns J; pyadjoint replays it for every later evaluation.

    Args:
      args: the command line.
      say: the log function.
    """

    def __init__(self, args, say, path, restart=False):
        self.args, self.say = args, say
        tic = time.time()
        # The meshes, the fields and the record of the prepare run, from the
        # input file or from the state file of an inversion; on a restart,
        # also the last bed of the earlier run.
        loaded = common.read_inputs(path, last_control=restart)
        (self.parent, self.mantle, self.fields, self.ice,
         self.record) = loaded[:5]
        self.restart_bed = loaded[5] if restart else None
        say(f"  inputs: {path}, experiment {self.record['experiment']}, "
            f"mesh {self.record['mesh']['md5']} refine "
            f"{self.record['mesh']['refine_levels']}, "
            f"{self.record['mesh']['cells_mantle']} mantle cells "
            f"({time.time() - tic:.1f} s)")
        # The fields must come back as the prepare run wrote them.
        found = common.input_statistics(self.mantle, self.fields, self.ice)
        bad = common.compare_statistics(self.record["statistics"], found)
        if bad:
            raise RuntimeError(f"the input file did not load as written: "
                               f"{bad[:5]}")
        say("  inputs: min, max and integral of every field agree with the "
            "prepare record")
        self.scales = common.Scales.from_dict(self.record["scales"])

        # The forcing Functions that the sea-level equation reads. `SL_init`
        # is written from the control on the tape; `I` takes the ice of each
        # step on the tape. The run starts from the undeformed state with the
        # ice of step 0 on it.
        self.SL_obs = self.fields["SL_obs"]
        self.SL_init = Function(self.SL_obs.function_space(),
                                name="SL_init").assign(self.SL_obs)
        self.I_init = Function(self.ice[0].function_space(),
                               name="ice_init").assign(self.ice[0])
        self.I = Function(self.ice[0].function_space(),
                          name="ice").assign(self.ice[0])

        # The solver.
        # The time step, a Real Function assigned on the tape (see
        # `common.time_step`).
        self.dt = common.time_step(self.mantle)
        tic = time.time()
        self.solver, self.layout, self.solver_record = common.build_solver(
            self.parent, self.mantle, self.fields, self.scales,
            self.record["solver_constants"],
            {"SL_init": self.SL_init, "I": self.I, "I_init": self.I_init},
            self.dt, args, record=self.record)
        say(f"  solver: {self.solver_record['unknowns']} unknowns, "
            f"{self.solver_record['real_rows']} Real rows, k_s {args.k_s}, "
            f"C - A {self.solver_record['C_minus_A_nondim']:.6g} "
            f"({time.time() - tic:.1f} s)")

        # The surface mesh of the control (P1 geometry, see
        # `common.surface_mesh`) and the CG2 space of the control.
        self.surface = common.surface_mesh(self.mantle)
        self.V_bed = FunctionSpace(self.surface, "CG", 2)
        # The exact node map between the control space and the CG2 space of
        # `SL_init` on the mantle. Every move of a field between the surface
        # and the mantle goes through it; `interpolate` between the two
        # meshes evaluates fields at points and is not exact here (see
        # `common.SurfaceNodeMap`).
        self.bed_map = common.SurfaceNodeMap(
            self.V_bed, self.SL_init.function_space(), common.SURF_RE)
        # The L2 Riesz map of the control, built once and given to the
        # Control. Without it, every gradient that ROL asks for builds a new
        # RieszMap, with a new solver and a new mass-matrix assembly.
        # `constant_jacobian=True`: the mass matrix does not depend on the
        # control, so it is assembled once.
        self.riesz_map = RieszMap(self.V_bed, "L2", constant_jacobian=True,
                                  solver_parameters=RIESZ_SOLVER_PARAMETERS)
        # The observed bed on the surface, m: the nodal values of `SL_obs` on
        # Re, copied by the node map and scaled. `extend` of this bed gives
        # back `SL_obs` on Re to round-off (one multiplication and one
        # division by D), so the cold start begins at the observed bed.
        self.bed_obs = Function(self.V_bed, name="bed_observed_m")
        self.bed_map.restrict(self.SL_obs, out=self.bed_obs)
        self.bed_obs.assign(-D_SCALE * self.bed_obs)

        # The steps of the march: 1 .. last; the last is year 0 unless
        # `--steps` stops earlier.
        self.years = np.asarray(self.record["march"]["years"], dtype=float)
        year0 = int(self.record["march"]["year0_step"])
        self.last = year0 if args.steps is None else min(year0, args.steps)

        # The surface measure and area of the misfit (the solver's own sea-
        # level measure, so the misfit integrates SL as the solver does).
        self.dss = self.solver.sea_level_measure()(common.SURF_RE)
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
          bed: the control, the bed at the start of the run on the surface, m.

        Returns:
          J, an `AdjFloat`.
        """
        args, say, solver = self.args, self.say, self.solver
        tape = get_working_tape()

        # 1. The control to the initial sea level: `SL_init = -bed / D` at
        # the mantle nodes on Re, an exact copy of the nodal values, and
        # zero at the other nodes, which the solver does not read (it reads
        # `SL_init` only on the facets of Re). One `SurfaceExtendBlock` on
        # the tape, with the transpose of the node map as its adjoint.
        self.bed_map.extend(bed, self.SL_init, scale=-1.0 / D_SCALE)

        previous_dt = None
        for index in tape.timestepper(iter(range(1, self.last + 1))):
            dt_yr = float(self.years[index] - self.years[index - 1])
            # `dt` on the tape when it changes: a change made with annotation
            # paused would leave the replay with the first value.
            if dt_yr != previous_dt:
                self.dt.assign(dt_yr / common.T_BAR_YR)
                previous_dt = dt_yr
            # The ice of the step, its own Function from the input file, in
            # one assignment on the tape (backward Euler: the load at the end
            # of the step).
            self.I.assign(self.ice[index])

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
                f"{common.peak_memory_gb():.2f} GB")
            if args.log_memory:
                common.memory_now(f"tape step {index}", say)
            # The block of this solve knows its step, for the block log.
            block = getattr(solver, "adjoint_block", None)
            if block is not None:
                block.giamip_step = index

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
            step = getattr(self, "giamip_step", "?")
            say(f"  {label} step {step}: {time.time() - tic:.1f} s")
            common.memory_now(f"{label} step {step}", say)
            return out
        setattr(cls, name, logged)

    for name in names:
        wrap(name)


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
# The optimiser with the stop rule
# --------------------------------------------------------------------------


class StoppingLinMoreOptimiser(LinMoreOptimiser):
    """`LinMoreOptimiser` that also stops when J falls below a value.

    The status test of `LinMoreOptimiser` runs the callbacks and then the
    ROL status test (iteration limit, gradient and step tolerances). This
    one first stores the objective value of the accepted iterate
    (`status.value`) in `last_value`, runs the callbacks, and ends the
    optimisation when the value is at or below `stop_value`.

    Args:
      problem: the `MinimizationProblem`.
      parameters: the ROL parameters.
      stop_value: the value of J at which ROL stops.
    """

    def __init__(self, problem, parameters, stop_value):
        self.stop_value = float(stop_value)
        self.last_value = None
        self.stopped_by_value = False
        super().__init__(problem, parameters, checkpoint_dir=None)

    def _add_statustest(self):
        optimiser = self

        class StatusTest(ROL.StatusTest):
            def check(inner_self, status):
                optimiser.iteration += 1
                optimiser.last_value = float(status.value)
                optimiser.last_gnorm = float(status.gnorm)
                for callback in optimiser.callbacks:
                    callback()
                if optimiser.last_value <= optimiser.stop_value:
                    optimiser.stopped_by_value = True
                    return False
                return super().check(status)

        # Do not chain with the default status test.
        self.rol_algorithm.setStatusTest(StatusTest(self.rol_parameters), False)


# --------------------------------------------------------------------------
# Taylor test
# --------------------------------------------------------------------------


def taylor_direction(gradient, problem, args):
    """The direction of the Taylor test, with a largest value of `--taylor_h_m`.

    `random`: uniform random values on the surface nodes. J is quadratic in
    the bed, and a random direction is not correlated with the misfit, so
    the second-order term of J dominates the first-order term at every
    `eps` of the test. R1 then converges at rate 2 whether the gradient is
    right or not, and the test cannot tell them apart.

    `gradient`: the L2 Riesz representer of the gradient. The first-order
    term `eps dJ/dm . h` is then as large as it can be for a direction of
    this size, and a wrong gradient leaves an R1 that converges at rate 1.

    Args:
      gradient: dJ/dm, a cofunction on the space of the control.
      problem: the `BedInversion`.
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
# Restart
# --------------------------------------------------------------------------


def keep_previous(path):
    """Rename an existing file to `<stem>_previous<n>.h5` (rank 0).

    Returns the new name, or None if there was no file.
    """
    new = None
    if COMM_WORLD.rank == 0 and os.path.exists(path):
        stem, ext = os.path.splitext(path)
        n = 1
        while os.path.exists(f"{stem}_previous{n}{ext}"):
            n += 1
        new = f"{stem}_previous{n}{ext}"
        os.replace(path, new)
    return COMM_WORLD.bcast(new, root=0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True,
                   help="the input file of prepare_exp08.py")
    p.add_argument("--output_path", default="./")
    p.add_argument("--mode", choices=["taylor", "invert"], required=True)
    p.add_argument("--steps", type=int, default=None,
                   help="stop the march after this many steps (debugging)")
    # Taylor test.
    p.add_argument("--taylor_h_m", type=float, default=100.0,
                   help="largest value of the Taylor direction, m")
    p.add_argument("--taylor_direction", choices=["gradient", "random"],
                   default="gradient",
                   help="the direction of the Taylor test (see "
                        "taylor_direction)")
    p.add_argument("--seed", type=int, default=12345)
    # Optimisation.
    p.add_argument("--iterations", type=int, default=10,
                   help="ROL iterations of this run")
    p.add_argument("--stop_rms_m", type=float, default=1.0e-3,
                   help="stop when the RMS bed misfit is below this, m")
    p.add_argument("--tr_initial_radius", type=float, default=1000.0,
                   help="initial trust-region radius, in the L2 norm of the "
                        "bed on the surface (a uniform 1 m change has norm "
                        "sqrt(A) = 7.8, so 1000 allows an RMS change of "
                        "about 130 m)")
    p.add_argument("--bed_bound_m", type=float, default=15000.0,
                   help="the bed stays in [-bound, bound] m; ROL Lin-More "
                        "needs bounds, and these are far outside any bed")
    p.add_argument("--restart_state", default=None, metavar="FILE",
                   help="start from the bed of the last iteration in this "
                        "state.h5 of an earlier run")
    # Checkpointing.
    p.add_argument("--checkpoint_dir", default=None,
                   help="directory of the tape checkpoints, one per node "
                        "(default $TMPDIR)")
    # Debugging.
    p.add_argument("--log_memory", action="store_true",
                   help="log the memory after every taped step")
    p.add_argument("--log_blocks", action="store_true",
                   help="log time and memory around every replay and "
                        "adjoint solve")
    # The solver options.
    p.add_argument("--k_s", type=float, default=0.9422,
                   help="the fluid Love number of the bulge")
    p.add_argument("--gia_rebuild", choices=["per_solve", "lagged", "always"],
                   default=None)
    p.add_argument("--gia_rebuild_every", type=int, default=None)
    p.add_argument("--gia_rebuild_max_outer", type=float, default=None)
    # The iteration cap of the block-0 FGMRES. On the L0 mesh the capped
    # block-0 solves stop decreasing after about 30 iterations, so a cap
    # near 50 removes iterations that do not reduce the residual.
    p.add_argument("--block0_max_it", type=int, default=50)
    # 0 (off): the low-rank operator removes the declared rigid rotation
    # from its action, so every outer solve converges on the relative
    # tolerance.
    p.add_argument("--outer_atol", type=float, default=0.0)
    p.add_argument("--outer_max_it", type=int, default=30)
    return p.parse_args()


def enable_node_checkpointing(args, mesh, say):
    """Disk checkpointing of the tape, one directory per node.

    One checkpoint communicator per node: the ranks of a node share one
    directory in `$TMPDIR` (or `--checkpoint_dir`), and the data is written
    with PETSc Vec I/O, which needs no mesh file. Must run before the first
    block is on the tape.

    Returns:
      The directory of this node.
    """
    base = args.checkpoint_dir or os.environ.get("TMPDIR", "/tmp")
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    node_comm = mesh.comm.Split_type(MPI.COMM_TYPE_SHARED)
    # The directory is per node, named after the first global rank of the
    # node: on a shared file system (`/scratch`) the nodes must not write
    # into one directory.
    leader = node_comm.bcast(COMM_WORLD.rank, root=0)
    directory = os.path.join(base, f"giamip-adjoint-{job}", f"node{leader}")
    if node_comm.rank == 0:
        os.makedirs(directory, exist_ok=True)
    node_comm.Barrier()
    enable_disk_checkpointing(checkpoint_comm=node_comm,
                              checkpoint_dir=directory)
    tape = get_working_tape()
    tape.enable_checkpointing(SingleDiskStorageSchedule(move_data=False),
                              gc_timestep_frequency=1)
    say(f"  tape checkpoints: {directory} and the other nodes (one "
        f"communicator per node, {node_comm.size} ranks)")
    return directory


def main():
    """Run the Taylor test or the inversion."""
    args = parse_args()
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()
    say = common.RunLog(os.path.join(args.output_path, "invert_exp08.log"))
    tic_run = time.time()
    from gadopt.dtn_coupled_adjoint import CACHED_SOLVER_BLOCKS
    say("=" * 78)
    say(f"GIAMIP bed inversion, mode {args.mode}, {COMM_WORLD.size} ranks")
    say(f"  cached solver blocks (Firedrake PR 4638): {CACHED_SOLVER_BLOCKS}")
    say("=" * 78)

    # The state file of an inversion: a copy of the input file (cold start)
    # or of the state file of an earlier run (restart), into which every
    # iterate of the control is appended. A file of the same name from an
    # earlier job in this directory is kept under a new name first; a
    # restart from that file copies the kept file.
    controls_path = os.path.join(args.output_path, "controls.h5")
    state_path = os.path.join(args.output_path, "state.h5")
    restart = args.mode == "invert" and args.restart_state is not None
    if args.mode == "invert":
        source = (os.path.realpath(args.restart_state) if restart
                  else os.path.realpath(args.inputs))
        kept = keep_previous(state_path)
        if source == os.path.realpath(state_path):
            source = kept
        keep_previous(controls_path)
        tic = time.time()
        if COMM_WORLD.rank == 0:
            shutil.copyfile(source, state_path)
        COMM_WORLD.barrier()
        say(f"  state file: {state_path}, a copy of {source} "
            f"({time.time() - tic:.1f} s)")
        load_path = state_path
    else:
        source, load_path = None, args.inputs

    # Everything that is not on the tape, with annotation paused.
    problem = BedInversion(args, say, load_path, restart=restart)
    common.memory_now("after the setup", say)
    say(f"  march: steps 1..{problem.last}, years {problem.years[0]:.0f} .. "
        f"{problem.years[problem.last]:.0f}")

    # The control: the observed bed (cold start), or the last bed of the
    # earlier run. The node map copies the stored values at the mantle
    # nodes on Re onto the surface nodes, so the restart begins at the bed
    # that the earlier run stored, bit for bit.
    restart_from = None
    bed = Function(problem.V_bed, name=CONTROL_NAME)
    if restart:
        bed_mantle, iteration = problem.restart_bed
        problem.bed_map.restrict(bed_mantle, out=bed)
        restart_from = {"file": source, "iteration": iteration}
        say(f"  restart: bed of iteration {iteration} of {source}")
    else:
        bed.assign(problem.bed_obs)
    # The entries of this run in the state file continue after those of the
    # earlier run, so no index is written twice.
    first_index = 0 if restart_from is None else restart_from["iteration"] + 1

    tape = get_working_tape()
    tape.clear_tape()
    enable_node_checkpointing(args, problem.mantle, say)
    if args.log_blocks:
        install_block_log(say)

    # The one taped forward run.
    continue_annotation()
    tic = time.time()
    J = problem.tape_forward(bed)
    pause_annotation()
    taped_s = time.time() - tic
    say(f"  taped forward: J {float(J):.10e} (RMS bed misfit "
        f"{problem.rms_misfit_m(J):.4f} m), {taped_s:.1f} s, "
        f"{len(tape.get_blocks())} blocks")
    common.memory_now("after the taped forward run", say)

    # The control carries its L2 Riesz map, so the gradient representer of
    # every ROL iteration uses the same solver.
    control = Control(bed, riesz_map=problem.riesz_map)
    record = {"args": vars(args), "experiment": problem.record["experiment"],
              "inputs_record": {key: problem.record[key] for key in
                                ("mesh", "march", "scales", "solver_constants",
                                 "giamip_version", "inputs")},
              "solver": problem.solver_record, "J_taped": float(J),
              "rms_taped_m": problem.rms_misfit_m(J),
              "taped_forward_s": taped_s, "forward_rows": problem.forward_rows,
              "cached_solver_blocks": CACHED_SOLVER_BLOCKS,
              "restart_from": restart_from,
              "provenance": (common.provenance(HERE)
                             if COMM_WORLD.rank == 0 else None)}
    summary_path = os.path.join(args.output_path, "summary_invert.json")

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
        common.memory_now("after the adjoint", say)
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
        # J at which the RMS bed misfit is `--stop_rms_m`: RMS = sqrt(2 J / A).
        stop_value = 0.5 * problem.area * args.stop_rms_m ** 2
        optimiser = StoppingLinMoreOptimiser(minimisation, parameters,
                                             stop_value)

        # Every iteration: the control to `controls.h5` and one log line.
        with CheckpointFile(controls_path, "w") as chk:
            chk.save_mesh(problem.surface)
            chk.save_function(problem.bed_obs)
        iterations = []
        # The surface measure and area of the control, for the RMS change.
        surface_dx = dx(domain=problem.surface)
        surface_area = float(assemble(Constant(1.0) * surface_dx))

        # The bed on the mantle for the state file: the nodes on Re carry
        # the bed, every other node is zero. It must be in the space of the
        # node map, CG2 on the mantle (`common.CONTROL_SPACE`).
        bed_mantle = Function(problem.bed_map.V_mantle,
                              name=common.CONTROL_NAME)

        def after_iteration():
            with stop_annotating():
                index = len(iterations)
                current = optimiser.rol_solver.rolvector.dat[0]
                change = Function(problem.V_bed).assign(current - problem.bed_obs)
                rms_change = np.sqrt(float(assemble(change ** 2 * surface_dx))
                                     / surface_area)
                J_value = optimiser.last_value
                rms = problem.rms_misfit_m(J_value)
                with CheckpointFile(controls_path, "a") as chk:
                    chk.save_function(
                        Function(problem.V_bed, name=CONTROL_NAME).assign(current),
                        idx=index,
                        timestepping_info={"J": J_value, "rms_misfit_m": rms})
                # The same bed into the state file, for a restart, copied
                # exactly onto the mantle nodes on Re by the node map.
                problem.bed_map.extend(current, bed_mantle)
                common.append_control(state_path, problem.parent, bed_mantle,
                                      first_index + index,
                                      {"J": J_value, "rms_misfit_m": rms})
            node_gb = common.memory_now(f"ROL iteration {index}", say)
            iterations.append({"iteration": index,
                               "state_index": first_index + index,
                               "J": J_value,
                               "rms_misfit_m": rms,
                               "gnorm": optimiser.last_gnorm,
                               "rms_change_from_observed_m": rms_change,
                               "node_memory_gb": node_gb,
                               "wall_s": time.time() - tic_run})
            say(f"  ROL iteration {index}: J {J_value:.6e}, RMS bed misfit "
                f"{rms:.4f} m, gnorm {optimiser.last_gnorm:.3e}, RMS change "
                f"of the bed from the observed bed {rms_change:.3f} m, peak "
                f"memory {common.peak_memory_gb():.2f} GB, "
                f"{time.time() - tic_run:.0f} s")
            common.write_json(summary_path, {**record, "iterations": iterations,
                                             "state": "running"})

        optimiser.add_callback(after_iteration)
        tic = time.time()
        optimiser.run()
        reason = ("RMS misfit below the stop value" if optimiser.stopped_by_value
                  else "ROL status test (iteration limit or tolerance)")
        say(f"  optimisation: {len(iterations)} status checks, "
            f"{time.time() - tic:.1f} s, stopped by the {reason}")
        record["iterations"] = iterations
        record["stopped_by"] = reason

    record["total_s"] = time.time() - tic_run
    record["peak_memory_gb"] = common.peak_memory_gb()
    common.write_json(summary_path, {**record, "state": "finished"})
    say(f"  total {record['total_s']:.1f} s, peak memory "
        f"{record['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
