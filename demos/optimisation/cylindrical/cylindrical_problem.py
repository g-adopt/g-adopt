"""The inverse problem of the cylindrical optimiser comparison.

This module builds the twin experiment of ``tests/adjoint_2d_cylindrical/inverse.py``
in the form that the optimiser back-ends of the box benchmark expect
(``demos/optimisation/box/optimisers.py``). It does not restate the forward model or
the objective. It calls ``generate_inverse_problem`` of that test, which tapes the
production model once, and then wraps the taped objective in a new reduced functional
whose control carries the regularisation metric as its Riesz map.

The objective is the one that ``inverse.inverse()`` minimises with ROL today:

    J(m) = alpha_T int (T(t_F) - Tobs)^2 dx
         + (D_obs / D_u) (alpha_u / N) sum_n int_top |u_n - uobs_n|^2 ds
         + c_d int (T0 - Taverage)^2 dx
         + c_s int |grad(T0 - Taverage)|^2 dx

with

    c_d = alpha_d D_obs / D_damp,   c_s = alpha_s D_obs / D_smooth,
    D_obs    = int Tobs^2 dx,
    D_damp   = int (Tobs - Taverage)^2 dx,
    D_smooth = int |grad(Tobs - Taverage)|^2 dx,
    D_u      = int_top |uobs_{N-1}|^2 ds.

``m`` is the control, a Q1 (CG1) temperature at the start of the trajectory. ``T0``
is its L2 projection into the DQ2 state space. Q1 is a subspace of DQ2 and the
temperature boundary conditions of a DQ2 field are weak, so ``energy_solver.strong_bcs``
is empty and ``T0 = m`` up to quadrature. The production weights are alpha_T = 1,
alpha_u = 0.1, alpha_d = 0.01 and alpha_s = 0.1.

The metric. The two regularisation terms are exact quadratic forms in ``m``, so
their Hessian is the constant operator

    A = 2 c_d M + 2 c_s K,

with ``M`` the Q1 mass matrix and ``K`` the Q1 stiffness matrix. This is the
regularisation metric of the cylindrical case. The comparison uses it as the inner
product of the control for every optimiser: the Riesz map of the gradient, the norm
of ROL's vectors, the gradient norm of TAO, and the Krylov preconditioner of BNK
(``riesz_hessian.RieszPC`` applies ``A^{-1}``). The form here uses the default
quadrature, as the objective does; ``checks.py metric`` measures how far ``A`` is from
the Hessian of the taped regularisation terms.

Units. Everything is non-dimensional. The temperature lies in [0, 1], the annulus
runs from r = 1.22 to r = 2.22, and the timestep is ``delta_t`` in diffusion time
(production 6e-6).

Taping. ``generate_inverse_problem`` clears the working tape, annotates one forward
run, and pauses annotation before it returns. Nothing here annotates. Every optimiser
runs under ``stop_annotating``.

Files. The production code reads ``Checkpoint230.h5`` and ``Checkpoint_State.h5`` by
bare file name, so the process must run in a directory that holds them. The Hessian
harness ``tests/hessian/cylindrical_hvp_taylor.py`` already solves this for the same
problem (``prepare_rundir``, ``set_max_timesteps``, ``make_reference``), and this
module imports those functions instead of copying them. A reference state is valid
for one trajectory length and one timestep only.
"""

import gc
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mpi4py import MPI

# ---------------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------------
# The cylindrical test case and the Hessian harness are directories of plain scripts,
# not packages, and so is the box demo. They must be importable by name. The order
# matters: tests/adjoint_2d_cylindrical holds an ``inverse.py`` and a ``forward.py``,
# and so does the box demo. The cylindrical directory goes first, so that ``import
# inverse`` and ``import forward`` give the cylindrical case. From the box demo only
# ``optimisers``, ``riesz_hessian``, ``recording`` and ``problem`` are imported, and
# none of these names exists in the other two directories.
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
CYLINDRICAL_DIR = REPOSITORY / "tests" / "adjoint_2d_cylindrical"
HESSIAN_DIR = REPOSITORY / "tests" / "hessian"
BOX_DIR = REPOSITORY / "demos" / "optimisation" / "box"
for _directory in (BOX_DIR, HESSIAN_DIR, CYLINDRICAL_DIR):
    # Insert in reverse priority, each at position 0, so the final order is
    # CYLINDRICAL_DIR, HESSIAN_DIR, BOX_DIR.
    if str(_directory) in sys.path:
        sys.path.remove(str(_directory))
    sys.path.insert(0, str(_directory))

from gadopt import *  # noqa: E402,F401,F403
from gadopt.inverse import *  # noqa: E402,F401,F403 -- pyadjoint names
from pyadjoint import stop_annotating  # noqa: E402

import cylindrical_hvp_taylor as harness  # noqa: E402 -- tests/hessian
from problem import InnerProduct, make_inner_product  # noqa: E402,F401 -- the box demo

# The weights of ``inverse.inverse()``, the ROL inversion that g-adopt runs on this
# case. ``generate_inverse_problem`` switches a term off when its weight is not
# positive, so a single-term objective for a check is a tuple with -1 elsewhere.
PRODUCTION_WEIGHTS = {"alpha_T": 1.0, "alpha_u": 1e-1, "alpha_d": 1e-2, "alpha_s": 1e-1}

# The trajectory of ``forward.get_reference_values()``: 125 steps of 6e-6.
PRODUCTION_TIMESTEPS = 125
PRODUCTION_DELTA_T = 6e-6


@dataclass
class CylindricalProblem:
    """Everything an optimiser of the box benchmark needs, for the cylindrical case.

    The attribute names are those of ``box/problem.InverseProblem`` that
    ``optimisers.py``, ``riesz_hessian.py`` and ``recording.py`` read:
    ``reduced_functional``, ``control``, ``Tic``, ``bounds``, ``inner_product``,
    ``wall_nodes``, ``measures`` and ``prior_term``. The rest is for the run record.

    Attributes:
        reduced_functional: the reduced functional over the control, with the metric
            as the Riesz map of its control.
        control: the ``Control``. Its block variable is the one of the control of
            ``generate_inverse_problem``, so it refers to the same tape.
        Tic: the control Function, a Q1 temperature.
        Tic_true: the true initial temperature, DQ2, step 0 of the reference state.
        Tobs: the observed final temperature, DQ2, the last step of the reference.
        Taverage: the layer average of the initial reference temperature, Q1, the
            reference of the two regularisation terms.
        T: the DQ2 temperature of the forward model. ``T.block_variable`` is the
            final temperature on the tape, and its ``checkpoint`` holds the final
            temperature of the most recent forward replay.
        bounds: ``(lower, upper)`` Functions in the control space, 0 and 1, as in
            ``inverse.inverse()``.
        inner_product: the ``InnerProduct`` of the control, the metric ``A`` or the
            L2 mass matrix.
        weights: the objective weights alpha, the coefficients c_d and c_s of the
            objective, the metric weights and the metric coefficients.
        timesteps: the number of timesteps of the trajectory.
        delta_t: the timestep, non-dimensional.
        window_time: ``timesteps * delta_t``, the time span of the trajectory.
        norm_obs: ``int Tobs^2 dx``.
        norm_ic: ``int Tic_true^2 dx``.
        initial_misfit_floor: ``int (P Tic_true - Tic_true)^2 dx`` with ``P`` the L2
            projection into Q1. No Q1 control can have a smaller initial misfit.
        mesh: the extruded annulus mesh.
        rundir: the directory that holds the reference state.
        wall_nodes: empty. The box recorder splits the active set by the two walls
            of the box; this case has none to report, and the split is skipped.
        taped_objective: the value of J when the tape was recorded.
    """

    reduced_functional: object
    control: object
    Tic: Function
    Tic_true: Function
    Tobs: Function
    Taverage: Function
    T: Function
    bounds: tuple
    inner_product: InnerProduct
    weights: dict = field(default_factory=dict)
    timesteps: int = 0
    delta_t: float = 0.0
    window_time: float = 0.0
    norm_obs: float = 1.0
    norm_ic: float = 1.0
    initial_misfit_floor: Optional[float] = None
    mesh: object = None
    rundir: Optional[Path] = None
    wall_nodes: dict = field(default_factory=dict)
    taped_objective: Optional[float] = None

    # -- the twin-experiment measures --------------------------------------------------

    def _value(self, m):
        """The control value ``m``, or the current tape value when ``m`` is None."""
        return self.control.tape_value() if m is None else m

    def initial_misfit(self, m=None):
        """``int (m - Tic_true)^2 dx``: the error of the control against the truth.

        This is the cylindrical analogue of ``initial_misfit`` of the box records
        and of panel (c) of Fig. 5 of Ghelichkhan et al. (2024, GMD). It is
        unnormalised. Its smallest possible value is ``initial_misfit_floor``,
        because the truth is a DQ2 field and the control is Q1.

        Args:
            m: a Function in the control space; the tape value when None.

        Returns:
            float
        """
        m = self._value(m)
        return float(assemble((m - self.Tic_true) ** 2 * dx))

    def control_error(self, m=None):
        """``sqrt(initial_misfit / norm_ic)``, the same error as a relative norm."""
        return (self.initial_misfit(m) / self.norm_ic) ** 0.5

    def final_misfit(self):
        """``int (T(t_F) - Tobs)^2 dx`` of the most recent forward replay.

        This is the temperature misfit, the first term of the objective without its
        weight alpha_T, and panel (b) of Fig. 5 of the same paper. It belongs to the
        control of the last replay, which in a reduced-functional callback is the
        control the callback is about.

        Returns:
            float
        """
        return float(assemble((self.T.block_variable.checkpoint - self.Tobs) ** 2 * dx))

    def regularisation_terms(self, m=None):
        """The damping and the smoothing terms of the objective at ``m``, off the tape.

        The projection into DQ2 is repeated, with the same default quadrature as
        the taped objective, so that the two numbers are those of the objective.

        Args:
            m: a Function in the control space; the tape value when None.

        Returns:
            ``(damping, smoothing)``, each already multiplied by its coefficient
            ``c_d`` or ``c_s``. A term that is off in the objective is 0.0, and
            nothing is assembled for it.
        """
        c_d, c_s = self.weights["c_d"], self.weights["c_s"]
        if c_d == 0.0 and c_s == 0.0:
            return 0.0, 0.0
        m = self._value(m)
        with stop_annotating():
            # T0 of the objective: the control projected into the DQ2 state space.
            T0 = Function(self.Tic_true.function_space()).project(m)
            damping = smoothing = 0.0
            if c_d > 0.0:
                damping = c_d * float(assemble((T0 - self.Taverage) ** 2 * dx))
            if c_s > 0.0:
                smoothing = c_s * float(assemble(
                    dot(grad(T0 - self.Taverage), grad(T0 - self.Taverage)) * dx))
        return damping, smoothing

    def prior_term(self, m=None):
        """The regularisation part of J at ``m``, damping plus smoothing.

        The name is the one of the box records, where the same part of the
        objective is the prior term of the prior-covariance campaign.
        """
        return float(sum(self.regularisation_terms(m)))

    def measures(self, m=None):
        """The per-gradient measures of the run record.

        Args:
            m: control value for the control error; the tape value when None. The
                final misfit always belongs to the last forward replay.

        Returns:
            dict with ``initial_misfit``, ``control_error``, ``final_misfit``,
            ``relative_final_misfit``, ``damping_term`` and ``smoothing_term``.
            The velocity term is not here: it is the remainder of J and needs J,
            which the recorder adds (``CylindricalRecorder``).
        """
        damping, smoothing = self.regularisation_terms(m)
        final = self.final_misfit()
        return {
            "initial_misfit": self.initial_misfit(m),
            "control_error": float(self.control_error(m)),
            "final_misfit": final,
            "relative_final_misfit": final / self.norm_obs,
            "damping_term": damping,
            "smoothing_term": smoothing,
        }

    def record(self):
        """The problem settings for the JSON record, as plain numbers."""
        return {
            "timesteps": self.timesteps,
            "delta_t": self.delta_t,
            "window_time": self.window_time,
            "weights": self.weights,
            "control_dofs": self.Tic.function_space().dim(),
            "norm_obs": self.norm_obs,
            "norm_ic": self.norm_ic,
            "initial_misfit_floor": self.initial_misfit_floor,
            "taped_objective": self.taped_objective,
            "inner_product": self.inner_product.name,
            "rundir": str(self.rundir),
        }


def _final_temperature(tape, state_space):
    """The DQ2 temperature Function of the taped forward model.

    ``generate_inverse_problem`` does not return the state. Every energy solve on
    the tape writes the same Python Function ``T``, named "Temperature", so the
    output of any such block gives the object, and ``T.block_variable`` is the last
    one, the final temperature.

    Args:
        tape: the working tape.
        state_space: the DQ2 space of the temperature.

    Returns:
        Function

    Raises:
        RuntimeError: when no block writes a Function of that name and space.
    """
    for block in reversed(tape.get_blocks()):
        for output in block.get_outputs():
            obj = output.output
            if (isinstance(obj, Function) and obj.name() == "Temperature"
                    and obj.function_space() == state_space):
                return obj
    raise RuntimeError("no taped DQ2 'Temperature' found; the cylindrical inverse.py "
                       "no longer names its state as this module expects")


def prepare_reference(rundir, timesteps, delta_t, make=False):
    """Set the trajectory, enter the run directory, and make the reference if asked.

    Args:
        rundir: directory to run in. It receives a copy of ``Checkpoint230.h5`` and
            holds ``Checkpoint_State.h5``.
        timesteps: the number of timesteps of the trajectory.
        delta_t: the timestep, non-dimensional.
        make: run the forward model first to write ``Checkpoint_State.h5``.

    Returns:
        Path: the run directory. The process is inside it on return.

    Raises:
        FileNotFoundError: when the reference state is missing and ``make`` is
            False.
    """
    rundir = harness.prepare_rundir(rundir)
    os.chdir(rundir)
    harness.set_max_timesteps(timesteps, delta_t)
    stamp = rundir / "reference.json"
    if make:
        # ``from gadopt.inverse import *`` turns annotation on. The reference run is
        # data and must not be taped: at 125 steps its tape would hold tens of GB
        # until ``generate_inverse_problem`` clears it.
        with stop_annotating():
            harness.make_reference(timesteps, delta_t)
        if MPI.COMM_WORLD.rank == 0:
            # A record of the trajectory that this state belongs to. The state file
            # itself does not say, and a state of another length gives a valid-looking
            # but different problem.
            stamp.write_text(json.dumps({"timesteps": timesteps, "delta_t": delta_t}))
        MPI.COMM_WORLD.Barrier()
    if not (rundir / "Checkpoint_State.h5").exists():
        raise FileNotFoundError(f"no Checkpoint_State.h5 in {rundir}; pass "
                                "--make-reference once for this trajectory")
    if stamp.exists():
        made = json.loads(stamp.read_text())
        if made.get("timesteps") != timesteps or made.get("delta_t") != delta_t:
            raise ValueError(f"the reference state in {rundir} is for {made}, and this "
                             f"run asks for timesteps={timesteps}, delta_t={delta_t}")
    return rundir


def build_problem(rundir, timesteps=PRODUCTION_TIMESTEPS, delta_t=PRODUCTION_DELTA_T,
                  alpha_T=1.0, alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1,
                  inner_product="regularisation", metric_alpha_d=None,
                  metric_alpha_s=None, make_reference=False):
    """Tape the cylindrical forward model once and return the inverse problem.

    Args:
        rundir: run directory; see ``prepare_reference``.
        timesteps: number of timesteps of the trajectory.
        delta_t: timestep, non-dimensional. 6e-6 is the production value. The
            negative curvature of NOTES/CYLINDRICAL-HESSIAN-2026-11.md was measured
            at 125 steps of 1.2e-5.
        alpha_T, alpha_u, alpha_d, alpha_s: objective weights, with the meaning of
            ``generate_inverse_problem``. A weight that is not positive switches its
            term off.
        inner_product: ``"regularisation"`` for the metric ``A`` above, ``"L2"`` for
            the mass matrix, which is what ``gadopt.inverse.LinMoreOptimiser`` uses.
        metric_alpha_d, metric_alpha_s: weights of the metric. None takes the
            objective weights, so that ``A`` is the Hessian of the regularisation.
            A single-term objective for a check has no regularisation, and then the
            production weights 0.01 and 0.1 are used, because the metric needs a
            positive mass weight.
        make_reference: write the reference state first.

    Returns:
        CylindricalProblem
    """
    rundir = prepare_reference(rundir, timesteps, delta_t, make=make_reference)
    import inverse as production  # tests/adjoint_2d_cylindrical/inverse.py

    produced = production.generate_inverse_problem(
        alpha_T=alpha_T, alpha_u=alpha_u, alpha_d=alpha_d, alpha_s=alpha_s,
        checkpointing_schedule=None, uimposed=False)
    # No checkpoint schedule: every state stays in memory. The Hessian-vector product
    # needs this, because pyadjoint has no checkpoint-manager path for the
    # tangent-linear and second-order adjoint sweeps.
    rf_production = produced["reduced_functional"]
    Tic = produced["control"]
    objective = rf_production.functional
    tape = get_working_tape()

    with stop_annotating():
        mesh = Tic.function_space().mesh()
        Q1 = Tic.function_space()
        DQ = FunctionSpace(mesh, "DQ", 2)
        # The data, read again from the reference state. These are data, not tape
        # entries, so a second copy is harmless.
        with CheckpointFile("Checkpoint_State.h5", "r") as f:
            Tobs = f.load_function(mesh, "Temperature", idx=timesteps - 1)
            Tic_true = f.load_function(mesh, "Temperature", idx=0)
            Taverage = f.load_function(mesh, "Average_Temperature", idx=0)
        Tobs.rename("Observed_Temperature")
        Tic_true.rename("True_Initial_Temperature")
        T = _final_temperature(tape, Tobs.function_space())

        # The normalisations of the objective, with the definitions and the default
        # quadrature of inverse.py, so that c_d and c_s are its numbers.
        norm_obs = float(assemble(Tobs ** 2 * dx))
        norm_damping = float(assemble((Tobs - Taverage) ** 2 * dx))
        norm_smoothing = float(assemble(dot(grad(Tobs - Taverage),
                                            grad(Tobs - Taverage)) * dx))
        c_d = alpha_d * norm_obs / norm_damping if alpha_d > 0 else 0.0
        c_s = alpha_s * norm_obs / norm_smoothing if alpha_s > 0 else 0.0

        # The metric weights. They follow the objective unless set, and fall back to
        # the production weights for an objective without regularisation.
        if metric_alpha_d is None:
            metric_alpha_d = alpha_d if alpha_d > 0 else PRODUCTION_WEIGHTS["alpha_d"]
        if metric_alpha_s is None:
            metric_alpha_s = alpha_s if alpha_s > 0 else PRODUCTION_WEIGHTS["alpha_s"]
        metric_c_d = metric_alpha_d * norm_obs / norm_damping
        metric_c_s = metric_alpha_s * norm_obs / norm_smoothing
        if inner_product == "regularisation" and (metric_c_d <= 0 or metric_c_s < 0):
            # The mass term makes the form positive definite. K alone has the
            # constants in its null space and a direct solver would factorise it
            # without a complaint.
            raise ValueError("the regularisation metric needs a positive mass weight "
                             f"and got metric_alpha_d={metric_alpha_d}")
        # The factor 2 makes the form the Hessian of c_d int m^2 + c_s int |grad m|^2.
        ip = make_inner_product(Q1, inner_product, c_d=2 * metric_c_d,
                                c_s=2 * metric_c_s)

        # The best initial misfit that a Q1 control can reach.
        best = Function(Q1).project(Tic_true)
        floor = float(assemble((best - Tic_true) ** 2 * dx))
        norm_ic = float(assemble(Tic_true ** 2 * dx))

        T_lb = Function(Q1, name="Lower_bound_temperature").assign(0.0)
        T_ub = Function(Q1, name="Upper_bound_temperature").assign(1.0)

    # The control with the metric as its Riesz map. ``Control`` stores the block
    # variable that ``Tic`` holds now. Nothing has written to ``Tic`` since
    # ``generate_inverse_problem`` made its own Control, so the two refer to the same
    # tape entry; the assertion guards that.
    control = Control(Tic, riesz_map=ip.riesz_map)
    if control.block_variable is not rf_production.controls[0].block_variable:
        raise RuntimeError("the control no longer refers to the taped control")

    def collect_after_hessian(*args):
        """Collect garbage after each Hessian-vector product.

        A product leaves reference cycles that CPython frees only when its cyclic
        collector runs, so the memory of a long run grows by a constant amount per
        product until then. The box demo measured this and collects here too.
        """
        gc.collect()

    reduced_functional = ReducedFunctional(objective, control,
                                           hessian_cb_post=collect_after_hessian)
    # The tape lives for the whole run. Freezing takes its objects out of every later
    # collection, so the collection above stays cheap.
    gc.collect()
    gc.freeze()

    return CylindricalProblem(
        reduced_functional=reduced_functional,
        control=control,
        Tic=Tic,
        Tic_true=Tic_true,
        Tobs=Tobs,
        Taverage=Taverage,
        T=T,
        bounds=(T_lb, T_ub),
        inner_product=ip,
        weights={"alpha_T": alpha_T, "alpha_u": alpha_u, "alpha_d": alpha_d,
                 "alpha_s": alpha_s, "c_d": c_d, "c_s": c_s,
                 "metric_alpha_d": metric_alpha_d, "metric_alpha_s": metric_alpha_s,
                 "metric_c_d": metric_c_d, "metric_c_s": metric_c_s,
                 "norm_damping": norm_damping, "norm_smoothing": norm_smoothing},
        timesteps=timesteps,
        delta_t=delta_t,
        window_time=timesteps * delta_t,
        norm_obs=norm_obs,
        norm_ic=norm_ic,
        initial_misfit_floor=floor,
        mesh=mesh,
        rundir=rundir,
        taped_objective=float(produced["objective"]),
    )
