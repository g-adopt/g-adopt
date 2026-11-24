"""Optimiser back-ends for the box benchmark: ROL Lin-More and PETSc TAO.

Every runner takes an ``InverseProblem`` from ``problem.py`` and returns a
``Recorder`` with the run history. The tape is never annotated here: each solve is
wrapped in ``stop_annotating()``, and every callback runs inside a pyadjoint call that
already carries ``no_annotations``.

Cost is counted at the tape level, through the reduced functional callbacks:

- one forward replay per ``eval_cb_post``,
- one adjoint sweep per ``derivative_cb_post``,
- one tangent-linear plus one second-order adjoint sweep per ``hessian_cb_post``.

These are the counts that carry over to any solver configuration. Wall time is
recorded too, but it is the secondary measure.

Optimiser variants, selected by name:

ROL, through ``ROL.TypeB.LinMoreAlgorithm`` with an L-BFGS secant:

    rol-lbfgs          secant used as the Hessian (what g-adopt runs today)
    rol-newton         true Hessian-vector products, secant as the CG preconditioner
    rol-newton-noprec  true Hessian-vector products, no preconditioner

ROL, through ``ROL.TypeB.NewtonKrylovAlgorithm``:

    rol-newton-krylov  true Hessian-vector products, secant as the CG preconditioner.
                       One Krylov solve per outer iteration on the free variables, then
                       a projected backtracking line search that applies no Hessian.
                       Lin-More instead applies the model Hessian many times per outer
                       iteration, in its Cauchy point stage and in every round of its
                       minor loop, which is free for a secant and costs 2.2 forward
                       calls per application for a true Hessian. This variant has the
                       same structure as TAO's ``bnls``, so the pair separates the
                       algorithm from the library.

TAO, through ``pyadjoint.TAOSolver``:

    tao-blmvm          bounded L-BFGS, initial inverse Hessian = the Riesz map
    tao-bqnktr         bounded quasi-Newton-Krylov trust region: Krylov on the LMVM
                       matrix, which replaces the Hessian (bqnk.c sets
                       ``tao->hessian = B``), so no true Hessian-vector products
    tao-bqnkls         the same with a line search
    tao-bntr           bounded Newton trust region, true Hessian, LMVM preconditioner
    tao-bntr-noprec    the same with no preconditioner
    tao-bnls           bounded Newton line search, true Hessian, LMVM preconditioner

Any TAO option can be added or overridden with ``extra_options`` in the usual PETSc
form, for example ``{"tao_bnk_ksp_max_it": 20, "tao_ksp_ew": None}``.
"""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from gadopt import *
from gadopt.inverse import ROLSolver, minimisation_parameters
import pyadjoint.optimization.rol_solver as pyadjoint_rol
from pyadjoint import MinimizationProblem, stop_annotating
from pyadjoint.enlisting import Enlist
from pyadjoint.optimization.tao_solver import PETScVecInterface, RieszMapMat, TAOConvergenceError, TAOSolver
from riesz_hessian import build_preconditioner_matrix, build_shifted_hessian
from recording import PETSC_LIBRARY, KrylovRecorder, gradient_norms, tao_view_text
from gauss_newton import HESSIAN_CHOICES, LINEAR_DEFECT_TOLERANCE, build_gauss_newton_hessian
import ROL


class WorkBudgetReached(Exception):
    """Raised inside the TAO monitor when the work budget is spent.

    TAO has no option for a budget of this kind, and a monitor cannot end a solve by
    setting the converged reason, because the convergence test that runs after the
    monitor overwrites it. Raising is the way to stop.

    petsc4py does not pass the exception through: it turns any Python exception
    raised inside a callback into ``PETSc.Error`` with code 101, which is
    ``PETSC_ERR_PYTHON``, and the original type is gone by the time ``TAO.solve``
    returns. So the recorder also carries a flag, and ``run_tao`` reads the flag to
    tell a deliberate stop from a real PETSc failure.

    ROL is stopped through its status test and needs none of this.
    """


ROL_VARIANTS = ("rol-lbfgs", "rol-newton", "rol-newton-noprec", "rol-newton-krylov")
# The variants that run through NewtonKrylovAlgorithm instead of LinMoreAlgorithm.
ROL_NEWTON_KRYLOV_VARIANTS = ("rol-newton-krylov",)
TAO_VARIANTS = ("tao-blmvm", "tao-bqnktr", "tao-bqnkls", "tao-bntr", "tao-bntr-noprec",
                "tao-bntr-riesz", "tao-bnls", "tao-bnls-riesz")
VARIANTS = ROL_VARIANTS + TAO_VARIANTS
# The TAO types that solve a Krylov system per iteration through the BNK kernel. The
# per-solve recording of ``recording.KrylovRecorder`` applies to these alone.
TAO_KRYLOV_TYPES = ("bnls", "bntr", "bntl", "bqnkls", "bqnktr", "bqnktl")


# ---------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------

@dataclass
class Recorder:
    """Counts tape sweeps and records the state at every gradient evaluation.

    The recorder attaches to the reduced functional's callbacks. ``gradients`` gets one
    row per adjoint sweep, which is every point where an optimiser looked at the
    gradient, accepted or not. ``iterations`` gets one row per optimiser iteration,
    added by the optimiser's own monitor, and refers to the sweep counts at that
    moment.

    Attributes:
        problem: the ``InverseProblem``.
        output: path of the JSON file, or None. When it is set, the record is
            rewritten after every optimiser iteration, so a run that is stopped by a
            wall-clock limit still leaves its history on disk.
        metadata: the run settings, written into every dump.
        work_budget: stop when the forward-equivalent work passes this number. Zero
            removes the limit. Every method gets the same budget, so that a method
            which spends little per iteration and one which spends a lot are read at
            the same total cost. The test runs at the end of an optimiser iteration,
            so a run can pass the budget by up to the cost of one iteration: about
            two forward calls for a gradient-only method and about 2 + 2k for a
            Newton-Krylov method with k Krylov iterations.
        hessian_price: cost of one Hessian-vector product in forward calls, used to
            convert the counts into work. It is replaced by this run's own measured
            value as soon as there is one.
        gn_hessian_price: the same for one Gauss-Newton product
            (``gauss_newton.GaussNewtonOperator.action``). The default 2.0 is used
            only before the first Gauss-Newton product of the run ends: a product is
            one tangent-linear and one adjoint sweep, about the cost of a gradient
            plus a forward run. Afterwards the measured ratio replaces it.
        forward_times: wall time of every call to the reduced functional.
        gradient_times: wall time of every adjoint sweep.
        hessian_times: wall time of every Hessian-vector product.
        gn_hessian_times: wall time of every Gauss-Newton product, measured by the
            operator around its three sweeps and the release step, before the
            garbage collection that follows.
        forward: number of forward replays so far.
        adjoint: number of adjoint sweeps so far.
        hessian: number of Hessian-vector products so far (pyadjoint's full product,
            counted through ``hessian_cb_post``).
        gn_hessian: number of Gauss-Newton products so far. The operator calls
            ``end_gn`` after each one; the reduced-functional
            callbacks never fire for it, so the two counts never overlap.
        gradients: list of dict, one per gradient evaluation.
        iterations: list of dict, one per optimiser iteration.
        last_dual_gradient: local entries of the most recent gradient, when the
            optimiser asked for it in the dual space (TAO), in the layout of the
            PETSc vector of the control; None before the first gradient and for
            ROL. ``recording.gradient_norms`` needs it, because BNK keeps the
            unprojected gradient private.
        final_diagnostics: entries that exist only at the end of a run, such as the
            text of ``-tao_view``. They go into every dump at the top level.
    """

    problem: object
    forward: int = 0
    adjoint: int = 0
    hessian: int = 0
    gn_hessian: int = 0
    gradients: list = field(default_factory=list)
    iterations: list = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)
    output: object = None
    metadata: dict = field(default_factory=dict)
    _chained: dict = field(default_factory=dict)
    stop_reason: str = ""
    work_budget: float = 0.0
    hessian_price: float = 1.96
    gn_hessian_price: float = 2.0
    budget_hit: bool = False
    forward_times: list = field(default_factory=list)
    gradient_times: list = field(default_factory=list)
    hessian_times: list = field(default_factory=list)
    gn_hessian_times: list = field(default_factory=list)
    _call_start: dict = field(default_factory=dict)
    last_dual_gradient: object = None
    final_diagnostics: dict = field(default_factory=dict)

    def attach(self):
        """Install the callbacks on the reduced functional.

        pyadjoint calls a ``_pre`` callback before each operation and a ``_post``
        callback after it, so a pair of them times the operation itself. This is the
        only way to separate the cost of one call from the number of calls; the
        cumulative time per iteration cannot tell a slow call from many calls.
        """
        rf = self.problem.reduced_functional
        # The callbacks already on the reduced functional are kept and called after
        # the recorder's own. ``problem.build_problem`` installs a garbage collection
        # on ``hessian_cb_post``, and replacing it here would remove the one thing
        # that keeps the memory of a long run flat.
        self._chained = {
            "eval_cb_post": rf.eval_cb_post,
            "derivative_cb_post": rf.derivative_cb_post,
            "hessian_cb_post": rf.hessian_cb_post,
        }
        rf.eval_cb_pre = self._start("forward")
        rf.eval_cb_post = self._eval_cb
        rf.derivative_cb_pre = self._start("gradient", passthrough=True)
        rf.derivative_cb_post = self._derivative_cb
        rf.hessian_cb_pre = self._start("hessian")
        rf.hessian_cb_post = self._hessian_cb

    def _chain(self, name, *args):
        """Call the callback that was on the reduced functional before ``attach``.

        pyadjoint installs a no-op lambda by default, so there is always something to
        call. The return value of the ``_post`` callbacks is not used by pyadjoint,
        except for ``derivative_cb_post``, which the recorder handles itself.
        """
        callback = self._chained.get(name)
        if callback is not None:
            return callback(*args)

    def _start(self, kind, passthrough=False):
        """Build a ``_pre`` callback that stamps the start time of one operation.

        Args:
            kind: ``forward``, ``gradient`` or ``hessian``.
            passthrough: ``derivative_cb_pre`` must return the controls it is given,
                because pyadjoint passes the return value on.

        Returns:
            the callback.
        """
        def callback(*args):
            self._call_start[kind] = time.perf_counter()
            if passthrough:
                return args[0] if len(args) == 1 else args
        return callback

    def _elapsed(self, kind):
        """Time since the matching ``_pre`` callback, or None if there was none."""
        start = self._call_start.pop(kind, None)
        return None if start is None else time.perf_counter() - start

    def call_cost_summary(self):
        """Mean time of each operation and the ratios that price a Hessian product.

        The ratio of a Hessian-vector product to a forward call is what converts a
        Hessian count into forward-equivalent work, which is the only axis on which
        a Newton method and a quasi-Newton method can be compared honestly.

        Returns:
            dict
        """
        def mean(values):
            return float(sum(values) / len(values)) if values else None

        f, g, h = mean(self.forward_times), mean(self.gradient_times), mean(self.hessian_times)
        gn = mean(self.gn_hessian_times)
        return {
            "mean_forward_time": f,
            "mean_gradient_time": g,
            "mean_hessian_time": h,
            "mean_gn_hessian_time": gn,
            "gradient_per_forward": None if not (f and g) else g / f,
            "hessian_per_forward": None if not (f and h) else h / f,
            "gn_hessian_per_forward": None if not (f and gn) else gn / f,
        }

    def end_gn(self, elapsed):
        """Count one Gauss-Newton product and store its wall time.

        Args:
            elapsed: the time the operator measured around its three sweeps and its
                release step, before its garbage collection.
        """
        self.gn_hessian += 1
        self.gn_hessian_times.append(float(elapsed))

    def _eval_cb(self, J, values):
        self.forward += 1
        dt = self._elapsed("forward")
        if dt is not None:
            self.forward_times.append(dt)
        self._chain("eval_cb_post", J, values)

    def _hessian_cb(self, J, r, values):
        # The time is taken before the chained callback runs, because that callback is
        # the garbage collection and its cost belongs to neither the product nor the
        # next one.
        dt = self._elapsed("hessian")
        self.hessian += 1
        if dt is not None:
            self.hessian_times.append(dt)
        self._chain("hessian_cb_post", J, r, values)

    def _derivative_cb(self, J, dJ, values):
        """Record the gradient norm and the twin-experiment error, then pass dJ on.

        ``dJ`` is a Cofunction when the optimiser asked for the dual derivative
        (TAO) and a Function when it asked for the Riesz representative (ROL). Both
        norms below are the norm in the control inner product.
        """
        self.adjoint += 1
        dt = self._elapsed("gradient")
        if dt is not None:
            self.gradient_times.append(dt)
        ip = self.problem.inner_product
        g = dJ[0] if isinstance(dJ, (list, tuple)) else dJ
        if isinstance(g, Cofunction):
            gnorm = ip.dual_norm(g)
        else:
            gnorm = ip.norm(g)
        m = values[0] if isinstance(values, (list, tuple)) else values
        if isinstance(g, Cofunction):
            # A copy, in the layout of the PETSc vector that TAO holds: the owned
            # entries of the dat, which is what ``PETScVecInterface.to_petsc`` copies
            # into TAO's gradient for a single control.
            self.last_dual_gradient = np.array(g.dat.data_ro, dtype=float)
        # The two misfits of Ghelichkhan et al. (2024, GMD) Fig. 5: the initial misfit
        # against the reference initial condition, and the final misfit against Tobs.
        # Both are taken here, at the point where the adjoint was just computed, so
        # the forward state on the tape belongs to ``m``.
        row = {
            "J": float(J),
            "gradient_norm": float(gnorm),
            "forward": self.forward,
            "adjoint": self.adjoint,
            "hessian": self.hessian,
            "gn_hessian": self.gn_hessian,
            "cost": self.forward + self.adjoint,
            "time": time.perf_counter() - self.t0,
        }
        row.update(self.problem.measures(m))
        # The two parts of the objective, under the names of the prior-covariance
        # campaign. The misfit term is the final misfit, the same integral as the
        # first term of the taped objective; the prior term is the rest. J is their
        # sum.
        row["misfit_term"] = row["final_misfit"]
        row["prior_term"] = self.problem.prior_term(m)
        self.gradients.append(row)
        self._chain("derivative_cb_post", J, dJ, values)
        return dJ

    def mark_iteration(self, k, **extra):
        """Record one optimiser iteration. Called by the optimiser's monitor."""
        row = {"iteration": int(k), "forward": self.forward, "adjoint": self.adjoint,
               "hessian": self.hessian, "gn_hessian": self.gn_hessian,
               "cost": self.forward + self.adjoint,
               "work": self.work(),
               "time": time.perf_counter() - self.t0}
        if self.gradients:
            # Carry the measures of the most recent gradient evaluation, which is the
            # accepted iterate for every optimiser used here.
            row.update({key: self.gradients[-1][key]
                        for key in ("J", "gradient_norm", "control_error",
                                    "initial_misfit", "final_misfit",
                                    "relative_final_misfit", "misfit_term",
                                    "prior_term", "initial_misfit_bottom",
                                    "initial_misfit_top", "initial_misfit_interior",
                                    "wall_max_dev_bottom", "wall_max_dev_top")
                        if key in self.gradients[-1]})
        row.update(extra)
        self.iterations.append(row)
        self.write()
        if MPI.COMM_WORLD.rank == 0:
            log(f"iter {k:3d}  J {row.get('J', float('nan')):.6e}  "
                f"|g| {row.get('gradient_norm', float('nan')):.3e}  "
                f"IC {row.get('initial_misfit', float('nan')):.4e}  "
                f"FIN {row.get('final_misfit', float('nan')):.4e}  "
                f"err {row.get('control_error', float('nan')):.4e}  "
                f"fwd {self.forward} adj {self.adjoint} hvp {self.hessian}"
                + (f" gn {self.gn_hessian}" if self.gn_hessian else ""))

    def write(self):
        """Write the record to ``self.output``, if there is one. Rank 0 only.

        The whole record is rewritten each time. It holds a few hundred rows, so the
        cost is far below one tape sweep, and a whole file is always valid JSON even
        if the process is killed during the next write.
        """
        if self.output is None or MPI.COMM_WORLD.rank != 0:
            return
        record = dict(self.metadata)
        record.update(self.summary())
        tmp = f"{self.output}.tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=1)
        os.replace(tmp, self.output)

    def work(self):
        """Forward-equivalent work spent so far.

        ``forward + adjoint + p_full hessian + p_gn gn_hessian``. Each product is
        priced at its measured cost in forward calls (mean product time over mean
        forward time of this run), so the number is what the run has actually paid
        and not a count of sweeps. Before the first product of a kind ends, its
        default price applies.

        Returns:
            float
        """
        costs = self.call_cost_summary()
        price = self.hessian_price
        if costs["hessian_per_forward"]:
            price = costs["hessian_per_forward"]
        gn_price = self.gn_hessian_price
        if costs["gn_hessian_per_forward"]:
            gn_price = costs["gn_hessian_per_forward"]
        work = self.forward + self.adjoint + price * self.hessian
        if self.gn_hessian:
            # A run without a Gauss-Newton product adds no term, so its work is
            # ``forward + adjoint + price * hessian`` exactly.
            work += gn_price * self.gn_hessian
        return work

    def over_budget(self):
        """Whether the work budget is set and has been passed."""
        return self.work_budget > 0.0 and self.work() >= self.work_budget

    def summary(self):
        """Dictionary for JSON output."""
        return {
            "forward": self.forward,
            "adjoint": self.adjoint,
            "hessian": self.hessian,
            "gn_hessian": self.gn_hessian,
            "wall_time": time.perf_counter() - self.t0,
            "forward_times": self.forward_times,
            "gradient_times": self.gradient_times,
            "hessian_times": self.hessian_times,
            "gn_hessian_times": self.gn_hessian_times,
            "call_cost": self.call_cost_summary(),
            "work": self.work(),
            "work_budget": self.work_budget,
            "stop_reason": self.stop_reason,
            "gradients": self.gradients,
            "iterations": self.iterations,
            **self.final_diagnostics,
        }


# ---------------------------------------------------------------------------------
# ROL
# ---------------------------------------------------------------------------------

class ShiftedROLObjective(pyadjoint_rol.ROLObjective):
    """A reduced functional whose Hessian action carries a shift ``lambda``.

    The action returned is ``H v + lambda v`` in the control space. pyadjoint returns
    the Hessian action already mapped through the Riesz map of the control inner
    product, so the operator ROL's Krylov solver sees is ``A^{-1} H``, with ``A`` the
    inner-product operator. Adding ``lambda v`` to the mapped action therefore makes
    that operator ``A^{-1} H + lambda I``, which is ``A^{-1} (H + lambda A)``: the
    shift is by a multiple of the inner-product operator, and not by a multiple of the
    identity in nodal coordinates.

    Why this exists. The misfit Hessian of this problem with no regularisation is
    indefinite: ROL's truncated CG met negative curvature within two or three
    iterations in 18 of the 25 outer iterations of the first
    ``rol-newton-krylov`` run. ROL has no answer to that in any of its algorithms or
    parameters; it truncates CG and returns the partial iterate, which after one step
    is a preconditioned steepest-descent direction. PETSc's bounded Newton types
    instead shift the Hessian whenever they see negative curvature, an indefinite
    matrix, a direction that is not descent, or a failed line search, with
    ``pert = min(100, max(1e-4, 0.1 gnorm))`` grown tenfold on each further failure
    (``bnk.c:563-580``). This class gives ROL a shift from outside the library, so
    that both methods regularise an indefinite model.

    The two shifts are not the same operator. ROL gets ``lambda A`` as described
    above. BNK adds ``pert * I`` in nodal coordinates (``MatShift``), unless the run
    sets ``--bnk-shift metric``, which makes BNK's shift ``pert * A`` as well
    (``riesz_hessian.ShiftedHessianMat``). The nodal identity is far larger than the
    Hessian of this CG1 problem, whose nodal scale is that of the mass matrix, so
    with the identity shift the two methods are regularised very differently.

    ``shift`` is set by the caller once per outer iteration and held fixed for every
    Hessian application inside that iteration, which is what BNK does.

    Attributes:
        shift: the current value of lambda. Zero applies no shift.
    """

    def __init__(self, rf, scale=1.0):
        super().__init__(rf, scale=scale)
        self.shift = 0.0

    def hessVec(self, hv, v, x, tol):
        """Apply the shifted Hessian to ``v`` and write the result into ``hv``."""
        super().hessVec(hv, v, x, tol)
        if self.shift:
            # ``hv.dat`` and ``v.dat`` are lists of controls. Enlist gives a list for a
            # single control as well, so one loop covers both cases.
            for h, direction in zip(Enlist(hv.dat), Enlist(v.dat)):
                h.assign(h + self.shift * direction)


class FormROLVector(pyadjoint_rol.ROLVector):
    """A ROL vector whose inner product is a user bilinear form on the control space.

    pyadjoint's ``ROLVector`` accepts only the strings ``l2``, ``L2`` and ``H1`` for
    the inner product, because ``Function._ad_dot`` accepts only those. This class
    takes an ``InnerProduct`` instead, so that the regularisation operator can be the
    metric. ROL's algorithms are only consistent if ``dot`` and the Riesz map that
    produced the gradient agree, so the same ``InnerProduct`` must be on the Control.

    Args:
        dat: list of Functions (the pyadjoint convention).
        ip: an ``InnerProduct`` from ``problem.py``.
        inner_product: ignored; accepted because ``ROLSolver`` passes it by keyword.
    """

    def __init__(self, dat, ip, inner_product=None):
        # ``gadopt.inverse.ROLSolver`` calls ``vector_class(x, *vector_args,
        # inner_product=...)``, so the string arrives as a keyword. The base class
        # stores it for its own ``dot`` and ``riesz_map``; both are overridden here,
        # so the stored value is only a tag.
        super().__init__(dat, inner_product=ip.name)
        self.ip = ip

    def dot(self, yy):
        return sum(float(assemble(self.ip.form(x, y))) for x, y in zip(self.dat, yy.dat))

    def riesz_map(self, derivs):
        derivs = derivs if isinstance(derivs, (list, tuple)) else [derivs]
        return [self.ip.riesz_map(d) for d in derivs]

    def clone(self):
        res = FormROLVector([x._ad_copy() for x in self.dat], self.ip)
        res.scale(0.0)
        return res


def rol_parameters(variant, max_it, initial_radius=1e-2, secant_storage=10,
                   krylov_max_it=20, krylov_rtol=1e-2, krylov_atol=1e-4,
                   overrides=None):
    """ROL parameter list for one variant, from g-adopt's ``minimisation_parameters``.

    Args:
        variant: one of ``ROL_VARIANTS``.
        max_it: iteration limit.
        initial_radius: initial trust-region radius.
        secant_storage: L-BFGS memory.
        krylov_max_it: truncated CG iteration limit. Each iteration is one
            Hessian-vector product when the true Hessian is used.
        krylov_rtol, krylov_atol: truncated CG tolerances.
        overrides: dict mapping a tuple of keys to a value, applied last. Lin-More
            prices a model-Hessian application at zero, which is right for the secant
            and wrong for a true Hessian-vector product that costs about 2.2 forward
            calls, so its inner limits need to be tunable from a run script. The
            entries that matter are the radius rule under ``Step/Trust Region``, the
            Krylov tolerances under ``General/Krylov``, and the projected-search and
            minor-iteration limits under ``Step/Trust Region/Lin-More``.

    Returns:
        dict, a deep copy with the variant applied.
    """
    import copy
    params = copy.deepcopy(minimisation_parameters)
    params["Status Test"]["Iteration Limit"] = max_it
    params["Status Test"]["Gradient Tolerance"] = 0.0
    params["Step"]["Trust Region"]["Initial Radius"] = initial_radius
    params["General"]["Secant"]["Maximum Storage"] = secant_storage
    params["General"]["Krylov"]["Iteration Limit"] = krylov_max_it
    params["General"]["Krylov"]["Relative Tolerance"] = krylov_rtol
    params["General"]["Krylov"]["Absolute Tolerance"] = krylov_atol
    # ``Use as Hessian`` routes every model Hessian-vector product to the secant.
    # ``Use as Preconditioner`` applies the inverse secant inside truncated CG.
    if variant == "rol-lbfgs":
        params["General"]["Secant"]["Use as Hessian"] = True
        params["General"]["Secant"]["Use as Preconditioner"] = False
    elif variant == "rol-newton":
        params["General"]["Secant"]["Use as Hessian"] = False
        params["General"]["Secant"]["Use as Preconditioner"] = True
    elif variant == "rol-newton-noprec":
        params["General"]["Secant"]["Use as Hessian"] = False
        params["General"]["Secant"]["Use as Preconditioner"] = False
    elif variant == "rol-newton-krylov":
        # NewtonKrylovAlgorithm reads the same two keys and gives them the same
        # meaning, so the settings match rol-newton: the true Hessian in the Krylov
        # operator, the inverse secant as the Krylov preconditioner.
        params["General"]["Secant"]["Use as Hessian"] = False
        params["General"]["Secant"]["Use as Preconditioner"] = True
        # The Krylov type must be named. NewtonKrylovAlgorithm builds its solver
        # through KrylovFactory, which reads this key with the default "GMRES"
        # (ROL_KrylovFactory.hpp:135), while the algorithm reads the SAME key for its
        # display name with the default "Conjugate Gradients"
        # (ROL_TypeB_NewtonKrylovAlgorithm_Def.hpp:66). Leaving the key out therefore
        # runs GMRES under a label that says CG. GMRES also sets its flag to 0 or 1
        # only, so the negative-curvature branch of the run loop can never fire, and
        # an ascent direction would pass the Armijo test, because that test compares
        # against ``-c1 * gs`` with ``gs`` the inner product of the step and the
        # gradient. CG detects negative curvature and the fallback to steepest descent
        # works, so name it here.
        params["General"]["Krylov"]["Type"] = "Conjugate Gradients"
    else:
        raise ValueError(f"unknown ROL variant {variant!r}")
    params["General"]["Print Verbosity"] = 1 if MPI.COMM_WORLD.rank == 0 else 0
    params["General"]["Output Level"] = 1 if MPI.COMM_WORLD.rank == 0 else 0
    # Applied last, so that a run script can change any entry, including the ones the
    # variant set above. Every key on the path must already exist: a typo in a ROL
    # parameter name is silently ignored by ROL itself, so it is caught here instead.
    for path, value in (overrides or {}).items():
        node = params
        for key in path[:-1]:
            if key not in node:
                raise KeyError(f"no ROL parameter group {'/'.join(path)!r}: {key!r}")
            node = node[key]
        if path[-1] not in node:
            raise KeyError(f"no ROL parameter {'/'.join(path)!r}: {path[-1]!r}")
        node[path[-1]] = value
    return params


def run_rol(problem, variant, max_it=20, output=None, metadata=None,
            gradient_rtol=0.0, work_budget=0.0, hessian_shift=0.0, **kwargs):
    """Run one ROL Lin-More variant on the problem.

    This mirrors ``gadopt.inverse.LinMoreOptimiser`` without the ROL checkpointing,
    and with the control inner product as the vector metric. The status test is
    replaced so that every iteration is recorded before the default test decides
    whether to stop.

    Args:
        problem: an ``InverseProblem``.
        variant: one of ``ROL_VARIANTS``.
        max_it: iteration limit.
        output: JSON file to rewrite after every iteration, or None.
        metadata: run settings written into every dump.
        work_budget: stop when the forward-equivalent work passes this number.
        gradient_rtol: stop when the gradient norm falls to this fraction of its
            value at the first iteration. Zero runs to the iteration limit. ROL's
            own ``Gradient Tolerance`` is an absolute number, and the starting norm
            is not known before the run, so the test is applied here instead.
        **kwargs: forwarded to ``rol_parameters``.

    Returns:
        Recorder
    """
    recorder = Recorder(problem, output=output, metadata=metadata or {},
                        work_budget=work_budget)
    recorder.attach()

    params = rol_parameters(variant, max_it, **kwargs)
    rol_params = ROL.ParameterList(params, "Parameters")
    minimisation_problem = MinimizationProblem(problem.reduced_functional, bounds=problem.bounds)
    solver = ROLSolver(minimisation_problem, params, inner_product=problem.inner_product.name,
                       vector_class=FormROLVector, vector_args=[problem.inner_product])
    if hessian_shift > 0.0:
        # Replace the objective with one that shifts the Hessian action. ``run`` below
        # is given this object explicitly, so swapping it here is enough.
        solver.rolobjective = ShiftedROLObjective(problem.reduced_functional)
    secant = ROL.lBFGS(params["General"]["Secant"]["Maximum Storage"])
    if variant in ROL_NEWTON_KRYLOV_VARIANTS:
        # NewtonKrylovAlgorithm ignores the whole Step/Trust Region block, including
        # the initial radius, so ``initial_radius`` has no effect on this variant. It
        # is a line-search method and it starts every outer iteration at a step length
        # of 1.0, because "Use Previous Step Length as Initial Guess" is false.
        algorithm = ROL.NewtonKrylovAlgorithm(rol_params, secant)
    else:
        algorithm = ROL.LinMoreAlgorithm(rol_params, secant)

    class StatusTest(ROL.StatusTest):
        """Record the iterate, apply the relative gradient test, then defer to ROL.

        ROL's ``gnorm`` is the norm of the projected gradient in the vector's own
        inner product, which is the control inner product here. The first value seen
        is the reference for the relative test.
        """
        iteration = 0
        gnorm0 = None

        def check(inner_self, status):
            gnorm = float(status.gnorm)
            if inner_self.gnorm0 is None and gnorm > 0.0:
                inner_self.gnorm0 = gnorm
            if hessian_shift > 0.0:
                # BNK's initial rule, ``pert = min(1e2, max(1e-4, 0.1 |g|))``
                # (bnk.c:570 with the defaults at 1264-1273). The value is set once per
                # outer iteration and stays fixed for every Hessian application inside
                # it, which is what BNK does. BNK also grows the shift tenfold on each
                # further failure; that part is not reproduced, so this is the mildest
                # form of the treatment.
                solver.rolobjective.shift = min(1e2, max(1e-4, hessian_shift * gnorm))
            recorder.mark_iteration(inner_self.iteration, rol_gnorm=gnorm,
                                    snorm=float(status.snorm))
            inner_self.iteration += 1
            if (gradient_rtol > 0.0 and inner_self.gnorm0
                    and gnorm <= gradient_rtol * inner_self.gnorm0):
                recorder.stop_reason = (
                    f"gradient fell to {gnorm / inner_self.gnorm0:.2e} of its "
                    f"starting value")
                return False
            if recorder.over_budget():
                recorder.stop_reason = (
                    f"work budget of {recorder.work_budget:.0f} forward-equivalents "
                    f"spent")
                return False
            return super().check(status)

    algorithm.setStatusTest(StatusTest(rol_params), False)

    with stop_annotating():
        algorithm.run(solver.rolvector, solver.rolobjective, solver.bounds)
    if not recorder.stop_reason:
        # ROL's own status test ended the run (its iteration limit or its own
        # tolerances). A complete run always carries a stop reason, so that a record
        # without one is known to be cut by a wall limit or a crash.
        recorder.stop_reason = "ROL status test"
    return recorder


# ---------------------------------------------------------------------------------
# TAO
# ---------------------------------------------------------------------------------

class InnerProductMatCtx:
    """PETSc Mat context applying the control inner-product operator, primal to dual.

    For a Function ``x`` in the control space the action is the Cofunction
    ``a(x, .)``. This is the forward operator whose inverse is the Riesz map, and it
    is what an LMVM matrix needs as ``J0`` (its initial Hessian approximation).

    Args:
        problem: an ``InverseProblem``, for the control and the inner product.
    """

    def __init__(self, problem):
        self.ip = problem.inner_product
        self.interface = PETScVecInterface((problem.control.control,))
        self.x = problem.control.control._ad_init_zero()
        self.v = TestFunction(problem.control.control.function_space())

    def mult(self, mat, x, y):
        self.interface.from_petsc(x, (self.x,))
        out = assemble(self.ip.form(self.x, self.v))
        self.interface.to_petsc(y, (out,))


def _set_lmvm_initial_hessian(tao, problem):
    """Give a BQNK solver's LMVM matrix the control inner product as ``J0``.

    pyadjoint sets the Riesz map as initial inverse Hessian for ``lmvm`` and ``blmvm``
    only. For the quasi-Newton-Krylov types the LMVM matrix is the Krylov
    preconditioner, and its initial guess is the identity in nodal coordinates unless
    this is done. The forward ``J0`` is the inner-product operator; its inverse is
    applied through a KSP that only calls the Riesz map.
    """
    from petsc4py import PETSc

    B = tao.getLMVMMat()
    n, N = B.getSizes()[0]
    comm = B.getComm()

    J0 = PETSc.Mat().createPython(((n, N), (n, N)), InnerProductMatCtx(problem), comm=comm)
    J0.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    J0.setUp()

    Minv = RieszMapMat(problem.reduced_functional.controls, comm=comm)

    class RieszPC:
        def apply(self, pc, x, y):
            Minv.mult(x, y)

    pc = PETSc.PC().createPython(RieszPC(), comm=comm)
    pc.setOperators(J0)
    pc.setUp()
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.setOperators(J0)
    ksp.setPC(pc)
    ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
    ksp.setUp()

    B.setLMVMJ0(J0)
    B.setLMVMJ0KSP(ksp)
    # The caller keeps these alive for the life of the solve; a petsc4py TAO object
    # cannot carry Python attributes.
    return (J0, Minv, pc, ksp)


# The Eisenstat-Walker settings of ``--krylov-ew``. ``tao_ksp_ew`` turns the forcing
# term on (``TaoKSPSetUseEW``). The two tolerances are read with the prefix of BNK's
# KSP, ``tao_bnk_``: the tolerance of the first solve and the upper cap of every later
# one. Both are 0.5, the coarse tolerance that hIPPYlib pairs with Gauss-Newton far
# from the solution. The rule between them is PETSc's version 2,
# ``rtol_k = (|b_k| / |b_{k-1}|)^1.618``, with the safeguard
# ``rtol_k >= rtol_{k-1}^1.618`` while that bound is above 0.1, where ``b`` is the
# right-hand side of the solve (the inactive gradient in nodal coordinates).
KRYLOV_EW_OPTIONS = {"tao_ksp_ew": True, "tao_bnk_ksp_ew_rtol0": 0.5,
                     "tao_bnk_ksp_ew_rtolmax": 0.5}

# The values of a PETSc boolean option that mean true. A flag given with no value
# (``--tao-opt tao_ksp_ew``) arrives as None and means true as well.
_PETSC_TRUE = (None, True, "true", "True", "TRUE", "1", "yes", "on")


def check_newton_options(params, variant, hessian="full", krylov_ew=False,
                         bnk_shift="identity"):
    """Refuse option sets that would run something other than what was asked for.

    A misspelt or overridden PETSc option is ignored by PETSc without an error, so a
    run asked to use the Gauss-Newton product or the Eisenstat-Walker tolerance could
    run with neither and still look valid. This function runs before the solver is
    built and raises instead.

    - ``hessian`` other than ``"full"`` needs a ``-riesz`` variant (the metric that
      preconditions the Krylov solve), BNK's shift at zero (``tao_bnk_imax`` equal to
      0) and no metric shift. A positive-semi-definite product never needs a shift,
      and a nodal shift ``pert * I`` is far larger than the Hessian of a CG1 control
      (``riesz_hessian.ShiftedHessianMat``).
    - ``krylov_ew`` needs ``tao_ksp_ew`` true and the two tolerances of
      ``KRYLOV_EW_OPTIONS`` at their values, also after ``--tao-opt`` has been merged.

    Args:
        params: the merged TAO options of ``tao_parameters``.
        variant: the optimiser variant.
        hessian: one of ``gauss_newton.HESSIAN_CHOICES``.
        krylov_ew: whether ``--krylov-ew`` was given.
        bnk_shift: ``"identity"`` or ``"metric"``.

    Raises:
        ValueError
    """
    if hessian not in HESSIAN_CHOICES:
        raise ValueError(f"hessian must be one of {HESSIAN_CHOICES}, got {hessian!r}")
    if hessian != "full":
        if not variant.endswith("-riesz"):
            raise ValueError(f"hessian={hessian!r} needs a tao-*-riesz variant, got "
                             f"{variant!r}")
        if bnk_shift == "metric":
            raise ValueError(f"hessian={hessian!r} needs the zero shift, not the metric "
                             f"shift")
        imax = params.get("tao_bnk_imax", "absent")
        try:
            imax_zero = imax is not None and float(imax) == 0.0
        except (TypeError, ValueError):
            imax_zero = False
        if not imax_zero:
            raise ValueError(f"hessian={hessian!r} needs tao_bnk_imax=0 (the zero shift), "
                             f"and the options hold tao_bnk_imax={imax!r}")
    if krylov_ew:
        if params.get("tao_ksp_ew", "absent") not in _PETSC_TRUE:
            raise ValueError(f"krylov_ew needs tao_ksp_ew true, and the options hold "
                             f"{params.get('tao_ksp_ew', 'absent')!r}")
        for key, value in KRYLOV_EW_OPTIONS.items():
            if key == "tao_ksp_ew":
                continue
            given = params.get(key, "absent")
            try:
                same = float(given) == value
            except (TypeError, ValueError):
                same = False
            if not same:
                raise ValueError(f"krylov_ew needs {key}={value}, and the options hold "
                                 f"{given!r}")


def tao_parameters(variant, max_it, initial_radius=1e-2, extra_options=None,
                   gradient_rtol=0.0, krylov_rtol=1e-2, krylov_ew=False,
                   tr_norm="nodal"):
    """TAO options dictionary for one variant.

    Args:
        variant: one of ``TAO_VARIANTS``.
        max_it: iteration limit.
        initial_radius: initial trust-region radius, and the radius is held to it
            instead of being estimated, so that the cost of the first iteration is
            comparable with ROL.
        extra_options: dict of PETSc options merged last.
        gradient_rtol: value for ``tao_gttol``, the ratio of the current convergence
            measure to its starting value. A relative test is used because the
            absolute one is not comparable between the TAO types: ``blmvm`` measures
            the gradient norm through ``setGradientNorm``, which is the control inner
            product, while the bounded Newton types report the Fischer-Burmeister
            residual of the bound-constrained conditions in the plain Euclidean norm
            (``bnk.c:69-72``). Those two differ by a mesh-dependent factor, near 150
            on a 150-cell mesh, so one ``tao_gatol`` would mean two different things.
        krylov_rtol: the fixed relative tolerance of BNK's Krylov solve,
            ``tao_bnk_ksp_rtol``. The default is 1e-2.
        krylov_ew: add ``KRYLOV_EW_OPTIONS``, so that PETSc sets the tolerance of
            each solve by the Eisenstat-Walker rule. The fixed tolerance stays in the
            options; the forcing term overwrites it before each solve, and a solve
            that still shows 1e-2 in the record shows that the rule did not act.
        tr_norm: ``"nodal"`` leaves STCG's trust-region norm at its default, the
            Euclidean norm of the nodal values, and sets no option. ``"metric"`` sets
            ``tao_bnk_ksp_cg_dtype preconditioned``, the norm of the preconditioner
            ``A^{-1}`` (the name that PETSc 3.26.0 knows; later releases accept it as
            a deprecated alias of ``ksp_cg_tr_norm_type``).

    Returns:
        dict of PETSc options.
    """
    if tr_norm not in ("nodal", "metric"):
        raise ValueError(f"tr_norm must be 'nodal' or 'metric', got {tr_norm!r}")
    _, tao_type, *flags = variant.split("-")
    params = {
        "tao_type": tao_type,
        "tao_max_it": max_it,
        "tao_gatol": 0.0,
        "tao_grtol": 0.0,
        "tao_gttol": gradient_rtol,
        "tao_monitor": None,
        "tao_converged_reason": None,
        "tao_trust0": initial_radius,
    }
    if tao_type.startswith("bn") or tao_type.startswith("bq"):
        # Newton and quasi-Newton-Krylov share the BNK options.
        params["tao_bnk_init_type"] = "constant"
        params["tao_bnk_ksp_max_it"] = 20
        params["tao_bnk_ksp_rtol"] = krylov_rtol
        if krylov_ew:
            params.update(KRYLOV_EW_OPTIONS)
        if tr_norm == "metric":
            params["tao_bnk_ksp_cg_dtype"] = "preconditioned"
    if "noprec" in flags:
        # The default BNK preconditioner is an LMVM matrix built from the gradient
        # history, with the identity as its initial guess. This variant removes it,
        # so the two measure what the quasi-Newton memory adds to Newton-Krylov.
        params["tao_bnk_pc_type"] = "none"
    if "riesz" in flags:
        # The Riesz map of the control inner product as the Krylov preconditioner.
        # PETSc builds the class by name, and it reads everything it needs from the
        # preconditioning matrix that ``run_tao`` passes to ``TAOSolver``.
        params["tao_bnk_pc_type"] = "python"
        params["tao_bnk_pc_python_type"] = "riesz_hessian.RieszPC"
    if extra_options:
        params.update(extra_options)
    return params


def run_tao(problem, variant, max_it=20, initial_radius=1e-2, extra_options=None,
            riesz_initial_hessian=True, output=None, metadata=None,
            gradient_rtol=0.0, work_budget=0.0, bnk_shift="identity",
            hessian="full", gn_switch_it=5, gn_switch_gtol=0.0, krylov_rtol=1e-2,
            krylov_ew=False, tr_norm="nodal", gn_expected_terms=None):
    """Run one TAO variant on the problem.

    Args:
        problem: an ``InverseProblem``.
        variant: one of ``TAO_VARIANTS``.
        max_it: iteration limit.
        initial_radius: initial trust-region radius.
        extra_options: dict of PETSc options merged over the defaults.
        riesz_initial_hessian: for the ``bqnk*`` types, set the control inner product
            as the LMVM initial Hessian. pyadjoint already does the equivalent for
            ``blmvm``.
        output: JSON file to rewrite after every iteration, or None.
        metadata: run settings written into every dump.
        bnk_shift: ``"identity"`` keeps BNK's Hessian shift ``pert * I`` in nodal
            coordinates, PETSc's own behaviour. ``"metric"`` installs
            ``riesz_hessian.ShiftedHessianMat`` as the Hessian, so that the same
            ``pert`` shifts by ``pert * A`` with ``A`` the metric of the control.
            Needs a ``-riesz`` variant, which supplies ``A``.
        hessian: the product of BNK's Krylov solve. ``"full"`` is pyadjoint's
            Hessian and installs nothing: the TAO object keeps pyadjoint's Hessian
            matrix. ``"gn"`` installs ``gauss_newton.GaussNewtonHessianMat``
            with the Gauss-Newton product for the whole run. ``"gn-then-full"``
            starts with Gauss-Newton and switches to the full Hessian at iteration
            ``gn_switch_it``, with the fall-back to Gauss-Newton after a full solve
            that stops on negative curvature. Both need a ``-riesz`` variant and the
            zero shift (``check_newton_options``).
        gn_switch_it: the iteration of the switch of ``gn-then-full``.
        gn_switch_gtol: switch also when TAO's gradient norm, the
            Fischer-Burmeister residual in nodal coordinates for BNK, falls to this
            fraction of its first value; 0 turns the test off.
        krylov_rtol: fixed relative tolerance of the Krylov solve.
        krylov_ew: Eisenstat-Walker tolerance, see ``tao_parameters``.
        tr_norm: the norm of the trust radius, see ``tao_parameters``.
        gn_expected_terms: the number of quadratic terms of the objective, which the
            Gauss-Newton operator checks at construction; None skips that check.

    Returns:
        Recorder
    """
    if bnk_shift not in ("identity", "metric"):
        raise ValueError(f"bnk_shift must be 'identity' or 'metric', got {bnk_shift!r}")
    if bnk_shift == "metric" and not variant.endswith("-riesz"):
        raise ValueError("bnk_shift='metric' needs a -riesz variant, whose "
                         "preconditioning matrix carries the metric")
    recorder = Recorder(problem, output=output, metadata=metadata or {},
                        work_budget=work_budget)
    recorder.attach()

    params = tao_parameters(variant, max_it, initial_radius, extra_options,
                            gradient_rtol=gradient_rtol, krylov_rtol=krylov_rtol,
                            krylov_ew=krylov_ew, tr_norm=tr_norm)
    check_newton_options(params, variant, hessian=hessian, krylov_ew=krylov_ew,
                         bnk_shift=bnk_shift)
    minimisation_problem = MinimizationProblem(problem.reduced_functional, bounds=problem.bounds)

    # The ``riesz`` variants precondition the Newton-Krylov solve with the control
    # inner product. That needs a separate preconditioning matrix, because the
    # Hessian matrix of pyadjoint carries no Riesz map and cannot be restricted to
    # the inactive set of the bounds. See ``riesz_hessian``.
    pmat = pmat_context = None
    if variant.endswith("-riesz"):
        pmat, pmat_context = build_preconditioner_matrix(problem)

    solver = TAOSolver(minimisation_problem, params, Pmat=pmat)
    tao = solver.tao

    # pyadjoint's Hessian context, read before a metric shift replaces the matrix,
    # so that the Krylov recorder can observe the Hessian actions in both modes.
    hessian_context = tao.getHessian()[0].getPythonContext()
    shifted_mat = shifted_context = None
    if bnk_shift == "metric":
        shifted_mat, shifted_context = build_shifted_hessian(tao, pmat_context)
    # The Gauss-Newton matrix replaces pyadjoint's as the Hessian of TAO. The Krylov
    # recorder then taps its ``mult``, so the curvature record of a negative-curvature
    # stop is that of the product that the solve used, in either mode.
    gn_mat = gn_context = None
    if hessian != "full":
        gn_mat, gn_context = build_gauss_newton_hessian(
            tao, problem, recorder=recorder, choice=hessian, switch_it=gn_switch_it,
            switch_gtol=gn_switch_gtol, expected_count=gn_expected_terms)
        hessian_context = gn_context

    def hessian_mode():
        """The product of the next Krylov solve: ``"gn"`` or ``"full"``.

        None for the quasi-Newton-Krylov types, whose Krylov operator is the LMVM
        matrix and no Hessian product at all.
        """
        if gn_context is not None:
            return gn_context.mode
        return "full" if tao.getType() in ("bnls", "bntr", "bntl") else None

    j0_refs = None
    if riesz_initial_hessian and tao.getType() in ("bqnktr", "bqnkls", "bqnktl", "bqnls"):
        j0_refs = _set_lmvm_initial_hessian(tao, problem)

    # The per-solve record of the Krylov solver, for the types that have one.
    krylov = None
    if tao.getType() in TAO_KRYLOV_TYPES:
        krylov = KrylovRecorder(tao, pmat_context=pmat_context,
                                comm=problem.Tic.function_space().mesh().comm,
                                wall_nodes=problem.wall_nodes,
                                hessian_context=hessian_context,
                                shifted_hessian=shifted_context,
                                hessian_mode=hessian_mode)
        krylov.attach()
    diagnostic_errors = []

    def diagnostics(tao_obj, xdiff, gnorm):
        """The entries of the run record that TAO does not report through petsc4py.

        Every entry only reads state, so the optimiser takes the same path with or
        without them. A failure here is stored and never raised, because an
        exception would reach PETSc as error 101 and end the run.

        Args:
            tao_obj: the TAO object.
            xdiff: the ``xdiff`` of ``getSolutionStatus``, the step length of the
                line search for the BNK types (``bnls.c:167``).
            gnorm: the ``gnorm`` of ``getSolutionStatus``, which for the BNK types is
                the Fischer-Burmeister residual norm that TAO tests.

        Returns:
            dict of new row entries.
        """
        extra = {
            # The step length of the last line search, under its own name. The
            # field ``radius`` holds the same number, and stays for the records
            # made before this field existed.
            "step_length": float(xdiff),
            # ``tao->trust`` after the update that follows the step; BNK gives this
            # radius to the next Krylov solve.
            "trust_radius": None,
            "gradient_norm_euclidean": None,
            "gradient_norm_metric": None,
            "gradient_norm_fischer_burmeister": None,
            "fischer_burmeister_rel_diff": None,
        }
        try:
            extra["trust_radius"] = PETSC_LIBRARY.trust_radius(tao_obj)
            norms = gradient_norms(tao_obj, recorder.last_dual_gradient,
                                   comm=problem.Tic.function_space().mesh().comm)
            extra["gradient_norm_euclidean"] = norms["euclidean"]
            extra["gradient_norm_metric"] = norms["metric"]
            extra["gradient_norm_fischer_burmeister"] = norms["fischer_burmeister"]
            if norms["fischer_burmeister"] is not None and gnorm:
                # The check that the recomputed residual is the one TAO tested. It
                # differs from zero by roundoff unless the last adjoint sweep did
                # not belong to the accepted iterate, for example after a failed
                # line search, where BNK restores the previous point.
                extra["fischer_burmeister_rel_diff"] = (
                    abs(norms["fischer_burmeister"] - gnorm) / abs(gnorm))
        except Exception as error:  # noqa: BLE001 -- see the docstring
            diagnostic_errors.append(f"iteration diagnostics: {type(error).__name__}: {error}")
        # The product that the solves of the next iteration will use. The solves of
        # this row carry their own ``hessian_product``, which is the record of what
        # each solve used.
        extra["hessian_product"] = hessian_mode()
        if krylov is not None:
            extra["krylov_solves"] = krylov.pop_solves()
        return extra

    def monitor(tao_obj):
        its = tao_obj.getIterationNumber()
        _, f, gnorm, cnorm, xdiff, reason = tao_obj.getSolutionStatus()
        recorder.mark_iteration(its, tao_gnorm=float(gnorm), radius=float(xdiff),
                                **diagnostics(tao_obj, xdiff, gnorm))
        if recorder.over_budget():
            recorder.stop_reason = (
                f"work budget of {recorder.work_budget:.0f} forward-equivalents spent")
            recorder.budget_hit = True
            raise WorkBudgetReached(recorder.stop_reason)

    tao.setMonitor(monitor)

    # ``TAOSolver.solve`` carries no annotation guard of its own.
    with stop_annotating():
        try:
            solver.solve()
        except WorkBudgetReached as e:
            # Kept for the case where petsc4py ever passes the exception through.
            log(f"TAO stopped: {e}")
        except PETSc.Error:
            # petsc4py reports a Python exception from a callback as error code 101.
            # The flag says whether that exception was the budget stop.
            if not recorder.budget_hit:
                raise
            log(f"TAO stopped: {recorder.stop_reason}")
        except TAOConvergenceError as e:
            # Hitting the iteration limit counts as divergence for TAO, and pyadjoint
            # raises on any non-positive reason. The history up to there is the result.
            log(f"TAO stopped: {e}")
    if not recorder.budget_hit:
        recorder.stop_reason = f"TAO converged reason {tao.getConvergedReason()}"
    # The output of ``-tao_view``, taken after the solve. It holds the counts of the
    # step types and of the Krylov termination reasons. It does not show the
    # Newton shift; each Krylov solve record holds that as ``hessian_shift``.
    view_path = None if output is None else f"{os.path.splitext(str(output))[0]}.tao_view.txt"
    try:
        recorder.final_diagnostics["tao_view"] = tao_view_text(tao, view_path)
    except Exception as error:  # noqa: BLE001 -- the view must never fail a run
        diagnostic_errors.append(f"tao_view: {type(error).__name__}: {error}")
    if gn_context is not None:
        recorder.final_diagnostics.update(gn_context.record())
        worst = recorder.final_diagnostics["gn_linear_defect_max"]
        if worst is not None and worst > LINEAR_DEFECT_TOLERANCE:
            diagnostic_errors.append(
                f"Gauss-Newton: sum_k w_k f_k differs from J by {worst:.3e} (relative) "
                f"at a Hessian update, above {LINEAR_DEFECT_TOLERANCE:g}")
    recorder.final_diagnostics["recording_errors"] = (
        diagnostic_errors + (krylov.errors if krylov is not None else []))
    if krylov is not None and krylov.solves:
        # Solves after the last monitor call, which can only happen when TAO stops
        # inside an iteration. They are kept, so that no solve is lost.
        recorder.final_diagnostics["krylov_solves_unassigned"] = krylov.pop_solves()
    # Both references were only kept alive for the solve; a petsc4py TAO cannot hold
    # Python attributes, so the Python side has to own them until here.
    del j0_refs, pmat, pmat_context, shifted_mat, shifted_context, gn_mat, gn_context
    return recorder


def run(problem, variant, **kwargs):
    """Dispatch on the variant name."""
    if variant in ROL_VARIANTS:
        return run_rol(problem, variant, **kwargs)
    if variant in TAO_VARIANTS:
        return run_tao(problem, variant, **kwargs)
    raise ValueError(f"unknown optimiser {variant!r}; choose from {VARIANTS}")
