"""Diagnostics of the TAO bounded Newton runs that the run record did not hold before.

The recorder of ``optimisers.py`` counts tape sweeps and records the twin-experiment
measures at every gradient. This module adds what a comparison of preconditioners and
priors needs on top of that, per Krylov solve and per outer iteration:

- the Krylov iteration count, the converged reason, and a flag for a count at the cap;
- the extreme Ritz values of the preconditioned Hessian of every solve;
- the trust radius, held apart from the step length of the line search;
- three norms of the projected gradient: Euclidean, metric and Fischer-Burmeister;
- the size of the active set, split into lower, upper and fixed bounds;
- the Hessian perturbation (shift) that BNK added before each Krylov solve;
- the text of ``-tao_view`` at the end of the run.

Every function here reads state. None of them writes into a vector that TAO, the
Krylov solver or the tape uses, so the optimiser takes the same path with or without
this module. The run of step S0 of the prior-covariance campaign tests that claim: two
runs, one with and one without this module, must agree in every field that both
records hold.

The code follows the PETSc 3.25.0 sources, which is the version that the Setonix
container runs. The functions and line references below are from
``src/tao/bound/impls/bnk/{bnk.c,bnls.c}``, ``src/ksp/ksp/impls/cg/stcg/stcg.c`` and
``src/tao/util/tao_util.c`` at the tag ``v3.25.0``.
"""

import ctypes
import os
import tempfile

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


# ---------------------------------------------------------------------------------
# State that petsc4py does not expose
# ---------------------------------------------------------------------------------

class _PetscLibrary:
    """Access to two PETSc getters that petsc4py 3.25 has no binding for.

    ``TaoGetCurrentTrustRegionRadius`` returns ``tao->trust``, the radius that BNK
    passes to the Krylov solver before each solve and updates after each step. The
    only value petsc4py gives for it is the step length of the line search, which
    ``TaoGetSolutionStatus`` returns as ``xdiff``; that is a different number.
    ``KSPCGGetNormD`` returns the norm of the step that the trust-region conjugate
    gradient solver computed, which tells whether the step reached the radius.

    Both are pure getters. They are reached through ``ctypes`` on the petsc4py
    extension module: ``dlsym`` on that handle also searches the libraries the
    module links, and ``libpetsc`` is one of them. If the symbol cannot be found the
    value is recorded as None, so a missing symbol never stops a run.
    """

    def __init__(self):
        self._tao_trust = None
        self._ksp_normd = None
        try:
            library = ctypes.CDLL(PETSc.__file__)
            self._tao_trust = library.TaoGetCurrentTrustRegionRadius
            self._tao_trust.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            self._tao_trust.restype = ctypes.c_int
            self._ksp_normd = library.KSPCGGetNormD
            self._ksp_normd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            self._ksp_normd.restype = ctypes.c_int
        except (OSError, AttributeError):
            # A build without these symbols leaves both getters at None.
            pass

    @staticmethod
    def _call(function, obj):
        """Call a getter of signature ``(object, PetscReal *)`` and return the value.

        Returns None when the getter is unavailable or returns a PETSc error code.
        PetscReal is a C double in every build this demo uses (real, double
        precision), which is what the ctypes signature states.
        """
        if function is None:
            return None
        value = ctypes.c_double(float("nan"))
        if function(ctypes.c_void_p(obj.handle), ctypes.byref(value)) != 0:
            return None
        return float(value.value)

    def trust_radius(self, tao):
        """``tao->trust`` of a TAO object, or None."""
        return self._call(self._tao_trust, tao)

    def step_norm(self, ksp):
        """Norm of the last step of a trust-region CG solver, or None.

        STCG measures this norm in the norm of its ``-ksp_cg_dtype`` option, which
        is the unpreconditioned (Euclidean nodal) norm by default (``stcg.c:640``).
        """
        return self._call(self._ksp_normd, ksp)


PETSC_LIBRARY = _PetscLibrary()


def _reason_names(enum_class):
    """Map the integer codes of a petsc4py reason enum to their names."""
    return {value: name for name, value in vars(enum_class).items()
            if not name.startswith("_") and isinstance(value, int)}


KSP_REASONS = _reason_names(PETSc.KSP.ConvergedReason)


# ---------------------------------------------------------------------------------
# Norms of the gradient
# ---------------------------------------------------------------------------------

def _fischer(a, b):
    """The Fischer-Burmeister function ``sqrt(a^2 + b^2) - a - b``, elementwise.

    This is the stable form of PETSc ``tao_util.c``: where ``a + b > 0`` the
    difference of two nearly equal numbers is replaced by the algebraically equal
    ``-2ab / (sqrt(a^2 + b^2) + a + b)``. The function is zero exactly when
    ``a >= 0``, ``b >= 0`` and ``ab = 0``, so it measures the violation of one
    complementarity condition.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    root = np.sqrt(a * a + b * b)
    out = np.empty_like(a)
    small = (a + b) <= 0
    out[small] = root[small] - (a[small] + b[small])
    big = ~small
    out[big] = -2.0 * a[big] * b[big] / (root[big] + (a[big] + b[big]))
    return out


def fischer_burmeister(x, g, lower, upper):
    """Fischer-Burmeister residual of the bound-constrained optimality conditions.

    The same vector as PETSc ``VecFischer(X, F, L, U, FB)``, which the bounded
    Newton types use as their convergence measure: ``bnls.c`` computes it from the
    solution and the UNPROJECTED gradient after every step, takes its Euclidean norm
    and passes that norm to ``TaoMonitor`` and to the convergence test. So the norm
    of this vector is the number ``-tao_gatol`` and ``-tao_gttol`` are tested
    against. Per entry, with ``inf`` meaning an absent bound:

    - no bound: ``-g``;
    - upper only: ``-phi(u - x, -g)``;
    - lower only: ``phi(x - l, g)``;
    - equal bounds: ``l - x``;
    - both: ``phi(x - l, phi(u - x, -g))``.

    Args:
        x: the control, an array of the local entries.
        g: the gradient in the dual space, nodal coordinates, local entries.
        lower: the lower bounds, local entries.
        upper: the upper bounds, local entries.

    Returns:
        numpy array of the local entries of the residual vector.
    """
    x, g = np.asarray(x, dtype=float), np.asarray(g, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    inf = PETSc.INFINITY
    no_lower = lower <= -inf
    no_upper = upper >= inf
    fb = np.empty_like(x)

    free = no_lower & no_upper
    fb[free] = -g[free]
    upper_only = no_lower & ~no_upper
    fb[upper_only] = -_fischer(upper[upper_only] - x[upper_only], -g[upper_only])
    lower_only = ~no_lower & no_upper
    fb[lower_only] = _fischer(x[lower_only] - lower[lower_only], g[lower_only])
    boxed = ~no_lower & ~no_upper
    fixed = boxed & (lower == upper)
    fb[fixed] = lower[fixed] - x[fixed]
    both = boxed & ~fixed
    inner = _fischer(upper[both] - x[both], -g[both])
    fb[both] = _fischer(x[both] - lower[both], inner)
    return fb


def _global_norm(local, comm):
    """Euclidean norm of a distributed vector from its local entries."""
    return float(np.sqrt(comm.allreduce(float(np.dot(local, local)), op=MPI.SUM)))


def _global_dot(a, b, comm):
    """Inner product of two distributed vectors from their local entries."""
    return float(comm.allreduce(float(np.dot(a, b)), op=MPI.SUM))


def gradient_norms(tao, unprojected_gradient, comm=MPI.COMM_WORLD):
    """The three gradient norms of item 4 of the run record, at the current iterate.

    Called from the TAO monitor. At that moment BNK has estimated the active set at
    the new iterate and stored the gradient with the active entries set to zero in
    ``tao->gradient`` (``bnls.c:148-152``); the next Krylov solve uses this same
    active set. The three numbers are:

    - ``euclidean``: the Euclidean norm of that projected gradient.
    - ``metric``: ``sqrt(g^T R g)`` of the same vector, with ``R`` the matrix that
      pyadjoint sets with ``TaoSetGradientNorm``, which is the Riesz map of the
      control (``tao_solver.py``, ``tao.setGradientNorm(Minv_mat)``). This is the
      computation of ``TaoGradientNorm`` (``taosolver.c:2910-2916``), so the value
      is BNK's own ``bnk->gnorm``: the number that scales the Hessian shift and the
      trust radius rules, and that petsc4py cannot read.
    - ``fischer_burmeister``: the norm of ``fischer_burmeister`` above, from the
      unprojected gradient. This is the convergence measure of BNK.

    Args:
        tao: the TAO object.
        unprojected_gradient: the local entries of the full gradient at the current
            iterate in TAO's vector layout, or None. BNK keeps this vector private,
            so the caller passes the gradient from the last adjoint sweep.
        comm: communicator of the control.

    Returns:
        dict with ``euclidean``, ``metric`` and ``fischer_burmeister``. A value that
        cannot be computed is None.
    """
    projected, _ = tao.getGradient()
    out = {"euclidean": float(projected.norm(PETSc.NormType.NORM_2)),
           "metric": None, "fischer_burmeister": None}

    metric_matrix = tao.getGradientNorm()
    if metric_matrix is not None and metric_matrix.handle:
        # A work vector of our own, so that ``tao->gradient_norm_tmp`` is untouched.
        work = projected.duplicate()
        metric_matrix.mult(projected, work)
        out["metric"] = float(np.sqrt(projected.dot(work)))
        work.destroy()

    if unprojected_gradient is not None:
        x = tao.getSolution().getArray(readonly=True)
        lower, upper = tao.getVariableBounds()
        fb = fischer_burmeister(x, unprojected_gradient,
                                lower.getArray(readonly=True),
                                upper.getArray(readonly=True))
        out["fischer_burmeister"] = _global_norm(fb, comm)
    return out


# ---------------------------------------------------------------------------------
# Ritz values from the conjugate gradient coefficients
# ---------------------------------------------------------------------------------

def lanczos_ritz_values(rz, z, d, comm=MPI.COMM_WORLD):
    """Ritz values of the preconditioned operator from one conjugate gradient solve.

    Conjugate gradient builds the Lanczos tridiagonal matrix of the preconditioned
    operator ``M^{-1} Q`` as a by-product. With step lengths ``alpha_j = r_j^T z_j /
    p_j^T Q p_j`` and ``beta_j = r_{j+1}^T z_{j+1} / r_j^T z_j``, the matrix after
    ``k`` steps is

        T[0, 0]     = 1 / alpha_0
        T[j, j]     = 1 / alpha_j + beta_{j-1} / alpha_{j-1}
        T[j, j + 1] = T[j + 1, j] = sqrt(beta_j) / alpha_j

    and its eigenvalues are the Ritz values of ``M^{-1} Q`` on the Krylov space.
    The extreme ones converge first, to the extreme eigenvalues. For a
    preconditioner equal to the Hessian they all sit at 1.

    PETSc can return these values for ``KSPCG`` but not for ``KSPSTCG``, the
    trust-region solver that BNK uses: STCG has no ``computeextremesingularvalues``
    method in 3.25.0, so ``KSPComputeExtremeSingularValues`` returns -1 for both.
    So the coefficients are rebuilt here from what STCG exposes:

    - ``r_j^T z_j`` and ``z_j`` from the preconditioner, which STCG applies to each
      residual ``r_j`` exactly once, in order (``stcg.c:139`` and ``:358``);
    - the iterates ``d_j`` from the monitor, which STCG calls after each update of
      ``d`` (``stcg.c:171`` and ``:417``), with ``d_0 = 0``.

    The search directions follow from ``p_0 = z_0`` and ``p_j = z_j +
    beta_{j-1} p_{j-1}``, the same recursion as ``stcg.c:459``, and
    ``alpha_j = (d_{j+1} - d_j)^T p_j / p_j^T p_j`` because ``d_{j+1} = d_j +
    alpha_j p_j``. The operator ``Q`` is the shifted Hessian on the inactive set,
    so the Ritz values include the Newton shift of BNK.

    A step that STCG cuts at the trust radius moves ``d`` by a shorter length and
    is not followed by a monitor call, and a step of negative curvature has no
    positive ``alpha``. Neither gives a coefficient, so the matrix is built from the
    completed steps alone.

    How to read the result. Conjugate gradient in floating point loses the
    orthogonality of its Lanczos vectors once a Ritz value has converged, and the
    converged value then appears again as a "ghost" copy. In the 20-step solve of
    the S0 run D2, 981038 appears three times and 23597 twice. Therefore:

    - the list is not a count of clustered eigenvalues, and repeated entries must
      not be read as multiplicity;
    - ``ritz_max`` is a lower bound on the largest eigenvalue of ``M^{-1} Q`` and
      ``ritz_min`` is an upper bound on the smallest, because Ritz values lie
      inside the spectrum;
    - so ``ritz_max / ritz_min`` is a lower bound on the condition number of the
      preconditioned operator, never an estimate of it from above.

    Args:
        rz: list of ``r_j^T z_j``, one per preconditioner application.
        z: list of the local entries of ``z_j``, same length as ``rz``.
        d: list of the local entries of ``d_j`` from the monitor, ``d[0] = 0``.
        comm: communicator of the vectors.

    Returns:
        numpy array of the Ritz values in ascending order, empty when no step
        completed.
    """
    steps = min(len(d) - 1, len(z))
    if steps <= 0:
        return np.array([])
    alpha = []
    beta = []
    p = None
    for j in range(steps):
        if j == 0:
            p = np.array(z[0], dtype=float)
        else:
            b = rz[j] / rz[j - 1]
            beta.append(b)
            p = z[j] + b * p
        pp = _global_dot(p, p, comm)
        if pp <= 0.0:
            break
        alpha.append(_global_dot(d[j + 1] - d[j], p, comm) / pp)
    k = len(alpha)
    if k == 0 or min(alpha) <= 0.0:
        # A non-positive step length means the coefficients are not those of a
        # positive definite Lanczos process; report nothing instead of a wrong value.
        return np.array([])
    T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1.0 / alpha[j] + (beta[j - 1] / alpha[j - 1] if j > 0 else 0.0)
        if j + 1 < k:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(beta[j]) / alpha[j]
    return np.linalg.eigvalsh(T)


# ---------------------------------------------------------------------------------
# Per-solve recording of the Krylov solver of BNK
# ---------------------------------------------------------------------------------

class KrylovRecorder:
    """Records every Krylov solve of a TAO bounded Newton run.

    It installs three read-only hooks on the KSP of the TAO object: a pre-solve and
    a post-solve callback, which bracket one ``KSPSolve``, and a monitor, which STCG
    calls after each update of the iterate. When the preconditioning matrix comes
    from ``riesz_hessian.py``, it also registers as the observer of the Riesz
    preconditioner, which hands it each residual and preconditioned residual, and
    as the observer of the ``MatShift`` calls on that matrix.

    The Hessian perturbation. BNK adds ``pert * I`` to the reduced Hessian before a
    Krylov solve when ``pert > 0`` (``TaoBNKComputeStep``, ``bnk.c:414-421`` at
    v3.25.0), and it shifts the preconditioning matrix by the same ``pert`` in the
    same place. ``riesz_hessian.py`` passes that value here. Each solve record then
    holds ``hessian_shift``: the exact ``pert`` of BNK for that solve, or 0.0 when
    BNK made no shift call, which in BNK means ``pert`` was zero. The value changes
    between solves by the rules of ``TaoBNKSafeguardStep`` and
    ``TaoBNKPerformLineSearch``; ``prior_campaign/summarise_shift.py`` names the
    rule that each change followed. The recorded value is in the nodal (Euclidean)
    units of the Hessian matrix, not in the metric of the control.

    BNK can call ``KSPSolve`` more than once in one outer iteration, so the record
    is per solve. ``pop_solves`` hands the solves since the last call to the TAO
    monitor, which stores them in the row of that iteration.

    A failure inside a hook is caught and stored in ``errors``, and the recorder
    then stops recording. An exception that left a hook would reach PETSc as error
    101 and end the run, and a diagnostic must never end a run.

    Args:
        tao: the TAO object, after ``setUp``.
        pmat_context: the ``InnerProductMat`` of ``riesz_hessian.py`` that the run
            passed as the preconditioning matrix, or None.
        comm: communicator of the control.
        wall_nodes: ``{"bottom": array, "top": array}`` of local indices of the
            wall nodes in the layout of the control vector, or None. With it, each
            solve also records which wall nodes are in the active set, so that the
            entries into and exits from the active set can be counted.

    Attributes:
        solves: the solve records not yet handed out by ``pop_solves``.
        errors: messages of failures inside the hooks.
    """

    def __init__(self, tao, pmat_context=None, comm=MPI.COMM_WORLD, wall_nodes=None):
        self.tao = tao
        self.wall_nodes = wall_nodes or {}
        self.ksp = tao.getKSP()
        self.pmat_context = pmat_context
        self.comm = comm
        self.solves = []
        self.errors = []
        self.enabled = True
        self._current = None
        self._solve_index = 0
        # The values of the MatShift calls since the last solve opened. BNK makes
        # at most one call per solve, just before ``KSPSolve``, so the list holds
        # zero or one entry when the next solve opens.
        self._pending_shifts = []

    def attach(self):
        """Install the hooks. Call once, after ``TAOSolver`` has set TAO up."""
        self.ksp.setPreSolve(self._guard(self._pre_solve))
        self.ksp.setPostSolve(self._guard(self._post_solve))
        self.ksp.setMonitor(self._guard(self._monitor))
        if self.pmat_context is not None:
            self.pmat_context.observer = self._guard(self._observe_preconditioner)
            self.pmat_context.shift_observer = self._guard(self._observe_shift)

    def _guard(self, function):
        """Wrap a hook so that a failure disables the recorder and never propagates."""
        def hook(*args, **kwargs):
            if not self.enabled:
                return None
            try:
                return function(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 -- see the class docstring
                self.enabled = False
                self.errors.append(f"{function.__name__}: {type(error).__name__}: {error}")
                return None
        return hook

    # -- hooks ---------------------------------------------------------------------

    def _pre_solve(self, ksp, rhs, x):
        """Open a solve record: the radius, the size and split of the active set."""
        operator, pmat = ksp.getOperators()
        n_inactive = operator.getSize()[0]
        n_total = self.tao.getSolution().getSize()
        record = {
            "solve": self._solve_index,
            # The radius BNK has just given STCG with ``KSPCGSetRadius``
            # (``bnk.c:441``), in the norm of ``-ksp_cg_dtype``.
            "trust_radius": PETSC_LIBRARY.trust_radius(self.tao),
            "max_it": int(ksp.getTolerances()[3]),
            "inactive": int(n_inactive),
            "active": int(n_total - n_inactive),
        }
        record.update(self._active_split(pmat, n_total))
        # BNK's perturbation for this solve. The shift call came before KSPSolve,
        # so it is already in the pending list. No call means pert was zero
        # (``bnk.c:414`` shifts only when ``pert > 0``). The call count is kept so
        # that a second call in one solve, which BNLS never makes, would show.
        shifts, self._pending_shifts = self._pending_shifts, []
        record["hessian_shift"] = float(sum(shifts)) if shifts else 0.0
        record["hessian_shift_calls"] = len(shifts)
        self._current = {"record": record, "rz": [], "z": [], "d": [], "rnorm": []}
        self._solve_index += 1

    def _observe_shift(self, alpha):
        """Store the value of one ``MatShift`` on the preconditioning matrix."""
        self._pending_shifts.append(float(alpha))

    def _observe_preconditioner(self, r, z):
        """Store ``r^T z`` and ``z`` of one preconditioner application in a solve."""
        if self._current is None:
            # An application outside a KSP solve belongs to no Lanczos process.
            return
        r_local = r.getArray(readonly=True)
        z_local = np.array(z.getArray(readonly=True), dtype=float)
        self._current["rz"].append(_global_dot(r_local, z_local, self.comm))
        self._current["z"].append(z_local)

    def _monitor(self, ksp, its, rnorm):
        """Store the residual norm and a copy of the iterate after each update."""
        if self._current is None:
            return
        self._current["rnorm"].append(float(rnorm))
        self._current["d"].append(np.array(ksp.getSolution().getArray(readonly=True),
                                           dtype=float))

    def _post_solve(self, ksp, rhs, x):
        """Close the solve record: count, reason, step norm and Ritz values."""
        current, self._current = self._current, None
        if current is None:
            return
        record = current["record"]
        iterations = int(ksp.getIterationNumber())
        reason = int(ksp.getConvergedReason())
        record.update({
            # STCG counts one iteration per application of the operator, which is
            # one Hessian-vector product, including the application of a step that
            # is then cut at the trust radius (``stcg.c:181-182`` and ``:500-502``).
            "iterations": iterations,
            "reason": reason,
            "reason_name": KSP_REASONS.get(reason, str(reason)),
            "at_cap": bool(iterations >= record["max_it"]
                           or reason == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT),
            "step_norm": PETSC_LIBRARY.step_norm(ksp),
            "residual_norms": current["rnorm"],
        })
        ritz = (lanczos_ritz_values(current["rz"], current["z"], current["d"], self.comm)
                if current["z"] else np.array([]))
        record["ritz_values"] = [float(v) for v in ritz]
        record["ritz_min"] = float(ritz[0]) if ritz.size else None
        record["ritz_max"] = float(ritz[-1]) if ritz.size else None
        self.solves.append(record)

    # -- helpers -------------------------------------------------------------------

    def _active_split(self, pmat, n_total):
        """Split the active set of this solve into lower, upper and fixed bounds.

        BNK keeps its three index sets private. The inactive set reaches Python as
        the index set of ``createSubMatrix`` on the preconditioning matrix of
        ``riesz_hessian.py``, and its complement is the active set. Each active
        entry is then named by the nearer bound: fixed when the two bounds are
        equal, lower when ``x - l <= u - x``, upper otherwise. BNK's Bertsekas
        estimate (``TaoEstimateActiveBounds``) calls an entry lower-active only
        when it lies within its small tolerance of the lower bound, so for bounds
        far apart, as ``[0, 1]`` here, the nearer bound always gives BNK's split.

        Returns:
            dict with ``active_lower``, ``active_upper`` and ``active_fixed``; all
            None when the preconditioning matrix is not from ``riesz_hessian.py``.
        """
        split = {"active_lower": None, "active_upper": None, "active_fixed": None}
        for name in self.wall_nodes:
            split[f"active_wall_{name}"] = None
        context = pmat.getPythonContext() if pmat.getType() == PETSc.Mat.Type.PYTHON else None
        if context is None:
            return split
        iset = getattr(context, "iset", None)
        x = self.tao.getSolution()
        lower, upper = self.tao.getVariableBounds()
        start, end = x.getOwnershipRange()
        active = np.ones(end - start, dtype=bool)
        if iset is not None:
            inactive = iset.getIndices() - start
            active[inactive] = False
        else:
            # The full matrix: BNK found no active entry and solves on every dof.
            active[:] = False
        xa = x.getArray(readonly=True)[active]
        la = lower.getArray(readonly=True)[active]
        ua = upper.getArray(readonly=True)[active]
        fixed = la == ua
        near_lower = ~fixed & ((xa - la) <= (ua - xa))
        near_upper = ~fixed & ~near_lower
        for key, mask in (("active_lower", near_lower), ("active_upper", near_upper),
                          ("active_fixed", fixed)):
            split[key] = int(self.comm.allreduce(int(mask.sum()), op=MPI.SUM))
        # The wall nodes in the active set, as global indices of the control
        # vector, sorted. One wall has 65 nodes at 64 cells, so the list is short,
        # and two consecutive lists give the nodes that entered and left the set.
        for name, nodes in self.wall_nodes.items():
            local = [int(i) + start for i in nodes if active[i]]
            gathered = self.comm.allgather(local)
            split[f"active_wall_{name}"] = sorted(i for part in gathered for i in part)
        return split

    def pop_solves(self):
        """Return the solve records since the last call, and forget them."""
        solves, self.solves = self.solves, []
        return solves


# ---------------------------------------------------------------------------------
# The output of -tao_view
# ---------------------------------------------------------------------------------

def tao_view_text(tao, path=None):
    """The text that ``-tao_view`` prints, as a string.

    The view holds what petsc4py cannot read: the counts of Newton, BFGS, scaled
    gradient and gradient steps, the counts of the Krylov termination reasons, the
    trust radius, and the settings of the KSP and the line search.

    Args:
        tao: the TAO object.
        path: file to keep the text in, or None for a temporary file.

    Returns:
        str on rank 0, None on the other ranks or on failure.
    """
    comm = tao.getComm().tompi4py()
    keep = path is not None
    if not keep:
        name = None
        if comm.rank == 0:
            handle, name = tempfile.mkstemp(suffix=".tao_view.txt")
            os.close(handle)
        path = comm.bcast(name, root=0)
    viewer = PETSc.Viewer().createASCII(str(path), comm=tao.getComm())
    tao.view(viewer)
    viewer.destroy()
    text = None
    if comm.rank == 0:
        with open(path) as f:
            text = f.read()
        if not keep:
            os.remove(path)
    return text
