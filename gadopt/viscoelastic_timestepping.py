"""Recorded BE/variable-BDF2 coefficients and accepted constitutive history.

Algebraic GIA fields are always evaluated at the new time. Only material
histories receive a time derivative. A prescribed schedule is recorded;
this module does not implement adaptive error control or step rejection.
"""
import math

import firedrake as fd


class BDF2History:
    """Two accepted levels, independent of the solver's mixed solution copies.

    ``prepare`` precedes a solve; ``accept`` follows a successful solve and
    any nullspace projection. Function assignments are recorded by pyadjoint.
    Python counters are used only during the original forward march.

    Args:
      histories: the history Functions at the start of the march.
      dt: the time step, a number, a `Constant` or a `Real` Function.
      order: 2 for variable-step BDF2 with a backward-Euler first step, 1 for
        backward Euler throughout.
      own_current: `False` (the default) keeps private copies of `histories`
        as the accepted level. `True` makes the given Functions the accepted
        level itself, so that a solver can publish one history Function that
        is also the state the next step reads, with no second copy to keep in
        step.
      keep_previous: whether to store the level before the accepted one.
        `None` (the default) stores it always. A solver that knows it runs
        backward Euler passes `order == 2`: at order 1 the previous level has
        coefficient zero, and one level of a DG1 history is the largest field
        of a GIA run.
    """

    def __init__(self, histories, dt, order=2, *, own_current=False,
                 keep_previous=None):
        self.dt = dt
        self.order = order
        self.current = (list(histories) if own_current
                        else [m.copy(deepcopy=True) for m in histories])
        if keep_previous is None:
            keep_previous = True
        if keep_previous:
            self.previous = [m.copy(deepcopy=True) for m in histories]
        elif order == 2:
            raise ValueError("BDF2 needs the previous history level.")
        else:
            self.previous = None
        R = fd.FunctionSpace(histories[0].function_space().mesh(), "R", 0)
        self.alpha = fd.Function(R, name="bdf_alpha").assign(1)
        self.beta = fd.Function(R, name="bdf_beta").assign(1)
        self.gamma = fd.Function(R, name="bdf_gamma").assign(0)
        self.previous_dt = None
        self.prepared_dt = None

    def prepare(self):
        h = float(self.dt)
        if not math.isfinite(h) or h <= 0:
            raise ValueError("BDF2 dt must be finite and positive")
        if self.order == 1 or self.previous_dt is None:
            alpha, beta, gamma = 1., 1., 0.
        else:
            r = h / self.previous_dt
            if r >= 1 + math.sqrt(2):
                raise ValueError("BDF2 requires dt_new/dt_old < 1 + sqrt(2)")
            alpha, beta, gamma = (1 + 2*r)/(1 + r), 1 + r, -r*r/(1 + r)
        self.alpha.assign(alpha)
        self.beta.assign(beta)
        self.gamma.assign(gamma)
        self.prepared_dt = h

    def combinations(self):
        """The history the next step relaxes from, one expression per level.

        `beta m_n + gamma m_{n-1}`, the BDF2 combination, which is `m_n` on a
        backward-Euler step. Without a stored previous level the accepted
        level is returned itself.
        """
        if self.previous is None:
            return list(self.current)
        return [self.beta*m + self.gamma*p
                for m, p in zip(self.current, self.previous)]

    def accept(self, histories, output_of=None):
        """Accept `histories` as the new level, after a successful solve.

        With BDF2 the accepted level first moves to the previous level by a
        recorded assignment. Its saved value on the tape refers to the saved
        value of the accepted level (`DelegatedFunctionCheckpoint`), so it
        costs no copy.

        Args:
          histories: the new value of every history Function.
          output_of: the tape block whose map produced `histories`, or None.
            With a block, the new accepted level is written without
            annotation, and its new block variable becomes an output of that
            block, in the order of `histories`. Firedrake's solve wrapper
            makes the unknown of a solve an output of its solve block in the
            same way. The tape then holds one block variable of the new
            history per step, and no assign block between the block's own
            history buffer and the accepted level. The block must return the
            value of this output in a replay and take its adjoint seed and
            its tangent (`gadopt.constitutive_integrators.
            ConstitutiveStepBlock` does). Without a block, the new level is
            assigned, which the tape records as a block of its own when it
            annotates.
        """
        previous_levels = self.previous or [None]*len(self.current)
        for previous, current, updated in zip(previous_levels, self.current, histories):
            if previous is not None:
                previous.assign(current)
            if output_of is None:
                current.assign(updated)
            else:
                current.assign(updated, annotate=False)
                output_of.add_output(current.create_block_variable())
        self.previous_dt = self.prepared_dt
        self.prepared_dt = None

    def restart(self):
        """Use BE on the next step, e.g. after a resolved loading jump."""
        self.previous_dt = None


def assign_timestep(solver, dt, h):
    """Set a prescribed driver step, restarting BE at a large BDF2 increase.

    A finite number of BE startup steps is compatible with second-order
    global convergence. This keeps a driver's prescribed output/forcing grid
    intact rather than inserting extra load evaluations. The library solver
    itself continues to reject excessive BDF2 ratios unless explicitly reset.
    Return whether a restart was needed, for driver logging.
    """
    h = float(h)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Timestep must be finite and positive")
    history = getattr(solver, "time_integrator", None)
    restarted = (history is not None and history.order == 2
                 and history.previous_dt is not None
                 and h/history.previous_dt >= 1 + math.sqrt(2))
    if restarted:
        solver.restart_time_integration()
    dt.assign(h)
    return restarted
