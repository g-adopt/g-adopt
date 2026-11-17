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
    """

    def __init__(self, histories, dt, order=2):
        self.dt = dt
        self.order = order
        self.current = [m.copy(deepcopy=True) for m in histories]
        self.previous = [m.copy(deepcopy=True) for m in histories]
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
        return [self.beta*m + self.gamma*p
                for m, p in zip(self.current, self.previous)]

    def accept(self, histories):
        for previous, current, updated in zip(self.previous, self.current, histories):
            previous.assign(current)
            current.assign(updated)
        self.previous_dt = self.prepared_dt
        self.prepared_dt = None

    def restart(self):
        """Use BE on the next step, e.g. after a resolved loading jump."""
        self.previous_dt = None
