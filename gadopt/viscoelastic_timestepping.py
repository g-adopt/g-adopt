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


def time_history_fields(solver):
    """Named accepted history levels needed by an application checkpoint."""
    history = getattr(solver, "time_integrator", None)
    if history is None:
        return []
    return [(f"time_{level}_{i}", field)
            for level in ("current", "previous")
            for i, field in enumerate(getattr(history, level))]


def time_history_metadata(solver):
    """Numeric metadata, suitable for CheckpointFile timestepping info."""
    history = getattr(solver, "time_integrator", None)
    order = 2 if getattr(solver, "time_scheme", "backward_euler") == "bdf2" else 1
    previous = None if history is None else history.previous_dt
    values = {"time_scheme_order": float(order), "time_history_version": 1.,
              "time_previous_dt": -1. if previous is None else previous}
    if getattr(solver, "constitutive_method", None) == "direct":
        values.update(material_solver_kind=1., material_quad_degree=float(solver.quad_degree))
    return values


def restore_time_history_metadata(solver, metadata):
    """Validate a restart's scheme and restore its accepted step size.

    Validate before loading coefficient fields. Legacy metadata is BE-only;
    it must never silently start a BDF2 continuation with missing histories.
    """
    requested = time_history_metadata(solver)
    for key in ("material_solver_kind", "material_quad_degree"):
        if metadata.get(key, 0.) != requested.get(key, 0.):
            raise ValueError(f"Checkpoint {key} does not match the requested material solver")
    expected = requested["time_scheme_order"]
    written = metadata.get("time_scheme_order", 1)
    if written != expected:
        raise ValueError(f"Checkpoint time order {written} does not match requested order {expected}")
    history = getattr(solver, "time_integrator", None)
    if history is None:
        return
    if metadata.get("time_history_version") != 1 or "time_previous_dt" not in metadata:
        raise ValueError("Checkpoint is missing the accepted time-history metadata")
    previous = float(metadata["time_previous_dt"])
    if previous != -1. and (not math.isfinite(previous) or previous <= 0):
        raise ValueError("Checkpoint previous timestep must be positive or the startup sentinel")
    history.previous_dt = None if previous == -1. else previous
    history.prepared_dt = None


def add_material_solver_arguments(parser):
    """Shared driver selection; the existing coupled formulation stays default."""
    def positive_degree(value):
        degree = int(value)
        if degree < 1:
            raise ValueError("quadrature degree must be positive")
        return degree

    parser.add_argument("--material_solver", choices=("coupled", "eliminated"),
                        default="coupled", help="Global DG histories or local quadrature elimination")
    parser.add_argument("--material_quad_degree", type=positive_degree, default=4,
                        help="Mechanics/history quadrature degree for elimination (default 4)")


def material_solver_options(method, quad_degree, dg_degree):
    """Matching space and equation options for the two material formulations."""
    if method not in ("coupled", "eliminated"):
        raise ValueError("material_solver must be coupled or eliminated")
    if not isinstance(quad_degree, int) or quad_degree < 1:
        raise ValueError("material_quad_degree must be a positive integer")
    if method == "coupled":
        return {"internal_variable_degree": dg_degree}, {}
    return ({"condense_internal_variables": True,
             "internal_variable_family": "Quadrature",
             "internal_variable_degree": quad_degree},
            {"constitutive_method": "direct", "quad_degree": quad_degree})
