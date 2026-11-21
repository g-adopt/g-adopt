"""Driver of the `time_convergence` case: the order in time of the GIA schemes.

Usage: `python gia_time_convergence.py --entry <name>`, with the entry names
of `gia_entries.TIME_CONVERGENCE` (`maxwell`, `power`).

An entry marches the configuration `SEA` of `gia_march.py` (with the
entry's changes) from rest to `TC_END_TIME` under an ice load that is
applied at `t = 0` and kept constant. It marches first a reference, BDF2
with `TC_REFERENCE_STEPS` equal steps, then each scheme of `TC_SCHEMES`
(backward Euler, BDF2 with equal steps, BDF2 with alternating steps) at
each step count of `TC_LEVELS`. All marches use the same meshes, so the
fields of two marches are in the same spaces and their difference is
exact.

For each scheme and level the driver writes the errors at the end time
against the reference: of the displacement and the potential,
`errornorm(reference, field) / norm(reference)`, and of the sea-level
shift, `|shift - reference| / |reference|`. With `e(n)` the error at `n`
steps, the order between `n` and `2n` steps is `log2(e(n) / e(2n))`, which
tends to `p` for a scheme of order `p` as the step goes to zero; the test
`test_gia_time_convergence.py` computes the orders and asserts them. This
script asserts nothing. The output is `name value` lines to
`out/time_convergence-<entry>.dat`, with the names of
`gia_entries.tc_key`.

The ice load `.002 (1 - cos(theta - .3)) / 2 (1 + .5 cos 2 theta)` has
degree 0, 1 and 2 parts, so the shift, the centre of mass and the rotation
all move, and it is not symmetric about an axis of the mesh.
"""

import argparse
import time
from dataclasses import replace

import gadopt  # imported before firedrake
import firedrake as fd

import gia_march as M
import gia_cases as C
from gia_entries import (CASE_DIR, TC_ALTERNATION, TC_END_TIME, TC_LEVELS,
                         TC_REFERENCE_STEPS, TC_SCHEMES, TIME_CONVERGENCE, tc_key,
                         time_convergence_values)


def set_ice(sea):
    """Set the ice thickness of the case, constant in time.

    A cap centred at `theta = pi + .3` (off the axes of the mesh) whose
    thickness is modulated by `1 + .5 cos 2 theta`, so that the load has a
    second moment. Its largest thickness is about 0.003, the size of the
    largest ice of the forward scenario, for which Newton converges from
    rest at the step 0.08.

    Args:
      sea: the `gia_cases.SeaFields` of the run.
    """
    X = fd.SpatialCoordinate(sea.I.function_space().mesh())
    theta = fd.atan2(X[1], X[0])
    sea.I.interpolate(.002 * (1 - fd.cos(theta - .3)) / 2 * (1 + .5 * fd.cos(2 * theta)))


def keep_ice(sea, k):
    """The ice history of `gia_march.advance` for a load constant in time: no change."""


def step_lengths(pattern, n):
    """The `n` step lengths of a march to `TC_END_TIME`.

    Args:
      pattern: `"equal"` for `n` steps of `TC_END_TIME / n`; `"alternating"`
        for the factors `TC_ALTERNATION` of that mean step in turn, starting
        with the shorter step. `n` must be even for `"alternating"`, so
        that the steps end at `TC_END_TIME`.
      n: the step count.

    Returns:
      A list of `n` floats.
    """
    mean = TC_END_TIME / n
    if pattern == "equal":
        return [mean] * n
    assert pattern == "alternating" and n % 2 == 0, (pattern, n)
    return [factor * mean for _ in range(n // 2) for factor in TC_ALTERNATION]


def march(meshes, case, steps, parameters):
    """Build the solver of `case` and march it from rest over `steps`.

    The direct preset with the tight Newton tolerances of
    `gia_cases.build_new` (`snes_rtol` 1e-11), so that the error of the
    nonlinear solve is far below the smallest time error measured.

    Args:
      meshes: the parent annulus and its mantle submesh.
      case: the configuration, with the time scheme of the march.
      steps: the step lengths.
      parameters: replaced parameters of `gia_cases.build_new`, or `None`.

    Returns:
      A dictionary: `u`, `psi` (copies of the fields at the end), `shift`,
      `time` (the sum of the steps, the end time reached), `newton_max`
      (the largest Newton count of a step) and `min_power_factor` (the
      smallest power-law factor at the end; 1 for a Newtonian rheology).
    """
    run = M.build(meshes, case, preset="direct", parameters=parameters)
    set_ice(run.sea)
    newton = []

    def on_step(k, run):
        newton.append(run.solver.appctx["gia_last_solve_newton_its"])

    M.advance(run, steps, on_step=on_step, ice=keep_ice)
    return {"u": run.u.copy(deepcopy=True), "psi": run.psi.copy(deepcopy=True),
            "shift": M._real(run.solver, "sea_level_shift"), "time": run.time,
            "newton_max": max(newton),
            "min_power_factor": M._min_power_factor(run.solver)}


def run_entry(name):
    """March the reference and every scheme and level of entry `name`; write the errors."""
    entry = TIME_CONVERGENCE[name]
    base = replace(M.SEA, **entry["changes"])
    parameters = entry.get("parameters")
    meshes = C.annulus_meshes()
    start = time.time()
    out = {}
    # The reference: BDF2 on equal steps, 4 times finer than the finest
    # level (see `gia_entries.TC_REFERENCE_STEPS` for its effect on the
    # measured orders).
    ref = march(meshes, replace(base, scheme="bdf2"),
                step_lengths("equal", TC_REFERENCE_STEPS), parameters)
    u_norm, psi_norm = fd.norm(ref["u"]), fd.norm(ref["psi"])
    key = lambda quantity: tc_key("reference", TC_REFERENCE_STEPS, quantity)  # noqa: E731
    out.update({key("u_norm"): u_norm, key("psi_norm"): psi_norm,
                key("shift_value"): ref["shift"], key("time"): ref["time"],
                key("newton_max"): ref["newton_max"],
                key("min_power_factor"): ref["min_power_factor"]})
    print(f"reference: |u| {u_norm:.6e} |psi| {psi_norm:.6e} shift {ref['shift']:.6e} "
          f"min factor {ref['min_power_factor']:.4f} ({time.time() - start:.0f} s)",
          flush=True)
    for scheme, (time_scheme, pattern) in TC_SCHEMES.items():
        for n in TC_LEVELS:
            result = march(meshes, replace(base, scheme=time_scheme),
                           step_lengths(pattern, n), parameters)
            errors = {
                "u": fd.errornorm(ref["u"], result["u"]) / u_norm,
                "psi": fd.errornorm(ref["psi"], result["psi"]) / psi_norm,
                "shift": abs(result["shift"] - ref["shift"]) / abs(ref["shift"])}
            for quantity, value in errors.items():
                out[tc_key(scheme, n, quantity)] = value
            out[tc_key(scheme, n, "time")] = result["time"]
            out[tc_key(scheme, n, "newton_max")] = result["newton_max"]
            print(f"{scheme} {n}: errors u {errors['u']:.4e} psi {errors['psi']:.4e} "
                  f"shift {errors['shift']:.4e} newton max {result['newton_max']} "
                  f"({time.time() - start:.0f} s)", flush=True)
    M.write_values(CASE_DIR / time_convergence_values(name), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(TIME_CONVERGENCE))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    run_entry(name)
