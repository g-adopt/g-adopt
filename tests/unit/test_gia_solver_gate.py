"""Gate 1: the new self-gravitating GIA solver writes the old solver's equations.

For every configuration the old solver (`gadopt.gia_gravity`) supports, both
solvers are built on the same annulus, one fixed nonzero state is written
into both (every field, both history levels, the BDF2 coefficients after a
startup step), and the assembled residual and the Jacobian action on two
random directions are compared field by field, mapped by name between the
two mixed spaces. They must agree to round-off.

On the Quadrature family the residual and the Jacobian are those of the
step backend, after the local constitutive root is solved at the state: that
is the system the step solves. On the low-rank DtN path the low-rank update
`theta_psi B` is added to both, as the solver's callbacks add it.

Configurations the old solver reached only through its pointwise
substitution (the condensed layout with `constitutive_method="pointwise"`,
which accepts `un = 0`) are not compared: the new solver has no pointwise
path. A rigid core is compared on the DG family. The old solver refuses a
rigid core with backstress, and the Quadrature history with `un`.

Set `GIA_GATE_REPORT` to a file name to append one JSON line per case.
"""

import json
import os

import firedrake as fd
import numpy as np
import pytest

import gia_solver_cases as C

Case = C.Case

#: Every configuration of gate 1. The label lists family, scheme, DtN
#: representation, rheology, core, rotation, centre of mass and load.
CASES = [
    # DG family, rigid core.
    Case(rotation=True),
    Case(representation="lowrank", rotation=True),
    Case(scheme="bdf2", model="maxwell2", rotation=True),
    Case(scheme="bdf2", representation="lowrank", model="power"),
    Case(model="power", rotation=True, load=None),
    # DG family, fluid core.
    Case(representation="lowrank", model="power", core="fluid", rotation=True,
         centre=True),
    Case(model="backstress", core="fluid"),
    Case(scheme="bdf2", representation="lowrank", model="backstress",
         core="fluid", centre=True, load="sea_fixed"),
    Case(model="maxwell2", core="fluid", rotation=True, centre=True,
         load="sea_fixed"),
    Case(scheme="bdf2", representation="lowrank", model="maxwell2",
         core="fluid", rotation=True, centre=True, load="sea_moving"),
    Case(model="power", core="fluid", centre=True, load="sea_moving"),
    Case(representation="lowrank", core="fluid", load=None),
    # Quadrature family (natural displacement boundaries, so a fluid core).
    Case(family="Quadrature", representation="lowrank", core="fluid"),
    Case(family="Quadrature", scheme="bdf2", model="maxwell2", core="fluid",
         rotation=True, centre=True, load="sea_fixed"),
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="power", core="fluid", rotation=True, centre=True,
         load="sea_moving"),
    Case(family="Quadrature", model="backstress", core="fluid"),
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="backstress", core="fluid", centre=True, load="sea_moving"),
    Case(family="Quadrature", model="power", core="fluid", rotation=True),
]

#: Round-off bound on the relative difference of the whole vector.
RTOL = 1e-12


@pytest.fixture(scope="module")
def meshes():
    return C.annulus_meshes()


def old_directions(old, layout, new, directions):
    """The directions of the new space, mapped by name into the old space."""
    out = []
    for d in directions:
        od = fd.Function(old.solution_space)
        for i, name in enumerate(C.old_field_names(layout)):
            C.fill(od.subfunctions[i], d.subfunctions[new.field_index[name]])
        out.append(od)
    return out


def gate(meshes, case, *, perturb=None):
    """Build both solvers at one fixed state and compare. Returns the numbers."""
    sea = C.SeaFields(meshes[1], ice=True)
    new, _, _ = C.build_new(meshes, case, sea=sea)
    old, _, layout = C.build_old(meshes, case, sea=sea)
    if perturb is not None:
        perturb(new)
    rng = np.random.default_rng(20261119)
    state = C.random_state(new, rng)
    C.set_state(new, old, layout, state,
                previous_dt=0.07 if case.scheme == "bdf2" else None)
    if case.scheme == "bdf2":
        # The BDF2 coefficients of a step after a startup step of 0.07.
        r = C.DT0 / .07
        assert abs(float(new.time_integrator.alpha) - (1 + 2 * r) / (1 + r)) < 1e-14
    directions = [C.random_function(new.solution_space, rng, 1.0) for _ in range(2)]
    r_new, a_new = C.residual_and_actions(new, directions)
    r_old, a_old = C.residual_and_actions(
        old, old_directions(old, layout, new, directions))
    result = {"case": case.label,
              "residual": C.compare(r_old, r_new, layout, new)[:2],
              "actions": [C.compare(o, n, layout, new)[:2]
                          for o, n in zip(a_old, a_new)]}
    with r_old.dat.vec_ro as v:
        result["residual_norm"] = v.norm()
    return result


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_residual_and_jacobian_action_match_the_old_solver(meshes, case):
    result = gate(meshes, case)
    report = os.environ.get("GIA_GATE_REPORT")
    if report and fd.COMM_WORLD.rank == 0:
        with open(report, "a") as f:
            f.write(json.dumps({k: (list(map(float, v)) if k == "residual" else
                                    [list(map(float, a)) for a in v] if k == "actions"
                                    else v) for k, v in result.items()}) + "\n")
    print(result)
    assert result["residual_norm"] > 0
    overall, worst = result["residual"]
    assert overall < RTOL, result
    for overall, worst in result["actions"]:
        assert overall < RTOL, result


def test_the_gate_sees_a_changed_residual(meshes):
    """Negative control: one small extra term is a difference far above round-off.

    The term `1e-3 (u, w)` is about a thousandth of the elastic stiffness,
    the size of a modelling error the gate must catch.
    """
    def add_a_term(new):
        new.F = new.F + 1e-3 * fd.inner(new.solution_split[0], new.tests[0]) * new.gravity.dx_m

    result = gate(meshes, Case(rotation=True), perturb=add_a_term)
    print(result)
    # The state is small (1e-3) against the load, so the term moves the
    # residual by about 6e-7 of its size and the actions by about 1e-3:
    # both are many orders above round-off.
    assert result["residual"][0] > 1e3 * RTOL
    assert all(overall > 1e3 * RTOL for overall, _ in result["actions"])
