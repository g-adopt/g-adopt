"""Gate 1: the self-gravitating GIA solver writes the equations it was checked against.

For every configuration the earlier solver (`gadopt.gia_gravity` at commit
2f4b1397) supported, that solver and the present one were
built on the same annulus, one fixed nonzero state was written into both
(every field, both history levels, the BDF2 coefficients after a startup
step), and the assembled residual and the Jacobian action on two fixed
directions agreed field by field to round-off. The earlier solver's vectors
are stored as fingerprints (`gia_cases.fingerprint`: per field block
the norm and the pairing with a fixed probe) in
`data/gia_gate_reference.json`, so this file now compares the present solver
against those stored values. The state, the directions and the probes are
interpolated from fixed expressions of the coordinates
(`gia_cases.coordinate_function`), so the comparison does not depend
on the order of the degrees of freedom and runs on any number of ranks.

On the Quadrature family the residual and the Jacobian are those of the
step backend, after the local constitutive root is solved at the state: that
is the system the step solves. On the low-rank DtN path the low-rank update
`theta_psi B` is added, as the solver's callbacks add it.

Configurations the earlier solver reached only through its pointwise
substitution are not compared: the present solver has no pointwise path. A
rigid core is compared on the DG family only.
"""

import firedrake as fd
import pytest

import gia_cases as C

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

# The prescribed load of the gate has a mean, so its second moment, the
# load's term in the rotation row, is nonzero and compared.
CASES = [c.but(load_mean=0.5) if c.load == "surface" else c for c in CASES]

#: Round-off bound on the relative difference of one field block.
RTOL = C.FIELD_RTOL

#: The stored fingerprints of the earlier solver, by case label.
REFERENCE = C.reference_values("gate")["2d"]


@pytest.fixture(scope="module")
def meshes():
    return C.annulus_meshes()


def gate(meshes, case, *, perturb=None):
    """Build the solver at the fixed state of `case` and fingerprint its vectors.

    Args:
      meshes: the parent annulus and its mantle submesh.
      case: the configuration.
      perturb: `None`, or a function that changes the built solver (the
        negative controls).

    Returns:
      Three fingerprints: the residual and the two Jacobian actions.
    """
    sea = C.SeaFields(meshes[1], ice=True)
    new, _, _ = C.build_new(meshes, case, sea=sea)
    if perturb is not None:
        perturb(new)
    # The labels of the state and the directions select their expressions;
    # they are those of the recording, and a change here changes the state
    # so that the stored values no longer apply.
    state = C.coordinate_state(new)
    C.write_state(new, state, previous_dt=0.07 if case.scheme == "bdf2" else None)
    if case.scheme == "bdf2":
        # The BDF2 coefficients of a step after a startup step of 0.07.
        r = C.DT0 / .07
        assert abs(float(new.time_integrator.alpha) - (1 + 2 * r) / (1 + r)) < 1e-14
    directions = [C.coordinate_mixed_function(new, f"direction{k}", 1.0)
                  for k in range(2)]
    residual, actions = C.residual_and_actions(new, directions)
    return [C.fingerprint(v, new.field_index) for v in [residual, *actions]]


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_residual_and_jacobian_action_match_the_reference(meshes, case):
    """Residual and two Jacobian actions agree with the stored values in every block.

    Every field block is compared, so that a difference in a small block (a
    `Real` row) is not hidden in the norm of the whole vector.
    """
    fingerprints = gate(meshes, case)
    reference = REFERENCE[case.label]
    # The state is nonzero, so the residual is not trivially equal.
    assert fingerprints[0]["displacement"][0] > 0
    for new, ref in zip(fingerprints, reference):
        C.assert_fingerprints_agree(new, ref, rtol=RTOL)


def _add(term):
    """A perturbation that adds `term(new)` to the new solver's residual."""
    def perturb(new):
        new.F = new.F + term(new)
    return perturb


NEGATIVE = {
    # About a thousandth of the elastic stiffness of the displacement block.
    "displacement": lambda new: 1e-3 * fd.inner(new.solution_split[0], new.tests[0]) * new.gravity.dx_m,
    # About a thousandth of the Laplacian of the potential row.
    "potential": lambda new: 1e-3 * new.field("potential") * new.tests[new.field_index["potential"]] * new.gravity.dx_g,
    # A thousandth of the diagonal of the rotation row, a `Real` row.
    "m3": lambda new: 1e-3 * new.field("m3") * new.tests[new.field_index["m3"]] * new.gravity.dx_m,
}


@pytest.mark.parametrize("row", list(NEGATIVE))
def test_the_gate_sees_a_changed_row(meshes, row):
    """Negative controls: one small extra term in one row is seen in that block.

    The block of the changed row differs from the stored values far above
    round-off in the residual and in the Jacobian actions, and every other
    block still agrees, so the comparison maps the blocks by name and does
    not hide a small block.
    """
    case = Case(rotation=True).but(load_mean=0.5)
    fingerprints = gate(meshes, case, perturb=_add(NEGATIVE[row]))
    for new, ref in zip(fingerprints, REFERENCE[case.label]):
        differences = C.fingerprint_differences(new, ref)
        print(row, f"{differences[row]:.1e}")
        assert differences[row] > 1e2 * RTOL, (row, differences[row])
        others = {name: value for name, value in differences.items() if name != row}
        assert max(others.values()) <= RTOL, others
