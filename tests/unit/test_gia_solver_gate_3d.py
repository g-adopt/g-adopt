"""Gate 1 in 3-D: the GIA solver assembles the equations it was checked against.

Assembly only, no solve, on the 24-cell radially extruded cubed sphere
(level-1 `CubedSphereMesh`, two radial layers; one mesh for the
displacement and the potential). The 2-D gate cannot see what only 3-D has:
three rotation rows with the polar-wander pair closed by `C - A = Q k_s`, the
CMB of an extruded mesh reached as `"bottom"` through
`CombinedSurfaceMeasure`, and three centre-of-mass rows. The earlier solver
(`gadopt.gia_gravity` at commit 2f4b1397) was given the same
numbers and the same state, and its residual and two Jacobian actions are
stored as fingerprints in `data/gia_gate_reference.json` (key "3d"). The
present solver is compared against those values field by field. The state,
the directions and the probes are defined by the coordinates
(`gia_cases.coordinate_function`), so the comparison runs on any
number of ranks.
"""

import firedrake as fd
import pytest

import gadopt
import gia_cases as C
from gadopt import gia_couplings as cp
from gadopt.gia_solver import SelfGravitatingGIASolver as NewSolver

OMEGA_SQ = 1.566e-3
K_S = 0.9
RADIUS = 2.0      # the outer radius of the extruded sphere: 1 + 2 x 0.5


@pytest.fixture(scope="module")
def mesh():
    base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                           extrusion_type="radial")
    mesh.cartesian = False
    return mesh


def material(model):
    return C.approximation(C.Case(model=model))


def build(mesh, representation, model, family, core_boundary="bottom"):
    """The solver of one 3-D gate case: fluid core, rotation, centre of mass, load.

    Args:
      mesh: the extruded sphere, for the displacement and the potential.
      representation: "multiplier" or "lowrank".
      model: the rheology name of `gia_cases.approximation`.
      family: "DG" or "Quadrature".
      core_boundary: the boundary of the `FluidCore`; the CMB of the
        extruded sphere is `"bottom"`, and a test of the refusal passes
        another one.
    """
    X = fd.SpatialCoordinate(mesh)
    C_polar = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
    # Every moment the rows read: a mean (mass), `z` (the third
    # centre-of-mass row) and `x z`, `y z` (the polar-wander rows, through
    # `p_1 = -x z` and `p_2 = -y z`).
    sigma = C.SIGMA_HAT * (0.3 + X[2] + X[0] * X[2] + X[1] * X[2])
    quadrature = family == "Quadrature"
    dtn = {"top": {"dtn": gadopt.SphericalDtN(1)},
           "bottom": {"dtn": gadopt.SphericalDtN(1)}}
    approximation = material(model)
    new = NewSolver(
        fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)),
        fd.Function(fd.FunctionSpace(mesh, "CG", 2)), approximation,
        dt=fd.Constant(C.DT0), gravity_bcs=dtn, internal_variable_family=family,
        quad_degree=C.QUAD if quadrature else None,
        dtn_representation=representation,
        couplings=[cp.FluidCore(boundary=core_boundary, rho_core=2.0),
                   cp.Rotation(C=C_polar, k_s=K_S, surface_radius=RADIUS,
                               Omega_sq=OMEGA_SQ),
                   cp.CentreOfMass(),
                   cp.SurfaceLoad(boundary="top", sigma=sigma, g_surface=1.0)],
        solver_parameters="direct")
    return new


CASES = [("lowrank", "maxwell2", "Quadrature")]


#: The stored fingerprints of the earlier solver, by case label.
REFERENCE = C.reference_values("gate")["3d"]

#: The blocks with a bound of their own, one dictionary for each compared
#: vector: the residual, then the two Jacobian actions. The earlier solver
#: integrated three coupling integrals on a measure without a degree, so at
#: the degree UFL estimated: the mantle term of the rotation rows
#: (`Rotation.inertia_form`), the mantle term of the centre-of-mass rows
#: (`CentreOfMass.dipole_form`) and the displacement source of the potential
#: row (`GravityPotential.volume_source`). This solver integrates all three
#: at the explicit mechanics quadrature degree, the degree of their
#: transposes in the momentum row. On this coarse sphere the two rules
#: differ by more than on the annulus. Measured differences, residual /
#: action 0 / action 1:
#:
#: - potential: 3.48e-5 / 6.98e-5 / 1.20e-5;
#: - `m1`, `m2`: 1.99e-6 / 1.20e-7 / 9.08e-7;
#: - `m3`: 1.04e-5 / 1.46e-7 / 7.64e-7;
#: - `centre_of_mass_0`, `centre_of_mass_1`: 2.55e-7 / 1.10e-8 / 4.2e-11;
#: - `centre_of_mass_2`: 5.57e-8 / 1.10e-8 / 4.0e-11.
#:
#: Each bound is about 1.25 to 1.3 times its own measured value, so a later
#: change of one row that is larger than that in one vector is seen, also
#: if it stays below the bound of another vector. The centre-of-mass rows of
#: action 1 are below `FIELD_RTOL` and keep it. The potential row moves only
#: through the displacement source: with that source put back on UFL's
#: estimate the block agrees to 1e-12, and the `(u, psi)` pair of the
#: Jacobian is then a transpose only to 4.1e-4 (3.0e-16 with the explicit
#: degree). The displacement block agrees to 1e-12.
FIELD_BOUNDS = [
    {"potential": 4.5e-5, "m1": 2.5e-6, "m2": 2.5e-6, "m3": 1.3e-5,
     "centre_of_mass_0": 3.2e-7, "centre_of_mass_1": 3.2e-7, "centre_of_mass_2": 7e-8},
    {"potential": 9e-5, "m1": 1.5e-7, "m2": 1.5e-7, "m3": 1.9e-7,
     **{f"centre_of_mass_{i}": 1.4e-8 for i in range(3)}},
    {"potential": 1.5e-5, "m1": 1.15e-6, "m2": 1.15e-6, "m3": 1e-6},
]


@pytest.mark.parametrize("representation, model, family", CASES,
                         ids=["-".join(c) for c in CASES])
def test_3d_residual_and_jacobian_action_match_the_reference(
        mesh, representation, model, family):
    """Residual and two Jacobian actions agree with the stored values in every block."""
    new = build(mesh, representation, model, family)
    # Three rotation rows and three centre-of-mass rows, by name.
    assert {"m1", "m2", "m3", "centre_of_mass_2"} <= set(new.field_index)
    # The labels of the state and the directions are those of the recording.
    state = C.coordinate_state(new)
    C.write_state(new, state, previous_dt=None)
    directions = [C.coordinate_mixed_function(new, f"direction{k}", 1.0)
                  for k in range(2)]
    residual, actions = C.residual_and_actions(new, directions)
    reference = REFERENCE["-".join((representation, model, family))]
    for vector, ref, bounds in zip([residual, *actions], reference, FIELD_BOUNDS,
                                   strict=True):
        C.assert_fingerprints_agree(C.fingerprint(vector, new.field_index), ref,
                                    field_rtol=bounds)


def test_a_fluid_core_tag_with_no_facets_is_still_refused(mesh):
    """The extruded measure resolves more names, and refuses no fewer.

    `CombinedSurfaceMeasure` sends every integer tag to `ds_v`, the
    vertical (side) facets, and a radially extruded closed sphere has none.
    So an integer tag names an empty facet set here and must reach the
    fluid core's empty-measure refusal, exactly as a wrong tag does on the
    annulus. Without this check, a branch that resolves every name it is
    given leaves the geometry check dead on extruded meshes.
    """
    with pytest.raises(ValueError, match="empty measure"):
        build(mesh, "lowrank", "maxwell2", "Quadrature", core_boundary=4)
