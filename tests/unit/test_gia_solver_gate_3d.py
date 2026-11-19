"""Gate 1 in 3-D: the GIA solver assembles the equations it was checked against.

Assembly only, no solve, on the 24-cell radially extruded cubed sphere of
`test_gia_nested_condensation.py::TestThreeDimensions` (one mesh for the
displacement and the potential). The 2-D gate cannot see what only 3-D has:
three rotation rows with the polar-wander pair closed by `C - A = Q k_s`, the
CMB of an extruded mesh reached as `"bottom"` through
`CombinedSurfaceMeasure`, and three centre-of-mass rows. The earlier solver
(`gadopt.gia_gravity` at commit 2f4b1397) was given the same
numbers and the same random state, and its residual and two Jacobian actions
are stored as fingerprints in `data/gia_gate_reference.json` (key "3d"). The
present solver is compared against those values field by field, in serial,
because the probes follow the serial order of the degrees of freedom.
"""

import firedrake as fd
import numpy as np
import pytest

import gadopt
import gia_solver_cases as C
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


def build(mesh, representation, model, family):
    """The solver of one 3-D gate case: fluid core, rotation, centre of mass, load.

    Args:
      mesh: the extruded sphere, for the displacement and the potential.
      representation: "multiplier" or "lowrank".
      model: the rheology name of `gia_solver_cases.approximation`.
      family: "DG" or "Quadrature".
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
        couplings=[cp.FluidCore(boundary="bottom", rho_core=2.0),
                   cp.Rotation(C=C_polar, k_s=K_S, surface_radius=RADIUS,
                               Omega_sq=OMEGA_SQ),
                   cp.CentreOfMass(),
                   cp.SurfaceLoad(boundary="top", sigma=sigma, g_surface=1.0)],
        solver_parameters="direct")
    return new


CASES = [("multiplier", "maxwell", "DG"), ("lowrank", "maxwell2", "DG"),
         ("lowrank", "power", "DG"), ("lowrank", "maxwell2", "Quadrature")]


#: The stored fingerprints of the earlier solver, by case label.
REFERENCE = C.reference_values("gate")["3d"]


@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason="the stored probes are serial")
@pytest.mark.parametrize("representation, model, family", CASES,
                         ids=["-".join(c) for c in CASES])
def test_3d_residual_and_jacobian_action_match_the_reference(
        mesh, representation, model, family):
    """Residual and two Jacobian actions agree with the stored values in every block."""
    new = build(mesh, representation, model, family)
    # Three rotation rows and three centre-of-mass rows, by name.
    assert {"m1", "m2", "m3", "centre_of_mass_2"} <= set(new.field_index)
    # The generator, its seed and the order of the draws are those of the
    # recording.
    rng = np.random.default_rng(7)
    state = C.random_state(new, rng)
    C.write_state(new, state, previous_dt=None)
    directions = [C.random_function(new.solution_space, rng, 1.0) for _ in range(2)]
    residual, actions = C.residual_and_actions(new, directions)
    reference = REFERENCE["-".join((representation, model, family))]
    for vector, ref in zip([residual, *actions], reference):
        C.assert_fingerprints_agree(C.fingerprint(vector, new.field_index), ref)
