"""Gate 1 in 3-D: the new solver assembles the old solver's equations.

Assembly only, no solve, on the 24-cell radially extruded cubed sphere of
`test_gia_nested_condensation.py::TestThreeDimensions` (one mesh for the
displacement and the potential). The 2-D gate cannot see what only 3-D has:
three rotation rows with the polar-wander pair closed by `C - A = Q k_s`, the
CMB of an extruded mesh reached as `"bottom"` through
`CombinedSurfaceMeasure`, and three centre-of-mass rows. Both solvers get the
same numbers, one random state is written into both, and the residual and
two Jacobian actions are compared field by field, mapped by name.
"""

import firedrake as fd
import numpy as np
import pytest

import gadopt
import gia_solver_cases as C
from gadopt import gia_couplings as cp
from gadopt import gia_gravity as old_gia
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
    X = fd.SpatialCoordinate(mesh)
    C_polar = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
    # Every moment the rows read: a mean (mass), `z` (the third
    # centre-of-mass row) and `x z`, `y z` (the polar-wander rows, through
    # `p_1 = -x z` and `p_2 = -y z`).
    sigma = C.SIGMA_HAT * (0.3 + X[2] + X[0] * X[2] + X[1] * X[2])
    quadrature = family == "Quadrature"
    dtn = {"top": {"dtn": gadopt.SphericalDtN(1)},
           "bottom": {"dtn": gadopt.SphericalDtN(1)}}
    # The old solver: one mesh, the load given twice.
    old_dtn = {"top": {"dtn": gadopt.SphericalDtN(1), "sigma": sigma},
               "bottom": {"dtn": gadopt.SphericalDtN(1)}}
    approximation = material(model)
    Z, layout = gadopt.self_gravitating_gia_space(
        mesh, mesh, gravity_bcs=old_dtn, rotation=True, fluid_core=True,
        centre_of_mass=True, n_internal_variables=approximation.n_history_tensors,
        condense_internal_variables=quadrature,
        internal_variable_family=family, internal_variable_degree=C.QUAD if quadrature else 1,
        self_gravity_number=C.LAMBDA, dtn_representation=representation)
    kwargs = dict(quad_degree=C.QUAD, constitutive_method="direct") if quadrature else {}
    old = old_gia.SelfGravitatingGIASolver(
        fd.Function(Z), approximation, layout=layout, dt=fd.Constant(C.DT0),
        bcs={"top": {"normal_stress": C.B_MU * sigma}},
        rotation_moments={"C": C_polar, "k_s": K_S}, surface_radius=RADIUS,
        Omega_sq=OMEGA_SQ,
        fluid_core=old_gia.FluidCore(boundary="bottom", rho_core=2.0),
        solver_parameters="direct", **kwargs)
    new = NewSolver(
        fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)),
        fd.Function(fd.FunctionSpace(mesh, "CG", 2)), material(model),
        dt=fd.Constant(C.DT0), gravity_bcs=dtn, internal_variable_family=family,
        quad_degree=C.QUAD if quadrature else None,
        dtn_representation=representation,
        couplings=[cp.FluidCore(boundary="bottom", rho_core=2.0),
                   cp.Rotation(C=C_polar, k_s=K_S, surface_radius=RADIUS,
                               Omega_sq=OMEGA_SQ),
                   cp.CentreOfMass(),
                   cp.SurfaceLoad(boundary="top", sigma=sigma, g_surface=1.0)],
        solver_parameters="direct")
    return new, old, layout


CASES = [("multiplier", "maxwell", "DG"), ("lowrank", "maxwell2", "DG"),
         ("lowrank", "power", "DG"), ("lowrank", "maxwell2", "Quadrature")]


@pytest.mark.parametrize("representation, model, family", CASES,
                         ids=["-".join(c) for c in CASES])
def test_3d_residual_and_jacobian_action_match_the_old_solver(
        mesh, representation, model, family):
    new, old, layout = build(mesh, representation, model, family)
    names = C.old_field_names(layout)
    # Three rotation rows and three centre-of-mass rows, by name.
    assert {"m1", "m2", "m3", "centre_of_mass_2"} <= set(names)
    assert names == list(new.field_names)
    rng = np.random.default_rng(7)
    state = C.random_state(new, rng)
    C.set_state(new, old, layout, state, previous_dt=None)
    directions = [C.random_function(new.solution_space, rng, 1.0) for _ in range(2)]
    old_directions = []
    for d in directions:
        od = fd.Function(old.solution_space)
        for i, name in enumerate(names):
            C.fill(od.subfunctions[i], d.subfunctions[new.field_index[name]])
        old_directions.append(od)
    r_new, a_new = C.residual_and_actions(new, directions)
    r_old, a_old = C.residual_and_actions(old, old_directions)
    results = [C.compare(r_old, r_new, layout, new)]
    results += [C.compare(o, n, layout, new) for o, n in zip(a_old, a_new)]
    print(family, representation, model,
          [(f"{whole:.1e}", f"{worst:.1e}") for whole, worst, _ in results])
    for whole, _, per_field in results:
        assert whole < 1e-12
        C.assert_fields_agree(per_field)
