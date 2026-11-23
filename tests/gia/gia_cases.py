"""Builders shared by the unit tests and the case runs of `gadopt.gia_solver`.

`Case` describes one configuration of the coupled GIA system and `build_new`
builds it with `gadopt.gia_solver.SelfGravitatingGIASolver`.
`coordinate_state`, `write_state` and `residual_and_actions` give the
residual and the Jacobian actions at one fixed state, and `fingerprint` and
`reference_values` compare them with the values stored from the earlier
solver.

The geometry is the small triangle annulus of `test_gia_direct_bdf2.py`: a
parent with a stand-off buffer, the mantle a `Submesh`, both P2-curved. The
linear mesh is stored in `tests/unit/data/gia_triangle_annulus.msh`.

Everything that enters a stored value is defined by the coordinates and not
by the order of the degrees of freedom: the state and the directions are
interpolated from fixed expressions of `SpatialCoordinate`, and a probe is a
Function interpolated in the same way and paired with the vector. The order
of the degrees of freedom depends on the mesh reader, on the reordering of
DMPlex and on the number of ranks, so values defined by that order do not
transfer between machines or rank counts. Values defined by the
coordinates do, up to round-off.
"""

import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import firedrake as fd
import numpy as np

import gadopt
from gadopt import gia_couplings as new_couplings
from gadopt.gia_solver import SelfGravitatingGIASolver as NewSolver

# The constants and tags of `test_gia_gravity.py`.
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3
CELL_MANTLE = 101
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5
DT0 = 0.1
#: The quadrature degree of every Quadrature-family case (the drivers' 4).
QUAD = 4


#: The stored data of the GIA unit tests (the mesh and the reference values)
#: is in `tests/unit/data`. The path is resolved from this file, which is in
#: `tests/gia`, so that it does not depend on the working directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "unit" / "data"


#: The linear triangle annulus of the GIA unit tests. Written once by
#: `demos/gravity/generate_selfgrav_annulus.generate(path, dr_mantle=.3,
#: n_azimuthal=16, quads=False)` with gmsh 4.15.2 and stored, so that the
#: cells are the same on every machine and no test needs gmsh.
MESH_FILE = DATA_DIR / "gia_triangle_annulus.msh"


def curve_mesh(linear_mesh):
    """Remap a linear annulus mesh to P2 coordinates, so its circles are curved.

    Edge midpoints are pushed radially onto the linear interpolant of the
    vertex radii, so an edge whose two vertices lie on a circle becomes a
    quadratic arc on that circle. The same map as `curve_mesh` of
    `demos/gravity/validate_selfgrav_annulus.py`; it is repeated here because
    that module imports gmsh.

    Args:
      linear_mesh: a mesh with straight edges and no vertex at the origin.

    Returns:
      A new mesh with P2 coordinates. A `Submesh` built before the remap keeps
      its entity maps to the parent.
    """
    X = fd.SpatialCoordinate(linear_mesh)
    r = fd.sqrt(X[0] ** 2 + X[1] ** 2)
    r_p1 = fd.Function(fd.FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = fd.Function(fd.VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    return fd.Mesh(X_p2)


def annulus_meshes(comm=fd.COMM_WORLD):
    """The parent annulus and its mantle submesh, both P2-curved.

    The parent is read from `MESH_FILE`. `Submesh` does not inherit the P2
    coordinates of its parent, so the submesh is curved separately.

    Args:
      comm: the communicator of the meshes; `COMM_SELF` gives one rank its
        own serial meshes inside a parallel test.
    """
    parent = curve_mesh(fd.Mesh(str(MESH_FILE), comm=comm))
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


@dataclass(frozen=True)
class Case:
    """One configuration of the coupled system.

    Attributes:
      family: "DG" or "Quadrature".
      scheme: "backward_euler" or "bdf2".
      representation: "multiplier" or "lowrank".
      model: "maxwell", "maxwell2", "power", "backstress".
      core: "rigid" (`un = 0` at the CMB) or "fluid".
      rotation, centre: rotation rows, centre-of-mass rows.
      load: None, "surface" (prescribed), "sea_fixed", "sea_moving".
    """

    family: str = "DG"
    scheme: str = "backward_euler"
    representation: str = "multiplier"
    model: str = "maxwell"
    core: str = "rigid"
    rotation: bool = False
    centre: bool = False
    load: str | None = "surface"
    #: The mean of the prescribed load, in units of its amplitude.
    load_mean: float = 0.0

    @property
    def label(self):
        parts = [self.family, self.scheme.replace("backward_euler", "be"),
                 self.representation, self.model, self.core]
        if self.rotation:
            parts.append("rot")
        if self.centre:
            parts.append("com")
        if self.load:
            parts.append(self.load)
        return "-".join(parts)

    def but(self, **kwargs):
        return replace(self, **kwargs)


def approximation(case, control=None, B_mu=B_MU, transition_stress=.003,
                  self_gravity_number=LAMBDA):
    """A fresh approximation of `case.model`.

    Args:
      case: the configuration; `case.model` selects the rheology.
      control: `None`, or a field that scales every viscosity by `exp(control)`.
      B_mu: the self-gravity coupling number.
      self_gravity_number: the number `Lambda` of the potential equation; a
        `Real` Function makes it a control.
      transition_stress: the stress above which the power law (`"power"` and
        `"backstress"`) departs from the Newtonian viscosity. A lower value
        makes the same load more strongly nonlinear.
    """
    factor = 1.0 if control is None else fd.exp(control)
    common = dict(bulk_modulus=1.0, density=1.0, g=1.0, B_mu=B_mu,
                  self_gravity_number=self_gravity_number)
    if case.model == "maxwell":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=1.0, viscosity=1.0 * factor, **common)
    if case.model == "maxwell2":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=[.4, .6], viscosity=[.12 * factor, .6 * factor], **common)
    if case.model == "power":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=[.4, .6], viscosity=[.12 * factor, .6 * factor],
            exponent=3, transition_stress=transition_stress, **common)
    if case.model == "backstress":
        return gadopt.BackstressApproximation(
            shear_modulus=1., viscosity=.3 * factor, backstress_modulus=.8,
            recovery_time=.4, exponent=3, transition_stress=transition_stress, **common)
    raise ValueError(case.model)


def load_expression(mesh, mean=0.0):
    """The prescribed surface density `SIGMA_HAT (mean + cos 2 theta)` on `mesh`.

    A nonzero mean gives the load a second moment, so its term in the
    rotation row (`int sigma r^2 dS`) is nonzero, and a net mass for the 2-D
    monopole datum. The gate uses it; the marches use the mean-free load.
    """
    X = fd.SpatialCoordinate(mesh)
    return SIGMA_HAT * (mean + fd.cos(2 * fd.atan2(X[1], X[0])))


def polar_moment(meshes):
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub, intersect_measures=(fd.dx(domain=parent),))
    return fd.assemble(fd.dot(X, X) * dx_m)


class SeaFields:
    """The prescribed fields of the sea-level cases, on the mechanics mesh."""

    def __init__(self, sub, ice=False):
        Q = fd.FunctionSpace(sub, "CG", 2)
        X = fd.SpatialCoordinate(sub)
        theta = fd.atan2(X[1], X[0])
        # The reference sea level changes sign, so the coastline crosses the
        # surface and the live sheet is nonlinear where it is sampled.
        self.SL_init = fd.Function(Q).interpolate(.005 * fd.cos(theta))
        self.I_init = fd.Function(Q)
        self.I = fd.Function(Q)
        if ice:
            # An ice cap on one side, with a grounding line: used where the
            # residual is compared at a fixed state. A march from rest under
            # this sudden load is not solved by Newton from zero.
            self.I.interpolate(fd.conditional(
                fd.cos(theta) < -.3, .02 * (-fd.cos(theta) - .3), 0.))

    def kwargs(self, case):
        return dict(boundary=CURVE_RE, rho_w=.1, rho_i=.09, g_surface=1.,
                    SL_init=self.SL_init, I=self.I, I_init=self.I_init,
                    N_init=fd.Constant(0.), ur_init=fd.Constant(0.),
                    softplus_width=.001, fixed_ocean=case.load == "sea_fixed")


def gravity_bcs(parent, truncation=3):
    return {CURVE_OUTER: {"dtn": gadopt.CylindricalDtN(truncation)},
            CURVE_INNER: {"dtn": gadopt.CylindricalDtN(truncation)}}


def dt_function(sub, value=DT0):
    return fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(value)


def build_new(meshes, case, *, control=None, dt=None, sea=None, nullspace=None,
              solver_parameters="direct", extra=None, default_options=True,
              parameters=None, extra_bcs=None, solver_class=NewSolver, **kwargs):
    """The new solver for `case`. Returns `(solver, u, psi)`.

    Args:
      meshes: the parent annulus and its mantle submesh.
      case: the configuration.
      control: `None`, or a `Real` Function that scales the viscosity.
      dt: the time step, a `Real` Function; `None` gives `DT0`.
      sea: the prescribed fields of a sea-level case; `None` makes new ones.
      nullspace: the `nullspace` argument of the solver.
      solver_parameters: the preset name or dictionary.
      extra: options added to the tight default tolerances.
      default_options: `False` leaves out the tight default tolerances.
      parameters: values that replace coupling parameters, keyed
        `"Coupling.argument"`, for example `{"FluidCore.g": g}`, and
        `"approximation.B_mu"` for the self-gravity coupling number,
        `"approximation.self_gravity_number"` for `Lambda` and
        `"approximation.transition_stress"` for the power law.
      extra_bcs: displacement conditions added to the ones the case gives,
        `{boundary: {name: value}}`, merged per boundary, for example
        `{CURVE_RE: {"normal_stress": sigma}}`.
      solver_class: `SelfGravitatingGIASolver` or a subclass of it, for
        example the solver with a frozen datum of the `adjoint` case
        (`gia_adjoint.FrozenDatumSolver`).
      kwargs: further arguments of the solver.
    """
    parent, sub = meshes
    parameters = dict(parameters or {})

    def value(key, default):
        # A replaced coupling parameter, or the value of the test geometry.
        return parameters.pop(key, default)
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    bcs = {}
    couplings = []
    if case.core == "rigid":
        bcs[CURVE_RC] = {"un": 0.0}
    else:
        couplings.append(new_couplings.FluidCore(
            boundary=CURVE_RC, rho_core=value("FluidCore.rho_core", 1.2),
            g=value("FluidCore.g", 1.)))
    # The caller's conditions, merged per boundary into the case's own.
    for bc_id, conditions in (extra_bcs or {}).items():
        bcs.setdefault(bc_id, {}).update(conditions)
    if case.rotation:
        couplings.append(new_couplings.Rotation(
            C=polar_moment(meshes), Omega_sq=value("Rotation.Omega_sq", 1.566e-3)))
    if case.centre:
        couplings.append(new_couplings.CentreOfMass())
    if case.load == "surface":
        couplings.append(new_couplings.SurfaceLoad(
            boundary=CURVE_RE, sigma=load_expression(sub, case.load_mean),
            g_surface=value("SurfaceLoad.g_surface", 1.0)))
    elif case.load in ("sea_fixed", "sea_moving"):
        sea = sea or SeaFields(sub)
        couplings.append(new_couplings.SeaLevel(**{
            **sea.kwargs(case),
            "g_surface": value("SeaLevel.g_surface", 1.0)}))
    # Tight tolerances, so that the direct presets compare at round-off.
    options = (dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11)
               if default_options else {})
    options.update(extra or {})
    material = approximation(
        case, control, B_mu=value("approximation.B_mu", B_MU),
        transition_stress=value("approximation.transition_stress", .003),
        self_gravity_number=value("approximation.self_gravity_number", LAMBDA))
    # Every replacement must have reached a coupling of the case.
    assert not parameters, f"parameters of couplings the case lacks: {sorted(parameters)}"
    solver = solver_class(
        u, psi, material, dt=dt or dt_function(sub),
        time_scheme=case.scheme, internal_variable_family=case.family,
        quad_degree=QUAD if case.family == "Quadrature" else None,
        gravity_bcs=gravity_bcs(parent), bcs=bcs, couplings=couplings,
        dtn_representation=case.representation, nullspace=nullspace,
        solver_parameters=solver_parameters, solver_parameters_extra=options,
        **kwargs)
    return solver, u, psi


def _unit(seed, k):
    """A number in [0, 1) from two integers, the same on every platform.

    Integer arithmetic modulo 2^32 (Knuth's multiplicative hash), so the
    value does not depend on a NumPy generator, whose stream NumPy does not
    promise to keep across versions.
    """
    return ((seed * 2654435761 + (k + 1) * 40503 + 12345) % 2 ** 32) / 2 ** 32


def _component(X, seed, k):
    """One smooth scalar of the coordinates `X` with parameters from `(seed, k)`.

    `sin(a . X + p) + 0.5 cos(b . X + q) + 0.3`, with every component of the
    wave vectors `a` and `b` between -3 and 3 in units of the inverse mantle
    depth. The annulus spans radii 0.6 to 4.4, so each term has a few
    oscillations across it and no symmetry of the annulus; the constant part
    keeps the mean of the field nonzero.
    """
    dim = len(X)
    values = [_unit(seed, 8 * k + j) for j in range(2 * dim + 2)]
    a = [6 * values[j] - 3 for j in range(dim)]
    b = [6 * values[dim + j] - 3 for j in range(dim)]
    p, q = 2 * np.pi * values[2 * dim], 2 * np.pi * values[2 * dim + 1]
    return (fd.sin(sum(a_j * X[j] for j, a_j in enumerate(a)) + p)
            + .5 * fd.cos(sum(b_j * X[j] for j, b_j in enumerate(b)) + q) + .3)


def coordinate_function(space, name, amplitude):
    """A Function of `space` defined by the coordinates and by `name`.

    Every component is a smooth expression of `SpatialCoordinate` whose
    parameters come from `crc32(name)` and the component number, scaled by
    `amplitude`, and interpolated into `space`. A `Real` space gets one
    number per component from the same parameters. The values at a node
    depend only on where the node is, so the Function is the same on every
    machine and rank count up to round-off.

    Args:
      space: a function space that is not mixed.
      name: a label that selects the expression, for example the field name.
      amplitude: the scale of the values.

    Returns:
      The new Function.
    """
    seed = zlib.crc32(name.encode())
    f = fd.Function(space)
    shape = space.value_shape
    size = int(np.prod(shape)) if shape else 1
    if space.ufl_element().family() == "Real":
        values = [amplitude * (_unit(seed, k) - .5) for k in range(size)]
        f.assign(fd.Constant(np.reshape(values, shape) if shape else values[0]))
        return f
    X = fd.SpatialCoordinate(space.mesh())
    components = [amplitude * _component(X, seed, k) for k in range(size)]
    expression = (fd.as_tensor(np.reshape(np.array(components, dtype=object), shape).tolist())
                  if shape else components[0])
    return f.interpolate(expression)


def coordinate_mixed_function(new, name, amplitude):
    """A Function of the solver's mixed space, field by field from the coordinates.

    Each sub-function is `coordinate_function` of its space with the label
    `name:field`, so a field gets the same values whatever its position in
    the mixed space.

    Args:
      new: a `SelfGravitatingGIASolver`.
      name: the label of the whole direction.
      amplitude: the scale of the values.
    """
    out = fd.Function(new.solution_space)
    for field, index in new.field_index.items():
        sub = out.subfunctions[index]
        fill(sub, coordinate_function(sub.function_space(), f"{name}:{field}", amplitude))
    return out


def fill(function, values):
    """Copy the coefficient vector `values` (a Function) into `function`."""
    with values.dat.vec_ro as src, function.dat.vec_wo as dst:
        src.copy(dst)


def coordinate_state(new, amplitude=1e-3):
    """Coordinate-defined values for every field and both history levels, by name.

    The displacement of amplitude 1e-3 with wave numbers up to about 4 gives
    strains of a few times 1e-3, above the transition stress 0.003 of the
    power-law cases, so the nonlinear part of those rheologies is reached.

    Args:
      new: a `SelfGravitatingGIASolver`.
      amplitude: the scale of the values.

    Returns:
      `{name: Function}` with the entries of `new.state()` (without
      `previous_dt`) and, for a solver with a history unknown,
      `history_iterate`, the Newton iterate of the history.
    """
    state = {}
    for name, f in new.state().items():
        if name == "previous_dt":
            continue
        state[name] = coordinate_function(f.function_space(), name, amplitude)
    if "history" in new.field_index:
        state["history_iterate"] = coordinate_function(
            new.solution.subfunctions[new.field_index["history"]].function_space(),
            "history_iterate", amplitude)
    return state


def write_state(new, state, *, previous_dt):
    """Write one fixed state into the new solver, field by field and by name.

    Also prepares the time coefficients: a BDF2 solver with a previous step
    `previous_dt` gets its BDF2 coefficients; `None` gives backward Euler.
    The 2-D monopole datum is updated at this state.

    Args:
      new: a `SelfGravitatingGIASolver` of `gadopt.gia_solver`.
      state: `{name: Function}` from `coordinate_state`.
      previous_dt: the length of the previous step, or `None`.
    """
    for name, index in new.field_index.items():
        # The DG history unknown is the Newton iterate, a value of its own;
        # the accepted levels are the integrator's.
        source = state["history_iterate"] if name == "history" else state[name]
        fill(new.solution.subfunctions[index], source)
    fill(new.time_integrator.current[0], state["history"])
    if new.time_integrator.previous is not None:
        fill(new.time_integrator.previous[0], state["history_previous"])
    new.time_integrator.previous_dt = previous_dt
    new.time_integrator.prepare()
    new.update_total_mass()


def _lowrank(solver, x, y):
    """`y += theta_psi B x`, the low-rank part of the residual or the Jacobian."""
    if solver.dtn_operator is None:
        return
    with x.dat.vec_ro as xv, y.dat.vec_rw as yv:
        solver.dtn_operator.apply_local(xv.array_r, yv.array_w)


def residual_and_actions(solver, directions):
    """The assembled residual and the Jacobian actions at the present state.

    On the Quadrature family the residual and the Jacobian are those of the
    step backend: the local constitutive root is refreshed at the state,
    and the private inputs are set to the live ones.
    """
    if hasattr(solver._step, "backend"):
        backend = solver._step.backend
    else:
        backend = None
    if backend is not None:
        # The mixed solution is one of the inputs, so this also copies the
        # state into `backend.work`.
        for target, value in zip(backend.private_inputs, backend.inputs):
            target.assign(value)
        backend.material.refresh(backend.work)
        state, F, J = backend.work, backend.F, backend.J
    else:
        state, F = solver.solution, solver.F
        J = fd.derivative(F, solver.solution)
    residual = fd.assemble(F)
    _lowrank(solver, state, residual)
    actions = []
    for d in directions:
        direction = fd.Function(state.function_space())
        fill(direction, d)
        action = fd.assemble(fd.action(J, direction))
        _lowrank(solver, direction, action)
        actions.append(action)
    return residual, actions


#: Bound on the relative difference of one field block. Every block is
#: nonzero in the gate cases, so round-off of one block stays near 1e-15.
FIELD_RTOL = 1e-10


def probe_pairing(vector, name):
    """`[norm, probe, probe norm]` of one vector, with a coordinate-defined probe.

    `vector` is a Function or a Cofunction that is not mixed. The probe is
    `coordinate_function` of the primal space with the label `probe:name`,
    and `probe` is the sum over the degrees of freedom of the vector entry
    times the probe value at that node. The norms are Euclidean norms of
    the coefficient vectors. All three are reductions over every rank, so
    they do not depend on the order of the degrees of freedom or on the rank
    count, and every rank gets the same numbers.

    Args:
      vector: the vector to fingerprint.
      name: the field name, which selects the probe.
    """
    space = vector.function_space()
    if isinstance(vector, fd.Cofunction):
        space = space.dual()
    probe = coordinate_function(space, f"probe:{name}", 1.0)
    with vector.dat.vec_ro as v, probe.dat.vec_ro as w:
        return [float(v.norm()), float(v.dot(w)), float(w.norm())]


def fingerprint(vector, index_of):
    """`{field: [norm, probe, probe norm]}` of a mixed residual or action vector.

    Each field block is paired with its own coordinate-defined probe
    (`probe_pairing`), so two vectors with the same fingerprint agree in norm
    and in one fixed direction per block.

    Args:
      vector: a Cofunction on a mixed space.
      index_of: `{field name: index in that mixed space}`.
    """
    return {name: probe_pairing(vector.subfunctions[index], name)
            for name, index in index_of.items()}


def fingerprint_differences(new, reference):
    """`{field: relative difference}` of two fingerprints, block by block.

    The difference of a block is the larger of the norm difference over the
    reference norm and the probe difference over the largest probe value the
    block can give (reference norm times probe norm). Both are near 1e-15
    when the two vectors agree to round-off.

    Args:
      new: a fingerprint from `fingerprint`.
      reference: a fingerprint of the same fields, for example from
        `reference_values`.
    """
    assert set(new) == set(reference), (sorted(new), sorted(reference))
    out = {}
    for name, (norm, probe, probe_norm) in reference.items():
        new_norm, new_probe, _ = new[name]
        scale = norm if norm > 0 else 1e-300
        out[name] = max(abs(new_norm - norm) / scale,
                        abs(new_probe - probe) / (scale * probe_norm))
    return out


def assert_fingerprints_agree(new, reference, rtol=FIELD_RTOL, field_rtol=None):
    """Assert two fingerprints agree to round-off, block by block.

    Args:
      new, reference: fingerprints from `fingerprint` or `reference_values`.
      rtol: the bound of every block that `field_rtol` does not name.
      field_rtol: `{field: bound}` for the blocks with a bound of their own,
        or `None`. A caller that gives one states the cause and the measured
        difference of each named block where it defines the dictionary.
    """
    field_rtol = field_rtol or {}
    bad = {name: value for name, value in
           fingerprint_differences(new, reference).items()
           if value > field_rtol.get(name, rtol)}
    assert not bad, f"field blocks differ from the reference: {bad}"


#: The stored values of the earlier solver that the comparisons read:
#: `gia_gate_reference.json` holds the fingerprints of the residual and of two
#: Jacobian actions per gate case (keys "2d" and "3d", then the case label),
#: `gia_forward_reference.json` the fingerprints of the three-step forward
#: states and the `Real` values after each step. Both were recorded from the
#: earlier solver (`gadopt.gia_gravity` at commit 2f4b1397) with the
#: coordinate-defined states, directions and probes of this module, so the
#: comparisons hold on any machine and any number of ranks.
REFERENCE_DIR = DATA_DIR


def reference_values(kind):
    """The stored reference values of `kind` ("gate" or "forward")."""
    import json
    with open(REFERENCE_DIR / f"gia_{kind}_reference.json") as f:
        return json.load(f)
