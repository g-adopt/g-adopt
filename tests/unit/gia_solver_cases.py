"""Builders shared by the tests of `gadopt.gia_solver`.

`Case` describes one configuration of the coupled GIA system. `build_new`
builds it with the new solver (`gadopt.gia_solver.SelfGravitatingGIASolver`)
and `build_old` with the old one (`gadopt.gia_gravity`), on the same meshes
and with the same numbers, so that a test can compare the two field by field.

The geometry is the small triangle annulus of `test_gia_direct_bdf2.py`: a
parent with a stand-off buffer, the mantle a `Submesh`, both P2-curved.
"""

import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import firedrake as fd
import numpy as np

import gadopt
from gadopt import gia_couplings as new_couplings
from gadopt import gia_gravity as old_gia
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


def annulus_meshes():
    """The parent annulus and its mantle submesh, both P2-curved."""
    import pytest
    pytest.importorskip("gmsh")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "demos" / "gravity"))
    from generate_selfgrav_annulus import generate
    from validate_selfgrav_annulus import curve_mesh
    path = Path(tempfile.gettempdir()) / "gadopt_bdf2_triangle_annulus.msh"
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        generate(str(path), dr_mantle=.3, n_azimuthal=16, quads=False)
    fd.COMM_WORLD.barrier()
    parent = curve_mesh(fd.Mesh(str(path)))
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


def approximation(case, control=None):
    """A fresh approximation of `case.model`; `control` scales the viscosity."""
    factor = 1.0 if control is None else fd.exp(control)
    common = dict(bulk_modulus=1.0, density=1.0, g=1.0, B_mu=B_MU,
                  self_gravity_number=LAMBDA)
    if case.model == "maxwell":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=1.0, viscosity=1.0 * factor, **common)
    if case.model == "maxwell2":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=[.4, .6], viscosity=[.12 * factor, .6 * factor], **common)
    if case.model == "power":
        return gadopt.CompressibleInternalVariableApproximation(
            shear_modulus=[.4, .6], viscosity=[.12 * factor, .6 * factor],
            exponent=3, transition_stress=.003, **common)
    if case.model == "backstress":
        return gadopt.BackstressApproximation(
            shear_modulus=1., viscosity=.3 * factor, backstress_modulus=.8,
            recovery_time=.4, exponent=3, transition_stress=.003, **common)
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
              **kwargs):
    """The new solver for `case`. Returns `(solver, u, psi)`."""
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    bcs = {}
    couplings = []
    if case.core == "rigid":
        bcs[CURVE_RC] = {"un": 0.0}
    else:
        couplings.append(new_couplings.FluidCore(boundary=CURVE_RC, rho_core=1.2, g=1.))
    if case.rotation:
        couplings.append(new_couplings.Rotation(C=polar_moment(meshes), Omega_sq=1.566e-3))
    if case.centre:
        couplings.append(new_couplings.CentreOfMass())
    if case.load == "surface":
        couplings.append(new_couplings.SurfaceLoad(
            boundary=CURVE_RE, sigma=load_expression(sub, case.load_mean),
            g_surface=1.0))
    elif case.load in ("sea_fixed", "sea_moving"):
        sea = sea or SeaFields(sub)
        couplings.append(new_couplings.SeaLevel(**sea.kwargs(case)))
    # Tight tolerances, so that the direct presets compare at round-off.
    options = (dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11)
               if default_options else {})
    options.update(extra or {})
    solver = NewSolver(
        u, psi, approximation(case, control), dt=dt or dt_function(sub),
        time_scheme=case.scheme, internal_variable_family=case.family,
        quad_degree=QUAD if case.family == "Quadrature" else None,
        gravity_bcs=gravity_bcs(parent), bcs=bcs, couplings=couplings,
        dtn_representation=case.representation, nullspace=nullspace,
        solver_parameters=solver_parameters, solver_parameters_extra=options,
        **kwargs)
    return solver, u, psi


def build_old(meshes, case, *, control=None, dt=None, sea=None,
              declare_nullspace=False, solver_parameters="direct", extra=None):
    """The old solver for `case`. Returns `(solver, z, layout)`."""
    parent, sub = meshes
    quadrature = case.family == "Quadrature"
    g_bcs = gravity_bcs(parent)
    bcs = {}
    if case.load == "surface":
        g_bcs[CURVE_RE] = {"interior_sigma": load_expression(parent, case.load_mean)}
        bcs[CURVE_RE] = {"normal_stress": B_MU * load_expression(sub, case.load_mean)}
    if case.core == "rigid":
        bcs[CURVE_RC] = {"un": 0.0}
    material = approximation(case, control)
    Z, layout = gadopt.self_gravitating_gia_space(
        sub, parent, gravity_bcs=g_bcs, rotation=case.rotation,
        fluid_core=case.core == "fluid", centre_of_mass=case.centre,
        sea_level=case.load in ("sea_fixed", "sea_moving"),
        n_internal_variables=material.n_history_tensors,
        condense_internal_variables=quadrature,
        internal_variable_family="Quadrature" if quadrature else "DG",
        internal_variable_degree=QUAD if quadrature else 1,
        self_gravity_number=LAMBDA, dtn_representation=case.representation)
    z = fd.Function(Z)
    kwargs = {}
    if case.core == "fluid":
        kwargs["fluid_core"] = old_gia.FluidCore(boundary=CURVE_RC, rho_core=1.2, g=1.)
    if case.load in ("sea_fixed", "sea_moving"):
        sea = sea or SeaFields(sub)
        kwargs["sea_level"] = old_gia.SeaLevel(**sea.kwargs(case))
    if quadrature:
        kwargs["quad_degree"] = QUAD
        kwargs["constitutive_method"] = "direct"
    if declare_nullspace:
        ns = gadopt.rigid_rotation_nullspace(Z, layout)
        kwargs.update(nullspace=ns, transpose_nullspace=ns)
    options = dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11)
    options.update(extra or {})
    solver = old_gia.SelfGravitatingGIASolver(
        z, material, layout=layout, dt=dt or dt_function(sub),
        time_scheme=case.scheme, bcs=bcs,
        rotation_moments={"C": polar_moment(meshes)} if case.rotation else None,
        solver_parameters=solver_parameters, solver_parameters_extra=options,
        **kwargs)
    return solver, z, layout


def old_field_names(layout):
    """The state names of the old layout's fields, in its space order."""
    names = {layout.displacement: "displacement", layout.potential: "potential"}
    for i in layout.internal_variables:
        names[i] = "history"
    for k, i in enumerate(layout.multipliers):
        names[i] = f"dtn_multiplier_{k}"
    if layout.core_pressure is not None:
        names[layout.core_pressure] = "core_pressure"
    for name, i in layout.rotation.items():
        names[i] = name
    for k, i in enumerate(layout.centre_of_mass):
        names[i] = f"centre_of_mass_{k}"
    if layout.sea_level is not None:
        names[layout.sea_level] = "sea_level_shift"
    return [names[i] for i in range(layout.n_fields)]


def random_function(space, rng, amplitude):
    f = fd.Function(space)
    with f.dat.vec_wo as v:
        v.array[:] = amplitude * (rng.random(v.array.shape) - .5)
    return f


def fill(function, values):
    """Copy the coefficient vector `values` (a Function) into `function`."""
    with values.dat.vec_ro as src, function.dat.vec_wo as dst:
        src.copy(dst)


def random_state(new, rng, amplitude=1e-3):
    """Random values for every field and both history levels, by name."""
    state = {}
    for name, f in new.state().items():
        if name == "previous_dt":
            continue
        state[name] = random_function(f.function_space(), rng, amplitude)
    if "history" in new.field_index:
        state["history_iterate"] = random_function(
            new.solution.subfunctions[1].function_space(), rng, amplitude)
    return state


def write_state(new, state, *, previous_dt):
    """Write one fixed state into the new solver, field by field and by name.

    Also prepares the time coefficients: a BDF2 solver with a previous step
    `previous_dt` gets its BDF2 coefficients; `None` gives backward Euler.
    The 2-D monopole datum is updated at this state.

    Args:
      new: a `SelfGravitatingGIASolver` of `gadopt.gia_solver`.
      state: `{name: Function}` from `random_state`.
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


def set_state(new, old, old_layout, state, *, previous_dt):
    """Write one fixed state into both solvers, field by field and by name.

    Also prepares the time coefficients: a BDF2 solver with a previous step
    `previous_dt` gets its BDF2 coefficients; `None` gives backward Euler.
    """
    names = old_field_names(old_layout)
    for i, name in enumerate(names):
        # The DG history unknown is the Newton iterate, a value of its own;
        # the accepted levels are the integrator's.
        source = state["history_iterate"] if name == "history" else state[name]
        fill(old.solution.subfunctions[i], source)
        fill(new.solution.subfunctions[new.field_index[name]], source)
    # The history levels.
    fill(new.time_integrator.current[0], state["history"])
    if new.time_integrator.previous is not None:
        fill(new.time_integrator.previous[0], state["history_previous"])
    if old.time_integrator is None:
        # The old DG backward-Euler path reads the old history from
        # `solution_old`.
        fill(old.solution_old.subfunctions[1], state["history"])
    else:
        fill(old.time_integrator.current[0], state["history"])
        fill(old.time_integrator.previous[0], state.get(
            "history_previous", state["history"]))
        old.time_integrator.previous_dt = previous_dt
        old.time_integrator.prepare()
        if old.layout.condensed:
            fill(old.internal_variables, state["history"])
    new.time_integrator.previous_dt = previous_dt
    new.time_integrator.prepare()
    # The 2-D monopole datum at this state.
    old.update_total_mass()
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
    if getattr(solver, "_direct_backend", None) is not None:
        backend = solver._direct_backend
    elif hasattr(solver, "_step") and hasattr(solver._step, "backend"):
        backend = solver._step.backend
    else:
        backend = None
    if backend is not None:
        for target, value in zip(backend.private_inputs, backend.inputs):
            target.assign(value)
        backend.work.assign(solver.solution)
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


def compare(old_vec, new_vec, old_layout, new):
    """`(overall, worst field, per field)` relative differences, mapped by name."""
    names = old_field_names(old_layout)
    comm = new.mesh.comm
    per_field = {}
    num = den = 0.0
    for i, name in enumerate(names):
        # The owned values of each rank, summed over the ranks.
        a = np.ravel(old_vec.subfunctions[i].dat.data_ro)
        b = np.ravel(new_vec.subfunctions[new.field_index[name]].dat.data_ro)
        diff2 = comm.allreduce(float(np.sum((a - b) ** 2)))
        size2 = comm.allreduce(float(np.sum(a ** 2)))
        num += diff2
        den += size2
        per_field[name] = np.sqrt(diff2 / size2) if size2 > 0 else np.sqrt(diff2)
    return np.sqrt(num / den), max(per_field.values()), per_field


#: Bound on the relative difference of one field block. Every block is
#: nonzero in the gate cases (the smallest, the `m3` row, is about 1e-4 of the
#: whole vector), so round-off of one block stays near 1e-15.
FIELD_RTOL = 1e-10


def assert_fields_agree(per_field, rtol=FIELD_RTOL):
    """Assert every field block of a comparison agrees to round-off."""
    bad = {name: value for name, value in per_field.items() if value > rtol}
    assert not bad, f"field blocks differ: {bad}"


def fingerprint(vector, index_of):
    """`{field: [norm, probe]}` of a mixed residual or action vector.

    `probe` is the dot product of the field block with a fixed vector drawn
    from a generator seeded by the field name, so two vectors with the same
    fingerprint agree in norm and in one fixed direction per block. Serial
    only: the probe follows the local order of the degrees of freedom.

    Args:
      vector: a Cofunction on a mixed space.
      index_of: `{field name: index in that mixed space}`.
    """
    import zlib
    out = {}
    for name, index in index_of.items():
        values = np.ravel(vector.subfunctions[index].dat.data_ro)
        probe = np.random.default_rng(zlib.crc32(name.encode())).random(values.shape) - .5
        out[name] = [float(np.linalg.norm(values)), float(values @ probe),
                     float(np.linalg.norm(probe))]
    return out


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


def assert_fingerprints_agree(new, reference, rtol=FIELD_RTOL):
    """Assert two fingerprints agree to round-off, block by block."""
    bad = {name: value for name, value in
           fingerprint_differences(new, reference).items() if value > rtol}
    assert not bad, f"field blocks differ from the reference: {bad}"


#: The stored values that replace the old solver in the comparisons:
#: `gia_gate_reference.json` holds the fingerprints of the residual and of two
#: Jacobian actions per gate case (keys "2d" and "3d", then the case label),
#: `gia_forward_reference.json` the fingerprints of the three-step forward
#: states and the `Real` values after each step. Both were recorded from the
#: old solver (`gadopt.gia_gravity`, commit 2f4b1397) in serial, with the
#: generators the tests use, so the probes follow the serial order of the
#: degrees of freedom and the comparisons run in serial only.
REFERENCE_DIR = Path(__file__).resolve().parent / "data"


def reference_values(kind):
    """The stored reference values of `kind` ("gate" or "forward")."""
    import json
    with open(REFERENCE_DIR / f"gia_{kind}_reference.json") as f:
        return json.load(f)
