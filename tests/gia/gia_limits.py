"""Driver of the `limits` case: solves whose answer is known in a limit or exactly.

Usage: `python gia_limits.py --entry <name>`, with the entry names of
`gia_entries.LIMITS` (`sea`, `mechanics`, `rheology`).

Each entry runs a group of rows, on the Quadrature history family
(`internal_variable_family="Quadrature"`, `quad_degree=gia_cases.QUAD`).
The driver writes, for each row, the measured number and the number it is
checked against (a closed form, a limit, an independent solve or an exact
identity) as `name value` lines to `out/limits-<entry>.dat`. The test
`test_gia_limits.py` holds the tolerances and asserts; this script asserts
nothing. No row uses a recorded value.

The configurations come from the helper modules of `gia_models`, which the
unit tests also use:

- `gia_models.sea_level`: `build`, `surface_fields`, `coastline_fields`,
  `indicator`, `re_measure`, `solve_settings`, `remove_rigid_rotation`,
  `sheet`, and the constants `RHO_I`, `RHO_W`, `EPS`, `GROUNDED`, `SHIFT`,
  `ELASTIC_EARTH`;
- `gia_models.sea_level_sheet`: `marine_fields`, `width`;
- `gia_models.annulus`: `approximation`, `gravity_bcs`,
  `FluidCoreBuilder.build_fluid`, `CentreOfMassFrame` (its `build` and its
  reference gravity), and the constants `B_MU`, `LAMBDA`, `SIGMA_HAT`,
  `CURVE_RE`, `CURVE_RC`, `RE`.

The stiff power-law march is configured in this file
(`stiff_power_law_march`).

The meshes are those of the unit fixture `test_gia_gravity.meshes` (the
annulus written by gmsh with a mantle 0.2 deep per layer and 32 cells
around), because the sector indicators of the sea-level rows rely on its Re
vertices at multiples of `pi/16`.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import gadopt
import firedrake as fd
import numpy as np
from firedrake.adjoint import stop_annotating
from gadopt.gia_couplings import CentreOfMass, FluidCore, Rotation, SeaLevel
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.internal_variable_equation import history_slices
from gadopt.nullspaces import apply_kernel_projection

import gia_cases as C
import gia_march as M
from gia_entries import CASE_DIR, LIMITS, REPRESENTATIONS, STIFF_STEPS, limits_values

from gia_models import annulus as G
from gia_models import sea_level as S
from gia_models import sea_level_sheet as SH

#: The keywords that put a solver on the Quadrature history family.
QUADRATURE = dict(internal_variable_family="Quadrature", quad_degree=C.QUAD)


def unit_meshes():
    """The parent annulus and its mantle submesh of the unit fixture, P2-curved.

    The body of `test_gia_gravity.meshes`: gmsh writes the annulus once to
    the temporary directory, and both meshes are curved, because `Submesh`
    does not inherit the parent's P2 coordinates.
    """
    root = CASE_DIR.parents[1]
    sys.path.insert(0, str(root / "demos" / "gravity"))
    import generate_selfgrav_annulus as gen
    path = Path(tempfile.gettempdir()) / (
        f"gadopt_selfgrav_gia_{G.DR_MANTLE}_{G.N_AZIMUTHAL}.msh")
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        gen.generate(str(path), dr_mantle=G.DR_MANTLE, n_azimuthal=G.N_AZIMUTHAL)
    fd.COMM_WORLD.barrier()
    parent = C.curve_mesh(fd.Mesh(str(path)))
    parent.cartesian = False
    sub = C.curve_mesh(fd.Submesh(parent, 2, G.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def converged(solver):
    """1.0 if the last nonlinear solve reported convergence, else 0.0."""
    return 1.0 if solver.solver.snes.getConvergedReason() > 0 else 0.0


def try_solve(solver):
    """Solve, and return 1.0 if the solve converged, else 0.0.

    A `ConvergenceError` is caught, so that one row that does not converge
    does not stop the rows after it; its flag in the output is 0 and the
    test of that row fails on it.
    """
    try:
        solver.solve()
    except fd.ConvergenceError:
        pass
    return converged(solver)


def sea_build(meshes, fields, representation, **kwargs):
    """`test_sea_level.build` on the Quadrature family with the settings of `representation`."""
    with stop_annotating():
        return S.build(meshes, fields, representation=representation,
                       **S.solve_settings(representation), **QUADRATURE, **kwargs)


def field_differences(solver_a, solver_b):
    """The relative differences of displacement, potential and `Shift` of two solves.

    The rule of `test_sea_level.solved_fields_agree`: each field in the max
    norm relative to the max norm of the same field of `solver_b`, the
    displacement after its best-fitting rigid rotation is removed (no
    physical term fixes that mode, so its multiple follows the Krylov path).
    The entries run on one rank, so the max norm is a norm of the whole
    vector.

    Returns:
      `{name: relative difference}` for `displacement`, `potential`, `shift`.
    """
    out = {}
    for name, key in (("displacement", "displacement"), ("potential", "potential"),
                      ("shift", S.SHIFT)):
        f_a = solver_a.solution.subfunctions[solver_a.field_index[key]]
        f_b = solver_b.solution.subfunctions[solver_b.field_index[key]]
        scale = np.abs(f_b.dat.data_ro).max()
        if name == "displacement":
            du = fd.Function(f_b.function_space()).assign(f_a - f_b)
            difference = np.abs(S.remove_rigid_rotation(du, solver_b.mesh).dat.data_ro).max()
        else:
            difference = np.abs(f_a.dat.data_ro - f_b.dat.data_ro).max()
        out[name] = difference / scale if scale > 0 else np.inf
    return out


# -- Entry `sea` --------------------------------------------------------------------

def eustatic_rows(meshes, representation):
    """The eustatic shift of a melting cap on a stiff, non-gravitating Earth.

    `test_sea_level.TestEustaticLimit` (sheet of `SeaLevel`) and
    `test_sea_level_sheet.TestSolves::test_the_shift_is_the_eustatic_value`
    (the softplus width of that file): deep ocean, grounded ice melts by
    1e-2 on the two opposite caps, `Lambda = 1e-6`, moduli 1e6. The closed
    form `Shift = -(rho_i / rho_w) int dI dS / L_ocean`, with `L_ocean` the
    length of Re outside the caps.
    """
    _, sub = meshes
    d_ice = -1e-2
    out = {}
    for label, overrides in (("eustatic", None), ("eustatic_softplus", SH.width())):
        solver, _, sea = sea_build(meshes, S.surface_fields(sub, "caps", d_ice=d_ice),
                                   representation, lam=1e-6, stiff=True,
                                   sea_level_overrides=overrides)
        try_solve(solver)
        dss = S.re_measure(solver)
        caps = S.indicator(sub, "caps")
        eustatic = (-(S.RHO_I / S.RHO_W) * fd.assemble(d_ice * caps * dss)
                    / fd.assemble((1 - caps) * dss))
        out[f"{label}_shift_{representation}"] = float(sea.shift())
        out[f"{label}_closed_{representation}"] = eustatic
        out[f"{label}_converged_{representation}"] = converged(solver)
    return out


def fixed_coastline_rows(meshes, representation):
    """The rows of `test_sea_level.TestFixedOcean` that solve.

    - `saturated`: a land sector with grounded ice in a deep ocean; the
      fixed and the live coastline solve the same problem
      (`test_a_saturated_coastline_gives_the_live_solution`, 1e-8);
    - `ocean_ice`: `GROUNDED` ice over the reference ocean does not change a
      fixed-coastline solve (`test_ice_over_the_ocean_changes_nothing`,
      1e-8, the ocean ice more than 100 times the land ice);
    - `fixed_eustatic`: on a rigid Earth (moduli 1e8, `Lambda = 1e-6`) the
      fixed-coastline shift is `-rho_i int (1 - C0) dI / (rho_w int C0)`
      (`test_the_shift_is_the_eustatic_value`, 1e-6).
    """
    _, sub = meshes
    rep = representation
    out = {}
    states = {}
    fields = S.coastline_fields(sub, "cap", d_ice=1e-2)
    for fixed_ocean in (False, True):
        solver, _, _ = sea_build(meshes, fields, rep,
                                 sea_level_overrides=dict(fixed_ocean=fixed_ocean))
        try_solve(solver)
        out[f"saturated_converged_{fixed_ocean}_{rep}"] = converged(solver)
        states[fixed_ocean] = solver
    out[f"saturated_live_shift_{rep}"] = float(states[False].real_field(S.SHIFT))
    for name, value in field_differences(states[True], states[False]).items():
        out[f"saturated_{name}_{rep}"] = value

    states = {}
    for ocean_ice in (0.0, S.GROUNDED):
        fields = S.coastline_fields(sub, "cap", d_ice=1e-2, ocean_ice=ocean_ice)
        solver, _, _ = sea_build(meshes, fields, rep,
                                 sea_level_overrides=dict(fixed_ocean=True))
        try_solve(solver)
        out[f"ocean_ice_converged_{ocean_ice:g}_{rep}"] = converged(solver)
        states[ocean_ice] = solver
    dss = S.re_measure(states[S.GROUNDED])
    land = S.indicator(sub, "cap")
    out[f"ocean_ice_ocean_mass_{rep}"] = fd.assemble(S.RHO_I * S.GROUNDED * (1 - land) * dss)
    out[f"ocean_ice_land_mass_{rep}"] = fd.assemble(S.RHO_I * 1e-2 * land * dss)
    for name, value in field_differences(states[S.GROUNDED], states[0.0]).items():
        out[f"ocean_ice_{name}_{rep}"] = value

    d_ice = 1e-2
    fields = S.coastline_fields(sub, "caps", d_ice=d_ice, ocean_ice=S.GROUNDED)
    solver, _, sea = sea_build(meshes, fields, rep, lam=1e-6, stiff=True,
                               earth=dict(shear_modulus=1e8, bulk_modulus=1e8),
                               sea_level_overrides=dict(fixed_ocean=True))
    try_solve(solver)
    dss = S.re_measure(solver)
    C0 = S.sheet().ocean_fraction(fields["SL_init"], S.EPS, S.RHO_W, S.RHO_I)
    dI = fields["I"] - fields["I_init"]
    out[f"fixed_eustatic_shift_{rep}"] = float(sea.shift())
    out[f"fixed_eustatic_closed_{rep}"] = (
        -S.RHO_I * fd.assemble((1 - C0) * dI * dss) / (S.RHO_W * fd.assemble(C0 * dss)))
    out[f"fixed_eustatic_converged_{rep}"] = converged(solver)
    return out


def sign_and_repeat_rows(meshes, representation):
    """The sign of the geoid response and the repeatability of a solve.

    - `sign`: on a stiff Earth with ice growing on the two caps, the mean
      sea-level change on the ocean next to the caps exceeds the mean on the
      ocean farthest from them by more than 1e-3 of the eustatic value
      (`test_sea_level.TestSignAgainstPhysics`, with the rigid rotation
      declared as the nullspace);
    - `repeat`: a second solve with nothing changed gives the same `Shift`
      and potential to 1e-8 (`test_sea_level.TestRepeatedSolve`, elastic
      Earth, full `Lambda`); it fails if the datum of the first solve lags;
    - `sink`: in the same elastic solve the surface under the growing ice
      moves down: the mean of `u . n` over the caps is negative. No unit
      test checked this sign in a solve; it is the row that fails when the
      weight of the sea-level sheet enters the displacement row with the
      wrong sign, which every other row here does not see (the stiff Earth
      hides it, and a comparison of two solves carries it on both sides).
    """
    _, sub = meshes
    rep = representation
    out = {}
    d_ice = 1e-2
    fields = S.surface_fields(sub, "caps", d_ice=d_ice)
    # The stiff Earth declares the rigid rotation as its nullspace. With
    # moduli 1e6 the Jacobian annihilates the rotation to 1.8e-12 relative
    # (8e-6 on the default Earth), so without the declaration the system is
    # singular to round-off along it: the outer FGMRES of the iterative
    # low-rank preset stalls between 1e-9 and 6e-9 relative, above its rtol of
    # 1e-12. With the declaration it converges in 4 outer iterations and one
    # Newton step. The rotation does not change `u . n` on Re to first
    # order, so the sea level does not depend on it.
    solver, _, sea = sea_build(meshes, fields, rep, stiff=True, nullspace="rigid_rotation")
    try_solve(solver)
    dss = S.re_measure(solver)
    delta = sea.sea_level() - fields["SL_init"]
    near, far = S.indicator(sub, "near"), S.indicator(sub, "far")
    out[f"sign_near_minus_far_{rep}"] = (
        fd.assemble(near * delta * dss) / fd.assemble(near * dss)
        - fd.assemble(far * delta * dss) / fd.assemble(far * dss))
    out[f"sign_eustatic_{rep}"] = S.RHO_I / S.RHO_W * d_ice
    out[f"sign_converged_{rep}"] = converged(solver)

    solver, z, sea = sea_build(meshes, S.surface_fields(sub, "caps", d_ice=d_ice), rep,
                               earth=S.ELASTIC_EARTH)
    first_converged = try_solve(solver)
    shift_first = float(sea.shift())
    psi_first = z.subfunctions[solver.potential_index].dat.data_ro.copy()
    second_converged = try_solve(solver)
    psi_second = z.subfunctions[solver.potential_index].dat.data_ro
    out[f"repeat_shift_{rep}"] = shift_first
    out[f"repeat_shift_change_{rep}"] = abs(float(sea.shift()) - shift_first)
    out[f"repeat_psi_scale_{rep}"] = np.abs(psi_first).max()
    out[f"repeat_psi_change_{rep}"] = np.abs(psi_second - psi_first).max()
    out[f"repeat_converged_{rep}"] = min(first_converged, second_converged)
    # The surface sinks under added ice: the mean radial displacement of Re
    # under the caps is negative, and large against the root mean square of
    # the radial displacement over the whole of Re.
    dss = S.re_measure(solver)
    caps = S.indicator(sub, "caps")
    u_r = fd.dot(solver.field("displacement"), fd.FacetNormal(solver.mesh))
    out[f"sink_caps_mean_{rep}"] = (fd.assemble(caps * u_r * dss)
                                    / fd.assemble(caps * dss))
    out[f"sink_rms_{rep}"] = float(np.sqrt(fd.assemble(u_r ** 2 * dss)
                                           / fd.assemble(fd.Constant(1.0) * dss)))
    return out


def grounding_line_rows(meshes):
    """The marine grounding line moves by more than half a facet.

    `test_sea_level_sheet.TestSolves::test_the_marine_grounding_line_moves`,
    on the multiplier path: a marine ice sheet thins by 10 percent, and the
    integral of `|W - W_ref|` over Re, the arc over which the water fraction
    changed, exceeds half the length of one of the 32 facets of Re.
    """
    from gadopt import sea_level_sheet as sls
    _, sub = meshes
    solver, _, sea = sea_build(meshes, SH.marine_fields(sub), "multiplier",
                               sea_level_overrides=SH.width())
    try_solve(solver)
    dss = S.re_measure(solver)
    W_ref = sls.water_fraction(sea.I_init, sea.SL_init, S.EPS, S.RHO_W, S.RHO_I)
    return {"grounding_change": fd.assemble(abs(sea.water_fraction() - W_ref) * dss),
            "grounding_facet": fd.assemble(fd.Constant(1.0) * dss) / G.N_AZIMUTHAL,
            "grounding_converged": converged(solver)}


def run_sea(meshes):
    """The rows of the entry `sea`, on both DtN representations where the original ran both."""
    out = {}
    for rep in REPRESENTATIONS:
        out.update(eustatic_rows(meshes, rep))
        out.update(fixed_coastline_rows(meshes, rep))
        out.update(sign_and_repeat_rows(meshes, rep))
    out.update(grounding_line_rows(meshes))
    return out


# -- Entry `mechanics` --------------------------------------------------------------

def rigid_modes(V):
    """The three rigid motions of the annulus on `V`: the rotation and two translations."""
    X = fd.SpatialCoordinate(V.mesh())
    return [fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]])),
            fd.Function(V).interpolate(fd.as_vector([1.0, 0.0])),
            fd.Function(V).interpolate(fd.as_vector([0.0, 1.0]))]


def null_coupling_rows(meshes):
    """`B_mu = 0` with a fluid core is the uncoupled mechanics with the core constraint.

    `test_gia_gravity.py::TestFluidCore::
    test_zero_B_mu_matches_independently_constrained_mechanics` and
    `TestNullCoupling::test_it_reproduces_the_uncoupled_solver`. The
    Quadrature family refuses the rigid core `un = 0` of the second test,
    so both become this one row with a fluid core.

    The GIA solver at `B_mu = 0` (traction `B_MU SIGMA_HAT (cos 2 phi +
    0.25)` on Re, multiplier DtN, direct preset) against
    `CoupledInternalVariableSolver`, which has no gravity, with its history
    on the Quadrature element of the same degree, so that its history
    equation is the pointwise equation that the Quadrature family
    eliminates. The core pressure `p` of the reference is found by
    superposition: the flux `int u . n dS_cmb` is affine in `p`, so two
    solves at `p = 0` and `p = 1` give the `p` with zero flux.

    With a fluid core and `B_mu = 0` nothing fixes a rigid motion of the
    mantle, so block 0 is singular. Both solvers declare the rotation and
    the two translations as their nullspace (the GIA solver without its
    density check, which refuses a translation: at `B_mu > 0` a translation
    moves mass) and the displacement difference is compared after the
    projection of these modes. The strain, the history, the flux and `p` do
    not see a rigid motion.
    """
    from gadopt import CoupledInternalVariableSolver
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)
    load = G.B_MU * G.SIGMA_HAT * (fd.cos(2 * fd.atan2(X[1], X[0])) + fd.Constant(0.25))
    surface_bcs = {G.CURVE_RE: {"normal_stress": load}}
    V = fd.VectorFunctionSpace(sub, "CG", 2)
    basis = fd.VectorSpaceBasis(rigid_modes(V))
    basis.orthonormalize()
    with stop_annotating():
        solver, u, _ = G.FluidCoreBuilder.build_fluid(
            meshes, sheet=False, approximation_kwargs={"B_mu": 0.0},
            dtn_representation="multiplier", extra_bcs=surface_bcs,
            nullspace=basis, check_nullspace=False, solver_parameters="direct",
            **QUADRATURE)
    solver.solve()

    S_q = fd.TensorFunctionSpace(sub, "Quadrature", C.QUAD)
    Zm = fd.MixedFunctionSpace([V, S_q])
    zm = fd.Function(Zm)
    pressure = fd.Constant(0.0)
    w = fd.TestFunctions(Zm)[0]
    n = fd.FacetNormal(sub)
    dss = fd.Measure("ds", domain=sub)(G.CURVE_RC)
    mixed_basis = fd.VectorSpaceBasis(
        [fd.Function(Zm.sub(0)).assign(mode) for mode in rigid_modes(V)])
    mixed_basis.orthonormalize()
    reference = CoupledInternalVariableSolver(
        zm, G.approximation(B_mu=0.0), dt=1.0, bcs=surface_bcs, quad_degree=C.QUAD,
        additional_forcing_term=pressure * fd.dot(w, n) * dss,
        solver_parameters="direct",
        nullspace=fd.MixedVectorSpaceBasis(Zm, [mixed_basis, Zm.sub(1)]))

    def solve_at(value):
        # One solve from rest at the core pressure `value`.
        pressure.assign(value)
        zm.assign(0.0)
        reference.solution_old.assign(0.0)
        reference.solve()
        return zm.copy(deepcopy=True)

    base = solve_at(0.0)
    response = solve_at(1.0)
    response -= base
    base_flux = fd.assemble(fd.dot(base.subfunctions[0], n) * dss)
    response_flux = fd.assemble(fd.dot(response.subfunctions[0], n) * dss)
    p_ref = float(-base_flux / response_flux)
    u_ref = fd.Function(V).assign(base.subfunctions[0] + p_ref * response.subfunctions[0])
    m_ref = fd.Function(S_q).assign(base.subfunctions[1] + p_ref * response.subfunctions[1])

    dx_q = fd.dx(domain=sub, metadata={"quadrature_degree": C.QUAD})

    def relative(a, b):
        # The L2 norm of `a - b` over the norm of `b`, at the history's points.
        return float(np.sqrt(fd.assemble(fd.inner(a - b, a - b) * dx_q)
                             / fd.assemble(fd.inner(b, b) * dx_q)))

    du = fd.Function(V).assign(u - u_ref)
    apply_kernel_projection(basis, du)
    core = [c for c in solver.couplings if isinstance(c, FluidCore)][0]
    return {"null_converged": converged(solver),
            "null_flux": fd.assemble(fd.dot(u, n) * dss),
            "null_flux_reference": fd.assemble(fd.dot(u_ref, n) * dss),
            "null_response_flux": response_flux,
            "null_pressure": float(core.pressure()),
            "null_pressure_reference": p_ref,
            "null_strain": relative(fd.sym(fd.grad(u)), fd.sym(fd.grad(u_ref))),
            "null_history": relative(history_slices(solver.internal_variables)[0], m_ref),
            "null_displacement": fd.norm(du) / fd.norm(u_ref),
            "null_u_norm": fd.norm(u_ref)}


def fluid_core_rows(meshes):
    """The fluid-core annulus under the `cos 2 phi` sheet: core volume and degree.

    One solve of `TestFluidCore.build_fluid` with rotation, on the
    Quadrature family. Two rows read it:

    - `cmb`: `|int u . n dS_cmb| <= 1e-10 sqrt(int (u . n)^2 int 1) + 1e-13`
      (`TestFluidCore::test_solve_enforces_zero_integrated_cmb_displacement`);
    - `degree`: a `cos 2 phi` load excites the `cos 2 phi` trace coefficient
      of each DtN boundary, above 1e-6, and every other coefficient stays
      below 1e-9 of it plus 1e-15 (`test_gia_gravity.py::
      test_the_answer_is_degree_two`, which ran with a rigid core; the
      Quadrature family needs the fluid core).
    """
    with stop_annotating():
        solver, u, _ = G.FluidCoreBuilder.build_fluid(meshes, rotation=True, **QUADRATURE)
    solver.solve()
    core = [c for c in solver.couplings if isinstance(c, FluidCore)][0]
    dss = core.measure()
    un = fd.dot(u, fd.FacetNormal(solver.mesh))
    out = {"cmb_converged": converged(solver),
           "cmb_flux": fd.assemble(un * dss),
           "cmb_scale": float(np.sqrt(fd.assemble(un * un * dss)
                                      * fd.assemble(fd.Constant(1.0) * dss))),
           "cmb_u_norm": fd.norm(u)}
    # The largest ratio of an inactive coefficient to the active one, over
    # both DtN boundaries, and the smallest active coefficient.
    ratios, active = [], []
    for modes in solver.coefficients().values():
        cos2 = abs(float(modes["cos2"]))
        active.append(cos2)
        ratios += [(abs(float(value)) - 1e-15) / cos2
                   for key, value in modes.items() if key != "cos2"]
    out["degree_active_min"] = min(active)
    out["degree_inactive_ratio_max"] = max(ratios)
    return out


def centre_of_mass_rows(meshes, representation):
    """After a solve with a degree-1 load the first mass moment `D` is zero.

    `test_gia_gravity.py::TestCentreOfMassFrame::
    test_solve_holds_the_centre_of_mass`: the frame configuration (fluid
    core, a degree-1 sheet on Re), `|D_i| < 1e-8` of the moment of the load
    sheet `SIGMA_HAT pi Re^2`. The low-rank arm uses the iterative preset
    with the dense Schur complement on the `Real` block and tight
    tolerances, as the original.
    """
    kwargs = {}
    if representation == "lowrank":
        kwargs = dict(solver_parameters="iterative", preset_options=dict(
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
            outer_rtol=1e-12, block0_rtol=1e-4, snes_rtol=1e-10))
    with stop_annotating():
        solver, _, _ = G.CentreOfMassFrame().build(
            meshes, load_degree=1, dtn_representation=representation,
            **QUADRATURE, **kwargs)
    solver.solve()
    centre = [c for c in solver.couplings if isinstance(c, CentreOfMass)][0]
    D = [fd.assemble(form) for form in centre.dipole()]
    rep = representation
    return {f"com_dipole_0_{rep}": D[0], f"com_dipole_1_{rep}": D[1],
            f"com_load_moment_{rep}": G.SIGMA_HAT * np.pi * G.RE ** 2,
            f"com_multipliers_finite_{rep}": float(all(
                np.isfinite([float(lam) for lam in centre.multipliers()]))),
            f"com_converged_{rep}": converged(solver)}


def rotation_mode_rows(meshes):
    """The declared rigid rotation is absent from the answer.

    `test_gia_gravity.py::TestRigidRotationNullspace::
    test_declaring_it_removes_it_from_the_answer`, on the fluid-core annulus
    (the Quadrature family refuses the rigid core): the content of the mode
    `(-y, x)` in the displacement, `|c| |r0| / |u|` with `c` the L2
    projection coefficient, is below 1e-14 with the mode declared and below
    1e-3 of the content without it.
    """
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)

    def rotation_content(u):
        # The fraction of `u` along the rigid rotation, in the L2 norm.
        r0 = fd.Function(u.function_space()).interpolate(fd.as_vector([-X[1], X[0]]))
        dxm = fd.Measure("dx", domain=sub)
        c = fd.assemble(fd.dot(u, r0) * dxm) / fd.assemble(fd.dot(r0, r0) * dxm)
        return abs(c) * fd.norm(r0) / fd.norm(u)

    out = {}
    for label, nullspace in (("plain", None), ("declared", "rigid_rotation")):
        with stop_annotating():
            solver, u, _ = G.FluidCoreBuilder.build_fluid(meshes, nullspace=nullspace,
                                                          **QUADRATURE)
        solver.solve()
        out[f"rotation_content_{label}"] = rotation_content(u)
        out[f"rotation_converged_{label}"] = converged(solver)
    return out


def accepted_state_rows(meshes):
    """When `solve` returns, the accepted state is the solved state, bit for bit.

    `test_gia_gravity.py::TestAcceptedStateAfterAStep`, on the Quadrature
    family (fluid core, multiplier DtN), with and without the declared
    rigid rotation: the accepted history level `internal_variables` equals
    the history the step solved (`solved_history`), and `u`, `psi` equal the
    displacement and the potential of the mixed solution. Each difference
    is the l2 norm of the difference vector; it must be exactly 0.
    """
    out = {}
    for label, nullspace in (("plain", None), ("declared", "rigid_rotation")):
        with stop_annotating():
            solver, u, psi = G.FluidCoreBuilder.build_fluid(
                meshes, dtn_representation="multiplier", nullspace=nullspace,
                **QUADRATURE)
        solver.solve()
        z, index = solver.solution, solver.field_index
        pairs = (("history", solver.internal_variables, solver.solved_history()),
                 ("displacement", u, z.subfunctions[index["displacement"]]),
                 ("potential", psi, z.subfunctions[index["potential"]]))
        for name, accepted, solved in pairs:
            with accepted.dat.vec_ro as a, solved.dat.vec_ro as b:
                difference = a.copy()
                difference.axpy(-1.0, b)
                out[f"accepted_{name}_difference_{label}"] = difference.norm()
                out[f"accepted_{name}_norm_{label}"] = b.norm()
    return out


def run_mechanics(meshes):
    """The rows of the entry `mechanics`."""
    out = {}
    out.update(null_coupling_rows(meshes))
    out.update(fluid_core_rows(meshes))
    for rep in REPRESENTATIONS:
        out.update(centre_of_mass_rows(meshes, rep))
    out.update(rotation_mode_rows(meshes))
    out.update(accepted_state_rows(meshes))
    return out


# -- Entry `rheology` ---------------------------------------------------------------

def newtonian_limit_rows(meshes):
    """A power law that is Newtonian solves the Newtonian problem.

    `test_gia_gravity.py::TestNewtonianLimit`, on the fluid-core annulus
    with the `cos 2 phi` sheet (the Quadrature family refuses the rigid core
    of the original), direct preset:

    - `exponent_one`: exponent 1 with a transition stress 1e-4 and a
      background stress 0.5 gives the Newtonian displacement, history and
      potential to 1e-12 (the `0^0` guard of `power_law_factor` makes the
      factor exactly 1);
    - `near_one`: exponent `1 + 1e-6` with background stress equal to the
      transition stress 1e-3 gives the Newtonian displacement to 1e-6
      (`|f - 1| <= (eps / 2) ln(1 + sigma_max / sigma*)`);
    - `zero_background`: the same exponent with background stress 0 differs
      from the Newtonian displacement by more than 0.1, because the factor
      tends to 1/2 as the exponent approaches 1 from above.
    """
    _, sub = meshes
    dx_q = fd.dx(domain=sub, metadata={"quadrature_degree": C.QUAD})

    def solve(**approximation_kwargs):
        # One step of the fluid-core annulus with the given power-law settings.
        with stop_annotating():
            solver, u, psi = G.FluidCoreBuilder.build_fluid(
                meshes, solver_parameters="direct",
                approximation_kwargs=approximation_kwargs or None, **QUADRATURE)
        solver.solve()
        return solver, u, psi

    def relative(a, b, measure=None):
        # The L2 norm of `a - b` over the norm of `b`.
        if measure is None:
            return fd.norm(a - b) / fd.norm(b)
        return float(np.sqrt(fd.assemble(fd.inner(a - b, a - b) * measure)
                             / fd.assemble(fd.inner(b, b) * measure)))

    plain, u_p, psi_p = solve()
    m_p = history_slices(plain.internal_variables)[0]
    power, u_q, psi_q = solve(exponent=1, transition_stress=1e-4, background_stress=0.5)
    m_q = history_slices(power.internal_variables)[0]
    out = {"exponent_one_u": relative(u_q, u_p),
           "exponent_one_history": relative(m_q, m_p, dx_q),
           "exponent_one_psi": relative(psi_q, psi_p),
           "exponent_one_u_norm": fd.norm(u_p),
           "exponent_one_history_norm": float(np.sqrt(fd.assemble(fd.inner(m_p, m_p) * dx_q)))}
    near, u_n, _ = solve(exponent=1 + 1e-6, transition_stress=1e-3, background_stress=1e-3)
    out["near_one_u"] = relative(u_n, u_p)
    out["near_one_converged"] = converged(near)
    out["near_one_newtonls"] = float(near.solver_parameters["snes_type"] == "newtonls")
    _, u_z, _ = solve(exponent=1 + 1e-6, transition_stress=1e-3, background_stress=0.0)
    out["zero_background_u"] = relative(u_z, u_p)
    return out


#: The stiff power-law march: three Maxwell elements (shear moduli and
#: Maxwell times; the viscosity of element j is `mu_j tau_j`), ten times
#: faster than the Exp10 regime, so `dt / tau_1 = 5` before the power law;
#: exponent 3.5 in the cells between 0.2 and 0.8 below Re and 1 elsewhere;
#: transition stress 1e-2; the shoreline state of `test_sea_level.py`
#: rotated by 0.5, with 10 percent of the initial ice melting at each of
#: the `STIFF_STEPS` steps of length 1; softplus width 0.4.
STIFF_SHEAR_MODULI = (0.5, 0.3, 0.2)
STIFF_MAXWELL_TIMES = (0.2, 2.0, 20.0)
STIFF_EXPONENT = 3.5
STIFF_LAYER = (0.2, 0.8)
STIFF_TRANSITION_STRESS = 1e-2
STIFF_PHASE = 0.5
STIFF_MELT_PER_STEP = 0.1
STIFF_SOFTPLUS_WIDTH = 0.4


def stiff_power_law_march(meshes):
    """Every step of a stiff power-law march with a moving coastline converges.

    The configuration of `test_newton_converges_on_a_stiff_power_law` (the
    power-law gate of the GIAMIP experiments 10 and 12) on the Quadrature
    family: three Maxwell elements, a layered power law that lowers the
    viscosity by up to about 100, the fluid core, rotation, the centre of
    mass and the sea level with a moving coastline, the low-rank DtN under
    the iterative preset with `newtonls`, BDF2. The original measured 5, 5
    and 6 Newton iterations and bounded each step by 10.

    Returns:
      `{stiff_newton_<k>, stiff_reason_<k>}` for the steps `k = 1, 2, 3`.
    """
    parent, sub = meshes
    frame = G.CentreOfMassFrame()
    Q = fd.FunctionSpace(sub, "DG", 0)
    X = fd.SpatialCoordinate(sub)
    r = fd.sqrt(fd.dot(X, X))
    with stop_annotating():
        # The exponent per cell, exactly 3.5 or exactly 1 at the centroid
        # depth, so that the `n = 1` guard of the power law holds exactly.
        depth = fd.Function(Q).interpolate(G.RE - r).dat.data_ro
        exponent = fd.Function(Q)
        exponent.dat.data[:] = np.where(
            (depth > STIFF_LAYER[0]) & (depth < STIFF_LAYER[1]), STIFF_EXPONENT, 1.0)
        approximation = gadopt.CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=1.0,
            shear_modulus=[fd.Function(Q).assign(mu) for mu in STIFF_SHEAR_MODULI],
            viscosity=[fd.Function(Q).assign(mu * tau)
                       for mu, tau in zip(STIFF_SHEAR_MODULI, STIFF_MAXWELL_TIMES)],
            bulk_shear_ratio=1, g=frame.gravity_of_r(r), B_mu=G.B_MU,
            self_gravity_number=G.LAMBDA, exponent=exponent,
            transition_stress=fd.Function(Q).assign(STIFF_TRANSITION_STRESS),
            background_stress=0.0)
        V2 = fd.FunctionSpace(sub, "CG", 2)
        phi = fd.atan2(X[1], X[0])
        SL_init = fd.Function(V2).interpolate(fd.cos(phi - STIFF_PHASE) + 0.2)
        I_init = fd.Function(V2).interpolate(
            0.6 * fd.max_value(0.0, -fd.cos(phi - STIFF_PHASE) - 0.4))
        ice = fd.Function(V2).assign(I_init)
        sea = SeaLevel(boundary=G.CURVE_RE, rho_w=S.RHO_W, rho_i=S.RHO_I,
                       g_surface=frame.gravity_of_r(G.RE), SL_init=SL_init, I=ice,
                       I_init=I_init, N_init=fd.Constant(0.0), ur_init=fd.Constant(0.0),
                       softplus_width=STIFF_SOFTPLUS_WIDTH, fixed_ocean=False)
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        dt = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0)
        solver = gadopt.SelfGravitatingGIASolver(
            fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
            fd.Function(fd.FunctionSpace(parent, "CG", 2)), approximation, dt=dt,
            gravity_bcs=G.gravity_bcs(parent, sheet=False), bcs={},
            couplings=[FluidCore(boundary=G.CURVE_RC, rho_core=frame.RHO_CORE,
                                 g=fd.Constant(frame.gravity_of_r(G.RC))),
                       Rotation(C=fd.assemble(fd.dot(X, X) * dx_m),
                                Omega_sq=OMEGA_SQ_EARTH),
                       CentreOfMass(), sea],
            dtn_representation="lowrank", time_scheme="bdf2",
            nullspace="rigid_rotation", solver_parameters="iterative",
            preset_options=dict(outer_rtol=1e-11, block0_rtol=1e-4, snes_rtol=1e-10,
                                snes_type="newtonls"),
            solver_parameters_extra={"snes_atol": 1e-14}, **QUADRATURE)
    out = {}
    for k in range(1, STIFF_STEPS + 1):
        ice.assign((1.0 - k * STIFF_MELT_PER_STEP) * I_init)
        reason = 0
        try:
            solver.solve()
            reason = solver.solver.snes.getConvergedReason()
        except fd.ConvergenceError:
            reason = solver.solver.snes.getConvergedReason()
        out[f"stiff_newton_{k}"] = solver.solver.snes.getIterationNumber()
        out[f"stiff_reason_{k}"] = reason
        out[f"stiff_min_power_factor_{k}"] = M._min_power_factor(solver)
        if reason <= 0:
            break
    return out


def run_rheology(meshes):
    """The rows of the entry `rheology`."""
    out = {}
    out.update(newtonian_limit_rows(meshes))
    out.update(stiff_power_law_march(meshes))
    return out


RUNS = {"sea": run_sea, "mechanics": run_mechanics, "rheology": run_rheology}
assert set(RUNS) == set(LIMITS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(LIMITS))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    values = RUNS[name](unit_meshes())
    M.write_values(CASE_DIR / limits_values(name), values)
