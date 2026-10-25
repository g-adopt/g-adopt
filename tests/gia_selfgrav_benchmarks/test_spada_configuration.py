"""Small checks for Spada load choices and independent diagnostic cutoffs.

Run with the Firedrake environment; these do not run a GIA time integration.
"""

from pathlib import Path

import numpy as np
import pytest
import spada
from firedrake import (
    Function,
    FunctionSpace,
    SpatialCoordinate,
    UnitCubeMesh,
    UnitOctahedralSphereMesh,
    VectorFunctionSpace,
    as_vector,
)


def test_independent_cli_cutoffs():
    defaults = spada.parse_args([])
    assert defaults.load_nmax == defaults.diagnostic_nmax == 10
    assert defaults.load_shape == "cap"
    assert defaults.bulk_shear_ratio == 1000.0
    assert defaults.dtn_degree == 5
    assert defaults.outer_rtol == 1.0e-6
    assert defaults.block0_rtol == 1.0e-4
    assert defaults.load_representation == "spatial"
    assert spada.run_settings(defaults)["load_nmax"] is None
    disc = spada.parse_args(["--case", "polar-motion", "--load_shape", "disc"])
    assert spada.run_settings(disc)["load_shape"] == "disc"
    args = spada.parse_args(["--load_representation", "spectral", "--load_nmax", "128"])
    assert args.load_nmax == 128
    assert args.diagnostic_nmax == 10
    assert args.load_representation == "spectral"
    args = spada.parse_args(["--load_nmax", "20", "--diagnostic_nmax", "40"])
    assert spada.run_settings(args)["diagnostic_nmax"] == 40


@pytest.mark.parametrize("option", ["--load_nmax", "--diagnostic_nmax", "--dtn_degree"])
def test_invalid_cutoff(option):
    with pytest.raises(SystemExit):
        spada.parse_args([option, "1"])


@pytest.mark.parametrize("shape", ["cap", "disc"])
def test_reference_above_spectral_load_cutoff(shape):
    sigma = spada.diagnostic_load_coefficients(
        40, load_nmax=10, load_shape=shape, load_representation="spectral")
    reference = spada.TabooReference()
    for values in reference.coefficients(1.0, sigma, 40):
        np.testing.assert_array_equal(values[9:], 0.0)  # degrees 11..40
    np.testing.assert_array_equal(sigma[:2], 0.0)
    np.testing.assert_array_equal(sigma[2:11], spada.load_coefficients(10, shape)[2:])


@pytest.mark.parametrize("shape", ["cap", "disc"])
def test_spatial_reference_retains_high_degrees(shape):
    sigma = spada.diagnostic_load_coefficients(
        40, load_nmax=10, load_shape=shape)
    np.testing.assert_array_equal(sigma[2:], spada.load_coefficients(40, shape)[2:])
    settings = spada.run_settings(spada.parse_args(["--load_nmax", "128"]))
    assert settings["load_nmax"] is None
    assert settings["load_representation"] == "spatial"


@pytest.mark.parametrize("load_nmax,diagnostic_nmax", [(128, 10), (10, 20)])
@pytest.mark.parametrize("shape", ["cap", "disc"])
def test_comparison_uses_diagnostic_cutoff(monkeypatch, load_nmax, diagnostic_nmax, shape):
    reference = spada.TabooReference()
    sigma = spada.diagnostic_load_coefficients(diagnostic_nmax, load_nmax=load_nmax,
                                               load_shape=shape, load_representation="spectral")
    expected = reference.coefficients(0., sigma, diagnostic_nmax)
    calls = []

    def spectra(solver, parent, mantle, nmax, **kwargs):
        calls.append(nmax)
        # A model equal to the reference, including unloaded diagnostic modes.
        return tuple(np.concatenate(([0., 0.], values)) for values in expected)

    monkeypatch.setattr(spada, "surface_spectra", spectra)
    monkeypatch.setattr(spada, "log", lambda *args: None)
    with np.errstate(divide="raise", invalid="raise"):
        row = spada.compare_cap(0., None, None, None, reference,
                                load_nmax=load_nmax, diagnostic_nmax=diagnostic_nmax,
                                load_shape=shape, load_representation="spectral")
    assert calls == [diagnostic_nmax]
    np.testing.assert_array_equal(row["degrees"], np.arange(2, diagnostic_nmax + 1))
    for quantity, values in zip(["U", "V", "N"], expected):
        np.testing.assert_array_equal(row[f"{quantity}_n"], values)
        np.testing.assert_array_equal(row[f"{quantity}_n_ref"], values)


def test_analytic_cap_low_modes():
    # Independent closed-form integrals of the cap's mass and dipole.
    delta = 1 - np.cos(np.deg2rad(spada.CAP_RADIUS_DEG))
    amplitude = spada.RHO_ICE * spada.CAP_THICKNESS_M
    expected = [amplitude * delta / 3,
                amplitude * delta * (1 - 2 * delta / 5)]
    np.testing.assert_allclose(spada.cap_load(1), expected, rtol=1e-11)


def test_reference_range():
    reference = spada.TabooReference()
    with pytest.raises(ValueError, match="TABOO reference supports"):
        reference.love_time(0.0, reference.nmax + 1)


def test_checkpoint_records_distinguish_loads():
    spectral = spada.run_settings(spada.parse_args(
        ["--load_representation", "spectral", "--load_nmax", "128"]))
    spatial = spada.run_settings(spada.parse_args([]))
    assert spectral != spatial
    assert spada.checkpoint_run_settings({"run": spectral}) == spectral
    legacy = {"run": {"case": "cap", "bulk_shear_ratio": 100., "smoke": False},
              "rows": [{"degrees": list(range(2, 41))}]}
    upgraded = spada.checkpoint_run_settings(legacy)
    assert upgraded["load_nmax"] == upgraded["diagnostic_nmax"] == 40
    assert upgraded["load_representation"] == "spectral"
    assert upgraded["load_shape"] == "cap"
    assert upgraded["dtn_degree"] == 5
    assert upgraded["load_representation"] != spatial["load_representation"]
    # Existing records with split cutoffs also predate the shape option.
    old_spectral = {k: v for k, v in spectral.items()
                    if k not in ("load_shape", "dtn_degree", "outer_rtol", "block0_rtol")}
    assert spada.checkpoint_run_settings({"run": old_spectral}) == spectral
    disc = spada.run_settings(spada.parse_args(["--load_shape", "disc"]))
    assert disc != spada.run_settings(spada.parse_args([]))


def test_independent_dtn_degree():
    baseline = spada.run_settings(spada.parse_args([]))
    coarse = spada.run_settings(spada.parse_args(["--dtn_degree", "2"]))
    assert coarse == {**baseline, "dtn_degree": 2}
    assert spada.checkpoint_run_settings({"run": coarse}) == coarse
    assert spada.checkpoint_run_settings({"run": baseline}) != coarse


def test_coarse_solver_tolerances():
    baseline = spada.run_settings(spada.parse_args([]))
    coarse = spada.run_settings(spada.parse_args(
        ["--outer_rtol", "1e-5", "--block0_rtol", "1e-3"]))
    assert coarse == {**baseline, "outer_rtol": 1.0e-5, "block0_rtol": 1.0e-3}
    assert spada.checkpoint_run_settings({"run": coarse}) == coarse
    legacy = {k: v for k, v in baseline.items() if k not in ("outer_rtol", "block0_rtol")}
    assert spada.checkpoint_run_settings({"run": legacy}) == baseline
    assert spada.checkpoint_run_settings({"run": legacy}) != coarse


def test_study_options_and_restart_compatibility():
    from spada_study import run_matrix

    for run in run_matrix():
        args = spada.parse_args(run["arguments"])
        settings = spada.run_settings(args)
        for key, value in run["settings"].items():
            assert settings[key] == value
        assert args.write_surface_output == run["surface_output"]
        assert args.smoke == run["smoke"]
        assert settings["load_centre_deg"] == run["load_centre_deg"]
    baseline = spada.run_settings(spada.parse_args([]))
    legacy = {key: value for key, value in baseline.items()
              if key not in ("timestep_scale", "mesh_h_km", "litho_layers")}
    assert spada.checkpoint_run_settings({"run": legacy}) == baseline
    coarse = spada.run_settings(spada.parse_args(["--mesh_h_km", "500", "--litho_layers", "1"]))
    assert spada.checkpoint_run_settings({"run": legacy}, {"h_km": 500., "litho_layers": 1}) == coarse
    assert coarse != baseline


@pytest.mark.parametrize("scale,steps", [(1, 138), (2, 69)])
def test_timestep_scaling_preserves_output_epochs(scale, steps):
    args = spada.parse_args(["--timestep_scale", str(scale)])
    segments = spada.time_segments(args, spada.EPOCHS_KYR)
    assert sum(segment[3] for segment in segments) == steps
    assert [segment[1]/1000 for segment in segments if segment[4]] == list(spada.EPOCHS_KYR[1:])
    assert segments[-1][1] == 20000.
    assert segments[0][2] == 10.*scale


@pytest.mark.parametrize("option,value", [("--mesh_h_km", "0"), ("--mesh_h_km", "nan"),
                                          ("--timestep_scale", "-1"), ("--timestep_scale", "inf"),
                                          ("--litho_layers", "0"), ("--litho_layers", "1.5")])
def test_invalid_mesh_and_timestep_options(option, value):
    with pytest.raises(SystemExit):
        spada.parse_args([option, value])


@pytest.mark.parametrize("option", ["--outer_rtol", "--block0_rtol"])
@pytest.mark.parametrize("value", ["0", "-1", "1", "nan", "inf"])
def test_invalid_solver_tolerance(option, value):
    with pytest.raises(SystemExit):
        spada.parse_args([option, value])


@pytest.mark.parametrize("n_real", [1, 4])
def test_coarse_solver_keeps_cached_schur(n_real):
    from types import SimpleNamespace

    layout = SimpleNamespace(condensed=False, real_fields=tuple(range(n_real)))
    baseline = spada.common.solver_parameters(layout, "ksponly")
    coarse = spada.common.solver_parameters(
        layout, "ksponly", outer_rtol=1.0e-5, block0_rtol=1.0e-3, ainvb=True)
    assert baseline["dtn_schur_ainvb"] is coarse["dtn_schur_ainvb"] is True
    assert baseline.keys() == coarse.keys()
    changed = {k for k in baseline if baseline[k] != coarse[k]}
    assert changed == {"ksp_rtol", "dtn_fieldsplit_0_condensed_ksp_rtol"}


def test_rotating_profile_defaults():
    args = spada.parse_args(["--case", "polar-motion", "--write_profiles"])
    geometry = spada.profile_geometry(args.case, spada.LOAD_CENTRE_DEG[args.case],
                                      longitudes=args.profile_longitudes,
                                      bearings=args.profile_bearings)
    assert geometry["profile_kind"] == "load-centred"
    np.testing.assert_array_equal(geometry["bearing_deg"], [0, 90])


def test_profile_writer_samples_vector_components(tmp_path, monkeypatch):
    from types import SimpleNamespace

    mesh = UnitCubeMesh(2, 2, 2)
    tmp_path = Path(mesh.comm.bcast(str(tmp_path), root=0))
    monkeypatch.setattr(spada.common, "RE", .5)
    monkeypatch.setattr(spada.common, "D_SCALE", 2.)
    monkeypatch.setattr(spada.common, "gravity_exact_ufl", lambda r: spada.Constant(1.))
    geometry = spada.profile_geometry("polar-motion", (55, 45), extent_deg=5, spacing_deg=5)
    profiles = spada.build_profile_samplers(mesh, mesh, 1e-6, geometry)
    displacement = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
        as_vector((1., 2., 3.)))
    potential = Function(FunctionSpace(mesh, "CG", 1)).assign(4.)
    path = tmp_path / "profiles.npz"
    spada.write_profiles(path, SimpleNamespace(displacement=displacement, potential=potential),
                          mesh, profiles, 1.)
    mesh.comm.barrier()
    with np.load(path, allow_pickle=False) as data:
        assert np.all(data["found"])
        for key, basis in [("U_m", "unit_vectors"), ("V_m", "south_vectors"),
                           ("W_m", "east_vectors"), ("along_m", "tangent_vectors"),
                           ("across_m", "right_vectors")]:
            expected = geometry[basis] @ np.array([2., 4., 6.])
            np.testing.assert_allclose(data[key], expected, atol=1e-12)
        np.testing.assert_allclose(data["N_m"], 8.)
        np.testing.assert_array_equal(data["centre_deg"], [55., 45.])


@pytest.mark.parametrize("representation", ["spectral", "spatial"])
@pytest.mark.parametrize("case", ["cap", "polar-motion"])
@pytest.mark.parametrize("shape", ["cap", "disc"])
def test_load_interpolation(representation, case, shape):
    mesh = UnitOctahedralSphereMesh(refinement_level=1, degree=2)
    axis = spada.load_axis(case)
    scale = spada.common.RHO_BAR * spada.common.D_SCALE
    # A non-default cutoff checks that load_field follows the supplied
    # coefficients, rather than the default diagnostic degree.
    sigma = spada.load_coefficients(20 if representation == "spectral" else 1, shape) / scale
    field = spada.load_field(mesh, sigma, axis, representation=representation, load_shape=shape)
    coordinates = Function(VectorFunctionSpace(mesh, "CG", 2)).interpolate(
        SpatialCoordinate(mesh)).dat.data_ro
    cosine = coordinates @ axis / np.linalg.norm(coordinates, axis=1)
    if representation == "spectral":
        coefficients = sigma.copy()
        coefficients[:2] = 0
        expected = np.polynomial.legendre.legval(cosine, coefficients)
    else:
        cos_alpha = np.cos(np.deg2rad(spada.CAP_RADIUS_DEG))
        if shape == "cap":
            density = (spada.RHO_ICE * spada.CAP_THICKNESS_M / scale
                       * np.sqrt(np.maximum((cosine - cos_alpha) / (1 - cos_alpha), 0)))
        else:
            density = (spada.RHO_ICE * spada.DISC_THICKNESS_M / scale
                       * (cosine >= cos_alpha))
        expected = density - sigma[0] - sigma[1] * cosine
    np.testing.assert_allclose(field.dat.data_ro, expected, rtol=1e-11, atol=1e-16)
    assert field.function_space() == FunctionSpace(mesh, "CG", 2)


def test_disc_polar_motion_reference():
    reference = spada.TabooReference()
    scale = spada.polar_motion_scale("disc")
    for t in spada.EPOCHS_KYR:
        np.testing.assert_allclose(reference.polar_motion(t, "disc"),
                                   np.array(reference.polar_motion(t)) * scale)


def test_load_location_is_independent_of_rotation():
    off = spada.parse_args(["--case", "cap", "--load_centre", "25", "-285"])
    on = spada.parse_args(["--case", "polar-motion"])
    assert off.load_centre == on.load_centre == [25., 75.]
    np.testing.assert_allclose(spada.load_axis(off.case, off.load_centre),
                               spada.load_axis(on.case, on.load_centre))
    baseline = spada.run_settings(spada.parse_args([]))
    assert spada.run_settings(off) == {**baseline, "load_centre_deg": [25., 75.]}
    legacy = {k: v for k, v in baseline.items() if k != "load_centre_deg"}
    assert spada.checkpoint_run_settings({"run": legacy}) == baseline
    assert spada.checkpoint_run_settings({"run": legacy}) != spada.run_settings(off)


@pytest.mark.parametrize("centre", [(181, 75), (-1, 75), (25, "nan"), ("inf", 75)])
def test_invalid_load_location(centre):
    with pytest.raises(SystemExit):
        spada.parse_args(["--load_centre", *map(str, centre)])


def test_off_axis_projection_uses_load_axis():
    mesh = UnitCubeMesh(2, 2, 2)
    x = SpatialCoordinate(mesh)
    axis = spada.load_axis("cap", [25., 75.])
    cosine = spada.dot(x, spada.Constant(axis)) / spada.sqrt(spada.dot(x, x))
    field = (3*cosine**2 - 1) / 2
    coefficients = spada.project_surface(field, mesh, spada.ds(domain=mesh), False,
                                         nmax=2, axis=axis)
    assert coefficients[2] == pytest.approx(1., abs=1e-12)
    wrong_axis = spada.project_surface(field, mesh, spada.ds(domain=mesh), False, nmax=2)
    assert abs(wrong_axis[2] - 1) > .1


def test_reference_polar_motion_rotates_with_load():
    ref = spada.TabooReference()
    mx, my = ref.polar_motion(1.)
    np.testing.assert_allclose(ref.polar_motion(1., load_centre=[25, 165]), [-my, mx])
    np.testing.assert_allclose(ref.polar_motion(1., load_centre=[155, 75]), [-mx, -my])
    np.testing.assert_allclose(ref.polar_motion(1., load_centre=[0, 75]), [0, 0], atol=1e-15)


def test_lod_conversion_and_epoch_export(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(spada, "log", lambda *args: None)
    solver = SimpleNamespace(rotation_values=lambda: {"m1": 1e-4, "m2": -2e-4, "m3": -1e-8})
    values = spada.rotation_diagnostics(solver)
    assert values["m3"] == -1e-8
    assert values["lod_ms"] == pytest.approx(.864)
    row = spada.compare_polar_motion(1., solver, spada.TabooReference())
    assert row["lod_ms"] == values["lod_ms"]
    assert row["lod_eq33_ref_ms"] > 0.
    assert row["phase_ref"] == pytest.approx(-105.)
    zero = spada.compare_polar_motion(1., solver, spada.TabooReference(), load_centre=[0, 0])
    assert zero["phase_ref"] is None


def test_lod_eq33_reproduces_independent_gs_archive_value():
    reference = spada.TabooReference()
    # giamip converted T02-03 length_of_day/cap/GS, first row. Its source
    # is lod_GS.dat, SHA256 29f630465a57b165b9624fb31c084d57c3c795952647e56697d0bcab906e5290.
    assert reference.length_of_day_eq33(.001) == pytest.approx(56.9639388465671, rel=1e-7)
    for t in (0., 1., 20.):
        assert reference.length_of_day_eq33(t, "disc") == pytest.approx(
            reference.length_of_day_eq33(t) * spada.polar_motion_scale("disc"))


def test_profile_rates_use_adjacent_steps_and_survive_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from firedrake import Constant, MixedFunctionSpace

    monkeypatch.setattr(spada.common, "RE", .5)
    monkeypatch.setattr(spada.common, "D_SCALE", 2.)
    monkeypatch.setattr(spada.common, "gravity_exact_ufl", lambda r: Constant(1.))
    mesh = UnitCubeMesh(2, 2, 2)
    tmp_path = Path(mesh.comm.bcast(str(tmp_path), root=0))
    space = MixedFunctionSpace([VectorFunctionSpace(mesh, "CG", 1),
                                FunctionSpace(mesh, "CG", 1)])
    geometry = spada.profile_geometry("polar-motion", (55, 45), extent_deg=5, spacing_deg=5)
    samplers = spada.build_profile_samplers(mesh, mesh, 1e-6, geometry)
    segments = [(0., 10., 10., 1, True), (10., 20., 5., 2, True)]

    def run(start_step, initial_time):
        solution, old = Function(space), Function(space)
        u, phi = solution.subfunctions
        dt = Constant(spada.common.DT_ELASTIC)
        clock = [initial_time]

        def assign(t_yr):
            f = 1 + .1*t_yr + .001*t_yr**2
            u.assign(Constant((f, 2*f, 3*f)))
            phi.assign(4*f)

        if start_step:
            assign(initial_time)

        def solve():
            clock[0] = 0. if clock[0] is None else clock[0] + float(dt)*spada.common.T_BAR_YR
            assign(clock[0])
            old.assign(solution)

        solver = SimpleNamespace(solution=solution, solution_old=old,
                                  displacement=u, potential=phi, solve=solve)
        history = spada.ProfileRateHistory(
            solver, initial_time=None if initial_time is None else initial_time/1000.)
        paths = []

        def on_epoch(t):
            path = tmp_path / f"{start_step}_{t:g}.npz"
            spada.write_profiles(path, solver, mesh, samplers, t, rates=history)
            mesh.comm.barrier()
            paths.append(path)

        spada.common.march(solver, dt, segments, elastic=True, start_step=start_step,
                           before_step=lambda t: history.before_step(solver, t),
                           on_step=lambda t, *args: history.after_step(solver, t),
                           on_epoch=on_epoch)
        return paths

    full = run(0, None)
    with np.load(full[0]) as data:
        assert not data["rate_valid"]
        assert np.all(np.isnan(data["U_m_per_yr"]))
    for path, dt_yr in zip(full[1:], [10., 5.]):
        with np.load(path) as data:
            assert data["rate_valid"]
            assert data["rate_dt_yr"] == pytest.approx(dt_yr)
            factor = 1 + .01 * (data["rate_interval_start_kyr"]
                                + data["rate_interval_end_kyr"]) * 1000
            for key, basis in [("U", "unit_vectors"), ("V", "south_vectors"),
                               ("W", "east_vectors"), ("along", "tangent_vectors"),
                               ("across", "right_vectors")]:
                np.testing.assert_allclose(data[key+"_m_per_yr"],
                                           geometry[basis] @ (factor*np.array([.2, .4, .6])),
                                           atol=1e-12)
            np.testing.assert_allclose(data["N_m_per_yr"], .8*factor, atol=1e-12)
    resumed = run(2, 15.)
    assert len(resumed) == 1
    with np.load(full[-1]) as a, np.load(resumed[0]) as b:
        for key in ["U_m_per_yr", "along_m_per_yr", "N_m_per_yr", "rate_dt_yr"]:
            np.testing.assert_allclose(a[key], b[key], atol=1e-12)
