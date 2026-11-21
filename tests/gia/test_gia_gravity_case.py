"""Tests of the `gravity` case: `GravitySolver` alone, against closed forms.

`gia_gravity.py` writes the measured values of each entry of
`gia_entries.GRAVITY` to `out/gravity-<entry>.dat`, and the CG counts of a
closed-form entry to `out/gravity-<entry>-counts.dat`. Each test here holds
the tolerance of the unit test that its rows come from, named in its
docstring. The only recorded values are the CG counts at each mesh level of
the serial closed-form entries (`expected/gravity-counts.txt` in 2-D,
`expected/gravity-counts-3d.txt` in 3-D); every other check is against a
closed form, a theoretical order, an exact identity or another entry.

An entry with `local` False runs a 3-D solve and runs on CI only. A
missing output of such an entry fails its tests, or skips them where
`gia_entries.LOCAL_ENV` is 1 (the development machine). A missing output
of any other entry is a failure. The basename is not `test_gia_gravity.py`, because
`tests/unit/test_gia_gravity.py` exists and pytest refuses two test modules
of one name.

This file does not import Firedrake; it runs where the outputs are.
"""

import math
import re
import warnings

import pytest

import gia_entries as E


def read(path):
    """The `name value` lines of a file as a dictionary; `#` starts a comment."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                name, value = line.split()
                out[name] = float(value)
    return out


def values(entry):
    """The values of `entry`.

    A missing file of an entry that runs on CI only fails the test, or skips
    it where `gia_entries.LOCAL_ENV` is 1 (`gia_entries.require_output`).
    """
    path = E.CASE_DIR / E.gravity_values(entry)
    if not E.GRAVITY[entry].get("local", True):
        E.require_output(path, entry)
    return read(path)


def order(coarse, fine):
    """The observed order `log2(coarse / fine)` of two errors on meshes of ratio 2."""
    return math.log(coarse / fine) / math.log(2.0)


def relative(measured, expected):
    """`|measured / expected - 1|`."""
    return abs(measured / expected - 1.0)


CLOSED_2D = ("closed-2d-multiplier", "closed-2d-lowrank")
CLOSED_3D = ("closed-3d-multiplier", "closed-3d-lowrank")


def test_each_entry_ran_on_its_ranks():
    """Each entry that ran, ran on the number of ranks `GRAVITY` gives it.

    An entry for two ranks that ran on one would pass its comparison with
    the serial entry without testing what it claims.
    """
    for entry, item in E.GRAVITY.items():
        path = E.CASE_DIR / E.gravity_values(entry)
        if not item.get("local", True):
            if not path.exists() and E.on_local_machine():
                continue  # this machine does not run the CI-only entries
            # On CI a missing file of a CI-only entry fails here.
            E.require_output(path, entry)
        ranks = read(path)["ranks"]
        assert ranks == item["cores"], (
            f"{entry}: ran on {ranks!r} rank(s), expected {item['cores']!r}")


# -- Errors against closed forms, and their order -----------------------------------

ORDER_CASES = ([(entry, row, bound, E.GRAVITY_LEVELS_2D)
                for entry in CLOSED_2D for row, bound in E.GRAVITY_ORDER_ROWS_2D.items()]
               + [(entry, row, bound, E.GRAVITY_LEVELS_3D)
                  for entry in CLOSED_3D for row, bound in E.GRAVITY_ORDER_ROWS_3D.items()])
ORDER_IDS = [f"{entry}-{row}" for entry, row, _, _ in ORDER_CASES]


@pytest.mark.parametrize("entry,row,bound,levels", ORDER_CASES, ids=ORDER_IDS)
def test_the_finest_error_is_within_the_unit_bound(entry, row, bound, levels):
    """The relative L2 error against the closed form on the finest mesh.

    The bound of the unit test: `TestCylindrical` 1e-4 (sheets, shell), 2e-5
    (Dirichlet and flux), 1e-6 (mass channels); `TestSpherical` 5e-3. The
    finest 2-D meshes are finer than the unit meshes.
    """
    error = values(entry)[f"{row}_error_{levels - 1}"]
    assert error <= bound, f"{entry} {row}: finest error {error!r} above {bound!r}"


@pytest.mark.parametrize("entry,row,bound,levels", ORDER_CASES, ids=ORDER_IDS)
def test_the_error_converges_at_the_order_of_cg2(entry, row, bound, levels):
    """The order between the two finest meshes is at least `GRAVITY_ORDER_MIN`.

    CG2 on degree-2 coordinates converges at order 3 in L2 for a smooth
    potential; each mesh has half the facet size of the one before. An order
    below `GRAVITY_ORDER_REPORT` passes with a warning, so that it is
    reported.
    """
    v = values(entry)
    coarse, fine = v[f"{row}_error_{levels - 2}"], v[f"{row}_error_{levels - 1}"]
    measured = order(coarse, fine)
    assert measured >= E.GRAVITY_ORDER_MIN, (
        f"{entry} {row}: order {measured!r} (errors {coarse!r}, {fine!r}) below "
        f"{E.GRAVITY_ORDER_MIN!r}")
    if measured < E.GRAVITY_ORDER_REPORT:
        warnings.warn(f"{entry} {row}: order {measured:.3f} below {E.GRAVITY_ORDER_REPORT}")


@pytest.mark.parametrize("entry", CLOSED_2D)
@pytest.mark.parametrize("sheet", E.GRAVITY_SHEETS, ids=lambda s: f"{s[0]}{s[1]}")
def test_the_sheet_coefficient_is_the_amplitude(entry, sheet):
    """The trace coefficient of the active mode, and the silence of the others.

    `TestCylindrical::test_boundary_sheet`: on the boundary that carries the
    sheet the coefficient of `cos m` is the amplitude `2 pi a / m` to 1e-4
    (finest mesh); every other coefficient on both boundaries is below 1e-10
    on every mesh, by the orthogonality of the modes.
    """
    v = values(entry)
    name = f"sheet_{sheet[0]}{sheet[1]}"
    finest = E.GRAVITY_LEVELS_2D - 1
    coefficient, amplitude = v[f"{name}_coefficient_{finest}"], v[f"{name}_amplitude"]
    assert relative(coefficient, amplitude) <= E.GRAVITY_COEFFICIENT_RTOL_2D, (
        f"{entry} {name}: coefficient {coefficient!r}, amplitude {amplitude!r}, relative "
        f"difference above {E.GRAVITY_COEFFICIENT_RTOL_2D!r}")
    for k in range(E.GRAVITY_LEVELS_2D):
        inactive = v[f"{name}_inactive_{k}"]
        assert inactive < E.GRAVITY_INACTIVE_2D, (
            f"{entry} {name} mesh {k}: inactive coefficient {inactive!r} above "
            f"{E.GRAVITY_INACTIVE_2D!r}")


@pytest.mark.parametrize("entry", CLOSED_3D)
def test_the_sphere_coefficients_are_the_closed_forms(entry):
    """The 3-D trace coefficients (`TestSpherical`).

    The `Y2,1` coefficient of the sheet is the amplitude `4 pi a / 5` to 5e-4
    and the inactive coefficients are below 1e-9; the `Y0,0` coefficients of
    the monopole are `sqrt(4 pi) 4 pi sigma RMIN^2 / R` to 5e-3. All on the
    finest shell.
    """
    v = values(entry)
    finest = E.GRAVITY_LEVELS_3D - 1
    coefficient = v[f"sphere_sheet_coefficient_{finest}"]
    amplitude = v["sphere_sheet_amplitude"]
    assert relative(coefficient, amplitude) <= E.GRAVITY_COEFFICIENT_RTOL_3D, (
        f"{entry}: Y2,1 coefficient {coefficient!r}, amplitude {amplitude!r}")
    for k in range(E.GRAVITY_LEVELS_3D):
        inactive = v[f"sphere_sheet_inactive_{k}"]
        assert inactive < E.GRAVITY_INACTIVE_3D, (
            f"{entry} shell {k}: inactive coefficient {inactive!r} above "
            f"{E.GRAVITY_INACTIVE_3D!r}")
    for marker in ("top", "bottom"):
        measured = v[f"sphere_monopole_{marker}_coefficient_{finest}"]
        expected = v[f"sphere_monopole_{marker}_expected"]
        assert relative(measured, expected) <= E.GRAVITY_MONOPOLE_COEFFICIENT_RTOL, (
            f"{entry} {marker}: Y0,0 coefficient {measured!r}, expected {expected!r}")


@pytest.mark.parametrize("entry", CLOSED_2D)
def test_the_uniform_disc_converges_with_the_polygon(entry):
    """The uniform disc, the sharpest check of the 2-D monopole datum.

    `TestCylindrical::test_monopole_uniform_disc`: the error is that of the
    polygonal boundary (order 2): below 2e-2 at refinement 2 and 2e-3 at
    refinement 4, each refinement below 0.35 of the one before; the boundary
    mean (the gauge) follows the same geometry error, below 1e-3 at the end.
    """
    v = values(entry)
    n = len(E.GRAVITY_DISC_LEVELS)
    errors = [v[f"disc_error_{k}"] for k in range(n)]
    means = [v[f"disc_mean_{k}"] for k in range(n)]
    assert errors[0] < E.GRAVITY_DISC_COARSE, f"{entry}: coarsest error {errors[0]!r}"
    assert errors[-1] < E.GRAVITY_DISC_FINE, f"{entry}: finest error {errors[-1]!r}"
    assert means[-1] < E.GRAVITY_DISC_MEAN, f"{entry}: finest boundary mean {means[-1]!r}"
    for label, series in (("error", errors), ("mean", means)):
        for coarse, fine in zip(series, series[1:]):
            assert fine < E.GRAVITY_DISC_RATIO * coarse, (
                f"{entry}: {label} {fine!r} not below {E.GRAVITY_DISC_RATIO!r} x {coarse!r}")


@pytest.mark.parametrize("entry", CLOSED_2D)
@pytest.mark.parametrize("channel", E.GRAVITY_MASS_CHANNELS)
def test_every_channel_encloses_its_mass(entry, channel):
    """Density, a sheet and a prescribed flux all enclose mass.

    `TestCylindrical::test_monopole_counts_every_mass_channel`: on the finest
    mesh, which is the unit mesh, the enclosed mass equals its closed form to
    1e-6 and the mean of `psi` on the outer boundary, which is the gauge, is
    below 1e-10. The mean is not zero on the coarser meshes: the datum is
    applied as `-2 G M / R` and weighted by the discrete boundary length,
    which differs from `2 pi R` there (measured 5.5e-9, 8.7e-11, 2.1e-12 on
    the three meshes).
    """
    v = values(entry)
    finest = E.GRAVITY_LEVELS_2D - 1
    mass, expected = v[f"mass_{channel}_mass_{finest}"], v[f"mass_{channel}_expected"]
    assert relative(mass, expected) <= E.GRAVITY_MASS_RTOL, (
        f"{entry} {channel}: enclosed mass {mass!r}, expected {expected!r}")
    mean = v[f"mass_{channel}_mean_{finest}"]
    assert mean < E.GRAVITY_GAUGE_MEAN, (
        f"{entry} {channel}: boundary mean {mean!r} above {E.GRAVITY_GAUGE_MEAN!r}")


@pytest.mark.parametrize("entry", CLOSED_2D)
def test_the_gauge_does_not_depend_on_the_robin_shift(entry):
    """`test_monopole_gauge_does_not_depend_on_the_robin_shift`.

    Precondition: the discrete boundary length is `2 pi R` to 1e-9. Then for
    `alpha` = 0.5, 1, 3 the boundary mean is below 1e-9 and the solutions
    agree to 1e-9 (relative L2).
    """
    v = values(entry)
    ratio = v["gauge_length_ratio"]
    assert abs(ratio - 1.0) < E.GRAVITY_LENGTH_RTOL, f"{entry}: length / 2 pi R = {ratio!r}"
    for k in range(3):
        assert v[f"gauge_mean_{k}"] < E.GRAVITY_ALPHA_RTOL, (
            f"{entry}: boundary mean {v[f'gauge_mean_{k}']!r} at shift {k}")
    for k in (1, 2):
        assert v[f"gauge_difference_{k}"] < E.GRAVITY_ALPHA_RTOL, (
            f"{entry}: difference {v[f'gauge_difference_{k}']!r} at shift {k}")


# -- Low-rank against multiplier -----------------------------------------------------

def coefficient_group(name):
    """The problem and mesh of a trace coefficient: the name up to its mesh index.

    The names are `coefficient_<row>_<k>_<boundary>_<mode>`, and `<row>`
    holds no `_<digits>_`, so the group is the name up to the first one.
    """
    match = re.match(r"(coefficient_.*?_\d+)_", name)
    return match.group(1) if match else name


@pytest.mark.parametrize("pair", [CLOSED_2D, CLOSED_3D], ids=["2d", "3d"])
def test_the_representations_give_the_same_coefficients(pair):
    """Every trace coefficient of every closed-form row, on every mesh.

    The two representations discretise the same operator. In 2-D both
    entries solve with the direct preset and the coefficients agree to
    `GRAVITY_PARITY_COEFFICIENTS` (1e-12) absolute. In 3-D both solve with
    the iterative preset, so they agree to the Krylov tolerance: the bound
    is `GRAVITY_PARITY_COEFFICIENTS_3D_RTOL` (1e-9) of the largest
    coefficient of the same problem and mesh (the measurement is in
    `gia_entries`). A wrong weight in the low-rank update (M9: the sign of
    the interior mean mode) moves the coefficients at the size of the error
    it makes.
    """
    multiplier, lowrank = values(pair[0]), values(pair[1])
    names = sorted(name for name in multiplier if name.startswith("coefficient_"))
    assert names and set(names) == {n for n in lowrank if n.startswith("coefficient_")}
    if pair == CLOSED_2D:
        allowed = {name: E.GRAVITY_PARITY_COEFFICIENTS for name in names}
    else:
        # The scale of each problem and mesh, from the multiplier entry.
        scale = {}
        for name in names:
            group = coefficient_group(name)
            scale[group] = max(scale.get(group, 0.0), abs(multiplier[name]))
        assert all(value > 0.0 for value in scale.values()), f"a problem with no load: {scale!r}"
        allowed = {name: E.GRAVITY_PARITY_COEFFICIENTS_3D_RTOL * scale[coefficient_group(name)]
                   for name in names}
    worst = max(names, key=lambda n: abs(multiplier[n] - lowrank[n]) / allowed[n])
    difference = abs(multiplier[worst] - lowrank[worst])
    assert difference <= allowed[worst], (
        f"{worst}: multiplier {multiplier[worst]!r}, low-rank {lowrank[worst]!r}, "
        f"difference {difference!r} above {allowed[worst]!r}")


@pytest.mark.parametrize("problem", ["two_boundaries", "sheet", "strong_bc"])
def test_the_parity_problems_agree_in_one_process(problem):
    """The parity problems of `test_dtn_lowrank_parity.py`, both paths in one process.

    Potential 1e-14 relative L2, coefficients 1e-12 absolute.
    """
    v = values("closed-2d-lowrank")
    psi, coefficients = v[f"parity_{problem}_psi"], v[f"parity_{problem}_coefficients"]
    assert psi <= E.GRAVITY_PARITY_PSI, f"{problem}: potential difference {psi!r}"
    assert coefficients <= E.GRAVITY_PARITY_COEFFICIENTS, (
        f"{problem}: coefficient difference {coefficients!r}")


def test_the_3d_parity_is_limited_by_the_tolerance():
    """`test_parity_shell` and `test_shell_parity_is_tolerance_limited`.

    At `rtol` 1e-14 the two paths agree to 1e-14 (potential) and 1e-12
    (coefficients). Over `rtol` 1e-9, 1e-11, 1e-13 the gap falls by more
    than 30 per hundredfold tightening and stays below 10 times `rtol`.
    """
    v = values("closed-3d-lowrank")
    assert v["parity_shell_psi"] <= E.GRAVITY_PARITY_PSI, f"{v['parity_shell_psi']!r}"
    assert v["parity_shell_coefficients"] <= E.GRAVITY_PARITY_COEFFICIENTS, (
        f"{v['parity_shell_coefficients']!r}")
    gaps = [v[f"parity_shell_gap_{k}"] for k in range(len(E.GRAVITY_TOLERANCE_SWEEP))]
    for coarse, fine in zip(gaps, gaps[1:]):
        assert coarse / fine > E.GRAVITY_SWEEP_FACTOR, f"gaps {gaps!r}"
    for rtol, gap in zip(E.GRAVITY_TOLERANCE_SWEEP, gaps):
        assert gap < E.GRAVITY_SWEEP_BOUND * rtol, f"gap {gap!r} at rtol {rtol!r}"


# -- CG counts -----------------------------------------------------------------------

def count_allowance(name):
    """The allowed difference of a recorded CG count, by the preset of its series."""
    preset = "direct" if "_direct_" in name else "iterative"
    return E.GRAVITY_COUNT_ALLOWANCE[preset]


def compare_counts(entries, expected_path):
    """Compare the CG counts of `entries`, level by level, with their recorded counts.

    Every count that an entry writes must be recorded, and every recorded
    count of the entries must be written, so that a mesh level that is added
    or lost also fails.
    """
    expected = read(expected_path)
    measured = {}
    for entry in entries:
        measured.update(read(E.CASE_DIR / E.gravity_counts(entry)))
    recorded = {name: value for name, value in expected.items()
                if name.startswith(tuple(f"{entry}_" for entry in entries))}
    assert measured and set(measured) == set(recorded), (
        f"counts written {sorted(measured)!r}, recorded {sorted(recorded)!r}")
    changed = {name: (value, recorded[name]) for name, value in measured.items()
               if abs(value - recorded[name]) > count_allowance(name)}
    assert not changed, (
        "CG counts (measured, recorded) that differ by more than "
        f"{E.GRAVITY_COUNT_ALLOWANCE!r}: {changed!r}")


@pytest.mark.parametrize("entry", E.GRAVITY_COUNT_ENTRIES)
def test_the_cg_counts_match_the_recording(entry):
    """The CG count at each mesh level against `expected/gravity-counts.txt`.

    `cg`: multigrid-preconditioned CG under the iterative preset (the
    potential block of the multiplier path; CG on `A + B` on the low-rank
    path). `sheet_top2_cg_direct`: on the low-rank path, CG with an LU of
    `A` (direct preset), whose count is set by the low-rank update. Each
    count must equal its recorded value within `GRAVITY_COUNT_ALLOWANCE`, so
    that a change of the preconditioner, of the operator or of PETSc shows
    as a changed count.
    """
    compare_counts((entry,), E.CASE_DIR / E.GRAVITY_EXPECTED_COUNTS)


def test_the_3d_cg_counts_match_the_recording():
    """The CG count at each 3-D shell against `expected/gravity-counts-3d.txt`.

    The same rule as in 2-D, for the two 3-D closed-form entries. Those
    entries run on CI only, so their counts are recorded from a reviewed
    run of them. Until the expected file exists, the test skips and prints
    the measured counts.
    """
    for entry in E.GRAVITY_COUNT_ENTRIES_3D:
        E.require_output(E.CASE_DIR / E.gravity_values(entry), entry)
    path = E.CASE_DIR / E.GRAVITY_EXPECTED_COUNTS_3D
    if not path.exists():
        measured = {entry: read(E.CASE_DIR / E.gravity_counts(entry))
                    for entry in E.GRAVITY_COUNT_ENTRIES_3D}
        pytest.skip(f"{E.GRAVITY_EXPECTED_COUNTS_3D} is not recorded yet; measured {measured!r}")
    compare_counts(E.GRAVITY_COUNT_ENTRIES_3D, path)


def test_two_ranks_give_the_serial_answers():
    """`closed-2d-lowrank-np2` against `closed-2d-lowrank`.

    The sheet and mass-channel solves on two ranks: every error, trace
    coefficient and enclosed mass agrees with the serial value to
    `|a - b| <= 1e-10 |b| + 1e-13`. A boundary mean
    (`_mean_`) is an integral of a potential of size 10 that cancels to
    1e-12 on the finest meshes, so it is compared absolutely to
    `GRAVITY_NP2_MEAN_ATOL` (measured differences up to 8.4e-13). The
    boundary quadrature degree comes from reductions over the ranks; a
    degree chosen differently on two ranks changes the discretisation and
    shows here. Counts are not compared.
    """
    serial, parallel = values("closed-2d-lowrank"), values("closed-2d-lowrank-np2")
    compared = [name for name in parallel
                if name != "ranks" and "_cg_" not in name and not name.startswith("cg_")]
    assert set(compared) <= set(serial)
    assert any(n.startswith("sheet_") for n in compared)
    assert any(n.startswith("mass_") for n in compared)
    for name in compared:
        if "_mean_" in name:
            bound = E.GRAVITY_NP2_MEAN_ATOL
        else:
            bound = E.GRAVITY_NP2_RTOL * abs(serial[name]) + E.GRAVITY_NP2_ATOL
        assert abs(parallel[name] - serial[name]) <= bound, (
            f"{name}: two ranks {parallel[name]!r}, one rank {serial[name]!r}, bound {bound!r}")


# -- Setup and split (multiplier entry) ----------------------------------------------

def test_the_net_mass_refusals_and_warnings():
    """`TestSetup`: what the 2-D net-mass checks refuse, warn about and carry.

    A net mass with one DtN boundary solves, and the enclosed mass is the
    area (1e-12); a net mass with a strong `psi` condition is refused at
    every scale of `rho` (1, 1e-3, 1e-10); a zero-mean density with that
    condition solves to a potential of L2 norm above 1e-3; a mass of
    leakage size warns, a zero-mean and a deliberate density do not.
    """
    v = values("closed-2d-multiplier")
    assert relative(v["setup_net_mass"], v["setup_net_mass_area"]) <= E.GRAVITY_NET_MASS_RTOL, (
        f"mass {v['setup_net_mass']!r}, area {v['setup_net_mass_area']!r}")
    assert v["setup_dirichlet_net_mass_refused"] == 1.0
    for k in range(3):
        assert v[f"setup_leakage_refused_{k}"] == 1.0, f"scale {k} not refused"
    assert v["setup_zero_mean_dirichlet_norm"] > E.GRAVITY_ZERO_MEAN_NORM, (
        f"{v['setup_zero_mean_dirichlet_norm']!r}")
    assert v["setup_leakage_warns"] == 1.0
    assert v["setup_zero_mean_warns"] == 0.0
    assert v["setup_deliberate_warns"] == 0.0


@pytest.mark.parametrize("preset", ["iterative", "direct"])
def test_the_index_set_split_matches_the_field_number_split(preset):
    """`TestIndexSetSplit::test_matches_the_field_number_split`.

    Potential and coefficients to 1e-12, equal outer counts.
    """
    v = values("closed-2d-multiplier")
    assert v[f"split_{preset}_psi"] < E.GRAVITY_SPLIT_RTOL, f"{v[f'split_{preset}_psi']!r}"
    assert v[f"split_{preset}_coefficients"] < E.GRAVITY_SPLIT_RTOL, (
        f"{v[f'split_{preset}_coefficients']!r}")
    assert v[f"split_{preset}_count_new"] == v[f"split_{preset}_count_old"]


def test_3d_has_no_monopole_datum():
    """`TestSetup::test_monopole_absent_in_3d`: no datum, no enclosed mass."""
    v = values("closed-3d-multiplier")
    assert v["absent_monopole_boundaries"] == 0.0
    assert v["absent_enclosed_mass"] == 0.0


# -- submesh --------------------------------------------------------------------------

def test_a_submesh_density_gives_the_same_mesh_solution():
    """The cross-mesh density (`test_cross_mesh_consistency`,
    `test_cross_mesh_iterative_preset`,
    `test_submesh_parity_and_the_mixed_space_disappearing`).

    The submesh density reproduces the same-mesh solution to 1e-8, the
    iterative preset the direct one to 1e-6 (relative L2), and the low-rank
    path the multiplier one to 1e-14 (potential) and 1e-12 (coefficients).
    The multiplier path carries 8 fields (potential, the DG0 dummy and 6
    multipliers, offset 2); the low-rank path one space and no multiplier.
    """
    v = values("submesh")
    assert v["same_mesh_cross_mesh"] == 0.0 and v["submesh_cross_mesh"] == 1.0
    assert v["submesh_vs_same_mesh"] < E.GRAVITY_SUBMESH_RTOL, f"{v['submesh_vs_same_mesh']!r}"
    assert v["iterative_offset"] == 2.0
    assert v["iterative_vs_direct"] < E.GRAVITY_SUBMESH_ITERATIVE_RTOL, (
        f"{v['iterative_vs_direct']!r}")
    assert (v["multiplier_fields"], v["multiplier_count"], v["multiplier_offset"]) == (
        8.0, 6.0, 2.0)
    assert v["lowrank_single_space"] == 1.0 and v["lowrank_count"] == 0.0
    assert v["lowrank_vs_multiplier_psi"] <= E.GRAVITY_PARITY_PSI, (
        f"{v['lowrank_vs_multiplier_psi']!r}")
    assert v["lowrank_vs_multiplier_coefficients"] <= E.GRAVITY_PARITY_COEFFICIENTS, (
        f"{v['lowrank_vs_multiplier_coefficients']!r}")


# -- adjoint ------------------------------------------------------------------------

TAYLOR_CASES = [(entry, row) for entry, rows in E.GRAVITY_TAYLOR_ROWS.items() for row in rows]
REPLAY_CASES = [(entry, row) for entry, rows in E.GRAVITY_REPLAY_ROWS.items() for row in rows]


@pytest.mark.parametrize("entry,row", TAYLOR_CASES, ids=[f"{e}-{r}" for e, r in TAYLOR_CASES])
def test_the_gradient_passes_the_guarded_taylor_test(entry, row):
    """`test_gravity_adjoint.assert_taylor_with_guards`, row by row.

    The directional derivative is at least 1e-6 `|J|` and the functional
    responds to the largest step by at least 1e-8 `|J|` (so the test is not
    run on a flat map); the rate with the gradient is at least 1.90 and the
    rate without it is in [0.85, 1.15] (so the same steps reach only first
    order without the gradient).
    """
    v = values(entry)
    J = v[f"{row}_J"]
    dJdm, response = v[f"{row}_dJdm"], v[f"{row}_response"]
    rate, rate0 = v[f"{row}_rate"], v[f"{row}_rate0"]
    assert abs(dJdm) >= E.GRAVITY_DERIVATIVE_FLOOR * J, (
        f"{row}: directional derivative {dJdm!r} below {E.GRAVITY_DERIVATIVE_FLOOR!r} |J| = "
        f"{E.GRAVITY_DERIVATIVE_FLOOR * J!r}")
    assert response >= E.GRAVITY_RESPONSE_FLOOR * J, f"{row}: response {response!r}, |J| {J!r}"
    assert rate >= E.GRAVITY_TAYLOR_RATE_MIN, (
        f"{row}: Taylor rate {rate!r} below {E.GRAVITY_TAYLOR_RATE_MIN!r}")
    low, high = E.NO_GRADIENT_BAND
    assert low <= rate0 <= high, f"{row}: rate without gradient {rate0!r} outside {low, high!r}"


@pytest.mark.parametrize("entry,row", REPLAY_CASES, ids=[f"{e}-{r}" for e, r in REPLAY_CASES])
def test_the_replay_repeats_the_recording_and_fresh_solves(entry, row):
    """`assert_replay_b1` and `assert_replay_b2` (1e-9 relative).

    The replay at the recorded control repeats the recorded functional; where
    a second control is written, the replay there equals a fresh, untaped
    solve.
    """
    v = values(entry)
    recorded, replay = v[f"{row}_recorded"], v[f"{row}_replay"]
    assert abs(replay - recorded) <= E.GRAVITY_REPLAY_RTOL * abs(recorded), (
        f"{row}: replay {replay!r}, recorded {recorded!r}")
    if f"{row}_fresh2" in v:
        fresh, replay2 = v[f"{row}_fresh2"], v[f"{row}_replay2"]
        assert abs(replay2 - fresh) <= E.GRAVITY_REPLAY_RTOL * abs(fresh), (
            f"{row}: replay {replay2!r} at the second control, fresh solve {fresh!r}")


@pytest.mark.parametrize("entry,G", [(entry, f"G_{d}") for d, entries in
                                     (("2d", E.GRAVITY_ADJOINT_2D), ("3d", E.GRAVITY_ADJOINT_3D))
                                     for entry in entries])
def test_the_derivative_in_G_is_2J_over_G(entry, G):
    """`test_gradient_G_constant`: `dJ/dG = 2 J / G`, exactly.

    `psi` is linear in `G`, so `J = int psi^2` is quadratic: the adjoint
    derivative equals `2 J / G` and the central difference to 1e-9; a
    gradient wrong by 1 % gives a Taylor rate at most 1.15.
    """
    v = values(entry)
    dJdG = v[f"{G}_dJdG"]
    assert abs(dJdG - v[f"{G}_analytic"]) <= E.GRAVITY_G_RTOL * abs(dJdG), (
        f"{entry}: dJ/dG {dJdG!r}, 2J/G {v[f'{G}_analytic']!r}")
    assert abs(v[f"{G}_central"] - dJdG) <= E.GRAVITY_G_RTOL * abs(dJdG), (
        f"{entry}: central difference {v[f'{G}_central']!r}, adjoint {dJdG!r}")
    assert v[f"{G}_sabotage"] <= E.SCALED_GRADIENT_RATE_MAX, (
        f"{entry}: rate {v[f'{G}_sabotage']!r} with a gradient wrong by 1 %")


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_the_tangent_is_the_transpose_of_the_adjoint(entry):
    """`test_tlm_adjoint_dot_product`: native and exact tangents (1e-8).

    The native tangent sweep of pyadjoint (on the multiplier entry, the path
    without multipliers, which is the only one it can run) against the
    adjoint directional derivative; and, `rho -> psi` being linear, the
    exact tangent `2 int psi(m) psi(h)` from two independent solves against
    the adjoint on the DtN configuration.
    """
    v = values(entry)
    for adjoint, tangent in (("tlm_adjoint", "tlm_native"), ("tlm_adjoint2", "tlm_exact")):
        assert abs(v[adjoint]) >= 1e-12 and abs(v[tangent]) >= 1e-12
        assert relative(v[tangent], v[adjoint]) <= E.GRAVITY_TANGENT_RTOL, (
            f"{entry}: {tangent} {v[tangent]!r}, {adjoint} {v[adjoint]!r}")


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_a_gradient_wrong_by_one_percent_is_detected(entry):
    """`test_taylor_detects_sabotaged_gradient`.

    The clean rate is at least 1.90; the derivative times 1.01, and the
    gradient plus 1 % of its norm along a density that `h` sees, both give a
    rate at most 1.15 and a residual at the smallest step at least 10 times
    the clean one.
    """
    v = values(entry)
    assert v["sabotage_rate_clean"] >= E.GRAVITY_TAYLOR_RATE_MIN, f"{v['sabotage_rate_clean']!r}"
    assert abs(v["sabotage_overlap"]) > 1e-8
    for kind in ("multiplicative", "noise"):
        assert v[f"sabotage_rate_{kind}"] <= E.SCALED_GRADIENT_RATE_MAX, (
            f"{kind}: rate {v[f'sabotage_rate_{kind}']!r}")
    for kind in ("multiplicative", "noise"):
        assert v[f"sabotage_residual_{kind}"] >= 10.0, (
            f"{kind}: residual ratio {v[f'sabotage_residual_{kind}']!r}")


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_the_first_order_ladder(entry):
    """`test_taylor_rho_fieldsplit_2d` / `test_taylor_rho_2d`: R0 and R1 rates.

    Without the gradient the remainder converges at order 1 (rate at least
    0.9), with it at order 2 (at least 1.90), and the smallest remainder is
    above round-off (1e-15); the gradient is nonzero somewhere.
    """
    v = values(entry)
    assert v["rho_2d_R0_rate"] >= E.GRAVITY_R0_RATE_MIN, f"{v['rho_2d_R0_rate']!r}"
    assert v["rho_2d_R1_rate"] >= E.GRAVITY_TAYLOR_RATE_MIN, f"{v['rho_2d_R1_rate']!r}"
    assert v["rho_2d_R1_min"] >= E.GRAVITY_R1_FLOOR, f"{v['rho_2d_R1_min']!r}"
    assert v["rho_2d_gradient_max"] > 0.0


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_a_net_mass_control_replays_as_a_fresh_solve(entry):
    """`test_reduced_functional_replay`: the enclosed mass is on the tape.

    A control with net mass replays to the value of a fresh solve (1e-9).
    """
    v = values(entry)
    assert abs(v["replay_mass"] - v["replay_mass_fresh"]) <= (
        E.GRAVITY_REPLAY_RTOL * abs(v["replay_mass_fresh"])), (
        f"replay {v['replay_mass']!r}, fresh {v['replay_mass_fresh']!r}")


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_solution_and_subfunction_give_one_gradient(entry):
    """`test_solution_assign_preserves_tape`: `solution` against `subfunctions[0]`.

    Same value and gradient to 1e-14, and replays that repeat both. On the
    multiplier path a raw `dat.data` copy of the field has an exactly zero
    gradient, which differs from the true one by more than 1e-3.
    """
    v = values(entry)
    J_sol, J_mix = v["assign_J_solution"], v["assign_J_mixed"]
    assert abs(J_sol - J_mix) <= E.GRAVITY_ASSIGN_RTOL * abs(J_mix), f"{J_sol!r}, {J_mix!r}"
    assert v["assign_gradient_difference"] <= E.GRAVITY_ASSIGN_RTOL, (
        f"{v['assign_gradient_difference']!r}")
    for route in ("solution", "mixed"):
        recorded, replay = v[f"assign_J_{route}"], v[f"assign_replay_{route}"]
        assert abs(replay - recorded) <= E.GRAVITY_REPLAY_RTOL * abs(recorded)
    if entry == "adjoint-2d-multiplier":
        assert v["assign_raw_gradient_max"] == 0.0, f"{v['assign_raw_gradient_max']!r}"
        assert v["assign_raw_difference"] > 1e-3


@pytest.mark.parametrize("entry,control", [(e, c) for e in E.GRAVITY_ADJOINT_2D
                                           for c in E.GRAVITY_MONOPOLE_CONTROLS])
def test_the_monopole_gradient_matches_fresh_solves(entry, control):
    """`test_monopole_gradient_against_fresh_solves`, per control (1e-6).

    The adjoint derivative against a central difference of solves of new
    solvers, which the tape cannot influence: a mass channel severed from
    the tape passes a Taylor test and fails this.
    """
    v = values(entry)
    adjoint, central = v[f"monopole_{control}_adjoint"], v[f"monopole_{control}_central"]
    assert relative(adjoint, central) < E.GRAVITY_MONOPOLE_FRESH_RTOL, (
        f"{entry} {control}: adjoint {adjoint!r}, fresh central difference {central!r}")


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_the_sheet_gradient_is_the_closed_form_2d(entry):
    """`test_gradient_matches_closed_form_sheet_2d`: `dc/ds = 2 pi G a / m`.

    To 1e-3, with the nearest wrong closed form (the factor of `m + 1`)
    more than 0.1 away.
    """
    v = values(entry)
    gradient, analytic = v["sheet_gradient"], v["sheet_gradient_analytic"]
    assert relative(gradient, analytic) <= E.GRAVITY_SHEET_GRADIENT_RTOL_2D, (
        f"{entry}: dc/ds {gradient!r}, 2 pi a / m {analytic!r}")
    assert abs(gradient - v["sheet_gradient_wrong"]) / abs(analytic) > (
        E.GRAVITY_SHEET_WRONG_MARGIN)


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_3D)
def test_the_sheet_gradient_is_the_closed_form_3d(entry):
    """`test_gradient_matches_closed_form_sheet_3d`: `dc/ds = 4 pi G a / (2l + 1)` (5e-3)."""
    v = values(entry)
    gradient, analytic = v["sphere_sheet_gradient"], v["sphere_sheet_gradient_analytic"]
    assert relative(gradient, analytic) <= E.GRAVITY_SHEET_GRADIENT_RTOL_3D, (
        f"{entry}: dc/ds {gradient!r}, 4 pi a / (2l + 1) {analytic!r}")


def test_the_lowrank_l0_weight_is_zero():
    """`test_taylor_monopole_constraint_row_3d` (low-rank): its premise.

    With `alpha = 1` and `lam_0 = 1 / R` the monopole weight is zero to 1e-14.
    """
    v = values("adjoint-3d-lowrank")
    assert abs(v["l0_weight"]) < 1e-14, f"{v['l0_weight']!r}"


@pytest.mark.parametrize("entry", E.GRAVITY_ADJOINT_2D)
def test_the_gradient_exists_past_the_field_wall(entry):
    """`M = 35` on both boundaries: past 128 fields, or with no field at all.

    Multiplier (`test_taylor_rho_past_field_enumeration_wall`,
    `test_solves_past_the_field_enumeration_wall`): 142 fields, the solve
    converges, the coefficients of modes up to 5 move by less than 1e-8 of
    their scale from the `M = 5` solve, the field-number split is refused.
    Low-rank (`test_taylor_at_a_truncation_the_multiplier_path_cannot_reach`):
    141 modes, no multiplier. Both: a nonzero gradient (the replay and the
    guarded Taylor test are rows of the parametrized tests above).
    """
    v = values(entry)
    if entry == "adjoint-2d-multiplier":
        assert v["wall_fields"] == 1 + v["wall_multipliers"] and v["wall_fields"] > 128
        assert v["wall_taped_fields"] > 128
        assert v["wall_converged"] > 0
        assert v["wall_low_mode_change"] < 1e-8 * v["wall_scale"], (
            f"{v['wall_low_mode_change']!r}, scale {v['wall_scale']!r}")
        assert v["wall_field_numbers_refused"] == 1.0
    else:
        assert v["wall_modes"] == 141.0 and v["wall_multipliers"] == 0.0
    assert v["wall_gradient_max"] > 0.0


def test_the_multiplier_tape_has_its_structure():
    """`test_tape_structure` (clean and dirty construction).

    One solve on the mixed space and one on the `Real` space of the enclosed
    mass, both reaching `J`; no assemble block inside the solve; two assign
    blocks and at least one subfunction block; the objective assemble
    reaches `J`. Built under annotation, the construction leaves assemble
    blocks of which none reaches `J`.
    """
    v = values("adjoint-2d-multiplier")
    for p in ("tape_clean", "tape_dirty"):
        assert v[f"{p}_solve_blocks"] == 2 and v[f"{p}_mixed_solves"] == 1
        assert v[f"{p}_mass_solves"] == 1 and v[f"{p}_mass_reaches"] == 1
        assert v[f"{p}_mixed_reaches"] == 1 and v[f"{p}_solve_assembles"] == 0
        assert v[f"{p}_subfunction_blocks"] >= 1 and v[f"{p}_assign_blocks"] == 2
        assert v[f"{p}_objective_assembles"] == 1 and v[f"{p}_objective_reaches"] == 1
        assert v[f"{p}_J"] > 0.0
    assert v["tape_dirty_construction_assembles"] > 0
    assert v["tape_dirty_construction_reaching"] == 0


def test_float_severs_the_multiplier_tape():
    """`test_coefficient_float_severs_tape`: the canonical negative control.

    A functional built from `coefficients()` on the multiplier path has an
    exactly zero gradient, and all four perturbed replays return the
    recorded value.
    """
    v = values("adjoint-2d-multiplier")
    assert v["severed_gradient_max"] == 0.0, f"{v['severed_gradient_max']!r}"
    assert v["severed_constant_replays"] == 4.0


def test_the_taped_integral_and_the_multiplier_give_one_gradient():
    """`test_coefficient_taped_integral_taylor` and `test_taylor_through_multiplier`.

    The integral form of the coefficient replays at the recorded control and
    at a second one (1e-9), equals the multiplier read (1e-8), and the
    gradients of the two routes agree (1e-8); a pure `cos 3` direction is
    blind to the `cos 2` coefficient (1e-9 of the active derivative).
    """
    v = values("adjoint-2d-multiplier")
    recorded = v["integral_recorded"]
    assert abs(v["integral_replay"] - recorded) <= E.GRAVITY_REPLAY_RTOL * abs(recorded)
    assert abs(v["integral_replay2"] - v["integral_fresh2"]) <= (
        E.GRAVITY_REPLAY_RTOL * abs(v["integral_fresh2"])), (
        f"{v['integral_replay2']!r}, {v['integral_fresh2']!r}")
    assert abs(recorded - v["integral_read"]) <= E.GRAVITY_ROUTE_RTOL * abs(recorded)
    assert v["integral_gradient_difference"] <= E.GRAVITY_ROUTE_RTOL
    assert v["read_gradient_difference"] <= E.GRAVITY_ROUTE_RTOL
    assert abs(v["read_orthogonal"]) <= 1e-9 * abs(v["read_active"]), (
        f"{v['read_orthogonal']!r} against {v['read_active']!r}")


def test_coefficients_is_taped_on_the_lowrank_path():
    """`test_coefficient_taped_integral_taylor` (low-rank).

    `coefficients()` equals the taped integral (1e-8); both gradients are
    nonzero (an untaped read gives exactly zero) and agree to 1e-8.
    """
    v = values("adjoint-2d-lowrank")
    assert relative(v["coefficients_recovered"], v["coefficients_integral"]) <= (
        E.GRAVITY_ROUTE_RTOL)
    assert v["coefficients_gradient_max"] > 0.0
    assert v["coefficients_integral_gradient_max"] > 0.0
    assert v["coefficients_gradient_difference"] <= E.GRAVITY_ROUTE_RTOL, (
        f"{v['coefficients_gradient_difference']!r}")


def test_a_firedrake_constant_is_not_a_control():
    """`test_firedrake_constant_is_not_differentiable_canary`.

    The gradient with respect to a `Constant` is exactly zero and its replay
    does not move. If this fails, Firedrake made `Constant` differentiable,
    and the advice to control `G` through a `Real` Function can be revised.
    """
    v = values("adjoint-2d-multiplier")
    assert v["canary_gradient"] == 0.0, f"{v['canary_gradient']!r}"
    assert v["canary_replay_unchanged"] == 1.0


def test_the_submesh_gradient_is_finite():
    """`test_taylor_rho_submesh`: the cross-mesh adjoint returns a gradient.

    On the low-rank path no mixed space exists (one space). The replay and
    the guarded Taylor test are rows of the parametrized tests above.
    """
    for entry in E.GRAVITY_ADJOINT_2D:
        v = values(entry)
        assert math.isfinite(v["submesh_gradient_max"]) and v["submesh_recorded"] > 0.0
    assert values("adjoint-2d-lowrank")["submesh_single_space"] == 1.0


# -- quadrature ---------------------------------------------------------------------

#: The sweep of `test_silence_implies_adequacy`. Warp 0.20 at `L = 8` is the
#: residual documented on `QuadratureCalibration`: a scalar distortion bound
#: cannot describe a two-dimensional calibration set. It is a strict xfail
#: with the measured numbers, so that a change of the rule that closes the
#: gap shows as an unexpected pass.
SWEEP = [pytest.param(warp, L, marks=pytest.mark.xfail(
    strict=True,
    reason="Known residual of the rule, documented on QuadratureCalibration: warp "
           "0.20 at L = 8 is silent (distortion 1.586 below 1.9, resolution 4.43 below "
           "11.1, degree 17 of the incumbent 18) and integrates to 1.90e-8, 1.9 times "
           "the target 1e-8 (measured 2026-11-21).")) if (warp, L) == (0.20, 8)
         else pytest.param(warp, L) for warp, L in E.GRAVITY_QUADRATURE_SWEEP]


@pytest.mark.parametrize("warp,L", SWEEP,
                         ids=[E.gravity_quadrature_entry(*s) for s in E.GRAVITY_QUADRATURE_SWEEP])
def test_silence_implies_adequacy(warp, L):
    """`test_silence_implies_adequacy`: either the degree works, or it was said.

    A warped shell that the constructor accepts without a boundary
    quadrature warning integrates every mode to the calibration target 1e-8.
    Only quadrature warnings count. With `sqrt(mean(area))` in
    `boundary_facet_scale` (M12) the distortion bound reads the meshes as
    more uniform than they are, and the warp 0.30 meshes go silent while
    missing the target.
    """
    v = values(E.gravity_quadrature_entry(warp, L))
    if v["sweep_warned"]:
        return
    assert v["sweep_error"] < E.GRAVITY_QUADRATURE_TARGET, (
        f"warp {warp}, L {L}: no quadrature warning, but error {v['sweep_error']!r} above "
        f"{E.GRAVITY_QUADRATURE_TARGET!r} (degree {v['sweep_degree']!r}, distortion "
        f"{v['sweep_distortion']!r} of {v['sweep_max_distortion']!r})")


def test_the_degree_follows_the_mesh_and_stays_odd():
    """`test_follows_the_mesh_not_the_truncation`, `test_odd_because_odd_is_free`,
    `test_never_costs_more_than_the_default_it_replaces`,
    `test_element_degree_costs_two_each`.

    One truncation on three boundary resolutions: the degree does not grow
    and falls overall, and stays below the incumbent `2 (M + 1)`; the degree
    is odd and `(q + 2) // 2 - 1 == q // 2`; it never exceeds the incumbent
    `2 (L + p)`; each element degree costs `QUADRATURE_ELEMENT_DEGREE_COST`.
    """
    v = values("quadrature-rule")
    degrees = [v[f"follows_degree_{k}"] for k in range(3)]
    assert degrees == sorted(degrees, reverse=True) and degrees[0] > degrees[-1], f"{degrees!r}"
    assert all(q < v["follows_incumbent"] for q in degrees)
    for k in range(3):
        q = int(v[f"odd_degree_{k}"])
        assert q % 2 == 1 and (q + 2) // 2 - 1 == q // 2, f"degree {q!r}"
    for k in range(5):
        assert v[f"cost_degree_{k}"] <= v[f"cost_incumbent_{k}"], (
            f"configuration {k}: {v[f'cost_degree_{k}']!r} > {v[f'cost_incumbent_{k}']!r}")
    for k in range(2):
        steps = [v[f"element_{k}_{p + 1}"] - v[f"element_{k}_{p}"] for p in (1, 2)]
        assert steps == [v["element_cost"]] * 2, f"mesh {k}: steps {steps!r}"


def test_the_free_warnings_fire_and_stay_quiet():
    """`test_an_extrapolating_mesh_is_flagged`, `test_distortion_alone_triggers_the_warning`,
    `test_a_well_covered_mesh_is_silent`.

    An irregular annulus at `M = 64` warns; a warp 0.30 shell at `L = 8` has
    its distortion above the bound, its resolution below its bound and the
    cap silent, so the distortion branch alone warns; three ordinary
    configurations give no warning at all.
    """
    v = values("quadrature-rule")
    assert v["extrapolating_warned"] == 1.0
    assert v["distortion_value"] > v["distortion_max"], (
        f"{v['distortion_value']!r}, {v['distortion_max']!r}")
    assert v["distortion_resolution"] < v["distortion_max_resolution"]
    assert v["distortion_same_rule"] == 1.0 and v["distortion_warned"] == 1.0
    assert v["silent_warned"] == 0.0


@pytest.mark.parametrize("entry", ["quadrature-rule", "quadrature-rank-np2"])
def test_the_degree_and_check_are_rank_independent(entry):
    """`test_degree_and_check_are_rank_independent`, on one and on two ranks.

    The degree is 9 for the 96-facet annulus at `M = 16` and 12 for the
    level-2 shell at `L = 5`, and both integrate to 1e-12, whatever the
    number of ranks.
    """
    v = values(entry)
    for dimension, degree in E.GRAVITY_RANK_DEGREES.items():
        assert v[f"rank_{dimension}_degree"] == degree, (
            f"{entry} {dimension}: degree {v[f'rank_{dimension}_degree']!r}, expected {degree!r}")
        assert v[f"rank_{dimension}_error"] < E.GRAVITY_RANK_CHECK, (
            f"{entry} {dimension}: check {v[f'rank_{dimension}_error']!r}")


def test_triangular_facets_carry_their_own_constants():
    """`test_triangular_facets_carry_their_own_constants`.

    The triangular constants have the larger intercept and slope; the
    icosahedral shell at `L = 10` uses them, takes degree 15, below the
    incumbent, and integrates to the target.
    """
    v = values("quadrature-rule")
    assert v["triangle_taller"] == 1.0 and v["triangle_steeper"] == 1.0
    assert v["triangle_calibration"] == 1.0
    assert v["triangle_degree"] == E.GRAVITY_TRIANGLE_DEGREE, f"{v['triangle_degree']!r}"
    assert v["triangle_degree"] < v["triangle_incumbent"]
    assert v["triangle_error"] < E.GRAVITY_QUADRATURE_TARGET, f"{v['triangle_error']!r}"


@pytest.mark.parametrize("k", range(6))
def test_the_default_degree_integrates_every_mode(k):
    """`TestDefaultIsAdequate::test_every_mode_is_integrated` (six meshes, 1e-8)."""
    error = values("quadrature-rule")[f"adequate_error_{k}"]
    assert error < E.GRAVITY_QUADRATURE_TARGET, f"mesh {k}: error {error!r}"


def test_the_check_is_an_instrument():
    """`TestCheckIsAnInstrument`: the check measures quadrature, not sphericity.

    Over degrees 5, 9, 13, 21 on one shell the check falls and spans more
    than four orders; a rich rule (25) integrates to 1e-12 on a coarse
    sphere; the recovered rule reproduces the assembly to 1e-13; the simplex
    fallback integrates to the target; an explicit inadequate degree has no
    report and is refused by the check, while the same degree on the
    uniform annulus integrates to 1e-11; an even degree and the next odd one
    give bit for bit the same check.
    """
    v = values("quadrature-rule")
    sweep = [v[f"instrument_sweep_{k}"] for k in range(4)]
    assert sweep == sorted(sweep, reverse=True), f"{sweep!r}"
    assert sweep[0] / sweep[-1] > E.GRAVITY_INSTRUMENT_SPAN, f"{sweep!r}"
    assert v["instrument_rich_rule"] < E.GRAVITY_RANK_CHECK, f"{v['instrument_rich_rule']!r}"
    assert v["instrument_recovered_rule"] <= E.GRAVITY_RECOVERED_RTOL, (
        f"{v['instrument_recovered_rule']!r}")
    assert v["instrument_simplex_error"] < E.GRAVITY_QUADRATURE_TARGET
    assert v["instrument_explicit_report_none"] == 1.0
    assert v["instrument_inadequate_refused"] == 1.0
    assert v["instrument_uniform_error"] < E.GRAVITY_UNIFORM_CHECK, (
        f"{v['instrument_uniform_error']!r}")
    assert v["instrument_even_odd_equal"] == 1.0
