"""Offline plotting checks with known reference offsets, without Firedrake."""

import csv
import json
from pathlib import Path

import giamip
from giamip.benchmarks.results import BenchmarkResult, Curve
import matplotlib.pyplot as plt
import numpy as np
import pytest

import plot_martinec_profiles as plot


@pytest.fixture
def reference(monkeypatch):
    references = {}
    for letter in ("B", "D"):
        case = giamip.case(f"martinec2018-{letter}")
        result = BenchmarkResult(letter, "reference")
        x = np.linspace(0., 180., 181)
        for spec in case.profiles:
            for q in spec.quantities:
                values = .01 * x + {"U": -1., "N": .5, "S": -119.5}[q]
                result.add_profile(spec.name, q, x, values, built_from="synthetic test reference")
        if letter == "D":
            result.add_time_series("h_UF", [0., 15.], [0., -120.])
        references[letter] = result
    monkeypatch.setattr(BenchmarkResult, "reference", staticmethod(lambda letter: references[letter]))

    def envelope(letter, profile, quantity, **kwargs):
        case = giamip.case(f"martinec2018-{letter}")
        return {"max": .5, "rms": .5, "members": {"TEST": {}},
                "features": case.feature_colatitudes(profile)}

    monkeypatch.setattr(plot.metrics, "contributor_envelope", envelope)
    return references


def write_run(directory, reference, offset=.2):
    directory.mkdir()
    letter = reference.case
    case = giamip.case(f"martinec2018-{letter}")
    times = np.array([0., 1., case.terminal_time_kyr])
    files = []
    for epoch in times:
        filename = f"martinec-{letter}-profiles_{epoch:g}kyr.npz"
        files.append(filename)
        data = {"t_kyr": epoch}
        for spec in case.profiles:
            x = reference.profile(spec.name, "U").axis
            data[f"{spec.name}_colatitude_deg"] = x
            data[f"{spec.name}_longitude_deg"] = spec.longitude_deg
            data[f"{spec.name}_found"] = np.ones(x.shape, dtype=bool)
            for q in spec.quantities:
                data[f"{spec.name}_{q}_m"] = reference.profile(spec.name, q).values + offset
        np.savez(directory / filename, **data)
    filename = f"martinec-{letter}-timeseries.npz"
    np.savez(directory / filename, t_kyr=times, h_UF_m=np.full(times.shape, -120. + offset),
             dt_yr=[.03168809, 1000., 1000.])
    summary = {"case": letter, "smoke": False, "steps": len(times),
               "timeseries_file": filename, "profile_files": files}
    path = directory / f"summary_{letter}.json"
    path.write_text(json.dumps(summary))
    return path


def test_case_B_intervals_epoch_and_residuals(tmp_path, reference):
    path = write_run(tmp_path / "ladder", reference["B"])
    run = plot.load_run(path, "Ladder")
    case = giamip.case("martinec2018-B")
    for spec in case.profiles:
        fig = plot.plot_terminal([run], case, reference["B"], spec, plot.DEFAULT_BAND_DEG)
        assert "10 kyr" in fig._suptitle.get_text()
        assert "15 kyr" not in fig._suptitle.get_text()
        assert len(fig.axes) == 2 * len(spec.quantities)
        for ax in fig.axes:
            assert ax.get_xlim() == spec.colatitude_interval_deg
        for ax in fig.axes[len(spec.quantities):]:
            np.testing.assert_allclose(ax.lines[0].get_ydata(), .2, atol=1e-12)
        plt.close(fig)
        fig = plot.plot_evolution([run], case, spec)
        assert all(ax.get_xlim() == spec.colatitude_interval_deg for ax in fig.axes)
        plt.close(fig)


def test_B_history_has_only_terminal_reference_and_excludes_elastic_dt(tmp_path, reference):
    path = write_run(tmp_path / "ladder", reference["B"])
    run = plot.load_run(path, "Ladder")
    fig = plot.plot_history([run], giamip.case("martinec2018-B"), reference["B"])
    terminal = fig.axes[0].lines[-1]
    np.testing.assert_array_equal(terminal.get_xdata(), [10.])
    np.testing.assert_allclose(terminal.get_ydata(), [-120.])
    steps = fig.axes[1].lines[0]
    np.testing.assert_array_equal(steps.get_xdata(), [0., 1., 10.])
    np.testing.assert_array_equal(steps.get_ydata(), [1000., 1000., 1000.])
    assert fig.axes[1].get_yscale() == "log"
    plt.close(fig)


def test_D_reference_history_and_endpoint(tmp_path, reference):
    path = write_run(tmp_path / "D", reference["D"])
    run = plot.load_run(path, "D")
    case = giamip.case("martinec2018-D")
    fig = plot.plot_history([run], case, reference["D"])
    np.testing.assert_array_equal(fig.axes[0].lines[-1].get_xdata(), [0., 15.])
    assert fig.axes[0].get_xlim() == (0., 15.)
    plt.close(fig)


@pytest.fixture
def contributors(monkeypatch, reference):
    results = {}
    for member in ("GA", "WW", "ZM"):
        result = BenchmarkResult("B", member)
        x = np.array([5., 16., 37., 45., 65., 78., 125., 135.])
        for key in reference["B"].profiles:
            if member == "WW" and key == ("basin", "S"):
                continue
            result.add_profile(*key, x, np.full(x.shape, 2.))
            # Deliberately distinct from profile minus reference to check that
            # the published residual is used, without resampling its grid.
            result.differences[key] = Curve(x, np.full(x.shape, .03))
        results[member] = result
    monkeypatch.setattr(BenchmarkResult, "reference_path", staticmethod(lambda letter: Path("unused.nc")))
    monkeypatch.setattr(BenchmarkResult, "members", staticmethod(lambda path: list(results)))
    monkeypatch.setattr(BenchmarkResult, "read", staticmethod(lambda path, member, case: results[member]))
    return plot.load_contributors("B")


def test_contributor_native_grids_published_residuals_and_missing_quantity(tmp_path, reference, contributors):
    assert [result.member for result in contributors] == ["GA", "WW"]
    run = plot.load_run(write_run(tmp_path / "ladder", reference["B"]), "Ladder")
    case = giamip.case("martinec2018-B")
    spec = next(spec for spec in case.profiles if spec.name == "basin")
    fig = plot.plot_terminal([run], case, reference["B"], spec, plot.DEFAULT_BAND_DEG, contributors)
    for j, q in enumerate(spec.quantities):
        for row in (0, 1):
            lines = {line.get_label(): line for line in fig.axes[row * len(spec.quantities) + j].lines}
            np.testing.assert_array_equal(lines["GA"].get_xdata(), [65., 78., 125., 135.])
            np.testing.assert_allclose(lines["GA"].get_ydata(), 2. if row == 0 else .03)
            assert ("WW" in lines) == (q != "S")
        np.testing.assert_allclose(lines["Ladder"].get_ydata(), .2, atol=1e-12)
    assert {text.get_text() for text in fig.legends[0].get_texts()} == {
        "GA", "WW", "VEGA reference", "Ladder"}
    plt.close(fig)


@pytest.mark.parametrize("overlay", [False, True])
def test_multiple_runs_render_and_export_scores_without_changing_inputs(tmp_path, reference, contributors, overlay):
    paths = [write_run(tmp_path / name, reference["B"], offset)
             for name, offset in (("ladder", .2), ("uniform", .1))]
    before = {path: path.read_bytes() for directory in ("ladder", "uniform")
              for path in (tmp_path / directory).iterdir()}
    output = tmp_path / "plots"
    plot.main([*map(str, paths), "--labels", "Ladder", "Uniform 10 yr", "--output", str(output),
               *(["--contributors"] if overlay else [])])
    manifest = json.loads((output / "plot_manifest.json").read_text())
    assert manifest["case"] == "B"
    assert manifest["contributors"] == (["GA", "WW"] if overlay else [])
    assert [row["label"] for row in manifest["runs"]] == ["Ladder", "Uniform 10 yr"]
    assert len(manifest["figures"]) == 10
    for name in manifest["figures"]:
        signature = b"\x89PNG" if Path(name).suffix == ".png" else b"%PDF"
        assert (output / name).read_bytes().startswith(signature)
    with (output / "profile_scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 10
    for row in scores:
        expected = .2 if row["run"] == "Ladder" else .1
        assert float(row["max"]) == pytest.approx(expected, abs=1e-12)
        assert float(row["rms"]) == pytest.approx(expected, abs=1e-12)
        assert row["inside"] == "True"
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not plt.get_fignums()


def test_missing_profile_points_are_rejected(tmp_path, reference):
    path = write_run(tmp_path / "bad", reference["B"])
    filename = path.parent / "martinec-B-profiles_10kyr.npz"
    data = plot.read_npz(filename)
    data["load_found"][25] = False
    np.savez(filename, **data)
    with pytest.raises(ValueError, match="unlocated load"):
        plot.load_run(path, "Bad")


def test_incomplete_history_is_rejected(tmp_path, reference):
    path = write_run(tmp_path / "partial", reference["B"])
    filename = path.parent / "martinec-B-timeseries.npz"
    data = plot.read_npz(filename)
    np.savez(filename, **{key: values[:-1] for key, values in data.items()})
    with pytest.raises(ValueError, match="ascend to 10 kyr"):
        plot.load_run(path, "Partial")


def test_mixed_cases_are_rejected(tmp_path, reference):
    b = write_run(tmp_path / "B", reference["B"])
    d = write_run(tmp_path / "D", reference["D"])
    with pytest.raises(SystemExit) as error:
        plot.main([str(b), str(d)])
    assert error.value.code == 2
