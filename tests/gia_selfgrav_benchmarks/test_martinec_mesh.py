"""Independent cap/lithosphere refinements and compatibility with saved meshes."""

from copy import deepcopy
import math
from types import SimpleNamespace

import martinec
import numpy as np
import pytest


def mesh_spec(arguments):
    args = martinec.parse_args(arguments)
    kind = "smoke" if args.smoke else "martinec"
    return {**martinec.common.MESHES[kind], **martinec.mesh_overrides(args)}


def test_independent_mesh_changes_and_unchanged_defaults():
    original = deepcopy(martinec.common.MESHES)
    baseline = mesh_spec([])
    cap = mesh_spec(["--cap_h_km", "39"])
    layers = mesh_spec(["--litho_layers", "4"])
    assert cap == {**baseline, "refinement": {**baseline["refinement"], "cap_h_km": 39.}}
    assert layers == {**baseline, "litho_layers": 4}
    assert mesh_spec([]) == original["martinec"]
    assert martinec.common.MESHES == original
    assert mesh_spec(["--smoke", "--litho_layers", "2"]) == {**original["smoke"], "litho_layers": 2}


def size_expressions(spec):
    """Capture the expressions handed to gmsh, without generating a volume mesh."""
    expressions = {}

    def add(kind):
        tag = len(expressions) + 1
        expressions[tag] = None
        return tag

    field = SimpleNamespace(add=add, setString=lambda tag, name, value: expressions.update({tag: value}))
    hook = martinec.common.martinec_size_fields(
        spec["h_km"] / martinec.common.D_KM, spec["grade"], **spec["refinement"])
    hook(SimpleNamespace(model=SimpleNamespace(mesh=SimpleNamespace(field=field))))
    return list(expressions.values())


def test_cap_size_field_changes_without_changing_coastline_fields():
    baseline = size_expressions(mesh_spec([]))
    finer = size_expressions(mesh_spec(["--cap_h_km", "39"]))
    assert baseline == size_expressions(mesh_spec(["--cap_h_km", "78"]))
    assert finer[1:] == baseline[1:]  # Both coastline fields are identical.
    common = martinec.common
    points = [(common.MARTINEC_CAP_CENTRE, 78., 39.),
              ((100. - common.MARTINEC_COAST_PSI_DEG, 320.), 78., 78.)]
    for centre, expected_baseline, expected_finer in points:
        x, y, z = common.RE * common._unit_vector(*centre)
        namespace = {"x": x, "y": y, "z": z, "sqrt": math.sqrt,
                     "acos": math.acos, "tanh": math.tanh, "max": max, "min": min, "abs": abs}
        for expressions, expected in ((baseline, expected_baseline), (finer, expected_finer)):
            spacing = min(eval(expr, {"__builtins__": {}}, namespace) for expr in expressions)
            assert spacing * common.D_KM == pytest.approx(expected)


def test_lithosphere_subdivision_preserves_material_interfaces():
    common = martinec.common
    coarse = np.array(common.sphere_radii(2))
    fine = np.array(common.sphere_radii(4))
    np.testing.assert_array_equal(coarse[coarse <= common.R_LITHO], fine[fine <= common.R_LITHO])
    np.testing.assert_array_equal(coarse[coarse >= common.RE], fine[fine >= common.RE])
    lithosphere = fine[(fine >= common.R_LITHO) & (fine <= common.RE)]
    # The driver's stored dimensionless interface radii are rounded.
    np.testing.assert_allclose(np.diff(lithosphere) * common.D_KM, [17.5] * 4, atol=1e-4)


@pytest.mark.parametrize("arguments", [
    ["--cap_h_km", "0"], ["--cap_h_km", "-1"], ["--cap_h_km", "nan"], ["--cap_h_km", "inf"],
    ["--cap_h_km", "39", "--smoke"], ["--litho_layers", "0"],
    ["--litho_layers", "-1"], ["--litho_layers", "2.5"],
])
def test_invalid_mesh_options(arguments):
    with pytest.raises(SystemExit) as error:
        martinec.parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("case", list("BCDE"))
def test_all_cases_accept_independent_mesh_options(case):
    assert mesh_spec(["--case", case, "--cap_h_km", "39"])["litho_layers"] == 2
    assert mesh_spec(["--case", case, "--litho_layers", "4"])["refinement"]["h_fine_km"] == 78.


@pytest.mark.parametrize("arguments", [["--cap_h_km", "39"], ["--litho_layers", "4"]])
@pytest.mark.parametrize("case", ["B", "E"])
def test_new_runs_pass_overrides_to_mesh_builder(tmp_path, monkeypatch, arguments, case):
    args = martinec.parse_args(["--case", case, *arguments, "--output_path", str(tmp_path)])
    monkeypatch.setattr(martinec, "parse_args", lambda: args)

    def build(kind, directory, stem, *, mesh_overrides):
        assert kind == "martinec"
        assert mesh_overrides == martinec.mesh_overrides(args)
        raise RuntimeError("mesh builder reached")

    monkeypatch.setattr(martinec.common, "build_meshes", build)
    with pytest.raises(RuntimeError, match="mesh builder reached"):
        martinec.main()


@pytest.mark.parametrize("arguments,stored", [
    (["--cap_h_km", "39"], []),
    (["--litho_layers", "4"], []),
    (["--cap_h_km", "78"], ["--cap_h_km", "39"]),
    (["--litho_layers", "2"], ["--litho_layers", "4"]),
])
@pytest.mark.parametrize("case", ["B", "E"])
def test_restart_rejects_explicit_mesh_changes(tmp_path, monkeypatch, arguments, stored, case):
    args = martinec.parse_args(["--case", case, *arguments, "--restart", "checkpoint.h5",
                                "--output_path", str(tmp_path)])
    epochs = martinec.EPOCHS_KYR["T0" if case == "B" else "T1"]
    ladder, _ = martinec.time_schedule(args, epochs)
    state = SimpleNamespace(record={"run": martinec.run_settings(args, martinec.BULK_SHEAR_RATIO[case], ladder)})
    saved_mesh = mesh_spec(stored)
    monkeypatch.setattr(martinec, "parse_args", lambda: args)
    monkeypatch.setattr(martinec.common, "load_checkpoint", lambda *a: (None, None, saved_mesh, state))
    with pytest.raises(ValueError, match="Checkpoint"):
        martinec.main()


def test_restart_inherits_saved_mesh_and_accepts_legacy_defaults():
    # Old checkpoints have h_fine_km but no cap_h_km. Explicit matching
    # settings remain valid, and omitted options retain any saved mesh.
    martinec.check_mesh_restart(martinec.parse_args(["--cap_h_km", "78", "--litho_layers", "2"]), mesh_spec([]))
    refined = ["--cap_h_km", "39", "--litho_layers", "4"]
    martinec.check_mesh_restart(martinec.parse_args(refined), mesh_spec(refined))
    martinec.check_mesh_restart(martinec.parse_args([]), mesh_spec(refined))
