"""Check the registered pilot against the published inter-code range."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def result():
    return json.loads(Path(__file__).with_name("king_result.json").read_text())


def test_converged_convective_branch(result):
    assert result["snes_reason"] > 0
    assert result["snes_residual"] < 1e-8
    assert result["vrms"] > 1
    assert abs(result["reaction_energy_defect"]) < 1e-8


def test_published_intercode_range(result):
    references = json.loads(Path(__file__).with_name("king_reference.json").read_text())
    references = [r for r in references if r["ra"] == result["ra"] and r["di"] == result["di"]]
    # This is a coarse registered smoke case, not a refinement certificate.
    # No fitted G-ADOPT expected data are used. The allowances account for
    # the coarsest printed rounding of these published columns.
    for field, reference_key, rounding in [
        ("nu_top_reaction", "nu", 0.005),
        ("vrms", "vrms", 0.05),
        ("mean_surface_relative_temperature", "mean_temperature", 0.0005),
    ]:
        values = [r[reference_key] for r in references]
        assert min(values) - rounding <= result[field] <= max(values) + rounding
