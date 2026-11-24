"""Tests of ``summarise_gn.py`` on synthetic records (PLAN.md section 4.6).

Pure Python, no Firedrake. Run with the default Python:

    ~/Workplace/python3.12/bin/python3.12 -m pytest test_summarise_gn.py

The records have the JSON layout of ``box/optimisers.py``: one iteration row per
iteration with ``forward``, ``adjoint``, ``hessian``, ``gn_hessian`` and ``J``. Most
tests price every product at 1 forward-equivalent (``timing.json`` with
``full_per_forward = gn_over_full = 1``) and put the whole work of a row into
``forward``, so that the work of a row is the number written in the test. The tests
of the side readings split the work into products instead.

The three references are the same in most tests:

- T-ROL: J(200) = 0.3, J(400) = 0.18;
- T-N is the better full-Hessian run at 200 (0.4 against 0.45), T-N-R at 400 (0.19
  against 0.2), so ``N*`` takes its value from a different run at each level;
- both Newton runs fall to 0.03 at the end, so ``J_min`` is far below ``J_N*(600) =
  0.14`` and the floor guard leaves every level valid.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarise_gn as sg  # noqa: E402

UNIT_TIMING = {"full_per_forward": 1.0, "gn_over_full": 1.0}

ROL = [(2, 1.0), (100, 0.5), (200, 0.3), (300, 0.22), (400, 0.18), (500, 0.16),
       (600, 0.15), (700, 0.14), (800, 0.13)]
TN = [(2, 1.0), (100, 0.6), (200, 0.4), (300, 0.25), (400, 0.2), (500, 0.17),
      (600, 0.14), (700, 0.05), (800, 0.03)]
TNR = [(2, 1.0), (100, 0.7), (200, 0.45), (300, 0.3), (400, 0.19), (500, 0.16),
       (600, 0.145), (700, 0.06), (800, 0.04)]

# One Gauss-Newton arm per verdict. The comments give the crossings that decide it.
ARMS = {
    # A: 0.3 at 130 (S 1.54), 0.18 at 200 (S 2). B: 0.4 at 100 (S 2), 0.19 at 200
    # (S 2). C: 0.14 at 200 (S 3).
    "V1": [(2, 1.0), (50, 0.5), (100, 0.35), (130, 0.25), (200, 0.15), (260, 0.10),
           (400, 0.07), (600, 0.06), (800, 0.05)],
    # As V1 up to work 200, then a stall above 0.14: C never reached, S = 0.
    "V1b": [(2, 1.0), (50, 0.5), (100, 0.35), (130, 0.25), (200, 0.17), (300, 0.16),
            (400, 0.155), (600, 0.15), (800, 0.15)],
    # A: 0.3 at 190 (1.05), 0.18 at 250 (1.6). B: 0.4 at 150 (1.33 < 1.5). C: 0.14
    # at 400 (1.5).
    "V2": [(2, 1.0), (100, 0.6), (150, 0.38), (190, 0.29), (250, 0.17), (400, 0.12),
           (600, 0.1), (800, 0.09)],
    # A: 0.3 at 300 (0.67). B: 0.4 at 100 (2), 0.19 at 380 (1.05). C: 0.14 at 500.
    "V3": [(2, 1.0), (100, 0.38), (250, 0.32), (300, 0.25), (380, 0.185), (500, 0.13),
           (800, 0.1)],
    # A: 0.3 at 300 (0.67). B: 0.4 at 300 (0.67). C: 0.14 at 600 (1.0).
    "V4": [(2, 1.0), (100, 0.6), (200, 0.42), (300, 0.26), (400, 0.2), (500, 0.17),
           (600, 0.135), (800, 0.1)],
    # Slow everywhere and never reaches 0.14.
    "V5": [(2, 1.0), (100, 0.8), (300, 0.6), (600, 0.5), (800, 0.45)],
}


def record(points, split=None, **extra):
    """A synthetic run record.

    Args:
        points: list of ``(work, J)``, one per iteration row.
        split: None to put the whole work into ``forward``, or ``(f, a, h, g)``,
            the fractions of the work at unit prices that are forward runs, adjoint
            sweeps, full products and Gauss-Newton products.
        **extra: top-level entries (``krylov_ew``, the time lists).
    """
    rows = []
    for k, (work, J) in enumerate(points):
        if split is None:
            counts = {"forward": work, "adjoint": 0, "hessian": 0, "gn_hessian": 0}
        else:
            f, a, h, g = split
            counts = {"forward": f * work, "adjoint": a * work, "hessian": h * work,
                      "gn_hessian": g * work}
        rows.append(dict(iteration=k, J=J, misfit_term=0.9 * J, prior_term=0.1 * J,
                         **counts))
    out = {"iterations": rows}
    out.update(extra)
    return out


def records_with(arm_points, rol=ROL, tn=TN, tnr=TNR, name="T-GN-R"):
    """The four records of the rule with one Gauss-Newton arm."""
    return {"T-ROL": record(rol), "T-N": record(tn), "T-N-R": record(tnr),
            name: record(arm_points)}


# ---------------------------------------------------------------------------------
# 1 and 2: every verdict, and an arm that never reaches the value
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["V1", "V1b", "V2", "V3", "V4", "V5"])
def test_each_verdict(code):
    result = sg.apply_rule(records_with(ARMS[code]), 1.0, 1.0)
    assert result["verdict"]["code"] == code
    assert result["chosen"] == "T-GN-R"


def test_test_pattern_of_each_verdict():
    expected = {"V1": (True, True, True), "V1b": (True, True, False),
                "V2": (True, False, True), "V3": (False, True, True),
                "V4": (False, False, True), "V5": (False, False, False)}
    for code, pattern in expected.items():
        arm = sg.apply_rule(records_with(ARMS[code]), 1.0, 1.0)["arms"]["T-GN-R"]
        assert (arm["A"]["pass"], arm["B"]["pass"], arm["C"]["pass"]) == pattern, code


def test_never_reaches_gives_zero():
    arm = sg.apply_rule(records_with(ARMS["V5"]), 1.0, 1.0)["arms"]["T-GN-R"]
    part = arm["C"]["parts"][0]
    assert part["S"] == 0.0
    assert part["W_interval"] is None
    assert not arm["C"]["pass"]


def test_v1b_has_c_failed_by_a_stall():
    arm = sg.apply_rule(records_with(ARMS["V1b"]), 1.0, 1.0)["arms"]["T-GN-R"]
    assert arm["C"]["parts"][0]["S"] == 0.0
    assert arm["C"]["parts"][0]["level_used"] == 600.0


# ---------------------------------------------------------------------------------
# 3: the floor guard
# ---------------------------------------------------------------------------------

def test_floor_guard_replaces_600_by_400():
    # Every run flattens near 0.1 from work 500 on, so J_N*(600) is within 5 percent
    # of J0 - J_min of the floor, and 600 is not a valid level for N*. 400 is.
    tn = [(2, 1.0), (100, 0.6), (200, 0.4), (300, 0.25), (400, 0.2), (500, 0.102),
          (600, 0.101), (800, 0.1005)]
    tnr = [(2, 1.0), (100, 0.7), (200, 0.45), (300, 0.3), (400, 0.19), (500, 0.103),
           (600, 0.102), (800, 0.101)]
    arm = [(2, 1.0), (50, 0.5), (100, 0.35), (130, 0.25), (200, 0.15), (300, 0.12),
           (600, 0.1002), (800, 0.1)]
    result = sg.apply_rule(records_with(arm, tn=tn, tnr=tnr), 1.0, 1.0)
    c = result["arms"]["T-GN-R"]["C"]["parts"][0]
    assert c["level"] == 600.0
    assert c["level_used"] == 400.0
    assert c["J_ref"] == 0.19
    notes = result["arms"]["T-GN-R"]["notes"]
    assert any("600 is not valid" in note and "400 replaces it" in note for note in notes)


def test_floor_guard_unevaluable_counts_as_failed_and_says_so():
    # Every run reaches the floor 0.1 before work 150: no level is valid for N*.
    tn = [(2, 1.0), (100, 0.1001), (800, 0.1)]
    tnr = [(2, 1.0), (100, 0.1002), (800, 0.1)]
    arm = [(2, 1.0), (50, 0.5), (100, 0.2), (140, 0.1), (800, 0.1)]
    result = sg.apply_rule(records_with(arm, tn=tn, tnr=tnr), 1.0, 1.0)
    arm = result["arms"]["T-GN-R"]
    assert not arm["C"]["pass"]
    assert arm["C"]["parts"][0]["S"] is None
    assert any("needs a decision by a person" in note for note in arm["notes"])


# ---------------------------------------------------------------------------------
# 4: a value reached exactly at an iteration boundary
# ---------------------------------------------------------------------------------

def test_value_reached_exactly_at_a_row():
    # J_ROL(200) = 0.3, and the arm has J = 0.3 exactly at work 180. The row counts
    # as reaching the value, and the interval is that of the row before and the row.
    arm = [(2, 1.0), (100, 0.5), (180, 0.3), (250, 0.17), (400, 0.12), (800, 0.05)]
    result = sg.apply_rule(records_with(arm), 1.0, 1.0)
    part = result["arms"]["T-GN-R"]["A"]["parts"][0]
    assert part["J_ref"] == 0.3
    assert part["W_interval"] == (100.0, 180.0)
    assert part["S"] == pytest.approx(200.0 / 180.0, rel=1e-15)


def test_work_to_reach_reads_lower_end():
    rows = sg.priced_rows(record([(2, 1.0), (50, 0.5), (90, 0.2)]), 1.0, 1.0)
    assert sg.work_to_reach(rows, 0.2) == (50.0, 90.0)
    assert sg.work_to_reach(rows, 0.19) is None
    assert sg.speedup(rows, 0.5, 100.0) == (2.0, (2.0, 50.0))


# ---------------------------------------------------------------------------------
# 5: N* takes the better Newton run at each level
# ---------------------------------------------------------------------------------

def test_nstar_from_a_different_run_at_each_level():
    result = sg.apply_rule(records_with(ARMS["V1"]), 1.0, 1.0)
    b200, b400 = result["arms"]["T-GN-R"]["B"]["parts"]
    assert b200["J_ref"] == 0.4      # T-N at 200 (T-N-R has 0.45)
    assert b400["J_ref"] == 0.19     # T-N-R at 400 (T-N has 0.2)
    assert b200["reference"] == b400["reference"] == "N* = min(T-N, T-N-R)"


def test_newton_runs_get_test_a():
    result = sg.apply_rule(records_with(ARMS["V1"]), 1.0, 1.0)
    assert set(result["newton_test_A"]) == {"T-N", "T-N-R"}
    # T-N reaches 0.3 at work 300: S = 0.67, test A fails.
    assert not result["newton_test_A"]["T-N"]["pass"]


def test_several_arms_choose_the_largest_s400():
    records = records_with(ARMS["V4"], name="T-GN-R")
    records["T-GN-R-EW"] = record(ARMS["V1"])
    result = sg.apply_rule(records, 1.0, 1.0)
    assert result["chosen"] == "T-GN-R-EW"
    assert result["verdict"]["code"] == "V1"


# ---------------------------------------------------------------------------------
# 6: the prices of the rule and the two side readings
# ---------------------------------------------------------------------------------

def priced_records(gn_times=None):
    """Records whose work is split into products, for the side readings.

    Newton runs: 0.4 of the work forward and adjoint, 0.6 full products. The
    Gauss-Newton arm: the same split with Gauss-Newton products. ROL: no products.
    At unit prices the work is that of the lists above, and the arm is V1.
    """
    newton_split = (0.2, 0.2, 0.6, 0.0)
    gn_split = (0.2, 0.2, 0.0, 0.6)
    times = {"forward_times": [1.0, 1.0, 1.0], "hessian_times": [1.0, 1.0, 1.0]}
    gn_extra = {"forward_times": [1.0, 1.0, 1.0],
                "gn_hessian_times": gn_times or [1.0, 1.0, 1.0]}
    return {"T-ROL": record(ROL, forward_times=[1.0]),
            "T-N": record(TN, split=newton_split, **times),
            "T-N-R": record(TNR, split=newton_split, **times),
            "T-GN-R": record(ARMS["V1"], split=gn_split, **gn_extra)}


def test_prices_come_from_timing_json():
    records = priced_records()
    lines, result = sg.summarise(records, {"full_per_forward": 1.0, "gn_over_full": 1.0})
    assert result["prices"] == {"p_full": 1.0, "r_GN": 1.0}
    assert result["rule"]["verdict"]["code"] == "V1"
    # A larger product price makes every Newton and Gauss-Newton row dearer, and the
    # arm no longer catches ROL at work 200.
    lines, result = sg.summarise(records, {"full_per_forward": 2.0, "gn_over_full": 1.0})
    assert result["prices"] == {"p_full": 2.0, "r_GN": 1.0}
    assert result["rule"]["arms"]["T-GN-R"]["A"]["pass"] is False


def test_gn_price_is_p_full_times_r_gn():
    records = priced_records()
    rows = sg.apply_rule(records, 2.0, 1.5)["rows"]["T-GN-R"]
    row = records["T-GN-R"]["iterations"][3]
    expected = row["forward"] + row["adjoint"] + 2.0 * 1.5 * row["gn_hessian"]
    assert rows[3]["work_rule"] == pytest.approx(expected, rel=1e-15)


def test_side_ratio_changes_the_verdict_and_says_so():
    lines, result = sg.summarise(priced_records(), UNIT_TIMING, side_ratio=2.4822)
    assert result["rule"]["verdict"]["code"] == "V1"
    assert result["side_ratio"]["verdict"]["code"] != "V1"
    text = "\n".join(lines)
    assert "NOTE: common ratio 2.4822: the verdict of T-GN-R is" in text


def test_own_ratios_change_the_verdict_and_say_so():
    # The arm's own Gauss-Newton product costs 3 forward runs (median 3 s against a
    # median forward of 1 s), against 1 in timing.json.
    lines, result = sg.summarise(priced_records(gn_times=[3.0, 3.0, 3.0]), UNIT_TIMING)
    assert result["rule"]["verdict"]["code"] == "V1"
    assert result["own"]["verdict"]["code"] != "V1"
    text = "\n".join(lines)
    assert "NOTE: own per-call ratios: the verdict of T-GN-R is" in text
    assert "T-GN-R: full product 1.0000, GN product 3.0000" in text


def test_no_note_when_the_readings_agree():
    lines, _ = sg.summarise(priced_records(), UNIT_TIMING, side_ratio=1.0)
    assert not any(line.startswith("NOTE: common ratio") for line in lines)
    assert not any(line.startswith("NOTE: own per-call") for line in lines)


# ---------------------------------------------------------------------------------
# 7: the errors
# ---------------------------------------------------------------------------------

def test_no_timing_json_is_an_error():
    with pytest.raises(sg.RuleError, match="no timing.json"):
        sg.summarise(records_with(ARMS["V1"]), None)


def test_timing_json_without_prices_is_an_error():
    with pytest.raises(sg.RuleError):
        sg.summarise(records_with(ARMS["V1"]), {"full_per_forward": 2.0})


def krylov_row(rtols):
    """An iteration row list whose solves have the given tolerances."""
    rows = record(ARMS["V1"])["iterations"]
    for row, rtol in zip(rows[1:], rtols):
        row["krylov_solves"] = [{"rtol": rtol, "iterations": 3,
                                 "reason_name": "CONVERGED_RTOL", "hessian_product": "gn"}]
    return rows


def test_ew_record_with_every_rtol_1e2_is_an_error():
    records = records_with(ARMS["V4"])
    records["T-GN-R-EW"] = {"iterations": krylov_row([1e-2] * 8), "krylov_ew": True}
    with pytest.raises(sg.RuleError, match="Eisenstat-Walker rule did not act"):
        sg.summarise(records, UNIT_TIMING)


def test_ew_record_with_ew_tolerances_is_accepted():
    records = records_with(ARMS["V4"])
    records["T-GN-R-EW"] = {"iterations": krylov_row([0.5, 0.4, 0.3, 0.2, 0.2, 0.2,
                                                      0.2, 0.2]), "krylov_ew": True}
    lines, result = sg.summarise(records, UNIT_TIMING)
    assert result["rule"]["chosen"] == "T-GN-R-EW"


def test_main_exits_with_an_error(tmp_path, capsys):
    paths = {}
    for name, rec in records_with(ARMS["V1"]).items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(rec))
    argv = ["--rol", str(paths["T-ROL"]), "--newton-small", str(paths["T-N"]),
            "--newton-large", str(paths["T-N-R"]), "--gn", str(paths["T-GN-R"])]
    assert sg.main(argv) == 2
    assert "no timing.json" in capsys.readouterr().err
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps(UNIT_TIMING))
    out_json = tmp_path / "verdict.json"
    assert sg.main(argv + ["--timing-json", str(timing), "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "VERDICT (rule): V1" in printed
    assert json.loads(out_json.read_text())["rule"]["verdict"]["code"] == "V1"
