"""Tests of ``summarise.py`` on synthetic records, for the Gauss-Newton changes.

Pure Python, no Firedrake. Run with the default Python:

    ~/Workplace/python3.12/bin/python3.12 -m pytest test_summarise.py

Every record has forward times of 1 s, so a price is the median time of the call in
seconds. The work of a row is ``forward + p_G adjoint + p_full hessian + p_gn
gn_hessian``.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarise  # noqa: E402


def record(points, counts, times, radius=None):
    """A synthetic record.

    Args:
        points: list of J, one per iteration row.
        counts: list of ``(forward, adjoint, hessian, gn_hessian)``, one per row.
        times: dict of the per-call time lists.
        radius: the ``initial_radius`` of the run, or None.
    """
    rows = []
    for k, (J, (f, a, h, g)) in enumerate(zip(points, counts)):
        rows.append({"iteration": k, "J": J, "forward": f, "adjoint": a, "hessian": h,
                     "gn_hessian": g, "initial_misfit": 0.1 * J})
    out = {"iterations": rows, **times}
    if radius is not None:
        out["initial_radius"] = radius
    return out


# ROL: one forward and one gradient per iteration, gradient 0.5 s: work 1.5 per row.
ROL_TIMES = {"forward_times": [1.0] * 5, "gradient_times": [0.5] * 5}


def rol_record():
    """P-ROL-L2: J falls by 0.02 per 1.5 work units, to 0.6 at work 30."""
    n = 101
    return record([1.0 - 0.02 * min(k, 20) - 0.001 * max(k - 20, 0) for k in range(n)],
                  [(k + 1, k + 1, 0, 0) for k in range(n)], ROL_TIMES)


def gn_record(gn_time=2.0, speed=1.0, radius=50.6):
    """P-GN: one forward, one gradient and 4 GN products per iteration."""
    n = 60
    J = [1.0 - min(0.08 * speed * k, 0.5) for k in range(n)]
    times = {"forward_times": [1.0] * 5, "gradient_times": [0.5] * 5,
             "gn_hessian_times": [gn_time] * 5}
    return record(J, [(k + 1, k + 1, 0, 4 * k) for k in range(n)], times, radius=radius)


def newton_record(radius=0.79):
    """P-Newton: one forward, one gradient and 2 full products per iteration."""
    n = 60
    J = [1.0 - min(0.03 * k, 0.5) for k in range(n)]
    times = {"forward_times": [1.0] * 5, "gradient_times": [0.5] * 5,
             "hessian_times": [1.25] * 5}
    return record(J, [(k + 1, k + 1, 2 * k, 0) for k in range(n)], times, radius=radius)


def write(tmp_path, name, rec):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(rec))
    return str(path)


def test_pooled_prices_hold_the_gn_median():
    gp, hp, medians = summarise.pooled_prices([rol_record(), gn_record(gn_time=2.0)])
    assert gp == 0.5
    assert hp is None
    assert medians["gn_hessian"] == 2.0
    assert summarise.gn_price(medians) == 2.0
    _, _, medians = summarise.pooled_prices([rol_record()])
    assert medians["gn_hessian"] is None
    assert summarise.gn_price(medians) is None


def test_priced_rows_add_the_gn_term_only_when_present():
    rows = summarise.priced_rows(gn_record(), 0.5, None, 2.0)
    row = rows[3]
    assert row["work_priced"] == 4 + 0.5 * 4 + 2.0 * 12
    # A record without Gauss-Newton products: the same numbers as the old formula.
    old_rows = summarise.priced_rows(newton_record(), 0.5, 1.25)
    new_rows = summarise.priced_rows(newton_record(), 0.5, 1.25, 7.0)
    assert [r["work_priced"] for r in old_rows] == [r["work_priced"] for r in new_rows]
    assert old_rows[5]["work_priced"] == 6 + 0.5 * 6 + 1.25 * 10


def test_newton_mode_is_unchanged(tmp_path, capsys):
    out = tmp_path / "out.json"
    summarise.main(["--reference", write(tmp_path, "rol", rol_record()),
                    "--newton", write(tmp_path, "newton", newton_record()),
                    "--levels", "20", "30", "--json", str(out)])
    result = json.loads(out.read_text())
    assert result["prices"]["gradient"] == 0.5
    assert result["prices"]["hessian"] == 1.25
    assert result["pooled"]["W_star"] == 30.0
    assert "P-GN" not in result["pooled"]


def test_newton_is_required_without_gn(tmp_path):
    with pytest.raises(SystemExit):
        summarise.main(["--reference", write(tmp_path, "rol", rol_record())])


def run_gn(tmp_path, gn, extra=()):
    out = tmp_path / "gn-out.json"
    summarise.main(["--reference", write(tmp_path, "rol", rol_record()),
                    "--gn", write(tmp_path, "gn", gn), "--levels", "20", "30",
                    "--json", str(out), *extra])
    return json.loads(out.read_text())


def test_gn_rule_uses_the_gn_price_and_gives_a_verdict(tmp_path):
    result = run_gn(tmp_path, gn_record(gn_time=0.25, speed=4.0))
    assert result["prices"]["gn_hessian"] == 0.25
    assert result["pooled"]["W_star"] == 30.0
    gn = result["pooled"]["P-GN"]
    # J_ref = J of ROL at work 30, the row of iteration 19: 1 - 0.38 = 0.62. P-GN
    # with speed 4 falls 0.32 per iteration and reaches 0.62 at iteration 2, whose
    # work is 3 + 0.5 * 3 + 0.25 * 8 = 6.5. S = 30 / 6.5 > 3.
    assert gn["J_ref"] == pytest.approx(0.62)
    assert gn["W_N"][1] == pytest.approx(6.5)
    assert gn["verdict"] == "wins"


def test_gn_undecided_names_the_second_run(tmp_path):
    # Speed 4 and a GN product of 1.5 s: P-GN reaches 0.62 at iteration 2, whose work
    # is 3 + 0.5 * 3 + 1.5 * 8 = 16.5, so S = 30 / 16.5 = 1.82: undecided.
    result = run_gn(tmp_path, gn_record(gn_time=1.5, speed=4.0))
    assert result["pooled"]["P-GN"]["verdict"].startswith("undecided (a second P-GN run")


def test_own_ratios_include_the_gn_price(tmp_path, capsys):
    gn = gn_record(gn_time=0.25, speed=4.0)
    # The own forward median of P-GN is 2 s, so its own GN price is 0.125 and its
    # own gradient price 0.25, against the pooled 0.25 and 0.5.
    gn["forward_times"] = [2.0] * 5
    result = run_gn(tmp_path, gn)
    printed = capsys.readouterr().out
    assert "P-GN: gradient 0.250, product nan, GN product 0.125" in printed
    assert result["own"]["P-GN"]["W_N"][1] == pytest.approx(3 + 0.25 * 3 + 0.125 * 8)


def test_prices_json_reading(tmp_path, capsys):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"gradient": 0.65, "hessian": 1.25}))
    result = run_gn(tmp_path, gn_record(gn_time=0.25, speed=4.0),
                    ["--prices-json", str(prices)])
    fixed = result["fixed_prices"]["P-GN"]
    # Gradient at 0.65 from the file, GN product at 0.25 from P-GN's own times.
    assert fixed["W_N"][1] == pytest.approx(3 + 0.65 * 3 + 0.25 * 8)


def test_against_newton_states_the_radius(tmp_path, capsys):
    newton = write(tmp_path, "newton", newton_record(radius=0.79))
    run_gn(tmp_path, gn_record(gn_time=0.25, speed=4.0, radius=50.6), ["--newton", newton])
    printed = capsys.readouterr().out
    assert "changes the product and the radius together" in printed
    run_gn(tmp_path, gn_record(gn_time=0.25, speed=4.0, radius=0.79), ["--newton", newton])
    printed = capsys.readouterr().out
    assert "changes the product and the radius together" not in printed
    assert "P-GN against P-Newton" in printed
