"""Tests of the `time_convergence` case: the orders in time of backward Euler and BDF2.

`gia_time_convergence.py` writes, for each entry, scheme and step count, the
error at the end time against a BDF2 reference with
`gia_entries.TC_REFERENCE_STEPS` steps to `out/time_convergence-<entry>.dat`.
With `e(n)` the error at `n` steps, the measured order between `n` and `2n`
steps is `log2(e(n) / e(2n))`: halving the step divides the error of a
scheme of order `p` by `2^p` once the step is small enough. The expected
values are the theoretical orders, 1 for backward Euler and 2 for BDF2 on
equal and on variable steps. No value is recorded from an earlier run.

This file does not import Firedrake; it runs where the outputs are.
"""

import math

import pytest

import gia_entries as E


def values(entry):
    """The `name value` lines of `out/time_convergence-<entry>.dat` as a dictionary."""
    path = E.CASE_DIR / E.time_convergence_values(entry)
    out = {}
    with open(path) as f:
        for line in f:
            name, value = line.split()
            out[name] = float(value)
    return out


def order(v, scheme, quantity, coarse, fine):
    """The measured order of `scheme` for `quantity` between `coarse` and `fine` steps.

    Args:
      v: the values of one entry.
      scheme: a name of `E.TC_SCHEMES`.
      quantity: a name of `E.TC_QUANTITIES`.
      coarse, fine: two step counts of `E.TC_LEVELS`, with `fine = 2 coarse`.

    Returns:
      `log2(e(coarse) / e(fine))`.
    """
    e_coarse = v[E.tc_key(scheme, coarse, quantity)]
    e_fine = v[E.tc_key(scheme, fine, quantity)]
    assert e_coarse > 0.0 and e_fine > 0.0, (
        f"{scheme} {quantity}: errors {e_coarse!r} ({coarse} steps) and {e_fine!r} "
        f"({fine} steps) must be positive")
    return math.log2(e_coarse / e_fine)


ENTRIES = list(E.TIME_CONVERGENCE)


@pytest.mark.parametrize("entry", ENTRIES)
def test_every_march_reaches_the_end_time(entry):
    """Every march ends at `TC_END_TIME`, so all errors are taken at the same time.

    The alternating steps end there only for an even step count; the
    reference has a nonzero displacement, potential and shift, so the
    relative errors have a scale.
    """
    v = values(entry)
    marches = [("reference", E.TC_REFERENCE_STEPS)]
    marches += [(scheme, n) for scheme in E.TC_SCHEMES for n in E.TC_LEVELS]
    for scheme, n in marches:
        reached = v[E.tc_key(scheme, n, "time")]
        assert abs(reached - E.TC_END_TIME) <= 1e-12 * E.TC_END_TIME, (
            f"{scheme} {n}: end time {reached!r}, expected {E.TC_END_TIME!r}")
    for quantity in ("u_norm", "psi_norm", "shift_value"):
        size = v[E.tc_key("reference", E.TC_REFERENCE_STEPS, quantity)]
        assert abs(size) > 0.0, f"reference {quantity} is {size!r}"


@pytest.mark.parametrize("quantity", E.TC_BE_ORDER_QUANTITIES)
@pytest.mark.parametrize("entry", ENTRIES)
def test_backward_euler_is_first_order(entry, quantity):
    """The order of backward Euler between 32 and 64 steps is in `TC_BE_ORDER_BAND`."""
    measured = order(values(entry), "backward_euler", quantity, *E.TC_BE_ORDER_LEVELS)
    low, high = E.TC_BE_ORDER_BAND
    assert low <= measured <= high, (
        f"{entry} backward Euler {quantity}: order {measured!r} between "
        f"{E.TC_BE_ORDER_LEVELS} steps, expected in [{low!r}, {high!r}]")


def bdf2_cases():
    """The parameters of `test_bdf2_is_second_order`: entry, scheme, quantity, step pair.

    An entry takes its pairs from `TC_BDF2_ORDER_PAIRS_BY_ENTRY`, or else
    from `TC_BDF2_ORDER_PAIRS`.
    """
    for entry in ENTRIES:
        pairs = E.TC_BDF2_ORDER_PAIRS_BY_ENTRY.get(entry, E.TC_BDF2_ORDER_PAIRS)
        for scheme in ("bdf2", "bdf2_alternating"):
            for quantity in E.TC_QUANTITIES:
                for pair in pairs:
                    yield pytest.param(entry, scheme, quantity, pair,
                                       id=f"{entry}-{scheme}-{quantity}-{pair[0]}-{pair[1]}")


@pytest.mark.parametrize("entry, scheme, quantity, pair", list(bdf2_cases()))
def test_bdf2_is_second_order(entry, scheme, quantity, pair):
    """The order of BDF2 between 16 and 32 and between 32 and 64 steps is at least 1.8.

    On equal steps and on steps that alternate between 0.8 and 1.2 of the
    mean step. The power-law entry is checked between 32 and 64 steps only
    (`TC_BDF2_ORDER_PAIRS_BY_ENTRY`). The variable-step coefficients keep the scheme second order
    on the alternating steps; the equal-step coefficients there make it
    inconsistent (mutation M4).
    """
    measured = order(values(entry), scheme, quantity, *pair)
    assert measured >= E.TC_BDF2_ORDER_MIN, (
        f"{entry} {scheme} {quantity}: order {measured!r} between {pair} steps, "
        f"expected at least {E.TC_BDF2_ORDER_MIN!r}")


@pytest.mark.parametrize("quantity", E.TC_QUANTITIES)
@pytest.mark.parametrize("entry", ENTRIES)
def test_both_schemes_have_the_same_limit(entry, quantity):
    """The error of backward Euler against the BDF2 reference at 64 steps is below 1e-3.

    A first-order error of this size at 64 steps means that both schemes
    converge to the same solution; a wrong term in one scheme only would
    leave a gap that does not shrink with the step.
    """
    measured = values(entry)[E.tc_key("backward_euler", E.TC_BE_LIMIT_STEPS, quantity)]
    assert measured < E.TC_BE_LIMIT_ERROR, (
        f"{entry} backward Euler {quantity}: error {measured!r} at "
        f"{E.TC_BE_LIMIT_STEPS} steps, expected below {E.TC_BE_LIMIT_ERROR!r}")
