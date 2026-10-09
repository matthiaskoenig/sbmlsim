"""The reductions of a formula over the time of each simulation."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulator.formula import compile_formula, evaluate_reduced, reduce_formula

#: two simulations: the second has a change at 1 (a duplicated time) and padding
T = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, np.nan]])
X = np.array([[1.0, 3.0, 2.0, 0.0], [5.0, 4.0, 6.0, np.nan]])


def test_max_and_min_reduce_every_simulation() -> None:
    np.testing.assert_array_equal(
        evaluate_reduced("max(x)", {"x": X}, T), [[3.0], [6.0]]
    )
    np.testing.assert_array_equal(
        evaluate_reduced("min(x)", {"x": X}, T), [[0.0], [4.0]]
    )


def test_a_reduction_broadcasts_over_the_time() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("x / max(x)", {"x": X}, T), X / np.array([[3.0], [6.0]])
    )


def test_the_mean_is_weighted_by_the_time() -> None:
    # (1+3)/2 + (3+2)/2 + (2+0)/2 = 5.5 over 3; (5+4)/2 over 1, the change adds 0
    np.testing.assert_allclose(
        evaluate_reduced("mean(x)", {"x": X}, T), [[5.5 / 3.0], [4.5]]
    )


def test_the_mean_does_not_depend_on_the_steps() -> None:
    coarse = np.linspace(0.0, 2.0, 3)[None, :]
    fine = np.linspace(0.0, 2.0, 101)[None, :]
    for t in (coarse, fine):
        mean = evaluate_reduced("mean(x)", {"x": 2.0 * t + 1.0}, t)
        assert mean[0, 0] == pytest.approx(3.0)


def test_at_interpolates_and_takes_the_value_after_a_change() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("at(x, 1.5)", {"x": X}, T), [[2.5], [np.nan]]
    )
    np.testing.assert_allclose(
        evaluate_reduced("at(x, 1)", {"x": X}, T), [[3.0], [6.0]]
    )
    assert np.isnan(evaluate_reduced("at(x, 5)", {"x": X}, T)).all()


def test_the_time_of_at_is_a_value_per_simulation() -> None:
    when = np.array([[0.5], [0.0]])
    np.testing.assert_allclose(
        evaluate_reduced("at(x, when)", {"x": X, "when": when}, T), [[2.0], [5.0]]
    )


def test_the_padding_and_the_steady_state_are_no_time_points() -> None:
    t = np.array([[0.0, 1.0, np.inf]])
    x = np.array([[1.0, 2.0, 100.0]])
    assert evaluate_reduced("max(x)", {"x": x}, t)[0, 0] == 2.0
    assert evaluate_reduced("mean(x)", {"x": x}, t)[0, 0] == pytest.approx(1.5)
    assert np.isnan(evaluate_reduced("at(x, 2)", {"x": x}, t)[0, 0])


def test_a_simulation_without_values_reduces_to_nan_without_a_warning() -> None:
    t = np.full((1, 3), np.nan)
    x = np.full((1, 3), np.nan)
    for formula in ("max(x)", "min(x)", "mean(x)", "at(x, 1)"):
        assert np.isnan(evaluate_reduced(formula, {"x": x}, t)).all()


def test_a_single_time_point_is_its_own_mean() -> None:
    t = np.array([[2.0, np.nan]])
    x = np.array([[7.0, np.nan]])
    assert evaluate_reduced("mean(x)", {"x": x}, t)[0, 0] == 7.0


def test_a_value_per_simulation_is_constant_in_time() -> None:
    s = np.array([[2.0], [3.0]])
    np.testing.assert_array_equal(
        evaluate_reduced("max(s) + mean(s) + at(s, 9)", {"s": s}, T), [[6.0], [9.0]]
    )


def test_an_inner_reduction_is_reduced_first() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("max(x - min(x))", {"x": X}, T), [[3.0], [2.0]]
    )


def test_a_reduction_keeps_the_unit() -> None:
    q = Q(X, "mmol/l")
    assert evaluate_reduced("max(x)", {"x": q}, T).units == q.units
    assert evaluate_reduced("mean(x)", {"x": q}, T).units == q.units
    assert evaluate_reduced("at(x, 1)", {"x": q}, T).units == q.units


@pytest.mark.parametrize("formula", ["mean(x)", "at(x, 1)"])
def test_mean_and_at_need_the_time(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_reduced(formula, {"x": X})


@pytest.mark.parametrize(
    "formula", ["max()", "mean(x, 2)", "at(x)", "at(x, 1, 2)", "max(x", "x +"]
)
def test_a_wrong_formula_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        reduce_formula(formula)


def test_an_identifier_without_a_value_is_reported() -> None:
    with pytest.raises(ValueError, match="'y'"):
        evaluate_reduced("x + max(y)", {"x": X}, T)


def test_the_symbols_of_a_reduced_formula() -> None:
    reduced = reduce_formula("ins / at(ins, t0) + max([glc])")
    assert reduced.symbols == ("[glc]", "ins", "t0")
    assert reduced.outer_symbols == ("ins",)
    assert reduced.time_symbols == ("t0",)
    assert len(reduced.reductions) == 2


def test_an_identifier_with_a_dot_is_one_symbol() -> None:
    compiled = compile_formula("hctz.auc_inf_obs / hctz.cmax + 1.5")
    assert compiled.symbols == ("hctz.auc_inf_obs", "hctz.cmax")
    assert compiled.evaluate([10.0, 2.0]) == pytest.approx(6.5)
