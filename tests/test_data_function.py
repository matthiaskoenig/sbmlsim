"""The formula of a Data of type FUNCTION is the math of PEtab."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.data import evaluate_function


def test_a_ratio_of_quantities_keeps_the_units() -> None:
    x = Q(np.array([1.0, 2.0]), "mmol/l")
    y = Q(np.array([2.0, 4.0]), "mmol/l")
    ratio = evaluate_function("x / y", {"x": x, "y": y})
    np.testing.assert_allclose(ratio.to("dimensionless").magnitude, [0.5, 0.5])
    amount = evaluate_function("x * v", {"x": x, "v": Q(2.0, "l")})
    assert amount.to("mmol").magnitude.tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Y/max(Y)", [0.25, 0.5, 1.0]),
        ("Y - min(Y)", [0.0, 1.0, 3.0]),
        ("max(Y, 2)", [2.0, 2.0, 4.0]),
        ("min(Y, 2)", [1.0, 2.0, 2.0]),
        ("Y/max(Y + Z)", [1 / 8, 2 / 8, 4 / 8]),
        ("max(max(Y), 2) + 0*Y", [4.0, 4.0, 4.0]),
        ("Ymax/max(Y)", [0.25, 0.25, 0.25]),
        ("Y^2 + ln(Z)", [1.0, 4.0, 16.0 + np.log(4.0)]),
        ("piecewise(1, Y > 1.5, 0)", [0.0, 1.0, 1.0]),
    ],
)
def test_reductions_and_petab_math(formula: str, expected: list[float]) -> None:
    variables = {
        "Y": np.array([1.0, 2.0, 4.0]),
        "Z": np.array([1.0, 1.0, 4.0]),
        "Ymax": np.array([1.0, 1.0, 1.0]),
    }
    np.testing.assert_allclose(evaluate_function(formula, variables), expected)


def test_a_reduction_ignores_the_padding() -> None:
    y = np.array([1.0, 2.0, np.nan])
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}), [0.5, 1.0, np.nan]
    )


def test_a_reduction_of_quantities_keeps_the_units() -> None:
    y = Q(np.array([1.0, 2.0, 4.0]), "mmol/l")
    normalized = evaluate_function("Y/max(Y)", {"Y": y})
    np.testing.assert_allclose(
        normalized.to("dimensionless").magnitude, [0.25, 0.5, 1.0]
    )
    shifted = evaluate_function("Y - min(Y)", {"Y": y})
    assert str(shifted.units) == str(y.units)


def test_a_formula_of_parameters_is_a_number() -> None:
    assert evaluate_function("2 * k", {"k": 3.0}) == pytest.approx(6.0)


@pytest.mark.parametrize("formula", ["Y +", "max(Y", "foo(Y)"])
def test_invalid_math_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        evaluate_function(formula, {"Y": np.array([1.0])})


def test_an_unknown_identifier_is_reported() -> None:
    with pytest.raises(ValueError, match="W"):
        evaluate_function("Y / W", {"Y": np.array([1.0])})


def test_a_reduction_of_a_scan_is_per_simulation() -> None:
    y = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 2.0]])
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}), [[0.25, 0.5, 1.0], [0.5, 0.5, 1.0]]
    )
    np.testing.assert_allclose(evaluate_function("max(Y)", {"Y": y}), [4.0, 2.0])


def test_a_reduction_of_a_simulation_is_a_number() -> None:
    value = evaluate_function("max(Y) + k", {"Y": np.array([1.0, 3.0]), "k": 1.0})
    assert np.ndim(value) == 0
    assert value == pytest.approx(4.0)


@pytest.mark.parametrize("formula", ["mean(Y)", "at(Y, 1)"])
def test_mean_and_at_are_no_reductions_of_data(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_function(formula, {"Y": np.array([1.0, 2.0])})
