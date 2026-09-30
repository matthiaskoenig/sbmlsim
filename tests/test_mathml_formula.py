"""Tests of the formulas of SBML which are read with libsbml and sbmlmath."""

from typing import Any

import libsbml
import numpy as np
import pytest
import sympy

from sbmlsim.mathml import (
    evaluate_formula,
    expression_to_astnode,
    expression_to_formula,
    formula_expression,
    formula_symbols,
)


@pytest.mark.parametrize(
    ("formula", "variables", "expected"),
    [
        ("alpha", {"alpha": 1.3}, 1.3),
        ("prey + (alpha - 1.3)", {"prey": 1.0, "alpha": 2.0}, 1.7),
        ("0.5", {}, 0.5),
        ("2 * x^2", {"x": 3.0}, 18.0),
        ("piecewise(1, x > 2, 0)", {"x": 3.0}, 1.0),
        ("piecewise(1, x > 2, 0)", {"x": 1.0}, 0.0),
        ("exp(x) + ln(y)", {"x": 0.0, "y": 1.0}, 1.0),
        ("log10(x)", {"x": 100.0}, 2.0),
        ("log(x)", {"x": 100.0}, 2.0),
        ("time / 2", {"time": 4.0}, 2.0),
        ("pi", {}, np.pi),
    ],
)
def test_a_formula_is_evaluated(
    formula: str, variables: dict[str, float], expected: float
) -> None:
    """A formula is evaluated on the values of its identifiers."""
    assert evaluate_formula(formula, variables) == pytest.approx(expected)


@pytest.mark.parametrize(
    "symbol", ["beta", "gamma", "lambda", "S", "I", "E", "N", "O", "Q", "zeta", "re"]
)
def test_an_identifier_which_is_a_name_of_sympy(symbol: str) -> None:
    """An identifier of a model is a symbol, whatever sympy calls by its name.

    `sympify` reads `beta` and `gamma` as functions and `S`, `I` and `E` as
    its registry, the imaginary unit and the number of Euler, which
    `sbmlsim.mathml.expr_from_formula` hands the text of a formula to.
    """
    assert formula_symbols(f"2 * {symbol} + 1") == {symbol}
    assert evaluate_formula(f"2 * {symbol} + 1", {symbol: 3.0}) == pytest.approx(7.0)


def test_the_values_of_a_formula_are_arrays() -> None:
    """A formula of arrays is an array."""
    values = evaluate_formula("a * time", {"a": 2.0, "time": np.array([1.0, 2.0])})
    np.testing.assert_allclose(values, [2.0, 4.0])


def test_the_symbols_of_a_formula() -> None:
    """The identifiers are the symbols, the time of the model is `time`."""
    assert formula_symbols("prey + (alpha - 1.3) * time") == {"prey", "alpha", "time"}
    assert formula_symbols("0.5") == set()
    assert formula_symbols("pi * 2") == set()


def test_values_which_are_not_used() -> None:
    """The values of other identifiers are ignored."""
    assert evaluate_formula("a", {"a": 1.0, "b": 2.0}) == 1.0


@pytest.mark.parametrize("formula", ["", "  ", "x +", "1 +* 2", "(x"])
def test_a_formula_which_is_not_math(formula: str) -> None:
    """A formula which cannot be read names itself."""
    with pytest.raises(ValueError, match=r"The formula '.*' is (empty|not valid math)"):
        formula_expression(formula)


@pytest.mark.parametrize("formula", [None, 1.0, ["x"]])
def test_a_formula_which_is_not_text(formula: Any) -> None:
    """A formula is text, a number is not read as one."""
    with pytest.raises(ValueError, match="is empty"):
        formula_expression(formula)


@pytest.mark.parametrize("formula", ["f(x)", "rateOf(x)", "delay(x, 1)"])
def test_a_function_without_a_value(formula: str) -> None:
    """A function which sympy cannot evaluate is an error and not a symbol."""
    with pytest.raises(ValueError, match="uses the functions"):
        formula_expression(formula)


def test_an_identifier_without_a_value() -> None:
    """An identifier without a value names itself and the values."""
    with pytest.raises(ValueError, match=r"'a \+ b' uses \['b'\].*\['a', 'c'\]"):
        evaluate_formula("a + b", {"a": 1.0, "c": 2.0})


@pytest.mark.parametrize(
    ("expression", "formula"),
    [
        (sympy.log(sympy.Symbol("x")), "ln(x)"),
        (sympy.tanh(sympy.Symbol("x")), "tanh(x)"),
        (sympy.Symbol("x") ** 2, "x^2"),
        (sympy.Float(1.5) * sympy.Symbol("net1__layer1__weight__0_1"), None),
        (sympy.Piecewise((sympy.Symbol("x"), sympy.Symbol("x") > 0), (0, True)), None),
        (sympy.Abs(sympy.Symbol("x")), "abs(x)"),
        (sympy.exp(-sympy.Symbol("x")), "exp(-x)"),
    ],
)
def test_an_expression_is_a_formula(
    expression: sympy.Basic, formula: str | None
) -> None:
    """An expression is written as a formula which has its value."""
    written = expression_to_formula(expression)
    if formula is not None:
        assert written == formula
    # no unit of a number, which the math of SBML level 2 does not have
    assert "dimensionless" not in written
    symbols = sorted(expression.free_symbols, key=str)
    for value in (-0.75, 0.5, 2.0):
        if expression.has(sympy.log) and value <= 0:
            continue
        variables = {str(symbol): value for symbol in symbols}
        expected = float(sympy.N(expression.subs(dict.fromkeys(symbols, value))))
        assert evaluate_formula(written, variables) == pytest.approx(expected)


def test_the_logarithm_of_a_formula_is_not_the_natural_logarithm() -> None:
    """`log` is the logarithm to the base 10 in a formula of SBML."""
    x = sympy.Symbol("x")
    assert expression_to_formula(sympy.log(x)) == "ln(x)"
    assert evaluate_formula("ln(x)", {"x": np.e}) == pytest.approx(1.0)
    assert evaluate_formula("log(x)", {"x": 10.0}) == pytest.approx(1.0)


def test_an_expression_is_math_of_sbml() -> None:
    """The syntax tree is what a rule of a model takes."""
    a, b = sympy.symbols("a b")
    astnode = expression_to_astnode(a * b + 1)
    assert astnode.isWellFormedASTNode()
    document = libsbml.SBMLDocument(2, 4)
    model = document.createModel()
    for sid in ("a", "b", "c"):
        parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setConstant(sid != "c")
        parameter.setValue(2.0)
    rule = model.createAssignmentRule()
    rule.setVariable("c")
    assert rule.setMath(astnode) == libsbml.LIBSBML_OPERATION_SUCCESS
    assert "units" not in libsbml.writeSBMLToString(document)


def test_an_expression_without_mathml() -> None:
    """The error function is not a function of the MathML of SBML."""
    with pytest.raises(ValueError, match=r"'erf\(x\)' has no MathML.*\['erf'\]"):
        expression_to_astnode(sympy.erf(sympy.Symbol("x")))
