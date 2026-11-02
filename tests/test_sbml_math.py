"""The math of SBML compiled for the events of a model (#299)."""

import math

import libsbml
import numpy as np
import pytest

from sbmlsim.sbml_math import AVOGADRO, compile_math, math_formula

#: the values of the identifiers of the formulas
VALUES = {"a": 2.5, "b": 0.5, "q": 49.0, "n": -8.0}


def _evaluate(formula: str) -> float:
    """Compile an L3 formula of SBML and evaluate it on `VALUES`."""
    settings = libsbml.L3ParserSettings()
    settings.setParseUnits(False)
    astnode = libsbml.parseL3FormulaWithSettings(formula, settings)
    assert astnode is not None, libsbml.getLastParseL3Error()
    compiled = compile_math(libsbml.writeMathMLToString(astnode))
    return compiled.evaluate([VALUES[s] for s in compiled.symbols])


#: the functions which the math of PEtab names otherwise or does not have
FUNCTIONS = {
    "root(3, a)": 2.5 ** (1 / 3),
    "floor(a)": 2.0,
    "ceiling(a)": 3.0,
    "factorial(3)": 6.0,
    "factorial(a)": math.gamma(3.5),
    "quotient(7, 2)": 3.0,
    "quotient(-7, 3)": -2.0,
    "rem(7, 2)": 1.0,
    "rem(-7, 3)": -1.0,
    "xor(a > b, a < b)": 1.0,
    "xor(a > b, a > 0, b > 0)": 1.0,
    "xor(a > b, a > 0)": 0.0,
    "implies(a > b, a > 0)": 1.0,
    "implies(a > b, a < 0)": 0.0,
    "implies(a < b, a < 0)": 1.0,
    "geq(a, b, b)": 1.0,
    "eq(a, a, a)": 1.0,
    "eq(a, a, b)": 0.0,
    "lt(b, a, b)": 0.0,
    "arcsin(b)": math.asin(0.5),
    "arccos(b)": math.acos(0.5),
    "arctan(a)": math.atan(2.5),
    "arcsec(a)": math.acos(1 / 2.5),
    "infinity": math.inf,
    "-infinity": -math.inf,
    "pi": math.pi,
    "exponentiale": math.e,
    "avogadro": AVOGADRO,
    "log(2, 8)": 3.0,
    "log10(1000)": 3.0,
    "ln(a)": math.log(2.5),
    "piecewise(1, a > b, 2)": 1.0,
    "max(a, b, 3)": 3.0,
}


@pytest.mark.parametrize(("formula", "value"), FUNCTIONS.items(), ids=FUNCTIONS)
def test_the_functions_of_sbml_are_evaluated(formula: str, value: float) -> None:
    assert _evaluate(formula) == pytest.approx(value, rel=1e-15)


def test_notanumber_is_not_a_number() -> None:
    assert math.isnan(_evaluate("notanumber"))


#: math whose value is no real number, which roadrunner computes as a double
DOUBLES = {
    "root(3, n)": math.nan,
    "n^0.5": math.nan,
    "sqrt(n)": math.nan,
    "1 / (a - a)": math.inf,
    "ln(a - a)": -math.inf,
    "piecewise(1, a < b)": math.nan,
}


@pytest.mark.parametrize(("formula", "value"), DOUBLES.items(), ids=DOUBLES)
def test_a_value_is_a_double_without_an_error(formula: str, value: float) -> None:
    """Python raises and numpy warns, roadrunner computes a double."""
    np.testing.assert_equal(_evaluate(formula), value)


#: math whose value is the double python computes, bitwise
EXACT = {
    "q / 49": 49.0 / 49,
    "quotient(q, 49)": 1.0,
    "rem(q, 49)": 0.0,
    "1.1e-5 * a": 1.1e-5 * 2.5,
    "0.1 + 0.2": 0.1 + 0.2,
    "avogadro / 1e23": AVOGADRO / 1e23,
}


@pytest.mark.parametrize(("formula", "value"), EXACT.items(), ids=EXACT)
def test_the_operations_of_the_math_are_kept(formula: str, value: float) -> None:
    """`q / 49` is not `q * (1/49)`, which is less than 1 for `q = 49`."""
    assert _evaluate(formula) == value


@pytest.mark.parametrize("formula", ["rateOf(a)", "delay(a, 1)"])
def test_math_without_a_value_at_a_time_is_not_compiled(formula: str) -> None:
    with pytest.raises(ValueError, match="no value at a time"):
        _evaluate(formula)


def test_the_identifiers_are_selections() -> None:
    """A concentration `[S]` is an identifier of MathML, the time a csymbol."""
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>'
        '<ci> [S] </ci><ci> k </ci><csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol>'
        "</apply></math>"
    )
    compiled = compile_math(mathml)
    assert compiled.symbols == ("[S]", "k", "time")
    assert compiled.evaluate([1.0, 2.0, 4.0]) == 7.0
    assert math_formula(mathml) == compiled.formula == "[S] + k + time"
