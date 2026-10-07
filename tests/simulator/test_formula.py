"""The formulas of the changes of a simulation."""

import re

import pytest

from sbmlsim.simulator.formula import compile_formula


def test_formula_with_concentration_and_time() -> None:
    """`[S]` is a concentration and `time` the time of the change."""
    f = compile_formula("[A] + 2 * time + k1")
    assert f.symbols == ("[A]", "k1", "time")
    assert f.evaluate([1.0, 0.5, 3.0]) == pytest.approx(1.0 + 6.0 + 0.5)


def test_formula_math_of_petab() -> None:
    """The math is the math of PEtab, e.g. `log` is the natural logarithm."""
    assert compile_formula("log(exp(2))").evaluate([]) == pytest.approx(2.0)
    assert compile_formula("piecewise(1, time > 5, 0)").evaluate([6.0]) == 1.0
    assert compile_formula("piecewise(1, time > 5, 0)").evaluate([4.0]) == 0.0


def test_formula_is_compiled_once() -> None:
    """A formula is compiled once and cached."""
    assert compile_formula("A + 1") is compile_formula("A + 1")


def test_invalid_formula() -> None:
    """A formula which is not math is an error which names it."""
    with pytest.raises(ValueError, match=re.escape("'A +'")):
        compile_formula("A +")
