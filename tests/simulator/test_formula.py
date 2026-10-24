"""The formulas of the changes of a simulation."""

import re
import subprocess
import sys

import pytest

from sbmlsim.simulator.formula import compile_formula, parse_math


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


def test_compiling_a_formula_does_not_import_petab() -> None:
    """The parser of the math of PEtab is read without the package petab.

    `petab.v2.math` imported petab, which imports its SciML extension with
    torch, matplotlib.pyplot and scipy.stats: 1.0 to 1.4 s for the first
    formula of a process and of every worker of a pool.
    """
    code = (
        "import sys\n"
        "from sbmlsim.simulator.formula import compile_formula\n"
        "assert compile_formula('[S] * k + max(a, 2)').symbols == ('[S]', 'a', 'k')\n"
        "print(sorted(m for m in ('petab', 'torch', 'matplotlib.pyplot', 'scipy.stats')"
        " if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "formula",
    [
        "-x^2",
        "2^3^2",
        "a && b || !c",
        "piecewise(1, x > 2, 0)",
        "log(x, 2) + ln(y) + log10(z) + log2(w)",
        "x == y",
        "true + false",
        "abs(sign(-x)) * exp(pi) / inf",
        "min(a, b) - max(a, b)",
        "1e-3 * x - 0.5",
    ],
)
def test_the_parser_is_the_one_of_petab(formula: str) -> None:
    from petab.v2.math import sympify_petab

    assert parse_math(formula) == sympify_petab(formula)
