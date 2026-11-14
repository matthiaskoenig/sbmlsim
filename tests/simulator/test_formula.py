"""The formulas of the changes of a simulation."""

import re
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from sbmlsim.resources import HCTZ_SBML
from sbmlsim.simulation import Change, Formula, Simulation
from sbmlsim.simulation.scan import Scan
from sbmlsim.simulator import Simulator
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


def test_the_math_of_petab_v2_is_the_parser_of_petab_v1() -> None:
    """`parse_math` reads the parser of `petab.v1.math`, which v2 re-exports."""
    import petab.v1.math
    import petab.v2.math

    assert petab.v2.math.sympify_petab is petab.v1.math.sympify_petab


#: values of `x`, `y` and `z`, many of which a reciprocal of 15 digits misses
_X = np.linspace(0.1, 1000.0, 997)
_Y = np.linspace(-3.0, 3.0, 997)
_Z = np.geomspace(1e-6, 1e6, 997)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("x/90", lambda x, y, z: x / 90.0),
        ("x / 90 * y", lambda x, y, z: x / 90.0 * y),
        ("2*x/3", lambda x, y, z: 2.0 * x / 3.0),
        ("2*(x/3)", lambda x, y, z: 2.0 * (x / 3.0)),
        ("x / (y + 7) / z", lambda x, y, z: x / (y + 7.0) / z),
        ("x - y - z", lambda x, y, z: x - y - z),
        ("x - (y - z)", lambda x, y, z: x - (y - z)),
        ("x + (y + z)", lambda x, y, z: x + (y + z)),
        ("x * (y * z)", lambda x, y, z: x * (y * z)),
        ("1e-3 * x", lambda x, y, z: 1e-3 * x),
        ("1.5e+2 / x + 2.5E-7", lambda x, y, z: 150.0 / x + 2.5e-7),
        ("7 / 90 * x", lambda x, y, z: 7.0 / 90.0 * x),
        ("-3 / x", lambda x, y, z: -3.0 / x),
        ("x * -0.1", lambda x, y, z: x * -0.1),
        ("-x^2", lambda x, y, z: -(x**2.0)),
        ("(-y)^2", lambda x, y, z: (-y) ** 2.0),
        ("x^(1/3)", lambda x, y, z: x ** (1.0 / 3.0)),
        ("pow(x, 0.5) + sqrt(x)", lambda x, y, z: x**0.5 + np.sqrt(x)),
        ("2^3^y", lambda x, y, z: 2.0 ** (3.0**y)),
        ("exp(y / 3)", lambda x, y, z: np.exp(y / 3.0)),
        ("log(z / 7)", lambda x, y, z: np.log(z / 7.0)),
        ("ln(z) / 3", lambda x, y, z: np.log(z) / 3.0),
        ("log10(z) + log2(z)", lambda x, y, z: np.log10(z) + np.log2(z)),
        ("log(z, 10) + log(z, 2)", lambda x, y, z: np.log10(z) + np.log2(z)),
        ("log(z, 3)", lambda x, y, z: np.log(z) / np.log(3.0)),
        ("abs(y) / 9 + sign(y)", lambda x, y, z: np.abs(y) / 9.0 + np.sign(y)),
        ("sin(y) / cos(y) - tan(y)", lambda x, y, z: np.sin(y) / np.cos(y) - np.tan(y)),
        ("sec(y)", lambda x, y, z: 1.0 / np.cos(y)),
        ("arccot(x)", lambda x, y, z: np.arctan(1.0 / x)),
        (
            "max(min(x / 3, exp(y) / 7), log(z) / 11)",
            lambda x, y, z: np.maximum(
                np.minimum(x / 3.0, np.exp(y) / 7.0), np.log(z) / 11.0
            ),
        ),
        (
            "piecewise(x / 3, x > 10 && y <= 2, y / 7, !(x == y), 1e-3)",
            lambda x, y, z: np.where(
                (x > 10.0) & (y <= 2.0), x / 3.0, np.where(x != y, y / 7.0, 1e-3)
            ),
        ),
        (
            "piecewise(z / 3, y, 0.1 + 0.2)",
            lambda x, y, z: np.where(y != 0.0, z / 3.0, 0.1 + 0.2),
        ),
        ("x - x", lambda x, y, z: x - x),
        ("(x + y) * z / 3", lambda x, y, z: (x + y) * z / 3.0),
        ("-(x - y) / 3", lambda x, y, z: -(x - y) / 3.0),
        ("(2 > 1) * x / 3", lambda x, y, z: 1.0 * x / 3.0),
    ],
)
def test_formula_operations_as_written(
    formula: str, expected: Callable[..., np.ndarray]
) -> None:
    """A number is the double of its text and the operations are the ones written.

    sympy folds `x/90` into `0.0111111111111111*x`, a reciprocal of 15 digits,
    and `2*x/3` into `0.666666666666667*x`; the compiled formula divides once,
    in the order written, as numpy does.
    """
    compiled = compile_formula(formula)
    values = {"x": _X, "y": _Y, "z": _Z}
    with np.errstate(all="ignore"):
        result = compiled.apply([values[s] for s in compiled.symbols])
        reference = expected(_X, _Y, _Z)
    np.testing.assert_array_equal(np.broadcast_to(result, _X.shape), reference)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("min(45 / 90, 1)", 0.5),
        ("45 / 90", 0.5),
        ("0.1 + 0.2", 0.1 + 0.2),
        ("(0.1 + 0.2) + 0.3", (0.1 + 0.2) + 0.3),
        ("0.1 + (0.2 + 0.3)", 0.1 + (0.2 + 0.3)),
        ("7 / 90", 7.0 / 90.0),
        ("10 / 3", 10.0 / 3.0),
        ("0.30000000000000004", 0.30000000000000004),
        ("12345678901234567891", 12345678901234567891.0),
        ("1e-3", 0.001),
        ("2^-1", 0.5),
        ("-2^2", -4.0),
        ("log(1000, 10)", 3.0),
        ("log10(1000)", 3.0),
        ("log(8, 2)", 3.0),
        ("max(-1 / 3, -0.5)", -1.0 / 3.0),
        ("piecewise(1 / 3, 2 > 1, 0)", 1.0 / 3.0),
        ("true + 1 / 3", 1.0 + 1.0 / 3.0),
        ("1 / inf", 0.0),
        ("log(0)", -np.inf),
        ("arccoth(1)", np.inf),
    ],
)
def test_formula_numbers_are_doubles(formula: str, expected: float) -> None:
    """A formula of numbers is evaluated in doubles, not folded at 15 digits.

    A function at its pole is infinite without the warning of numpy.
    """
    assert compile_formula(formula).evaluate([]) == expected


def test_formula_reads_every_symbol_it_names() -> None:
    """The formula is not simplified: `x - x` reads `x` and is `NaN` at infinity."""
    compiled = compile_formula("x - x + 0 * y")
    assert compiled.symbols == ("x", "y")
    assert np.isnan(compiled.evaluate([np.inf, 1.0]))


def test_formula_negation_of_a_comparison_with_nan() -> None:
    """`!(x > 1)` is true for `NaN`, which `x <= 1` is not."""
    compiled = compile_formula("piecewise(1, !(x > 1), 0)")
    assert compiled.evaluate([np.nan]) == 1.0
    assert compiled.evaluate([2.0]) == 0.0


@pytest.mark.parametrize("formula", ["1/0", "sqrt(-1)", "(x > 1) + 1", "nan + x"])
def test_formula_which_petab_refuses(formula: str) -> None:
    """A formula the parser of PEtab refuses is refused as before."""
    with pytest.raises(ValueError, match="is not valid math"):
        compile_formula(formula)


def test_observable_divides_as_written() -> None:
    """An observable `[Cve_hctz] / 90` is the selection divided by 90, bitwise."""
    times = np.linspace(0, 24 * 60, 145).tolist()
    simulation = Simulation(
        end=24 * 60, preinit_changes={"PODOSE_hctz": 10.0}, times=times
    )
    result = Simulator().run(
        HCTZ_SBML,
        Scan(simulation, []),
        [Formula("c90", "[Cve_hctz] / 90", unit="mM")],
        keep=["[Cve_hctz]", "c90"],
    )
    np.testing.assert_array_equal(
        result.ds["c90"].values, result.ds["[Cve_hctz]"].values / 90.0
    )


def test_change_and_preinit_formulas_divide_as_written() -> None:
    """The value of a formula of a change and of the pre-initialization."""
    simulation = Simulation(
        end=60,
        preinit_changes={"PODOSE_hctz": "10 / 3"},
        changes=[Change(30.0, {"PODOSE_hctz": "PODOSE_hctz + 7 / 90"})],
        times=[0.0, 30.0, 60.0],
    )
    result = Simulator().run(HCTZ_SBML, Scan(simulation, []), keep=["PODOSE_hctz"])
    dose = result.ds["PODOSE_hctz"].values
    assert dose[0] == 10.0 / 3.0
    # the state before and after the change at its time
    assert dose[2] == dose[1] + 7.0 / 90.0


def test_formula_of_many_terms() -> None:
    """A sum of many terms is added in the order written, without nesting."""
    n = 250
    compiled = compile_formula(" + ".join(f"x{i} / {i + 1}" for i in range(n)))
    terms = np.linspace(0.1, 9.0, n)
    values = {f"x{i}": term for i, term in enumerate(terms)}
    expected = terms[0] / 1.0
    for i in range(1, n):
        expected = expected + terms[i] / (i + 1)
    assert compiled.evaluate([values[s] for s in compiled.symbols]) == expected
