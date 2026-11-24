"""The formulas of the changes of a simulation."""

import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from sbmlsim.resources import HCTZ_SBML
from sbmlsim.simulation import Change, Formula, Simulation
from sbmlsim.simulation.scan import Scan
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.formula import (
    compile_formula,
    formula_mathml,
    linear_form,
    parse_math,
)


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


@pytest.mark.parametrize(
    ("formula", "values", "expected"),
    [
        ("1 / x", [0.0], np.inf),
        ("1 / x", [-0.0], -np.inf),
        ("x / y", [1, 0], np.inf),
        ("x / y", [np.int64(1), np.float32(0.0)], np.inf),
        ("x ^ 0.5", [-4.0], np.nan),
        ("(x - 9) ^ (1 / 3)", [1.0], np.nan),
        ("x / (2 - 2)", [1.0], np.inf),
        # the product underflows to 0 in doubles, sympy keeps it
        ("1 / (1e-200 * 1e-200)", [], np.inf),
        ("log(x)", [0.0], -np.inf),
        ("exp(x)", [1000.0], np.inf),
    ],
)
def test_formula_computes_in_doubles_of_numpy(
    formula: str, values: list[Any], expected: float
) -> None:
    """A number, a python float included, is a double of numpy, as in the math of SBML.

    A division by zero is infinite and the root of a negative number `NaN`,
    never a `ZeroDivisionError` or a complex number, and without a warning.
    """
    np.testing.assert_array_equal(compile_formula(formula).evaluate(values), expected)


def test_formula_on_arrays_gives_doubles_without_warnings() -> None:
    """The arrays of `apply` compute without the warnings of numpy."""
    compiled = compile_formula("1 / x + log(y)")
    value = compiled.apply([np.array([0.0, 2.0]), np.array([1.0, 0.0])])
    np.testing.assert_array_equal(value, [np.inf, -np.inf])


class _Leaf:
    """A number of a fake parse tree."""

    def __init__(self, text: str) -> None:
        self.text = text

    def getText(self) -> str:
        return self.text

    def accept(self, visitor: Any) -> Any:
        return visitor.visitNumber(self)


class AddExprContext:
    """An operation of a fake parse tree, of the name of the node of the grammar."""

    def __init__(self, *children: Any) -> None:
        self.children = children

    def getChild(self, i: int) -> Any:
        return self.children[i]

    def getText(self) -> str:
        return "".join(child.getText() for child in self.children)

    def accept(self, visitor: Any) -> Any:
        return visitor.visitAddExpr(self)


@pytest.mark.parametrize("token", ["; import os", "**", "//", "+ 1 +"])
def test_formula_operator_is_one_of_a_table(token: str) -> None:
    """An operator token is mapped through a fixed table, no text reaches the code."""
    from sbmlsim.simulator.formula import _NumpyCode

    tree = AddExprContext(_Leaf("1"), _Leaf(token), _Leaf("2"))
    with pytest.raises(ValueError, match="Unexpected operator"):
        _NumpyCode().compile(tree)


def test_a_function_of_a_reciprocal_reads_its_whole_argument() -> None:
    """`arcsec(x + 1)` is `arccos(1 / (x + 1))`, not `arccos(1 / x + 1)`."""
    x = 1.7
    assert compile_formula("arcsec(x + 1)").evaluate([x]) == np.arccos(1.0 / (x + 1))
    assert compile_formula("arccoth(x * 2)").evaluate([x]) == np.arctanh(1.0 / (x * 2))
    assert compile_formula("arccsc(x / 0.5)").evaluate([x]) == np.arcsin(
        1.0 / (x / 0.5)
    )


#: the values of the identifiers of the formulas of the MathML
MATHML_VALUES = {"x": 1.7, "y": 0.4, "z": -2.5}

#: formulas which cover every construct of the math of PEtab, and whether
#: roadrunner computes them bitwise as numpy does: the operations IEEE 754
#: rounds correctly (arithmetic, `sqrt`, comparisons, `abs`, `piecewise`) are
#: the same on every platform, while the elementary functions (`exp`, `log`,
#: the trigonometric and hyperbolic functions and `pow` of a fraction) are the
#: ones of the libm of numpy and of roadrunner, which may differ in the last
#: bits by platform (`arcsin` on windows and on python 3.15), a logarithm of a
#: base is `log(x) / log(b)` in roadrunner and `log10` or `log2` in numpy, and
#: a function of a reciprocal of a sum rounds twice
MATHML_FORMULAS = [
    ("x / 90", True),
    ("0.1 + 0.2 + x", True),
    ("x - y - 3", True),
    ("x - (y - 3)", True),
    ("-x^2", False),
    ("+x * -y", True),
    ("2^3^2 + x", True),
    ("pow(x, y)", False),
    ("x * y / 7", True),
    ("1e-3 * x - 2.5e-7", True),
    ("1e300 * x * 1e10", True),
    ("inf - x", True),
    ("min(x, y) + max(x, z)", True),
    ("min(x / 110, 1)", True),
    ("sign(z) + sign(x) + sign(y - y)", True),
    ("abs(z) + sqrt(x)", True),
    ("exp(x) + ln(x) + log(y)", False),
    ("log(x, 10)", False),
    ("log(x, 2) + log2(y)", False),
    ("log(x, 3)", False),
    ("log10(x)", False),
    ("sin(x) + cos(x) + tan(x)", False),
    ("sinh(x) + cosh(x) + tanh(y)", False),
    ("arcsin(y) + arccos(y) + arctan(z)", False),
    ("arcsinh(x) + arccosh(x) + arctanh(y)", False),
    ("sec(x) + csc(x) + cot(x)", False),
    ("sech(x) + csch(x) + coth(x)", False),
    ("arcsec(x + 1) + arccsc(x + 1) + arccot(x)", False),
    ("arcsech(y) + arccsch(x) + arccoth(x + 1)", False),
    ("piecewise(1, x > 2, 2, y < 1, 3)", True),
    ("piecewise(x, x >= 2 && y <= 3 || !(x == y), y)", True),
    ("piecewise(x, x != y, y) + piecewise(z, x, y)", True),
    ("true + false + x", True),
    ("piecewise(1, true, 0) * x", True),
]


def _evaluated_by_roadrunner(mathml: str, values: dict[str, float]) -> float:
    """Evaluate MathML as the initial assignment of a parameter in roadrunner."""
    import libsbml
    import roadrunner

    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("formula")
    for sid, value in values.items():
        parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setValue(value)
        parameter.setConstant(True)
    out = model.createParameter()
    out.setId("out")
    out.setValue(0.0)
    out.setConstant(True)
    math = libsbml.readMathMLFromString(mathml)
    assert math is not None, mathml
    assignment = model.createInitialAssignment()
    assignment.setSymbol("out")
    assignment.setMath(math)
    assert doc.checkInternalConsistency() == 0
    r = roadrunner.RoadRunner(libsbml.writeSBMLToString(doc))
    return float(r.getValue("out"))


@pytest.mark.parametrize(("formula", "bitwise"), MATHML_FORMULAS)
def test_the_mathml_of_a_formula_computes_the_compiled_formula(
    formula: str, bitwise: bool
) -> None:
    """The MathML is the formula of `compile_formula`, evaluated by roadrunner."""
    compiled = compile_formula(formula)
    expected = compiled.evaluate([MATHML_VALUES[s] for s in compiled.symbols])
    actual = _evaluated_by_roadrunner(formula_mathml(formula), MATHML_VALUES)
    if bitwise:
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_the_mathml_of_a_formula_names_the_identifiers_of_the_model() -> None:
    mathml = formula_mathml("min(crcl / 110, 1)", {"crcl": "panel_knob__crcl"})
    assert "<ci> panel_knob__crcl </ci>" in mathml
    assert "crcl <" not in mathml.replace("panel_knob__crcl", "")
    assert _evaluated_by_roadrunner(mathml, {"panel_knob__crcl": 55.0}) == 0.5


def test_the_mathml_of_a_formula_writes_numbers_as_their_doubles() -> None:
    """A number is written with the shortest digits of its double."""
    mathml = formula_mathml("0.1 * x + 1e-300 + 12345678901234567890")
    assert '<cn type="real"> 0.1 </cn>' in mathml
    assert '<cn type="real"> 12345678901234567000.0 </cn>' in mathml
    assert (
        _evaluated_by_roadrunner(formula_mathml("1e-300 * x"), {"x": 2.0})
        == 1e-300 * 2.0
    )


def test_the_time_of_a_formula_is_the_time_of_sbml() -> None:
    mathml = formula_mathml("time + 1")
    assert "http://www.sbml.org/sbml/symbols/time" in mathml
    import libsbml

    assert libsbml.formulaToL3String(libsbml.readMathMLFromString(mathml)) == (
        "time + 1"
    )


@pytest.mark.parametrize("formula", ["[S] + 1", "hctz.cmax * 2", "x +", "1 / 0"])
def test_the_mathml_of_a_formula_refuses_what_sbml_does_not_read(formula: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"'{formula}'")):
        formula_mathml(formula)


class TestLinearForm:
    """A formula read as a linear form, `constant + sum(factor * symbol)`.

    Two callers read one and both ask it of this function: the executor over
    the entities of a model, for a form of a change the engine evaluates from
    its own state (`sbmlsim.simulator.executor_core.form_of`), and a fit over
    the parameters it estimates, for a derived value of a condition the engine
    evaluates at the values its optimizer is at (`sbmlsim.fit.fit_core`).
    """

    @pytest.mark.parametrize(
        "formula, expected",
        [
            ("a", (0.0, [(0, 1.0)])),
            ("a + b", (0.0, [(0, 1.0), (1, 1.0)])),
            ("2 * a + 3", (3.0, [(0, 2.0)])),
            ("a - b", (0.0, [(0, 1.0), (1, -1.0)])),
            ("a / 2", (0.0, [(0, 0.5)])),
            ("3 * b", (0.0, [(1, 3.0)])),
        ],
    )
    def test_a_linear_form_is_read_off_a_formula(
        self, formula: str, expected: tuple[float, list[tuple[int, float]]]
    ) -> None:
        """A formula linear in the named symbols gives its constant and factors."""
        assert linear_form(formula, {"a": 0, "b": 1}) == expected

    @pytest.mark.parametrize(
        "formula",
        [
            "a * b",  # not linear
            "a * a",  # of a higher degree
            "exp(a)",  # no polynomial
            "1 / a",  # no polynomial
            "3",  # a number the caller folds itself
            "a + c",  # reads a symbol the caller did not name
            "c",  # the same
        ],
    )
    def test_a_formula_which_is_no_linear_form_is_none(self, formula: str) -> None:
        """A formula which is no linear form over the named symbols is refused."""
        assert linear_form(formula, {"a": 0, "b": 1}) is None

    def test_the_factor_of_a_symbol_is_the_one_of_the_formula(self) -> None:
        """The form is over the symbols of the model, so it carries no unit."""
        # a form of a fit reads the entity of the model a parameter writes,
        # whose value is the one the engine sets, so a factor of the unit of
        # that parameter is already in the coefficient of the formula
        assert linear_form("0.25 * a", {"a": 0}) == (0.0, [(0, 0.25)])

    def test_a_constant_of_the_form_is_substituted_before_it_is_read(self) -> None:
        """A symbol of a known value is a number of the form, not a term."""
        assert linear_form("a + 2 * c", {"a": 0}, {"c": 3.0}) == (6.0, [(0, 1.0)])

    def test_a_form_of_constants_alone_is_none(self) -> None:
        """A formula whose symbols are all known is a number the caller folds."""
        assert linear_form("2 * c", {"a": 0}, {"c": 3.0}) is None

    @pytest.mark.parametrize(
        "formula, expected",
        [
            ("[A] + 5", (5.0, [(0, 1.0)])),
            ("c + [A]", (0.0, [(1, 1.0), (0, 1.0)])),
            ("2 * [A]", (0.0, [(0, 2.0)])),
            ("pk.cmax / 2", (0.0, [(2, 0.5)])),
        ],
    )
    def test_a_form_reads_the_selections_of_roadrunner(
        self, formula: str, expected: tuple[float, list[tuple[int, float]]]
    ) -> None:
        """The concentration of a species and a dotted name are symbols (#630).

        The parser of the math of PEtab reads neither the brackets of a
        concentration nor a dotted name, so such a formula raised instead of
        being read as a form and every caller folded it. The formula is
        escaped for the parser as `compile_formula` escapes it and every
        symbol is mapped back, so it is the same formula to both.
        """
        assert linear_form(formula, {"[A]": 0, "c": 1, "pk.cmax": 2}) == expected

    def test_a_concentration_of_a_known_value_is_substituted(self) -> None:
        """A constant of the form is named by its selection (#630)."""
        assert linear_form("a + 2 * [A]", {"a": 0}, {"[A]": 3.0}) == (
            6.0,
            [(0, 1.0)],
        )

    def test_a_concentration_which_is_no_form_is_none(self) -> None:
        """A formula of a concentration which is not linear is refused (#630)."""
        assert linear_form("[A] * [A]", {"[A]": 0}) is None
        assert linear_form("[B] + 1", {"[A]": 0}) is None
