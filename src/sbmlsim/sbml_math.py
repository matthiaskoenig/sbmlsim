"""The math of SBML as sympy expressions and numpy functions.

The MathML of SBML is read with `sbmlmath` into sympy expressions, with the
meaning SBML gives it (`MathMLParser`): `rem` and `quotient` truncate, a
number in e-notation is the double of its text, and a logarithm with a base is
a quotient of natural logarithms (`log_quotient`). `sbmlsim.sciml.formula`
reads the formulas of the networks with it.

`compile_math` compiles the math of the events of a model, which the executor
evaluates when a change of a simulation makes the trigger of an event true
(`sbmlsim.model.symbols.EventSymbols`, `sbmlsim.simulator.executor`): every
function of the MathML of SBML (#299), the function definitions of the model
are expanded when its symbols are read. `rateOf` and `delay` have no value
at a single time, a math with them is no math `compile_math` compiles. The
values are doubles as roadrunner computes them: a division by zero is
infinite, the root of a negative number is not a number, and every number of
the math is the double of its text.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import libsbml
import numpy as np
import sympy
from sbmlmath import SBMLMathMLParser, TimeSymbol
from sbmlmath.csymbol import CSymbol
from sympy.printing.numpy import NumPyPrinter

from sbmlsim.simulator.formula import CompiledFormula

#: the namespace of MathML
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"

#: the value of the Avogadro constant which roadrunner evaluates, the one of
#: SBML Level 3 Version 1
AVOGADRO = 6.02214179e23

#: the name of the time of the model in a compiled math
TIME = "time"


def _truncate(quotient: sympy.Expr) -> sympy.Expr:
    """Round a quotient towards zero, as `quotient` and `rem` of SBML do.

    Args:
        quotient: the expression to round.

    Returns:
        The expression, a piecewise of `floor` and `ceiling`, which the MathML
        of SBML has.
    """
    return sympy.Piecewise(
        (sympy.floor(quotient), quotient >= 0), (sympy.ceiling(quotient), True)
    )


class MathMLParser(SBMLMathMLParser):
    """The MathML parser of `sbmlmath` with the meaning SBML gives the math.

    `sbmlmath` reads `rem` and `quotient` with the modulo of sympy, which
    rounds towards negative infinity. The `rem` of SBML is the remainder of
    the truncated division, `rem(-7, 3)` is -1, and the `quotient` is the
    truncated quotient, `quotient(-7, 3)` is -2. A number in e-notation,
    `1.1 <sep/> -5`, is the double of `1.1e-5`, which `sbmlmath` computes as
    `1.1 * 10**-5`, a different double.
    """

    def handle_apply(self, element: Any) -> sympy.Expr:
        """Parse an `apply`, `rem` and `quotient` truncate.

        Args:
            element: the `apply` element.

        Returns:
            The expression.
        """
        operator, *operands = element
        tag = str(operator.tag).removeprefix(f"{{{MATHML_NAMESPACE}}}")
        if tag in ("rem", "quotient") and len(operands) == 2:
            x, y = (self._parse_element(operand) for operand in operands)
            if y.is_zero:
                # not defined; sympy would simplify `x - 0 * zoo` to `x`
                return sympy.nan
            # `x / y`, not `x * (1/y)`: `49 * (1/49)` is less than 1
            with sympy.evaluate(False):
                truncated = _truncate(x / y)
                return truncated if tag == "quotient" else x - y * truncated
        return super().handle_apply(element)

    def handle_cn(self, element: Any) -> sympy.Expr:
        """Parse a number, one in e-notation as the double of its text.

        Args:
            element: the `cn` element.

        Returns:
            The number.
        """
        if element.attrib.get("type") == "e-notation" and len(element):
            mantissa = (element.text or "").strip()
            exponent = (element[0].tail or "").strip()
            return sympy.Float(float(f"{mantissa}e{exponent}"))
        return super().handle_cn(element)


def log_quotient(expression: sympy.Basic) -> sympy.Basic:
    """Write the logarithms with a base as quotients of natural logarithms.

    `sbmlmath` reads `log(x)`, `log10(x)` and `log(b, x)` as `log(x, b)`,
    which neither the printer of MathML nor `lambdify` at full precision
    handle.

    Args:
        expression: the expression.

    Returns:
        The expression, in which `log(x, b)` is `log(x) / log(b)`.
    """
    return expression.replace(
        lambda node: isinstance(node, sympy.log) and len(node.args) == 2,
        lambda node: sympy.log(node.args[0]) / sympy.log(node.args[1]),
    )


def log10_of_quotient(expression: sympy.Basic) -> sympy.Basic:
    """Write a quotient with the logarithm of 10 as `log10`.

    `log(1000) / log(10)` is 2.9999999999999996, `log10(1000)` is 3.

    Args:
        expression: the expression, in which a base is a quotient of
            logarithms, see `log_quotient`.

    Returns:
        The expression, in which `log(x) / log(10)` is `log10(x)`, which a
        function of `lambdify` evaluates with `numpy.log10`.
    """
    base = 1 / sympy.log(10)
    log10: Any = sympy.Function("log10")

    def replace(product: sympy.Mul) -> sympy.Basic:
        logarithm = next(arg for arg in product.args if isinstance(arg, sympy.log))
        rest = [arg for arg in product.args if arg not in (logarithm, base)]
        return sympy.Mul(*rest, log10(logarithm.args[0]))

    return expression.replace(
        lambda node: (
            isinstance(node, sympy.Mul)
            and base in node.args
            and any(isinstance(arg, sympy.log) for arg in node.args)
        ),
        replace,
    )


class _DoublePrinter(NumPyPrinter):
    """The printer of `lambdify` which writes a number as its double.

    The printer of sympy writes a float with 15 digits, `0.30000000000000004`
    as `0.3`, a different double, and the factorial of a number with the
    factorial of python, which takes integers only.
    """

    def _print_Float(self, expr: sympy.Float) -> str:
        return repr(float(expr))

    def _print_factorial(self, expr: sympy.factorial) -> str:
        # the factorial of SBML of a real number, see `_factorial`
        return f"factorial({self._print(expr.args[0])})"


def _factorial(x: Any) -> Any:
    """Get the factorial of SBML, the gamma function of `x + 1`."""
    from scipy.special import gamma

    return gamma(np.asarray(x, dtype=float) + 1)


def math_formula(mathml: str) -> str:
    """Write the MathML of SBML as an L3 formula, e.g. for a message.

    Args:
        mathml: the MathML.

    Returns:
        The formula, the MathML itself if libsbml does not read it.
    """
    astnode: libsbml.ASTNode | None = libsbml.readMathMLFromString(mathml)
    if astnode is None:
        return mathml
    settings = libsbml.L3ParserSettings()
    settings.setParseUnits(False)
    return str(libsbml.formulaToL3StringWithSettings(astnode, settings))


@functools.cache
def math_function(mathml: str) -> tuple[tuple[str, ...], Callable[..., Any]]:
    """Get the numpy function of the MathML of SBML and the names of its arguments.

    The function computes the operations of the math as they are written,
    `q / 3` and not `(1/3) * q`, and every number is the double of its text.
    It computes on the values it is given: on numbers python raises for a
    complex value, an overflow or a division by zero, on arrays numpy gives
    `nan` and `inf`; `compile_math` gives doubles, as roadrunner computes them.

    Args:
        mathml: the MathML; the csymbol of the time is `time`, the Avogadro
            constant `AVOGADRO`. The units of its numbers are ignored.

    Returns:
        The names of the identifiers, sorted, and the function of their
        values, in this order.

    Raises:
        ValueError: if the math is not read or uses a function which has no
            value at a time: `rateOf`, `delay` or a function the MathML of
            SBML does not have.
    """
    formula = math_formula(mathml)
    try:
        expression = sympy.sympify(
            MathMLParser(ignore_units=True, floats_as_rationals=False).parse_str(mathml)
        )
    except Exception as err:
        raise ValueError(
            f"The math '{formula}' cannot be read: {type(err).__name__}: {err}"
        ) from err
    functions = sorted(
        {
            str(function.func)
            for function in expression.atoms(sympy.Function)
            if isinstance(function, sympy.core.function.AppliedUndef)
        }
    )
    if functions:
        raise ValueError(
            f"The math '{formula}' uses {functions}, which have no value at a "
            f"time of a simulation"
        )
    # the expression keeps the operations of the math, which roadrunner
    # computes, e.g. `q / 3` and not `(1/3) * q`
    with sympy.evaluate(False):
        # the time is a csymbol as well, the other one is the Avogadro constant
        expression = log_quotient(expression).xreplace(
            {
                symbol: sympy.Symbol(TIME)
                if isinstance(symbol, TimeSymbol)
                else sympy.Float(AVOGADRO)
                for symbol in expression.free_symbols
                if isinstance(symbol, CSymbol)
            }
        )
        symbols = sorted(expression.free_symbols, key=str)
        # the symbols are arguments by position: an identifier is not always
        # a name of python, e.g. `lambda` or the selection `[S]`
        arguments = [sympy.Dummy() for _ in symbols]
        function = sympy.lambdify(
            arguments,
            log10_of_quotient(
                expression.xreplace(dict(zip(symbols, arguments, strict=True)))
            ),
            modules=[{"log10": np.log10, "factorial": _factorial}, "numpy"],
            # the settings lambdify gives a printer of its own
            printer=_DoublePrinter(
                {
                    "fully_qualified_modules": False,
                    "inline": True,
                    "allow_unknown_functions": True,
                    "user_functions": {},
                }
            ),
        )
    return tuple(str(symbol) for symbol in symbols), function


@functools.cache
def compile_math(mathml: str) -> CompiledFormula:
    """Compile the MathML of SBML whose identifiers are selections, see the module.

    Args:
        mathml: the MathML, its identifiers selections of roadrunner, e.g.
            `[S]` for the concentration of a species, see `math_function`.

    Returns:
        The compiled math, its `formula` the L3 formula of the math. Its
        values are doubles, also where numpy would warn or python raise.

    Raises:
        ValueError: if the math is not read or uses a function which has no
            value at a time, see `math_function`.
    """
    symbols, function = math_function(mathml)

    def evaluate(*values: Any) -> Any:
        with np.errstate(all="ignore"):
            return function(*(np.asarray(value, dtype=float) for value in values))

    return CompiledFormula(
        formula=math_formula(mathml), symbols=symbols, _function=evaluate
    )
