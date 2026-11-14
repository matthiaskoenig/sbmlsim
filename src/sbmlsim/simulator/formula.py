"""Formulas of the math of PEtab over the selections of roadrunner.

A formula is a string of the math of PEtab whose symbols are selections of
roadrunner: `S` is the amount of a species, `[S]` its concentration and `time`
the time. The brackets are not math, so `[S]` is replaced by an identifier
before the formula is parsed and mapped back afterwards; so is an identifier
with a dot, the parameter of a PK observable, e.g. `hctz.cmax`.

A formula is compiled to a numpy function once and cached, a `CompiledFormula`
is not pickled: a plan keeps the formula as a string. The parser of PEtab
decides what is valid math, `parse_math`, and the function is written from
its parse tree with the meaning the package gives the math of SBML as well
(`sbmlsim.sbml_math`): a number is the double of its text and the operations
are the ones written, in the order written, `x / 90` is one division by 90 and
`0.1 + 0.2` is 0.30000000000000004, see `_NumpyCode`.

The formulas of observables extend the math by four reductions over the time
of one simulation, see `reduce_formula` and `evaluate_reduced`: `max(x)` and
`min(x)` with a single argument (with two or more they are the elementwise
functions of PEtab), `mean(x)`, the time weighted mean, and `at(x, t)`, the
value at a time. A value is an array whose last axis is the time of a
simulation, so a scan of many simulations reduces every simulation on its
own. The formulas of data share `reduce_formula` and have only `max` and
`min`, which reduce a labelled array along its dimension of the time or of
the rows of a dataset by name, see `sbmlsim.data.evaluate_function`.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from sbmlsim.result.timecourse import apply_weights, grid_weights

#: a concentration `[S]` in a formula
_BRACKETS = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")

#: prefix of the identifier which stands for a concentration while parsing
_PREFIX = "sbmlsim_concentration__"

#: an identifier with a dot, e.g. the parameter `cmax` of the observable `hctz`
_DOTTED = re.compile(
    r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_.])"
)

#: prefix of the identifier which stands for an identifier with a dot
_DOT = "sbmlsim_dotted__"

#: the reductions over the time of a simulation and how they are called
REDUCTIONS: dict[str, str] = {
    "max": "max(x)",
    "min": "min(x)",
    "mean": "mean(x)",
    "at": "at(x, t)",
}

#: a call of a reduction which is not the end of a longer identifier
_REDUCTION_CALL = re.compile(r"(?<![A-Za-z0-9_.])(max|min|mean|at)\s*\(")

#: prefix of the symbol which stands for the value of a reduction
_REDUCTION_PREFIX = "sbmlsim_reduction__"


@dataclass(frozen=True)
class CompiledFormula:
    """A formula compiled to a numpy function of its symbols.

    Attributes:
        formula: the formula as it was given.
        symbols: the selections the formula reads, sorted.
    """

    formula: str
    symbols: tuple[str, ...]
    _function: Callable[..., Any] = field(repr=False, compare=False)

    def evaluate(self, values: Sequence[float]) -> float:
        """Evaluate the formula.

        A function of numpy at a pole or outside of its domain gives `inf` or
        `NaN` without a warning, e.g. `log(0)` is `-inf`.

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
        with np.errstate(all="ignore"):
            return float(self._function(*values))

    def apply(self, values: Sequence[Any]) -> Any:
        """Evaluate the formula on values of any type numpy operates on.

        Unlike `evaluate`, the result is not converted to a float, arrays and
        the quantities of pint keep their shape and their units.

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
        return self._function(*values)

    def evaluate_array(self, values: Sequence[np.ndarray], size: int) -> np.ndarray:
        """Evaluate the formula on arrays of the values of its symbols.

        Args:
            values: the values of the symbols, in the order of `symbols`, one
                array of `size` values per symbol.
            size: the number of values, which a formula without symbols needs.

        Returns:
            The `size` values of the formula, `inf` or `NaN` without a warning
            where numpy gives them, see `evaluate`.
        """
        with np.errstate(all="ignore"):
            value = np.asarray(self._function(*values), dtype=float)
        return np.broadcast_to(value, (size,)).copy()


#: the name the parser of the math of PEtab is imported as, see `_parser`
_PARSER = "sbmlsim._petab_math"


def parse_math(formula: str) -> sp.Basic:
    """Parse the math of PEtab into a sympy expression, `petab.v2.math.sympify_petab`.

    Args:
        formula: the math, without the brackets of a concentration.

    Returns:
        The expression.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    return _parser()(formula)


@functools.cache
def _parser() -> Callable[[str], sp.Basic]:
    """Get the parser of the math of PEtab without importing the package petab.

    The parser of petab is the package `petab.v1.math`, which imports only
    sympy, numpy and the runtime of ANTLR, but the package `petab` imports
    the problems of PEtab v1 with matplotlib.pyplot and scipy.stats and
    `petab.v2` its SciML extension with torch: 1.0 to 1.4 s for the first
    formula of a process and of every worker of a pool. The directory of the
    parser is imported as the package `_PARSER` instead, the same code under
    another name. A process which imported petab already uses it, and a
    petab whose parser is elsewhere is imported as a whole.
    """
    if "petab" in sys.modules:
        from petab.v2.math import sympify_petab

        return sympify_petab
    spec = importlib.util.find_spec("petab")
    locations = list(spec.submodule_search_locations or []) if spec else []
    directory = Path(locations[0]) / "v1" / "math" if locations else None
    if directory is None or not (directory / "sympify.py").is_file():
        from petab.v2.math import sympify_petab

        return sympify_petab
    package = importlib.util.spec_from_file_location(
        _PARSER,
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    if package is None or package.loader is None:
        raise ImportError(f"The parser of petab in '{directory}' cannot be read.")
    module = importlib.util.module_from_spec(package)
    sys.modules[_PARSER] = module
    try:
        package.loader.exec_module(module)
    except BaseException:
        del sys.modules[_PARSER]
        raise
    return module.sympify_petab


@functools.cache
def compile_formula(formula: str) -> CompiledFormula:
    """Compile a formula, see the module.

    The formula is checked by the parser of PEtab, `parse_math`, and compiled
    from its parse tree by `_NumpyCode`: every number is the double of its
    text and every operation the one written, in the order written, e.g.
    `x / 90` divides once by `90.0` and `0.1 + 0.2` is 0.30000000000000004.
    The sympy expression of `parse_math` is not compiled, sympy evaluates on
    construction with numbers of 15 digits and writes `x / 90` as
    `0.0111111111111111 * x`.

    Args:
        formula: the formula.

    Returns:
        The compiled formula, which reads every identifier of the formula.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    dotted: dict[str, str] = {}

    def escape(match: re.Match[str]) -> str:
        return dotted.setdefault(match.group(1), f"{_DOT}{len(dotted)}")

    escaped = _DOTTED.sub(escape, formula)
    escaped = _BRACKETS.sub(lambda m: f"{_PREFIX}{m.group(1)}", escaped)
    names = {identifier: name for name, identifier in dotted.items()}

    def selection(name: str) -> str:
        if name in names:
            return names[name]
        if name.startswith(_PREFIX):
            return f"[{name.removeprefix(_PREFIX)}]"
        return name

    try:
        # the parser of PEtab decides what is valid math
        parse_math(escaped)
        code = _NumpyCode()
        body = code.compile(_parse_tree(escaped))
    except Exception as err:
        raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    ordered = sorted(code.arguments, key=selection)
    symbols = tuple(selection(identifier) for identifier in ordered)
    arguments = ", ".join(code.arguments[identifier] for identifier in ordered)
    # the code is written from the tree alone: numbers, the generated names of
    # the arguments and the functions of numpy, never a text of the formula
    function = eval(f"lambda {arguments}: {body}", {"numpy": np})
    return CompiledFormula(formula=formula, symbols=symbols, _function=function)


def _parse_tree(formula: str) -> Any:
    """Parse the math of PEtab into the parse tree of its grammar.

    Args:
        formula: the math, without the brackets of a concentration.

    Returns:
        The tree, a `PetabExpressionContext` of the parser of PEtab.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    from antlr4 import CommonTokenStream, InputStream

    grammar = sys.modules[_parser().__module__]
    lexer = grammar.PetabMathExprLexer(InputStream(formula))
    lexer.removeErrorListeners()
    lexer.addErrorListener(grammar.MathErrorListener())
    parser = grammar.PetabMathExprParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser.addErrorListener(grammar.MathErrorListener())
    return parser.petabExpression()


#: the precedences of python of the code of `_NumpyCode`, lowest first
_ADD, _MUL, _UNARY, _POW, _ATOM = range(5)

#: the functions of PEtab of one argument which numpy has
_NUMPY_FUNCTIONS: dict[str, str] = {
    "exp": "exp",
    "ln": "log",
    "log10": "log10",
    "log2": "log2",
    "sqrt": "sqrt",
    "abs": "abs",
    "sign": "sign",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arcsinh": "arcsinh",
    "arccosh": "arccosh",
    "arctanh": "arctanh",
}

#: the functions of PEtab which are the reciprocal of a function of numpy
_RECIPROCALS: dict[str, str] = {
    "sec": "cos",
    "csc": "sin",
    "cot": "tan",
    "sech": "cosh",
    "csch": "sinh",
    "coth": "tanh",
}

#: the functions of PEtab which are a function of numpy of the reciprocal
_OF_RECIPROCAL: dict[str, str] = {
    "arcsec": "arccos",
    "arccsc": "arcsin",
    "arccot": "arctan",
    "arcsech": "arccosh",
    "arccsch": "arcsinh",
    "arccoth": "arctanh",
}

#: the comparisons of PEtab as functions of numpy
_COMPARISONS: dict[str, str] = {
    "==": "equal",
    "!=": "not_equal",
    "<": "less",
    ">": "greater",
    "<=": "less_equal",
    ">=": "greater_equal",
}

#: the nodes of the operations which group to the left, see `_NumpyCode._chain`
_CHAINED = frozenset({"AddExprContext", "MultExprContext"})

#: the logarithms with a base which numpy has
_LOGARITHMS: dict[float, str] = {10.0: "log10", 2.0: "log2"}


@dataclass(frozen=True)
class _Code:
    """The python code of a node of a parse tree, see `_NumpyCode`.

    Attributes:
        text: the code.
        level: its precedence in python, `_ADD` to `_ATOM`.
        boolean: whether it is a truth value.
        literal: the value of a literal, a number or a truth value.
    """

    text: str
    level: int
    boolean: bool = False
    literal: float | bool | None = None


class _NumpyCode:
    """Write the parse tree of the math of PEtab as the code of a numpy function.

    The code computes the operations as written: a number is the double of its
    text (`repr(float(text))`, which python reads back as the same double),
    `a / b` is one division, `(a + b) + c` and `a + (b + c)` are added in
    their order, a function is the one of numpy of its name, `sec(x)` is
    `1.0 / numpy.cos(x)` and `log(x, b)` is `numpy.log(x) / numpy.log(b)`, of
    the base 10 or 2 `numpy.log10` or `numpy.log2`. A comparison is the one
    of numpy, `piecewise` is `numpy.select`, a truth value read as a number
    is 1 or 0 and a number read as a truth value is true when it is not 0, as
    the parser of PEtab reads them. Parentheses are written only where python
    would group otherwise, so a sum of many terms does not nest.

    Attributes:
        arguments: the identifiers of the formula with the names of the
            arguments of the function which stand for them.
    """

    def __init__(self) -> None:
        """Create the writer of one formula."""
        self.arguments: dict[str, str] = {}

    def compile(self, tree: Any) -> str:
        """Write the code of the parse tree of a formula.

        Args:
            tree: the parse tree, see `_parse_tree`.

        Returns:
            The code of the value of the formula; a truth value literal is 1
            or 0, as the parser of PEtab gives it.
        """
        code = self._visit(tree)
        return self._numeric(code).text if code.literal is not None else code.text

    def _visit(self, ctx: Any) -> _Code:
        result = ctx.accept(self)
        if not isinstance(result, _Code):
            raise ValueError(f"Unexpected expression: {ctx.getText()}")
        return result

    def visitChildren(self, ctx: Any) -> _Code:
        """Visit the only child of a node which wraps a rule."""
        if ctx.getChildCount() != 1:
            raise ValueError(f"Unexpected expression: {ctx.getText()}")
        return self._visit(ctx.getChild(0))

    def visitPetabExpression(self, ctx: Any) -> _Code:
        """Visit the root."""
        return self._visit(ctx.getChild(0))

    def visitParenExpr(self, ctx: Any) -> _Code:
        """Visit an expression in parentheses, which the tree groups."""
        return self._visit(ctx.getChild(1))

    def visitNumber(self, ctx: Any) -> _Code:
        """Write a number as the double of its text."""
        value = float(ctx.getText())
        if math.isinf(value):
            return _Code("numpy.inf", _ATOM, literal=value)
        return _Code(repr(value), _ATOM, literal=value)

    def visitVar(self, ctx: Any) -> _Code:
        """Write an identifier as the argument which stands for it."""
        name = self.arguments.setdefault(ctx.getText(), f"_{len(self.arguments)}")
        return _Code(name, _ATOM)

    def visitBooleanLiteral(self, ctx: Any) -> _Code:
        """Write `true` or `false`."""
        value = ctx.TRUE() is not None
        return _Code(str(value), _ATOM, boolean=True, literal=value)

    def visitUnaryExpr(self, ctx: Any) -> _Code:
        """Write `-a` or `+a`."""
        operand = self._numeric(self._visit(ctx.getChild(1)))
        if ctx.getChild(0).getText() == "+":
            return operand
        return _Code(f"-{_grouped(operand, operand.level <= _UNARY)}", _UNARY)

    def visitMultExpr(self, ctx: Any) -> _Code:
        """Write `a * b` or `a / b`."""
        return self._chain(ctx)

    def visitAddExpr(self, ctx: Any) -> _Code:
        """Write `a + b` or `a - b`."""
        return self._chain(ctx)

    def visitPowerExpr(self, ctx: Any) -> _Code:
        """Write `a ^ b` as `a ** b`, which groups to the right in python."""
        base = self._numeric(self._visit(ctx.getChild(0)))
        exponent = self._numeric(self._visit(ctx.getChild(2)))
        return _power(base, exponent)

    def _chain(self, ctx: Any) -> _Code:
        """Write a chain of operations which group to the left in python.

        The tree of `a + b + c + ...` nests to the left, one node per
        operation; it is walked down without recursion, so a sum of many terms
        is written in the depth of one.
        """
        spine = [ctx]
        while type(spine[-1].getChild(0)).__name__ in _CHAINED:
            spine.append(spine[-1].getChild(0))
        code = self._numeric(self._visit(spine[-1].getChild(0)))
        for node in reversed(spine):
            operator = node.getChild(1).getText()
            level = _MUL if operator in ("*", "/") else _ADD
            right = self._numeric(self._visit(node.getChild(2)))
            code = _Code(
                f"{_grouped(code, code.level < level)} {operator} "
                f"{_grouped(right, right.level <= level)}",
                level,
            )
        return code

    def visitComparisonExpr(self, ctx: Any) -> _Code:
        """Write a comparison as the function of numpy."""
        left = self._numeric(self._visit(ctx.getChild(0)))
        right = self._numeric(self._visit(ctx.getChild(2)))
        function = _COMPARISONS[ctx.getChild(1).getText()]
        return _Code(
            f"numpy.{function}({left.text}, {right.text})", _ATOM, boolean=True
        )

    def visitBooleanNotExpr(self, ctx: Any) -> _Code:
        """Write `!a`."""
        operand = self._condition(self._visit(ctx.getChild(1)))
        return _Code(f"numpy.logical_not({operand.text})", _ATOM, boolean=True)

    def visitBooleanAndOrExpr(self, ctx: Any) -> _Code:
        """Write `a && b` or `a || b`."""
        left = self._condition(self._visit(ctx.getChild(0)))
        right = self._condition(self._visit(ctx.getChild(2)))
        function = "logical_and" if ctx.BOOLEAN_AND() else "logical_or"
        return _Code(
            f"numpy.{function}({left.text}, {right.text})", _ATOM, boolean=True
        )

    def visitArgumentList(self, ctx: Any) -> list[_Code]:
        """Write the arguments of a call."""
        return [self._visit(child) for child in ctx.children[::2]]

    def visitFunctionCall(self, ctx: Any) -> _Code:
        """Write a call of a function of PEtab."""
        name = ctx.getChild(0).getText()
        arguments: list[_Code] = (
            ctx.getChild(2).accept(self) if ctx.getChildCount() >= 4 else []
        )
        if name == "piecewise":
            return self._piecewise(ctx, arguments)
        values = [self._numeric(argument) for argument in arguments]
        arity = 2 if name in ("pow", "min", "max") else 1
        if name == "log" and len(values) == 2:
            arity = 2
        if len(values) != arity:
            raise ValueError(
                f"Unexpected number of arguments: {len(values)} in {ctx.getText()}"
            )
        texts = [value.text for value in values]
        if name in _NUMPY_FUNCTIONS or (name == "log" and arity == 1):
            return _Code(
                f"numpy.{_NUMPY_FUNCTIONS.get(name, 'log')}({texts[0]})", _ATOM
            )
        if name in _RECIPROCALS:
            return _Code(f"1.0 / numpy.{_RECIPROCALS[name]}({texts[0]})", _MUL)
        if name in _OF_RECIPROCAL:
            return _Code(f"numpy.{_OF_RECIPROCAL[name]}(1.0 / {texts[0]})", _ATOM)
        if name == "pow":
            return _power(values[0], values[1])
        if name in ("min", "max"):
            function = "minimum" if name == "min" else "maximum"
            return _Code(f"numpy.{function}({texts[0]}, {texts[1]})", _ATOM)
        if name == "log":
            base = values[1].literal
            if isinstance(base, float) and base in _LOGARITHMS:
                return _Code(f"numpy.{_LOGARITHMS[base]}({texts[0]})", _ATOM)
            return _Code(f"numpy.log({texts[0]}) / numpy.log({texts[1]})", _MUL)
        raise ValueError(f"Unknown function: {ctx.getText()}")

    def _piecewise(self, ctx: Any, arguments: list[_Code]) -> _Code:
        """Write `piecewise(value, condition, ..., otherwise)` as `numpy.select`."""
        if len(arguments) % 2 != 1:
            raise ValueError(
                f"Unexpected number of arguments: {len(arguments)} in {ctx.getText()}"
            )
        values = [self._numeric(value).text for value in arguments[::2]]
        conditions = [self._condition(c).text for c in arguments[1::2]] + ["True"]
        return _Code(
            f"numpy.select([{', '.join(conditions)}], [{', '.join(values)}], "
            f"default=numpy.nan)",
            _ATOM,
        )

    @staticmethod
    def _numeric(code: _Code) -> _Code:
        """Read a truth value as a number, 1 or 0, as `petab.v1.math` does."""
        if not code.boolean:
            return code
        if code.literal is not None:
            value = 1.0 if code.literal else 0.0
            return _Code(repr(value), _ATOM, literal=value)
        return _Code(f"numpy.where({code.text}, 1.0, 0.0)", _ATOM)

    @staticmethod
    def _condition(code: _Code) -> _Code:
        """Read a number as a truth value, true when it is not 0."""
        if code.boolean:
            return code
        return _Code(f"numpy.not_equal({code.text}, 0.0)", _ATOM, boolean=True)


def _grouped(code: _Code, parentheses: bool) -> str:
    """Get the text of the code, in parentheses if they are needed."""
    return f"({code.text})" if parentheses else code.text


def _power(base: _Code, exponent: _Code) -> _Code:
    """Write `base ** exponent`; `**` groups to the right in python."""
    return _Code(
        f"{_grouped(base, base.level <= _POW)} ** "
        f"{_grouped(exponent, exponent.level < _POW)}",
        _POW,
    )


@dataclass(frozen=True)
class Reduction:
    """A reduction over the time of a simulation, see `reduce_formula`.

    Attributes:
        symbol: the identifier which stands for its value in the formula.
        function: `max`, `min`, `mean` or `at`.
        arguments: the formulas of its arguments, which read the symbols of
            the reductions inside of them.
    """

    symbol: str
    function: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class ReducedFormula:
    """A formula whose reductions are replaced by symbols, see `reduce_formula`.

    Attributes:
        formula: the formula as it was given.
        outer: the formula with every reduction replaced by its symbol.
        reductions: the reductions, an inner one before the one whose
            argument it is.
    """

    formula: str
    outer: str
    reductions: tuple[Reduction, ...]

    @property
    def placeholders(self) -> frozenset[str]:
        """Get the symbols which stand for the reductions."""
        return frozenset(reduction.symbol for reduction in self.reductions)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Get the identifiers the formula reads, also inside of the reductions."""
        parts = (self.outer, *(a for r in self.reductions for a in r.arguments))
        found = {s for part in parts for s in compile_formula(part).symbols}
        return tuple(sorted(found - self.placeholders))

    @property
    def outer_symbols(self) -> tuple[str, ...]:
        """Get the identifiers the formula reads outside of its reductions."""
        return tuple(
            s for s in compile_formula(self.outer).symbols if s not in self.placeholders
        )

    @property
    def time_symbols(self) -> tuple[str, ...]:
        """Get the identifiers the times of the `at` reductions read."""
        found = {
            s
            for r in self.reductions
            if r.function == "at"
            for s in compile_formula(r.arguments[1]).symbols
        }
        return tuple(sorted(found - self.placeholders))


@functools.cache
def reduce_formula(formula: str) -> ReducedFormula:
    """Find the reductions of a formula, see the module.

    Args:
        formula: the formula.

    Returns:
        The formula with its reductions.

    Raises:
        ValueError: if the parentheses are not balanced, a reduction has not
            its number of arguments or a part of the formula is not valid math.
    """
    reductions: list[Reduction] = []
    outer = _replace_reductions(formula, formula, reductions)
    for part in (outer, *(a for r in reductions for a in r.arguments)):
        try:
            compile_formula(part)
        except ValueError as err:
            raise ValueError(
                f"The formula '{formula}' is not valid math: {err}"
            ) from err
    return ReducedFormula(formula=formula, outer=outer, reductions=tuple(reductions))


def _closing_parenthesis(text: str, start: int, formula: str) -> int:
    """Find the parenthesis which closes the one opened before `start`.

    Raises:
        ValueError: if the parentheses of the formula are not balanced.
    """
    depth = 1
    for k in range(start, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"The parentheses of the formula '{formula}' are not balanced.")


def _split_arguments(text: str) -> list[str]:
    """Split the arguments of a call at the commas outside of parentheses."""
    arguments: list[str] = []
    depth = 0
    start = 0
    for k, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:k])
            start = k + 1
    arguments.append(text[start:])
    return arguments


def _replace_reductions(text: str, formula: str, reductions: list[Reduction]) -> str:
    """Replace every reduction of a text by a symbol, inner reductions first.

    Raises:
        ValueError: if the parentheses are not balanced or a reduction has not
            its number of arguments.
    """
    parts: list[str] = []
    position = 0
    while (match := _REDUCTION_CALL.search(text, position)) is not None:
        end = _closing_parenthesis(text, match.end(), formula)
        arguments = [
            _replace_reductions(argument, formula, reductions)
            for argument in _split_arguments(text[match.end() : end])
        ]
        name = match.group(1)
        parts.append(text[position : match.start()])
        if name in ("max", "min") and len(arguments) > 1:
            parts.append(f"{name}({','.join(arguments)})")
        else:
            expected = 2 if name == "at" else 1
            if len(arguments) != expected or not all(a.strip() for a in arguments):
                raise ValueError(
                    f"The reduction '{name}' of the formula '{formula}' is called "
                    f"as {REDUCTIONS[name]}."
                )
            symbol = f"{_REDUCTION_PREFIX}{len(reductions)}"
            reductions.append(
                Reduction(symbol=symbol, function=name, arguments=tuple(arguments))
            )
            parts.append(symbol)
        position = end + 1
    parts.append(text[position:])
    return "".join(parts)


def evaluate_reduced(
    formula: str,
    values: Mapping[str, Any],
    time: np.ndarray | None = None,
    tolerance: float | np.ndarray = 0.0,
    changes: Sequence[np.ndarray] | None = None,
) -> Any:
    """Evaluate a formula with reductions over the time of every simulation.

    A value is a number, a numpy array or a quantity of pint. The last axis of
    an array is the time of a simulation and the axes before it are the
    simulations, e.g. `(n_points, n_rows)` padded with `NaN` in a worker or
    `(*dims, time)` of a scan. A number, or an array whose last
    axis has one element, is a value per simulation, which is constant in
    time. A reduction keeps its axis with one element, so its value broadcasts
    against the timecourses, and keeps the unit of its argument:

    - `max(x)`, `min(x)`: the largest and the smallest value, ignoring `NaN`;
    - `mean(x)`: the trapezoidal integral divided by the time between the
      first and the last time point, the value itself for a single time point;
      the two time points of the states before and after a change at its time
      are an interval without time, which adds nothing;
    - `at(x, t)`: the value at the time `t`, a number or a value per
      simulation, interpolated linearly; the value after a change or an event
      of the model at its time, toward the state before a change at a time
      before it (time points at most `tolerance` apart are a jump, see
      `sbmlsim.result.timecourse.grid_weights`) and `NaN` outside of the time
      points.

    The times which are not finite, the padding (`NaN`) and the steady state
    after the end (`inf`), are no time points of a reduction.

    Args:
        formula: the formula.
        values: the values of its identifiers.
        time: the time points of the timecourses, of their shape; `None` where
            there are no times, e.g. for the units of a formula, which allows
            `max` and `min` of a timecourse.
        tolerance: the resolution of the time of every simulation, a number or
            an array of the shape of the simulations, e.g. `(n_points,)`, see
            `sbmlsim.simulator.plan.time_tolerance`; `at` takes two time points
            at most this apart as a jump.
        changes: the times of the changes after the start of every
            simulation, in the order of the simulations flattened, see
            `grid_weights`; `None` for none.

    Returns:
        The value of the formula.

    Raises:
        ValueError: if the formula is not valid, reads an identifier without a
            value, reduces a timecourse with `mean` or `at` without `time`,
            or `changes` are not the ones of every simulation.
    """
    reduced = reduce_formula(formula)
    scope = dict(values)
    for reduction in reduced.reductions:
        x = _apply(reduction.arguments[0], scope, formula)
        if reduction.function == "at":
            when = _apply(reduction.arguments[1], scope, formula)
            scope[reduction.symbol] = _at(x, when, time, tolerance, formula, changes)
        elif reduction.function == "mean":
            scope[reduction.symbol] = _mean(x, time, formula)
        else:
            scope[reduction.symbol] = _extreme(reduction.function, x, time)
    return _apply(reduced.outer, scope, formula)


def _apply(part: str, values: Mapping[str, Any], formula: str) -> Any:
    """Evaluate a part of a formula without reductions on the values.

    Raises:
        ValueError: if the part reads an identifier which has no value.
    """
    compiled = compile_formula(part)
    missing = [symbol for symbol in compiled.symbols if symbol not in values]
    if missing:
        raise ValueError(
            f"The formula '{formula}' reads {missing}, which have no values."
        )
    return compiled.apply([values[symbol] for symbol in compiled.symbols])


def _split(x: Any) -> tuple[np.ndarray, Any]:
    """Split a value into a float array and its unit, `None` without one."""
    units = getattr(x, "units", None)
    if units is not None and hasattr(x, "magnitude"):
        return np.asarray(x.magnitude, dtype=float), units
    return np.asarray(x, dtype=float), None


def _join(magnitude: np.ndarray, units: Any) -> Any:
    """Give an array its unit again."""
    return magnitude if units is None else magnitude * units


def _constant(magnitude: np.ndarray) -> bool:
    """Check whether a value is constant in time: a number or a value per simulation."""
    return magnitude.ndim == 0 or magnitude.shape[-1] == 1


def _times(magnitude: np.ndarray, time: np.ndarray | None) -> np.ndarray:
    """Get the time points of a reduction: the finite times, every point without times."""
    if time is None:
        return np.ones(magnitude.shape, dtype=bool)
    return np.broadcast_to(np.isfinite(np.asarray(time, dtype=float)), magnitude.shape)


def _extreme(function: str, x: Any, time: np.ndarray | None) -> Any:
    """Reduce to the largest or the smallest value, ignoring `NaN`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    masked = np.where(_times(magnitude, time), magnitude, np.nan)
    ufunc = np.fmax if function == "max" else np.fmin
    # fmax and fmin ignore NaN and give NaN for a row of NaN, without a warning
    return _join(ufunc.reduce(masked, axis=-1, keepdims=True), units)


def _no_time(function: str, formula: str) -> ValueError:
    """Get the error of a reduction which needs the times and has none."""
    return ValueError(
        f"'{function}' in the formula '{formula}' needs the time points of the "
        f"simulation, which were not given; it is a reduction of the Formula "
        f"observables of a scan."
    )


def _mean(x: Any, time: np.ndarray | None, formula: str) -> Any:
    """Reduce to the time weighted mean, see `evaluate_reduced`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    if time is None:
        raise _no_time("mean", formula)
    t = np.broadcast_to(np.asarray(time, dtype=float), magnitude.shape)
    valid = np.isfinite(t)
    segment = valid[..., 1:] & valid[..., :-1]
    with np.errstate(invalid="ignore", over="ignore"):
        pieces = 0.5 * (magnitude[..., 1:] + magnitude[..., :-1]) * np.diff(t, axis=-1)
    # a sequential sum: the trailing zeros of the padding leave it unchanged,
    # which the pairwise sum of np.sum does not guarantee for another length
    area = np.cumsum(np.where(segment, pieces, 0.0), axis=-1)[..., -1:]
    finite = np.where(valid, t, np.nan)
    span = np.fmax.reduce(finite, axis=-1, keepdims=True) - np.fmin.reduce(
        finite, axis=-1, keepdims=True
    )
    last = _last(magnitude, valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(span > 0, area / np.where(span > 0, span, 1.0), last)
    return _join(mean, units)


def _last(magnitude: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Get the value at the last time point of every simulation, `NaN` without one."""
    index = valid.shape[-1] - 1 - np.argmax(valid[..., ::-1], axis=-1)
    value = np.take_along_axis(magnitude, index[..., None], axis=-1)
    return np.where(valid.any(axis=-1, keepdims=True), value, np.nan)


def _at(
    x: Any,
    when: Any,
    time: np.ndarray | None,
    tolerance: float | np.ndarray,
    formula: str,
    changes: Sequence[np.ndarray] | None = None,
) -> Any:
    """Reduce to the value at a time, see `evaluate_reduced`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    if time is None:
        raise _no_time("at", formula)
    target, _ = _split(when)
    shape = magnitude.shape
    times = np.broadcast_to(np.asarray(time, dtype=float), shape).reshape(-1, shape[-1])
    rows = magnitude.reshape(-1, shape[-1])
    targets = np.broadcast_to(target, (*shape[:-1], 1)).reshape(-1)
    tolerances = np.broadcast_to(
        np.asarray(tolerance, dtype=float)[..., None], (*shape[:-1], 1)
    ).reshape(-1)
    if changes is not None and len(changes) != rows.shape[0]:
        raise ValueError(
            f"'{formula}': the changes of {len(changes)} simulations for the "
            f"timecourses of {rows.shape[0]}, the changes of every simulation are "
            f"given or none."
        )
    out = np.array(
        [
            apply_weights(
                grid_weights(
                    times[k],
                    targets[k : k + 1],
                    float(tolerances[k]),
                    None if changes is None else changes[k],
                ),
                rows[k],
            )[0]
            for k in range(rows.shape[0])
        ]
    )
    return _join(out.reshape(*shape[:-1], 1), units)
