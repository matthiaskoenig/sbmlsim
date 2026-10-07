"""Formulas of the changes of a simulation.

A formula is a string of the math of PEtab whose symbols are selections of
roadrunner: `S` is the amount of a species, `[S]` its concentration and `time`
the time of the change. The brackets are not math, so `[S]` is replaced by an
identifier before the formula is parsed and mapped back afterwards.

A formula is compiled to a numpy function once and cached, a `CompiledFormula`
is not pickled: a plan keeps the formula as a string.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import sympy as sp
from petab.v2.math import sympify_petab

#: a concentration `[S]` in a formula
_BRACKETS = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")

#: prefix of the identifier which stands for a concentration while parsing
_PREFIX = "sbmlsim_concentration__"


@dataclass(frozen=True)
class CompiledFormula:
    """A formula compiled to a numpy function of its symbols.

    Attributes:
        formula: the formula as it was given.
        symbols: the selections the formula reads, sorted.
    """

    formula: str
    symbols: tuple[str, ...]
    _function: Callable[..., float] = field(repr=False, compare=False)

    def evaluate(self, values: Sequence[float]) -> float:
        """Evaluate the formula.

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
        return float(self._function(*values))


def _selection(name: str) -> str:
    """Map an identifier of a parsed formula back to its selection."""
    if name.startswith(_PREFIX):
        return f"[{name.removeprefix(_PREFIX)}]"
    return name


@functools.cache
def compile_formula(formula: str) -> CompiledFormula:
    """Compile a formula, see the module.

    Args:
        formula: the formula.

    Returns:
        The compiled formula.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    escaped = _BRACKETS.sub(lambda m: f"{_PREFIX}{m.group(1)}", formula)
    try:
        expression = sympify_petab(escaped)
    except Exception as err:
        raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    ordered = sorted(expression.free_symbols, key=lambda s: _selection(str(s)))
    symbols = tuple(_selection(str(s)) for s in ordered)
    function = sp.lambdify(ordered, expression, modules="numpy")
    return CompiledFormula(formula=formula, symbols=symbols, _function=function)
