"""Formulas of the math of PEtab over the selections of roadrunner.

A formula is a string of the math of PEtab whose symbols are selections of
roadrunner: `S` is the amount of a species, `[S]` its concentration and `time`
the time. The brackets are not math, so `[S]` is replaced by an identifier
before the formula is parsed and mapped back afterwards; so is an identifier
with a dot, the parameter of a PK observable, e.g. `hctz.cmax`.

A formula is compiled to a numpy function once and cached, a `CompiledFormula`
is not pickled: a plan keeps the formula as a string.

The formulas of observables and of data extend the math by four reductions
over the time of one simulation, see `reduce_formula` and `evaluate_reduced`:
`max(x)` and `min(x)` with a single argument (with two or more they are the
elementwise functions of PEtab), `mean(x)`, the time weighted mean, and
`at(x, t)`, the value at a time. A value is an array whose last axis is the
time of a simulation, so a scan of many simulations reduces every simulation
on its own.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
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
            The `size` values of the formula.
        """
        value = np.asarray(self._function(*values), dtype=float)
        return np.broadcast_to(value, (size,)).copy()


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
    # petab.v2 imports its SciML extension and torch, which costs seconds;
    # only a simulation with a formula pays it
    from petab.v2.math import sympify_petab

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
        expression = sympify_petab(escaped)
    except Exception as err:
        raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    ordered = sorted(expression.free_symbols, key=lambda s: selection(str(s)))
    symbols = tuple(selection(str(s)) for s in ordered)
    function = sp.lambdify(ordered, expression, modules="numpy")
    return CompiledFormula(formula=formula, symbols=symbols, _function=function)


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
    formula: str, values: Mapping[str, Any], time: np.ndarray | None = None
) -> Any:
    """Evaluate a formula with reductions over the time of every simulation.

    A value is a number, a numpy array or a quantity of pint. The last axis of
    an array is the time of a simulation and the axes before it are the
    simulations, e.g. `(n_points, n_rows)` padded with `NaN` in a worker or
    `(*dims, time)` of the data of a scan. A number, or an array whose last
    axis has one element, is a value per simulation, which is constant in
    time. A reduction keeps its axis with one element, so its value broadcasts
    against the timecourses, and keeps the unit of its argument:

    - `max(x)`, `min(x)`: the largest and the smallest value, ignoring `NaN`;
    - `mean(x)`: the trapezoidal integral divided by the time between the
      first and the last time point, the value itself for a single time point;
    - `at(x, t)`: the value at the time `t`, a number or a value per
      simulation, interpolated linearly; the value after a change at its time
      and `NaN` outside of the time points.

    The times which are not finite, the padding (`NaN`) and the steady state
    after the end (`inf`), are no time points of a reduction.

    Args:
        formula: the formula.
        values: the values of its identifiers.
        time: the time points of the timecourses, of their shape; `None` where
            there are no times, e.g. for data, which allows `max` and `min`.

    Returns:
        The value of the formula.

    Raises:
        ValueError: if the formula is not valid, reads an identifier without a
            value, or reduces a timecourse with `mean` or `at` without `time`.
    """
    reduced = reduce_formula(formula)
    scope = dict(values)
    for reduction in reduced.reductions:
        x = _apply(reduction.arguments[0], scope, formula)
        if reduction.function == "at":
            when = _apply(reduction.arguments[1], scope, formula)
            scope[reduction.symbol] = _at(x, when, time, formula)
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
        f"simulation, which data has not; it is a reduction of the Formula "
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


def _at(x: Any, when: Any, time: np.ndarray | None, formula: str) -> Any:
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
    out = np.array(
        [
            apply_weights(grid_weights(times[k], targets[k : k + 1]), rows[k])[0]
            for k in range(rows.shape[0])
        ]
    )
    return _join(out.reshape(*shape[:-1], 1), units)
