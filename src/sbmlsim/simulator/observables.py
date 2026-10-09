"""The observables of a run, compiled against its models and evaluated.

`compile_observables` orders the observables of a run by what they read,
checks every symbol against the model, derives the kind (a timecourse or a
value per simulation) and the unit of every observable and gives an
`ObservableGraph`: frozen, picklable and free of pint, so the chunks of a scan
carry it into the workers. A run without observables has the graph of the
selections of its model, each a timecourse of its own name (`identity_graph`).

`ObservableGraph.evaluate` computes the observables on the native solutions
of the points of a chunk, stacked into arrays `(n_points, n_rows)` padded
with `NaN`: a timecourse is such an array, a value per simulation an array
`(n_points, 1)` while the graph is evaluated and `(n_points,)` in its result.
The observables are evaluated before any interpolation onto a grid, so the
reductions and the analysis of a PK observable are as exact as the output of
the simulation. An observable which no kept observable needs is not
evaluated, and the selections only it reads are not selected.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.result.scan import TIME
from sbmlsim.simulation.observables import (
    PK,
    Custom,
    Formula,
    Observable,
    ObservableKind,
)
from sbmlsim.simulator.formula import compile_formula, evaluate_reduced, reduce_formula
from sbmlsim.simulator.pk import PKNode, compile_pk, evaluate_pk
from sbmlsim.simulator.plan import Plan
from sbmlsim.units import Quantity, ureg

SCALAR = ObservableKind.SCALAR
TIMECOURSE = ObservableKind.TIMECOURSE


@dataclass(frozen=True)
class FormulaNode:
    """A Formula observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        formula: the formula.
        kind: a timecourse or a value per simulation.
        factor: the factor from the derived unit of the formula into its
            declared unit, `1` without a conversion; `evaluate` applies it to
            the values of the formula, so every observable which reads it sees
            the unit of the observable.
    """

    id: str
    formula: str
    kind: ObservableKind
    factor: float


@dataclass(frozen=True)
class CustomNode:
    """A Custom observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        function: the function of a module.
        kind: a timecourse or a value per simulation.
        symbols: what it reads.
    """

    id: str
    function: Callable[[np.ndarray, dict[str, Any]], Any]
    kind: ObservableKind
    symbols: tuple[str, ...]


#: an observable compiled against the models of a run
Node = FormulaNode | CustomNode | PKNode


class ObservableError(RuntimeError):
    """An observable failed on the points of a chunk.

    Attributes:
        row: the row of the point which failed in the arrays of the chunk,
            `None` for every point.
        message: the error.
    """

    def __init__(self, row: int | None, message: str) -> None:
        """Create the error, the arguments pickle."""
        super().__init__(row, message)
        self.row = row
        self.message = message

    def __str__(self) -> str:
        """Get the message."""
        return self.message


@dataclass(frozen=True)
class ObservableGraph:
    """The observables of a run, see the module.

    Attributes:
        nodes: the observables the kept ones need, in the order of what they
            read.
        selections: the selections of roadrunner the nodes read, without
            `time`.
        kinds: the kind of every symbol and output: `time` and the selections
            are timecourses.
        units: the unit of every symbol and output, for a formula the declared
            unit if it has one (the values of `evaluate` are in it), else the
            derived one.
        keep: the outputs of the result, in order.
        outputs: every output: the ids of the formulas and customs and the
            parameters of the PK observables; the selections without
            observables.
        doses: the dose targets of the PK observables the run evaluates,
            whose units the models of a run share.
    """

    nodes: tuple[Node, ...]
    selections: tuple[str, ...]
    kinds: dict[str, ObservableKind]
    units: dict[str, str]
    keep: tuple[str, ...]
    outputs: tuple[str, ...]
    doses: tuple[str, ...] = ()

    @property
    def timecourses(self) -> tuple[str, ...]:
        """Get the kept timecourses."""
        return tuple(k for k in self.keep if self.kinds[k] is TIMECOURSE)

    @property
    def scalars(self) -> tuple[str, ...]:
        """Get the kept values per simulation."""
        return tuple(k for k in self.keep if self.kinds[k] is SCALAR)

    def evaluate(
        self,
        time: np.ndarray,
        columns: Mapping[str, np.ndarray],
        plans: Sequence[Plan],
    ) -> dict[str, np.ndarray]:
        """Evaluate the observables on the points of a chunk, see the module.

        Args:
            time: the time points `(n_points, n_rows)`, padded with `NaN`.
            columns: the selections, arrays of the shape of `time`.
            plans: the plan of every point, with its values, which gives the
                doses of the PK observables.

        Returns:
            The kept outputs: a timecourse `(n_points, n_rows)`, a value per
            simulation `(n_points,)`.

        Raises:
            ObservableError: if an observable fails for a point or for all.
        """
        values: dict[str, np.ndarray] = {TIME: time, **columns}
        for node in self.nodes:
            if isinstance(node, FormulaNode):
                values[node.id] = _formula(node, time, values)
            elif isinstance(node, CustomNode):
                values[node.id] = _custom(node, time, values, self.kinds)
            else:
                try:
                    values.update(
                        evaluate_pk(node, time, values[node.selection], plans)
                    )
                except Exception as err:
                    raise ObservableError(
                        None,
                        f"The PK observable '{node.id}' failed: "
                        f"{type(err).__name__}: {err}",
                    ) from err
        out: dict[str, np.ndarray] = {}
        for key in self.keep:
            out[key] = values[key][:, 0] if self.kinds[key] is SCALAR else values[key]
        return out


def _formula(
    node: FormulaNode, time: np.ndarray, values: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Evaluate a formula; a floating point error gives `inf` or `NaN`.

    The padding of a timecourse is `NaN`, the steady state (`inf`) is kept.

    Raises:
        ObservableError: if the formula fails, for every point.
    """
    try:
        with np.errstate(all="ignore"):
            value = np.asarray(
                evaluate_reduced(node.formula, values, time), dtype=float
            )
            shape = (time.shape[0], 1) if node.kind is SCALAR else time.shape
            value = np.broadcast_to(value, shape).copy()
            if node.factor != 1.0:
                value *= node.factor
            if node.kind is TIMECOURSE:
                value[np.isnan(time)] = np.nan
    except Exception as err:
        raise ObservableError(
            None,
            f"The formula of the observable '{node.id}' failed: "
            f"{type(err).__name__}: {err}",
        ) from err
    return value


def _custom(
    node: CustomNode,
    time: np.ndarray,
    values: Mapping[str, np.ndarray],
    kinds: Mapping[str, ObservableKind],
) -> np.ndarray:
    """Call the function of a custom observable once per point.

    Raises:
        ObservableError: if the function fails for a point, with its row.
    """
    scalar = node.kind is SCALAR
    out = np.full((time.shape[0], 1) if scalar else time.shape, np.nan)
    for row in range(time.shape[0]):
        valid = np.isfinite(time[row])
        try:
            arguments = {
                symbol: float(values[symbol][row, 0])
                if kinds[symbol] is SCALAR
                else values[symbol][row, valid]
                for symbol in node.symbols
            }
            result = node.function(time[row, valid], arguments)
            if scalar:
                out[row, 0] = float(result)
            else:
                array = np.asarray(result, dtype=float)
                if array.shape != (int(valid.sum()),):
                    raise ValueError(
                        f"the function returned the shape {array.shape} for "
                        f"{int(valid.sum())} time points"
                    )
                out[row, valid] = array
        except Exception as err:
            raise ObservableError(
                row,
                f"The function of the observable '{node.id}' failed: "
                f"{type(err).__name__}: {err}",
            ) from err
    return out


def identity_graph(
    selections: Sequence[str],
    units: Mapping[str, str] | None = None,
    keep: Sequence[str] | None = None,
) -> ObservableGraph:
    """Get the graph of a run without observables.

    Every selection is a timecourse of its own name.

    Args:
        selections: the selections, without `time`.
        units: the unit of the time and of every selection, `""` by default.
        keep: the selections of the result, every one by default.

    Returns:
        The graph.

    Raises:
        ValueError: if `keep` names no selection.
    """
    names = tuple(dict.fromkeys(s for s in selections if s != TIME))
    units = dict(units or {})
    return ObservableGraph(
        nodes=(),
        selections=names,
        kinds={TIME: TIMECOURSE, **dict.fromkeys(names, TIMECOURSE)},
        units={TIME: units.get(TIME, ""), **{s: units.get(s, "") for s in names}},
        keep=_keep(keep, names, {}),
        outputs=names,
    )


def compile_observables(
    observables: Sequence[Observable] | None,
    model: RoadrunnerSBMLModel,
    *,
    keep: Sequence[str] | None = None,
    plans: Sequence[Plan] = (),
) -> ObservableGraph:
    """Compile the observables of a run against its first model, see the module.

    Args:
        observables: the observables; `None` or none for the selections of
            the model, see `identity_graph`.
        model: the loaded model, whose selections and units they read.
        keep: the outputs of the result, every output by default; the id of a
            PK observable keeps all its parameters.
        plans: the plans of the run, each with the values of a point, whose
            dosing decides the parameters of the PK observables.

    Returns:
        The graph.

    Raises:
        TypeError: if the observables are no sequence of observables or `keep`
            is a string.
        ValueError: for two observables of one id; an id which is a selection
            of the model; a symbol which is neither an observable nor a
            selection; a cycle; a time of `at` which is a timecourse; a unit
            which cannot be derived without `unit=` or not be converted into
            it; a time of `at` which has not the time unit of the model; a
            formula which mixes units of one dimension at different scales; a
            PK observable which does not fit the model (see
            `compile_pk`); or a `keep` which names no output.

    A declared unit is the unit of the observable: the values are converted
    when the formula is evaluated and the observables which read it see them in
    that unit. A formula which adds units of one dimension at different scales
    is refused, but a comparison or a piecewise, which pint cannot evaluate, is
    not checked for mixed scales.
    """
    uinfo = model.uinfo
    time_unit = uinfo.get(TIME, "") or ""
    if not observables:
        selections = [s for s in model.selections or [] if s != TIME]
        units = {s: uinfo.get(s, "") or "" for s in selections}
        return identity_graph(selections, {TIME: time_unit, **units}, keep)
    if isinstance(observables, Observable):
        raise TypeError("The observables of a run are a sequence of observables.")
    definitions: dict[str, Observable] = {}
    for observable in observables:
        if not isinstance(observable, Observable):
            raise TypeError(f"{observable!r} is no observable (Formula, PK, Custom).")
        if observable.id in definitions:
            raise ValueError(f"Two observables have the id '{observable.id}'.")
        definitions[observable.id] = observable
    clash = sorted(name for name in definitions if model.has_selection(name))
    if clash:
        raise ValueError(
            f"The observable ids {clash} are selections of the model: an observable "
            f"and a selection share no name, choose other ids."
        )
    kinds: dict[str, ObservableKind] = {TIME: TIMECOURSE}
    units: dict[str, str] = {TIME: time_unit}
    selections: dict[str, None] = {}
    compiled: dict[str, Node] = {}
    outputs: dict[str, tuple[str, ...]] = {}
    for name in _order(definitions):
        observable = definitions[name]
        for symbol in observable.reads:
            _resolve(
                symbol, name, model, definitions, outputs, kinds, units, selections
            )
        if isinstance(observable, Formula):
            node, unit = _compile_formula(observable, kinds, units)
            kinds[name], units[name] = node.kind, unit
            outputs[name] = (name,)
        elif isinstance(observable, Custom):
            node = CustomNode(
                id=name,
                function=observable.function,
                kind=observable.kind,
                symbols=tuple(observable.symbols),
            )
            kinds[name], units[name] = observable.kind, str(ureg.Unit(observable.unit))
            outputs[name] = (name,)
        elif isinstance(observable, PK):
            node, pk_units = _compile_pk(observable, model, kinds, units, plans)
            for output, unit in pk_units.items():
                kinds[output], units[output] = SCALAR, unit
            outputs[name] = tuple(pk_units)
        else:
            raise TypeError(f"{observable!r} is no observable (Formula, PK, Custom).")
        compiled[name] = node
    every = tuple(o for name in definitions for o in outputs[name])
    kept = _keep(keep, every, outputs)
    needed = _needed(kept, definitions)
    nodes = tuple(node for name, node in compiled.items() if name in needed)
    read = {s for name in needed for s in definitions[name].reads}
    doses = tuple(
        node.dose.target
        for node in nodes
        if isinstance(node, PKNode) and node.dose is not None and node.dose.target
    )
    return ObservableGraph(
        nodes=nodes,
        selections=tuple(s for s in selections if s in read),
        kinds=kinds,
        units=units,
        keep=kept,
        outputs=every,
        doses=tuple(dict.fromkeys(doses)),
    )


def _owner(symbol: str, definitions: Mapping[str, Observable]) -> str | None:
    """Get the observable a symbol is an output of, `None` for a selection."""
    if symbol in definitions:
        return symbol
    head, dot, _ = symbol.partition(".")
    if dot and isinstance(definitions.get(head), PK):
        return head
    return None


def _order(definitions: Mapping[str, Observable]) -> list[str]:
    """Order the observables so that every one comes after what it reads.

    Raises:
        ValueError: if the observables read each other in a cycle.
    """
    order: list[str] = []
    state: dict[str, bool] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) is True:
            return
        if state.get(name) is False:
            cycle = (*path[path.index(name) :], name)
            raise ValueError(
                f"The observables read each other in a cycle: {' -> '.join(cycle)}."
            )
        state[name] = False
        for symbol in definitions[name].reads:
            owner = _owner(symbol, definitions)
            if owner is not None:
                visit(owner, (*path, name))
        state[name] = True
        order.append(name)

    for name in definitions:
        visit(name, ())
    return order


def _resolve(
    symbol: str,
    name: str,
    model: RoadrunnerSBMLModel,
    definitions: Mapping[str, Observable],
    outputs: Mapping[str, tuple[str, ...]],
    kinds: dict[str, ObservableKind],
    units: dict[str, str],
    selections: dict[str, None],
) -> None:
    """Resolve a symbol an observable reads: an output or a selection.

    Raises:
        ValueError: if it is a parameter a PK observable does not give, or
            neither an observable nor a selection of the model.
    """
    if symbol in kinds:
        return
    owner = _owner(symbol, definitions)
    if owner is not None:
        raise ValueError(
            f"The observable '{name}' reads '{symbol}', which the PK observable "
            f"'{owner}' does not give: {list(outputs.get(owner, ()))}."
        )
    if not model.has_selection(symbol):
        raise ValueError(
            f"The observable '{name}' reads '{symbol}', which is neither an "
            f"observable nor a selection of the model."
        )
    kinds[symbol] = TIMECOURSE
    units[symbol] = model.uinfo.get(symbol, "") or ""
    selections[symbol] = None


def _derive_unit(formula: str, units: Mapping[str, str]) -> str | None:
    """Derive the unit of a formula, `None` where pint cannot.

    The formula is applied to quantities of one in the units of its symbols;
    a symbol without a unit is dimensionless. A warning of pint, e.g. a unit
    which numpy strips, means that no unit can be derived.
    """
    reduced = reduce_formula(formula)
    try:
        values = {
            symbol: ureg.Quantity(1.0, units[symbol] or "dimensionless")
            for symbol in reduced.symbols
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="ignore"):
                value = evaluate_reduced(formula, values)
    except Exception:
        return None
    if isinstance(value, Quantity):
        return str(value.units)
    try:
        float(value)
    except (TypeError, ValueError):
        return None
    return "dimensionless"


def _symbol_quantities(symbols: Sequence[str], units: Mapping[str, str]) -> dict:
    """Get the quantities of one in the units of the symbols."""
    return {
        symbol: ureg.Quantity(1.0, units[symbol] or "dimensionless")
        for symbol in symbols
    }


def _mixes_scales(formula: str, units: Mapping[str, str]) -> bool:
    """Check whether a formula adds units of one dimension at different scales.

    The formula is evaluated on quantities of one in the units of its
    symbols and on plain ones; pint converts the unit of a sum, so a different
    magnitude means that the scales differ, e.g. `ng/ml - mg/l`. A formula
    which pint cannot evaluate, e.g. a comparison or a piecewise, is not
    checked.
    """
    symbols = reduce_formula(formula).symbols
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="ignore"):
                quantity = evaluate_reduced(formula, _symbol_quantities(symbols, units))
                plain = evaluate_reduced(formula, dict.fromkeys(symbols, 1.0))
        magnitude = quantity.magnitude if isinstance(quantity, Quantity) else quantity
        return not np.allclose(magnitude, plain, rtol=1e-9, atol=1e-12)
    except Exception:
        return False


def _time_units(formula: str, units: Mapping[str, str]) -> list[Any]:
    """Derive the values of the times of the `at` reductions of a formula.

    Returns:
        The times applied to quantities of one in the units of the
        symbols, in the order of the reductions; empty where pint cannot.
    """
    reduced = reduce_formula(formula)
    scope = _symbol_quantities(reduced.symbols, units)
    times: list[Any] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="ignore"):
                for reduction in reduced.reductions:
                    x = compile_formula(reduction.arguments[0])
                    scope[reduction.symbol] = x.apply([scope[s] for s in x.symbols])
                    if reduction.function == "at":
                        when = compile_formula(reduction.arguments[1])
                        times.append(when.apply([scope[s] for s in when.symbols]))
    except Exception:
        return []
    return times


def _check_times(observable: Formula, units: Mapping[str, str]) -> None:
    """Check that the time of every `at` is a number or in the time unit.

    Raises:
        ValueError: if a time has another unit than the time of the model.
    """
    time_unit = units[TIME] or "dimensionless"
    for when in _time_units(observable.formula, units):
        if not isinstance(when, Quantity):
            continue
        try:
            factor = float(ureg.Quantity(1.0, when.units).to(time_unit).magnitude)
        except Exception:
            factor = None
        if factor is None or abs(factor - 1.0) > 1e-9:
            raise ValueError(
                f"The time of 'at' in the formula of the observable "
                f"'{observable.id}' has the unit '{when.units}', not the time unit "
                f"of the model '{units[TIME]}'; give a number or a time in the "
                f"time unit of the model."
            )


def _compile_formula(
    observable: Formula,
    kinds: Mapping[str, ObservableKind],
    units: Mapping[str, str],
) -> tuple[FormulaNode, str]:
    """Compile a formula: its kind, its unit and the factor into it.

    Returns:
        The node and the unit of the observable: the declared one, else the
        derived one.

    Raises:
        ValueError: if a time of `at` is a timecourse or has no time unit, the
            formula mixes scales, or the unit cannot be derived without `unit=`
            or not be converted into it.
    """
    reduced = reduce_formula(observable.formula)
    for symbol in reduced.time_symbols:
        if kinds[symbol] is TIMECOURSE:
            raise ValueError(
                f"The time of 'at' in the formula of the observable "
                f"'{observable.id}' reads the timecourse '{symbol}'; it is a number "
                f"or a value per simulation."
            )
    _check_times(observable, units)
    if _mixes_scales(observable.formula, units):
        raise ValueError(
            f"The formula '{observable.formula}' of the observable '{observable.id}' "
            f"mixes units of different scale (e.g. ng/ml and mg/l); the values are "
            f"in the units of their symbols: declare one of them as a Formula with "
            f"unit=..., a factor in the formula does not convert it."
        )
    kind = (
        SCALAR if all(kinds[s] is SCALAR for s in reduced.outer_symbols) else TIMECOURSE
    )
    derived = _derive_unit(observable.formula, units)
    declared = None if observable.unit is None else str(ureg.Unit(observable.unit))
    if declared is None:
        if derived is None:
            raise ValueError(
                f"The unit of the formula '{observable.formula}' of the observable "
                f"'{observable.id}' cannot be derived, e.g. of a comparison or of "
                f"piecewise; give it as unit=."
            )
        node = FormulaNode(observable.id, observable.formula, kind, 1.0)
        return node, derived
    unitless = all(not units[s] for s in reduced.symbols)
    if derived is None or (derived == "dimensionless" and unitless):
        node = FormulaNode(observable.id, observable.formula, kind, 1.0)
        return node, declared
    try:
        factor = float(ureg.Quantity(1.0, derived).to(declared).magnitude)
    except Exception as err:
        raise ValueError(
            f"The formula of the observable '{observable.id}' has the unit "
            f"'{derived}', which cannot be converted into its unit "
            f"'{observable.unit}'."
        ) from err
    node = FormulaNode(observable.id, observable.formula, kind, factor)
    return node, declared


def _compile_pk(
    observable: PK,
    model: RoadrunnerSBMLModel,
    kinds: Mapping[str, ObservableKind],
    units: Mapping[str, str],
    plans: Sequence[Plan],
) -> tuple[PKNode, dict[str, str]]:
    """Compile a PK observable against the model, see `compile_pk`.

    Raises:
        ValueError: if its timecourse is a value per simulation, the dose is no
            target of the model, or `compile_pk` raises.
    """
    if kinds[observable.selection] is not TIMECOURSE:
        raise ValueError(
            f"The PK observable '{observable.id}' analyses '{observable.selection}', "
            f"which is a value per simulation, not a timecourse."
        )
    dose_unit = None
    if isinstance(observable.dose, str):
        if not model.has_selection(observable.dose):
            raise ValueError(
                f"The dose '{observable.dose}' of the observable '{observable.id}' is "
                f"no target of the model."
            )
        dose_unit = model.uinfo.get(observable.dose, "") or ""
    return compile_pk(
        observable,
        unit=units[observable.selection],
        time_unit=units[TIME],
        dose_unit=dose_unit,
        plans=plans,
    )


def _keep(
    keep: Sequence[str] | None,
    outputs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Get the kept outputs; a group, the id of a PK observable, keeps all of its.

    Raises:
        TypeError: if `keep` is a string.
        ValueError: if `keep` names no output or keeps nothing.
    """
    if keep is None:
        return tuple(outputs)
    if isinstance(keep, str):
        raise TypeError(f"'keep' is a sequence of ids, not the string {keep!r}.")
    kept: dict[str, None] = {}
    for key in keep:
        names = groups.get(key) or ([key] if key in outputs else None)
        if names is None:
            raise ValueError(
                f"'keep' names '{key}', which is no observable of the run: "
                f"{list(outputs)}."
            )
        kept.update(dict.fromkeys(names))
    if not kept:
        raise ValueError("'keep' keeps nothing; name at least one observable.")
    return tuple(kept)


def _needed(kept: Sequence[str], definitions: Mapping[str, Observable]) -> set[str]:
    """Get the observables the kept outputs need, the kept ones among them."""
    needed: set[str] = set()
    stack = [owner for k in kept if (owner := _owner(k, definitions)) is not None]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        stack.extend(
            owner
            for symbol in definitions[name].reads
            if (owner := _owner(symbol, definitions)) is not None
        )
    return needed
