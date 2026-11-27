"""The curves of a group which keeps the output of its simulation, for a fit of the engine.

A simulation group whose data reads a value per simulation, a reduction or a
`PK` observable keeps the output of its simulation: the problem simulates the
plan of the experiment as it is, e.g. the steps of the integrator, and
evaluates the observables of the group on that output
(`OptimizationProblem._group_graphs`). A curve of a request of `fit/1` reads
such a value as well (sbmlsim-core 0.32.0, the floor of sbmlsim), a `reduction` of a column of its
condition (sbmlsim-core `Reduction`), so the condition of such a group is the
request of the plan the problem simulates and every training mapping of it is
one curve:

- a value per simulation of a `Formula` which is one reduction, `max(x)`,
  `min(x)`, `mean(x)` or `at(x, t)` of a time `t` which is a number, is a
  curve of that reduction of the column `x`, times the factor of the declared
  unit of the observable (`FormulaNode.factor`), which python applies after
  the reduction as well;
- a parameter `<id>.<parameter>` of a `PK` observable is a curve of the
  reduction `pk`: the batch of pkpdutils of the analysis python runs
  (`sbmlsim.simulator.pk.evaluate_pk`), whose time points and values the engine
  replaces with the ones of every simulation, with the options of the
  observable;
- a timecourse over the time is a curve of the reduction `at` at the times of
  its data, the linear interpolation with the jumps python interpolates a
  group which keeps its output with (`_interpolate_after_jumps`), the state
  before a change where the point takes it.

The column a curve reduces is a selection of the model, or a formula of the
selections the model of the fit carries as a parameter of an assignment rule
(`sbmlsim.fit.observable_sbml`), as it carries the observables of a group which
outputs the times of its data: a timecourse observable without a reduction of
the experiment, or the argument of a reduction which is no selection.

Every time of a reduction is the time of the output, the time of the plan
shifted by its `time_shift`, which the curve carries: the engine simulates on
the time of the plan.

What stays a gap of a group which keeps its output: a scan (#624), a group
with a timecourse on output times it added, a formula of a value per
simulation which is more than one reduction (`Aurine / max(Aurine)`), a time of
`at` which is no number, a `Custom` observable, the flags of a `PK`
observable, a dose of a `PK` observable which the fit moves or which pkpdutils
does not analyse in its kernel, an observable model of placeholders, a mapping
over another x than the time, and the logarithmic or estimated noise of a
likelihood, whose column is a value at a row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from sbmlsim.result.scan import TIME
from sbmlsim.simulator.formula import compile_formula, reduce_formula
from sbmlsim.simulator.observables import FormulaNode, ObservableKind
from sbmlsim.simulator.pk import PKNode
from sbmlsim.simulator.plan import time_tolerance

if TYPE_CHECKING:  # pragma: no cover
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.simulator.plan import Plan

#: the reductions of a formula which are a curve of the engine
FORMULA_REDUCTIONS = ("max", "min", "mean", "at")

#: the reductions which read the time of the output, and take its shift
_TIMED = ("mean", "at", "pk")

#: the contract of the batch of pkpdutils a curve of `pk` carries
BATCH_CONTRACT = "batch/1"


class KeptGap(ValueError):
    """A training mapping of a group which keeps its output, which no curve reads."""


@dataclass(frozen=True)
class KeptFormula:
    """A column the model of the fit carries for the curves of a group.

    Attributes:
        target: the id of the parameter of the assignment rule.
        formula: the formula of the selections.
        names: the parameters of the observables the formula reads, by their
            ids in the experiment.
    """

    target: str
    formula: str
    names: dict[str, str]


@dataclass(frozen=True)
class KeptCurve:
    """The curve of a training mapping of a group which keeps its output.

    Attributes:
        mapping: index of the fit mapping.
        column: the column of the condition the curve reduces.
        reduction: the reduction of the contract, a dictionary.
        times: the times of an interpolation (`at`), empty for a reduction to
            one value.
    """

    mapping: int
    column: str
    reduction: dict[str, Any]
    times: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class KeptGroup:
    """The columns and the curves of a group which keeps its output.

    Attributes:
        formulas: the columns the model of the fit carries, in the order of
            what they read.
        curves: the curve of every training mapping of the group.
    """

    formulas: tuple[KeptFormula, ...]
    curves: dict[int, KeptCurve]

    @property
    def columns(self) -> list[str]:
        """Get the columns the condition of the group selects, in order."""
        return list(dict.fromkeys(curve.column for curve in self.curves.values()))


def kept_group(
    problem: OptimizationProblem, group: int, training: list[int]
) -> KeptGroup:
    """Get the columns and the curves of a group which keeps its output.

    Args:
        problem: the initialized problem.
        group: the simulation group, whose `_group_graphs` is set.
        training: the training mappings of the group.

    Returns:
        The columns of the model of the fit and the curves of the mappings.

    Raises:
        KeptGap: for a group or a mapping which no curve of the engine reads,
            with the reason.
    """
    from sbmlsim.fit import fit_core

    graph = problem._group_graphs[group]
    if graph is None:
        raise ValueError(f"the group {group} outputs the times of its data")
    if problem._group_points[group] is not None:
        raise KeptGap(f"the simulation of the group {group} is a scan")
    if problem._group_timecourse_graphs[group] is not None:
        raise KeptGap(
            f"the group {group} has a timecourse on output times it adds to the "
            f"times of its simulation"
        )
    plan = problem.plans[group]
    nodes = {node.id: node for node in graph.nodes}
    formulas: dict[str, KeptFormula] = {}

    def column_of(name: str) -> str:
        """Get the column of a name, a selection or a pointwise observable."""
        node = nodes.get(name)
        if node is None:
            return name
        if not (
            isinstance(node, FormulaNode)
            and node.kind is ObservableKind.TIMECOURSE
            and not reduce_formula(node.formula).reductions
        ):
            raise KeptGap(
                f"the observable '{name}' is no formula of the values at a time, "
                f"which is a column of the model of the fit"
            )
        names = {
            symbol: column_of(symbol)
            for symbol in compile_formula(node.formula).symbols
            if symbol in nodes
        }
        target = fit_core._group_observable(group, name)
        formula = node.formula
        if node.factor != 1.0:
            formula = f"({formula}) * {node.factor!r}"
        formulas.setdefault(target, KeptFormula(target, formula, names))
        return target

    def argument_of(owner: str, argument: str) -> str:
        """Get the column of the argument of a reduction."""
        stripped = argument.strip()
        symbols = compile_formula(argument).symbols
        if symbols == (stripped,):
            return column_of(stripped)
        names = {symbol: column_of(symbol) for symbol in symbols if symbol in nodes}
        target = fit_core._group_observable(group, f"{owner}__reduced")
        formulas.setdefault(target, KeptFormula(target, argument, names))
        return target

    timed = _timed(plan)
    curves: dict[int, KeptCurve] = {}
    targets = fit_core._parameter_targets(problem)
    for k in training:
        key = problem.mapping_keys[k]
        if problem.observable_models[k] is not None:
            raise KeptGap(
                f"the mapping '{key}' reads an observable model of a group which "
                f"keeps its output"
            )
        if fit_core._is_log(problem, k) or fit_core._noise_column(problem, k):
            raise KeptGap(
                f"the noise of the mapping '{key}' is logarithmic or a column of "
                f"the simulation, which a reduction has no row of"
            )
        kind = problem.observation_kinds[k].value
        y = problem.yid_observable[k]
        if kind == "timecourse":
            if problem.xid_observable[k] != TIME:
                raise KeptGap(f"the mapping '{key}' is over another x than the time")
            curves[k] = _interpolated(problem, k, column_of(y), timed)
            continue
        if kind != "scalar":
            raise KeptGap(f"the mapping '{key}' is over a dimension of a scan")
        node = nodes.get(y)
        if isinstance(node, FormulaNode):
            size = len(problem.y_references[k])
            curves[k] = _reduced(k, key, node, timed, argument_of, size)
            continue
        pk = next(
            (n for n in graph.nodes if isinstance(n, PKNode) and y in n.outputs),
            None,
        )
        if pk is None:
            raise KeptGap(
                f"the mapping '{key}' reads '{y}', which is no formula of one "
                f"reduction and no parameter of a PK observable"
            )
        if pk.dose is not None and pk.dose.target in targets:
            raise KeptGap(
                f"the dose '{pk.dose.target}' of the PK observable '{pk.id}' is a "
                f"parameter of the fit, which the batch of a request holds at a value"
            )
        parameter = y[len(pk.id) + 1 :]
        if parameter == "flags":
            raise KeptGap(f"the mapping '{key}' reads the flags of an analysis")
        curves[k] = KeptCurve(
            mapping=k,
            column=column_of(pk.selection),
            reduction={
                "kind": "pk",
                "parameter": parameter,
                **timed,
                "changes": [float(t) for t in plan.output_change_times],
                **pk_batch(pk, plan),
            },
        )
    return KeptGroup(tuple(formulas.values()), curves)


def _timed(plan: Plan) -> dict[str, float]:
    """Get the shift of the time of a reduction, the one of the output."""
    return {"time_shift": float(plan.time_shift)}


def _reduced(
    k: int,
    key: str,
    node: FormulaNode,
    timed: dict[str, float],
    argument_of: Callable[[str, str], str],
    size: int,
) -> KeptCurve:
    """Get the curve of a value per simulation of a formula of one reduction.

    Every one of the `size` measurements is compared with the reduced value;
    those of `at` interpolate at its one time each.

    Raises:
        KeptGap: if the formula is no single reduction of a column, or a time
            of `at` is no number.
    """
    reduced = reduce_formula(node.formula)
    if len(reduced.reductions) != 1 or reduced.outer.strip() != (
        reduced.reductions[0].symbol
    ):
        raise KeptGap(
            f"the mapping '{key}' reads '{node.formula}', which is no single "
            f"reduction of a column"
        )
    reduction = reduced.reductions[0]
    function = reduction.function
    if function not in FORMULA_REDUCTIONS:  # pragma: no cover - the four of python
        raise KeptGap(f"the reduction '{function}' is none of the engine")
    column = argument_of(node.id, reduction.arguments[0])
    block: dict[str, Any] = {"kind": function, "factor": float(node.factor)}
    if function in _TIMED:
        block.update(timed)
    times: list[float] = []
    if function == "at":
        when = compile_formula(reduction.arguments[1])
        if when.symbols:
            raise KeptGap(
                f"the mapping '{key}' reads '{node.formula}', whose time reads "
                f"{list(when.symbols)} and is no number"
            )
        times = [float(when.evaluate([]))] * size
    return KeptCurve(mapping=k, column=column, reduction=block, times=times)


def _interpolated(
    problem: OptimizationProblem, k: int, column: str, timed: dict[str, float]
) -> KeptCurve:
    """Get the curve of a timecourse of a group which keeps its output.

    The times are the ones python interpolates at, the time of a change for a
    point which takes the state before it (`_x_at_data`), and the flag of a
    point which takes the state before a jump is the one of
    `_before_change`.
    """
    plan = problem.plans[k_group_of(problem, k)]
    x_at_data = problem._x_at_data[k]
    times = np.asarray(
        problem.x_references[k] if x_at_data is None else x_at_data, dtype=float
    )
    before = problem._before_change[k]
    block: dict[str, Any] = {
        "kind": "at",
        **timed,
        "tolerance": float(time_tolerance(plan.start, plan.end)),
        "changes": [float(t) for t in plan.output_change_times],
    }
    if before is not None and bool(np.any(before >= 0)):
        block["before"] = [bool(flag) for flag in before >= 0]
    return KeptCurve(
        mapping=k,
        column=column,
        reduction=block,
        times=[float(t) for t in times],
    )


def k_group_of(problem: OptimizationProblem, mapping: int) -> int:
    """Get the simulation group of a fit mapping."""
    return next(
        group
        for group, mappings in enumerate(problem.mapping_groups)
        if mapping in mappings
    )


def pk_batch(node: PKNode, plan: Plan) -> dict[str, Any]:
    """Get the batch and the options of the analysis of a PK observable.

    The batch is the one `sbmlsim.simulator.pk.evaluate_pk` hands to
    pkpdutils, written by `pkpdutils.nca.engine.batch_dict`: one sample with
    the units of the observable, the doses the plan assigns to its target and
    its route. Its time points and values are the synthetic timecourse of
    `compile_pk`, which the engine replaces with the column of every
    simulation; nothing else of the batch depends on them.

    Args:
        node: the PK observable.
        plan: the plan of the group, with the values of the start of the fit.

    Returns:
        `batch` and `options`, the dictionaries of the contract of the kernel
        of pkpdutils.

    Raises:
        KeptGap: if the kernel of pkpdutils does not analyse the batch, which
            python then analyses in its own core.
    """
    import pkpdutils as pk
    from pkpdutils.nca.engine import batch_dict, options_dict, unsupported_reason

    from sbmlsim.simulator.pk import _padded, _probe, doses_of

    empty = (np.empty(0), np.empty(0))
    times, amounts = empty if node.dose is None else doses_of(plan, node.dose)
    t, c = _probe(plan, times, node.route)
    length = _padded(t.size)
    time = np.full((1, length), np.nan)
    values = np.full((1, length), np.nan)
    time[0, : t.size] = t
    values[0, : c.size] = c
    dose = None
    if node.dose is not None and times.size:
        dose = {
            "amount": amounts[None, :],
            "time": times[None, :],
            "unit": node.dose.unit,
        }
    timecourses = pk.Timecourses.from_arrays(
        time,
        values,
        time_unit=node.time_unit,
        unit=node.unit,
        dims=("_sim",),
        dose=dose,
        route=node.route if dose is not None else None,
    )
    options = node.options if node.options is not None else pk.NCAOptions()
    reason = unsupported_reason(timecourses, options)
    if reason is not None:
        raise KeptGap(
            f"the analysis of the PK observable '{node.id}' is none of the kernel "
            f"of pkpdutils: {reason}"
        )
    batch = {name: _plain(value) for name, value in batch_dict(timecourses).items()}
    batch["contract"] = BATCH_CONTRACT
    return {"batch": batch, "options": _plain(options_dict(options))}


def _plain(value: Any) -> Any:
    """Get a value of the contract of the kernel as python objects, arrays flat."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.ravel().tolist()]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
