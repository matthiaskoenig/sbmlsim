"""The non-compartmental analysis of a `PK` observable, with pkpdutils.

A PK observable analyses one timecourse of every simulation of a chunk with
`pkpdutils.nca`: the timecourses are handed over as `Timecourses.from_arrays`
with the time points of every simulation (the padding with `NaN` is trimmed
away in groups of the same length), and the doses
the plan of every point assigns to the dose target. Every parameter pkpdutils
derives is a value per simulation `<id>.<parameter>`, and `<id>.flags` the
flags of the analysis (`pkpdutils.NCAFlag`).

Which parameters pkpdutils derives depends on the dosing (a multiple dosing
adds the parameters of the dosing interval, a dose the clearance and the
volume) and on the options. `compile_pk` finds them and their units by an
analysis of a synthetic timecourse with the dosing of every plan of a run,
so every point of the result has the same variables; the points of a chunk
are analysed in groups of the same number of doses and of time points, each
group trimmed to its time points, so no row is padded and a parameter does
not depend on the chunking.

pkpdutils is imported when a PK observable is compiled or evaluated, not on
the import of sbmlsim: it imports pandas, scipy and xarray, about a second.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sbmlsim.result.scan import FLAGS
from sbmlsim.simulation.observables import PK
from sbmlsim.simulator.plan import Plan

#: the time points of the synthetic timecourse which finds the parameters
_PROBE_POINTS = 97


@dataclass(frozen=True)
class DoseSpec:
    """Where the doses of a PK observable come from.

    Attributes:
        target: the target of the model whose values the plans assign, `None`
            for a fixed dose.
        amount: the fixed dose at the start, ignored for a target.
        unit: the unit of the amounts, for a target its unit in the model.
    """

    target: str | None
    amount: float
    unit: str


@dataclass(frozen=True)
class PKNode:
    """A PK observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        selection: the timecourse it analyses.
        time_unit: the unit of the time.
        unit: the unit of the timecourse.
        dose: where its doses come from, `None` without doses.
        route: the route of the doses, the name of a `pkpdutils.Route`.
        options: the `pkpdutils.NCAOptions`, `None` for the defaults.
        parameters: the parameters it gives, `flags` among them.
    """

    id: str
    selection: str
    time_unit: str
    unit: str
    dose: DoseSpec | None
    route: str | None
    options: Any
    parameters: tuple[str, ...]

    def output(self, parameter: str) -> str:
        """Get the id of the value of a parameter, `<id>.<parameter>`."""
        return f"{self.id}.{parameter}"

    @property
    def outputs(self) -> tuple[str, ...]:
        """Get the ids of the values of its parameters."""
        return tuple(self.output(parameter) for parameter in self.parameters)


def doses_of(plan: Plan, dose: DoseSpec) -> tuple[np.ndarray, np.ndarray]:
    """Get the times and the amounts of the doses of the plan of a point.

    A fixed dose is one dose at the start. The doses of a target are the
    positive values the plan assigns to it: a value before the initialization
    at the start and the values of its changes at their times, where a change
    at the start replaces the value before the initialization; the values of a
    presimulation are no doses. The times are the times of the result, i.e.
    shifted by the `time_shift` of the plan.

    Args:
        plan: the plan of a point, with its values.
        dose: where the doses come from.

    Returns:
        The times and the amounts, sorted by time.

    Raises:
        ValueError: if the plan assigns a formula to the target.
    """
    if dose.target is None:
        return (
            np.array([plan.start + plan.time_shift]),
            np.array([dose.amount], dtype=float),
        )
    found: dict[float, float] = {}
    assignments = [(plan.start, a) for a in plan.preinit] + [
        (event.time, a) for event in plan.events for a in event.assignments
    ]
    for time, assignment in assignments:
        if assignment.target != dose.target:
            continue
        if assignment.value is None:
            raise ValueError(
                f"The dose '{dose.target}' is the formula '{assignment.formula}' in a "
                f"simulation; a PK observable reads the dose as a number."
            )
        found[time] = assignment.value
    doses = sorted((time, value) for time, value in found.items() if value > 0)
    times = np.array([time for time, _ in doses], dtype=float) + plan.time_shift
    amounts = np.array([value for _, value in doses], dtype=float)
    return times, amounts


def _analyse(
    *,
    time_unit: str,
    unit: str,
    dose_unit: str | None,
    route: str | None,
    options: Any,
    time: np.ndarray,
    values: np.ndarray,
    dose_times: np.ndarray,
    dose_amounts: np.ndarray,
) -> Any:
    """Run pkpdutils on timecourses `(n, n_rows)` and doses `(n, n_doses)`."""
    import pkpdutils as pk

    dose = None
    if dose_unit is not None and dose_times.shape[-1] > 0:
        dose = {"amount": dose_amounts, "time": dose_times, "unit": dose_unit}
    timecourses = pk.Timecourses.from_arrays(
        time,
        values,
        time_unit=time_unit,
        unit=unit,
        dims=("_sim",),
        dose=dose,
        route=route if dose is not None else None,
    )
    return pk.nca(timecourses, options=options)


def _probe(
    plan: Plan, times: np.ndarray, route: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Get a timecourse of a one-compartment model dosed at the times."""
    start, end = plan.start + plan.time_shift, plan.end + plan.time_shift
    t = np.linspace(start, end, _PROBE_POINTS)
    ke = 8.0 / (end - start)
    ka = 4.0 * ke
    c = np.zeros_like(t)
    for at in times if times.size else np.array([start]):
        tau = np.clip(t - at, 0.0, None)
        if route == "iv_bolus":
            shape = np.exp(-ke * tau)
        else:
            shape = np.exp(-ke * tau) - np.exp(-ka * tau)
        c += np.where(t >= at, shape, 0.0)
    return t, c


def compile_pk(
    observable: PK,
    *,
    unit: str,
    time_unit: str,
    dose_unit: str | None,
    plans: Sequence[Plan],
) -> tuple[PKNode, dict[str, str]]:
    """Compile a PK observable, see the module.

    Args:
        observable: the definition.
        unit: the unit of its timecourse.
        time_unit: the unit of the time of the model.
        dose_unit: the unit of the dose target in the model; `None` for a fixed
            dose or without a dose.
        plans: the plans of the run, each with the values of a point, whose
            dosing decides the parameters.

    Returns:
        The compiled observable and the unit of each of its outputs.

    Raises:
        ValueError: if the timecourse or the dose target has no unit, the
            route is none of pkpdutils or an infusion, a plan doses by a
            formula, a simulation has no time span, there is no plan, or a
            parameter to keep is none pkpdutils derives.
    """
    import pkpdutils as pk

    if not unit:
        raise ValueError(
            f"The timecourse '{observable.selection}' of the observable "
            f"'{observable.id}' has no unit, which the analysis needs; give the "
            f"model units or analyse a Formula observable with a unit."
        )
    if not plans:
        raise ValueError(
            f"The observable '{observable.id}' has no simulation to analyse."
        )
    route = None
    if observable.route is not None:
        try:
            route = str(pk.Route(observable.route))
        except ValueError as err:
            raise ValueError(
                f"The route '{observable.route}' of the observable '{observable.id}' "
                f"is none of pkpdutils: {[str(r) for r in pk.Route]}."
            ) from err
        if route == str(pk.Route.IV_INFUSION):
            raise ValueError(
                f"The observable '{observable.id}' has the route of an infusion, "
                f"whose duration the changes of a simulation do not carry; analyse "
                f"it as 'iv_bolus' or 'oral'."
            )
    dose: DoseSpec | None = None
    if isinstance(observable.dose, str):
        if not dose_unit:
            raise ValueError(
                f"The dose '{observable.dose}' of the observable '{observable.id}' "
                f"has no unit in the model."
            )
        dose = DoseSpec(target=observable.dose, amount=0.0, unit=dose_unit)
    elif observable.dose is not None:
        dose = DoseSpec(
            target=None,
            amount=float(observable.dose.magnitude),
            unit=str(observable.dose.units),
        )
    units: dict[str, str] = {}
    for plan in plans:
        if plan.end <= plan.start:
            raise ValueError(
                f"The observable '{observable.id}' analyses a simulation without a "
                f"time span, [{plan.start}, {plan.end}]."
            )
        times, amounts = (
            (np.empty(0), np.empty(0)) if dose is None else doses_of(plan, dose)
        )
        t, c = _probe(plan, times, route)
        result = _analyse(
            time_unit=time_unit,
            unit=unit,
            dose_unit=None if dose is None else dose.unit,
            route=route,
            options=observable.options,
            time=t[None, :],
            values=c[None, :],
            dose_times=times[None, :],
            dose_amounts=amounts[None, :],
        )
        for parameter in (*result.parameters, FLAGS):
            units.setdefault(parameter, result.units(parameter))
    parameters = tuple(units)
    if observable.parameters is not None:
        unknown = [p for p in observable.parameters if p not in units]
        if unknown:
            raise ValueError(
                f"The observable '{observable.id}' keeps the parameters {unknown}, "
                f"which pkpdutils does not derive for its dosing: {sorted(units)}."
            )
        parameters = tuple(dict.fromkeys((*observable.parameters, FLAGS)))
    node = PKNode(
        id=observable.id,
        selection=observable.selection,
        time_unit=time_unit,
        unit=unit,
        dose=dose,
        route=route,
        options=observable.options,
        parameters=parameters,
    )
    return node, {node.output(p): units[p] for p in parameters}


def evaluate_pk(
    node: PKNode, time: np.ndarray, values: np.ndarray, plans: Sequence[Plan]
) -> dict[str, np.ndarray]:
    """Analyse the timecourses of the points of a chunk.

    The points are analysed in groups of the same number of doses and of the
    same number of time points, each group trimmed to that number, so pkpdutils
    sees no padding: it sums the rows pairwise, and padding would change the
    last bits of a parameter with the chunking.

    Args:
        node: the observable.
        time: the time points `(n_points, n_rows)`, padded with `NaN`.
        values: the timecourse, of the shape of `time`.
        plans: the plan of every point, with its values.

    Returns:
        The output id of every parameter -> its values `(n_points, 1)`, `NaN`
        where pkpdutils gives no value for a point.
    """
    finite = np.isfinite(time)
    time = np.where(finite, time, np.nan)
    values = np.where(finite, values, np.nan)
    n = time.shape[0]
    out = {output: np.full((n, 1), np.nan) for output in node.outputs}
    empty = (np.empty(0), np.empty(0))
    doses = [empty if node.dose is None else doses_of(p, node.dose) for p in plans]
    counts = np.array([times.size for times, _ in doses], dtype=int)
    points = finite.sum(axis=1)
    groups = np.unique(np.stack([counts, points], axis=1), axis=0)
    for count, n_points in groups:
        if n_points == 0:
            continue
        rows = np.flatnonzero((counts == count) & (points == n_points))
        result = _analyse(
            time_unit=node.time_unit,
            unit=node.unit,
            dose_unit=None if node.dose is None else node.dose.unit,
            route=node.route,
            options=node.options,
            time=time[rows, :n_points],
            values=values[rows, :n_points],
            dose_times=np.array([doses[r][0] for r in rows]).reshape(rows.size, count),
            dose_amounts=np.array([doses[r][1] for r in rows]).reshape(
                rows.size, count
            ),
        )
        for parameter in node.parameters:
            if parameter in result.ds:
                out[node.output(parameter)][rows, 0] = np.asarray(
                    result.ds[parameter].values, dtype=float
                )
    return out
