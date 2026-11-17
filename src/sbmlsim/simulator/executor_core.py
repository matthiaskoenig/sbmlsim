"""The executor of the engine core runs a plan on a model of the engine core.

A plan is one request of the contract `simulation/1` (`request_of`) which one
call of the engine simulates: the engine initializes the model with the values
of the request, reaches the steady state of a presimulation, integrates in
absolute time from the start in the segments between the changes, fires the
events of the model and answers with the table of the selections, see
`sbmlsim.model.model_core` and the documentation of sbmlsim-core. The executor
therefore holds no loop over the segments and no semantics of its own, unlike
the executors of COPASI and AMICI; the table becomes a `TimecourseResult` with
`output_columns` and `output_time` of `sbmlsim.simulator.executor`, which
every executor shares, so the column `time`, a later column `time` and a model
with an entity named `time` are the ones of roadrunner.

The engine has the semantics of the executor of roadrunner, see
`sbmlsim.simulator.executor`: the values before the initialization, which the
initial assignments see, the pre-equilibration of a steady state from the
time 0 whatever the start, the changes of PEtab v2 (the values of a change set
at once, a compartment which keeps the concentrations of its species, the
events a change triggers fired with it) and the output of the steps of the
integrator or of the given times, with two rows at the time of a change after
the start, the state before the change and the state after it.

A simulation which fails raises the error of its cause and not one of the
bindings (`_failed`): a simulation the engine does not support is an
`UnsupportedModelError`, which a run raises for the whole scan, a
presimulation which reaches no steady state a `SteadyStateError`, and every
other failure of the engine a `RuntimeError` with its message, so a point of a
scan is flagged and an evaluation of a fit is infinite as with the other
engines. A scan does not run its points through `execute` but hands the
requests of a chunk to the engine as one batch, which answers with a
`PointError` instead of raising (`point_error`, the same mapping of the
errors), see `sbmlsim.simulator.worker._run_chunk_core`.

The known differences to the executor of roadrunner, the first five of which
the engine documents:

1. A change which sets a state exactly to the threshold of a strict
   comparison of a trigger fires its event right after the change, as the
   executors of COPASI and AMICI do; the CVODE of roadrunner ignores a root
   which is 0 at the start of a segment.
2. A change may set the variable of an assignment rule which has no math: the
   engine ignores such a rule, as roadrunner does. A plan never carries such
   a change, since `ModelSymbols.kind` refuses the variable of an assignment
   rule as a target, so only a request built by hand reaches it.
3. A change of an entity named `time` sets that entity, where the executor of
   roadrunner sets the clock of roadrunner.
4. Events which accumulate end the simulation with an error of the
   integration (`CV_TOO_MUCH_WORK`), e.g. the bouncing ball at its
   accumulation point, which roadrunner falls through and goes on from: the
   executions of the events within the relative tolerance of the last root
   share the budget of `maximum_num_steps` steps towards one output time.
5. The roots of a condition which chatters share that budget as well, so a
   model which roadrunner finishes after millions of steps may end with
   `CV_TOO_MUCH_WORK` here.
6. A steady state after the end of a simulation, the output at the time
   `inf` of `Plan.steady_state_output`, is no part of the contract: the
   `steady_state` of a request is the pre-equilibration before the start
   alone, so such a plan cannot be expressed as one request and
   `check_supported` refuses it rather than leaving its last row out.
7. A value of a request is a number, so a value which a plan sets by a
   formula is folded before the request is sent and refused where the
   executor cannot fold it (`check_supported`).

   A formula at a change is folded with the state of the model at the time of
   the change, which the executor reads as the outputs of the segment which
   ends there: a request of its own, the one of the plan with its end at the
   time of the change and the symbols of the formula among its selections,
   whose last row is that state (`_state_at`). The outputs of the engine are
   the selections of roadrunner, which is what `apply_changes` of
   `sbmlsim.simulator.changes` reads of the model of roadrunner, so a formula
   of a parameter, a compartment, an amount `S` or a concentration `[S]` is
   the value of roadrunner. Two cases are refused (`_check_change`): a symbol
   which is no output of the engine, which only a plan built by hand reaches,
   since `compile_simulation` refuses a symbol the model has not, and a model
   with an event of its own, whose state the engine answers after the events
   of a time have fired where roadrunner reads it before, so a formula of
   such a model would disagree silently.

   The folded value is the state of a request which ends at the change, where
   the integrator stops, and not of the request of the plan, which integrates
   past it: the two differ by the error of the integration, 1.8e-8 of the
   value at the default tolerances and 4.5e-12 at the tolerances the tests
   compare the engines at, measured on a change in the middle of the probe
   model of `tests.simulator.models`, which is the size of that error on that
   model and no bound of it. So the step of a relative change is the one of
   roadrunner within the integration and not exactly, which no other executor
   needs a segment of its own for.

   A segment resumes from the state of the segment before it where the
   executor can set the whole state of the model with a change
   (`_state_targets`, every model of the package): it starts at the time of
   the change before it, sets the state there and runs neither the
   pre-equilibration of the plan again nor the interval before it, so the
   fold integrates the simulation once over all its changes. A model with a
   state the executor cannot set, e.g. a species reference which is not
   constant, folds every segment from the start of the simulation, which
   costs one pass per change: measured on the HCTZ model with a dose a day
   and 200 output times, 3, 30 and 100 doses cost 8.8, 88 and 290 ms with the
   resume and 9.7, 562 and 5870 ms without it, against the 4.7, 35 and 116 ms
   of the same protocol with the doses as numbers (roadrunner: 5.2, 40 and
   133 ms). So a relative protocol costs about 2.5 times its numeric one
   whatever the number of doses, and the fold of a plan with a single such
   change asks for the symbols of its formula alone, since no later segment
   resumes from it.

   The segments of the points of a scan are simulated where the requests are
   built, one after the other in the python process of the run and not on the
   threads of a batch, so a scan of 20 points of three doses of HCTZ costs
   171 ms against the 84 ms of the doses as numbers and the 92 ms of
   roadrunner; folding the k-th change of every point of a chunk as one batch
   would run them on the threads again and is not done.

   A formula before the initialization is folded where the executor
   evaluates it without the model: a formula of a parameter whose symbols
   are the values of the model as loaded or the numbers the same
   pre-initialization sets, see `_preinit_value`. The model of the engine is
   stateless, so the executor cannot set the values of a pre-initialization
   on it and read what a formula reads, as the executor of roadrunner does,
   and no request answers them either, since they are set before the
   initialization of the simulation the request is. Only a formula of a
   parameter is folded: the size of a compartment and the amount or the
   concentration of a species are set with the bookkeeping of the volumes
   and the amounts of the initialization, which the engine does as every
   other executor does since sbmlsim-core 0.7.0, so a formula of a
   compartment and of a species can be folded as well and is not yet.
8. The engine integrates with the absolute tolerance of every state of
   roadrunner (`CoreSBMLModel.request_settings`), so it needs no factor of
   one tolerance as COPASI and AMICI do, but its own build of CVODE chooses
   its own steps and its own evaluation of the model as loaded can differ
   from roadrunner's by a unit in the last place, e.g. the volume of a
   compartment of an assignment rule of many terms, which scales the
   tolerance of a concentration species in it (two of the 28 states of the
   HCTZ model): a result is not roadrunner's bitwise and the error of the
   engine is not bounded by roadrunner's. Measured as the largest error of
   a variable relative to its largest value, against roadrunner at the
   relative tolerance 1e-12 and the absolute 1e-16, on the scan of three
   doses of the HCTZ model of `tests.simulator.test_simulator_engines`: at
   the defaults of `Simulator` the error of the engine is 1.0003 of
   roadrunner's on that model and 0.98 and 1.03 of it on the two other
   packaged models which have an error at all (the error of roadrunner is
   exactly 0 on the demo and on the midazolam model, where the ratio is
   undefined and not a percentage), and at the relative tolerance 1e-8 and
   the absolute 1e-10 of the same scan it is 1.237 of it. `ACCURACY_MARGIN`
   of that module is therefore no bound of the agreement of the two
   integrators but a check of the class of the error: a core which does not
   receive the tolerance vector of roadrunner is wrong by orders of
   magnitude, while the step selection of two builds of CVODE moves the
   error by a few percent at equal tolerances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

from sbmlsim.engine import UnsupportedModelError, import_core
from sbmlsim.model.symbols import EventSymbols, ModelSymbols, TargetKind
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.executor import SteadyStateError, output_columns, output_time
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.simulator.plan import (
    Assignment,
    OutputMode,
    Plan,
    PlanEvent,
    SteadyStatePlan,
    preinit_targets,
    time_tolerance,
)

if TYPE_CHECKING:
    from sbmlsim.model.model_core import CoreSBMLModel

#: the contract of a request of the engine
CONTRACT: Final[str] = "simulation/1"

#: the output of the steps of the integrator, a variant of the output of the
#: contract without fields, which is its name and no object
INTEGRATOR_STEPS: Final[str] = "integrator_steps"

#: the output of the given times, the variant which carries them
TIMES: Final[str] = "times"

#: the clock of a simulation, the symbol of the time in a formula and the
#: first column of every output of the engine
TIME: Final[str] = "time"

#: the flag of an integration which did not reach a steady state, the
#: `STEADY_STATE_FLAG` of the engine, which is no flag of CVODE: the engine
#: reports a presimulation which runs out of time as an error of its
#: integration, where the other executors raise a `SteadyStateError`
STEADY_STATE_FLAG: Final[int] = -1004

#: a simulation the engine does not support, the `kind` of a `PointError` of
#: a batch and the `UnsupportedError` of a simulation
UNSUPPORTED: Final[str] = "unsupported"

#: a failure of the engine itself, the `kind` of a `PointError` and the
#: `InternalError` of a simulation
INTERNAL: Final[str] = "internal"

#: an integration which failed, the `kind` of a `PointError` and the
#: `IntegrationError` of a simulation, whose `flag` is the one of CVODE
INTEGRATION: Final[str] = "integration"

#: a request the engine refuses, the `kind` of a `PointError` and the
#: `RequestError` of a simulation
REQUEST: Final[str] = "request"


def execute(
    plan: Plan, model: CoreSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a model of the engine core.

    The contract of `sbmlsim.simulator.executor.execute`: the output has two
    rows at the time of a change after the start, the state before the change
    and the state after it, and the steps of the integrator have two at an
    event of the model, see the module.

    Args:
        plan: the plan, compiled against the model.
        model: the model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.

    Returns:
        The result, with the time shifted by `plan.time_shift`.

    Raises:
        UnsupportedModelError: for a plan or a simulation the engine does not
            support, see `check_supported`, which `request_of` calls.
        SteadyStateError: if a presimulation does not reach a steady state.
        RuntimeError: if the engine fails to simulate the request, with its
            message, e.g. an integration which fails, see `_failed`.
    """
    core = import_core()
    columns = output_columns(selections)
    request = request_of(plan, model, columns)
    # the engine is stateless, the request carries every value of the
    # simulation: the model only records what the request sends and that it
    # was simulated, see `CoreSBMLModel.initialize`
    model.initialize(preinit_targets(plan))
    try:
        output = model.loaded.instance().simulate(request)
    except core.SimulationError as err:
        raise _failed(err, model, core) from err
    finally:
        model.simulated()
    # the time is shifted in the array the bindings own, which `output.values`
    # gives out again by every access: the output is read once here and
    # dropped with this call, and a caller which keeps several outputs, e.g.
    # of a batch of points, reads the values of each of them once as well
    values = output_time(output.values, columns, plan, model.symbols)
    return TimecourseResult(columns=tuple(columns), values=values)


def request_of(
    plan: Plan,
    model: CoreSBMLModel,
    selections: Sequence[str],
    settings: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Get the request of the contract `simulation/1` of a plan.

    The request carries the whole simulation, since the engine holds no state
    of its own: the interval, the values before the initialization, the steady
    state before the start, the changes at their absolute times, the output
    and the settings of the integrator. It has no seed, so the engine orders
    simultaneous events of equal priority with the default of the contract; a
    plan has none.

    A value of the request is a number, so every formula of the plan is folded
    here and a plan which one request cannot carry is refused
    (`check_supported`, which every request goes through): a request of the
    engine is always the one of a plan it simulates as roadrunner.

    A formula at a change reads the state of the model at the time of the
    change, so building the request simulates: the executor asks the engine
    for the symbols the formula reads over the segment which ends at the
    change, one request per such change, each resuming from the state of the
    segment before it, see `_changes` and `_state_at`. A plan whose changes
    are numbers and formulas of the time, which every plan of
    `compile_simulation` without a formula of an entity is, simulates nothing
    here.

    Args:
        plan: the plan, compiled against the model.
        model: the model, which gives the settings of the integrator, the
            values a formula before the initialization reads and the state a
            formula at a change reads.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.
        settings: the settings of the request, `model.request_settings()` by
            default, which computes the absolute tolerance of every state once
            per call: a caller which builds the requests of the points of a
            scan computes them once and passes them here.

    Returns:
        The request, a dictionary of the fields of the contract.

    Raises:
        UnsupportedModelError: for a plan the engine does not simulate as
            roadrunner, see `check_supported`.
        SteadyStateError: if the presimulation of a segment which a formula at
            a change is folded with reaches no steady state.
        RuntimeError: if the engine fails to simulate such a segment, see
            `_failed`.
    """
    check_supported(plan, model)
    output: Any = (
        INTEGRATOR_STEPS
        if plan.output is OutputMode.INTEGRATOR
        else {TIMES: list(plan.times)}
    )
    plain = _plain_values(preinit_targets(plan))
    request: dict[str, Any] = {
        "contract": CONTRACT,
        "start": plan.start,
        "end": plan.end,
        "output": output,
        "preinit": _preinit_values(plan.preinit, plain, model),
        "changes": [],
        "steady_state": _steady_state(plan.steady_state, plain, model),
        "selections": output_columns(selections),
        "settings": model.request_settings() if settings is None else settings,
    }
    # the changes last: a formula of a change is folded with the state at its
    # time, which the executor reads with a request of its own, see `_changes`
    request["changes"] = _changes(plan, model, request)
    return request


def check_supported(plan: Plan, model: CoreSBMLModel) -> None:
    """Refuse a plan the engine does not simulate as roadrunner.

    `request_of` checks every plan, and `Simulator.run` the plan of the first
    point of every simulation and model before it runs a point, see
    `sbmlsim.simulator.engines.check_plan`, so a plan the engine refuses fails
    the whole scan whatever `on_error` says, which is what a plan one request
    cannot carry deserves: it fails every point of the scan alike.

    The steady state after the end of a simulation is no field of the
    contract and a value of a request is a number, so a formula the executor
    cannot fold is refused, see the module and `_preinit_value`. The check is
    the one of the plan alone and simulates nothing: a formula at a change is
    folded with the state of the model at its time, which `request_of` reads
    with a request of the segment which ends there (`_state_at`), and only the
    symbols of such a formula are checked here, against the outputs of the
    engine (`_check_change`).

    Args:
        plan: the plan, compiled against the model, with the values of a
            point.
        model: the model, which gives the values a formula before the
            initialization reads and the outputs a formula at a change may
            read.

    Raises:
        UnsupportedModelError: a `NotImplementedError`, for a plan with a
            steady state after the end of its simulation and for a formula
            the executor cannot fold into a number.
    """
    if plan.steady_state_output is not None:
        raise UnsupportedModelError(
            f"The engine 'core' does not simulate the steady state after the end "
            f"of the simulation of the model '{model.label}', the output at the "
            f"time 'inf': the steady state of a request is the pre-equilibration "
            f"before the start of the simulation, so one request cannot continue "
            f"from the state the timecourse ended in."
        )
    # the values before the initialization the request carries, folded here
    # and discarded: a formula the executor cannot fold raises, see
    # `_preinit_values`
    plain = _plain_values(preinit_targets(plan))
    _preinit_values(plan.preinit, plain, model)
    if plan.steady_state is not None:
        _preinit_values(plan.steady_state.preinit, plain, model)
    for event in plan.events:
        _check_change(event, model, plan.symbols.events)


def point_error(point: Any, model: CoreSBMLModel) -> Exception:
    """Get the error of a failed point of a batch as the error of an executor.

    A batch of the engine answers with a `PointError` instead of raising, so
    that one point never ends the batch, see
    `sbmlsim.simulator.worker._run_chunk_core`: its `kind` is the error a
    single simulation would raise and its `message` is the message of that
    error, so a point of a batch and a point of its own fail alike, see
    `_failed`, which this shares the mapping with.

    Args:
        point: the `PointError` of the bindings, with its `kind`, `message`
            and the `flag` of an integration which failed.
        model: the model which was simulated.

    Returns:
        The error to raise or to record as the failure of the point.
    """
    return _error_of(point.kind, point.message, point.flag, model)


def _failed(err: Exception, model: CoreSBMLModel, core: ModuleType) -> Exception:
    """Get the error of a simulation of the engine as the error of an executor.

    The error of its cause, so that a caller handles the engines alike and no
    error of the bindings reaches it, as none reaches a caller of `load_model`,
    see `sbmlsim.model.model_core`. The four errors of a simulation of the
    engine map onto the three a run tells apart:

    - a simulation the engine does not support is an `UnsupportedModelError`
      which names the feature with the message of the engine. It is a
      `NotImplementedError`, the one error a run raises instead of flagging
      the point which hit it (`sbmlsim.simulator.worker.run_chunk`), which is
      what a feature the engine does not simulate deserves: it fails every
      point of the scan alike.
    - a presimulation which found no steady state is a `SteadyStateError`, the
      integration which reports the `STEADY_STATE_FLAG`.
    - every other integration which failed is a `RuntimeError`, a point a run
      flags, as a failed integration of roadrunner is.
    - a request the engine refuses is a `RuntimeError` as well, so the point is
      flagged: a value which is not finite reaches the engine as such a
      request, and roadrunner flags the point of a scan whose dimension
      carries one (measured with `on_error="flag"`, where its integration
      fails with `CV_CONV_FAILURE`), so a user sees the same of both engines.
    - an internal error of the engine, a trap of it or of the model, is a
      `RuntimeError` which says so, so the point is flagged without hiding
      what failed.

    Args:
        err: the `SimulationError` of the engine, whose `flag` is the one of
            the integration which failed.
        model: the model which was simulated.
        core: the bindings of sbmlsim-core, whose errors are told apart.

    Returns:
        The error to raise.
    """
    kind, flag = REQUEST, None
    if isinstance(err, core.UnsupportedError):
        kind = UNSUPPORTED
    elif isinstance(err, core.InternalError):
        kind = INTERNAL
    elif isinstance(err, core.IntegrationError):
        kind, flag = INTEGRATION, err.flag
    return _error_of(kind, str(err), flag, model)


def _error_of(
    kind: str, message: str, flag: int | None, model: CoreSBMLModel
) -> Exception:
    """Get the error of a failed simulation of its kind, see `_failed`.

    Args:
        kind: what failed, `unsupported`, `internal`, `integration` or
            `request`, the kinds of a `PointError` of a batch, which are the
            errors of a single simulation.
        message: the message of the engine.
        flag: the flag of CVODE of an integration which failed, `None`
            otherwise.
        model: the model which was simulated, which the error names.

    Returns:
        The error to raise.
    """
    label = f"the model '{model.label}'"
    if kind == UNSUPPORTED:
        return UnsupportedModelError(
            f"The engine 'core' does not simulate {label} as roadrunner: {message}"
        )
    if kind == INTERNAL:
        return RuntimeError(
            f"The engine 'core' failed internally on {label}, see the bindings of "
            f"sbmlsim-core: {message}"
        )
    text = f"The engine 'core' failed to simulate {label}: {message}"
    if kind == INTEGRATION and flag == STEADY_STATE_FLAG:
        return SteadyStateError(text)
    return RuntimeError(text)


def _steady_state(
    steady_state: SteadyStatePlan | None,
    plain: Mapping[str, Assignment],
    model: CoreSBMLModel,
) -> dict[str, Any] | None:
    """Get the steady state of a request, `None` without a presimulation.

    Args:
        steady_state: the presimulation of the plan.
        plain: the pre-initialization assignments of a number, see
            `_plain_values`.
        model: the model, which gives the values a formula reads.

    Returns:
        The steady state of the contract: its values before the
        initialization, which the engine sets together with the ones of the
        request, and the tolerances and the longest time of its criterion.

    Raises:
        UnsupportedModelError: for a formula the executor cannot fold, see
            `_preinit_value`.
    """
    if steady_state is None:
        return None
    return {
        "preinit": _preinit_values(steady_state.preinit, plain, model),
        "absolute_tolerance": steady_state.absolute_tolerance,
        "relative_tolerance": steady_state.relative_tolerance,
        "max_time": steady_state.max_time,
    }


def _plain_values(assignments: Sequence[Assignment]) -> dict[str, Assignment]:
    """Get the assignments of a number by their target.

    Args:
        assignments: the pre-initialization assignments of a plan and of its
            steady state, which every executor applies as one
            pre-initialization, see `preinit_targets`.

    Returns:
        The assignments which carry a number, by their target.
    """
    return {a.target: a for a in assignments if a.value is not None}


def _preinit_values(
    assignments: Sequence[Assignment],
    plain: Mapping[str, Assignment],
    model: CoreSBMLModel,
) -> list[list[Any]]:
    """Get pre-initialization assignments as the `(id, value)` pairs of the contract.

    Args:
        assignments: the assignments of the plan or of its steady state.
        plain: the pre-initialization assignments of a number, see
            `_plain_values`.
        model: the model, which gives the values a formula reads.

    Returns:
        A pair of the target and its value per assignment, in their order.

    Raises:
        UnsupportedModelError: for a formula the executor cannot fold, see
            `_preinit_value`.
    """
    return [
        [a.target, a.value if a.value is not None else _preinit_value(a, plain, model)]
        for a in assignments
    ]


def _changes(
    plan: Plan, model: CoreSBMLModel, request: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Get the changes of a request, the values of a plan folded into numbers.

    A formula at a change is evaluated by the executor of roadrunner with the
    state of the model at the time of the change (`apply_changes` of
    `sbmlsim.simulator.changes`), which a request carries no formula of: the
    executor reads that state as the outputs of the segment which ends at the
    change, with a request of its own, and sends the number, see `_state_at`.
    The changes are folded in their order, the one of their times, so the
    request of a segment carries the changes before it, already folded, and
    resumes from the state of the segment before it.

    Args:
        plan: the plan, whose events are sorted by their time.
        model: the model, which the engine reads the state of.
        request: the request which is built, whose fields the request of a
            segment takes, its `changes` excluded.

    Returns:
        The `changes` of the request, one entry per event of the plan.

    Raises:
        SteadyStateError: if the presimulation of a segment reaches no steady
            state.
        RuntimeError: if the engine fails to simulate a segment, see
            `_failed`.
    """
    changes: list[dict[str, Any]] = []
    # the symbols of the model every change reads, by the position of its
    # event: the state of the model is read only where a formula reads it
    reads = [_change_symbols(event) for event in plan.events]
    # the targets a segment resumes from, `None` for a model whose state the
    # executor cannot hand to the engine, whose segments start at `plan.start`
    targets = _state_targets(model)
    # the time of the last segment which was simulated and the state after the
    # change at that time, which the next segment resumes from
    resume: tuple[float, list[list[Any]]] | None = None
    for index, event in enumerate(plan.events):
        symbols = reads[index]
        if not symbols:
            changes.append({"time": event.time, "values": _change_values(event, {})})
            continue
        # the state is read only where a later change resumes from it, so a
        # plan with one such change asks for the symbols of its formula alone
        resumed = targets if any(reads[index + 1 :]) else None
        selections = sorted({*symbols, *(resumed or ())})
        state = _state_at(plan, model, request, changes, resume, event.time, selections)
        values = _change_values(event, state)
        changes.append({"time": event.time, "values": values})
        if resumed is not None:
            resume = (event.time, _resume_values(resumed, model, state, values))
    return changes


def _state_targets(model: CoreSBMLModel) -> list[str] | None:
    """Get the targets a segment of the fold resumes from, or `None`.

    A segment which resumes sets the state of the model at its start with a
    change there, so it integrates its own interval instead of the one from
    the start of the simulation, see `_state_at`. The targets are every
    parameter, every compartment and the amount of every species which is not
    the variable of an assignment rule, which follows the others, so every
    species is a target of such a change and the rule that a compartment
    keeps the concentrations of the species in it (PEtab v2) keeps none of
    them, see `apply_changes` of `sbmlsim.simulator.changes`. A reaction is an
    output of the engine and no target.

    Args:
        model: the model.

    Returns:
        The targets, sorted, or `None` if the model has a state which is none
        of them, e.g. a species reference which is not constant: such a state
        would silently keep the value of the model as loaded, so the segments
        of that model start at the start of the simulation instead.
    """
    symbols = model.symbols
    entities = symbols.parameters | symbols.compartments | symbols.species
    targets = sorted(
        t for t in entities - symbols.assignment_rules if model.has_selection(t)
    )
    if not set(model.state_ids()) <= set(targets):
        return None
    return targets


def _resume_values(
    targets: Sequence[str],
    model: CoreSBMLModel,
    state: Mapping[str, float],
    values: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """Get the values the next segment resumes from, the state after a change.

    The state at the time of the change for every target the change does not
    set, and the values of the change for the ones it does: the change is
    applied by the segment which resumes, so its own targets carry its values.
    A target of the change is matched by its entity, so a change of the
    concentration `[S]` replaces the amount `S` of the state and the segment
    sets the concentration, as the change of the plan does.

    A value of a request is a number, so a target whose value is not a number
    is left out where it is no state of the model: it is then no value of the
    simulation but one of the model, which the initialization of the segment
    computes again from the pre-initialization the segment keeps (the three
    compartments of the HCTZ model whose size is `NaN`). The value of a state
    is kept whatever it is, so a simulation whose state is not finite reaches
    the engine as a request it refuses and its point is flagged, as every
    value which is not finite does, instead of being hidden here.

    Args:
        targets: the targets of the state, see `_state_targets`.
        model: the model, whose states are told from its other targets.
        state: the values at the time of the change, which carry every target.
        values: the `(id, value)` pairs of the change, already folded.

    Returns:
        The `(id, value)` pairs of the change at the start of the next
        segment.
    """
    assigned = {ModelSymbols.entity(str(target)) for target, _ in values}
    states = set(model.state_ids())
    kept = [
        [t, state[t]]
        for t in targets
        if ModelSymbols.entity(t) not in assigned
        and (t in states or isfinite(state[t]))
    ]
    return [*kept, *(list(v) for v in values)]


def _change_symbols(event: PlanEvent) -> list[str]:
    """Get the symbols of the model which the formulas of a change read, sorted.

    Args:
        event: the assignments at one time.

    Returns:
        Every symbol but `time`, which is the time of the change and no value
        of the model, as `apply_changes` reads it.
    """
    symbols: set[str] = set()
    for a in event.assignments:
        if a.value is None:
            symbols |= set(compile_formula(str(a.formula)).symbols)
    return sorted(symbols - {TIME})


def _change_values(event: PlanEvent, state: Mapping[str, float]) -> list[list[Any]]:
    """Get the values of a change as the `(id, value)` pairs of the contract.

    Every formula is evaluated with the state at the time of the change and
    `time` is that time, which is what `apply_changes` of
    `sbmlsim.simulator.changes` does on the model of roadrunner: a value of a
    change is evaluated before any of them is set, so the formulas of one
    change read the state before it.

    Args:
        event: the assignments at one time.
        state: the values of the symbols the formulas read, see `_state_at`,
            empty for a change without a formula of the model.

    Returns:
        A pair of the target and its value per assignment, in their order.
    """
    values: list[list[Any]] = []
    for a in event.assignments:
        if a.value is not None:
            values.append([a.target, a.value])
            continue
        formula = compile_formula(str(a.formula))
        values.append(
            [
                a.target,
                formula.evaluate(
                    [event.time if s == TIME else state[s] for s in formula.symbols]
                ),
            ]
        )
    return values


def _state_at(
    plan: Plan,
    model: CoreSBMLModel,
    request: Mapping[str, Any],
    changes: Sequence[dict[str, Any]],
    resume: tuple[float, list[list[Any]]] | None,
    time: float,
    selections: Sequence[str],
) -> dict[str, float]:
    """Get the values of the selections at the time of a change.

    The engine hands out no state of a model, only the outputs of a request,
    so the executor asks it for the values a formula reads over the segment
    which ends at the change, with a request of its own. The state at the
    change is the last row of its output, the one the engine integrates up to
    before it applies a change there (the first row for a change at the start,
    whose segment is empty), and its selections are the executor's own and no
    column of the result of the caller.

    A segment resumes from the state of the segment before it where the
    executor can hand the whole state of the model to the engine
    (`_state_targets`): its request starts at the time of that change, sets
    the state there with a change of its own (`_resume_values`), carries the
    changes between the two times and neither the pre-initialization nor the
    presimulation of the plan, which that state already holds. The fold then
    integrates the simulation once over all its changes instead of once from
    the start per change; the first segment, and every segment of a model
    whose state the executor cannot set, starts at the start of the plan with
    its pre-initialization and its presimulation.

    The outputs are the selections of roadrunner, `S` the amount of a
    species, `[S]` its concentration and the id of a parameter or a
    compartment, which is what `apply_changes` reads of the model of
    roadrunner, so the amounts and the concentrations are the ones of
    roadrunner. A symbol which is no output of the engine is refused before
    the request is built, see `_check_change`.

    Args:
        plan: the plan, which gives the output of the segment.
        model: the model, which the engine simulates the segment on.
        request: the request which is built, whose fields the one of the
            segment takes, its `start`, `end`, `output`, `preinit`,
            `steady_state`, `changes` and `selections` excluded.
        changes: the changes before the time, folded into numbers.
        resume: the time of the segment before this one and the state after
            its change, `None` for the first segment and for a model whose
            state the executor cannot set.
        time: the time of the change.
        selections: the values to read, the symbols the formulas of the change
            read and the state a later segment resumes from.

    Returns:
        The value of every selection at the time of the change.

    Raises:
        SteadyStateError: if the presimulation reaches no steady state.
        RuntimeError: if the engine fails to simulate the segment, see
            `_failed`, which it fails the simulation of the plan with as
            well.
    """
    core = import_core()
    columns = output_columns(selections)
    start = plan.start if resume is None else resume[0]
    segment = {
        **request,
        "start": start,
        "end": time,
        "output": _segment_output(plan, start, time),
        "selections": columns,
    }
    if resume is None:
        segment["changes"] = list(changes)
    else:
        # the change at the start of the segment sets the state, so the
        # presimulation is not run again; the pre-initialization is kept, with
        # the one of the presimulation, which every executor applies as one
        # (`preinit_targets`), so that the initialization of the segment gives
        # every value which is no state what it has in the plan
        steady = request["steady_state"]
        segment["preinit"] = [
            *request["preinit"],
            *([] if steady is None else steady["preinit"]),
        ]
        segment["steady_state"] = None
        segment["changes"] = [
            {"time": start, "values": resume[1]},
            *(c for c in changes if c["time"] > start),
        ]
    try:
        output = model.loaded.instance().simulate(segment)
    except core.SimulationError as err:
        raise _failed(err, model, core) from err
    row = output.values[-1 if time > start else 0]
    return dict(zip(columns, (float(v) for v in row), strict=True))


def _segment_output(plan: Plan, start: float, time: float) -> Any:
    """Get the output of the segment of a change, from a start to its time.

    The output of the plan over the interval of the segment: the steps of the
    integrator, or the output times of the plan inside it and its end, so that
    the integrator of the segment is given the output times of the plan and
    stops at the change. An output time within the resolution of the time of
    the change (`time_tolerance`) is the time of the change itself and is left
    out, so no two output times are a few units in the last place apart.

    Args:
        plan: the plan.
        start: the start of the segment, the start of the plan or the time of
            the change it resumes from.
        time: the time of the change, the end of the segment.

    Returns:
        The `output` of the request of the segment.
    """
    if plan.output is OutputMode.INTEGRATOR:
        return INTEGRATOR_STEPS
    tolerance = time_tolerance(plan.start, plan.end)
    times = [t for t in plan.times if start <= t < time - tolerance]
    return {TIMES: [*times, time]}


def _check_change(
    event: PlanEvent, model: CoreSBMLModel, events: Sequence[EventSymbols]
) -> None:
    """Refuse a change whose formula the executor cannot fold as roadrunner does.

    A formula is folded with the state at the time of the change, which the
    executor reads as the outputs of the engine (`_state_at`), and two cases
    are refused.

    - A symbol which is no output of the compiled model, neither a parameter
      nor a compartment nor an amount or a concentration of a species, has no
      value the executor could read. `compile_simulation` refuses such a
      formula already, so only a plan built by hand carries one.
    - A model with an event of its own: the engine answers the state of a time
      after the events of that time have fired, where the executor of
      roadrunner reads it before they fire (it evaluates the triggers and
      applies the change before `fire_events`, see
      `sbmlsim.simulator.executor._simulate`), so the two engines would
      disagree without an error. Measured on a model whose event adds 10 to a
      state at the time 5: roadrunner folds the state 0.0067 and a request of
      the engine which ends at 5 answers 10.0067. The refusal is on the
      presence of an event in the model and not on an analysis of its trigger,
      which is an arbitrary formula whose times the executor cannot tell: an
      analysis which is wrong is the silent disagreement again. A change of
      numbers is no formula and runs on a model with events as before.

    The engine does expose the state before the events of a time, as the first
    of the two rows which the steps of the integrator have at an event
    (measured: the rows at 5 of that model are 0.0393 and 10.0393), but
    nothing in the output says that two rows at the end of a segment are an
    event and not something else, and the state roadrunner folds is the one at
    one unit in the last place before the time (`segment_stop`) with the
    triggers evaluated there, so reading that row would be a guess. The fix
    belongs in the contract of the engine, which has no field for the state
    before the events of a time.

    Args:
        event: the assignments at one time.
        model: the model, whose outputs the symbols are checked against.
        events: the events of the model, whose presence refuses a formula.

    Raises:
        UnsupportedModelError: for a formula which reads a symbol the engine
            has no output of, and for a formula of a model with an event.
    """
    for a in event.assignments:
        if a.value is not None:
            continue
        formula = compile_formula(str(a.formula))
        symbols = [s for s in formula.symbols if s != TIME]
        unknown = sorted(s for s in symbols if not model.has_selection(s))
        if unknown:
            raise _no_fold(
                a,
                model,
                f"at the time {event.time}",
                f"it reads {unknown}, which the engine has no output of, so the "
                f"executor cannot read the state of the model at the time of the "
                f"change",
            )
        if symbols and events:
            raise UnsupportedModelError(
                f"The engine 'core' does not simulate the formula '{a.formula}' of "
                f"'{a.target}' at the time {event.time} of the model "
                f"'{model.label}', which has the event '{events[0].eid}': the "
                f"formula reads {sorted(symbols)}, and the state of the model at "
                f"the time of a change is the one a request of the engine answers "
                f"there, which is the state after the events of that time have "
                f"fired, while the executor of roadrunner reads it before they "
                f"fire. The two engines would disagree without an error, so the "
                f"formula is refused, see `sbmlsim.simulator.executor_core`."
            )


def _preinit_value(
    assignment: Assignment, plain: Mapping[str, Assignment], model: CoreSBMLModel
) -> float:
    """Get the value of a formula before the initialization, or refuse it.

    Every executor sets the values of the pre-initialization of a plan and of
    its steady state on the model and evaluates every formula of it against
    the model with those values, before it sets the result of any of them
    (`RoadrunnerSBMLModel.initialize`, which COPASI and AMICI follow). The
    model of the engine is stateless, so the executor cannot set the values
    and read what a formula reads: it folds the two cases whose symbols are
    known without the model and refuses the others.

    - the pre-initialization sets no number, only formulas: no value was set
      before the formulas are evaluated, so every symbol is the value of the
      model as loaded (`CoreSBMLModel.loaded_value`, which is what roadrunner
      reads then, the assignment rules of the model included).
    - every symbol of the formula is a target which the same
      pre-initialization sets to a number: the symbol is that number.

    Everything else is refused, the mixed case included: a symbol which the
    pre-initialization does not set follows the values it sets wherever an
    assignment rule or an initial assignment of the model relates them, e.g.
    the rule `kk := 3*f` of a pre-initialization which sets `f`, so the value
    of the model as loaded would be the wrong one.

    Only a formula of a parameter is folded and only a number of a parameter
    is read: the size of a compartment and the amount or the concentration of
    a species are set with the bookkeeping of the volumes and the amounts of
    the initialization, which the engine does as every other executor does,
    see the module.

    The symbol `time` is the value of an entity of that name in the model as
    loaded, which is what roadrunner reads (measured on a model with a
    parameter `time`, case 01820), and otherwise the clock, 0 before the
    initialization in both cases, since the pre-initialization does not set
    it. A plan compiled by `compile_simulation` carries the clock in no such
    formula: a formula before the initialization reads parameters only
    (PEtab v2).

    Args:
        assignment: the assignment of the formula, whose `formula` is set.
        plain: the pre-initialization assignments of a number, see
            `_plain_values`.
        model: the model, which gives the values of the symbols as loaded.

    Returns:
        The value of the formula.

    Raises:
        UnsupportedModelError: for a formula of another target, with another
            symbol or of a mixed pre-initialization, see above.
    """
    where = "before the initialization"
    if assignment.kind is not TargetKind.PARAMETER:
        raise _no_fold(
            assignment,
            model,
            where,
            f"'{assignment.target}' is a {assignment.kind}, which is set with the "
            f"bookkeeping of the volumes and the amounts of the initialization",
        )
    formula = compile_formula(str(assignment.formula))
    values: list[float] = []
    for symbol in formula.symbols:
        assigned = plain.get(symbol)
        if assigned is not None:
            if assigned.kind is not TargetKind.PARAMETER:
                raise _no_fold(
                    assignment,
                    model,
                    where,
                    f"it reads '{symbol}', which the pre-initialization sets as a "
                    f"{assigned.kind}",
                )
            values.append(float(assigned.value))  # ty: ignore[invalid-argument-type]
            continue
        try:
            loaded = model.loaded_value(symbol)
        except KeyError:
            if symbol != TIME:
                raise _no_fold(
                    assignment,
                    model,
                    where,
                    f"it reads '{symbol}', which the model has no value of before "
                    f"the initialization",
                ) from None
            # the clock of the simulation, which roadrunner reads as 0 before
            # the initialization whatever the pre-initialization sets, so it
            # is folded in both cases
            values.append(0.0)
            continue
        if plain:
            raise _no_fold(
                assignment,
                model,
                where,
                f"it reads '{symbol}', which the pre-initialization does not set "
                f"to a number, and the values it sets are not on the model when "
                f"the formula is evaluated",
            )
        values.append(loaded)
    return formula.evaluate(values)


def _no_fold(
    assignment: Assignment, model: CoreSBMLModel, where: str, reason: str
) -> UnsupportedModelError:
    """Get the error of a formula the executor cannot fold into a number.

    Args:
        assignment: the assignment of the formula.
        model: the model.
        where: where the formula is, `before the initialization` or at a time.
        reason: why the executor cannot fold it.

    Returns:
        The error to raise.
    """
    return UnsupportedModelError(
        f"The engine 'core' does not simulate the formula "
        f"'{assignment.formula}' of '{assignment.target}' {where} of the model "
        f"'{model.label}': {reason}. A value of a request of the engine is a "
        f"number, see `sbmlsim.simulator.executor_core`."
    )
