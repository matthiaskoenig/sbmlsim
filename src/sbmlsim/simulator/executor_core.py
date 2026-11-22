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
errors), see `sbmlsim.simulator.worker._run_chunk_core`; the folds of the
changes of those points are batched over the points as well (`requests_of`),
so an error of a fold is the error of its point and of no other.

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
   alone, so such a plan cannot be expressed as one request. The executor
   continues from the state the timecourse ended in with a second request
   whose pre-initialization carries that state and whose `steady_state` is
   the criterion of the plan (`steady_state_rows`, `steady_state_request`; a
   pre-initialization value holds over the initial assignment of its target,
   as PEtab v2 says, and the requests of the points of a chunk are simulated
   as one batch, as their folds are). Three models keep the refusal of
   `check_supported`, each named by `steady_state_gap`: a model with events
   of its own, whose triggers a request cannot be told the state of, so one
   which the carried state makes true fires again; a model which reads the
   time, since the steady state of a request is integrated from the time 0
   whatever the start of its simulation, while the executor of roadrunner
   continues at the time the timecourse ended at; and a model whose state the
   executor cannot set (`resume_targets`).
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
   the value of roadrunner. The probe of a model with events ends one unit in
   the last place before the time of the change (`probe_end`, `segment_stop`),
   where the executor of roadrunner reads the state as well, and two cases are
   refused (`_check_change`): a symbol which is no output of the engine, which
   only a plan built by hand reaches, since `compile_simulation` refuses a
   symbol the model has not, and a symbol an event of the model assigns or
   which follows such a target through an assignment rule (`_event_moved`).
   The engine fires an event of a time for an end within about 1e-9 of it, far
   wider than the unit in the last place CVODE resolves, so the probe cannot
   tell the state before the events of the time of the change from the state
   after them: a formula which reads what such an event moves would disagree
   with roadrunner silently and is refused instead. A formula which reads only
   what no event moves is folded, whatever events the model has.

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

   The folds of the points of a chunk are batched over its points: the k-th
   segment of every point of the chunk is one batch of the engine
   (`requests_of`, #550), since the segment of a point depends on the
   segments of that point alone, so a chunk pays as many rounds of the
   threads as its longest protocol has changes which read the model and no
   fold runs in the python process of the run. Measured on the HCTZ model
   with 200 output times and four threads, against the same scan with the
   doses as numbers: 20 points of 3 doses cost 120 ms against 109, where the
   folds one after the other cost 136 ms, 20 points of 10 doses 170 against
   133 (249 ms), and 100 points of 3 doses 202 against 159 (305 ms). So the
   fold of a relative protocol costs a third of what it did, near the four
   threads it now runs on, and the scan of 20 points of three doses is
   faster than roadrunner's 138 ms.

   A formula before the initialization is folded where the executor
   evaluates it without the model: a formula of a parameter whose symbols
   are the values of the model as loaded or the numbers the same
   pre-initialization sets, see `_preinit_value`. The model of the engine is
   stateless, so the executor cannot set the values of a pre-initialization
   on it and read what a formula reads, as the executor of roadrunner does,
   and no request answers them either, since they are set before the
   initialization of the simulation the request is. The target of the formula
   may be any entity: the fold gives the number and the engine applies it
   with the bookkeeping of the volumes and the amounts of its initialization,
   which it does as every other executor does since sbmlsim-core 0.7.0, so a
   formula of a compartment or of a species is folded as one of a parameter
   is (#607, which a PEtab v2 condition that estimates a concentration
   needs).
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from sbmlsim.engine import UnsupportedModelError, import_core
from sbmlsim.model.symbols import EventSymbols, ModelSymbols, TargetKind
from sbmlsim.result.timecourse import TimecourseResult, TimecourseSensitivities
from sbmlsim.simulator.executor import (
    SteadyStateError,
    integrable,
    output_columns,
    output_time,
    segment_stop,
)
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
    # the steady state after the end continues from the state the timecourse
    # ended in, which the output of the timecourse carries (`steady_state_rows`)
    run = steady_state_columns([plan], model, columns)
    request = request_of(plan, model, run)
    # the engine is stateless, the request carries every value of the
    # simulation: the model only records what the request sends and that it
    # was simulated, see `CoreSBMLModel.initialize`
    model.initialize(preinit_targets(plan))
    try:
        output = model.instance().simulate(request)
    except core.SimulationError as err:
        raise _failed(err, model, core) from err
    finally:
        model.simulated()
    # the time is shifted in the array the bindings own, which `output.values`
    # gives out again by every access: the output is read once here and
    # dropped with this call, and a caller which keeps several outputs, e.g.
    # of a batch of points, reads the values of each of them once as well
    values = output_time(output.values, run, plan, model.symbols)
    if plan.steady_state_output is not None:
        row = steady_state_rows(
            [(plan, request, values)], model, run, _one_by_one(model)
        )[0]
        if isinstance(row, Exception):
            raise row
        values = np.vstack([values, row])
    return TimecourseResult(
        columns=tuple(columns), values=steady_state_kept(values, run, columns)
    )


def execute_sensitivities(
    plan: Plan,
    model: CoreSBMLModel,
    selections: Sequence[str],
    parameters: Sequence[str],
) -> TimecourseSensitivities:
    """Run a plan with the forward sensitivities of the parameters.

    `execute` with the `sensitivities` of the request of `simulation/1`: the
    engine integrates one sensitivity vector per parameter along with the
    states (CVODES, `CV_STAGGERED`), so one run of `k` parameters answers what
    the `2 k + 1` runs of a central difference answer, and to the tolerance of
    the integration instead of the step of a difference. The states of such a
    run are the ones of the same run without it, the sensitivities being no
    part of the error test of a step.

    The engine integrates them only for a run it carries them for: a plan with
    a change, with a presimulation or of a model with events is refused by the
    engine, whose message the error names.

    Args:
        plan: the plan, compiled against the model.
        model: the model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.
        parameters: the ids the derivatives are taken in, in their order: a
            parameter of the model, a compartment, or a species as its amount
            (`S`) or its concentration (`[S]`).

    Returns:
        The result and the derivatives of its columns.

    Raises:
        ValueError: if no parameter is given, or one is named twice.
        UnsupportedModelError: for a plan, a simulation or a request of
            sensitivities the engine does not carry.
        SteadyStateError: if a presimulation does not reach a steady state.
        RuntimeError: if the engine fails to simulate the request.
    """
    ids = list(parameters)
    if not ids:
        raise ValueError("The forward sensitivities need at least one parameter.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"The parameters of the sensitivities repeat: {ids}.")
    core = import_core()
    columns = output_columns(selections)
    request = request_of(plan, model, columns)
    request["sensitivities"] = ids
    model.initialize(preinit_targets(plan))
    try:
        output = model.instance().simulate(request)
    except core.SimulationError as err:
        raise _failed(err, model, core) from err
    finally:
        model.simulated()
    # `None` for a request without them, and missing in the bindings before
    # 0.14.0, which carried the sensitivities of a run only inside Rust
    sensitivities = getattr(output, "sensitivities", None)
    if sensitivities is None:
        raise RuntimeError(
            f"The engine answered the request of the sensitivities of {ids} without "
            f"them; sbmlsim-core {core.__version__} carries them on the output of a "
            f"simulation since 0.14.0."
        )
    values = output_time(output.values, columns, plan, model.symbols)
    return TimecourseSensitivities(
        result=TimecourseResult(columns=tuple(columns), values=values),
        parameters=tuple(ids),
        # the array of the bindings, as the values of the result are, read once
        values=sensitivities.values,
    )


def execute_many(
    points: Sequence[tuple[Plan, Sequence[str]]],
    model: CoreSBMLModel,
    threads: int,
) -> list[TimecourseResult | Exception]:
    """Run the plans of a batch on a model of the engine core as one batch.

    What `execute` is for one plan: the engine simulates the requests of a
    batch on threads of its own and answers with the output of every point or
    with the error which failed it, so the points of the batch run at once in
    this process rather than one after the other, see
    `sbmlsim.simulator.worker._run_chunk_core` for the points of a chunk of a
    scan and `sbmlsim.fit.optimization.OptimizationProblem._simulate_batch`
    for the simulations of an evaluation of a fit.

    The result is the one of the plans run one by one, bitwise: every thread
    of the batch simulates on an instance of its own and the engine holds no
    state between two requests. The model records no pre-initialization,
    unlike `execute`: the engine is stateless and every point of the batch
    carries the values of its simulation in its request.

    The points are simulated in chunks of at most `CORE_MAX_CHUNK`, so a
    batch of thousands of points holds the tables of one chunk and not the
    ones of the batch; the threads of the batch are the ones of every chunk.

    Args:
        points: the plan of every point and the columns of its output, in
            their order.
        model: the model of every point of the batch.
        threads: the threads of the engine the batch runs on, at least one.

    Returns:
        The output of every point in the order of the points, the error which
        failed it instead: the error of its integration, of its presimulation
        or of the request which could not be built for it, see `point_error`
        and `_failed`, which an evaluation of a fit handles as the one of a
        simulation of its own.

    Raises:
        NotImplementedError: for a plan the engine does not support, which
            fails for every value of the parameters of a fit and for every
            point of a scan, see `check_supported`.
        ValueError: if the engine refuses the batch, e.g. for more threads
            than it runs on.
    """
    # the cap of the points of one batch, which the simulator of a scan
    # chunks its points with as well; imported here, as the simulator imports
    # the executors
    from sbmlsim.simulator.simulator import CORE_MAX_CHUNK

    core = import_core()
    # the tolerance of every state is computed once for the whole batch
    settings = model.request_settings()
    results: list[TimecourseResult | Exception] = []
    for start in range(0, len(points), CORE_MAX_CHUNK):
        chunk = points[start : start + CORE_MAX_CHUNK]
        of_point: dict[int, TimecourseResult | Exception] = {}
        requests: list[dict[str, Any]] = []
        built: list[tuple[int, Plan, Sequence[str]]] = []
        # the requests of the chunk, the folds of its points batched (#550); a
        # point whose request cannot be built fails alone
        for k, (request, (plan, selections)) in enumerate(
            zip(
                requests_of(
                    chunk,
                    model,
                    lambda segments: _batch(model, segments, threads),
                    settings,
                ),
                chunk,
                strict=True,
            )
        ):
            if isinstance(request, Exception):
                of_point[k] = request
                continue
            requests.append(request)
            built.append((k, plan, output_columns(selections)))
        outputs = _batch(model, requests, threads)
        # the requests of the batch are released with it, the engine has read them
        del requests
        for position, (k, plan, columns) in enumerate(built):
            output = outputs[position]
            # the table of the point is released once its values are read
            outputs[position] = None
            if isinstance(output, core.PointError):
                err = point_error(output, model)
                if isinstance(err, NotImplementedError):
                    raise err
                of_point[k] = err
                continue
            # the time is shifted in the array the bindings own, which
            # `output.values` answers with as it is, so it is read once here
            values = output_time(output.values, columns, plan, model.symbols)
            of_point[k] = TimecourseResult(columns=tuple(columns), values=values)
        # every point of the chunk has an output or the error which failed it
        results.extend(of_point[k] for k in range(len(chunk)))
    return results


def _batch(
    model: CoreSBMLModel, requests: list[dict[str, Any]], threads: int
) -> list[Any]:
    """Simulate the requests of a batch on the executor of a model.

    The batch runs on the executor of the model (`CoreSBMLModel.batch_executor`,
    one per number of threads), which keeps the pool of its threads and the
    instance of every one of them across the batches of a model, so a batch
    pays for neither.

    Args:
        model: the model of the batch.
        requests: the request of every point of the batch, in their order.
        threads: the threads of the engine the batch runs on.

    Returns:
        The output of every request in the order of the requests, the
        `PointError` of a point which failed among them.

    Raises:
        ValueError: if the engine refuses the batch, e.g. for more threads
            than it runs on.
    """
    if not requests:
        return []
    try:
        return model.batch_executor(threads).simulate_many(requests)
    except ValueError as err:
        raise ValueError(
            f"The engine 'core' refused a batch of {len(requests)} points on "
            f"{threads} threads: {err}"
        ) from err


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
    segment before it, see `_Fold`. The segments of one plan run one after the
    other here; the points of a chunk fold their k-th segment as one batch,
    see `requests_of`. A plan whose changes are numbers and formulas of the
    time, which every plan of `compile_simulation` without a formula of an
    entity is, simulates nothing here.

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
    request = requests_of([(plan, selections)], model, _one_by_one(model), settings)[0]
    if isinstance(request, Exception):
        raise request
    return request


def requests_of(
    points: Sequence[tuple[Plan, Sequence[str]]],
    model: CoreSBMLModel,
    simulate: Callable[[list[dict[str, Any]]], Sequence[Any]],
    settings: dict[str, object] | None = None,
) -> list[dict[str, Any] | Exception]:
    """Get the requests of the points of a chunk, their folds batched (#550).

    The plural of `request_of`: the request of every point of a chunk of a
    scan or of an evaluation of a fit, with the k-th segment of the fold of
    every point of the chunk simulated as one batch instead of the segments of
    one point after the other. The k-th segment of a point is independent of
    the ones of the other points - every point folds its own state, as it does
    alone - so the folds of a chunk run on the threads of the engine and not
    in this process one at a time, which is where a relative protocol spends
    its time: a scan of 20 points of three doses of HCTZ costs 171 ms with the
    folds of its points one after the other against 84 ms with the doses as
    numbers (roadrunner: 92 ms).

    The batch of a round is the points which still have a segment to fold, so
    a chunk pays as many rounds as its longest protocol has changes which read
    the model, and a point whose changes are numbers is in none of them. The
    chunk holds the tables of one round, one segment per point of the round,
    which carry the symbols of the folds and the output times inside the
    segment and no selection of the result; every one of them is read and
    released as soon as its round is answered.

    Args:
        points: the plan of every point and the columns of its output, in
            their order.
        model: the model of every point, which gives the settings of the
            integrator and the values the folds read.
        simulate: what simulates the segments of a round, the batch of the
            caller: it answers with the output of every segment in the order
            of the requests, or with the error of a segment which failed
            among them (a `PointError` of the engine, or an exception), see
            `_one_by_one` for the segments of a single plan.
        settings: the settings of the requests, `model.request_settings()` by
            default, which computes the absolute tolerance of every state once
            for every point of the chunk.

    Returns:
        The request of every point in the order of the points, the exception
        which failed it instead: a plan the executor refuses, a formula it
        cannot evaluate or the error of a segment of its fold, which the
        caller flags as the failure of that point.

    Raises:
        NotImplementedError: for a plan or a simulation the engine does not
            support, which fails every point of a chunk alike, see
            `check_supported` and `point_error`.
    """
    request_settings = model.request_settings() if settings is None else settings
    folds: list[_Fold | Exception] = []
    for plan, selections in points:
        try:
            folds.append(
                _fold_of(plan, model, output_columns(selections), request_settings)
            )
        except NotImplementedError:
            # a feature the engine does not simulate fails every point
            raise
        except Exception as err:  # the point fails alone
            folds.append(err)
    while True:
        segments: list[dict[str, Any]] = []
        waiting: list[_Fold] = []
        for fold in folds:
            if isinstance(fold, Exception):
                continue
            segment = fold.step()
            if segment is not None:
                segments.append(segment)
                waiting.append(fold)
        if not segments:
            break
        outputs = list(simulate(segments))
        del segments
        for position, fold in enumerate(waiting):
            output = outputs[position]
            # the table of a segment is released once its row is read
            outputs[position] = None
            fold.take(output)
    return [fold if isinstance(fold, Exception) else fold.result() for fold in folds]


def _one_by_one(
    model: CoreSBMLModel,
) -> Callable[[list[dict[str, Any]]], Sequence[Any]]:
    """Get the `simulate` of a fold which runs its segments one after the other.

    What `request_of` folds a single plan with: the segments of one plan
    resume from each other, so they are one request after the other anyway
    and run on an instance of the model, as the simulation of the plan does,
    and not on the executor of a batch.

    Args:
        model: the model the segments run on.

    Returns:
        What simulates the segments of a round, see `requests_of`.
    """
    core = import_core()

    def simulate(requests: list[dict[str, Any]]) -> Sequence[Any]:
        outputs: list[Any] = []
        for request in requests:
            try:
                outputs.append(model.instance().simulate(request))
            except core.SimulationError as err:
                failure = _failed(err, model, core)
                failure.__cause__ = err
                outputs.append(failure)
        return outputs

    return simulate


def _fold_of(
    plan: Plan,
    model: CoreSBMLModel,
    columns: Sequence[str],
    settings: dict[str, object],
) -> _Fold:
    """Get the fold of the changes of a plan, its request without them.

    Args:
        plan: the plan, compiled against the model.
        model: the model.
        columns: the columns of the result of the point.
        settings: the settings of the request.

    Returns:
        The fold, whose request is every field of the contract but the
        changes.

    Raises:
        UnsupportedModelError: for a plan the engine does not simulate as
            roadrunner, see `check_supported`.
        ValueError: for a formula before the initialization which the
            executor cannot evaluate, see `_preinit_value`.
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
        "selections": output_columns(columns),
        "settings": settings,
    }
    return _Fold(plan=plan, model=model, request=request)


class _Fold:
    """The changes of one plan, folded into the numbers of its request.

    A formula at a change is evaluated by the executor of roadrunner with the
    state of the model at the time of the change (`apply_changes` of
    `sbmlsim.simulator.changes`), which a request carries no formula of: the
    executor reads that state as the outputs of the segment which ends at the
    change, with a request of its own, and sends the number. The changes are
    folded in their order, the one of their times, so the request of a segment
    carries the changes before it, already folded, and resumes from the state
    of the segment before it.

    The fold is a step at a time so that the folds of the points of a chunk
    run as one batch per step (`requests_of`): `step` gives the request of the
    next segment, which the caller simulates, `take` folds the change that
    segment was simulated for, and `result` is the request of the plan once
    `step` answers with nothing. Every change whose values are numbers or
    formulas of the time is folded by `step` itself, so a plan without a
    formula of an entity asks for no segment at all.

    A failure of the fold is the failure of its point alone and is kept
    (`error`): the plan of another point of the chunk folds whatever happened
    here. A feature the engine does not support is raised instead, it fails
    every point of the chunk alike.
    """

    def __init__(self, plan: Plan, model: CoreSBMLModel, request: dict[str, Any]):
        """Create the fold of the changes of a plan.

        Args:
            plan: the plan, whose events are sorted by their time.
            model: the model, which the engine reads the state of.
            request: the request which is built, whose fields the request of a
                segment takes, its changes excluded.
        """
        self.plan = plan
        self.model = model
        self.request = request
        #: the symbols of the model every change reads, by the position of
        #: its event: the state of the model is read only where a formula
        #: reads it
        self.reads = [_change_symbols(event) for event in plan.events]
        #: the targets a segment resumes from, `None` for a model whose state
        #: the executor cannot hand to the engine, whose segments start at
        #: `plan.start`
        self.targets = _state_targets(model)
        #: the changes which are folded, in their order
        self.changes: list[dict[str, Any]] = []
        #: the event which is folded, the one the pending segment ends at
        self.index = 0
        #: the time of the last segment which was simulated and the state
        #: after the change at that time, which the next segment resumes from
        self.resume: tuple[float, list[list[Any]]] | None = None
        #: the selections of the pending segment
        self.columns: list[str] = []
        #: what failed the point, if anything
        self.error: Exception | None = None

    def step(self) -> dict[str, Any] | None:
        """Get the request of the next segment of the fold, or nothing.

        Every change until the next one which reads the model is folded here,
        a change of numbers and of formulas of the time among them.

        Returns:
            The request of the segment which ends at the next change which
            reads the state of the model, `None` if the fold has no such
            change left or failed.

        Raises:
            NotImplementedError: for a feature the engine does not support.
        """
        if self.error is not None:
            return None
        try:
            return self._segment()
        except NotImplementedError:
            raise
        except Exception as err:
            self.error = err
            return None

    def take(self, output: Any) -> None:
        """Fold the change the segment of `step` was simulated for.

        Args:
            output: the output of the segment, or the error which failed it
                (a `PointError` of the engine or an exception).

        Raises:
            NotImplementedError: for a feature the engine does not support.
        """
        try:
            self._consume(output)
        except NotImplementedError:
            raise
        except Exception as err:
            self.error = err

    def result(self) -> dict[str, Any] | Exception:
        """Get the request of the plan, or the error which failed its point."""
        if self.error is not None:
            return self.error
        self.request["changes"] = self.changes
        return self.request

    def _segment(self) -> dict[str, Any] | None:
        """Fold the changes of numbers and build the request of the next segment."""
        while self.index < len(self.plan.events):
            event = self.plan.events[self.index]
            symbols = self.reads[self.index]
            if symbols:
                self.columns = output_columns(
                    sorted({*symbols, *(self._resumed() or ())})
                )
                return _segment_request(
                    self.plan,
                    self.request,
                    self.changes,
                    self.resume,
                    event.time,
                    self.columns,
                )
            self.changes.append(
                {"time": event.time, "values": _change_values(event, {})}
            )
            self.index += 1
        return None

    def _consume(self, output: Any) -> None:
        """Read the state of the segment and fold the change at its end."""
        core = import_core()
        if isinstance(output, core.PointError):
            raise point_error(output, self.model)
        if isinstance(output, BaseException):
            raise output
        event = self.plan.events[self.index]
        start = self.plan.start if self.resume is None else self.resume[0]
        state = _state_of(
            output, self.columns, probe_end(self.plan, start, event.time), start
        )
        values = _change_values(event, state)
        self.changes.append({"time": event.time, "values": values})
        resumed = self._resumed()
        if resumed is not None:
            self.resume = (
                event.time,
                _resume_values(resumed, self.model, state, values),
            )
        self.index += 1

    def _resumed(self) -> list[str] | None:
        """Get the targets the next segment resumes from, or nothing.

        The state is read only where a later change resumes from it, so a plan
        with one such change asks for the symbols of its formula alone.
        """
        return self.targets if any(self.reads[self.index + 1 :]) else None


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
    if plan.steady_state_output is not None and (
        reason := steady_state_gap(plan, model)
    ):
        raise UnsupportedModelError(
            f"The engine 'core' does not simulate the steady state after the end "
            f"of the simulation of the model '{model.label}', the output at the "
            f"time 'inf': the steady state of a request is the pre-equilibration "
            f"before the start of the simulation, so the executor continues from "
            f"the state the timecourse ended in with a request of its own, which "
            f"it cannot do here, since {reason}."
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


def _state_targets(model: CoreSBMLModel) -> list[str] | None:
    """Get the targets a segment of the fold resumes from, or `None`.

    `CoreSBMLModel.resume_targets`, which computes them once per model and
    selections: they are the same for every point of a scan, which called
    this for each of them. It stays a function of this module because it is
    the seam the tests replace to run the fold without resuming.

    Args:
        model: the model.

    Returns:
        The targets, sorted, or `None`, see `CoreSBMLModel.resume_targets`.
    """
    return model.resume_targets()


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


def probe_end(plan: Plan, start: float, time: float) -> float:
    """Get the time the probe of a change integrates to.

    The formulas of a change read the state of the model **before** the
    events of its time have fired, which is what `apply_changes` of
    `sbmlsim.simulator.changes` reads on the model of roadrunner: the
    executor of roadrunner applies a change to the state its segment stopped
    at, one unit in the last place before the time (`segment_stop`), and
    fires the events after it.

    The engine answers the state of a time after the events of that time have
    fired, so the probe of a model with events ends at that same stop instead
    of at the time of the change, which is the state roadrunner folds, to its
    last unit in the last place. A model without events has no such state to
    tell apart and its probe ends at the time of the change, so its request
    is the one of before, bitwise; so is the probe of a segment too short to
    integrate (`integrable`), whose state is the one at its start.

    Args:
        plan: the plan, whose model gives the events.
        start: the start of the segment.
        time: the time of the change.

    Returns:
        The end of the probe, the time of the change or the stop before it.
    """
    if not plan.symbols.events:
        return time
    stop = segment_stop(start, time)
    return stop if integrable(start, stop) else time


def _segment_request(
    plan: Plan,
    request: Mapping[str, Any],
    changes: Sequence[dict[str, Any]],
    resume: tuple[float, list[list[Any]]] | None,
    time: float,
    selections: Sequence[str],
) -> dict[str, Any]:
    """Get the request which reads the state of the model at the time of a change.

    The engine hands out no state of a model, only the outputs of a request,
    so the executor asks it for the values a formula reads over the segment
    which ends at the change, with a request of its own. The state at the
    change is the last row of its output, see `_state_of`, and its selections
    are the executor's own and no column of the result of the caller.

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
        The request of the segment.
    """
    start = plan.start if resume is None else resume[0]
    end = probe_end(plan, start, time)
    segment = {
        **request,
        "start": start,
        "end": end,
        "output": _segment_output(plan, start, end),
        "selections": output_columns(selections),
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
    return segment


def _state_of(
    output: Any, columns: Sequence[str], time: float, start: float
) -> dict[str, float]:
    """Get the state of the model at the time of a change from the output of a segment.

    The last row of the output of the segment, the one the engine integrates
    up to before it applies a change there, the first row for a change at the
    start of the segment, whose interval is empty.

    Args:
        output: the output of the segment, see `_segment_request`.
        columns: the selections of the segment, the columns of its output.
        time: the time of the change, the end of the segment.
        start: the start of the segment.

    Returns:
        The value of every selection at the time of the change.
    """
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


def steady_state_columns(
    plans: Sequence[Plan], model: CoreSBMLModel, columns: Sequence[str]
) -> list[str]:
    """Get the columns a run with a steady state after the end reads.

    The caller's columns and, for a plan which has a steady state after the
    end, the targets the executor carries the state of the timecourse into
    the request of the steady state with (`_state_targets`), which the output
    of the timecourse must have for `steady_state_rows` to read it.

    Args:
        plans: the plans of the run, one for a single simulation and one per
            point for a chunk of a scan.
        model: the model.
        columns: the columns of the result.

    Returns:
        The columns of the run, the caller's first and in their order.
    """
    if all(plan.steady_state_output is None for plan in plans):
        return list(columns)
    targets = _state_targets(model) or ()
    seen = set(columns)
    return [*columns, *(t for t in targets if t not in seen)]


def steady_state_kept(
    values: np.ndarray, run: Sequence[str], columns: Sequence[str]
) -> np.ndarray:
    """Drop the columns a run added for the state of the steady state.

    Args:
        values: the values of the run, over the columns of the run.
        run: the columns of the run, see `steady_state_columns`.
        columns: the columns of the result, the first of the run.

    Returns:
        The values over the columns of the result, the array itself when the
        run read nothing else.
    """
    if len(run) == len(columns):
        return values
    return values[:, : len(columns)]


def steady_state_rows(
    points: Sequence[tuple[Plan, dict[str, Any], np.ndarray]],
    model: CoreSBMLModel,
    columns: Sequence[str],
    simulate: Callable[[list[dict[str, Any]]], Sequence[Any]],
) -> list[np.ndarray | Exception]:
    """Get the row of the steady state after the end of every point which has one.

    The contract has no steady state after the end of a simulation, so the
    executor continues from the state the timecourse ended in with a second
    request per point (`steady_state_request`), and the requests of the points
    of a chunk are simulated as one batch, as their folds are, so a chunk pays
    one round for all of them.

    Args:
        points: the plan, the request of the timecourse and the values of its
            output, in their order; a plan without a steady state after the
            end is left out by the caller.
        model: the model, which reads the state out of the output.
        columns: the columns of the output, which carry the state.
        simulate: what simulates the requests, the batch of the caller, as in
            `requests_of`.

    Returns:
        The row of the steady state of every point, in its order, with the
        time `inf`, or the exception which failed it.
    """
    targets = _state_targets(model)
    assert targets is not None
    requests: list[dict[str, Any]] = []
    for plan, request, values in points:
        state = _state_of(_Values(values), columns, 1.0, 0.0)
        requests.append(
            steady_state_request(request, plan, {t: state[t] for t in targets}, columns)
        )
    rows: list[np.ndarray | Exception] = []
    for output in simulate(requests):
        if isinstance(output, BaseException):
            rows.append(
                output if isinstance(output, Exception) else RuntimeError(str(output))
            )
            continue
        core = import_core()
        if isinstance(output, core.PointError):
            rows.append(point_error(output, model))
            continue
        row = np.array(output.values[-1], dtype=float)
        row[0] = np.inf
        # the time of the row is `inf` in every `time` column, as
        # `output_time` copies it, except in a model with an entity of that
        # name, whose later column is the value of the entity (#445)
        rows.append(output_time(row[np.newaxis, :], columns, _AT_INF, model.symbols)[0])
    return rows


#: the plan of the row of a steady state: no shift of the time, which is
#: `inf` there, so `output_time` only copies it into a later `time` column
_AT_INF: Final[Any] = SimpleNamespace(time_shift=0.0)


@dataclass(frozen=True)
class _Values:
    """The output of a request as `_state_of` reads it, its values alone."""

    values: np.ndarray


def steady_state_gap(plan: Plan, model: CoreSBMLModel) -> str | None:
    """Get why the executor cannot continue a steady state from the end, or `None`.

    The contract has no steady state after the end of a simulation, so the
    executor continues from the state the timecourse ended in with a second
    request, whose pre-initialization carries that state and whose
    `steady_state` is the criterion of the plan (`steady_state_request`). Two
    things keep it from doing so:

    - a model with events of its own, whose triggers the second request
      cannot be told the state of: a request initializes the model, where a
      trigger which the state carried into it makes true fires again, while
      the executor of roadrunner continues with the triggers as the
      timecourse left them and such an event does not fire twice (case 0023
      of the PEtab test suite, whose event adds 1000 to its state at the
      threshold the steady state is past);
    - a model whose state it cannot set, the one `resume_targets` answers
      `None` for, e.g. a species reference which is not constant: the state
      of the second request would silently be the one of the model as loaded;
    - a model which reads the time, since the steady state of a request is
      integrated from the time 0 whatever the start of its simulation, while
      the executor of roadrunner continues at the time the timecourse ended
      at (`executor.execute`, which starts at 0 for a model which does not
      read the time, so the two agree exactly there).

    Args:
        plan: the plan, whose model symbols give the events and say whether
            it reads the time.
        model: the model.

    Returns:
        The reason, a clause of the message of the refusal, or `None` when
        the executor continues the steady state.
    """
    if plan.symbols.events:
        return (
            "the model has events of its own, whose triggers a request "
            "cannot be told the state of, so one which the state carried "
            "into it makes true would fire again"
        )
    if plan.symbols.time_dependent:
        return (
            "the model reads the time and the steady state of a request is "
            "integrated from the time 0, while the executor of roadrunner "
            "continues at the time the timecourse ended at"
        )
    if _state_targets(model) is None:
        return (
            "the model has a state the executor cannot set, so the state the "
            "timecourse ended in cannot be carried into a request"
        )
    return None


def steady_state_request(
    request: Mapping[str, Any],
    plan: Plan,
    state: Mapping[str, float],
    columns: Sequence[str],
) -> dict[str, Any]:
    """Get the request of the steady state after the end of a simulation.

    The state the timecourse ended in is set before the initialization, where
    it holds over the initial assignment of its target (PEtab v2, which the
    engine applies), and the `steady_state` of the request is the criterion of
    the plan, so the engine initializes the model in that state and
    equilibrates it. The simulation which follows is the one output time at
    the end of the plan, whose single row is the steady state; the caller
    gives it the time `inf`, as `Plan.steady_state_output` has it.

    The pre-initialization of the plan and its presimulation are not carried:
    the state replaces them, and a value of theirs which is no state, e.g. a
    parameter of a rule, is the one of the model as loaded, which the
    initialization gives it again.

    Args:
        request: the request of the timecourse, whose fields it takes but its
            interval, output, pre-initialization, steady state, changes and
            selections.
        plan: the plan, which gives the end and the criterion.
        state: the state at the end of the timecourse, by `_state_targets`.
        columns: the columns of the result.

    Returns:
        The request of the steady state.
    """
    steady = plan.steady_state_output
    assert steady is not None
    return {
        **request,
        "start": plan.end,
        "end": plan.end,
        "output": {TIMES: [plan.end]},
        "preinit": [[target, value] for target, value in state.items()],
        "changes": [],
        "steady_state": {
            "preinit": [],
            "absolute_tolerance": steady.absolute_tolerance,
            "relative_tolerance": steady.relative_tolerance,
            "max_time": steady.max_time,
        },
        "selections": output_columns(columns),
    }


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
        moved = sorted(s for s in symbols if s in _event_moved(events, model))
        if moved:
            raise UnsupportedModelError(
                f"The engine 'core' does not simulate the formula '{a.formula}' of "
                f"'{a.target}' at the time {event.time} of the model "
                f"'{model.label}', which has the event '{_moving_event(events, moved)}'"
                f": it reads {moved}, which an event of the model "
                f"assigns or which follows such a target through an assignment "
                f"rule, and the state of the model at the time of a change is the "
                f"one a request of the engine answers there, which is the state "
                f"after the events of that time have fired, while the executor of "
                f"roadrunner reads it before they fire. An event of the model "
                f"which fires at the time of the change would make the two engines "
                f"disagree without an error, so the formula is refused, see "
                f"`sbmlsim.simulator.executor_core`."
            )


def _moving_event(events: Sequence[EventSymbols], moved: Sequence[str]) -> str:
    """Get the id of an event which assigns one of the symbols, for the message.

    Args:
        events: the events of the model, at least one.
        moved: the symbols the formula reads and an event could move.

    Returns:
        The id of the first event which assigns one of them, or of the first
        event when they follow their targets through an assignment rule.
    """
    wanted = {name[1:-1] if name.startswith("[") else name for name in moved}
    for event in events:
        targets = {target for target, _ in event.assignments}
        if targets & (wanted | set(moved)):
            return event.eid
    return events[0].eid


def _event_moved(
    events: Sequence[EventSymbols], model: CoreSBMLModel
) -> frozenset[str]:
    """Get the selections an event of the model could move at the time of a change.

    These are the targets of the assignments of its events and, since an
    assignment rule follows its inputs at once, every variable of an
    assignment rule, whose value an event could move through it without
    being its target. A model without events moves nothing.

    The concentration and the amount of a species go together, so a target of
    either names both, and so does the compartment of a species of a
    concentration, whose size scales the amount.

    Args:
        events: the events of the model.
        model: the model, which gives the species and their compartments.

    Returns:
        Every selection the executor must not read at the time of a change.
    """
    if not events:
        return frozenset()
    symbols = model.symbols
    moved: set[str] = set(symbols.assignment_rules)
    for event in events:
        moved |= {target for target, _ in event.assignments}
    # a target names the amount, the concentration and, for a species of a
    # concentration, the size of its compartment, which scales them
    for name in list(moved):
        sid = name[1:-1] if name.startswith("[") else name
        if sid in symbols.species:
            moved |= {sid, f"[{sid}]"}
    for sid, compartment in symbols.species_compartment.items():
        if compartment in moved:
            moved |= {sid, f"[{sid}]"}
    return frozenset(moved)


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

    - the symbol is a constant parameter of the model, the variable of no
      initial assignment and of no rule (`_constant_parameter`): nothing the
      pre-initialization sets can move it, so the value of the model as
      loaded is the one roadrunner reads there whatever else is set. The
      parameters of a PEtab problem are such parameters, which is what a
      condition like `[A] = initial_A1 + initial_A2` reads.

    Everything else is refused: a symbol which the pre-initialization does not
    set and which is no constant parameter follows the values it sets
    wherever an assignment rule or an initial assignment of the model relates
    them, e.g. the rule `kk := 3*f` of a pre-initialization which sets `f`, so
    the value of the model as loaded would be the wrong one.

    A formula of any target is folded, a parameter, a compartment and the
    amount or the concentration of a species alike: the fold computes the
    number the formula has, and the engine then applies that number to the
    target with the bookkeeping of the volumes and the amounts of its
    initialization, exactly as it applies a number the pre-initialization
    states for the same target (#607). Only a number of a parameter is read as
    the value of a symbol, though: a pre-initialization which sets the
    concentration `[S]` does not give the amount `S` that number, the
    bookkeeping of the initialization does, so a formula which reads a symbol
    the pre-initialization sets as anything but a parameter is refused.

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
        if plain and not _constant_parameter(model, symbol):
            raise _no_fold(
                assignment,
                model,
                where,
                f"it reads '{symbol}', which the pre-initialization does not set "
                f"to a number and which is no constant parameter of the model, so "
                f"the values it sets could move it and they are not on the model "
                f"when the formula is evaluated",
            )
        values.append(loaded)
    return formula.evaluate(values)


def _constant_parameter(model: CoreSBMLModel, symbol: str) -> bool:
    """Whether a symbol is a constant parameter of the model.

    Such a parameter is the variable of no initial assignment and of no rule,
    so its value is the one its SBML states and nothing a pre-initialization
    sets can move it: the value of the model as loaded is the one roadrunner
    reads when it evaluates a formula of the pre-initialization, whatever
    else that pre-initialization sets (`_preinit_value`). The parameters of a
    PEtab problem are such parameters, which is why a condition like
    `[A] = initial_A1 + initial_A2` is folded.

    Args:
        model: the model.
        symbol: the symbol of the formula.

    Returns:
        Whether the symbol is a constant parameter.
    """
    symbols = model.symbols
    return (
        symbol in symbols.parameters
        and symbol not in symbols.initial_assignments
        and symbol not in symbols.assignment_rules
        and symbol not in symbols.rate_rules
    )


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
