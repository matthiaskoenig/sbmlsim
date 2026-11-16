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
   formula is folded where the executor evaluates the formula without the
   model and refused otherwise (`check_supported`): a formula of the time at
   a change, whose time the plan carries, and a formula of a parameter
   before the initialization whose symbols are the values of the model as
   loaded or the numbers the same pre-initialization sets, see
   `_preinit_value`. The model of the engine is stateless, so the executor
   cannot set the values of a pre-initialization on it and read what a
   formula reads, as the executor of roadrunner does. Only a formula of a
   parameter is folded: the size of a compartment and the amount or the
   concentration of a species are set with the bookkeeping of the volumes
   and the amounts of the initialization, which the engine does otherwise,
   see the next difference; the day the engine holds the concentrations of
   a pre-initialization across the initialization, a formula of a
   compartment and of a species can be folded as well.
8. The engine converts the concentration of a species a pre-initialization
   sets into an amount with the volume its compartment has at the moment of
   the assignment, where every other executor holds the concentration
   across the initialization and converts it with the size the compartment
   has when the model is initialized (PEtab v2, a compartment keeps the
   concentrations of the species in it). The two differ wherever the same
   pre-initialization changes that size, whether it sets the compartment
   itself or a symbol its size follows by an initial assignment:
   `{"[A]": 5.0, "C": 3.0}` of the probe model, whose `C` is 2 as loaded, is
   `A = 15` in roadrunner and `A = 10` in the engine, and on a model whose
   compartment is `c = V*f` the pre-initialization `{"[A]": 5.0, "f": 3.0}`
   is `A = 30` against `A = 10` (measured). The engine applies the values of
   a request one after the other, so the direct pair comes out right in the
   other order, `{"C": 3.0, "[A]": 5.0}`, which is a symptom of that one
   rule and no second difference. `check_supported` refuses a
   pre-initialization which sets the concentration of a species and changes
   the size of its compartment (`_check_concentrations`) rather than
   simulate a state neither engine would answer for, and the static bundle
   of the panels of a model, whose recipe the engine runs in the browser
   without a plan at all, shares that refusal (`concentration_problem`, the
   one place the rule lives, see `sbmlsim.serve.export`). Measured and not
   refused, since the engines agree on them: an amount of such a
   compartment, the amount of a substance species among them, a
   pre-initialization which changes the size of a compartment without
   setting a species in it, whose concentrations the engine carries across
   the initialization itself, and the values of a change, which the engine
   applies at once.
9. The engine integrates with the absolute tolerance of every state of
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

from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

from sbmlsim.engine import UnsupportedModelError, import_core
from sbmlsim.model.symbols import ModelSymbols, TargetKind
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

    Args:
        plan: the plan, compiled against the model.
        model: the model, which gives the settings of the integrator and the
            values a formula before the initialization reads.
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
    """
    check_supported(plan, model)
    output: Any = (
        INTEGRATOR_STEPS
        if plan.output is OutputMode.INTEGRATOR
        else {TIMES: list(plan.times)}
    )
    plain = _plain_values(preinit_targets(plan))
    return {
        "contract": CONTRACT,
        "start": plan.start,
        "end": plan.end,
        "output": output,
        "preinit": _preinit_values(plan.preinit, plain, model),
        "changes": [
            {"time": event.time, "values": _change_values(event, model)}
            for event in plan.events
        ],
        "steady_state": _steady_state(plan.steady_state, plain, model),
        "selections": output_columns(selections),
        "settings": model.request_settings() if settings is None else settings,
    }


def check_supported(plan: Plan, model: CoreSBMLModel) -> None:
    """Refuse a plan the engine does not simulate as roadrunner.

    `request_of` checks every plan, and `Simulator.run` the plan of the first
    point of every simulation and model before it runs a point, see
    `sbmlsim.simulator.engines.check_plan`, so a plan the engine refuses fails
    the whole scan whatever `on_error` says, which is what a plan one request
    cannot carry deserves: it fails every point of the scan alike.

    The steady state after the end of a simulation is no field of the
    contract, a value of a request is a number, so a formula the executor
    cannot fold is refused, and the engine converts the concentration of a
    species with the volume its compartment has at the moment of the
    assignment, so a pre-initialization which sets a concentration and
    changes the size of its compartment is refused, see the module,
    `_preinit_value` and `_check_concentrations`.

    Args:
        plan: the plan, compiled against the model, with the values of a
            point.
        model: the model, which gives the values a formula before the
            initialization reads.

    Raises:
        UnsupportedModelError: a `NotImplementedError`, for a plan with a
            steady state after the end of its simulation, for a formula the
            executor cannot fold into a number and for a pre-initialization
            of a concentration whose volume it changes.
    """
    if plan.steady_state_output is not None:
        raise UnsupportedModelError(
            f"The engine 'core' does not simulate the steady state after the end "
            f"of the simulation of the model '{model.label}', the output at the "
            f"time 'inf': the steady state of a request is the pre-equilibration "
            f"before the start of the simulation, so one request cannot continue "
            f"from the state the timecourse ended in."
        )
    _check_concentrations(plan, model)
    # the values the request carries, which every formula of the plan is
    # folded into and which are discarded here: a formula the executor cannot
    # fold raises, see `_preinit_values` and `_change_values`
    plain = _plain_values(preinit_targets(plan))
    _preinit_values(plan.preinit, plain, model)
    if plan.steady_state is not None:
        _preinit_values(plan.steady_state.preinit, plain, model)
    for event in plan.events:
        _change_values(event, model)


def _check_concentrations(plan: Plan, model: CoreSBMLModel) -> None:
    """Refuse a pre-initialization of a concentration whose volume it changes.

    The rule is `concentration_problem`, which the static bundle of the panels
    of a model shares; the plan applies its pre-initialization and the one of
    its steady state as one pre-initialization (`preinit_targets`), and its
    symbols give the compartment of a species and what its size depends on.

    Args:
        plan: the plan.
        model: the model, which the error names.

    Raises:
        UnsupportedModelError: if the pre-initialization sets the
            concentration of a species and changes the size of its
            compartment.
    """
    problem = concentration_problem(
        [a.target for a in preinit_targets(plan)],
        plan.symbols,
        f"of the model '{model.label}'",
    )
    if problem is not None:
        raise UnsupportedModelError(problem)


def concentration_problem(
    targets: Iterable[str], symbols: ModelSymbols, where: str
) -> str | None:
    """Tell why the engine gets a pre-initialization of a concentration wrong.

    The one place the rule of difference 8 of the module lives, so that the
    executor and the static bundle of the panels of a model, whose recipe the
    engine runs in the browser without a plan at all
    (`sbmlsim.serve.export`), refuse the same pre-initializations with the
    same words. It goes, with both of its callers, the day sbmlsim-core#7 is
    released: the engine then holds the concentration across the
    initialization itself and nothing is left to refuse. Two texts go with
    them, which `grep sbmlsim-core#7` finds as well: the refusals of the
    engine core in `docs/simulation.md` and the clause of the engines in
    `CLAUDE.md`.

    The engine converts the concentration of a species a pre-initialization
    sets into an amount with the volume its compartment has at the moment of
    the assignment, where every other executor holds the concentration across
    the initialization and converts it with the size the compartment has when
    the model is initialized (PEtab v2, a compartment keeps the concentrations
    of the species in it, `RoadrunnerSBMLModel.initialize`). The two differ
    wherever the same pre-initialization changes that size: `{"[A]": 5.0,
    "C": 3.0}` of the probe model, whose `C` is 2 as loaded, is `A = 15` in
    roadrunner and `A = 10` in the engine, and so is `{"[A]": 5.0, "f": 3.0}`
    of a model whose compartment is `c = V*f`, where no value of the
    pre-initialization is the compartment at all. The engine applies the
    values of a request one after the other, so the direct pair comes out
    right in the other order, which is a symptom of that one rule.

    Every other pre-initialization of such a compartment is simulated, since
    the engines agree on them (measured, see the module): an amount, the
    amount of a substance species among them, which is set as it is whenever
    the volume settles, and a change of the size alone, whose concentrations
    the engine carries across the initialization itself.

    Args:
        targets: the targets of the pre-initialization, in the order a
            problem reports them: `[S]` for the concentration of a species
            and the id of the entity for every other value. A target the
            model has no species or compartment of is no problem of this
            rule.
        symbols: the symbols of the model, which give the compartment of a
            species and the symbols its size depends on.
        where: the model and whatever else names the pre-initialization, which
            the problem names after the species, e.g. `of the model 'hctz'`
            or `of the panel 'patient_plasma' of the model 'hctz' in its
            variant 'default'`.

    Returns:
        The problem of the first concentration whose volume the
        pre-initialization changes, `None` for a pre-initialization the
        engine simulates as every other executor does.
    """
    given = list(targets)
    entities = {symbols.entity(target) for target in given}
    for target in given:
        species = symbols.entity(target)
        if species == target:
            continue
        compartment = symbols.species_compartment.get(species)
        if compartment is None:
            continue
        changed = sorted(_volume_symbols(symbols, compartment) & entities)
        if not changed:
            continue
        named = ", ".join(f"'{symbol}'" for symbol in changed)
        return (
            f"The engine 'core' does not simulate the pre-initialization of "
            f"the concentration of the species '{species}' {where} together "
            f"with {named}, which the size of its compartment '{compartment}' "
            f"follows: the engine converts the concentration into an amount "
            f"with the volume at the moment of the assignment, where "
            f"roadrunner holds the concentration across the initialization "
            f"and converts it with the size the compartment has once the "
            f"model is initialized (PEtab v2), so the two agree only where "
            f"the pre-initialization leaves that size alone."
        )
    return None


def _volume_symbols(symbols: ModelSymbols, compartment: str) -> set[str]:
    """Get a compartment and every symbol its size depends on.

    The size of a compartment of an initial assignment follows the symbols of
    that assignment, and they follow theirs, so the dependencies are walked to
    their end: on a model of `c = V*f` and `V = 2*g` the size of `c` follows
    `V`, `f` and `g`. `ModelSymbols.initial_assignment_dependencies` has
    expanded the assignment rules of every such symbol already, see
    `sbmlsim.model.symbols`.

    Args:
        symbols: the symbols of the model.
        compartment: the compartment.

    Returns:
        The compartment and the symbols its size depends on.
    """
    dependencies = symbols.initial_assignment_dependencies or {}
    reached: set[str] = set()
    stack = [compartment]
    while stack:
        symbol = stack.pop()
        if symbol in reached:
            continue
        reached.add(symbol)
        stack.extend(dependencies.get(symbol, ()))
    return reached


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


def _change_values(event: PlanEvent, model: CoreSBMLModel) -> list[list[Any]]:
    """Get the values of a change as the `(id, value)` pairs of the contract.

    A formula at a change is evaluated by the executor of roadrunner with the
    state of the model at the time of the change (`apply_changes` of
    `sbmlsim.simulator.changes`), which the engine does not hand out, so only
    a formula of the time is folded, with the time of the change, which the
    plan carries. Every other formula is refused: a value of a request is a
    number.

    Args:
        event: the assignments at one time.
        model: the model, which the error names.

    Returns:
        A pair of the target and its value per assignment, in their order.

    Raises:
        UnsupportedModelError: for a formula which reads anything but the
            time.
    """
    values: list[list[Any]] = []
    for a in event.assignments:
        if a.value is not None:
            values.append([a.target, a.value])
            continue
        formula = compile_formula(str(a.formula))
        other = sorted(set(formula.symbols) - {"time"})
        if other:
            raise _no_fold(
                a,
                model,
                f"at the time {event.time}",
                f"it reads {other}, which only the state of the model at the time "
                f"of the change has",
            )
        # every symbol is the time, the time of the change
        values.append([a.target, formula.evaluate([event.time] * len(formula.symbols))])
    return values


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
    the initialization, which the engine does otherwise, see the module and
    `_check_concentrations`.

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
            if symbol != "time":
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
