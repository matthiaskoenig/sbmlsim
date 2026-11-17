"""The model of an SBML model in the engine core, see `sbmlsim.model.model_sbml`.

`CoreSBMLModel` compiles the SBML of a model with the bindings of sbmlsim-core
(`sbmlsim.engine.import_core`): the compiler writes it as a WebAssembly module
with a description of the contract `model/2` (its states, parameters and
selections), which the runtime of the engine loads into a `LoadedModel`. The
compiled model of an SBML is kept per process and shared by every model of
that SBML (`COMPILED`), and the runtime, which compiles the engine itself, is
created once (`runtime`): the HCTZ model compiles in 3 ms and loads in 4 ms,
while the runtime takes 51 ms.

The engine is stateless: a simulation is a request of the contract
`simulation/1` which carries the values set before the initialization, the
changes, the output and the settings of the integrator, and a model holds
nothing of the simulations which ran. `initialize` therefore records the
pre-initialization assignments which the executor sends with its request,
`reset` records none and `simulated` records nothing, and `value` is the value
of the model as loaded, as `loaded_value`.

The values of the model are read in the selections of roadrunner, `S` the
amount of a species, `[S]` its concentration, the id of a parameter, a
compartment or a reaction (its rate) and `time`, which every request answers
as its first column and the description does not list.

The settings of the integrator have the names of roadrunner, CVODE in either
engine: the ones a request has a field for are taken (`REQUEST_SETTINGS`), the
other settings of roadrunner are ignored with one report per model
(`IGNORED_SETTINGS`), and `request_settings` is the complete `settings` of a
request, with the absolute tolerance of every state as roadrunner computes its
vector of CVODE, see `sbmlsim.model.tolerances`.

A model the compiler refuses, e.g. one with an algebraic rule, the delay
csymbol, a fast reaction or a package it requires, raises
`UnsupportedModelError` with the message of the compiler, which names the
feature, and so does a model the engine cannot load, so that nothing of the
bindings reaches a caller of `load_model`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, override

from sbmlsim.engine import Engine, UnsupportedModelError, import_core
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.model_sbml import (
    PORTABLE_SETTINGS,
    ROADRUNNER_ONLY_SETTINGS,
    EngineSettings,
    IntegratorSetting,
    SBMLModel,
)
from sbmlsim.model.tolerances import AbsoluteTolerance, state_tolerances
from sbmlsim.units import UnitRegistry

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

logger = logging.getLogger(__name__)

#: the name of the integrator of the engine, CVODE as in roadrunner
INTEGRATOR: Final[str] = "cvode"

#: the time of a simulation, the first column of every request and no entry of
#: the selections of the description
TIME: Final[str] = "time"

#: whether the compiler optimizes the module it writes, the compile option of
#: `COMPILED` with the Jacobian
OPTIMIZE: Final[bool] = True

#: the Jacobian the compiler writes, `auto` the analytic one of a model it can
#: differentiate, the compile option of `COMPILED` with the optimization
JACOBIAN: Final[str] = "auto"

#: the kind of a compartment in the description of a model, `model/2`
COMPARTMENT: Final[str] = "compartment"

#: the settings of roadrunner which a request of the engine has a field for,
#: see `CoreSBMLModel.set_integrator_settings`
REQUEST_SETTINGS: Final[frozenset[str]] = PORTABLE_SETTINGS | {
    "initial_time_step",
    "stiff",
}

#: the settings of roadrunner which a request has no field for, ignored with
#: one report per model: the settings of CVODE which the engine does not set
#: and the ones of the output of roadrunner, which a request states itself
IGNORED_SETTINGS: Final[frozenset[str]] = (
    PORTABLE_SETTINGS | ROADRUNNER_ONLY_SETTINGS
) - REQUEST_SETTINGS

#: the settings of the integrator of a request which a model does not set, the
#: defaults of `simulation/1`, which are the ones of roadrunner
DEFAULT_SETTINGS: Final[dict[str, float | int | bool]] = {
    "relative_tolerance": 1e-6,
    "maximum_num_steps": 20000,
    "maximum_time_step": 0.0,
    "initial_time_step": 0.0,
    "stiff": True,
}

#: the settings of a request which are no settings of roadrunner, left at the
#: choice of the engine: the linear solver and the Jacobian of the integration
ENGINE_SETTINGS: Final[dict[str, str]] = {
    "linear_solver": "auto",
    "jacobian": "auto",
}

#: the models which were compiled and loaded in this process, by the SBML and
#: the compile options, see `compiled_model`; the values are `LoadedModel`s of
#: the bindings, which are not imported statically
COMPILED: dict[tuple[str, bool, str], Any] = {}


@cache
def runtime() -> Any:
    """Get the runtime of the engine of this process, created once.

    The runtime compiles the engine and its math library, 51 ms, and loads
    the models of the process, see `compiled_model`.

    Returns:
        The `Runtime` of the bindings.

    Raises:
        ImportError: if the bindings of sbmlsim-core are not installed.
    """
    return import_core().Runtime()


def compiled_model(
    sbml: str,
    *,
    optimize: bool = OPTIMIZE,
    jacobian: str = JACOBIAN,
    label: str = "model",
) -> Any:
    """Compile an SBML with the engine and load it, once per SBML in this process.

    The compiled model of an SBML and its compile options is kept in
    `COMPILED` and shared by every model of them: it is the module of the
    engine and its description, which a simulation does not change, while the
    values of a simulation are the ones of its request.

    Args:
        sbml: the SBML of the model, level 3, as the engine compiles it, see
            `SBMLModel._take_engine_sbml`.
        optimize: whether the compiler optimizes the module.
        jacobian: the Jacobian the compiler writes.
        label: the name of the model in a message.

    Returns:
        The `LoadedModel` of the bindings.

    Raises:
        UnsupportedModelError: naming the feature, for a model the compiler
            refuses, and for one the engine cannot load, so that nothing of
            the bindings reaches a caller of `load_model`.
        ImportError: if the bindings of sbmlsim-core are not installed.
    """
    key = (sbml, optimize, jacobian)
    loaded = COMPILED.get(key)
    if loaded is None:
        core = import_core()
        try:
            compiled = core.compile_sbml(sbml, optimize=optimize, jacobian=jacobian)
        except core.CompileError as err:
            raise UnsupportedModelError(
                f"The engine 'core' cannot compile the model '{label}': {err}"
            ) from err
        try:
            loaded = runtime().load(compiled.wasm, compiled.description)
        except core.LoadError as err:
            raise UnsupportedModelError(
                f"The engine 'core' cannot load the model '{label}': {err}"
            ) from err
        COMPILED[key] = loaded
    return loaded


class CoreSBMLModel(SBMLModel):
    """The model of an SBML model in the engine core, see the module.

    Attributes:
        loaded: the compiled model of the engine, a `LoadedModel` of the
            bindings, shared by every model of the same SBML, see `COMPILED`.
        description: the description of the compiled model, `model/2`.
        assignments: the pre-initialization assignments which were recorded
            last and which the executor sends with its request, see
            `initialize`.
        absolute_tolerance: the absolute tolerance set last, see
            `set_integrator_settings`.
    """

    engine = Engine.CORE

    def __init__(
        self,
        source: str | Path | Source,
        base_path: Path | None = None,
        changes: dict | None = None,
        sid: str | None = None,
        name: str | None = None,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
        parameters: dict[str, float] | None = None,
    ):
        """Compile the model with the engine, with changes, selections and settings.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`; they are not selected by default. The compiled model
        is shared by every model of the same SBML, see `compiled_model`.

        Raises:
            UnsupportedModelError: for a model the compiler of the engine
                refuses, e.g. one with an algebraic rule.
            ValueError: for a selection the model has not, or an unknown
                setting.
            ImportError: if the bindings of sbmlsim-core are not installed.
        """
        super().__init__(
            source=source,
            base_path=base_path,
            changes=changes,
            sid=sid,
            name=name,
            selections=selections,
            ureg=ureg,
            parameters=parameters,
        )
        self.loaded: Any = compiled_model(self._take_engine_sbml(), label=self.label)
        self.description: dict[str, Any] = self.loaded.description
        #: the states which the integrator integrates, in the order of the
        #: state vector of the engine, see `state_ids`
        self._state_ids: tuple[str, ...] = tuple(
            e["id"] for e in self.description["states"]
        )
        #: the selections of the compiled model, see `has_selection`
        self._selection_ids: frozenset[str] = frozenset(
            s["id"] for s in self.description["selections"]
        )
        #: the compartments of the model, the states and the parameters of the
        #: kind `compartment`, see `_initial_volumes`
        self._compartments: tuple[str, ...] = tuple(
            e["id"]
            for e in (*self.description["states"], *self.description["parameters"])
            if e["kind"] == COMPARTMENT
        )
        #: the values of the model as loaded, read once, see `loaded_value`
        self._loaded_values: dict[str, float] = dict(self.loaded.initial_values())
        self.assignments: tuple[Assignment, ...] = ()
        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        #: the instance of the engine the simulations of this model run on,
        #: made at the first one and kept, see `instance`
        self._instance: Any = None
        #: the executors of the batches of this model by their threads, see
        #: `batch_executor`
        self._executors: dict[int, Any] = {}
        #: the absolute tolerance `_tolerance_values` was computed for
        self._tolerance_of: AbsoluteTolerance | None = None
        #: the absolute tolerance of every state, see `_request_tolerance`
        self._tolerance_values: list[float] | None = None
        #: the targets of `resume_targets` with the selections they were
        #: computed for, `_selections_version`
        self._resume_targets: tuple[int, list[str] | None] | None = None
        #: counts the times the selections of the model were set, which is
        #: what `resume_targets` depends on
        self._selections_version: int = 0
        #: the settings of the integrator of a request which were set, by
        #: their names in roadrunner, see `request_settings`
        self._settings: dict[str, float | int | bool] = {}
        #: the settings of roadrunner, which reports the ignored ones once
        self._engine_settings = EngineSettings(
            "the engine 'core'",
            INTEGRATOR,
            self.label,
            logger,
            taken=REQUEST_SETTINGS,
            ignored=IGNORED_SETTINGS,
        )

        self.set_selections(selections)
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
    ) -> CoreSBMLModel:
        """Create from AbstractModel, see `CoreSBMLModel`."""
        return CoreSBMLModel(
            source=abstract_model.source,
            changes=abstract_model.changes,
            parameters=abstract_model.parameters,
            sid=abstract_model.sid,
            name=abstract_model.name,
            base_path=abstract_model.base_path,
            selections=abstract_model.selections if selections is None else selections,
            ureg=ureg,
            settings=settings,
        )

    @override
    def initialize(self, assignments: Sequence[Assignment]) -> None:
        """Record the values which are set before the initialization of the model.

        The engine is stateless, the values of a simulation are the `preinit`
        of its request: the assignments are recorded for the executor, which
        sends them, and nothing is set on the model, see the module.

        Args:
            assignments: the pre-initialization assignments of a plan.
        """
        self.assignments = tuple(assignments)

    @override
    def reset(self) -> None:
        """Set the model to the state it was loaded in, without a simulation.

        The engine holds no state of a simulation, so only the recorded
        pre-initialization is dropped, see `initialize`.
        """
        self.initialize([])

    @override
    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization.

        Nothing is recorded: a request of the engine carries every value of
        its simulation, so a simulation does not depend on the one before it,
        see the module.
        """

    @override
    def loaded_value(self, selection: str) -> float:
        """Get the value of a parameter, a compartment or a species of the model as loaded.

        Args:
            selection: the id of a parameter, a compartment or a species (its
                amount), or `[S]`, the concentration of the species `S`.

        Returns:
            The value as loaded.

        Raises:
            KeyError: if the selection is none of these.
        """
        return self._loaded_values[selection]

    @override
    def value(self, selection: str) -> float:
        """Get the current value of a selection.

        The engine is stateless, a model holds no values of a simulation: the
        value is the one of the model as loaded, and `time` is 0, the time a
        model is initialized at, see the module.

        Args:
            selection: the selection, e.g. `S`, `[S]`, a parameter, a
                compartment, a reaction or `time`.

        Returns:
            The value of the model as loaded.

        Raises:
            RuntimeError: if the engine has no selection of the name, as
                roadrunner.
        """
        if selection == TIME:
            return 0.0
        try:
            return self._loaded_values[selection]
        except KeyError:
            raise RuntimeError(
                f"The engine 'core' has no selection '{selection}' in the model "
                f"'{self.label}'."
            ) from None

    @override
    def has_selection(self, selection: str) -> bool:
        """Check whether the engine has a selection of a name in the model.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it. `time` is the first
            column of every request and no selection of the description.
        """
        return selection == TIME or selection in self._selection_ids

    @override
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, `default_selections` for `None`.

        Raises:
            ValueError: if the engine has no selection of a name.
        """
        chosen = self.default_selections() if selections is None else list(selections)
        unknown = [s for s in chosen if not self.has_selection(s)]
        if unknown:
            raise ValueError(
                f"The engine 'core' has no selections {unknown} in the model "
                f"'{self.label}'."
            )
        self.selections = chosen
        self.selections_given = selections is not None
        # `resume_targets` reads the selections, so its cache is dropped
        self._selections_version += 1

    @override
    def set_integrator_settings(self, **settings: IntegratorSetting) -> None:
        """Set settings of the integrator, CVODE of the engine.

        The settings have the names of roadrunner; a request of the engine has
        a field for `absolute_tolerance` (a float or an `AbsoluteTolerance`,
        sent as the tolerance of every state, see `request_settings`),
        `relative_tolerance`, `maximum_num_steps`, `maximum_time_step`,
        `initial_time_step` and `stiff` (`REQUEST_SETTINGS`). A setting of
        roadrunner which it has no field for, e.g. `variable_step_size`, which
        the output of a request states instead, is ignored and reported once
        per model (`IGNORED_SETTINGS`).

        Args:
            **settings: the settings by their names in roadrunner.

        Raises:
            ValueError: if neither the engine nor roadrunner has a setting of
                a name, or a value is not valid.
        """
        # every setting is checked before one is set, so a value which is not
        # valid leaves the settings of the model as they were
        for key, value in self._engine_settings.checked(settings).items():
            if isinstance(value, AbsoluteTolerance):
                self.absolute_tolerance = value
            else:
                self._settings[key] = value

    def request_settings(self) -> dict[str, object]:
        """Get the settings of the integrator of a request of the engine.

        The complete `settings` of a request of `simulation/1`: the settings
        which were set (`set_integrator_settings`), the defaults of the
        contract for the ones which were not (`DEFAULT_SETTINGS`, the defaults
        of roadrunner) and the settings which the engine chooses
        (`ENGINE_SETTINGS`).

        The absolute tolerance is the tolerance of every state in the order of
        `state_ids`, `{"per_state": [...]}`, computed as roadrunner computes
        its vector of CVODE, from the initial volumes and the initial values
        of the model as loaded, see `sbmlsim.model.tolerances`: a float
        setting is the tolerance of every kind (`AbsoluteTolerance.of`), as in
        roadrunner, so the engine integrates with its tolerances and not with
        one number, which `{"scalar": value}` of the contract would be.

        Returns:
            The settings of a request.
        """
        return {
            **DEFAULT_SETTINGS,
            **self._settings,
            **ENGINE_SETTINGS,
            "absolute_tolerance": self._request_tolerance(),
        }

    def _request_tolerance(self) -> dict[str, object]:
        """Get the absolute tolerance of a request, see `request_settings`.

        The tolerances are computed once per `absolute_tolerance` of the
        model: everything else they read is the model as loaded, which does
        not change, so only `set_integrator_settings` moves them. Every
        request gets a list of its own, so a caller which changes the values
        of its request does not reach the ones of the next.

        Returns:
            The tolerance of every state in the order of `state_ids`.
        """
        if (
            self._tolerance_values is None
            or self._tolerance_of != self.absolute_tolerance
        ):
            states = state_tolerances(
                self.state_ids(),
                self.symbols,
                self._initial_volumes(),
                self.absolute_tolerance,
                initial_values=self._loaded_values,
            )
            self._tolerance_values = [state.absolute_tolerance for state in states]
            self._tolerance_of = self.absolute_tolerance
        return {"per_state": list(self._tolerance_values)}

    def resume_targets(self) -> list[str] | None:
        """Get the targets a segment which resumes sets the state with, or `None`.

        A segment of the fold of the executor which resumes sets the state of
        the model at its start with a change there, so it integrates its own
        interval instead of the one from the start of the simulation, see
        `_state_at` of `sbmlsim.simulator.executor_core`. The targets are
        every parameter, every compartment and the amount of every species
        which is not the variable of an assignment rule, which follows the
        others, so every species is a target of such a change and the rule
        that a compartment keeps the concentrations of the species in it
        (PEtab v2) keeps none of them, see `apply_changes` of
        `sbmlsim.simulator.changes`. A reaction is an output of the engine
        and no target.

        They are the same for every point of a scan, so they are computed
        once per model and selections, which is what they read.

        Returns:
            The targets, sorted, or `None` if the model has a state which is
            none of them, e.g. a species reference which is not constant:
            such a state would silently keep the value of the model as
            loaded, so the segments of that model start at the start of the
            simulation instead.
        """
        cached = self._resume_targets
        if cached is not None and cached[0] == self._selections_version:
            return cached[1]
        symbols = self.symbols
        entities = symbols.parameters | symbols.compartments | symbols.species
        targets: list[str] | None = sorted(
            t for t in entities - symbols.assignment_rules if self.has_selection(t)
        )
        if targets is not None and not set(self.state_ids()) <= set(targets):
            targets = None
        self._resume_targets = (self._selections_version, targets)
        return targets

    def instance(self) -> Any:
        """Get the instance of the engine the simulations of this model run on.

        The engine holds no state between two requests (the contract
        `simulation/1`, see the module), so one instance is made and kept
        instead of one per simulation: the result is the one of a new
        instance bitwise and making one costs more than a simulation of a
        small model (0.33 ms against 0.25 ms of one simulation of the HCTZ
        model, which is why it is kept). The instance lives as long as the
        model, as its compiled module does.

        Returns:
            The instance, made at the first simulation.
        """
        if self._instance is None:
            self._instance = self.loaded.instance()
        return self._instance

    def batch_executor(self, threads: int) -> Any:
        """Get the executor the batches of `threads` threads of this model run on.

        The executor of the bindings keeps the pool of its threads and the
        instance of every one of them across its calls, where
        `LoadedModel.simulate_many` builds both for every batch, which is
        most of the time of a chunk of a small model (a chunk of 250 points
        of the demo model takes 5.0 ms against 3.1 ms). An executor runs on
        at most the threads it was made with, so one is kept per number of
        threads, and it holds one instance per thread for as long as the
        model does.

        Args:
            threads: the threads of the batch, at least one.

        Returns:
            The executor, made at the first batch of that number of threads.

        Raises:
            ValueError: if the engine refuses an executor of these threads.
        """
        executor = self._executors.get(threads)
        if executor is None:
            executor = self.loaded.batch_executor(threads=threads)
            self._executors[threads] = executor
        return executor

    def _initial_volumes(self) -> dict[str, float]:
        """Get the initial volume of every compartment of the model as loaded.

        The reference volumes of the absolute tolerances of the species of a
        concentration, see `sbmlsim.model.tolerances`. They are the volumes of
        the model as loaded, so the tolerances of a simulation do not depend
        on the pre-initialization of another one.
        """
        return {cid: self._loaded_values[cid] for cid in self._compartments}

    @override
    def integrator_state(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Get the name of the integrator of the model and every one of its settings.

        The settings have the names of roadrunner and are every setting of a
        request, the defaults of the contract for the ones which were not set,
        so that two models of equal settings have one state whatever was set
        on them (the key of a worker of a pool, see
        `sbmlsim.simulator.worker.ModelSpec`) and `set_integrator_state` is
        the inverse of this. The absolute tolerance is the
        `AbsoluteTolerance` of the model and not the tolerances of its states,
        see `request_settings`.

        Returns:
            The name of the integrator and its settings, sorted by name.
        """
        settings: dict[str, Any] = {
            **DEFAULT_SETTINGS,
            **self._settings,
            "absolute_tolerance": self.absolute_tolerance,
        }
        return INTEGRATOR, tuple((key, settings[key]) for key in sorted(settings))

    @override
    def set_integrator_state(
        self, state: tuple[str, tuple[tuple[str, Any], ...]]
    ) -> None:
        """Set the settings of the integrator, see `integrator_state`.

        Args:
            state: the name of the integrator, `cvode`, and its settings.

        Raises:
            ValueError: if the integrator is not the one of the engine, or it
                has no setting of a name.
        """
        integrator, settings = state
        if integrator != INTEGRATOR:
            raise ValueError(
                f"The engine 'core' integrates with '{INTEGRATOR}', not with "
                f"'{integrator}'."
            )
        self.set_integrator_settings(**dict(settings))

    @override
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates.

        The states of the state vector of the engine, in the order of the
        `states` of the description, which a tolerance per state follows: the
        species which reactions or a rate rule change, in the order of the
        SBML, then the other targets of rate rules.
        """
        return list(self._state_ids)
