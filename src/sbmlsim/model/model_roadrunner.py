"""RoadRunner model."""

import logging
import os
import re
import weakref
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import libsbml
import numpy as np
import pandas as pd
import roadrunner
from roadrunner import _roadrunner  # ty: ignore[import-not-found]

from sbmlsim.engine import Engine, JitOptimization
from sbmlsim.memory import release_freed_memory
from sbmlsim.model.event_rows import with_event_markers
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.model_sbml import (
    INITIAL_SUFFIX,
    IntegratorSetting,
    SBMLModel,
    _flatten,
    _is_helper,
)
from sbmlsim.model.symbols import TargetKind
from sbmlsim.model.tolerances import (
    AbsoluteTolerance,
    StateTolerance,
    state_tolerances,
)
from sbmlsim.units import UnitRegistry

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

#: what `RoadRunner.resetAll` resets: every variable to its initial value, with
#: the initial assignments evaluated
RESET_ALL = (
    roadrunner.SelectionRecord.TIME
    | roadrunner.SelectionRecord.RATE
    | roadrunner.SelectionRecord.FLOATING
    | roadrunner.SelectionRecord.BOUNDARY
    | roadrunner.SelectionRecord.COMPARTMENT
    | roadrunner.SelectionRecord.GLOBAL_PARAMETER
    | roadrunner.SelectionRecord.STOICHIOMETRY
)


def reset_all(r: roadrunner.RoadRunner) -> None:
    """Reset a model to its initial values, as `RoadRunner.resetAll`.

    roadrunner exposes the symbols of a model as attributes, so a model with a
    species or parameter named `reset` hides the method `reset`, which
    `resetAll` calls (case 00952 of the SBML Test Suite). The binding is
    called directly.

    The reset queues the events which fire at the time 0 at the integration
    start time of the integrator and of the model, which every `simulate` sets
    to its start: after a simulation whose last integration started later,
    e.g. at the end of a simulation with events, the events at the time 0
    fired at that time plus their delay, i.e. after the end of the next
    simulation (#305). Both are set to 0 first.

    Args:
        r: the roadrunner instance.
    """
    r.getIntegrator().setIntegrationStartTime(0.0)
    r.model.setIntegrationStartTime(0.0)
    _roadrunner.RoadRunner_reset(r, RESET_ALL)


def reset_values(r: roadrunner.RoadRunner) -> None:
    """Reset a model to its initial values without the events at the time 0.

    `reset_all` resets the model and then fires the events whose trigger is
    false before the time 0 and true at it, with the values of the reset; the
    values of a pre-initialization, which are set after the reset, are not
    seen by them (#439). This is the reset of the model alone, which leaves
    the time of the model before the time 0, so that `fire_initial_events`
    evaluates the triggers with the values set after it.

    Args:
        r: the roadrunner instance.
    """
    model = r.model
    r.getIntegrator().setIntegrationStartTime(0.0)
    model.setIntegrationStartTime(0.0)
    _roadrunner.ExecutableModel_reset(model, RESET_ALL)


def fire_initial_events(r: roadrunner.RoadRunner) -> None:
    """Fire the events at the time 0 with the current values, as `reset_all` does.

    The time of the model is set before the time 0, where roadrunner takes
    the trigger of an event as its initial value, and the integrator is
    restarted at the time 0, which evaluates the triggers there: an event
    whose trigger is false before and true at the time 0 fires (an event
    with a delay is queued), with the current values, also the ones a
    pre-initialization set (#439). The integration start times are 0 as in
    `reset_all`.

    Args:
        r: the roadrunner instance.
    """
    integrator, model = r.getIntegrator(), r.model
    integrator.setIntegrationStartTime(0.0)
    model.setIntegrationStartTime(0.0)
    _roadrunner.ExecutableModel_setTime(model, -np.inf)
    _roadrunner.Integrator_restart(integrator, 0.0)


#: the level of a model loaded without one, unless `JIT_OPTIMIZATION_VARIABLE`
#: sets another
DEFAULT_JIT_OPTIMIZATION = JitOptimization.O2

#: the environment variable which sets the level of a model loaded without one,
#: e.g. `SBMLSIM_JIT_OPTIMIZATION=O0`
JIT_OPTIMIZATION_VARIABLE = "SBMLSIM_JIT_OPTIMIZATION"

#: the optimization level of roadrunner of every level
_LLJIT_LEVELS: dict[JitOptimization, int] = {
    JitOptimization.O0: roadrunner.LoadSBMLOptions.NONE,
    JitOptimization.O1: roadrunner.LoadSBMLOptions.LESS,
    JitOptimization.O2: roadrunner.LoadSBMLOptions.DEFAULT,
    JitOptimization.O3: roadrunner.LoadSBMLOptions.AGGRESSIVE,
}


def resolve_jit_optimization(
    level: JitOptimization | str | None = None,
) -> JitOptimization:
    """Get the optimization level of the compilation of a model.

    Args:
        level: the level, e.g. `"O0"`. Without one, the level of
            `JIT_OPTIMIZATION_VARIABLE` or else `DEFAULT_JIT_OPTIMIZATION`.

    Returns:
        The level.

    Raises:
        ValueError: if the level is none of `JitOptimization`.
    """
    source = "jit_optimization"
    if level is None:
        level = os.environ.get(JIT_OPTIMIZATION_VARIABLE)
        source = JIT_OPTIMIZATION_VARIABLE
        if not level:
            return DEFAULT_JIT_OPTIMIZATION
    try:
        return JitOptimization(level)
    except ValueError:
        levels = ", ".join(repr(str(member)) for member in JitOptimization)
        raise ValueError(
            f"{source} {level!r} is no optimization level of the JIT, one of {levels}."
        ) from None


def load_roadrunner(
    source: str | Path, optimization: JitOptimization | str | None = None
) -> roadrunner.RoadRunner:
    """Load an instance of roadrunner.

    The model is compiled with LLJIT at the given optimization level, see
    `JitOptimization`, which compiles faster than MCJIT. roadrunner keeps the
    compiled models of a process in its cache, keyed by the SBML, the JIT and
    its optimization level (roadrunner #1420), and a model taken from the
    cache integrates as a compiled one (roadrunner #1383, sbmlsim #313), so a
    model loaded again is not compiled again. The memory the compilation
    freed is returned to the system (`sbmlsim.memory.release_freed_memory`).

    A model which references models of other files (`externalModelDefinition`)
    is flattened first, its references resolved relative to its file (to the
    current directory for SBML given as a string), as `SBMLModel` simulates
    it.

    Args:
        source: the path of an SBML file or the SBML itself.
        optimization: the optimization level of the compilation, see
            `resolve_jit_optimization` for a level which is not given.

    Returns:
        The instance.
    """
    options = roadrunner.LoadSBMLOptions()
    options.setLLVMBackend(roadrunner.LoadSBMLOptions.LLJIT)
    level = resolve_jit_optimization(optimization)
    options.setLLJitOptimizationLevel(_LLJIT_LEVELS[level])
    instance = roadrunner.RoadRunner(_loadable(source), options)
    # the compilation leaves the memory of its allocations free in the heap
    release_freed_memory()
    return instance


def _sbml_text(data: bytes) -> str:
    """Decode an SBML file in the encoding of its XML declaration, UTF-8 without one.

    A byte order mark of UTF-8 is dropped.
    """
    match = _ENCODING.match(data)
    encoding = match.group(1).decode("ascii") if match else "utf-8"
    if encoding.lower().replace("_", "-") in ("utf-8", "utf8"):
        encoding = "utf-8-sig"
    return data.decode(encoding)


#: the encoding of an XML declaration
_ENCODING = re.compile(
    rb'^(?:\xef\xbb\xbf)?\s*<\?xml[^>]*?encoding\s*=\s*["\']([A-Za-z0-9._-]+)["\']'
)


def _loadable(source: str | Path) -> str:
    """Get the SBML of a source which roadrunner loads, see `load_roadrunner`.

    A path is read (in the encoding of its XML declaration), a string which
    starts with `<` (after a byte order mark and whitespace) is the SBML. A
    hierarchical model with external model definitions is flattened, its
    references resolved relative to its file (to the current directory for
    SBML given as a string), as `SBMLModel` simulates it.

    Args:
        source: the path of an SBML file or the SBML itself.

    Returns:
        The SBML.
    """
    path: Path | None = None
    text = str(source)
    if isinstance(source, Path) or not text.lstrip("\ufeff \t\r\n").startswith("<"):
        path = Path(source)
        text = _sbml_text(path.read_bytes())
    if "externalModelDefinition" in text:
        text = _flatten(text, path=path)
    return text


logger = logging.getLogger(__name__)


class RoadrunnerSBMLModel(SBMLModel):
    """The model of an SBML model in roadrunner, see `SBMLModel`."""

    engine = Engine.ROADRUNNER

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
        jit_optimization: JitOptimization | str | None = None,
    ):
        """Load the model into roadrunner, with changes, selections and settings.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`; they are not selected by default. The model is
        compiled at the optimization level `jit_optimization`, see
        `resolve_jit_optimization` for a level which is not given.
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
        sbml = self._take_engine_sbml()
        #: entity with an initial assignment -> the parameter whose
        #: assignment rule is the math of the initial assignment, see
        #: `initialize`
        self.initial_helpers: dict[str, str] = {}
        if self.symbols.initial_assignments:
            sbml, self.initial_helpers = _with_initial_helpers(sbml)
        #: the parameters which mark the events of the model, which the
        #: executor selects as the last columns of its integrations, see
        #: `sbmlsim.model.event_rows`
        self.event_markers: tuple[str, ...] = ()
        if self.symbols.events:
            sbml, self.event_markers = with_event_markers(sbml)

        #: the optimization level the model is compiled with, which a worker
        #: compiles it with as well, see `sbmlsim.simulator.worker.ModelSpec`
        self.jit_optimization: JitOptimization = resolve_jit_optimization(
            jit_optimization
        )

        # load model
        self.r: roadrunner.RoadRunner | None = (
            load_roadrunner(sbml, self.jit_optimization)
            if self.hierarchical
            or self.initial_helpers
            or self.event_markers
            or self.parameters
            or self.source.content is not None
            else self.load_roadrunner_model(
                source=self.source, optimization=self.jit_optimization
            )
        )

        #: whether the instance was reset or loaded and not simulated since,
        #: see `initialize`
        self._reset_pending: bool = True
        #: the values of the instance of roadrunner as loaded (a weak
        #: reference), which a reset restores, see `_loaded_state`
        self._state_of: tuple[weakref.ref[roadrunner.RoadRunner], _State] | None = None
        #: whether the model has an event with a delay, which can still be
        #: pending when a simulation ends, see `discard_pending_events`
        self._delayed_events: bool = any(e.delayed for e in self.symbols.events)
        #: whether the model has an event with a delay which can fire at the
        #: time 0, which every initialization queues, see `initialize`
        self._delayed_initial_events: bool = any(
            e.delayed and not e.initial_value for e in self.symbols.events
        )
        #: whether events at the time 0 were queued since the last simulation
        #: or regeneration, the load counts as one, see `initialize`
        self._initial_events_queued: bool = True
        if self.r is not None and self.symbols.events:
            # the load fired the events at the time 0 with the values as
            # loaded; the values as loaded are the ones before them, which an
            # initialization fires with its own values (#439), and the model
            # is left as roadrunner loaded it
            reset_values(self.r)
            self._loaded_state()
            fire_initial_events(self.r)
        else:
            self._loaded_state()
        #: the time of roadrunner the last simulation since the load or the
        #: last regeneration reached, `inf` before any, see
        #: `discard_pending_events`
        self._reached: float = np.inf

        # set selections
        self.selections = self.set_timecourse_selections(
            self.r, selections=self.selections, exclude=set(self.parameters)
        )

        #: the absolute tolerance set last and the tolerances of the states,
        #: see `set_integrator_settings`
        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        self._state_tolerances: list[StateTolerance] = []
        #: what the tolerances of the states were set for: the instance of
        #: roadrunner (a weak reference, which keeps no replaced instance
        #: alive) and the vector of CVODE, see `_has_absolute_tolerance`
        self._tolerances_of: (
            tuple[weakref.ref[roadrunner.RoadRunner], list[float]] | None
        ) = None
        #: the initial volumes of the compartments of the instance of
        #: roadrunner as loaded, see `_loaded_volumes`
        self._volumes_of: (
            tuple[weakref.ref[roadrunner.RoadRunner], dict[str, float]] | None
        ) = None
        #: compartments whose volume was reported as raised to the floor
        self._raised_reported: set[str] = set()
        #: the instance of roadrunner (a weak reference) and the ids of its
        #: SBML, see `sbml_ids`
        self._sbml_ids_of: tuple[weakref.ref[roadrunner.RoadRunner], SBMLIds] | None = (
            None
        )
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )

    @property
    def r_loaded(self) -> roadrunner.RoadRunner:
        """Get the roadrunner instance of the model.

        Raises:
            ValueError: if no model is loaded.
        """
        if self.r is None:
            raise ValueError(f"The model '{self.sid}' is not loaded.")
        return self.r

    @override
    def initialize(self, assignments: Sequence[Assignment]) -> None:
        """Initialize the model with values before its initialization.

        PEtab v2 sets the values of the parameter table and of the first
        conditions on the model before it is initialized, so an initial
        assignment follows a changed parameter, and a target which is set
        replaces its own initial assignment. roadrunner reinitializes the
        model when an initial value is set with `init(...)`, which costs a
        compilation of the model per value. The model is therefore
        initialized as it was loaded, the values are set as the current
        values, and the initial assignments which read a changed entity are
        evaluated again, in their order, from the assignment rule of their
        helper parameter, which roadrunner evaluates on the current values.

        roadrunner's reset fires the events at the time 0 with the values of
        the reset, before the values of a pre-initialization are set, so an
        event whose trigger they make true would not fire and one whose
        trigger they make false would (#439). The model is therefore reset
        without these events (`reset_values`), and they fire after the values
        are set (`fire_initial_events`): an event whose trigger is false
        before the time 0 and true at it fires once per initialization, with
        the values of the pre-initialization, as in libroadrunner when the
        values are set before the reset, and the output at the start is the
        state after it. A model which was not simulated since its last reset
        is not reset again, the values as loaded, the ones before the events
        at the time 0, are set back instead (the values of an earlier
        initialization and its events included), see `_reset`; `simulated`
        records a simulation. An event with a delay is queued when it fires
        at the time 0 and a reset keeps the queue, so a model with such an
        event is generated again before it is initialized again without a
        simulation in between, the load included, which empties the queue.

        A compartment keeps the concentration of the species whose initial
        value is a concentration, as at the initialization of SBML, and of a
        species which is set as a concentration, and the amount of every
        other species; a species of an assignment rule follows its rule. So
        does a compartment of an assignment rule which reads a changed entity
        (#344): roadrunner evaluates the rule with the new values and keeps
        the amounts, the concentration of such a species is set again after
        the entities its compartment depends on, before the initial
        assignments which read it.

        Args:
            assignments: the pre-initialization assignments of a plan. A
                formula reads the parameters after the values are set.
        """
        if self._delayed_initial_events and self._initial_events_queued:
            self._regenerate()
        self._reset()
        r = self.r_loaded
        symbols = self.symbols
        entities: set[str] = set()
        concentrations: set[str] = set()
        amounts: set[str] = set()
        for a in assignments:
            entity = symbols.entity(a.target)
            entities.add(entity)
            if a.kind is TargetKind.SPECIES_CONCENTRATION:
                concentrations.add(entity)
            elif a.kind is TargetKind.SPECIES_AMOUNT:
                amounts.add(entity)
        # the values first, then the formulas, which read the values and are
        # evaluated before any of them is set
        values = [a for a in assignments if a.value is not None]
        for a in values:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, float(a.value), concentrations, amounts)  # ty: ignore[invalid-argument-type]
        for a in values:
            if a.kind is not TargetKind.COMPARTMENT:
                self._set(a.target, float(a.value))  # ty: ignore[invalid-argument-type]
        # the package of the simulator imports the models
        from sbmlsim.simulator.formula import compile_formula

        evaluated: list[tuple[Assignment, float]] = []
        for a in assignments:
            if a.formula is not None:
                formula = compile_formula(a.formula)
                evaluated.append(
                    (a, formula.evaluate([r.getValue(s) for s in formula.symbols]))
                )
        for a, value in evaluated:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, value, concentrations, amounts)
            else:
                self._set(a.target, value)
        # the concentrations which are set, in the size of their compartment
        # after the initialization
        set_concentrations = {
            symbols.entity(a.target): float(a.value)  # ty: ignore[invalid-argument-type]
            for a in values
            if a.kind is TargetKind.SPECIES_CONCENTRATION
        } | {
            symbols.entity(a.target): value
            for a, value in evaluated
            if a.kind is TargetKind.SPECIES_CONCENTRATION
        }

        dependencies = symbols.initial_assignment_dependencies or {}
        changed = set(entities)
        for entity in symbols.initial_assignment_order:
            if not (dependencies[entity] & changed):
                continue
            if entity in set_concentrations:
                value = set_concentrations[entity]
            elif entity in entities:
                continue
            elif entity in self.initial_helpers:
                value = float(r.getValue(self.initial_helpers[entity]))
            elif entity in symbols.initial_concentration_values:
                value = symbols.initial_concentration_values[entity]
            else:
                # an initial amount, which roadrunner keeps
                continue
            if entity in symbols.compartments:
                self._set_compartment(entity, value, concentrations, amounts)
            elif entity in symbols.species and entity not in symbols.only_substance:
                self._set(f"[{entity}]", value)
            else:
                self._set(entity, value)
            changed.add(entity)
        # also without events, which sets the time to 0 and restarts the
        # integrator, as roadrunner's reset does
        fire_initial_events(r)
        self._initial_events_queued = True

    def _reset(self) -> None:
        """Set the model to the state it was loaded in, bitwise, see `initialize`.

        After a simulation roadrunner resets the model and the values which
        its reset misses are set back (`_restore_loaded_state`); the
        parameters and the compartments as loaded are set before the reset as
        well, which evaluates the initial values with them (#318). The reset
        fires no events, the values as loaded are the ones before the events
        at the time 0, which `initialize` fires (#439). A model which was not
        simulated since its last reset, i.e. initialized at most, is not
        reset again, every value which differs from the model as loaded is
        set back instead (#322):
        the amount of a species, which roadrunner reads and writes through
        the current volume of its compartment, is the amount as loaded, also
        when an initialization changed the volume and the species.
        """
        r = self.r_loaded
        # the values as loaded of another instance of roadrunner are read
        # before it is reset, see `_loaded_state`
        self._loaded_state()
        if not self._reset_pending:
            # the reset evaluates the initial values with the current
            # parameters and compartments, which are the ones as loaded first
            # (#318)
            self._restore_loaded_state(species=False)
            reset_values(r)
            self._reset_pending = True
        self._restore_loaded_state()

    @override
    def reset(self) -> None:
        """Set the model to the state it was loaded in, bitwise, without a simulation.

        A simulation leaves the model with the values of its
        pre-initialization and its state at its end until the next
        initialization, see `initialize`; a reader of the model, e.g. the
        references of a design, resets it after it read what it needs, so
        that nothing which reads the model later sees them (#315). Every
        parameter, compartment and species amount which is not the variable
        of an assignment rule is the one of the model as loaded afterwards,
        see `_reset`, the values before the events at the time 0, which the
        next initialization fires; roadrunner resets the model once per
        simulation, see `initialize`.
        """
        self._reset()

    @override
    def loaded_value(self, selection: str) -> float:
        """Get the value of a parameter, a compartment or a species of the model as loaded.

        The current value of the model depends on the simulations before it,
        see `reset`, the value as loaded does not (#315).

        Args:
            selection: the id of a parameter, a compartment or a species (its
                amount), or `[S]`, the concentration of the species `S`.

        Returns:
            The value as loaded.

        Raises:
            KeyError: if the selection is none of these.
        """
        state = self._loaded_state()
        index = state.index
        if selection.startswith("[") and selection.endswith("]"):
            k = index.get(selection[1:-1], -1)
            if k < state.first_species:
                raise KeyError(selection)
            return float(state.concentrations[k - state.first_species])
        if selection not in index:
            raise KeyError(selection)
        return float(state.values[index[selection]])

    def _loaded_state(self) -> _State:
        """Get the values of the model as loaded, read once per instance of roadrunner.

        They are read when the model is loaded, before an initialization
        changes a value. sbmlsim never replaces its instance of roadrunner
        (`regenerateModel` keeps it); an instance which replaces `r` from
        outside is read at its first reset, before it is reset, so it has to
        be in the state it was loaded in, see `_restore_loaded_state`.
        """
        r = self.r_loaded
        if self._state_of is None or self._state_of[0]() is not r:
            rules = set(r.getAssignmentRuleIds())
            ids = _state_ids(r)
            self._state_of = (
                weakref.ref(r),
                _State(
                    ids=ids,
                    values=_state_values(r),
                    settable=np.array([sid not in rules for sid in ids], dtype=bool),
                    first_species=len(r.model.getGlobalParameterIds())
                    + len(r.model.getCompartmentIds()),
                    concentrations=np.concatenate(
                        [
                            np.asarray(
                                r.model.getBoundarySpeciesConcentrations(), dtype=float
                            ),
                            np.asarray(
                                r.model.getFloatingSpeciesConcentrations(), dtype=float
                            ),
                        ]
                    ),
                ),
            )
        return self._state_of[1]

    def _restore_loaded_state(self, species: bool = True) -> None:
        """Set the values back which a reset of roadrunner misses in the last bits.

        roadrunner's resetAll evaluates the initial values of a model with the
        current values of the parameters, i.e. with the values the simulation
        before set, e.g. the initial volume of a compartment whose initial
        assignment reads an assignment rule of the body weight, and the amount
        of a concentration species in it differs from the amount as loaded in
        the last bits. A simulation would then depend on the simulation before
        it on the same model, a point of a scan on the points of its chunk,
        and a serial run would differ from a pool (#303). Every parameter,
        compartment and species which is not the variable of an assignment
        rule gets its value of the model as loaded where it differs, the
        parameters and the compartments first and the amounts of the species
        last, which a compartment does not change.

        `_reset` restores the parameters and the compartments before the
        reset as well (`species=False`), so that the reset evaluates the
        initial values as at the load: roadrunner keeps the amount of a
        concentration species and reads and writes it through the current
        volume of its compartment (`(amount / volume) * volume`), so an
        amount which a reset computed from the volume of the simulation
        before can read as the amount as loaded under the volume as loaded,
        where the restore after the reset does not see it, and differ from it
        in the last bits under the volume of the next simulation, which also
        a set of the amount cannot avoid (#318).

        Args:
            species: restore the amounts of the species as well.
        """
        r = self.r_loaded
        state = self._loaded_state()
        current = _state_values(r) if species else _parameter_values(r)
        for k in _changed(current, state):
            r.setValue(state.ids[k], float(state.values[k]))

    @override
    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization.

        The next `initialize` resets the model, see there, and the time the
        simulation reached is kept, see `discard_pending_events`.
        """
        self._reset_pending = False
        self._initial_events_queued = False
        self._reached = float(self.r_loaded.model.getTime())

    def discard_pending_events(self, horizon: float) -> None:
        """Discard the events of the simulation before which are still pending.

        roadrunner's reset keeps the queue of the events of a model: an event
        with a delay which the simulation before triggered and which had not
        fired when it ended fires in the next simulation at its time, if that
        simulation reaches it (#305), also after an initialization without a
        simulation in between, e.g. the references of a design. Such an event
        is after the time the last simulation reached, so only a simulation
        which integrates beyond it can see it; for a model with an event with
        a delay the model is then generated again, which empties the queue
        (it costs milliseconds for a small model and tenths of a second for a
        large one). Every point of a scan reaches the same end, so a scan
        regenerates no model, unless it integrates to a steady state.

        The selections and the absolute tolerances of sbmlsim are set again
        after it. The next initialization resets the new model, which queues
        its events at t0.

        Args:
            horizon: the time of roadrunner the next simulation integrates to,
                `inf` for an integration to a steady state.
        """
        if not self._delayed_events or horizon <= self._reached:
            return
        self._regenerate()

    def _regenerate(self) -> None:
        """Generate the model again, which empties the queue of its events.

        See `discard_pending_events` and `initialize`.
        """
        r = self.r_loaded
        selections = list(r.timeCourseSelections)
        try:
            r.regenerateModel(True)
        finally:
            # also after a failed regeneration, which the next simulation
            # repeats
            r.timeCourseSelections = selections
            self._set_absolute_tolerance(self.absolute_tolerance)
        self._reset_pending = False
        self._initial_events_queued = False
        self._reached = np.inf

    def _set(self, selection: str, value: float) -> None:
        """Set a value in an initialization.

        The next initialization sets every value back which differs from the
        model as loaded, see `_reset`, so no value needs to be recorded: a
        value which roadrunner refuses to set did not change, and an amount
        which an initialization read under a changed volume is not set back
        under another one (#300, #322).
        """
        self.r_loaded.setValue(selection, value)

    def _set_compartment(
        self,
        compartment: str,
        value: float,
        concentrations: set[str],
        amounts: set[str],
    ) -> None:
        """Set a compartment before the initialization.

        Args:
            compartment: the compartment.
            value: its size.
            concentrations: species which are set as concentrations.
            amounts: species which are set as amounts.
        """
        r = self.r_loaded
        symbols = self.symbols
        # a species of an assignment rule follows its rule, roadrunner refuses
        # to set it (#300)
        kept = {
            s: r.getValue(f"[{s}]")
            for s, c in symbols.species_compartment.items()
            if c == compartment
            and s not in amounts
            and s not in symbols.assignment_rules
            and (s in symbols.initial_concentration or s in concentrations)
        }
        self._set(compartment, value)
        for species, concentration in kept.items():
            r.setValue(f"[{species}]", concentration)

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
        jit_optimization: JitOptimization | str | None = None,
    ):
        """Create from AbstractModel, see `RoadrunnerSBMLModel`."""
        logger.debug("RoadrunnerSBMLModel from AbstractModel")
        return RoadrunnerSBMLModel(
            source=abstract_model.source,
            changes=abstract_model.changes,
            parameters=abstract_model.parameters,
            sid=abstract_model.sid,
            name=abstract_model.name,
            base_path=abstract_model.base_path,
            selections=abstract_model.selections if selections is None else selections,
            ureg=ureg,
            settings=settings,
            jit_optimization=jit_optimization,
        )

    @classmethod
    def load_roadrunner_model(
        cls,
        source: Source,
        optimization: JitOptimization | str | None = None,
    ) -> roadrunner.RoadRunner:
        """Load model from given source.

        :param source: path to SBML model or SBML string
        :param optimization: the optimization level of the compilation, see
            `load_roadrunner`
        :return: roadrunner instance
        """
        if isinstance(source, (str, Path)):
            source = Source.from_source(source=source)

        # load model
        if source.path is not None:
            r = load_roadrunner(source.path, optimization)
        elif source.is_content():
            r = load_roadrunner(str(source.content), optimization)

        return r

    @classmethod
    def set_timecourse_selections(
        cls,
        r: roadrunner.RoadRunner,
        selections: list[str] | None = None,
        exclude: Collection[str] = (),
    ) -> list[str]:
        """Set the selections of the simulations.

        Without selections every entity of the model is selected, except a
        compartment without a size (`NaN`), e.g. a membrane whose area the
        model does not use, which has no value to record.

        Args:
            r: the roadrunner instance.
            selections: the selections, every entity of the model without.
            exclude: ids which are not selected by default, e.g. the
                parameters a problem added to the model.

        Returns:
            The selections.
        """
        if selections is None:
            r_model: roadrunner.ExecutableModel = r.model
            compartments = [
                cid
                for cid, size in zip(
                    r_model.getCompartmentIds(),
                    r_model.getCompartmentVolumes(),
                    strict=True,
                )
                if not np.isnan(size)
            ]

            r.timeCourseSelections = [
                "time",
                *r_model.getFloatingSpeciesIds(),
                *r_model.getBoundarySpeciesIds(),
                *[
                    pid
                    for pid in r_model.getGlobalParameterIds()
                    if not _is_helper(pid) and pid not in exclude
                ],
                *r_model.getReactionIds(),
                *compartments,
            ]
            r.timeCourseSelections += [
                f"[{key}]"
                for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
                )
            ]
        else:
            r.timeCourseSelections = selections
        return list(r.timeCourseSelections)

    @override
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, every entity of the model for `None`,
                see `set_timecourse_selections`; the parameters added to the
                model are not selected by default.

        Raises:
            RuntimeError: if roadrunner has no selection of a name.
        """
        self.selections = self.set_timecourse_selections(
            self.r_loaded,
            selections=None if selections is None else list(selections),
            exclude=set(self.parameters),
        )
        self.selections_given = selections is not None

    @override
    def value(self, selection: str) -> float:
        """Get the current value of a selection of roadrunner.

        Args:
            selection: the selection, e.g. `S`, `[S]`, a parameter or a
                reaction.

        Returns:
            The value in the current state of the model.

        Raises:
            RuntimeError: if roadrunner has no selection of the name.
        """
        return float(self.r_loaded.getValue(selection))

    @override
    def has_selection(self, selection: str) -> bool:
        """Check whether roadrunner has a selection of a name in the model.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it.
        """
        try:
            self.r_loaded.getValue(selection)
        except RuntimeError:
            return False
        return True

    @override
    def set_integrator_settings(self, **kwargs: IntegratorSetting) -> None:
        """Set settings of the integrator.

        Every setting of the integrator of roadrunner is passed on, for CVODE
        e.g. `relative_tolerance`, `stiff`, `variable_step_size`,
        `initial_time_step`, `minimum_time_step`, `maximum_time_step` and
        `maximum_num_steps`. `absolute_tolerance`, a float or an
        `AbsoluteTolerance`, is set as one tolerance per state, see
        `sbmlsim.model.tolerances`.

        A setting the integrator has already is not set again, so that a
        simulator which applies its settings before every simulation costs
        nothing: the tolerances of the states are computed again only for
        another tolerance, another instance of roadrunner or a vector of
        CVODE which was set elsewhere. They are computed from the initial
        volumes of the compartments of the model as loaded, so a simulation
        does not depend on the pre-initialization of the one before it.

        Args:
            **kwargs: the settings by their names in roadrunner.

        Raises:
            ValueError: if the integrator has no setting of a name, or an
                override of the absolute tolerance is not a state.
        """
        integrator: roadrunner.Integrator = self.r_loaded.getIntegrator()
        names = _setting_names(integrator)
        unknown = sorted(set(kwargs) - names)
        if unknown:
            raise ValueError(
                f"The integrator '{integrator.getName()}' has no settings "
                f"{unknown}, its settings are {sorted(names)}."
            )
        for key, value in kwargs.items():
            if key == "absolute_tolerance":
                if isinstance(value, bool):
                    raise ValueError("The absolute tolerance is a number.")
                tolerance = AbsoluteTolerance.of(value)
                if not self._has_absolute_tolerance(tolerance, integrator):
                    self._set_absolute_tolerance(tolerance)
            elif integrator.getValue(key) != value:
                integrator.setValue(key, value)
                logger.debug("Integrator setting: '%s = %s'", key, value)

    @override
    def integrator_state(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Get the name of the integrator of the model and every one of its settings.

        The absolute tolerance is the `AbsoluteTolerance` of the model, from
        which `set_integrator_settings` computes the tolerance of every
        state, and not the vector of the integrator.

        Returns:
            The name of the integrator and its settings, sorted by name.
        """
        integrator: roadrunner.Integrator = self.r_loaded.getIntegrator()
        settings: list[tuple[str, Any]] = []
        for name in sorted(str(name) for name in integrator.getSettings()):
            if name == "absolute_tolerance":
                settings.append((name, self.absolute_tolerance))
            else:
                settings.append((name, integrator.getValue(name)))
        return str(integrator.getName()), tuple(settings)

    @override
    def set_integrator_state(
        self, state: tuple[str, tuple[tuple[str, Any], ...]]
    ) -> None:
        """Set the integrator of the model and its settings, see `integrator_state`.

        Args:
            state: the name of the integrator, e.g. `cvode`, and its settings.

        Raises:
            ValueError: if the integrator has no setting of a name.
        """
        integrator, settings = state
        r = self.r_loaded
        if r.getIntegrator().getName() != integrator:
            r.setIntegrator(integrator)
        self.set_integrator_settings(**dict(settings))

    def _has_absolute_tolerance(
        self, tolerance: AbsoluteTolerance, integrator: roadrunner.Integrator
    ) -> bool:
        """Check whether CVODE integrates with the tolerances of a tolerance.

        The tolerances of the states depend on the instance of roadrunner,
        from whose initial volumes as loaded they are computed, see
        `_loaded_volumes`; the vector of CVODE is compared as well, it may
        have been set on the integrator directly.

        Args:
            tolerance: the tolerance.
            integrator: the integrator of the model.
        """
        if tolerance != self.absolute_tolerance or self._tolerances_of is None:
            return False
        instance, vector = self._tolerances_of
        return (
            instance() is self.r_loaded
            and _floats(integrator.getAbsoluteToleranceVector()) == vector
        )

    def _loaded_volumes(self) -> dict[str, float]:
        """Get the initial volumes of the compartments of the model as loaded.

        roadrunner answers the initial volume of a compartment with an initial
        assignment or an assignment rule from the current values of the
        model, i.e. from the pre-initialization of the simulation which ran
        last. The volumes are therefore read once per instance of roadrunner,
        when the model sets its first tolerances, which it does when it is
        loaded, and kept: the tolerances of a simulation do not depend on the
        simulations before it, and a worker of a pool, which loads the model,
        has the same tolerances.

        Returns:
            The initial volume by the id of the compartment.
        """
        r = self.r_loaded
        if self._volumes_of is None or self._volumes_of[0]() is not r:
            self._volumes_of = (
                weakref.ref(r),
                dict(
                    zip(
                        r.model.getCompartmentIds(),
                        _floats(r.model.getCompartmentInitVolumes()),
                        strict=True,
                    )
                ),
            )
        return self._volumes_of[1]

    def _loaded_values(self) -> dict[str, float]:
        """Get the values of the model as loaded by id, the amounts of the species."""
        state = self._loaded_state()
        return dict(zip(state.ids, _floats(state.values), strict=True))

    def sbml_ids(self) -> SBMLIds:
        """Get the constant parameters and the species of the SBML roadrunner simulates.

        They are read from the SBML of the instance of roadrunner
        (`getSBML()`, i.e. with the added parameters and the helpers of the
        initial assignments) once per instance and kept: serializing and
        parsing the SBML again on every call took 4.8 s of the 8.6 s of the
        initialization of the experiments of the acetaminophen model of
        pkdb_models (403 kB of SBML), which call `sampling.parameters_of`
        for 133 simulations (#332). An instance which replaces `r` is read
        again. sbmlsim never changes the structure of its instance
        (`regenerateModel` keeps the SBML); an edit of the instance itself,
        e.g. `r.addParameter(...)` or `r.removeSpecies(...)`, is not tracked,
        and the ids stay those of the instance as first read until `r` is
        replaced.

        Returns:
            The ids, in the order of the SBML.
        """
        r = self.r_loaded
        if self._sbml_ids_of is None or self._sbml_ids_of[0]() is not r:
            doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(r.getSBML())
            sbml_model: libsbml.Model = doc.getModel()
            self._sbml_ids_of = (
                weakref.ref(r),
                SBMLIds(
                    constant_parameters=tuple(
                        p.getId()
                        for p in sbml_model.getListOfParameters()
                        if p.getConstant()
                    ),
                    species=tuple(s.getId() for s in sbml_model.getListOfSpecies()),
                ),
            )
        return self._sbml_ids_of[1]

    @override
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates.

        The states of roadrunner's state vector, the targets of rate rules
        first, then the independent floating species, in the order of
        `getRatesOfChange` and of the tolerance vector of CVODE. They are
        counted by the rates of change, not by the tolerance vector, which is
        empty for another integrator than CVODE.
        """
        r = self.r_loaded
        n = len(r.getRatesOfChange())
        return [r.model.getStateVectorId(k) for k in range(n)]

    def _set_absolute_tolerance(self, tolerance: AbsoluteTolerance) -> None:
        """Set the absolute tolerance of every state, see `tolerances`.

        roadrunner turns a single value into a vector by its own scaling, so
        the tolerances are set as the vector of CVODE, in the order of
        `state_ids`, in one call. The reference volumes are
        the initial volumes of the model as loaded, see `_loaded_volumes`,
        and the initial values which scale the tolerances of `relative` are
        the values of the model as loaded, see `_loaded_state`, so the
        tolerances do not depend on a simulation before.
        """
        r = self.r_loaded
        volumes = self._loaded_volumes()
        states = state_tolerances(
            self.state_ids(),
            self.symbols,
            volumes,
            tolerance,
            initial_values=self._loaded_values(),
        )
        for state in states:
            if (
                state.volume_raised
                and state.compartment is not None
                and state.compartment not in self._raised_reported
            ):
                self._raised_reported.add(state.compartment)
                logger.warning(
                    "The compartment '%s' of the model '%s' has the initial volume "
                    "%s; the absolute tolerances of its species use the volume %s.",
                    state.compartment,
                    self.sid or r.model.getModelName(),
                    volumes.get(state.compartment),
                    state.volume,
                )
        integrator: roadrunner.Integrator = r.getIntegrator()
        integrator.setValue(
            "absolute_tolerance", [state.absolute_tolerance for state in states]
        )
        self.absolute_tolerance = tolerance
        self._state_tolerances = states
        self._tolerances_of = (
            weakref.ref(r),
            _floats(integrator.getAbsoluteToleranceVector()),
        )

    def tolerances(self) -> pd.DataFrame:
        """Get the absolute tolerance of every state.

        Returns:
            A row per state with `sid`, `kind`, `compartment` (the compartment
            of a species, amount and concentration, empty for another state),
            `volume` (the reference volume of a concentration species, which
            scales its tolerance, empty for another state),
            `absolute_tolerance` and `scale` (the scale of a tolerance
            relative to it, see `AbsoluteTolerance.relative`, empty for the
            tolerance of a kind or an override).
        """
        compartments = self.symbols.species_compartment
        return pd.DataFrame(
            [
                {
                    "sid": s.sid,
                    "kind": s.kind.value,
                    "compartment": compartments.get(s.sid, s.compartment),
                    "volume": s.volume,
                    "absolute_tolerance": s.absolute_tolerance,
                    "scale": np.nan if s.scale is None else s.scale,
                }
                for s in self._state_tolerances
            ],
            columns=[
                "sid",
                "kind",
                "compartment",
                "volume",
                "absolute_tolerance",
                "scale",
            ],
        )

    @staticmethod
    def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """Create GlobalParameter DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(r.getCurrentSBML())
        model: libsbml.Model = doc.getModel()
        # the helpers of the initial assignments and the markers of the events
        # are not parameters of the model, see `initialize`
        sids = [sid for sid in r_model.getGlobalParameterIds() if not _is_helper(sid)]
        parameters: list[libsbml.Parameter] = [model.getParameter(sid) for sid in sids]
        data = {
            "sid": sids,
            "value": [r.getValue(sid) for sid in sids],
            "unit": [p.getUnits() for p in parameters],
            "constant": [p.getConstant() for p in parameters],
            "name": [p.getName() for p in parameters],
        }
        return pd.DataFrame(
            data, columns=pd.Index(["sid", "value", "unit", "constant", "name"])
        )

    @staticmethod
    def species_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """Create FloatingSpecies DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        sbml_str = r.getCurrentSBML()

        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
        model: libsbml.Model = doc.getModel()

        sids = r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
        species: list[libsbml.Species] = [model.getSpecies(sid) for sid in sids]

        data = {
            "sid": sids,
            "concentration": np.concatenate(
                [
                    r_model.getFloatingSpeciesConcentrations(),
                    r_model.getBoundarySpeciesConcentrations(),
                ],
                axis=0,
            ),
            "amount": np.concatenate(
                [
                    r.model.getFloatingSpeciesAmounts(),
                    r.model.getBoundarySpeciesAmounts(),
                ],
                axis=0,
            ),
            "unit": [s.getUnits() for s in species],
            "constant": [s.getConstant() for s in species],
            "boundaryCondition": [s.getBoundaryCondition() for s in species],
            "name": [s.getName() for s in species],
        }

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "sid",
                    "concentration",
                    "amount",
                    "unit",
                    "constant",
                    "boundaryCondition",
                    "name",
                ]
            ),
        )


@dataclass(frozen=True)
class SBMLIds:
    """The ids of the SBML of a model, see `RoadrunnerSBMLModel.sbml_ids`.

    Attributes:
        constant_parameters: the parameters which are constant.
        species: the species.
    """

    constant_parameters: tuple[str, ...]
    species: tuple[str, ...]


#: the names of the settings of an integrator by its name, see `_setting_names`
_SETTING_NAMES: dict[str, frozenset[str]] = {}


def _setting_names(integrator: roadrunner.Integrator) -> frozenset[str]:
    """Get the names of the settings of an integrator, read once per kind."""
    name = str(integrator.getName())
    names = _SETTING_NAMES.get(name)
    if names is None:
        names = frozenset(integrator.getSettings())
        _SETTING_NAMES[name] = names
    return names


@dataclass(frozen=True)
class _State:
    """The values of a model, see `RoadrunnerSBMLModel._loaded_state`.

    Attributes:
        ids: the global parameters, compartments, boundary and floating
            species, in this order, see `_state_ids`.
        values: their values, the amounts of the species.
        settable: whether a value is not the variable of an assignment rule.
        first_species: the index of the first species, the number of the
            parameters and the compartments.
        concentrations: the concentrations of the boundary and the floating
            species, in the order of their ids.
    """

    ids: list[str]
    values: np.ndarray
    settable: np.ndarray
    first_species: int
    concentrations: np.ndarray

    @cached_property
    def index(self) -> dict[str, int]:
        """Get the index of every id in `ids`."""
        return {sid: k for k, sid in enumerate(self.ids)}

    @cached_property
    def nan(self) -> np.ndarray | None:
        """Get whether a value as loaded is `NaN`, `None` if none is."""
        nan = np.isnan(self.values)
        return nan if nan.any() else None


def _changed(current: np.ndarray, state: _State) -> np.ndarray:
    """Get the indices of the values which differ from the model as loaded.

    A value which is `NaN` as loaded and now, e.g. a compartment without a
    size, is unchanged, and the variable of an assignment rule, which
    roadrunner refuses to set, is never one of them.

    Args:
        current: the values of `_state_values` now, or its first values, e.g.
            the parameters and the compartments of `_parameter_values`.
        state: the values as loaded.

    Returns:
        The indices into `state.ids`, in their order.
    """
    n = current.size
    differ = (current != state.values[:n]) & state.settable[:n]
    if state.nan is not None:
        differ &= ~(state.nan[:n] & np.isnan(current))
    return np.flatnonzero(differ)


def _state_ids(r: roadrunner.RoadRunner) -> list[str]:
    """Get the ids of the values of `_state_values`."""
    m = r.model
    return [
        *m.getGlobalParameterIds(),
        *m.getCompartmentIds(),
        *m.getBoundarySpeciesIds(),
        *m.getFloatingSpeciesIds(),
    ]


def _state_values(r: roadrunner.RoadRunner) -> np.ndarray:
    """Get the values of the parameters, compartments and the amounts of the species."""
    m = r.model
    return np.concatenate(
        [
            np.asarray(m.getGlobalParameterValues(), dtype=float),
            np.asarray(m.getCompartmentVolumes(), dtype=float),
            np.asarray(m.getBoundarySpeciesAmounts(), dtype=float),
            np.asarray(m.getFloatingSpeciesAmounts(), dtype=float),
        ]
    )


def _parameter_values(r: roadrunner.RoadRunner) -> np.ndarray:
    """Get the values of the parameters and the compartments, the first ones of `_state_values`."""
    m = r.model
    return np.concatenate(
        [
            np.asarray(m.getGlobalParameterValues(), dtype=float),
            np.asarray(m.getCompartmentVolumes(), dtype=float),
        ]
    )


def _floats(values: Sequence[float] | np.ndarray) -> list[float]:
    """Get the values of an array of roadrunner as floats."""
    return np.asarray(values, dtype=float).tolist()


def _with_initial_helpers(sbml: str) -> tuple[str, dict[str, str]]:
    """Add a helper parameter for every initial assignment of a model.

    The helper `<entity>__initial` is a parameter with an assignment rule
    which is the math of the initial assignment of the entity, so roadrunner
    evaluates the math on the current values whenever the helper is read, see
    `RoadrunnerSBMLModel.initialize`. The initial assignments stay.

    Args:
        sbml: the SBML of the model.

    Returns:
        The SBML with the helpers and the helper of every entity.

    Raises:
        ValueError: if the model already has an entity of the id of a helper.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model = doc.getModel()
    helpers: dict[str, str] = {}
    for assignment in list(model.getListOfInitialAssignments()):
        if assignment.getMath() is None:
            # an initial assignment without math assigns nothing
            continue
        entity = assignment.getSymbol()
        helper = f"{entity}{INITIAL_SUFFIX}"
        if model.getElementBySId(helper) is not None:
            raise ValueError(
                f"The model already has an entity '{helper}', the id of the "
                f"helper of the initial assignment of '{entity}'."
            )
        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(helper)
        parameter.setConstant(False)
        rule: libsbml.AssignmentRule = model.createAssignmentRule()
        rule.setVariable(helper)
        rule.setMath(assignment.getMath().deepCopy())
        helpers[entity] = helper
    return libsbml.writeSBMLToString(doc), helpers
