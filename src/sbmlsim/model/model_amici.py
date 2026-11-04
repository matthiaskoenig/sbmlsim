"""The model of an SBML model in AMICI, see `sbmlsim.model.model_sbml`.

`AmiciSBMLModel` compiles the SBML of a model with AMICI (the optional extra
`amici`) into a python extension, which is cached on disk (`compiled_dir`),
and owns an instance of the compiled model and of its solver, CVODES. Its
values are read and written in the selections of roadrunner, `S` the amount
of a species, `[S]` its concentration, the id of a parameter or a
compartment, the id of a reaction its rate, the id of a species reference
its stoichiometry (a number, which AMICI folds into its equations) and
`time`, with the numbers of the SBML, which AMICI keeps as they are. Every
selection roadrunner makes by default is a state, a free parameter or an
expression of AMICI or a number of the SBML, so the model is compiled
without observables. A model of SBML level 1 or 2 is converted to level 3
version 2 first.

AMICI integrates the concentration of a species whose `hasOnlySubstanceUnits`
is false and the amount of the other species, folds a compartment without a
rule into the equations as a number and turns a parameter with an initial
assignment into an expression, which follows its math during a simulation.
The SBML which AMICI compiles (`amici_sbml`) therefore keeps every entity a
change can set apart:

- a parameter without a rule, an initial assignment or an event is a free
  parameter of AMICI, which is set directly;
- a compartment without an assignment rule and a parameter with an initial
  assignment, a rate rule or an event are the targets of one event which
  never fires (`TARGETS_EVENT`), which makes them states of AMICI whose rate
  is 0 (or the one of their rate rule), so their value is the one at the
  initialization until a change or an event sets it;
- the species are states of AMICI anyway.

The initial value of every such state is an initial assignment of two free
parameters which sbmlsim adds per entity, its value and its kind
(`VALUE_PREFIX`, `KIND_PREFIX`): `piecewise(value as amount, kind == 1, value
as concentration, kind == 2, initial value of the SBML)`, the initial value of
the SBML being its initial assignment or its number, an amount of a species
whose state is a concentration divided by its compartment and vice versa (a
conversion factor, a constant of SBML, stays a free parameter of AMICI). A
pre-initialization sets these parameters and the free parameters, AMICI
evaluates the initial state of the model from them in the order of its
dependencies, so a target replaces its own initial assignment and every
initial value which depends on it follows, as in roadrunner
(`RoadrunnerSBMLModel.initialize`). A compartment whose number in the SBML is
not finite (`NaN`) and which has no initial assignment stays a number of
AMICI, which integrates no state which is not finite, and a change of it
raises.

The current state of the model is its time, the states of AMICI and its free
parameters, which the executor sets between two integrations; the values of
the assignment rules and the rates of the reactions are the expressions of
AMICI, which an integration of no length at the current time evaluates.

AMICI finds the roots of the triggers of the events of the model, the
executor fires the events, see `sbmlsim.simulator.executor_amici`: every
event of the model keeps its trigger in the SBML AMICI compiles and, instead
of its assignments, sets two states of AMICI, that it fired in the
integration and the time of its first execution (`FIRED_PREFIX`,
`FIRST_PREFIX`, `fired`, `first_times`), which the executor sets to 0 before
every integration (`reset_events`), so the first root of an integration is
known exactly and the state of the model is the one before it. AMICI
evaluates a comparison exactly at its threshold at the start of an
integration as written (`time <= 10` at the time 10 true, `time > 10` false:
its Heaviside variables are initialized from the sign of the root functions,
a root of 0 as true) and keeps the value for the whole integration, since
CVODES does not report a root at the start and the root function does not
change its sign afterwards. The executor therefore shifts every comparison
which is exactly at its threshold at the start of an integration by a little,
so that AMICI finds the crossing when the operands move: every strict
comparison of a trigger, `a > b` or `a < b`, is compiled shifted by a free
parameter of AMICI, `a > b + shift` (`SHIFT_PREFIX`, `comparisons`,
`set_shifts`), and so is every comparison of the maths AMICI integrates, the
kinetic laws, the assignment rules and the rate rules, e.g. the condition of
a piecewise in a rate (`SWITCH_PREFIX`, `switches`, `set_switch_shifts`,
`Switch`): `a > b + shift`, `a + shift < b`, `a >= b - shift` and
`a <= b + shift`, so a shifted comparison keeps its value at its threshold
until the operands move by the shift in the direction which changes it, as
roadrunner evaluates it at every step; `a == b` and `a != b` of a math are
compiled as `a <= b + shift && a >= b - shift` and `a < b - shift ||
a > b + shift`, which is how AMICI compiles them without the shift. The
trigger of every event starts true in the compiled SBML, so AMICI fires no
event at the start of an integration; the events whose trigger is false
before the time 0 (`initial_value_false`) are fired by the executor at the
start of a simulation, as the ones a change triggers. A feature which AMICI
does not compile or simulates otherwise than roadrunner raises at the load,
see `_check_supported`; a trigger which compares for equality, `a == b` or
`a != b`, is refused there: AMICI has no root of it, and roadrunner fires it
only where a step of its integrator lands on the equality.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time as time_module
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, override

import libsbml
import numpy as np

from sbmlsim.engine import Engine, import_amici
from sbmlsim.model.comparisons import (
    COMPARISONS,
    math_comparisons,
    shift_comparison,
    trigger_comparisons,
)
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.model_sbml import (
    EngineSettings,
    EngineValues,
    Feature,
    IntegratorSetting,
    SBMLModel,
    conversion_factors,
    rate_rule_entities,
    reads_rate_of,
    sbml_maths,
    variable_features,
)
from sbmlsim.model.symbols import ModelSymbols, TargetKind, selection_math
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.units import UnitRegistry

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

logger = logging.getLogger(__name__)

#: the environment variable of the directory of the compiled models, see
#: `cache_root`
CACHE_ENV = "SBMLSIM_AMICI_CACHE"

#: the version of the way sbmlsim compiles a model with AMICI, part of the key
#: of the cache: a change of the options or of the files of a compiled model
#: raises it
CACHE_FORMAT = 3

#: the options of AMICI's compilation of a model: no observables (the values
#: are the states and the expressions; AMICI 1.2 fails to build an event
#: observable of a model whose trigger reads an assignment rule), every species
#: a state (no conservation laws), no sensitivities
IMPORT_OPTIONS: dict[str, Any] = {
    "observation_model": [],
    "compute_conservation_laws": False,
    "generate_sensitivity_code": False,
}

#: the id of the event which makes the entities a change sets states of AMICI
TARGETS_EVENT = "__sbmlsim_targets"

#: the prefix of the id sbmlsim gives an event of the model without one, by
#: its position among the events of the model
EVENT_PREFIX = "__sbmlsim_event_"

#: the prefix of the state of AMICI which records that an event of the model
#: fired in an integration, 1 after its first execution, by the position of
#: the event, see `AmiciSBMLModel.fired`
FIRED_PREFIX = "__sbmlsim_fired_"

#: the prefix of the state of AMICI of the time of the first execution of an
#: event of the model in an integration, by the position of the event, see
#: `AmiciSBMLModel.first_times`
FIRST_PREFIX = "__sbmlsim_first_"

#: the prefix of the free parameter of AMICI by which a strict comparison of
#: the trigger of an event is shifted, by the position of the event and of the
#: comparison in its trigger, see `StrictComparison`
SHIFT_PREFIX = "__sbmlsim_shift_"

#: the prefix of the free parameter of AMICI by which a comparison of a math
#: of the model is shifted, by the position of the comparison among them, see
#: `Switch`
SWITCH_PREFIX = "__sbmlsim_switch_"

#: the prefix of the free parameter of the value of an entity which a
#: pre-initialization sets, see the module
VALUE_PREFIX = "__sbmlsim_value_"

#: the prefix of the free parameter of the kind of the value of an entity, 0
#: for its initial value of the SBML, 1 for an amount or the value of a
#: parameter or a compartment, 2 for a concentration
KIND_PREFIX = "__sbmlsim_kind_"

#: the kinds of the value of an entity, see `KIND_PREFIX`
_KIND_SBML, _KIND_VALUE, _KIND_CONCENTRATION = 0.0, 1.0, 2.0

#: the name of the integrator, CVODES of SUNDIALS
INTEGRATOR = "cvodes"

#: the factor from the absolute tolerance of roadrunner to the one of CVODES
#: in AMICI, so AMICI integrates at least as accurately as roadrunner at equal
#: settings. roadrunner gives every state the absolute tolerance of its kind,
#: a concentration species its tolerance times the volume of its compartment,
#: see `sbmlsim.model.tolerances`; AMICI has one tolerance, of the
#: concentration of a species whose state is a concentration and of the amount
#: of the other species. Measured on the HCTZ model, doses of 10, 25 and 50 mg
#: over 24 hr (240 output times), against roadrunner at an absolute tolerance
#: of 1e-16 and a relative one of 1e-12, the largest error relative to the
#: largest value of a variable: roadrunner 1.5e-7 at the defaults of
#: `Simulator` (1e-10 and 1e-10), AMICI 9.1e-7 with the tolerance as it is,
#: 1.5e-7 with 1e-12 and 1.1e-7 with 1e-13, i.e. the factor 1e-3.
AMICI_ATOL_FACTOR: float = 1e-3

#: the tool with which AMICI builds a model, see `_build_tools_on_path`
BUILD_TOOL = "ninja"

#: the compiled models of the process by the name of their module, a module
#: of a name is imported once
_MODULES: dict[str, ModuleType] = {}


class _Store(StrEnum):
    """Where AMICI keeps the value of an entity."""

    STATE = "state"
    FREE = "free parameter"
    EXPRESSION = "expression"
    NUMBER = "number"


@dataclass(frozen=True)
class AmiciSBML:
    """The SBML which AMICI compiles, see the module.

    Attributes:
        sbml: the SBML.
        flagged: the entities whose initial value a pre-initialization sets
            through the parameters `VALUE_PREFIX` and `KIND_PREFIX`.
        not_a_number: the compartments which are numbers of AMICI, their
            number in the SBML is not finite.
        initial_value_false: the events whose trigger is false before the
            time 0 in the SBML of the model, in the order of the SBML.
        events: the ids of the events of the model in AMICI, in the order of
            the SBML, `EVENT_PREFIX` and the position of an event without an
            id; the states which record that an event fired and when first
            are named by its position (`FIRED_PREFIX`, `FIRST_PREFIX`).
        comparisons: the strict comparisons of the triggers of the events,
            in the order of the shift parameters (`SHIFT_PREFIX`).
        switches: the comparisons of the maths of the model, in the order of
            their shift parameters (`SWITCH_PREFIX`).
    """

    sbml: str
    flagged: frozenset[str]
    not_a_number: frozenset[str]
    initial_value_false: tuple[str, ...]
    events: tuple[str, ...] = ()
    comparisons: tuple[StrictComparison, ...] = ()
    switches: tuple[Switch, ...] = ()


@dataclass(frozen=True)
class StrictComparison:
    """A strict comparison of the trigger of an event, `a > b` or `a < b`.

    AMICI compiles it shifted, `a > b + shift` or `a + shift < b`, see the
    module; the operands are MathML of selections, which
    `sbmlsim.sbml_math.compile_math` compiles, so the executor tells where
    the comparison is exactly at its threshold, `a == b`.

    Attributes:
        event: the position of the event among the events of the model.
        left: the MathML of the left operand `a`.
        right: the MathML of the right operand `b`.
    """

    event: int
    left: str
    right: str


@dataclass(frozen=True)
class Switch:
    """A comparison of a math AMICI integrates, e.g. the condition of a piecewise in a rate.

    AMICI compiles it shifted by a free parameter so that it keeps its value
    at its threshold until the operands move, see the module; the operands
    are MathML of selections, which `sbmlsim.sbml_math.compile_math`
    compiles, so the executor tells where the comparison is exactly at its
    threshold, `a == b`, and shifts it there.

    Attributes:
        left: the MathML of the left operand `a`.
        right: the MathML of the right operand `b`.
    """

    left: str
    right: str


class AmiciSBMLModel(SBMLModel):
    """The model of an SBML model in AMICI, see the module.

    Attributes:
        amici_model: the model of AMICI, whose free parameters, time and
            initial state an integration sets.
        solver: the solver of AMICI, CVODES.
        initial_value_false: the events whose trigger is false before the
            time 0, which the executor fires at the start of a simulation
            when their trigger is true, see the module.
        event_ids: the ids of the events of the model in AMICI, in the order
            of the SBML, see `AmiciSBML.events`.
        comparisons: the strict comparisons of the triggers of the events,
            see `StrictComparison` and `set_shifts`.
        switches: the comparisons of the maths of the model, see `Switch`
            and `set_switch_shifts`.
        absolute_tolerance: the absolute tolerance set last, see
            `set_integrator_settings`.
    """

    engine = Engine.AMICI

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
        """Compile the model with AMICI or take it from the cache, with changes, selections and settings.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`; they are not selected by default.

        Raises:
            ValueError: for a feature AMICI does not simulate as roadrunner,
                an error of AMICI when it imports the model, a selection
                AMICI has not, or an unknown setting.
            RuntimeError: if AMICI fails to compile the model.
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
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
        sbml_model: libsbml.Model = doc.getModel()
        _check_supported(sbml_model, self.label)
        #: the stoichiometries of the species references with an id, which are
        #: constant (`_check_supported`) and AMICI folds into its equations
        self._stoichiometries: dict[str, float] = {
            # a reference of level 3 without a stoichiometry has the
            # stoichiometry 1 in roadrunner and in the equations of AMICI
            sr.getId(): float(sr.getStoichiometry()) if sr.isSetStoichiometry() else 1.0
            for reaction in sbml_model.getListOfReactions()
            for sr in (*reaction.getListOfReactants(), *reaction.getListOfProducts())
            if sr.isSetId()
        }
        self._state_ids: tuple[str, ...] = _integrated(sbml_model, self.symbols)

        compiled = amici_sbml(sbml, self.symbols, self.label)
        self.initial_value_false: tuple[str, ...] = compiled.initial_value_false
        self.event_ids: tuple[str, ...] = compiled.events
        self.comparisons: tuple[StrictComparison, ...] = compiled.comparisons
        self.switches: tuple[Switch, ...] = compiled.switches
        self._flagged: frozenset[str] = compiled.flagged
        self._not_a_number: frozenset[str] = compiled.not_a_number
        module = load_module(compiled.sbml, self.label)
        self.amici_model: Any = module.get_model()
        self.solver: Any = self.amici_model.create_solver()

        amodel = self.amici_model
        self._where: dict[str, tuple[_Store, int]] = {}
        for k, sid_ in enumerate(amodel.get_state_ids()):
            self._where[str(sid_)] = (_Store.STATE, k)
        free_ids = [str(p) for p in amodel.get_free_parameter_ids()]
        for k, pid in enumerate(free_ids):
            self._where[pid] = (_Store.FREE, k)
        expression_ids = [str(e) for e in amodel.get_expression_ids()]
        for k, eid in enumerate(expression_ids):
            self._where.setdefault(eid, (_Store.EXPRESSION, k))
        #: the numbers of the SBML which AMICI folds into its equations
        self._numbers: dict[str, float] = {
            **dict.fromkeys(self._not_a_number, math.nan),
            **self._stoichiometries,
        }
        for sid_ in self._numbers:
            self._where[sid_] = (_Store.NUMBER, -1)
        reactions = [r.getId() for r in sbml_model.getListOfReactions()]
        #: the index of the expression of the rate of every reaction, the
        #: rates are the last expressions of AMICI in the order of the SBML
        self._fluxes: dict[str, int] = {
            rid: len(expression_ids) - len(reactions) + k
            for k, rid in enumerate(reactions)
        }
        if any(expression_ids[k] != f"flux_{rid}" for rid, k in self._fluxes.items()):
            raise ValueError(
                f"AMICI has no rates of the reactions of the model '{self.label}' "
                f"in its last expressions {expression_ids[-len(reactions) :]}."
            )
        missing = sorted(
            sid_
            for sid_ in (
                *self.symbols.parameters,
                *self.symbols.compartments,
                *self.symbols.species,
            )
            if sid_ not in self._where
        )
        if missing:
            raise ValueError(
                f"AMICI has no values of {missing} of the model '{self.label}'."
            )
        self._value_index: dict[str, int] = {
            sid_[len(VALUE_PREFIX) :]: k
            for k, sid_ in enumerate(free_ids)
            if sid_.startswith(VALUE_PREFIX)
        }
        self._kind_index: dict[str, int] = {
            sid_[len(KIND_PREFIX) :]: k
            for k, sid_ in enumerate(free_ids)
            if sid_.startswith(KIND_PREFIX)
        }

        #: the free parameters as loaded, the parameters of sbmlsim 0
        self._loaded_parameters: np.ndarray = np.array(
            amodel.get_free_parameters(), dtype=float
        )
        self._time: float = 0.0
        self._x: np.ndarray = np.zeros(0)
        self._p: np.ndarray = self._loaded_parameters.copy()
        #: the expressions and the rates of the states of AMICI of the current
        #: state, evaluated when they are read
        self._w: np.ndarray | None = None
        self._xdot: np.ndarray | None = None
        state_ids = [str(x) for x in amodel.get_state_ids()]
        #: the states which record that the events fired in an integration
        self._fired: np.ndarray = np.array(
            [state_ids.index(f"{FIRED_PREFIX}{k}") for k in range(len(self.event_ids))],
            dtype=int,
        )
        #: the states of the times of the first executions of the events
        self._first: np.ndarray = np.array(
            [state_ids.index(f"{FIRST_PREFIX}{k}") for k in range(len(self.event_ids))],
            dtype=int,
        )
        #: the free parameters of the shifts of the strict comparisons of the
        #: triggers
        self._shifts: np.ndarray = np.array(
            [free_ids.index(pid) for pid in shift_ids(self.comparisons)], dtype=int
        )
        #: the free parameters of the shifts of the comparisons of the maths
        self._switch_shifts: np.ndarray = np.array(
            [free_ids.index(pid) for pid in switch_ids(len(self.switches))],
            dtype=int,
        )
        self._start(self._loaded_parameters.copy())

        #: the values of the model as loaded, see `loaded_value`
        self._loaded: dict[str, float] = {}
        for sid_ in (*self.symbols.parameters, *self.symbols.compartments):
            self._loaded[sid_] = self.value(sid_)
        for sid_ in self.symbols.species:
            self._loaded[sid_] = self.value(sid_)
            self._loaded[f"[{sid_}]"] = self.value(f"[{sid_}]")

        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        #: the settings of roadrunner, which reports the ignored ones once
        self._settings = EngineSettings("AMICI", INTEGRATOR, self.label, logger)

        self.set_selections(selections)
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )

    @property
    def values(self) -> EngineValues:
        """Get the current values of the model in the selections of roadrunner."""
        return EngineValues(self)

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
    ) -> AmiciSBMLModel:
        """Create from AbstractModel, see `AmiciSBMLModel`."""
        return AmiciSBMLModel(
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

    @property
    def time(self) -> float:
        """Get the current time of the model."""
        return self._time

    @property
    def state(self) -> np.ndarray:
        """Get a copy of the current states of AMICI, in the order of its state ids."""
        return self._x.copy()

    @property
    def free_parameters(self) -> np.ndarray:
        """Get a copy of the current free parameters of AMICI, in the order of their ids."""
        return self._p.copy()

    def set_time(self, time: float) -> None:
        """Set the current time of the model, between two integrations.

        Args:
            time: the time.
        """
        self._time = float(time)
        self._w = None
        self._xdot = None

    def set_state(self, time: float, state: Sequence[float] | np.ndarray) -> None:
        """Set the current time and states of the model, e.g. after an integration.

        Args:
            time: the time.
            state: the states of AMICI, in the order of its state ids.

        Raises:
            ValueError: if the number of states is not the one of AMICI.
        """
        x = np.array(state, dtype=float)
        if x.shape != self._x.shape:
            raise ValueError(
                f"The model '{self.label}' has {self._x.size} states in AMICI, "
                f"not {x.size}."
            )
        self._x = x
        self.set_time(time)

    def prepare(self, timepoints: Sequence[float] | np.ndarray) -> None:
        """Set the model of AMICI to an integration from the current state.

        The free parameters, the start time (the current time) and the
        initial state (the current states) of the model of AMICI are the
        ones of the model; the integration outputs at the time points. The
        solver is a new one with the settings of the one before: CVODES keeps
        a root function which is exactly 0 at the start of an integration
        and still 0 after its first small step out of the root finding of
        its memory (`gactive` of `cvRcheck1`), which AMICI reuses for the
        next integration without setting it back (`CVodeRootInit` with the
        same function returns at once), so the trigger would not be found
        again, e.g. after an integration from a change which set a state to
        the threshold of a trigger.

        Args:
            timepoints: the output times, from the current time on.
        """
        amodel = self.amici_model
        before = self.solver
        solver = amodel.create_solver()
        solver.set_relative_tolerance(before.get_relative_tolerance())
        solver.set_absolute_tolerance(before.get_absolute_tolerance())
        solver.set_max_steps(before.get_max_steps())
        solver.set_max_step_size(before.get_max_step_size())
        self.solver = solver
        amodel.set_free_parameters(self._p.tolist())
        amodel.set_t0(self._time)
        amodel.set_initial_state(self._x.tolist())
        amodel.set_timepoints(list(map(float, timepoints)))

    def _start(self, parameters: np.ndarray) -> None:
        """Start the model at the time 0 from the initial state of AMICI of free parameters.

        Args:
            parameters: the free parameters of AMICI, the ones of sbmlsim of a
                pre-initialization included.
        """
        amodel = self.amici_model
        amodel.set_free_parameters(parameters.tolist())
        amodel.set_t0(0.0)
        # no custom initial state, AMICI evaluates it from the parameters
        amodel.set_initial_state([])
        self._p = parameters
        self._x = np.array(amodel.get_initial_state(), dtype=float)
        self.set_time(0.0)

    def _expressions(self) -> np.ndarray:
        """Get the expressions of AMICI of the current state.

        AMICI evaluates them in an integration of no length at the current
        time from the current states; it fires no event at its start, see
        the module.

        Raises:
            RuntimeError: if AMICI fails to evaluate them.
        """
        if self._w is None:
            if self.amici_model.nw == 0:
                self._w = np.zeros(0)
            else:
                self._evaluate()
        assert self._w is not None
        return self._w

    def _evaluate(self) -> None:
        """Evaluate the expressions and the rates of the states of AMICI of the current state.

        Raises:
            RuntimeError: if AMICI fails to evaluate them.
        """
        amodel = self.amici_model
        self.prepare([self._time])
        rdata = run_simulation()(amodel, self.solver)
        if rdata.status != 0:
            raise RuntimeError(
                f"AMICI cannot evaluate the model '{self.label}' at the "
                f"time {self._time}, status {rdata.status}."
            )
        self._w = (
            np.array(rdata.w, dtype=float).reshape(1, amodel.nw)[0]
            if amodel.nw
            else np.zeros(0)
        )
        self._xdot = np.array(rdata.xdot, dtype=float)

    def rates(self, states: Sequence[str]) -> np.ndarray:
        """Get the current rates of change of states.

        The rate of a species is the rate of its amount, `V dc/dt + c dV/dt`
        for a species whose state of AMICI is its concentration `c` in the
        compartment of the size `V`; `dV/dt` is the rate of a compartment
        which is a state of AMICI and 0 for one which is not (a number, a
        free parameter, or an assignment rule, which is constant in a steady
        state of the states it reads).

        Args:
            states: the ids of species or of the targets of rate rules, see
                `state_ids`.

        Returns:
            The rate of every state.

        Raises:
            RuntimeError: if AMICI fails to evaluate them.
        """
        if self._xdot is None:
            self._evaluate()
        xdot = self._xdot
        assert xdot is not None
        symbols = self.symbols

        def rate(sid: str) -> float:
            store, index = self._where[sid]
            if store is not _Store.STATE:
                return 0.0
            return float(xdot[index])

        result = []
        for sid in states:
            value = rate(sid)
            if sid in symbols.species and sid not in symbols.only_substance:
                compartment = symbols.species_compartment[sid]
                value = value * self._stored(compartment) + self._stored(sid) * rate(
                    compartment
                )
            result.append(value)
        return np.array(result, dtype=float)

    def fired(self, states: np.ndarray) -> np.ndarray:
        """Tell the events AMICI fired in an integration from states of AMICI.

        Args:
            states: states of AMICI, a row or an array of rows.

        Returns:
            Whether every event, in the order of `event_ids`, fired since the
            records were set back (`reset_events`), of every row.
        """
        return np.asarray(states, dtype=float)[..., self._fired] > 0.5

    def first_times(self, states: np.ndarray) -> np.ndarray:
        """Get the times of the first executions of the events by AMICI from states of AMICI.

        Args:
            states: states of AMICI, a row or an array of rows.

        Returns:
            The time of the first execution of every event, in the order of
            `event_ids`, since the records were set back (`reset_events`), of
            every row; 0 for an event which did not fire, see `fired`.
        """
        return np.asarray(states, dtype=float)[..., self._first]

    def reset_events(self) -> None:
        """Set the records of the events back, before an integration, see `fired`."""
        self._x[self._fired] = 0.0
        self._x[self._first] = 0.0

    def set_shifts(self, shifts: Sequence[float]) -> None:
        """Set the shifts of the strict comparisons of the triggers, see `StrictComparison`.

        Args:
            shifts: a shift per comparison, in the order of `comparisons`, 0
                for a comparison as written.
        """
        self._p[self._shifts] = np.asarray(shifts, dtype=float)
        self._w = None
        self._xdot = None

    def set_switch_shifts(self, shifts: Sequence[float]) -> None:
        """Set the shifts of the comparisons of the maths of the model, see `Switch`.

        Args:
            shifts: a shift per comparison, in the order of `switches`, 0 for
                a comparison as written.
        """
        self._p[self._switch_shifts] = np.asarray(shifts, dtype=float)
        self._w = None
        self._xdot = None

    @property
    def not_a_number(self) -> frozenset[str]:
        """Get the compartments which are numbers of AMICI, their number in the SBML is not finite."""
        return self._not_a_number

    def output_transform(
        self, selections: Sequence[str]
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Get the conversion of the output of AMICI into the values of selections.

        Args:
            selections: the selections, see `value`.

        Returns:
            A function of the times, the states and the expressions of AMICI
            of an output (a row per time), which gives the values of the
            selections, a row per time and a column per selection, with the
            current free parameters.

        Raises:
            ValueError: if AMICI has no selection of a name.
        """
        for selection in selections:
            if not self._resolves(selection):
                raise ValueError(self._selection_error(selection))
        symbols = self.symbols

        def stored(sid: str) -> Callable[..., np.ndarray | float]:
            store, index = self._where[sid]
            if store is _Store.STATE:
                return lambda t, x, w: x[:, index]
            if store is _Store.EXPRESSION:
                return lambda t, x, w: w[:, index]
            if store is _Store.FREE:
                return lambda t, x, w: self._p[index]
            number = self._numbers[sid]
            return lambda t, x, w: number

        def column(selection: str) -> Callable[..., np.ndarray | float]:
            if selection == "time":
                return lambda t, x, w: t
            if selection in self._fluxes:
                index = self._fluxes[selection]
                return lambda t, x, w: w[:, index]
            concentration = selection.startswith("[")
            sid = selection[1:-1] if concentration else selection
            value = stored(sid)
            if sid not in symbols.species:
                return value
            volume = stored(symbols.species_compartment[sid])
            only_substance = sid in symbols.only_substance
            if concentration and only_substance:
                return lambda t, x, w: value(t, x, w) / volume(t, x, w)
            if not concentration and not only_substance:
                return lambda t, x, w: value(t, x, w) * volume(t, x, w)
            return value

        columns = [column(s) for s in selections]

        def transform(t: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
            t = np.asarray(t, dtype=float)
            result = np.empty((t.size, len(columns)))
            for k, function in enumerate(columns):
                result[:, k] = function(t, x, w)
            return result

        return transform

    def _stored(self, sid: str) -> float:
        """Get the value AMICI keeps of an entity.

        The value of a parameter or a compartment, the concentration of a
        species whose state is a concentration and the amount of the other
        species.
        """
        store, index = self._where[sid]
        if store is _Store.STATE:
            return float(self._x[index])
        if store is _Store.FREE:
            return float(self._p[index])
        if store is _Store.EXPRESSION:
            return float(self._expressions()[index])
        return self._numbers[sid]

    def _volume(self, species: str) -> float:
        """Get the current size of the compartment of a species."""
        return self._stored(self.symbols.species_compartment[species])

    def _amount(self, species: str) -> float:
        """Get the current amount of a species."""
        stored = self._stored(species)
        if species in self.symbols.only_substance:
            return stored
        return stored * self._volume(species)

    def _concentration(self, species: str) -> float:
        """Get the current concentration of a species."""
        stored = self._stored(species)
        if species in self.symbols.only_substance:
            return stored / self._volume(species)
        return stored

    def _selection_error(self, selection: str) -> str:
        """Get the message of a selection AMICI has not."""
        return f"AMICI has no selection '{selection}'."

    def _resolves(self, selection: str) -> bool:
        """Whether AMICI has a selection of a name."""
        if (
            selection == "time"
            or selection in self._fluxes
            or selection in self._stoichiometries
        ):
            return True
        if selection.startswith("[") and selection.endswith("]"):
            return selection[1:-1] in self.symbols.species
        return selection in self.symbols.parameters or (
            selection in self.symbols.compartments or selection in self.symbols.species
        )

    @override
    def value(self, selection: str) -> float:
        """Get the current value of a selection.

        Args:
            selection: the selection, e.g. `S`, `[S]`, a parameter, a
                compartment, a reaction or `time`.

        Returns:
            The value in the current state of the model.

        Raises:
            RuntimeError: if AMICI has no selection of the name, as
                roadrunner.
        """
        if not self._resolves(selection):
            raise RuntimeError(self._selection_error(selection))
        if selection == "time":
            return self._time
        if selection in self._fluxes:
            return float(self._expressions()[self._fluxes[selection]])
        if selection.startswith("["):
            return self._concentration(selection[1:-1])
        if selection in self.symbols.species:
            return self._amount(selection)
        return self._stored(selection)

    def set_value(self, selection: str, value: float) -> None:
        """Set the current value of a selection, between two integrations.

        The value is written into the states or the free parameters of
        AMICI, the amount of a species whose state is a concentration as the
        concentration in the current size of its compartment and vice versa.
        As in roadrunner, which integrates the amounts of the species, every
        other species keeps its amount, also when the size of its compartment
        changes (a compartment which is set, or one of an assignment rule
        which reads what is set).

        Args:
            selection: `S`, `[S]`, a parameter or a compartment.
            value: the value.

        Raises:
            RuntimeError: if AMICI has no selection of the name, or it cannot
                be set: the time, a reaction, the variable of an assignment
                rule, as roadrunner, or a compartment which is a number of
                AMICI.
        """
        if not self._resolves(selection):
            raise RuntimeError(self._selection_error(selection))
        concentration = selection.startswith("[")
        entity = selection[1:-1] if concentration else selection
        if (
            selection == "time"
            or selection in self._fluxes
            or selection in self._stoichiometries
        ):
            raise RuntimeError(f"AMICI cannot set '{selection}'.")
        if entity in self.symbols.assignment_rules:
            raise RuntimeError(
                f"AMICI cannot set '{selection}', it is the variable of an "
                f"assignment rule."
            )
        if entity in self._not_a_number:
            raise RuntimeError(
                f"AMICI cannot set '{selection}' during a simulation, its value "
                f"in the SBML is not a number, which AMICI keeps out of its states."
            )
        symbols = self.symbols
        # the species whose state is a concentration keep their amounts
        kept = [
            sid
            for sid in symbols.species
            if sid != entity
            and sid not in symbols.only_substance
            and self._where[sid][0] is _Store.STATE
        ]
        volumes = {sid: self._volume(sid) for sid in kept}
        number = value
        if entity in symbols.species:
            only_substance = entity in symbols.only_substance
            if concentration and only_substance:
                number = value * self._volume(entity)
            elif not concentration and not only_substance:
                number = value / self._volume(entity)
        store, index = self._where[entity]
        if store is _Store.STATE:
            self._x[index] = number
        elif store is _Store.FREE:
            self._p[index] = number
        else:
            raise RuntimeError(
                f"AMICI cannot set '{selection}', it is {store.value} of AMICI."
            )
        self._w = None
        self._xdot = None
        for sid in kept:
            volume = self._volume(sid)
            if volume != volumes[sid]:
                state = self._where[sid][1]
                self._x[state] = self._x[state] * volumes[sid] / volume

    @override
    def initialize(self, assignments: Sequence[Assignment]) -> None:
        """Initialize the model with values before its initialization.

        The free parameters of AMICI are the ones as loaded with the values
        which are set, a parameter of AMICI directly and an entity whose
        initial value is a state of AMICI through the parameters of sbmlsim
        of its value and its kind, from which AMICI evaluates the initial
        state, see the module; the initialization does not depend on the one
        before it. A target replaces its own initial assignment, every
        initial value which depends on it follows. A compartment whose size
        changes keeps the concentration of the species whose initial value
        is a concentration and of a species which is set as a concentration,
        and the amount of every other species, as roadrunner does
        (`RoadrunnerSBMLModel.initialize`); the variable of an assignment
        rule follows its rule, a value of it is not used.

        A formula reads what it reads in roadrunner, the values as loaded
        with the values which are set, before the initial assignments are
        evaluated again, and an assignment rule on that state; the values of
        all formulas are evaluated before any of them is set (PEtab v2).

        Args:
            assignments: the pre-initialization assignments of a plan. A
                formula reads parameters only, see `Plan.preinit`.

        Raises:
            NotImplementedError: for a compartment which is a number of
                AMICI, see the module.
        """
        symbols = self.symbols
        outside = sorted(
            {symbols.entity(a.target) for a in assignments} & self._not_a_number
        )
        if outside:
            raise NotImplementedError(
                f"AMICI cannot set {outside} before the initialization of the "
                f"model '{self.label}', its value in the SBML is not a number, "
                f"which AMICI keeps out of its states."
            )
        set_values: list[tuple[Assignment, float]] = [
            (a, float(a.value)) for a in assignments if a.value is not None
        ]
        formulas = [a for a in assignments if a.formula is not None]
        if formulas:
            # the package of the simulator imports the models
            from sbmlsim.simulator.formula import compile_formula

            compiled = [(a, compile_formula(str(a.formula))) for a in formulas]
            values = self._formula_values(
                set_values, {s for _, f in compiled for s in f.symbols}
            )
            set_values += [
                (a, f.evaluate([values[s] for s in f.symbols])) for a, f in compiled
            ]
        parameters = self._loaded_parameters.copy()
        for a, value in set_values:
            entity = symbols.entity(a.target)
            if entity in symbols.assignment_rules:
                continue
            if entity in self._flagged:
                parameters[self._value_index[entity]] = value
                parameters[self._kind_index[entity]] = (
                    _KIND_CONCENTRATION
                    if a.kind is TargetKind.SPECIES_CONCENTRATION
                    else _KIND_VALUE
                )
            else:
                store, index = self._where[entity]
                if store is not _Store.FREE:
                    raise RuntimeError(
                        f"AMICI cannot set '{entity}' before the initialization of "
                        f"the model '{self.label}', it is {store.value} of AMICI."
                    )
                parameters[index] = value
        self._start(parameters)

    def _formula_values(
        self, set_values: Sequence[tuple[Assignment, float]], read: set[str]
    ) -> dict[str, float]:
        """Get the values of the parameters the formulas of an initialization read.

        A formula reads what it reads in roadrunner
        (`RoadrunnerSBMLModel.initialize`): the values which are set and
        otherwise the values as loaded, before the initial assignments are
        evaluated again, and the variable of an assignment rule follows its
        rule on that state. The state is the one as loaded with the values
        set as roadrunner sets them: the compartments first, which keep the
        concentration of their species whose initial value is a
        concentration or which are set as a concentration and the amount of
        every other species, then the other values.

        Args:
            set_values: the values which are set.
            read: the parameters the formulas read.

        Returns:
            The value of every parameter which is read.
        """
        symbols = self.symbols
        self._start(self._loaded_parameters.copy())
        concentrations = {
            symbols.entity(a.target)
            for a, _ in set_values
            if a.kind is TargetKind.SPECIES_CONCENTRATION
        }
        amounts = {
            symbols.entity(a.target)
            for a, _ in set_values
            if a.kind is TargetKind.SPECIES_AMOUNT
        }
        for a, value in set_values:
            if a.kind is not TargetKind.COMPARTMENT:
                continue
            kept = {
                sid: self.value(f"[{sid}]")
                for sid, cid in symbols.species_compartment.items()
                if cid == a.target
                and sid not in amounts
                and sid not in symbols.assignment_rules
                and (sid in symbols.initial_concentration or sid in concentrations)
            }
            self.set_value(a.target, value)
            for sid, concentration in kept.items():
                self.set_value(f"[{sid}]", concentration)
        for a, value in set_values:
            if (
                a.kind is not TargetKind.COMPARTMENT
                and symbols.entity(a.target) not in symbols.assignment_rules
            ):
                self.set_value(a.target, value)
        return {s: self.value(s) for s in read}

    @override
    def reset(self) -> None:
        """Set the model to the state it was loaded in, bitwise, without a simulation."""
        self.initialize([])

    @override
    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization.

        Nothing is recorded: an initialization starts from the free
        parameters as loaded whatever ran before it, see `initialize`.
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
        return self._loaded[selection]

    @override
    def has_selection(self, selection: str) -> bool:
        """Check whether AMICI has a selection of a name in the model.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it.
        """
        return self._resolves(selection)

    @override
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, `default_selections` for `None`.

        Raises:
            ValueError: if AMICI has no selection of a name.
        """
        chosen = self.default_selections() if selections is None else list(selections)
        for selection in chosen:
            if not self._resolves(selection):
                raise ValueError(self._selection_error(selection))
        self.selections = chosen
        self.selections_given = selections is not None

    @override
    def set_integrator_settings(self, **settings: IntegratorSetting) -> None:
        """Set settings of the integrator, CVODES of AMICI.

        The settings have the names of roadrunner: `relative_tolerance`,
        `absolute_tolerance`, `maximum_num_steps` and `maximum_time_step`.
        AMICI has one absolute tolerance, the smallest tolerance of an
        `AbsoluteTolerance` (of the kinds and of the overrides, reported once
        when they differ) times `AMICI_ATOL_FACTOR`, so AMICI integrates at
        least as accurately as roadrunner at equal settings. A setting of
        roadrunner which AMICI has no
        equivalent of, e.g. `variable_step_size`, is ignored and reported
        once per model.

        Args:
            **settings: the settings by their names in roadrunner.

        Raises:
            ValueError: if neither AMICI nor roadrunner has a setting of a
                name, or a value is not valid.
        """
        solver = self.solver
        for key, value in self._settings.checked(settings).items():
            if isinstance(value, AbsoluteTolerance):
                solver.set_absolute_tolerance(
                    self._settings.scalar_tolerance(value, AMICI_ATOL_FACTOR)
                )
                self.absolute_tolerance = value
            elif key == "maximum_num_steps":
                solver.set_max_steps(int(value))
            elif key == "relative_tolerance":
                solver.set_relative_tolerance(float(value))
            else:
                solver.set_max_step_size(float(value))

    @override
    def integrator_state(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Get the name of the integrator of the model and every one of its settings.

        The settings have the names of roadrunner, the absolute tolerance is
        the `AbsoluteTolerance` of the model as it was set; the one absolute
        tolerance of AMICI is its smallest tolerance, see
        `set_integrator_settings`.

        Returns:
            The name of the integrator and its settings, sorted by name.
        """
        solver = self.solver
        return INTEGRATOR, (
            ("absolute_tolerance", self.absolute_tolerance),
            ("maximum_num_steps", int(solver.get_max_steps())),
            ("maximum_time_step", float(solver.get_max_step_size())),
            ("relative_tolerance", float(solver.get_relative_tolerance())),
        )

    @override
    def set_integrator_state(
        self, state: tuple[str, tuple[tuple[str, Any], ...]]
    ) -> None:
        """Set the settings of the integrator, see `integrator_state`.

        Args:
            state: the name of the integrator, `cvodes`, and its settings.

        Raises:
            ValueError: if the integrator is not the one of AMICI, or it has
                no setting of a name.
        """
        integrator, settings = state
        if integrator != INTEGRATOR:
            raise ValueError(
                f"AMICI integrates with '{INTEGRATOR}', not with '{integrator}'."
            )
        self.set_integrator_settings(**dict(settings))

    @override
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates.

        The species which reactions or a rate rule change, in the order of
        the SBML, then the other targets of rate rules, in the order of the
        SBML; the states sbmlsim adds to AMICI, whose rate is 0, are not
        part of them.
        """
        return list(self._state_ids)


def _integrated(model: libsbml.Model, symbols: ModelSymbols) -> tuple[str, ...]:
    """Get the ids of the states which a model integrates, see `AmiciSBMLModel.state_ids`."""
    states = [
        s.getId()
        for s in model.getListOfSpecies()
        if s.getId() not in symbols.assignment_rules
        and (
            s.getId() in symbols.rate_rules
            or not (s.getBoundaryCondition() or s.getConstant())
        )
    ]
    states += rate_rule_entities(model, symbols)
    return tuple(states)


def _check_supported(model: libsbml.Model, label: str) -> None:
    """Check that AMICI simulates a model as roadrunner does.

    AMICI does not import the delay csymbol, a delay of an event, a
    priority of an event which is no number, a trigger which is not
    persistent, a stoichiometry math and a fast reaction, and fails to build
    a model which reads the csymbol rateOf (in an assignment rule, AMICI
    1.2); it integrates a model with algebraic rules as a system of
    differential algebraic equations, which the executor does not, and makes
    the stoichiometry of a species reference which is not constant, which
    roadrunner changes, and an initial assignment of a species reference or
    of a conversion factor a math which follows its parameters during a
    simulation instead of a number. A trigger which compares for equality,
    `a == b` or `a != b`, has no root AMICI finds (the root function of
    `a - b` touches 0 without a change of its sign), and roadrunner fires it
    only where a step of its integrator lands on the equality, so its result
    depends on the output times. These raise before AMICI compiles the
    model; an error of AMICI on another feature raises when it imports the
    model, see `_compile`.

    Args:
        model: the model of the SBML.
        label: the name of the model in a message.

    Raises:
        ValueError: naming the feature and its element, for a model with one.
    """

    def unsupported(feature: str) -> ValueError:
        return ValueError(
            f"The model '{label}' has {feature}; AMICI does not simulate it as "
            f"roadrunner, the model cannot be simulated with AMICI."
        )

    for feature, element in variable_features(model):
        reason = (
            ", which AMICI integrates as a differential algebraic equation"
            if feature is Feature.ALGEBRAIC_RULE
            else ""
        )
        raise unsupported(element + reason)
    for reaction in model.getListOfReactions():
        if reaction.isSetFast() and reaction.getFast():
            raise unsupported(f"the fast reaction '{reaction.getId()}'")
    factors = conversion_factors(model)
    for assignment in model.getListOfInitialAssignments():
        if assignment.getSymbol() in factors:
            raise unsupported(
                f"an initial assignment of the conversion factor "
                f"'{assignment.getSymbol()}'"
            )
    level = model.getLevel()
    functions: libsbml.ListOfFunctionDefinitions = model.getListOfFunctionDefinitions()
    for event in model.getListOfEvents():
        eid = event.getId()
        if event.isSetDelay():
            raise unsupported(f"the delay of the event '{eid}'")
        if event.isSetPriority():
            priority = event.getPriority().getMath()
            if priority is not None and not priority.isNumber():
                raise unsupported(
                    f"the priority of the event '{eid}', which is no number"
                )
        trigger = event.getTrigger()
        if level >= 3 and trigger is not None and not trigger.getPersistent():
            raise unsupported(
                f"the trigger of the event '{eid}', which is not persistent"
            )
        if trigger is not None and trigger.isSetMath():
            node: libsbml.ASTNode = trigger.getMath().deepCopy()
            if functions.size():
                libsbml.SBMLTransforms.replaceFD(node, functions)
            for _, _, comparison in math_comparisons(node, EQUALITIES):
                raise unsupported(
                    f"the trigger of the event '{eid}', which compares for "
                    f"equality ('{libsbml.formulaToL3String(comparison)}')"
                )
    for where, ast in sbml_maths(model):
        if reads_rate_of(ast):
            raise unsupported(f"the csymbol rateOf in the {where}")


def amici_sbml(sbml: str, symbols: ModelSymbols, label: str) -> AmiciSBML:
    """Get the SBML which AMICI compiles, see the module.

    Args:
        sbml: the SBML of the model, with the added parameters.
        symbols: the symbols of the model.
        label: the name of the model in a message.

    Returns:
        The SBML and what sbmlsim changed in it.

    Raises:
        ValueError: if the model has an entity of the id of a parameter of
            sbmlsim, or a model of a level before 3 cannot be converted to
            level 3.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if doc.getLevel() < 3 and not doc.setLevelAndVersion(3, 2, False):
        # the initial assignments and the attributes of the triggers which
        # sbmlsim sets are elements of level 3 (initial assignments from
        # level 2 version 2), libsbml converts the semantics of the levels
        raise ValueError(
            f"The model '{label}' of level {doc.getLevel()} version "
            f"{doc.getVersion()} cannot be converted to SBML level 3 version 2 "
            f"for AMICI."
        )
    model: libsbml.Model = doc.getModel()
    level = doc.getLevel()
    rules = symbols.assignment_rules
    factors = conversion_factors(model)
    event_targets = {
        assignment.getVariable()
        for event in model.getListOfEvents()
        for assignment in event.getListOfEventAssignments()
    }
    initial: dict[str, libsbml.ASTNode] = {
        a.getSymbol(): a.getMath()
        for a in model.getListOfInitialAssignments()
        if a.getMath() is not None
    }

    flagged: list[str] = []
    targets: list[str] = []
    not_a_number: set[str] = set()
    for parameter in model.getListOfParameters():
        pid = parameter.getId()
        if pid in rules or pid in factors:
            continue
        if pid in initial or pid in symbols.rate_rules or pid in event_targets:
            flagged.append(pid)
            targets.append(pid)
    for compartment in model.getListOfCompartments():
        cid = compartment.getId()
        if cid in rules:
            continue
        size = compartment.getSize() if compartment.isSetSize() else 1.0
        if cid not in initial and not math.isfinite(size):
            not_a_number.add(cid)
            continue
        flagged.append(cid)
        targets.append(cid)
    flagged += [s.getId() for s in model.getListOfSpecies() if s.getId() not in rules]

    for sid in flagged:
        entity = model.getElementBySId(sid)
        for pid in (f"{VALUE_PREFIX}{sid}", f"{KIND_PREFIX}{sid}"):
            if model.getElementBySId(pid) is not None:
                raise ValueError(
                    f"The model '{label}' has the entity '{pid}', the id of a "
                    f"parameter sbmlsim adds for AMICI."
                )
            parameter: libsbml.Parameter = model.createParameter()
            parameter.setId(pid)
            parameter.setValue(0.0)
            parameter.setConstant(True)
        math_ = _initial_math(entity, initial.get(sid))
        if sid in initial:
            model.removeInitialAssignment(sid)
        assignment: libsbml.InitialAssignment = model.createInitialAssignment()
        assignment.setSymbol(sid)
        assignment.setMath(math_)

    # AMICI finds the roots of the triggers, the executor fires the events,
    # see the module: an event keeps its trigger, with every strict comparison
    # shifted by a parameter, and records instead of its assignments that it
    # fired and the time of its first execution
    events: list[str] = []
    comparisons: list[StrictComparison] = []
    concentrations = frozenset(symbols.species - symbols.only_substance)
    functions: libsbml.ListOfFunctionDefinitions = model.getListOfFunctionDefinitions()

    def added(sid: str) -> None:
        if model.getElementBySId(sid) is not None:
            raise ValueError(
                f"The model '{label}' has the entity '{sid}', the id of an "
                f"element sbmlsim adds for AMICI."
            )

    def parameter(sid: str, constant: bool) -> None:
        added(sid)
        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setValue(0.0)
        parameter.setConstant(constant)

    for k, event in enumerate(model.getListOfEvents()):
        added(f"{EVENT_PREFIX}{k}")
        if not event.isSetId():
            event.setId(f"{EVENT_PREFIX}{k}")
        events.append(event.getId())
        trigger = event.getTrigger()
        if trigger is not None and trigger.isSetMath():
            node: libsbml.ASTNode = trigger.getMath().deepCopy()
            if functions.size():
                libsbml.SBMLTransforms.replaceFD(node, functions)
            for shifted, (_, _, comparison) in enumerate(trigger_comparisons(node)):
                left, right = _operands(comparison, concentrations, functions)
                comparisons.append(StrictComparison(event=k, left=left, right=right))
                pid = f"{SHIFT_PREFIX}{k}_{shifted}"
                parameter(pid, constant=True)
                shift_comparison(comparison, pid)
            trigger.setMath(node)
        while event.getNumEventAssignments():
            event.removeEventAssignment(0)
        fired_id, first_id = f"{FIRED_PREFIX}{k}", f"{FIRST_PREFIX}{k}"
        parameter(fired_id, constant=False)
        parameter(first_id, constant=False)
        fired: libsbml.EventAssignment = event.createEventAssignment()
        fired.setVariable(fired_id)
        fired.setMath(_real(1.0))
        # the time at the first execution, unchanged by the later ones:
        # `first + (time - first) * (1 - fired)`, the values of the state at
        # the trigger
        first: libsbml.EventAssignment = event.createEventAssignment()
        first.setVariable(first_id)
        first.setMath(
            _apply(
                libsbml.AST_PLUS,
                _name(first_id),
                _apply(
                    libsbml.AST_TIMES,
                    _apply(
                        libsbml.AST_MINUS,
                        libsbml.ASTNode(libsbml.AST_NAME_TIME),
                        _name(first_id),
                    ),
                    _apply(libsbml.AST_MINUS, _real(1.0), _name(fired_id)),
                ),
            )
        )

    # every comparison of a math AMICI integrates is shifted by a parameter,
    # so that it keeps its value at its threshold until the operands move,
    # see the module
    switches: list[Switch] = []
    maths: list[tuple[Any, libsbml.ASTNode, dict[str, float]]] = []
    for reaction in model.getListOfReactions():
        law: libsbml.KineticLaw | None = reaction.getKineticLaw()
        if law is not None and law.isSetMath():
            maths.append(
                (
                    law,
                    law.getMath(),
                    {p.getId(): p.getValue() for p in law.getListOfLocalParameters()},
                )
            )
    for rule in model.getListOfRules():
        if rule.isSetMath() and not rule.isAlgebraic():
            maths.append((rule, rule.getMath(), {}))
    for element, math_, local in maths:
        node = math_.deepCopy()
        if functions.size():
            libsbml.SBMLTransforms.replaceFD(node, functions)
        _substitute_numbers(node, local)
        found = math_comparisons(node, (*COMPARISONS, *EQUALITIES))
        if not found:
            continue
        # the comparisons inside the operands of another one first, so the
        # replacement of the other one copies them shifted
        for parent, position, comparison in reversed(found):
            pid = f"{SWITCH_PREFIX}{len(switches)}"
            switches.append(Switch(*_operands(comparison, concentrations, functions)))
            parameter(pid, constant=True)
            if comparison.getType() in COMPARISONS:
                shift_comparison(comparison, pid)
                continue
            replacement = _shifted_equality(comparison, pid)
            if parent is None:
                node = replacement
            else:
                parent.replaceChild(position, replacement, True)
        element.setMath(node)

    if targets:
        for sid in targets:
            model.getElementBySId(sid).setConstant(False)
        event: libsbml.Event = model.createEvent()
        event.setId(TARGETS_EVENT)
        event.setUseValuesFromTriggerTime(True)
        trigger: libsbml.Trigger = event.createTrigger()
        trigger.setMath(libsbml.ASTNode(libsbml.AST_CONSTANT_FALSE))
        if level >= 3:
            trigger.setPersistent(True)
        for sid in targets:
            target: libsbml.EventAssignment = event.createEventAssignment()
            target.setVariable(sid)
            target.setMath(_name(sid))

    initial_value_false: list[str] = []
    if level >= 3:
        for event in model.getListOfEvents():
            trigger = event.getTrigger()
            if trigger is None:
                continue
            if not trigger.getInitialValue() and event.getId() != TARGETS_EVENT:
                initial_value_false.append(event.getId())
            trigger.setInitialValue(True)

    return AmiciSBML(
        sbml=libsbml.writeSBMLToString(doc),
        flagged=frozenset(flagged),
        not_a_number=frozenset(not_a_number),
        initial_value_false=tuple(initial_value_false),
        events=tuple(events),
        comparisons=tuple(comparisons),
        switches=tuple(switches),
    )


def shift_ids(comparisons: Sequence[StrictComparison]) -> list[str]:
    """Get the ids of the parameters of the shifts of strict comparisons, see `SHIFT_PREFIX`."""
    counts: dict[int, int] = {}
    ids = []
    for comparison in comparisons:
        j = counts.get(comparison.event, 0)
        counts[comparison.event] = j + 1
        ids.append(f"{SHIFT_PREFIX}{comparison.event}_{j}")
    return ids


def switch_ids(count: int) -> list[str]:
    """Get the ids of the parameters of the shifts of the comparisons of the maths, see `SWITCH_PREFIX`."""
    return [f"{SWITCH_PREFIX}{j}" for j in range(count)]


#: the comparisons for equality, which a trigger must not have, see
#: `_check_supported`
EQUALITIES = (libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ)


def _operands(
    comparison: libsbml.ASTNode,
    concentrations: frozenset[str],
    functions: libsbml.ListOfFunctionDefinitions,
) -> tuple[str, str]:
    """Get the operands of a comparison as MathML of selections, see `selection_math`."""
    return (
        str(selection_math(comparison.getChild(0), concentrations, functions)),
        str(selection_math(comparison.getChild(1), concentrations, functions)),
    )


def _shifted_equality(comparison: libsbml.ASTNode, pid: str) -> libsbml.ASTNode:
    """Get a comparison for equality shifted by a parameter, see the module.

    `a == b` becomes `a <= b + shift && a >= b - shift` and `a != b`
    `a < b - shift || a > b + shift`.

    Args:
        comparison: the node of the comparison.
        pid: the id of the parameter of the shift.
    """
    equal = comparison.getType() == libsbml.AST_RELATIONAL_EQ
    left: libsbml.ASTNode = comparison.getChild(0)
    right: libsbml.ASTNode = comparison.getChild(1)
    plus, minus = libsbml.AST_PLUS, libsbml.AST_MINUS
    below = _apply(
        libsbml.AST_RELATIONAL_LEQ if equal else libsbml.AST_RELATIONAL_LT,
        left.deepCopy(),
        _apply(plus if equal else minus, right.deepCopy(), _name(pid)),
    )
    above = _apply(
        libsbml.AST_RELATIONAL_GEQ if equal else libsbml.AST_RELATIONAL_GT,
        left.deepCopy(),
        _apply(minus if equal else plus, right.deepCopy(), _name(pid)),
    )
    return _apply(
        libsbml.AST_LOGICAL_AND if equal else libsbml.AST_LOGICAL_OR, below, above
    )


def _substitute_numbers(math: libsbml.ASTNode, numbers: dict[str, float]) -> None:
    """Replace identifiers of a math by numbers, e.g. the local parameters of a kinetic law."""
    if not numbers:
        return
    stack = [math]
    while stack:
        node = stack.pop()
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
        if node.isName() and not node.getDefinitionURLString():
            value = numbers.get(node.getName())
            if value is not None:
                node.setType(libsbml.AST_REAL)
                node.setValue(float(value))


def _name(sid: str) -> libsbml.ASTNode:
    """Get the math of an identifier."""
    node = libsbml.ASTNode(libsbml.AST_NAME)
    node.setName(sid)
    return node


def _real(value: float) -> libsbml.ASTNode:
    """Get the math of a number."""
    node = libsbml.ASTNode(libsbml.AST_REAL)
    node.setValue(float(value))
    return node


def _apply(kind: int, *children: libsbml.ASTNode) -> libsbml.ASTNode:
    """Get the math of an operation of its arguments."""
    node = libsbml.ASTNode(kind)
    for child in children:
        node.addChild(child)
    return node


def _initial_math(entity: Any, initial: libsbml.ASTNode | None) -> libsbml.ASTNode:
    """Get the initial value of an entity of the parameters of sbmlsim, see the module.

    Args:
        entity: a parameter, a compartment or a species of the SBML.
        initial: the math of its initial assignment, `None` without one.

    Returns:
        `piecewise(value as amount, kind == 1, value as concentration, kind
        == 2, initial value of the SBML)`, in the value AMICI keeps of the
        entity: the concentration of a species whose `hasOnlySubstanceUnits`
        is false, the amount of the other species.
    """
    sid = entity.getId()
    value, kind = _name(f"{VALUE_PREFIX}{sid}"), _name(f"{KIND_PREFIX}{sid}")
    if isinstance(entity, libsbml.Species):
        compartment = entity.getCompartment()
        only_substance = entity.getHasOnlySubstanceUnits()
        if initial is not None:
            original = initial.deepCopy()
        elif entity.isSetInitialAmount():
            original = _real(entity.getInitialAmount())
            if not only_substance:
                original = _apply(libsbml.AST_DIVIDE, original, _name(compartment))
        elif entity.isSetInitialConcentration():
            original = _real(entity.getInitialConcentration())
            if only_substance:
                original = _apply(libsbml.AST_TIMES, original, _name(compartment))
        else:
            original = _real(0.0)
        if only_substance:
            as_amount = value.deepCopy()
            as_concentration = _apply(
                libsbml.AST_TIMES, value.deepCopy(), _name(compartment)
            )
        else:
            as_amount = _apply(libsbml.AST_DIVIDE, value.deepCopy(), _name(compartment))
            as_concentration = value.deepCopy()
        return _apply(
            libsbml.AST_FUNCTION_PIECEWISE,
            as_amount,
            _apply(libsbml.AST_RELATIONAL_EQ, kind.deepCopy(), _real(_KIND_VALUE)),
            as_concentration,
            _apply(
                libsbml.AST_RELATIONAL_EQ, kind.deepCopy(), _real(_KIND_CONCENTRATION)
            ),
            original,
        )
    if initial is not None:
        original = initial.deepCopy()
    elif isinstance(entity, libsbml.Compartment):
        original = _real(entity.getSize() if entity.isSetSize() else 1.0)
    else:
        original = _real(entity.getValue() if entity.isSetValue() else 0.0)
    return _apply(
        libsbml.AST_FUNCTION_PIECEWISE,
        value,
        _apply(libsbml.AST_RELATIONAL_EQ, kind, _real(_KIND_VALUE)),
        original,
    )


def cache_root() -> Path:
    """Get the directory of the compiled models.

    `SBMLSIM_AMICI_CACHE` if it is set, else `~/.cache/sbmlsim/amici`.
    """
    root = os.environ.get(CACHE_ENV)
    if root:
        return Path(root)
    return Path.home() / ".cache" / "sbmlsim" / "amici"


def cache_key(sbml: str) -> str:
    """Get the key of a compiled model in the cache.

    The SHA-256 of the SBML AMICI compiles, the version of AMICI, the
    version and the implementation of python, the machine, the options of the
    compilation and `CACHE_FORMAT`.

    Args:
        sbml: the SBML AMICI compiles, see `amici_sbml`.

    Returns:
        The key, a hexadecimal digest.
    """
    amici = import_amici()
    digest = hashlib.sha256()
    for part in (
        sbml,
        str(amici.__version__),
        sys.version,
        sys.implementation.cache_tag or "",
        platform.machine(),
        repr(sorted(IMPORT_OPTIONS.items())),
        str(CACHE_FORMAT),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def module_name(key: str) -> str:
    """Get the name of the module of a compiled model, see `cache_key`."""
    return f"sbmlsim_amici_{key[:32]}"


def compiled_dir(sbml: str, label: str = "model") -> Path:
    """Get the directory of the compiled model of an SBML, compiling it if it is not cached.

    The model is compiled into a temporary directory next to its directory
    in the cache (`cache_root`, `cache_key`), which is renamed to it when
    the compilation is complete, so the directory of a model is complete if
    it exists. The compilation runs under a lock of the key, so processes
    which need the same model at once compile it once.

    Args:
        sbml: the SBML AMICI compiles, see `amici_sbml`.
        label: the name of the model in a message.

    Returns:
        The directory of the compiled model.

    Raises:
        ValueError: if AMICI cannot import the model.
        RuntimeError: if AMICI fails to compile the model.
    """
    key = cache_key(sbml)
    root = cache_root()
    target = root / key
    if target.is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root / f"{key}.lock"):
        if not target.is_dir():
            # the directories of a compilation of the key which did not end,
            # e.g. of a process which was killed; no other process compiles
            # the key while the lock is held
            for left in root.glob(f"{key}.*"):
                if left.is_dir():
                    shutil.rmtree(left, ignore_errors=True)
            _compile(sbml, key, target, label)
    return target


def load_module(sbml: str, label: str = "model") -> ModuleType:
    """Get the module of the compiled model of an SBML, see `compiled_dir`.

    A module is imported once per process.

    Args:
        sbml: the SBML AMICI compiles, see `amici_sbml`.
        label: the name of the model in a message.

    Returns:
        The module of the model, whose `get_model` creates a model of AMICI.
    """
    directory = compiled_dir(sbml, label)
    name = module_name(directory.name)
    module = _MODULES.get(name)
    if module is None:
        amici = import_amici()
        module = amici.import_model_module(name, directory)
        _MODULES[name] = module
    return module


def _compile(sbml: str, key: str, target: Path, label: str) -> None:
    """Compile a model with AMICI into its directory in the cache.

    Args:
        sbml: the SBML AMICI compiles.
        key: the key of the model, see `cache_key`.
        target: the directory of the model in the cache, which must not
            exist.
        label: the name of the model in a message.

    Raises:
        ValueError: if AMICI cannot import the model.
        RuntimeError: if a tool of the build is missing, or AMICI fails to
            compile the model.
    """
    amici = import_amici()
    sbml_errors = (
        importlib.import_module("amici.importers.utils").SBMLException,
        NotImplementedError,
        ValueError,
        KeyError,
        TypeError,
    )
    name = module_name(key)
    building = Path(tempfile.mkdtemp(prefix=f"{key}.", dir=target.parent))
    logger.info("AMICI compiles the model '%s' into '%s'.", label, target)
    start = time_module.perf_counter()
    try:
        try:
            importer = amici.SbmlImporter(sbml, from_file=False)
        except sbml_errors as err:
            raise ValueError(f"AMICI cannot import the model '{label}': {err}") from err
        # AMICI prints the output of a failed build, which the error carries
        with (
            _build_tools_on_path(),
            _quiet(),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if shutil.which(BUILD_TOOL) is None:
                raise RuntimeError(
                    f"AMICI builds a model with '{BUILD_TOOL}', which is not on the "
                    f"path; it is part of the extra amici, `pip install "
                    f"sbmlsim[amici]`."
                )
            try:
                importer.sbml2amici(
                    name,
                    building,
                    verbose=logging.ERROR,
                    **IMPORT_OPTIONS,
                )
            except subprocess.CalledProcessError as err:
                output = err.output.decode("utf-8", "replace") if err.output else ""
                missing = re.search(r"Required command '([^']+)' not found", output)
                if missing:
                    raise RuntimeError(
                        f"AMICI builds a model with '{missing.group(1)}', which is "
                        f"not on the path."
                    ) from err
                raise RuntimeError(
                    f"AMICI fails to compile the model '{label}': {output[-2000:]}"
                ) from err
            except OSError as err:
                raise RuntimeError(
                    f"AMICI cannot build the model '{label}', a tool of the build "
                    f"is missing or fails: {err}"
                ) from err
            except sbml_errors as err:
                raise ValueError(
                    f"AMICI cannot import the model '{label}': {err}"
                ) from err
        # the objects of the build are not needed by the module
        shutil.rmtree(building / "build", ignore_errors=True)
        building.rename(target)
    finally:
        shutil.rmtree(building, ignore_errors=True)
    logger.info(
        "AMICI compiled the model '%s' in %.1f s.",
        label,
        time_module.perf_counter() - start,
    )


@contextlib.contextmanager
def _build_tools_on_path() -> Generator[None]:
    """Put the scripts of the python environment first on the path during a compilation.

    AMICI builds a model with `ninja`, which the extra `amici` installs into
    the scripts of the environment, which are not on the path of a python
    which is not run from an activated environment.
    """
    before = os.environ.get("PATH")
    scripts = sysconfig.get_path("scripts")
    os.environ["PATH"] = os.pathsep.join(p for p in (scripts, before) if p)
    try:
        yield
    finally:
        if before is None:
            del os.environ["PATH"]
        else:
            os.environ["PATH"] = before


@contextlib.contextmanager
def _quiet() -> Generator[None]:
    """Keep the loggers of AMICI at the level of errors during a compilation."""
    amici_logger = logging.getLogger("amici")
    level = amici_logger.level
    amici_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        amici_logger.setLevel(level)


@contextlib.contextmanager
def _locked(path: Path) -> Generator[None]:
    """Hold the lock of a file, which other processes wait for.

    POSIX locks the file with `fcntl.flock`; windows creates it exclusively
    and waits while it exists, a lock older than an hour is taken as left
    over by a process which ended.
    """
    if sys.platform == "win32":
        while True:
            try:
                handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time_module.time() - path.stat().st_mtime > 3600:
                        path.unlink()
                time_module.sleep(0.5)
                continue
            break
        try:
            yield
        finally:
            os.close(handle)
            with contextlib.suppress(OSError):
                path.unlink()
    else:
        import fcntl

        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_simulation() -> Any:
    """Get `run_simulation` of AMICI, which integrates a model with a solver."""
    import_amici()
    return importlib.import_module("amici.sim.sundials").run_simulation
