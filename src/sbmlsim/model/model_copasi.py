"""The model of an SBML model in COPASI, see `sbmlsim.model.model_sbml`.

`CopasiSBMLModel` loads the SBML of a model into a data model of COPASI
(python-copasi, the optional extra `copasi`), which it owns until it is
released (`close`). Its values are read and written in the selections of
roadrunner, `S` the amount of a species, `[S]` its concentration, the id of a
parameter or a compartment, the id of a reaction its rate and `time`, with the
numbers of the SBML: COPASI holds the amount of a species as a particle
number, which the model converts with the factor of COPASI from a quantity to
a number.

COPASI reads a model with its own units, without converting the numbers of
the SBML (a species in other units than the model keeps its number, as in
roadrunner). The model checks after the import that every plain initial
number of the SBML is the one of COPASI and raises otherwise, e.g. for an
amount whose particle number is not finite. It raises as well for an error of
the import (of COPASI or of libsbml) and for the features COPASI loads and
simulates otherwise than roadrunner, see `_check_supported`. COPASI
multiplies the stoichiometries of a species by its conversion factor when it
imports a model, so a model with conversion factors is compiled again after
the import and a plan which changes a conversion factor is refused, see
`_compile_conversion_factors`.

COPASI folds a constant entity into the constants of its model, a value
written into it during a simulation is not seen. The SBML which COPASI loads
therefore makes every entity which a change can set (the parameters,
compartments and species which are not the variable of an assignment rule)
non-constant and the target of one event which never fires
(`TARGETS_EVENT`), which makes them part of the state of COPASI, where a
change sets them (`_set_current_values`); an entity whose number in the SBML
is not finite (`NaN`) is left out, COPASI integrates no state which is not
finite, and a change of it raises.

LSODA in COPASI finds no crossing of a comparison whose root function is
exactly 0 when it starts. In a model with events, the SBML which COPASI loads
therefore has every comparison of a trigger and of a math COPASI integrates
shifted by a parameter (`SHIFT_PREFIX`, `SWITCH_PREFIX`), which is 0 but
before an integration from a state in which a comparison of a trigger is
exactly at its threshold, see `CopasiSBMLModel.shift_comparisons`.

A result of COPASI is reproducible within the integration and not bitwise
across loads of a model: COPASI sums the fluxes of the reactions of a species
into its rate in an order which depends on the addresses of its objects, i.e.
on the memory the process allocated and freed before the load (a released
model, whose data model is removed, or any memory of python), so the rate of
a species of three or more reactions differs in its last bits and the
integration amplifies it, on the HCTZ model with a dose to a few tens of
times the tolerances of the integrator (copasi/basico#77). A loaded model is
bitwise reproducible, also when another model is released.
"""

from __future__ import annotations

import logging
import math
import sys
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import libsbml
import numpy as np

from sbmlsim.engine import (
    Engine,
    UnsupportedModelError,
    import_copasi,
    kept_encoding,
)
from sbmlsim.model.comparisons import (
    COMPARISONS,
    STRICT_COMPARISONS,
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
    variable_features,
)
from sbmlsim.model.symbols import TargetKind
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.units import UnitRegistry

# the optional extra, its error names the extra
import_copasi()
import COPASI  # noqa: E402

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

logger = logging.getLogger(__name__)

#: the id of the event which makes the entities a change sets part of the state
TARGETS_EVENT = "__sbmlsim_targets"

#: the prefix of the parameter by which a comparison of the trigger of an
#: event is shifted, by the position of the event and of the comparison in its
#: trigger, see `CopasiSBMLModel.shift_comparisons`
SHIFT_PREFIX = "__sbmlsim_shift_"

#: the prefix of the parameter by which a comparison of a math COPASI
#: integrates is shifted, by its position among them, see
#: `CopasiSBMLModel.shift_comparisons`
SWITCH_PREFIX = "__sbmlsim_switch_"

#: the units in the last place of its threshold, and of the time, by which a
#: comparison exactly at its threshold on a plateau is shifted, see
#: `CopasiSBMLModel._threshold_shift`: more than LSODA steps over after a root
#: of another function which is exactly 0 where LSODA found it, e.g. the end
#: of a plateau of a rate at `time < 1.5` (`DRCHEK` takes the root functions
#: about a hundred units in the last place of the time later and misses a
#: crossing in between), and less than the error of LSODA in the location of
#: a root
THRESHOLD_ULPS = 1024.0

#: the shift of a comparison by which its roots are told, see
#: `CopasiSBMLModel._comparison_roots`
_PROBE_SHIFT = 1e300

#: the name of the integrator, the method `deterministic` of COPASI
INTEGRATOR = "lsoda"

#: the settings of roadrunner and the parameters of LSODA in COPASI
LSODA_PARAMETERS: dict[str, str] = {
    "relative_tolerance": "Relative Tolerance",
    "absolute_tolerance": "Absolute Tolerance",
    "maximum_num_steps": "Max Internal Steps",
    "maximum_time_step": "Max Internal Step Size",
}

#: the factor from the absolute tolerance of roadrunner to the one of LSODA
#: in COPASI, so COPASI integrates at least as accurately as roadrunner at
#: equal settings. roadrunner gives every state the absolute tolerance of its
#: kind, a concentration species its tolerance times the volume of its
#: compartment, see `sbmlsim.model.tolerances`. COPASI has one tolerance,
#: which `CLsodaMethod::initializeAtolVector` scales per state: a species by
#: `min(V * Q2N, |N0|)` (V the volume of its compartment, Q2N the factor from
#: amounts to particle numbers, N0 its initial particle number), so its
#: tolerance shrinks with a small initial amount, and a global quantity or a
#: compartment by `min(1, |x0|)`. Measured on the HCTZ model, doses of 10,
#: 25 and 50 mg over 24 hr against roadrunner at an absolute tolerance of
#: 1e-16 and a relative one of 1e-12, the largest error relative to the
#: largest value of a variable: roadrunner 7.6e-8 at the defaults of
#: `Simulator` (1e-10 and 1e-10), COPASI 1.4e-6 with the tolerance as it is
#: (3.6e-4 relative to a value) and 1.8e-8 with 1e-12, i.e. the factor 1e-2.
COPASI_ATOL_FACTOR: float = 1e-2

#: the messages of COPASI which do not change the numbers of a model, logged
#: at the level `DEBUG`: the default units of the model (91, 92), the units of
#: species and of the model which COPASI does not import and whose numbers it
#: keeps (25, 66), a compartment without a size, which has the size 1 as in
#: roadrunner (45), a compartment without spatial dimensions, which COPASI
#: reads as one of three, whose size is the same number (90), the
#: stoichiometries COPASI multiplies by the conversion factors, which the
#: model compiles (100, see `_compile_conversion_factors`), an unknown
#: annotation and the warnings of libsbml
BENIGN_MESSAGES = (
    "SBML (100)",
    "SBML (90)",
    "SBML (91)",
    "SBML (92)",
    "SBML (25)",
    "SBML (66)",
    "SBML (45)",
    "CAnnotation (4)",
    "LIBSBML WARNING",
)

#: the types of the messages of COPASI which are errors
_ERROR_TYPES = frozenset(
    {
        COPASI.CCopasiMessage.ERROR,
        COPASI.CCopasiMessage.EXCEPTION,
        COPASI.CCopasiMessage.ERROR_FILTERED,
        COPASI.CCopasiMessage.EXCEPTION_FILTERED,
    }
)


class _Kind(StrEnum):
    """The kinds of a selection."""

    TIME = "time"
    AMOUNT = "amount"
    CONCENTRATION = "concentration"
    VALUE = "value"
    FLUX = "flux"


@dataclass(frozen=True)
class _Shifted:
    """A comparison shifted by a parameter, see `_with_shifted_comparisons`.

    Attributes:
        pid: the id of the parameter of the shift.
        strict: whether the comparison is strict, `>` or `<`, false at its
            threshold; `>=` and `<=` are true there.
        event: the position of the event of the trigger of the comparison
            among the events of the model, `None` for a comparison of a math
            COPASI integrates, e.g. the condition of a piecewise of a rate.
    """

    pid: str
    strict: bool
    event: int | None

    @property
    def trigger(self) -> bool:
        """Whether it is a comparison of the trigger of an event."""
        return self.event is not None


@dataclass(frozen=True)
class _Selection:
    """A selection of roadrunner in COPASI.

    Attributes:
        kind: the kind of the selection, see `_Kind`.
        reference: the object of COPASI of its value, a particle number for an
            amount.
        entity: the id of the entity, `None` for the time.
    """

    kind: _Kind
    reference: Any
    entity: str | None


def drain_messages(label: str) -> None:
    """Take the messages of COPASI, raise for an error and log the others.

    COPASI collects its messages in a queue of the process instead of
    printing them, so the messages of a call of COPASI are taken right after
    it. An error of COPASI and an error of libsbml, which COPASI passes on as
    a message of another type, raise; a message which is known not to change
    the numbers (`BENIGN_MESSAGES`) is logged at the level `DEBUG`, every
    other one as a warning.

    Args:
        label: the name of the model in a message.

    Raises:
        ValueError: with the text of the errors, if a message is an error.
    """
    errors: list[str] = []
    while COPASI.CCopasiMessage.size():
        message = COPASI.CCopasiMessage.getFirstMessage()
        text = str(message.getText()).strip()
        if message.getType() in _ERROR_TYPES or "LIBSBML ERROR" in text:
            errors.append(text)
        elif any(benign in text for benign in BENIGN_MESSAGES):
            logger.debug("COPASI on the model '%s': %s", label, text)
        else:
            logger.warning("COPASI on the model '%s': %s", label, text)
    if errors:
        raise ValueError(
            f"COPASI reports errors on the model '{label}': " + " ".join(errors)
        )


def _set_current_values(
    mc: COPASI.CMathContainer, values: Sequence[tuple[Any, float]], label: str
) -> None:
    """Set current values of the state of a math container of COPASI, between two integrations.

    The python bindings have no setter of a value of the state of a math
    container (copasi/basico#79): the state is pushed into the objects of the
    model (`pushState`), the values are set on them, the time with
    `CModel.setTime` and an entity with `CModelEntity.setValue`, and the state
    is fetched back (`fetchState`), which copies the doubles, so a value is
    set bitwise. Every value is read back: a value which COPASI does not take,
    e.g. of an entity which is not part of its state, raises. The caller
    updates the values which depend on the state afterwards
    (`_update_current_values`).

    Args:
        mc: the math container.
        values: the objects of the model, the `CModel` for the time or a
            `CModelEntity` (a particle number for a species), and their values.
        label: the name of the model in a message.

    Raises:
        RuntimeError: if COPASI does not take a value.
    """
    mc.pushState()
    for entity, value in values:
        if isinstance(entity, COPASI.CModel):
            entity.setTime(value)
        else:
            entity.setValue(value)
    mc.fetchState()
    for entity, value in values:
        taken = float(mc.getMathObject(entity.getValueReference()).getValue())
        if _differs(taken, value):
            raise RuntimeError(
                f"COPASI does not take the value {value!r} of "
                f"'{entity.getObjectName()}' of the model '{label}', it is not "
                f"part of the state of COPASI."
            )


class CopasiSBMLModel(SBMLModel):
    """The model of an SBML model in COPASI, see the module.

    Attributes:
        entities: the parameters, compartments and species of COPASI by their
            SBML id.
        reactions: the reactions of COPASI by their SBML id.
        absolute_tolerance: the absolute tolerance set last, see
            `set_integrator_settings`.
    """

    engine = Engine.COPASI

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
        """Load the model into COPASI, with changes, selections and settings.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`; they are not selected by default.

        Raises:
            ValueError: for an error of COPASI when it imports the model, a
                model with algebraic rules, an initial number which COPASI
                reads otherwise, a selection COPASI has not, or an unknown
                setting.
        """
        self._dm: COPASI.CDataModel | None = None
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
        #: the ids of the species references, which COPASI has no values of
        self._species_references: frozenset[str] = frozenset(
            sr.getId()
            for reaction in sbml_model.getListOfReactions()
            for sr in (*reaction.getListOfReactants(), *reaction.getListOfProducts())
            if sr.isSetId()
        )

        COPASI.CCopasiMessage.clearDeque()
        dm: COPASI.CDataModel = COPASI.CRootContainer.addDatamodel()
        self._dm = dm
        # the data model is removed when the model is released, also when the
        # model is never used, see `close`
        self._finalizer = weakref.finalize(self, _remove_datamodel, dm)
        #: the comparisons of the triggers and of the maths, shifted, see
        #: `shift_comparisons`
        self._shifted: tuple[_Shifted, ...]
        #: whether an event of the model assigns an entity which a comparison
        #: of a trigger reads, so that it may set it exactly to its threshold
        #: in an integration of COPASI
        self.lands: bool
        shifted, self._shifted, self.lands = _with_shifted_comparisons(sbml, self)
        #: the entities whose value in the SBML is not finite, which COPASI
        #: keeps out of its state, see `_with_targets_event`
        self._not_a_number: frozenset[str]
        #: by the position of an event of the model, whether the executor
        #: fires it at the start, its trigger false before the time 0, see
        #: `_with_triggers_true_at_the_start`
        self.start_events: tuple[bool, ...]
        started, self.start_events = _with_triggers_true_at_the_start(shifted, self)
        targeted, self._not_a_number = _with_targets_event(started, self)
        try:
            with kept_encoding():
                imported = dm.importSBMLFromString(targeted)
        except COPASI.CCopasiException:
            imported = False
        drain_messages(self.label)
        if not imported:
            raise ValueError(f"COPASI cannot load the model '{self.label}'.")
        #: the conversion factors of the model and the entities their initial
        #: assignments read, by entity the conversion factor it sets, which a
        #: plan must not change, see `_compile_conversion_factors`
        self.conversion_factors: dict[str, str] = _conversion_factor_inputs(
            conversion_factors(sbml_model), self.symbols.initial_assignment_dependencies
        )
        if self.conversion_factors:
            _compile_conversion_factors(dm.getModel())
            drain_messages(self.label)

        cmodel: COPASI.CModel = dm.getModel()
        self.entities: dict[str, Any] = {}
        for k in range(cmodel.getNumModelValues()):
            entity = cmodel.getModelValue(k)
            self.entities[entity.getSBMLId()] = entity
        for k in range(cmodel.getNumCompartments()):
            entity = cmodel.getCompartment(k)
            self.entities[entity.getSBMLId()] = entity
        for k in range(cmodel.getNumMetabs()):
            entity = cmodel.getMetabolite(k)
            self.entities[entity.getSBMLId()] = entity
        #: the states, read once, see `state_ids`
        self._state_ids: tuple[str, ...] | None = None
        self.reactions: dict[str, Any] = {}
        for k in range(cmodel.getNumReactions()):
            reaction = cmodel.getReaction(k)
            self.reactions[reaction.getSBMLId()] = reaction
        #: the selections resolved in COPASI, see `_selection`
        self._selections: dict[str, _Selection] = {}
        #: the roots of COPASI of every shifted comparison, see
        #: `_comparison_roots`
        self._roots_of: tuple[tuple[int, ...], ...] | None = None
        #: the states of the roots in the values of the math container of
        #: COPASI, `None` without a comparison of a trigger, see
        #: `_comparison_roots`
        self._root_states: _RootStates | None = None
        #: the values of the roots of COPASI as LSODA saw them last, at the end
        #: of the last integration or at the initialization, see
        #: `shift_comparisons`
        self._seen: np.ndarray | None = None

        #: the values of the model as loaded, see `loaded_value`
        self._loaded: dict[str, float] = {}
        q2n = self.mc.getQuantity2NumberFactor()
        for sid, entity in self.entities.items():
            if sid in self.symbols.species:
                self._loaded[sid] = entity.getInitialValue() / q2n
                self._loaded[f"[{sid}]"] = entity.getInitialConcentration()
            else:
                self._loaded[sid] = entity.getInitialValue()
        self._check_initial_numbers(sbml_model)

        #: the initial expressions of the entities as loaded, of COPASI
        self._expressions: dict[str, str] = {
            sid: str(entity.getInitialExpression())
            for sid, entity in self.entities.items()
            if entity.getInitialExpression()
        }
        #: the entities whose initial expression is removed, see `initialize`
        self._cleared: frozenset[str] = frozenset()
        #: the complete initial state of COPASI as loaded, in the layout of the
        #: entities whose expressions are removed, see `_restore`
        self._loaded_state: COPASI.FloatVector = COPASI.FloatVector(
            self.mc.getCompleteInitialState()
        )
        self._apply_initial_values()

        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        #: the settings of roadrunner, which reports the ignored ones once
        self._settings = EngineSettings("COPASI", INTEGRATOR, self.label, logger)

        self.set_selections(selections)
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )

    @property
    def dm(self) -> COPASI.CDataModel:
        """Get the data model of COPASI of the model.

        Raises:
            RuntimeError: if the model was released, see `close`.
        """
        if self._dm is None:
            raise RuntimeError(f"The model '{self.label}' of COPASI was released.")
        return self._dm

    @property
    def cmodel(self) -> COPASI.CModel:
        """Get the model of COPASI."""
        return self.dm.getModel()

    @property
    def mc(self) -> COPASI.CMathContainer:
        """Get the math container of COPASI, the values a simulation integrates."""
        return self.cmodel.getMathContainer()

    @property
    def values(self) -> EngineValues:
        """Get the current values of the model in the selections of roadrunner."""
        return EngineValues(self)

    def close(self) -> None:
        """Release the data model of COPASI, the model cannot be used afterwards.

        A model which is garbage collected releases it as well.
        """
        if self._dm is not None:
            self._dm = None
            self._finalizer()

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
    ) -> CopasiSBMLModel:
        """Create from AbstractModel, see `CopasiSBMLModel`."""
        return CopasiSBMLModel(
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

    def _check_initial_numbers(self, sbml_model: libsbml.Model) -> None:
        """Check that COPASI reads every plain initial number of the SBML.

        A plain initial number is the `value` of a parameter, the `size` of a
        compartment and the `initialAmount` or `initialConcentration` of a
        species of an entity without an initial assignment and an assignment
        rule; a species is compared as the amount or the concentration the
        SBML gives.

        Raises:
            ValueError: if COPASI reads a number otherwise: in other units, or
                as a particle number out of the range of a double, e.g. an
                amount of `1e300` mole.
        """
        symbols = self.symbols
        computed = symbols.initial_assignments | symbols.assignment_rules
        numbers: list[tuple[str, str, float]] = []
        for p in sbml_model.getListOfParameters():
            if p.isSetValue():
                numbers.append((p.getId(), p.getId(), p.getValue()))
        for c in sbml_model.getListOfCompartments():
            if c.isSetSize():
                numbers.append((c.getId(), c.getId(), c.getSize()))
        for s in sbml_model.getListOfSpecies():
            if s.isSetInitialAmount():
                numbers.append((s.getId(), s.getId(), s.getInitialAmount()))
            elif s.isSetInitialConcentration():
                numbers.append(
                    (s.getId(), f"[{s.getId()}]", s.getInitialConcentration())
                )
        for sid, selection, number in numbers:
            if sid in computed or sid not in self.entities:
                continue
            value = self._loaded[selection]
            if not (
                math.isclose(value, number, rel_tol=1e-12)
                or (math.isnan(value) and math.isnan(number))
            ):
                raise ValueError(
                    f"COPASI reads '{sid}' of the model '{self.label}' otherwise: "
                    f"{number!r} in the SBML, {value!r} in COPASI; COPASI reads it "
                    f"in other units, or its particle number (COPASI holds the "
                    f"amounts as particle numbers) is out of the range of a double."
                )

    @override
    def initialize(self, assignments: Sequence[Assignment]) -> None:
        """Initialize the model with values before its initialization.

        The model is set to its initial state as loaded first (`_restore`),
        so an initialization does not depend on the one before it. The values
        are then set as the initial values of their targets and COPASI
        evaluates the initial assignments and the assignment rules which read
        them again; a target replaces its own initial assignment, whose
        initial expression is removed in COPASI while a target sets it. A
        formula reads the parameters as in roadrunner, the values which are
        set and otherwise the values as loaded, before the initial
        assignments are evaluated again (a parameter of an initial assignment
        has its value as loaded, also when an entity it depends on is set) and
        an assignment rule on that state, see `_formula_values`; the values
        of all formulas are evaluated before any of them is set (PEtab v2).

        A compartment whose size changes keeps the concentration of the
        species whose initial value is a concentration and of a species which
        is set as a concentration, and the amount of every other species, as
        roadrunner does (`RoadrunnerSBMLModel.initialize`); a species of an
        assignment rule or of an initial assignment follows its math.

        Args:
            assignments: the pre-initialization assignments of a plan. A
                formula reads parameters only, see `Plan.preinit`.
        """
        symbols = self.symbols
        targets = {symbols.entity(a.target): a.kind for a in assignments}
        self._restore(frozenset(t for t in targets if t in self._expressions))
        references: list[Any] = []
        for a in assignments:
            if a.value is not None:
                references.append(self._set_initial(a.target, a.kind, float(a.value)))
        formulas = [a for a in assignments if a.formula is not None]
        if formulas:
            # the package of the simulator imports the models
            from sbmlsim.simulator.formula import compile_formula

            compiled = [(a, compile_formula(str(a.formula))) for a in formulas]
            values = self._formula_values(
                assignments, {s for _, f in compiled for s in f.symbols}
            )
            evaluated = [
                (a, f.evaluate([values[s] for s in f.symbols])) for a, f in compiled
            ]
            for a, value in evaluated:
                references.append(self._set_initial(a.target, a.kind, value))
        self._update_initial_values(references, targets)
        self._apply_initial_values()

    def _formula_values(
        self, assignments: Sequence[Assignment], symbols: set[str]
    ) -> dict[str, float]:
        """Get the values of the parameters the formulas of an initialization read.

        A formula reads what it reads in roadrunner
        (`RoadrunnerSBMLModel.initialize`): the values which are set and
        otherwise the values as loaded, before the initial assignments are
        evaluated again and before the value of any formula is set, so a
        parameter of an initial assignment has its value as loaded, also when
        an entity it depends on is set. The variable of an assignment rule
        follows its rule on that state, which COPASI evaluates on the current
        state of the model (`_values_set`).

        Args:
            assignments: the pre-initialization assignments.
            symbols: the parameters the formulas read.

        Returns:
            The value of every symbol.
        """
        values = {
            self.symbols.entity(a.target): float(a.value)
            for a in assignments
            if a.value is not None
        }
        read = {s: values.get(s, self._loaded[s]) for s in symbols}
        rules = sorted(s for s in symbols if s in self.symbols.assignment_rules)
        if rules:
            self._values_set(assignments, rules)
            read |= {s: self.value(s) for s in rules}
        return read

    def _values_set(self, assignments: Sequence[Assignment], rules: list[str]) -> None:
        """Set the values of an initialization as the current values of COPASI.

        The current state is the one as loaded with the values set as
        roadrunner sets them (`RoadrunnerSBMLModel.initialize`): the
        compartments first, which keep the concentration of their species
        whose initial value is a concentration or which are set as a
        concentration and the amount of every other species, then the other
        values; the assignment rules are evaluated on it. The initial state
        is not changed, the initialization applies it after.

        Args:
            assignments: the pre-initialization assignments.
            rules: the variables of assignment rules a formula reads.

        Raises:
            NotImplementedError: if an entity which is set is not part of the
                state of COPASI, its value in the SBML is not a number.
        """
        symbols = self.symbols
        values = [a for a in assignments if a.value is not None]
        outside = sorted(
            symbols.entity(a.target)
            for a in values
            if symbols.entity(a.target) in self._not_a_number
        )
        if outside:
            raise NotImplementedError(
                f"COPASI cannot evaluate a formula before the initialization "
                f"which reads the assignment rules {rules} while {outside} is "
                f"set, whose value in the SBML is not a number, which COPASI "
                f"keeps out of its state."
            )
        mc = self.mc
        mc.applyInitialValues()
        _update_current_values(mc)
        amounts = {
            symbols.entity(a.target)
            for a in assignments
            if a.kind is TargetKind.SPECIES_AMOUNT
        }
        concentrations = {
            symbols.entity(a.target)
            for a in assignments
            if a.kind is TargetKind.SPECIES_CONCENTRATION
        }
        for a in values:
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
            self.set_value(a.target, float(a.value))  # ty: ignore[invalid-argument-type]
            for sid, concentration in kept.items():
                self.set_value(f"[{sid}]", concentration)
        for a in values:
            if a.kind is not TargetKind.COMPARTMENT:
                self.set_value(a.target, float(a.value))  # ty: ignore[invalid-argument-type]

    def _set_initial(self, target: str, kind: TargetKind, value: float) -> Any:
        """Set the initial value of a target, see `initialize`.

        Returns:
            The reference of the initial value which is set.
        """
        entity = self.entities[self.symbols.entity(target)]
        if kind is TargetKind.SPECIES_CONCENTRATION:
            entity.setInitialConcentration(value)
            return entity.getInitialConcentrationReference()
        if kind is TargetKind.SPECIES_AMOUNT:
            entity.setInitialValue(value * self.mc.getQuantity2NumberFactor())
        else:
            entity.setInitialValue(value)
        return entity.getInitialValueReference()

    def _update_initial_values(
        self, references: list[Any], targets: dict[str, TargetKind]
    ) -> None:
        """Evaluate the initial values which depend on the values which are set.

        COPASI keeps the particle numbers of the species in a compartment
        whose size changes; a species which keeps its concentration (see
        `initialize`) gets its concentration as loaded again, and the values
        which depend on it are evaluated again.

        Args:
            references: the references of the initial values which are set.
            targets: the targets which are set and their kinds.
        """
        if not references:
            return
        cmodel = self.cmodel
        cmodel.updateInitialValues(_vector(references))
        symbols = self.symbols
        changed = {
            cid
            for cid in symbols.compartments
            if _differs(self.entities[cid].getInitialValue(), self._loaded[cid])
        }
        if not changed:
            return
        kept: list[Any] = []
        for sid, cid in symbols.species_compartment.items():
            if (
                cid not in changed
                or sid in targets
                or sid in symbols.assignment_rules
                or sid in self._expressions
                or sid not in symbols.initial_concentration
            ):
                continue
            entity = self.entities[sid]
            entity.setInitialConcentration(self._loaded[f"[{sid}]"])
            kept.append(entity.getInitialConcentrationReference())
        if kept:
            cmodel.updateInitialValues(_vector(kept))

    def _restore(self, cleared: frozenset[str]) -> None:
        """Set the initial state of COPASI back to the one as loaded.

        The initial state as loaded is set back as a whole, bitwise. The
        entities whose initial expression is removed for an initialization
        are `cleared`: a change of them compiles the model of COPASI again
        (the expressions of the other entities are set back first), so a scan
        which sets the same targets compiles it once.

        Args:
            cleared: the entities whose initial expression is removed.
        """
        mc = self.mc
        if cleared != self._cleared:
            # the model of COPASI as loaded, which the compilation reads
            mc.setCompleteInitialState(self._loaded_state)
            mc.pushInitialState()
            for sid in self._cleared - cleared:
                self.entities[sid].setInitialExpression(self._expressions[sid])
            for sid in cleared - self._cleared:
                self.entities[sid].setInitialExpression("")
            self.cmodel.compileIfNecessary()
            drain_messages(self.label)
            self._cleared = cleared
            self._roots_of = None
            mc = self.mc
            self._loaded_state = COPASI.FloatVector(mc.getCompleteInitialState())
        mc.setCompleteInitialState(self._loaded_state)
        mc.pushInitialState()

    def _apply_initial_values(self) -> None:
        """Make the initial values the current values of the model.

        COPASI evaluates the triggers of the events in the initial state and
        fires an event whose comparison is exactly at its threshold and turns
        right after it by the rate of change of its root (a strict one which
        rises, one which is not strict which falls); the roots are recorded
        there, see `shift_comparisons`.
        """
        mc = self.mc
        mc.applyInitialValues()
        _update_current_values(mc)
        if self._shifted and self._roots_of is None:
            # once per compilation, its probes set the states of the roots,
            # which applying the initial values sets again
            self._comparison_roots()
            mc.applyInitialValues()
            _update_current_values(mc)
        self.record_roots()

    @override
    def reset(self) -> None:
        """Set the model to the state it was loaded in, bitwise, without a simulation."""
        self.initialize([])

    @override
    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization.

        Nothing is recorded: an initialization sets the initial state as
        loaded back whatever ran before it, see `initialize`.
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

    def _selection(self, selection: str) -> _Selection:
        """Resolve a selection of roadrunner in COPASI.

        Raises:
            ValueError: if COPASI has no such selection.
        """
        resolved = self._selections.get(selection)
        if resolved is not None:
            return resolved
        entity = self.entities.get(self.symbols.entity(selection))
        if selection == "time":
            resolved = _Selection(_Kind.TIME, self.cmodel.getValueReference(), None)
        elif selection.startswith("[") and selection.endswith("]"):
            if entity is not None and selection[1:-1] in self.symbols.species:
                resolved = _Selection(
                    _Kind.CONCENTRATION,
                    entity.getConcentrationReference(),
                    selection[1:-1],
                )
        elif selection in self.reactions:
            resolved = _Selection(
                _Kind.FLUX, self.reactions[selection].getFluxReference(), selection
            )
        elif entity is not None:
            resolved = _Selection(
                _Kind.AMOUNT if selection in self.symbols.species else _Kind.VALUE,
                entity.getValueReference(),
                selection,
            )
        if resolved is None:
            reason = (
                ", the stoichiometry of a species reference, which COPASI has no "
                "value of"
                if selection in self._species_references
                else ""
            )
            raise ValueError(f"COPASI has no selection '{selection}'{reason}.")
        self._selections[selection] = resolved
        return resolved

    def _current(self, selection: str) -> _Selection:
        """Resolve a selection of a current value, see `_selection`.

        Raises:
            RuntimeError: if COPASI has no such selection, the error of
                roadrunner.
        """
        try:
            return self._selection(selection)
        except ValueError as err:
            raise RuntimeError(str(err)) from None

    @override
    def value(self, selection: str) -> float:
        """Get the current value of a selection.

        Args:
            selection: the selection, e.g. `S`, `[S]`, a parameter, a
                compartment, a reaction or `time`.

        Returns:
            The value in the current state of the model.

        Raises:
            RuntimeError: if COPASI has no selection of the name, as
                roadrunner.
        """
        resolved = self._current(selection)
        mc = self.mc
        value = float(mc.getMathObject(resolved.reference).getValue())
        if resolved.kind == _Kind.AMOUNT:
            return value / mc.getQuantity2NumberFactor()
        return value

    def set_value(self, selection: str, value: float) -> None:
        """Set the current value of a selection, between two integrations.

        The value is set in the state of COPASI (`_set_current_values`), the
        amount of a species as its particle number, a concentration as the
        amount in the current size of its compartment, and the values which
        depend on it are updated.

        Args:
            selection: `S`, `[S]`, a parameter or a compartment.
            value: the value.

        Raises:
            RuntimeError: if COPASI has no selection of the name, or it cannot
                be set: the time, a reaction or the variable of an assignment
                rule, as roadrunner.
        """
        resolved = self._current(selection)
        if resolved.entity is None or resolved.kind == _Kind.FLUX:
            raise RuntimeError(f"COPASI cannot set '{selection}'.")
        if resolved.entity in self.symbols.assignment_rules:
            raise RuntimeError(
                f"COPASI cannot set '{selection}', it is the variable of an "
                f"assignment rule."
            )
        if resolved.entity in self._not_a_number:
            raise RuntimeError(
                f"COPASI cannot set '{selection}' during a simulation, its value "
                f"in the SBML is not a number, which COPASI keeps out of its state."
            )
        mc = self.mc
        entity = self.entities[resolved.entity]
        number = value
        if resolved.kind == _Kind.AMOUNT:
            number = value * mc.getQuantity2NumberFactor()
        elif resolved.kind == _Kind.CONCENTRATION:
            compartment = self.entities[
                self.symbols.species_compartment[resolved.entity]
            ]
            size = mc.getMathObject(compartment.getValueReference()).getValue()
            number = value * size * mc.getQuantity2NumberFactor()
        _set_current_values(mc, [(entity, number)], self.label)
        _update_current_values(mc)

    def pending_event_time(self) -> float:
        """Get the time of the next delayed event COPASI has scheduled.

        Returns:
            The time at which the next scheduled event executes, infinity
            without one.
        """
        return float(self.mc.getProcessQueueExecutionTime())

    def fire_pending_events(self, time: float) -> None:
        """Execute the delayed events COPASI has scheduled at a time.

        The model is set to the time and COPASI executes the events of its
        queue which are due, in the current state, and the events they
        trigger; the values which depend on the state are updated.

        Args:
            time: the time of the events, see `pending_event_time`.

        Raises:
            RuntimeError: if COPASI reports an error, with its message.
        """
        self.set_time(time)
        mc = self.mc
        # the flags of the changes of the state, a type SWIG has no destructor
        # of: disowned, its few bytes are left to the process instead of a
        # report of a leak on stdout per call
        mc.processQueue(True).disown()
        _update_current_values(mc)
        try:
            drain_messages(self.label)
        except ValueError as err:
            raise RuntimeError(str(err)) from None

    def fire_queued_events(self, time: float) -> None:
        """Execute the events COPASI queued at a time, e.g. at the initial values, and the events they trigger.

        COPASI executes them at the start of an integration, the ones of a
        comparison for equality first (`processQueue(false)`, then `true`,
        as `CTrajectoryTask::processStep`).

        Args:
            time: the time of the events, the current time.

        Raises:
            RuntimeError: if COPASI reports an error, with its message.
        """
        self.set_time(time)
        mc = self.mc
        # the flags of the changes of the state, see `fire_pending_events`
        mc.processQueue(False).disown()
        mc.processQueue(True).disown()
        _update_current_values(mc)
        try:
            drain_messages(self.label)
        except ValueError as err:
            raise RuntimeError(str(err)) from None

    def set_time(self, time: float) -> None:
        """Set the current time of the model, between two integrations.

        An integration of COPASI continues from the current time; the values
        which depend on the time are updated.

        Args:
            time: the time.
        """
        mc = self.mc
        _set_current_values(mc, [(self.cmodel, time)], self.label)
        _update_current_values(mc)

    def shift_comparisons(self, resolution: float, written: bool = False) -> None:
        """Shift the comparisons which are exactly at their threshold, before an integration.

        LSODA in COPASI finds the root of a comparison where its root
        function changes its sign. A root function which is exactly 0 when
        LSODA starts to watch it is masked while it stays 0, and when one is
        0 LSODA takes every root function at a little after the start
        (`DRCHEK`), so a comparison of a trigger at its threshold is not found
        when it turns later, e.g. `X > 2` after a change which sets `X = 2`
        and a rise of `X` right after the change or after a plateau, which
        roadrunner fires. COPASI handles such a root only when it applies the
        initial values, by the sign of its rate of change.

        Every comparison of a trigger and of a math COPASI integrates is
        compiled shifted by a parameter, see `_with_shifted_comparisons`,
        which is 0 but before an integration from a state in which a
        comparison of a trigger is exactly at its threshold with its value as
        written, e.g. after a change which sets a value to it or the time to
        the time of a change. Then every comparison at its threshold is
        shifted, so that no root function of COPASI is 0 at the start and
        LSODA finds the crossings: a comparison keeps its value as written
        until the operands move by the shift in the direction which changes
        it, a comparison of a trigger which COPASI found turned at its
        threshold where LSODA saw it last (`record_roots`: at the end of the
        integration before or at the initial values, and the state of its
        root in COPASI is the turned value), e.g. `time > 10` at the end of a
        horizon of a steady state at 10, keeps its turned value, so its event
        does not fire again. The size of a shift is `_threshold_shift`.
        Without such a comparison of a trigger every shift is 0, the model as
        written.

        Args:
            resolution: the resolution of the time of the simulation, see
                `sbmlsim.simulator.plan.time_tolerance`.
            written: whether every comparison of a trigger at its threshold
                has its value as written, after an event of the model which
                COPASI fired, whose processing of the states of the roots at
                a threshold the executor does not follow.
        """
        if not self._shifted:
            return
        roots_of = self._comparison_roots()
        shifts = [self.entities[s.pid] for s in self._shifted]
        self._set_shifts([(shift, 0.0) for shift in shifts])
        roots = self._roots()
        at_threshold = [any(roots[i] == 0.0 for i in indices) for indices in roots_of]
        states = (
            None
            if written or self._root_states is None
            else self._root_states.read(self.mc, roots)
        )
        turned = [
            shifted.trigger
            and at
            and states is not None
            and self._turned(shifted, indices, states)
            for shifted, indices, at in zip(
                self._shifted, roots_of, at_threshold, strict=True
            )
        ]
        if any(
            shifted.trigger and at and not t
            for shifted, at, t in zip(self._shifted, at_threshold, turned, strict=True)
        ):
            rates = self._root_rates(len(roots))
            for j, indices in enumerate(roots_of):
                if at_threshold[j]:
                    sign = -1.0 if turned[j] else 1.0
                    rate = max(abs(rates[i]) for i in indices)
                    value = self._threshold_shift(
                        shifts[j], indices, sign, rate * resolution
                    )
                    self._set_shifts([(shifts[j], value)])
        self.mc.updateRootValues(False)

    def _set_shifts(self, values: Sequence[tuple[Any, float]]) -> None:
        """Set the current values of parameters of shifts, see `_set_current_values`.

        The values which depend on them are not updated, the roots are, where
        they are read (`_roots`).
        """
        _set_current_values(self.mc, values, self.label)

    def record_roots(self) -> None:
        """Record the roots of COPASI where LSODA saw them last, after an integration or at the initial values, see `shift_comparisons`."""
        if self._shifted:
            self._seen = self._roots()

    def landed(self) -> tuple[bool, list[int]]:
        """Tell the comparisons of the triggers an integration left at their threshold or whose crossing COPASI missed.

        An event of the model which COPASI fires in an integration may set a
        comparison of a trigger exactly to its threshold, e.g. `G: X > 0`
        after `E: X = 0`, which LSODA does not find when it turns, see
        `shift_comparisons`: after the integration the comparison is at its
        threshold, or, if it turned, the state of its root in COPASI is not
        its value. COPASI processes a root it finds at once, so a state which
        differs from the sign of its root is a crossing it missed.

        Returns:
            Whether a comparison of a trigger is exactly at its threshold, and
            the positions of the events among the events of the model whose
            crossing COPASI missed, in their order.
        """
        if not self._shifted or self._root_states is None:
            return False, []
        roots = self._roots()
        states = self._root_states.read(self.mc, roots)
        at_threshold = False
        missed: set[int] = set()
        for shifted, indices in zip(
            self._shifted, self._comparison_roots(), strict=True
        ):
            if shifted.event is None:
                continue
            for i in indices:
                root = roots[i]
                if root == 0.0:
                    at_threshold = True
                elif (states[i] == 1.0) != (root > 0.0):
                    missed.add(shifted.event)
        return at_threshold, sorted(missed)

    def snapshot(self) -> tuple[COPASI.FloatVector, np.ndarray | None]:
        """Get the current state of the model, of the shifts and of the roots as LSODA saw them last, see `restore`."""
        return COPASI.FloatVector(self.mc.getState(False)), (
            None if self._seen is None else self._seen.copy()
        )

    def restore(self, snapshot: tuple[COPASI.FloatVector, np.ndarray | None]) -> None:
        """Set the model back to a state of `snapshot`, the time and the shifts included.

        The states of the triggers in COPASI are the ones of the integration
        after it; an integration sets them from the current values first.
        """
        state, seen = snapshot
        mc = self.mc
        mc.setState(state)
        _update_current_values(mc)
        mc.updateRootValues(False)
        self._seen = None if seen is None else seen.copy()

    def _turned(
        self, shifted: _Shifted, indices: Sequence[int], states: np.ndarray
    ) -> bool:
        """Whether COPASI found a comparison of a trigger at its threshold turned where LSODA saw it last.

        The roots of the comparison were 0 where they were recorded last
        (`record_roots`), and the states of the roots in COPASI (`states`, see
        `_RootStates`) are the values after the crossing, true for a strict
        comparison and false for one which is not strict; a root COPASI did
        not find there, or a value set since, has the value as written.
        """
        seen = self._seen
        if seen is None:
            return False
        turned = 1.0 if shifted.strict else 0.0
        return all(seen[i] == 0.0 and states[i] == turned for i in indices)

    def _roots(self) -> np.ndarray:
        """Get the current values of the roots of COPASI, of the triggers and the discontinuities."""
        mc = self.mc
        mc.updateRootValues(False)
        roots = mc.getRoots()
        return np.array([roots.get(i) for i in range(roots.size())], dtype=float)

    def _root_rates(self, count: int) -> np.ndarray:
        """Get the current rates of change of the roots of COPASI in the time."""
        mc = self.mc
        _update_current_values(mc)
        rates = COPASI.FloatVector(count)
        mc.calculateRootDerivatives(rates)
        return np.array([rates.get(i) for i in range(count)], dtype=float)

    def _comparison_roots(self) -> tuple[tuple[int, ...], ...]:
        """Get the positions of the roots of COPASI of every shifted comparison.

        A root of a comparison is one whose value changes with its shift,
        told once per compilation of the model of COPASI, see
        `_apply_initial_values`, together with the states of the roots, see
        `_RootStates.locate`, which moves the shift of a comparison far in
        both directions and changes the states of its roots after
        `resetEventsFound` to false and true, so the states COPASI keeps are
        set again afterwards.

        Raises:
            RuntimeError: if the states of the roots are not found.
        """
        if self._roots_of is None:
            mc = self.mc
            base = self._roots()
            result: list[tuple[int, ...]] = []
            probes: list[tuple[_Shifted, Any, tuple[int, ...]]] = []
            for shifted in self._shifted:
                shift = self.entities[shifted.pid]
                value = float(mc.getMathObject(shift.getValueReference()).getValue())
                self._set_shifts([(shift, _PROBE_SHIFT)])
                roots = self._roots()
                self._set_shifts([(shift, value)])
                changed = (roots != base) & ~(np.isnan(roots) & np.isnan(base))
                indices = tuple(int(i) for i in np.flatnonzero(changed))
                result.append(indices)
                if indices and shifted.trigger:
                    probes.append((shifted, shift, indices))
            self._root_states = _RootStates.locate(self, probes) if probes else None
            mc.updateRootValues(False)
            self._roots_of = tuple(result)
        return self._roots_of

    def _threshold_shift(
        self, shift: Any, indices: Sequence[int], sign: float, change: float
    ) -> float:
        """Get the shift of a comparison exactly at its threshold, see `shift_comparisons`.

        Its operands moving at their current rate, the change of the
        comparison over the resolution of the time, so it turns at most one
        resolution late, and at least twice the unit in the last place of the
        threshold, the smallest power of two which moves every root of the
        comparison off 0, found by bisection of the exponent. On a plateau,
        where the rate is 0, `THRESHOLD_ULPS` units in the last place of the
        threshold, at least the absolute tolerance of LSODA and
        `THRESHOLD_ULPS` units in the last place of the time, the change of
        operands which move by one per unit of the time later: more than
        LSODA steps over after a root of another function which is exactly 0
        where LSODA found it, e.g. the end of the plateau at `time < 1.5`
        (`DRCHEK`, about a hundred units in the last place of the time), also
        for a threshold 0, whose units in the last place tell nothing.

        Args:
            shift: the parameter of the shift in COPASI, at 0.
            indices: the positions of the roots of the comparison.
            sign: 1 for a shift which keeps the value of the comparison as
                written at its threshold, -1 for one which keeps the other.
            change: the change of the root functions of the comparison over
                the resolution of the time at their current rate.

        Returns:
            The shift with its sign, 0 if no shift moves the roots, e.g. for
            an operand which is not finite.
        """

        def moves(exponent: int) -> bool:
            self._set_shifts([(shift, sign * math.ldexp(1.0, exponent))])
            roots = self._roots()
            return all(roots[i] != 0.0 for i in indices)

        low, high = (
            sys.float_info.min_exp - sys.float_info.mant_dig,
            sys.float_info.max_exp - 1,
        )
        if not moves(high):
            self._set_shifts([(shift, 0.0)])
            return 0.0
        # the smallest exponent which moves the roots is in (low, high]
        if moves(low):
            high = low
        while high - low > 1:
            middle = (low + high) // 2
            if moves(middle):
                high = middle
            else:
                low = middle
        unit = math.ldexp(1.0, high)
        if change > 0.0:
            shifted = max(2.0 * unit, change)
        else:
            method = self.dm.getTask("Time-Course").getMethod()
            absolute = method.getParameter(LSODA_PARAMETERS["absolute_tolerance"])
            time = abs(self.value("time"))
            shifted = max(
                THRESHOLD_ULPS * unit,
                float(absolute.getDblValue()),
                THRESHOLD_ULPS * float(np.spacing(max(time, 1.0))),
            )
        return sign * min(shifted, sys.float_info.max)

    def rates(self, states: Sequence[str]) -> np.ndarray:
        """Get the current rates of change of states.

        Args:
            states: the ids of species, whose rates are rates of their amounts,
                or of the targets of rate rules, see `state_ids`.

        Returns:
            The rate of every state.
        """
        mc = self.mc
        _update_current_values(mc)
        n2q = 1.0 / mc.getQuantity2NumberFactor()
        return np.array(
            [
                mc.getMathObject(self.entities[sid].getRateReference()).getValue()
                * (n2q if sid in self.symbols.species else 1.0)
                for sid in states
            ],
            dtype=float,
        )

    @override
    def has_selection(self, selection: str) -> bool:
        """Check whether COPASI has a selection of a name in the model.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it.
        """
        try:
            self._selection(selection)
        except ValueError:
            return False
        return True

    @override
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, `default_selections` for `None`.

        Raises:
            ValueError: if COPASI has no selection of a name.
        """
        chosen = self.default_selections() if selections is None else list(selections)
        for selection in chosen:
            self._selection(selection)
        self.selections = chosen
        self.selections_given = selections is not None

    def output_cns(self, selections: Sequence[str]) -> list[str]:
        """Get the common names of COPASI of the values of selections.

        The values of an output of COPASI (`CDataHandler`), the amount of a
        species as its particle number, see `output_transform`.

        Args:
            selections: the selections.

        Returns:
            The common name of each selection.

        Raises:
            ValueError: if COPASI has no selection of a name.
        """
        return [
            str(self._selection(s).reference.getCN().getString()) for s in selections
        ]

    def output_transform(
        self, selections: Sequence[str]
    ) -> Callable[[Sequence[float] | np.ndarray], np.ndarray]:
        """Get the conversion of the output of COPASI into the values of selections.

        Args:
            selections: the selections of the output, see `output_cns`.

        Returns:
            A function of the rows of an output (a row or an array of rows,
            the columns in the order of the selections), which turns the
            particle numbers into amounts.
        """
        n2q = 1.0 / self.mc.getQuantity2NumberFactor()
        scale = np.array(
            [
                n2q if self._selection(s).kind == _Kind.AMOUNT else 1.0
                for s in selections
            ]
        )

        def transform(rows: Sequence[float] | np.ndarray) -> np.ndarray:
            return np.asarray(rows, dtype=float) * scale

        return transform

    @override
    def set_integrator_settings(self, **settings: IntegratorSetting) -> None:
        """Set settings of the integrator, LSODA of COPASI.

        The settings have the names of roadrunner: `relative_tolerance`,
        `absolute_tolerance`, `maximum_num_steps` and `maximum_time_step`.
        COPASI has one absolute tolerance, the smallest tolerance of an
        `AbsoluteTolerance` (of the kinds and of the overrides, reported once
        when they differ) times `COPASI_ATOL_FACTOR`, so COPASI integrates at
        least as accurately as roadrunner at equal settings. A setting of
        roadrunner which COPASI has no equivalent of, e.g.
        `variable_step_size`, is ignored and reported once per model.

        Args:
            **settings: the settings by their names in roadrunner.

        Raises:
            ValueError: if neither COPASI nor roadrunner has a setting of a
                name, or a value is not valid.
        """
        method = self.dm.getTask("Time-Course").getMethod()
        for key, value in self._settings.checked(settings).items():
            parameter = method.getParameter(LSODA_PARAMETERS[key])
            if isinstance(value, AbsoluteTolerance):
                parameter.setDblValue(
                    self._settings.scalar_tolerance(value, COPASI_ATOL_FACTOR)
                )
                self.absolute_tolerance = value
            elif key == "maximum_num_steps":
                parameter.setUIntValue(int(value))
            else:
                parameter.setDblValue(float(value))

    @override
    def integrator_state(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Get the name of the integrator of the model and every one of its settings.

        The settings have the names of roadrunner, the absolute tolerance is
        the `AbsoluteTolerance` of the model as it was set; the one absolute
        tolerance of LSODA is its smallest tolerance times
        `COPASI_ATOL_FACTOR`, see `set_integrator_settings`.

        Returns:
            The name of the integrator and its settings, sorted by name.
        """
        method = self.dm.getTask("Time-Course").getMethod()
        settings: list[tuple[str, Any]] = []
        for key in sorted(LSODA_PARAMETERS):
            parameter = method.getParameter(LSODA_PARAMETERS[key])
            if key == "absolute_tolerance":
                settings.append((key, self.absolute_tolerance))
            elif key == "maximum_num_steps":
                settings.append((key, int(parameter.getUIntValue())))
            else:
                settings.append((key, float(parameter.getDblValue())))
        return INTEGRATOR, tuple(settings)

    @override
    def set_integrator_state(
        self, state: tuple[str, tuple[tuple[str, Any], ...]]
    ) -> None:
        """Set the settings of the integrator, see `integrator_state`.

        Args:
            state: the name of the integrator, `lsoda`, and its settings.

        Raises:
            ValueError: if the integrator is not the one of COPASI, or it has
                no setting of a name.
        """
        integrator, settings = state
        if integrator != INTEGRATOR:
            raise ValueError(
                f"COPASI integrates with '{INTEGRATOR}', not with '{integrator}'."
            )
        self.set_integrator_settings(**dict(settings))

    @override
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates.

        The species which reactions or a rate rule change, in the order of
        the SBML, then the other targets of rate rules, in the order of the
        SBML; read once, the states of a model do not change.
        """
        if self._state_ids is None:
            integrated = (
                COPASI.CModelEntity.Status_REACTIONS,
                COPASI.CModelEntity.Status_ODE,
            )
            doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(self.sbml)
            sbml_model: libsbml.Model = doc.getModel()
            states = [
                s.getId()
                for s in sbml_model.getListOfSpecies()
                if self.entities[s.getId()].getStatus() in integrated
            ]
            states += rate_rule_entities(sbml_model, self.symbols)
            self._state_ids = tuple(states)
        return list(self._state_ids)


def _check_supported(model: libsbml.Model, label: str) -> None:
    """Check that COPASI simulates a model as roadrunner does.

    COPASI loads the following without an error and simulates them otherwise
    than roadrunner: it ignores algebraic rules, makes the stoichiometry of a
    non-constant species reference, of a stoichiometry math and of an initial
    assignment of a species reference a constant, and simulates the delay
    csymbol otherwise. The conversion factors are simulated after a
    compilation of the model, see `_compile_conversion_factors`.

    Args:
        model: the model of the SBML.
        label: the name of the model in a message.

    Raises:
        UnsupportedModelError: naming the feature and its element, for a
            model with one.
    """

    def unsupported(feature: str) -> UnsupportedModelError:
        return UnsupportedModelError(
            f"The model '{label}' has {feature}; COPASI does not simulate it as "
            f"roadrunner, the model cannot be simulated with COPASI."
        )

    features = list(variable_features(model))
    reasons = {
        Feature.ALGEBRAIC_RULE: ", which COPASI ignores",
        Feature.STOICHIOMETRY_MATH: ", which COPASI makes a constant",
        Feature.NON_CONSTANT_REFERENCE: ", whose stoichiometry COPASI makes a constant",
        Feature.REFERENCE_INITIAL_ASSIGNMENT: (
            ", which COPASI evaluates once into a constant"
        ),
    }
    for feature, element in features:
        if feature is Feature.ALGEBRAIC_RULE:
            raise unsupported(element + reasons[feature])
    for feature, element in features:
        raise unsupported(element + reasons.get(feature, ""))


def _compile_conversion_factors(cmodel: COPASI.CModel) -> None:
    """Compile a model of COPASI with conversion factors again after its import.

    The import of COPASI multiplies the stoichiometries of the species with a
    conversion factor by the value of the factor (`SBML (100)`), after it
    compiled the model, without marking the model for a compilation: a time
    course integrates the stoichiometries of the SBML (copasi/basico#82). A
    compilation forced after the import compiles the stoichiometries with
    the factors. The stoichiometries keep the value of a factor at the
    import, a change of a factor or of an entity its initial assignment
    reads is refused, see `_conversion_factor_inputs`. The compilation of a
    model whose conversion factor has an initial assignment sets the locale
    of the process to `C`, see `kept_encoding`.

    Args:
        cmodel: the model of COPASI, right after its import.
    """
    cmodel.setCompileFlag(True)
    with kept_encoding():
        cmodel.compileIfNecessary()


def _conversion_factor_inputs(
    factors: set[str] | frozenset[str],
    dependencies: dict[str, frozenset[str]] | None,
) -> dict[str, str]:
    """Get the conversion factors and the entities their initial values depend on.

    COPASI multiplies the stoichiometries by the initial value of a
    conversion factor when it imports the model, see
    `_compile_conversion_factors`, so neither the factor nor an entity which
    its initial assignment reads, also through other initial assignments,
    may change afterwards.

    Args:
        factors: the conversion factors of the model.
        dependencies: entity -> the entities its initial assignment reads, see
            `ModelSymbols.initial_assignment_dependencies`.

    Returns:
        Entity -> the conversion factor whose value depends on it, a factor
        itself included.
    """
    inputs: dict[str, str] = {}
    for factor in sorted(factors):
        pending = [factor]
        while pending:
            sid = pending.pop()
            if sid in inputs:
                continue
            inputs[sid] = factor
            pending.extend(sorted((dependencies or {}).get(sid, frozenset())))
    return inputs


def _remove_datamodel(dm: COPASI.CDataModel) -> None:
    """Remove a data model from the root container of COPASI."""
    COPASI.CRootContainer.removeDatamodel(dm)


def _differs(a: float, b: float) -> bool:
    """Whether two numbers differ, `NaN` equals `NaN`."""
    return a != b and not (math.isnan(a) and math.isnan(b))


def _update_current_values(mc: COPASI.CMathContainer) -> None:
    """Update the current values of a math container which depend on its state.

    The simulated values are the ones the integrator needs, the transient
    data values the others, e.g. an assignment rule which no rate reads.
    """
    mc.updateSimulatedValues(False)
    mc.updateTransientDataValues()


@dataclass(frozen=True)
class _RootStates:
    """The states of the roots of COPASI in the values of its math container.

    COPASI keeps the state of every root of the triggers (1 true, 0 false),
    which tells a crossing it found and how it processed a root at its
    threshold, but no API gives them (copasi/basico#87). They follow right
    after the values of the roots in the values of the math container
    (`getValues`), a layout of COPASI which this class alone reads: it is
    located once per compilation of the model and validated by probes
    (`locate`), and every read checks that the values of the roots are still
    right before the states (`read`), so another layout raises rather than
    gives wrong states.

    Attributes:
        position: the position of the state of the first root in the values.
        count: the number of roots.
    """

    position: int
    count: int

    @staticmethod
    def locate(
        model: CopasiSBMLModel,
        probes: Sequence[tuple[_Shifted, Any, tuple[int, ...]]],
    ) -> _RootStates:
        """Locate the states of the roots in the values of the math container of a model.

        A candidate is a block of the values equal to the roots followed by
        states. It is the one if, for every probed comparison with its shift
        moved far to either side, the states which `resetEventsFound` sets
        at the candidate are the values of the comparison, false then true
        for a strict one, true then false for one which is not strict.

        Args:
            model: the model, whose roots are read and whose shifts are set.
            probes: the comparisons with roots, their parameters of the shift
                (at 0) and the positions of their roots.

        Returns:
            The states of the roots.

        Raises:
            RuntimeError: if no candidate validates.
        """
        mc = model.mc
        roots = model._roots()
        n = len(roots)
        values = mc.getValues()
        count = values.size()
        candidates = [
            k + n
            for k in range(count - 2 * n + 1)
            if not any(_differs(values.get(k + i), roots[i]) for i in range(n))
            and all(values.get(k + n + i) in (0.0, 1.0) for i in range(n))
        ]
        for at in reversed(candidates):
            valid = True
            for shifted, shift, indices in probes:
                for probe in (_PROBE_SHIFT, -_PROBE_SHIFT):
                    model._set_shifts([(shift, probe)])
                    mc.updateRootValues(False)
                    mc.resetEventsFound()
                    # a strict comparison shifted forward is false, one which
                    # is not strict true
                    expected = 0.0 if (probe > 0) == shifted.strict else 1.0
                    states = mc.getValues()
                    valid = valid and all(
                        states.get(at + i) == expected for i in indices
                    )
                model._set_shifts([(shift, 0.0)])
            mc.updateRootValues(False)
            mc.resetEventsFound()
            if valid:
                return _RootStates(at, n)
        raise RuntimeError(
            f"The states of the roots of the model '{model.label}' are not found "
            f"in the values of COPASI, whose layout sbmlsim reads, since COPASI "
            f"has no API for them (copasi/basico#87)."
        )

    def read(self, mc: COPASI.CMathContainer, roots: np.ndarray) -> np.ndarray:
        """Read the states of the roots.

        Args:
            mc: the math container.
            roots: the current values of the roots, see
                `CopasiSBMLModel._roots`.

        Returns:
            The state of every root, 1 true and 0 false.

        Raises:
            RuntimeError: if the values of the roots are not right before the
                states, the layout of COPASI changed.
        """
        values = mc.getValues()
        at, n = self.position, self.count
        if (
            len(roots) != n
            or values.size() < at + n
            or any(_differs(values.get(at - n + i), roots[i]) for i in range(n))
        ):
            raise RuntimeError(
                "The states of the roots are not where they were found in the "
                "values of COPASI, whose layout sbmlsim reads, since COPASI has "
                "no API for them (copasi/basico#87)."
            )
        return np.array([values.get(at + i) for i in range(n)], dtype=float)


def _vector(references: Sequence[Any]) -> COPASI.ObjectStdVector:
    """Get the vector of COPASI of references to objects."""
    vector = COPASI.ObjectStdVector()
    for reference in references:
        vector.append(reference)
    return vector


def _with_shifted_comparisons(
    sbml: str, model: SBMLModel
) -> tuple[str, tuple[_Shifted, ...], bool]:
    """Shift the comparisons of the triggers and of the maths COPASI integrates by parameters.

    For a model with events, the comparisons through the logical operators of
    a trigger, `>`, `<`, `>=` and `<=`, and the ones anywhere in a kinetic
    law, an assignment rule or a rate rule, e.g. the condition of a piecewise
    of a rate, of which COPASI finds the roots as well, are shifted by a
    parameter of the value 0, `a > b + shift` (`SHIFT_PREFIX`,
    `SWITCH_PREFIX`, see `sbmlsim.model.comparisons.shift_comparison`), which
    a simulation sets at the threshold of the comparison, see
    `CopasiSBMLModel.shift_comparisons`; a math is rewritten with the
    function definitions it calls expanded only if it has such a comparison.
    A shift of 0 is the comparison as written, bitwise. The parameters are
    made targets of `TARGETS_EVENT` with the other parameters, see
    `_with_targets_event`.

    Args:
        sbml: the SBML of the model.
        model: the model, whose events and label are used.

    Returns:
        The SBML with the shifted comparisons, the shifted comparisons in the
        order of their parameters, and whether an event of the model assigns
        an entity a comparison of a trigger reads, see `_assigns_operands`.

    Raises:
        ValueError: if the model has an element of the id of a parameter of
            a shift.
    """
    if not model.symbols.events:
        return sbml, (), False
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    sbml_model: libsbml.Model = doc.getModel()
    functions: libsbml.ListOfFunctionDefinitions = (
        sbml_model.getListOfFunctionDefinitions()
    )
    shifted: list[_Shifted] = []
    operands: set[str] = set()

    def shift(comparison: libsbml.ASTNode, pid: str, event: int | None) -> None:
        if sbml_model.getElementBySId(pid) is not None:
            raise ValueError(
                f"The model '{model.label}' has the entity '{pid}', the id of a "
                f"parameter sbmlsim adds for COPASI."
            )
        parameter: libsbml.Parameter = sbml_model.createParameter()
        parameter.setId(pid)
        parameter.setValue(0.0)
        parameter.setConstant(False)
        strict = comparison.getType() in STRICT_COMPARISONS
        shifted.append(_Shifted(pid=pid, strict=strict, event=event))
        shift_comparison(comparison, pid)

    def expanded(math: libsbml.ASTNode) -> libsbml.ASTNode:
        node: libsbml.ASTNode = math.deepCopy()
        if functions.size():
            libsbml.SBMLTransforms.replaceFD(node, functions)
        return node

    for k, event in enumerate(sbml_model.getListOfEvents()):
        trigger: libsbml.Trigger | None = event.getTrigger()
        if trigger is None or not trigger.isSetMath():
            continue
        node = expanded(trigger.getMath())
        found = trigger_comparisons(node, COMPARISONS)
        for j, (_, _, comparison) in enumerate(found):
            operands |= _names(comparison)
            shift(comparison, f"{SHIFT_PREFIX}{k}_{j}", event=k)
        if found:
            trigger.setMath(node)
    if not shifted:
        return sbml, (), False
    lands = _assigns_operands(sbml_model, operands)

    elements: list[Any] = [
        reaction.getKineticLaw()
        for reaction in sbml_model.getListOfReactions()
        if reaction.isSetKineticLaw()
    ]
    elements += [rule for rule in sbml_model.getListOfRules() if not rule.isAlgebraic()]
    switches = 0
    for element in elements:
        if not element.isSetMath():
            continue
        node = expanded(element.getMath())
        found = math_comparisons(node, COMPARISONS)
        # a comparison in an operand of another one first, which the shift of
        # the other one copies
        for _, _, comparison in reversed(found):
            shift(comparison, f"{SWITCH_PREFIX}{switches}", event=None)
            switches += 1
        if found:
            element.setMath(node)
    return libsbml.writeSBMLToString(doc), tuple(shifted), lands


def _names(math: libsbml.ASTNode) -> set[str]:
    """Get the identifiers a math reads."""
    names: set[str] = set()
    stack = [math]
    while stack:
        node = stack.pop()
        if node.isName():
            names.add(node.getName())
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return names


def _assigns_operands(sbml_model: libsbml.Model, operands: set[str]) -> bool:
    """Whether an event of a model assigns an entity which the comparisons of the triggers read.

    The entities a comparison reads through the assignment rules, and the
    compartments of the species it reads, whose concentrations their sizes
    change. Such an event may set a comparison exactly to its threshold in
    an integration of COPASI, see `CopasiSBMLModel.lands`.

    Args:
        sbml_model: the model.
        operands: the identifiers the comparisons of the triggers read.
    """
    rules = {
        rule.getVariable(): _names(rule.getMath())
        for rule in sbml_model.getListOfRules()
        if rule.isAssignment() and rule.isSetMath()
    }
    read: set[str] = set()
    stack = list(operands)
    while stack:
        sid = stack.pop()
        if sid in read:
            continue
        read.add(sid)
        stack.extend(rules.get(sid, ()))
        species: libsbml.Species | None = sbml_model.getSpecies(sid)
        if species is not None:
            stack.append(species.getCompartment())
    return any(
        assignment.getVariable() in read
        for event in sbml_model.getListOfEvents()
        for assignment in event.getListOfEventAssignments()
    )


def _with_triggers_true_at_the_start(
    sbml: str, model: SBMLModel
) -> tuple[str, tuple[bool, ...]]:
    """Make the trigger of every event true before the time 0, the executor fires the events of the start.

    COPASI queues an event whose trigger is false before the time 0
    (`initialValue="false"`) and true in the initial state at the initial
    values, together with an event whose comparison is exactly at its
    threshold and turns right after the start, see
    `CopasiSBMLModel._apply_initial_values`; the executor executes the queue
    of the start of a run whose events may set a trigger to its threshold
    first and has the state before it at the start, see
    `sbmlsim.simulator.executor_copasi._TimeCourse._integrate`, while an
    event of the start has the state after it there, as in roadrunner and
    AMICI. The triggers of such events are therefore true before the time 0
    in the SBML which COPASI loads, so COPASI does not fire them at the
    start, and the executor fires them in the initial state, see
    `sbmlsim.simulator.executor_copasi.execute`. A model with a delayed
    event is left as it is, its delayed events of the start COPASI
    schedules itself; so is a model of level 1 or 2, whose triggers are
    true before the time 0.

    Args:
        sbml: the SBML of the model.
        model: the model, whose symbols are used.

    Returns:
        The SBML and, by the position of an event of the model, whether its
        trigger was false before the time 0.
    """
    events = model.symbols.events
    if not events or any(event.delayed for event in events):
        return sbml, tuple(False for _ in events)
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if doc.getLevel() < 3:
        return sbml, tuple(False for _ in events)
    sbml_model: libsbml.Model = doc.getModel()
    start: list[bool] = []
    for event in sbml_model.getListOfEvents():
        trigger: libsbml.Trigger | None = event.getTrigger()
        start.append(trigger is not None and not trigger.getInitialValue())
        if trigger is not None:
            trigger.setInitialValue(True)
    if not any(start):
        return sbml, tuple(start)
    return libsbml.writeSBMLToString(doc), tuple(start)


def _with_targets_event(sbml: str, model: SBMLModel) -> tuple[str, frozenset[str]]:
    """Make the entities a change can set part of the state of COPASI.

    Every parameter, compartment and species which is not the variable of an
    assignment rule or a conversion factor, which SBML requires to be
    constant, is made non-constant and the target of an event which
    never fires (`TARGETS_EVENT`), which assigns it its own value, see the
    module. An entity whose value in the SBML is a number which is not finite
    (`NaN`, e.g. a compartment of the HCTZ model, or an infinity) and which
    has no initial assignment is left out: COPASI integrates no state which
    is not finite (`CTrajectoryMethod (25)`, an invalid state).

    Args:
        sbml: the SBML of the model.
        model: the model, whose symbols and label are used.

    Returns:
        The SBML which COPASI loads, and the entities which are left out
        because their value is not finite.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if doc.getLevel() < 2:
        doc.setLevelAndVersion(2, 4)
    sbml_model: libsbml.Model = doc.getModel()
    symbols = model.symbols
    # a conversion factor is a constant parameter in SBML
    constants = conversion_factors(sbml_model)
    targets: list[str] = []
    not_a_number: set[str] = set()
    for entity in (
        *sbml_model.getListOfParameters(),
        *sbml_model.getListOfCompartments(),
        *sbml_model.getListOfSpecies(),
    ):
        sid = entity.getId()
        if sid in symbols.assignment_rules or sid in constants:
            continue
        if sid not in symbols.initial_assignments and not math.isfinite(
            _number(entity)
        ):
            not_a_number.add(sid)
            continue
        entity.setConstant(False)
        targets.append(sid)
    if not targets:
        return sbml, frozenset(not_a_number)
    event: libsbml.Event = sbml_model.createEvent()
    event.setId(TARGETS_EVENT)
    trigger: libsbml.Trigger = event.createTrigger()
    trigger.setMath(libsbml.parseL3Formula("false"))
    if doc.getLevel() >= 3:
        event.setUseValuesFromTriggerTime(True)
        trigger.setInitialValue(False)
        trigger.setPersistent(True)
    for target in targets:
        assignment: libsbml.EventAssignment = event.createEventAssignment()
        assignment.setVariable(target)
        assignment.setMath(libsbml.parseL3Formula(target))
    return libsbml.writeSBMLToString(doc), frozenset(not_a_number)


def _number(entity: Any) -> float:
    """Get the number of a parameter, a compartment or a species in the SBML.

    Returns:
        The value, size, initial amount or initial concentration, 0 without
        one.
    """
    if isinstance(entity, libsbml.Species):
        if entity.isSetInitialAmount():
            return entity.getInitialAmount()
        if entity.isSetInitialConcentration():
            return entity.getInitialConcentration()
        return 0.0
    if isinstance(entity, libsbml.Compartment):
        return entity.getSize() if entity.isSetSize() else 0.0
    return entity.getValue() if entity.isSetValue() else 0.0
