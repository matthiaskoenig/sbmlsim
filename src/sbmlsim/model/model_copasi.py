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
simulates otherwise than roadrunner, see `_check_supported`.

COPASI folds a constant entity into the constants of its model, a value
written into it during a simulation is not seen. The SBML which COPASI loads
therefore makes every entity which a change can set (the parameters,
compartments and species which are not the variable of an assignment rule)
non-constant and the target of one event which never fires
(`TARGETS_EVENT`), which makes them part of the state of COPASI; an entity
whose number in the SBML is not finite (`NaN`) is left out, COPASI integrates
no state which is not finite, and a change of it raises.
"""

from __future__ import annotations

import ctypes
import logging
import math
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import libsbml
import numpy as np

from sbmlsim.engine import Engine, import_copasi
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.model_sbml import IntegratorSetting, SBMLModel
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

#: the settings of roadrunner which COPASI has no equivalent of, ignored with
#: a warning: the settings of CVODE and the optimization of the compilation
ROADRUNNER_ONLY_SETTINGS = frozenset(
    {
        "initial_time_step",
        "jit_optimization",
        "max_output_rows",
        "maximum_adams_order",
        "maximum_bdf_order",
        "minimum_time_step",
        "multiple_steps",
        "stiff",
        "variable_step_size",
    }
)

#: the messages of COPASI which do not change the numbers of a model, logged
#: at the level `DEBUG`: the default units of the model (91, 92), the units of
#: species and of the model which COPASI does not import and whose numbers it
#: keeps (25, 66), a compartment without a size, which has the size 1 as in
#: roadrunner (45), a compartment without spatial dimensions, which COPASI
#: reads as one of three, whose size is the same number (90), an unknown
#: annotation and the warnings of libsbml
BENIGN_MESSAGES = (
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


def _write_value(math_object: Any, value: float) -> None:
    """Write the value of an object of the math container of COPASI.

    The python bindings of COPASI have no setter of the current value of an
    object of the math container, which a change between two integrations
    sets; the value is the double at the value pointer of the object, written
    here. This is the only place sbmlsim writes the memory of COPASI. The
    caller updates the values which depend on it afterwards
    (`_update_current_values`).

    Args:
        math_object: the object of the math container (`CMathObject`).
        value: the value.
    """
    ctypes.c_double.from_address(int(math_object.getValuePointer())).value = value


class CopasiValues:
    """The current values of a model of COPASI in the selections of roadrunner.

    The `ModelValues` of `sbmlsim.simulator.changes`, with which an executor
    reads and sets values between two integrations.
    """

    def __init__(self, model: CopasiSBMLModel) -> None:
        """Get the values of a model."""
        self._model = model

    def getValue(self, selection: str, /) -> float:
        """Get the current value of a selection, see `CopasiSBMLModel.value`."""
        return self._model.value(selection)

    def setValue(self, selection: str, value: float, /) -> None:
        """Set the current value of a selection, see `CopasiSBMLModel.set_value`."""
        self._model.set_value(selection, value)


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
        #: the entities whose value in the SBML is not finite, which COPASI
        #: keeps out of its state, see `_with_targets_event`
        self._not_a_number: frozenset[str]
        targeted, self._not_a_number = _with_targets_event(sbml, self)
        try:
            imported = dm.importSBMLFromString(targeted)
        except COPASI.CCopasiException:
            imported = False
        drain_messages(self.label)
        if not imported:
            raise ValueError(f"COPASI cannot load the model '{self.label}'.")

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
        #: the settings of roadrunner which were reported as ignored
        self._ignored_reported: set[str] = set()
        #: whether the absolute tolerance was reported as one number
        self._tolerance_reported: bool = False

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
    def values(self) -> CopasiValues:
        """Get the current values of the model in the selections of roadrunner."""
        return CopasiValues(self)

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
            mc = self.mc
            self._loaded_state = COPASI.FloatVector(mc.getCompleteInitialState())
        mc.setCompleteInitialState(self._loaded_state)
        mc.pushInitialState()

    def _apply_initial_values(self) -> None:
        """Make the initial values the current values of the model."""
        mc = self.mc
        mc.applyInitialValues()
        _update_current_values(mc)

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

        The value is written into the state of COPASI (`_write_value`), the
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
        _write_value(mc.getMathObject(entity.getValueReference()), number)
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

    def set_time(self, time: float) -> None:
        """Set the current time of the model, between two integrations.

        An integration of COPASI continues from the current time; the values
        which depend on the time are updated.

        Args:
            time: the time.
        """
        mc = self.mc
        _write_value(mc.getMathObject(self.cmodel.getValueReference()), time)
        _update_current_values(mc)

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
        unknown = sorted(
            set(settings) - LSODA_PARAMETERS.keys() - ROADRUNNER_ONLY_SETTINGS
        )
        if unknown:
            raise ValueError(
                f"The integrator '{INTEGRATOR}' of COPASI has no settings {unknown}, "
                f"its settings are {sorted(LSODA_PARAMETERS)}."
            )
        ignored = sorted(
            (set(settings) & ROADRUNNER_ONLY_SETTINGS) - self._ignored_reported
        )
        if ignored:
            self._ignored_reported.update(ignored)
            logger.warning(
                "COPASI has no equivalent of the settings %s of roadrunner, the "
                "model '%s' ignores them.",
                ignored,
                self.label,
            )
        method = self.dm.getTask("Time-Course").getMethod()
        for key, value in settings.items():
            if key not in LSODA_PARAMETERS:
                continue
            parameter = method.getParameter(LSODA_PARAMETERS[key])
            if key == "absolute_tolerance":
                if isinstance(value, bool):
                    raise ValueError("The absolute tolerance is a number.")
                tolerance = AbsoluteTolerance.of(value)
                parameter.setDblValue(self._scalar_tolerance(tolerance))
                self.absolute_tolerance = tolerance
            elif key == "maximum_num_steps":
                if (
                    isinstance(value, bool | AbsoluteTolerance)
                    or not float(value).is_integer()
                    or value < 1
                ):
                    raise ValueError(
                        f"The maximum number of steps is a positive integer, not "
                        f"{value}."
                    )
                parameter.setUIntValue(int(value))
            else:
                if isinstance(value, bool | AbsoluteTolerance) or not value >= 0:
                    raise ValueError(f"The setting '{key}' is a number, not {value}.")
                parameter.setDblValue(float(value))

    def _scalar_tolerance(self, tolerance: AbsoluteTolerance) -> float:
        """Get the one absolute tolerance of COPASI of a tolerance per kind.

        The smallest tolerance of the kinds and of the overrides, reported
        once per model when they differ, times `COPASI_ATOL_FACTOR`.
        """
        values = {tolerance.amount, tolerance.concentration, tolerance.other}
        values |= set(dict(tolerance.ids).values())
        smallest = min(values)
        if len(values) > 1 and not self._tolerance_reported:
            self._tolerance_reported = True
            logger.warning(
                "COPASI has one absolute tolerance, the model '%s' integrates "
                "with the smallest one of %s, %s, times %s.",
                self.label,
                tolerance,
                smallest,
                COPASI_ATOL_FACTOR,
            )
        return smallest * COPASI_ATOL_FACTOR

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
            states += [
                e.getId()
                for e in (
                    *sbml_model.getListOfParameters(),
                    *sbml_model.getListOfCompartments(),
                )
                if e.getId() in self.symbols.rate_rules
            ]
            self._state_ids = tuple(states)
        return list(self._state_ids)


def _check_supported(model: libsbml.Model, label: str) -> None:
    """Check that COPASI simulates a model as roadrunner does.

    COPASI loads the following without an error and simulates them otherwise
    than roadrunner: it ignores algebraic rules and the conversion factors
    of the model and of the species in a time course, makes the stoichiometry
    of a non-constant species reference, of a stoichiometry math and of an
    initial assignment of a species reference a constant, and simulates the
    delay csymbol otherwise.

    Args:
        model: the model of the SBML.
        label: the name of the model in a message.

    Raises:
        ValueError: naming the feature and its element, for a model with one.
    """

    def unsupported(feature: str) -> ValueError:
        return ValueError(
            f"The model '{label}' has {feature}; COPASI does not simulate it as "
            f"roadrunner, the model cannot be simulated with COPASI."
        )

    for rule in model.getListOfRules():
        if rule.isAlgebraic():
            raise unsupported("an algebraic rule, which COPASI ignores")
    if model.isSetConversionFactor():
        raise unsupported(
            f"the conversion factor '{model.getConversionFactor()}' of the model, "
            f"which a time course of COPASI ignores"
        )
    for species in model.getListOfSpecies():
        if species.isSetConversionFactor():
            raise unsupported(
                f"the conversion factor '{species.getConversionFactor()}' of the "
                f"species '{species.getId()}', which a time course of COPASI ignores"
            )
    level = model.getLevel()
    references: set[str] = set()
    for reaction in model.getListOfReactions():
        for sr in (*reaction.getListOfReactants(), *reaction.getListOfProducts()):
            if sr.isSetId():
                references.add(sr.getId())
            if sr.isSetStoichiometryMath():
                raise unsupported(
                    f"the stoichiometry math of the species '{sr.getSpecies()}' in "
                    f"the reaction '{reaction.getId()}', which COPASI makes a "
                    f"constant"
                )
            if level >= 3 and sr.isSetConstant() and not sr.getConstant():
                name = sr.getId() or sr.getSpecies()
                raise unsupported(
                    f"the non-constant species reference '{name}' of the reaction "
                    f"'{reaction.getId()}', whose stoichiometry COPASI makes a "
                    f"constant"
                )
    for assignment in model.getListOfInitialAssignments():
        if assignment.getSymbol() in references:
            raise unsupported(
                f"an initial assignment of the species reference "
                f"'{assignment.getSymbol()}', which COPASI evaluates once into a "
                f"constant"
            )
    for where, ast in _maths(model):
        if _reads_delay(ast):
            raise unsupported(f"the delay csymbol in the {where}")


def _maths(model: libsbml.Model) -> Iterator[tuple[str, libsbml.ASTNode]]:
    """Get every math of a model with the name of its element."""
    for function in model.getListOfFunctionDefinitions():
        yield f"function definition '{function.getId()}'", function.getMath()
    for assignment in model.getListOfInitialAssignments():
        yield f"initial assignment of '{assignment.getSymbol()}'", assignment.getMath()
    for rule in model.getListOfRules():
        kind = "rate rule" if rule.isRate() else "assignment rule"
        if rule.isAlgebraic():
            kind = "algebraic rule"
        yield f"{kind} of '{rule.getVariable()}'", rule.getMath()
    for constraint in model.getListOfConstraints():
        yield "constraint", constraint.getMath()
    for reaction in model.getListOfReactions():
        if reaction.isSetKineticLaw():
            law = reaction.getKineticLaw()
            yield f"kinetic law of the reaction '{reaction.getId()}'", law.getMath()
    for event in model.getListOfEvents():
        where = f"event '{event.getId()}'"
        if event.isSetTrigger():
            yield where, event.getTrigger().getMath()
        if event.isSetDelay():
            yield where, event.getDelay().getMath()
        if event.isSetPriority():
            yield where, event.getPriority().getMath()
        for assignment in event.getListOfEventAssignments():
            yield where, assignment.getMath()


#: the definition URL of the delay csymbol
_DELAY = "http://www.sbml.org/sbml/symbols/delay"


def _reads_delay(math: libsbml.ASTNode | None) -> bool:
    """Whether a math reads the delay csymbol, found by its definition URL."""
    stack = [math] if math is not None else []
    while stack:
        node = stack.pop()
        if node.getDefinitionURLString() == _DELAY:
            return True
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return False


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


def _vector(references: Sequence[Any]) -> COPASI.ObjectStdVector:
    """Get the vector of COPASI of references to objects."""
    vector = COPASI.ObjectStdVector()
    for reference in references:
        vector.append(reference)
    return vector


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
    constants = {
        species.getConversionFactor()
        for species in sbml_model.getListOfSpecies()
        if species.isSetConversionFactor()
    }
    if sbml_model.isSetConversionFactor():
        constants.add(sbml_model.getConversionFactor())
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
