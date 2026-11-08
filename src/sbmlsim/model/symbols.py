"""The symbols of a model which a simulation can change.

A target of a change is named like a selection of roadrunner: `S` is the
amount of a species, `[S]` its concentration, a parameter or a compartment is
its id. `ModelSymbols` reads what the targets of a model are from its SBML once
and answers what kind a target is, which the compile step and the executor
need: a compartment change keeps the concentration of the species in it, a
parameter with an initial assignment cannot be set before the initialization
without deriving the model, and the target of an assignment rule cannot be set
at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import libsbml

#: the prefix of the id sbmlsim gives an event of a model without one, by its
#: position among the events of the model, see `EventSymbols.eid`
EVENT_PREFIX = "__sbmlsim_event_"


@dataclass(frozen=True)
class EventSymbols:
    """An event of a model in the selections of roadrunner.

    The executor fires an event whose trigger a change of a simulation makes
    true, which roadrunner does not, see `sbmlsim.simulator.executor`.

    Attributes:
        eid: id of the event, `EVENT_PREFIX` and the position of the event
            among the events of the model for an event without an id, the
            id the engines give it and the messages name it by.
        trigger: the trigger, MathML of selections which
            `sbmlsim.sbml_math.compile_math` compiles, `None` without a
            trigger. A trigger which it does not compile, one with `rateOf`
            or `delay`, is reported by the executor and a change does not
            fire its event.
        delayed: whether the event has a delay.
        assignments: the selection and the MathML of every assignment.
        priority: the MathML of the priority, or `None` without a priority.
        use_values_from_trigger_time: evaluate assignments when triggered.
        persistent: keep the event when its trigger becomes false.
        initial_value: the value of the trigger before the time 0, an event
            whose trigger is false then and true at the start fires at the
            start.
    """

    eid: str
    trigger: str | None
    delayed: bool
    assignments: tuple[tuple[str, str], ...]
    priority: str | None = None
    use_values_from_trigger_time: bool = True
    persistent: bool = True
    initial_value: bool = True


class TargetKind(StrEnum):
    """The kind of a target of a change."""

    PARAMETER = "parameter"
    COMPARTMENT = "compartment"
    SPECIES_AMOUNT = "species amount"
    SPECIES_CONCENTRATION = "species concentration"


def _read(sbml: str | Path) -> libsbml.SBMLDocument:
    """Read the document of SBML, a path or the SBML itself.

    Raises:
        ValueError: if the SBML has no model.
    """
    text = str(sbml)
    doc: libsbml.SBMLDocument = (
        libsbml.readSBMLFromString(text)
        if text.lstrip().startswith("<")
        else libsbml.readSBMLFromFile(text)
    )
    if doc.getModel() is None:
        raise ValueError(f"The SBML has no model: '{text[:200]}'")
    return doc


@dataclass(frozen=True)
class ModelSymbols:
    """The entities of a model a simulation refers to.

    Attributes:
        parameters: ids of the global parameters.
        compartments: ids of the compartments.
        species: ids of the species.
        species_compartment: species -> its compartment.
        only_substance: species with `hasOnlySubstanceUnits=true`.
        initial_assignments: entities with an initial assignment.
        assignment_rules: entities which are the variable of an assignment
            rule.
        rate_rules: entities which are the variable of a rate rule.
        initial_assignment_order: the keys of
            `initial_assignment_dependencies`, an entity after the ones it
            depends on.
        initial_assignment_dependencies: entity whose initial value the
            initialization evaluates again -> the entities it depends on,
            through assignment rules: the math of an initial assignment, and
            for a species in a compartment of an assignment rule, whose
            initial value is converted with the size of the compartment at
            the initialization, the compartment (#344). Such a species is a
            key also without an initial assignment.
        initial_concentration: species whose initial value is a
            concentration, which a change of their compartment before the
            initialization keeps.
        initial_concentration_values: species without an initial assignment
            -> its initial concentration, which the initialization sets again
            when the size of its compartment of an assignment rule changed.
        events: the symbols of the events.
        time_dependent: whether a math of the model reads the csymbol time or
            delay, or an event has a delay, see `sbmlsim.simulator.executor`.
            roadrunner keeps a pending event at a time of its clock, so an
            event with a delay needs the absolute time.
    """

    parameters: frozenset[str]
    compartments: frozenset[str]
    species: frozenset[str]
    species_compartment: dict[str, str]
    only_substance: frozenset[str]
    initial_assignments: frozenset[str]
    assignment_rules: frozenset[str]
    rate_rules: frozenset[str]
    initial_assignment_order: tuple[str, ...] = ()
    initial_assignment_dependencies: dict[str, frozenset[str]] | None = None
    initial_concentration: frozenset[str] = frozenset()
    initial_concentration_values: dict[str, float] = field(default_factory=dict)
    events: tuple[EventSymbols, ...] = ()
    time_dependent: bool = False

    @classmethod
    def from_sbml(cls, sbml: str | Path) -> ModelSymbols:
        """Read the symbols of a model.

        Args:
            sbml: path of the SBML file or the SBML itself.

        Returns:
            The symbols of the model.
        """
        doc = _read(sbml)
        model: libsbml.Model = doc.getModel()
        species: list[libsbml.Species] = list(model.getListOfSpecies())
        rules: list[libsbml.Rule] = list(model.getListOfRules())
        rule_symbols = {
            r.getVariable(): _names(r.getMath()) for r in rules if r.isAssignment()
        }
        compartment_of = {
            sp.getId(): sp.getCompartment()
            for sp in species
            if not sp.getHasOnlySubstanceUnits()
        }
        assigned = frozenset(a.getSymbol() for a in model.getListOfInitialAssignments())
        dependencies: dict[str, frozenset[str]] = {}
        for assignment in model.getListOfInitialAssignments():
            reads = _expand(_names(assignment.getMath()), rule_symbols)
            # the concentration of a species depends on its compartment
            reads |= {compartment_of[s] for s in reads if s in compartment_of}
            dependencies[assignment.getSymbol()] = frozenset(reads)
        # the initial value of a concentration species is converted into its
        # amount with the size of its compartment at the initialization, which
        # an assignment rule may change; a compartment of an initial assignment
        # or which is set keeps the concentrations itself (#344)
        for sid, compartment in compartment_of.items():
            if compartment in rule_symbols and sid not in rule_symbols:
                dependencies[sid] = dependencies.get(sid, frozenset()) | frozenset(
                    _expand({compartment}, rule_symbols)
                )
        concentration_species = frozenset(compartment_of)
        maths: list[libsbml.ASTNode | None] = [r.getMath() for r in rules]
        maths += [
            reaction.getKineticLaw().getMath()
            for reaction in model.getListOfReactions()
            if reaction.isSetKineticLaw()
        ]
        maths += [a.getMath() for a in model.getListOfInitialAssignments()]
        # SBML does not allow the time in a function definition, a model may
        # still have one
        maths += [f.getMath() for f in model.getListOfFunctionDefinitions()]
        for event in model.getListOfEvents():
            maths.append(event.getTrigger().getMath() if event.isSetTrigger() else None)
            maths.append(event.getDelay().getMath() if event.isSetDelay() else None)
            maths.append(
                event.getPriority().getMath() if event.isSetPriority() else None
            )
            maths += [a.getMath() for a in event.getListOfEventAssignments()]
        return cls(
            time_dependent=any(_reads_time(m) for m in maths)
            or any(event.isSetDelay() for event in model.getListOfEvents()),
            events=tuple(
                _event_symbols(
                    event,
                    k,
                    concentration_species,
                    model.getListOfFunctionDefinitions(),
                )
                for k, event in enumerate(model.getListOfEvents())
            ),
            initial_assignment_order=_order(dependencies),
            initial_assignment_dependencies=dependencies,
            initial_concentration_values={
                sp.getId(): sp.getInitialConcentration()
                for sp in species
                if sp.getId() in dependencies
                and sp.isSetInitialConcentration()
                and sp.getId() not in assigned
            },
            initial_concentration=frozenset(
                sp.getId()
                for sp in species
                if (
                    sp.isSetInitialConcentration()
                    or (
                        not sp.isSetInitialAmount()
                        and not sp.getHasOnlySubstanceUnits()
                    )
                )
                or (sp.getId() in assigned and not sp.getHasOnlySubstanceUnits())
            ),
            parameters=frozenset(p.getId() for p in model.getListOfParameters()),
            compartments=frozenset(c.getId() for c in model.getListOfCompartments()),
            species=frozenset(s.getId() for s in species),
            species_compartment={s.getId(): s.getCompartment() for s in species},
            only_substance=frozenset(
                s.getId() for s in species if s.getHasOnlySubstanceUnits()
            ),
            initial_assignments=assigned,
            assignment_rules=frozenset(
                r.getVariable() for r in rules if r.isAssignment()
            ),
            rate_rules=frozenset(r.getVariable() for r in rules if r.isRate()),
        )

    @staticmethod
    def entity(target: str) -> str:
        """Get the entity of a target, the species of a concentration `[S]`."""
        if target.startswith("[") and target.endswith("]"):
            return target[1:-1]
        return target

    def kind(self, target: str) -> TargetKind:
        """Get the kind of a target.

        Args:
            target: `S` for the amount of a species, `[S]` for its
                concentration, or the id of a parameter or a compartment.

        Returns:
            The kind of the target.

        Raises:
            ValueError: if the model has no such entity, if a concentration is
                not one of a species, or if the entity is the variable of an
                assignment rule, which a change cannot set.
        """
        entity = self.entity(target)
        if entity in self.assignment_rules:
            raise ValueError(
                f"'{target}' is the variable of an assignment rule, its value "
                f"is the rule and a change cannot set it."
            )
        if entity != target:
            if entity not in self.species:
                raise ValueError(
                    f"'{target}' is a concentration, but '{entity}' is no "
                    f"species of the model."
                )
            return TargetKind.SPECIES_CONCENTRATION
        if entity in self.species:
            return TargetKind.SPECIES_AMOUNT
        if entity in self.compartments:
            return TargetKind.COMPARTMENT
        if entity in self.parameters:
            return TargetKind.PARAMETER
        raise ValueError(
            f"'{target}' is not a parameter, a compartment or a species of the model."
        )


def _names(math: libsbml.ASTNode | None) -> set[str]:
    """Get the identifiers a math reads, without the names of functions."""
    names: set[str] = set()
    if math is None:
        return names
    stack = [math]
    while stack:
        node = stack.pop()
        # `isName` and not the type: with a second SWIG module of libsbml
        # loaded, e.g. by `sbmlmath`, `getType` answers a pointer of an enum
        if node.isName() and node.getName():
            names.add(node.getName())
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return names


#: the definition URLs of the csymbols which read the time
_TIME_CSYMBOLS = frozenset(
    {
        "http://www.sbml.org/sbml/symbols/time",
        "http://www.sbml.org/sbml/symbols/delay",
    }
)


def _reads_time(math: libsbml.ASTNode | None) -> bool:
    """Get whether a math reads the time or a delay (the csymbols, not an id).

    The csymbols are found by their definition URL and not by the type of the
    node, which is not reliable with a second SWIG module of libsbml, see
    `_names`.
    """
    if math is None:
        return False
    stack = [math]
    while stack:
        node = stack.pop()
        if node.getDefinitionURLString() in _TIME_CSYMBOLS:
            return True
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return False


def _expand(names: set[str], rules: dict[str, set[str]]) -> set[str]:
    """Expand identifiers through the assignment rules which set them."""
    expanded: set[str] = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in expanded:
            continue
        expanded.add(name)
        stack.extend(rules.get(name, ()))
    return expanded


def _order(dependencies: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """Order the initial assignments, an assignment after the ones it reads.

    Raises:
        ValueError: if the initial assignments read each other in a cycle.
    """
    order: list[str] = []
    done: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = sorted(
            target
            for target, reads in remaining.items()
            if not (reads & set(remaining)) - {target}
        )
        if not ready:
            raise ValueError(
                f"The initial assignments of {sorted(remaining)} read each other "
                f"in a cycle."
            )
        for target in ready:
            order.append(target)
            done.add(target)
            del remaining[target]
    return tuple(order)


def selection_math(
    math: libsbml.ASTNode | None,
    concentrations: frozenset[str],
    functions: libsbml.ListOfFunctionDefinitions,
) -> str | None:
    """Get the math of a model as MathML of selections.

    A species whose value is a concentration in the math of the model, i.e. a
    species without only substance units, is its concentration `[S]`, every
    other identifier its id. The function definitions of the model are
    expanded, the units of the numbers removed, see
    `sbmlsim.sbml_math.compile_math`.

    Args:
        math: the math.
        concentrations: the species which are concentrations in the math.
        functions: the function definitions of the model.

    Returns:
        The MathML, `None` without math.
    """
    if math is None:
        return None
    node: libsbml.ASTNode = math.deepCopy()
    if functions.size():
        libsbml.SBMLTransforms.replaceFD(node, functions)
    stack = [node]
    while stack:
        n = stack.pop()
        stack.extend(n.getChild(k) for k in range(n.getNumChildren()))
        # `isName` and not the type, see `_names`; a csymbol has a definition
        if (
            n.isName()
            and not n.getDefinitionURLString()
            and n.getName() in concentrations
        ):
            n.setName(f"[{n.getName()}]")
        if n.isSetUnits():
            n.unsetUnits()
    return str(libsbml.writeMathMLToString(node))


def _event_symbols(
    event: libsbml.Event,
    position: int,
    concentrations: frozenset[str],
    functions: libsbml.ListOfFunctionDefinitions,
) -> EventSymbols:
    """Read an event of a model, an event without an id named `EVENT_PREFIX` and its position."""
    trigger: libsbml.Trigger | None = event.getTrigger()
    assignments = []
    for assignment in event.getListOfEventAssignments():
        variable = assignment.getVariable()
        target = f"[{variable}]" if variable in concentrations else variable
        math = selection_math(assignment.getMath(), concentrations, functions)
        if math is not None:
            assignments.append((target, math))
    return EventSymbols(
        eid=event.getId() or f"{EVENT_PREFIX}{position}",
        trigger=None
        if trigger is None
        else selection_math(trigger.getMath(), concentrations, functions),
        delayed=event.isSetDelay(),
        assignments=tuple(assignments),
        priority=selection_math(
            event.getPriority().getMath(), concentrations, functions
        )
        if event.isSetPriority()
        else None,
        use_values_from_trigger_time=event.getUseValuesFromTriggerTime(),
        persistent=trigger.getPersistent() if trigger is not None else True,
        initial_value=trigger.getInitialValue() if trigger is not None else True,
    )
