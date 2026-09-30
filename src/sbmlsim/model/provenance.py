"""The record of what was derived from a model.

`sbmlsim` writes models which are derived from the model of a problem: the
model with the compiled networks of a hybrid problem
(`sbmlsim.sciml.compiler.compile_network`) and the model with the observables
which are formulas (`sbmlsim.fit.petab_v2.observables.add_observables`). A fit
simulates the derived model, an export writes the model the problem was
defined with. The derived model therefore carries what was added to it, as an
element of the annotation of the model in the namespace `NAMESPACE`:

    <derived xmlns="https://github.com/matthiaskoenig/sbmlsim/derived" source="lv.xml">
      <created>net1__layer1__weight__0_0 net1__output0__0</created>
      <target id="gamma" constant="true"/>
    </derived>

`created` lists the parameters which were added, with their rules, and
`target` lists the parameters of the source which got a rule, with the value
of `constant` they had. `strip_derivation` undoes both, which gives the source
model. A derivation of a derived model extends the record, so the source is
always the model the problem was defined with.

The element is added to the annotation of the model in place: libsbml checks
an annotation it is given as a whole against the `metaid` of the model, and
the RDF annotation of a model of the wild does not always have one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import libsbml

logger = logging.getLogger(__name__)

#: the namespace of the record
NAMESPACE = "https://github.com/matthiaskoenig/sbmlsim/derived"

#: the element of the record
ELEMENT = "derived"


@dataclass(frozen=True)
class Derivation:
    """What was added to a model.

    Attributes:
        source: name of the file of the model the derivation started from.
        created: ids of the parameters which were added, in the order they
            were added; a rule of such a parameter was added with it.
        targets: the parameters of the source which got a rule, as pairs of
            the id and whether it was constant before, in the order they got
            it; `dict(derivation.targets)` is the lookup. Pairs and not a
            mapping, so the record is a value which hashes, pickles and copies.
    """

    source: str
    created: tuple[str, ...] = ()
    targets: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        """Freeze the attributes, a list becomes a tuple."""
        object.__setattr__(self, "created", tuple(self.created))
        object.__setattr__(
            self,
            "targets",
            tuple((str(sid), bool(constant)) for sid, constant in self.targets),
        )

    def xml(self) -> str:
        """Get the record as the element of the annotation."""
        targets = "".join(
            f"<target id={quoteattr(sid)} "
            f'constant="{"true" if constant else "false"}"/>'
            for sid, constant in self.targets
        )
        return (
            f'<{ELEMENT} xmlns="{NAMESPACE}" source={quoteattr(self.source)}>'
            f"<created>{escape(' '.join(self.created))}</created>{targets}"
            f"</{ELEMENT}>"
        )


def _record_index(model: libsbml.Model) -> int:
    """Get the index of the record in the annotation of a model, `-1` for none."""
    if not model.isSetAnnotation():
        return -1
    annotation: libsbml.XMLNode = model.getAnnotation()
    for k in range(annotation.getNumChildren()):
        child: libsbml.XMLNode = annotation.getChild(k)
        if child.getName() == ELEMENT and child.getURI() == NAMESPACE:
            return k
    return -1


def derivation_of(model: libsbml.Model) -> Derivation | None:
    """Read the record of a model.

    Args:
        model: the model.

    Returns:
        The derivation, `None` for a model which is not derived.

    Raises:
        ValueError: if the record is not valid.
    """
    k = _record_index(model)
    if k < 0:
        return None
    record: libsbml.XMLNode = model.getAnnotation().getChild(k)
    source = record.getAttrValue("source")
    if not source:
        raise ValueError(
            f"The model '{model.getId()}' has a record of its derivation without "
            f"the source"
        )
    created: list[str] = []
    targets: dict[str, bool] = {}
    for i in range(record.getNumChildren()):
        child: libsbml.XMLNode = record.getChild(i)
        if child.getName() == "created":
            text = child.getChild(0).getCharacters() if child.getNumChildren() else ""
            created.extend(text.split())
        elif child.getName() == "target":
            sid = child.getAttrValue("id")
            constant = child.getAttrValue("constant")
            if not sid or constant not in ("true", "false"):
                raise ValueError(
                    f"The model '{model.getId()}' has a record of its derivation "
                    f"with the target '{sid}' and constant '{constant}'"
                )
            targets[sid] = constant == "true"
    return Derivation(
        source=source, created=tuple(created), targets=tuple(targets.items())
    )


def _remove_record(model: libsbml.Model) -> None:
    """Remove the record from the annotation of a model, if it has one."""
    k = _record_index(model)
    if k < 0:
        return
    annotation: libsbml.XMLNode = model.getAnnotation()
    annotation.removeChild(k)
    if annotation.getNumChildren() == 0:
        model.unsetAnnotation()


def record_derivation(
    model: libsbml.Model,
    source: Path,
    created: Iterable[str],
    targets: Mapping[str, bool],
) -> Derivation:
    """Write the record of a derivation into a model.

    A record the model has is extended: the source stays, the created ids
    and the targets are added. A target which the earlier record created is
    a created id and not a target.

    Args:
        model: the derived model.
        source: the file of the model the derivation started from.
        created: ids of the parameters which were added.
        targets: id of a parameter which got a rule -> whether it was
            constant.

    Returns:
        The record which was written.

    Raises:
        ValueError: if the record cannot be written.
    """
    earlier = derivation_of(model)
    if earlier is None:
        derivation = Derivation(
            source=Path(source).name,
            created=tuple(created),
            targets=tuple(targets.items()),
        )
    else:
        derivation = Derivation(
            source=earlier.source,
            created=(*earlier.created, *created),
            targets=tuple(
                {
                    **dict(earlier.targets),
                    **{
                        sid: constant
                        for sid, constant in targets.items()
                        if sid not in earlier.created
                    },
                }.items()
            ),
        )
        _remove_record(model)
    if model.isSetAnnotation():
        node = libsbml.XMLNode.convertStringToXMLNode(derivation.xml())
    else:
        node = libsbml.XMLNode.convertStringToXMLNode(
            f"<annotation>{derivation.xml()}</annotation>"
        )
    if node is None:
        raise ValueError(
            f"The record of the derivation of '{source}' is not XML: {derivation.xml()}"
        )
    if model.isSetAnnotation():
        success = model.getAnnotation().addChild(node)
    else:
        success = model.setAnnotation(node)
    if success != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"The record of the derivation of '{source}' cannot be written into "
            f"the model '{model.getId()}': "
            f"{libsbml.OperationReturnValue_toString(success)}"
        )
    return derivation


def _symbols_of(math: libsbml.ASTNode | None) -> set[str]:
    """Get the names in a math expression, i.e. its `ci` elements."""
    if math is None:
        return set()
    names: set[str] = set()
    stack: list[libsbml.ASTNode] = [math]
    while stack:
        node = stack.pop()
        if node.isName():
            names.add(node.getName())
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return names


def _check_target_rule(
    model: libsbml.Model, sid: str, created: set[str], name: str
) -> None:
    """Check that the rule of a target is the one the derivation wrote.

    The rule of a target sets it to an output of a network, i.e. to a
    parameter the derivation created. Any other rule was written by hand,
    and removing it would drop it without notice.

    Args:
        model: the derived model.
        sid: id of the target.
        created: ids of the parameters the derivation created.
        name: the file of the model, for the messages.

    Raises:
        ValueError: if the target has no rule, or its rule does not refer to
            an output of a network only.
    """
    rule: libsbml.Rule | None = model.getRuleByVariable(sid)
    if rule is None:
        raise ValueError(
            f"The model '{name}' has no rule for the target '{sid}', which its "
            f"derivation set"
        )
    symbols = _symbols_of(rule.getMath())
    if not symbols:
        raise ValueError(
            f"The rule of the target '{sid}' of the model '{name}' does not refer "
            f"to an output of a network, it was changed after the derivation"
        )
    for symbol in sorted(symbols - created):
        raise ValueError(
            f"The rule of the target '{sid}' of the model '{name}' refers to "
            f"'{symbol}', which its derivation did not create, it was changed "
            f"after the derivation"
        )


def _math_elements(model: libsbml.Model) -> Iterator[tuple[str, Any]]:
    """Iterate the elements of a model which have math or refer to a variable.

    Yields:
        The description of the element for a message and the element.
    """
    for rule in model.getListOfRules():
        yield f"{rule.getElementName()} '{rule.getVariable()}'", rule
    for assignment in model.getListOfInitialAssignments():
        yield f"initialAssignment of '{assignment.getSymbol()}'", assignment
    for constraint in model.getListOfConstraints():
        yield "constraint", constraint
    for function in model.getListOfFunctionDefinitions():
        yield f"functionDefinition '{function.getId()}'", function
    for reaction in model.getListOfReactions():
        if reaction.isSetKineticLaw():
            yield (
                f"kineticLaw of the reaction '{reaction.getId()}'",
                reaction.getKineticLaw(),
            )
    for k, event in enumerate(model.getListOfEvents()):
        label = event.getId() or f"#{k + 1}"
        for part in ("Trigger", "Delay", "Priority"):
            if getattr(event, f"isSet{part}")():
                yield (
                    f"{part.lower()} of the event '{label}'",
                    getattr(event, f"get{part}")(),
                )
        for assignment in event.getListOfEventAssignments():
            yield (
                f"eventAssignment of '{assignment.getVariable()}' of the event "
                f"'{label}'",
                assignment,
            )


def _check_references(model: libsbml.Model, removed: set[str], name: str) -> None:
    """Check that nothing in the model refers to a removed parameter.

    Args:
        model: the model, after the removal.
        removed: ids of the parameters which were removed.
        name: the file of the model, for the messages.

    Raises:
        ValueError: if a rule, kinetic law, initial assignment, event,
            constraint or function refers to a removed parameter, which
            happens if the model was changed by hand after the derivation.
    """
    for what, element in _math_elements(model):
        names = _symbols_of(element.getMath())
        for getter in ("getVariable", "getSymbol"):
            if hasattr(element, getter):
                names.add(getattr(element, getter)())
        for sid in sorted(names & removed):
            raise ValueError(
                f"The {what} of the model '{name}' refers to '{sid}', which its "
                f"derivation created, it was changed after the derivation"
            )


def strip_derivation(sbml_path: Path) -> tuple[libsbml.SBMLDocument, Derivation]:
    """Undo the derivation of a model.

    Args:
        sbml_path: the derived model.

    Returns:
        The document of the source model, i.e. the model without the
        parameters and rules which were added and without the record, and
        the record.

    Raises:
        ValueError: if the model cannot be read, is not derived, a created
            parameter or a target is not in it, the rule of a target is not
            the one of the network, or the model refers to a created
            parameter outside of the parts the derivation wrote, i.e. it was
            changed by hand after the derivation.
    """
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        raise ValueError(f"'{sbml_path}' is not an SBML model")
    derivation = derivation_of(model)
    if derivation is None:
        raise ValueError(f"The model '{sbml_path}' is not derived from a model")
    for sid in derivation.created:
        if model.getParameter(sid) is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation created"
            )
    created = set(derivation.created)
    for sid, _ in derivation.targets:
        if model.getParameter(sid) is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation set"
            )
        _check_target_rule(model, sid, created, Path(sbml_path).name)
    for sid in derivation.created:
        model.removeRuleByVariable(sid)
        model.removeParameter(sid)
    for sid, constant in derivation.targets:
        model.removeRuleByVariable(sid)
        model.getParameter(sid).setConstant(constant)
    _check_references(model, created, Path(sbml_path).name)
    _remove_record(model)
    logger.info(
        "The model '%s' is the model '%s' without %d parameters",
        Path(sbml_path).name,
        derivation.source,
        len(derivation.created),
    )
    return document, derivation
