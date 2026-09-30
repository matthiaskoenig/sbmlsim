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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

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
        targets: id of a parameter of the source which got a rule -> whether
            it was constant before.
    """

    source: str
    created: tuple[str, ...] = ()
    targets: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the attributes."""
        object.__setattr__(self, "created", tuple(self.created))
        object.__setattr__(self, "targets", dict(self.targets))

    def xml(self) -> str:
        """Get the record as the element of the annotation."""
        targets = "".join(
            f'<target id="{sid}" constant="{"true" if constant else "false"}"/>'
            for sid, constant in self.targets.items()
        )
        return (
            f'<{ELEMENT} xmlns="{NAMESPACE}" source="{self.source}">'
            f"<created>{' '.join(self.created)}</created>{targets}</{ELEMENT}>"
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
    return Derivation(source=source, created=tuple(created), targets=targets)


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
            source=Path(source).name, created=tuple(created), targets=dict(targets)
        )
    else:
        derivation = Derivation(
            source=earlier.source,
            created=(*earlier.created, *created),
            targets={
                **earlier.targets,
                **{
                    sid: constant
                    for sid, constant in targets.items()
                    if sid not in earlier.created
                },
            },
        )
        _remove_record(model)
    node = libsbml.XMLNode.convertStringToXMLNode(derivation.xml())
    if node is None:
        raise ValueError(
            f"The record of the derivation of '{source}' is not XML: {derivation.xml()}"
        )
    if model.isSetAnnotation():
        success = model.getAnnotation().addChild(node)
    else:
        success = model.setAnnotation(
            libsbml.XMLNode.convertStringToXMLNode(
                f"<annotation>{derivation.xml()}</annotation>"
            )
        )
    if success != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"The record of the derivation of '{source}' cannot be written into "
            f"the model '{model.getId()}': "
            f"{libsbml.OperationReturnValue_toString(success)}"
        )
    return derivation


def strip_derivation(sbml_path: Path) -> tuple[libsbml.SBMLDocument, Derivation]:
    """Undo the derivation of a model.

    Args:
        sbml_path: the derived model.

    Returns:
        The document of the source model, i.e. the model without the
        parameters and rules which were added and without the record, and
        the record.

    Raises:
        ValueError: if the model cannot be read, is not derived, or a created
            parameter or a target is not in it.
    """
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        raise ValueError(f"'{sbml_path}' is not an SBML model")
    derivation = derivation_of(model)
    if derivation is None:
        raise ValueError(f"The model '{sbml_path}' is not derived from a model")
    for sid in derivation.created:
        model.removeRuleByVariable(sid)
        if model.removeParameter(sid) is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation created"
            )
    for sid, constant in derivation.targets.items():
        model.removeRuleByVariable(sid)
        parameter: libsbml.Parameter | None = model.getParameter(sid)
        if parameter is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation set"
            )
        parameter.setConstant(constant)
    _remove_record(model)
    logger.info(
        "The model '%s' is the model '%s' without %d parameters",
        Path(sbml_path).name,
        derivation.source,
        len(derivation.created),
    )
    return document, derivation
