"""The interactive panels of a model in an annotation of its SBML.

A panel (`sbmlsim.panel`) belongs to the model and not to a python checkout, so
it is declared as an element of the annotation of the `<model>` element, in the
namespace `PANEL_NS`, and travels with the SBML and with the COMBINE archive of
a release:

    <annotation>
      <panels xmlns="https://sbmlsim.org/ns/panel/1">
        <panel id="renal" title="Renal impairment" subtitle="hctz, single oral dose" xLabel="time [hr]" yLabel="plasma concentration" yMax="2" xMax="24">
          <simulation variant="hctz_body" end="1440" points="200" xUnit="hr">
            <preinit target="BW" value="75"/>
            <derived target="f_renal_function" formula="min(crcl / 110, 1)"/>
          </simulation>
          <knob id="dose" target="PODOSE_hctz" label="dose" unit="mg" kind="range" min="12.5" max="100" steps="9" default="25"/>
          <knob id="crcl" label="creatinine clearance" unit="ml/min" kind="range" min="10" max="150" steps="8" default="110">
            <option label="normal" value="110"/>
            <option label="severe impairment" value="20"/>
          </knob>
          <curve id="[Cve_hctz]" label="hctz, plasma" unit="uM" scale="linear"/>
          <data id="vormfelde2006" label="Vormfelde 2006" curve="[Cve_hctz]" dataset="Vormfelde2006_Fig2" x="time" y="mean" yerr="mean_sd" conversion="1/(297.7 g/mole)">
            <select column="label" value="hctz25"/>
            <at knob="crcl" value="110"/>
          </data>
        </panel>
      </panels>
    </annotation>

An attribute is the field of the object of `sbmlsim.panel` it carries, with the
exceptions `xLabel` for `x_label`, `yLabel` for `y_label`, `yMin`, `yMax` and
`xMax` for the limits of the axes `y_min`, `y_max` and `x_max` and `xUnit` for
`x_unit`, which are camel case; an attribute which is not set means the default
of its field, e.g. an axis without a limit or a knob without a target, which
is an input of the formulas of the derived targets alone. The
`end` of the simulation is in the time unit of the model (a day of a model of
minutes above) and `xUnit` is the unit its time axis is delivered in. The
`option` children of a knob are its `choices`, the choices of a knob of the
kind `choice` and the named stops of one of the kind `range`,
the `preinit` children of the simulation are its `preinit`, its `derived`
children its `derived`, each of a `target` and its `formula`, the `select`
children of a series of the data are its `sel`, the rows of its dataset, each
of them of a `column` and a `value`, and its `at` children are its `at`, the
state of the knobs it was measured under, each of them of a `knob` and a
`value`. The `dataset` of a series is the id of a figure or a table of a study
and no path, its `curve` is the curve of the panel it belongs under, whose
unit it is delivered in, and its `conversion` is a quantity its values are
multiplied by, the molecular weight of a substance above. The tables
`_PANEL_ATTRIBUTES`, `_SIMULATION_ATTRIBUTES`, `_KNOB_ATTRIBUTES`,
`_CHOICE_ATTRIBUTES`, `_CURVE_ATTRIBUTES` and `_DATA_ATTRIBUTES` own these
names, so the reader and the writer cannot disagree: `panels_from_sbml` reads
the annotation, `panels_to_annotation` and `write_panels` write it, each the
inverse of the other for every field.

A text which is empty is no attribute either, because libsbml does not write
one, and is read back as the default of its field: the default of `subtitle`
is empty, and a panel, a knob, a curve and a series of the data each need a
title or a label which is not.

`panels_from_sbml` never raises for a panel it cannot read: it reports every
such panel as a `PanelProblem` and reads the others, so one malformed panel
does not take the model down. The element is read and written with libsbml
alone, so neither a simulator nor a project is imported.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import quoteattr

import libsbml

from sbmlsim.panel import (
    CONTRACT,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
    panel_from_payload,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PANEL_NS",
    "PanelProblem",
    "panels_from_sbml",
    "panels_to_annotation",
    "write_panels",
]

#: the namespace of the annotation of the panels, the version of its XML
PANEL_NS: Final = "https://sbmlsim.org/ns/panel/1"

#: the element of the annotation which carries the panels
PANELS_ELEMENT: Final = "panels"
#: the element of one panel
PANEL_ELEMENT: Final = "panel"
#: the element of the simulation of a panel
SIMULATION_ELEMENT: Final = "simulation"
#: the element of one value of the pre-initialization of the simulation
PREINIT_ELEMENT: Final = "preinit"
#: the element of one derived target of the simulation and its formula
DERIVED_ELEMENT: Final = "derived"
#: the element of one knob of a panel
KNOB_ELEMENT: Final = "knob"
#: the element of one choice of a knob of the kind `choice` or one named stop
#: of a knob of the kind `range`
OPTION_ELEMENT: Final = "option"
#: the element of one curve of a panel
CURVE_ELEMENT: Final = "curve"
#: the element of one series of the study data of a panel
DATA_ELEMENT: Final = "data"
#: the element of one column which selects the rows of such a series
SELECT_ELEMENT: Final = "select"
#: the element of one knob of the state such a series was measured under
AT_ELEMENT: Final = "at"

#: the fields of a panel by the attribute which carries them
_PANEL_ATTRIBUTES: Final = {
    "id": "id",
    "title": "title",
    "subtitle": "subtitle",
    "xLabel": "x_label",
    "yLabel": "y_label",
    "yMin": "y_min",
    "yMax": "y_max",
    "xMax": "x_max",
}
#: the fields of the simulation of a panel, its `preinit` and its `derived`
#: are children
_SIMULATION_ATTRIBUTES: Final = {
    "variant": "variant",
    "end": "end",
    "points": "points",
    "xUnit": "x_unit",
}
#: the fields of a knob, its `choices` are the `option` children
_KNOB_ATTRIBUTES: Final = {
    name: name
    for name in (
        "id",
        "label",
        "target",
        "unit",
        "kind",
        "min",
        "max",
        "steps",
        "default",
    )
}
#: the fields of one choice of a knob
_CHOICE_ATTRIBUTES: Final = {"label": "label", "value": "value"}
#: the fields of a curve
_CURVE_ATTRIBUTES: Final = {name: name for name in ("id", "label", "unit", "scale")}
#: the fields of one series of the study data, its `sel` the `select` children
#: and its `at` the `at` children
_DATA_ATTRIBUTES: Final = {
    name: name
    for name in (
        "id",
        "label",
        "curve",
        "dataset",
        "x",
        "y",
        "yerr",
        "conversion",
        "unit",
    )
}

#: a panel or one of its parts, which the writer reads the fields of
type _Part = Panel | PanelSimulation | Knob | Choice | PanelCurve | PanelData


@dataclass(frozen=True)
class PanelProblem:
    """A panel of the annotation of a model which cannot be read.

    Attributes:
        panel_id: the id of the panel, empty for an element without one.
        reason: why the panel cannot be read, the message of the error.
    """

    panel_id: str
    reason: str


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def panels_from_sbml(
    sbml: Path | str,
) -> tuple[dict[str, Panel], tuple[PanelProblem, ...]]:
    """Read the panels of the annotation of the model of an SBML.

    Args:
        sbml: the path of the SBML file or the SBML itself, i.e. a string
            which starts with `<`.

    Returns:
        The panels by their id, in the order of the annotation, and the
        problems: a panel which is not valid, one of another contract version
        and one of an id which another panel of the model has as well, which
        keeps the first of them. A model without the annotation has neither.

    Raises:
        FileNotFoundError: for a path which is no file.
        ValueError: for a source which is no SBML model; a panel which cannot
            be read is a problem and no error.
    """
    model: libsbml.Model = _document_of(sbml).getModel()
    panels: dict[str, Panel] = {}
    problems: list[PanelProblem] = []
    for element in _panel_elements(model):
        panel_id: str = element.getAttrValue("id")
        try:
            panel = panel_from_payload(_panel_payload(element))
        except (TypeError, ValueError) as error:
            problems.append(PanelProblem(panel_id, str(error)))
            continue
        if panel.id in panels:
            problems.append(
                PanelProblem(
                    panel.id,
                    f"The model declares the panel '{panel.id}' twice; the first "
                    "one is read.",
                )
            )
            continue
        panels[panel.id] = panel
    return panels, tuple(problems)


def _document_of(sbml: Path | str) -> libsbml.SBMLDocument:
    """Read the document of an SBML file or of the SBML itself.

    Args:
        sbml: the path of the SBML file or the SBML itself.

    Returns:
        The document, which has a model.

    Raises:
        FileNotFoundError: for a path which is no file.
        ValueError: for a source which is no SBML model.
    """
    content = sbml if isinstance(sbml, str) and sbml.lstrip().startswith("<") else None
    if content is None and not Path(sbml).is_file():
        raise FileNotFoundError(f"The SBML file '{sbml}' does not exist.")
    document: libsbml.SBMLDocument = (
        libsbml.readSBMLFromString(content)
        if content is not None
        else libsbml.readSBMLFromFile(str(sbml))
    )
    if document.getModel() is None:
        where = "The SBML" if content is not None else f"'{sbml}'"
        raise ValueError(f"{where} is no SBML model, it has no panels.")
    return document


def _panels_index(model: libsbml.Model) -> int:
    """Get the index of the `panels` element in the annotation, `-1` for none."""
    if not model.isSetAnnotation():
        return -1
    annotation: libsbml.XMLNode = model.getAnnotation()
    for k in range(annotation.getNumChildren()):
        child: libsbml.XMLNode = annotation.getChild(k)
        if child.getName() == PANELS_ELEMENT and child.getURI() == PANEL_NS:
            return k
    return -1


def _panel_elements(model: libsbml.Model) -> Iterator[libsbml.XMLNode]:
    """Iterate the `panel` elements of the annotation of a model, in their order.

    Yields:
        The elements; none for a model without the annotation.
    """
    k = _panels_index(model)
    if k < 0:
        return
    yield from _children(model.getAnnotation().getChild(k), PANEL_ELEMENT)


def _children(element: libsbml.XMLNode, name: str) -> Iterator[libsbml.XMLNode]:
    """Iterate the children of an element which have a name, in their order.

    Yields:
        The children; the text between two elements has no name and is left
        out.
    """
    for k in range(element.getNumChildren()):
        child: libsbml.XMLNode = element.getChild(k)
        if child.getName() == name:
            yield child


def _fields_of(element: libsbml.XMLNode, names: Mapping[str, str]) -> dict[str, Any]:
    """Get the fields an element declares, by the name of the field.

    Args:
        element: the element.
        names: the field of every attribute, by the attribute.

    Returns:
        The text of every attribute which is set, by its field; an attribute
        which is not set is left out, so the field keeps its default.
    """
    return {
        field: element.getAttrValue(attribute)
        for attribute, field in names.items()
        if element.hasAttr(attribute)
    }


def _panel_payload(element: libsbml.XMLNode) -> dict[str, Any]:
    """Get the `panel/1` payload of a `panel` element, `panel_from_payload` reads it."""
    return {
        "contract": CONTRACT,
        "panel": _fields_of(element, _PANEL_ATTRIBUTES),
        "axes": [_knob_payload(knob) for knob in _children(element, KNOB_ELEMENT)],
        "curves": [
            _fields_of(curve, _CURVE_ATTRIBUTES)
            for curve in _children(element, CURVE_ELEMENT)
        ],
        "data": [_data_payload(data) for data in _children(element, DATA_ELEMENT)],
        "simulation": _simulation_payload(element),
    }


def _data_payload(element: libsbml.XMLNode) -> dict[str, Any]:
    """Get the payload of a `data` element, its `select` and `at` children.

    Args:
        element: the `data` element.

    Returns:
        The fields of the element with its `select` children as its `sel` and
        its `at` children as its `at`.

    Raises:
        ValueError: naming the series, for a `select` child without a
            `column`, which selects no rows, and for an `at` child without a
            `knob`, which is the state of no knob; `panels_from_sbml` reports
            either as a problem of its panel.
    """
    payload = _fields_of(element, _DATA_ATTRIBUTES)
    sel: dict[str, str] = {}
    for select in _children(element, SELECT_ELEMENT):
        column = select.getAttrValue("column")
        if not column:
            raise ValueError(
                f"A '{SELECT_ELEMENT}' of the series "
                f"'{element.getAttrValue('id')}' has no 'column': the rows of a "
                f"series are selected by the value of a column of its dataset."
            )
        sel[column] = select.getAttrValue("value")
    if sel:
        payload["sel"] = sel
    at: dict[str, str] = {}
    for condition in _children(element, AT_ELEMENT):
        knob = condition.getAttrValue("knob")
        if not knob:
            raise ValueError(
                f"An '{AT_ELEMENT}' of the series "
                f"'{element.getAttrValue('id')}' has no 'knob': the state a "
                f"series was measured under is the value a knob of the panel "
                f"takes."
            )
        at[knob] = condition.getAttrValue("value")
    if at:
        payload["at"] = at
    return payload


def _knob_payload(element: libsbml.XMLNode) -> dict[str, Any]:
    """Get the payload of a `knob` element, its `option` children its choices."""
    payload = _fields_of(element, _KNOB_ATTRIBUTES)
    choices = [
        _fields_of(option, _CHOICE_ATTRIBUTES)
        for option in _children(element, OPTION_ELEMENT)
    ]
    if choices:
        payload["choices"] = choices
    return payload


def _simulation_payload(element: libsbml.XMLNode) -> dict[str, Any]:
    """Get the payload of the `simulation` of a `panel` element.

    Args:
        element: the `panel` element.

    Returns:
        The fields of its first `simulation` element with its `preinit` and
        its `derived` children, empty for a panel without one, which is no
        panel.
    """
    simulations = list(_children(element, SIMULATION_ELEMENT))
    if not simulations:
        return {}
    payload = _fields_of(simulations[0], _SIMULATION_ATTRIBUTES)
    preinit = {
        value.getAttrValue("target"): value.getAttrValue("value")
        for value in _children(simulations[0], PREINIT_ELEMENT)
    }
    if preinit:
        payload["preinit"] = preinit
    derived = {
        value.getAttrValue("target"): value.getAttrValue("formula")
        for value in _children(simulations[0], DERIVED_ELEMENT)
    }
    if derived:
        payload["derived"] = derived
    return payload


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
def panels_to_annotation(panels: Sequence[Panel]) -> str:
    """Get the XML of the annotation of the panels of a model.

    Args:
        panels: the panels, in the order they are written.

    Returns:
        The `annotation` element with the `panels` element of the namespace
        `PANEL_NS`, which `panels_from_sbml` reads back.
    """
    return f"<annotation>{_panels_xml(panels)}</annotation>"


def write_panels(sbml_in: Path, sbml_out: Path, panels: Sequence[Panel]) -> None:
    """Write an SBML with the annotation of the panels of its model.

    The `panels` element of an annotation the model carries is replaced and
    its other annotations, e.g. its RDF metadata, stay; no panel removes the
    element. Nothing of the model itself is changed, so it simulates as it
    did.

    Args:
        sbml_in: the SBML file which is read.
        sbml_out: the SBML file which is written.
        panels: the panels of the model, in the order they are written.

    Raises:
        FileNotFoundError: if the SBML which is read is no file.
        ValueError: for a source which is no SBML model, and if the
            annotation cannot be written into the model.
        OSError: if the SBML cannot be written.
    """
    document: libsbml.SBMLDocument = _document_of(sbml_in)
    model: libsbml.Model = document.getModel()
    _remove_panels(model)
    if panels:
        _add_panels(model, panels)
    if libsbml.writeSBMLToFile(document, str(sbml_out)) != 1:
        raise OSError(f"The SBML with the panels cannot be written into '{sbml_out}'.")
    logger.info(
        "The annotation of %d panels was written into '%s'", len(panels), sbml_out
    )


def _remove_panels(model: libsbml.Model) -> None:
    """Remove the `panels` element from the annotation of a model, if it has one."""
    k = _panels_index(model)
    if k < 0:
        return
    annotation: libsbml.XMLNode = model.getAnnotation()
    annotation.removeChild(k)
    if annotation.getNumChildren() == 0:
        model.unsetAnnotation()


def _add_panels(model: libsbml.Model, panels: Sequence[Panel]) -> None:
    """Add the `panels` element to the annotation of a model.

    The element is added to the annotation in place, as the record of a
    derivation is (`sbmlsim.model.provenance`): libsbml checks an annotation
    it is given as a whole against the `metaid` of the model, which a model of
    the wild does not always have.

    Args:
        model: the model, without a `panels` element.
        panels: the panels, in the order they are written.

    Raises:
        ValueError: if the annotation cannot be written into the model.
    """
    xml = (
        _panels_xml(panels) if model.isSetAnnotation() else panels_to_annotation(panels)
    )
    node: libsbml.XMLNode | None = libsbml.XMLNode.convertStringToXMLNode(xml)
    if node is None:
        raise ValueError(f"The annotation of the panels is no XML: {xml}")
    success: int = (
        model.getAnnotation().addChild(node)
        if model.isSetAnnotation()
        else model.setAnnotation(node)
    )
    if success != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"The annotation of {len(panels)} panels cannot be written into the "
            f"model '{model.getId()}': "
            f"{libsbml.OperationReturnValue_toString(success)}"
        )


def _panels_xml(panels: Sequence[Panel]) -> str:
    """Get the XML of the `panels` element of the panels."""
    elements = "".join(_panel_xml(panel) for panel in panels)
    return f'<{PANELS_ELEMENT} xmlns="{PANEL_NS}">{elements}</{PANELS_ELEMENT}>'


def _panel_xml(panel: Panel) -> str:
    """Get the XML of the `panel` element of a panel and of its parts."""
    children = "".join(
        (
            _simulation_xml(panel.simulation),
            *(_knob_xml(knob) for knob in panel.knobs),
            *(
                _element_xml(CURVE_ELEMENT, curve, _CURVE_ATTRIBUTES)
                for curve in panel.curves
            ),
            *(_data_xml(data) for data in panel.data),
        )
    )
    return _element_xml(PANEL_ELEMENT, panel, _PANEL_ATTRIBUTES, children)


def _simulation_xml(simulation: PanelSimulation) -> str:
    """Get the XML of the `simulation` element, its `preinit` and `derived` children."""
    preinit = "".join(
        f"<{PREINIT_ELEMENT} target={quoteattr(target)} value={quoteattr(str(value))}/>"
        for target, value in simulation.preinit.items()
    )
    derived = "".join(
        f"<{DERIVED_ELEMENT} target={quoteattr(target)} formula={quoteattr(formula)}/>"
        for target, formula in simulation.derived.items()
    )
    return _element_xml(
        SIMULATION_ELEMENT, simulation, _SIMULATION_ATTRIBUTES, preinit + derived
    )


def _data_xml(data: PanelData) -> str:
    """Get the XML of a `data` element, its rows and its state its children."""
    select = "".join(
        f"<{SELECT_ELEMENT} column={quoteattr(column)} value={quoteattr(value)}/>"
        for column, value in (data.sel or {}).items()
    )
    at = "".join(
        f"<{AT_ELEMENT} knob={quoteattr(knob)} value={quoteattr(str(value))}/>"
        for knob, value in (data.at or {}).items()
    )
    return _element_xml(DATA_ELEMENT, data, _DATA_ATTRIBUTES, select + at)


def _knob_xml(knob: Knob) -> str:
    """Get the XML of a `knob` element, its choices the `option` children."""
    options = "".join(
        _element_xml(OPTION_ELEMENT, choice, _CHOICE_ATTRIBUTES)
        for choice in knob.choices
    )
    return _element_xml(KNOB_ELEMENT, knob, _KNOB_ATTRIBUTES, options)


def _element_xml(
    name: str, part: _Part, names: Mapping[str, str], children: str = ""
) -> str:
    """Get the XML of the element of a panel or of one of its parts.

    Args:
        name: the name of the element.
        part: the panel or the part of it the attributes are read from.
        names: the field of every attribute, by the attribute.
        children: the XML of the children of the element, none without.

    Returns:
        The element; a field which is `None` or empty is no attribute, so it
        is read back as the default of its field.
    """
    attributes = "".join(
        f" {attribute}={quoteattr(str(value))}"
        for attribute, field in names.items()
        if (value := getattr(part, field)) is not None and value != ""
    )
    if not children:
        return f"<{name}{attributes}/>"
    return f"<{name}{attributes}>{children}</{name}>"
