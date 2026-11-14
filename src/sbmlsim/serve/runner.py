"""The models a request may name and the runner of their panels.

`ModelRegistry` is the models of the service. It is built once at startup from
a mapping of model id to project reference (`ModelRegistry.of`) and is the only
way from a request to a model: `entry` answers an id of that mapping and
raises `UnknownModel` for everything else, so a request can never name a
module, a project reference or a path. A model whose project declares an app
(`sbmlsim.panel.App`) is checked there as well (`_check_app`): a page which
names a panel the model does not declare, a variant the project does not
have, two panels of a page which declare one knob id differently
(`ModelEntry.shared`) and a scenario of a page which sets a knob its panels do
not declare or a value which is none of that knob (`_check_scenarios`) are
refused at startup, since the panels of a model are
read from its SBML, which a project is imported without. So are the formulas of
the derived targets of every panel of every variant (`_check_derived`), which
are math of PEtab the declaration does not parse.

`PanelRunner` runs a panel of such a model: the values of its knobs become the
pre-initialization of one `Simulation`, the knobs of the grid become a
`Dimension` of the product of the values of `Knob.values(n)`, a derived target
is the value of its formula over the knobs, a value of that dimension when it
reads a knob of the grid and of the pre-initialization otherwise, and the scan
of the two runs on the warm model of the entry. The curves of the `PanelResult` are the declared
curves of the panel, one row per point of the grid in C order and one row for a
request without a grid, in the unit the curve declares or else in the unit of
the model; the x of the result is the grid of `points + 1` times from 0 to the
`end` of the panel, which are in the time unit of the model, delivered in the
`x_unit` the panel declares or else in that time unit, which `units["x"]`
names and which `units(entry, panel)` answers without a run, so the study
data of a panel (`sbmlsim.serve.data`) is drawn on the axis of its curves. A
point which does not integrate is `NaN` and its flat index is in
`failed`, never an exception: a knob value a reader turns to may be a value
the model does not simulate.

A panel declares its units as text, which may be a unit id of its model, e.g.
`mmole_per_l`, so every such text is resolved in the registry of the model
(`DeliveredUnit`, `PanelUnits`, `_resolved`): the runner is the one place which
holds the model and therefore the one place which resolves them, for the x and
the curves of a run and for the study data of the same panel alike. The unit of
a curve is the one rule `_curve_unit` states, the unit it declares and else the
unit the model gives its selection, so a series of the study data under a curve
is delivered in the unit that curve is drawn in whether the panel declares it
or not. A declared unit is checked against the unit the model gives those
values (`_checked`), the `x_unit` against the time of the model and the unit of
a curve against the unit of its selection, so a declaration which does not
convert is refused once and alike whether a run converts the values or only the
units of the panel are asked for.

A model and its simulator are loaded once per model id and SBML file and kept
warm, and every run of one of them holds its lock: roadrunner instances are
not thread safe and a service runs its sync endpoints in a thread pool.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from tokenize import TokenError
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from pint.errors import PintError
from pint.facets.plain import PlainUnit

from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.panel import AppPage, Knob, Panel, PanelCurve, PanelData
from sbmlsim.project import Project, load_project
from sbmlsim.result.scan import STATUS, TIME
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator.formula import CompiledFormula, compile_formula

if TYPE_CHECKING:
    from sbmlsim.result import ScanResult
    from sbmlsim.simulator import Simulator

logger = logging.getLogger(__name__)

__all__ = [
    "UNITLESS",
    "DeliveredUnit",
    "GridAxis",
    "ModelEntry",
    "ModelRegistry",
    "PanelResult",
    "PanelRunner",
    "PanelUnits",
    "SharedKnob",
    "UnknownKnob",
    "UnknownModel",
    "UnknownPanel",
    "UnknownVariant",
    "X",
    "check_knobs",
    "check_variant",
]

#: the key of the time of a result in `PanelResult.units`
X: Final = "x"

#: the errors the parsing of a unit raises: the ones of pint, whose
#: `UndefinedUnitError` is an `AttributeError`, and the ones of its parser, see
#: `sbmlsim.units.short_unit`
_UNIT_ERRORS: Final = (PintError, TokenError, ValueError, TypeError, AttributeError)


class UnknownModel(LookupError):
    """A model id which the registry does not serve."""


class UnknownPanel(LookupError):
    """A panel id which a served model does not declare."""


class UnknownKnob(ValueError):
    """A knob id which the panel of a run does not declare.

    It is the one error of a run which a request is answered with, so a
    service answers it and no other `ValueError` of the simulation, which
    would carry what the service is made of into a response.
    """


class UnknownVariant(ValueError):
    """A variant which the project of a model does not have.

    Like `UnknownKnob` it is an error of a request, with a message which names
    the model and its variants and nothing of the service.
    """


def check_variant(entry: ModelEntry, variant: str | None) -> None:
    """Check the variant of a request against the variants of a project.

    Args:
        entry: the entry of the model of the request.
        variant: the variant the request names, `None` for the one the panel
            declares, which needs no check.

    Raises:
        UnknownVariant: naming the model and its variants, for a variant the
            project does not have.
    """
    if variant is not None and variant not in entry.project.variants:
        raise UnknownVariant(
            f"The model '{entry.id}' has no variant '{variant}'; "
            f"its variants are {sorted(entry.project.variants)}."
        )


def check_knobs(
    panel: Panel, values: Mapping[str, float], grid: Mapping[str, int]
) -> None:
    """Check the ids of a request against the knobs a panel declares.

    It is the one check of a request which needs the panel and no model, so a
    request which names something else costs nothing: a run does it before it
    loads a model and a service does it before it answers from its cache,
    which must not turn an id the panel does not declare into an answer.

    Args:
        panel: the panel of the request.
        values: the value of a knob by its id.
        grid: the points of a knob of the grid by its id.

    Raises:
        UnknownKnob: naming the ids, if `values` or `grid` name something
            which is no knob of the panel.
    """
    knobs = sorted(knob.id for knob in panel.knobs)
    unknown = sorted((set(values) | set(grid)).difference(knobs))
    if unknown:
        raise UnknownKnob(
            f"{unknown} are no knobs of the panel '{panel.id}', which has {knobs}."
        )


@dataclass(frozen=True)
class SharedKnob:
    """A knob the shell of a page of an app owns, see `ModelEntry.shared`.

    It is a knob more than one panel of the page declares or one a scenario
    of the page sets. A shell shows one control of it for the page and sets
    every panel of `panels` from that control, so the plots of the page are of
    one state.

    Attributes:
        knob: its declaration, which every panel of `panels` declares alike.
        panels: the panels of the page which declare it, in the order the page
            shows them.
    """

    knob: Knob
    panels: tuple[Panel, ...]


@dataclass(frozen=True)
class ModelEntry:
    """A model of the registry, see `ModelRegistry`.

    Attributes:
        id: the id of the model, the one a request names.
        project: the project of the model package.
        title: the title of the model, as it is shown.
    """

    id: str
    project: Project
    title: str

    @cached_property
    def pages(self) -> dict[str, tuple[Panel, ...]]:
        """Get the panels of each page of the app of the model.

        They are the panels `ModelRegistry.panel` serves, read once, so the
        knobs a page shares (`shared`) and the heads of its panels which the
        app carries (`sbmlsim.serve.payload.AppResponse`) are of one reading
        of the SBML of the model.

        Returns:
            The panels of every page by its id, in the order the page shows
            them; a model without an app has no pages.

        Raises:
            KeyError: for a page which names a panel the model does not
                declare, which `_check_app` refuses first.
        """
        app = self.project.app
        if app is None:
            return {}
        panels = self.project.panels_of()
        return {
            page.id: tuple(panels[one] for one in page.panels) for page in app.pages
        }

    @cached_property
    def shared(self) -> dict[str, tuple[SharedKnob, ...]]:
        """Get the knobs the shell of each page of the app of the model owns.

        They are derived from the panels and are no part of the declaration of
        the app (`sbmlsim.panel.App`): the app names the panels of a page, the
        panels declare their knobs, and a knob id which more than one panel of
        a page declares is one knob of that page. A knob a scenario of the
        page sets (`sbmlsim.panel.Scenario`) is a knob of the page as well,
        even when one panel declares it: the shell sets it when a reader picks
        the scenario, so it owns its control, and a panel shows no control of
        a knob the shell sets. They are worked out once, from the panels
        `ModelRegistry.panel` serves, which a service does when its registry
        is built (`ModelRegistry.of`), so a page whose panels declare one id
        differently is refused at startup.

        Returns:
            The shared knobs of every page by its id, each in the order the
            first panel which declares one lists them (`_shared_knobs`); a page
            whose panels share none and which has no scenario has none, and a
            model without an app has no pages.

        Raises:
            ValueError: naming the model, the page, the knob id, the two panels
                and the fields, for a page two of whose panels declare one knob
                id differently, see `_shared_knobs`.
            KeyError: for a page which names a panel the model does not
                declare, which `_check_app` refuses first.
        """
        app = self.project.app
        if app is None:
            return {}
        pages = self.pages
        return {
            page.id: _shared_knobs(
                self.id,
                page.id,
                list(pages[page.id]),
                {knob for scenario in page.scenarios for knob in scenario.values},
            )
            for page in app.pages
        }


class ModelRegistry:
    """The models a request may name, see the module."""

    def __init__(self, entries: Mapping[str, ModelEntry]) -> None:
        """Create a registry of entries, see `of` for the one of a service.

        Args:
            entries: the entry of every model id, in the order they are
                served.
        """
        self._entries: dict[str, ModelEntry] = dict(entries)

    def __repr__(self) -> str:
        """Get the representation."""
        return f"ModelRegistry({list(self._entries)})"

    @classmethod
    def of(cls, specs: Mapping[str, str]) -> ModelRegistry:
        """Create the registry of the projects of their references.

        Every project is loaded once, here, so that a request reaches a model
        of the mapping and nothing else.

        Args:
            specs: the project reference `module:attribute` of every model id.

        Returns:
            The registry.

        Raises:
            ValueError: for a reference which is no project, see
                `sbmlsim.project.load_project`; naming the model, the page and
                the panels, for an app whose page names a panel the model does
                not declare, naming the model and the variants, for an app
                which names a variant the project does not have, and naming
                the model, the page, the knob id and the two panels, for a page
                whose panels declare one knob id differently, and naming the
                model, the page, the scenario, the knob and the panel, for a
                scenario of a value no panel of its page takes, see
                `_check_app`; naming the model, the variant, the panel and
                the target or the knob, for a formula of a derived target
                which is no math or reads no knob of its panel and for a knob
                without a target which no formula reads, see `_check_derived`.
            ImportError: if the module of a reference is not imported.
        """
        entries: dict[str, ModelEntry] = {}
        for model_id, reference in specs.items():
            project = load_project(reference)
            entry = ModelEntry(
                id=model_id,
                project=project,
                title=project.title or project.name,
            )
            _check_app(entry)
            _check_derived(entry)
            entries[model_id] = entry
            logger.info("The model '%s' is the project '%s'.", model_id, project.name)
        return cls(entries)

    def ids(self) -> tuple[str, ...]:
        """Get the ids of the served models, in the order they are served."""
        return tuple(self._entries)

    def entry(self, model_id: str) -> ModelEntry:
        """Get the entry of a model id.

        Args:
            model_id: the id a request names.

        Returns:
            The entry.

        Raises:
            UnknownModel: naming the id, for an id which is not served.
        """
        try:
            return self._entries[model_id]
        except KeyError:
            raise UnknownModel(f"'{model_id}' is no model of this service.") from None

    def panel(self, model_id: str, panel_id: str) -> Panel:
        """Get a panel of a served model.

        The panels are the ones of the default variant of the model, the
        annotated ones with the declaration of the project over them
        (`sbmlsim.project.Project.panels_of`), which reads the SBML once and
        keeps them.

        Args:
            model_id: the id of the model.
            panel_id: the id of the panel.

        Returns:
            The panel.

        Raises:
            UnknownModel: for a model which is not served.
            UnknownPanel: naming the id, for a panel the model does not
                declare.
        """
        panels = self.entry(model_id).project.panels_of()
        try:
            return panels[panel_id]
        except KeyError:
            raise UnknownPanel(
                f"'{panel_id}' is no panel of the model '{model_id}'."
            ) from None


def _check_app(entry: ModelEntry) -> None:
    """Check the app of a model against the panels and variants of its project.

    A page of an app is a list of panel ids and a variant of an app is a
    variant of the model (`sbmlsim.panel.App`), neither of which the
    declaration can check itself: the panels of a model are read from its SBML
    and a project is imported without it. It is checked here, where the models
    of a service are registered, so an app which names something the model does
    not declare is a problem of the startup of the service, before any request,
    and never one of a request.

    The panels are the ones of the default variant of the model, the panels
    `ModelRegistry.panel` serves; the variants are checked first, so an app of
    an unknown variant is refused without reading an SBML.

    A model which declares no app is not checked and reads nothing.

    Args:
        entry: the model of the registry.

    Raises:
        ValueError: naming the model and the variants, for an app which names
            a variant the project does not have; naming the model, the page and
            the panels, for a page which names a panel the model does not
            declare; naming the model, the page, the knob id and the two
            panels, for a page whose panels declare one knob id differently,
            see `ModelEntry.shared`; naming the model, the page, the scenario,
            the knob and the panel, for a scenario which sets a knob no panel
            of its page declares or a value which is no value of that knob,
            see `_check_scenarios`; if the project has no model variants, see
            `sbmlsim.project.Project.panels_of`.
        FileNotFoundError: if the SBML the panels are read from is no file,
            which is a problem of the service and not of a request.
    """
    app = entry.project.app
    if app is None:
        return
    variants = sorted(entry.project.variants)
    unknown = sorted({variant.id for variant in app.variants}.difference(variants))
    if unknown:
        raise ValueError(
            f"The app of the model '{entry.id}' names the variants {unknown}, "
            f"which its project does not have; its variants are {variants}."
        )
    panels = entry.project.panels_of()
    declared = sorted(panels)
    for page in app.pages:
        missing = sorted(set(page.panels).difference(declared))
        if missing:
            raise ValueError(
                f"The page '{page.id}' of the app of the model '{entry.id}' shows "
                f"the panels {missing}, which the model does not declare; its "
                f"panels are {declared}."
            )
    # the shared knobs of every page, which raise for one id declared twice,
    # so the knobs a scenario sets are of one declaration per page
    _ = entry.shared
    pages = entry.pages
    for page in app.pages:
        _check_scenarios(entry.id, page, pages[page.id])


def _check_scenarios(model_id: str, page: AppPage, panels: Sequence[Panel]) -> None:
    """Check the scenarios of a page of an app against the knobs of its panels.

    A scenario (`sbmlsim.panel.Scenario`) sets knobs of its page, which the
    declaration cannot check, since the panels are read from the SBML: a key
    which no panel of the page declares sets nothing a reader sees, and a
    value which is no value of its knob in a panel which declares it, outside
    the bounds of a range knob or none of the choices of a choice knob, is a
    state that panel cannot be in.

    Args:
        model_id: the id of the model.
        page: the page of the app.
        panels: the panels of the page, in the order it shows them.

    Raises:
        ValueError: naming the model, the page, the scenario and the knob, for
            a knob no panel of the page declares, and naming the panel as
            well, for a value which is no value of the knob in that panel.
    """
    for scenario in page.scenarios:
        for knob_id, value in scenario.values.items():
            where = (
                f"The scenario '{scenario.id}' of the page '{page.id}' of the app "
                f"of the model '{model_id}'"
            )
            declaring = [
                (panel, knob)
                for panel in panels
                for knob in panel.knobs
                if knob.id == knob_id
            ]
            if not declaring:
                raise ValueError(
                    f"{where} sets the knob '{knob_id}', which no panel of the page "
                    f"declares; the panels are {[panel.id for panel in panels]}."
                )
            for panel, knob in declaring:
                problem = _no_value_of(knob, value)
                if problem:
                    raise ValueError(
                        f"{where} sets the knob '{knob_id}' to {value}, which is "
                        f"{problem} of the knob in the panel '{panel.id}'."
                    )


def _no_value_of(knob: Knob, value: float) -> str:
    """Tell why a value is no value of a knob, see `_check_scenarios`.

    Args:
        knob: the knob.
        value: the value, in the unit of the knob.

    Returns:
        What the value is not, e.g. `outside the bounds [10.0, 150.0]`, empty
        for a value of the knob: one within the bounds of a range knob, a stop
        or not, and one of the choices of a choice knob.
    """
    if knob.kind == "choice":
        choices = [choice.value for choice in knob.choices]
        return "" if value in choices else f"none of the choices {choices}"
    low, high = knob.min, knob.max
    if low is not None and high is not None and not low <= value <= high:
        return f"outside the bounds [{low}, {high}]"
    return ""


def _check_derived(entry: ModelEntry) -> None:
    """Check the derived targets of every panel of every variant of a model.

    A formula of a derived target (`sbmlsim.panel.PanelSimulation.derived`)
    is math of PEtab, which the declaration does not parse, since
    `sbmlsim.panel` is pure data: it is compiled here, where the models of a
    service are registered, so a formula which is no math, which reads a
    symbol which is no knob of its panel, and a knob without a target which no
    formula reads, an input of nothing, are problems of the startup of the
    service and never of a request. The panels are the ones of every variant
    of the project (`sbmlsim.project.Project.panels_of`), whether an app shows
    them or not, since a request may name any variant; the SBML of each
    variant is read once.

    Args:
        entry: the model of the registry.

    Raises:
        ValueError: naming the model, the variant, the panel, the target and
            the formula, for a formula which is no math of PEtab; naming them
            and the symbols, for a formula which reads a symbol which is no
            knob of its panel; naming the model, the variant, the panel and
            the knob, for a knob without a target which no formula reads.
        FileNotFoundError: if the SBML of a variant is no file.
    """
    for variant in entry.project.variants:
        for panel in entry.project.panels_of(variant).values():
            where = f" of the model '{entry.id}' in its variant '{variant}'"
            formulas = _formulas(panel, where)
            read = {s for formula in formulas.values() for s in formula.symbols}
            for knob in panel.knobs:
                if knob.target is None and knob.id not in read:
                    raise ValueError(
                        f"The knob '{knob.id}' of the panel '{panel.id}'{where} sets "
                        f"no target and no formula of its derived targets reads it; "
                        f"a knob without a target is an input of a formula."
                    )


def _shared_knobs(
    model_id: str,
    page_id: str,
    panels: Sequence[Panel],
    named: Set[str] = frozenset(),
) -> tuple[SharedKnob, ...]:
    """Get the knobs the panels of a page share, and check they declare them alike.

    An app shows one control for a knob id which more than one panel of a page
    declares and sets every one of those panels from it, so the declarations of
    that id must be one declaration: two panels which gave it other bounds, an
    other default or an other target would be set from a control which is
    right for one of them only. The declarations are compared field for field,
    and the fields which differ are named. The same id on different pages is
    no such knob, since a page shows its own controls.

    A knob a scenario of the page sets (`named`) is a knob of the page whether
    one panel declares it or more, since the shell sets it, see
    `ModelEntry.shared`.

    The shared knobs are listed in the order the first panel which declares one
    lists them: the panels are walked in the order of the page and the knobs of
    each in the order of its declaration.

    Args:
        model_id: the id of the model.
        page_id: the id of the page.
        panels: the panels of the page, in the order it shows them, each once.
        named: the ids of the knobs the scenarios of the page set.

    Returns:
        The knobs more than one of the panels declares and the knobs of
        `named` which a panel declares, with the panels which declare each.

    Raises:
        ValueError: naming the model, the page, the knob id, the two panels
            and the fields, for a knob id which two panels of the page declare
            differently.
    """
    first: dict[str, tuple[Panel, Knob]] = {}
    declaring: dict[str, list[Panel]] = {}
    for panel in panels:
        for knob in panel.knobs:
            seen = first.setdefault(knob.id, (panel, knob))
            declaring.setdefault(knob.id, []).append(panel)
            other_panel, other = seen
            fields = [
                name
                for name in type(knob).model_fields
                if getattr(knob, name) != getattr(other, name)
            ]
            if fields:
                raise ValueError(
                    f"The page '{page_id}' of the app of the model '{model_id}' "
                    f"shows the panels '{other_panel.id}' and '{panel.id}', which "
                    f"declare the knob '{knob.id}' differently ({', '.join(fields)}); "
                    "the panels of a page share a knob they both declare, so they "
                    "must declare it alike."
                )
    return tuple(
        SharedKnob(knob=knob, panels=tuple(declaring[knob_id]))
        for knob_id, (_, knob) in first.items()
        if len(declaring[knob_id]) > 1 or knob_id in named
    )


@dataclass(frozen=True)
class GridAxis:
    """An axis of the grid a run of a panel covered, one of its knobs.

    Attributes:
        id: the id of the knob of the axis.
        label: the knob as it is shown.
        unit: the unit of the values, the one the knob declares or else the
            unit of its target in the model.
        values: the values of the knob the run covered, in their order.
    """

    id: str
    label: str
    unit: str | None
    values: tuple[float, ...]


@dataclass(frozen=True)
class DeliveredUnit:
    """A unit a panel is delivered in: the text it reports and that unit.

    A panel declares a unit as text, e.g. `uM`, which may be a unit id of its
    model, e.g. `mmole_per_l`: the text is resolved in the registry of the
    model (`sbmlsim.model.model_sbml.SBMLModel.Q`), which is the only registry
    which knows these ids, and the values are converted with the resolved unit
    while a result reports the declared text, see the module.

    Attributes:
        text: the unit as the panel declares it or as the model gives it,
            which a result reports, e.g. `PanelResult.units` and
            `sbmlsim.serve.data.PanelSeries.unit`; empty for values which
            carry no unit, which both of them report as `None`.
        unit: that unit, the one the values are converted to, `None` for
            values which carry none (`UNITLESS`), which are delivered as they
            are and whose dataset must carry none either.
    """

    text: str
    unit: PlainUnit | None


#: the unit of values which carry none: a curve whose selection the model has
#: no unit for and a model whose time has no unit. They are delivered as they
#: are and the dataset of a series under them must carry no unit either, see
#: `sbmlsim.serve.data`.
UNITLESS: Final = DeliveredUnit(text="", unit=None)


@dataclass(frozen=True)
class PanelUnits:
    """The units the x and the study data of a panel are delivered in.

    Attributes:
        x: the unit of the x of a run of the panel and of the times of its
            study data, `UNITLESS` for a model whose time has no unit and a
            panel which declares none.
        series: the unit of every series of the study data of the panel, by
            the id of the series: the unit of the curve it belongs under
            (`_curve_unit`), `UNITLESS` for a curve the model gives no unit
            and `None` for a series whose own `id` is no curve of the panel,
            which is delivered in the unit of its own dataset.
    """

    x: DeliveredUnit
    series: Mapping[str, DeliveredUnit | None]


@dataclass(frozen=True)
class PanelResult:
    """The result of a run of a panel, see the module.

    Attributes:
        x: the output times, in the unit `units[X]` names: the `x_unit` the
            panel declares, else the time unit of the model.
        dims: the axes of the grid, empty for a run of one point.
        curves: the values of every declared curve, a row per point of the
            grid in C order.
        units: the unit of every curve and, under `X`, of the x, `None` for
            values which carry none, as `sbmlsim.serve.data.PanelSeries.unit`
            and `x_unit` of the study data of the same panel are, so the two
            sides of an axis state one unit.
        failed: the flat indices in C order of the points which did not
            integrate, whose rows are `NaN`.
    """

    x: tuple[float, ...]
    dims: tuple[GridAxis, ...]
    curves: Mapping[str, list[list[float]]]
    units: Mapping[str, str | None]
    failed: tuple[int, ...]


#: the key of a warm model: the id of its entry and the SBML file it is loaded
#: from, so that the default variant and its name are one model
type _WarmKey = tuple[str, Path]


@dataclass(frozen=True)
class _Warm:
    """A loaded model with its simulator and the lock of its runs.

    Attributes:
        model: the model, in the engine and at the optimization level of the
            project.
        simulator: the serial simulator of the settings of the project.
        lock: the lock a run of the model holds, see the module.
    """

    model: SBMLModel
    simulator: Simulator
    lock: threading.Lock = field(default_factory=threading.Lock)


class PanelRunner:
    """Runs the panels of the models of a registry, see the module.

    Attributes:
        registry: the models a request may name.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        """Create a runner of the models of a registry, which loads none yet.

        Args:
            registry: the models a request may name.
        """
        self.registry = registry
        #: the warm model of every model id and SBML file, see `_warm_model`
        self._warm: dict[_WarmKey, _Warm] = {}
        #: the lock of the load of one such key, see `_warm_model`
        self._loads: dict[_WarmKey, threading.Lock] = {}

    def __repr__(self) -> str:
        """Get the representation."""
        return f"PanelRunner({self.registry!r})"

    def run(
        self,
        model_id: str,
        panel: Panel,
        values: Mapping[str, float],
        grid: Mapping[str, int],
        variant: str | None = None,
    ) -> PanelResult:
        """Run a panel of a model at a point of its knobs or over a grid.

        A knob of `grid` is an axis of the result, of the points of
        `Knob.values(n)`; every other knob is set to its value in `values` or
        else to its `default`, over the pre-initialization the panel declares.
        The axes are in the order of the knobs of the panel, so the result does
        not depend on the order of a request.

        Args:
            model_id: the id of the model, see `ModelRegistry`.
            panel: the panel, e.g. `ModelRegistry.panel(model_id, panel_id)`.
            values: the value of a knob by its id, in the unit of the knob.
            grid: the points of a knob of the grid by its id, see
                `Knob.values`.
            variant: the variant of the model to simulate, the one the panel
                declares without.

        Returns:
            The result, see `PanelResult`; a point which did not integrate is
            `NaN` and its flat index is in `failed`.

        Raises:
            UnknownModel: for a model which is not served.
            UnknownKnob: naming the ids, if `values` or `grid` name something
                which is no knob of the panel; it is raised before a model is
                loaded, so a request which names one costs nothing.
            UnknownVariant: for a variant the project does not have, also
                before a model is loaded.
            ValueError: for a grid of fewer than one point, see `Knob.values`;
                naming both units, for an `x_unit` of the panel which is no
                time unit of the model, see `_times`, and for a unit of a
                curve which is no unit of its values, see `_curve_unit`;
                naming the panel, for limits of its axes which are no limits
                of the units it is delivered in, see `_check_limits`. The
                limits of a request (`sbmlsim.panel.MAX_KNOBS`, `MAX_GRID`)
                are checked where a request is read.
        """
        entry = self.registry.entry(model_id)
        check_knobs(panel, values, grid)
        check_variant(entry, variant)
        warm = self._warm_model(
            entry, panel.simulation.variant if variant is None else variant
        )
        times = np.linspace(0.0, panel.simulation.end, panel.simulation.points + 1)
        _check_limits(warm.model, panel)
        with warm.lock:
            scan, axes = _scan(warm.model, panel, values, grid)
            result = warm.simulator.run(
                warm.model,
                scan,
                time=times,
                keep=[curve.id for curve in panel.curves],
                on_error="flag",
                progress=False,
            )
            return _result(warm.model, panel, axes, result)

    def units(self, entry: ModelEntry, panel: Panel) -> PanelUnits:
        """Get the units a panel is delivered in, see `PanelUnits`.

        The x is the one `PanelResult.units[X]` of a run of the panel names,
        and every unit is resolved in the registry of the warm model of the
        entry, the model a run of the panel integrates on, so the study data
        of a panel (`sbmlsim.serve.data.panel_data`) is delivered on the axis
        of its curves and in their units whatever the declaration names.

        Args:
            entry: the entry of the model, the one the caller holds.
            panel: the panel.

        Returns:
            The units, see `PanelUnits`.

        Raises:
            ValueError: naming both units, for an `x_unit` which is no time
                unit of the model, see `_x_unit`, or for a unit of a curve
                which is no unit of its values, see `_curve_unit`; naming the
                series and its unit, for a unit of a series which is no unit
                of pint and none of the model; naming the panel, for y limits
                of curves which are delivered in different units and for an
                `x_max` past the end of its simulation, see `_check_limits`;
                if the project of the entry has no model variants.
            KeyError: for a variant the project does not have.
        """
        warm = self._warm_model(entry, panel.simulation.variant)
        return _panel_units(warm.model, panel)

    def _warm_model(self, entry: ModelEntry, variant: str | None) -> _Warm:
        """Get the warm model of a variant of an entry, loaded on its first use.

        A model which is warm is answered without any lock, and a model which
        is not is loaded under the lock of its key alone, so a request for one
        model never waits for the compilation of another; the key is read
        again under that lock, so two threads which ask for one cold model at
        once compile it once and the second one waits for the first. The two
        dictionaries are read and written with the atomic operations of a
        `dict` (`get`, `setdefault` and an assignment of a key), and only a
        model which is ready is published.

        Args:
            entry: the entry of the model.
            variant: the variant of the model, its default variant without.

        Returns:
            The warm model.

        Raises:
            ValueError: if the project of the entry has no model variants.
            KeyError: for a variant the project does not have.
        """
        path = entry.project.variant_path(variant).resolve()
        key: _WarmKey = (entry.id, path)
        warm = self._warm.get(key)
        if warm is not None:
            return warm
        with self._loads.setdefault(key, threading.Lock()):
            warm = self._warm.get(key)
            if warm is None:
                warm = _load(entry, path)
                self._warm[key] = warm
                logger.info(
                    "The model '%s' is loaded from '%s' and stays warm.",
                    entry.id,
                    path,
                )
            return warm


def _load(entry: ModelEntry, path: Path) -> _Warm:
    """Load the model of an entry from an SBML file with its simulator.

    The model is loaded in the engine and at the optimization level of the
    project and the simulator is the serial one of its settings, so a project
    which pins an engine or the tolerances of its integrator is honoured.

    Args:
        entry: the entry of the model.
        path: the SBML file of the variant, see `PanelRunner._warm_model`.

    Returns:
        The warm model.
    """
    settings = entry.project.simulator
    simulator = settings.simulator(n_workers=1)
    model = simulator.load(
        load_model(
            path,
            settings.engine,
            jit_optimization=settings.jit_optimization,
        )
    )
    return _Warm(model=model, simulator=simulator)


def _scan(
    model: SBMLModel,
    panel: Panel,
    values: Mapping[str, float],
    grid: Mapping[str, int],
) -> tuple[Scan, tuple[GridAxis, ...]]:
    """Get the scan of a request and the axes of its grid.

    The knobs of the grid span one `Dimension` of the points of their product
    in C order, in the order of the knobs of the panel: the target of every
    such knob and every derived target whose formula reads one of them
    (`sbmlsim.panel.PanelSimulation.derived`) are values of that dimension,
    coupled, so a formula which reads two knobs of the grid has its value at
    every point of the product. A run of the product in one dimension has the
    rows of the product of one dimension per knob, in the same order, which is
    the order of the rows of `PanelResult`. Every other knob is a value of the
    pre-initialization, and so is every derived target whose formula reads no
    knob of the grid.

    A formula is evaluated with numpy over the points of the grid at once, on
    the numbers of the knobs in their declared units, and its value is the
    value of its target in the unit of the model, see `PanelSimulation`.

    Args:
        model: the model, which resolves the units of the knobs.
        panel: the panel.
        values: the value of a knob by its id.
        grid: the points of a knob of the grid by its id.

    Returns:
        The scan, of one dimension for a request of a grid and none for one of
        a point, and the axes of the grid, one per knob of the grid in the
        order of the knobs of the panel.

    Raises:
        ValueError: if a formula of a derived target is no math of PEtab or
            reads no knob of the panel, and naming the knob, for a knob of the
            grid which sets nothing, which the registry of a service refuses
            when it starts (`_check_derived`).
    """
    preinit: dict[str, Any] = dict(panel.simulation.preinit)
    gridded = [knob for knob in panel.knobs if knob.id in grid]
    axes = tuple(
        GridAxis(
            id=knob.id,
            label=knob.label,
            unit=_unit_of(model, knob),
            values=knob.values(grid[knob.id]),
        )
        for knob in gridded
    )
    # the numbers of every knob of the grid at the points of the product in C
    # order, and the number of every other knob
    mesh = np.meshgrid(*(np.asarray(axis.values) for axis in axes), indexing="ij")
    points = {axis.id: one.ravel() for axis, one in zip(axes, mesh, strict=True)}
    size = math.prod(len(axis.values) for axis in axes)
    numbers = {
        knob.id: float(values.get(knob.id, knob.default))
        for knob in panel.knobs
        if knob.id not in grid
    }
    changed: dict[str, Any] = {}
    for knob in panel.knobs:
        if knob.target is None:
            continue
        if knob.id in points:
            changed[knob.target] = _in_model_units(model, knob, points[knob.id])
        else:
            preinit[knob.target] = _in_model_units(model, knob, numbers[knob.id])
    for target, formula in _formulas(panel).items():
        if points.keys().isdisjoint(formula.symbols):
            preinit[target] = formula.evaluate([numbers[s] for s in formula.symbols])
            continue
        arguments = [
            points[s] if s in points else np.full(size, numbers[s])
            for s in formula.symbols
        ]
        changed[target] = formula.evaluate_array(arguments, size)
    if gridded and not changed:
        raise ValueError(
            f"The knobs {[knob.id for knob in gridded]} of the grid of the panel "
            f"'{panel.id}' set no target and no derived target reads them."
        )
    dimensions = (
        [Dimension("__".join(axis.id for axis in axes), values=changed)]
        if gridded
        else []
    )
    simulation = Simulation(end=panel.simulation.end, preinit_changes=preinit)
    return Scan(simulation, dimensions), axes


def _formulas(panel: Panel, where: str = "") -> dict[str, CompiledFormula]:
    """Compile the formulas of the derived targets of a panel.

    Args:
        panel: the panel.
        where: the model and the variant of the panel, which a problem names
            after the panel, e.g. ` of the model 'hctz' in its variant
            'default'`; empty for a run, which names the panel alone.

    Returns:
        The compiled formula of every derived target, by the target, in the
        order the panel declares them.

    Raises:
        ValueError: naming the panel, the target and the formula, for a
            formula which is no math of PEtab; naming the panel, the target
            and the symbols, for a formula which reads a symbol which is no
            knob of the panel.
    """
    knobs = sorted(knob.id for knob in panel.knobs)
    formulas: dict[str, CompiledFormula] = {}
    for target, formula in panel.simulation.derived.items():
        try:
            compiled = compile_formula(formula)
        except ValueError as error:
            raise ValueError(
                f"The formula '{formula}' of the derived target '{target}' of the "
                f"panel '{panel.id}'{where} is no math of PEtab: {error}"
            ) from error
        unknown = sorted(set(compiled.symbols).difference(knobs))
        if unknown:
            raise ValueError(
                f"The formula '{formula}' of the derived target '{target}' of the "
                f"panel '{panel.id}'{where} reads {unknown}, which are no knobs of "
                f"the panel; a formula reads the knobs of its panel, which are "
                f"{knobs}."
            )
        formulas[target] = compiled
    return formulas


def _panel_units(model: SBMLModel, panel: Panel) -> PanelUnits:
    """Resolve the units a panel declares in the registry of its model.

    Args:
        model: the model of the panel.
        panel: the panel.

    Returns:
        The units, see `PanelUnits`.

    Raises:
        ValueError: naming both units, for an `x_unit` which is no time unit
            of the model, see `_x_unit`, or for a unit of a curve which is no
            unit of its values, see `_curve_unit`; naming the series and its
            unit, for a unit of a series which is no unit, see `_series_unit`;
            naming the panel, for limits of its axes which are no limits of
            the units it is delivered in, see `_check_limits`.
    """
    _check_limits(model, panel)
    return PanelUnits(
        x=_x_unit(model, panel),
        series={data.id: _series_unit(model, panel, data) for data in panel.data},
    )


def _check_limits(model: SBMLModel, panel: Panel) -> None:
    """Check the limits of the axes of a panel against the units it is delivered in.

    A limit of the value axis (`Panel.y_min`, `Panel.y_max`) is a number in
    the unit the curves of the panel are delivered in (`_curve_unit`), which
    is therefore one unit: curves of different units share no axis a number
    could be a limit of. The `x_max` is a number in the unit the x is
    delivered in (`_x_unit`) and lies within the simulation, whose `end` is in
    the time unit of the model. Both are checked where the units of a panel
    are resolved, for a run of the panel and for its study data alike, see
    `_panel_units`.

    Args:
        model: the model of the panel, whose unit ids the units may name.
        panel: the panel.

    Raises:
        ValueError: naming the panel and the curves with their units, for y
            limits of curves which are delivered in different units; naming
            the panel, the `x_max`, the end and its unit, for an `x_max` past
            the end of the simulation; naming both units, for a unit of a
            curve or an `x_unit` which is no unit of its values, see
            `_curve_unit` and `_x_unit`.
    """
    if panel.y_min is not None or panel.y_max is not None:
        delivered = [(curve.id, _curve_unit(model, curve)) for curve in panel.curves]
        first = delivered[0][1]
        if not all(_same_unit(first, unit) for _, unit in delivered):
            named = ", ".join(
                f"the curve '{curve_id}' in "
                + (f"'{unit.text}'" if unit.text else "no unit")
                for curve_id, unit in delivered
            )
            raise ValueError(
                f"The limits of the value axis of the panel '{panel.id}' are in "
                f"the unit its curves are delivered in, which they do not share: "
                f"{named}."
            )
    if panel.x_max is not None:
        x_unit = _x_unit(model, panel)
        end = panel.simulation.end
        if x_unit.unit is not None:
            end = float(
                model.Q(end, _time_unit(model) or "dimensionless").to(x_unit.unit).m
            )
        if panel.x_max > end and not math.isclose(panel.x_max, end, rel_tol=1e-12):
            unit = f" '{x_unit.text}'" if x_unit.text else " no unit"
            raise ValueError(
                f"The x_max={panel.x_max} of the panel '{panel.id}' lies beyond the "
                f"end of its simulation, which is {end} in{unit}, the unit of its x."
            )


def _same_unit(one: DeliveredUnit, other: DeliveredUnit) -> bool:
    """Tell whether two delivered units are one unit, whatever their texts.

    A unit is compared by what it is and not by how it is written: `uM` and
    `umol/l` are one unit, which pint does not tell by the equality of its
    units, so a value of one is converted into the other.

    Args:
        one: a unit.
        other: another unit.

    Returns:
        Whether a value in `one` is the same value in `other`; values without
        a unit (`UNITLESS`) are one unit with each other and with no other.
    """
    if one.unit is None or other.unit is None:
        return one.unit is None and other.unit is None
    try:
        factor = float((1.0 * one.unit).to(other.unit).m)
    except _UNIT_ERRORS:
        return False
    return math.isclose(factor, 1.0, rel_tol=1e-12)


def _resolved(model: SBMLModel, text: str, what: str) -> DeliveredUnit:
    """Resolve a unit of a panel or of its model, see `DeliveredUnit`.

    It is the one place a unit of a panel is read: in the registry of the
    model, whose unit ids a declaration may name (`SBMLModel.Q`) and whose own
    units are expressions of pint, so the x and the curves of a run and the
    study data under them are converted with one unit and report one text.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        text: the unit, empty for values which carry none.
        what: the declaration the unit belongs to, which a problem names, e.g.
            `the curve 'Cve_hctz'`.

    Returns:
        The text and that unit, `UNITLESS` for an empty text.

    Raises:
        ValueError: naming the unit and `what`, for a text which is no unit of
            pint and no unit of the model.
    """
    if not text:
        return UNITLESS
    try:
        return DeliveredUnit(text=text, unit=model.Q(1.0, text).units)
    except _UNIT_ERRORS as err:
        raise ValueError(
            f"The unit '{text}' of {what} is no unit of pint and no unit of the "
            f"model: {err}"
        ) from err


def _checked(
    model: SBMLModel, declared: str, model_text: str, what: str
) -> DeliveredUnit:
    """Resolve a declared unit of a panel and check it against its values.

    It is the one check of a declaration: of the `x_unit` against the time
    unit of the model (`_x_unit`) and of the unit of a curve against the unit
    the model gives its selection (`_curve_unit`). A declaration which does
    not convert is therefore refused where the units of a panel are resolved,
    the same way for a run of the panel, which converts the values, and for
    its study data, which only asks for the units.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        declared: the declared unit, which is not empty.
        model_text: the unit the model gives the values, empty for values
            which carry none.
        what: the declaration the unit belongs to, which a problem names, e.g.
            `the curve 'Cve_hctz'`.

    Returns:
        The declared unit, see `DeliveredUnit`.

    Raises:
        ValueError: naming the declared unit and the unit of the model, for a
            declaration which is no unit at all and for one the values of the
            model do not convert to.
    """
    try:
        delivered = _resolved(model, declared, what)
        model.Q(1.0, model_text or "dimensionless").to(delivered.unit)
    except _UNIT_ERRORS as err:
        gives = f"in '{model_text}'" if model_text else "no unit"
        raise ValueError(
            f"The unit '{declared}' of {what} is no unit of its values, which "
            f"the model gives {gives}."
        ) from err
    return delivered


def _x_unit(model: SBMLModel, panel: Panel) -> DeliveredUnit:
    """Get the unit the x of a panel is delivered in, see `PanelUnits`.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        panel: the panel.

    Returns:
        The `x_unit` the panel declares, else the time unit of the model,
        `UNITLESS` for a model whose time has no unit and a panel which
        declares none, whose times are delivered as they are.

    Raises:
        ValueError: naming the declared unit and the time unit of the model,
            for an `x_unit` which is no unit at all or no time of the model.
    """
    model_unit = _time_unit(model)
    declared = panel.simulation.x_unit
    if not declared:
        return _resolved(model, model_unit, "the time of the model")
    return _checked(model, declared, model_unit, "the x of the panel")


def _curve_unit(model: SBMLModel, curve: PanelCurve) -> DeliveredUnit:
    """Get the unit a curve of a panel is delivered in, see `PanelUnits`.

    It is the unit the curve declares, else the unit the model gives its
    selection: the one rule of the curves of a run (`_result`) and of the
    study data under them (`sbmlsim.serve.data`), so the points of a series
    are drawn in the unit of its curve.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        curve: the curve, one of `Panel.curves`.

    A declared unit is checked against the unit the model gives the
    selection (`_checked`), so a curve which declares a unit its values do not
    convert to is refused where the units are resolved and not only where a
    run converts them.

    Returns:
        The unit, `UNITLESS` for a curve which declares none and whose
        selection the model gives none either.

    Raises:
        ValueError: naming both units, for a declared unit which is no unit at
            all or which the values of the model do not convert to.
    """
    model_text = model.uinfo.get(curve.id, "") or ""
    what = f"the curve '{curve.id}'"
    if not curve.unit:
        return _resolved(model, model_text, what)
    return _checked(model, curve.unit, model_text, what)


def _series_unit(
    model: SBMLModel, panel: Panel, data: PanelData
) -> DeliveredUnit | None:
    """Get the unit a series of the study data of a panel is delivered in.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        panel: the panel of the series.
        data: the declaration of the series, one of `Panel.data`.

    The unit a series declares itself is delivered as it is, it is not
    checked against the unit of its curve (`sbmlsim.panel.PanelData.unit`),
    while the unit of the curve is the checked one of `_curve_unit`, so one
    rule holds for a curve and the series under it.

    Returns:
        The unit the series declares itself, else the unit of the curve it
        belongs under (`_curve_unit`, the unit of the model for a curve which
        declares none), `None` for a series whose own `id` is no curve of the
        panel, which is delivered in the unit of its own dataset.

    Raises:
        ValueError: naming the series and the unit, for a unit of the series
            which is no unit of pint and none of the model; naming both units,
            for a unit of its curve which is no unit of its values.
    """
    if data.unit:
        return _resolved(model, data.unit, f"the series '{data.id}'")
    curve = panel.curve_of(data)
    return None if curve is None else _curve_unit(model, curve)


def _time_unit(model: SBMLModel) -> str:
    """Get the time unit of a model, the unit the time of a result is in.

    It is the unit the scan core puts into `ScanResult.units[TIME]`, read
    from the same place, so the x of a result and the x of its study data are
    the same unit, see `PanelRunner.units`.

    Args:
        model: the model.

    Returns:
        The unit, empty for a model whose time has no unit. An SBML without
        time units has the `second` the units of a model fall back to
        (`sbmlsim.units.UnitsInformation.from_sbml`), so the empty text is the
        time of a `UnitsInformation` which carries none.
    """
    return model.uinfo.get(TIME, "") or ""


def _unit_of(model: SBMLModel, knob: Knob) -> str | None:
    """Get the unit of the values of a knob, the one of its target without one.

    Args:
        model: the model of the knob.
        knob: the knob.

    Returns:
        The unit the knob declares, else the unit of its target in the model,
        `None` for a target which has none and for a knob without a target
        which declares none.
    """
    if knob.unit is not None or knob.target is None:
        return knob.unit
    unit: str | None = model.uinfo.get(knob.target)
    return unit


def _in_model_units(model: SBMLModel, knob: Knob, value: Any) -> Any:
    """Get a value or an array of values of a knob in the units of the model.

    A knob which declares its unit gives a quantity, which the simulation
    converts into the unit of the target; a knob without one gives the value
    as it is, i.e. in the unit of its target.

    Args:
        model: the model of the knob, whose unit ids the unit may name.
        knob: the knob.
        value: the value or the array of the values.

    Returns:
        The quantity or the value.
    """
    if knob.unit is None:
        return value
    return model.Q(value, knob.unit)


def _times(
    model: SBMLModel, panel: Panel, result: ScanResult
) -> tuple[tuple[float, ...], str | None]:
    """Get the output times of a result in the x unit of the panel, and that unit.

    The `end` and the `points` of a panel are in the time unit of the model,
    the `x_unit` it declares is the unit the times are delivered in, e.g. a
    day of a model of minutes as `end=1440` delivered in hours.

    The unit is the one `_x_unit` resolves, the one the study data of the
    panel is delivered in as well, so the two cannot disagree.

    Args:
        model: the model of the panel, whose unit ids the unit may name.
        panel: the panel.
        result: the result of the scan of the run, on a grid of times.

    Returns:
        The times and their unit, which is `None` for a model whose time has
        no unit and a panel which declares none.

    Raises:
        ValueError: naming the declared unit and the time unit of the model,
            for an `x_unit` which is no unit of pint or of the model, or no
            time of it, see `_x_unit`.
    """
    times = result.quantity(TIME)
    delivered = _x_unit(model, panel)
    if delivered.unit is not None:
        times = times.to(delivered.unit)
    return tuple(float(time) for time in times.magnitude), delivered.text or None


def _result(
    model: SBMLModel,
    panel: Panel,
    axes: tuple[GridAxis, ...],
    result: ScanResult,
) -> PanelResult:
    """Read the declared curves of a panel out of the result of its scan.

    Args:
        model: the model of the panel, whose unit ids a curve may name.
        panel: the panel.
        axes: the axes of the grid of the run.
        result: the result of the scan of the run, on a grid of times.

    Returns:
        The result of the panel.

    Raises:
        KeyError: for a curve which is no variable of the result.
        ValueError: naming both units, for an `x_unit` which is no time unit
            of the model, see `_times`, and for a unit of a curve which is no
            unit of its values, see `_curve_unit`.
    """
    x, x_unit = _times(model, panel, result)
    curves: dict[str, list[list[float]]] = {}
    units: dict[str, str | None] = {X: x_unit}
    for curve in panel.curves:
        quantity = result.quantity(curve.id)
        delivered = _curve_unit(model, curve)
        if delivered.unit is not None:
            quantity = quantity.to(delivered.unit)
        values = np.asarray(quantity.magnitude, dtype=float).reshape(-1, len(x))
        curves[curve.id] = values.tolist()
        units[curve.id] = delivered.text or None
    status = result.ds.get(STATUS)
    failed: tuple[int, ...] = ()
    if status is not None:
        flags = np.asarray(status.values).reshape(-1)
        failed = tuple(int(index) for index in np.flatnonzero(flags))
    return PanelResult(x=x, dims=axes, curves=curves, units=units, failed=failed)
