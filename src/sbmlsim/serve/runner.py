"""The models a request may name and the runner of their panels.

`ModelRegistry` is the models of the service. It is built once at startup from
a mapping of model id to project reference (`ModelRegistry.of`) and is the only
way from a request to a model: `entry` answers an id of that mapping and raises
`UnknownModel` for everything else, so a request can never name a module, a
project reference or a path. A model whose project declares an app
(`sbmlsim.panel.App`) is checked there as well (`_check_app`): a page which
names a panel the model does not declare, a variant the project does not have,
two panels of a page which declare one knob id differently
(`ModelEntry.shared`) and a scenario of a page which sets a knob its panels do
not declare or a value which is none of that knob (`_check_scenarios`) are
refused at startup, since the panels of a model are read from its SBML, which a
project is imported without. The declaration of a panel in a variant is the one
of the SBML of that variant (`sbmlsim.project.Project.panels_of`), so the app
is checked in every variant it is served in (`ModelEntry.app_variants`), and
the formulas of the derived targets of every panel of every variant are checked
as well (`_check_derived`), which are math of PEtab the declaration does not
parse.

`PanelRunner` runs a panel of such a model: the values of its knobs become the
pre-initialization of one `Simulation`, the knobs of the grid become a
`Dimension` of the product of the values of `Knob.values(n)`, a derived target
is the value of its formula over the knobs, a value of that dimension when it
reads a knob of the grid and of the pre-initialization otherwise, and the scan
of the two runs on the warm model of the entry. The curves of the `PanelResult`
are the declared curves of the panel, one row per point of the grid in C order
and one row for a request without a grid, in the unit the curve declares or
else in the unit of the model; the x of the result is the grid of `points + 1`
times from 0 to the `end` of the panel, which are in the time unit of the
model, delivered in the `x_unit` the panel declares or else in that time unit,
which `units["x"]` names and which `units(entry, panel)` answers without a run,
so the study data of a panel (`sbmlsim.serve.data`) is drawn on the axis of its
curves. A point which does not integrate is `NaN` and its flat index is in
`failed`, never an exception: a knob value a reader turns to may be a value the
model does not simulate.

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

A request which waits for the lock of a model, or for its load, holds a thread
of that pool, so the requests of one model could take every thread of the
server and stall the other models and every other sync route. The requests of
one model id which are in the runner at once are therefore bounded
(`MAX_WAITING`, over its variants): one integrates and the others wait, and a
request beyond the bound is refused at once with `ModelBusy`, before it loads
or waits for anything, which a service answers with a 503 to retry. A model
thus holds at most `MAX_WAITING` threads of the server whatever its callers
send, and the threads of the other models and routes stay free.

The panels of a model which share a simulation (`sbmlsim.panel.PanelSimulation`,
the same variant, `end`, `points`, `x_unit`, `preinit` and `derived`) share a
run as well: a page of several panels of one model asks for every panel at the
same state of its knobs at once, and the scan of that state is run once with
the selections of every panel of the variant which shares the simulation
(`_selections`) and kept (`MAX_RUNS`), so the requests of the other panels at
that state read their curves out of it without integrating. A stored run is
keyed on what decides its numbers (`_run_key`): the warm model, the sha256 of
its SBML, the engine and the settings of its simulator, the times and the
resolved scan, i.e. the values of the pre-initialization and of the grid by
their targets, which `_scan` gives the simulator in the canonical order of the
targets, so two panels which declare one state with their knobs in another
order share it, never on the panel. The key is read under the lock of the warm
model, after it was taken, so a request whose state is running waits for that
run and reads it, and every read and write of a stored run is under the lock of
the model which ran it. A run is kept by its last use (`BoundedCache`), and a
panel whose curves a stored run of its state lacks runs it again with the
curves of both, so two groups of panels of one state do not evict each other.

A run is shared only where sharing is bitwise, which is the engine roadrunner
(`SHARED_ENGINES`): its integrator takes its steps whatever a scan selects, so
a curve does not change with the selections a scan keeps and an answer read
out of a shared run is the answer of a run of the panel alone, bitwise. AMICI
outputs a substitute of the steps of its integrator which it refines where the
interpolation between two rows misses a selection
(`sbmlsim.simulator.executor_amici`), so a curve of another panel refines the
rows of this one, and a result of COPASI is not bitwise reproducible across
two runs at all; a panel of either engine is therefore run alone, with its own
curves, as a runner which shares nothing (`max_runs=0`) runs every panel.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Generator, Iterable, Mapping, Sequence, Set
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tokenize import TokenError
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from pint.errors import PintError
from pint.facets.plain import PlainUnit

from sbmlsim.engine import Engine
from sbmlsim.hashing import canonical_json_sha256, file_sha256
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.panel import AppPage, Knob, Panel, PanelCurve, PanelData
from sbmlsim.project import Project, load_project
from sbmlsim.result.scan import STATUS, TIME
from sbmlsim.serve.cache import BoundedCache
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator.formula import CompiledFormula, compile_formula
from sbmlsim.units import Quantity

if TYPE_CHECKING:
    from sbmlsim.result import ScanResult
    from sbmlsim.simulator import Simulator

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_RUNS",
    "MAX_WAITING",
    "SHARED_ENGINES",
    "UNITLESS",
    "DeliveredUnit",
    "GridAxis",
    "KnobValueError",
    "ModelBusy",
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
    "check_app_variant",
    "check_knobs",
    "check_values",
    "check_variant",
]

#: the key of the time of a result in `PanelResult.units`
X: Final = "x"

#: the runs a runner keeps for the panels which share them, see the module: the
#: panels of a page ask for one state at once, so a run is read within a second
#: of its run and a few entries cover the pages several readers turn at once,
#: while every answer is kept by the cache of the responses
#: (`sbmlsim.serve.cache.MAX_ENTRIES`) anyway. It bounds the number of runs and
#: not their bytes: a run holds every curve of the panels which share its
#: simulation, each at most `MAX_GRID` points of `MAX_POINTS + 1` times of 8
#: bytes, 1.0 MB, so the runs of a runner hold at most `MAX_RUNS` times the
#: curves of the largest such group of a served model in MB, each group of
#: panels at most `MAX_CURVES` curves per panel. The four curves of the page of
#: glimepiride of pkdb_models at the limits, a grid of 64 doses of its 501
#: times, are 1.03 MB of a run
MAX_RUNS: Final = 32

#: the requests of one model id which may be in the runner at once, the one
#: which integrates and the ones which wait for its lock or its load, see the
#: module: a page asks for its panels at once, so a few requests of one model
#: are its normal load, while every request beyond the bound would hold a
#: thread of the server for a run of seconds and is refused instead. The thread
#: pool of the server (anyio's, 40 threads) thus keeps threads for the other
#: models and routes as long as fewer than 40 / `MAX_WAITING` models are
#: saturated at once
MAX_WAITING: Final = 4

#: the engines whose runs the panels of one simulation share, see the module:
#: the ones whose curves do not change with the selections of a scan
SHARED_ENGINES: Final = frozenset({Engine.ROADRUNNER})

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


class KnobValueError(UnknownKnob):
    """A value of a knob, or the points of a grid over it, which its declaration does not take.

    A request is answered with it as with `UnknownKnob`, a 422 whose message
    names the knob, the value and what the knob takes, and nothing of the
    service: a value outside the bounds of a range knob, a value which is none
    of the choices of a choice knob, or a grid over a choice knob of other
    points than its choices.
    """


class ModelBusy(RuntimeError):
    """A request of a model which has as many requests in the runner as it may.

    It is refused before it loads or waits for anything, see the module and
    `MAX_WAITING`; a service answers it with a 503 to retry, whose message
    names nothing of the service.
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


def check_app_variant(entry: ModelEntry, variant: str | None) -> None:
    """Check the variant of a request of the app of a model.

    The app is served in the variants of `ModelEntry.app_variants`, the ones
    its pages were checked in when the registry was built, and in no other.

    Args:
        entry: the entry of the model of the request.
        variant: the variant the request names, `None` for the default one.

    Raises:
        UnknownVariant: naming the model and its variants, for a variant the
            project does not have, and naming the model and the variants of
            its app, for a variant the app does not name.
    """
    check_variant(entry, variant)
    offered = entry.app_variants()
    if variant is not None and variant not in offered:
        raise UnknownVariant(
            f"The app of the model '{entry.id}' offers no variant '{variant}'; "
            f"its variants are {sorted(str(name) for name in offered)}."
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


def check_values(
    panel: Panel, values: Mapping[str, float], grid: Mapping[str, int]
) -> None:
    """Check the values of a request against the knobs a panel declares.

    A value is a value its knob takes: within the bounds of a range knob,
    whether it is a named stop of it or not, or one of the choices of a choice
    knob; and a grid over a choice knob runs its choices, so it has as many
    points as the knob has choices. A request of another value runs a model
    in a state the panel does not declare, e.g. a renal function a build of
    the model has no class of, and is refused before it runs or is answered
    from the cache. The ids are checked first (`check_knobs`), so every id
    here is a knob of the panel; it is a check of a request and not of a run,
    which `PanelRunner.run` runs as it is given.

    Args:
        panel: the panel of the request, the declaration of the variant the
            request names.
        values: the value of a knob by its id.
        grid: the points of a knob of the grid by its id.

    Raises:
        KnobValueError: naming the knob, the value and what the knob takes,
            for the first value of `values` or `grid` its knob does not take.
    """
    knobs = {knob.id: knob for knob in panel.knobs}
    for knob_id, value in values.items():
        knob = knobs.get(knob_id)
        if knob is None:
            continue
        taken = _no_value_of(knob, value)
        if taken:
            raise KnobValueError(
                f"The value {value!r} of the knob '{knob_id}' of the panel "
                f"'{panel.id}' is {taken}."
            )
    for knob_id, points in grid.items():
        knob = knobs.get(knob_id)
        if knob is None or knob.kind != "choice" or points == len(knob.choices):
            continue
        raise KnobValueError(
            f"A grid of {points} points over the knob '{knob_id}' of the panel "
            f"'{panel.id}' is none it runs: a grid over it runs its "
            f"{len(knob.choices)} choices."
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

    The panels of a model may differ by its variant: the declaration of a
    panel in a variant is the one of the SBML of that variant
    (`sbmlsim.project.Project.panels_of`), so the pages of the app and the
    knobs they share are worked out per variant (`pages`, `shared`), once for
    each and kept.

    Attributes:
        id: the id of the model, the one a request names.
        project: the project of the model package.
        title: the title of the model, as it is shown.
    """

    id: str
    project: Project
    title: str
    #: the panels of each page of the app, by the name of the variant
    _pages: dict[str | None, dict[str, tuple[Panel, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    #: the knobs the shell of each page owns, by the name of the variant
    _shared: dict[str | None, dict[str, tuple[SharedKnob, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def variant_name(self, variant: str | None) -> str | None:
        """Get the name of a variant of the model, the default one for `None`.

        Args:
            variant: the name of a variant, `None` for the default variant.

        Returns:
            The name, the first variant of the project for `None`, which is
            its default; `None` for a project without variants, whose panels
            cannot be read (`sbmlsim.project.Project.panels_of`).
        """
        if variant is not None or not self.project.variants:
            return variant
        return next(iter(self.project.variants))

    def app_variants(self) -> tuple[str | None, ...]:
        """Get the variants the app of the model is served in.

        They are the default variant of the project, which a request without
        a variant is answered with, and the variants the app names
        (`sbmlsim.panel.App.variants`), which a reader switches between: the
        pages of the app are checked in each of them when the registry of a
        service is built (`_check_app`), and the app is served in no other.

        Returns:
            The names, the default variant first and the others in the order
            the app names them; the default variant alone for a model without
            an app.
        """
        default = self.variant_name(None)
        app = self.project.app
        named = [] if app is None else [variant.id for variant in app.variants]
        return (default, *(name for name in named if name != default))

    def pages(self, variant: str | None = None) -> dict[str, tuple[Panel, ...]]:
        """Get the panels of each page of the app of the model in a variant.

        They are the panels `ModelRegistry.panel` serves for that variant,
        read once, so the knobs a page shares (`shared`) and the heads of its
        panels which the app carries (`sbmlsim.serve.payload.AppResponse`) are
        of one reading of the SBML of the variant.

        Args:
            variant: the name of the variant, the default variant without.

        Returns:
            The panels of every page by its id, in the order the page shows
            them; a model without an app has no pages.

        Raises:
            KeyError: for a page which names a panel the variant does not
                declare, which `_check_app` refuses first for the variants of
                `app_variants`, and for a variant the project does not have.
        """
        name = self.variant_name(variant)
        if name not in self._pages:
            app = self.project.app
            if app is None:
                return {}
            panels = self.project.panels_of(name)
            self._pages[name] = {
                page.id: tuple(panels[one] for one in page.panels) for page in app.pages
            }
        return self._pages[name]

    def shared(self, variant: str | None = None) -> dict[str, tuple[SharedKnob, ...]]:
        """Get the knobs the shell of each page of the app of the model owns.

        They are derived from the panels and are no part of the declaration of
        the app (`sbmlsim.panel.App`): the app names the panels of a page, the
        panels declare their knobs, and a knob id which more than one panel of
        a page declares is one knob of that page. A knob a scenario of the
        page sets (`sbmlsim.panel.Scenario`) is a knob of the page as well,
        even when one panel declares it: the shell sets it when a reader picks
        the scenario, so it owns its control, and a panel shows no control of
        a knob the shell sets. They are worked out once per variant, from the
        panels `ModelRegistry.panel` serves for it, which a service does when
        its registry is built (`ModelRegistry.of`), so a page whose panels
        declare one id differently is refused at startup.

        Args:
            variant: the name of the variant, the default variant without.

        Returns:
            The shared knobs of every page by its id, each in the order the
            first panel which declares one lists them (`_shared_knobs`); a page
            whose panels share none and which has no scenario has none, and a
            model without an app has no pages.

        Raises:
            ValueError: naming the model, the variant, the page, the knob id,
                the two panels and the fields, for a page two of whose panels
                declare one knob id differently, see `_shared_knobs`.
            KeyError: for a page which names a panel the variant does not
                declare, which `_check_app` refuses first.
        """
        name = self.variant_name(variant)
        if name not in self._shared:
            app = self.project.app
            if app is None:
                return {}
            pages = self.pages(name)
            self._shared[name] = {
                page.id: _shared_knobs(
                    f"the model '{self.id}' in its variant '{name}'",
                    page.id,
                    list(pages[page.id]),
                    {knob for scenario in page.scenarios for knob in scenario.values},
                )
                for page in app.pages
            }
        return self._shared[name]


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

    def panel(self, model_id: str, panel_id: str, variant: str | None = None) -> Panel:
        """Get a panel of a served model in a variant.

        The panels are the ones of the variant, the annotated ones of its SBML
        with the declarations of the project over them
        (`sbmlsim.project.Project.panels_of`), which reads the SBML once and
        keeps them: the declaration of a panel in a variant is the one of that
        variant.

        Args:
            model_id: the id of the model.
            panel_id: the id of the panel.
            variant: the name of the variant, the default variant without.

        Returns:
            The panel.

        Raises:
            UnknownModel: for a model which is not served.
            UnknownVariant: for a variant the project does not have.
            UnknownPanel: naming the id, for a panel the variant does not
                declare.
        """
        entry = self.entry(model_id)
        check_variant(entry, variant)
        panels = entry.project.panels_of(variant)
        try:
            return panels[panel_id]
        except KeyError:
            within = "" if variant is None else f" in its variant '{variant}'"
            raise UnknownPanel(
                f"'{panel_id}' is no panel of the model '{model_id}'{within}."
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

    The panels of a model may differ by its variant, so the app is checked in
    every variant it is served in (`ModelEntry.app_variants`): the default
    variant and the variants the app names. In each, every panel of every page
    is a panel of the variant, the panels of a page declare a knob id alike
    (`ModelEntry.shared`) and the scenarios of a page set knobs of its panels
    to values of them (`_check_scenarios`). The variants are checked first, so
    an app of an unknown variant is refused without reading an SBML.

    A model which declares no app is not checked here and reads nothing.

    Args:
        entry: the model of the registry.

    Raises:
        ValueError: naming the model and the variants, for an app which names
            a variant the project does not have; naming the model, the
            variant, the page and the panels, for a page which names a panel
            the variant does not declare; naming the model, the variant, the
            page, the knob id and the two panels, for a page whose panels
            declare one knob id differently, see `ModelEntry.shared`; naming
            the model, the variant, the page, the scenario, the knob and the
            panel, for a scenario which sets a knob no panel of its page
            declares or a value which is no value of that knob, see
            `_check_scenarios`; if the project has no model variants, see
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
    for variant in entry.app_variants():
        model = f"the model '{entry.id}' in its variant '{variant}'"
        declared = sorted(entry.project.panels_of(variant))
        for page in app.pages:
            missing = sorted(set(page.panels).difference(declared))
            if missing:
                raise ValueError(
                    f"The page '{page.id}' of the app of {model} shows the panels "
                    f"{missing}, which the variant does not declare; its panels "
                    f"are {declared}."
                )
        # the shared knobs of every page, which raise for one id declared
        # twice, so the knobs a scenario sets are of one declaration per page
        _ = entry.shared(variant)
        pages = entry.pages(variant)
        for page in app.pages:
            _check_scenarios(model, page, pages[page.id])


def _check_scenarios(model: str, page: AppPage, panels: Sequence[Panel]) -> None:
    """Check the scenarios of a page of an app against the knobs of its panels.

    A scenario (`sbmlsim.panel.Scenario`) sets knobs of its page, which the
    declaration cannot check, since the panels are read from the SBML: a key
    which no panel of the page declares sets nothing a reader sees, and a
    value which is no value of its knob in a panel which declares it, outside
    the bounds of a range knob or none of the choices of a choice knob, is a
    state that panel cannot be in.

    Args:
        model: the model and the variant of the panels, which a problem
            names, e.g. `the model 'hctz' in its variant 'default'`.
        page: the page of the app.
        panels: the panels of the page, in the order it shows them.

    Raises:
        ValueError: naming the model, the variant, the page, the scenario and
            the knob, for a knob no panel of the page declares, and naming the
            panel as well, for a value which is no value of the knob in that
            panel.
    """
    for scenario in page.scenarios:
        for knob_id, value in scenario.values.items():
            where = (
                f"The scenario '{scenario.id}' of the page '{page.id}' of the app "
                f"of {model}"
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
    """Tell why a value is no value of a knob, see `_check_scenarios` and `check_values`.

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
    """Check the targets and the derived targets of every panel of every variant.

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

    The targets of the knobs, the derived targets and the keys of `preinit`
    are checked here as well, against the entities of the model of the
    variant, which are read with its panels (`_check_entities`); a
    hierarchical model, whose entities are the ones of its flattened model, is
    not checked.

    Args:
        entry: the model of the registry.

    Raises:
        ValueError: naming the model, the variant, the panel, the target and
            the formula, for a formula which is no math of PEtab; naming them
            and the symbols, for a formula which reads a symbol which is no
            knob of its panel; naming the model, the variant, the panel and
            the knob, for a knob without a target which no formula reads; and
            naming the model, the variant, the panel and the id, for a target
            which is no entity of the model, see `_check_entities`.
        FileNotFoundError: if the SBML of a variant is no file.
    """
    for variant in entry.project.variants:
        entities = entry.project.model_entities(variant)
        for panel in entry.project.panels_of(variant).values():
            where = f" of the model '{entry.id}' in its variant '{variant}'"
            if entities is not None:
                _check_entities(panel, entities, where)
            formulas = _formulas(panel, where)
            read = {s for formula in formulas.values() for s in formula.symbols}
            for knob in panel.knobs:
                if knob.target is None and knob.id not in read:
                    raise ValueError(
                        f"The knob '{knob.id}' of the panel '{panel.id}'{where} sets "
                        f"no target and no formula of its derived targets reads it; "
                        f"a knob without a target is an input of a formula."
                    )


def _check_entities(panel: Panel, entities: Set[str], where: str) -> None:
    """Check that a panel sets entities of its model and no other ids.

    The targets of the knobs, the derived targets and the keys of `preinit`
    of a panel are the values of a pre-initialization, which the simulator
    refuses for an id which is no entity of the model, e.g. a typo, on every
    request of the panel; it is refused once, when the service starts,
    against the entities read from the SBML of the variant
    (`sbmlsim.project.Project.model_entities`).

    Args:
        panel: the panel.
        entities: the ids a pre-initialization of the model can set, see
            `sbmlsim.panel_sbml.ModelPanels.entities`.
        where: the model and the variant of the panel, which a problem names
            after the panel, e.g. ` of the model 'hctz' in its variant
            'default'`.

    Raises:
        ValueError: naming the panel, the model, the variant, what sets the id
            and the id, for an id which is no entity of the model.
    """
    targets = [
        *(
            (f"the target of the knob '{knob.id}'", knob.target)
            for knob in panel.knobs
            if knob.target is not None
        ),
        *(("a derived target", target) for target in panel.simulation.derived),
        *(("a target of the preinit", target) for target in panel.simulation.preinit),
    ]
    for what, target in targets:
        if target not in entities:
            raise ValueError(
                f"The panel '{panel.id}'{where} sets '{target}', {what}, which is "
                f"no entity of the model: a target is a parameter, a species, a "
                f"compartment or a species reference of the model, or '[S]', the "
                f"concentration of a species 'S'."
            )


def _shared_knobs(
    model: str,
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
        model: the model and the variant of the panels, which a problem
            names, e.g. `the model 'hctz' in its variant 'default'`.
        page_id: the id of the page.
        panels: the panels of the page, in the order it shows them, each once.
        named: the ids of the knobs the scenarios of the page set.

    Returns:
        The knobs more than one of the panels declares and the knobs of
        `named` which a panel declares, with the panels which declare each.

    Raises:
        ValueError: naming the model, the variant, the page, the knob id, the
            two panels and the fields, for a knob id which two panels of the
            page declare differently.
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
                    f"The page '{page_id}' of the app of {model} "
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
        key: the model id and the SBML file of the model, see `_WarmKey`.
        sha256: the sha256 of the SBML file the model was loaded from.
        model: the model, in the engine and at the optimization level of the
            project.
        simulator: the serial simulator of the settings of the project.
        lock: the lock a run of the model holds, see the module.
    """

    key: _WarmKey
    sha256: str
    model: SBMLModel
    simulator: Simulator
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class _Resolved:
    """The scan of a request of a panel, see `_scan`.

    Attributes:
        scan: the scan.
        axes: the axes of its grid, one per knob of the grid in the order of
            the knobs of the panel.
        preinit: the values of the pre-initialization by their target, in the
            order of the targets, which the simulation gets them in.
        grid: the values of the dimension of the grid by their target, in the
            order of the targets, which the dimension gets them in, empty for
            a request of a point.
    """

    scan: Scan
    axes: tuple[GridAxis, ...]
    preinit: Mapping[str, Any]
    grid: Mapping[str, Any]


@dataclass(frozen=True)
class _Run:
    """A run of a scan which the panels of its simulation share, see the module.

    Attributes:
        result: the result of the scan.
        selections: the selections the result keeps, the curves of every panel
            of the simulation which the model can select.
    """

    result: ScanResult
    selections: frozenset[str]


class PanelRunner:
    """Runs the panels of the models of a registry, see the module.

    Attributes:
        registry: the models a request may name.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        max_runs: int = MAX_RUNS,
        max_waiting: int = MAX_WAITING,
    ) -> None:
        """Create a runner of the models of a registry, which loads none yet.

        Args:
            registry: the models a request may name.
            max_runs: the runs the runner keeps for the panels which share
                them, see the module; 0 shares none, so every request runs the
                scan of its panel with the curves of that panel alone.
            max_waiting: the requests of one model id which may be in the
                runner at once, see `MAX_WAITING`.

        Raises:
            ValueError: for a negative number of runs and for fewer than one
                request of a model.
        """
        if max_runs < 0:
            raise ValueError(f"A runner keeps no negative number of runs: {max_runs}.")
        if max_waiting < 1:
            raise ValueError(
                f"A runner takes at least one request of a model, not {max_waiting}."
            )
        self.registry = registry
        self._max_waiting = max_waiting
        #: the requests of every model id which are in the runner, see
        #: `_admitted`
        self._in_runner: dict[str, int] = {}
        #: the lock of `_in_runner`, never held over anything slow
        self._admission = threading.Lock()
        #: the warm model of every model id and SBML file, see `_warm_model`
        self._warm: dict[_WarmKey, _Warm] = {}
        #: the lock of the load of one such key, see `_warm_model`
        self._loads: dict[_WarmKey, threading.Lock] = {}
        #: the runs the panels of a simulation share by `_run_key`, `None` for
        #: a runner which shares none
        self._runs: BoundedCache[_Run] | None = (
            BoundedCache(max_runs, what="run") if max_runs else None
        )

    def __repr__(self) -> str:
        """Get the representation."""
        return f"PanelRunner({self.registry!r})"

    @contextmanager
    def _admitted(self, model_id: str) -> Generator[None]:
        """Admit a request of a model into the runner, see the module.

        Args:
            model_id: the id of the model of the request.

        Yields:
            Nothing; the request is in the runner until the block ends.

        Raises:
            ModelBusy: if the model has `max_waiting` requests in the runner,
                at once and before anything is loaded or waited for.
        """
        with self._admission:
            count = self._in_runner.get(model_id, 0)
            if count >= self._max_waiting:
                raise ModelBusy(
                    f"The model '{model_id}' has {count} requests in the runner."
                )
            self._in_runner[model_id] = count + 1
        try:
            yield
        finally:
            with self._admission:
                self._in_runner[model_id] -= 1

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

        The scan of the state is run once for every panel of the variant which
        shares the simulation of the panel, with the curves of all of them,
        and a state which ran already is read out of that run, see the module.

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
            ModelBusy: for a model which has as many requests in the runner
                as it may, see the module; before a model is loaded.
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
        name = panel.simulation.variant if variant is None else variant
        with self._admitted(entry.id):
            return self._run_admitted(entry, panel, values, grid, name)

    def _run_admitted(
        self,
        entry: ModelEntry,
        panel: Panel,
        values: Mapping[str, float],
        grid: Mapping[str, int],
        name: str | None,
    ) -> PanelResult:
        """Run a panel of a request which was admitted into the runner, see `run`.

        Args:
            entry: the entry of the model.
            panel: the panel.
            values: the value of a knob by its id.
            grid: the points of a knob of the grid by its id.
            name: the variant of the model to simulate.

        Returns:
            The result, see `run`.
        """
        model_id = entry.id
        warm = self._warm_model(entry, name)
        _check_limits(warm.model, panel)
        with warm.lock:
            resolved = _scan(warm.model, panel, values, grid)
            curves = [curve.id for curve in panel.curves]
            runs = self._runs
            if runs is None or entry.project.simulator.engine not in SHARED_ENGINES:
                result = _run(warm, panel, resolved.scan, curves)
                return _result(warm.model, panel, resolved.axes, result)
            key = _run_key(warm, entry, panel, resolved)
            stored = runs.get(key)
            if stored is not None and stored.selections.issuperset(curves):
                logger.debug(
                    "The panel '%s' of the model '%s' is read out of the run '%s'.",
                    panel.id,
                    model_id,
                    key,
                )
                return _result(warm.model, panel, resolved.axes, stored.result)
            selections = _selections(
                warm.model, panel, entry.project.panels_of(name).values()
            )
            # the curves of the run it replaces as well, so the panels which
            # read that run read this one and two groups of one state do not
            # replace each other's run back and forth
            if stored is not None:
                selections += sorted(stored.selections.difference(selections))
            result = _run(warm, panel, resolved.scan, selections)
            runs.store(key, _Run(result=result, selections=frozenset(selections)))
            return _result(warm.model, panel, resolved.axes, result)

    def units(
        self, entry: ModelEntry, panel: Panel, variant: str | None = None
    ) -> PanelUnits:
        """Get the units a panel is delivered in, see `PanelUnits`.

        The x is the one `PanelResult.units[X]` of a run of the panel names,
        and every unit is resolved in the registry of the warm model of the
        entry, the model a run of the panel integrates on, so the study data
        of a panel (`sbmlsim.serve.data.panel_data`) is delivered on the axis
        of its curves and in their units whatever the declaration names.

        Args:
            entry: the entry of the model, the one the caller holds.
            panel: the panel.
            variant: the variant of the model whose units are resolved, the
                one the panel declares without, as for `run`.

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
            ModelBusy: for a model which has as many requests in the runner
                as it may, see the module, since a cold model is loaded here.
        """
        with self._admitted(entry.id):
            warm = self._warm_model(
                entry, panel.simulation.variant if variant is None else variant
            )
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
    which pins an engine or the tolerances of its integrator is honoured. The
    SBML is hashed before it is loaded, so a file which is rewritten during
    the load gives a hash of another model and never the hash of this one to
    a model of another file.

    Args:
        entry: the entry of the model.
        path: the SBML file of the variant, see `PanelRunner._warm_model`.

    Returns:
        The warm model.
    """
    sha256 = file_sha256(path)
    settings = entry.project.simulator
    simulator = settings.simulator(n_workers=1)
    model = simulator.load(
        load_model(
            path,
            settings.engine,
            jit_optimization=settings.jit_optimization,
        )
    )
    return _Warm(key=(entry.id, path), sha256=sha256, model=model, simulator=simulator)


def _run(
    warm: _Warm, panel: Panel, scan: Scan, selections: Sequence[str]
) -> ScanResult:
    """Run the scan of a panel on a warm model, which the caller holds the lock of.

    Args:
        warm: the warm model.
        panel: the panel, whose simulation gives the times.
        scan: the scan, see `_scan`.
        selections: the selections the result keeps.

    Returns:
        The result, on the grid of `points + 1` times from 0 to the `end` of
        the simulation of the panel; a point which did not integrate is `NaN`
        and flagged.
    """
    times = np.linspace(0.0, panel.simulation.end, panel.simulation.points + 1)
    return warm.simulator.run(
        warm.model,
        scan,
        time=times,
        keep=list(selections),
        on_error="flag",
        progress=False,
    )


def _selections(model: SBMLModel, panel: Panel, panels: Iterable[Panel]) -> list[str]:
    """Get the selections of a run which the panels of a simulation share.

    They are the curves of the panel, in their order, and then the curves of
    every other panel of the variant which shares its simulation, the same
    variant, `end`, `points`, `x_unit`, `preinit` and `derived`, so the panels
    of a page which ask for one state are answered by one run. A curve of
    another panel which the model cannot select is left out, so a panel which
    shares the simulation never fails the run of another one; a curve of the
    panel itself is kept whatever it is, so the panel fails as it fails alone.

    Args:
        model: the warm model the run integrates on.
        panel: the panel of the request.
        panels: the panels of the variant, see
            `sbmlsim.project.Project.panels_of`.

    Returns:
        The selections, each once.
    """
    selections = [curve.id for curve in panel.curves]
    seen = set(selections)
    for other in panels:
        if other.simulation != panel.simulation:
            continue
        for curve in other.curves:
            if curve.id not in seen and model.has_selection(curve.id):
                selections.append(curve.id)
                seen.add(curve.id)
    return selections


def _run_key(warm: _Warm, entry: ModelEntry, panel: Panel, resolved: _Resolved) -> str:
    """Get the key of a run, of what decides its numbers and not of the panel.

    It is the warm model, which a stored run is read on under its lock, the
    sha256 of its SBML, the engine and the settings of the simulator of the
    project, the times and the resolved scan: the values of the
    pre-initialization and of the dimension of the grid by their target, in
    the canonical order of the targets in which `_scan` gives them to the
    simulator, whatever order the panel declares its knobs in. Every number is
    the exact double (`_exact`), so two states which differ in the last bit
    are two runs. The ids of the knobs and the panel are no part of it: a
    panel whose knobs resolve to the scan of another one is answered by its
    run.

    Args:
        warm: the warm model.
        entry: the entry of the model, whose project gives the settings.
        panel: the panel, whose simulation gives the times.
        resolved: the scan of the request, see `_scan`.

    Returns:
        The sha256 of the canonical JSON of the key, see
        `sbmlsim.hashing.canonical_json_sha256`.
    """
    model_id, path = warm.key
    return canonical_json_sha256(
        {
            "model": model_id,
            "path": str(path),
            "model_sha256": warm.sha256,
            "engine": entry.project.simulator.engine.value,
            "simulator": entry.project.simulator.model_dump(mode="json"),
            "end": _exact(panel.simulation.end),
            "points": panel.simulation.points,
            "preinit": [[t, _exact(v)] for t, v in resolved.preinit.items()],
            "grid": [[t, _exact(v)] for t, v in resolved.grid.items()],
        }
    )


def _exact(value: Any) -> Any:
    """Get a value of a scan as JSON which holds it exactly, see `_run_key`.

    Args:
        value: a number, an array of numbers or a quantity of either, e.g. the
            value of a knob in its unit.

    Returns:
        A number as the hexadecimal text of its double, an array as its shape
        and its numbers, a quantity as its magnitude and its unit, and any
        other value as its type and its representation.
    """
    if isinstance(value, Quantity):
        return {"magnitude": _exact(value.magnitude), "unit": str(value.units)}
    if isinstance(value, np.ndarray):
        numbers = np.asarray(value, dtype=float)
        return {
            "shape": list(numbers.shape),
            "values": [float(number).hex() for number in numbers.ravel()],
        }
    if isinstance(value, (int, float, np.number)):
        return float(value).hex()
    return {"type": type(value).__name__, "repr": repr(value)}


def _scan(
    model: SBMLModel,
    panel: Panel,
    values: Mapping[str, float],
    grid: Mapping[str, int],
) -> _Resolved:
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
        a point, the axes of the grid, one per knob of the grid in the order
        of the knobs of the panel, and the values it sets, see `_Resolved`.

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
    # the values in the order of their targets and not of the declaration of
    # the knobs, so one state is one scan whichever panel resolves it
    preinit = dict(sorted(preinit.items()))
    changed = dict(sorted(changed.items()))
    dimensions = (
        [Dimension("__".join(axis.id for axis in axes), values=changed)]
        if gridded
        else []
    )
    simulation = Simulation(end=panel.simulation.end, preinit_changes=preinit)
    return _Resolved(
        scan=Scan(simulation, dimensions), axes=axes, preinit=preinit, grid=changed
    )


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
