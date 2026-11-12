"""The models a request may name and the runner of their panels.

`ModelRegistry` is the models of the service. It is built once at startup from
a mapping of model id to project reference (`ModelRegistry.of`) and is the only
way from a request to a model: `entry` answers an id of that mapping and
raises `UnknownModel` for everything else, so a request can never name a
module, a project reference or a path. A model whose project declares an app
(`sbmlsim.panel.App`) is checked there as well (`_check_app`): a page which
names a panel the model does not declare and a variant the project does not
have are refused at startup, since the panels of a model are read from its
SBML, which a project is imported without.

`PanelRunner` runs a panel of such a model: the values of its knobs become the
pre-initialization of one `Simulation`, a knob of the grid becomes a
`Dimension` of the values of `Knob.values(n)`, and the scan of the two runs on
the warm model of the entry. The curves of the `PanelResult` are the declared
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
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tokenize import TokenError
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from pint.errors import PintError
from pint.facets.plain import PlainUnit

from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.panel import Knob, Panel, PanelCurve, PanelData
from sbmlsim.project import Project, load_project
from sbmlsim.result.scan import STATUS, TIME
from sbmlsim.simulation import Dimension, Scan, Simulation

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
    "UnknownKnob",
    "UnknownModel",
    "UnknownPanel",
    "X",
    "check_knobs",
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
                not declare, and naming the model and the variants, for an app
                which names a variant the project does not have, see
                `_check_app`.
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
            declare; if the project has no model variants, see
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
    declared = sorted(entry.project.panels_of())
    for page in app.pages:
        missing = sorted(set(page.panels).difference(declared))
        if missing:
            raise ValueError(
                f"The page '{page.id}' of the app of the model '{entry.id}' shows "
                f"the panels {missing}, which the model does not declare; its "
                f"panels are {declared}."
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

        Returns:
            The result, see `PanelResult`; a point which did not integrate is
            `NaN` and its flat index is in `failed`.

        Raises:
            UnknownModel: for a model which is not served.
            UnknownKnob: naming the ids, if `values` or `grid` name something
                which is no knob of the panel; it is raised before a model is
                loaded, so a request which names one costs nothing.
            ValueError: for a grid of fewer than one point, see `Knob.values`;
                naming both units, for an `x_unit` of the panel which is no
                time unit of the model, see `_times`, and for a unit of a
                curve which is no unit of its values, see `_curve_unit`. The
                limits of a request (`sbmlsim.panel.MAX_KNOBS`, `MAX_GRID`)
                are checked where a request is read.
        """
        entry = self.registry.entry(model_id)
        check_knobs(panel, values, grid)
        warm = self._warm_model(entry, panel.simulation.variant)
        times = np.linspace(0.0, panel.simulation.end, panel.simulation.points + 1)
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
                of pint and none of the model; if the project of the entry has
                no model variants.
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

    Args:
        model: the model, which resolves the units of the knobs.
        panel: the panel.
        values: the value of a knob by its id.
        grid: the points of a knob of the grid by its id.

    Returns:
        The scan, one dimension per knob of the grid in the order of the
        knobs of the panel, and the axes of these dimensions.
    """
    preinit: dict[str, Any] = dict(panel.simulation.preinit)
    dimensions: list[Dimension] = []
    axes: list[GridAxis] = []
    for knob in panel.knobs:
        if knob.id in grid:
            points = knob.values(grid[knob.id])
            axes.append(
                GridAxis(
                    id=knob.id,
                    label=knob.label,
                    unit=_unit_of(model, knob),
                    values=points,
                )
            )
            dimensions.append(
                Dimension(
                    knob.id,
                    values={
                        knob.target: _in_model_units(
                            model, knob, np.asarray(points, dtype=float)
                        )
                    },
                )
            )
        else:
            value = float(values.get(knob.id, knob.default))
            preinit[knob.target] = _in_model_units(model, knob, value)
    simulation = Simulation(end=panel.simulation.end, preinit_changes=preinit)
    return Scan(simulation, dimensions), tuple(axes)


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
            unit, for a unit of a series which is no unit, see `_series_unit`.
    """
    return PanelUnits(
        x=_x_unit(model, panel),
        series={data.id: _series_unit(model, panel, data) for data in panel.data},
    )


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
        `None` for a target which has none.
    """
    if knob.unit is not None:
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
