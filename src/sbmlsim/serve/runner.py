"""The models a request may name and the runner of their panels.

`ModelRegistry` is the models of the service. It is built once at startup from
a mapping of model id to project reference (`ModelRegistry.of`) and is the only
way from a request to a model: `entry` answers an id of that mapping and
raises `UnknownModel` for everything else, so a request can never name a
module, a project reference or a path.

`PanelRunner` runs a panel of such a model: the values of its knobs become the
pre-initialization of one `Simulation`, a knob of the grid becomes a
`Dimension` of the values of `Knob.values(n)`, and the scan of the two runs on
the warm model of the entry. The curves of the `PanelResult` are the declared
curves of the panel, one row per point of the grid in C order and one row for a
request without a grid, in the unit the curve declares or else in the unit of
the model; the x of the result is the grid of `points + 1` times from 0 to the
`end` of the panel, both in the time unit of the model, which `units["x"]`
names. A point which does not integrate is `NaN` and its flat index is in
`failed`, never an exception: a knob value a reader turns to may be a value
the model does not simulate.

A model and its simulator are loaded once per `(model_id, variant)` and kept
warm, and every run of one of them holds its lock: roadrunner instances are
not thread safe and a service runs its sync endpoints in a thread pool.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.panel import Knob, Panel
from sbmlsim.project import Project, load_project
from sbmlsim.result.scan import STATUS, TIME
from sbmlsim.simulation import Dimension, Scan, Simulation

if TYPE_CHECKING:
    from sbmlsim.result import ScanResult
    from sbmlsim.simulator import Simulator

logger = logging.getLogger(__name__)

__all__ = [
    "GridAxis",
    "ModelEntry",
    "ModelRegistry",
    "PanelResult",
    "PanelRunner",
    "UnknownModel",
    "UnknownPanel",
    "X",
]

#: the key of the time of a result in `PanelResult.units`
X: Final = "x"


class UnknownModel(LookupError):
    """A model id which the registry does not serve."""


class UnknownPanel(LookupError):
    """A panel id which a served model does not declare."""


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
                `sbmlsim.project.load_project`.
            ImportError: if the module of a reference is not imported.
        """
        entries: dict[str, ModelEntry] = {}
        for model_id, reference in specs.items():
            project = load_project(reference)
            entries[model_id] = ModelEntry(
                id=model_id,
                project=project,
                title=project.title or project.name,
            )
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
class PanelResult:
    """The result of a run of a panel, see the module.

    Attributes:
        x: the output times, in the time unit of the model.
        dims: the axes of the grid, empty for a run of one point.
        curves: the values of every declared curve, a row per point of the
            grid in C order.
        units: the unit of every curve and, under `X`, of the x.
        failed: the flat indices in C order of the points which did not
            integrate, whose rows are `NaN`.
    """

    x: tuple[float, ...]
    dims: tuple[GridAxis, ...]
    curves: Mapping[str, list[list[float]]]
    units: Mapping[str, str]
    failed: tuple[int, ...]


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
        #: the warm model of every model id and variant, see `_warm_model`
        self._warm: dict[tuple[str, str | None], _Warm] = {}
        #: held while a model is looked up or loaded, so it is loaded once
        self._loading = threading.Lock()

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
            ValueError: naming the ids, if `values` or `grid` name something
                which is no knob of the panel; for a grid of fewer than one
                point, see `Knob.values`. The limits of a request
                (`sbmlsim.panel.MAX_KNOBS`, `MAX_GRID`) are checked where a
                request is read.
        """
        entry = self.registry.entry(model_id)
        knobs = {knob.id: knob for knob in panel.knobs}
        unknown = sorted((set(values) | set(grid)) - set(knobs))
        if unknown:
            raise ValueError(
                f"{unknown} are no knobs of the panel '{panel.id}', which has "
                f"{sorted(knobs)}."
            )
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

    def _warm_model(self, entry: ModelEntry, variant: str | None) -> _Warm:
        """Get the warm model of a variant of an entry, loaded on its first use.

        A model is loaded once: the lock is held while it is looked up and
        loaded, so two threads which ask for the same model at once do not
        compile it twice and the second one waits for the first.

        Args:
            entry: the entry of the model.
            variant: the variant of the model, its default variant without.

        Returns:
            The warm model.

        Raises:
            ValueError: if the project of the entry has no model variants.
            KeyError: for a variant the project does not have.
        """
        key = (entry.id, variant)
        with self._loading:
            warm = self._warm.get(key)
            if warm is None:
                warm = _load(entry, variant)
                self._warm[key] = warm
                logger.info(
                    "The model '%s' (%s) is loaded and stays warm.",
                    entry.id,
                    variant or "default variant",
                )
            return warm


def _load(entry: ModelEntry, variant: str | None) -> _Warm:
    """Load a variant of the model of an entry with its simulator.

    The model is loaded in the engine and at the optimization level of the
    project and the simulator is the serial one of its settings, so a project
    which pins an engine or the tolerances of its integrator is honoured.

    Args:
        entry: the entry of the model.
        variant: the variant of the model, its default variant without.

    Returns:
        The warm model.
    """
    settings = entry.project.simulator
    simulator = settings.simulator(n_workers=1)
    model = simulator.load(
        load_model(
            entry.project.variant_path(variant),
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
        pint.DimensionalityError: for a curve whose declared unit is not the
            unit of its values in the model.
    """
    x = tuple(float(time) for time in np.asarray(result.ds.coords[TIME].values))
    curves: dict[str, list[list[float]]] = {}
    units: dict[str, str] = {X: result.units.get(TIME, "")}
    for curve in panel.curves:
        quantity = result.quantity(curve.id)
        if curve.unit is not None:
            quantity = quantity.to(model.Q(1.0, curve.unit).units)
        values = np.asarray(quantity.magnitude, dtype=float).reshape(-1, len(x))
        curves[curve.id] = values.tolist()
        units[curve.id] = (
            curve.unit if curve.unit is not None else result.units.get(curve.id, "")
        )
    status = result.ds.get(STATUS)
    failed: tuple[int, ...] = ()
    if status is not None:
        flags = np.asarray(status.values).reshape(-1)
        failed = tuple(int(index) for index in np.flatnonzero(flags))
    return PanelResult(x=x, dims=axes, curves=curves, units=units, failed=failed)
