"""The simulation experiments over the draws of the fitted parameters of a model.

A fit of a project becomes part of its model in three steps:

1. **Adopt** the fit: the kind `fit` with `adopt` (or the kind `adopt`) writes
   the record of a finished fit (its parameters, the profile likelihood and the
   Fisher information) and the adopted fits of the project with it, which a
   person copies into the directory of the fit records of the project
   (`Project.fits_dir`), see `sbmlsim.fit.records.adopt_fit`.
2. **Use** it in the factory: a model file takes the value of a fitted
   parameter from the records of the adopted fits, together with its
   uncertainty, a distribution of the SBML package distrib, see
   `sbmlsim.fit.records.AdoptedRecords.parameter`:

   ```python
   # not executed
   FITS = PROJECT.fit_records()
   Parameter("k", **FITS.parameter("k", unit=U.per_min), name="rate")
   ```

   `check_fit_parameters` checks that every variant of the model has the
   values and the distributions of the records.
3. **Draw** it: a run of the experiments with `UncertaintySettings`
   (`Project.run_experiments(..., uncertainty=...)`, the field `uncertainty` of
   the kind `simulate`) draws the parameters of the default variant of the
   model which have a distribution once (`UncertaintyRun.create`, a Latin
   hypercube into the dimension `sbmlsim.distrib.UNCERTAINTY_DIMENSION`) and
   gives every experiment the draws as its setting `uncertainty`;
   `SimulationExperiment.initialize` then runs every simulation over the draws
   and every curve of a simulation becomes the band of a quantile range with
   the median, see `apply_uncertainty`.

The draws are independent per parameter: distrib states one distribution per
parameter, the correlations of the Fisher information stay in the record.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import libsbml

from sbmlsim.data import Data
from sbmlsim.distrib import check_band
from sbmlsim.plot.plotting import Curve
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.task import Task

if TYPE_CHECKING:
    from sbmlsim.experiment.experiment import SimulationExperiment
    from sbmlsim.project import Project
    from sbmlsim.simulation.sampling import Distribution

# the sampler is imported where the draws are made: it imports the fits,
# which import the experiments

__all__ = [
    "FITTED_SUFFIX",
    "VALUE_RTOL",
    "UncertaintyRun",
    "UncertaintySettings",
    "apply_uncertainty",
    "check_fit_parameters",
]

logger = logging.getLogger(__name__)

#: the relative tolerance of the value of a fitted parameter in the model
VALUE_RTOL = 1e-12

#: the suffix of the simulation and the task of the fitted values, which the
#: curves draw whose x is neither the time nor the values of a dimension of the scan
FITTED_SUFFIX = "_fitted"


# ----------------------------------------------------------------------------
# the settings and the draws of a run
# ----------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class UncertaintySettings:
    """The settings of a run over the draws of the fitted parameters.

    Attributes:
        draws: the number of draws.
        seed: the seed of the draws.
        quantiles: the lower and the upper quantile of the bands.
    """

    draws: int = 100
    seed: int = 1234
    quantiles: tuple[float, float] = (0.05, 0.95)

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            ValueError: for less than two draws or quantiles which are not an
                increasing pair within `(0, 1)`, see `sbmlsim.distrib.check_band`.
        """
        object.__setattr__(
            self, "quantiles", check_band(self.draws, tuple(self.quantiles))
        )


@dataclass(frozen=True)
class UncertaintyRun:
    """The draws and the settings of a run, a setting of the experiments.

    Attributes:
        settings: the settings of the run.
        dimension: the draws of the fitted parameters.
    """

    settings: UncertaintySettings
    dimension: Dimension

    @property
    def quantiles(self) -> tuple[float, float]:
        """The quantiles of the bands."""
        return self.settings.quantiles

    @property
    def id(self) -> str:
        """The id of the dimension of the draws."""
        return self.dimension.id

    @classmethod
    def create(
        cls,
        project: Project,
        settings: UncertaintySettings,
        variant: str | None = None,
    ) -> UncertaintyRun:
        """Draw the fitted parameters of the model of a project.

        The draws are those of a variant of the model, the default variant
        (`Project.model_path`) without; every variant has the same fitted
        parameters with the same distributions (`check_fit_parameters`), so
        the draws run on the experiments of every variant.

        Args:
            project: the project, whose variant has the distributions.
            settings: the settings of the run.
            variant: the variant whose distributions are drawn, the default
                variant without.

        Returns:
            The run.

        Raises:
            ValueError: if the project has no variants, or no parameter of the
                variant has a distribution.
            KeyError: naming the variants, for an unknown variant.
        """
        from sbmlsim.simulation.sampling import uncertainty_dimension

        return cls(
            settings=settings,
            dimension=uncertainty_dimension(
                project.variant_path(variant), settings.draws, settings.seed
            ),
        )


# ----------------------------------------------------------------------------
# the experiments over the draws
# ----------------------------------------------------------------------------
def _set_targets(simulation: Simulation) -> set[str]:
    """Get the targets a simulation sets itself."""
    targets = set(simulation.preinit_changes)
    if simulation.presimulation is not None:
        targets |= set(simulation.presimulation.preinit_changes)
    for change in simulation.changes:
        targets |= set(change.values)
    return targets


def _without(dimension: Dimension, targets: set[str]) -> Dimension | None:
    """Get the dimension without the values of some targets, `None` if none is left."""
    shared = targets & set(dimension.values)
    if not shared:
        return dimension
    values = {k: v for k, v in dimension.values.items() if k not in shared}
    if not values:
        return None
    return Dimension(
        dimension.id, values=values, labels=dimension.labels, design=dimension.design
    )


def _over_draws(
    simulation: Simulation | Scan,
    dimension: Dimension,
    key: str,
    sid: str,
    available: set[str] | None = None,
) -> Simulation | Scan:
    """Get a simulation or a scan over the draws of the fitted parameters.

    A parameter which the simulation, or a dimension of the scan, sets itself
    keeps its value: it is left out of the draws of this simulation, as is a
    parameter which the model of the simulation does not have.

    Args:
        simulation: the simulation or scan.
        dimension: the draws.
        key: the key of the simulation, for the messages.
        sid: the id of the experiment, for the messages.
        available: the parameters of the model of the simulation, `None` for
            all of the draws.

    Returns:
        The scan over the draws, the simulation itself if no drawn parameter
        is left.

    Raises:
        ValueError: if the scan has a dimension of the id of the draws.
    """
    if isinstance(simulation, Scan):
        if any(d.id == dimension.id for d in simulation.dimensions):
            raise ValueError(
                f"The scan '{key}' of the experiment '{sid}' has a dimension "
                f"'{dimension.id}', the id of the draws of the fitted parameters."
            )
        base = simulation.simulation
        targets = _set_targets(base)
        for d in simulation.dimensions:
            targets |= set(d.values)
            for s in d.simulations.values():
                targets |= _set_targets(s)
        dimensions = list(simulation.dimensions)
    else:
        base = simulation
        targets = _set_targets(simulation)
        dimensions = []
    left_out = sorted(targets & set(dimension.values))
    if left_out:
        logger.info(
            "The simulation '%s' of '%s' sets the fitted parameters %s itself, "
            "they are not drawn.",
            key,
            sid,
            left_out,
        )
    excluded = set(targets)
    if available is not None:
        excluded |= set(dimension.values) - available
    draws = _without(dimension, excluded)
    if draws is None:
        return simulation
    return Scan(base, [*dimensions, draws])


def _has_dimension(simulation: Simulation | Scan, dimension_id: str) -> bool:
    """Check whether a simulation is a scan with a dimension of an id."""
    return isinstance(simulation, Scan) and any(
        d.id == dimension_id for d in simulation.dimensions
    )


def _curve_color(curve: Curve) -> str | None:
    """Get the color of a curve, `None` for the color cycle."""
    color = (curve.kwargs or {}).get("color")
    if color is None:
        line = getattr(curve.style, "line", None)
        line_color = getattr(line, "color", None)
        color = getattr(line_color, "color", line_color)
    return None if color is None else str(color)


def _model_parameters(
    experiment: SimulationExperiment, model_id: str
) -> set[str] | None:
    """Get the parameters of a model of an experiment, `None` if it is not loaded.

    The global parameters of the SBML of the model, whatever its engine.
    """
    model = experiment._models.get(model_id)
    if model is None:
        return None
    return set(model.symbols.parameters)


def _fitted_task(
    experiment: SimulationExperiment,
    task_id: str,
    originals: Mapping[str, Simulation | Scan],
) -> str:
    """Get the task of the simulation of a task with the fitted values, created once.

    Returns:
        The id of the task.
    """
    task = experiment._tasks[task_id]
    fitted_task = f"{task_id}{FITTED_SUFFIX}"
    if fitted_task not in experiment._tasks:
        fitted_simulation = f"{task.simulation_id}{FITTED_SUFFIX}"
        experiment._simulations[fitted_simulation] = originals[task.simulation_id]
        experiment._tasks[fitted_task] = Task(
            model=task.model_id, simulation=fitted_simulation
        )
    return fitted_task


def _on_task(data: Data | None, task_id: str) -> Data | None:
    """Get the data of a task from another task."""
    if data is None or data.task_id is None:
        return data
    return Data(data.selection, task=task_id, sel=data.sel)


def _over_scan_dimension(data: Data, simulation: Simulation | Scan) -> bool:
    """Check whether data are the values of a dimension of a scan.

    The values of a dimension (`<dimension>.<target>`) are the same in every
    draw, a curve over them has a band across the draws.
    """
    if not isinstance(simulation, Scan):
        return False
    dimension = str(data.index).split(".", 1)[0]
    return any(d.id == dimension for d in simulation.dimensions)


def _draw_data(data: Data, run_id: str, k: int) -> Data:
    """Get the data of a draw, an id of its own (the id does not depend on `sel`)."""
    sid = Data(data.selection, task=data.task_id, sel=data.sel).sid
    return Data(
        data.selection,
        task=data.task_id,
        sel={**data.sel, run_id: k},
        sid=f"{sid}__{run_id}_{k}",
    )


def _draw_curves(curve: Curve, run: UncertaintyRun) -> list[Curve]:
    """Get a line of every draw of a curve over another variable than the time.

    x differs between the draws, so the quantiles of y have no common x: the
    lines of the draws, thin, translucent and without a legend entry, in the
    colour of the curve, give the area of the draws; their opacity falls with
    the number of draws, so that the area is denser where more draws lie.
    """
    draws = run.settings.draws
    kwargs: dict[str, Any] = {
        **(curve.kwargs or {}),
        "label": None,
        "alpha": min(0.3, 10 / draws),
        "linewidth": 0.5,
        "marker": "",
    }
    kwargs.pop("markeredgecolor", None)
    color = _curve_color(curve)
    if color is not None:
        kwargs["color"] = color
    return [
        Curve(
            x=_draw_data(curve.x, run.id, k),
            y=_draw_data(curve.y, run.id, k),
            yaxis_position=curve.yaxis_position,
            **kwargs,
        )
        for k in range(draws)
    ]


def apply_uncertainty(experiment: SimulationExperiment, run: UncertaintyRun) -> None:
    """Run an initialized experiment over the draws of the fitted parameters.

    `SimulationExperiment.initialize` applies it for the setting
    `uncertainty`. Every simulation becomes a scan over the draws (a scan gets
    the dimension of the draws appended); a parameter which the simulation
    sets itself, or which the model of the simulation does not have, is not
    drawn, and a simulation without a drawn parameter runs once. Every curve
    of a figure over the time of a task over the draws, or over the values of
    a dimension of its scan (e.g. an amount at a time over the scanned renal
    function), becomes the band of the quantiles of the draws with the
    median, with the name and the colour of the curve. A curve whose x is
    another variable of the task (e.g. an effect over a concentration) has no
    common x across the draws: it draws the simulation with the fitted
    values, which runs once next to the draws (`<simulation>_fitted`), over a
    thin translucent line of every draw in its colour (the area of the draws,
    without a legend entry; not for a curve with a line per point of a
    dimension). The curves of the data stay.

    Args:
        experiment: the initialized experiment.
        run: the draws and the settings.

    Raises:
        ValueError: if a scan of the experiment has a dimension of the id of
            the draws, or a curve of a task draws a line per point of more
            than one dimension (a band is drawn per point of one).
    """
    sid = experiment.sid
    simulations = experiment._simulations
    originals = dict(simulations)
    available: dict[str, set[str] | None] = {}
    for task in experiment._tasks.values():
        parameters = _model_parameters(experiment, task.model_id)
        known = available.get(task.simulation_id)
        if task.simulation_id not in available or known is None:
            available[task.simulation_id] = parameters
        elif parameters is not None:
            available[task.simulation_id] = known & parameters
    for key, simulation in originals.items():
        simulations[key] = _over_draws(
            simulation, run.dimension, key, sid, available.get(key)
        )
    # the tasks whose simulation runs over the draws; a simulation without a
    # drawn parameter runs once and keeps its curves
    drawn = {
        task_id
        for task_id, task in experiment._tasks.items()
        if _has_dimension(simulations[task.simulation_id], run.id)
    }
    for figure in experiment._figures.values():
        for plot in figure.get_plots():
            curves: list[Curve] = []
            # the lines of the draws come first, below every curve of the plot
            draw_curves: list[Curve] = []
            for curve in plot.curves:
                task_id = None if curve.y.task_id is None else str(curve.y.task_id)
                if task_id is None or task_id not in drawn:
                    curves.append(curve)
                    continue
                original = originals[experiment._tasks[task_id].simulation_id]
                if curve.x.index != "time" and not _over_scan_dimension(
                    curve.x, original
                ):
                    if not curve.over:
                        draw_curves.extend(_draw_curves(curve, run))
                    fitted = _fitted_task(experiment, task_id, originals)
                    curve.x = _on_task(curve.x, fitted)  # ty: ignore[invalid-assignment]
                    curve.y = _on_task(curve.y, fitted)  # ty: ignore[invalid-assignment]
                    curve.xerr = _on_task(curve.xerr, fitted)
                    curve.yerr = _on_task(curve.yerr, fitted)
                    curve.name = f"{curve.name} (fit)" if curve.name else curve.name
                    curves.append(curve)
                    continue
                if len(curve.over) > 1:
                    raise ValueError(
                        f"The curve '{curve.sid}' of the figure '{figure.sid}' of "
                        f"'{sid}' draws a line per point of {list(curve.over)}; a "
                        f"band is drawn per point of one dimension."
                    )
                plot.band(
                    x=curve.x,
                    y=curve.y,
                    across=run.id,
                    quantiles=run.quantiles,
                    over=curve.over or None,
                    name=curve.name,
                    color=_curve_color(curve),
                    yaxis_position=curve.yaxis_position,
                )
            plot.curves = draw_curves + curves


# ----------------------------------------------------------------------------
# the check of a model
# ----------------------------------------------------------------------------
def check_fit_parameters(project: Project) -> list[str]:
    """Check that the model has the fitted values and the uncertainty of the adopted fits.

    Every parameter of an adopted fit must be a parameter of every variant of
    the model (`Project.variants`) with the value of the record of the fit
    (within `VALUE_RTOL`) and a distribution of its draws. A run over the
    draws draws the parameters of the default variant only
    (`UncertaintyRun.create`), and every draw replaces the parameter in every
    variant: a variant whose value or distribution of a fitted parameter
    differs would be simulated with the draws of another value, so it is a
    problem, as is a value of the default variant other than the record's.

    Args:
        project: the project.

    Returns:
        The problems, one line each; empty for a project without adopted fits.
    """
    from sbmlsim.simulation.sampling import parameter_distributions

    try:
        if not project.adopted_fits():
            return []
        records = project.fit_records()
    except (FileNotFoundError, ValueError) as err:
        return [str(err)]
    problems: list[str] = []
    default: dict[str, Distribution] | None = None
    single = len(project.variants) == 1
    for index, (variant, path) in enumerate(project.variants.items()):
        name = "model" if single else f"variant '{variant}'"
        if not path.exists():
            problems.append(f"the {name} {path} does not exist")
            continue
        sbml_model = libsbml.readSBMLFromFile(str(path)).getModel()
        distributions = parameter_distributions(path)
        if index == 0:
            default = distributions
        where = "" if single else f" (the {name}, {path.name})"
        for pid, parameter in records.parameters.items():
            p = sbml_model.getParameter(pid) if sbml_model is not None else None
            if p is None:
                problems.append(
                    f"the parameter '{pid}' of the fit '{parameter.fit_id}' is no "
                    f"parameter of {path.name}{where}"
                )
                continue
            if not math.isclose(p.getValue(), parameter.value, rel_tol=VALUE_RTOL):
                problems.append(
                    f"the parameter '{pid}' is {p.getValue()!r} in the model, the "
                    f"fit '{parameter.fit_id}' gives {parameter.value!r}: take it "
                    f"from the record (`AdoptedRecords.parameter`){where}"
                )
            if pid not in distributions:
                problems.append(
                    f"the parameter '{pid}' of the fit '{parameter.fit_id}' has no "
                    f"distribution in the model{where}"
                )
            elif (
                default is not None
                and pid in default
                and distributions[pid] != default[pid]
            ):
                problems.append(
                    f"the distribution of the parameter '{pid}' of the fit "
                    f"'{parameter.fit_id}' differs from the default variant"
                    f"{where}: {distributions[pid]} instead of {default[pid]}"
                )
    return problems
