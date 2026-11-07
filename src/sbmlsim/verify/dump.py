"""Run the experiments of a project and store what they produce, `verify dump`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import traceback
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import numpy as np

from sbmlsim.verify.common import UsageError, load_project_of, report

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment
    from sbmlsim.fit import FitData, FitMapping
    from sbmlsim.plot import Figure
    from sbmlsim.project import Project
    from sbmlsim.simulation import Scan, Simulation

__all__ = [
    "RESOLVED",
    "TIME_DIMENSIONS",
    "Dump",
    "defined_experiments",
    "describe_mapping",
    "dump",
    "figure_spec",
    "load",
    "mapping_range",
    "metadata_names",
    "on_grid",
    "package_versions",
    "serialize",
    "store_experiment",
    "variant_of",
]

logger = logging.getLogger(__name__)

#: dimensions along which the simulations of a task are compared: the time of a
#: grid and the points of a ragged timecourse
TIME_DIMENSIONS = ("time", "_point")

#: the key of the specification of a figure with the settings of its experiment
#: and its own and the size it is rendered with, without the settings of a run,
#: see `figure_spec`
RESOLVED = "__resolved__"


class Dump(TypedDict):
    """Everything `dump` stores of the experiments of a model."""

    #: the name of the project
    project: str
    #: names of the experiment classes in the order they were run
    order: list[str]
    #: traceback per experiment (or `import:<module>`) which failed
    errors: dict[str, str]
    #: (experiment, task, variable) -> (dimensions, values)
    results: dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]]
    #: (experiment, dataset, column) -> values
    datasets: dict[tuple[str, str, str], np.ndarray]
    #: experiment -> description of every fit mapping
    mappings: dict[str, list[str]]
    #: experiment -> figure -> specification of the figure (json)
    figures: dict[str, dict[str, str]]
    #: experiment -> fit mapping -> (data min, data max, simulation min, simulation
    #: max, unit) of the x values, or the error message
    ranges: dict[str, dict[str, tuple[float, float, float, float, str] | str]]
    #: the versions of the packages which made the dump
    versions: NotRequired[dict[str, str]]
    #: experiment -> task -> the variant of the model the task ran on
    #: (`variant_of`), of the tasks which run on one
    builds: NotRequired[dict[str, dict[str, str]]]
    #: the intervals of the common output grid of every simulation (`--grid`),
    #: missing for the steps of the integrator
    grid: NotRequired[int]


# --- dump -----------------------------------------------------------------------


def serialize(obj: object, depth: int = 0) -> object:
    """Serialize the attributes of an object (a figure) into json values.

    References back to the experiment and to parents are replaced by their type,
    so that the specification of a figure only contains the figure.
    """
    from sbmlsim.experiment import SimulationExperiment

    if depth > 12:
        return "<depth>"
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Enum):
        return f"{type(obj).__name__}.{obj.name}"
    if isinstance(obj, SimulationExperiment):
        return f"<experiment {type(obj).__name__}>"
    if isinstance(obj, dict):
        return {str(k): serialize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [serialize(v, depth + 1) for v in obj]
    if isinstance(obj, type):
        return f"<class {obj.__qualname__}>"
    if hasattr(obj, "__dict__"):
        skip = {"experiment", "_experiment", "figure", "_figure", "parent", "_parent"}
        return {"__type__": type(obj).__name__} | {
            k: f"<{type(v).__name__}>" if k in skip else serialize(v, depth + 1)
            for k, v in vars(obj).items()
        }
    return repr(obj)


def metadata_names(metadata: object) -> dict[str, str]:
    """Get the metadata of a fit mapping as {field: name}.

    Enum members are stored by name, so a change of their values or of the enum
    type (`StrEnum`) does not count as a change.
    """
    if metadata is None:
        return {}
    fields = getattr(metadata, "__dataclass_fields__", None)
    items = (
        {k: getattr(metadata, k) for k in fields} if fields else dict(vars(metadata))
    )
    return {
        k: v.name if isinstance(v, Enum) else str(v)
        for k, v in items.items()
        if not k.startswith("_")
    }


def describe_mapping(key: str, mapping: FitMapping) -> str:
    """Describe a fit mapping by its data, its observable, weight and metadata."""

    def fit_data(d: FitData) -> str:
        # the data of a mapping without x values have no x
        x = d.x.index if d.x else None
        sd = d.y_sd.index if d.y_sd else None
        se = d.y_se.index if d.y_se else None
        return f"{d.dset_id}|{d.task_id}|{x}|{d.y.index}|{d.count}|{sd}|{se}"

    metadata = sorted(metadata_names(mapping.metadata).items())
    try:
        weight = mapping.weight
    except ValueError:  # neither a weight nor a count: the collection weights it
        weight = None
    return (
        f"{key}|{fit_data(mapping.reference)}|{fit_data(mapping.observable)}"
        f"|weight={weight}|{metadata}"
    )


def mapping_range(
    mapping: FitMapping,
) -> tuple[float, float, float, float, str] | str:
    """Get the range of the x values of the data and of the simulation of a mapping.

    The data are converted into the unit of the simulation.
    """
    try:
        data = mapping.reference.get_data().x
        simulation = mapping.observable.get_data().x
        if data is None or simulation is None:
            return "no x values"
        x = np.asarray(data.to(simulation.units).magnitude, dtype=float)
        s = np.asarray(simulation.magnitude, dtype=float)
        return (
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(s)),
            float(np.nanmax(s)),
            str(simulation.units),
        )
    except Exception as err:  # reported per mapping
        return f"{type(err).__name__}: {err}"


def _time_number(time: object, time_unit: str | None) -> float:
    """Get a time of a simulation as a number in its time unit.

    Raises:
        ValueError: for a quantity of a simulation without a time unit, whose
            unit is that of the model, which the grid does not know.
    """
    if getattr(time, "magnitude", None) is None:
        return float(time)  # ty: ignore[invalid-argument-type]
    if not time_unit:
        raise ValueError(
            f"the time {time} is a quantity and the simulation has no time unit"
        )
    return float(time.m_as(time_unit))  # ty: ignore[unresolved-attribute]


def on_grid(simulation: Simulation | Scan, n: int) -> Simulation | Scan:
    """Get a simulation with a common fixed output grid (`dump --grid`).

    The output times are `n + 1` equidistant times from the start to the end
    of the simulation and the times of its changes (and of the dimensions of a
    scan with a time), in the time unit of the simulation; the steps or times
    of the simulation are replaced, everything else stays. The output of the
    integrator differs between two runs whose model or tolerances differ;
    on this grid both dumps of a compare have the same times, which are
    matched exactly, without interpolation, so a compare resolves its
    threshold (a migration compare dumps both models with `--grid`).

    A scan has its simulation on the grid and every simulation of a dimension
    of simulations, each on the grid of its own start, end and changes.

    Args:
        simulation: the simulation or the scan.
        n: the number of intervals of the grid.

    Returns:
        The simulation or the scan on the grid, a new object.

    Raises:
        ValueError: for an object which is no simulation or scan, or a time
            which is a quantity of a simulation without a time unit.
    """
    from sbmlsim.simulation import Scan, Simulation
    from sbmlsim.simulation.scan import DimensionKind

    if isinstance(simulation, Scan):
        at = [d.at for d in simulation.dimensions if d.at is not None]
        dimensions = [
            dataclasses.replace(
                d,
                simulations={
                    label: _on_grid(sim, n, at) for label, sim in d.simulations.items()
                },
            )
            if d.kind == DimensionKind.SIMULATIONS
            else d
            for d in simulation.dimensions
        ]
        return Scan(
            simulation=_on_grid(simulation.simulation, n, at), dimensions=dimensions
        )
    if not isinstance(simulation, Simulation):
        raise ValueError(f"{simulation!r} is no simulation or scan")
    return _on_grid(simulation, n, [])


def _on_grid(simulation: Simulation, n: int, at: Sequence[object]) -> Simulation:
    """Put a simulation on a grid, see `on_grid`."""
    from sbmlsim.simulation import Simulation

    unit = simulation.time_unit
    start = _time_number(simulation.start, unit)
    end = _time_number(simulation.end, unit)
    changes = [t for change in simulation.changes for t in change.times]
    times = {float(t) for t in np.linspace(start, end, n + 1)}
    times |= {
        t for t in (_time_number(x, unit) for x in [*changes, *at]) if start <= t <= end
    }
    return Simulation(
        time_unit=unit,
        start=simulation.start,
        end=simulation.end,
        preinit_changes={
            **simulation.preinit_changes,
            **simulation.uncertain_preinit,
        },
        changes=list(simulation.changes),
        presimulation=simulation.presimulation,
        times=sorted(times),
        time_shift=simulation.time_shift,
    )


def defined_experiments(
    project: Project, errors: dict[str, str]
) -> list[type[SimulationExperiment]]:
    """Get the experiments defined in the modules of the experiments of a project.

    Every subclass of `SimulationExperiment` defined in the modules of the
    project (`CheckSettings.experiment_modules` below its package) counts,
    except the classes from which another found class derives (the base
    experiments), see `sbmlsim.project_checks.defined_experiments`.

    Args:
        project: the project.
        errors: the traceback of every module which does not import is added
            here, as `import:<module>`.

    Returns:
        The experiment classes in the order of module name and definition.
    """
    import inspect

    from sbmlsim import project_checks

    modules, import_errors = project_checks.experiment_modules(project)
    errors.update({f"import:{name}": text for name, text in import_errors.items()})
    classes = project_checks.defined_experiments(modules)

    def position(cls: type[SimulationExperiment]) -> tuple[str, int]:
        try:
            line = inspect.getsourcelines(cls)[1]
        except OSError, TypeError:
            line = 0
        return cls.__module__, line

    return sorted(classes, key=position)


def variant_of(
    project: Project, experiment: SimulationExperiment, task: str
) -> str | None:
    """Get the variant of the model a task of an experiment runs on.

    The experiment decides (`SimulationExperiment.variant_of`, the hook of a
    framework which builds a model per group of subjects); else the variant of
    the project whose file is the source of the model of the task; `None` for a
    model which is none of the variants.

    Args:
        project: the project.
        experiment: the experiment, initialized.
        task: the key of the task.

    Returns:
        The name of the variant, or `None`.
    """
    chosen = experiment.variant_of(experiment._tasks[task].simulation_id)
    if chosen is not None:
        if str(chosen) not in project.variants:
            logger.warning(
                "The variant '%s' of the task '%s' of '%s' is no variant of the "
                "project '%s': %s.",
                chosen,
                task,
                experiment.sid,
                project.name,
                sorted(project.variants),
            )
        return str(chosen)
    model = experiment._models.get(experiment._tasks[task].model_id)
    path = getattr(getattr(model, "source", None), "path", None)
    if path is None:
        return None
    for name in project.variants:
        if project.variant_path(name).resolve() == Path(path).resolve():
            return name
    return None


def package_versions(project: Project) -> dict[str, str]:
    """Get the versions of the packages which make a dump.

    sbmlsim, the distribution of the package of the project and, for a project
    with release settings, the libraries of the release (`core_libraries`) which
    are installed.

    Args:
        project: the project.

    Returns:
        The version by name of the distribution.
    """
    from sbmlsim.fit.provenance import project_versions

    libraries = project.release.core_libraries if project.release is not None else []
    return project_versions(project, libraries)


def _tolerance(
    name: str, text: str | None, project: Project, option: str
) -> float | object | None:
    """Get a tolerance of `dump`: a number, `model` for that of the project, or none.

    Raises:
        UsageError: for a text which is neither, or a project without it.
    """
    if text is None:
        return None
    if text == "model":
        value = getattr(project.simulator, f"{name}_tolerance")
        if value is None:
            raise UsageError(f"{project.name} has no {name} tolerance of its own")
        return value
    try:
        return float(text)
    except ValueError:
        raise UsageError(f"{option} is a number or `model`, not {text}") from None


def dump(args: argparse.Namespace) -> int:
    """Run the experiments of a project and store everything in a pickle.

    Returns:
        1 if an experiment failed, else 0.

    Raises:
        UsageError: for an unknown experiment or tolerance, or a project
            without experiments.
    """
    from sbmlsim.experiment import ExperimentRunner
    from sbmlsim.simulator import Simulator

    project = load_project_of(args)
    errors: dict[str, str] = {}
    try:
        classes = project.experiment_classes()
    except (ValueError, TypeError) as err:
        raise UsageError(str(err)) from err
    if args.all:
        classes += [c for c in defined_experiments(project, errors) if c not in classes]
    if args.order:
        with Path(args.order).open("rb") as f:
            rank = {name: k for k, name in enumerate(pickle.load(f)["order"])}
        classes.sort(key=lambda c: rank.get(c.__name__, len(rank)))
    if args.experiments:
        unknown = set(args.experiments) - {c.__name__ for c in classes}
        if unknown:
            raise UsageError(f"unknown experiments: {sorted(unknown)}")
        classes = [c for c in classes if c.__name__ in args.experiments]
    classes = [c for c in classes if c.__name__ not in args.skip]
    logger.info("%d experiments", len(classes))

    tolerances: dict[str, Any] = {}
    for name, text in (("absolute", args.atol), ("relative", args.rtol)):
        value = _tolerance(name, text, project, f"--{name[0]}tol")
        if value is not None:
            tolerances[f"{name}_tolerance"] = value

    versions = package_versions(project)
    result = Dump(
        project=project.name,
        order=[c.__name__ for c in classes],
        errors=errors,
        results={},
        datasets={},
        mappings={},
        figures={},
        ranges={},
        versions=versions,
        builds={},
    )
    if args.grid is not None:
        result["grid"] = args.grid
    for cls in classes:
        try:
            runner = ExperimentRunner(
                experiment_classes=[cls],
                data_path=project.data_path,
                base_path=project.base_path or project.root,
                # a dump runs serially, the order of its experiments is the
                # order of the run
                simulator=Simulator(n_workers=1, **tolerances),
                # an experiment which cannot be initialized raises and is
                # recorded below
                on_error="raise",
                # the models compiled at the level of the project, as its runs do
                jit_optimization=project.simulator.jit_optimization,
            )
            for sid, experiment in runner.experiments.items():
                if args.grid is not None:
                    # a common output grid instead of the steps of the integrator
                    gridded = {}
                    for key, simulation in experiment._simulations.items():
                        try:
                            gridded[key] = on_grid(simulation, args.grid)
                        except ValueError as err:
                            raise ValueError(
                                f"The simulation '{key}' of the experiment '{sid}' "
                                f"cannot be put on the grid of --grid: {err}."
                            ) from err
                    experiment._simulations = gridded
                experiment.run(
                    runner.simulator,
                    output_path=None,
                    reduced_selections=args.reduced_selections,
                )
                store_experiment(result, sid, experiment, project)
        except Exception:  # every error of an experiment is reported
            result["errors"][cls.__name__] = traceback.format_exc()
            logger.error("%s failed\n%s", cls.__name__, result["errors"][cls.__name__])
            continue
        logger.info("%s: ok", cls.__name__)

    with Path(args.out).open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    report(
        f"{len(classes)} experiments, {len(result['results'])} arrays, "
        f"{len(result['errors'])} errors -> {args.out}"
    )
    return 1 if result["errors"] else 0


def store_experiment(
    result: Dump, sid: str, experiment: SimulationExperiment, project: Project
) -> None:
    """Store the results, datasets, fit mappings and figures of a run experiment.

    The time of a grid is a coordinate of a result; it is stored as the
    variable `time` over `("time",)`, which the comparison reads. The variant of
    the build of every task is stored (`builds`, see `variant_of`), so that a
    study which moves to the build of another variant is a difference of its
    own.
    """
    builds = {key: variant_of(project, experiment, key) for key in experiment._tasks}
    if any(variant is not None for variant in builds.values()):
        result.setdefault("builds", {})[sid] = {
            key: variant for key, variant in builds.items() if variant is not None
        }
    for task, res in experiment.results.items():
        ds = res.ds
        for var in ds.data_vars:
            result["results"][sid, task, str(var)] = (
                tuple(str(d) for d in ds[var].dims),
                np.asarray(ds[var].values),
            )
        if "time" in ds.coords and "time" not in ds.data_vars:
            result["results"][sid, task, "time"] = (
                ("time",),
                np.asarray(ds["time"].values),
            )
    for key, dset in experiment.datasets().items():
        for column in dset.columns:
            values = np.asarray(dset[column].values)
            if values.dtype.kind not in "fiub":
                values = values.astype(str)
            result["datasets"][sid, key, str(column)] = values
    mappings = experiment.fit_mappings()
    result["mappings"][sid] = sorted(
        describe_mapping(k, m) for k, m in mappings.items()
    )
    result["ranges"][sid] = {k: mapping_range(m) for k, m in mappings.items()}
    result["figures"][sid] = {
        fid: figure_spec(fig, experiment) for fid, fig in experiment.figures().items()
    }


def figure_spec(figure: Figure, experiment: SimulationExperiment) -> str:
    """Get the specification of a figure, its attributes as json.

    `RESOLVED` adds the settings of the experiment and of the figure (the
    `figure_settings` of the experiment overridden by the `settings` of the
    figure) and the width and height the figure is rendered with, as sbmlsim
    resolves them without the settings of a run, so that `compare` sees a change
    of a setting of the experiment or of the size of the panels.
    """
    spec = cast("dict[str, object]", serialize(figure))
    width, height = figure.size(figure.resolved_settings(experiment))
    spec[RESOLVED] = {
        "settings": serialize({**experiment.figure_settings, **figure.settings}),
        "width": width,
        "height": height,
    }
    return json.dumps(spec, sort_keys=True, default=repr)


# --- compare --------------------------------------------------------------------


def load(path: str) -> Dump:
    """Load a dump."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
