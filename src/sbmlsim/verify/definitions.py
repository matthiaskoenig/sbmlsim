"""The fit definitions and the run settings of a project, `verify fits`, `definitions`, `settings`."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sbmlsim.verify.common import UsageError, load_project_of, report
from sbmlsim.verify.dump import serialize

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment
    from sbmlsim.project import Project

logger = logging.getLogger(__name__)


def fits(args: argparse.Namespace) -> int:
    """Run a minimal fit (one start, a few evaluations) of every fit definition.

    Returns:
        1 if a fit failed, else 0.

    Raises:
        UsageError: for a project without fits or an unknown subset.
    """
    from sbmlsim.fit.cli import run_fit

    project = load_project_of(args)
    definitions = project.load_fit_definitions()
    if not definitions:
        raise UsageError(f"{project.name} has no fit definitions")
    if unknown := set(args.subsets) - set(definitions):
        raise UsageError(
            f"unknown fit definitions {sorted(unknown)}, the project has "
            f"{sorted(definitions)}"
        )
    status: dict[str, str] = {}
    for key in args.subsets or list(definitions):
        try:
            # the problems load their models at the level of the project, their
            # definitions carry it (`Project.load_fit_definitions`)
            run_fit(
                definitions[key],
                opid=f"smoke_{key}",
                size=1,
                seed=1234,
                output_dir=Path(args.out),
                max_nfev=args.max_nfev,
            )
            status[key] = "ok"
        except Exception:  # every failing subset is reported
            logger.exception("The fit definition '%s' failed.", key)
            status[key] = "FAILED"
    report(status)
    return 1 if "FAILED" in status.values() else 0


def definitions(args: argparse.Namespace) -> int:
    """Write the fit definitions of a project as json.

    The settings, the parameters and the mappings of the collections of
    every fit definition, so that the definitions of two checkouts are
    compared with `diff`. The checkout directory (`--root`, else the root of
    the project) is replaced by `<root>`.

    Raises:
        UsageError: for a project without fits.
    """
    project = load_project_of(args)
    root_path = Path(args.root or project.root).resolve()
    root = str(root_path)
    fit_definitions = project.load_fit_definitions()
    if not fit_definitions:
        raise UsageError(f"{project.name} has no fit definitions")
    content: dict[str, object] = {}
    for key, definition in fit_definitions.items():
        collections = definition.mapping_collections()
        content[key] = {
            "settings": serialize(definition.settings),
            "parameters": [serialize(p) for p in definition.parameters],
            "collections": {
                experiment: [serialize(c) for c in items]
                for experiment, items in sorted(collections.items())
            },
        }
    text = json.dumps(content, indent=2, sort_keys=True, default=str)
    Path(args.out).write_text(
        text.replace(json.dumps(root)[1:-1], "<root>").replace(root, "<root>"),
        encoding="utf-8",
    )
    return 0


# --- settings -------------------------------------------------------------------


def _plain(value: object) -> object:
    """A json value of a setting: paths as posix strings, classes by name."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return serialize(value)


def _bound(
    original: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    skip: Collection[str] = (),
) -> dict[str, object]:
    """The arguments of a call bound to the signature of `original`, defaults applied.

    A positional and a keyword call of the same values give the same result.
    `self` and the names in `skip` are left out.
    """
    signature = inspect.signature(original)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments: dict[str, object] = {}
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        else:
            arguments[name] = value
    return {
        name: _plain(value)
        for name, value in sorted(arguments.items())
        if name != "self" and name not in skip
    }


def capture_settings(
    project: Project,
    experiment: type[SimulationExperiment],
    root: Path,
    run: Callable[[type[SimulationExperiment], Path], object] | None = None,
) -> dict[str, Any]:
    """Capture the settings a run of an experiment hands to sbmlsim.

    The run is intercepted where it reaches sbmlsim: the arguments of the
    simulator (`Simulator`), of the runner (without the simulator) and of the
    run (without the output path; with the figure settings of the run), bound
    to the signatures, so positional and keyword calls agree, and every
    matplotlib `rcParams` entry which differs from the value before the run, at
    the time of the run. `calls` counts the intercepted calls. The simulator is
    the one created before the runner: the designs of an experiment
    (`sbmlsim.simulation.sampling.parameters_of`, `local`) create simulators of
    their own while the runner initializes it, which are neither recorded nor
    counted. Nothing is simulated and no report is written.

    Args:
        project: the project, whose `run_experiments` is intercepted.
        experiment: the experiment class the run gets.
        root: the checkout; it is replaced by `<root>` in the paths.
        run: the run to intercept, called with the experiment and an output
            directory; `project.run_experiments` with its settings without.

    Returns:
        The settings as json values.

    Raises:
        RuntimeError: if the run does not reach the simulator, the runner and
            the run of the experiments of sbmlsim.
    """
    from unittest import mock

    import matplotlib

    from sbmlsim.experiment import ExperimentRunner
    from sbmlsim.report.experiment_report import ExperimentReport
    from sbmlsim.simulator import Simulator

    captured: dict[str, Any] = {"experiment": experiment.__name__}
    calls = dict.fromkeys(("simulator", "runner", "run"), 0)
    simulator_init = Simulator.__init__
    runner_init = ExperimentRunner.__init__
    run_original = ExperimentRunner.run_experiments
    baseline = dict(matplotlib.rcParams)

    def record_simulator(*args: object, **kwargs: object) -> None:
        # a design of an experiment creates simulators of its own while the
        # runner initializes it
        if calls["runner"] == 0:
            calls["simulator"] += 1
            captured["simulator"] = _bound(simulator_init, args, kwargs)
        simulator_init(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def record_runner(*args: object, **kwargs: object) -> None:
        calls["runner"] += 1
        # the simulator is captured on its own; as an object it carries memory
        # addresses, which differ between any two runs
        captured["runner"] = _bound(runner_init, args, kwargs, skip={"simulator"})
        runner_init(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def record_run(*args: object, **kwargs: object) -> list[object]:
        calls["run"] += 1
        captured["run"] = _bound(run_original, args, kwargs, skip={"output_path"})
        captured["rc"] = {
            key: _plain(value)
            for key, value in sorted(dict(matplotlib.rcParams).items())
            if key not in baseline or bool(value != baseline[key])
        }
        return []

    def no_report(
        self: object, output_path: Path, *args: object, **kwargs: object
    ) -> Path:
        return output_path

    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(Simulator, "__init__", record_simulator),
        mock.patch.object(ExperimentRunner, "__init__", record_runner),
        mock.patch.object(ExperimentRunner, "run_experiments", record_run),
        mock.patch.object(ExperimentReport, "create_report", no_report),
    ):
        output = Path(tmp) / "settings"
        if run is None:
            project.run_experiments(experiment, output)
        else:
            run(experiment, output)
    missing = [
        key for key in ("simulator", "runner", "run", "rc") if key not in captured
    ]
    if missing:
        raise RuntimeError(
            f"the run did not reach sbmlsim, nothing captured for {missing}"
        )
    captured["calls"] = calls
    text = json.dumps(captured, sort_keys=True, default=str)
    for path in (root.resolve(), root.absolute()):
        for form in (path.as_posix(), str(path), json.dumps(str(path))[1:-1]):
            text = text.replace(form, "<root>")
    return json.loads(text)


def settings(args: argparse.Namespace) -> int:
    """Write the settings of a run of the experiments of a project as json.

    Raises:
        UsageError: for a project without experiments, an unknown experiment
            or a run which does not reach sbmlsim.
    """
    project = load_project_of(args)
    try:
        classes = project.experiment_classes()
    except (ValueError, TypeError) as err:
        raise UsageError(str(err)) from err
    if args.experiment:
        experiment = {c.__name__: c for c in classes}.get(args.experiment)
    else:
        experiment = next(iter(classes), None)
    if experiment is None:
        raise UsageError(
            f"no experiment {args.experiment or ''} in the experiments of "
            f"{project.name}"
        )
    try:
        captured = capture_settings(
            project, experiment, Path(args.root or project.root).absolute()
        )
    except RuntimeError as err:
        raise UsageError(f"{project.name}: {err}") from err
    Path(args.out).write_text(
        json.dumps(captured, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0
