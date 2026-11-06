"""The checks of a project as functions: the registry, the experiments, the fits.

A model package, or a repository of model packages, calls them from its own
tests, so a package which is exported into a repository of its own keeps them
(`check_model` checks that the project belongs to its package):

```python
# not executed
@pytest.mark.parametrize("project", PROJECTS)
def test_experiments_registered(project: Project) -> None:
    project_checks.check_registry(project)
```

The known exceptions of the checks (experiments which are not registered on
purpose, modules which do not import, the reason to run the experiments one
at a time, the checks of a battery which fail) are the `checks` of the
project, `sbmlsim.project.CheckSettings`, each with its reason.

The functions raise `AssertionError` with a message which names every problem,
also for experiments which fail to initialize or to run.

`import sbmlsim` does not import this module; it imports the experiments and
the fits when a check runs.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from sbmlsim.project import Project, resolve

if TYPE_CHECKING:
    from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
    from sbmlsim.fit.cli import FitRun

__all__ = [
    "Battery",
    "BatteryResult",
    "check_battery",
    "check_model",
    "check_registry",
    "defined_experiments",
    "experiment_modules",
    "initialize_experiments",
    "registered_experiments",
    "run_all_experiments",
    "start_fits",
]


def _package(project: Project) -> str:
    """Get the package of a project.

    Raises:
        AssertionError: for a project without a package.
    """
    if project.package is None:
        raise AssertionError(
            f"The project '{project.name}' has no package, whose modules the "
            "checks import."
        )
    return project.package


def experiment_modules(project: Project) -> tuple[list[ModuleType], dict[str, str]]:
    """Import the modules of the experiments of a project.

    The module is `CheckSettings.experiment_modules` below the package of the
    project, a package whose modules are imported with it, or a module. The
    modules in `CheckSettings.not_importable` are skipped.

    Args:
        project: the project.

    Returns:
        The imported modules and the traceback of every module which does not
        import, by its name relative to the package of the project.

    Raises:
        AssertionError: for a project without a package.
    """
    prefix = f"{_package(project)}."
    skipped = project.checks.not_importable
    modules: list[ModuleType] = []
    errors: dict[str, str] = {}

    def on_error(name: str) -> None:
        errors[name.removeprefix(prefix)] = traceback.format_exc()

    name = f"{prefix}{project.checks.experiment_modules}"
    try:
        package = importlib.import_module(name)
    except Exception:  # every module which does not import is reported
        on_error(name)
        return modules, errors
    modules.append(package)
    path = getattr(package, "__path__", None)
    if path is None:
        return modules, errors
    for info in pkgutil.walk_packages(
        path, prefix=f"{package.__name__}.", onerror=on_error
    ):
        if info.name.removeprefix(prefix) in skipped:
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:  # every module which does not import is reported
            on_error(info.name)
    return modules, errors


def defined_experiments(
    modules: Sequence[ModuleType],
) -> set[type[SimulationExperiment]]:
    """Get the experiments defined in modules, without the base experiments.

    A base experiment is a class from which another of the found classes
    derives.
    """
    from sbmlsim.experiment import SimulationExperiment

    classes = {
        cls
        for module in modules
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and issubclass(cls, SimulationExperiment)
    }
    return {
        cls
        for cls in classes
        if not any(other is not cls and issubclass(other, cls) for other in classes)
    }


def registered_experiments(project: Project) -> list[type[SimulationExperiment]]:
    """Get the experiments of the group `all` of a project."""
    return project.experiment_classes()


def _module_exists(name: str) -> bool:
    """Check that a module exists without importing it (its parents are imported)."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def check_registry(project: Project) -> None:
    """Check that every experiment a project defines is one of its experiments.

    Every module of the experiments imports (`experiment_modules`), every
    experiment is in the group `all` once, and the known exceptions of
    `project.checks` are still exceptions (a listed class is defined and not
    registered, a listed module exists).

    Args:
        project: the project.

    Raises:
        AssertionError: naming every problem.
    """
    modules, errors = experiment_modules(project)
    checks = project.checks
    problems = [f"{name} does not import:\n{tb}" for name, tb in errors.items()]
    names = [cls.__name__ for cls in registered_experiments(project)]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        problems.append(f"registered twice: {twice}")
    defined = defined_experiments(modules)
    unregistered = sorted(
        cls.__name__
        for cls in defined
        if cls.__name__ not in names and cls.__name__ not in checks.not_registered
    )
    if unregistered:
        problems.append(f"defined, but not in the group 'all': {unregistered}")
    stale = sorted(set(checks.not_registered) & set(names))
    if stale:
        problems.append(f"registered, but listed as not registered: {stale}")
    missing = sorted(
        set(checks.not_registered) - {cls.__name__ for cls in defined} - set(names)
    )
    if missing:
        problems.append(f"listed as not registered, but not defined: {missing}")
    absent = sorted(
        name
        for name in checks.not_importable
        if not _module_exists(f"{project.package}.{name}")
    )
    if absent:
        problems.append(f"listed as not importable, but no module: {absent}")
    if problems:
        raise AssertionError("\n".join(problems))


def _package_problems(project: Project) -> list[str]:
    """Check that a project belongs to its package, see `check_model`."""
    package = project.package
    if package is None:
        return []
    problems: list[str] = []
    try:
        module = importlib.import_module(package)
    except ImportError as err:
        return [f"the package '{package}' does not import: {err}"]
    if not any(value is project for value in vars(module).values()):
        problems.append(f"no attribute of the package '{package}' is this project")
    file = getattr(module, "__file__", None)
    if file is None or project.root != Path(file).parent:
        problems.append(f"the root {project.root} is not the directory of {file}")
    component = package.rsplit(".", 1)[-1]
    if project.name != component and not component.endswith(f"_{project.name}"):
        problems.append(
            f"the name '{project.name}' is not the name of the package '{component}'"
        )
    return problems


def _path_problems(project: Project) -> list[str]:
    """Check the paths of a project, see `check_model`."""
    root = project.root
    paths: list[tuple[str, Path | None]] = [
        *((f"the variant '{name}'", path) for name, path in project.variants.items()),
        ("the factory directory", project.factory_dir),
        ("the directory of the fit records", project.fits_dir),
        ("the file of the adopted fits", project.adopted_file),
    ]
    problems = [
        f"{label} {path} is not below the root {root}"
        for label, path in paths
        if path is not None and not path.is_relative_to(root)
    ]
    problems += [
        f"the model of the variant '{name}' {path} does not exist"
        for name, path in project.variants.items()
        if not path.exists()
    ]
    data = project.data_path
    data_paths = [] if data is None else [data] if isinstance(data, Path) else data
    problems += [
        f"the data path {path} does not exist"
        for path in data_paths
        if not path.exists()
    ]
    return problems


def check_model(project: Project) -> None:
    """Check that a project belongs to its package and its references hold.

    A slip in `package` or `root` of a project resolves the references of
    another project, or names the paths wrongly, without an error otherwise.
    With a package, the package imports, has the project as an attribute,
    its directory is `root`, and the name of the project is the last
    component of the package (or the component ends with `_<name>`). The
    variants, the directory of the factory, the directory of the fit records
    and the file of the adopted fits lie below `root`, the variants and the
    data paths exist, the references (the factory, the experiments, the fits,
    the credibility declaration, the actions) resolve, every action takes the
    arguments of its call, the experiments load
    (`Project.load_experiments`), the fits of a reference are not empty, and
    every parameter of an adopted fit has its value and a distribution in
    every variant (`sbmlsim.experiment.draws.check_fit_parameters`).

    Args:
        project: the project.

    Raises:
        AssertionError: naming every problem.
    """
    from sbmlsim.experiment.draws import check_fit_parameters

    problems = _package_problems(project) + _path_problems(project)
    references: dict[str, Any] = {
        "factory": project.factory,
        "experiments": project.experiments,
        "fits": project.fits if isinstance(project.fits, str) else None,
        "credibility": project.credibility,
        **{
            f"action '{name}'": action.function
            for name, action in project.actions.items()
        },
    }
    resolved: dict[str, Any] = {}
    for name, reference in references.items():
        if not isinstance(reference, str):
            continue
        try:
            resolved[name] = resolve(reference, project.package)
        except Exception as err:  # reported, whatever the reference does
            problems.append(f"{name} '{reference}' does not resolve: {err!r}")
    if project.experiments is not None and (
        not isinstance(project.experiments, str) or "experiments" in resolved
    ):
        try:
            project.load_experiments()
        except (TypeError, ValueError) as err:
            problems.append(f"the experiments do not load: {err}")
    if "fits" in resolved:
        fits = resolved["fits"]
        if not isinstance(fits, Mapping) or not fits:
            problems.append("the fit definitions are not a non-empty mapping")
    for name, action in project.actions.items():
        function = resolved.get(f"action '{name}'", action.function)
        if isinstance(function, str):
            continue
        if not callable(function):
            problems.append(f"the action '{name}' is no function")
            continue
        try:
            signature = inspect.signature(function)
        except TypeError, ValueError:
            continue
        try:
            signature.bind_partial(project, project.root, workers=1)
        except TypeError as err:
            problems.append(
                f"the action '{name}' is not called as `function(project, out, *, "
                f"workers, **arguments)`: {err}"
            )
    # the values of the adopted fits and their uncertainty are in the model
    problems += check_fit_parameters(project)
    if problems:
        raise AssertionError("\n".join(problems))


def _runner(project: Project) -> ExperimentRunner:
    """Initialize the experiments of a project, as its runs do."""
    from sbmlsim.experiment import ExperimentRunner

    return ExperimentRunner(
        experiment_classes=registered_experiments(project),
        base_path=project.base_path or project.root,
        data_path=project.data_path,
        simulator=None,
        on_error="log",
        jit_optimization=project.simulator.jit_optimization,
    )


def initialize_experiments(project: Project) -> dict[str, SimulationExperiment]:
    """Initialize the experiments with their data, figures, mappings.

    Every experiment is initialized, also after one which cannot be.

    Returns:
        The experiments by name.

    Raises:
        AssertionError: naming every experiment which cannot be initialized
            with its error; the tracebacks are in the log of sbmlsim.
    """
    runner = _runner(project)
    if runner.failed:
        total = len(runner.failed) + len(runner.experiments)
        lines = [f"{sid}: {result.error}" for sid, result in runner.failed.items()]
        raise AssertionError(
            f"{len(runner.failed)} of {total} experiments cannot be initialized "
            "(the tracebacks are in the log):\n" + "\n".join(lines)
        )
    return dict(runner.experiments)


def run_all_experiments(
    project: Project, output_dir: Path, *, n_workers: int | None = 1
) -> None:
    """Run the experiments of a project, as the kind `simulate` does.

    Every experiment runs, also after one which failed, with the settings of
    the simulator and the figures of the project
    (`Project.run_experiments`, the implementation of `Project`).

    Args:
        project: the project.
        output_dir: the directory of the results. With
            `CheckSettings.one_at_a_time` every experiment runs on its own,
            into a directory named after it: the memory of a run of all
            experiments of a large model exceeds that of a runner of a
            continuous integration; else they run together into `all`, with
            their figures collected.
        n_workers: the processes of the run, serial by default, since the
            tests of a package run in processes of their own.

    Raises:
        AssertionError: naming every failed experiment and figure, and every
            run which wrote no report.
    """
    from sbmlsim.experiment import ExperimentRunError

    failures: list[str] = []
    if project.checks.one_at_a_time is not None:
        for cls in registered_experiments(project):
            path = output_dir / cls.__name__
            try:
                Project.run_experiments(project, cls, path, n_workers=n_workers)
            except ExperimentRunError as err:
                failures += err.failures
            if not list(path.glob("**/*.html")):
                failures.append(f"{cls.__name__} wrote no report")
    else:
        path = output_dir / "all"
        try:
            Project.run_experiments(
                project,
                registered_experiments(project),
                path,
                n_workers=n_workers,
                collect_figures=True,
            )
        except ExperimentRunError as err:
            failures += err.failures
        if not list(path.glob("**/*.html")):
            failures.append("the run of all experiments wrote no report")
    if failures:
        raise AssertionError("\n".join(failures))


class BatteryResult(Protocol):
    """The result of a check of a battery, see `check_battery`.

    Attributes:
        check: the check, its name as its `value` (an `Enum`) or as `str`.
        passed: whether the check passed, `None` if it does not apply.
        status: the status of the check, as its `value` (an `Enum`) or as
            `str`, e.g. `not_applicable`.
        problems: the problems of a failed check, one line each.
    """

    @property
    def check(self) -> object:
        """The check."""
        ...

    @property
    def passed(self) -> bool | None:
        """Whether the check passed, `None` if it does not apply."""
        ...

    @property
    def status(self) -> object:
        """The status of the check."""
        ...

    @property
    def problems(self) -> Sequence[str]:
        """The problems of a failed check."""
        ...


class Battery(Protocol):
    """A battery of checks of the models of a project, see `check_battery`."""

    @property
    def results(self) -> Sequence[BatteryResult]:
        """The results of the checks."""
        ...


def check_battery[B: Battery](project: Project, run: Callable[[Project], B]) -> B:
    """Run a battery of the checks of the models of a project.

    A failed check is a problem unless the project knows it
    (`CheckSettings.battery`); a known failure whose check passes or does not
    apply no longer applies and is a problem as well.

    Args:
        project: the project.
        run: runs the battery of the project, e.g. the battery of a framework
            with its settings.

    Returns:
        The battery.

    Raises:
        AssertionError: naming every problem.
    """
    battery = run(project)
    known = project.checks.battery
    problems: list[str] = []
    for result in battery.results:
        check = str(getattr(result.check, "value", result.check))
        if result.passed is False and check not in known:
            problems.append(f"the check '{check}' failed:")
            problems += [f"  {problem}" for problem in result.problems]
        elif result.passed is not False and check in known:
            status = str(getattr(result.status, "value", result.status))
            status = status.replace("_", " ")
            problems.append(
                f"the known failure of the check '{check}' no longer applies, it "
                f"{status}: remove it from `checks.battery`"
            )
    if problems:
        raise AssertionError("\n".join(problems))
    return battery


def start_fits(
    project: Project, output_dir: Path, max_nfev: int = 2
) -> dict[str, dict[str, FitRun]]:
    """Start a fit of every fit problem: one start with a few evaluations.

    Args:
        project: the project.
        output_dir: the directory of the results of the fits.
        max_nfev: evaluations of the cost per start.

    Returns:
        The runs of every fit problem by its subset.

    Raises:
        AssertionError: if a fit did not run.
    """
    from sbmlsim.fit.cli import run_fit

    runs: dict[str, dict[str, FitRun]] = {}
    # the problems load their models at the level of the project
    for subset, definition in project.load_fit_definitions().items():
        runs[subset] = run_fit(
            definition,
            opid=f"test_{subset}",
            size=1,
            seed=1234,
            output_dir=output_dir,
            max_nfev=max_nfev,
        )
        if not runs[subset]:
            raise AssertionError(f"the fit {subset} did not run")
    return runs
