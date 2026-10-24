"""Runner for SimulationExperiments.

The ExperimentRunner is used to execute simulation experiments.
This includes
- loading of datasets
- loading of models
- running tasks (simulation on models)
- creating outputs

The experiments of a run are independent of each other, a run of several
experiments runs them in a pool of worker processes (`n_workers` of
`ExperimentRunner.run_experiments`): every worker initializes an experiment
again from its class, loads its models once and keeps them for the next
experiment, runs it serially (its scans included, a worker starts no pool) and
writes its outputs; the parent keeps its own initialized experiments, which
the report is created from, and logs the messages of the workers in the order
of the experiments. A run is serial when an experiment cannot be sent to a
worker, see `ExperimentRunner._serial_reason`.
"""

import copy
import dataclasses
import inspect
import logging
import math
import os
import pickle
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Literal, cast, override

import matplotlib

from sbmlsim import parallel
from sbmlsim.console import console
from sbmlsim.experiment.experiment import (
    DATASETS_DIR,
    STATIC_FORMAT,
    ExperimentResult,
    ExperimentRunError,
    SimulationExperiment,
)
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.model_roadrunner import (
    JitOptimization,
    _is_hierarchical,
    resolve_jit_optimization,
)
from sbmlsim.plot import FigureSettings
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulation import Scan
from sbmlsim.simulator import Simulator
from sbmlsim.units import UnitRegistry
from sbmlsim.units import ureg as package_ureg
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)

#: the experiments per worker of a pool of `run_experiments` for
#: `n_workers=None`: a worker imports sbmlsim and the modules of the
#: experiments and loads their models before its first experiment, about 2 s,
#: which it wins back over a few experiments. The 28 experiments of the HCTZ
#: model of pkdb_models take 20.8 s serially, 12.9 s on 4 workers, 11.9 s on
#: 8 and 12.8 s on 20, where the most expensive experiment is most of the time
#: and every worker keeps 330 MB.
EXPERIMENTS_PER_WORKER: int = 4

#: the cost of a figure in simulations, which orders the experiments of a pool
#: by their cost, see `_cost`: drawing and writing a figure takes about as
#: long as 30 simulations of a model of a whole body
FIGURE_COST: int = 30


#: the model of an experiment, i.e. what decides whether two experiments can
#: share one loaded roadrunner instance
ModelKey = tuple[
    tuple[str, str | None],
    str,
    tuple[tuple[str, str], ...],
    tuple[tuple[str, float], ...],
    tuple[str, ...] | None,
]


def model_key(abstract_model: AbstractModel) -> ModelKey:
    """Get the key a loaded model is cached under.

    Two experiments share a loaded model when they resolve to the same source,
    the same language, the same changes, the same added parameters and the
    same selections; the changes and the selections are part of it because
    they are applied to the roadrunner instance when it is created. The source
    of a file is its resolved path, the one of inline SBML its content and, for
    a hierarchical model, the directory its external models are resolved
    against, see `RoadrunnerSBMLModel`. The key is
    the description of the model and not the object: an experiment builds its
    own `AbstractModel`, which has no equality of its own, so caching on the
    object would never find anything and every experiment would load and
    compile the model again.

    Args:
        abstract_model: the model of an experiment.

    Returns:
        The key, hashable and equal for two descriptions of the same model.
    """
    changes = tuple(
        sorted((key, str(value)) for key, value in abstract_model.changes.items())
    )
    source = abstract_model.source
    identity: tuple[str, str | None]
    if source.content is None:
        identity = (str(source.path), None)
    elif _is_hierarchical(source.content):
        directory = (abstract_model.base_path or Path.cwd()).resolve()
        identity = (source.content, str(directory))
    else:
        identity = (source.content, None)
    selections = abstract_model.selections
    return (
        identity,
        str(abstract_model.language_type),
        changes,
        tuple(sorted(abstract_model.parameters.items())),
        tuple(selections) if selections is not None else None,
    )


class ExperimentRunner:
    """Class for running simulation experiments."""

    def __init__(
        self,
        experiment_classes: type[SimulationExperiment]
        | Iterable[type[SimulationExperiment]],
        base_path: Path | None,
        data_path: Path | Iterable[Path] | None,
        simulator: Simulator | None = None,
        ureg: UnitRegistry | None = None,  # FIXME: is this needed on ExperimentRunner?
        on_error: Literal["raise", "log"] = "log",
        jit_optimization: JitOptimization | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the runner.

        Args:
            experiment_classes: the simulation experiments.
            base_path: base path of the simulation experiments.
            data_path: path or paths of the datasets of the simulation experiments.
            simulator: the simulator of the tasks.
            ureg: the unit registry of the experiments.
            on_error: what an error in the definition of an experiment does (its
                models, datasets, simulations, tasks, data, figures, fit mappings
                and their checks). `"log"` (the default) logs it with its
                traceback and goes on with the next experiment: the experiment
                is in `failed` and not in `experiments`, and `run_experiments`
                reports it as failed without running it. `"raise"` raises it,
                which is what a fit uses.
            jit_optimization: the optimization level every model is compiled
                with, see `RoadrunnerSBMLModel`.
            **kwargs: settings of the experiments, `SimulationExperiment.settings`.
        """
        # single UnitRegistry per runner
        if not ureg:
            ureg = package_ureg
        self.ureg = ureg

        # initialize experiments
        self.base_path = base_path
        self.data_path = data_path
        self.experiments: dict[str, SimulationExperiment] = {}
        #: the experiments whose definition raised, by sid, as the result of a
        #: failed experiment without an output path
        self.failed: dict[str, ExperimentResult] = {}
        self.on_error: Literal["raise", "log"] = on_error
        self.jit_optimization: JitOptimization = resolve_jit_optimization(
            jit_optimization
        )
        self.models: dict[ModelKey, RoadrunnerSBMLModel] = {}
        self.simulator: Simulator | None = None
        # the sids of the experiments in the order of their classes
        self._order: list[str] = []
        # the settings every experiment was created with, which a worker
        # creates it with again
        self._settings: dict[str, dict[str, Any]] = {}

        classes: list[type[SimulationExperiment]] = (
            list(experiment_classes)
            if isinstance(experiment_classes, (list, tuple, set))
            else [cast(type[SimulationExperiment], experiment_classes)]
        )
        self.initialize(classes, **kwargs)
        self.set_simulator(simulator)

    def set_simulator(self, simulator: Simulator | None) -> None:
        """Set simulator on the runner and experiments."""
        if simulator is None:
            logger.debug(
                "No simulator set in ExperimentRunner. This warning can be "
                "ignored in parameter fitting."
            )
        else:
            self.simulator = simulator
            for experiment in self.experiments.values():
                experiment.simulator = simulator

    def initialize(
        self,
        experiment_classes: list[type[SimulationExperiment]]
        | tuple[type[SimulationExperiment]]
        | set[type[SimulationExperiment]],
        **kwargs,
    ):
        """Initialize ExperimentRunner.

        Initialization is required in addition to construction to allow serialization
        of information for parallelization.
        """
        if not isinstance(experiment_classes, (list, tuple, set)):
            experiment_classes = [experiment_classes]

        for exp_class in experiment_classes:
            if not isinstance(exp_class, type):
                raise ValueError(
                    f"All 'experiment_classes' must be a class definition deriving "
                    f"from 'SimulationExperiment', but '{exp_class}' is "
                    f"'{type(exp_class)}'."
                )

            logger.debug("Initialize SimulationExperiment: %s", exp_class.__name__)
            experiment: SimulationExperiment = exp_class(
                base_path=self.base_path,
                data_path=self.data_path,
                ureg=self.ureg,
                **kwargs,
            )

            sid = experiment.sid
            if sid not in self._order:
                self._order.append(sid)
            self._settings[sid] = kwargs
            # an experiment of the same sid replaces an earlier one
            self.experiments.pop(sid, None)
            self.failed.pop(sid, None)
            try:
                self._initialize_experiment(experiment)
            except Exception as err:
                if self.on_error == "raise":
                    raise
                logger.exception(
                    "The experiment '%s' cannot be initialized and is not run", sid
                )
                self.failed[sid] = ExperimentResult(
                    experiment=experiment,
                    output_path=None,
                    error=f"{type(err).__name__}: {err}",
                )
                continue
            self.experiments[sid] = experiment

    def _initialize_experiment(self, experiment: SimulationExperiment) -> None:
        """Resolve the models of an experiment and initialize it.

        The models are loaded once per runner, see `model_key`.
        """
        _models = {}
        for model_id, source in experiment.models().items():
            abstract_model = (
                source
                if isinstance(source, AbstractModel)
                else AbstractModel(source=source)
            )
            key = model_key(abstract_model)
            if key not in self.models:
                # not cached yet, cache the model for lookup
                self.models[key] = RoadrunnerSBMLModel.from_abstract_model(
                    abstract_model=abstract_model,
                    ureg=self.ureg,
                    jit_optimization=self.jit_optimization,
                )
            _models[model_id] = self.models[key]

        # set resolved models in experiment
        experiment._models = _models
        # only after model loading the unit registry is filled
        experiment.initialize()

    @timeit
    def run_experiments(
        self,
        output_path: Path,
        show_figures: bool = False,
        save_results: bool = False,
        figure_formats: list[str] | None = None,
        reduced_selections: bool = True,
        keep_results: bool = False,
        raise_on_failure: bool = False,
        figure_settings: FigureSettings | Mapping[str, Any] | None = None,
        n_workers: int | None = None,
    ) -> list[ExperimentResult]:
        """Run the experiments and write their outputs.

        Args:
            output_path: directory of the outputs, one directory per experiment.
            show_figures: show the matplotlib figures.
            save_results: write the results of the tasks.
            figure_formats: formats of the figures.
            reduced_selections: simulate only the selections an experiment uses.
            keep_results: keep the results of every experiment after its outputs
                are written. By default they are released, so a run holds only
                the results of the experiment it runs and not of all experiments
                until its end; `SimulationExperiment.results` then raises.
            raise_on_failure: raise an `ExperimentRunError` after every
                experiment ran when an experiment or a figure failed. By default
                the failures are logged and recorded in the results
                (`ExperimentResult.failed`) only.
            figure_settings: the settings of the figures of every experiment,
                a `FigureSettings` or overrides of its defaults, which the
                `figure_settings` of an experiment and the settings of a figure
                override.
            n_workers: the processes which run the experiments: `1` runs them
                in the calling process, a number in a pool of that many
                workers (at most one per experiment), `None` in a pool of a
                worker per `EXPERIMENTS_PER_WORKER` experiments, at most one
                per CPU, i.e. serially for a few experiments, and serially in a
                worker. In a pool every experiment
                runs serially in its worker, its scans included; a run with
                `show_figures`, `keep_results`, a unit registry of its own or
                an experiment which cannot be pickled is serial, see the
                module. The pool starts worker processes, a script which runs
                it needs the guard `if __name__ == "__main__":`.

        Returns:
            The results of the experiments, which the report is created from.

        Raises:
            ExperimentRunError: with `raise_on_failure`, if something failed.
            ValueError: if a key of `figure_settings` is no setting or
                `n_workers` is less than 1, before any experiment runs.
        """
        parallel.check_workers(n_workers)
        settings = (
            FigureSettings.resolve(run=figure_settings)
            if figure_settings is not None
            else None
        )
        if settings == FigureSettings():
            # the run sets nothing, e.g. `{}`: an overridden `run` which does
            # not take the settings misses nothing
            settings = None
        if not output_path.exists():
            output_path.mkdir(parents=True)

        formats = figure_formats if figure_formats is not None else [STATIC_FORMAT]
        options: dict[str, Any] = {
            "show_figures": show_figures,
            "save_results": save_results,
            "reduced_selections": reduced_selections,
            "keep_results": keep_results,
            "figure_settings": settings,
        }
        jobs = {
            sid: _Job(
                experiment_class=type(self.experiments[sid]),
                settings=self._settings[sid],
                base_path=self.base_path,
                data_path=self.data_path,
                jit_optimization=self.jit_optimization,
                on_error=self.on_error,
                simulator=self.simulator,
                output_path=output_path / sid,
                formats=formats,
                figure_formats=figure_formats,
                options=options,
                rc=_changed_rc_params(),
                levels=_logger_levels(),
            )
            for sid in self._order
            if sid in self.experiments
        }
        workers = self._workers(n_workers, jobs, options)
        executor = (
            parallel.start_pool(workers, preload=_modules(jobs.values()))
            if workers > 1
            else None
        )
        try:
            # the most expensive experiments first, so that none of them
            # starts last and keeps the others waiting
            futures: dict[str, Future[_Outcome]] = (
                {
                    sid: executor.submit(_run_job, jobs[sid])
                    for sid in sorted(
                        jobs, key=lambda sid: -_cost(self.experiments[sid])
                    )
                }
                if executor is not None
                else {}
            )
            exp_results = []
            for sid in self._order:
                console.rule(style="white")
                if sid in self.failed:
                    # the definition raised, the experiment is reported, not run
                    exp_results.append(
                        self._failed_result(
                            self.failed[sid], output_path / sid, formats
                        )
                    )
                    continue
                logger.info("Running SimulationExperiment: '%s'", sid)
                # ExperimentResult used to create report; an error of the
                # experiment or of a figure is logged and recorded in it
                outcome = self._outcome(futures.get(sid))
                exp_results.append(
                    self._run_experiment(
                        sid,
                        output_path / sid,
                        formats,
                        figure_formats=figure_formats,
                        **options,
                    )
                    if outcome is None
                    else outcome.result(self.experiments[sid], output_path / sid)
                )
        finally:
            if executor is not None:
                parallel.stop(executor)
        self._log_summary(exp_results)
        if raise_on_failure and any(r.failed for r in exp_results):
            raise ExperimentRunError(exp_results)
        return exp_results

    def _workers(
        self, n_workers: int | None, jobs: dict[str, "_Job"], options: dict[str, Any]
    ) -> int:
        """Get the number of processes which run the experiments, see `run_experiments`."""
        if n_workers is None:
            if parallel.in_worker():
                return 1
            n_workers = min(
                os.process_cpu_count() or 1,
                math.ceil(len(jobs) / EXPERIMENTS_PER_WORKER),
            )
        n_workers = min(n_workers, len(jobs))
        if n_workers <= 1:
            return 1
        reason = self._serial_reason(jobs, options)
        if reason is not None:
            logger.info("The experiments run in this process: %s", reason)
            return 1
        return n_workers

    def _serial_reason(
        self, jobs: dict[str, "_Job"], options: dict[str, Any]
    ) -> str | None:
        """Get why the experiments cannot run in workers, `None` if they can.

        A worker creates an experiment from its class, which pickle sends by
        its module and name: a class defined in a function or at runtime, or
        in the main module of an interactive session, which a worker cannot
        import, is not found. The figures shown and the results kept are in
        the worker, and a worker has the unit registry of the package.
        """
        if parallel.in_worker():
            return "a worker starts no pool"
        if options["show_figures"]:
            return "the figures are shown"
        if options["keep_results"]:
            return "the results of the experiments are kept"
        if self.ureg is not package_ureg:
            return "the runner has a unit registry of its own"
        main = sys.modules.get("__main__")
        for job in jobs.values():
            cls = job.experiment_class
            if cls.__module__ == "__main__" and not hasattr(main, "__file__"):
                return f"'{cls.__name__}' is defined in an interactive session"
            try:
                pickle.dumps(job)
            except Exception as err:
                return f"'{cls.__name__}' cannot be sent to a worker: {err}"
        return None

    @staticmethod
    def _outcome(future: "Future[_Outcome] | None") -> "_Outcome | None":
        """Get the outcome of an experiment in a worker and log its messages.

        `None` without a worker or when the worker died, e.g. by a crash of
        roadrunner; the experiment then runs in this process.
        """
        if future is None:
            return None
        try:
            outcome = future.result()
        except BrokenProcessPool:
            logger.error(
                "The worker of the experiment died, it runs in this process again"
            )
            return None
        for record in outcome.records:
            logging.getLogger(record.name).handle(record)
        return outcome

    def _run_experiment(
        self,
        sid: str,
        output_path: Path,
        formats: list[str],
        *,
        figure_formats: list[str] | None,
        **options: Any,
    ) -> ExperimentResult:
        """Run an experiment, an error which escapes its `run` is recorded.

        The base `run` records its errors itself. This is the backstop for a
        subclass which overrides `run` (e.g. to post-process its results) and
        raises, or which has the signature before `keep_results` and `on_error`:
        it must not stop the experiments after it. The new keyword arguments are
        passed only to a `run` which accepts them; the figures of a `run` which
        does not accept the `figure_settings` of the run are rendered without
        them, which is logged.

        Args:
            sid: the experiment.
            output_path: the directory of the experiment.
            formats: the figure formats of the run.
            figure_formats: the figure formats as given by the caller.
            **options: the other arguments of `SimulationExperiment.run`.
        """
        experiment = self.experiments[sid]
        kwargs: dict[str, Any] = {
            "simulator": self.simulator,
            "output_path": output_path,
            "figure_formats": figure_formats,
            **options,
            "on_error": "log",
        }
        parameters = inspect.signature(experiment.run).parameters
        if not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            kwargs = {k: v for k, v in kwargs.items() if k in parameters}
            if options.get("figure_settings") is not None and (
                "figure_settings" not in kwargs
            ):
                logger.warning(
                    "The run of '%s' does not take `figure_settings`, its figures "
                    "are rendered without the figure settings of the run",
                    sid,
                )
        started = time.time()
        try:
            result = experiment.run(**kwargs)
        except Exception as err:
            if self.on_error == "raise":
                raise
            logger.exception("The experiment '%s' failed", sid)
            result = self._written_result(
                ExperimentResult(
                    experiment=experiment,
                    output_path=output_path,
                    error=f"{type(err).__name__}: {err}",
                ),
                formats,
                started,
            )
        # the keys of a failing `figures_mpl()` are not known
        self._remove_unknown_figure_files(result, formats)
        return result

    @staticmethod
    def _written_result(
        result: ExperimentResult, formats: list[str], started: float
    ) -> ExperimentResult:
        """Add the files which a `run` wrote before an error escaped it.

        A file of the directory of the experiment which is not older than the
        start of the run is one of this run and is listed, an older one is
        removed by the caller as stale.

        Args:
            result: the failed result, with the directory of the experiment.
            formats: the figure formats of the run.
            started: the time the run started.
        """
        path = result.output_path
        if path is None:
            return result
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Cannot create the directory '%s'", path)
            return result
        sid = result.experiment.sid
        # the clock of a file system is coarser than the clock of the process
        since = started - 0.5

        def current(fmt: str, directory: Path = path) -> dict[str, Path]:
            return {
                p.name[len(sid) + 1 : -len(fmt) - 1]: p
                for p in directory.glob(f"{sid}_*.{fmt}")
                if p.is_file() and p.stat().st_mtime >= since
            }

        figures: dict[str, list[str]] = {}
        for fmt in formats:
            for key in current(fmt):
                figures.setdefault(key, []).append(fmt)
        datasets = [
            key
            for key in current("tsv", path / DATASETS_DIR)
            if key in result.experiment._datasets
        ]
        return dataclasses.replace(result, figures=figures, datasets=datasets)

    def _failed_result(
        self, failed: ExperimentResult, output_path: Path, formats: list[str]
    ) -> ExperimentResult:
        """Get the result of an experiment which failed outside of its `run`.

        Its directory is created for its report, and the files of its figures
        which an earlier run left there are removed, so that no figure looks
        like one of this run.

        Args:
            failed: the result of the failure, without an output path.
            output_path: the directory of the experiment.
            formats: the figure formats of the run.
        """
        result = dataclasses.replace(failed, output_path=output_path)
        sid = result.experiment.sid
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Cannot create the directory of '%s'", sid)
            return result
        experiment = result.experiment
        experiment._remove_figure_files(output_path, experiment._figures, formats)
        self._remove_unknown_figure_files(result, formats)
        return result

    @staticmethod
    def _remove_unknown_figure_files(
        result: ExperimentResult, formats: list[str]
    ) -> None:
        """Remove the figure files of earlier runs from the directory of a failure.

        When the definition of an experiment failed before its figures were
        known, or its `figures_mpl()` failed, the files of its figures cannot
        be named. The runner gives every experiment a directory of its own, so
        every `<sid>_*.<format>` of the formats of the run in it is from the
        experiment, except for the files this run wrote. Never done in a
        directory which an experiment shares with others.

        Args:
            result: the result of the failed experiment.
            formats: the figure formats of the run.
        """
        if not result.failed or result.output_path is None:
            return
        sid = result.experiment.sid
        kept = {
            f"{sid}_{key}.{fig_format}"
            for key, written in result.figures.items()
            for fig_format in written
        }
        for fig_format in formats:
            for path in result.output_path.glob(f"{sid}_*.{fig_format}"):
                if path.name in kept or not path.is_file():
                    continue
                logger.info(
                    "The file '%s' of '%s' is from an earlier run, it is removed",
                    path.name,
                    sid,
                )
                try:
                    path.unlink()
                except OSError:
                    logger.exception("Cannot remove the file '%s'", path)

    @staticmethod
    def _log_summary(results: list[ExperimentResult]) -> None:
        """Log which experiments and figures failed in a run."""
        failed = [r for r in results if r.failed]
        if not failed:
            return
        lines = []
        for result in failed:
            sid = result.experiment.sid
            if result.error:
                lines.append(f"  experiment '{sid}': {result.error}")
            for key, error in result.failed_figures.items():
                lines.append(f"  figure '{key}' of '{sid}': {error}")
        logger.error(
            "%s of %s experiments failed or have failed figures:\n%s",
            len(failed),
            len(results),
            "\n".join(lines),
        )


@dataclasses.dataclass(frozen=True)
class _Job:
    """What a worker needs to run an experiment, see `_run_job`."""

    experiment_class: type[SimulationExperiment]
    #: the settings the experiment is created with
    settings: dict[str, Any]
    base_path: Path | None
    data_path: Path | Iterable[Path] | None
    jit_optimization: JitOptimization
    on_error: Literal["raise", "log"]
    simulator: Simulator | None
    output_path: Path
    formats: list[str]
    figure_formats: list[str] | None
    #: the other arguments of `SimulationExperiment.run`
    options: dict[str, Any]
    #: the parameters of matplotlib which the run changed, see `_changed_rc_params`
    rc: dict[Any, Any]
    #: the levels of the loggers of the run, see `_logger_levels`
    levels: dict[str, int]


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """What a worker sends back of an experiment: its result and messages."""

    error: str | None
    failed_figures: dict[str, str]
    figures: dict[str, list[str]]
    datasets: list[str]
    records: list[logging.LogRecord]

    def result(
        self, experiment: SimulationExperiment, output_path: Path
    ) -> ExperimentResult:
        """Get the result of the experiment of the run."""
        return ExperimentResult(
            experiment=experiment,
            output_path=output_path,
            error=self.error,
            failed_figures=self.failed_figures,
            figures=self.figures,
            datasets=self.datasets,
        )


class _Collect(logging.Handler):
    """Collect the messages of a worker, which the parent logs, see `_Outcome`."""

    def __init__(self) -> None:
        """Collect every message the loggers pass on."""
        super().__init__(logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Keep the message with its arguments and traceback as text, which pickle sends."""
        record.msg = record.getMessage()
        record.args = None
        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        self.records.append(record)


def _cost(experiment: SimulationExperiment) -> int:
    """Estimate the cost of an experiment: its simulations and figures.

    Every point of a scan is a simulation, a figure costs `FIGURE_COST`.
    """
    points = 0
    for task in experiment._tasks.values():
        simulation = experiment._simulations.get(task.simulation_id)
        points += len(simulation) if isinstance(simulation, Scan) else 1
    return points + FIGURE_COST * len(experiment._figures)


def _modules(jobs: Iterable[_Job]) -> list[str]:
    """Get the modules which the forkserver imports once for all workers.

    The runner and the modules of the experiments, with what they import
    (sbmlsim, matplotlib, the packages of a model); the main module is not
    preloaded, see `sbmlsim.parallel`.
    """
    modules = {job.experiment_class.__module__ for job in jobs} - {"__main__"}
    return sorted({__name__, *modules})


def _changed_rc_params() -> dict[Any, Any]:
    """Get the parameters of matplotlib which differ from the ones as loaded.

    A caller sets them for the figures of a run, e.g. with `rc_context`; the
    worker loads the same `matplotlibrc` and sets them again. The backend is
    the one of the worker.
    """
    # `copy` reads the values without the warnings of deprecated parameters
    current, loaded = matplotlib.rcParams.copy(), matplotlib.rcParamsOrig.copy()
    return {
        key: value
        for key, value in current.items()
        if key != "backend" and value != loaded.get(key)
    }


def _logger_levels() -> dict[str, int]:
    """Get the levels of the loggers which have one, the root logger included."""
    loggers = logging.root.manager.loggerDict.values()
    levels = {
        logger.name: logger.level
        for logger in loggers
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET
    }
    levels[""] = logging.root.level
    return levels


def _run_job(job: _Job) -> _Outcome:
    """Run an experiment in a worker, see the module.

    The models of the experiments of the worker are kept for the next
    experiment, as the runner of the parent keeps them.
    """
    models: dict[ModelKey, RoadrunnerSBMLModel] = parallel.worker_cache(
        ("experiment models", job.jit_optimization), dict
    )
    for name, level in job.levels.items():
        logging.getLogger(name or None).setLevel(level)
    handler = _Collect()
    logging.root.addHandler(handler)
    simulator = copy.copy(job.simulator)
    if simulator is not None:
        # a worker runs the scans of its experiment itself
        simulator.n_workers = 1
    try:
        with matplotlib.rc_context(job.rc):
            runner = ExperimentRunner(
                [],
                base_path=job.base_path,
                data_path=job.data_path,
                simulator=simulator,
                on_error=job.on_error,
                jit_optimization=job.jit_optimization,
            )
            runner.models = models
            runner.initialize([job.experiment_class], **job.settings)
            (sid,) = runner._order
            result = (
                runner._failed_result(runner.failed[sid], job.output_path, job.formats)
                if sid in runner.failed
                else runner._run_experiment(
                    sid,
                    job.output_path,
                    job.formats,
                    figure_formats=job.figure_formats,
                    **job.options,
                )
            )
    finally:
        logging.root.removeHandler(handler)
    return _Outcome(
        error=result.error,
        failed_figures=result.failed_figures,
        figures=result.figures,
        datasets=result.datasets,
        records=handler.records,
    )


def run_experiments(
    experiments: type[SimulationExperiment] | list[type[SimulationExperiment]],
    output_path: Path,
    base_path: Path | None = None,
    data_path: Path | Iterable[Path] | None = None,
    show_report: bool = False,
    raise_on_failure: bool = False,
    figure_settings: FigureSettings | Mapping[str, Any] | None = None,
    n_workers: int | None = None,
) -> list[ExperimentResult]:
    """Run simulation experiments and write their report to the output path.

    Args:
        experiments: the simulation experiments.
        output_path: directory of the results and the report.
        base_path: base path of the simulation experiments.
        data_path: path or paths of the datasets of the simulation experiments.
        show_report: open the report in a web browser.
        raise_on_failure: raise an `ExperimentRunError` when an experiment or
            a figure failed, after every experiment ran and the report was
            written.
        figure_settings: the settings of the figures of every experiment, see
            `ExperimentRunner.run_experiments`.
        n_workers: the processes which run the experiments, see
            `ExperimentRunner.run_experiments`.

    Returns:
        The results of the experiments.

    Raises:
        ExperimentRunError: with `raise_on_failure`, if something failed.
    """
    if not isinstance(experiments, (list, tuple)):
        experiments = [experiments]
    simulator = Simulator()

    runner = ExperimentRunner(
        experiments,
        simulator=simulator,
        data_path=data_path,
        base_path=base_path,
    )
    results = runner.run_experiments(
        output_path=output_path,
        show_figures=False,
        figure_settings=figure_settings,
        n_workers=n_workers,
    )
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=output_path, show_report=show_report)
    if raise_on_failure and any(r.failed for r in results):
        raise ExperimentRunError(results)
    return results
