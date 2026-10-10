"""Runner for SimulationExperiments.

The ExperimentRunner is used to execute simulation experiments.
This includes
- loading of datasets
- loading of models
- running tasks (simulation on models)
- creating outputs
"""

import dataclasses
import inspect
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from sbmlsim.console import console
from sbmlsim.experiment.experiment import (
    STATIC_FORMAT,
    ExperimentResult,
    ExperimentRunError,
    SimulationExperiment,
)
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator import Simulator
from sbmlsim.units import UnitRegistry
from sbmlsim.units import ureg as package_ureg
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


#: the model of an experiment, i.e. what decides whether two experiments can
#: share one loaded roadrunner instance
ModelKey = tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, float], ...]]


def model_key(abstract_model: AbstractModel) -> ModelKey:
    """Get the key a loaded model is cached under.

    Two experiments share a loaded model when they name the same source, the
    same language and the same changes; the changes are part of it because
    they are applied to the roadrunner instance when it is created. The key is
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
    return (
        str(abstract_model.source.source),
        str(abstract_model.language_type),
        changes,
        tuple(sorted(abstract_model.parameters.items())),
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
        self.models: dict[ModelKey, RoadrunnerSBMLModel] = {}
        self.simulator: Simulator | None = None
        # the sids of the experiments in the order of their classes
        self._order: list[str] = []

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
                    abstract_model=abstract_model, ureg=self.ureg
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

        Returns:
            The results of the experiments, which the report is created from.

        Raises:
            ExperimentRunError: with `raise_on_failure`, if something failed.
        """
        if not output_path.exists():
            output_path.mkdir(parents=True)

        formats = figure_formats if figure_formats is not None else [STATIC_FORMAT]
        exp_results = []
        for sid in self._order:
            console.rule(style="white")
            if sid in self.failed:
                # the definition raised, the experiment is reported, not run
                exp_results.append(
                    self._failed_result(self.failed[sid], output_path / sid, formats)
                )
                continue
            logger.info("Running SimulationExperiment: '%s'", sid)
            # ExperimentResult used to create report; an error of the experiment
            # or of a figure is logged and recorded in it
            exp_results.append(
                self._run_experiment(
                    sid,
                    output_path / sid,
                    formats,
                    show_figures=show_figures,
                    save_results=save_results,
                    figure_formats=figure_formats,
                    reduced_selections=reduced_selections,
                    keep_results=keep_results,
                )
            )
        self._log_summary(exp_results)
        if raise_on_failure and any(r.failed for r in exp_results):
            raise ExperimentRunError(exp_results)
        return exp_results

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
        passed only to a `run` which accepts them.

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

        def current(fmt: str) -> dict[str, Path]:
            return {
                p.name[len(sid) + 1 : -len(fmt) - 1]: p
                for p in path.glob(f"{sid}_*.{fmt}")
                if p.is_file() and p.stat().st_mtime >= since
            }

        figures: dict[str, list[str]] = {}
        for fmt in formats:
            for key in current(fmt):
                figures.setdefault(key, []).append(fmt)
        datasets = [key for key in current("tsv") if key in result.experiment._datasets]
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


def run_experiments(
    experiments: type[SimulationExperiment] | list[type[SimulationExperiment]],
    output_path: Path,
    base_path: Path | None = None,
    data_path: Path | Iterable[Path] | None = None,
    show_report: bool = False,
    raise_on_failure: bool = False,
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
    )
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=output_path, show_report=show_report)
    if raise_on_failure and any(r.failed for r in results):
        raise ExperimentRunError(results)
    return results
