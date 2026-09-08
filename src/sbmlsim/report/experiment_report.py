"""Create report of simulation experiments."""

import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from sbmlsim import __version__
from sbmlsim.experiment.experiment import ExperimentResult, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.report.templates import TEMPLATE_DIR, template_environment

logger = logging.getLogger(__name__)
TEMPLATE_PATH = TEMPLATE_DIR


def _relative_path(path: Path, start: Path) -> Path:
    """Path relative to start, the absolute path if there is none.

    On Windows a path on another drive than `start` has no relative path.
    """
    try:
        return Path(os.path.relpath(path, str(start)))
    except ValueError:
        return path.resolve()


class ReportResults:
    """Results for a ExperimentReport."""

    def __init__(self):
        """Construct ReportResults."""
        self.data: dict[str, dict[str, Any]] = {}

    def to_json(self, json_path: Path) -> None:
        """Write to JSON.

        Args:
            json_path: Path to the JSON file.
        """
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, indent=2)

    @staticmethod
    def from_json(json_path: Path) -> "ReportResults":
        """Read from JSON.

        Args:
            json_path: Path to the JSON file.

        Returns:
            ReportResults read from the file.
        """
        with open(json_path, encoding="utf-8") as fp:
            data = json.load(fp)
        results = ReportResults()
        results.data = data
        return results

    def add_experiment_result(self, exp_result: ExperimentResult) -> None:
        """Retrieve information for report from the ExperimentResult.

        Args:
            exp_result: Result of the simulation experiment.

        Raises:
            ValueError: If a model has no resolvable path.
        """
        experiment: SimulationExperiment = exp_result.experiment
        abs_path = exp_result.output_path
        if abs_path is None:
            raise ValueError("ExperimentResult without output_path cannot be reported.")
        rel_path = Path(".")
        exp_id = experiment.sid

        # model links
        models: dict[str, Path] = {}
        for model_key, model in experiment.models().items():
            model_path: Path
            if isinstance(model, (Path, str)):
                model_path = Path(model)
            elif isinstance(model, AbstractModel):
                if model.source.path is None:
                    raise ValueError(f"Model '{model_key}' has no source path.")
                model_path = model.source.path
            else:
                raise ValueError(f"Unsupported model type: '{type(model)}'")

            models[model_key] = _relative_path(model_path, abs_path)

        # code path
        code_path = sys.modules[experiment.__module__].__file__
        if code_path is None:
            raise ValueError(f"No source file for experiment '{exp_id}'.")
        with open(code_path, encoding="utf-8") as f_code:
            code = f_code.read()
        code_rel_path = _relative_path(Path(code_path), abs_path)

        datasets = {
            key: rel_path / f"{exp_id}_{key}.tsv" for key in experiment._datasets
        }

        # parse meta data for figures (mapping based on figure keys)
        figures = {key: rel_path / f"{exp_id}_{key}" for key in experiment._mpl_figures}

        self.data[exp_id] = {
            "exp_id": exp_id,
            "models": models,
            "datasets": datasets,
            "figures": figures,
            "code_path": code_rel_path,
            "code": code,
        }


class ExperimentReport:
    """Report for an experiment."""

    class ReportType(Enum):
        """Type of report."""

        MARKDOWN = 1
        HTML = 2
        LATEX = 3

    def __init__(
        self,
        results: ReportResults | list[ExperimentResult],
        metadata: dict[str, Any] | None = None,
        template_path: Path = TEMPLATE_PATH,
    ):
        """Construct an ExperimentReport.

        Args:
            results: Report results or list of experiment results.
            metadata: Additional metadata for the report.
            template_path: Directory with the jinja2 templates.
        """
        report_results: ReportResults
        if isinstance(results, list):
            # FIXME: just a bugfix for handling the old outputs
            report_results = ReportResults()
            for exp_result in results:
                report_results.add_experiment_result(exp_result=exp_result)
        else:
            report_results = results

        # dictionary of exp_ids and information for report rendering
        self.data_dict = report_results.data
        self.metadata = metadata if metadata else {}
        self.template_path = template_path

    def create_report(
        self,
        output_path: Path,
        filename: str | None = None,
        report_type: ReportType = ReportType.HTML,
        f_filter_context: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Create report of SimulationExperiments.

        Processes ExperimentResults to generate overall report.

        All relative paths only can be resolved in the report if the
        paths are below the report or at the same level in the file
        hierarchy.

        Args:
            output_path: Directory for the report.
            filename: Name of the index file (without suffix).
            report_type: Type of the report.
            f_filter_context: Function filtering the context (latex reports).
            **kwargs: Additional arguments, e.g. `latex_path_prefix`.

        Returns:
            Path to the created index file.

        Raises:
            ValueError: If the report type is not supported.
        """
        env = template_environment(self.template_path)

        def write_report(
            filename: str, context: dict[str, Any], template_str: str
        ) -> Path:
            """Write the report file from given context and template."""
            template = env.get_template(template_str)
            text = template.render(context)
            suffix = template_str.split(".")[-1]
            out_file: Path = output_path / f"{filename}.{suffix}"
            with open(out_file, "w", encoding="utf-8") as f_out:
                f_out.write(text)
            return out_file

        if report_type == self.ReportType.HTML:
            suffix = "html"
        elif report_type == self.ReportType.MARKDOWN:
            suffix = "md"
        elif report_type == self.ReportType.LATEX:
            suffix = "tex"
        else:
            raise ValueError(f"Unsupported report type: '{report_type}'")

        if report_type in [self.ReportType.HTML, self.ReportType.MARKDOWN]:
            # report for individual simulation experiment
            for exp_id, context in self.data_dict.items():
                write_report(
                    filename=f"{exp_id}/{exp_id}",
                    context=context,
                    template_str=f"experiment.{suffix}",
                )

        # index file
        context: dict[str, Any] = {
            "version": __version__,
            "data": self.data_dict,
        }

        filename = filename if filename is not None else "index"
        # for latex report the pngs have to be collected with correct paths
        # adapt context
        if report_type == self.ReportType.LATEX:
            if f_filter_context:
                # filter subset of figures
                # FIXME: more robust
                f_filter_context(self.data_dict)

            # collect and copy figures

            context["latex_path_prefix"] = kwargs.get("latex_path_prefix", "")
            figure_base_path = output_path / f"{filename}_figures"
            if not figure_base_path.exists():
                figure_base_path.mkdir(parents=True)
            for exp_id, exp_context in self.data_dict.items():
                for fig_path in exp_context["figures"].values():
                    shutil.copy(
                        str(output_path / exp_id / f"{fig_path}.png"),
                        str(figure_base_path / f"{fig_path}.png"),
                    )

        report_path = write_report(
            filename=filename, context=context, template_str=f"index.{suffix}"
        )
        logger.info("report created: %s", report_path.resolve().as_uri())
        return report_path
