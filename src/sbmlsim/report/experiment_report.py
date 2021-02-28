"""Create report of simulation experiments."""
import json
import logging
import os
import shutil
import sys
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, List

import jinja2

from sbmlsim import RESOURCES_DIR, __version__
from sbmlsim.experiment import ExperimentResult, SimulationExperiment
from sbmlsim.model import AbstractModel


logger = logging.getLogger(__name__)
TEMPLATE_PATH = RESOURCES_DIR / "templates"


class ReportResults:
    """Results for a ExperimentReport."""

    def __init__(self):
        """Construct ReportResults."""
        self.data: Dict[str, Dict] = {}

    def to_json(self, json_path: Path):
        """Write to JSON."""
        with open(json_path, "w") as fp:
            json.dump(fp, self.data, indent=2)  # type: ignore

    @staticmethod
    def from_json(json_path: Path) -> "ReportResults":
        """Read from JSON."""
        with open(json_path, "r") as fp:
            data = json.load(fp)
        results = ReportResults()
        results.data = data
        return results

    def add_experiment_result(self, exp_result: ExperimentResult):
        """Retrieve information for report from the ExperimentResult."""
        experiment: SimulationExperiment = exp_result.experiment
        abs_path = exp_result.output_path
        rel_path = Path(".")
        exp_id = experiment.sid

        # model links
        models = {}
        for model_key, model in experiment.models().items():
            if isinstance(model, (Path, str)):
                model_path = Path(model)
            elif isinstance(model, AbstractModel):
                model_path = model.source.path  # type: ignore

            models[model_key] = Path(os.path.relpath(model_path, str(abs_path)))

        # code path
        code_path = sys.modules[experiment.__module__].__file__
        with open(code_path, "r") as f_code:
            code = f_code.read()
        code_path = Path(os.path.relpath(code_path, str(abs_path)))  # type: ignore

        datasets = {
            key: rel_path / f"{exp_id}_{key}.tsv" for key in experiment._datasets.keys()
        }

        # parse meta data for figures (mapping based on figure keys)
        figures = {
            key: rel_path / f"{exp_id}_{key}" for key in experiment._figures.keys()
        }

        self.data[exp_id] = {
            "exp_id": exp_id,
            "models": models,
            "datasets": datasets,
            "figures": figures,
            "code_path": code_path,
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
        self, results: ReportResults, metadata: Dict = None, template_path=TEMPLATE_PATH
    ):
        """Construct an ExperimentReport."""
        if isinstance(results, list):
            # FIXME: just a bugfix for handling the old outputs
            report_results = ReportResults()
            for exp_result in results:
                report_results.add_experiment_result(exp_result=exp_result)
        else:
            report_results = results

        self.data_dict = (
            report_results.data
        )  # dictionary of exp_ids and information for report rendering
        self.metadata = metadata if metadata else dict()
        self.template_path = template_path

    def create_report(
        self,
        output_path: Path,
        filename=None,
        report_type: ReportType = ReportType.HTML,
        f_filter_context=None,
        **kwargs,
    ):
        """Create report of SimulationExperiments.

        Processes ExperimentResults to generate overall report.

        All relative paths only can be resolved in the report if the
        paths are below the report or at the same level in the file
        hierarchy.
        """
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_path)),
            extensions=["jinja2.ext.autoescape"],
            trim_blocks=True,
            lstrip_blocks=True,
        )

        def write_report(filename: str, context: Dict, template_str: str):
            """Write the report file from given context and template."""
            template = env.get_template(template_str)
            text = template.render(context)
            suffix = template_str.split(".")[-1]
            out_file = output_path / f"{filename}.{suffix}"
            with open(out_file, "w") as f_out:
                f_out.write(text)

        if report_type == self.ReportType.HTML:
            suffix = "html"
        elif report_type == self.ReportType.MARKDOWN:
            suffix = "md"
        elif report_type == self.ReportType.LATEX:
            suffix = "tex"

        if report_type in [self.ReportType.HTML, self.ReportType.MARKDOWN]:
            # report for individual simulation experiment
            for exp_id, context in self.data_dict.items():
                # pprint(context)
                write_report(
                    filename=f"{exp_id}/{exp_id}",
                    context=context,
                    template_str=f"experiment.{suffix}",
                )

        # index file
        context = {
            "version": __version__,
            "data": self.data_dict,
        }

        filename = filename if filename is not None else "index"
        # for latex report the pngs have to be collected with correct paths
        # adapt context
        if report_type == self.ReportType.LATEX:
            # pprint(context)
            if f_filter_context:
                # filter subset of figures
                # FIXME: more robust
                f_filter_context(self.data_dict)
            # pprint(context)

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

        write_report(filename=filename, context=context, template_str=f"index.{suffix}")
