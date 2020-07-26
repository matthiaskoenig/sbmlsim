"""
Create markdown report of simulation experiments.
"""
import sys
import os
import logging
import jinja2
from collections import OrderedDict
from typing import Dict, List
from pathlib import Path
from enum import Enum

from sbmlsim import __version__
from sbmlsim.experiment import SimulationExperiment, ExperimentResult
from sbmlsim import BASE_PATH

TEMPLATE_PATH = BASE_PATH / "experiment" / "templates"
logger = logging.getLogger(__name__)


class ExperimentReport(object):

    class ReportType(Enum):
        MARKDOWN = 1
        HTML = 2

    def __init__(self, results: List[ExperimentResult],
                 metadata: Dict = None,
                 template_path=TEMPLATE_PATH):
        self.results = results
        if metadata is None:
            metadata = dict()

        self.metadata = metadata
        self.template_path = template_path

    def create_report(self, output_path: Path, report_type: ReportType=ReportType.HTML):
        """ Create report of SimulationExperiments.

        Processes ExperimentResults to generate overall report.

        All relative paths only can be resolved in the reports if the
        paths are below the reports or at the same level in the file
        hierarchy.
        """
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(self.template_path)),
                                 extensions=['jinja2.ext.autoescape'],
                                 trim_blocks=True,
                                 lstrip_blocks=True)

        def write_report(name: str, context: Dict, template_str: str):
            """Writes the report file from given context and template."""
            template = env.get_template(template_str)
            text = template.render(context)
            suffix = template_str.split(".")[-1]
            out_file = output_path / f'{name}.{suffix}'
            with open(out_file, "w") as f_out:
                f_out.write(text)

        if report_type == self.ReportType.HTML:
            suffix = "html"
        elif report_type == self.ReportType.MARKDOWN:
            suffix = "md"

        # report for simulation experiment
        exp_ids = OrderedDict()
        for exp_result in self.results:  # type: ExperimentResult
            experiment = exp_result.experiment  # type: SimulationExperiment
            exp_id = experiment.sid

            # relative paths to output path
            model_path = ""  # FIXME os.path.relpath(str(exp_result.model_path), output_path)
            data_path = os.path.relpath(str(experiment.data_path), os.path.join(output_path, exp_id))
            results_path = os.path.relpath(str(exp_result.output_path), os.path.join(output_path, exp_id))

            # code path
            code_path = sys.modules[experiment.__module__].__file__
            with open(code_path, "r") as f_code:
                code = f_code.read()
            code_path = os.path.relpath(code_path, os.path.join(output_path, exp_id))

            # parse meta data for figures (mapping based on figure keys)
            figures_keys = sorted(experiment._figures.keys())
            figures = {key: self.metadata.get(f"{exp_id}_{key}", None) for key in figures_keys}

            context = {
                'exp_id': exp_id,
                'results_path': results_path,  # prefix path for all results (figures, json, ...)
                'model_path': model_path,
                'data_path': data_path,
                'code_path': code_path,
                'datasets': sorted(experiment._datasets.keys()),
                'simulations': sorted(experiment._simulations.keys()),
                'figures': figures,
                'code': code,
            }
            exp_ids[exp_id] = context
            write_report(name=f"{exp_id}/{exp_id}", context=context,
                         template_str=f"experiment.{suffix}")

        # index file
        context = {
            'version': __version__,
            'exp_ids': exp_ids,
        }
        write_report(name="index", context=context,
                     template_str=f'index.{suffix}')
