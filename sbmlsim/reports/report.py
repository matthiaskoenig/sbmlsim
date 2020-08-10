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
from pprint import pprint

from sbmlsim import __version__
from sbmlsim.experiment import SimulationExperiment, ExperimentResult
from sbmlsim import BASE_PATH

logger = logging.getLogger(__name__)
TEMPLATE_PATH = BASE_PATH / "reports" / "templates"


class ReportResults:

    def __init__(self):
        self.data = OrderedDict()  # type: OrderedDict[str, Dict]

    def add_experiment_result(self, exp_result: ExperimentResult):
        """Retrieves information for report from the ExperimentResult.

        :param ExperimentResult:
        :return:
        """
        experiment = exp_result.experiment  # type: SimulationExperiment
        abs_path = exp_result.output_path
        rel_path = Path(".")
        exp_id = experiment.sid

        # model links
        models = {}
        for model_key, model_path in experiment.models().items():
            models[model_key] = Path(os.path.relpath(
                model_path, str(abs_path)
            ))

        # code path
        code_path = sys.modules[experiment.__module__].__file__
        with open(code_path, "r") as f_code:
            code = f_code.read()
        code_path = Path(os.path.relpath(code_path, str(abs_path)))

        datasets = {key: rel_path / f"{exp_id}_{key}.tsv" for key in experiment._datasets.keys()}

        # parse meta data for figures (mapping based on figure keys)
        figures = {key: rel_path / f"{exp_id}_{key}.svg" for key in experiment._figures.keys()}

        self.data[exp_id] = {
            'exp_id': exp_id,
            'models': models,
            'datasets': datasets,
            'figures': figures,
            'code_path': code_path,
            'code': code,
        }


class ExperimentReport:

    class ReportType(Enum):
        MARKDOWN = 1
        HTML = 2
        LATEX = 3

    def __init__(self, results: ReportResults,
                 metadata: Dict = None,
                 template_path=TEMPLATE_PATH):

        self.results = results
        self.metadata = metadata if metadata else dict()
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
        for exp_id, context in self.results.data.items():
            pprint(context)

            write_report(name=f"{exp_id}/{exp_id}", context=context,
                         template_str=f"experiment.{suffix}")

        # index file
        # context = {
        #     'version': __version__,
        #     'exp_ids': exp_ids,
        # }
        # write_report(name="index", context=context,
        #              template_str=f'index.{suffix}')
