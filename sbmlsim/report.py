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

from sbmlsim.experiment import ExperimentResult, SimulationExperiment
from sbmlsim import TEMPLATE_PATH

from sbmlsim import __version__

logger = logging.getLogger(__name__)


class Report(object):
    pass


# TODO: refactor in report class

def create_report(results: List[ExperimentResult],
                  output_path: Path,
                  metadata: Dict = None,
                  repository=None,
                  template_path=TEMPLATE_PATH):
    """ Creates markdown report.

    Processes dictionary of ExperimentResuls to generate
    overall report.

    All relative paths only can be resolved in the reports if the
    paths are below the reports or at the same level in the file
    hierarchy.
    """
    if metadata is None:
        metadata = dict()
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_path)),
                             extensions=['jinja2.ext.autoescape'],
                             trim_blocks=True,
                             lstrip_blocks=True)

    def write_report(name: str, context: Dict, template_str: str):
        """Writes the report file from given context and template."""
        template = env.get_template(template_str)
        text = template.render(context)
        suffix = template_str.split(".")[-1]
        out_file = output_path / f'{name}.{suffix}'
        with open(out_file, "w") as f_md:
            f_md.write(text)
            logger.info(f"Write {suffix}: 'file://{out_file}'")

    # --- {exp_id}.md ---
    exp_ids = OrderedDict()
    for exp_result in results:  # type: ExperimentResult
        experiment = exp_result.experiment  # type: SimulationExperiment
        exp_id = experiment.sid


        # relative paths to output path
        model_path = os.path.relpath(str(exp_result.model_path), output_path)
        data_path = os.path.relpath(str(exp_result.data_path), output_path)
        results_path = os.path.relpath(str(exp_result.results_path), output_path)

        # code path
        code_path = sys.modules[experiment.__module__].__file__
        with open(code_path, "r") as f_code:
            code = f_code.read()
        code_path = os.path.relpath(code_path, output_path)

        # parse meta data for figures (mapping based on figure keys)
        figures_keys = sorted(experiment.figures.keys())
        figures = {key: metadata.get(f"{exp_id}_{key}", None) for key in figures_keys}

        context = {
            'exp_id': exp_id,
            'results_path': results_path,  # prefix path for all results (figures, json, ...)
            'model_path': model_path,
            'data_path': data_path,
            'code_path': code_path,
            'datasets': sorted(experiment.datasets.keys()),
            'simulations': sorted(experiment.simulations.keys()),
            'scans': sorted(experiment.scans.keys()),
            'figures': figures,
            'code': code,
        }
        exp_ids[exp_id] = context

        # add additional information if existing
        if Path(f"{str(exp_result.model_path)[:-4]}.html").exists():
            context["report_path"] = f"{model_path[:-4]}.html"

        write_report(name=exp_id, context=context, template_str='experiment.md')
        write_report(name=exp_id, context=context, template_str='experiment.html')

    # --- index.md ---
    context = {
        'version': __version__,
        'exp_ids': exp_ids,
    }
    write_report(name="index", context=context, template_str='index.md')
    write_report(name="index", context=context, template_str='index.html')


if __name__ == "__main__":
    from pathlib import Path
    from sbmlsim.experiment import run_experiment

    from sbmlsim.examples.glucose.experiments.dose_response import DoseResponseExperiment
    BASE_PATH = Path(__file__).parent / "examples" / "glucose"

    results = []
    info = run_experiment(
        DoseResponseExperiment,
        output_path=BASE_PATH / "results",
        model_path=BASE_PATH / "model" / "liver_glucose.xml",
        data_path=BASE_PATH / "data",
        show_figures=False
    )
    results.append(info)
    create_report(results, output_path=BASE_PATH)
