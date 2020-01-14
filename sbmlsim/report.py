"""
Create markdown report of simulation experiments.
"""
import sys
import os
import logging
import jinja2
from typing import Dict, List
from pathlib import Path
from sbmlsim.experiment import ExperimentResult
from sbmlsim import TEMPLATE_PATH, BASE_PATH
from sbmlsim import __version__

logger = logging.getLogger(__name__)


def create_report(results: Dict[str, ExperimentResult], output_path: Path, repository=None):
    """ Creates markdown report.

    Processes dictionary of ExperimentResuls to generate
    overall report.
    """
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE_PATH)),
                             extensions=['jinja2.ext.autoescape'],
                             trim_blocks=True,
                             lstrip_blocks=True)

    exp_ids = []
    for exp_result in results:
        experiment = exp_result.experiment
        exp_id = experiment.sid
        exp_ids.append(exp_id)

        # relative paths to output path
        model_path = os.path.relpath(str(exp_result.model_path), output_path)
        report_path = f"{model_path[:-4]}.html"
        data_path = os.path.relpath(str(exp_result.data_path), output_path)

        code_path = sys.modules[experiment.__module__].__file__
        with open(code_path, "r") as f_code:
            code = f_code.read()
        code_path = os.path.relpath(code_path, BASE_PATH.parent)
        if repository:
            code_path = f"https:/{repository}/{code_path}"
        logger.debug(code_path)

        context = {
            'exp_id': exp_id,
            'model_path': model_path,
            'report_path': report_path,
            'data_path': data_path,
            'datasets': sorted(experiment.datasets.keys()),
            'simulations': sorted(experiment.simulations.keys()),
            'scans': sorted(experiment.simulations.keys()),
            'figures': sorted(experiment.figures.keys()),
            'code_path': code_path,
            'code': code,
        }
        template = env.get_template('experiment.md')
        md = template.render(context)
        md_file = output_path / f'{exp_id}.md'
        with open(md_file, "w") as f_index:
            f_index.write(md)
            logger.info(f"Create '{md_file}'")

    context = {
        'version': __version__,
        'exp_ids': exp_ids,
    }
    template = env.get_template('index.md')
    md = template.render(context)
    md_file = output_path / 'index.md'
    with open(md_file, "w") as f_index:
        f_index.write(md)
        logger.info(f"Create '{md_file}'")


if __name__ == "__main__":

    # Test experiment for report
    from sbmlsim.examples.experiments.glucose import RESULT_PATH, MODEL_PATH, DATA_PATH
    from sbmlsim.examples.experiments.glucose.dose_response import DoseResponseExperiment

    from sbmlsim.experiment import run_experiment

    results = []
    info = run_experiment(
        DoseResponseExperiment,
        output_path=RESULT_PATH,
        model_path=MODEL_PATH / "liver_glucose.xml",
        data_path=DATA_PATH,
        show_figures=False
    )
    results.append(info)
    create_report(results, output_path=RESULT_PATH)
