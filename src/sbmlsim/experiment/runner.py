"""
Runner for SimulationExperiments.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulator.simulation import SimulatorAbstract
from sbmlsim.units import UnitRegistry, Units
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


class ExperimentRunner(object):
    """Class for running simulation experiments."""

    def __init__(
        self,
        experiment_classes: Iterable[SimulationExperiment],
        base_path: Path,
        data_path: Path,
        simulator: SimulatorAbstract = None,
        ureg: UnitRegistry = None,
        **kwargs,
    ):

        # single UnitRegistry per runner
        if not ureg:
            ureg = Units.default_ureg()
        self.ureg = ureg
        self.Q_ = ureg.Quantity

        # initialize experiments
        self.base_path = base_path
        self.data_path = data_path
        self.experiments = {}
        self.models = {}

        self.initialize(experiment_classes, **kwargs)
        self.set_simulator(simulator)

    def set_simulator(self, simulator):
        """Set simulator on the runner and experiments."""
        if simulator is None:
            logger.warning(
                f"No simulator set in ExperimentRunner. This warning can be "
                f"ignored for parameter fitting which provides a simulator."
            )
        else:
            self.simulator = simulator  # type: SimulatorAbstract
            for experiment in self.experiments.values():
                experiment.simulator = simulator

    @timeit
    def initialize(self, experiment_classes, **kwargs):
        for exp_class in experiment_classes:
            logger.info(f"Initialize SimulationExperiment: {exp_class.__name__}")
            experiment = exp_class(
                base_path=self.base_path,
                data_path=self.data_path,
                ureg=self.ureg,
                **kwargs,
            )
            self.experiments[experiment.sid] = experiment

            # resolve models for experiment
            _models = {}
            for model_id, source in experiment.models().items():
                if source not in self.models:
                    self.models[source] = RoadrunnerSBMLModel(
                        source=source, ureg=self.ureg
                    )
                _models[model_id] = self.models[source]

            experiment._models = _models
            experiment.initialize()

    @timeit
    def run_experiments(
        self,
        output_path: Path,
        show_figures: bool = False,
        save_results: bool = False,
        figure_formats: List[str] = None,
        reduced_selections: bool = True,
    ) -> List:
        """Run the experiments."""
        if not output_path.exists():
            output_path.mkdir(parents=True)

        exp_results = []
        for sid, experiment in self.experiments.items():  # type: SimulationExperiment
            logger.info(f"Running SimulationExperiment: {sid}")

            # ExperimentResult used to create report
            result = experiment.run(
                simulator=self.simulator,
                output_path=output_path / sid,
                show_figures=show_figures,
                save_results=save_results,
                figure_formats=figure_formats,
                reduced_selections=reduced_selections,
            )
            exp_results.append(result)
        return exp_results
