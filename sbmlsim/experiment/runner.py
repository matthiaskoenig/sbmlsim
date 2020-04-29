from typing import List
from pathlib import Path
import logging

from collections import defaultdict
from dataclasses import dataclass

from .experiment import SimulationExperiment
from sbmlsim.simulator.simulation import SimulatorAbstract
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import UnitRegistry, Units
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


class ExperimentRunner(object):
    """Class for running simulation experiments."""

    def __init__(self, experiment_classes: List[SimulationExperiment],
                 base_path: Path, data_path: Path,
                 simulator: SimulatorAbstract = None,
                 ureg: UnitRegistry = None, **kwargs):

        if simulator is None:
            simulator = SimulatorSerial(
                absolute_tolerance=1E-14,
                relative_tolerance=1E-14
            )
            logger.warning(f"No simulator provided, using default simulator: "
                           f"'{simulator}'")
        self.simulator = simulator  # type: SimulatorAbstract

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

    @timeit
    def initialize(self, experiment_classes, **kwargs):
        for exp_class in experiment_classes:
            experiment = exp_class(
                base_path=self.base_path,
                data_path=self.data_path,
                ureg=self.ureg,
                **kwargs
            )
            self.experiments[experiment.sid] = experiment

            # resolve models for experiment
            _models = {}
            for model_id, source in experiment.models().items():
                if source not in self.models:
                    self.models[source] = RoadrunnerSBMLModel(source=source, ureg=self.ureg)
                _models[model_id] = self.models[source]

            experiment._models = _models
            experiment.initialize()
            experiment.simulator = self.simulator

    @timeit
    def run_experiments(self, output_path: Path,
                        show_figures: bool = False,
                        save_results: bool = False) -> List:
        """Run the experiments."""
        exp_results = []
        for sid, experiment in self.experiments.items():
            logger.info(f"Running SimulationExperiment: {sid}")
            result = experiment.run(
                simulator=self.simulator,
                output_path=output_path / sid,
                show_figures=show_figures,
                save_results=save_results,
            )
            exp_results.append(result)
        return exp_results

