"""Runner for SimulationExperiments.

The ExperimentRunner is used to execute simulation experiments.
This includes
- loading of datasets
- loading of models
- running tasks (simulation on models)
- creating outputs
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sbmlsim.experiment import ExperimentResult, SimulationExperiment
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulator import SimulatorSerial
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
        simulator: SimulatorSerial = None,
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
        self.experiments: Dict[str, SimulationExperiment] = {}
        self.models = {}
        self.simulator: Optional[SimulatorSerial] = None

        self.initialize(experiment_classes, **kwargs)
        self.set_simulator(simulator)

    def set_simulator(self, simulator: SimulatorSerial) -> None:
        """Set simulator on the runner and experiments."""
        if simulator is None:
            logger.debug(
                "No simulator set in ExperimentRunner. This warning can be "
                "ignored in parameter fitting."
            )
        else:
            self.simulator: SimulatorSerial = simulator
            for experiment in self.experiments.values():
                experiment.simulator = simulator

    @timeit
    def initialize(self, experiment_classes, **kwargs):
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

            logger.debug(f"Initialize SimulationExperiment: {exp_class.__name__}")
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
                    # not cashed yet, cash the model for lookup
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
    ) -> List[ExperimentResult]:
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
