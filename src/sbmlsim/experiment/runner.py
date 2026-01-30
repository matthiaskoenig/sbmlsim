"""Runner for SimulationExperiments.

The ExperimentRunner is used to execute simulation experiments.
This includes
- loading of datasets
- loading of models
- running tasks (simulation on models)
- creating outputs
"""

from pathlib import Path
from typing import Optional, Tuple, Type, Union

from pymetadata import log
from pymetadata.console import console

from sbmlsim.experiment import ExperimentResult, SimulationExperiment
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import UnitRegistry, UnitsInformation
from sbmlsim.utils import timeit


logger = log.get_logger(__name__)


class ExperimentRunner(object):
    """Class for running simulation experiments."""

    def __init__(
        self,
        experiment_classes: Union[
            Type[SimulationExperiment], list[Type[SimulationExperiment]]
        ],
        base_path: Path,
        data_path: Path,
        simulator: SimulatorSerial = None,
        ureg: UnitRegistry = None,  # FIXME: is this needed on ExperimentRunner?
        **kwargs,
    ):
        """Initialize the runner.

        FIXME: document arguments for the solver.

        """

        # single UnitRegistry per runner
        if not ureg:
            ureg = UnitsInformation._default_ureg()
        self.ureg = ureg
        self.Q_ = ureg.Quantity

        # initialize experiments
        self.base_path = base_path
        self.data_path = data_path
        self.experiments: dict[str, SimulationExperiment] = {}
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

    def initialize(
        self,
        experiment_classes: Union[
            list[Type[SimulationExperiment]],
            Tuple[Type[SimulationExperiment]],
            set[Type[SimulationExperiment]],
        ],
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

            logger.debug(f"Initialize SimulationExperiment: {exp_class.__name__}")
            experiment: SimulationExperiment = exp_class(
                base_path=self.base_path,
                data_path=self.data_path,
                ureg=self.ureg,
                **kwargs,
            )

            # resolve models for experiment
            _models = {}
            for model_id, source in experiment.models().items():
                if source not in self.models:
                    # not cashed yet, cash the model for lookup
                    self.models[source] = RoadrunnerSBMLModel.from_abstract_model(
                        abstract_model=source, ureg=self.ureg
                    )
                _models[model_id] = self.models[source]

            # set resolved models in experiment
            experiment._models = _models
            # only after model loading the unit registry is filled
            experiment.initialize()
            self.experiments[experiment.sid] = experiment

    @timeit
    def run_experiments(
        self,
        output_path: Path,
        show_figures: bool = False,
        save_results: bool = False,
        figure_formats: list[str] = None,
        reduced_selections: bool = True,
    ) -> list[ExperimentResult]:
        """Run the experiments."""
        if not output_path.exists():
            output_path.mkdir(parents=True)

        exp_results = []
        experiment: SimulationExperiment
        for sid, experiment in self.experiments.items():
            console.rule(style="white")
            logger.info(f"Running SimulationExperiment: '{sid}'")

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


def run_experiments(
    experiments: Union[Type[SimulationExperiment], list[Type[SimulationExperiment]]],
    output_path: Path,
    base_path: Path = None,
    data_path: Union[list[Path], Tuple[Path], Optional[Path]] = None,
) -> Path:
    """Run simulation experiments."""
    if not isinstance(experiments, (list, tuple)):
        experiments = [experiments]
    simulator = SimulatorSerial()

    runner = ExperimentRunner(
        experiments,
        simulator=simulator,
        data_path=data_path,
        base_path=base_path,
    )
    results = runner.run_experiments(
        output_path=output_path,
        show_figures=True,
    )
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=output_path)
