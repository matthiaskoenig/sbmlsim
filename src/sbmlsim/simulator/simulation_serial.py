"""Serial simulator.

`SimulatorSerial` loads a model and runs simulations on it: a `Simulation` is
compiled into a `sbmlsim.simulator.plan.Plan` and run by
`sbmlsim.simulator.executor.execute`, a `ScanSim` is a plan per combination of
its dimensions.
"""

import logging
from pathlib import Path

import roadrunner

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.result import TimecourseResult, XResult
from sbmlsim.simulation import ScanSim, Simulation
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation
from sbmlsim.units import UnitsInformation

logger = logging.getLogger(__name__)


class SimulatorSerial:
    """Serial simulator using a single core.

    A single simulator can run many different models.
    See the parallel simulator to run simulations on multiple
    cores.
    """

    def __init__(
        self,
        model: str | Path | RoadrunnerSBMLModel | AbstractModel | None = None,
        **kwargs,
    ):
        """Initialize serial simulator.

        :param model: Path to model or model
        :param kwargs: settings of the integrator, every setting of roadrunner,
            see `RoadrunnerSBMLModel.set_integrator_settings`
        """
        self.model: RoadrunnerSBMLModel | None = None

        # integrator settings
        self.integrator_settings = {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
            **kwargs,
        }

        # set model
        self.set_model(model)

    def set_model(
        self, model: str | Path | RoadrunnerSBMLModel | AbstractModel | None
    ) -> None:
        """Set model for simulator and updates the integrator settings."""
        # logger.info("SimulatorSerial.set_model")
        self.model = None
        if model is not None:
            if isinstance(model, RoadrunnerSBMLModel):
                # logger.info("SimulatorSerial.set_model from RoadrunnerSBMLModel")
                self.model = model
            elif isinstance(model, AbstractModel):
                # logger.info("SimulatorSerial.set_model from AbstractModel")
                self.model = RoadrunnerSBMLModel.from_abstract_model(
                    abstract_model=model
                )
            elif isinstance(model, (str, Path)):
                # logger.info("SimulatorSerial.set_model from Path")
                self.model = RoadrunnerSBMLModel(
                    source=model,
                )

            if self.model is None:
                raise ValueError(f"Unsupported model type: {type(model)}")
            # logger.info("set integrator settings")
            self.set_integrator_settings(**self.integrator_settings)
            # logger.info("model loading finished")

    def set_integrator_settings(self, **kwargs: float | int | bool) -> None:
        """Set settings of the integrator.

        See `RoadrunnerSBMLModel.set_integrator_settings`. The settings apply to the loaded model and to every model set later,
        e.g. the models of the tasks of an experiment.
        """
        if self.model is not None:
            RoadrunnerSBMLModel.set_integrator_settings(self.r_loaded, **kwargs)
        self.integrator_settings.update(kwargs)

    def set_timecourse_selections(self, selections: list[str] | None) -> None:
        """Set the selections of the simulations, all of the model for `None`."""
        model = self.model_loaded
        model.selections = RoadrunnerSBMLModel.set_timecourse_selections(
            self.r_loaded, selections=selections, exclude=set(model.parameters)
        )

    @property
    def r(self) -> roadrunner.RoadRunner | None:
        """Get the roadrunner instance of the model, `None` without a model.

        The instance belongs to the model, see `RoadrunnerSBMLModel`.
        """
        return None if self.model is None else self.model.r

    @property
    def model_loaded(self) -> RoadrunnerSBMLModel:
        """Model of the simulator, raises if no model is set."""
        if self.model is None:
            raise ValueError("No model set on the simulator.")
        return self.model

    @property
    def r_loaded(self) -> roadrunner.RoadRunner:
        """Roadrunner instance of the simulator, raises if no model is set."""
        if self.r is None:
            raise ValueError("No model set on the simulator.")
        return self.r

    @property
    def uinfo(self) -> UnitsInformation:
        """Get model unit information."""
        return self.model_loaded.uinfo

    def compile(self, simulation: Simulation) -> Plan:
        """Compile a simulation against the model of the simulator."""
        model = self.model_loaded
        return compile_simulation(simulation, model.symbols, model.uinfo)

    def simulate(self, simulation: Simulation | Plan) -> TimecourseResult:
        """Run one simulation with the selections of the simulator.

        Args:
            simulation: the simulation or its plan.

        Returns:
            The result of the simulation.
        """
        plan = simulation if isinstance(simulation, Plan) else self.compile(simulation)
        model = self.model_loaded
        return execute(plan, model, model.selections or ["time"])

    def run_simulation(self, simulation: Simulation) -> XResult:
        """Run a simulation.

        Returns:
            The result, without a dimension of a scan.
        """
        return XResult.from_timecourses(
            results=[self.simulate(simulation)], uinfo=self.uinfo
        )

    def run_scan(self, scan: ScanSim) -> XResult:
        """Run a scan, a simulation per combination of its dimensions.

        Returns:
            The result with a dimension per dimension of the scan.
        """
        _indices, simulations = scan.to_simulations()
        return XResult.from_timecourses(
            results=[self.simulate(simulation) for simulation in simulations],
            scan=scan,
            uinfo=self.uinfo,
        )
