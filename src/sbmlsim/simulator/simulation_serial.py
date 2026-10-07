"""Serial simulator.

`SimulatorSerial` loads a model and runs simulations on it: a `Simulation` is
compiled into a `sbmlsim.simulator.plan.Plan` and run by
`sbmlsim.simulator.executor.execute`, a `ScanSim` is a plan per combination of
its dimensions.
"""

import logging
from pathlib import Path

import numpy as np
import roadrunner

from sbmlsim.model import AbstractModel, ModelChange, RoadrunnerSBMLModel
from sbmlsim.result import TimecourseResult, XResult
from sbmlsim.simulation import ScanSim, Simulation, Timecourse, TimecourseSim
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation
from sbmlsim.units import Quantity, UnitsInformation

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
        :param kwargs: integrator settings
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

    def set_integrator_settings(self, **kwargs):
        """Set settings in the integrator."""
        RoadrunnerSBMLModel.set_integrator_settings(self.r_loaded, **kwargs)

    def set_timecourse_selections(self, selections: list[str] | None) -> None:
        """Set the selections of the simulations, all of the model for `None`."""
        self.model_loaded.selections = RoadrunnerSBMLModel.set_timecourse_selections(
            self.r_loaded, selections=selections
        )

    @property
    def r(self) -> roadrunner.RoadRunner | None:
        """Get the roadrunner instance of the model, `None` without a model.

        The instance belongs to the model, which loads it again when it
        derives the model, see `RoadrunnerSBMLModel.free_initial_assignments`.
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

    @property
    def Q_(self) -> type[Quantity]:
        """Quantity of the unit registry of the model."""
        return self.model_loaded.uinfo.ureg.Quantity

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

    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(
                f"'run_timecourse' requires TimecourseSim, but '{type(simulation)}'"
            )
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    def run_scan(self, scan: ScanSim) -> XResult:
        """Run a scan simulation."""
        if isinstance(scan.simulation, Simulation):
            _indices, definitions = scan.to_simulations()
            return XResult.from_timecourses(
                results=[self.simulate(simulation) for simulation in definitions],
                scan=scan,
                uinfo=self.uinfo,
            )
        # normalize the scan (simulation and dimensions)
        scan.normalize(uinfo=self.uinfo)

        # create all possible combinations of the scan
        _indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        results = self._timecourses(simulations)

        # based on the indices the result structure must be created
        return XResult.from_timecourses(results=results, scan=scan, uinfo=self.uinfo)

    def _timecourses(self, simulations: list[TimecourseSim]) -> list[TimecourseResult]:
        """Run timecourse simulations.

        Args:
            simulations: unit normalized simulations.

        Returns:
            The result of every simulation.
        """
        return [self._timecourse(sim) for sim in simulations]

    def _timecourse(self, simulation: TimecourseSim) -> TimecourseResult:
        """Timecourse simulation.

        Requires for all timecourse definitions in the timecourse simulation
        to be unit normalized. The changes have no units any more
        for parallel simulations.
        You should never call this function directly!

        The result is the array roadrunner returns, a DataFrame per simulation
        would cost as much as the simulation of a small model.

        Args:
            simulation: Simulation definition(s).

        Returns:
            The values of the timecourse selections, the timecourses which are
            not discarded one after the other.

        Raises:
            ValueError: if every timecourse of the simulation is discarded.
        """
        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        r = self.r_loaded
        if simulation.reset:
            r.resetToOrigin()

        results: list[TimecourseResult] = []
        t_offset = simulation.time_offset
        for k, tc in enumerate(simulation.timecourses):
            if k == 0 and tc.model_changes:
                # [1] apply model changes of first simulation
                logger.debug("Applying model changes")
                for key, item in tc.model_changes.items():
                    if key.startswith("init"):
                        logger.error(
                            "Initial model changes should be provided without 'init': '%s = %s'",
                            key,
                            item,
                        )
                    # FIXME: implement model changes via init
                    # init_key = f"init({key})"
                    init_key = key
                    try:
                        value = item.magnitude
                    except AttributeError:
                        value = item

                    try:
                        r[init_key] = value
                    except RuntimeError:
                        logger.error(
                            "roadrunner RuntimeError: '%s = %s'", init_key, item
                        )
                        # boundary condition=true species, trying direct fallback
                        # see https://github.com/sys-bio/roadrunner/issues/711
                        init_key = key
                        r[key] = value

                    logger.debug("	%s = %s", init_key, item)

                # [2] re-evaluate initial assignments
                # https://github.com/sys-bio/roadrunner/issues/710
                # logger.debug("Reevaluate initial conditions")
                # FIXME/TODO: support initial model changes
                # r.resetAll()
                # r.reset(SelectionRecord.DEPENDENT_FLOATING_AMOUNT)
                # r.reset(SelectionRecord.DEPENDENT_INITIAL_GLOBAL_PARAMETER)

            # [3] apply model manipulations
            # model manipulations are applied to model
            if len(tc.model_manipulations) > 0:
                # FIXME: update to support roadrunner model changes
                for key, value in tc.model_changes.items():
                    if key == ModelChange.CLAMP_SPECIES:
                        for sid, formula in value.items():
                            ModelChange.clamp_species(r, sid, formula)
                    else:
                        raise ValueError(
                            f"Unsupported model change: "
                            f"'{key}': {value}. Supported changes are: "
                            f"['{ModelChange.CLAMP_SPECIES}']"
                        )

            # [4] apply changes
            if tc.changes:
                logger.debug("Applying simulation changes")
            for key, item in tc.changes.items():
                # FIXME: handle concentrations/amounts/default
                # TODO: Figure out the hasOnlySubstanceUnit flag! (roadrunner)
                # r: roadrunner.ExecutableModel = self.r

                r[key] = (
                    float(item.magnitude) if isinstance(item, Quantity) else float(item)
                )
                logger.debug("	%s = %s", key, item)

            # run simulation
            integrator = r.integrator
            # FIXME: support simulation by times
            if integrator.getValue("variable_step_size"):
                s = r.simulate(start=tc.start, end=tc.end)
            else:
                s = r.simulate(start=tc.start, end=tc.end, steps=tc.steps)

            if not tc.discard:
                # discard timecourses (pre-simulation)
                result = TimecourseResult(
                    columns=tuple(s.colnames), values=np.array(s, dtype=float)
                )
                if t_offset != 0.0:
                    result.time[:] += t_offset
                t_offset += tc.end
                results.append(result)

        if not results:
            raise ValueError(
                "Every timecourse of the simulation is discarded, there are no results."
            )
        return TimecourseResult.concatenate(results)
