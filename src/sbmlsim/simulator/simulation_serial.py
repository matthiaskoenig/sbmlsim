"""Serial simulator."""

from typing import Optional, Union
from pathlib import Path
import roadrunner
import pandas as pd
from pint import Quantity
from pymetadata import log

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel, ModelChange
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, Timecourse, TimecourseSim
from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


class SimulatorSerial:
    """Serial simulator using a single core.

    A single simulator can run many different models.
    See the parallel simulator to run simulations on multiple
    cores.
    """

    def __init__(
        self,
        model: Union[str | Path | RoadrunnerSBMLModel | AbstractModel] = None,
        **kwargs,
    ):
        """Initialize serial simulator.

        :param model: Path to model or model
        :param kwargs: integrator settings
        """
        self.r: Optional[roadrunner.RoadRunner] = None
        self.model: Optional[RoadrunnerSBMLModel] = None

        # integrator settings
        self.integrator_settings = {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
            **kwargs,
        }

        # set model
        self.set_model(model)

    def set_model(self, model: Union[str | Path | RoadrunnerSBMLModel | AbstractModel]):
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

            # logger.info(f"get roadrunner instance from model: {type(self.model)}")
            self.r = self.model.r
            # logger.info("set integrator settings")
            self.set_integrator_settings(**self.integrator_settings)
            # logger.info("model loading finished")

    def set_integrator_settings(self, **kwargs):
        """Set settings in the integrator."""
        RoadrunnerSBMLModel.set_integrator_settings(self.r, **kwargs)

    def set_timecourse_selections(self, selections):
        """Set timecourse selection in model."""
        RoadrunnerSBMLModel.set_timecourse_selections(self.r, selections=selections)

    @property
    def uinfo(self) -> UnitsInformation:
        """Get model unit information."""
        return self.model.uinfo

    @property
    def Q_(self) -> Quantity:
        """Get model unit information."""
        return self.model.uinfo.ureg.Quantity

    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(
                f"'run_timecourse' requires TimecourseSim, but " f"'{type(simulation)}'"
            )
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    def run_scan(self, scan: ScanSim) -> XResult:
        """Run a scan simulation."""
        # normalize the scan (simulation and dimensions)
        scan.normalize(uinfo=self.uinfo)

        # create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        dfs = self._timecourses(simulations)

        # based on the indices the result structure must be created
        return XResult.from_dfs(dfs=dfs, scan=scan, uinfo=self.uinfo)

    def _timecourses(self, simulations: list[TimecourseSim]) -> list[pd.DataFrame]:
        return [self._timecourse(sim) for sim in simulations]

    def _timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """Timecourse simulation.

        Requires for all timecourse definitions in the timecourse simulation
        to be unit normalized. The changes have no units any more
        for parallel simulations.
        You should never call this function directly!

        :param simulation: Simulation definition(s)
        :return: DataFrame with results
        """
        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        if simulation.reset:
            self.r.resetToOrigin()

        frames = []
        t_offset = simulation.time_offset
        for k, tc in enumerate(simulation.timecourses):
            if k == 0 and tc.model_changes:
                # [1] apply model changes of first simulation
                logger.debug("Applying model changes")
                for key, item in tc.model_changes.items():
                    if key.startswith("init"):
                        logger.error(
                            f"Initial model changes should be provided "
                            f"without 'init': '{key} = {item}'"
                        )
                    # FIXME: implement model changes via init
                    # init_key = f"init({key})"
                    init_key = key
                    try:
                        value = item.magnitude
                    except AttributeError:
                        value = item

                    try:
                        self.r[init_key] = value
                    except RuntimeError:
                        logger.error(f"roadrunner RuntimeError: '{init_key} = {item}'")
                        # boundary condition=true species, trying direct fallback
                        # see https://github.com/sys-bio/roadrunner/issues/711
                        init_key = key
                        self.r[key] = value

                    logger.debug(f"\t{init_key} = {item}")

                # [2] re-evaluate initial assignments
                # https://github.com/sys-bio/roadrunner/issues/710
                # logger.debug("Reevaluate initial conditions")
                # FIXME/TODO: support initial model changes
                # self.r.resetAll()
                # self.r.reset(SelectionRecord.DEPENDENT_FLOATING_AMOUNT)
                # self.r.reset(SelectionRecord.DEPENDENT_INITIAL_GLOBAL_PARAMETER)

            # [3] apply model manipulations
            # model manipulations are applied to model
            if len(tc.model_manipulations) > 0:
                # FIXME: update to support roadrunner model changes
                for key, value in tc.model_changes.items():
                    if key == ModelChange.CLAMP_SPECIES:
                        for sid, formula in value.items():
                            ModelChange.clamp_species(self.r, sid, formula)
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

                try:
                    self.r[key] = float(item.magnitude)
                except AttributeError:
                    self.r[key] = float(item)
                logger.debug(f"\t{key} = {item}")

            # run simulation
            integrator = self.r.integrator
            # FIXME: support simulation by times
            if integrator.getValue("variable_step_size"):
                s = self.r.simulate(start=tc.start, end=tc.end)
            else:
                s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)

            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset

            if not tc.discard:
                # discard timecourses (pre-simulation)
                t_offset += tc.end
                frames.append(df)

        return pd.concat(frames, sort=False)
