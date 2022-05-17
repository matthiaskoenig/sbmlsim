"""Classes for running simulations with SBML models."""

import pandas as pd
from sbmlutils import log

from sbmlsim.model import ModelChange
from sbmlsim.simulator.model_rr import roadrunner
from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim
from sbmlsim.result import XResult

# TODO: handle unit information
# from pint import Quantity
# from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


# FIXME: This can probably be all on the roadrunner model.

from abc import ABC, abstractmethod


class SimulatorRR(ABC):
    """Abstract base class for roadrunner simulator."""
    @abstractmethod
    def set_model(self, model):
        """Set model."""
        pass

    @abstractmethod
    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set timecourse selections."""
        pass

    @abstractmethod
    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        pass

    @abstractmethod
    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Run timecourses."""
        pass

    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(
                f"'run_timecourse' requires TimecourseSim, but " f"'{type(simulation)}'"
            )
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    def run_scan(self, scan: ScanSim) -> XResult:
        """Run scan simulation."""
        # normalize the scan (simulation and dimensions)
        # FIXME: units
        scan.normalize(uinfo=self.uinfo)

        # create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        dfs = self._timecourses(simulations)

        # based on the indices the result structure must be created
        # FIXME: units
        # return XResult.from_dfs(dfs=dfs, scan=scan, uinfo=self.uinfo)

        return XResult.from_dfs(dfs=dfs, scan=scan)


class SimulationWorkerRR:
    """Worker running simulations with roadrunner.

    Implements the timecourse simulation once which can be reused by
    the different simulators.
    """

    def __init__(self, model_state: str):
        """Initialize worker with state str."""

        # TODO: use a roadrunner model here

        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner()
        self.r.loadStateS(model_state)

        # default settings
        self.integrator_settings = {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
        }
        self.integrator_settings.update(kwargs)
        self.set_model(model)

    # self.model: Optional[AbstractModel, RoadrunnerSBMLModel] = None

    def __init__(self, model=None, **kwargs):
        """Initialize serial simulator.

        :param model: Path to model or model
        :param kwargs: integrator settings
        """
        # default settings
        self.integrator_settings = {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
        }
        self.integrator_settings.update(kwargs)
        self.set_model(model)
        # self.model: Optional[AbstractModel, RoadrunnerSBMLModel] = None

    def set_model(self, model):
        """Set model for simulator and updates the integrator settings."""
        logger.debug("SimulatorSerial.set_model")
        if model is None:
            self.model = None
        else:
            # if isinstance(model, AbstractModel):
            #     self.model = model
            # else:
            # handle path, urn, ...

            # FIXME: this is probably the issue
            self.model = RoadrunnerSBMLModel(
                source=model, settings=self.integrator_settings
            )

            self.set_integrator_settings(**self.integrator_settings)


    def set_model(self, state: bytes) -> None:
        """Set model using the state."""
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner()
        if state is not None:
            # write state to temporary file for reading
            with tempfile.NamedTemporaryFile("wb") as f_temp:
                f_temp.write(state)
                filename = f_temp.name
                logger.debug(f"load state: {filename}")

                # FIXME: this must be in a lock
                self.r.loadState(str(filename))

    def set_model(self, model):
        """Set model."""
        super(SimulatorParallel, self).set_model(model)
        if model:
            if not self.model.state_path:
                raise ValueError("State path does not exist.")

            # read state only once
            logger.debug("Read state")
            state: bytes
            with open(self.model.state_path, "rb") as f_state:
                state = f_state.read()
            logger.debug("Set remote state")
            for simulator in self.simulators:
                simulator.set_model.remote(state)
            self.set_timecourse_selections(self.r.selections)

    def set_model(self, model):
        """Set model for simulator and updates the integrator settings."""
        logger.debug("SimulatorSerial.set_model")
        if model is None:
            self.model = None
        else:
            # if isinstance(model, AbstractModel):
            #     self.model = model
            # else:
            # handle path, urn, ...

            # FIXME: this is probably the issue
            self.model = RoadrunnerSBMLModel(
                source=model, settings=self.integrator_settings
            )

            self.set_integrator_settings(**self.integrator_settings)


    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set the timecourse selections."""
        logger.info(f"'set_timecourse_selections':{selections}")
        try:
            if selections is None:
                r_model: roadrunner.ExecutableModel = self.r.model
                self.r.timeCourseSelections = (
                    ["time"]
                    + r_model.getFloatingSpeciesIds()
                    + r_model.getBoundarySpeciesIds()
                    + r_model.getGlobalParameterIds()
                    + r_model.getReactionIds()
                    + r_model.getCompartmentIds()
                )
                self.r.timeCourseSelections += [
                    f"[{key}]"
                    for key in (
                        r_model.getFloatingSpeciesIds()
                        + r_model.getBoundarySpeciesIds()
                    )
                ]
            else:
                self.r.timeCourseSelections = selections
        except RuntimeError as err:
            logger.error(f"{err}")
            raise (err)


    def set_integrator_settings(self, **kwargs):
        """Set settings in the integrator."""
        if isinstance(self.model, RoadrunnerSBMLModel):
            RoadrunnerSBMLModel.set_integrator_settings(self.model.r, **kwargs)
        else:
            logger.warning(
                "Integrator settings can only be set on RoadrunnerSBMLModel."
            )

    def set_timecourse_selections(self, selections):
        """Set timecourse selection in model."""
        logger.debug(f"set_timecourse_selections: {selections}")
        RoadrunnerSBMLModel.set_timecourse_selections(self.r, selections=selections)

    @property
    def r(self) -> roadrunner.ExecutableModel:
        """Get the RoadRunner model."""
        return self.model._model

    @property
    def uinfo(self) -> UnitsInformation:
        """Get model unit information."""
        return self.model.uinfo

    @property
    def Q_(self) -> Quantity:
        """Get model unit information."""
        return self.model.uinfo.ureg.Quantity

    def set_timecourse_selections(self, selections):
        """Set timecourse selection in model."""
        logger.debug(f"set_timecourse_selections: {selections}")
        RoadrunnerSBMLModel.set_timecourse_selections(self.r, selections=selections)


    def set_integrator_settings(self, **kwargs):
        """Set settings in the integrator."""
        if isinstance(self.model, RoadrunnerSBMLModel):
            RoadrunnerSBMLModel.set_integrator_settings(self.model.r, **kwargs)
        else:
            logger.warning(
                "Integrator settings can only be set on RoadrunnerSBMLModel."
            )

    @property
    def r(self) -> roadrunner.ExecutableModel:
        """Get the RoadRunner model."""
        return self.model._model

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

            # debug model state
            # FIXME: report issue
            # sbml_str = self.r.getCurrentSBML()
            # with open("/home/mkoenig/git/pkdb_models/pkdb_models/models/dextromethorphan/results/debug/tests.xml", "w") as f_out:
            #     f_out.write(sbml_str)

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
