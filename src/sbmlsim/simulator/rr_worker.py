"""Classes for running simulations with SBML models."""
from typing import Iterator, Optional, Any, List

import pandas as pd
from sbmlutils import log

from sbmlsim.model import ModelChange
from sbmlsim.model.rr_model import roadrunner, IntegratorSettingKeys
from sbmlsim.simulation import Timecourse, TimecourseSim

logger = log.get_logger(__name__)


class SimulationWorkerRR:
    """Worker running simulations with roadrunner.

    Implements the timecourse simulation once which can be reused by
    the different simulators.
    """

    def __init__(self):
        """Initialize worker with roadrunner instance.

        Sets the default settings for the integrator.
        """

        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner()
        self.integrator_settings = {
            "absolute_tolerance": 1e-8,
            "relative_tolerance": 1e-8,
            "variable_step_size": False,
            "stiff": True,
        }

    def set_model(self, model_state: str) -> None:
        """Set model for simulator and updates the integrator settings."""
        self.r.loadStateS(model_state)
        self.set_integrator_settings()

    def set_timecourse_selections(self, selections: Optional[Iterator[str]] = None) -> None:
        """Set the timecourse selections.

        If the selections are None the complee selections will be used.

        :raises RuntimeError:
        """
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
            raise err

    def get_timecourse_selections(self) -> List[str]:
        """Get timecourse selections."""
        return self.r.timeCourseSelections

    def set_integrator_settings(self, **kwargs) -> roadrunner.Integrator:
        """Set integrator settings.

        Keys are:
            variable_step_size [boolean]
            stiff [boolean]
            absolute_tolerance [float]
            relative_tolerance [float]

        """
        settings = self.integrator_settings.copy()
        settings.update(kwargs)

        integrator: roadrunner.Integrator = self.r.getIntegrator()
        for key, value in settings.items():
            if key not in IntegratorSettingKeys:
                logger.debug(
                    f"Unsupported integrator key for roadrunner " f"integrator: '{key}'"
                )
                continue

            if key == "absolute_tolerance":
                # hack to handle amount and concentration absolute tolerances
                # for small volumes
                value = min(value, value * min(self.r.model.getCompartmentVolumes()))

            integrator.setValue(key, value)
            logger.debug(f"Integrator setting: '{key} = {value}'")
        return integrator

    def get_integrator_setting(self, key: str) -> Any:
        """Get integrator setting for given key."""
        integrator: roadrunner.Integrator = self.r.getIntegrator()
        return integrator.getSetting(key)



    # @property
    # def uinfo(self) -> UnitsInformation:
    #     """Get model unit information."""
    #     return self.model.uinfo
    #
    # @property
    # def Q_(self) -> Quantity:
    #     """Get model unit information."""
    #     return self.model.uinfo.ureg.Quantity
    #
    # @property
    # def r(self) -> roadrunner.ExecutableModel:
    #     """Get the RoadRunner model."""
    #     return self.model._model


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
