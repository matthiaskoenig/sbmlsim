import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import libsbml
import numpy as np
import pandas as pd
import roadrunner

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.units import Quantity, UnitRegistry, Units
from sbmlsim.utils import md5_for_path


logger = logging.getLogger(__name__)


class RoadrunnerSBMLModel(AbstractModel):
    """Roadrunner model wrapper."""

    IntegratorSettingKeys = {
        "variable_step_size",
        "stiff",
        "absolute_tolerance",
        "relative_tolerance",
    }

    def __init__(
        self,
        source: str,
        base_path: Path = None,
        changes: Dict = None,
        sid: str = None,
        name: str = None,
        selections: List[str] = None,
        ureg: UnitRegistry = None,
        settings: Dict = None,
    ):
        super(RoadrunnerSBMLModel, self).__init__(
            source=source,
            language_type=AbstractModel.LanguageType.SBML,
            changes=changes,
            sid=sid,
            name=name,
            base_path=base_path,
            selections=selections,
        )
        if self.language_type != AbstractModel.LanguageType.SBML:
            raise ValueError(
                f"{self.__class__.__name__} only supports "
                f"language_type '{AbstractModel.LanguageType.SBML}'."
            )

        # load the model
        self.state_path = self.get_state_path()
        self._model = self.load_roadrunner_model(
            source=self.source, state_path=self.state_path
        )
        # set selections
        self.selections = self.set_timecourse_selections(
            self._model, selections=self.selections
        )

        # set integrator settings
        if settings is not None:
            RoadrunnerSBMLModel.set_integrator_settings(self._model, **settings)

        # every model has its own unit registry (in a simulation experiment one
        # global unit registry per experiment should be used)
        if not ureg:
            ureg = Units.default_ureg()
        self.udict, self.ureg = self.parse_units(ureg)

    @property
    def Q_(self) -> Quantity:
        """Quantity to create quantities for model changes."""
        return self.ureg.Quantity

    @property
    def r(self) -> roadrunner.RoadRunner:
        """Roadrunner instance."""
        return self._model

    def get_state_path(self):
        if self.source.is_path():
            mdf5 = md5_for_path(self.source.path)
            return Path(f"{self.source.path}_rr{roadrunner.__version__}_{mdf5}.state")
        else:
            return None

    @classmethod
    def load_roadrunner_model(
        cls, source: Source, state_path: Path = None
    ) -> roadrunner.RoadRunner:
        """Loads model from given source

        :param source: path to SBML model or SBML string
        :param state_path: path to rr state
        :return: roadrunner instance
        """
        if isinstance(source, (str, Path)):
            source = Source.from_source(source=source)

        # load model
        if source.is_path():
            if state_path and state_path.exists():
                # logging.info(f"Load model from state: '{state_path}'")
                r = roadrunner.RoadRunner()
                r.loadState(str(state_path))
            else:
                logging.warning(f"Load model from SBML: '{source.path.resolve()}'")
                r = roadrunner.RoadRunner(str(source.path))
                # save state path
                if state_path:
                    r.saveState(str(state_path))
                    logging.warning(f"Save state: '{state_path}'")
        elif source.is_content():
            r = roadrunner.RoadRunner(str(source.content))

        return r

    @classmethod
    def copy_roadrunner_model(cls, r: roadrunner.RoadRunner) -> roadrunner.RoadRunner:
        """Copy roadrunner model by using the state.

        :param r:
        :return:
        """
        ftmp = tempfile.NamedTemporaryFile()
        filename = ftmp.name
        r.saveState(filename)
        r2 = roadrunner.RoadRunner()
        r2.loadState(filename)
        return r2

    def parse_units(self, ureg):
        """Parse units from SBML model"""
        if self.source.is_content():
            model_path = self.source.content
        elif self.source.is_path():
            model_path = self.source.path

        return Units.get_units_from_sbml(model_path, ureg)

    @classmethod
    def set_timecourse_selections(
        cls, r: roadrunner.RoadRunner, selections: List[str] = None
    ) -> None:
        """ Sets the full model selections. """
        if selections is None:
            r_model = r.model  # type: roadrunner.ExecutableModel

            r.timeCourseSelections = (
                ["time"]
                + r_model.getFloatingSpeciesIds()
                + r_model.getBoundarySpeciesIds()
                + r_model.getGlobalParameterIds()
                + r_model.getReactionIds()
                + r_model.getCompartmentIds()
            )
            r.timeCourseSelections += [
                f"[{key}]"
                for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
                )
            ]
        else:
            r.timeCourseSelections = selections
        return selections

    @staticmethod
    def set_integrator_settings(r: roadrunner.RoadRunner, **kwargs) -> None:
        """Set integrator settings.

        Keys are:
            variable_step_size [boolean]
            stiff [boolean]
            absolute_tolerance [float]
            relative_tolerance [float]

        """
        integrator = r.getIntegrator()
        for key, value in kwargs.items():
            if key not in RoadrunnerSBMLModel.IntegratorSettingKeys:
                logger.debug(
                    f"Unsupported integrator key for roadrunner " f"integrator: '{key}'"
                )
                continue

            # adapt the absolute_tolerance relative to the amounts
            if key == "absolute_tolerance":
                # special hack to acount for amount and concentration absolute
                # tolerances
                value = min(value, value * min(r.model.getCompartmentVolumes()))
            integrator.setValue(key, value)
            logger.debug(f"Integrator setting: '{key} = {value}'")
        return integrator

    @staticmethod
    def set_default_settings(r: roadrunner.RoadRunner, **kwargs):
        """ Set default settings of integrator. """
        RoadrunnerSBMLModel.set_integrator_settings(
            r,
            variable_step_size=True,
            stiff=True,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
        )

    @staticmethod
    def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """
        Create GlobalParameter DataFrame.
        :return: pandas DataFrame
        """
        r_model = r.model  # type: roadrunner.ExecutableModel
        doc = libsbml.readSBMLFromString(
            r.getCurrentSBML()
        )  # type: libsbml.SBMLDocument
        model = doc.getModel()  # type: libsbml.Model
        sids = r_model.getGlobalParameterIds()
        parameters = [
            model.getParameter(sid) for sid in sids
        ]  # type: List[libsbml.Parameter]
        data = {
            "sid": sids,
            "value": r_model.getGlobalParameterValues(),
            "unit": [p.units for p in parameters],
            "constant": [p.constant for p in parameters],
            "name": [p.name for p in parameters],
        }
        df = pd.DataFrame(data, columns=["sid", "value", "unit", "constant", "name"])
        return df

    @staticmethod
    def species_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """
        Create FloatingSpecies DataFrame.
        :return: pandas DataFrame
        """
        r_model = r.model  # type: roadrunner.ExecutableModel
        sbml_str = r.getCurrentSBML()

        doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
        model = doc.getModel()  # type: libsbml.Model

        sids = r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
        species = [model.getSpecies(sid) for sid in sids]  # type: List[libsbml.Species]

        data = {
            "sid": sids,
            "concentration": np.concatenate(
                [
                    r_model.getFloatingSpeciesConcentrations(),
                    r_model.getBoundarySpeciesConcentrations(),
                ],
                axis=0,
            ),
            "amount": np.concatenate(
                [
                    r.model.getFloatingSpeciesAmounts(),
                    r.model.getBoundarySpeciesAmounts(),
                ],
                axis=0,
            ),
            "unit": [s.getUnits() for s in species],
            "constant": [s.getConstant() for s in species],
            "boundaryCondition": [s.getBoundaryCondition() for s in species],
            "name": [s.getName() for s in species],
        }

        return pd.DataFrame(
            data,
            columns=[
                "sid",
                "concentration",
                "amount",
                "unit",
                "constant",
                "boundaryCondition",
                "species",
                "name",
            ],
        )
