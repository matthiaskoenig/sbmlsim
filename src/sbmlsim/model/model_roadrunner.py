"""RoadRunner model."""

import tempfile
from pathlib import Path
from typing import Optional, Union

import libsbml
import numpy as np
import pandas as pd
import roadrunner
from pymetadata import log

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.units import Quantity, UnitRegistry, UnitsInformation
from sbmlsim.utils import md5_for_path


logger = log.get_logger(__name__)


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
        source: Union[str, Path],
        base_path: Path = None,
        changes: dict = None,
        sid: str = None,
        name: str = None,
        selections: list[str] = None,
        ureg: UnitRegistry = None,
        settings: dict = None,
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

        # check SBML
        if self.language_type != AbstractModel.LanguageType.SBML:
            raise ValueError(f"language_type not supported '{self.language_type}'.")

        # load model
        # logger.info("load model")
        self.r: Optional[roadrunner.RoadRunner] = self.load_roadrunner_model(
            source=self.source
        )
        # logger.info(self.r)

        # set selections
        # logger.info("set selections")
        self.selections = self.set_timecourse_selections(
            self.r, selections=self.selections
        )

        # set integrator settings
        # logger.info("set integrator settings")
        if settings:
            RoadrunnerSBMLModel.set_integrator_settings(self.r, **settings)

        # normalize model changes
        self.uinfo = self.parse_units(ureg)
        self.normalize(uinfo=self.uinfo)

    @property
    def Q_(self) -> Quantity:
        """Quantity to create quantities for model changes."""
        return self.uinfo.ureg.Quantity

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] = None,
        ureg: UnitRegistry = None,
        settings: dict = None,
    ):
        """Create from AbstractModel."""
        logger.debug("RoadrunnerSBMLModel from AbstractModel")
        return RoadrunnerSBMLModel(
            source=abstract_model.source.source,
            changes=abstract_model.changes,
            sid=abstract_model.sid,
            name=abstract_model.name,
            base_path=abstract_model.base_path,
            selections=selections,
            ureg=ureg,
            settings=settings,
        )

    @classmethod
    def load_roadrunner_model(
        cls,
        source: Source,
    ) -> roadrunner.RoadRunner:
        """Load model from given source.

        :param source: path to SBML model or SBML string
        :param state_path: path to rr state
        :return: roadrunner instance
        """
        if isinstance(source, (str, Path)):
            source = Source.from_source(source=source)

        # load model
        if source.is_path():
            sbml_path: Path = source.path
            # state_path: Path = RoadrunnerSBMLModel.get_state_path(sbml_path=sbml_path)

            r = roadrunner.RoadRunner(str(sbml_path))
            # FIXME: see https://github.com/sys-bio/roadrunner/issues/963
            # if state_path.exists():
            #     logger.debug(f"Load model from state: '{state_path}'")
            #     r = roadrunner.RoadRunner()
            #     r.loadState(str(state_path))
            #     # with open(state_path, "rb") as fin:
            #     #     r.loadStateS(fin.read())
            #     logger.debug(f"Model loaded from state: '{state_path}'")
            # else:
            #     logger.info(f"Load model from SBML: '{sbml_path}'")
            #     r = roadrunner.RoadRunner(str(sbml_path))
            #     # save state
            #     r.saveState(str(state_path))
            #     # with open(state_path, "wb") as fout:
            #     #     fout.write(r.saveStateS(opt="b"))
            #     logger.info(f"Save state: '{state_path}'")

        elif source.is_content():
            r = roadrunner.RoadRunner(str(source.content))

        return r

    @staticmethod
    def get_state_path(sbml_path: Path) -> Optional[Path]:
        """Get path of the state file.

        The state file is a binary file which allows fast model loading.
        """
        md5 = md5_for_path(sbml_path)
        return Path(f"{sbml_path}_rr{roadrunner.__version__}_{md5}.state")

    @classmethod
    def copy_roadrunner_model(cls, r: roadrunner.RoadRunner) -> roadrunner.RoadRunner:
        """Copy roadrunner model by using the state."""
        ftmp = tempfile.NamedTemporaryFile()
        filename = ftmp.name
        r.saveState(filename)
        r2 = roadrunner.RoadRunner()
        r2.loadState(filename)
        return r2

    def parse_units(self, ureg: UnitRegistry) -> UnitsInformation:
        """Parse units from SBML model."""
        uinfo: UnitsInformation
        if self.source.is_content():
            uinfo = UnitsInformation.from_sbml(sbml=self.source.content, ureg=ureg)
        elif self.source.is_path():
            uinfo = UnitsInformation.from_sbml(sbml=self.source.path, ureg=ureg)

        return uinfo

    @classmethod
    def set_timecourse_selections(
        cls, r: roadrunner.RoadRunner, selections: list[str] = None
    ) -> list[str]:
        """Set the model selections for timecourse simulation."""
        if selections is None:
            r_model: roadrunner.ExecutableModel = r.model

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
    def set_integrator_settings(
        r: roadrunner.RoadRunner, **kwargs
    ) -> roadrunner.Integrator:
        """Set integrator settings.

        Keys are:
            variable_step_size [boolean]
            stiff [boolean]
            absolute_tolerance [float]
            relative_tolerance [float]

        """
        integrator: roadrunner.Integrator = r.getIntegrator()
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
                compartment_values = r.model.getCompartmentVolumes()
                if len(compartment_values) > 0:
                    value = min(value, value * min(compartment_values))

            integrator.setValue(key, value)
            logger.debug(f"Integrator setting: '{key} = {value}'")
        return integrator

    @staticmethod
    def set_default_settings(r: roadrunner.RoadRunner, **kwargs):
        """Set default settings of integrator."""
        RoadrunnerSBMLModel.set_integrator_settings(
            r,
            variable_step_size=True,
            stiff=True,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
        )

    @staticmethod
    def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """Create GlobalParameter DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(r.getCurrentSBML())
        model: libsbml.Model = doc.getModel()
        sids = r_model.getGlobalParameterIds()
        parameters: list[libsbml.Parameter] = [model.getParameter(sid) for sid in sids]
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
        """Create FloatingSpecies DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        sbml_str = r.getCurrentSBML()

        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
        model: libsbml.Model = doc.getModel()

        sids = r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
        species: list[libsbml.Species] = [model.getSpecies(sid) for sid in sids]

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
