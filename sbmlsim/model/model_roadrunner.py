from typing import List, Dict
from pathlib import Path
import logging
import roadrunner
import libsbml
import pandas as pd
import numpy as np
import tempfile

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.units import Units, UnitRegistry
from sbmlsim.utils import deprecated

logger = logging.getLogger(__name__)


class RoadrunnerSBMLModel(AbstractModel):
    """Roadrunner model wrapper."""

    def __init__(self, source: str, base_path: Path = None,
                 changes: Dict = None,
                 sid: str = None, name: str = None,
                 selections: List[str] = None,
                 ureg: UnitRegistry = None,
                 ):
        super(RoadrunnerSBMLModel, self).__init__(
            source=source,
            language_type=AbstractModel.LanguageType.SBML,
            changes=changes,
            sid=sid,
            name=name,
            base_path=base_path,
            selections=selections,
            ureg=ureg
        )

    def load_model(self):
        """Loads the model from the given source information."""
        if self.language_type != AbstractModel.LanguageType.SBML:
            raise ValueError(f"{self.__class__.__name__} only supports "
                             f"language_type '{AbstractModel.LanguageType.SBML}'.")

        r = RoadrunnerSBMLModel.load_roadrunner_model(
            self.source, selections=self.selections)
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

    @classmethod
    def load_roadrunner_model(cls, source: Source, selections: List[str] = None) -> roadrunner.RoadRunner:
        """ Loads model from given source

        :param source: path to SBML model or SBML string
        :param selections: boolean flag to set selections
        :return: roadrunner instance
        """
        if isinstance(source, (str, Path)):
            source = Source.from_source(source=source)

        # load model
        if source.is_path():
            logging.info(f"Load model: '{source.path.resolve()}'")
            r = roadrunner.RoadRunner(str(source.path))
        elif source.is_content():
            r = roadrunner.RoadRunner(str(source.content))

        # set selections
        cls.set_timecourse_selections(r, selections)
        return r

    def parse_units(self, ureg):
        """Parse units from SBML model"""
        if self.source.is_content():
            model_path = self.source.content
        elif self.source.is_path():
            model_path = self.source.path

        return Units.get_units_from_sbml(model_path, ureg)

    @classmethod
    def apply_change(cls, model, change):
        """Applies change to model"""
        return

    @classmethod
    def set_timecourse_selections(cls, r: roadrunner.RoadRunner,
                                  selections: List[str] = None) -> None:
        """ Sets the full model selections. """
        if not selections:
            r_model = r.model  # type: roadrunner.ExecutableModel

            r.timeCourseSelections = ["time"] \
                                     + r_model.getFloatingSpeciesIds() \
                                     + r_model.getBoundarySpeciesIds() \
                                     + r_model.getGlobalParameterIds() \
                                     + r_model.getReactionIds() \
                                     + r_model.getCompartmentIds()
            r.timeCourseSelections += [f'[{key}]' for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds())]
        else:
            r.timeCourseSelections = selections

    @staticmethod
    def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """
        Create GlobalParameter DataFrame.
        :return: pandas DataFrame
        """
        r_model = r.model  # type: roadrunner.ExecutableModel
        doc = libsbml.readSBMLFromString(
            r.getCurrentSBML())  # type: libsbml.SBMLDocument
        model = doc.getModel()  # type: libsbml.Model
        sids = r_model.getGlobalParameterIds()
        parameters = [model.getParameter(sid) for sid in
                      sids]  # type: List[libsbml.Parameter]
        data = {
            'sid': sids,
            'value': r_model.getGlobalParameterValues(),
            'unit': [p.units for p in parameters],
            'constant': [p.constant for p in parameters],
            'name': [p.name for p in parameters],
        }
        df = pd.DataFrame(data,
                          columns=['sid', 'value', 'unit', 'constant', 'name'])
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
        species = [model.getSpecies(sid) for sid in
                   sids]  # type: List[libsbml.Species]

        data = {
            'sid': sids,
            'concentration': np.concatenate([
                r_model.getFloatingSpeciesConcentrations(),
                r_model.getBoundarySpeciesConcentrations()
            ], axis=0),
            'amount': np.concatenate([
                r.model.getFloatingSpeciesAmounts(),
                r.model.getBoundarySpeciesAmounts()
            ], axis=0),
            'unit': [s.getUnits() for s in species],
            'constant': [s.getConstant() for s in species],
            'boundaryCondition': [s.getBoundaryCondition() for s in species],
            'name': [s.getName() for s in species],
        }

        return pd.DataFrame(data, columns=['sid', 'concentration', 'amount', 'unit',
                                           'constant',
                                           'boundaryCondition', 'species', 'name'])


    @classmethod
    @deprecated
    def clamp_species(cls, r: roadrunner.RoadRunner, sids,
                      boundary_condition=True) -> roadrunner.RoadRunner:
        """ Clamp/Free specie(s) via setting boundaryCondition=True/False.

        This requires changing the SBML and ODE system.

        :param r: roadrunner.RoadRunner
        :param sids: sid or iterable of sids
        :param boundary_condition: boolean flag to clamp (True) or free (False) species
        :return: modified roadrunner.RoadRunner
        """
        # TODO: implement via Roadrunner model changes

        # get model for current SBML state
        sbml_str = r.getCurrentSBML()
        # FIXME: bug in concentrations!

        # print(sbml_str)

        doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
        model = doc.getModel()  # type: libsbml.Model

        if isinstance(sids, str):
            sids = [sids]

        for sid in sids:
            # set boundary conditions
            sbase = model.getElementBySId(sid)  # type: libsbml.SBase
            if not sbase:
                logging.error("No element for SId in model: {}".format(sid))
                return None
            else:
                if sbase.getTypeCode() == libsbml.SBML_SPECIES:
                    species = sbase  # type: libsbml.Species
                    species.setBoundaryCondition(boundary_condition)
                else:
                    logging.error(
                        "SId in clamp does not match species: {}".format(sbase))
                    return None

        # create modified roadrunner instance
        sbmlmod_str = libsbml.writeSBMLToString(doc)
        rmod = cls.load_model(sbmlmod_str)  # type: roadrunner.RoadRunner
        cls.set_timecourse_selections(rmod, r.timeCourseSelections)

        return rmod


if __name__ == "__main__":

    from sbmlsim.tests.constants import MODEL_REPRESSILATOR
    model = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)
    print(model)