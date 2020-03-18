from typing import List, Dict
from pathlib import Path
import logging
import roadrunner
import libsbml
import pandas as pd
import numpy as np

from pint import UnitRegistry

from sbmlsim.models.model_resources import Source
from sbmlsim.models.model import AbstractModel
from sbmlsim.units import Units
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
    def load_roadrunner_model(cls, source: Source, selections: List[str] = None) -> roadrunner.RoadRunner:
        """ Loads model from given source

        :param source: path to SBML model or SBML string
        :param selections: boolean flag to set selections
        :return: roadrunner instance
        """

        # load model
        if source.is_path():
            logging.info(f"Load model: '{source.path}'")
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
    def copy_model(cls, r: roadrunner.RoadRunner) -> roadrunner.RoadRunner:
        """Copy current model.

        :param r:
        :return:
        """
        # independent copy by parsing SBML
        sbml_str = r.getCurrentSBML()  # type: str
        r_copy = roadrunner.RoadRunner(sbml_str)

        # copy state of instance
        cls.copy_model_state(r_from=r, r_to=r_copy)
        return r_copy

    @classmethod
    @deprecated
    def copy_model_state(cls, r_from: roadrunner.RoadRunner, r_to: roadrunner.RoadRunner,
                         copy_selections=True,
                         copy_integrator=True,
                         copy_states=True):
        """ Copy roadrunner state between model instances

        :param r_from:
        :param r_to:
        :return:
        """
        if copy_selections:
            # copy of selections (by value)
            r_to.timeCourseSelections = r_from.timeCourseSelections
            r_to.steadyStateSelections = r_from.steadyStateSelections

        if copy_integrator:
            # copy integrator state
            integrator = r_from.getIntegrator()  # type: roadrunner.Integrator
            integrator_name = integrator.getName()
            r_to.setIntegrator(integrator_name)

            settings_keys = integrator.getSettings()  # type: Tuple[str]
            print(settings_keys)
            for key in settings_keys:
                r_to.integrator.setValue(key, integrator.getValue(key))

        if copy_states:
            # FIXME: implement: copying of current state to initial state
            # for state variables
            pass

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

    @classmethod
    @deprecated
    def reset_all(cls, r):
        """ Reset all model variables to CURRENT init(X) values.

        This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
        parameters back to the values they had when the model was first loaded.
        """
        r.reset(roadrunner.SelectionRecord.TIME |
                roadrunner.SelectionRecord.RATE |
                roadrunner.SelectionRecord.FLOATING |
                roadrunner.SelectionRecord.GLOBAL_PARAMETER)


if __name__ == "__main__":
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR
    model = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)
    print(model)