"""
Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.

"""

# FIXME: general model and model changes


from pathlib import Path
from typing import List, Tuple, Dict
import warnings
import abc
from

from sbmlsim.models import biomodels


MODEL_CHANGE_BOUNDARY_CONDITION = "boundary_condition"


class ModelChange(object):
    # TODO: encode all the possible model changes which can be performed with
    # a model. These are reused in the setting up of models in the beginning,
    # or later on in simulations
    pass


class Model(object):

    def __init__(self, sid: str, source: str, language: str=None, changes: Dict=None,
                 name: str=None):
        """

        :param sid:
        :param source: resolvable absolute path, no support for relative paths
        :param language:
        :param changes:
        :param name:
        """

        if not language or len(language) == 0:
            warnings.warn(f"No model language specified, defaulting to SBML for: '{source}'")
            language = "sbml"

        self.sid = sid
        self.name = name
        self.language = language
        self.source = source

        if changes is None:
            changes = {}
        self.changes = changes
        self.model = None



        # read CellML
        if 'cellml' in language:
            warnings.warn("CellML model encountered, sbmlsim does not support CellML".format(language))
            raise ValueError("CellML models not supported yet")
        # other
        else:
            warnings.warn("Unsupported model language: '{}'.".format(language))

    @abc.abstractmethod
    def load_model(self):
        """Loads the model from the given source information."""
        return

    @abc.abstractmethod
    def apply_change(self, model, change):
        return

    def apply_model_changes(self):
        # apply model changes
        for change in self.model_changes[mid]:
            self._apply_model_change(model, change)


class SBMLModel(Model):
    """An SBML model."""

    def resolve_sbml(self):
        def is_urn():
            return self.source.lower().startswith('urn')

        def is_http():
            return self.source.lower().startswith('http') or source.startswith('HTTP')

            # read SBML

        if 'sbml' in language or len(language) == 0:
            sbml_str = None
            if is_urn():
                sbml_str = biomodels.sbml_from_biomodels_urn(source)
            elif is_http():
                sbml_str = biomodels.sbml_from_biomodels_url(source)
            if sbml_str:
                model = load_model(sbml_str)
            else:
                # load file, by resolving path relative to working dir
                # FIXME: support absolute paths?
                sbml_path = self.working_dir / source
                model = load_model(sbml_path)
