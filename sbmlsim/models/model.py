"""
Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.

"""
from typing import List, Tuple, Dict
import logging
import abc

logger = logging.getLogger(__name__)


class ModelChange(object):
    MODEL_CHANGE_BOUNDARY_CONDITION = "boundary_condition"

    def __init__(self):
        # TODO: encode all the possible model changes which can be performed with
        # a model. These are reused in the setting up of models in the beginning,
        # or later on in simulations
        raise NotImplementedError


class Model(object):
    """Abstract base class to store a model in sbmlsim.

    Depending on the model language different subclasses are implemented.
    """
    MODEL_LANGUAGE_SBML = "sbml"
    MODEL_LANGUAGE_CELLML = "cellml"

    def __init__(self, mid: str, source: str, language: str = None, changes: Dict = None,
                 name: str = None):
        """

        Currently only absolute paths are supported in path.
        FIXME: relative paths.

        :param mid: model id
        :param source: path string or urn string
        :param language:
        :param changes:
        :param name:
        """
        if not language or len(language) == 0:
            logger.warning(f"No model language specified, defaulting to "
                           f"SBML for: '{source}'")
            language = Model.MODEL_LANGUAGE_SBML
        if 'sbml' not in language:
            logger.warning(f"Unsupported model language: '{language}'")

        self.sid = mid
        self.name = name
        self.language = language
        self.source = source

        if changes is None:
            changes = {}
        self.changes = changes

        self._model = None  # field for loaded model with changes
        self.load_model()

    @abc.abstractmethod
    def load_model(self):
        """Loads the model from the given source information."""
        return

    @abc.abstractclassmethod
    def apply_change(cls, model, change):
        """Applies change to model"""
        return

    def apply_model_changes(self, changes):
        """Applies dictionary of model changes."""
        for change in self.changes:
            Model.apply_change(self._model, change)
