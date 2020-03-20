"""
Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.

"""
from typing import List, Tuple, Dict
from pathlib import Path
from enum import Enum
import logging
import abc
from sbmlsim.units import Units
from pint import UnitRegistry


from sbmlsim.models.model_resources import Source, resolve_source

logger = logging.getLogger(__name__)


class ModelChange(object):
    MODEL_CHANGE_BOUNDARY_CONDITION = "boundary_condition"

    def __init__(self):
        # TODO: encode all the possible model changes which can be performed with
        # a model. These are reused in the setting up of models in the beginning,
        # or later on in simulations
        raise NotImplementedError


class AbstractModel(object):
    """Abstract base class to store a model in sbmlsim.

    Depending on the model language different subclasses are implemented.
    """
    class LanguageType(Enum):
        SBML = 1
        CELLML = 2

    class SourceType(Enum):
        PATH = 1
        URN = 2
        URL = 3

    def __init__(self, source: str,
                 language: str = None, language_type: LanguageType = None,
                 base_path: Path = None,
                 changes: Dict = None,
                 sid: str = None, name: str = None,
                 selections: List[str] = None,
                 ureg: UnitRegistry = None,
                 ):
        """

        :param mid: model id
        :param source: path string or urn string
        :param language:
        :param changes:
        :param base_path: base path relative to which the sources are resolved
        :param name:
        """
        if not language and not language_type:
            raise ValueError("Either 'language' or 'language_type' argument are"
                             "required")
        if language and language_type:
            raise ValueError("Either 'language' or 'language_type' can be set,"
                             "but not both.")

        # parse language_type
        if language:
            if isinstance(language, str):
                if 'sbml' in language:
                    language_type = AbstractModel.LanguageType.SBML
                else:
                    raise ValueError(f"Unsupported model language: '{language}'")

        self.sid = sid
        self.name = name
        self.language = language
        self.language_type = language_type
        self.base_path = base_path
        self.source = resolve_source(source, base_dir=base_path)  # type: Source

        if changes is None:
            changes = {}
        self.changes = changes

        self.selections = selections

        # load the model
        self._model = self.load_model()  # field for loaded model with changes

        # every model has its own unit registry (in a simulation experiment one
        # global unit registry per experiment should be used)
        if not ureg:
            ureg = Units.default_ureg()
        self.udict, self.ureg = self.parse_units(ureg)
        self.Q_ = self.ureg.Quantity

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "sid": self.sid,
            "name": self.name,
            "language": self.language_type,
            "language_type": self.language_type,
            "source": self.source.to_dict(),
            "changes": self.changes,
        }
        return d

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def parse_units(self, ureg: UnitRegistry):
        """Parses the units from the model"""
        return {}, ureg

    @abc.abstractmethod
    def load_model(self):
        """Loads the model from the current information."""
        return None

    @abc.abstractclassmethod
    def apply_change(cls, model, change):
        """Applies change to model"""
        return

    def apply_model_changes(self, changes):
        """Applies dictionary of model changes."""
        for change in self.changes:
            AbstractModel.apply_change(self._model, change)
