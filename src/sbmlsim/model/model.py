"""Models.

Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.
"""
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sbmlsim.model.model_resources import Source
from sbmlsim.units import Units


logger = logging.getLogger(__name__)


class AbstractModel(object):
    """Abstract base class to store a model in sbmlsim.

    Depending on the model language different subclasses are implemented.
    """

    class LanguageType(Enum):
        """Language types."""

        SBML = 1
        CELLML = 2

    class SourceType(Enum):
        """Source types."""

        PATH = 1
        URN = 2
        URL = 3

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.language_type.name}({self.source.source}, changes={len(self.changes)})"

    def __init__(
        self,
        source: Union[str, Path],
        sid: Optional[str] = None,
        name: Optional[str] = None,
        language: Optional[str] = None,
        language_type: LanguageType = LanguageType.SBML,
        base_path: Optional[Path] = None,
        changes: Dict = None,
        selections: List[str] = None,
    ):

        if not language and not language_type:
            raise ValueError(
                "Either 'language' or 'language_type' argument are required"
            )
        if language and language_type:
            raise ValueError(
                "Either 'language' or 'language_type' can be set, but not both."
            )

        # parse language_type
        if language:
            if isinstance(language, str):
                if "sbml" in language:
                    language_type = AbstractModel.LanguageType.SBML
                else:
                    raise ValueError(f"Unsupported model language: '{language}'")

        self.sid = sid
        self.name = name
        self.language = language
        self.language_type = language_type
        self.base_path = base_path
        self.source: Source = Source.from_source(source, base_dir=base_path)

        if changes is None:
            changes = {}
        self.changes = changes
        self.selections = selections

        # normalize parameters at end of initialization

    def normalize(self, udict, ureg):
        """Normalize values to model units for all changes."""
        self.changes = Units.normalize_changes(self.changes, udict=udict, ureg=ureg)

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "sid": self.sid,
            "name": self.name,
            "language": self.language_type,
            "language_type": self.language_type,
            "source": self.source.to_dict(),
            "changes": self.changes,
        }
        return d

    '''
    # SED-ML HANDLING

    def apply_change(self, target, value):
        """Applies change to model"""
        if target.startswith("/"):
            # xpath expression
            target = AbstractModel._resolve_xpath(self._model, target)


    def apply_model_changes(self, changes):
        """Applies dictionary of model changes."""
        for key, value in self.changes.items():
            AbstractModel.apply_change(target=key, value=value)


    @staticmethod
    def _target_from_xpath(model: 'AbstractModel', xpath: str):
        """ Resolve the target from the xpath expression.

        A single target in the model corresponding to the modelId is resolved.
        Currently, the model is not used for xpath resolution.

        :param xpath: xpath expression.
        :type xpath: str
        :param modelId: id of model in which xpath should be resolved
        :type modelId: str
        :return: single target of xpath expression
        :rtype: Target (namedtuple: id type)
        """
        # TODO: via better xpath expression
        #   get type from the SBML document for the given id.
        #   The xpath expression can be very general and does not need to contain the full
        #   xml path
        #   For instance:
        #   /sbml:sbml/sbml:model/descendant::*[@id='S1']
        #   has to resolve to species.
        # TODO: figure out concentration or amount (from SBML document)
        # FIXME: getting of sids, pids not very robust, handle more cases (rules, reactions, ...)

        Target = namedtuple('Target', 'id type')

        def getId(xpath):
            xpath = xpath.replace('"', "'")
            match = re.findall(r"id='(.*?)'", xpath)
            if (match is None) or (len(match) is 0):
                logger.warn("Xpath could not be resolved: {}".format(xpath))
            return match[0]

        # parameter value change
        if ("model" in xpath) and ("parameter" in xpath):
            return Target(getId(xpath), 'parameter')
        # species concentration change
        elif ("model" in xpath) and ("species" in xpath):
            return Target(getId(xpath), 'concentration')
        # other
        elif ("model" in xpath) and ("id" in xpath):
            return Target(getId(xpath), 'other')
        # cannot be parsed
        else:
            raise ValueError("Unsupported target in xpath: {}".format(xpath))

    @staticmethod
    def set_xpath_value(xpath: str, value: float, model):
        """ Creates python line for given xpath target and value.
        :param xpath:
        :type xpath:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        target = SEDMLParser._resolve_xpath(xpath)
        if target:
            if target.type == "concentration":
                # initial concentration
                expr = f'init([{target.id}])'
            elif target.type == "amount":
                # initial amount
                expr = f'init({target.id})'
            else:
                # other (parameter, flux, ...)
                expr = target.id
            print(f"{expr} = {value}")
            model[expr] = value
        else:
            logger.error(f"Unsupported target xpath: {xpath}")

    @staticmethod
    def selectionFromVariable(var, model):
        """ Resolves the selection for the given variable.

        First checks if the variable is a symbol and returns the symbol.
        If no symbol is set the xpath of the target is resolved
        and used in the selection

        :param var: variable to resolve
        :type var: SedVariable
        :return: a single selection
        :rtype: Selection (namedtuple: id type)
        """
        Selection = namedtuple('Selection', 'id type')

        # parse symbol expression
        if var.isSetSymbol():
            cvs = var.getSymbol()
            astr = cvs.rsplit("symbol:")
            sid = astr[1]
            return Selection(sid, 'symbol')
        # use xpath
        elif var.isSetTarget():
            xpath = var.getTarget()
            target = SEDMLParser._resolveXPath(xpath, model)
            return Selection(target.id, target.type)

        else:
            warnings.warn(f"Unrecognized Selection in variable: {var}")
            return None
    '''
