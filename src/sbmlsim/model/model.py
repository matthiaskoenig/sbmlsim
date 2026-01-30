"""Models.

Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pymetadata import log

from sbmlsim.model.model_resources import Source
from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


class AbstractModel:
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
        changes: dict = None,
        selections: list[str] = None,
    ):
        """Initialize SourceType."""

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

    def normalize(self, uinfo: UnitsInformation):
        """Normalize values to model units for all changes."""
        self.changes = UnitsInformation.normalize_changes(self.changes, uinfo=uinfo)

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
