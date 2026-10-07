"""Models.

Functions for model loading, model manipulation and settings on the integrator.
Model can be in different formats, main supported format being SBML.

Other formats could be supported like CellML or NeuroML.
"""

import logging
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from sbmlsim.model.model_resources import Source
from sbmlsim.units import UnitsInformation

logger = logging.getLogger(__name__)


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
        source: str | Path,
        sid: str | None = None,
        name: str | None = None,
        language: str | None = None,
        language_type: LanguageType | None = None,
        base_path: Path | None = None,
        changes: dict | None = None,
        selections: list[str] | None = None,
        parameters: Mapping[str, float] | None = None,
    ):
        """Initialize the model description.

        Args:
            source: path, URN or URL of the model, or the SBML itself.
            sid: id of the model.
            name: name of the model.
            language: language of the model, e.g. `sbml`.
            language_type: language of the model as `LanguageType`.
            base_path: directory a relative path is resolved against.
            changes: changes of the model, applied before the initialization
                of every simulation of it.
            selections: selections of the simulations.
            parameters: constant parameters which are added to the model, by
                their id and value, e.g. the parameters of the parameter
                table of a PEtab problem which are not entities of the model.
        """
        if not language and language_type is None:
            # SBML is the default language
            language_type = AbstractModel.LanguageType.SBML
        if language and language_type is not None:
            raise ValueError(
                "Either 'language' or 'language_type' can be set, but not both."
            )

        # parse language_type
        if language and isinstance(language, str):
            if "sbml" in language:
                language_type = AbstractModel.LanguageType.SBML
            else:
                raise ValueError(f"Unsupported model language: '{language}'")

        if language_type is None:
            raise ValueError("Either 'language' or 'language_type' is required.")
        self.sid = sid
        self.name = name
        self.language = language
        self.language_type: AbstractModel.LanguageType = language_type
        self.base_path = base_path
        self.source: Source = Source.from_source(source, base_dir=base_path)

        if changes is None:
            changes = {}
        self.changes = changes
        self.selections = selections
        self.parameters: dict[str, float] = dict(parameters or {})

        # normalize parameters at end of initialization

    def normalize(self, uinfo: UnitsInformation):
        """Normalize values to model units for all changes."""
        self.changes = UnitsInformation.normalize_changes(self.changes, uinfo=uinfo)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "sid": self.sid,
            "name": self.name,
            "language": self.language_type,
            "language_type": self.language_type,
            "source": self.source.to_dict(),
            "changes": self.changes,
            "parameters": self.parameters,
        }
