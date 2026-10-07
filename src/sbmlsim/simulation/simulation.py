"""Abstract base simulation of `Timecourse` and `TimecourseSim`.

Removed with them, see `sbmlsim.simulation.definition` for the simulation
which replaces them.
"""

import abc
import logging
from abc import ABC
from typing import Any

from sbmlsim.units import UnitsInformation

logger = logging.getLogger(__name__)


class AbstractSim(ABC):
    """AbstractSim.

    Base class of simulations.
    """

    @abc.abstractmethod
    def normalize(self, uinfo: UnitsInformation) -> None:
        """Normalize simulation."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_model_changes(self, model_changes: dict[str, Any]) -> None:
        """Add model changes to model."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.__class__.__name__,
        }
