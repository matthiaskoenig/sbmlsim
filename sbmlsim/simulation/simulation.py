"""
Abstract base simulation.
"""
from typing import List, Dict, Iterable
import abc
from abc import ABC
import numpy as np


class Dimension(object):
    """Defines dimension for a simulation or scan.
    The dimension defines how the dimension is called,
    the index is the corresponding index of the dimension.
    """
    def __init__(self, dimension: str, index: np.ndarray, changes: Dict):
        self.dimension = dimension  # type: str
        self.index = index
        self.changes = changes

    def __repr__(self):
        return f"Dim({self.dimension}({len(self)}), " \
               f"{list(self.changes.keys())})"

    def __len__(self):
        return len(self.index)


class AbstractSim(ABC):

    @abc.abstractmethod
    def dimensions(self) -> List[Dimension]:
        """Returns the dimensions of the simulation."""
        pass

    @abc.abstractmethod
    def normalize(self, udict, ureg):
        pass

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            'type': self.__class__.__name__,
        }
        return d
