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
    def __init__(self, dimension: str, index: np.ndarray=None, changes: Dict=None):
        if index is None and changes is None:
            raise ValueError("Either 'index' or 'changes' required for Dimension.")
        self.dimension = dimension  # type: str

        if changes is None:
            changes = {}
        self.changes = changes
        if index is None:
            # figure out index from changes
            num = 1
            for key, values in changes.items():
                if isinstance(values, Iterable):
                    n = len(values)
                    if num != 1 and num != n:
                        raise ValueError(f"All changes must have same length: '{changes}'")
                    num = n
            index = np.arange(num)
        self.index = index

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
