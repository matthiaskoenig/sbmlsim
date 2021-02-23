"""Abstract base simulation."""
import abc
import itertools
from abc import ABC
from typing import Any, Dict, Iterable, List

import numpy as np


class Dimension(object):
    """Define dimension for a simulation or scan.

    The dimension defines how the dimension is called,
    the index is the corresponding index of the dimension.
    """

    def __init__(self, dimension: str, index: np.ndarray = None, changes: Dict = None):
        """Dimension.

        If no index is provided the index is calculated from the changes.
        So in most cases the index can be left empty (e.g., for scanning of
        parameters).

        :param dimension: unique id of dimension, should start with 'dim'
        :param index: index for values in dimension
        :param changes: changes to apply.
        """
        if index is None and changes is None:
            raise ValueError("Either 'index' or 'changes' required for Dimension.")
        self.dimension = dimension  # type: str

        if changes is None:
            changes = {}
        self.changes = changes
        if index is None:
            # figure out index from changes
            num = 1
            for values in changes.values():
                if isinstance(values, Iterable):
                    n = len(values)
                    if num != 1 and num != n:
                        raise ValueError(
                            f"All changes must have same length: '{changes}'"
                        )
                    num = n
            index = np.arange(num)
        self.index = index

    def __repr__(self) -> str:
        """Get representation."""
        return f"Dim({self.dimension}({len(self)}), " f"{list(self.changes.keys())})"

    def __len__(self) -> int:
        """Get length."""
        return len(self.index)

    @staticmethod
    def indices_from_dimensions(dimensions: List["Dimension"]):
        """Get indices of all combinations of dimensions."""
        index_vecs = [dim.index for dim in dimensions]
        return list(itertools.product(*index_vecs))


class AbstractSim(ABC):
    """AbstractSim.

    Base class of simulations.
    """

    @abc.abstractmethod
    def dimensions(self) -> List[Dimension]:
        """Get dimension of the simulation."""
        pass

    @abc.abstractmethod
    def normalize(self, udict: Dict[str, Any], ureg) -> None:
        """Normalize simulation."""
        pass

    @abc.abstractmethod
    def add_model_changes(self, changes: Dict) -> None:
        """Add model changes to model."""
        pass

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        d = {
            "type": self.__class__.__name__,
        }
        return d
