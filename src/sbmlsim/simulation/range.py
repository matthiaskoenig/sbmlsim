"""The dimensions of a scan."""

import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np


class Dimension:
    """Define dimension for a scan.

    The dimension defines how the dimension is called,
    the index is the corresponding index of the dimension.
    """

    def __init__(
        self,
        dimension: str,
        index: np.ndarray | None = None,
        changes: dict[str, Any] | None = None,
        at: Any = None,
    ):
        """Dimension.

        If no index is provided the index is calculated from the changes.
        The values of a dimension replace the values of their targets wherever
        the scanned simulation sets them; a dimension with `at` applies them
        as a `Change` at that time instead.
        So in most cases the index can be left empty (e.g., for scanning of
        parameters).

        :param dimension: unique id of dimension, should start with 'dim'
        :param index: index for values in dimension
        :param changes: changes to apply.
        :param at: time of the change of the values, a number in the time
            unit of the simulation or a quantity; `None` applies them where
            the simulation sets them or before the initialization.
        """
        if index is None and changes is None:
            raise ValueError("Either 'index' or 'changes' required for Dimension.")
        self.dimension: str = dimension
        self.at: Any = at

        if changes is None:
            changes = {}
        self.changes: dict[str, Any] = changes
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
        self.index: np.ndarray = index

    def __repr__(self) -> str:
        """Get representation."""
        return f"Dim({self.dimension}({len(self)}), {list(self.changes.keys())})"

    def __len__(self) -> int:
        """Get length."""
        return len(self.index)

    @staticmethod
    def indices_from_dimensions(dimensions: list["Dimension"]) -> list[tuple[Any, ...]]:
        """Get indices of all combinations of dimensions."""
        index_vecs = [dim.index for dim in dimensions]
        return list(itertools.product(*index_vecs))
