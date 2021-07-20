import itertools
from typing import Dict, Iterable, List, Union, Tuple

import numpy as np
from sbmlsim.simulation.base import BaseObject
from sbmlsim.console import console

# from rich import pretty
# pretty.install()
# from rich import print

# FORMAT = "%(message)s"
# logging.basicConfig(
#     level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
# )
#
# log = logging.getLogger("rich")
# log.info("Hello, World!")



# FIXME: common method to access data

class Range(BaseObject):
    """Range class.

    The Range class is the base class for the different types of ranges,
    i.e. UniformRange, VectorRange, FunctionalRange, and DataRange.
    """

    def __init__(self, sid: str, name: str = None):
        """Construct Range."""
        super(Range, self).__init__(sid=sid, name=name)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Range({self.sid}, {self.name}"


class VectorRange(Range):
    """VectorRange class.

    The VectorRange describes an ordered collection of real values, listing
    them explicitly within child value elements.
    """

    def __init__(self, sid: str, values: Union[List, Tuple, np.ndarray], name: str = None,):
        """Construct VectorRange."""
        super(VectorRange, self).__init__(sid=sid, name=name)
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if not isinstance(values, np.ndarray):
            raise ValueError(f"'values' in VectorRange must be numpy.ndarray, but '{type(values)}' for "
                             f"'{values}'")
        if values.ndim != 1:
            raise ValueError(
                f"'values' in VectorRange must be one-dimensional, but ndim='{values.ndim}' for "
                f"'{values}'")

        # values are sorted
        values_sorted = np.sort(values)
        if not np.allclose(values, values_sorted):
            console.log(
                f"'values' in VectorRange must be one-dimensional, but ndim='{values.ndim}' for "
                f"'{values}'")
        self.values: np.ndarray = values.sort()

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Range(sid={self.sid}, name={self.name})"

# TODO: implement
# @dataclass
# class UniformRange(Range):
#     start: float
#     end: float
#     steps: float
#     type: str  # linear/log
#
#
# @dataclass
# class DataRange(Range):
#     source: str
#
# @dataclass
# class FunctionalRange(Range):
#     range: Range
#     variables: List   # model variables
#     parameters: List
#     math: str



class Dimension:
    """Define dimension for a scan.

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
        self.dimension: str = dimension

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


if __name__ == "__main__":
    console.rule("[bold red]Range examples")
    vrange = VectorRange(sid="range1", values=[0, 2, 3])
    console.log(vrange)
    import time
    with console.status("Working..."):
        time.sleep(2)

