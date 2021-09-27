"""Range classes."""

import itertools
from enum import Enum, auto, unique
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

from sbmlsim.console import console
from sbmlsim.simulation.base import BaseObject


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
        self._values: np.ndarray

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Range({self.sid}, {self.name}"

    @property
    def values(self) -> np.ndarray:
        """Get values of the range."""
        return self._values

    @values.setter
    def values(self, data: np.ndarray):
        """Set values for range."""
        if not isinstance(data, np.ndarray):
            raise ValueError(
                f"'data' in Range must be numpy.ndarray, but '{type(data)}' for "
                f"'{data}'"
            )


class VectorRange(Range):
    """VectorRange class.

    The VectorRange describes an ordered collection of real values, listing
    them explicitly within child value elements.
    """

    def __init__(
        self,
        sid: str,
        values: Union[List, Tuple, np.ndarray],
        name: str = None,
    ):
        """Construct VectorRange."""
        super(VectorRange, self).__init__(sid=sid, name=name)
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' in VectorRange must be numpy.ndarray, but '{type(values)}' for "
                f"'{values}'"
            )
        if values.ndim != 1:
            raise ValueError(
                f"'values' in VectorRange must be one-dimensional, but ndim='{values.ndim}' for "
                f"'{values}'"
            )

        # values are sorted
        values_sorted: np.ndarray = np.sort(values)
        if not np.allclose(values, values_sorted):
            console.log(
                f"'values' in VectorRange must be one-dimensional, but ndim='{values.ndim}' for "
                f"'{values}'"
            )
        self.values: np.ndarray = values_sorted

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Range(sid={self.sid}, name={self.name})"


class UniformRangeType(Enum):
    """UniformRangeType.

    Attribute type that can take the values linear or log.
    Determines whether to draw the values logarithmically (with a base of 10) or linearly.
    """

    linear = auto()
    log = auto()


class UniformRange(Range):
    """UniformRange class.

    The UniformRange on the preceding page) allows the definition of a Range with uniformly spaced values.
    The range_type determines whether to draw the values logarithmically (with a base of 10) or linearly.

    """

    def __init__(
        self,
        sid: str,
        start: float,
        end: float,
        steps: int,
        range_type: UniformRangeType,
        name: str = None,
    ):
        """Construct VectorRange."""
        super(UniformRange, self).__init__(sid=sid, name=name)
        self.start: float = start
        self.end: float = end
        self.steps: int = steps
        self.range_type: UniformRangeType = range_type

        if self.range_type == UniformRangeType.linear:
            self.values = np.linspace(start=start, end=end, num=steps + 1)
        elif self.range_type == UniformRangeType.log:
            # In linear space, the sequence starts at ``base ** start``
            # (`base` to the power of `start`) and ends with ``base ** stop
            self.values = np.logspace(
                start=np.log10(self.start),
                end=np.log10(self.end),
                num=self.steps + 1,
                base=10,
            )
        else:
            raise ValueError(
                f"Unsupported range type in UniformRange: '{self.range_type}'"
            )

    def __repr__(self) -> str:
        """Get string representation."""
        return f"UniformRange(sid={self.sid}, name={self.name}, start={self.start}, stop={self.stop}, steps={self.steps})"


class DataRange(Range):
    """DataRange class.

    The DataRange constructs a range by reference to external data.
    The sourceRef must point to a DataDescription with a single dimension,
    whose values are used as the values of the range.
    """

    def __init__(self, sid: str, source_ref: str, name: str = None):
        """Construct DataRange."""
        super(DataRange, self).__init__(sid=sid, name=name)
        self.source_ref: str = source_ref

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"DataRange(sid={self.sid}, name={self.name}, source_ref={self.source_ref})"
        )

    @property
    def values(self) -> np.ndarray:
        """Resolve data from data generator."""

        # FIXME: implement; requires access to the resolved DataDescriptions of the experiment.
        raise NotImplementedError


class FunctionalRange(Range):
    """FunctionalRange class.

    The FunctionalRange constructs a range through calculations that
    determine the next value based on the value(s) of other range(s) or model variables.
    In this it is similar to the ComputeChange element, and shares some of the same
    child elements (but is not a subclass of ComputeChange).
    """

    def __init__(self, sid: str, range: str, name: str = None):
        """Construct DataRange."""
        super(DataRange, self).__init__(sid=sid, name=name)
        # self.source_ref: str = source_ref  # FIXME

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"DataRange(sid={self.sid}, name={self.name}, source_ref={self.source_ref})"
        )

    @property
    def values(self) -> np.ndarray:
        """Resolve data from data generator."""

        # FIXME: implement; requires access to the resolved DataDescriptions of the experiment.
        raise NotImplementedError


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
    vrange1 = VectorRange(sid="range1", values=[0, 2, 3])
    vrange2 = VectorRange(sid="range2", values=np.linspace(start=0, stop=10, num=10))

    console.log(vrange1)
    console.log(vrange2)
    # import time
    # with console.status("Working..."):
    #     time.sleep(2)
