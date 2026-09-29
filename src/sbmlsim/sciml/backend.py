"""The backends the layers and functions of a network are written against.

A layer is implemented once. It uses the array operations of numpy (`@`,
`reshape`, `sum`, `concatenate`), which work on arrays of numbers and on
`object` arrays of expressions alike, and takes everything which differs
between the two from a `Backend`: the elementwise functions and the functions
with a condition.

`NumpyBackend` works on `float` arrays and is the forward pass. A backend on
sympy expressions is the first half of the compilation of a network into an
SBML model; it subclasses `Backend`, sets `kind` to `BackendKind.SYMPY` and
implements the abstract methods, the layers do not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from scipy import special


class BackendKind(StrEnum):
    """The kinds of backends a layer declares its support for."""

    NUMPY = "numpy"
    SYMPY = "sympy"


#: the backends of a layer which only uses the methods of `Backend`
ALL_BACKENDS: frozenset[BackendKind] = frozenset(BackendKind)

#: the backends of a layer which needs numbers, e.g. to find a maximum
NUMPY_ONLY: frozenset[BackendKind] = frozenset({BackendKind.NUMPY})


class Backend(ABC):
    """The elementwise functions and the array type of a forward pass.

    Every method takes and returns arrays of the `dtype` of the backend and
    works elementwise, with the broadcasting of numpy.

    Attributes:
        kind: the kind of the backend, which a layer declares its support for.
        dtype: the dtype of the arrays, `float` or `object`.
    """

    kind: ClassVar[BackendKind]
    dtype: ClassVar[type]

    def asarray(self, values: Any) -> np.ndarray:
        """Convert values to an array of the backend.

        Args:
            values: an array, a nested sequence or a scalar.

        Returns:
            The array with the dtype of the backend.
        """
        return np.asarray(values, dtype=self.dtype)

    @abstractmethod
    def exp(self, x: np.ndarray) -> np.ndarray:
        """Calculate the exponential function."""

    @abstractmethod
    def log(self, x: np.ndarray) -> np.ndarray:
        """Calculate the natural logarithm."""

    @abstractmethod
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hyperbolic tangent."""

    @abstractmethod
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the square root."""

    @abstractmethod
    def erf(self, x: np.ndarray) -> np.ndarray:
        """Calculate the error function."""

    @abstractmethod
    def absolute(self, x: np.ndarray) -> np.ndarray:
        """Calculate the absolute value."""

    @abstractmethod
    def select(
        self,
        x: np.ndarray,
        threshold: float,
        below: np.ndarray | float,
        above: np.ndarray | float,
    ) -> np.ndarray:
        """Choose between two values by a condition on `x`.

        This is the one function with a condition, every piecewise activation
        is written with it.

        Args:
            x: the values the condition is evaluated on.
            threshold: the threshold of the condition.
            below: the result where `x <= threshold`.
            above: the result where `x > threshold`.

        Returns:
            `above` where `x > threshold` and `below` elsewhere.
        """

    @abstractmethod
    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
        """Get a shift which keeps the exponentials of `softmax` finite.

        `softmax` does not change when a value which is constant along `axis`
        is subtracted from `x`. A backend on numbers returns the maximum along
        the axis, a backend on expressions returns zero.

        Args:
            x: the input of `softmax`.
            axis: the axis `softmax` normalizes over.

        Returns:
            The shift, which broadcasts against `x`.
        """


class NumpyBackend(Backend):
    """The backend on arrays of numbers, i.e. the forward pass."""

    kind: ClassVar[BackendKind] = BackendKind.NUMPY
    dtype: ClassVar[type] = float

    def exp(self, x: np.ndarray) -> np.ndarray:
        """Calculate the exponential function.

        An overflow is `inf` and not a warning: the branch of a `select` which
        is not chosen is evaluated as well.
        """
        with np.errstate(over="ignore"):
            return np.exp(x)

    def log(self, x: np.ndarray) -> np.ndarray:
        """Calculate the natural logarithm."""
        return np.log(x)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hyperbolic tangent."""
        return np.tanh(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the square root."""
        return np.sqrt(x)

    def erf(self, x: np.ndarray) -> np.ndarray:
        """Calculate the error function."""
        return special.erf(x)

    def absolute(self, x: np.ndarray) -> np.ndarray:
        """Calculate the absolute value."""
        return np.abs(x)

    def select(
        self,
        x: np.ndarray,
        threshold: float,
        below: np.ndarray | float,
        above: np.ndarray | float,
    ) -> np.ndarray:
        """Choose between two values by a condition on `x`."""
        return np.where(x > threshold, above, below)

    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
        """Get the maximum along the axis."""
        return np.max(x, axis=axis, keepdims=True)
