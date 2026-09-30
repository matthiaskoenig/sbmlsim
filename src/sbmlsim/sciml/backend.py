"""The backends the layers and functions of a network are written against.

A layer is implemented once. It uses the array operations of numpy (`@`,
`reshape`, `sum`, `concatenate`), which work on arrays of numbers and on
`object` arrays of expressions alike, and takes everything which differs
between the two from a `Backend`: the elementwise functions and the functions
with a condition.

`NumpyBackend` works on `float` arrays and is the forward pass.
`SympyBackend` works on `object` arrays of sympy expressions and is the first
half of the compilation of a network into an SBML model, see
`sbmlsim.sciml.compiler`. The layers are the same for both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
import sympy
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
    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Get a shift which keeps the exponentials of `softmax` finite.

        `softmax` does not change when a value which is constant along `axis`
        is subtracted from `x`. Both backends return the maximum along the
        axis, a backend on expressions as the expression `Max`, so that a
        model with the network does not overflow where the forward pass does
        not.

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

    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Get the maximum along the axis."""
        return np.max(x, axis=axis, keepdims=True)


def _elementwise(function: Callable[..., Any], n_args: int = 1) -> np.ufunc:
    """Get the function which applies a function to every element of arrays.

    Args:
        function: the function of `n_args` elements.
        n_args: the number of arrays the function takes an element of.

    Returns:
        The function of `object` arrays, with the broadcasting of numpy.
    """
    return np.frompyfunc(function, n_args, 1)


def _exact(value: Any) -> Any:
    """Get a number as the integer it is, which keeps the expressions short.

    Args:
        value: a number or an expression.

    Returns:
        The integer for a float without a fraction, e.g. `0` for `0.0`, the
        value otherwise.
    """
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return sympy.Integer(int(value))
    return value


class SympyBackend(Backend):
    """The backend on `object` arrays of sympy expressions.

    The elements of the arrays are symbols, numbers and expressions of them.
    A function with a condition is a `sympy.Piecewise`, which is the
    `piecewise` of the MathML of SBML. `erf` is evaluated as `sympy.erf`,
    which the MathML of SBML does not have: the compiler rejects an expression
    with it.
    """

    kind: ClassVar[BackendKind] = BackendKind.SYMPY
    dtype: ClassVar[type] = object

    def exp(self, x: np.ndarray) -> np.ndarray:
        """Calculate the exponential function."""
        return np.asarray(_elementwise(sympy.exp)(x), dtype=object)

    def log(self, x: np.ndarray) -> np.ndarray:
        """Calculate the natural logarithm."""
        return np.asarray(_elementwise(sympy.log)(x), dtype=object)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hyperbolic tangent."""
        return np.asarray(_elementwise(sympy.tanh)(x), dtype=object)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the square root."""
        return np.asarray(_elementwise(sympy.sqrt)(x), dtype=object)

    def erf(self, x: np.ndarray) -> np.ndarray:
        """Calculate the error function."""
        return np.asarray(_elementwise(sympy.erf)(x), dtype=object)

    def absolute(self, x: np.ndarray) -> np.ndarray:
        """Calculate the absolute value."""
        return np.asarray(_elementwise(sympy.Abs)(x), dtype=object)

    def select(
        self,
        x: np.ndarray,
        threshold: float,
        below: np.ndarray | float,
        above: np.ndarray | float,
    ) -> np.ndarray:
        """Choose between two values by a condition on `x`, as a `Piecewise`."""
        limit = _exact(threshold)

        def piecewise(value: Any, low: Any, high: Any) -> sympy.Basic:
            return sympy.Piecewise(
                (_exact(high), sympy.sympify(value) > limit), (_exact(low), True)
            )

        return np.asarray(_elementwise(piecewise, 3)(x, below, above), dtype=object)

    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Get the maximum along the axis, as the expression `Max`."""
        moved = np.moveaxis(np.asarray(x, dtype=object), axis, -1)
        maximum = np.empty(moved.shape[:-1], dtype=object)
        for index in np.ndindex(maximum.shape):
            maximum[index] = sympy.Max(*(sympy.sympify(v) for v in moved[index]))
        return np.expand_dims(maximum, axis)
