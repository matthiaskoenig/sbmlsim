"""Tests of the backend on arrays of numbers."""

import inspect

import numpy as np

from sbmlsim.sciml.backend import (
    ALL_BACKENDS,
    NUMPY_ONLY,
    Backend,
    BackendKind,
    NumpyBackend,
)


def test_the_kinds_of_backends() -> None:
    """A layer declares its backends with these sets."""
    assert {BackendKind.NUMPY, BackendKind.SYMPY} == ALL_BACKENDS
    assert {BackendKind.NUMPY} == NUMPY_ONLY
    assert NumpyBackend.kind == BackendKind.NUMPY
    assert NumpyBackend.dtype is float


def test_a_backend_implements_the_elementwise_functions() -> None:
    """The abstract class names what a backend on expressions implements."""
    assert inspect.isabstract(Backend)
    assert Backend.__abstractmethods__ == {
        "exp",
        "log",
        "tanh",
        "sqrt",
        "erf",
        "absolute",
        "select",
        "stabilizer",
    }
    assert not inspect.isabstract(NumpyBackend)


def test_the_array_of_the_backend() -> None:
    """Values are converted to arrays of numbers in double precision."""
    backend = NumpyBackend()
    for values in ([1, 2], np.array([1, 2]), np.array([1.0, 2.0], dtype="f4")):
        array = backend.asarray(values)
        assert array.dtype == float
        np.testing.assert_array_equal(array, [1.0, 2.0])
    assert backend.asarray(3).shape == ()


def test_the_elementwise_functions() -> None:
    """The functions are the ones of numpy and scipy."""
    backend = NumpyBackend()
    x = np.array([[0.25, 1.0], [4.0, 9.0]])
    np.testing.assert_allclose(backend.exp(x), np.exp(x))
    np.testing.assert_allclose(backend.log(x), np.log(x))
    np.testing.assert_allclose(backend.tanh(x), np.tanh(x))
    np.testing.assert_allclose(backend.sqrt(x), [[0.5, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(backend.absolute(-x), x)
    np.testing.assert_allclose(backend.erf(np.array([0.0, 1.0])), [0.0, 0.8427007929])


def test_the_selection_by_a_condition() -> None:
    """`above` holds for `x > threshold`, the threshold itself is `below`."""
    backend = NumpyBackend()
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, -5.0, 5.0), [-5.0, -5.0, 5.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, 0.0, x), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, 2 * x, x), [-2.0, 0.0, 1.0])


def test_an_overflow_is_not_a_warning() -> None:
    """The branch of a selection which is not chosen may overflow."""
    backend = NumpyBackend()
    with np.errstate(over="raise"):
        assert backend.exp(np.array([800.0]))[0] == np.inf


def test_the_shift_of_softmax() -> None:
    """The maximum along the axis keeps its axis, so it broadcasts."""
    backend = NumpyBackend()
    x = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
    np.testing.assert_array_equal(backend.stabilizer(x, 1), [[5.0], [7.0]])
    np.testing.assert_array_equal(backend.stabilizer(x, 0), [[7.0, 5.0, 4.0]])
