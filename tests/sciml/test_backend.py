"""Tests of the backends on arrays of numbers and of expressions."""

import inspect

import numpy as np
import pytest
import sympy

from sbmlsim.sciml.backend import (
    ALL_BACKENDS,
    NUMPY_ONLY,
    Backend,
    BackendKind,
    NumpyBackend,
    SympyBackend,
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


# --- THE BACKEND ON EXPRESSIONS ---


def test_the_backend_on_expressions() -> None:
    """The backend on expressions is the second kind of backend."""
    assert SympyBackend.kind == BackendKind.SYMPY
    assert SympyBackend.dtype is object
    assert not inspect.isabstract(SympyBackend)
    x = SympyBackend().asarray([sympy.Symbol("a"), 1.5])
    assert x.dtype == object
    assert x.shape == (2,)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("exp", sympy.exp),
        ("log", sympy.log),
        ("tanh", sympy.tanh),
        ("sqrt", sympy.sqrt),
        ("erf", sympy.erf),
        ("absolute", sympy.Abs),
    ],
)
def test_the_elementwise_functions_on_expressions(name: str, expected: object) -> None:
    """A function is applied to every element and keeps the shape."""
    backend = SympyBackend()
    a, b = sympy.symbols("a b")
    x = np.array([[a, b], [a + b, 2 * a]], dtype=object)
    y = getattr(backend, name)(x)
    assert y.dtype == object
    assert y.shape == (2, 2)
    assert y[1, 0] == expected(a + b)  # ty: ignore[call-non-callable]


def test_a_function_of_a_value_without_axes_is_an_array() -> None:
    """numpy returns a scalar for an array without axes, the backend an array."""
    backend = SympyBackend()
    x = np.array(sympy.Symbol("a"), dtype=object)
    for y in (backend.tanh(x), backend.select(x, 0.0, 0.0, x)):
        assert isinstance(y, np.ndarray)
        assert y.shape == ()
        assert y.dtype == object


def test_a_condition_is_a_piecewise() -> None:
    """`select` is `above` where `x > threshold` and `below` elsewhere."""
    backend = SympyBackend()
    a = sympy.Symbol("a")
    (y,) = backend.select(np.array([a], dtype=object), 0.0, 0.0, np.array([a]))
    assert y == sympy.Piecewise((a, a > 0), (0, True))
    assert y.subs(a, 2.0) == 2.0
    assert y.subs(a, 0.0) == 0
    assert y.subs(a, -2.0) == 0
    # a number which is an integer is written as one, the others as they are
    assert y.atoms(sympy.Float) == set()
    (z,) = backend.select(np.array([a], dtype=object), 0.5, -1.25, 6.0)
    assert z == sympy.Piecewise((6, a > 0.5), (-1.25, True))


def test_a_condition_on_a_number() -> None:
    """An element which is a number is evaluated."""
    backend = SympyBackend()
    y = backend.select(np.array([2.0, -1.0], dtype=object), 0.0, 0.0, 7.0)
    assert list(y) == [7, 0]


def test_the_shift_of_softmax_on_expressions() -> None:
    """`softmax` on expressions is shifted by the maximum, as on numbers."""
    a, b, c, d = sympy.symbols("a b c d")
    x = np.array([[a, b], [c, 2.0]], dtype=object)
    backend = SympyBackend()
    assert backend.stabilizer(x, 0).tolist() == [[sympy.Max(a, c), sympy.Max(b, 2.0)]]
    assert backend.stabilizer(x, 1).tolist() == [[sympy.Max(a, b)], [sympy.Max(c, 2.0)]]
    assert backend.stabilizer(x, -1).shape == (2, 1)
    numbers = np.array([1.0, 3.0, 2.0], dtype=object)
    assert backend.stabilizer(numbers, 0).tolist() == [3.0]
    assert backend.stabilizer(np.array([d], dtype=object), 0).tolist() == [d]


def test_the_softmax_of_the_backends() -> None:
    """Numbers are shifted by the maximum, expressions need no maximum.

    `1 / sum_j exp(x_j - x_i)` overflows only in a term of the sum, which
    gives the limit `0`. Its size grows with the number of units, a maximum
    in every term with its square: roadrunner inlines the assignment rules.
    """
    x = np.array([[1.0, 800.0, -3.0], [0.5, 0.25, 2.0]])
    for axis in (0, 1, -1):
        shifted = np.exp(x - x.max(axis=axis, keepdims=True))
        np.testing.assert_array_equal(
            NumpyBackend().softmax(x, axis),
            shifted / shifted.sum(axis=axis, keepdims=True),
        )
        expressions = SympyBackend().softmax(x.astype(object), axis)
        assert expressions.shape == x.shape
        np.testing.assert_allclose(
            expressions.astype(float), NumpyBackend().softmax(x, axis), rtol=1e-14
        )

    a, b, c = sympy.symbols("a b c")
    y = SympyBackend().softmax(np.array([a, b, c], dtype=object), 0)
    assert y[0] == 1 / (1 + sympy.exp(b - a) + sympy.exp(c - a))
    assert not any(expression.atoms(sympy.Max) for expression in y)
