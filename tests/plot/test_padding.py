"""The padding of a ragged result is not drawn."""

import numpy as np

from sbmlsim.plot.padding import first_curve, without_padding


def test_first_curve_of_a_scan() -> None:
    """A scan has a column per simulation, the first one is drawn."""
    np.testing.assert_array_equal(first_curve(np.array([[1, 2], [3, 4]])), [1, 3])
    np.testing.assert_array_equal(first_curve(np.ones((2, 2, 3))), [1, 1])
    np.testing.assert_array_equal(first_curve(np.array([1, 2])), [1, 2])
    assert first_curve(None) is None


def test_padding_is_dropped_by_the_x() -> None:
    """The points whose x is NaN are padding, a NaN of y is a gap and stays."""
    x = np.array([0.0, 1.0, np.nan])
    y = np.array([1.0, np.nan, np.nan])
    xs, ys, none = without_padding(x, y, None)
    np.testing.assert_array_equal(xs, [0.0, 1.0])
    np.testing.assert_array_equal(ys, [1.0, np.nan])
    assert none is None
