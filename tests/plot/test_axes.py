"""Quantities plot with matplotlib in their units and label their axes."""

from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.units import ConversionError

from sbmlsim import Q
from sbmlsim.plot import label_axes
from sbmlsim.units import setup_matplotlib


def _q(values: list[float], unit: str) -> Any:
    """Get a quantity, which the stubs of matplotlib do not know as data."""
    return Q(np.array(values), unit)


def test_quantities_plot_in_the_unit_of_their_axis() -> None:
    """The first quantity sets the unit of an axis, the others are converted."""
    setup_matplotlib()
    ax = Figure().add_subplot()
    (first,) = ax.plot(_q([0.0, 60.0], "min"), _q([1.0, 2.0], "mg/l"))
    (second,) = ax.plot(_q([1.0, 2.0], "hr"), _q([1.0, 2.0], "ng/ml"))
    # the data of a line in the units of its axes
    np.testing.assert_allclose(
        np.asarray(first.get_xydata()), [[0.0, 1.0], [60.0, 2.0]]
    )
    np.testing.assert_allclose(
        np.asarray(second.get_xydata()), [[60.0, 0.001], [120.0, 0.002]]
    )
    # a quantity of another dimension does not fit the axis
    with pytest.raises(ConversionError):
        ax.plot(_q([1.0], "kg"), _q([1.0], "mg/l"))


def test_the_axes_are_labelled_with_their_names_and_units() -> None:
    """`label_axes` puts the short unit of an axis next to its name."""
    setup_matplotlib()
    ax = Figure().add_subplot()
    ax.plot(_q([0.0, 1.0], "hr"), _q([1.0, 2.0], "ng/ml"))
    label_axes(ax, "time", "HCTZ plasma")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("time [h]", "HCTZ plasma [ng/ml]")


def test_an_axis_without_a_unit_is_labelled_with_its_name() -> None:
    """Plain numbers and a dimensionless quantity have no unit in the label."""
    setup_matplotlib()
    ax = Figure().add_subplot()
    ax.plot([0.0, 1.0], _q([1.0, 2.0], "dimensionless"))
    label_axes(ax, "time", "ratio")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("time", "ratio")
    label_axes(ax, y="share")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("time", "share")
