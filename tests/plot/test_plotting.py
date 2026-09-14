"""Test plotting."""

import copy

import pytest

from sbmlsim.plot.plotting import Axis, ColorType

color_data = [
    # not supporting one-characters
    # ("r", "#ff0000ff", 1.0),  # depend on mpl settings and version
    # ("g", "#008000ff", 1.0),  # depend on mpl settings and version
    # ("b", "#0000ffff", 1.0),  # depend on mpl settings and version
    ("white", "#ffffffff", 1.0),
    ("black", "#000000ff", 1.0),
    ("#1234ff44", "#1234ff44", 1.0),
    ("#1234ff", "#1234ffff", 1.0),
    ("#56784a", "#56784a00", 0.0),
]


@pytest.mark.parametrize("color, hex, alpha", color_data)
def test_parse_color(color: str, hex: str, alpha: float) -> None:
    """Test parsing of color."""
    c = ColorType.parse_color(color, alpha=alpha)
    assert c is not None
    assert c.color == hex


# ---------------------------------------------------------------------------
# the axis label follows the label and the unit, see #167
# ---------------------------------------------------------------------------
axis_label_data = [
    ("time", "s", "time [s]"),
    # a part which is not given is left out rather than written as `None`
    ("time", None, "time"),
    (None, "week", "[week]"),
    (None, None, ""),
    # `dimensionless` is written as `-`
    ("x", "dimensionless", "x [-]"),
    (None, "dimensionless", "[-]"),
]


@pytest.mark.parametrize("label, unit, name", axis_label_data)
def test_the_axis_label_is_the_label_and_the_unit(
    label: str | None, unit: str | None, name: str
) -> None:
    """The label and the unit form the axis label."""
    assert Axis(label=label, unit=unit).name == name


def test_setting_the_unit_updates_the_axis_label() -> None:
    """A unit which is set is reflected in the label a figure renders.

    `name` was computed once in the constructor, so an axis which was given a
    new unit kept the label of the old one.
    """
    axis = Axis(label="time", unit="s")
    axis.unit = "week"
    assert axis.name == "time [week]"


def test_setting_the_label_updates_the_axis_label() -> None:
    """A label which is set is reflected the same way."""
    axis = Axis(label="time", unit="s")
    axis.label = "duration"
    assert axis.name == "duration [s]"


def test_a_name_overrides_the_label_and_the_unit() -> None:
    """`name` is the complete axis label and does not follow the parts."""
    axis = Axis(label="time", unit="s", name="Fixed")
    assert axis.name == "Fixed"
    axis.unit = "week"
    assert axis.name == "Fixed"


def test_a_name_of_none_restores_the_derived_label() -> None:
    """Dropping the override hands the axis back to its label and unit."""
    axis = Axis(label="time", unit="s", name="Fixed")
    axis.name = None
    assert axis.name == "time [s]"


def test_a_copied_axis_still_follows_its_unit() -> None:
    """A copy of a derived axis is derived, not pinned to the copied text."""
    axis = copy.copy(Axis(label="time", unit="s"))
    axis.unit = "week"
    assert axis.name == "time [week]"


def test_a_copied_axis_keeps_its_override() -> None:
    """A copy of an overridden axis keeps the override."""
    axis = copy.copy(Axis(label="time", unit="s", name="Fixed"))
    axis.unit = "week"
    assert axis.name == "Fixed"


def test_a_copied_axis_keeps_the_reverse_flag() -> None:
    """`reverse` is what a serializer inverts the axis by and was dropped."""
    assert copy.copy(Axis(label="time", unit="s", reverse=True)).reverse is True
