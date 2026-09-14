"""Test plotting."""

import copy

import pytest

from sbmlsim.plot.plotting import (
    Axis,
    ColorType,
    Figure,
    Fill,
    Line,
    LineType,
    Marker,
    MarkerType,
    Plot,
    Style,
)

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


# ---------------------------------------------------------------------------
# styles which derive from other styles
# ---------------------------------------------------------------------------
def test_resolving_a_style_does_not_change_the_style_it_derives_from() -> None:
    """A base style is shared, so resolving a derived style must not write it.

    `resolve_style` returned the base object itself and then assigned the
    attributes of the deriving style onto it, so every style with that base
    silently took over the values of the first one which was resolved.
    """
    base = Style(sid="base", line=Line(color=ColorType("#000000ff"), thickness=1.0))
    derived = Style(
        sid="derived",
        base_style=base,
        line=Line(color=ColorType("#ff0000ff"), thickness=5.0),
    )

    resolved = derived.resolve_style()

    assert resolved is not base
    assert base.line is not None
    assert base.line.thickness == 1.0
    assert base.line.color is not None
    assert base.line.color.color == "#000000ff"
    assert resolved.line is not None
    assert resolved.line.thickness == 5.0


def test_a_derived_style_takes_over_the_attributes_which_are_set() -> None:
    """The inherited attributes are the fields of `Line`, `Marker` and `Fill`.

    They were looked up under the camel case names of SED-ML, which none of
    those dataclasses ever had, so the line type, the marker type, the marker
    edge and the second fill colour were never inherited.
    """
    base = Style(
        sid="base",
        line=Line(type=LineType.SOLID, thickness=1.0),
        marker=Marker(type=MarkerType.CIRCLE, line_thickness=1.0),
        fill=Fill(color=ColorType("#000000ff")),
    )
    derived = Style(
        sid="derived",
        base_style=base,
        line=Line(type=LineType.DASH, thickness=3.0),
        marker=Marker(type=MarkerType.SQUARE, line_thickness=4.0),
        fill=Fill(second_color=ColorType("#ff0000ff")),
    )

    resolved = derived.resolve_style()

    assert resolved.line is not None
    assert resolved.line.type == LineType.DASH
    assert resolved.marker is not None
    assert resolved.marker.type == MarkerType.SQUARE
    assert resolved.marker.line_thickness == 4.0
    assert resolved.fill is not None
    assert resolved.fill.second_color is not None
    assert resolved.fill.second_color.color == "#ff0000ff"
    # and what the deriving style does not set stays that of the base
    assert resolved.fill.color is not None
    assert resolved.fill.color.color == "#000000ff"


def test_a_copied_style_keeps_the_style_it_derives_from() -> None:
    """`__copy__` dropped `base_style`, so a copy lost its inheritance."""
    base = Style(sid="base", line=Line(thickness=1.0))
    assert copy.copy(Style(sid="s", base_style=base)).base_style is base


# ---------------------------------------------------------------------------
# the matplotlib style vocabulary
# ---------------------------------------------------------------------------
linestyle_data = [
    (":", LineType.DOT),
    ("dotted", LineType.DOT),
    ("-", LineType.SOLID),
    ("--", LineType.DASH),
    ("-.", LineType.DASHDOT),
    ("None", LineType.NONE),
    (" ", LineType.NONE),
    ("", LineType.NONE),
]


@pytest.mark.parametrize("linestyle, line_type", linestyle_data)
def test_the_matplotlib_linestyles_are_understood(
    linestyle: str, line_type: LineType
) -> None:
    """`:` is the dotted linestyle of matplotlib, `.` is a marker.

    The mapping carried `.` and not `:`, so the canonical spelling raised a
    `KeyError` and a marker was accepted as a linestyle.
    """
    style = Style.from_mpl_kwargs(linestyle=linestyle)
    assert style.line is not None
    assert style.line.type == line_type


def test_a_linestyle_round_trips_through_the_figure_model() -> None:
    """What the model writes back is a linestyle matplotlib accepts."""
    for line_type in LineType:
        assert line_type in Style.SEDML2MPL_LINESTYLE_MAPPING


def test_an_unsupported_marker_says_what_is_supported() -> None:
    """A `KeyError` of the raw value does not say what to write instead."""
    with pytest.raises(ValueError, match="Unsupported marker"):
        Style.from_mpl_kwargs(marker="p")


# ---------------------------------------------------------------------------
# copying a plot and associating it with its figure
# ---------------------------------------------------------------------------
def test_a_copied_plot_keeps_its_right_y_axis() -> None:
    """`Plot.__copy__` did not pass `yaxis_right`, so a copy lost it."""
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s"),
        yaxis=Axis("y", unit="mM"),
        yaxis_right=Axis("r", unit="mM"),
    )
    assert copy.copy(plot).yaxis_right is not None


def test_a_plot_of_a_figure_knows_its_figure() -> None:
    """A plot resolves its data through its figure, see `Plot.experiment`.

    `Figure.from_plots` and `add_subplot` built the subplots without the
    association, so `plot.figure` raised for every plot which was not added
    through `add_plots`.
    """
    plot = Plot(sid="p", xaxis=Axis("x", unit="s"))
    figure = Figure.from_plots(sid="f", plots=[plot], experiment=None)
    assert plot.figure is figure

    other = Plot(sid="q", xaxis=Axis("x", unit="s"))
    figure.num_rows = 2
    assert figure.add_subplot(other, row=2, col=1).figure is figure


def test_a_right_y_axis_of_the_wrong_type_is_refused() -> None:
    """`xaxis` and `yaxis` were checked and `yaxis_right` was not."""
    with pytest.raises(ValueError, match="yaxis_right"):
        Plot(sid="p", yaxis_right="not an axis")  # ty: ignore[invalid-argument-type]
