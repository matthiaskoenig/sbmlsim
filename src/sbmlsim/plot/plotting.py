"""Classes for storing plotting information.

The general workflow of generating plotting information is the following.

1. Within simulation experiments abstract plotting information is stored.
    i.e., how from the data plots can be generated.


Working with multidimensional data !
Additional settings are required which allow to define how things
        are plotted.
        E.g. over which dimensions should an error be calculated and which
        dimensions should be plotted individually.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import numpy as np
from matplotlib.colors import to_hex, to_rgba

from sbmlsim.data import Data

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)

# The colors in the default property cycle have been changed
# to the category10 color palette used by Vega and d3 originally developed at Tableau.
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class BasePlotObject:
    """Base class for plotting objects."""

    def __init__(self, sid: str | None, name: str | None):
        """Initialize BasePlotObject.

        Args:
            sid: Identifier of the object.
            name: Name of the object.
        """
        self.sid = sid
        self.name = name


class LineType(Enum):
    """LineType options."""

    NONE = 1
    SOLID = 2
    DASH = 3
    DOT = 4
    DASHDOT = 5
    DASHDOTDOT = 6


class MarkerType(Enum):
    """MarkerType options."""

    NONE = 1
    SQUARE = 2
    CIRCLE = 3
    DIAMOND = 4
    XCROSS = 5
    PLUS = 6
    STAR = 7
    TRIANGLEUP = 8
    TRIANGLEDOWN = 9
    TRIANGLELEFT = 10
    TRIANGLERIGHT = 11
    HDASH = 12
    VDASH = 13


class CurveType(Enum):
    """CurveType options."""

    POINTS = 1
    BAR = 2
    BARSTACKED = 3
    HORIZONTALBAR = 4
    HORIZONTALBARSTACKED = 5


class ColorType:
    """ColorType class.

    Encoding color information used in plots.
    """

    def __init__(self, color: str):
        """Initialize ColorType.

        Args:
            color: Color as hex string with alpha channel.

        Raises:
            ValueError: If the color is None.
        """
        if color is None:
            raise ValueError("color cannot be NoneType")

        self.color: str = color

    def to_dict(self) -> str:
        """Convert for serialization.

        Returns:
            Color string.
        """
        return self.color

    def __repr__(self) -> str:
        """Get string representation."""
        return self.color

    @staticmethod
    def parse_color(color: str | None, alpha: float = 1.0) -> ColorType | None:
        """Parse given color and add alpha information.

        Args:
            color: Color as matplotlib color string or hex color.
            alpha: Alpha value in [0, 1].

        Returns:
            ColorType or None if no color is given.

        Raises:
            ValueError: If the hex color has an incorrect format.
        """
        # https://matplotlib.org/stable/tutorials/colors/colors.html
        if color is None or len(color) == 0:
            return None

        if isinstance(color, str) and color.startswith("#"):
            # handle hex colors
            if len(color) == 7:
                # parse alpha
                color_hex = f"{color}{round(alpha * 255):02x}"
            elif len(color) == 9:
                color_hex = color
                if alpha != 1.0:
                    logger.warning(
                        "alpha ignored for hex colors with alpha channel: "
                        "'%s', alpha=%s.",
                        color,
                        alpha,
                    )
            else:
                logger.error("Incorrect hex color: '%s'", color)
                raise ValueError(f"Incorrect hex color: '{color}'")

        else:
            rgba = to_rgba(color, alpha)
            color_hex = to_hex(rgba, keep_alpha=True)

        return ColorType(color_hex)


@dataclass
class Line:
    """Style of a line."""

    type: LineType = LineType.SOLID
    color: ColorType | None = None
    thickness: float | None = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary of the line attributes.
        """
        return {
            "type": self.type,
            "color": self.color,
            "thickness": self.thickness,
        }


@dataclass
class Marker:
    """Style of a marker."""

    size: float | None = 6.0
    type: MarkerType = MarkerType.NONE
    fill: ColorType | None = None
    line_color: ColorType | None = None
    line_thickness: float | None = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary of the marker attributes.
        """
        return {
            "size": self.size,
            "type": self.type,
            "fill": self.fill,
            "line_color": self.line_color,
            "line_thickness": self.line_thickness,
        }


@dataclass
class Fill:
    """Style of a fill."""

    color: ColorType | None = None
    second_color: ColorType | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary of the fill attributes.
        """
        return {
            "color": self.color,
            "second_color": self.second_color,
        }


# matplotlib line styles, see
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
MplLineStyle = str | tuple[int, tuple[int, ...]]


class Style(BasePlotObject):
    """Style class.

    Storing styling informatin about line, marker and fill.
    Styles can be derived from other styles based on the the
    base_style attribute.
    """

    #: the attributes a derived style takes over, i.e. the fields of `Line`,
    #: `Marker` and `Fill`; they were the camel case names of SED-ML, which no
    #: attribute of those dataclasses ever had, so nothing was inherited
    LINE_KEYS: ClassVar[tuple[str, ...]] = ("type", "color", "thickness")
    MARKER_KEYS: ClassVar[tuple[str, ...]] = (
        "type",
        "size",
        "fill",
        "line_color",
        "line_thickness",
    )
    FILL_KEYS: ClassVar[tuple[str, ...]] = ("color", "second_color")

    MPL2SEDML_LINESTYLE_MAPPING: ClassVar[dict[str, LineType]] = {
        "": LineType.NONE,
        " ": LineType.NONE,
        "none": LineType.NONE,
        "None": LineType.NONE,
        "-": LineType.SOLID,
        "solid": LineType.SOLID,
        ":": LineType.DOT,
        "dotted": LineType.DOT,
        "--": LineType.DASH,
        "dashed": LineType.DASH,
        "-.": LineType.DASHDOT,
        "dashdot": LineType.DASHDOT,
        "dashdotdotted": LineType.DASHDOTDOT,
    }
    SEDML2MPL_LINESTYLE_MAPPING: ClassVar[dict[LineType, MplLineStyle]] = {
        LineType.NONE: "",
        LineType.SOLID: "solid",
        LineType.DOT: "dotted",
        LineType.DASH: "dashed",
        LineType.DASHDOT: "dashdot",
        LineType.DASHDOTDOT: (0, (3, 5, 1, 5, 1, 5)),
    }

    MPL2SEDML_MARKER_MAPPING: ClassVar[dict[str, MarkerType]] = {
        "": MarkerType.NONE,
        "s": MarkerType.SQUARE,
        "o": MarkerType.CIRCLE,
        "D": MarkerType.DIAMOND,
        "x": MarkerType.XCROSS,
        "+": MarkerType.PLUS,
        "*": MarkerType.STAR,
        "^": MarkerType.TRIANGLEUP,
        "v": MarkerType.TRIANGLEDOWN,
        "<": MarkerType.TRIANGLELEFT,
        ">": MarkerType.TRIANGLERIGHT,
        "_": MarkerType.HDASH,
        "|": MarkerType.VDASH,
        # the fills and the spellings matplotlib accepts for the same shape
        "none": MarkerType.NONE,
        "None": MarkerType.NONE,
        "d": MarkerType.DIAMOND,
        "X": MarkerType.XCROSS,
        "P": MarkerType.PLUS,
    }
    SEDML2MPL_MARKER_MAPPING: ClassVar[dict[MarkerType, str]] = {
        MarkerType.NONE: "",
        MarkerType.SQUARE: "s",
        MarkerType.CIRCLE: "o",
        MarkerType.DIAMOND: "D",
        MarkerType.XCROSS: "x",
        MarkerType.PLUS: "+",
        MarkerType.STAR: "*",
        MarkerType.TRIANGLEUP: "^",
        MarkerType.TRIANGLEDOWN: "v",
        MarkerType.TRIANGLELEFT: "<",
        MarkerType.TRIANGLERIGHT: ">",
        MarkerType.HDASH: "_",
        MarkerType.VDASH: "|",
    }

    def __init__(
        self,
        sid: str | None = None,
        name: str | None = None,
        base_style: Style | None = None,
        line: Line | None = None,
        marker: Marker | None = None,
        fill: Fill | None = None,
    ):
        """Initialize Style.

        Args:
            sid: Identifier of the style.
            name: Name of the style.
            base_style: Style this style is derived from.
            line: Line style (default line if not provided).
            marker: Marker style (default marker if not provided).
            fill: Fill style.
        """
        super().__init__(sid, name)

        # using default styling if not otherwise provided
        if marker is None:
            marker = Marker()
        if line is None:
            line = Line()

        self.base_style: Style | None = base_style
        self.line: Line | None = line
        self.marker: Marker | None = marker
        self.fill: Fill | None = fill

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declared style and its inheritance."""
        return {
            "sid": self.sid,
            "name": self.name,
            "base_style": self.base_style,
            "line": self.line,
            "marker": self.marker,
            "fill": self.fill,
        }

    def resolve_style(self) -> Style:
        """Resolve all basestyle information.

        Resolves the actual style information.

        Returns:
            Style with all information of the base styles applied.
        """
        # recursive resolving of basestyle.
        if not self.base_style:
            return self

        # a copy of the base, so that resolving a derived style does not write
        # into the style it derives from, which every other style sharing that
        # base would see
        style = deepcopy(self.base_style.resolve_style())

        # overwrite information
        if self.line:
            if not style.line:
                style.line = deepcopy(self.line)
            else:
                Style._override(self.line, style.line, Style.LINE_KEYS)

        if self.marker:
            if not style.marker:
                style.marker = deepcopy(self.marker)
            else:
                Style._override(self.marker, style.marker, Style.MARKER_KEYS)

        if self.fill:
            if not style.fill:
                style.fill = deepcopy(self.fill)
            else:
                Style._override(self.fill, style.fill, Style.FILL_KEYS)

        return style

    @staticmethod
    def _override(source: Any, target: Any, keys: tuple[str, ...]) -> None:
        """Copy the attributes which are set from the source to the target.

        Args:
            source: object of the deriving style.
            target: object of the resolved base style, changed in place.
            keys: attributes to take over when they are set on the source.
        """
        for key in keys:
            value = getattr(source, key)
            if value:
                logger.debug("%s: %s = %s", type(source).__name__, key, value)
                setattr(target, key, value)

    def __repr__(self) -> str:
        """Get string presentation."""
        return (
            f"{self.sid} (base_style={self.base_style}) [marker={self.marker}; line={self.line}; "
            f"fill={self.fill}]"
        )

    def __copy__(self) -> Style:
        """Copy style object."""
        return Style(
            sid=self.sid,
            name=self.name,
            base_style=self.base_style,
            line=self.line,
            marker=self.marker,
            fill=self.fill,
        )

    def to_mpl_curve_kwargs(self) -> dict[str, Any]:
        """Convert to matplotlib curve keyword arguments.

        Returns:
            Keyword arguments for matplotlib curves.
        """
        kwargs: dict[str, Any] = {}
        if self.line:
            if self.line.color:
                kwargs["color"] = self.line.color.color
            if self.line.type is not None:
                kwargs["linestyle"] = Style.SEDML2MPL_LINESTYLE_MAPPING[self.line.type]
            if self.line.thickness:
                kwargs["linewidth"] = self.line.thickness
        if self.marker:
            if self.marker.type is not None:
                kwargs["marker"] = Style.SEDML2MPL_MARKER_MAPPING[self.marker.type]
            if self.marker.size:
                kwargs["markersize"] = self.marker.size
            if self.marker.fill:
                kwargs["markerfacecolor"] = self.marker.fill.color
            if self.marker.line_color:
                kwargs["markeredgecolor"] = self.marker.line_color.color
            if self.marker.line_thickness:
                kwargs["markeredgewidth"] = self.marker.line_thickness

        if self.fill:
            if self.fill.color:
                kwargs["fill.color"] = self.fill.color.color
            if self.fill.second_color:
                kwargs["fill.second_color"] = self.fill.second_color.color

        return kwargs

    #: length of the caps of an error bar, in points
    ERROR_CAPSIZE: ClassVar[float] = 3.0

    def _mpl_error_kwargs(self) -> dict[str, Any]:
        """Define keywords for error bars.

        `errorbar` takes them directly and `bar` takes them in `error_kw`, so
        the two callers unpack this differently.

        Returns:
            Keyword arguments for error bars.
        """
        return {"capsize": Style.ERROR_CAPSIZE}

    def to_mpl_points_kwargs(self) -> dict[str, Any]:
        """Convert to matplotlib point curve keyword arguments.

        Returns:
            Keyword arguments for matplotlib errorbar plots.
        """
        points_kwargs = self.to_mpl_curve_kwargs()
        for key in ["fill.color", "fill.second_color"]:
            if key in points_kwargs:
                points_kwargs.pop(key)
        return {
            **points_kwargs,
            **self._mpl_error_kwargs(),
        }

    def to_mpl_bar_kwargs(self) -> dict[str, Any]:
        """Convert to matplotlib bar curve keyword arguments.

        Returns:
            Keyword arguments for matplotlib bar plots.
        """
        bar_kwargs = self.to_mpl_curve_kwargs()
        for key in [
            "marker",
            "markersize",
            "markeredgewidth",
            "markeredgecolor",
            "markerfacecolor",
            "fill.second_color",
        ]:
            # pop line keys
            if key in bar_kwargs:
                bar_kwargs.pop(key)

        if "color" in bar_kwargs:
            bar_kwargs["edgecolor"] = bar_kwargs.pop("color")
        if "fill.color" in bar_kwargs:
            bar_kwargs["color"] = bar_kwargs.pop("fill.color")

        return {
            **bar_kwargs,
            "error_kw": self._mpl_error_kwargs(),
        }

    def to_mpl_area_kwargs(self) -> dict[str, Any]:
        """Define keyword dictionary for a shaded area.

        Returns:
            Keyword arguments for matplotlib fill_between.
        """
        kwargs: dict[str, Any] = {}

        if self.line:
            if self.line.color:
                kwargs["edgecolor"] = self.line.color.color
            if self.line.type is not None:
                kwargs["linestyle"] = Style.SEDML2MPL_LINESTYLE_MAPPING[self.line.type]
            if self.line.thickness:
                kwargs["linewidth"] = self.line.thickness

        if self.fill and self.fill.color:
            kwargs["color"] = self.fill.color.color
            # FIXME: second color not supported (gradients)
            # if self.fill.second_color:
            #    kwargs["second.color"] = self.fill.second_color

        return kwargs

    @staticmethod
    def _lookup(mapping: dict[str, Any], value: Any, kind: str) -> Any:
        """Translate a matplotlib style value into the type of the figure model.

        Args:
            mapping: the matplotlib value to type mapping of the kind.
            value: the matplotlib value, e.g. `":"` or `"o"`.
            kind: what is looked up, for the message.

        Returns:
            The type of the figure model.

        Raises:
            ValueError: If the figure model has no type for the value. The
                message names what is supported, a `KeyError` of the raw value
                says nothing about what to write instead.
        """
        if value is None:
            value = ""
        if value in mapping:
            return mapping[value]
        raise ValueError(
            f"Unsupported {kind} '{value}', the figure model describes "
            f"{kind}s as one of: {sorted(k for k in mapping if k)}"
        )

    @staticmethod
    def from_mpl_kwargs(**kwargs: Any) -> Style:
        """Create style from matplotlib arguments.

        Args:
            **kwargs: Matplotlib styling arguments, e.g. `color`, `alpha`,
                `linestyle`, `linewidth`, `marker`, `markersize`,
                `markerfacecolor`, `markeredgecolor`, `markeredgewidth`.

        Returns:
            Style corresponding to the matplotlib arguments.
        """
        color = ColorType.parse_color(
            color=kwargs.get("color"),
            alpha=kwargs.get("alpha", 1.0),
        )
        line_color = ColorType.parse_color(
            color=kwargs.get("markeredgecolor"),
        )
        fill_color: ColorType | None = color
        if "markerfacecolor" in kwargs:
            markerfacecolor = kwargs["markerfacecolor"]
            fill_color = (
                markerfacecolor
                if isinstance(markerfacecolor, ColorType)
                else ColorType.parse_color(markerfacecolor)
            )

        # Line
        linestyle = Style._lookup(
            Style.MPL2SEDML_LINESTYLE_MAPPING, kwargs.get("linestyle", "-"), "linestyle"
        )
        line = Line(color=color, type=linestyle, thickness=kwargs.get("linewidth", 1.0))

        # Marker
        marker_symbol = Style._lookup(
            Style.MPL2SEDML_MARKER_MAPPING, kwargs.get("marker", ""), "marker"
        )
        marker = Marker(
            type=marker_symbol,
            size=kwargs.get("markersize"),
            fill=fill_color,
            line_color=line_color,
            line_thickness=kwargs.get("markeredgewidth"),
        )

        # Fill
        fill = Fill(color=color)

        return Style(line=line, marker=marker, fill=fill)


#: the decades below the largest value which the automatic limits of a log
#: axis show at most, see `log_limits`
LOG_DECADES = 6

#: the margin of the automatic limits of a log axis, a share of its decades, as
#: matplotlib's default margin
LOG_MARGIN = 0.05


def log_limits(
    smallest: float,
    largest: float,
    low: float | None = None,
    high: float | None = None,
) -> tuple[float, float] | None:
    """Get the limits of a log axis whose data span too many decades.

    The steps of the integrator start with tiny steps, at which a value which
    starts at zero, e.g. a concentration after an oral dose, is tiny as well;
    the automatic limits of a log axis would show dozens of decades of these
    values. Data which span more than `LOG_DECADES` below the largest value
    are limited to these decades, with the margin `LOG_MARGIN` of the
    decades; a limit set on the axis is kept and the margin follows it.

    Args:
        smallest: the smallest positive value of the data.
        largest: the largest value of the data.
        low: the lower limit set on the axis, `None` without.
        high: the upper limit set on the axis, `None` without.

    Returns:
        The lower and the upper limit, `None` when the automatic limits are
        fine: no positive data, or data within `LOG_DECADES`, which keeps the
        figures of such data as they were.
    """
    if not (np.isfinite(smallest) and np.isfinite(largest)) or largest <= 0:
        return None
    floor = largest / 10**LOG_DECADES
    if smallest >= floor:
        return None
    bottom = low if low is not None else max(smallest, floor)
    top = high if high is not None else largest
    if top <= bottom:
        return None
    margin = 10 ** (LOG_MARGIN * np.log10(top / bottom))
    return (
        bottom if low is not None else bottom / margin,
        top if high is not None else top * margin,
    )


class AxisScale(Enum):
    """Scale of the axis."""

    LINEAR = 1
    LOG10 = 2


class YAxisPosition(Enum):
    """Position of yaxis."""

    LEFT = 1
    RIGHT = 2


class Axis(BasePlotObject):
    """Axis object."""

    def __init__(
        self,
        label: str | None = None,
        unit: str | None = None,
        name: str | None = None,
        scale: AxisScale | str = AxisScale.LINEAR,
        min: float | None = None,
        max: float | None = None,
        reverse: bool = False,
        grid: bool = False,
        label_visible: bool = True,
        ticks_visible: bool = True,
        style: Style | None = None,
    ):
        """Axis object.

        The label and the unit form the axis label together, i.e. `name` is
        `"<label> [<unit>]"`, and it follows both of them: setting `label` or
        `unit` on an axis updates it. `name` is the complete axis label and
        overrides them; setting it to `None` hands the axis back to its label
        and its unit.

        Args:
            label: label part of axis label
            unit: unit part of axis label
            name: complete axis label, overrides label and unit
            scale: Scale of the axis, i.e. "linear" or "log" axis.
            min: lower axis bound
            max: upper axis bound
            reverse: flag to reverse axis plot order
            grid: show grid lines along the axis
            label_visible: show/hide the label text
            ticks_visible: show/hide axis ticks
            style: style of the axis
        """
        self._label: str | None = label
        self._unit: str | None = unit
        # sets `name`, i.e. the override, which is `None` for a derived label
        super().__init__(sid=None, name=name)
        self.scale = scale
        self.min: float | None = min
        self.max: float | None = max
        self.reverse: bool = reverse
        self.grid: bool = grid
        self.label_visible: bool = label_visible
        self.ticks_visible: bool = ticks_visible
        self.style: Style | None = style

    def __repr__(self) -> str:
        """Get string."""
        return (
            f"Axis(sid={self.sid} name={self.name} scale={self.scale} "
            f"min={self.min} max={self.max})"
        )

    def __str__(self) -> str:
        """Get string."""
        return f"Axis({self.name, self.scale})"

    def __copy__(self) -> Axis:
        """Copy axis object."""
        return Axis(
            label=self.label,
            # the override and not `name`, so that a derived label stays derived
            name=self._name,
            unit=self.unit,
            scale=self.scale,
            min=self.min,
            max=self.max,
            reverse=self.reverse,
            grid=self.grid,
            label_visible=self.label_visible,
            ticks_visible=self.ticks_visible,
            style=copy.copy(self.style),
        )

    @property
    def label(self) -> str | None:
        """Get the label part of the axis label."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        """Set the label part of the axis label.

        Args:
            label: label part, which `name` follows unless it is overridden.
        """
        self._label = label

    @property
    def unit(self) -> str | None:
        """Get the unit part of the axis label."""
        return self._unit

    @unit.setter
    def unit(self, unit: str | None) -> None:
        """Set the unit part of the axis label.

        Args:
            unit: unit part, which `name` follows unless it is overridden.
        """
        self._unit = unit

    @property
    def name(self) -> str:
        """Get the complete axis label, which is what a figure renders.

        The override if one was set, the label and the unit otherwise, so that
        a change of either of them is reflected.
        """
        if self._name is not None:
            return self._name
        return self._derived_name()

    @name.setter
    def name(self, name: str | None) -> None:
        """Set the complete axis label, overriding the label and the unit.

        Args:
            name: complete axis label; `None` restores the derived one.
        """
        self._name: str | None = name

    def _derived_name(self) -> str:
        """Get the axis label which the label and the unit form.

        A unit of `dimensionless` is written as `-`, and a part which is not
        given is left out rather than written as `None`.
        """
        if not self._label and not self._unit:
            return ""
        if not self._unit:
            return f"{self._label}"
        unit = "-" if self._unit == "dimensionless" else self._unit
        if not self._label:
            return f"[{unit}]"
        return f"{self._label} [{unit}]"

    @property
    def scale(self) -> AxisScale:
        """Get axis scale."""
        return self._scale

    @scale.setter
    def scale(self, scale: AxisScale | str) -> None:
        """Set axis scale.

        Args:
            scale: AxisScale or one of the strings "linear", "log", "log10".

        Raises:
            ValueError: If the scale string is not supported.
        """
        if isinstance(scale, str):
            if scale == "linear":
                scale = AxisScale.LINEAR
            elif scale in {"log", "log10"}:
                scale = AxisScale.LOG10
            else:
                raise ValueError(f"Unsupported axis scale: '{scale}'")
        self._scale: AxisScale = scale

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary of the axis attributes.
        """
        return {
            "sid": self.sid,
            "name": self.name,
            "label": self.label,
            "unit": self.unit,
            "scale": self.scale,
            "min": self.min,
            "max": self.max,
            "grid": self.grid,
            "label_visible": self.label_visible,
            "ticks_visible": self.ticks_visible,
        }


class AbstractCurve(BasePlotObject):
    """Base class of Curves and ShadedAreas."""

    def __init__(
        self,
        sid: str | None,
        name: str | None,
        x: Data | None = None,
        order: int | None = None,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
    ):
        """Abstract base class of Curve and ShadedArea.

        Args:
            sid: identifier of the curve
            name: label of the curve
            x: x data
            order: order of the curve in the plot
            style: style of the curve
            yaxis_position: position of the yaxis for the curve
        """
        super().__init__(sid, name)
        self.x: Data | None = x
        self.order: int | None = order
        self.style: Style | None = style
        self.yaxis_position: YAxisPosition | None = yaxis_position


def _dimensions(
    over: str | Sequence[str] | None, kind: str, ident: str | None
) -> tuple[str, ...]:
    """Get the dimensions a curve or band draws one line per point of.

    Args:
        over: the dimensions, one or several.
        kind: what draws them, `curve` or `band`, for the error.
        ident: the id of the curve or band, else its name, for the error.

    Raises:
        ValueError: if a dimension is named twice.
    """
    names = (over,) if isinstance(over, str) else tuple(over or ())
    if len(set(names)) != len(names):
        what = f"The {kind} '{ident}'" if ident else f"A {kind}"
        raise ValueError(f"{what} names a dimension twice in over: {names}.")
    return names


class Curve(AbstractCurve):
    """Curve object."""

    def __init__(
        self,
        x: Data,
        y: Data,
        sid: str | None = None,
        name: str | None = None,
        xerr: Data | None = None,
        yerr: Data | None = None,
        order: int | None = None,
        type: CurveType = CurveType.POINTS,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
        over: str | Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize Curve.

        Args:
            x: x data
            y: y data
            sid: identifier of the curve
            name: label of the curve (name of y data if not provided)
            xerr: x error data
            yerr: y error data
            order: order of the curve in the plot
            type: type of the curve
            style: style of the curve (matplotlib kwargs are ignored if set)
            yaxis_position: position of the yaxis for the curve
            over: the dimensions of a scan with one line per point, see
                `sbmlsim.plot.points`
            **kwargs: matplotlib styling arguments, `label` sets the name

        Raises:
            ValueError: if a dimension is named twice in `over`, or a curve
                which is not of points names one.
        """
        super().__init__(
            sid=sid,
            name=name if name else y.name,
            x=x,
            order=order,
            style=style,
            yaxis_position=yaxis_position,
        )
        self.x: Data = x
        self.y: Data = y

        # set symmetrical
        self.xerr: Data | None = xerr
        self.yerr: Data | None = yerr

        # the label names the curve, it is no argument of the style
        if "label" in kwargs:
            self.name = kwargs.pop("label")

        self.type: CurveType = type
        self.over: tuple[str, ...] = _dimensions(over, "curve", sid or self.name)
        if self.over and type != CurveType.POINTS:
            raise ValueError(
                f"The curve '{sid or self.name}' is a bar curve, which draws no line "
                f"per point of {list(self.over)}; draw points or select a label."
            )

        # parse additional arguments and create style
        if style:
            if kwargs:
                logger.warning(
                    "'style' is set, the style arguments %s are ignored.",
                    sorted(kwargs),
                )
        else:
            kwargs = Curve._add_default_style_kwargs(kwargs, y.dtype)
            style = Style.from_mpl_kwargs(**kwargs)
        self.style = style
        self.kwargs = kwargs  # store for lookup

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Curve(sid={self.sid} name={self.name} type={self.type} order={self.order} "
            f"x={self.x is not None} y={self.y is not None} over={self.over} "
            f"xerr={self.xerr is not None} yerr={self.yerr is not None})"
        )

    def __str__(self) -> str:
        """Get string."""
        info = [
            "Curve(",
            f"\tsid={self.sid}",
            f"\tname={self.name}",
            f"\ttype={self.type}",
            f"\tx={self.x}",
            f"\ty={self.y}",
            f"\txerr={self.xerr}",
            f"\tyerr={self.yerr}",
            f"\tover={self.over}",
            f"\torder={self.order}",
            f"\tyaxis_position={self.yaxis_position}",
            ")",
        ]
        return "\n".join(info)

    @staticmethod
    def _add_default_style_kwargs(
        d: dict[str, Any], dtype: Data.Types
    ) -> dict[str, Any]:
        """Add the default plotting style arguments.

        Args:
            d: matplotlib keyword arguments
            dtype: type of the plotted data

        Returns:
            Keyword arguments with defaults added.
        """
        if dtype == Data.Types.TASK:
            if "linestyle" not in d:
                d["linestyle"] = "-"
            if "linewidth" not in d:
                d["linewidth"] = 2.0

        elif dtype == Data.Types.DATASET:
            if "linestyle" not in d:
                d["linestyle"] = "--"
            if "marker" not in d:
                d["marker"] = "s"

        return d

    def to_dict(self) -> dict[str, Any]:
        """Convert Curve to dictionary.

        Returns:
            Dictionary of the curve attributes.
        """
        return {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid if self.x else None,
            "y": self.y.sid if self.y else None,
            "xerr": self.xerr.sid if self.xerr else None,
            "yerr": self.yerr.sid if self.yerr else None,
            "over": list(self.over),
            "yaxis_position": self.yaxis_position,
            "style": self.style,
            "order": self.order,
        }


class ShadedArea(AbstractCurve):
    """ShadedArea class."""

    def __init__(
        self,
        x: Data,
        yfrom: Data,
        yto: Data,
        order: int | None = None,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
        **kwargs: Any,
    ):
        """Initialize ShadedArea.

        Args:
            x: x data
            yfrom: lower y data
            yto: upper y data
            order: order of the area in the plot
            style: style of the area
            yaxis_position: position of the yaxis for the area
            **kwargs: additional arguments, `label`, `sid` and `name` are used
        """
        super().__init__(
            sid=None,
            name=None,
            x=x,
            order=order,
            style=style,
            yaxis_position=yaxis_position,
        )
        self.x: Data = x
        self.yfrom: Data = yfrom
        self.yto: Data = yto

        if "label" in kwargs:
            self.name = kwargs["label"]
        if "sid" in kwargs:
            self.sid = kwargs["sid"]
        if "name" in kwargs:
            self.name = kwargs["name"]

        self.kwargs: dict[str, Any] = kwargs

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"ShadedArea(sid={self.sid} name={self.name} order={self.order} "
            f"x={self.x is not None} yfrom={self.yfrom is not None} "
            f"yto={self.yto is not None})"
        )

    def __str__(self) -> str:
        """Get string."""
        info = [
            "ShadedArea(",
            f"\tsid={self.sid}",
            f"\tname={self.name}",
            f"\tx={self.x}",
            f"\tyfrom={self.yfrom}",
            f"\tyto={self.yto}",
            f"\torder={self.order}",
            f"\tyaxis_position={self.yaxis_position}",
            ")",
        ]
        return "\n".join(info)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary of the area attributes.
        """
        return {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid if self.x else None,
            "yfrom": self.yfrom.sid if self.yfrom else None,
            "yto": self.yto.sid if self.yto else None,
            "yaxis_position": self.yaxis_position,
            "style": self.style,
            "order": self.order,
        }


class Band(AbstractCurve):
    """The median and a quantile band of data over a dimension of draws.

    The dimension `across` of y is reduced to the two quantiles, ignoring
    `NaN`, and the median, when the figure is drawn, see
    `sbmlsim.plot.points.band_lines`; `over` gives one band per point of
    other dimensions, in the colours of a curve over them.
    """

    def __init__(
        self,
        x: Data,
        y: Data,
        across: str,
        quantiles: tuple[float, float] = (0.05, 0.95),
        median: bool = True,
        over: str | Sequence[str] | None = None,
        sid: str | None = None,
        name: str | None = None,
        color: str | None = None,
        alpha: float = 0.3,
        order: int | None = None,
        yaxis_position: YAxisPosition | None = None,
    ):
        """Initialize a band.

        Args:
            x: x data.
            y: y data, with the dimension `across`.
            across: the dimension of the draws, which is reduced.
            quantiles: the lower and the upper quantile, `0 <= low < high <= 1`.
            median: draw the median as a line.
            over: the dimensions with one band per point.
            sid: identifier of the band.
            name: label of the band, the name of y by default.
            color: colour of the band; by default the colour of the colour
                cycle at the position of the band among the bands of its plot
                (`C0`, `C1`, ...).
            alpha: opacity of the area.
            order: order of the band in the plot.
            yaxis_position: position of the y axis of the band.

        Raises:
            ValueError: if the quantiles are not increasing within `[0, 1]`,
                `across` is in `over`, or `over` names more than one dimension.
        """
        super().__init__(
            sid=sid,
            name=name if name else y.name,
            x=x,
            order=order,
            yaxis_position=yaxis_position,
        )
        low, high = (float(q) for q in quantiles)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"The quantiles of a band are increasing within [0, 1], not {quantiles}."
            )
        self.x: Data = x
        self.y: Data = y
        self.across: str = across
        self.quantiles: tuple[float, float] = (low, high)
        self.median: bool = median
        label = sid or self.name
        self.over: tuple[str, ...] = _dimensions(over, "band", label)
        if across in self.over:
            raise ValueError(
                f"The band '{label}' reduces '{across}', which is also in over."
            )
        if len(self.over) > 1:
            raise ValueError(
                f"A band is drawn per point of one dimension, not of {self.over}."
            )
        self.color: str | None = color
        self.alpha: float = alpha

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Band(sid={self.sid} name={self.name} across={self.across} "
            f"over={self.over})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the band to a dictionary."""
        return {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid,
            "y": self.y.sid,
            "across": self.across,
            "quantiles": list(self.quantiles),
            "median": self.median,
            "over": list(self.over),
            "color": self.color,
            "alpha": self.alpha,
            "yaxis_position": self.yaxis_position,
            "order": self.order,
        }


class Plot(BasePlotObject):
    """Plot panel.

    A plot is the basic element of a plot. This corresponds to a single
    panel or axes combination in a plot. Multiple plots create a figure.
    """

    def __init__(
        self,
        sid: str,
        name: str | None = None,
        xaxis: Axis | None = None,
        yaxis: Axis | None = None,
        yaxis_right: Axis | None = None,
        curves: list[Curve] | None = None,
        areas: list[ShadedArea] | None = None,
        bands: list[Band] | None = None,
        legend: bool = True,
        facecolor: ColorType | None = None,
        title_visible: bool = True,
        height: float | None = None,
        width: float | None = None,
    ):
        """Initialize plot.

        Args:
            sid: Sid of the plot
            name: title of the plot
            xaxis: x-Axis
            yaxis: y-Axis
            yaxis_right: right y-Axis
            curves: list of curves for the plots
            areas: list of shaded areas for the plots
            bands: list of bands for the plots
            legend: boolean flag to show or hide legend
            facecolor: color of the plot.
            title_visible: boolean flag to show the title
            height: plot height (should be set on figure)
            width: plot width (should be set on figure)

        Raises:
            ValueError: If the axes are not of type Axis.
        """
        super().__init__(sid, name)
        if curves is None:
            curves = []
        if legend is None:
            # legend by default
            legend = True

        if xaxis and not isinstance(xaxis, Axis):
            raise ValueError(f"'xaxis' must be of type Axis but: '{type(xaxis)}'")
        if yaxis and not isinstance(yaxis, Axis):
            raise ValueError(f"'yaxis' must be of type Axis but: '{type(yaxis)}'")
        if yaxis_right and not isinstance(yaxis_right, Axis):
            raise ValueError(
                f"'yaxis_right' must be of type Axis but: '{type(yaxis_right)}'"
            )

        if facecolor is None:
            facecolor = ColorType.parse_color("white")
            assert facecolor is not None

        # property storage
        self._xaxis: Axis | None = None
        self._yaxis: Axis | None = None
        self._yaxis_right: Axis | None = None
        self._curves: list[Curve] = []
        self._areas: list[ShadedArea] = []
        self._bands: list[Band] = []
        self._figure: Figure | None = None

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.yaxis_right = yaxis_right
        self.curves = curves
        self.areas = areas
        self.bands = bands

        self.legend: bool = legend
        self.facecolor: ColorType = facecolor
        self.title_visible: bool = title_visible
        self.height: float | None = height
        self.width: float | None = width

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Plot(xaxis={self.xaxis} yaxis={self.yaxis} "
            f"yaxis_right={self.yaxis_right} #curves={len(self.curves)} "
            f"legend={self.legend})"
        )

    def __str__(self) -> str:
        """Get string."""
        return f"Plot({self.to_dict()})"

    def __copy__(self) -> Plot:
        """Copy the existing object."""
        if self.sid is None:
            raise ValueError("Plot requires an sid to be copied.")
        return Plot(
            sid=self.sid,
            name=self.name,
            xaxis=copy.copy(self.xaxis),
            yaxis=copy.copy(self.yaxis),
            yaxis_right=copy.copy(self.yaxis_right),
            curves=self.curves,
            areas=self.areas,
            bands=self.bands,
            legend=self.legend,
            facecolor=self.facecolor,
            title_visible=self.title_visible,
            height=self.height,
            width=self.width,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary of the plot attributes.
        """
        return {
            "sid": self.sid,
            "name": self.name,
            "xaxis": self.xaxis,
            "yaxis": self.yaxis,
            "yaxis_right": self.yaxis_right,
            "legend": self.legend,
            "facecolor": self.facecolor,
            "title_visible": self.title_visible,
            "curves": self.curves,
            "areas": self.areas,
            "bands": self.bands,
        }

    @property
    def figure(self) -> Figure:
        """Get figure for plot.

        Raises:
            ValueError: If the plot has no associated figure.
        """
        if not self._figure:
            raise ValueError(f"The plot '{self}' has no associated figure.")

        return self._figure

    @figure.setter
    def figure(self, value: Figure) -> None:
        """Set figure for plot."""
        self._figure = value

    @property
    def experiment(self) -> SimulationExperiment | None:
        """Get simulation experiment for this plot."""
        return self.figure.experiment

    @property
    def title(self) -> str | None:
        """Get title."""
        return self.name

    @title.setter
    def title(self, value: str) -> None:
        """Set title."""
        self.set_title(title=value)

    def set_title(self, title: str) -> None:
        """Set title.

        Args:
            title: Title of the plot.
        """
        self.name = title

    @property
    def xaxis(self) -> Axis | None:
        """Get xaxis."""
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value: str | Axis | None) -> None:
        """Set xaxis."""
        self.set_xaxis(label=value)

    def set_xaxis(
        self, label: str | Axis | None, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        Args:
            label: label of Axis or Axis object
            unit: unit of the Axis (added to label)
            **kwargs: additional Axis arguments
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_xaxis"
        self._xaxis = ax

    @property
    def yaxis(self) -> Axis | None:
        """Get yaxis."""
        return self._yaxis

    @yaxis.setter
    def yaxis(self, value: str | Axis | None) -> None:
        """Set yaxis."""
        self.set_yaxis(label=value)

    def set_yaxis(
        self, label: str | Axis | None, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        Args:
            label: label of Axis or Axis object
            unit: unit of the Axis (added to label)
            **kwargs: additional Axis arguments, e.g. `label_visible`
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_yaxis"
        self._yaxis = ax

    @property
    def yaxis_right(self) -> Axis | None:
        """Get right yaxis."""
        return self._yaxis_right

    @yaxis_right.setter
    def yaxis_right(self, value: str | Axis | None) -> None:
        """Set right yaxis."""
        self.set_yaxis_right(label=value)

    def set_yaxis_right(
        self, label: str | Axis | None, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        Args:
            label: label of Axis or Axis object
            unit: unit of the Axis (added to label)
            **kwargs: additional Axis arguments, e.g. `label_visible`
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_yaxis_right"
        self._yaxis_right = ax

    @staticmethod
    def _create_axis(
        label: str | Axis | None, unit: str | None = None, **kwargs: Any
    ) -> Axis | None:
        """Create axis from label or return given Axis.

        Args:
            label: label of Axis or Axis object
            unit: unit of the Axis (added to label)
            **kwargs: additional Axis arguments

        Returns:
            Axis or None if no label is given.
        """
        ax: Axis | None
        if not label:
            ax = None
        elif isinstance(label, Axis):
            ax = label
        else:
            ax = Axis(label=label, unit=unit, **kwargs)
        return ax

    def _set_order(self, abstract_curve: AbstractCurve) -> None:
        """Set order for given AbstractCurve.

        Args:
            abstract_curve: Curve or ShadedArea to set the order on.
        """
        if abstract_curve.order is None:
            orders = [
                ac.order
                for ac in [*self.curves, *self.areas, *self.bands]
                if ac.order is not None
            ]
            if not orders:
                abstract_curve.order = 0
            else:
                abstract_curve.order = max(orders) + 1

    def add_curve(self, curve: Curve) -> None:
        """Add Curve via the helper function.

        All additions must go via this function to ensure data registration.

        Args:
            curve: Curve to add.
        """
        if curve.sid is None:
            curve.sid = f"{self.sid}_curve{len(self.curves)}"

        self._set_order(curve)
        self.curves.append(curve)

    def add_area(self, area: ShadedArea) -> None:
        """Add ShadedArea via the helper function.

        All additions must go via this function to ensure data registration.

        Args:
            area: ShadedArea to add.
        """
        if area.sid is None:
            area.sid = f"{self.sid}_area{len(self.areas)}"

        self._set_order(area)
        self.areas.append(area)

    def add_band(self, band: Band) -> None:
        """Add a Band via the helper function.

        Args:
            band: Band to add.
        """
        if band.sid is None:
            band.sid = f"{self.sid}_band{len(self.bands)}"

        self._set_order(band)
        self.bands.append(band)

    @property
    def curves(self) -> list[Curve]:
        """Get curves."""
        return self._curves

    @curves.setter
    def curves(self, value: list[Curve] | None) -> None:
        """Set curves."""
        self._curves = []
        if value is not None:
            for curve in value:
                self.add_curve(curve)

    @property
    def areas(self) -> list[ShadedArea]:
        """Get areas."""
        return self._areas

    @areas.setter
    def areas(self, value: list[ShadedArea] | None) -> None:
        """Set areas."""
        self._areas = []
        if value is not None:
            for area in value:
                self.add_area(area)

    @property
    def bands(self) -> list[Band]:
        """Get bands."""
        return self._bands

    @bands.setter
    def bands(self, value: list[Band] | None) -> None:
        """Set bands."""
        self._bands = []
        if value is not None:
            for band in value:
                self.add_band(band)

    def band(
        self,
        x: Data,
        y: Data,
        across: str,
        quantiles: tuple[float, float] = (0.05, 0.95),
        median: bool = True,
        over: str | Sequence[str] | None = None,
        name: str | None = None,
        color: str | None = None,
        alpha: float = 0.3,
        yaxis_position: YAxisPosition | None = None,
        sid: str | None = None,
        order: int | None = None,
    ) -> Band:
        """Create a band of the median and the quantiles of y over `across` and add it.

        The arguments are those of `Band`.

        Returns:
            The band.
        """
        band = Band(
            x=x,
            y=y,
            across=across,
            quantiles=quantiles,
            median=median,
            over=over,
            sid=sid,
            name=name,
            color=color,
            alpha=alpha,
            order=order,
            yaxis_position=yaxis_position,
        )
        self.add_band(band)
        return band

    def curve(
        self,
        x: Data,
        y: Data,
        xerr: Data | None = None,
        yerr: Data | None = None,
        type: CurveType = CurveType.POINTS,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
        over: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create curve and add to plot.

        Args:
            x: x data
            y: y data
            xerr: x error data
            yerr: y error data
            type: type of curve (default points)
            style: style for curve
            yaxis_position: position of yaxis for this curve
            over: the dimensions of a scan with one line per point
            **kwargs: matplotlib styling kwargs
        """
        curve = Curve(
            x=x,
            y=y,
            xerr=xerr,
            yerr=yerr,
            type=type,
            style=style,
            yaxis_position=yaxis_position,
            over=over,
            **kwargs,
        )
        self.add_curve(curve)

    def add_data(
        self,
        xid: str,
        yid: str,
        xid_sd: str | None = None,
        xid_se: str | None = None,
        yid_sd: str | None = None,
        yid_se: str | None = None,
        count: int | str | None = None,
        dataset: str | None = None,
        task: str | None = None,
        label: str | None = "__yid__",
        type: CurveType = CurveType.POINTS,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
        sel: Mapping[str, Any] | None = None,
        over: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a data curve to the plot.

        Styling of curve is based on the provided style and matplotlib
        kwargs.

        Args:
            xid: index of x data
            yid: index of y data
            xid_sd: index of x SD data
            xid_se: index of x SE data
            yid_sd: index of y SD data
            yid_se: index of y SE data
            count: count for curve (number of subjects)
            dataset: dataset id
            task: task id
            label: label for curve (label=None for no label)
            type: type of curve (default points)
            style: style for curve
            yaxis_position: position of yaxis for this curve
            sel: labels of the dimensions of a task, or column values of the
                rows of a dataset, for every `Data` of the curve, see `Data`
            over: the dimensions of a scan with one line per point
            **kwargs: matplotlib styling kwargs

        Raises:
            ValueError: If the combination of arguments is not supported.
        """
        if yid_sd and yid_se:
            raise ValueError("Set either 'yid_sd' or 'yid_se', not both.")
        if dataset is not None and task is not None:
            raise ValueError("Set either 'dataset' or 'task', not both.")
        if dataset is None and task is None:
            raise ValueError("Set either 'dataset' or 'task'.")
        if count is not None and dataset is None:
            raise ValueError("'count' can only be set on a dataset")
        if label == "__nolabel__":
            logger.error(
                "'label' is set to '__nolabel__', to not add a label for "
                "a curve use 'label=None' instead."
            )
            label = None
        elif label == "__yid__":
            logger.debug(
                "No label provided on curve, using default label 'yid'. "
                "To not plot a label use 'label=None'"
            )
        if style is None and "markeredgecolor" not in kwargs:
            # the default of data, a style of the curve sets its own
            kwargs["markeredgecolor"] = "black"

        # xerr data
        xerr = None
        xerr_label = ""
        if xid_sd and xid_se:
            logger.warning("'xid_sd' and 'xid_se' set, using 'xid_sd'.")
        if xid_sd:
            if xid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            xerr_label = " ± SD"
            xerr = Data(xid_sd, dataset=dataset, task=task, sel=sel)
        elif xid_se:
            if xid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            xerr_label = " ± SE"
            xerr = Data(xid_se, dataset=dataset, task=task, sel=sel)

        _ = xerr_label

        # yerr data
        yerr = None
        yerr_label = ""
        if yid_sd and yid_se:
            logger.warning("'yid_sd' and 'yid_se' set, using 'yid_sd'.")
        if yid_sd:
            if yid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            yerr_label = " ± SD"
            yerr = Data(yid_sd, dataset=dataset, task=task, sel=sel)
        elif yid_se:
            if yid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            yerr_label = " ± SE"
            yerr = Data(yid_se, dataset=dataset, task=task, sel=sel)

        if label is not None:
            # add count information
            if count is None:
                count_label = ""
            else:
                # FIXME: this is duplicated in FitData
                if isinstance(count, int):
                    pass
                elif isinstance(count, str):
                    # resolve count data from dataset
                    experiment = self.experiment
                    if experiment is None:
                        raise ValueError(
                            f"The count '{count}' is a column of a dataset, but "
                            f"the plot belongs to no simulation experiment."
                        )
                    count_data = Data(index=count, dataset=dataset, task=task, sel=sel)
                    counts = count_data.get_data(experiment)
                    counts_unique = np.unique(np.asarray(counts.values))
                    if counts_unique.size > 1:
                        logger.warning("count is not unique for dataset: '%s'", counts)
                    count = int(counts_unique[0])
                else:
                    raise ValueError(
                        f"'count' must be integer or a column in a "
                        f"dataset, but type '{type(count)}'."
                    )
                count_label = f" (n={count})"

            label = f"{label}{yerr_label}{count_label}"

        self.curve(
            x=Data(xid, dataset=dataset, task=task, sel=sel),
            y=Data(yid, dataset=dataset, task=task, sel=sel),
            xerr=xerr,
            yerr=yerr,
            label=label,
            type=type,
            style=style,
            yaxis_position=yaxis_position,
            over=over,
            **kwargs,
        )


class SubPlot(BasePlotObject):
    """A SubPlot holds a plot in a Figure.

    The SubPlot defines the layout used by the plot, i.e., the position
    and number of panels the plot is spanning.
    """

    def __init__(
        self,
        plot: Plot,
        row: int | None = None,
        col: int | None = None,
        row_span: int = 1,
        col_span: int = 1,
        sid: str | None = None,
        name: str | None = None,
    ):
        """Initialize SubPlot.

        Args:
            plot: Plot of the subplot.
            row: row position of the plot in [1, num_rows]
            col: col position of the plot in [1, num_cols]
            row_span: number of rows the plot spans
            col_span: number of columns the plot spans
            sid: identifier of the subplot
            name: name of the subplot
        """
        super().__init__(sid=sid, name=name)
        self.plot = plot
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plot and its position in the figure."""
        return {
            "sid": self.sid,
            "name": self.name,
            "plot": self.plot,
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
        }

    def __str__(self) -> str:
        """Get string."""
        return f"<Subplot[{self.row},{self.col}]>"


#: the scopes of a legend, see `FigureSettings.legend_scope`: a legend per
#: plot or one per row of the grid
LEGEND_SCOPES: tuple[str, ...] = ("plot", "row")


@dataclass(frozen=True)
class FigureSettings:
    """The settings a figure is rendered with.

    The settings cascade, the more specific wins: the defaults of this class,
    the settings of the run (`figure_settings=` of `SimulationExperiment.run`,
    `ExperimentRunner.run_experiments` and the function `run_experiments`, a
    `FigureSettings` or a mapping of overrides of the defaults), the settings
    of the experiment (the class attribute `SimulationExperiment.figure_settings`,
    a mapping of overrides, which the subclasses of an experiment inherit) and
    the settings of the figure (`Figure(settings=...)`, a mapping of
    overrides), see `resolve`. An override is a field by its name, e.g.
    `{"fig_dpi": 300, "legend_fontsize": 10}`. The settings of a figure
    therefore do not depend on the experiments which ran before it.

    Attributes:
        fig_dpi: the resolution of a static image in dots per inch, which
            turns the size of a figure in inches into its pixels. An
            interactive page has a fixed 72 pixels per inch, see
            `serialization_plotly.PAGE_PIXELS_PER_INCH`, whatever `fig_dpi`.
        fig_facecolor: the colour of the figure.
        fig_subplots_wspace: the horizontal spacing between the panels, as a
            fraction of the panel width.
        fig_subplots_hspace: the vertical spacing between the panels, as a
            fraction of the panel height.
        panel_width: the width of a panel in inches, which gives the width of
            a figure without an explicit `width`.
        panel_height: the height of a panel in inches, which gives the height
            of a figure without an explicit `height`.
        fig_titlesize: the font size of the title of the figure.
        fig_titleweight: the font weight of the title of the figure.
        axes_titlesize: the font size of the title of a plot.
        axes_titleweight: the font weight of the title of a plot.
        axes_labelsize: the font size of the label of an axis.
        axes_labelweight: the font weight of the label of an axis.
        xtick_labelsize: the font size of the tick labels of an x axis.
        ytick_labelsize: the font size of the tick labels of a y axis.
        legend_fontsize: the font size of a legend.
        legend_position: `"inside"` a plot or `"outside"` right of it.
        legend_loc: the location of a legend inside a plot, a location of
            matplotlib, e.g. `"best"` or `"upper left"`.
        legend_scope: which plots a legend collects, `LEGEND_SCOPES`:
            `"plot"` a legend per plot, placed by `legend_position` and
            `legend_loc`; `"row"` one legend per row of the grid, right of
            the grid, with the entries of the plots of the row, an entry
            which reads and looks the same once (#327). The `legend` of a
            plot decides whether its entries enter the legend of its row.
    """

    fig_dpi: float = 72
    fig_facecolor: str = "white"
    fig_subplots_wspace: float = 0.3
    fig_subplots_hspace: float = 0.3
    panel_width: float = 7.0
    panel_height: float = 5.0
    fig_titlesize: float = 25
    fig_titleweight: str = "bold"
    axes_titlesize: float = 20
    axes_titleweight: str = "bold"
    axes_labelsize: float = 18
    axes_labelweight: str = "bold"
    xtick_labelsize: float = 15
    ytick_labelsize: float = 15
    legend_fontsize: float = 13
    legend_position: str = "inside"
    legend_loc: str = "best"
    legend_scope: str = "plot"

    def __post_init__(self) -> None:
        """Check the settings which take one of some values.

        Raises:
            ValueError: if `legend_scope` is none of `LEGEND_SCOPES`.
        """
        if self.legend_scope not in LEGEND_SCOPES:
            raise ValueError(
                f"The legend_scope of a figure is one of {list(LEGEND_SCOPES)}, "
                f"not {self.legend_scope!r}."
            )

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Get the names of the settings, in the order of the fields."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def check(cls, overrides: Mapping[str, Any] | None, owner: str) -> dict[str, Any]:
        """Check the overrides of a level of the cascade.

        Args:
            overrides: the overrides by the names of the settings.
            owner: the level for the message, e.g. "the experiment 'Exp'".

        Returns:
            A copy of the overrides, empty for `None`.

        Raises:
            TypeError: if the overrides are no mapping.
            ValueError: if a key is no setting, the message names the settings.
        """
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError(
                f"The figure settings of {owner} are a mapping of overrides by "
                f"the names of the settings, e.g. {{'fig_dpi': 300}}, not "
                f"'{type(overrides).__name__}'."
            )
        names = cls.names()
        unknown = [key for key in overrides if key not in names]
        if unknown:
            raise ValueError(
                f"The figure settings of {owner} have unknown keys "
                f"{', '.join(repr(k) for k in unknown)}. The settings are the "
                f"fields of `FigureSettings`: {', '.join(names)}."
            )
        return dict(overrides)

    @classmethod
    def resolve(
        cls,
        run: FigureSettings | Mapping[str, Any] | None = None,
        experiment: Mapping[str, Any] | None = None,
        figure: Mapping[str, Any] | None = None,
    ) -> FigureSettings:
        """Get the settings of the cascade, the more specific level wins.

        Args:
            run: the settings of the run, a `FigureSettings` or overrides of
                the defaults.
            experiment: the overrides of the experiment,
                `SimulationExperiment.figure_settings`.
            figure: the overrides of the figure, `Figure.settings`.

        Returns:
            The settings.

        Raises:
            ValueError: if a level has a key which is no setting.
        """
        settings = (
            run if isinstance(run, FigureSettings) else cls(**cls.check(run, "the run"))
        )
        for overrides, owner in [
            (experiment, "the experiment"),
            (figure, "the figure"),
        ]:
            checked = cls.check(overrides, owner)
            if checked:
                settings = replace(settings, **checked)
        return settings


def _removed_setting(name: str) -> str:
    """Get the message of a setting read, set or defined on `Figure` or a figure."""
    return (
        f"'{name}' is no attribute of the class `Figure` or of a figure any more: "
        f"set on the class it changed every later figure of the process, also "
        f"those of other experiments (#260). It is a field of `FigureSettings` "
        f"(the default is `FigureSettings().{name}`) and is set for a run with "
        f"`figure_settings={{'{name}': <value>}}` of `SimulationExperiment.run`, "
        f"`ExperimentRunner.run_experiments` or `run_experiments`; for an "
        f"experiment and its subclasses with `SimulationExperiment.figure_settings`, "
        f"i.e. `figure_settings = MappingProxyType({{'{name}': <value>}})` in the "
        f"class body; for one figure with `Figure(settings=...)`, i.e. "
        f"`Figure(..., settings={{'{name}': <value>}})`, or for a figure which "
        f"exists with `fig.settings = {{**fig.settings, '{name}': <value>}}`. A "
        f"figure of `figures_mpl()` reads the value of its experiment with "
        f"`FigureSettings.resolve(experiment=self.figure_settings).{name}`; the "
        f"settings of the run do not reach `figures_mpl()`."
    )


#: the settings of a figure, which `Figure` and a figure refuse as attributes
_SETTING_NAMES: frozenset[str] = frozenset(FigureSettings.names())


class _FigureType(type):
    """The type of `Figure`, which refuses the settings as class attributes.

    The settings were class attributes of `Figure`, which an experiment set for
    every later figure of the process (#260). Setting or reading one on the
    class raises with the way to set it now, rather than being ignored.
    """

    # hidden from the type checker, which would take any attribute of the class
    # for one of `__getattr__` and allow setting any, while without them it
    # reports a setting read or set on `Figure` as no attribute of it
    if not TYPE_CHECKING:

        def __getattr__(cls, name: str) -> NoReturn:
            """Raise for an attribute the class does not have."""
            if name in _SETTING_NAMES:
                raise AttributeError(_removed_setting(name), name=name, obj=cls)
            raise AttributeError(
                f"type object '{cls.__name__}' has no attribute '{name}'",
                name=name,
                obj=cls,
            )

        def __setattr__(cls, name: str, value: Any) -> None:
            """Set a class attribute, except a setting of a figure."""
            if name in _SETTING_NAMES:
                raise AttributeError(_removed_setting(name), name=name, obj=cls)
            super().__setattr__(name, value)


class Figure(BasePlotObject, metaclass=_FigureType):
    """A figure consists of multiple subplots.

    A reference to the experiment is required, so the plot can
    resolve the datasets and the simulations. The figure is rendered with
    the settings of `FigureSettings`, which its `settings` override.
    """

    def __init__(
        self,
        experiment: SimulationExperiment | None,
        sid: str,
        name: str | None = None,
        subplots: list[SubPlot] | None = None,
        height: float | None = None,
        width: float | None = None,
        num_rows: int = 1,
        num_cols: int = 1,
        settings: Mapping[str, Any] | None = None,
    ):
        """Initialize Figure.

        Args:
            experiment: Simulation experiment the figure belongs to.
            sid: identifier of the figure
            name: title of the figure
            subplots: subplots of the figure
            height: height of the figure in inches, without it the number of
                rows times the panel height of the settings it is rendered with
            width: width of the figure in inches, without it the number of
                columns times the panel width of the settings it is rendered with
            num_rows: number of panel rows
            num_cols: number of panel columns
            settings: the settings of this figure, overrides of the settings of
                the run and of the experiment by name, e.g.
                `{"legend_fontsize": 10}`, see `FigureSettings`.

        Raises:
            TypeError: if the settings are no mapping.
            ValueError: if a key of the settings is no field of `FigureSettings`.
        """
        super().__init__(sid, name)
        self.experiment: SimulationExperiment | None = experiment
        if subplots is None:
            subplots = []
        self.subplots: list[SubPlot] = subplots
        for subplot in self.subplots:
            # a plot resolves its data through its figure, see `Plot.experiment`;
            # `add_subplot` does this for the plots which are added later
            subplot.plot.figure = self
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        #: the explicit size in inches, `None` for the size of the panels of the
        #: settings the figure is rendered with, see `size`
        self.height: float | None = height
        self.width: float | None = width
        self._settings: dict[str, Any] = {}
        self.settings = settings

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a setting of `FigureSettings` in the body of a subclass.

        Defined in the body of a subclass, a setting would be an attribute the
        serializers never read, see `_FigureType`.

        Raises:
            AttributeError: for a name of a setting in the body of the subclass.
        """
        super().__init_subclass__(**kwargs)
        for name in sorted(_SETTING_NAMES.intersection(vars(cls))):
            raise AttributeError(_removed_setting(name), name=name, obj=cls)

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Figure(sid={self.sid} name={self.name} "
            f"shape=[{self.num_rows},{self.num_cols}] "
            f"#subplots={len(self.subplots)})"
        )

    # hidden from the type checker for the reason of `_FigureType`
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            """Raise for an attribute the figure does not have.

            A setting of `FigureSettings` raises with the way to set it, see
            `settings`.
            """
            if name in _SETTING_NAMES:
                raise AttributeError(_removed_setting(name), name=name, obj=self)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'",
                name=name,
                obj=self,
            )

        def __setattr__(self, name: str, value: Any) -> None:
            """Set an attribute, except a setting of `FigureSettings`."""
            if name in _SETTING_NAMES:
                raise AttributeError(_removed_setting(name), name=name, obj=self)
            super().__setattr__(name, value)

    @property
    def settings(self) -> Mapping[str, Any]:
        """Get the settings of this figure, overrides by name (read-only)."""
        return MappingProxyType(self._settings)

    @settings.setter
    def settings(self, value: Mapping[str, Any] | None) -> None:
        """Set the settings of this figure, overrides by name, `None` for none.

        Raises:
            TypeError: if the settings are no mapping.
            ValueError: if a key is no field of `FigureSettings`.
        """
        self._settings = FigureSettings.check(value, f"the figure '{self.sid}'")

    def resolved_settings(
        self,
        experiment: SimulationExperiment | None = None,
        run: FigureSettings | Mapping[str, Any] | None = None,
    ) -> FigureSettings:
        """Get the settings the figure is rendered with.

        The settings of the figure override those of its experiment, which
        override those of the run, see `FigureSettings.resolve`.

        Args:
            experiment: the experiment the figure is rendered for, its
                `figure_settings`; without it there is no level of an experiment.
            run: the settings of the run, a `FigureSettings` or overrides of
                the defaults.

        Returns:
            The settings.
        """
        return FigureSettings.resolve(
            run=run,
            experiment=experiment.figure_settings if experiment is not None else None,
            figure=self._settings,
        )

    def size(self, settings: FigureSettings) -> tuple[float, float]:
        """Get the width and the height in inches the figure is rendered with.

        The explicit `width` and `height`, else the number of columns and rows
        times the panel size of the settings.

        Args:
            settings: the settings the figure is rendered with, see
                `resolved_settings`.

        Returns:
            The width and the height.
        """
        width = (
            self.width
            if self.width is not None
            else self.num_cols * settings.panel_width
        )
        height = (
            self.height
            if self.height is not None
            else self.num_rows * settings.panel_height
        )
        return width, height

    def num_panels(self) -> int:
        """Get number of panel spots for plots.

        Plots can span multiple of these panels.

        Returns:
            Number of panels.
        """
        return self.num_cols * self.num_rows

    def set_title(self, title: str | None) -> None:
        """Set title.

        Args:
            title: Title of the figure.
        """
        self.name = title

    def create_plots(
        self,
        xaxis: Axis | None = None,
        yaxis: Axis | None = None,
        legend: bool = True,
    ) -> list[Plot]:
        """Create plots in the figure.

        Settings are applied to all generated plots. E.g. if an xaxis is provided
        all plots have a copy of this xaxis.

        Args:
            xaxis: xaxis copied to all plots
            yaxis: yaxis copied to all plots
            legend: flag to show legends

        Returns:
            Created plots.
        """
        plots = []
        for k in range(self.num_panels()):
            # create independent axis objects
            xax = deepcopy(xaxis) if xaxis else None
            yax = deepcopy(yaxis) if yaxis else None
            # create plot
            p = Plot(sid=f"{self.sid}__plot{k}", xaxis=xax, yaxis=yax, legend=legend)
            plots.append(p)
        self.add_plots(plots, copy_plots=False)
        return plots

    @property
    def plots(self) -> list[Plot]:
        """Get plots in this figure."""
        return self.get_plots()

    def get_plots(self) -> list[Plot]:
        """Get plots in this figure.

        Returns:
            Plots of all subplots.
        """
        return [subplot.plot for subplot in self.subplots]

    def add_subplot(
        self, plot: Plot, row: int, col: int, row_span: int = 1, col_span: int = 1
    ) -> Plot:
        """Add plot as subplot to figure.

        Be careful that individual subplots do not overlap when adding multiple
        subplots.

        Args:
            plot: Plot to add as subplot.
            row: row position for plot in [1, num_rows]
            col: col position for plot in [1, num_cols]
            row_span: span of figure with row + row_span <= num_rows
            col_span: span of figure with col + col_span <= num_cols

        Returns:
            The added plot.

        Raises:
            ValueError: If the position is outside of the figure.
        """
        if row <= 0:
            raise ValueError(f"row must be > 0, but 'row={row}'")
        if col <= 0:
            raise ValueError(f"col must be > 0, but 'col={col}'")
        if row > self.num_rows:
            raise ValueError(f"row must be <= num_rows, but '{row} > {self.num_rows}'")
        if col > self.num_cols:
            raise ValueError(f"col must be <= num_cols, but '{col} > {self.num_cols}'")
        if row + row_span - 1 > self.num_rows:
            raise ValueError(
                f"row + row_span - 1 must be <= num_rows, but "
                f"'{row + row_span - 1} > {self.num_rows}'"
            )
        if col + col_span - 1 > self.num_cols:
            raise ValueError(
                f"col + col_span - 1 must be <= num_cols, but "
                f"'{col + col_span - 1} > {self.num_cols}'"
            )

        if self.height and not plot.height:
            plot.height = self.height / self.num_rows * row_span
        if self.width and not plot.width:
            plot.width = self.width / self.num_cols * col_span

        # the plot resolves its data through the figure, see `Plot.experiment`
        plot.figure = self
        self.subplots.append(
            SubPlot(plot=plot, row=row, col=col, row_span=row_span, col_span=col_span)
        )
        return plot

    def add_plots(self, plots: list[Plot], copy_plots: bool = False) -> None:
        """Add plots to figure.

        For every plot a subplot is generated.

        Args:
            plots: Plots to add.
            copy_plots: Flag to copy the plots before adding.

        Raises:
            ValueError: If more plots than panels are provided.
        """
        # FIXME: handle correct copying of plots
        new_plots = [copy.copy(p) for p in plots] if copy_plots else plots

        if len(new_plots) > self.num_cols * self.num_rows:
            raise ValueError("Too many plots for figure")
        ridx = 1
        cidx = 1
        for plot in new_plots:
            # via `add_subplot`, which associates the plot with the figure and
            # gives it the size of its panel
            self.add_subplot(plot=plot, row=ridx, col=cidx)

            # increase indices for next plot
            if cidx == self.num_cols:
                cidx = 1
                ridx += 1
            else:
                cidx += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary of the figure attributes.
        """
        return {
            "sid": self.sid,
            "name": self.name,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "width": self.width,
            "height": self.height,
            "settings": self._settings,
            "subplots": self.subplots,
        }
