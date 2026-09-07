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
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

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

    MPL2SEDML_LINESTYLE_MAPPING: ClassVar[dict[str, LineType]] = {
        "": LineType.NONE,
        "-": LineType.SOLID,
        "solid": LineType.SOLID,
        ".": LineType.DOT,
        "dotted": LineType.DOT,
        "--": LineType.DASH,
        "dashed": LineType.DASH,
        "-.": LineType.DASHDOT,
        "dashdot": LineType.DASHDOT,
        "dashdotdotted": LineType.DASHDOTDOT,
    }
    SEDML2MPL_LINESTYLE_MAPPING: ClassVar[dict[LineType, MplLineStyle]] = {
        v: k for (k, v) in MPL2SEDML_LINESTYLE_MAPPING.items()
    }
    SEDML2MPL_LINESTYLE_MAPPING[LineType.DASHDOTDOT] = (0, (3, 5, 1, 5, 1, 5))

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
    }
    SEDML2MPL_MARKER_MAPPING: ClassVar[dict[MarkerType, str]] = {
        v: k for (k, v) in MPL2SEDML_MARKER_MAPPING.items()
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

    def resolve_style(self) -> Style:
        """Resolve all basestyle information.

        Resolves the actual style information.

        Returns:
            Style with all information of the base styles applied.
        """
        # recursive resolving of basestyle.
        if not self.base_style:
            return self

        # get base_style information
        logger.warning("Resolving base_style: %s", self.base_style)
        style = self.base_style.resolve_style()

        # overwrite information
        if self.line:
            if not style.line:
                style.line = deepcopy(self.line)
            else:
                for key in ["style", "color", "thickness"]:
                    if hasattr(self.line, key) and getattr(self.line, key):
                        logger.debug("line: %s = %s", key, getattr(self.line, key))
                        setattr(style.line, key, getattr(self.line, key))

        if self.marker:
            if not style.marker:
                style.marker = deepcopy(self.marker)
            else:
                for key in ["style", "size", "fill", "lineColor", "lineThickness"]:
                    if hasattr(self.marker, key) and getattr(self.marker, key):
                        logger.debug(
                            "marker: %s = %s", key, getattr(self.marker, key)
                        )
                        setattr(style.marker, key, getattr(self.marker, key))

        if self.fill:
            if not style.fill:
                style.fill = deepcopy(self.fill)
            else:
                for key in ["color", "secondColor"]:
                    if hasattr(self.fill, key) and getattr(self.fill, key):
                        logger.debug("fill: %s = %s", key, getattr(self.fill, key))
                        setattr(style.fill, key, getattr(self.fill, key))

        return style

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

    def _mpl_error_kwargs(self) -> dict[str, Any]:
        """Define keywords for error bars.

        Returns:
            Keyword arguments for error bars.
        """
        return {
            "error_kw": {
                # 'ecolor': "black",
                # 'elinewidth': 2.0,
            }
        }

    def to_mpl_points_kwargs(self) -> dict[str, Any]:
        """Convert to matplotlib point curve keyword arguments.

        Returns:
            Keyword arguments for matplotlib errorbar plots.
        """
        points_kwargs = self.to_mpl_curve_kwargs()
        for key in ["fill.color", "fill.second_color"]:
            if key in points_kwargs:
                points_kwargs.pop(key)
        error_kwargs = self._mpl_error_kwargs()
        return {
            **points_kwargs,
            **error_kwargs["error_kw"],
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
            **self._mpl_error_kwargs(),
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

        if self.fill:
            if self.fill.color:
                kwargs["color"] = self.fill.color.color
            # FIXME: second color not supported (gradients)
            # if self.fill.second_color:
            #    kwargs["second.color"] = self.fill.second_color

        return kwargs

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
        linestyle = Style.MPL2SEDML_LINESTYLE_MAPPING[kwargs.get("linestyle", "-")]
        line = Line(color=color, type=linestyle, thickness=kwargs.get("linewidth", 1.0))

        # Marker
        marker_symbol = Style.MPL2SEDML_MARKER_MAPPING[kwargs.get("marker", "")]
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

        Label and unit form together the axis label.
        To set the label directly use the name attribute.

        Args:
            label: label part of axis label
            unit: unit part of axis label
            name: complete axis label (overwrites label and unit)
            scale: Scale of the axis, i.e. "linear" or "log" axis.
            min: lower axis bound
            max: upper axis bound
            reverse: flag to reverse axis plot order
            grid: show grid lines along the axis
            label_visible: show/hide the label text
            ticks_visible: show/hide axis ticks
            style: style of the axis
        """
        super().__init__(sid=None, name=None)
        if label and name:
            ValueError("Either set label or name on Axis.")
        # if unit is None:
        #     unit = "?"
        if not name:
            if not label and not unit:
                name = ""
            elif unit != "dimensionless":
                name = f"{label} [{unit}]"
            else:
                name = f"{label} [-]"

        self.label: str | None = label
        self.name: str = name
        self.unit: str | None = unit
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
            name=self.name,
            unit=self.unit,
            scale=self.scale,
            min=self.min,
            max=self.max,
            grid=self.grid,
            label_visible=self.label_visible,
            ticks_visible=self.ticks_visible,
            style=copy.copy(self.style),
        )

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
            **kwargs: matplotlib styling arguments, `label` sets the name
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

        if "label" in kwargs:
            self.name = kwargs["label"]

        self.type: CurveType = type

        # parse additional arguments and create style
        if style:
            logger.warning("'style' is set, 'kwargs' style arguments are ignored.")
        else:
            kwargs = Curve._add_default_style_kwargs(kwargs, y.dtype)
            style = Style.from_mpl_kwargs(**kwargs)
        self.style = style
        self.kwargs = kwargs  # store for lookup

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Curve(sid={self.sid} name={self.name} type={self.type} order={self.order} "
            f"x={self.x is not None} y={self.y is not None}"
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

        if "capsize" not in d:
            d["capsize"] = 3
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
            f"x={self.x is not None} yfrom={self.yfrom is not None}"
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

        if facecolor is None:
            facecolor = ColorType.parse_color("white")
            assert facecolor is not None

        # property storage
        self._xaxis: Axis | None = None
        self._yaxis: Axis | None = None
        self._yaxis_right: Axis | None = None
        self._curves: list[Curve] = []
        self._areas: list[ShadedArea] = []
        self._figure: Figure | None = None

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.yaxis_right = yaxis_right
        self.curves = curves
        self.areas = areas

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
        return Plot(
            sid=self.sid,
            name=self.name,
            xaxis=copy.copy(self.xaxis),
            yaxis=copy.copy(self.yaxis),
            curves=self.curves,
            areas=self.areas,
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
    def experiment(self) -> SimulationExperiment:
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
                for ac in [*self.curves, *self.areas]
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

    def curve(
        self,
        x: Data,
        y: Data,
        xerr: Data | None = None,
        yerr: Data | None = None,
        type: CurveType = CurveType.POINTS,
        style: Style | None = None,
        yaxis_position: YAxisPosition | None = None,
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
        if "markeredgecolor" not in kwargs:
            kwargs["markeredgecolor"] = "black"

        # xerr data
        xerr = None
        xerr_label = ""
        if xid_sd and xid_se:
            logger.warning("'xid_sd' and 'xid_se' set, using 'xid_sd'.")
        if xid_sd:
            if xid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            xerr_label = "±SD"
            xerr = Data(xid_sd, dataset=dataset, task=task)
        elif xid_se:
            if xid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            xerr_label = "±SE"
            xerr = Data(xid_se, dataset=dataset, task=task)

        _ = xerr_label

        # yerr data
        yerr = None
        yerr_label = ""
        if yid_sd and yid_se:
            logger.warning("'yid_sd' and 'yid_se' set, using 'yid_sd'.")
        if yid_sd:
            if yid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            yerr_label = "±SD"
            yerr = Data(yid_sd, dataset=dataset, task=task)
        elif yid_se:
            if yid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            yerr_label = "±SE"
            yerr = Data(yid_se, dataset=dataset, task=task)

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
                    count_data = Data(index=count, dataset=dataset, task=task)
                    counts = count_data.get_data(self.experiment)
                    counts_unique = np.unique(counts.magnitude)
                    if counts_unique.size > 1:
                        logger.warning("count is not unique for dataset: '%s'", counts)
                    count = int(counts[0].magnitude)
                else:
                    raise ValueError(
                        f"'count' must be integer or a column in a "
                        f"dataset, but type '{type(count)}'."
                    )
                count_label = f" (n={count})"

            label = f"{label}{yerr_label}{count_label}"

        self.curve(
            x=Data(xid, dataset=dataset, task=task),
            y=Data(yid, dataset=dataset, task=task),
            xerr=xerr,
            yerr=yerr,
            label=label,
            type=type,
            style=style,
            yaxis_position=yaxis_position,
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

    def __str__(self) -> str:
        """Get string."""
        return f"<Subplot[{self.row},{self.col}]>"


class Figure(BasePlotObject):
    """A figure consists of multiple subplots.

    A reference to the experiment is required, so the plot can
    resolve the datasets and the simulations.
    """

    fig_dpi: int = 72
    fig_facecolor: str = "white"
    fig_subplots_wspace: float = 0.3  # vertical spacing of subplots (fraction of axes)
    fig_subplots_hspace: float = (
        0.3  # horizontal spacing of subplots (fraction of axes)
    )
    panel_width: float = 7.0
    panel_height: float = 5.0
    fig_titlesize: int = 25
    fig_titleweight: str = "bold"
    axes_titlesize: int = 20
    axes_titleweight: str = "bold"
    axes_labelsize: int = 18
    axes_labelweight: str = "bold"
    xtick_labelsize: int = 15
    ytick_labelsize: int = 15
    legend_fontsize: int = 13
    legend_position: str = "inside"  # "outside"
    legend_loc: str = "best"
    _area_interpolation_points: int = 300

    def __init__(
        self,
        experiment: SimulationExperiment,
        sid: str,
        name: str | None = None,
        subplots: list[SubPlot] | None = None,
        height: float | None = None,
        width: float | None = None,
        num_rows: int = 1,
        num_cols: int = 1,
    ):
        """Initialize Figure.

        Args:
            experiment: Simulation experiment the figure belongs to.
            sid: identifier of the figure
            name: title of the figure
            subplots: subplots of the figure
            height: height of the figure (calculated from panels if not set)
            width: width of the figure (calculated from panels if not set)
            num_rows: number of panel rows
            num_cols: number of panel columns
        """
        super().__init__(sid, name)
        self.experiment: SimulationExperiment = experiment
        if subplots is None:
            subplots = []
        self.subplots: list[SubPlot] = subplots
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self._height: float
        self._width: float
        self.height = height
        self.width = width

    def __repr__(self) -> str:
        """Get representation string."""
        return (
            f"Figure(sid={self.sid} name={self.name} "
            f"shape=[{self.num_rows},{self.num_cols}] "
            f"#subplots={len(self.subplots)})"
        )

    @property
    def height(self) -> float:
        """Get height."""
        return self._height

    @height.setter
    def height(self, value: float | None) -> None:
        """Set height."""
        if value is None:
            value = self.num_rows * self.panel_height
        self._height = value

    @property
    def width(self) -> float:
        """Get width."""
        return self._width

    @width.setter
    def width(self, value: float | None) -> None:
        """Set width."""
        if value is None:
            value = self.num_cols * self.panel_width
        self._width = value

    def num_subplots(self) -> int:
        """Get number of subplots.

        Returns:
            Number of subplots.
        """
        return len(self.subplots)

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
                f"row + row_span must be <= num_rows, but "
                f"'{row + row_span} > {self.num_rows}'"
            )
        if col + col_span - 1 > self.num_cols:
            raise ValueError(
                f"col + col_span - 1 must be <= num_cols, but "
                f"'{col + col_span} > {self.num_cols}'"
            )

        if self.height and not plot.height:
            plot.height = self.height / self.num_rows * row_span
        if self.width and not plot.width:
            plot.width = self.width / self.num_cols * col_span

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
            self.subplots.append(
                SubPlot(plot=plot, row=ridx, col=cidx, row_span=1, col_span=1)
            )

            # increase indices for next plot
            if cidx == self.num_cols:
                cidx = 1
                ridx += 1
            else:
                cidx += 1
            # set the figure for the plot
            plot.figure = self

    @staticmethod
    def from_plots(
        sid: str, plots: list[Plot], experiment: SimulationExperiment
    ) -> Figure:
        """Create figure object from list of plots.

        Args:
            sid: identifier of the figure
            plots: plots stacked in a single column
            experiment: simulation experiment of the figure

        Returns:
            Figure with the plots.
        """
        num_plots = len(plots)
        return Figure(
            experiment=experiment,
            sid=sid,
            num_rows=num_plots,
            num_cols=1,
            height=num_plots * Figure.panel_height,
            width=Figure.panel_width,
            subplots=[
                SubPlot(plot, row=(k + 1), col=1) for k, plot in enumerate(plots)
            ],
        )

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
            "subplots": self.subplots,
        }
