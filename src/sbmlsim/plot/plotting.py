"""
Classes for storing plotting information.

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
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from matplotlib.colors import to_hex, to_rgba
from pymetadata import log

from sbmlsim.data import Data


logger = log.get_logger(__name__)

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

    def __init__(self, sid: str, name: str):
        """Initialize BasePlotObject."""
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
        """Initialize ColorType."""
        if color is None:
            raise ValueError("color cannot be NoneType")

        self.color = color

    def to_dict(self):
        """Convert for serialization."""
        return self.color

    def __repr__(self) -> str:
        """Get string representation."""
        return self.color

    @staticmethod
    def parse_color(color: str, alpha: float = 1.0) -> Optional[ColorType]:
        """Parse given color and add alpha information.

        :param color:
        :param alpha:
        :return: ColorType or None
        """
        # https://matplotlib.org/stable/tutorials/colors/colors.html
        if color is None or len(color) == 0:
            return None

        elif isinstance(color, str) and color.startswith("#"):
            # handle hex colors
            if len(color) == 7:
                # parse alpha
                color_hex = color + "%02x" % round(alpha * 255)
            elif len(color) == 9:
                color_hex = color
                if alpha != 1.0:
                    logger.warning(
                        f"alpha ignored for hex colors with alpha channel: "
                        f"'{color}', alpha={alpha}."
                    )
            else:
                logger.error(f"Incorrect hex color: '{color}'")

        else:
            color = to_rgba(color, alpha)
            color_hex = to_hex(color, keep_alpha=True)

        return ColorType(color_hex)


@dataclass
class Line:
    """Style of a line."""

    type: LineType = LineType.SOLID
    color: ColorType = None
    thickness: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "color": self.color,
            "thickness": self.thickness,
        }


@dataclass
class Marker:
    """Style of a marker."""

    size: float = 6.0
    type: MarkerType = MarkerType.NONE
    fill: ColorType = None
    line_color: ColorType = None
    line_thickness: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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

    color: ColorType = None
    second_color: ColorType = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "color": self.color,
            "second_color": self.second_color,
        }


class Style(BasePlotObject):
    """Style class.

    Storing styling informatin about line, marker and fill.
    Styles can be derived from other styles based on the the
    base_style attribute.
    """

    def __init__(
        self,
        sid: str = None,
        name: str = None,
        base_style: Optional["Style"] = None,
        line: Optional[Line] = None,
        marker: Optional[Marker] = None,
        fill: Optional[Fill] = None,
    ):
        """Initialize Style."""
        super(Style, self).__init__(sid, name)

        # using default styling if not otherwise provided
        if marker is None:
            marker = Marker()
        if line is None:
            line = Line()

        self.base_style: Optional["Style"] = base_style
        self.line: Optional[Line] = line
        self.marker: Optional[Marker] = marker
        self.fill: Optional[Fill] = fill

    def resolve_style(self) -> "Style":
        """Resolve all basestyle information.

        Resolves the actual style information.
        """
        # recursive resolving of basestyle.
        if not self.base_style:
            return self

        # get base_style information
        logger.warning(f"Resolving base_style: {self.base_style}")
        style = self.base_style.resolve_style()

        # overwrite information
        if self.line:
            if not style.line:
                style.line = deepcopy(self.line)
            else:
                for key in ["style", "color", "thickness"]:
                    if hasattr(self.line, key) and getattr(self.line, key):
                        logger.debug(f"line: {key} = {getattr(self.line, key)}")
                        setattr(style.line, key, getattr(self.line, key))

        if self.marker:
            if not style.marker:
                style.marker = deepcopy(self.marker)
            else:
                for key in ["style", "size", "fill", "lineColor", "lineThickness"]:
                    if hasattr(self.marker, key) and getattr(self.marker, key):
                        logger.debug(f"marker: {key} = {getattr(self.marker, key)}")
                        setattr(style.marker, key, getattr(self.marker, key))

        if self.fill:
            if not style.fill:
                style.fill = deepcopy(self.fill)
            else:
                for key in ["color", "secondColor"]:
                    if hasattr(self.fill, key) and getattr(self.fill, key):
                        logger.debug(f"fill: {key} = {getattr(self.fill, key)}")
                        setattr(style.fill, key, getattr(self.fill, key))

        return style

    def __repr__(self) -> str:
        """Get string presentation."""
        return (
            f"{self.sid} (base_style={self.base_style}) [marker={self.marker}; line={self.line}; "
            f"fill={self.fill}]"
        )

    def __copy__(self) -> "Style":
        """Copy axis object."""
        return Style(
            sid=self.sid,
            name=self.name,
            line=self.line,
            marker=self.marker,
            fill=self.fill,
        )

    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
    MPL2SEDML_LINESTYLE_MAPPING = {
        "": LineType.NONE,
        "-": LineType.SOLID,
        "solid": LineType.SOLID,
        ".": LineType.DOT,
        "dotted": LineType.DOT,
        "--": LineType.DASH,
        "dashed": LineType.DASH.DASH,
        "-.": LineType.DASHDOT,
        "dashdot": LineType.DASHDOT,
        "dashdotdotted": LineType.DASHDOTDOT,
    }
    SEDML2MPL_LINESTYLE_MAPPING = {
        v: k for (k, v) in MPL2SEDML_LINESTYLE_MAPPING.items()
    }
    SEDML2MPL_LINESTYLE_MAPPING[LineType.DASHDOTDOT] = (0, (3, 5, 1, 5, 1, 5))

    MPL2SEDML_MARKER_MAPPING = {
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
    SEDML2MPL_MARKER_MAPPING = {v: k for (k, v) in MPL2SEDML_MARKER_MAPPING.items()}

    def to_mpl_curve_kwargs(self) -> dict:
        """Convert to matplotlib curve keyword arguments."""
        kwargs: dict[str, Any] = {}
        if self.line:
            if self.line.color:
                kwargs["color"] = self.line.color.color
            if self.line.type:
                kwargs["linestyle"] = Style.SEDML2MPL_LINESTYLE_MAPPING[self.line.type]
            if self.line.thickness:
                kwargs["linewidth"] = self.line.thickness
        if self.marker:
            if self.marker.type:
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
        """Define keywords for error bars."""
        error_kwargs = {
            "error_kw": {
                # 'ecolor': "black",
                # 'elinewidth': 2.0,
            }
        }
        return error_kwargs

    def to_mpl_points_kwargs(self) -> dict[str, Any]:
        """Convert to matplotlib point curve keyword arguments."""
        points_kwargs = self.to_mpl_curve_kwargs()
        for key in ["fill.color", "fill.second_color"]:
            if key in points_kwargs:
                points_kwargs.pop(key)
        error_kwargs = self._mpl_error_kwargs()
        return {
            **points_kwargs,
            **error_kwargs["error_kw"],
        }

    def to_mpl_bar_kwargs(self):
        """Convert to matplotlib bar curve keyword arguments."""
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
        """Define keyword dictionary for a shaded area."""
        kwargs: dict[str, Any] = {}

        if self.line:
            if self.line.color:
                kwargs["edgecolor"] = self.line.color.color
            if self.line.type:
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
    def from_mpl_kwargs(**kwargs) -> "Style":
        """Create style from matplotlib arguments.

        :keyword alpha: alpha setting
        :keyword color: color setting
        :param kwargs:
        :return:
        """
        color = ColorType.parse_color(
            color=kwargs.get("color", None),
            alpha=kwargs.get("alpha", 1.0),
        )
        line_color = ColorType.parse_color(
            color=kwargs.get("markeredgecolor", None),
        )

        # Line
        linestyle = Style.MPL2SEDML_LINESTYLE_MAPPING[kwargs.get("linestyle", "-")]
        line = Line(color=color, type=linestyle, thickness=kwargs.get("linewidth", 1.0))

        # Marker
        marker_symbol = Style.MPL2SEDML_MARKER_MAPPING[kwargs.get("marker", "")]
        marker = Marker(
            type=marker_symbol,
            size=kwargs.get("markersize", None),
            fill=kwargs.get("markerfacecolor", color),
            line_color=line_color,
            line_thickness=kwargs.get("markeredgewidth", None),
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
        label: str = None,
        unit: str = None,
        name: str = None,
        scale: AxisScale = AxisScale.LINEAR,
        min: float = None,
        max: float = None,
        reverse: bool = False,
        grid: bool = False,
        label_visible: bool = True,
        ticks_visible: bool = True,
        style: Style = None,
    ):
        """Axis object.

        Label and unit form together the axis label.
        To set the label directly use the name attribute.

        :param label: label part of axis label
        :param unit: unit part of axis label
        :param scale: Scale of the axis, i.e. "linear" or "log" axis.
        :param min: lower axis bound
        :param max: upper axis bound
        :param reverse: flag to reverse axis plot order
        :param grid: show grid lines along the axis
        :param label_visible: show/hide the label text
        :param ticks_visible: show/hide axis ticks
        """
        super(Axis, self).__init__(sid=None, name=None)
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

        self.label: str = label
        self.name: str = name
        self.unit: str = unit
        self.scale: AxisScale = scale
        self.min: float = min
        self.max: float = max
        self.reverse: bool = reverse
        self.grid: bool = grid
        self.label_visible: bool = label_visible
        self.ticks_visible: bool = ticks_visible
        self.style = style

    def __repr__(self) -> str:
        """Get string."""
        return (
            f"Axis(sid={self.sid} name={self.name} scale={self.scale} "
            f"min={self.min} max={self.max})"
        )

    def __str__(self) -> str:
        """Get string."""
        return f"Axis({self.name, self.scale})"

    def __copy__(self) -> "Axis":
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
            style=self.style.__copy__(),
        )

    @property
    def scale(self) -> AxisScale:
        """Get axis scale."""
        return self._scale

    @scale.setter
    def scale(self, scale: AxisScale) -> None:
        """Set axis scale."""
        if isinstance(scale, str):
            if scale == "linear":
                scale = AxisScale.LINEAR
            elif scale in {"log", "log10"}:
                scale = AxisScale.LOG10
            else:
                ValueError(f"Unsupported axis scale: '{scale}'")
        self._scale = scale

    def to_dict(self):
        """Convert to dictionary."""
        d = {
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
        return d


class AbstractCurve(BasePlotObject):
    """Base class of Curves and ShadedAreas."""

    def __init__(
        self,
        sid: str,
        name: str,
        x: Data = None,
        order: int = None,
        style: Style = None,
        yaxis_position: YAxisPosition = None,
    ):
        """Abstract base class of Curve and ShadedArea.

        :param sid:
        :param name: label of the curve
        :param x:
        :param order:
        :param style:
        :param yaxis_position:
        """
        super(AbstractCurve, self).__init__(sid, name)
        self.x = x
        self.order = order
        self.style = style
        self.yaxis_position = yaxis_position


class Curve(AbstractCurve):
    """Curve object."""

    def __init__(
        self,
        x: Data,
        y: Data,
        sid=None,
        name=None,
        xerr: Data = None,
        yerr: Data = None,
        order=None,
        type: CurveType = CurveType.POINTS,
        style: Style = None,
        yaxis_position: YAxisPosition = None,
        **kwargs,
    ):
        """Initialize Curve."""
        super(Curve, self).__init__(
            sid=sid,
            name=name if name else y.name,
            x=x,
            order=order,
            style=style,
            yaxis_position=yaxis_position,
        )
        self.y = y

        # set symmetrical
        self.xerr: Data = xerr
        self.yerr: Data = yerr

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
    def _add_default_style_kwargs(d: dict, dtype: str) -> dict:
        """Add the default plotting style arguments."""

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

    def to_dict(self):
        """Convert Curve to dictionary."""
        d = {
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
        return d


class ShadedArea(AbstractCurve):
    """ShadedArea class."""

    def __init__(
        self,
        x: Data,
        yfrom: Data,
        yto: Data,
        order: Optional[int] = None,
        style: Style = None,
        yaxis_position: YAxisPosition = None,
        **kwargs,
    ):
        """Initialize ShadedArea."""
        super(ShadedArea, self).__init__(
            sid=None,
            name=None,
            x=x,
            order=order,
            style=style,
            yaxis_position=yaxis_position,
        )
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

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid if self.x else None,
            "yfrom": self.yfrom.sid if self.yfrom else None,
            "yto": self.yto.sid if self.yto else None,
            "yaxis_position": self.yaxis_position,
            "style": self.style,
            "order": self.order,
        }
        return d


class Plot(BasePlotObject):
    """Plot panel.

    A plot is the basic element of a plot. This corresponds to a single
    panel or axes combination in a plot. Multiple plots create a figure.
    """

    def __init__(
        self,
        sid: str,
        name: str = None,
        xaxis: Axis = None,
        yaxis: Axis = None,
        yaxis_right: Axis = None,
        curves: list[Curve] = None,
        areas: list[ShadedArea] = None,
        legend: bool = True,
        facecolor: ColorType = None,
        title_visible: bool = True,
        height: float = None,
        width: float = None,
    ):
        """Initialize plot.

        :param sid: Sid of the plot
        :param name: title of the plot
        :param legend: boolean flag to show or hide legend
        :param xaxis: x-Axis
        :param yaxis: y-Axis
        :param curves: list of curves for the plots
        :param facecolor: color of the plot.
        :param title_visible: boolean flag to show the title
        :param height: plot height (should be set on figure)
        :param width: plot width (should be set on figure)
        """
        super(Plot, self).__init__(sid, name)
        if curves is None:
            curves = list()
        if legend is None:
            # legend by default
            legend = True

        if xaxis and not isinstance(xaxis, Axis):
            raise ValueError(f"'xaxis' must be of type Axis but: '{type(xaxis)}'")
        if yaxis and not isinstance(yaxis, Axis):
            raise ValueError(f"'yaxis' must be of type Axis but: '{type(yaxis)}'")

        if facecolor is None:
            facecolor = ColorType.parse_color("white")

        # property storage
        self._xaxis: Axis = None
        self._yaxis: Axis = None
        self._yaxis_right: Axis = None
        self._curves: list[Curve] = None
        self._areas: list[ShadedArea] = None
        self._figure: Figure = None

        self.xaxis: Axis = xaxis
        self.yaxis: Axis = yaxis
        self.yaxis_right: Axis = yaxis_right
        self.curves: list[Curve] = curves
        self.areas: list[ShadedArea] = areas

        self.legend: bool = legend
        self.facecolor: ColorType = facecolor
        self.title_visible: bool = title_visible
        self.height = height
        self.width = width

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

    def __copy__(self) -> "Plot":
        """Copy the existing object."""
        return Plot(
            sid=self.sid,
            name=self.name,
            xaxis=Axis.__copy__(self.xaxis),
            yaxis=Axis.__copy__(self.yaxis),
            curves=self.curves,
            areas=self.areas,
            legend=self.legend,
            facecolor=self.facecolor,
            title_visible=self.title_visible,
            height=self.height,
            width=self.width,
        )

    def to_dict(self):
        """Convert to dictionary."""
        d = {
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
        return d

    @property
    def figure(self) -> "Figure":
        """Get figure for plot."""
        if not self._figure:
            raise ValueError(f"The plot '{self}' has no associated figure.")

        return self._figure

    @figure.setter
    def figure(self, value: "Figure"):
        """Set figure for plot."""
        self._figure = value

    @property
    def experiment(self):
        """Get simulation experiment for this plot."""
        return self.figure.experiment

    @property
    def title(self) -> str:
        """Get title."""
        return self.name

    @title.setter
    def title(self, value: str) -> None:
        """Set title."""
        self.set_title(title=value)

    def set_title(self, title: str) -> None:
        """Set title."""
        self.name = title

    @property
    def xaxis(self) -> Axis:
        """Get xaxis."""
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value: Axis) -> None:
        """Set xaxis."""
        self.set_xaxis(label=value)

    def set_xaxis(
        self, label: Optional[Union[str, Axis]], unit: str = None, **kwargs
    ) -> None:
        """Set axis with all axes attributes.

        All argument of Axis are supported.
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_xaxis"
        self._xaxis = ax

    @property
    def yaxis(self) -> Axis:
        """Get yaxis."""
        return self._yaxis

    @yaxis.setter
    def yaxis(self, value: Axis) -> None:
        """Set yaxis."""
        self.set_yaxis(label=value)

    def set_yaxis(self, label: Union[str, Axis], unit: str = None, **kwargs):
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        :param label:
        :param unit:
        :keyword label_visible:
        :param kwargs:
        :return:
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_yaxis"
        self._yaxis = ax

    @property
    def yaxis_right(self) -> Axis:
        """Get right yaxis."""
        return self._yaxis_right

    @yaxis_right.setter
    def yaxis_right(self, value: Axis) -> None:
        """Set right yaxis."""
        self.set_yaxis_right(label=value)

    def set_yaxis_right(
        self, label: Union[str, Axis], unit: str = None, **kwargs
    ) -> None:
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        :param label: label of Axis
        :param unit: unit of the Axis (added to label)
        :keyword label_visible: boolean flag to make the axis visible or not.
        :param kwargs:
        :return:
        """
        ax = Plot._create_axis(label=label, unit=unit, **kwargs)
        if ax and ax.sid is None:
            ax.sid = f"{self.sid}_yaxis_right"
        self._yaxis_right = ax

    @staticmethod
    def _create_axis(
        label: Optional[Union[str, Axis]], unit: str = None, **kwargs
    ) -> Optional[Axis]:
        if not label:
            ax = None
        elif isinstance(label, Axis):
            ax = label
        else:
            ax = Axis(label=label, unit=unit, **kwargs)
        return ax

    def _set_order(self, abstract_curve: AbstractCurve):
        """Set order for given AbstractCurve."""
        if abstract_curve.order is None:
            if not self.curves and not self.areas:
                abstract_curve.order = 0
            else:
                abstract_curve.order = (
                    max([ac.order for ac in self.curves + self.areas]) + 1
                )

    def add_curve(self, curve: Curve):
        """Add Curve via the helper function.

        All additions must go via this function to ensure data registration.
        """
        if curve.sid is None:
            curve.sid = f"{self.sid}_curve{len(self.curves)}"

        self._set_order(curve)
        self.curves.append(curve)

    def add_area(self, area: ShadedArea):
        """Add ShadedArea via the helper function.

        All additions must go via this function to ensure data registration.
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
    def curves(self, value: list[Curve]):
        """Set curves."""
        self._curves = list()
        if value is not None:
            for curve in value:
                self.add_curve(curve)

    @property
    def areas(self) -> list[ShadedArea]:
        """Get areas."""
        return self._areas

    @areas.setter
    def areas(self, value: list[ShadedArea]) -> None:
        """Set areas."""
        self._areas = list()
        if value is not None:
            for area in value:
                self.add_area(area)

    def curve(
        self,
        x: Data,
        y: Data,
        xerr: Data = None,
        yerr: Data = None,
        type: CurveType = CurveType.POINTS,
        style: Style = None,
        yaxis_position: YAxisPosition = None,
        **kwargs,
    ):
        """Create curve and add to plot."""
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
        xid_sd=None,
        xid_se=None,
        yid_sd=None,
        yid_se=None,
        count: Union[int, str] = None,
        dataset: str = None,
        task: str = None,
        label: Optional[str] = "__yid__",
        type: CurveType = CurveType.POINTS,
        style: Style = None,
        yaxis_position: YAxisPosition = None,
        **kwargs,
    ):
        """Add a data curve to the plot.

        Styling of curve is based on the provided style and matplotlib
        kwargs.

        :param xid: index of x data
        :param yid: index of y data
        :param yid_sd: index of y SD data
        :param yid_se: index of y SE data
        :param count: count for curve (number of subjects)
        :param dataset: dataset id
        :param task: task id
        :param label: label for curve (label=None for no label)
        :param type: type of curve (default points)
        :param style: style for curve
        :param yaxis_position: position of yaxis for this curve
        :param kwargs: matplotlib styling kwargs
        :return:
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
                        logger.warning(f"count is not unique for dataset: '{counts}'")
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
        row: int = None,
        col: int = None,
        row_span: int = 1,
        col_span: int = 1,
        sid: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize SubPlot."""
        super(SubPlot, self).__init__(sid=sid, name=name)
        self.plot = plot
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span

    def __str__(self):
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
        experiment: "SimulationExperiment",  # noqa: F821
        sid: str,
        name: str = None,
        subplots: list[SubPlot] = None,
        height: float = None,
        width: float = None,
        num_rows: int = 1,
        num_cols: int = 1,
    ):
        """Initialize Figure."""
        super(Figure, self).__init__(sid, name)
        self.experiment: "SimulationExperiment" = experiment  # noqa: F821
        if subplots is None:
            subplots = list()
        self.subplots: list[SubPlot] = subplots
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self._height: float = height
        self._width: float = width
        self.height: float = height
        self.width: float = width
        # print(f"[{self.num_rows}, {self.num_cols}], ({self.height}, {self.width})")
        # print(f"Figure: [{self.panel_height}, {self.panel_width}]")

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
    def height(self, value: float) -> None:
        """Set height."""
        if value is None:
            value = self.num_rows * self.panel_height
        self._height = value

    @property
    def width(self) -> float:
        """Get width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set width."""
        if value is None:
            value = self.num_cols * self.panel_width
        self._width = value

    def num_subplots(self) -> int:
        """Get number ofsubplots."""
        return len(self.subplots)

    def num_panels(self) -> int:
        """Get number of panel spots for plots.

        Plots can span multiple of these panels.
        """
        return self.num_cols * self.num_rows

    def set_title(self, title):
        """Set title."""
        self.name = title

    def create_plots(
        self, xaxis: Axis = None, yaxis: Axis = None, legend: bool = True
    ) -> list[Plot]:
        """Create plots in the figure.

        Settings are applied to all generated plots. E.g. if an xaxis is provided
        all plots have a copy of this xaxis.
        """
        plots = []
        for k in range(self.num_panels()):
            # create independent axis objects
            xax = deepcopy(xaxis) if xaxis else None
            yax = deepcopy(yaxis) if yaxis else None
            # create plot
            p = Plot(sid=f"{self.sid}__plot{k}", xaxis=xax, yaxis=yax, legend=legend)
            p.set_figure = self
            plots.append(p)
        self.add_plots(plots, copy_plots=False)
        return plots

    @property
    def plots(self) -> list[Plot]:
        """Get plots in this figure."""
        return self.get_plots()

    def get_plots(self) -> list[Plot]:
        """Get plots in this figure."""
        return [subplot.plot for subplot in self.subplots]

    def add_subplot(
        self, plot: Plot, row: int, col: int, row_span: int = 1, col_span: int = 1
    ) -> Plot:
        """Add plot as subplot to figure.

        Be careful that individual subplots do not overlap when adding multiple
        subplots.

        :param plot: Plot to add as subplot.
        :param row: row position for plot in [1, num_rows]
        :param col: col position for plot in [1, num_cols]
        :param row_span: span of figure with row + row_span <= num_rows
        :param col_span: span of figure with col + col_span <= num_cols
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
        """

        # FIXME: handle correct copying of plots
        if copy_plots:
            new_plots = [copy.copy(p) for p in plots]
        else:
            new_plots = plots

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
    def from_plots(sid, plots: list[Plot]) -> "Figure":
        """Create figure object from list of plots."""
        num_plots = len(plots)
        return Figure(
            sid=sid,
            num_rows=num_plots,
            num_cols=1,
            height=num_plots * Figure.panel_height,
            width=Figure.panel_width,
            subplots=[
                SubPlot(plot, row=(k + 1), col=1) for k, plot in enumerate(plots)
            ],
        )

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "sid": self.sid,
            "name": self.name,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "width": self.width,
            "height": self.height,
            "subplots": self.subplots,
        }
        return d
