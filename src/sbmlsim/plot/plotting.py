"""
Base classes for storing plotting information.

The general workflow of generating plotting information is the following.

1. Within simulation experiments abstract plotting information is stored.
    i.e., how from the data plots can be generated.


"""
import copy
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from matplotlib.colors import to_hex, to_rgba

from sbmlsim.data import Data, DataSet


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


class BasePlotObject(object):
    """Base class for plotting objects."""

    def __init__(self, sid: str, name: str):
        self.sid = sid
        self.name = name


class LineType(Enum):
    NONE = 1
    SOLID = 2
    DASH = 3
    DOT = 4
    DASHDOT = 5
    DASHDOTDOT = 6


class MarkerType(Enum):
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


class ColorType(object):
    def __init__(self, color: str):
        if color is None:
            raise ValueError("color cannot be NoneType")

        self.color = color

    def to_dict(self):
        return self.color

    def __repr__(self):
        return self.color


@dataclass
class Line(object):
    type: LineType
    color: ColorType
    thickness: float

    def to_dict(self):
        return {
            "type": self.type,
            "color": self.color,
            "thickness": self.thickness,
        }


@dataclass
class Marker(object):
    size: float
    type: MarkerType
    fill: ColorType
    line_color: ColorType
    line_thickness: float = 1.0

    def to_dict(self):
        return {
            "size": self.size,
            "type": self.type,
            "fill": self.fill,
            "line_color": self.line_color,
            "line_thickness": self.line_thickness,
        }


@dataclass
class Fill(object):
    color: ColorType
    second_color: ColorType = None


class Style(BasePlotObject):
    def __init__(
        self,
        sid: str = None,
        name: str = None,
        base_style: "Style" = None,
        line: Line = None,
        marker: Marker = None,
        fill: Fill = None,
    ):

        # FIXME: base_style not handled
        super(Style, self).__init__(sid, name)
        self.line = line
        self.marker = marker
        self.fill = fill

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

    def to_mpl_kwargs(self) -> Dict:
        """Convert to matplotlib plotting arguments"""
        kwargs = {}
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
            pass

        return kwargs

    @staticmethod
    def parse_color(color: str, alpha: float = 1.0) -> Optional[ColorType]:
        """Parse given color and add alpha information.

        :param color:
        :param alpha:
        :return: ColorType or None
        """
        # https://matplotlib.org/stable/tutorials/colors/colors.html
        if color is None:
            return None

        elif color.startswith("#"):
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

    @staticmethod
    def from_mpl_kwargs(**kwargs) -> "Style":
        """Creates style from matplotlib arguments.

        :keyword alpha: alpha setting
        :keyword color: color setting
        :param kwargs:
        :return:
        """
        color = Style.parse_color(
            color=kwargs.get("color", None),
            alpha=kwargs.get("alpha", 1.0),
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
            line_color=kwargs.get("markeredgecolor", None),
            line_thickness=kwargs.get("markeredgewidth", None),
        )

        # Fill
        # FIXME: implement

        return Style(line=line, marker=marker, fill=None)


class Axis(BasePlotObject):
    class AxisScale(Enum):
        LINEAR = 1
        LOG10 = 2

    def __init__(
        self,
        label: str = None,
        unit: str = None,
        name: str = None,
        scale: AxisScale = AxisScale.LINEAR,
        min: float = None,
        max: float = None,
        grid: bool = False,
        label_visible=True,
        ticks_visible=True,
    ):
        """Axis object.

        Label and unit form together the axis label.
        To set the label directly use the name attribute.

        :param label: label part of axis label
        :param unit: unit part of axis label
        :param scale: Scale of the axis, i.e. "linear" or "log" axis.
        :param min: lower axis bound
        :param max: upper axis bound
        :param grid: show grid lines along the axis
        :param label_visible: show/hide the label text
        :param ticks_visible: show/hide axis ticks
        """
        super(Axis, self).__init__(sid=None, name=None)
        if label and name:
            ValueError("Either set label or name on Axis.")
        if unit is None:
            unit = "?"
        if not name:
            name = f"{label} [{unit}]"

        self.label = label
        self.name = name
        self.unit = unit
        self.scale = scale
        self.min = min
        self.max = max
        self.grid = grid
        self.label_visible = label_visible
        self.ticks_visible = ticks_visible

    def __copy__(self):
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
        )

    def __str__(self):
        return self.name

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale: AxisScale):
        if isinstance(scale, str):
            if scale == "linear":
                scale = self.AxisScale.LINEAR
            elif scale in {"log", "log10"}:
                scale = self.AxisScale.LOG10
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
    def __init__(
        self, sid: str, name: str, x: Data, order: int, style: Style, yaxis: Axis
    ):
        """
        :param sid:
        :param name: label of the curve
        :param xdata:
        :param order:
        :param style:
        :param yaxis:
        """
        super(AbstractCurve, self).__init__(sid, name)
        self.x = x
        self.order = order
        self.style = style
        self.yaxis = yaxis


class Curve(AbstractCurve):
    def __init__(
        self,
        x: Data,
        y: Data,
        xerr: Data = None,
        yerr: Data = None,
        single_lines: bool = False,
        dim_reductions: List[str] = None,
        order=None,
        style: Style = None,
        yaxis=None,
        **kwargs,
    ):
        super(Curve, self).__init__(None, None, x, order, style, yaxis)
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        self.single_lines = single_lines
        self.dim_reductions = dim_reductions

        if "label" in kwargs:
            self.name = kwargs["label"]

        # parse additional arguments and create style
        if style:
            logger.warning("'style' is set, 'kwargs' style arguments ignored")
        else:
            kwargs = Curve._add_default_style_kwargs(kwargs, y.dtype)
            style = Style.from_mpl_kwargs(**kwargs)
        self.style = style
        self.kwargs = kwargs  # store for lookup

    def __str__(self):
        info = f"x: {self.x}\ny: {self.y}\nxerr: {self.xerr}\nyerr: {self.yerr}"
        return info

    @staticmethod
    def _add_default_style_kwargs(d: Dict, dtype: str) -> Dict:
        """Default plotting styles"""

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
        """ Convert to dictionary. """
        d = {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid if self.x else None,
            "y": self.y.sid if self.y else None,
            "xerr": self.xerr.sid if self.xerr else None,
            "yerr": self.yerr.sid if self.yerr else None,
            "yaxis": self.yaxis,
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
        legend: bool = False,
        xaxis: Axis = None,
        yaxis: Axis = None,
        curves: List[Curve] = None,
        facecolor=Style.parse_color("white"),
        title_visible=True,
    ):
        """
        :param sid: Sid of the plot
        :param name: title of the plot
        :param legend: boolean flag to show or hide legend
        :param xaxis:
        :param yaxis:
        :param curves:
        """
        super(Plot, self).__init__(sid, name)
        if curves is None:
            curves = list()
        self.legend = legend
        self.facecolor = facecolor
        self.title_visible = title_visible

        if xaxis is not None:
            if not isinstance(xaxis, Axis):
                raise ValueError(f"'xaxis' must be of type Axis but: '{type(xaxis)}'")
        if yaxis is not None:
            if not isinstance(yaxis, Axis):
                raise ValueError(f"'yaxis' must be of type Axis but: '{type(yaxis)}'")

        self._xaxis = None  # type: Axis
        self._yaxis = None  # type: Axis
        self._curves = None
        self._figure = None  # type: Figure

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.curves = curves

    def __copy__(self):
        return Plot(
            sid=self.sid,
            name=self.name,
            xaxis=Axis.__copy__(self.xaxis),
            yaxis=Axis.__copy__(self.yaxis),
            curves=self.curves,
            facecolor=self.facecolor,
        )

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "sid": self.sid,
            "name": self.name,
            "xaxis": self.xaxis,
            "yaxis": self.yaxis,
            "legend": self.legend,
            "facecolor": self.facecolor,
            "curves": self.curves,
        }
        return d

    @property
    def figure(self) -> "Figure":
        if not self._figure:
            raise ValueError(f"The plot '{self}' has no associated figure.")

        return self._figure

    @figure.setter
    def figure(self, value: "Figure"):
        self._figure = value

    @property
    def experiment(self):
        return self.figure.experiment

    @property
    def title(self):
        return self.name

    @title.setter
    def title(self, value: str):
        self.set_title(title=value)

    def set_title(self, title: str):
        self.name = title

    @property
    def xaxis(self) -> Axis:
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value: Axis) -> None:
        self.set_xaxis(label=value)

    def set_xaxis(self, label: Union[str, Axis], unit: str = None, **kwargs):
        """Set axis with all axes attributes.

        All argument of Axis are supported.

        :param label:
        :param unit:
        :keyword label_visible:
        :param kwargs:
        :return:
        """
        if isinstance(label, Axis):
            ax = label
        else:
            ax = Axis(label=label, unit=unit, **kwargs)
        if ax.sid is None:
            ax.sid = f"{self.sid}_xaxis"
        self._xaxis = ax

    @property
    def yaxis(self) -> Axis:
        return self._yaxis

    @yaxis.setter
    def yaxis(self, value: Axis) -> None:
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
        if isinstance(label, Axis):
            ax = label
        else:
            ax = Axis(label=label, unit=unit, **kwargs)
        if ax.sid is None:
            ax.sid = f"{self.sid}_yaxis"
        self._yaxis = ax

    def add_curve(self, curve: Curve):
        """Curves are added via the helper function."""
        if curve.sid is None:
            curve.sid = f"{self.sid}_curve{len(self.curves)}"

        curve.order = len(self.curves)

        # inject default colors if no colors provided
        color = Style.parse_color(
            color=DEFAULT_COLORS[curve.order % len(DEFAULT_COLORS)],
            alpha=curve.kwargs.get("alpha", 1.0),
        )
        style = curve.style  # type: Style
        if (style.line.type != LineType.NONE) and (not style.line.color):
            style.line.color = color
            logger.warning(
                f"'{self.sid}.{curve.sid}': undefined line color set to: {color}"
            )
        if (style.marker.type != MarkerType.NONE) and (not style.marker.fill):
            style.marker.fill = color
            logger.error(
                f"'{self.sid}.{curve.sid}': undefined marker fill set to: {color}"
            )

        self.curves.append(curve)

    @property
    def curves(self) -> List[Curve]:
        return self._curves

    @curves.setter
    def curves(self, value: List[Curve]):
        self._curves = list()
        if value is not None:
            for curve in value:
                self.add_curve(curve)

    def curve(
        self,
        x: Data,
        y: Data,
        xerr: Data = None,
        yerr: Data = None,
        single_lines: bool = False,
        dim_reductions: List[str] = None,
        **kwargs,
    ):
        """Adds curve to the plot.

        Data can be high-dimensional data from a scan.
        Additional settings are required which allow to define how things
        are plotted.
        E.g. over which dimensions should an error be calculated and which
        dimensions should be plotted individually.
        """
        curve = Curve(x, y, xerr, yerr, single_lines=single_lines, **kwargs)
        self.add_curve(curve)

    def add_data(
        self,
        xid: str,
        yid: str,
        yid_sd=None,
        yid_se=None,
        count: Union[int, str] = None,
        dataset: str = None,
        task: str = None,
        label: str = "__yid__",
        single_lines=False,
        dim_reduction=None,
        **kwargs,
    ):
        """Wrapper around plotting."""
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
        if label == "__yid__":
            logger.warning(
                "No label provided on curve, using default label 'yid'. "
                "To not plot a label use 'label=None'"
            )

        # experiment to resolve data
        experiment = self.experiment

        # yerr data
        yerr = None
        yerr_label = ""
        if yid_sd and yid_se:
            logger.warning("'yid_sd' and 'yid_se' set, using 'yid_sd'.")
        if yid_sd:
            if yid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            yerr_label = "±SD"
            yerr = Data(experiment, yid_sd, dataset=dataset, task=task)
        elif yid_se:
            if yid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            yerr_label = "±SE"
            yerr = Data(experiment, yid_se, dataset=dataset, task=task)

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
                    count_data = Data(
                        experiment, index=count, dataset=dataset, task=task
                    )
                    counts = count_data.data
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

        # FIXME: here the data is not resolved yet, it is just the definition
        # Necessary to define how the scans should be plotted, i.e.
        # which curves should be generated
        self.curve(
            x=Data(experiment, xid, dataset=dataset, task=task),
            y=Data(experiment, yid, dataset=dataset, task=task),
            yerr=yerr,
            label=label,
            single_lines=single_lines,
            dim_reduction=dim_reduction,
            **kwargs,
        )


class SubPlot(BasePlotObject):
    """
    A SubPlot is a locate plot in a figure.
    """

    def __init__(
        self,
        plot: Plot,
        row: int = None,
        col: int = None,
        row_span: int = 1,
        col_span: int = 1,
    ):
        self.plot = plot
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span

    def __str__(self):
        return f"<Subplot[{self.row},{self.col}]>"


class Figure(BasePlotObject):
    """A figure consists of multiple subplots.

    A reference to the experiment is required, so the plot can
    resolve the datasets and the simulations.
    """

    fig_dpi = 72
    fig_facecolor = "white"
    fig_subplots_wspace = 0.15  # vertical spacing of subplots (fraction of axes)
    fig_subplots_hspace = 0.15  # horizontal spacing of subplots (fraction of axes)
    panel_width = 7
    panel_height = 5
    fig_titlesize = 25
    fig_titleweight = "bold"
    axes_titlesize = 20
    axes_titleweight = "bold"
    axes_labelsize = 18
    axes_labelweight = "bold"
    xtick_labelsize = 15
    ytick_labelsize = 15
    legend_fontsize = 13
    legend_loc = "best"
    _area_interpolation_points = 300

    def __init__(
        self,
        experiment,
        sid: str,
        name: str = None,
        subplots: List[SubPlot] = None,
        height: float = None,
        width: float = None,
        num_rows: int = 1,
        num_cols: int = 1,
    ):
        super(Figure, self).__init__(sid, name)
        self.experiment = experiment
        if subplots is None:
            subplots = list()
        self.subplots = subplots
        self.num_rows = num_rows
        self.num_cols = num_cols

        if width is None:
            width = num_cols * Figure.panel_width
        if height is None:
            height = num_rows * Figure.panel_height
        self.width = width
        self.height = height

    def num_subplots(self):
        """Number of existing subplots."""
        return len(self.subplots)

    def num_panels(self):
        """Number of available spots for plots."""
        return self.num_cols * self.num_rows

    def set_title(self, title):
        self.name = title

    def create_plots(
        self, xaxis: Axis = None, yaxis: Axis = None, legend: bool = True
    ) -> List[Plot]:
        """Template function for creating plots"""
        plots = []
        for k in range(self.num_panels()):
            # create independent axis objects
            xax = deepcopy(xaxis) if xaxis else None
            yax = deepcopy(yaxis) if yaxis else None
            # create plot
            p = Plot(sid=f"plot{k}", xaxis=xax, yaxis=yax, legend=legend)
            p.set_figure = self
            plots.append(p)
        self.add_plots(plots, copy_plots=False)
        return plots

    @property
    def plots(self):
        return self.get_plots()

    def get_plots(self) -> List[Plot]:
        """Returns list of plots."""
        return [subplot.plot for subplot in self.subplots]

    # FIXME
    def add_plots(self, plots: List[Plot], copy_plots: bool = False) -> None:
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
        for k, plot in enumerate(new_plots):
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
    def from_plots(sid, plots: List[Plot]) -> "Figure":
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
        """ Convert to dictionary. """
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
