"""
Base classes for storing plotting information.

The general workflow of generating plotting information is the following.

1. Within simulation experiments abstract plotting information is stored.
    i.e., how from the data plots can be generated.


"""
from typing import List, Dict
import logging
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

from sbmlsim.result import XResult
from sbmlsim.data import DataSet, Data

from matplotlib.colors import to_rgba, to_hex

logger = logging.getLogger(__name__)



class Base(object):
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
        self.color = color


@dataclass
class Line(object):
    type: LineType
    color: ColorType
    thickness: float


@dataclass
class Marker(object):
    size: float
    type: MarkerType
    fill: ColorType
    line_color: ColorType
    line_thickness: float = 1.0


@dataclass
class Fill(object):
    color: ColorType
    second_color: ColorType = None


class Style(Base):
    def __init__(self, sid: str = None,
                 name: str = None,
                 base_style: 'Style' = None,
                 line: Line = None,
                 marker: Marker = None,
                 fill: Fill = None):
        super(Style, self).__init__(sid, name)
        self.line = line
        self.marker = marker
        self.fill = fill

    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
    MPL2SEDML_LINESTYLE_MAPPING = {
        '-': LineType.SOLID,
        'solid': LineType.SOLID,
        '.': LineType.DOT,
        'dotted': LineType.DOT,
        '--': LineType.DASH,
        'dashed': LineType.DASH.DASH,
        '-.': LineType.DASHDOT,
        'dashdot': LineType.DASHDOT,
        'dashdotdotted': LineType.DASHDOTDOT
    }
    SEDML2MPL_LINESTYLE_MAPPING = {v: k for (k, v) in MPL2SEDML_LINESTYLE_MAPPING.items()}

    MPL2SEDML_MARKER_MAPPING = {
        '': MarkerType.NONE,
        's': MarkerType.SQUARE,
        'o': MarkerType.CIRCLE,
        'D': MarkerType.DIAMOND,
        'x': MarkerType.XCROSS,
        '+': MarkerType.PLUS,
        '*': MarkerType.STAR,
        '^': MarkerType.TRIANGLEUP,
        'v': MarkerType.TRIANGLEDOWN,
        '<': MarkerType.TRIANGLELEFT,
        '>': MarkerType.TRIANGLERIGHT,
        '_': MarkerType.HDASH,
        '|': MarkerType.VDASH,
    }
    SEDML2MPL_MARKER_MAPPING = {v: k for (k,v) in MPL2SEDML_MARKER_MAPPING.items()}

    def to_mpl_kwargs(self) -> Dict:
        """Convert to matplotlib plotting arguments"""
        kwargs = {}
        if self.line:
            if self.line.color:
                kwargs["color"] = self.line.color
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
                kwargs["markerfacecolor"] = self.marker.fill
            if self.marker.line_color:
                kwargs["markeredgecolor"] = self.marker.line_color
            if self.marker.line_thickness:
                kwargs["markeredgewidth"] = self.marker.line_thickness

        if self.fill:
            pass

        return kwargs

    @staticmethod
    def from_mpl_kwargs(**kwargs) -> 'Style':

        # FIXME: handle alpha colors
        # https://matplotlib.org/3.1.0/tutorials/colors/colors.html
        alpha = kwargs.get("alpha", 1.0)
        color = kwargs.get("color", None)
        if color:
            color = to_rgba(color, alpha)
            color = to_hex(color, keep_alpha=True)

        # Line
        linestyle = kwargs.get("linestyle", '-')
        if linestyle is not None:
            linestyle = Style.MPL2SEDML_LINESTYLE_MAPPING[linestyle]

        line = Line(
            color=color,
            type=linestyle,
            thickness=kwargs.get("linewidth", 1.0)
        )

        # Marker
        marker_symbol = kwargs.get("marker", None)
        if marker_symbol is not None:
            marker_symbol = Style.MPL2SEDML_MARKER_MAPPING[marker_symbol]
        marker = Marker(
            size=kwargs.get("markersize", None),
            type=marker_symbol,
            fill=kwargs.get("markerfacecolor", color),
            line_color=kwargs.get("markeredgecolor", None),
            line_thickness=kwargs.get("markeredgewidth", None)
        )

        # Fill
        # FIXME: implement

        return Style(line=line, marker=marker, fill=None)


class Axis(Base):

    class AxisScale(Enum):
        LINEAR = 1
        LOG10 = 2

    def __init__(self, name: str = None, unit: str = None,
                 scale: AxisScale = AxisScale.LINEAR,
                 min: float = None, max: float = None, grid: bool = False):
        """ Axis object.

        :param name:
        :param unit:
        :param scale: Scale of the axis, i.e. "linear" or "log" axis.
        :param min: lower axis bound
        :param max: upper axis bound
        """
        super(Axis, self).__init__(None, name)
        if unit is None:
            unit = "?"
        self.unit = unit
        self.scale = scale
        self.min = min
        self.max = max
        self.grid = grid

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

    @property
    def label(self):
        return f"{self.name} [{self.unit}]"


class AbstractCurve(Base):
    def __init__(self, sid: str, name: str,
                 x: Data, order: int, style: Style, yaxis: Axis):
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
    def __init__(self,
                 x: Data, y: Data, xerr: Data=None, yerr: Data=None,
                 single_lines: bool=False,
                 order=None, style: Style=None, yaxis=None, **kwargs):
        super(Curve, self).__init__(None, None, x, order, style, yaxis)
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        self.single_lines = single_lines

        if "label" in kwargs:
            self.name = kwargs["label"]

        # parse additional arguments and create style
        self.style = Style.from_mpl_kwargs(**kwargs)

    def __str__(self):
        info = f"x: {self.x}\ny: {self.y}\nxerr: {self.xerr}\nyerr: {self.yerr}"
        return info

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "sid": self.sid,
            "name": self.name,
            "order": self.order,
            "style": self.style,
            "x": self.x.sid if self.x else None,
            "y": self.y.sid if self.y else None,
            "xerr": self.xerr.sid if self.xerr else None,
            "yerr": self.yerr.sid if self.yerr else None,
            "yaxis": self.yaxis
        }
        return d


class Plot(Base):
    """ Plot panel.
    A plot is the basic element of a plot. This corresponds to a single
    panel or axes combination in a plot. Multiple plots create a figure.
    """
    def __init__(self, sid: str, name: str = None,
                 legend: bool = False,
                 xaxis: Axis = None,
                 yaxis: Axis = None,
                 curves: List[Curve] = None):
        """
        :param sid:
        :param name: title of the plot
        :param legend:
        :param xaxis:
        :param yaxis:
        :param curves:
        """
        super(Plot, self).__init__(sid, name)
        if curves is None:
            curves = list()
        self.legend = legend
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.curves = curves
        self._figure = None  # type: Figure

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "sid": self.sid,
            "name": self.name,
            "legend": self.legend,
            "xaxis": self.xaxis,
            "yaxis": self.yaxis,
            "curves": self.curves,
        }
        return d

    def get_figure(self):
        if not self._figure:
            raise ValueError(f"The plot '{self}' has no associated figure.")

        return self._figure

    def set_figure(self, value: 'Figure'):
        self._figure = value

    figure = property(get_figure, set_figure)

    @property
    def experiment(self):
        return self.figure.experiment

    def get_title(self):
        """Get title of plot panel."""
        return self.name

    def set_title(self, name: str):
        """Set title of plot panel."""
        self.name = name

    def set_xaxis(self, label: str, unit: str=None, **kwargs):
        self.xaxis = Axis(name=label, unit=unit, **kwargs)

    def set_yaxis(self, label: str, unit: str=None, **kwargs):
        self.yaxis = Axis(name=label, unit=unit, **kwargs)

    def add_curve(self, curve: Curve):
        """
        Curves are added via the helper function
        """
        if curve.sid is None:
            curve.sid = f"{self.sid}_curve{len(self.curves)+1}"
        self.curves.append(curve)

    def _default_kwargs(self, d, dtype):
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
                d['marker'] = 's'

        if 'capsize' not in d:
            d['capsize'] = 3
        return d

    def curve(self, x: Data, y: Data, xerr: Data=None, yerr: Data=None,
              single_lines: bool = False, **kwargs):
        """Adds curves to the plot.

        Data can be high-dimensional data from a scan.
        Additional settings are required which allow to define how things
        are plotted.
        E.g. over which dimensions should an error be calculated and which
        dimensions should be plotted individually.
        """

        kwargs = self._default_kwargs(kwargs, x.dtype)
        curve = Curve(x, y, xerr, yerr, single_lines=single_lines, **kwargs)
        self.add_curve(curve)

    def add_data(self,
                 xid: str, yid: str, yid_sd=None, yid_se=None, count: int=None,
                 dataset: str=None, task: str=None,
                 label='__nolabel__',
                 xf=1.0, yf=1.0, single_lines=False,
                 **kwargs):
        """Wrapper around plotting.

        """
        if yid_sd and yid_se:
            raise ValueError("Set either 'yid_sd' or 'yid_se', not both.")
        if dataset is not None and task is not None:
            raise ValueError("Set either 'dataset' or 'task', not both.")
        if dataset is None and task is None:
            raise ValueError("Set either 'dataset' or 'task'.")

        for f in [xf, yf]:
            if hasattr(f, "magnitude"):
                f = f.magnitude
                if abs(f-1.0) > 1E-8:
                    # FIXME: fix scaling factors
                    raise ValueError("Scaling factors not supported yet !!!")

        # experiment to resolve data
        experiment = self.experiment

        # yerr data
        yerr = None
        yerr_label = ''
        if yid_sd:
            yerr_label = "±SD"
            yerr = Data(experiment, yid_sd, dataset=dataset, task=task)
        elif yid_se:
            yerr_label = "±SE"
            yerr = Data(experiment, yid_se, dataset=dataset, task=task)

        # label
        if label != "__nolabel__":
            count_label = ""
            if count:
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
            **kwargs
        )


class SubPlot(Base):
    """
    A SubPlot is a locate plot in a figure.
    """
    def __init__(self, plot: Plot,
                 row: None, col: None,
                 row_span: int = 1,
                 col_span: int = 1):
        self.plot = plot
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span


class Figure(Base):
    """A figure consists of multiple subplots.

    A reference to the experiment is required, so the plot can
    resolve the datasets and the simulations.
    """
    panel_width = 7.0
    panel_height = 7.0

    def __init__(self, experiment, sid: str, name: str = None,
                 subplots: List[SubPlot] = None,
                 height: float = None,
                 width: float = None,
                 num_rows: int = 1, num_cols: int = 1):
        super(Figure, self).__init__(sid, name)
        self.experiment = experiment
        if subplots is None:
            subplots = list()
        self.subplots = subplots
        self.num_rows = num_rows
        self.num_cols = num_cols

        if width == None:
            width = num_cols * Figure.panel_width
        if height == None:
            height = num_rows * Figure.panel_height
        self.width = width
        self.height = height

    def num_subplots(self):
        """Number of existing subplots."""
        return len(self.subplots)

    def num_panels(self):
        """Number of available spots for plots."""
        return self.num_cols * self.num_rows

    def create_plots(self, xaxis: Axis=None,
                     yaxis: Axis=None, legend: bool=True) -> List[Plot]:
        """Template function for creating plots"""
        plots = []
        for k in range(self.num_panels()):
            # create independent axis objects
            xax = deepcopy(xaxis) if xaxis else None
            yax = deepcopy(yaxis) if yaxis else None
            # create plot
            p = Plot(sid=f"plot{k}", xaxis=xax, yaxis=yax, legend=legend)
            plots.append(p)
        self.add_plots(plots)
        return plots

    def add_plots(self, plots: List[Plot]):
        if len(plots) > self.num_cols*self.num_rows:
            raise ValueError("Too many plots for figure")
        ridx = 1
        cidx = 1
        for k, plot in enumerate(plots):
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
            plot.set_figure(value=self)

    @staticmethod
    def from_plots(sid, plots: List[Plot]) -> 'Figure':
        """
        Create figure object from list of plots.
        """
        num_plots = len(plots)
        return Figure(sid=sid,
                      num_rows=num_plots, num_cols=1,
                      height=num_rows*5.0, width=num_cols*5.0,
                      subplots=[
                        SubPlot(plot, row=(k+1), col=1) for k, plot in enumerate(plots)
                      ])

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