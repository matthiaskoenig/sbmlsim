"""
Base classes for storing plotting information.
"""
from typing import List
import logging
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from sbmlsim.result import Result
from sbmlsim.data import DataSet

logger = logging.getLogger(__name__)


class Base(object):
    """

    """
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
    def __init__(self, sid: str, name: str,
                 base_style: 'Style' = None,
                 line: Line = None,
                 marker: Marker = None,
                 fill: Fill = None):
        super(Style, self).__init__(sid, name)
        self.line = line
        self.marker = marker
        self.fill = fill


class Axis(Base):
    def __init__(self, sid: str = None, name: str = None):
        super(Axis, self).__init__(sid, name)


class AbstractCurve(Base):
    def __init__(self, sid: str, name: str,
                 xdata, order, style, yaxis):
        super(AbstractCurve, self).__init__(sid, name)
        self.xdata = xdata
        self.order = order
        self.style = style
        self.yaxis = yaxis


class Curve(AbstractCurve):
    def __init__(self, sid: str, name: str,
                 xdata, ydata, xerr=None, yerr=None,
                 order=None, style: Style=None, yaxis=None, **kwargs):
        super(Curve, self).__init__(sid, name, xdata, order, style, yaxis)
        self.ydata = ydata
        self.xerr = xerr
        self.yerr = yerr


class Plot(Base):
    """
    A plot is the basic element of a plot.
    This corresponds to an axis.
    """
    def __init__(self, sid: str, name: str,
                 legend: bool = False,
                 xaxis: Axis = None,
                 yaxis: Axis = None,
                 curves: List[Curve] = None):
        super(Plot, self).__init__(sid, name)
        if curves is None:
            curves = list()
        self.legend = legend
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.curves = curves

    def add_data(self, data: DataSet,
                 xid: str, yid: str, yid_sd=None, yid_se=None, count=None,
                 xunit=None, yunit=None,
                 xf=1.0, yf=1.0,
                 label='__nolabel__', **kwargs):
        """ Add experimental data

        :param ax:
        :param data:
        :param xid:
        :param yid:
        :param xunit:
        :param yunit:
        :param label:
        :param kwargs:
        :return:
        """
        if isinstance(data, DataSet):
            dset = data
        elif isinstance(data, pd.DataFrame):
            dset = DataSet.from_df(data=data, udict=None, ureg=None)

        if dset.empty:
            logger.error(f"Empty dataset in adding data: {dset}")

        if abs(xf - 1.0) > 1E-8:
            logger.warning("xf attributes are deprecated, use units instead.")
        if abs(yf - 1.0) > 1E-8:
            logger.warning("yf attributes are deprecated, use units instead.")

        # add default styles
        if 'marker' not in kwargs:
            kwargs['marker'] = 's'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '--'

        # data with units
        x = dset[xid].values * dset.ureg(dset.udict[xid]) * xf
        y = dset[yid].values * dset.ureg(dset.udict[yid]) * yf
        y_err = None
        y_err_type = None
        if yid_sd:
            y_err = dset[yid_sd].values * dset.ureg(dset.udict[yid]) * yf
            y_err_type = "SD"
        elif yid_se:
            y_err = dset[yid_se].values * dset.ureg(dset.udict[yid]) * yf
            y_err_type = "SE"

        # convert
        if xunit:
            x = x.to(xunit)
        if yunit:
            y = y.to(yunit)
            if y_err is not None:
                y_err = y_err.to(yunit)

        # labels
        if label != "__nolabel__":
            if y_err_type:
                label = f"{label} ± {y_err_type}"
            if count:
                label += f" (n={count})"

        # plot
        if y_err is not None:
            if 'capsize' not in kwargs:
                kwargs['capsize'] = 3
            #ax.errorbar(x.magnitude, y.magnitude, y_err.magnitude, label=label,
            #            **kwargs)
            curve = Curve(sid=None, label=label,
                  xdata=x.magnitude, ydata=y.magnitude, yerr=y_err.magnitude, **kwargs)
        else:
            curve = Curve(sid=None, label=label,
                  xdata=x, ydata=y, **kwargs)
            # ax.plot(x, y, label=label, **kwargs)

        self.curves.append(curve)

    def add_line(self, data: Result,
                 xid: str, yid: str,
                 xunit=None, yunit=None, xf=1.0, yf=1.0, all_lines=False,
                 label='__nolabel__', **kwargs):
        """
        :param ax: axis to plot to
        :param data: Result data structure
        :param xid: id for xdata
        :param yid: id for ydata
        :param all_lines: plot all individual lines
        :param xunit: target unit for x (conversion is performed automatically)
        :param yunit: target unit for y (conversion is performed automatically)

        :param color:
        :return:
        """
        if not isinstance(data, Result):
            raise ValueError("Only Result objects supported in plotting.")
        if (hasattr(xf, "magnitude") and abs(xf.magnitude - 1.0) > 1E-8) or abs(
                xf - 1.0) > 1E-8:
            logger.warning("xf attributes are deprecated, use units instead.")
        if (hasattr(yf, "magnitude") and abs(yf.magnitude - 1.0) > 1E-8) or abs(
                yf - 1.0) > 1E-8:
            logger.warning("yf attributes are deprecated, use units instead.")

        # data with units
        x = data.mean[xid].values * data.ureg(data.udict[xid]) * xf
        y = data.mean[yid].values * data.ureg(data.udict[yid]) * yf
        y_sd = data.std[yid].values * data.ureg(data.udict[yid]) * yf
        y_min = data.min[yid].values * data.ureg(data.udict[yid]) * yf
        y_max = data.max[yid].values * data.ureg(data.udict[yid]) * yf

        # convert
        if xunit:
            x = x.to(xunit)
        if yunit:
            y = y.to(yunit)
            y_sd = y_sd.to(yunit)
            y_min = y_min.to(yunit)
            y_max = y_min.to(yunit)

        # FIXME: move to matplotlib backend
        # get next color
        # prop_cycler = ax._get_lines.prop_cycler
        # color = kwargs.get("color", next(prop_cycler)['color'])
        # kwargs["color"] = color

        if all_lines:
            for df in data.frames:
                xk = df[xid].values * data.ureg(data.udict[xid]) * xf
                yk = df[yid].values * data.ureg(data.udict[yid]) * yf
                xk = xk.to(xunit)
                yk = yk.to(yunit)
                # ax.plot(xk, yk, '-', label=label, **kwargs)
                if "linestyle" not in kwargs:
                    kwargs["linestyle"] = "-"
                curve = Curve(sid=None, name=label, xdata=xk, ydata=yk, **kwargs)
                self.curves.append(curve)
        else:
            if len(data) > 1:
                # FIXME: std areas should be within min/max areas!
                ax.fill_between(x, y - y_sd, y + y_sd, color=color, alpha=0.4,
                                label="__nolabel__")

                ax.fill_between(x, y + y_sd, y_max, color=color, alpha=0.2,
                                label="__nolabel__")
                ax.fill_between(x, y - y_sd, y_min, color=color, alpha=0.2,
                                label="__nolabel__")

            if "linestyle" not in kwargs:
                kwargs["linestyle"] = "-"
            curve = Curve(sid=None, name=label, xdata=x, ydata=y, **kwargs)
            self.curves.append(curve)
            # ax.plot(x, y, '-', label="{}".format(label), **kwargs)


class SubPlot(Base):
    """
    A SubPlot is a locate plot in a figure.
    """
    def __init__(self, plot: Plot,
                 row: int = 1, col: int = 1,
                 row_span: int = 1,
                 col_span: int = 1):
        self.plot = plot
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span


class Figure(Base):
    """A figure consists of multiple subplots."""
    def __init__(self, sid: str, name: str = None,
                 subplots: List[SubPlot] = None,
                 height: float = None,
                 width: float = None,
                 num_rows: int = 1, num_cols: int = 1):
        super(Figure, self).__init__(sid, name)
        if subplots is None:
            subplots = list()
        self.subplots = subplots
        self.height = height
        self.width = width
        self.num_rows = num_rows
        self.num_cols = num_cols
