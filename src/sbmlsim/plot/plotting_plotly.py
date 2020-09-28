import logging

import plotly.tools as tls

from sbmlsim.plot import Axis, Curve, Figure, Plot, SubPlot
from sbmlsim.plot.plotting_matplotlib import MatplotlibFigureSerializer


logger = logging.getLogger(__name__)

kwargs_data = {"marker": "s", "linestyle": "--", "linewidth": 1, "capsize": 3}
kwargs_sim = {"marker": None, "linestyle": "-", "linewidth": 2}


class PlotlyFigureSerializer:
    """Serializing figure to matplotlib figure."""

    def to_figure(figure: Figure):
        """Convert sbmlsim.Figure to figure."""

        fig_mpl = MatplotlibFigureSerializer.to_figure(figure)
        fig_plotly = tls.mpl_to_plotly(fig_mpl)
        return fig_plotly
