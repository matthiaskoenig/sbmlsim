import logging
from typing import List
import pandas as pd
import numpy as np
import xarray as xr
import itertools

import plotly.tools as tls
import plotly.offline as py
# py.init_notebook_mode()

from sbmlsim.result import XResult
from sbmlsim.data import DataSet, Data
from sbmlsim.plot import Figure, SubPlot, Plot, Curve, Axis
from sbmlsim.plot.plotting_matplotlib import MatplotlibFigureSerializer

logger = logging.getLogger(__name__)

kwargs_data = {'marker': 's', 'linestyle': '--', 'linewidth': 1, 'capsize': 3}
kwargs_sim = {'marker': None, 'linestyle': '-', 'linewidth': 2}


class PlotlyFigureSerializer():
    """
    Serializing figure to matplotlib figure.
    """

    def to_figure(figure: Figure):
        """Convert sbmlsim.Figure to figure."""

        fig_mpl = MatplotlibFigureSerializer.to_figure(figure)

        fig_plotly = tls.mpl_to_plotly(fig_mpl)
        return fig_plotly



if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    # import plotly
    #
    # fig_mpl, ax = plt.subplots(ncols=1, nrows=1)
    #
    x = np.linspace(start=0, stop=10)
    y = np.sin(x)
    # ax.plot(x, y)
    # fig_plotly = tls.mpl_to_plotly(fig_mpl)
    # # fig_plotly = PlotlyFigureSerializer.to_figure(fig)
    # print(fig_plotly)
    # plotly.offline.plot(fig_plotly)
    #
    import plotly
    import plotly.graph_objs as go
    # plotly.offline.plot({
    #     "data": [go.Scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 4])],
    #     "layout": go.Layout(title="Chart1")
    # }, auto_open=True)

    import plotly.graph_objects as go

    # fig = go.Figure(
    #     data=[go.Bar(y=[2, 1, 3])],
    #     layout_title_text="A Figure Displayed with fig.show()"
    # )
    # fig.show()
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y*100)],
        layout_title_text="A Figure Displayed with fig.show()"
    )

    # pip install ipython, nbformat
    # fig.show(renderer="iframe")

    with open('p_graph.html', 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


