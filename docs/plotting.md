# Plots and reports

Figures of a simulation experiment are described independent of the plotting backend: a `Figure` holds `Plot` panels, a plot has axes and `Curve` objects, and a curve references `Data` with a `Style`. The description is serialized with the experiment, and the format chooses what renders it: matplotlib draws the static images, plotly the interactive pages.

## Figures and plots

A `Figure` belongs to an experiment and has a grid of `num_rows` x `num_cols` panels. `create_plots` creates one `Plot` per panel with the given axes:

```python
from sbmlsim.plot import Axis, Figure

fig = Figure(experiment=None, sid="fig1", name="Repressilator", num_rows=1, num_cols=2)
plots = fig.create_plots(
    xaxis=Axis("time", unit="second"),
    yaxis=Axis("concentration", unit="dimensionless"),
    legend=True,
)
plots[0].set_title("timecourse")
plots[1].set_title("phase plane")
plots[1].set_xaxis("[X]", unit="dimensionless")
print(fig, len(plots))
```

An `Axis` has a label and a unit, which together form the axis label `name`, a `scale` (`linear` or `log`), `min`, `max`, `grid` and visibility flags. `name` follows both parts, i.e. `axis.unit = "week"` updates the label a figure renders; setting `name` overrides them and setting it to `None` hands the axis back to its label and its unit. The data plotted on an axis are converted to its unit, see [Units](units.md).

## Curves

A curve plots `Data` against `Data`, with optional error data, see [Data](data.md). `Plot.curve` adds a curve with matplotlib style keywords, `Plot.add_data` is the shortcut which creates the `Data` objects from a task or dataset:

```python
from sbmlsim.data import Data

plots[0].curve(
    x=Data("time", task="task_tc"),
    y=Data("[X]", task="task_tc"),
    label="X",
    color="tab:blue",
    linewidth=2.0,
)
plots[0].add_data(task="task_tc", xid="time", yid="[Y]", label="Y", color="tab:red")
plots[1].add_data(task="task_tc", xid="[X]", yid="[Y]", label="Y ~ X", color="black")
print([c.name for c in plots[0].curves])
```

Experimental data are added from a dataset with their errors and the count of the measurements:

```python
plots[0].add_data(
    dataset="dset1",
    xid="time",
    yid="mean",
    yid_sd="mean_sd",
    label="data",
    color="black",
)
```

`count` names the column with the number of measurements behind a mean, which is shown in the legend and used as weight in a fit.

The `CurveType` of a curve is `POINTS` (lines and markers), `BAR`, `BARSTACKED`, `HORIZONTALBAR` or `HORIZONTALBARSTACKED`; `ShadedArea` fills the area between two data curves. `examples/curve_types` shows all of them.

## Curves over the points of a scan

A curve of task data draws one line. When the data has dimensions of a scan, the curve names the ones it draws one line per point of, `plot.curve(x=Data("time", task="task_doses"), y=Data("mid", task="task_doses"), over="dose")`, also `plot.add_data(..., over="dose")`; every other dimension of the scan is selected with `Data(sel=...)`, and a dimension which is neither the axis, in `over` nor selected raises when the experiment is initialized. The points are the labels left after `sel`, in their order, so `Data(..., sel={"dose": [2, 1]})` draws the third and the second dose, each named and coloured by its own value. A second dimension, `over=("dose", "condition")`, sets the line style (solid, dashed, dotted, dash-dot, so at most four points left after `sel`): the first dimension gets the colour entries of the legend and the second grey entries with its line styles. A curve of a bar type with `over` raises when it is defined.

The lines take the colour map viridis, or shades of the colour of the curve, at their values: when the dimension changes one target, the colour of a point is the colour map at its value through the norm of the colour bar, logarithmic when all values are positive and the largest is more than 100 times the smallest, else linear, so the colours of unevenly spaced or decreasing doses read right on the bar. A dimension which changes no or several targets (several targets, simulations or models) is coloured by position along its labels. A legend entry reads `<curve name>, <target> = <value> <unit>`, e.g. `mid, PODOSE_mid = 5 mg`, or `<dimension> = <label>` when its dimension changes no or several targets.

From eleven points a colour bar of the dimension replaces the legend entries of the points: labelled `<target> [<unit>]` (`<target>` without a unit) with the values as ticks, or labelled with the dimension id and discrete, one cell per point with the labels as ticks, when the dimension changes no or several targets. A named curve with a colour bar keeps one legend entry of its name, a line in the colour of the middle of the colour map. Every colour bar of a panel gets its own place right of the panel, side by side in the order of the curves and right of the ticks and the title of a right y axis; the figure is wider by the room the bars of a column need, so the panels keep their size, and an outside legend starts right of the last bar. A panel with a right y axis has one legend of the curves of both axes, placed where it covers the data of neither. In the interactive page the one legend of the figure is right of every panel and colour bar, and entries which read and look the same, e.g. the simulation of every panel, are shown once and switch all of their curves.

A value per simulation is drawn over the values a dimension sets: `plot.curve(x=Data("dose.PODOSE_mid", task="task_doses"), y=Data("pk.cmax", task="task_doses"))`.

## Bands

`plot.band(x, y, across="draw", quantiles=(0.05, 0.95), median=True)` reduces the dimension `across` of y, e.g. the draws of a Latin hypercube design (see [Sampling and uncertainty](sampling.md)), to two quantiles, ignoring `NaN`, and draws the area between them with the median as a line and the quantile boundaries as thin lines in the band colour; `over=` (at most one dimension) gives one band per point of another dimension, in the colours of a curve over it. A band takes `color=`, `alpha=`, `sid=` and `order=` and may sit on the right y axis; a band without a colour takes the colour of the colour cycle at its position among the bands of its plot (`C0`, `C1`, ...), in both backends. The areas are drawn under the lines in both backends. The legend has one entry per band, its area in its colour labelled with the name and the quantiles in percent, `<name> (<low>-<high> %)`, e.g. `C (5-95 %)`: one per point with `over` (`<name>, <point label> (<low>-<high> %)`), or one of the name in the middle colour of the colour map for a band with a colour bar. The quantiles are computed when the figure is drawn, so no summary of the result is needed; a band needs a common grid of times, a ragged scan raises. `examples/experiment_scans.py` draws curves per dose, cmax over dose and a band over draws.

## Styles

A `Style` bundles the `Line` (type, color, thickness), the `Marker` (type, size, fill, line color) and the `Fill` of a curve. Matplotlib keywords such as `color`, `linestyle`, `linewidth`, `marker` and `alpha` are translated into a style, so a curve is styled either way:

```python
from sbmlsim.plot.plotting import ColorType, Line, LineType, Marker, MarkerType, Style

style = Style(
    line=Line(color=ColorType("tab:green"), type=LineType.DASH, thickness=1.5),
    marker=Marker(type=MarkerType.SQUARE, size=4, fill=ColorType("white")),
)
plots[0].curve(
    x=Data("time", task="task_tc"), y=Data("[Z]", task="task_tc"), style=style
)
print(style)
```

Colors are `ColorType` objects, created from matplotlib color names or hex strings and normalized to `#RRGGBBAA`.

## Rendering

A `Figure` says what is drawn and not how, so it is rendered by more than one backend. The format asks for one: `figure_formats=["svg", "png"]` are static images drawn by matplotlib, `figure_formats=["html"]` are interactive pages drawn by plotly, and a run asks for both at once. The default is `["svg"]`.

| | matplotlib | plotly |
| --- | --- | --- |
| formats | `svg`, `png`, `pdf`, … | `html` |
| per figure | 117 ms | 23 ms |
| for | the images of a publication | the pages a reader zooms, pans and hovers over |

The split follows the measurements. plotly is five times faster because it never rasterises: a page carries its data and the browser draws it, so writing it is 3 ms and the rest is resolving the data, which both backends do. The other direction does not hold - plotly writes PNG and SVG through a headless browser, which is 301 ms per figure at best and needs a Chrome on the machine, so the static images stay with matplotlib. Both read the same `Figure`, so the image and the page cannot disagree about what they show.

`MatplotlibFigureSerializer.to_figure` renders a single figure from a run experiment, `PlotlyFigureSerializer.to_figure` is its counterpart, and `SimulationExperiment.save_interactive_figures` writes the pages with the javascript of plotly next to them, so a report loads nothing from the network.

plotly is not a dependency of `sbmlsim`, it is in the `dev` extra: a run which asks for `html` without it reports that and writes no page. A figure of `figures_mpl()` is a matplotlib figure already and has no interactive version.

### Settings

The resolution, the panel size, the fonts and the legend of a figure are the fields of `FigureSettings`: `fig_dpi`, `fig_facecolor`, `fig_subplots_wspace`, `fig_subplots_hspace`, `panel_width`, `panel_height`, `fig_titlesize`, `fig_titleweight`, `axes_titlesize`, `axes_titleweight`, `axes_labelsize`, `axes_labelweight`, `xtick_labelsize`, `ytick_labelsize`, `legend_fontsize`, `legend_position` and `legend_loc`. They cascade, the more specific level wins:

- the defaults of `FigureSettings`,
- the run: `figure_settings=` of `SimulationExperiment.run`, `ExperimentRunner.run_experiments` and `run_experiments`, a `FigureSettings` or a mapping of overrides of the defaults, for every figure of the run, e.g. the resolution of a publication,
- the experiment: the class attribute `figure_settings` of a `SimulationExperiment`, a read-only mapping of overrides (`MappingProxyType`, which also keeps the `RUF012` rule of ruff quiet), empty by default. A subclass inherits it, so the base experiment of a model sets the settings of all its experiments; a subclass which assigns its own replaces it, `MappingProxyType({**Base.figure_settings, "fig_dpi": 300})` extends it,
- the figure: `Figure(..., settings={...})`, a mapping of overrides for one figure.

```python
from types import MappingProxyType

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.plot import Figure, FigureSettings


class ModelExperiment(SimulationExperiment):
    """The base of the experiments of a model."""

    figure_settings = MappingProxyType({"legend_fontsize": 10, "axes_labelsize": 16})


fig = Figure(experiment=None, sid="fig1", num_cols=2, settings={"axes_labelsize": 14})
settings = FigureSettings.resolve(
    run={"fig_dpi": 300, "legend_fontsize": 12},
    experiment=ModelExperiment.figure_settings,
    figure=fig.settings,
)
print(settings.fig_dpi, settings.legend_fontsize, settings.axes_labelsize)  # 300 10 14
print(fig.size(settings))  # (14.0, 5.0)
print(fig.size(FigureSettings(panel_width=4.0)))  # (8.0, 5.0)
fig.settings = {**fig.settings, "legend_position": "outside"}
print(dict(fig.settings))  # {'axes_labelsize': 14, 'legend_position': 'outside'}
# runner.run_experiments(output_path, figure_settings={"fig_dpi": 300})
```

A serializer resolves the settings of a figure the same way, `Figure.resolved_settings(experiment, run=...)`. An unknown key raises a `ValueError` which names the settings: the one of a run when the run starts, the one of an experiment at `initialize` and the one of a figure when the figure is created. A figure without an explicit `width` or `height` is `num_cols * panel_width` wide and `num_rows * panel_height` high in the settings it is rendered with. `figure.width` and `figure.height` are the explicit size and `None` without it, also in the JSON of an experiment, and `figure.size(figure.resolved_settings(experiment))` is the size in inches it is rendered with. `fig_dpi` is the resolution of a static image, which is cropped to its content (`bbox_inches="tight"`), so its pixels are those of the drawn content rather than of the whole figure. An interactive page has a fixed 72 pixels per inch (`PAGE_PIXELS_PER_INCH` of `sbmlsim.plot.serialization_plotly`), so it is `width * 72` by `height * 72` pixels whatever `fig_dpi` is: a run at 300 dpi for a publication has pages of the same size as a run at the default. A figure therefore looks the same whether its experiment runs alone or after others.

The settings were class attributes of `Figure` (`Figure.fig_dpi = 300`), which changed every later figure of the process, also those of other experiments. Setting or reading one on `Figure` or on a figure, or defining one in the body of a subclass of `Figure`, raises an `AttributeError` which says where it goes: `Figure.legend_fontsize = 10` at the top of the module of a model becomes `figure_settings = MappingProxyType({"legend_fontsize": 10})` of its base experiment, `Figure.fig_dpi = 300` before a run becomes `figure_settings={"fig_dpi": 300}` of the run, a setting on a figure which exists, `fig.legend_position = "outside"`, becomes `fig.settings = {**fig.settings, "legend_position": "outside"}` (or `Figure(..., settings={"legend_position": "outside"})` where it is created), and the read of a default becomes `FigureSettings().fig_titlesize`. A figure of `figures_mpl()` is styled with the settings of its experiment, `FigureSettings.resolve(experiment=self.figure_settings).fig_titlesize`; the settings of the run do not reach `figures_mpl()`.

## Reports

`ExperimentReport` collects `ExperimentResult` objects (or a stored `ReportResults`) and renders an HTML report with an index page and one page per experiment, listing the models, simulations, tasks, datasets, data and figures with the rendered images. The report is written next to the results, so the relative paths of the images resolve:

```python
from pathlib import Path

from sbmlsim.report.experiment_report import ExperimentReport

# results = runner.run_experiments(output_path=Path.cwd() / "results")
# ExperimentReport(results).create_report(output_path=Path.cwd() / "results")
```

`ReportResults.to_json` and `from_json` store the report data, so reports of experiments run at different times are combined. The report templates are jinja2 templates in `sbmlsim/resources/templates/`; `create_report(report_type=ExperimentReport.ReportType.MARKDOWN)` renders markdown instead of HTML.
