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

A curve of task data draws one line. When the data has dimensions of a scan, the curve names the ones it draws one line per point of, `plot.curve(x=Data("time", task="task_doses"), y=Data("mid", task="task_doses"), over="dose")`; every other dimension of the scan is selected with `Data(sel=...)`, and a dimension which is neither the axis, in `over` nor selected raises when the experiment is initialized. The lines take shades of the colour of the curve, or the colour map viridis without one, along the labels of the dimension. A second dimension, `over=("dose", "condition")`, sets the line style (solid, dashed, dotted, dash-dot, so at most four points left after `sel`): the first dimension gets the colour entries of the legend and the second grey entries with its line styles. A legend entry reads `<curve name>, <target> = <value> <unit>`, e.g. `mid, PODOSE_mid = 5 mg`, `<dimension> = <label>` when its dimension changes no or several targets. From eleven points a colour bar of the dimension, labelled `<target> [<unit>]` (`<target>` without a unit, the dimension id without a single target) and placed beside its panel, replaces the legend entries. A curve of a bar type with `over` raises. A value per simulation is drawn over the values a dimension sets: `plot.curve(x=Data("dose.PODOSE_mid", task="task_doses"), y=Data("pk.cmax", task="task_doses"))`.

## Bands

`plot.band(x, y, across="draw", quantiles=(0.05, 0.95), median=True)` reduces the dimension `across` of y, e.g. the draws of a Latin hypercube design (see [Sampling and uncertainty](sampling.md)), to two quantiles, ignoring `NaN`, and draws the area between them with the median as a line and the quantile boundaries as thin lines in the band colour; `over=` (at most one dimension) gives one band per point of another dimension. A band takes `color=` and `alpha=` and may sit on the right y axis. The legend has one entry per point, the median labelled `<name>, <point label>` (`<name> median` without `over`; the upper boundary with `median=False`), and one grey entry of the quantile range, e.g. `5-95 %`. The quantiles are computed when the figure is drawn, so no summary of the result is needed; a band needs a common grid of times, a ragged scan raises. `examples/experiment_scans.py` draws curves per dose, cmax over dose and a band over draws.

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

`MatplotlibFigureSerializer.to_figure` renders a single figure from a run experiment; `Figure.fig_dpi`, `Figure.axes_labelsize` and the other class attributes of `Figure` are the global matplotlib settings of the rendering. `PlotlyFigureSerializer.to_figure` is its counterpart, and `SimulationExperiment.save_interactive_figures` writes the pages with the javascript of plotly next to them, so a report loads nothing from the network.

plotly is not a dependency of `sbmlsim`, it is in the `dev` extra: a run which asks for `html` without it reports that and writes no page. A figure of `figures_mpl()` is a matplotlib figure already and has no interactive version.

## Reports

`ExperimentReport` collects `ExperimentResult` objects (or a stored `ReportResults`) and renders an HTML report with an index page and one page per experiment, listing the models, simulations, tasks, datasets, data and figures with the rendered images. The report is written next to the results, so the relative paths of the images resolve:

```python
from pathlib import Path

from sbmlsim.report.experiment_report import ExperimentReport

# results = runner.run_experiments(output_path=Path.cwd() / "results")
# ExperimentReport(results).create_report(output_path=Path.cwd() / "results")
```

`ReportResults.to_json` and `from_json` store the report data, so reports of experiments run at different times are combined. The report templates are jinja2 templates in `sbmlsim/resources/templates/`; `create_report(report_type=ExperimentReport.ReportType.MARKDOWN)` renders markdown instead of HTML.
