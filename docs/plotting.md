# Plots and reports

Figures of a simulation experiment are described independent of the plotting backend: a `Figure` holds `Plot` panels, a plot has axes and `Curve` objects, and a curve references `Data` with a `Style`. The description is serialized with the experiment and exported to SED-ML; matplotlib renders it.

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

An `Axis` has a label and a unit, which together form the axis label, a `scale` (`linear` or `log`), `min`, `max`, `grid` and visibility flags. The data plotted on an axis are converted to its unit, see [Units](units.md).

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

Colors are `ColorType` objects, created from matplotlib color names or hex strings, which are serialized to the `#RRGGBBAA` colors of SED-ML.

## Rendering

The `ExperimentRunner` renders the figures of every experiment with matplotlib and writes them to the output path in the `figure_formats` (`svg` by default). `MatplotlibFigureSerializer.to_figure` renders a single figure from a run experiment; `Figure.fig_dpi`, `Figure.axes_labelsize` and the other class attributes of `Figure` are the global matplotlib settings of the rendering.

## Reports

`ExperimentReport` collects `ExperimentResult` objects (or a stored `ReportResults`) and renders an HTML report with an index page and one page per experiment, listing the models, simulations, tasks, datasets, data and figures with the rendered images. The report is written next to the results, so the relative paths of the images resolve:

```python
from pathlib import Path

from sbmlsim.report.experiment_report import ExperimentReport

# results = runner.run_experiments(output_path=Path.cwd() / "results")
# ExperimentReport(results).create_report(output_path=Path.cwd() / "results")
```

`ReportResults.to_json` and `from_json` store the report data, so reports of experiments run at different times are combined. The report templates are jinja2 templates in `sbmlsim/resources/templates/`; `create_report(report_type=ExperimentReport.ReportType.MARKDOWN)` renders markdown instead of HTML.
