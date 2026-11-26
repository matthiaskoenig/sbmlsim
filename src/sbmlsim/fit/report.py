"""Report of a parameter fit.

Reporting is separate from optimizing: a `FitReport` is created from the
definition of an `OptimizationProblem`, the `FitSettings` of the fit and one or
more `ParameterSets`. It does not need an optimization to have been run in the
same session, and several parameter sets can be compared in a single report,
e.g., the fitted parameters against the initial values of the model or the
results of two fits against each other.

The plots which describe an optimization run rather than a parameter set, i.e.,
the traces of the optimizers and the waterfall plot, are only created when the
`OptimizationResult` of the run is passed as well.

`FitReport.create` writes the report into a folder: the tables, the figures,
the text report and `index.html`, a page of two tabs. The tab "Report" is the
report document of the fit (`sbmlsim.fit.report_document`), i.e. its methods
with their formulas, its results and its references, followed by the overview,
the results and the identifiability of the fit; the tab "Fit mappings" is the
figures of the cost and weight and of the residuals of every fit mapping, the
table of the metrics of every fit mapping and a card per fit mapping. The same document is written as Typst (`report.typ`),
compiled to `report.pdf`, and as LaTeX (`report.tex`, a fragment, and
`report_standalone.tex`) with its references (`references.bib`, and
`references.csl`, the style of the Typst bibliography). A PDF which cannot be
compiled is stated in the HTML, the other outputs are written.

Every figure is a chart of the renderer of sbmlsim-core (`sbmlsim.plot.chart`),
saved as SVG or PNG with its spec (`sbmlsim.report.figures`), which `index.html`
shows inline; building a report imports no matplotlib.
"""

from __future__ import annotations

import datetime
import json
import logging
import textwrap
import webbrowser
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from tokenize import TokenError
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pint.errors import PintError

from sbmlsim import __version__
from sbmlsim.atomic import write_text_atomic
from sbmlsim.engine import import_core
from sbmlsim.fit import display
from sbmlsim.fit.derived import (
    HookSummary,
    ParameterGroup,
    group_parameters,
    hook_summaries,
)
from sbmlsim.fit.evaluation import (
    GROUP_METRICS_COLUMNS,
    Grouping,
    criteria_metrics,
    group_metrics,
    subset_agreement,
)
from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, plot_all
from sbmlsim.fit.methods import ReportContext
from sbmlsim.fit.metrics import (
    POINT_COLUMNS,
    FitMetrics,
    fold_counts,
    r_squared,
    rmse,
)
from sbmlsim.fit.objects import (
    EVALUATED_KINDS,
    GSD_ERROR,
    FitParameter,
    MappingKind,
    describe_array,
)
from sbmlsim.fit.optimization import ObservationKind, OptimizationProblem
from sbmlsim.fit.options import (
    RUN_IN_PYTHON,
    RUN_IN_THE_ENGINE,
    RUN_MIXED,
    FitSettings,
    ObjectiveType,
    ParameterScaleType,
)
from sbmlsim.fit.parameter_mapping import has_renamed_targets
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.posterior import POSTERIOR_FILE, ParameterPosterior
from sbmlsim.fit.report_document import FIGURES, fit_document
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.plot.chart import Chart, ChartAxis, ChartPlot
from sbmlsim.plot.colors import normalize_color
from sbmlsim.progress import activity as progress_activity
from sbmlsim.report.document import Document
from sbmlsim.report.figures import (
    FIGURE_SCRIPT,
    IMAGE_FORMATS,
    inline_figure,
    save_figure,
    write_figure_script,
)
from sbmlsim.report.references import write_bibliography
from sbmlsim.report.render import (
    HTMLRenderer,
    LaTeXRenderer,
    TypstRenderer,
    compile_pdf,
)
from sbmlsim.report.templates import template_environment
from sbmlsim.units import ureg
from sbmlsim.utils import paths_text

logger = logging.getLogger(__name__)

#: the rows of a figure of a grouping, more groups are drawn on more pages, see
#: `FitReport.plot_grouping`
MAX_GROUP_ROWS = 6

#: where the optimizer of a fit ran, for the overview of its report, by the
#: `OptimizationResult.optimizer_of` of its record; a record which does not
#: say it has no such text (`sbmlsim.fit.options.RUN_IN_THE_ENGINE`)
OPTIMIZER_TEXT: dict[str, str] = {
    RUN_IN_THE_ENGINE: "the optimizer in the engine",
    RUN_IN_PYTHON: "the optimizer in python",
    RUN_MIXED: "the optimizer in the engine and in python",
}


def grouping_page(plot: str, name: str, page: int) -> str:
    """Get the name of a page of a figure of a grouping.

    `goodness_of_fit_tissue` for the first page, `goodness_of_fit_tissue_2`
    for the second.
    """
    return f"{plot}_{name}" if page == 1 else f"{plot}_{name}_{page}"


#: colors of the kinds of fit mapping, the colors of the chips of the page
KIND_COLORS: dict[str, str] = {
    MappingKind.TRAINING.value: "#2f9e44",
    MappingKind.VALIDATION.value: "#1c7ed6",
    MappingKind.OUTLIER.value: "#e8590c",
}

#: color of the agreement of the points of a panel of a single parameter set,
#: apart from the black band of the training data, see `_draw_subset_agreement`
SUBSET_AGREEMENT_COLOR = "#7048e8"

#: the length of the id of a parameter set in a box of metrics, see `_box_id`
BOX_ID_LENGTH = 20

#: colors of the studies, i.e. of the simulation experiments of the problem, in
#: the order they appear in it. The data points of the goodness of fit and the
#: Bland-Altman plot carry them, so a study is the same color in every figure
STUDY_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)

#: the colormap the colors of the categories are sampled from where there are
#: more of them than `STUDY_COLORS` has colors, see `category_color`: `turbo`
#: varies the hue and the lightness, so two neighbours of a large sample are
#: told apart (#511)
CATEGORY_COLORMAP: str = "turbo"

#: the coefficients of the polynomials of the red, green and blue of `turbo`,
#: from the constant term on (Mikhailov 2019, the polynomial approximation of
#: the colormap of Google), see `_turbo`
_TURBO: tuple[tuple[float, ...], ...] = (
    (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
    (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
    (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
)

#: markers of the parameter sets in the figures which color by study, so that
#: the sets are told apart where the color is taken, the markers of the
#: renderer
SET_MARKERS: tuple[str, ...] = (
    "circle",
    "square",
    "triangle",
    "diamond",
    "cross",
    "plus",
)


def _turbo(position: float) -> str:
    """Get the color of `turbo` at a position of `[0, 1]` as `#rrggbb`."""
    x = min(max(float(position), 0.0), 1.0)
    channels = [
        min(max(sum(c * x**n for n, c in enumerate(coefficients)), 0.0), 1.0)
        for coefficients in _TURBO
    ]
    return "#" + "".join(f"{round(255 * c):02x}" for c in channels)


def category_color(index: int, total: int) -> str:
    """Get the color of one of the categories of a figure, e.g. of a study.

    The categorical colors of `STUDY_COLORS` while they are enough, so a
    report of few studies keeps its colors, and otherwise `total` colors
    sampled from `CATEGORY_COLORMAP`: the colors repeated every
    `len(STUDY_COLORS)` categories, so the first, the tenth and the
    nineteenth study of a fit of 35 studies shared a color and could not be
    told apart in a panel (#511).

    Args:
        index: the position of the category, `0 <= index < total`.
        total: the number of categories of the figure.

    Returns:
        The color, a name of `STUDY_COLORS` or a hex string, see
        `sbmlsim.plot.colors.normalize_color`.
    """
    if total <= len(STUDY_COLORS):
        return STUDY_COLORS[index % len(STUDY_COLORS)]
    position = min(max(index, 0), total - 1) / (total - 1)
    return _turbo(position)


#: styles of the agreement band. The goodness of fit and the Bland-Altman plot
#: draw the same band, once as lines parallel to the identity and once as
#: horizontal lines, so the two figures are read the same way
#: (the line and its width in pixels of a curve or a rule of the renderer)
IDENTITY_STYLE: dict[str, Any] = {"line": "solid", "width": 1.5}
BIAS_STYLE: dict[str, Any] = {"line": "dashdot", "width": 1.2}
LIMITS_STYLE: dict[str, Any] = {"line": "dashed", "width": 1.2}

#: opacity of the filled area between the limits of agreement
BAND_ALPHA: float = 0.12

#: style of the lines of twofold of the goodness of fit, `f = 2y` and
#: `f = y/2`, the criterion of `within_2fold`, in a color of their own, apart
#: from the black limits of agreement and the gray band, which they lie close
#: to for a fit of a typical scatter
TWOFOLD_STYLE: dict[str, Any] = {
    "line": "dotted",
    "width": 2.0,
    "color": "#8b0000",
}

#: the header of a column of the metrics whose name is not what a table shows
METRIC_HEADERS: dict[str, str] = {
    "within_2fold": "2-fold [%]",
    "within_1.25fold": "1.25-fold [%]",
    "within_1.5fold": "1.5-fold [%]",
    "within_1sd": "1 SD [%]",
    "within_2sd": "2 SD [%]",
}

#: the figures which cover a full row of the report, i.e. the goodness of fit
#: and the Bland-Altman plot with their panel per kind and their metrics
WIDE_PLOTS: frozenset[str] = frozenset({"goodness_of_fit", "bland_altman"})

#: resolution of the PNG copies of the figures of the report document, which
#: LaTeX includes, see `FitReport.create`
PNG_DPI: float = 300.0

#: the files of the report document in a report folder, as they are linked
DOCUMENT_FILES: tuple[tuple[str, str], ...] = (
    ("report.pdf", "PDF"),
    ("report.typ", "Typst"),
    ("report.tex", "LaTeX"),
    ("report_standalone.tex", "LaTeX document"),
    ("references.bib", "BibTeX"),
)

#: height in pixels of a row of the figures with a row per curve, i.e. of the
#: costs and of the residuals of the fit mappings
ROW_HEIGHT: float = 12.24
#: the font size of the labels of the rows of these figures
ROW_FONT: float = 8.0
#: width in pixels of a character of the label of a row, of `ROW_FONT` points
ROW_LABEL_CHAR: float = 4.7

#: width and height in pixels of a panel of the goodness of fit and the
#: Bland-Altman plot, whose legend is right of the panels
PANEL_SIZE: float = 396.0

#: width and height in pixels of the four panels of the figure of a fit
#: mapping, whose legends are right of its rows
MAPPING_SIZE: tuple[float, float] = (720.0, 700.0)

#: size in pixels and opacity of a data point of the agreement figures
POINT_SIZE: float = 9.0
POINT_OPACITY: float = 0.7

#: the box with the key metrics of a panel, see `FitReport._metrics_boxes`:
#: its edge, fill and opacity, the font size of a text element of the
#: renderer (the size of the x tick labels, 11 pixels by default) and the
#: distance of the lines of a text of several lines, 1.2 font sizes, its
#: padding and its distance from the corner of the panel in pixels
METRICS_BOX: dict[str, Any] = {
    "edge": "#999999",
    "fill": "#ffffff",
    "fill_opacity": 0.9,
}
TEXT_SIZE: float = 11.0
BOX_LINE: float = 1.2 * TEXT_SIZE
BOX_PAD: float = 5.0
BOX_MARGIN: float = 8.0

#: the font size of the title of a figure in pixels, the one of the renderer,
#: which draws it bold; the width of the figure of a fit mapping makes room
#: for it, see `_title_width`
TITLE_SIZE: float = 16.0

#: the labels of the axes of the agreement figures
GOF_X_LABEL: str = "Experiment y"
GOF_Y_LABEL: str = "Prediction f(x)"
ALTMAN_X_LABEL: str = "Geometric mean √(f(x) · y)"
ALTMAN_Y_LABEL: str = "log10(f(x) / y)"

#: colors of the parameter sets, in the order of the sets. Black is the color
#: of the reference data of a fit mapping, so no parameter set uses it
SET_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
)


def _hex(color: str) -> str:
    """Get a color of the report as `#rrggbb`, see `normalize_color`."""
    normalized = normalize_color(color)
    if normalized is None:
        raise ValueError(f"The color '{color}' of a figure is none.")
    return normalized


def _percent(value: Any) -> str:
    """Format a percentage of the metrics, `-` for one which is not defined."""
    return _cell(value, ".1f")


def _box_value(value: float, spec: str, unit: str = "") -> str:
    """Format a value of a box of metrics, `-` for a value which is not defined."""
    return f"{value:{spec}}{unit}" if np.isfinite(value) else "-"


def _metric_cell(column: str, value: Any) -> str:
    """Format a value of a table of metrics, a percentage of a criterion as one."""
    if isinstance(value, str):
        return value
    if column.startswith("within_"):
        return _percent(value)
    return _cell(value, ".4g")


def _cell(value: Any, spec: str) -> str:
    """Format a value of a table, `-` for a value which is not defined.

    Args:
        value: the value, a text is kept as it is.
        spec: format specification of a number, e.g. `.5g`.

    Returns:
        The text of the cell; `None` and `nan`, e.g. an error without degrees
        of freedom, are `-`.
    """
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:{spec}}"


def _panel_point(panel: Mapping[str, Any], dx: float, dy: float) -> tuple[float, float]:
    """Get the values of the axes of a plot at pixels from its upper left corner.

    Args:
        panel: the plot of `figure_geometry`, its `rect` and its scales.
        dx: the pixels to the right of the left edge of its panel.
        dy: the pixels below the top edge of its panel.

    Returns:
        The values of the x and of the y axis there.
    """
    rect = panel["rect"]
    return (
        _scale_value(panel["x"], dx / rect["width"]),
        _scale_value(panel["y"], 1.0 - dy / rect["height"]),
    )


def _scale_value(scale: Mapping[str, Any], fraction: float) -> float:
    """Get the value at a fraction of an axis of the geometry of a chart.

    Args:
        scale: the scale of the axis of `figure_geometry`: its `kind`, `low`,
            `high` and `reverse`.
        fraction: the fraction from the lower end of the axis, its left or
            its bottom.

    Returns:
        The value of the axis there, on a logarithmic axis in its decades.
    """
    if scale.get("reverse"):
        fraction = 1.0 - fraction
    low, high = float(scale["low"]), float(scale["high"])
    if scale["kind"] == "log":
        return float(
            10 ** (np.log10(low) + fraction * (np.log10(high) - np.log10(low)))
        )
    return low + fraction * (high - low)


def evaluate(problem: OptimizationProblem, pset: ParameterSet) -> dict[str, list[Any]]:
    """Evaluate a set in the units of the problem, its arrays are read only.

    The complete data of the set (`OptimizationProblem.residuals` with
    `complete_data`), which `FitReport` caches per set of the report.

    Args:
        problem: the initialized problem.
        pset: the set, bound to the problem (`bind_parameter_set`).

    Returns:
        The residual data of the mappings.
    """
    evaluation: dict[str, list[Any]] = problem.residuals(  # ty: ignore[invalid-assignment]
        xlog=problem.to_scale(pset.x(problem.pids)), complete_data=True
    )
    for values in evaluation.values():
        for value in values:
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return evaluation


def evaluate_metrics(problem: OptimizationProblem, pset: ParameterSet) -> FitMetrics:
    """Get the metrics of a parameter set on a problem, evaluated now.

    What `FitReport.metrics` gives for a set which is not one of the report,
    without a report and its caches: the problem is evaluated as it is, e.g.
    with the integrator settings of `OptimizationProblem.integrating`, or in
    a worker of a pool which holds the problem (`sbmlsim.fit.pool`).

    Args:
        problem: the initialized problem.
        pset: the parameter set.

    Returns:
        The metrics of the set, see `sbmlsim.fit.metrics`.
    """
    return FitMetrics(
        problem=problem,
        parameter_set=deepcopy(pset),
        evaluation=evaluate(problem, problem.bind_parameter_set(pset)),
    )


class FitReport:
    """Report of a fit for one or more parameter sets.

    Creates the figures, the text report and the HTML report of a fit.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        settings: FitSettings,
        parameter_sets: ParameterSets | list[ParameterSet] | ParameterSet,
        opt_result: OptimizationResult | None = None,
        identifiability: IdentifiabilityResult | None = None,
        fisher: FisherInformation | None = None,
        posterior: ParameterPosterior | None = None,
        show_titles: bool = True,
        image_format: str = "svg",
        mapping_figures: bool = True,
        context: ReportContext | None = None,
        groupings: Sequence[Grouping] = (),
        error_bars: bool = False,
        color_by: str | None = None,
    ) -> None:
        """Construct the report.

        A copy of the problem definition is initialized with the settings. Its
        runtime and parameter sets belong to the report, so later fits or edits
        to the original inputs cannot change the report.

        Args:
            problem: definition of the optimization problem.
            settings: settings the parameter sets were fitted with.
            parameter_sets: one or more sets of parameters to report. The first
                set is the reference the others are compared against.
            opt_result: result of an optimization, adds the traces, the
                waterfall plot and the table of the runs.
            identifiability: result of a profile likelihood analysis, adds
                the identifiability section with the profiles.
            fisher: Fisher information of the parameters, adds its table of
                errors and intervals and the correlation of the parameters to
                the identifiability section.
            posterior: the Laplace posterior of the parameters, adds the
                section of the posterior with its credible intervals and the
                correlation of the parameters.
            show_titles: add titles to the figures: the title of a fit
                mapping above the panels of its figures, the title of a
                figure or of its panel for the other figures.
            image_format: format of the figures, `svg` or `png`, see
                `sbmlsim.report.figures.IMAGE_FORMATS`; every figure is
                saved with its spec as `json` beside it.
            mapping_figures: draw the figure of every fit mapping, i.e. the
                data with the simulation and the goodness of fit and the
                Bland-Altman plot of the mapping, see `plot_mapping`, one
                figure per mapping. A report which is only read for its
                tables and its overview figures is created without them, and
                its mapping cards carry their metrics alone.
            context: what the report document says about the fit beyond the
                fit, e.g. its title and its description; sbmlsim reports a fit
                with an empty context.
            groupings: the groupings the data points are evaluated by, e.g.
                `Grouping.by("tissue")`: per grouping a goodness of fit and a
                Bland-Altman plot with a row per group and a column per kind,
                the criteria per group and their metrics, see
                `plot_grouping`. A report without groupings has none of them.
            error_bars: draw the standard deviation of the data points in the
                goodness of fit and the Bland-Altman plots, see `_draw_points`.
            color_by: a column of the data points which colors the points of
                the goodness of fit and the Bland-Altman plots, e.g. `tissue`,
                the study by default.
        """
        if image_format not in IMAGE_FORMATS:
            raise ValueError(
                f"The figures of a report are {' or '.join(IMAGE_FORMATS)}, "
                f"not '{image_format}'."
            )
        # Copy the definition, which excludes native models, then build a runtime
        # owned by this report. Later fits and reports cannot change its settings.
        self.problem = deepcopy(problem)
        self.settings = settings
        self.parameter_sets = deepcopy(ParameterSets.of(parameter_sets))
        self.opt_result = deepcopy(opt_result)
        self.identifiability = deepcopy(identifiability)
        self.fisher = deepcopy(fisher)
        self.posterior = deepcopy(posterior)
        self.show_titles = show_titles
        self.image_format = image_format
        self.mapping_figures = mapping_figures
        self.context = context if context is not None else ReportContext()
        self.groupings: tuple[Grouping, ...] = tuple(groupings)
        self.error_bars = error_bars
        self.color_by = color_by
        # resolution of the PNG copies of the figures, see `_create_figures`
        self._png_dpi: float = PNG_DPI

        # Resolve data in the runtime owned by this report.
        self.problem.initialize(settings)
        self.parameter_sets = ParameterSets(
            [self.problem.bind_parameter_set(pset) for pset in self.parameter_sets]
        )

        # residual data of the mappings, by parameter set
        self._res_data: dict[tuple[Any, ...], dict[str, list[Any]]] = {}

        # data points with their kind and predictions, by parameter set
        self._points: dict[tuple[Any, ...], pd.DataFrame] = {}
        self._check_columns()

    def _check_columns(self) -> None:
        """Check the columns `color_by` and the groupings name, and their names.

        The columns of the data points are known when the problem is
        initialized: the columns of `POINT_COLUMNS` and the fields of the
        metadata of the mappings. A typo fails here, not when the report is
        written, e.g. after a fit.

        Raises:
            ValueError: for a column which the data points do not have or two
                groupings of one name.
        """
        columns = [
            *POINT_COLUMNS,
            *dict.fromkeys(
                key for metadata in self.problem.mapping_metadata for key in metadata
            ),
        ]
        if self.color_by is not None and self.color_by not in columns:
            raise ValueError(
                f"color_by '{self.color_by}' is no column of the data points: "
                f"{columns}."
            )
        for grouping in self.groupings:
            missing = [c for c in grouping.columns if c not in columns]
            if missing:
                raise ValueError(
                    f"The grouping '{grouping.name}' groups by {missing}, which "
                    f"are no columns of the data points: {columns}."
                )
        names = [grouping.name for grouping in self.groupings]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"The groupings {twice} are given twice, their figures and tables "
                f"would overwrite each other; give them other names."
            )

    @staticmethod
    def from_optimization_result(
        problem: OptimizationProblem,
        opt_result: OptimizationResult,
        size: int = 1,
        with_model: bool = False,
        **kwargs: Any,
    ) -> FitReport:
        """Create the report of an optimization.

        Args:
            problem: definition of the optimization problem.
            opt_result: result of the optimization, it carries the settings.
            size: number of fitted parameter sets to report, the best first.
            with_model: report the initial values of the model as the reference
                set as well, so that the figures and the tables compare the fit
                against the model it started from. The report shows the fitted
                parameters alone by default.
            kwargs: additional arguments of `FitReport`.

        Returns:
            The report of the fit.
        """
        settings = opt_result.settings_stored
        report = FitReport(
            problem=problem,
            settings=settings,
            parameter_sets=opt_result.parameter_sets(size=size),
            opt_result=opt_result,
            **kwargs,
        )
        if with_model:
            report.parameter_sets = ParameterSets(
                [report.problem.parameter_set_model(), *report.parameter_sets]
            )
        return report

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{self.problem.opid}: "
            f"{[pset.sid for pset in self.parameter_sets]}>"
        )

    @property
    def reference_set(self) -> ParameterSet:
        """First parameter set, the reference the others are compared against."""
        return self.parameter_sets[0]

    @property
    def display_units(self) -> dict[str, str]:
        """Get the unit every table of the report shows for a parameter, by id.

        It is the unit as the user defined it (`FitParameter.unit`), e.g. the
        unit id `per_min` of the model, and `model` for the unit of the model.
        The values carry the units of pint, e.g. `1/min`, see
        `OptimizationProblem.parameter_units`, and so does the code
        `Q(value, unit)` of the optimal parameters of the text report, which
        runs with `sbmlsim.Q`.
        """
        return {p.pid: p.unit or "model" for p in self.problem.parameters}

    def mapping_title(self, k: int) -> str:
        """Get the title of the plots of a fit mapping, with its kind.

        The kind of every mapping is stated, the training data too, so that it
        is visible in the figures which curves the parameters were fitted on.
        """
        title = f"{self.problem.experiment_keys[k]} {self.problem.mapping_keys[k]}"
        return f"{title} [{self.problem.mapping_kinds[k].value}]"

    def color(self, pset: ParameterSet) -> str:
        """Get the color of a parameter set as `#rrggbb`."""
        index = [p.sid for p in self.parameter_sets].index(pset.sid)
        return _hex(SET_COLORS[index % len(SET_COLORS)])

    def studies(self) -> list[str]:
        """Get the studies of the problem, i.e. its simulation experiments.

        In the order the fit mappings of the problem name them, so the color of
        a study does not depend on which mappings a figure shows.
        """
        return list(dict.fromkeys(self.problem.experiment_keys))

    def study_color(self, study: str) -> str:
        """Get the color of a study, the same in every figure of the report."""
        studies = self.studies()
        return _hex(category_color(studies.index(study), len(studies)))

    def set_marker(self, pset: ParameterSet) -> str:
        """Get the marker of a parameter set in the figures colored by study."""
        index = [p.sid for p in self.parameter_sets].index(pset.sid)
        return SET_MARKERS[index % len(SET_MARKERS)]

    def _point_label(self, study: Any, pset: ParameterSet) -> str:
        """Get the legend entry of the points of a study and a parameter set."""
        if len(self.parameter_sets) == 1:
            return str(study)
        return f"{study} ({pset.sid})"

    def x(self, pset: ParameterSet) -> np.ndarray:
        """Get the values of a set in the parameter order of the problem."""
        return self._bound(pset).x(self.problem.pids)

    def _stored(self, pset: ParameterSet) -> ParameterSet | None:
        """Get the set of the report a parameter set stands for.

        A set stands for a set of the report when it is that set, or when it
        has its id, its values and its units in the units of the problem,
        e.g. the set the report was created with, which the report keeps in
        the units of the problem. Any other set, also one with the id of a
        set of the report, e.g. `model` or `fit`, is evaluated as it is,
        without the cache of the report.

        Args:
            pset: parameter set.

        Returns:
            The set of the report, `None` for another set.
        """
        for stored in self.parameter_sets:
            if stored is pset:
                return stored
        bound = self.problem.bind_parameter_set(pset)
        for stored in self.parameter_sets:
            if (stored.sid, stored.values, stored.units) == (
                bound.sid,
                bound.values,
                bound.units,
            ):
                return stored
        return None

    def _bound(self, pset: ParameterSet) -> ParameterSet:
        """Get a parameter set in the units of the problem."""
        stored = self._stored(pset)
        return stored if stored is not None else self.problem.bind_parameter_set(pset)

    def _evaluation_key(self, stored: ParameterSet) -> tuple[Any, ...]:
        """Identify the values of a set of the report, which may be edited.

        The caches keep the current evaluation of every set of the report:
        the evaluation of an earlier value of the set is dropped.
        """
        key = (stored.sid, *stored.x(self.problem.pids))
        sids = {s.sid for s in self.parameter_sets}
        for cache in (self._res_data, self._points):
            for previous in list(cache):
                if previous[0] not in sids or (
                    previous[0] == stored.sid and previous != key
                ):
                    del cache[previous]
        return key

    def metrics(self, pset: ParameterSet) -> FitMetrics:
        """Get the metrics of a parameter set on the problem.

        Args:
            pset: parameter set, a set of the report is evaluated once, see
                `residual_data`.

        Returns:
            The metrics of the set, see `sbmlsim.fit.metrics`.
        """
        stored = self._stored(pset)
        return FitMetrics(
            problem=self.problem,
            parameter_set=deepcopy(stored if stored is not None else pset),
            evaluation=self._residual_evaluation(pset),
        )

    def metrics_df(self) -> pd.DataFrame:
        """Get the metrics of every parameter set, per kind of fit mapping.

        A fit is evaluated on its training and on its validation data, so every
        parameter set has a row per kind.
        """
        return pd.concat(
            [self.metrics(pset).summary_df() for pset in self.parameter_sets],
            ignore_index=True,
        )

    def metrics_mappings_df(self) -> pd.DataFrame:
        """Get the metrics of every fit mapping and parameter set."""
        frames = []
        for pset in self.parameter_sets:
            df = self.metrics(pset).mappings_df()
            df.insert(0, "parameter_set", pset.sid)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def metrics_observables_df(self) -> pd.DataFrame:
        """Get the metrics of every observable, kind and parameter set."""
        frames = []
        for pset in self.parameter_sets:
            df = self.metrics(pset).observables_df()
            df.insert(0, "parameter_set", pset.sid)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def _fit_mapping(self, experiment: str, mapping: str) -> Any:
        """Get the fit mapping of an experiment and a mapping key."""
        runner = self.problem.runner
        if runner is None:
            raise ValueError(f"The problem '{self.problem.opid}' is not initialized.")
        return runner.experiments[experiment]._fit_mappings[mapping]

    def metrics_groups_df(self) -> pd.DataFrame:
        """Get the metrics of every grouping, group, kind and parameter set."""
        frames = []
        for pset in self.parameter_sets:
            for grouping in self.groupings:
                df = group_metrics(self.points(pset), grouping, self._fit_mapping)
                df.insert(0, "parameter_set", pset.sid)
                frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=pd.Index(["parameter_set", *GROUP_METRICS_COLUMNS])
            )
        return pd.concat(frames, ignore_index=True)

    def datapoints_df(self) -> pd.DataFrame:
        """Get the data points with their predictions for every parameter set."""
        frames = []
        for pset in self.parameter_sets:
            df = self.metrics(pset).datapoints_df()
            df.insert(0, "parameter_set", pset.sid)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def residual_data(self, pset: ParameterSet) -> dict[str, list[Any]]:
        """Get the complete residual data of the mappings for a parameter set.

        Every evaluation simulates all fit mappings, so the results of a set
        of the report are cached for the plots and tables which use it. Any
        other set is evaluated as it is, see `_stored`.
        """
        return deepcopy(self._residual_evaluation(pset))

    def _residual_evaluation(self, pset: ParameterSet) -> dict[str, list[Any]]:
        """Share one internal evaluation; public readers receive copies."""
        stored = self._stored(pset)
        if stored is None:
            return self._evaluate(self.problem.bind_parameter_set(pset))
        key = self._evaluation_key(stored)
        if key not in self._res_data:
            self._res_data[key] = self._evaluate(stored)
        return self._res_data[key]

    def _evaluate(self, pset: ParameterSet) -> dict[str, list[Any]]:
        """Evaluate a set in the units of the problem, see `evaluate`."""
        return evaluate(self.problem, pset)

    def points(self, pset: ParameterSet) -> pd.DataFrame:
        """Get the data points of a parameter set with their kind.

        The table of `FitMetrics.datapoints_df`, i.e. one row per data point
        with the measurement `DV`, the prediction `IPRED` and the `kind` of the
        fit mapping it belongs to. It is the source of the goodness of fit and
        the Bland-Altman plots and is cached for a set of the report, a data
        point costs a simulation; any other set is evaluated as it is, see
        `_stored`.
        """
        stored = self._stored(pset)
        if stored is None:
            return self.metrics(pset).datapoints_df()
        key = self._evaluation_key(stored)
        if key not in self._points:
            self._points[key] = self.metrics(stored).datapoints_df()
        return self._points[key].copy(deep=True)

    def point_kinds(self) -> list[str]:
        """Get the kinds of fit mapping the data points are shown in.

        The kinds the problem has, in the order of `EVALUATED_KINDS`, i.e. the
        training data, the validation data and the outliers. There is no panel
        over all data points: it pools data a fit was fitted on with data it
        dropped, which is not a number to read.
        """
        return [
            kind.value
            for kind in EVALUATED_KINDS
            if kind in self.problem.mapping_counts()
        ]

    @staticmethod
    def _of_kind(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Get the data points of one kind of fit mapping."""
        return df[df.kind == kind]

    # --------------------------------------------------------------------
    # report
    # --------------------------------------------------------------------
    def create(
        self,
        output_dir: Path,
        name: str | None = None,
        show_report: bool = False,
        png_dpi: float = PNG_DPI,
    ) -> Path:
        """Create the complete report.

        Writes the figures, the text report, the parameter sets, the report
        document as Typst, PDF and LaTeX with its references and the HTML
        report into `output_dir / name`. A PDF which Typst cannot compile is
        logged and stated in the HTML report; `report.typ` is kept and the
        other outputs are written.

        Args:
            output_dir: base directory of the reports.
            name: name of this report, the id of the optimization by default.
            show_report: open the HTML report in a web browser.
            png_dpi: the resolution of the PNG figures, the copies of the
                figures of the report document, which LaTeX includes, and
                the figures of a report of the format `png`.

        Returns:
            Path of the directory the report was written to.
        """
        if name is None:
            name = self.opt_result.sid if self.opt_result else self.problem.opid
        results_dir = output_dir / name
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        display.section("Report", icon=display.ICON_REPORT)
        display.key_values(
            {
                "directory": name,
                "parameter sets": ", ".join(pset.sid for pset in self.parameter_sets),
                "mappings": ", ".join(
                    f"{count} {kind.value}"
                    for kind, count in self.problem.mapping_counts().items()
                ),
            }
        )

        # the parameters are the input of a report, they are stored with it;
        # the records are written atomically, the report may be written next
        # to the records of its fit, which other runs read
        write_text_atomic(
            results_dir / "parameters.json", str(self.parameter_sets.to_json())
        )

        # metrics of the parameter sets
        for df, name in [
            (self.metrics_df(), "metrics.tsv"),
            (self.metrics_mappings_df(), "metrics_mappings.tsv"),
            (self.metrics_observables_df(), "metrics_observables.tsv"),
            (self.datapoints_df(), "datapoints.tsv"),
        ]:
            df.to_csv(results_dir / name, sep="\t", index=False)
        if self.groupings:
            self.metrics_groups_df().to_csv(
                results_dir / "metrics_groups.tsv", sep="\t", index=False
            )
        if self.opt_result:
            write_text_atomic(
                results_dir / "optimization_result.json",
                str(self.opt_result.to_json()),
            )
            self.opt_result.to_tsv(path=results_dir / "optimization_result.tsv")
        if self.fisher:
            write_text_atomic(
                results_dir / "fisher.json",
                json.dumps(self.fisher.to_dict(), indent=2),
            )
            self._fisher_summary_df().to_csv(
                results_dir / "fisher.tsv", sep="\t", index=False
            )
        if self.posterior:
            self.posterior.write(results_dir / POSTERIOR_FILE)
            self._posterior_summary_df().to_csv(
                results_dir / "posterior.tsv", sep="\t", index=False
            )
        if self.identifiability:
            write_text_atomic(
                results_dir / "identifiability.json",
                str(self.identifiability.to_json()),
            )
            self.identifiability.summary_df().to_csv(
                results_dir / "identifiability.tsv", sep="\t", index=False
            )

        self._write_text_report(path=results_dir / "report.txt")
        self._create_figures(plots_dir=plots_dir, png_dpi=png_dpi)
        # the document shows the figures which exist, so it follows them
        document = self.document(plots_dir)
        pdf_error = self._write_document(document, results_dir)
        self.html_report(
            path=results_dir / "index.html", document=document, pdf_error=pdf_error
        )

        report_path = results_dir / "index.html"
        display.link("report", report_path)
        if show_report:
            webbrowser.open(report_path.resolve().as_uri(), new=2)

        return results_dir

    def document(self, plots_dir: Path) -> Document:
        """Get the report document of the fit, its methods and its results.

        Args:
            plots_dir: the folder of the plots of the report; a figure of the
                document is shown if its file is in the folder.

        Returns:
            The document of `fit_document` with the context of the report.
        """
        return fit_document(self, self.context, plots_dir)

    @staticmethod
    def _write_document(document: Document, results_dir: Path) -> str | None:
        """Write the report document as Typst, PDF and LaTeX.

        Args:
            document: the report document.
            results_dir: the folder of the report.

        Returns:
            The reason the PDF could not be compiled, `None` if it was.
        """
        typ_path = results_dir / "report.typ"
        typ_path.write_text(TypstRenderer().render(document), encoding="utf-8")
        latex = LaTeXRenderer()
        (results_dir / "report.tex").write_text(
            latex.render(document), encoding="utf-8"
        )
        (results_dir / "report_standalone.tex").write_text(
            latex.render_standalone(document), encoding="utf-8"
        )
        write_bibliography(document.references, results_dir)

        # the PDF of an earlier report in the folder is not the PDF of this one
        pdf_path = results_dir / "report.pdf"
        pdf_path.unlink(missing_ok=True)
        try:
            compile_pdf(typ_path, pdf_path)
        except RuntimeError as err:
            logger.warning(
                "The PDF of the report could not be created, '%s' is kept: %s",
                typ_path.name,
                err,
            )
            return str(err)
        return None

    def _write_text_report(self, path: Path) -> None:
        """Write the text report of the problem, the parameters and the runs."""
        info = [self.problem.report(path=None, print_output=False)]
        info.append(self.parameters_report())
        info.extend(self.metrics(pset).report() for pset in self.parameter_sets)
        if self.opt_result:
            info.append(
                self.opt_result.report(
                    path=None, print_output=False, groups=self.parameter_groups()
                )
            )
        if self.identifiability:
            info.append(self.identifiability.report())

        with open(path, "w", encoding="utf-8") as f_report:
            f_report.write("\n".join(info))

    def parameters_report(self) -> str:
        """Get the report of the parameter sets.

        Reports the values of every set and the parameters which ended up close
        to one of their bounds.
        """
        table = self.parameter_sets.to_df(
            units=self.display_units, resolved_units=self.problem.parameter_units
        )
        # the elements of a network are one line per array and set
        grouped = {p.pid for _, members in self._grouping[1] for p in members}
        table = table[~table["parameter"].isin(grouped)]
        info = [
            "-" * 80,
            f"Parameter sets: {[pset.sid for pset in self.parameter_sets]}",
            "-" * 80,
        ]
        if not table.empty:
            # a fit of the elements of networks only has no single parameters
            info.extend([table.to_string(index=False), ""])
        for pset in self.parameter_sets:
            if pset.cost is not None:
                info.append(f"{pset.sid}: cost = {pset.cost:.6g}")
            info.extend(
                f"{pset.sid}: "
                + describe_array(
                    group.label,
                    len(group.ids),
                    members,
                    [pset.values.get(p.pid, float("nan")) for p in members],
                )
                for group, members in self._grouping[1]
            )
            for msg in bound_warnings(
                self.problem.parameters,
                self.x(pset),
                self.problem.scales_initialized,
                groups=self.parameter_groups(),
            ):
                info.append(f"\t>>> {pset.sid}: {msg} <<<")
        info.append("-" * 80)
        return "\n".join(info)

    #: caption of every plot which describes the whole fit
    #: what a value of the report means, shown as a tooltip on its column.
    #: The keys are the column names of the tables of the report
    HINTS: ClassVar[dict[str, str]] = {
        "elements": "Number of elements of the array.",
        "estimated": "Number of elements of the array the fit adjusts.",
        "norm": "Euclidean norm of the values of the estimated elements.",
        "n": "Number of data points which enter the value.",
        "k": "Number of parameters the fit adjusts.",
        "cost": (
            "The objective the optimization minimizes, 0.5 * sum of the squared "
            "weighted residuals. It is defined on the training data, so it is "
            "not reported for the validation data."
        ),
        "MSE": (
            "Mean squared error of the data and the prediction, unweighted, so "
            "the fit mappings with the largest values dominate it."
        ),
        "RMSE": (
            "Root mean squared error, the square root of the MSE, in the unit "
            "of the data."
        ),
        "NRMSE": (
            "Root mean square of the residuals normalized by the mean of their "
            "fit mapping, so every curve counts the same whatever its "
            "magnitude; the RMSE of the small curves is small even when they "
            "are missed by a factor."
        ),
        "RMSE_w": (
            "Root mean square of the residuals of the cost, i.e. the residuals "
            "of the residual type, weighted and with the loss function "
            "applied; the cost of the training data is 0.5 * n * RMSE_w². A "
            "parameter set can have a larger RMSE and smaller weighted "
            "residuals than another one."
        ),
        "R2": (
            "Coefficient of determination, 1 - SSE/SST. The prediction of a "
            "non-linear model is not a linear regression, so this is not the "
            "square of a correlation and is negative when the prediction is "
            "worse than the mean of the data."
        ),
        "within_2fold": (
            "Percentage of the data points whose prediction lies within twofold "
            "of the data, 0.5 <= prediction/data <= 2. Only the data points "
            "with a positive value and prediction have a ratio and are counted; "
            "- if no point has one."
        ),
        "within_1.25fold": (
            "Percentage of the data points with a ratio whose prediction lies "
            "within 1.25-fold of the data."
        ),
        "within_1.5fold": (
            "Percentage of the data points with a ratio whose prediction lies "
            "within 1.5-fold of the data."
        ),
        "within_1sd": (
            "Percentage of the data points with a standard deviation whose "
            "prediction lies within one standard deviation of the data; - if "
            "no point has one."
        ),
        "within_2sd": (
            "Percentage of the data points with a standard deviation whose "
            "prediction lies within two standard deviations of the data; - if "
            "no point has one."
        ),
        "AIC": (
            "Akaike information criterion, n*ln(MSE) + 2*k, up to a constant. "
            "Only differences between models fitted on the same data are "
            "meaningful; the smaller one is preferred."
        ),
        "BIC": (
            "Bayesian information criterion, n*ln(MSE) + k*ln(n), up to the "
            "same constant as the AIC. It charges a parameter more than the "
            "AIC from eight data points on, so it prefers the smaller model."
        ),
        "value": (
            "Value of the parameter in the unit of its fit parameter, the unit "
            "of the table of the parameters."
        ),
        "scale": "Space the optimizer searches the parameter in.",
        "se": (
            "Standard error of the parameter, the square root of the diagonal "
            "of the covariance, in the space the optimizer searches."
        ),
        "cv": (
            "The standard error relative to the value, in percent: the "
            "coefficient of variation of the parameter in the units of the "
            "model (to first order, also for a parameter searched as a "
            "logarithm)."
        ),
        "sd": (
            "Standard deviation of the posterior in the space the optimizer "
            "searches the parameter in, e.g. of its decadic logarithm."
        ),
        "q0.025": "Lower end of the 95 % credible interval of the posterior.",
        "q0.975": "Upper end of the 95 % credible interval of the posterior.",
        "target": "The entity of the model the parameter sets.",
        "ci_lower": "Lower bound of the confidence interval.",
        "ci_upper": "Upper bound of the confidence interval.",
        "identifiability": (
            "What the profile says: identifiable if it crosses the threshold "
            "on both sides, non_identifiable if it stays below it up to a "
            "bound of the parameter, structural if it is flat."
        ),
        "weight": "Weight of the fit mapping in the cost.",
        "start": "Value the optimization starts from.",
        "lower": "Lower bound of the parameter in the optimization.",
        "upper": "Upper bound of the parameter in the optimization.",
    }

    PLOT_CAPTIONS: ClassVar[dict[str, str]] = {
        "profiles": (
            "Profile likelihood of every parameter: the cost with the parameter "
            "fixed and the other parameters optimized, the threshold of the "
            "confidence level and the confidence interval"
        ),
        "traces": "Cost of the optimizers over their steps",
        "waterfall": "Cost of the optimization runs, ordered",
        "goodness_of_fit": (
            "Prediction against the measured data points, per kind of fit "
            "mapping, with the agreement of the training data, the lines of "
            "twofold and the metrics of the panel"
        ),
        "bland_altman": (
            "Agreement of prediction and measurement as a ratio, per kind of "
            "fit mapping, with the agreement of the training data and the "
            "agreement of the panel"
        ),
        "cost_bar": "Cost and weight of every fit mapping",
        "residual_boxplot": "Distribution of the squared weighted residuals",
        "cost_scatter": "Cost of the parameter sets against the reference",
    }

    #: what a figure of the report shows and what to look for in it
    PLOT_HINTS: ClassVar[dict[str, str]] = {
        "profiles": (
            "One profile per parameter. A profile which crosses the dashed "
            "threshold on both sides gives a finite confidence interval; one "
            "which stays below it up to a bound is practically "
            "non-identifiable; a flat profile is structurally "
            "non-identifiable, the parameter is compensated by the others."
        ),
        "traces": (
            "The cost of every optimization run over its steps. Runs which "
            "end at the same cost found the same optimum; a run which stops "
            "much higher is stuck in a local minimum."
        ),
        "waterfall": (
            "The final cost of the runs, ordered. A flat plateau at the left "
            "is the global optimum found repeatedly, which is the evidence "
            "that the multistart converged; steps to the right are local "
            "minima."
        ),
        "goodness_of_fit": (
            "Prediction against measurement, one point per data point, with a "
            "panel per kind of fit mapping. The points scatter around the "
            "diagonal when the model describes the data; a systematic "
            "deviation from it is a systematic error of the model. The band is "
            "the agreement of the training data, i.e. the same band the "
            "Bland-Altman plot draws, here as lines parallel to the diagonal. "
            "The dark red dotted lines are twofold, prediction = 2 x "
            "measurement and prediction = measurement / 2; the caption states "
            "the points between them of every kind. A point which is zero or "
            "negative is not drawn on the logarithmic axes and not counted, "
            "the title of a panel then states the points drawn of all. "
            "The validation panel shows how the fit describes data it was not "
            "fitted on, the outlier panel the data the fit dropped. The box "
            "of a panel carries its R², its normalized RMSE, the RMSE of "
            "the residuals of the cost, its fold errors GMFE and AFE and the "
            "percentages of its points within twofold and within two standard "
            "deviations of the data."
        ),
        "bland_altman": (
            "The ratio of prediction and measurement over the geometric mean "
            "of the two, i.e. the goodness of fit with the diagonal turned "
            "into the horizontal. The band is the same, the bias and the "
            "limits of agreement `bias ± 1.96 SD` of the training data as "
            "fold factors. A bias away from 1 is a systematic over- or "
            "underprediction, wide limits are a large scatter, and a trend "
            "over the mean is a model which describes the large or the small "
            "values better. The box of a panel carries the bias and the SD of "
            "its own points as fold factors and the share of them inside the "
            "limits of agreement of the training data."
        ),
        "cost_bar": (
            "How much every fit mapping contributes to the cost, with its "
            "weight. A mapping which dominates the cost dominates the fit."
        ),
        "residual_boxplot": (
            "The distribution of the squared weighted residuals per mapping. "
            "A mapping whose box sits far above the others is the one the "
            "model describes worst."
        ),
        "cost_scatter": (
            "The cost of every parameter set against the reference set, per "
            "fit mapping. Points below the diagonal are mappings the set "
            "describes better than the reference."
        ),
    }

    def twofold_statement(self) -> str:
        """Get the data points within twofold per kind, with their denominator.

        The points of every kind the goodness of fit has a panel of, counted
        as `within_2fold` counts them, i.e. over the points with a positive
        value and prediction, which are the points the logarithmic axes draw,
        e.g. `Within twofold of the data are 479 of the 523 training data
        points (91.6%) and 212 of the 243 validation data points (87.2%); only
        the data points with a positive value and prediction are counted`. A
        kind without a point with a ratio is left out. With several parameter
        sets the counts of every set follow `for <id>`.

        Returns:
            The statement without a full stop, empty if no kind has a point
            with a ratio.
        """
        statements: list[str] = []
        for pset in self.parameter_sets:
            points = self.points(pset)
            parts = []
            for kind in self.point_kinds():
                of_kind = self._of_kind(points, kind)
                inside, n_ratio = fold_counts(of_kind.DV, of_kind.IPRED)
                if n_ratio:
                    parts.append(
                        f"{inside} of the {n_ratio} {kind} data points "
                        f"({100.0 * inside / n_ratio:.1f}%)"
                    )
            if not parts:
                continue
            listed = (
                parts[0]
                if len(parts) == 1
                else ", ".join(parts[:-1]) + " and " + parts[-1]
            )
            prefix = f"for {pset.sid} " if len(self.parameter_sets) > 1 else ""
            statements.append(f"{prefix}{listed}")
        if not statements:
            return ""
        return (
            f"Within twofold of the data are {'; '.join(statements)}; only the "
            "data points with a positive value and prediction are counted"
        )

    def plot_caption(self, name: str) -> str:
        """Get the caption of a figure of the report.

        The caption of `PLOT_CAPTIONS`; the goodness of fit states the
        percentage of the data points within twofold of every kind as well,
        see `twofold_statement`.

        Args:
            name: name of the figure, e.g. `goodness_of_fit`.

        Returns:
            The caption, without a full stop.
        """
        caption = self.PLOT_CAPTIONS.get(name, name)
        if name == "goodness_of_fit":
            statement = self.twofold_statement()
            if statement:
                caption = f"{caption}. {statement}"
        return caption

    def _plots(self, plots_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
        """Get the plots of the given names which were created.

        A plot of `WIDE_PLOTS` is `wide`, i.e. it covers a full row. A plot
        with its spec is shown inline, see `_plot`.
        """
        return [
            self._plot(
                plots_dir,
                name,
                caption=self.plot_caption(name),
                hint=self.PLOT_HINTS.get(name, ""),
                wide=name in WIDE_PLOTS,
            )
            for name in names
            if (plots_dir / f"{name}.{self.image_format}").exists()
        ]

    def _plot(self, plots_dir: Path, name: str, **fields: Any) -> dict[str, Any]:
        """Get a plot of the page: its file, its fields and its inline figure.

        Args:
            plots_dir: the folder of the plots.
            name: the name of the plot, its file without the suffix.
            **fields: the fields of the plot on the page, e.g. its caption.

        Returns:
            `key`, `src` and `spec`, the file of its spec where it exists,
            the fields, and `svg` and `spec_text` of an SVG with its spec,
            which the page shows inline, see `inline_figure`.
        """
        path = plots_dir / f"{name}.{self.image_format}"
        spec = path.with_suffix(".json")
        return {
            "key": name,
            "src": f"plots/{path.name}",
            "spec": f"plots/{spec.name}" if spec.exists() else None,
            **fields,
            **inline_figure(path),
        }

    def html_context(
        self,
        results_dir: Path,
        name: str,
        document: Document | None = None,
        pdf_error: str | None = None,
    ) -> dict[str, Any]:
        """Collect everything the HTML report shows.

        Args:
            results_dir: directory of the report, the files are relative to it.
            name: name of the report, its directory.
            document: the report document of the tab "Report", the `document`
                of the plots of `results_dir` by default.
            pdf_error: the reason the PDF of the document could not be
                compiled, if it could not.

        Returns:
            The context of the `fit_report.html` template.
        """
        plots_dir = results_dir / "plots"
        if document is None:
            document = self.document(plots_dir)
        # the heading is the id of the fit, not the directory of the report,
        # which e.g. the run layer names `report`
        fit_id = self.opt_result.sid if self.opt_result else self.problem.opid
        counts = self.problem.mapping_counts()
        # the data the model does not describe is not resolved, so it has no
        # rows, no metrics and no figures: it is not part of the report
        kinds = [kind.value for kind in EVALUATED_KINDS]
        metrics = self.metrics_df()
        mapping_metrics = self.metrics_mappings_df()

        # the parameters with one column per set; the elements of a network
        # are one row per array, see `_array_rows`
        psets = list(self.parameter_sets)
        summaries = self._summaries
        single, groups = self._grouping
        parameters = [
            {
                "pid": p.pid,
                "target": p.target_id,
                # not `values`, jinja resolves that to `dict.values`
                "set_values": [
                    f"{pset.values.get(p.pid, float('nan')):.5g}" for pset in psets
                ],
                "lower": f"{p.lower_bound:.4g}",
                "upper": f"{p.upper_bound:.4g}",
                "unit": self.display_units[p.pid],
            }
            for p in single
        ]
        hooks = [
            {
                "name": summary.name,
                "kind": summary.kind,
                "description": summary.description,
                "targets": ", ".join(summary.targets),
            }
            for summary in summaries
        ]
        arrays = self._array_rows(groups, psets)
        # the warnings of the single parameters are under their table, the
        # warnings of the arrays under the table of the arrays
        warnings: list[str] = []
        array_warnings: list[str] = []
        # a set which carries values this problem does not apply: they have no
        # row of the table, which is the parameters of the problem, and they
        # are in `parameters.json`, so the reader of the report is told (#643)
        unapplied_warnings: list[str] = []
        for pset in psets:
            single_messages, array_messages = self._bound_warnings(pset)
            warnings.extend(f"{pset.sid}: {message}" for message in single_messages)
            array_warnings.extend(
                f"{pset.sid}: {message}" for message in array_messages
            )
            if pset.unapplied:
                unapplied_warnings.append(
                    f"{pset.sid}: the set carries {len(pset.unapplied)} value"
                    f"{'s' if len(pset.unapplied) > 1 else ''} this fit does not "
                    f"apply, the model keeps its own: "
                    f"{', '.join(pset.unapplied)}. They have no row below and "
                    f"are marked in parameters.json."
                )

        # the data per experiment and kind
        experiments: list[str] = []
        for experiment in self.problem.experiment_keys:
            if experiment not in experiments:
                experiments.append(experiment)
        data_summary = []
        for experiment in experiments:
            row_counts = [
                sum(
                    1
                    for k, exp in enumerate(self.problem.experiment_keys)
                    if exp == experiment and self.problem.mapping_kinds[k].value == kind
                )
                for kind in kinds
            ]
            data_summary.append(
                {
                    "experiment": experiment,
                    "counts": row_counts,
                    "total": sum(row_counts),
                }
            )
        totals = [counts.get(MappingKind(kind), 0) for kind in kinds]

        # one card per fit mapping, with the metrics of the last parameter set
        mapping_rows: list[dict[str, Any]] = mapping_metrics.to_dict(orient="records")
        metrics_by_mapping = {
            (row["parameter_set"], row["experiment"], row["mapping"]): row
            for row in mapping_rows
        }
        reference = psets[-1].sid
        captions = ["Data, simulation and agreement"]
        mappings: list[dict[str, Any]] = []
        for k, mapping_id in enumerate(self.problem.mapping_keys):
            sid = self.problem.experiment_keys[k]
            row = metrics_by_mapping.get((reference, sid, mapping_id))
            plots = self._plots(plots_dir, [f"{sid}_{mapping_id}"])
            for plot, caption in zip(plots, captions, strict=False):
                plot["caption"] = caption
            mappings.append(
                {
                    "experiment": sid,
                    "mapping": mapping_id,
                    "observable": self.problem.yid_observable[k],
                    "kind": self.problem.mapping_kinds[k].value,
                    "metrics": {
                        "n": int(row["n"]) if row else "-",
                        "RMSE": _cell(row["RMSE"], ".4g") if row else "-",
                        "R²": _cell(row["R2"], ".4g") if row else "-",
                        "2-fold": (
                            f"{_percent(row['within_2fold'])}%"
                            if row and np.isfinite(row["within_2fold"])
                            else "-"
                        ),
                        "GMFE": _cell(row["GMFE"], ".3g") if row else "-",
                    },
                    "plots": plots,
                }
            )

        files = [
            {"href": "report.txt", "label": "report.txt"},
            {"href": "parameters.json", "label": "parameters.json"},
            {"href": "metrics.tsv", "label": "metrics.tsv"},
            {"href": "metrics_mappings.tsv", "label": "metrics_mappings.tsv"},
            {"href": "metrics_observables.tsv", "label": "metrics_observables.tsv"},
            {"href": "datapoints.tsv", "label": "datapoints.tsv"},
        ]
        if self.opt_result:
            files.append(
                {
                    "href": "optimization_result.json",
                    "label": "optimization_result.json",
                }
            )
        if self.fisher:
            files.append({"href": "fisher.tsv", "label": "fisher.tsv"})
        if self.posterior:
            files.append({"href": POSTERIOR_FILE, "label": POSTERIOR_FILE})
            files.append({"href": "posterior.tsv", "label": "posterior.tsv"})
        if self.identifiability:
            files.append(
                {"href": "identifiability.json", "label": "identifiability.json"}
            )
            files.append(
                {"href": "identifiability.tsv", "label": "identifiability.tsv"}
            )

        badges = [
            {"label": "mappings", "value": len(self.problem.mapping_keys)},
            {"label": "parameters", "value": len(self.problem.parameters)},
        ]
        training = metrics[metrics.kind == MappingKind.TRAINING.value]
        if len(training):
            badges.append({"label": "cost", "value": f"{training.cost.min():.6g}"})
        if self.opt_result:
            badges.append({"label": "runs", "value": self.opt_result.size})
        if self.identifiability:
            badges.append(
                {
                    "label": "identifiable",
                    "value": (
                        f"{self.identifiability.n_identifiable}/"
                        f"{len(self.identifiability.profiles)}"
                    ),
                }
            )

        runs: list[list[str]] = []
        run_columns: list[str] = []
        if self.opt_result:
            run_columns = ["run", "success", "duration", "cost"]
            runs = [
                [
                    str(row["run"]),
                    str(row["success"]),
                    f"{row['duration']:.2f}",
                    f"{row['cost']:.6g}",
                ]
                for row in self.opt_result.df_fits[run_columns].to_dict(
                    orient="records"
                )
            ]

        return {
            # the HTML of the document is marked safe in the template; its
            # renderer escapes every text of the document
            "report_html": HTMLRenderer().fragment(document),
            "document_title": document.title,
            "pdf_error": pdf_error,
            "downloads": [
                {"href": file, "label": label}
                for file, label in DOCUMENT_FILES
                if (results_dir / file).exists()
            ],
            "title": f"{fit_id} | sbmlsim fit report",
            "fit_id": fit_id,
            # the id of the problem under the id of the fit, unless the same
            "opid": "" if self.problem.opid == fit_id else self.problem.opid,
            "version": __version__,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "badges": badges,
            "kinds": kinds,
            "fit_info": self.fit_info(),
            "parameters": parameters,
            "hooks": hooks,
            "arrays": arrays,
            "versioned_parameters": has_renamed_targets(single),
            "parameter_set_ids": [pset.sid for pset in psets],
            "bound_warnings": warnings,
            "array_bound_warnings": array_warnings,
            "unapplied_warnings": unapplied_warnings,
            "settings": {
                key.replace("_", " "): (
                    str(AbsoluteTolerance.of(self.settings.absolute_tolerance))
                    if key == "absolute_tolerance"
                    else value
                )
                for key, value in self.settings.to_dict().items()
            },
            "data_summary": data_summary,
            "data_total": {"counts": totals, "total": sum(totals)},
            # the hints by column and by the header a table shows
            "hints": {
                **self.HINTS,
                **{
                    header: self.HINTS[column]
                    for column, header in METRIC_HEADERS.items()
                },
            },
            "metrics_columns": [
                {
                    "name": METRIC_HEADERS.get(column, column),
                    "hint": self.HINTS.get(column),
                }
                for column in metrics.columns
            ],
            # a metric which is not defined, e.g. the cost of the validation
            # data, is `-`
            "metrics": [
                [
                    _percent(value)
                    if column.startswith("within_")
                    else _cell(value, ".6g")
                    for column, value in row.items()
                ]
                for row in metrics.to_dict(orient="records")
            ],
            "mapping_metrics_columns": [
                "parameter set",
                "experiment",
                "mapping",
                "kind",
                "n",
                "MSE",
                "RMSE",
                "NRMSE",
                "RMSE_w",
                "R2",
                METRIC_HEADERS["within_2fold"],
            ],
            "numeric_columns": [
                "n",
                "MSE",
                "RMSE",
                "NRMSE",
                "RMSE_w",
                "R2",
                METRIC_HEADERS["within_2fold"],
            ],
            "mapping_metrics": [
                {
                    "parameter_set": row["parameter_set"],
                    "experiment": row["experiment"],
                    "mapping": row["mapping"],
                    "kind": row["kind"],
                    "n": int(row["n"]),
                    "mse": _cell(row["MSE"], ".4g"),
                    "rmse": _cell(row["RMSE"], ".4g"),
                    "nrmse": _cell(row["NRMSE"], ".4g"),
                    "rmse_w": _cell(row["RMSE_w"], ".4g"),
                    "r2": _cell(row["R2"], ".4g"),
                    "within_2fold": _percent(row["within_2fold"]),
                }
                for row in mapping_rows
            ],
            "run_plots": self._plots(plots_dir, ["traces", "waterfall"]),
            "result_plots": self._plots(
                plots_dir,
                [
                    "goodness_of_fit",
                    "bland_altman",
                    "cost_scatter",
                ],
            ),
            # the figures with a row per fit mapping are in the tab of the
            # mappings, with the table of their metrics
            "mapping_plots": self._plots(plots_dir, ["cost_bar", "residual_boxplot"]),
            "mappings": mappings,
            "run_columns": run_columns,
            "runs": runs,
            "identifiability": self._identifiability_context(plots_dir),
            "fisher": self._fisher_context(),
            "posterior": self._posterior_context(),
            "priors": self._priors_context(),
            "groupings": self._groupings_context(plots_dir),
            "files": files,
        }

    def _groupings_context(self, plots_dir: Path) -> list[dict[str, Any]]:
        """Get the figures and the metrics of every grouping for the page."""
        if not self.groupings:
            return []
        metrics = self.metrics_groups_df()
        captions = {
            "goodness_of_fit": "Goodness of fit per group and kind",
            "bland_altman": "Bland-Altman per group and kind",
            "criteria": "Within twofold and within two SD per group and kind",
        }
        context: list[dict[str, Any]] = []
        for grouping in self.groupings:
            plots = []
            for plot, caption in captions.items():
                page = 1
                while (
                    plots_dir
                    / f"{grouping_page(plot, grouping.name, page)}.{self.image_format}"
                ).exists():
                    name = grouping_page(plot, grouping.name, page)
                    plots.append(
                        self._plot(
                            plots_dir,
                            name,
                            caption=f"{caption}, by {grouping.name}"
                            + (f", page {page}" if page > 1 else ""),
                            wide=plot != "criteria",
                        )
                    )
                    page += 1
            rows = metrics[metrics.grouping == grouping.name].drop(columns="grouping")
            context.append(
                {
                    "name": grouping.name,
                    "plots": plots,
                    "columns": [METRIC_HEADERS.get(c, c) for c in rows.columns],
                    "kinds": list(rows["kind"]),
                    "rows": [
                        [
                            _metric_cell(column, value)
                            for column, value in zip(rows.columns, row, strict=True)
                        ]
                        for row in rows.itertuples(index=False)
                    ],
                }
            )
        return context

    def _array_rows(
        self,
        groups: Sequence[tuple[ParameterGroup, Sequence[Any]]],
        psets: Sequence[ParameterSet],
    ) -> list[dict[str, Any]]:
        """Get the rows of the arrays of the networks for the overview.

        Args:
            groups: the arrays with the parameters of the fit which are their
                elements, see `sbmlsim.fit.derived.group_parameters`.
            psets: the parameter sets of the report.

        Returns:
            One row per array with the number of elements, the estimated
            ones, the bounds when the elements agree on them, and the
            minimum, the maximum and the norm of the values of every set.
            A versioned element is one element, its versions are values of
            the array.
        """
        rows: list[dict[str, Any]] = []
        for group, members in groups:
            pids = [p.pid for p in members]
            lower = {p.lower_bound for p in members}
            upper = {p.upper_bound for p in members}
            set_values: list[list[str]] = []
            for pset in psets:
                values = np.asarray(
                    [pset.values.get(pid, float("nan")) for pid in pids], dtype=float
                )
                set_values.append(
                    ["-", "-", "-"]
                    if values.size == 0
                    else [
                        f"{values.min():.4g}",
                        f"{values.max():.4g}",
                        f"{np.linalg.norm(values):.4g}",
                    ]
                )
            rows.append(
                {
                    "label": group.label,
                    "elements": len(group.ids),
                    "estimated": len({p.entity_id for p in members}),
                    "lower": f"{lower.pop():.4g}" if len(lower) == 1 else "-",
                    "upper": f"{upper.pop():.4g}" if len(upper) == 1 else "-",
                    "set_values": set_values,
                }
            )
        return rows

    @cached_property
    def _summaries(self) -> list[HookSummary]:
        """Get the summaries of the hooks of the problem, e.g. its networks."""
        return hook_summaries(self.problem.hybridizations)

    @cached_property
    def _grouping(
        self,
    ) -> tuple[list[FitParameter], list[tuple[ParameterGroup, list[FitParameter]]]]:
        """Split the parameters into single ones and the arrays of the hooks."""
        return group_parameters(self.problem.parameters, self._summaries)

    def _bound_warnings(self, pset: ParameterSet) -> tuple[list[str], list[str]]:
        """Get the bound warnings of a set, split into single ones and arrays.

        Args:
            pset: parameter set of the report.

        Returns:
            The messages of the parameters which are no elements of a network
            and the messages of the arrays, see `bound_warnings`.
        """
        single = {p.pid for p in self._grouping[0]}
        x = self.x(pset)
        scales = self.problem.scales_initialized
        parameters = self.problem.parameters
        parts: list[list[str]] = []
        for is_single in (True, False):
            ks = [k for k, p in enumerate(parameters) if (p.pid in single) == is_single]
            parts.append(
                bound_warnings(
                    [parameters[k] for k in ks],
                    x[ks],
                    [scales[k] for k in ks],
                    groups=self.parameter_groups(),
                )
            )
        return parts[0], parts[1]

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        """Get the arrays of the networks as groups of elements.

        Returns:
            label of the array -> the ids of all its elements, for
            `bound_warnings` and `OptimizationResult.report`.
        """
        return {group.label: group.ids for group, _ in self._grouping[1]}

    def _fisher_summary_df(self) -> pd.DataFrame:
        """Get the table of the Fisher information with the units of the report.

        The unit of a parameter is the one of `display_units`, the table of
        `FisherInformation.summary_df` has the units of pint of its values.

        Raises:
            ValueError: if the report has no Fisher information.
        """
        if self.fisher is None:
            raise ValueError("The report has no Fisher information.")
        df = self.fisher.summary_df
        units = self.display_units
        df["unit"] = [
            units.get(pid, unit)
            for pid, unit in zip(df["parameter"], df["unit"], strict=True)
        ]
        return df

    def _fisher_context(self) -> dict[str, Any] | None:
        """Collect what the Fisher information of the report shows."""
        fim = self.fisher
        if fim is None:
            return None

        df = self._fisher_summary_df()
        eigenvalues = fim.eigenvalues
        info: dict[str, Any] = {
            "parameter set": fim.sid,
            "parameter scale": (
                f"{fim.scale.name}, per parameter in the table"
                if fim.has_mixed_scales
                else fim.scale.name
            ),
            "data points": str(fim.n),
            "rank": f"{fim.rank} of {fim.k}",
            "condition number": f"{fim.condition_number:.4g}",
            "confidence level": f"{fim.alpha:.0%}",
        }
        return {
            "info": info,
            "identifiable": fim.is_identifiable,
            "columns": self._analysis_columns(df),
            "rows": self._analysis_rows(df, fim.parameter_scales, "se"),
            "eigenvalues": [f"{value:.4g}" for value in eigenvalues],
            **self._correlation_context(fim.pids, fim.correlation),
        }

    def _analysis_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Get the columns of the table of the parameters of an analysis."""
        return [
            {
                "name": column,
                "hint": self.HINTS.get(column),
                # the columns of text are left aligned
                "numeric": column not in {"parameter", "target", "scale", "unit"},
            }
            for column in df.columns
        ]

    def _analysis_rows(
        self, df: pd.DataFrame, scales: Sequence[ParameterScaleType], error: str
    ) -> list[list[str]]:
        """Get the rows of the table of the parameters of an analysis.

        The elements of a network are one row per array, with the norm of
        their values and the range of their errors.

        Args:
            df: the table, one row per parameter in the order of `scales`.
            scales: the scale of every parameter.
            error: the column of the errors.
        """
        groups = self._grouping[1]
        grouped = {p.pid for _, members in groups for p in members}
        rows: list[list[str]] = [
            [_cell(value, ".5g") for value in row.values()]
            for row in df.to_dict(orient="records")
            if row["parameter"] not in grouped
        ]
        for group, members in groups:
            sub = df[df["parameter"].isin([p.pid for p in members])]
            if not len(sub):
                continue
            n = len({p.entity_id for p in members})
            versions = f", {len(sub)} parameters" if len(sub) != n else ""
            names = {scales[i].name for i in sub.index}
            cells = {
                "parameter": (
                    f"{group.label} ({n} element{'s' if n != 1 else ''}{versions})"
                ),
                "value": f"norm {np.linalg.norm(sub['value'].to_numpy()):.4g}",
                "scale": names.pop() if len(names) == 1 else "mixed",
                # an error which is not defined, e.g. without degrees of
                # freedom, is no error
                error: (
                    f"{sub[error].min():.3g} to {sub[error].max():.3g}"
                    if sub[error].notna().any()
                    else "-"
                ),
                "unit": display.ELEMENT_UNIT_LABEL,
            }
            rows.append([cells.get(column, "-") for column in df.columns])
        return rows

    def _correlation_context(
        self, pids: Sequence[str], correlation: pd.DataFrame
    ) -> dict[str, Any]:
        """Get the correlation matrix of an analysis, without the elements of a network."""
        grouped = {p.pid for _, members in self._grouping[1] for p in members}
        keep = [i for i, pid in enumerate(pids) if pid not in grouped]
        return {
            "pids": [pids[i] for i in keep],
            "correlation": [
                [_cell(correlation.iloc[i, j], ".3f") for j in keep] for i in keep
            ],
            "note": (
                f"The correlation is shown for the {len(keep)} parameters which "
                f"are no elements of a network; the {len(pids) - len(keep)} "
                f"elements are left out."
                if grouped
                else None
            ),
        }

    def _posterior_summary_df(self) -> pd.DataFrame:
        """Get the table of the posterior with the units of the report.

        Raises:
            ValueError: if the report has no posterior.
        """
        if self.posterior is None:
            raise ValueError("The report has no posterior.")
        df = self.posterior.summary_df
        units = self.display_units
        df["unit"] = [
            units.get(pid, unit)
            for pid, unit in zip(df["parameter"], df["unit"], strict=True)
        ]
        return df

    def _posterior_context(self) -> dict[str, Any] | None:
        """Collect what the posterior of the report shows."""
        posterior = self.posterior
        if posterior is None:
            return None
        df = self._posterior_summary_df()
        if list(df["target"]) == list(df["parameter"]):
            df = df.drop(columns="target")
        scales = {s.name for s in posterior.scales}
        info: dict[str, Any] = {
            "parameter set": posterior.sid,
            "method": (
                "Laplace approximation"
                if posterior.method == "laplace"
                else posterior.method
            ),
            "objective": posterior.objective.value,
            "parameter scale": (
                scales.pop() if len(scales) == 1 else "per parameter in the table"
            ),
            "data points": str(posterior.n),
        }
        if posterior.objective is ObjectiveType.LEAST_SQUARES:
            info["residual variance"] = f"{posterior.sigma2:.4g}"
        info["credible level"] = "95%"
        return {
            "info": info,
            "objective": posterior.objective.value,
            "problems": list(posterior.problems),
            "columns": self._analysis_columns(df),
            "rows": self._analysis_rows(df, posterior.scales, "sd"),
            **self._correlation_context(posterior.pids, posterior.correlation),
        }

    def _priors_context(self) -> dict[str, Any] | None:
        """Collect the priors of the parameters and their correlation.

        `None` for a problem whose parameters have no prior.
        """
        parameters = self.problem.parameters
        if not any(p.prior is not None for p in parameters):
            return None
        rows = [
            {
                "pid": p.pid,
                "prior": str(p.prior) if p.prior is not None else "uniform(bounds)",
                "lower": f"{p.lower_bound:.4g}",
                "upper": f"{p.upper_bound:.4g}",
                "unit": self.display_units[p.pid],
            }
            for p in parameters
        ]
        correlation = self.problem.prior_correlation
        context: dict[str, Any] = {
            "rows": rows,
            "objective": self.settings.objective.value,
            "used": self.settings.objective is ObjectiveType.POSTERIOR,
            "pids": [],
            "correlation": [],
        }
        if correlation is not None:
            matrix = correlation.correlation_matrix()
            context["pids"] = list(correlation.targets)
            context["correlation"] = [
                [_cell(float(value), ".3f") for value in row] for row in matrix
            ]
        return context

    def _identifiability_context(self, plots_dir: Path) -> dict[str, Any] | None:
        """Collect what the identifiability section of the HTML report shows."""
        result = self.identifiability
        if result is None:
            return None

        def _bound(value: float | None, bound: float, sign: str) -> str:
            """Format a bound of a confidence interval, open sides as a bound."""
            return f"{sign} {bound:.4g}" if value is None else f"{value:.4g}"

        rows = []
        for pid, profile in result.profiles.items():
            p = result.parameter(pid)
            identifiability = profile.identifiability
            rows.append(
                {
                    "pid": pid,
                    "value": f"{profile.value_optimum:.5g}",
                    "ci_lower": _bound(profile.ci_lower, p.lower_bound, "<"),
                    "ci_upper": _bound(profile.ci_upper, p.upper_bound, ">"),
                    "unit": p.unit or "model",
                    "identifiability": identifiability.value if identifiability else "",
                    "label": identifiability.label if identifiability else "",
                    "n_points": len(profile),
                    "converged": bool(np.all(profile.converged)),
                }
            )
        info = {
            "parameter set": result.parameter_set.sid,
            "cost": f"{result.cost:.6g}",
            "minimal cost": f"{result.cost_min:.6g}",
            "confidence level": f"{result.settings.alpha:.0%}",
            "degrees of freedom": str(result.settings.degrees_of_freedom),
            "threshold": f"{result.threshold:.6g}",
            "reoptimize": str(result.settings.reoptimize),
            "duration": f"{result.duration:.1f} s",
        }
        plots = self._plots(plots_dir, ["profiles"])
        profile_plots = self._plots(
            plots_dir, [f"profile_{pid}" for pid in result.profiles]
        )
        for plot, pid in zip(profile_plots, result.profiles, strict=False):
            plot["caption"] = (
                f"Profile of '{pid}' and the paths of the other parameters along it"
            )
        return {
            "info": info,
            "rows": rows,
            "plots": plots,
            "profile_plots": profile_plots,
            "better_optimum": result.better_optimum,
            # the parameters the analysis gave up, so the report names them
            # instead of leaving them out silently (#522)
            "incomplete": [
                {"pid": pid, "reason": reason}
                for pid, reason in sorted(result.incomplete.items())
            ],
        }

    def fit_info(self) -> dict[str, str]:
        """Get the key facts of the fit, the same the console reports."""
        info: dict[str, str] = {
            "problem": self.problem.opid,
            # the engine which simulated the evaluations of the report
            "engine": self.problem.engine.value,
            "parameter sets": ", ".join(pset.sid for pset in self.parameter_sets),
            "fit mappings": ", ".join(
                f"{count} {kind.value}"
                for kind, count in self.problem.mapping_counts().items()
            ),
            # every class with fit mappings once, a class has one collection per
            # kind or selection, and a collection whose mappings are all
            # filtered out by the selection of the problem has none
            "experiments": ", ".join(
                dict.fromkeys(
                    self.problem.mapping_collections[k].experiment_class.__name__
                    for k in sorted(set(self.problem.collection_indices))
                )
            ),
        }
        if self._summaries:
            info["networks"] = ", ".join(
                f"{s.name} ({s.kind})" for s in self._summaries
            )
        info["base path"] = paths_text(self.problem.base_path)
        info["data path"] = paths_text(self.problem.data_path)
        if self.opt_result:
            # where the optimizer of the runs ran, for a record which says it
            place = OPTIMIZER_TEXT.get(self.opt_result.optimizer_of or "")
            info["optimization"] = f"{self.opt_result.size} runs" + (
                f", {place}" if place else ""
            )
        return info

    def html_report(
        self,
        path: Path,
        name: str | None = None,
        document: Document | None = None,
        pdf_error: str | None = None,
    ) -> None:
        """Create the interactive HTML report of the fit.

        The report is a single page with two tabs. "Report" is the report
        document with the methods and the results, followed by the overview of
        the fit, its results and its identifiability; "Fit mappings" is the
        figures of the cost and weight and of the residuals of every fit
        mapping, the table of the metrics of every fit mapping and a card per
        fit mapping. It is rendered from the `fit_report.html`
        template and needs no network access; its inline figures are
        interactive with the script `figures.js` beside it, see
        `write_figure_script`.

        Args:
            path: file to write.
            name: name of the report, the directory of `path` by default.
            document: the report document, see `html_context`.
            pdf_error: the reason the PDF could not be compiled, if it could not.
        """
        context = self.html_context(
            results_dir=path.parent,
            name=name if name else path.parent.name,
            document=document,
            pdf_error=pdf_error,
        )
        template = template_environment().get_template("fit_report.html")
        text = template.render(context)
        if 'data-figure="' in text:
            # the inline figures are interactive with the script beside the page
            write_figure_script(path.parent)
            text = template.render({**context, "figure_script": FIGURE_SCRIPT})
        with open(path, "w", encoding="utf-8") as f_html:
            f_html.write(text)

    # --------------------------------------------------------------------
    # figures
    # --------------------------------------------------------------------
    def _create_figures(self, plots_dir: Path, png_dpi: float = PNG_DPI) -> None:
        """Create all figures of the report.

        Every figure is a chart of the renderer of sbmlsim-core, saved as its
        image and its spec, see `sbmlsim.report.figures.save_figure`. The
        figures of the report document (`FIGURES`) are saved as PNG as well
        if the format of the report is another one, since LaTeX includes PNG
        and no SVG.

        Args:
            plots_dir: the folder of the figures.
            png_dpi: the resolution of the PNG figures.
        """
        self._png_dpi = float(png_dpi)
        if self.opt_result:
            self.plot_traces(path=plots_dir / f"traces.{self.image_format}")
            if self.opt_result.size > 1:
                self.plot_waterfall(path=plots_dir / f"waterfall.{self.image_format}")

        self.plot_goodness_of_fit(
            path=plots_dir / f"goodness_of_fit.{self.image_format}"
        )
        self.plot_bland_altman(path=plots_dir / f"bland_altman.{self.image_format}")
        self.plot_cost_bar(path=plots_dir / f"cost_bar.{self.image_format}")
        self.plot_residual_boxplot(
            path=plots_dir / f"residual_boxplot.{self.image_format}"
        )
        if len(self.parameter_sets) > 1:
            self.plot_cost_scatter(path=plots_dir / f"cost_scatter.{self.image_format}")

        for grouping in self.groupings:
            self.plot_grouping(grouping, plots_dir)
        if self.mapping_figures:
            self.plot_mapping(output_dir=plots_dir)
        if self.identifiability:
            plot_all(
                self.identifiability,
                plots_dir,
                self.image_format,
                png_dpi=self._png_dpi,
            )

    def _save_chart(self, chart: Chart, path: Path) -> None:
        """Save a figure as its image and its spec, see `save_figure`.

        A figure of the report document (`FIGURES`) in another format than
        PNG is saved as PNG next to it as well, which LaTeX includes; a
        figure of the format PNG is saved at the resolution of the report.
        """
        png = path.stem in FIGURES or path.suffix.lower() == ".png"
        save_figure(chart, path, png_dpi=self._png_dpi if png else None)

    @staticmethod
    def _metrics_boxes(chart: Chart, boxes: Sequence[tuple[ChartPlot, str]]) -> None:
        """Draw boxes of key metrics into the upper left corner of plots.

        The renderer places a text at values of the axes, so a box is placed
        by the geometry of the chart as it is drawn: one text of its lines,
        `BOX_LINE` pixels apart, below the top of the panel on a white box as
        wide as the renderer measures the longest line and no wider than the
        panel. The plots of the boxes are drawn before the boxes, and a box
        changes neither their axes nor their legends.

        Args:
            chart: the chart, with every other element drawn.
            boxes: the plots of the chart and the text of their box, a line
                per line of the box.
        """
        boxes = [(plot, text) for plot, text in boxes if text]
        if not boxes:
            return
        core = import_core()
        geometry = core.figure_geometry(chart.to_json())
        drawn = {plot["id"]: plot for plot in geometry["plots"]}
        for plot, text in boxes:
            panel = drawn[plot.id]
            lines = text.splitlines()
            # as wide as the renderer measures the widest line, never wider
            # than the panel
            width = min(
                core.text_width(text, TEXT_SIZE) + 2 * BOX_PAD,
                panel["rect"]["width"] - 2 * BOX_MARGIN,
            )
            height = BOX_LINE * len(lines) + 2 * BOX_PAD

            left, top = BOX_MARGIN, BOX_MARGIN
            plot.polygon(
                [
                    [
                        _panel_point(panel, left, top),
                        _panel_point(panel, left + width, top),
                        _panel_point(panel, left + width, top + height),
                        _panel_point(panel, left, top + height),
                    ]
                ],
                width=1.0,
                color=METRICS_BOX["edge"],
                fill=METRICS_BOX["fill"],
                fill_opacity=METRICS_BOX["fill_opacity"],
            )
            # the lines of a text are centred on its position
            x, y = _panel_point(panel, left + BOX_PAD, top + height / 2)
            plot.text(x, y, "\n".join(lines), color="#000000")

    @staticmethod
    def _log_limits(*data: Any, factor: float = 10.0) -> tuple[float, float]:
        """Get the limits of a logarithmic axis for the given data.

        Data points which are zero or negative cannot be shown on a logarithmic
        axis, only the positive values define the limits.
        """
        values = np.concatenate([np.asarray(d, dtype=float).ravel() for d in data])
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size == 0:
            # no positive data, the limits are a decade around one
            return 1.0 / factor, factor
        return float(np.min(positive)) / factor, float(np.max(positive)) * factor

    @staticmethod
    def _has_positive(*data: Any) -> bool:
        """Whether data has a positive finite value, which a log axis can show.

        A panel without positive values, e.g. the weighted residuals of a
        perfect fit, has no logarithm and stays linear.
        """
        for d in data:
            values = np.asarray(d, dtype=float).ravel()
            if np.any(np.isfinite(values) & (values > 0.0)):
                return True
        return False

    def _mapping_title(self, k: int) -> str | None:
        """Get the title of a figure of a fit mapping, once over its panels.

        The experiment, the mapping and its kind, and below them the observable
        with its unit and the metadata of the mapping, see `mapping_subtitle`.
        A long title is wrapped between its words, never inside an
        identifier. `None` without titles.
        """
        if not self.show_titles:
            return None
        details = ", ".join(
            part for part in (self._y_label(k), self.mapping_subtitle(k)) if part
        )
        return "\n".join(
            textwrap.fill(
                line, width=80, break_long_words=False, break_on_hyphens=False
            )
            for line in (self.mapping_title(k), details)
            if line
        )

    @staticmethod
    def _title_width(title: str | None, width: float) -> float:
        """Get the width of a chart so that its title fits, at least `width`."""
        if not title:
            return width
        measured = import_core().text_width(title, TITLE_SIZE, bold=True)
        return max(width, measured + 2 * BOX_MARGIN)

    def _is_scalar(self, k: int) -> bool:
        """Whether mapping `k` compares one value per simulation."""
        return self.problem.observation_kinds[k] is ObservationKind.SCALAR

    def _x_positions(self, k: int) -> np.ndarray:
        """Get the x positions of the reference points of mapping `k` in a figure.

        The reference points of a value per simulation have no x, they are drawn
        next to each other at `0..n-1`.
        """
        x_ref = np.asarray(self.problem.x_references[k], dtype=float)
        if self._is_scalar(k):
            return np.arange(x_ref.size, dtype=float)
        return x_ref

    @staticmethod
    def _with_unit(label: str, unit: str | None) -> str:
        """Get `<label> [<unit>]` with the unit in its short form, e.g. `mg/l`.

        The label stands alone without a unit and for a dimensionless one; a
        unit which the registry does not parse is written as it is.
        """
        if not unit:
            return label
        try:
            text = f"{ureg.Unit(unit):~P}"
        except PintError, TokenError, ValueError:
            text = unit
        return f"{label} [{text}]" if text else label

    def _x_label(self, k: int) -> str:
        """Get the label of the x axis of mapping `k`.

        `<x> [<unit>]` of a timecourse, `<target> [<unit>]` of values over a
        dimension and the mapping for a value per simulation, which has no
        quantity on x.
        """
        x_id = self.problem.xid_observable[k]
        if self._is_scalar(k) or x_id is None:
            return self.problem.mapping_keys[k]
        if self.problem.observation_kinds[k] is ObservationKind.DIMENSION:
            x_id = x_id.split(".", 1)[-1]
        return self._with_unit(x_id, self.problem.x_units[k])

    def _y_label(self, k: int) -> str:
        """Get the label `<y> [<unit>]` of the y axis of mapping `k`."""
        return self._with_unit(self.problem.yid_observable[k], self.problem.y_units[k])

    def _simulation_offset(self, ks: int) -> float:
        """Get the x offset of parameter set `ks` from the rows of a value per simulation.

        The simulated value is drawn next to every row of the data rather than
        on top of it; the offsets of all parameter sets stay within half the
        distance of two rows.
        """
        return 0.5 * (ks + 1) / (len(self.parameter_sets) + 1)

    def _x_limits(
        self, k: int, x_obs: Sequence[np.ndarray]
    ) -> tuple[float, float] | None:
        """Get the x limits of mapping `k`.

        The limits of a timecourse are the data, of values over a dimension the
        data and the simulated curves, and of a value per simulation the rows
        and the simulated values next to them. A single x, e.g. values over a
        dimension of one value, is drawn at a quarter of the width.

        Returns:
            The limits, `None` for an axis without a finite value, which the
            renderer scales.
        """
        x_pos = self._x_positions(k)
        if self._is_scalar(k):
            return -0.6, x_pos.size + 1.0
        if self.problem.observation_kinds[k] is ObservationKind.DIMENSION:
            values = np.concatenate(
                [x_pos, *[np.asarray(x, dtype=float) for x in x_obs]]
            )
        else:
            values = np.asarray(x_pos, dtype=float)
        # a measurement at infinity, a steady state, has no place on the axis
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        low, high = float(np.min(values)), float(np.max(values))
        if high == low:
            delta = abs(high) or 1.0
            return low - 0.1 * delta, high + 0.3 * delta
        delta = high - low
        return low - 0.1 * delta, high + 0.1 * delta

    def _reference_errors(self, k: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Get the lower and the upper error of the data of mapping `k`.

        A standard deviation is symmetric, a geometric mean is divided and
        multiplied by its geometric SD; an error which the weights filled is
        no error of the data and is `NaN`.

        Returns:
            The lower and the upper error of every point, `None` without
            errors.
        """
        y_ref = np.asarray(self.problem.y_references[k], dtype=float)
        y_ref_err = self.problem.y_errors[k]
        measured = self.problem.y_errors_measured[k]
        log_err = self.problem.y_errors_log[k]
        if log_err is not None:
            factor = np.exp(np.asarray(log_err, dtype=float))
            errors = np.array([y_ref - y_ref / factor, y_ref * factor - y_ref])
        elif y_ref_err is not None:
            err = np.asarray(y_ref_err, dtype=float)
            errors = np.array([err, err])
        else:
            return None
        if measured is not None:
            errors = np.where(np.asarray(measured, dtype=bool), errors, np.nan)
        return errors[0], errors[1]

    def _curve_y_axis(
        self, k: int, res_data: Mapping[str, dict[str, list[Any]]], log: bool
    ) -> ChartAxis:
        """Get the y axis of the curve of mapping `k`, linear or logarithmic.

        A logarithmic axis of a mapping without a positive value stays linear.
        Its bottom is 0.3 times the smallest reference value, its top 1.5 times
        the largest value of the panel, the error bars included, so the
        markers of the data stay inside the panel also when the data spans less
        than a decade.
        """
        y_ref = np.asarray(self.problem.y_references[k], dtype=float)
        simulated = [
            np.asarray(res_data[p.sid][key][k], dtype=float)
            for p in self.parameter_sets
            for key in ("y_obs", "y_obsip")
        ]
        label = self._y_label(k)
        if not log or not self._has_positive(y_ref, *simulated):
            return ChartAxis(label=label)
        errors = self._reference_errors(k)
        upper = y_ref + errors[1] if errors is not None else y_ref
        bottom = self._log_limits(y_ref, factor=1.0 / 0.3)[0]
        top = self._log_limits(y_ref, upper, *simulated, factor=1.5)[1]
        return ChartAxis(label=label, scale="log", min=bottom, max=top)

    def plot_mapping(self, output_dir: Path) -> None:
        """Plot the figure of every fit mapping: its curve and its agreement.

        Four panels: the data and the simulation of every parameter set,
        linear and with a logarithmic y axis, and the goodness of fit and the
        Bland-Altman plot of the points of the mapping, drawn as the summary
        figures draw them (`_draw_goodness_of_fit`, `_draw_bland_altman`), so
        a point reads the same in the figure of its mapping and in the summary
        figures. The prediction at the data points are hollow markers on the
        curve, the points of the lower panels. The box of the goodness of fit
        has the metrics of the mapping, see `mapping_text`. The figures are
        drawn in this process, one after the other: a figure of the renderer
        is milliseconds.
        """
        res_data = {pset.sid: self.residual_data(pset) for pset in self.parameter_sets}
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        mappings = range(len(self.problem.mapping_keys))
        with progress_activity(
            "report", "mapping figures", total=len(mappings)
        ) as drawn:
            for k in mappings:
                self.draw_mapping(k, output_dir, res_data, points)
                drawn.advance()

    def draw_mapping(
        self,
        k: int,
        output_dir: Path,
        res_data: Mapping[str, dict[str, list[Any]]],
        points: Mapping[str, pd.DataFrame],
    ) -> Chart:
        """Draw and save the figure of one fit mapping, see `plot_mapping`.

        The legend of the curves is right of the upper row, the legend of the
        agreement right of the lower one.

        Args:
            k: index of the fit mapping.
            output_dir: the folder of the figures.
            res_data: the residual data of every parameter set, see
                `residual_data`.
            points: the data points of every parameter set, see `points`.

        Returns:
            The figure, saved as `<experiment>_<mapping>` in the folder.
        """
        mapping_id = self.problem.mapping_keys[k]
        sid = self.problem.experiment_keys[k]
        title = self._mapping_title(k)
        chart = Chart(
            self._title_width(title, MAPPING_SIZE[0]),
            MAPPING_SIZE[1],
            title=title,
            rows=2,
            columns=2,
            legend_scope="row",
            legend_position="outside",
        )
        limits = self._x_limits(
            k, [res_data[p.sid]["x_obs"][k] for p in self.parameter_sets]
        )
        x = ChartAxis(
            label=self._x_label(k),
            min=None if limits is None else limits[0],
            max=None if limits is None else limits[1],
            ticks_visible=not self._is_scalar(k),
        )
        order = ["data", *(pset.sid for pset in self.parameter_sets)]
        for column, log in enumerate((False, True)):
            y = self._curve_y_axis(k, res_data, log)
            curve = chart.plot(0, column, x=x, y=y, legend_order=order)
            self._draw_curve(curve, k, res_data, log=y.scale == "log", bottom=y.min)

        of_mapping = {
            pset.sid: self._of_mapping(points[pset.sid], k)
            for pset in self.parameter_sets
        }
        gof_limits = self._log_limits(
            *[dp.DV for dp in of_mapping.values()],
            *[dp.IPRED for dp in of_mapping.values()],
        )
        x_gof, y_gof = self._goodness_axes(gof_limits, y_label=True)
        gof = chart.plot(1, 0, x=x_gof, y=y_gof)
        self._draw_goodness_of_fit(
            gof, of_mapping, gof_limits, label_band=True, point_labels=False
        )
        own_agreement = self.problem.mapping_kinds[k] is not MappingKind.OUTLIER
        x_altman, y_altman = self._altman_axes(of_mapping, own_agreement, y_label=True)
        altman = chart.plot(1, 1, x=x_altman, y=y_altman)
        self._draw_bland_altman(
            altman,
            of_mapping,
            label_band=True,
            point_labels=False,
            own_agreement=own_agreement,
        )
        self._metrics_boxes(
            chart,
            [
                (gof, self.mapping_text(k)),
                (altman, self._panel_text(of_mapping, "bland_altman")),
            ],
        )
        self._save_chart(chart, output_dir / f"{sid}_{mapping_id}.{self.image_format}")
        return chart

    def _of_mapping(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Get the data points of mapping `k`."""
        return df[
            (df.experiment == self.problem.experiment_keys[k])
            & (df.mapping == self.problem.mapping_keys[k])
        ]

    def _data_label(self, k: int) -> str:
        """Get the label of the data of mapping `k`, with its error and its count.

        `data (mean ± SD, N = 12)`: the error is a standard deviation, see
        `OptimizationProblem.y_errors`, or a geometric standard deviation of
        geometric means, `gmean ×/÷ GSD`, and `N` is the count of the
        subjects, one number when every data point has the same, else the
        range.
        """
        parts: list[str] = []
        if self.problem.y_errors_type[k] == GSD_ERROR:
            parts.append("gmean ×/÷ GSD")
        elif self.problem.y_errors[k] is not None:
            parts.append("mean ± SD")
        counts = self.problem.y_counts[k]
        if counts is not None:
            finite = np.asarray(counts, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                low, high = int(np.min(finite)), int(np.max(finite))
                parts.append(f"N = {low}" if low == high else f"N = {low}-{high}")
        return f"data ({', '.join(parts)})" if parts else "data"

    @staticmethod
    def _draw_errors(
        plot: ChartPlot,
        x: np.ndarray,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        along: Literal["x", "y"],
        color: str,
        legend: str | None,
        opacity: float = 1.0,
    ) -> None:
        """Draw the error bars of the points which have an error.

        Symmetric errors are the error bars of the renderer, asymmetric ones,
        e.g. a geometric SD or the error of a log ratio, are segments of a
        line broken by gaps; a point without a finite error has no bar.

        Args:
            plot: the plot.
            x: the x of every point.
            y: the y of every point.
            lower: the lower error of every point.
            upper: the upper error of every point.
            along: the axis of the errors.
            color: the color of the bars.
            legend: the legend group of the points, which toggles the bars
                with them.
            opacity: the opacity of the bars.
        """
        finite = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(x)
        finite &= np.isfinite(y)
        if not finite.any():
            return
        x, y, lower, upper = x[finite], y[finite], lower[finite], upper[finite]
        if np.allclose(lower, upper, rtol=1e-12, atol=0.0):
            plot.curve(
                x,
                y,
                xerr=lower if along == "x" else None,
                yerr=lower if along == "y" else None,
                line="none",
                width=1.0,
                color=color,
                opacity=opacity,
                legend=legend,
            )
            return
        gap = np.full(x.size, np.nan)
        if along == "y":
            xs = np.column_stack([x, x, gap]).ravel()
            ys = np.column_stack([y - lower, y + upper, gap]).ravel()
        else:
            xs = np.column_stack([x - lower, x + upper, gap]).ravel()
            ys = np.column_stack([y, y, gap]).ravel()
        plot.curve(xs, ys, width=1.0, color=color, opacity=opacity, legend=legend)

    def _draw_curve(
        self,
        plot: ChartPlot,
        k: int,
        res_data: Mapping[str, dict[str, list[Any]]],
        log: bool = False,
        bottom: float | None = None,
    ) -> None:
        """Draw the data of mapping `k` and the simulation of every parameter set.

        A timecourse is the simulated curve with the prediction at the data
        points as hollow markers on it, values over a dimension the simulated
        values with their markers and a value per simulation the one
        simulated value next to every row of the data. On a logarithmic axis
        a value which is not positive is a gap: a curve which is `0`, e.g. the
        urine at the start of a collection, ends at its last positive value
        and starts again at the next one. An error bar whose lower end is at
        or below zero, e.g. a standard deviation as large as its mean, goes to
        the bottom of the axis and keeps its upper half.

        Args:
            plot: the plot.
            k: index of the mapping.
            res_data: the residual data of every parameter set.
            log: whether the y axis is logarithmic.
            bottom: the lower end of a logarithmic y axis, which the lower
                end of an error bar at or below zero is drawn to.
        """

        def shown(values: Any) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            return np.where(values > 0.0, values, np.nan) if log else values

        x_ref = self._x_positions(k)
        y_ref = np.asarray(self.problem.y_references[k], dtype=float)
        scalar = self._is_scalar(k)
        dimension = self.problem.observation_kinds[k] is ObservationKind.DIMENSION

        # reference data, the same for all parameter sets
        black = "#000000"
        plot.curve(
            x_ref,
            shown(y_ref),
            line="none",
            marker="square",
            size=8,
            color=black,
            label=self._data_label(k),
            legend="data",
        )
        errors = self._reference_errors(k)
        if errors is not None:
            lower, upper = errors
            if log:
                # a lower end at or below zero has no place on the axis, the
                # bar goes to the bottom of the axis instead
                floor = bottom if bottom is not None else 0.0
                lower = np.where(
                    y_ref - lower > 0.0, lower, np.maximum(y_ref - floor, 0.0)
                )
            self._draw_errors(
                plot,
                x_ref,
                shown(y_ref),
                lower,
                upper,
                along="y",
                color=black,
                legend="data",
            )

        for ks, pset in enumerate(self.parameter_sets):
            data = res_data[pset.sid]
            color = self.color(pset)
            marker = self.set_marker(pset)
            if scalar:
                # the one simulated value, next to every row of the data
                plot.curve(
                    x_ref + self._simulation_offset(ks),
                    shown(np.full(x_ref.size, float(data["y_obs"][k][0]))),
                    marker=marker,
                    width=1.5,
                    color=color,
                    label=pset.sid,
                    legend=pset.sid,
                )
                continue
            # the points of a scan are the simulated values over a dimension,
            # a dimension of one value has no line
            plot.curve(
                data["x_obs"][k],
                shown(data["y_obs"][k]),
                marker=marker if dimension else "none",
                width=1.5,
                color=color,
                label=pset.sid,
                legend=pset.sid,
            )
            if not dimension:
                plot.curve(
                    x_ref,
                    shown(data["y_obsip"][k]),
                    line="none",
                    marker=marker,
                    size=7,
                    marker_fill="none",
                    marker_edge_width=1.5,
                    color=color,
                    legend=pset.sid,
                )

    def mapping_text(self, k: int) -> str:
        """Get the metrics of mapping `k`, one line per parameter set.

        The number of data points, the fold errors `GMFE` and `AFE`, the
        percentages of the points within twofold and within two standard
        deviations of the data, see `sbmlsim.fit.evaluation`, and the cost of
        the mapping with its share of the cost of the training data, which
        only a training mapping has.

        Args:
            k: index of the mapping.

        Returns:
            The text of the box, the lines are prefixed with the id of the set
            when several sets are compared.
        """
        training = self.problem.mapping_kinds[k] is MappingKind.TRAINING
        training_indices = [
            i
            for i, kind in enumerate(self.problem.mapping_kinds)
            if kind is MappingKind.TRAINING
        ]
        lines: list[str] = []
        for pset in self.parameter_sets:
            lines.extend(self._box_id(pset))
            dp = self._of_mapping(self.points(pset), k)
            metrics = criteria_metrics(dp)
            costs = self.residual_data(pset)["cost"]
            cost = float(costs[k])
            total = float(sum(costs[i] for i in training_indices))
            share = (
                f" ({_box_value(100.0 * cost / total, '.1f', '%')})"
                if training and total > 0
                else ""
            )
            lines.append(
                f"n = {len(dp)}, GMFE = {_box_value(metrics['GMFE'], '.3g')}, "
                f"AFE = {_box_value(metrics['AFE'], '.3g')}\n"
                f"2-fold = {_box_value(metrics['within_2fold'], '.1f', '%')}, "
                f"2 SD = {_box_value(metrics['within_2sd'], '.1f', '%')}, "
                f"cost = {_box_value(cost, '.3g')}{share}"
            )
        return "\n".join(lines)

    def mapping_subtitle(self, k: int) -> str:
        """Get the metadata of mapping `k`, the values which are reported.

        A value which is not reported (`NR`) or which is none (`NONE`, e.g. no
        coadministration) is left out.
        """
        values = [
            value
            for value in self.problem.mapping_metadata[k].values()
            if value not in ("", "NR", "NONE")
        ]
        return ", ".join(values)

    # --------------------------------------------------------------------
    # tables
    # --------------------------------------------------------------------
    def _cost_df(self, pset: ParameterSet) -> pd.DataFrame:
        """Calculate the cost of every mapping for a parameter set."""
        res_data = self.residual_data(pset)
        data = [
            {
                "id": f"{self.problem.experiment_keys[k]}_{self.problem.mapping_keys[k]}",
                "experiment": self.problem.experiment_keys[k],
                "mapping": self.problem.mapping_keys[k],
                "cost": res_data["cost"][k],
                "weight_curve": res_data["weights_curve"][k],
            }
            for k, _ in enumerate(self.problem.mapping_keys)
        ]
        return pd.DataFrame(
            data,
            columns=pd.Index(["id", "experiment", "mapping", "cost", "weight_curve"]),
        )

    def _datapoints_df(self, pset: ParameterSet) -> pd.DataFrame:
        """Calculate the data points and their residuals for a parameter set."""
        res_data = self.residual_data(pset)

        data = []
        for k, _ in enumerate(self.problem.mapping_keys):
            experiment = self.problem.experiment_keys[k]
            mapping = self.problem.mapping_keys[k]

            x_ref = self.problem.x_references[k]
            y_ref_err = self.problem.y_errors[k]
            y_ref_err_type = self.problem.y_errors_type[k]
            y_ref = self.problem.y_references[k]
            y_obs = res_data["y_obsip"][k]
            residuals = res_data["residuals"][k]
            for ix in range(len(y_obs)):
                data.append(
                    {
                        "experiment": experiment,
                        "mapping": mapping,
                        "x_ref": x_ref[ix],
                        "y_ref": y_ref[ix],
                        "y_ref_err": np.nan if not y_ref_err_type else y_ref_err[ix],
                        "y_obs": y_obs[ix],
                        "residual": residuals[ix],
                    }
                )

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "experiment",
                    "mapping",
                    "x_ref",
                    "y_ref",
                    "y_ref_err",
                    "y_obs",
                    "residual",
                ]
            ),
        )

    # --------------------------------------------------------------------
    # comparison of the parameter sets
    # --------------------------------------------------------------------
    def _panels(self, title: str, rows: int = 1) -> tuple[Chart, list[str]]:
        """Create a chart for a panel per subset of the data points and row.

        The figure covers a full row of the report, so the panels are large
        enough for the points, the band and the box with the metrics. A row
        has one legend right of its panels, whose entries the renderer merges
        over the panels; it widens the figure, the panels keep their size.

        Args:
            title: the title of the figure, shown with `show_titles`.
            rows: the rows of panels.

        Returns:
            The chart and the kinds of the columns.
        """
        kinds = self.point_kinds()
        chart = Chart(
            PANEL_SIZE * len(kinds),
            PANEL_SIZE * rows,
            title=title if self.show_titles else None,
            rows=rows,
            columns=len(kinds),
            wspace=0.25,
            hspace=0.35,
            legend_scope="row",
            legend_position="outside",
        )
        return chart, kinds

    def _legend_order(self, points: Mapping[str, pd.DataFrame]) -> list[str]:
        """Get the legend groups of the points of a figure, in their order.

        The points of the studies, or of the values of `color_by`, come first
        and the lines, i.e. the band and the lines of twofold, last, also when
        a study has points only in the last panel, which labels the lines.
        """
        column = self.color_by if self.color_by is not None else "experiment"
        order: list[str] = []
        for pset in self.parameter_sets:
            df = points[pset.sid]
            for key in dict.fromkeys(np.asarray(df[column], dtype=object)):
                order.append(self._point_label(key, pset))
        return list(dict.fromkeys(order))

    def _band_color(self, pset: ParameterSet) -> str:
        """Get the color of the agreement band of a parameter set.

        The points carry the color of their study, so the band is neutral and
        only takes a color when several sets are compared.
        """
        return "#000000" if len(self.parameter_sets) == 1 else self.color(pset)

    def _band_labels(self, pset: ParameterSet) -> tuple[str, str, str]:
        """Get the legend entries of the identity, the bias and the limits."""
        bias, half = self.agreement(pset)
        prefix = f"{pset.sid} " if len(self.parameter_sets) > 1 else ""
        return (
            "prediction = measurement",
            f"{prefix}bias {10**bias:.2f}x",
            f"{prefix}LoA {10 ** (bias - half):.2f}-{10 ** (bias + half):.2f}x",
        )

    def panel_metrics(self, kind: str, plot: str) -> str:
        """Get the key metrics of a panel, one line per parameter set.

        The goodness of fit shows how well the predictions describe the data
        of the kind: `R²`, the scale free `NRMSE` and the `RMSE_w` of the cost
        of `FitMetrics.summary`, the fold errors `GMFE` and `AFE` and the
        percentages of the points within twofold and within two standard
        deviations, see `sbmlsim.fit.evaluation`; the absolute RMSE is not shown, the data
        spans orders of magnitude and it only reads the largest curves. The
        Bland-Altman plot shows the agreement of the
        points of the panel itself: the bias and the SD of `log10(f(x)/y)` as
        fold factors and the share of the points inside the limits of
        agreement of the training data, i.e. inside the band of `agreement`.

        Args:
            kind: kind of fit mapping of the panel.
            plot: `goodness_of_fit` or `bland_altman`.

        Returns:
            The text of the box, the lines are prefixed with the id of the set
            when several sets are compared.
        """
        if plot not in ("goodness_of_fit", "bland_altman"):
            raise ValueError(f"Unknown plot '{plot}'.")
        return self._panel_text(
            {
                pset.sid: self._of_kind(self.points(pset), kind)
                for pset in self.parameter_sets
            },
            plot,
        )

    def _panel_text(self, points: Mapping[str, pd.DataFrame], plot: str) -> str:
        """Get the key metrics of the points of a panel, see `panel_metrics`.

        Args:
            points: the data points of the panel of every parameter set, by the
                id of the set, in the order of the sets.
            plot: `goodness_of_fit` or `bland_altman`.

        Returns:
            The text of the box, one line per set.

        Raises:
            ValueError: for an unknown plot.
        """
        lines: list[str] = []
        for pset in self.parameter_sets:
            lines.extend(self._box_id(pset))
            dp = points[pset.sid]
            if plot == "goodness_of_fit":
                metrics = criteria_metrics(dp)
                lines.append(
                    f"R² = {_box_value(r_squared(dp.DV, dp.IPRED), '.3f')}, "
                    f"NRMSE = {_box_value(rmse(dp.NRES), '.3g')}, "
                    f"RMSE_w = {_box_value(rmse(dp.IWRES), '.3g')}"
                )
                lines.append(
                    f"GMFE = {_box_value(metrics['GMFE'], '.3g')}, "
                    f"AFE = {_box_value(metrics['AFE'], '.3g')}, "
                    f"2-fold = {_box_value(metrics['within_2fold'], '.1f', '%')}, "
                    f"2 SD = {_box_value(metrics['within_2sd'], '.1f', '%')}"
                )
            elif plot == "bland_altman":
                _mean, difference, _mask = self._log_ratio(dp)
                if difference.size == 0:
                    lines.append("no ratio")
                    continue
                bias_training, half = self.agreement(pset)
                inside = np.mean(np.abs(difference - bias_training) <= half + 1e-12)
                bias = float(np.mean(difference))
                sd = (
                    _box_value(10 ** float(np.std(difference, ddof=1)), ".2f", "x")
                    if difference.size > 1
                    else "-"
                )
                lines.append(
                    f"bias = {10**bias:.2f}x, SD = {sd}, in LoA = {inside:.0%}"
                )
            else:
                raise ValueError(f"Unknown plot '{plot}'.")
        return "\n".join(lines)

    def _box_id(self, pset: ParameterSet) -> list[str]:
        """Get the line of the id of a set in a box of metrics, none for one set.

        When several sets are compared the metrics of a set follow a line of
        its id, shortened to `BOX_ID_LENGTH` characters: a box is inside its
        panel, and a long id in front of the metrics would take the width of
        the panel; the legend has the whole id.
        """
        if len(self.parameter_sets) == 1:
            return []
        sid = pset.sid
        if len(sid) > BOX_ID_LENGTH:
            sid = f"{sid[: BOX_ID_LENGTH - 3]}..."
        return [f"{sid}:"]

    def _value_color(self, value: Any) -> str:
        """Get the color of a value of the column `color_by`.

        The values in the order they appear in the data points of the
        reference set, so a value has the same color in every figure.
        """
        if self.color_by is None:
            raise ValueError("The report has no column to color by.")
        values = list(
            dict.fromkeys(
                str(v) for v in self.points(self.reference_set)[self.color_by]
            )
        )
        index = values.index(str(value)) if str(value) in values else len(values)
        # a value which is in no data point is a category of its own, so it
        # takes no color of a value which is
        return _hex(category_color(index, len(values) + 1))

    def _draw_points(
        self,
        plot: ChartPlot,
        dp: pd.DataFrame,
        pset: ParameterSet,
        x: np.ndarray,
        y: np.ndarray,
        point_labels: bool,
        xerr: np.ndarray | None = None,
        yerr: np.ndarray | None = None,
    ) -> None:
        """Draw the data points of a set in a panel of the agreement.

        The points of a study, or of a value of `color_by`, take its color and
        its legend entry; with `point_labels` false (the figure of a mapping)
        the points take the color of their set and no entry. With
        `error_bars` the errors are drawn, the columns of `xerr` and `yerr`
        per point, `NaN` for a point without a standard deviation.

        Args:
            plot: the panel.
            dp: the data points of the set which are drawn.
            pset: the parameter set.
            x: the x of every point.
            y: the y of every point.
            point_labels: label and color the points by their group.
            xerr: the lower and upper error of x of every point.
            yerr: the lower and upper error of y of every point.
        """
        column = self.color_by if self.color_by is not None else "experiment"
        keys = np.asarray(dp[column], dtype=object)
        for key in dict.fromkeys(keys):
            of_key = keys == key
            if not point_labels:
                color = self.color(pset)
            elif self.color_by is not None:
                color = self._value_color(key)
            else:
                color = self.study_color(str(key))
            label = self._point_label(key, pset) if point_labels else None
            plot.curve(
                x[of_key],
                y[of_key],
                line="none",
                marker=self.set_marker(pset),
                size=POINT_SIZE,
                marker_edge="#000000",
                marker_edge_width=1.0,
                opacity=POINT_OPACITY,
                color=color,
                label=label,
                legend=label,
            )
            if not self.error_bars:
                continue
            errors: tuple[tuple[np.ndarray | None, Literal["x", "y"]], ...] = (
                (xerr, "x"),
                (yerr, "y"),
            )
            for err, along in errors:
                if err is None:
                    continue
                part = np.asarray(err, dtype=float)[:, of_key]
                self._draw_errors(
                    plot,
                    x[of_key],
                    y[of_key],
                    part[0],
                    part[1],
                    along=along,
                    color=color,
                    legend=label,
                    opacity=0.6,
                )

    def _draw_subset_agreement(
        self,
        plot: ChartPlot,
        dp: pd.DataFrame,
        pset: ParameterSet,
        x_limits: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Draw the agreement of the points of a panel with its confidence.

        The bias and the limits of agreement of the points of the panel,
        thinner than the band of the training data, and the 95% confidence
        intervals of the bias and of the limits as shaded bands, see
        `sbmlsim.fit.evaluation.subset_agreement`; a panel of fewer than
        `MIN_POINTS_SUBSET_AGREEMENT` points has none. The lines are labelled
        in every panel which has them, the legend of a figure has an entry
        once; they are violet for a single set, which the band of the training
        data has in black, and in the color of a set when sets are compared.

        Args:
            plot: the panel.
            dp: the data points of the panel which have a ratio.
            pset: the parameter set.
            x_limits: the limits of the x axis, which the bands span.

        Returns:
            The lowest and the highest value drawn, `None` without an agreement.
        """
        agreement = subset_agreement(dp)
        if agreement is None:
            return None
        bias, half, bias_ci, limit_ci = agreement
        single = len(self.parameter_sets) == 1
        color = SUBSET_AGREEMENT_COLOR if single else self.color(pset)
        prefix = "" if single else f"{pset.sid} "
        edge = list(x_limits)
        plot.area(
            edge,
            [bias - bias_ci] * 2,
            [bias + bias_ci] * 2,
            color=color,
            fill_opacity=0.15,
        )
        for limit in (bias - half, bias + half):
            plot.area(
                edge,
                [limit - limit_ci] * 2,
                [limit + limit_ci] * 2,
                color=color,
                fill_opacity=0.1,
            )
        bias_label = f"{prefix}this panel bias"
        plot.rule(
            bias,
            line="dashdot",
            width=0.9,
            color=color,
            opacity=0.9,
            label=bias_label,
            legend=bias_label,
        )
        limits_label = f"{prefix}this panel LoA"
        for limit in (bias - half, bias + half):
            plot.rule(
                limit,
                line="dashed",
                width=0.9,
                color=color,
                opacity=0.9,
                label=limits_label,
                legend=limits_label,
            )
        return bias - half - limit_ci, bias + half + limit_ci

    @staticmethod
    def _goodness_axes(
        limits: tuple[float, float], y_label: bool
    ) -> tuple[ChartAxis, ChartAxis]:
        """Get the logarithmic axes of a panel of the goodness of fit.

        Args:
            limits: the limits of both axes.
            y_label: show the label of the y axis, which the first panel of a
                row does.
        """
        low, high = limits
        x = ChartAxis(label=GOF_X_LABEL, scale="log", min=low, max=high, grid=True)
        y = ChartAxis(
            label=GOF_Y_LABEL,
            scale="log",
            min=low,
            max=high,
            grid=True,
            label_visible=y_label,
        )
        return x, y

    def _draw_goodness_of_fit(
        self,
        plot: ChartPlot,
        points: Mapping[str, pd.DataFrame],
        limits: tuple[float, float],
        label_band: bool,
        point_labels: bool = True,
    ) -> None:
        """Draw a panel of the goodness of fit: the prediction over the data.

        The band of `agreement` of every set, the identity, the lines of
        twofold and the points by study and set, on the logarithmic axes of
        `_goodness_axes` within `limits`. A panel of the summary figure, of a
        grouping and of the figure of a mapping is drawn by it, so a point
        reads the same in all.

        Args:
            plot: the panel, of the axes of `_goodness_axes`.
            points: the data points of every parameter set, by the id of the
                set.
            limits: the limits of both axes.
            label_band: label the band and the lines, which the last panel
                of a row does: the renderer lists the entries of the panels of
                a row in the order of the panels, so the lines come after the
                points of every panel.
            point_labels: label and color the points by their study; the
                figure of a mapping, which has one study, colors them by their
                set, as its curves, and does not label them.
        """
        min_dp, max_dp = limits
        edge = np.array([min_dp, max_dp])
        for pset in self.parameter_sets:
            bias, half = self.agreement(pset)
            color = self._band_color(pset)
            _identity, bias_label, limits_label = self._band_labels(pset)
            plot.area(
                edge,
                edge * 10 ** (bias - half),
                edge * 10 ** (bias + half),
                color=color,
                fill_opacity=BAND_ALPHA,
            )
            plot.curve(
                edge,
                edge * 10**bias,
                color=color,
                label=bias_label if label_band else None,
                legend=bias_label if label_band else None,
                **BIAS_STYLE,
            )
            for limit in (bias - half, bias + half):
                plot.curve(
                    edge,
                    edge * 10**limit,
                    color=color,
                    label=limits_label if label_band else None,
                    legend=limits_label if label_band else None,
                    **LIMITS_STYLE,
                )
        identity = self._band_labels(self.reference_set)[0]
        plot.curve(
            edge,
            edge,
            color="#000000",
            label=identity if label_band else None,
            legend=identity if label_band else None,
            **IDENTITY_STYLE,
        )
        for factor in (2.0, 0.5):
            plot.curve(
                edge,
                edge * factor,
                label="2-fold" if label_band and factor > 1 else None,
                legend="2-fold" if label_band else None,
                **TWOFOLD_STYLE,
            )

        for pset in self.parameter_sets:
            # the points with a ratio, the others have no place on the axes
            dp = points[pset.sid][self._log_ratio(points[pset.sid])[2]]
            sd = np.asarray(dp.DV_SD, dtype=float)
            self._draw_points(
                plot,
                dp,
                pset,
                np.asarray(dp.DV, dtype=float),
                np.asarray(dp.IPRED, dtype=float),
                point_labels,
                xerr=np.vstack([sd, sd]),
            )

    def plot_goodness_of_fit(self, path: Path) -> Chart:
        """Plot the predicted against the measured data points, per kind.

        One panel per kind of fit mapping, i.e. the training data, the
        validation data and the outliers separately. The points scatter around
        the identity line when the model describes the data.

        The band is the agreement of `agreement`, i.e. the bias and the limits
        `bias ± 1.96 SD` of the training data, which on logarithmic axes are
        lines parallel to the identity: a point inside the band is a prediction
        the fit agrees to. It is the same band the Bland-Altman plot draws, in
        the same styles, so the two figures are read the same way.

        The dark red dotted lines are twofold, `f = 2y` and `f = y/2`: the
        percentage of the points between them is `within_2fold` of the
        metrics, which the box of a panel states, and the caption states the
        points with their denominator, see `twofold_statement`. The title of a
        panel states the points drawn, see `_panel_title`.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        chart, kinds = self._panels("Goodness of fit")
        order = self._legend_order(points)
        limits = self._log_limits(
            *[dp.DV for dp in points.values()], *[dp.IPRED for dp in points.values()]
        )
        boxes: list[tuple[ChartPlot, str]] = []
        for column, kind in enumerate(kinds):
            of_kind = {
                pset.sid: self._of_kind(points[pset.sid], kind)
                for pset in self.parameter_sets
            }
            x, y = self._goodness_axes(limits, y_label=column == 0)
            plot = chart.plot(
                0,
                column,
                x=x,
                y=y,
                title=self._panel_title(
                    kind, [points[pset.sid] for pset in self.parameter_sets]
                ),
                legend_order=order,
            )
            self._draw_goodness_of_fit(
                plot, of_kind, limits, label_band=column == len(kinds) - 1
            )
            boxes.append((plot, self._panel_text(of_kind, "goodness_of_fit")))
        self._metrics_boxes(chart, boxes)
        self._save_chart(chart, path)
        return chart

    @staticmethod
    def _log_ratio_errors(dp: pd.DataFrame, difference: np.ndarray) -> np.ndarray:
        """Get the error of the log ratio of the points from the SD of the data.

        `log10(f/(y ± SD))` around `log10(f/y)`: a larger data value is a
        smaller ratio. The upper end of a point whose SD is not smaller than
        its value has no logarithm and is drawn one decade above the point.

        Returns:
            The lower and the upper error of every point, `NaN` without an SD.
        """
        dv = np.asarray(dp.DV, dtype=float)
        sd = np.asarray(dp.DV_SD, dtype=float)
        ipred = np.asarray(dp.IPRED, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lower = difference - np.log10(ipred / (dv + sd))
            below = dv - sd
            upper = np.where(
                below > 0, np.log10(ipred / np.where(below > 0, below, 1.0)), np.nan
            )
            upper = np.where(below > 0, upper - difference, 1.0)
        upper = np.where(np.isfinite(sd), upper, np.nan)
        return np.vstack([lower, upper])

    @classmethod
    def _panel_title(cls, kind: str, points: Sequence[pd.DataFrame]) -> str:
        """Get the title of a panel of the data points, with the points drawn.

        The axes of the goodness of fit and of the Bland-Altman plot are
        logarithmic, so a data point which is zero or negative, in its value or
        its prediction, has no ratio and is not drawn. The title states the
        points drawn, of the set with the most of them, and of how many points
        of the reference set, e.g. `training (n=523 of 530)`; these are the
        points `within_2fold` counts.

        Args:
            kind: kind of fit mapping of the panel.
            points: the data points of every parameter set, see `points`, the
                reference set first.

        Returns:
            The title of the panel.
        """
        n_points = len(cls._of_kind(points[0], kind))
        n_shown = max(
            int(np.sum(cls._log_ratio(cls._of_kind(df, kind))[2])) for df in points
        )
        label = f"n={n_shown}" if n_shown == n_points else f"n={n_shown} of {n_points}"
        return f"{kind} ({label})"

    @staticmethod
    def _log_ratio(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the ratio of prediction and measurement of the data points.

        Args:
            df: data points, see `points`.

        Returns:
            The geometric mean of measurement and prediction, the difference of
            their logarithms and the mask of the points a ratio is defined for,
            i.e. the points where both values are positive.
        """
        dv = np.asarray(df.DV, dtype=float)
        ipred = np.asarray(df.IPRED, dtype=float)
        mask = np.isfinite(dv) & np.isfinite(ipred) & (dv > 0) & (ipred > 0)
        dv, ipred = dv[mask], ipred[mask]
        return np.sqrt(dv * ipred), np.log10(ipred / dv), mask

    def agreement(self, pset: ParameterSet) -> tuple[float, float]:
        """Get the bias and the half width of the limits of agreement.

        They are calculated on the training data alone, i.e. on the data the
        parameters were fitted on, and the Bland-Altman plot draws them in
        every panel: the limits are what the fit agrees to, and the validation
        data and the outliers are read against them. Calculating them per panel
        would give every subset its own reference and the panels could not be
        compared; pooling all data points would let the outliers, which are
        dropped exactly because they are far away, widen the limits.

        Args:
            pset: parameter set of the report.

        Returns:
            The bias `mean(log10(f(x)/y))` and `1.96 * SD` of it, both in
            decades. `10**bias` and `10**(bias ± half)` are the fold factors.
        """
        df = self.points(pset)
        training = df[df.kind == MappingKind.TRAINING.value]
        # a report of a problem without training data falls back to everything
        _mean, difference, _mask = self._log_ratio(training if len(training) else df)
        if difference.size == 0:
            return 0.0, 0.0
        half = 1.96 * float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0
        return float(np.mean(difference)), half

    def _altman_x_limits(
        self, points: Mapping[str, pd.DataFrame]
    ) -> tuple[float, float]:
        """Get the limits of the geometric mean of a panel of the Bland-Altman plot."""
        means = [self._log_ratio(dp)[0] for dp in points.values()]
        return self._log_limits(*means, factor=1.5)

    def _altman_axes(
        self,
        points: Mapping[str, pd.DataFrame],
        own_agreement: bool,
        y_label: bool,
    ) -> tuple[ChartAxis, ChartAxis]:
        """Get the axes of a panel of the Bland-Altman plot.

        The x axis is the logarithmic geometric mean of the points. The y axis
        shows the points, the lines of twofold and the agreement of the panel,
        not the error bars, which may reach far beyond the points, e.g. a
        standard deviation as large as the value; and the limits of agreement
        of the training data, which are drawn in every panel, so a panel whose
        own points stay well inside them does not crop them.

        Args:
            points: the data points of every parameter set, by the id of the
                set.
            own_agreement: whether the panel draws the agreement of its points.
            y_label: show the label of the y axis, which the first panel of a
                row does.
        """
        shown_y: list[float] = [np.log10(2.0), -np.log10(2.0)]
        for pset in self.parameter_sets:
            dp = points[pset.sid]
            _mean, difference, mask = self._log_ratio(dp)
            if difference.size == 0:
                continue
            shown_y.extend(difference.tolist())
            if own_agreement:
                agreement = subset_agreement(dp[mask])
                if agreement is not None:
                    bias, half, _bias_ci, limit_ci = agreement
                    shown_y.extend([bias - half - limit_ci, bias + half + limit_ci])
        low, high = min(shown_y), max(shown_y)
        pad = 0.08 * (high - low)
        low, high = low - pad, high + pad
        bounds = [
            bias + sign * half
            for bias, half in (self.agreement(pset) for pset in self.parameter_sets)
            for sign in (-1, 1)
        ]
        margin = 0.05 * max(high - low, max(bounds) - min(bounds), 1e-6)
        low, high = min(low, min(bounds) - margin), max(high, max(bounds) + margin)
        x_low, x_high = self._altman_x_limits(points)
        x = ChartAxis(
            label=ALTMAN_X_LABEL, scale="log", min=x_low, max=x_high, grid=True
        )
        y = ChartAxis(
            label=ALTMAN_Y_LABEL, min=low, max=high, grid=True, label_visible=y_label
        )
        return x, y

    def _draw_bland_altman(
        self,
        plot: ChartPlot,
        points: Mapping[str, pd.DataFrame],
        label_band: bool,
        point_labels: bool = True,
        own_agreement: bool = True,
    ) -> None:
        """Draw a panel of the Bland-Altman plot: the log ratio over the geometric mean.

        The points by study and set, the band of `agreement` of every set, the
        line of no difference and the lines of twofold, see
        `plot_bland_altman`, and the agreement of the points of the panel
        itself, see `_draw_subset_agreement`. A panel of the summary figure, of
        a grouping and of the figure of a mapping is drawn by it.

        Args:
            plot: the panel, of the axes of `_altman_axes`.
            points: the data points of every parameter set, by the id of the
                set.
            label_band: label the band and the lines, which the last panel
                of a row does: the renderer lists the entries of the panels of
                a row in the order of the panels, so the lines come after the
                points of every panel.
            point_labels: label the points by their study, see
                `_draw_goodness_of_fit`.
            own_agreement: draw the agreement of the points of the panel. The
                panel of the training data of the summary figure has the
                agreement of the band, the reference, and the outliers were
                dropped for not agreeing, so both leave it out.
        """
        x_limits = self._altman_x_limits(points)
        edge = list(x_limits)
        # the agreement of the training data, the same lines in every panel.
        # The points carry the color of their study, so the lines are neutral
        # and only take a color when several sets are compared; the same band
        # as the goodness of fit, here as horizontal lines
        for pset in self.parameter_sets:
            bias, half = self.agreement(pset)
            color = self._band_color(pset)
            _identity, bias_label, limits_label = self._band_labels(pset)
            plot.area(
                edge,
                [bias - half] * 2,
                [bias + half] * 2,
                color=color,
                fill_opacity=BAND_ALPHA,
            )
            plot.rule(
                bias,
                color=color,
                label=bias_label if label_band else None,
                legend=bias_label if label_band else None,
                **BIAS_STYLE,
            )
            for limit in (bias - half, bias + half):
                plot.rule(
                    limit,
                    color=color,
                    label=limits_label if label_band else None,
                    legend=limits_label if label_band else None,
                    **LIMITS_STYLE,
                )

        # the identity of the goodness of fit is no difference here, its lines
        # of twofold are the differences of `±log10(2)`
        identity = self._band_labels(self.reference_set)[0]
        plot.rule(
            0.0,
            color="#000000",
            label=identity if label_band else None,
            legend=identity if label_band else None,
            **IDENTITY_STYLE,
        )
        for sign in (1.0, -1.0):
            plot.rule(
                sign * np.log10(2.0),
                label="2-fold" if label_band and sign > 0 else None,
                legend="2-fold" if label_band else None,
                **TWOFOLD_STYLE,
            )

        for pset in self.parameter_sets:
            dp = points[pset.sid]
            mean, difference, mask = self._log_ratio(dp)
            if difference.size == 0:
                continue
            shown = dp[mask]
            if own_agreement:
                self._draw_subset_agreement(plot, shown, pset, x_limits)
            self._draw_points(
                plot,
                shown,
                pset,
                mean,
                difference,
                point_labels,
                yerr=self._log_ratio_errors(shown, difference),
            )

    def plot_bland_altman(self, path: Path) -> Chart:
        """Plot the agreement of prediction and measurement, per kind.

        A Bland-Altman plot of the ratio: the difference of the logarithms,
        `log10(f(x)/y)`, over the geometric mean of the two. The data of a fit
        spans orders of magnitude, so the agreement is multiplicative and the
        limits are read as fold factors.

        The bias and the limits of agreement `bias ± 1.96 SD` are those of
        `agreement`, i.e. of the training data, and they are the same in every
        panel, so the validation data and the outliers are read against what
        the fit agrees to. Every panel shows the limits even when its points
        are further out. It is the same band the goodness of fit draws, in the
        same styles: the identity there is no difference here.

        Data points which are zero or negative have no logarithm and are left
        out, i.e. the plot shows the points a ratio is defined for.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        chart, kinds = self._panels("Bland-Altman")
        order = self._legend_order(points)
        boxes: list[tuple[ChartPlot, str]] = []
        for column, kind in enumerate(kinds):
            of_kind = {
                pset.sid: self._of_kind(points[pset.sid], kind)
                for pset in self.parameter_sets
            }
            own_agreement = kind == MappingKind.VALIDATION.value
            x, y = self._altman_axes(of_kind, own_agreement, y_label=column == 0)
            plot = chart.plot(
                0,
                column,
                x=x,
                y=y,
                title=self._panel_title(
                    kind, [points[pset.sid] for pset in self.parameter_sets]
                ),
                legend_order=order,
            )
            self._draw_bland_altman(
                plot,
                of_kind,
                label_band=column == len(kinds) - 1,
                own_agreement=own_agreement,
            )
            boxes.append((plot, self._panel_text(of_kind, "bland_altman")))
        self._metrics_boxes(chart, boxes)
        self._save_chart(chart, path)
        return chart

    def plot_grouping(self, grouping: Grouping, plots_dir: Path) -> None:
        """Plot the figures of a grouping of the data points.

        `goodness_of_fit_<name>` and `bland_altman_<name>`: a row per group
        and a column per kind of fit mapping, the panels drawn as the summary
        figures draw theirs, with the band of the agreement of the training
        data and the box of the points of the panel; a panel without points
        says so. A figure has at most `MAX_GROUP_ROWS` rows, more groups are
        drawn on more pages, `goodness_of_fit_<name>_2` and so on, see
        `grouping_page`. `criteria_<name>`: the percentage of the points
        within twofold and within two standard deviations of the data per
        group and kind, of the reference set.

        Args:
            grouping: the grouping.
            plots_dir: the directory of the figures.
        """
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        groups = {
            sid: grouping.groups(df, self._fit_mapping) for sid, df in points.items()
        }
        labels = [label for label, _ in groups[self.reference_set.sid]]
        order = self._legend_order(points)
        limits = self._log_limits(
            *[dp.DV for dp in points.values()], *[dp.IPRED for dp in points.values()]
        )
        pages = [
            list(range(start, min(start + MAX_GROUP_ROWS, len(labels))))
            for start in range(0, max(len(labels), 1), MAX_GROUP_ROWS)
        ]
        for plot in ("goodness_of_fit", "bland_altman"):
            title = "Goodness of fit" if plot == "goodness_of_fit" else "Bland-Altman"
            for number, rows in enumerate(pages, start=1):
                page = f", {number} of {len(pages)}" if len(pages) > 1 else ""
                chart, kinds = self._panels(
                    f"{title} by {grouping.name}{page}", rows=max(len(rows), 1)
                )
                boxes: list[tuple[ChartPlot, str]] = []
                for r, g in enumerate(rows):
                    # the last panel of the row with points labels the band
                    drawn_kinds = [
                        kind
                        for kind in kinds
                        if any(
                            not self._of_kind(groups[sid][g][1], kind).empty
                            for sid in groups
                        )
                    ]
                    for c, kind in enumerate(kinds):
                        drawn = self._draw_group_panel(
                            chart,
                            r,
                            c,
                            plot,
                            labels[g],
                            kind,
                            groups,
                            g,
                            limits,
                            order,
                            label_band=bool(drawn_kinds) and kind == drawn_kinds[-1],
                        )
                        if drawn is not None:
                            boxes.append(drawn)
                self._metrics_boxes(chart, boxes)
                self._save_chart(
                    chart,
                    plots_dir
                    / f"{grouping_page(plot, grouping.name, number)}.{self.image_format}",
                )
        self._plot_criteria(grouping, plots_dir)

    def _draw_group_panel(
        self,
        chart: Chart,
        row: int,
        column: int,
        plot: str,
        label: str,
        kind: str,
        groups: Mapping[str, list[tuple[str, pd.DataFrame]]],
        g: int,
        limits: tuple[float, float],
        order: Sequence[str],
        label_band: bool,
    ) -> tuple[ChartPlot, str] | None:
        """Draw the panel of a group and a kind of a figure of a grouping.

        A panel without points says so. The last panel of a row with points
        labels the band, so the renderer lists the lines after the points of
        every panel of the row in the legend of the row.

        Args:
            chart: the figure.
            row: the row of the panel.
            column: the column of the panel.
            plot: `goodness_of_fit` or `bland_altman`.
            label: the label of the group.
            kind: the kind of the column.
            groups: the groups of every parameter set.
            g: the index of the group.
            limits: the limits of the axes of the goodness of fit.
            order: the legend groups of the points of the figure.
            label_band: label the band and the lines.

        Returns:
            The panel and the text of its box of metrics, `None` for a panel
            without points.
        """
        of_panel = {sid: self._of_kind(groups[sid][g][1], kind) for sid in groups}
        title = f"{label}: " + self._panel_title(kind, list(of_panel.values()))
        if all(df.empty for df in of_panel.values()):
            empty = ChartAxis(min=0.0, max=1.0, ticks_visible=False)
            panel = chart.plot(row, column, x=empty, y=empty, title=title)
            panel.text(0.5, 0.5, "no data", anchor="middle", color="#000000")
            return None
        own_agreement = kind != MappingKind.OUTLIER.value
        if plot == "goodness_of_fit":
            x, y = self._goodness_axes(limits, y_label=column == 0)
        else:
            x, y = self._altman_axes(of_panel, own_agreement, y_label=column == 0)
        panel = chart.plot(row, column, x=x, y=y, title=title, legend_order=order)
        if plot == "goodness_of_fit":
            self._draw_goodness_of_fit(panel, of_panel, limits, label_band=label_band)
        else:
            self._draw_bland_altman(
                panel, of_panel, label_band=label_band, own_agreement=own_agreement
            )
        return panel, self._panel_text(of_panel, plot)

    def _plot_criteria(self, grouping: Grouping, plots_dir: Path) -> Chart:
        """Plot the criteria of the groups of a grouping, see `plot_grouping`."""
        points = self.points(self.reference_set)
        metrics = group_metrics(points, grouping, self._fit_mapping)
        # every group, also one without points, e.g. an empty subset
        labels = [label for label, _ in grouping.groups(points, self._fit_mapping)]
        kinds = [kind for kind in self.point_kinds() if kind in set(metrics.kind)]
        criteria = (("within_2fold", "2-fold"), ("within_2sd", "2 SD"))
        chart = Chart(
            360.0 * len(criteria),
            72.0 + 36.0 * len(labels) * max(len(kinds), 1),
            title=f"Criteria by {grouping.name}" if self.show_titles else None,
            columns=len(criteria),
            legend_scope="row",
            legend_position="outside",
        )
        width = 0.8 / max(len(kinds), 1)
        y = np.arange(len(labels))
        for column_index, (column, name) in enumerate(criteria):
            plot = chart.plot(
                0,
                column_index,
                x=ChartAxis(label=f"within {name} [%]", min=0.0, max=110.0, grid=True),
                y=ChartAxis(
                    scale="category",
                    categories=labels,
                    reverse=True,
                    ticks_visible=column_index == 0,
                ),
            )
            for j, kind in enumerate(kinds):
                of_kind = metrics[metrics.kind == kind].set_index("group")
                values = np.array(
                    [
                        float(of_kind.loc[label, column])
                        if label in of_kind.index
                        else np.nan
                        for label in labels
                    ]
                )
                positions = y + (j - (len(kinds) - 1) / 2) * width
                plot.bar(
                    positions,
                    np.nan_to_num(values),
                    horizontal=True,
                    bar_width=width,
                    color=normalize_color(KIND_COLORS.get(kind, "gray")),
                    label=kind,
                    legend=kind,
                )
                # the value at every bar, `-` where the criterion applies to no
                # point, so that 0% and no value read differently
                for position, value, label in zip(
                    positions, values, labels, strict=True
                ):
                    if label not in of_kind.index:
                        continue
                    plot.text(
                        (value if np.isfinite(value) else 0.0) + 1.0,
                        position,
                        f"{value:.0f}%" if np.isfinite(value) else "-",
                        color="#000000",
                    )
            for position, label in zip(y, labels, strict=True):
                if label not in set(metrics.group):
                    plot.text(1.0, position, "no data", color="#000000")
        self._save_chart(
            chart, plots_dir / f"criteria_{grouping.name}.{self.image_format}"
        )
        return chart

    def _row_labels(self) -> list[str]:
        """Get the labels of the rows of the figures with a row per fit mapping."""
        return [
            f"{experiment}|{mapping}"
            for experiment, mapping in zip(
                self.problem.experiment_keys, self.problem.mapping_keys, strict=True
            )
        ]

    def plot_cost_bar(self, path: Path) -> Chart:
        """Plot the cost and the weight of every curve, per set.

        A curve is a row, in the order of the fit mappings from the top; the
        figure grows with the number of curves and the length of their labels,
        so that every label is readable, e.g. the 164 curves of a large fit.
        The weights of the curves do not depend on the parameters, they are
        the weights of the reference set.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        costs = {pset.sid: self._cost_df(pset) for pset in self.parameter_sets}
        costs_ref = costs[self.reference_set.sid]
        labels = self._row_labels()
        n_sets = len(self.parameter_sets)
        height = 0.8 / n_sets
        position = np.arange(len(labels))
        chart = Chart(
            *self._rows_size(labels, width=576.0, panels=360.0),
            title="Curve costs and weights" if self.show_titles else None,
            columns=3,
            wspace=0.1,
            fonts={"tick": ROW_FONT},
            legend_scope="row",
            legend_position="outside",
        )
        all_costs = np.concatenate(
            [np.asarray(df.cost, dtype=float) for df in costs.values()]
        )
        log = self._has_positive(all_costs)
        finite = all_costs[np.isfinite(all_costs)]
        cost_axis = (
            ChartAxis(label="Cost", scale="log", grid=True)
            if log
            # a cost is not negative, e.g. the zero cost of a perfect fit
            else ChartAxis(
                label="Cost",
                min=0.0,
                max=max(float(finite.max()) if finite.size else 0.0, 1.0) * 1.05,
                grid=True,
            )
        )
        rows = ChartAxis(scale="category", categories=labels, reverse=True)
        plot_cost = chart.plot(0, 0, column_span=2, x=cost_axis, y=rows)
        for ks, pset in enumerate(self.parameter_sets):
            cost = np.asarray(costs[pset.sid].cost, dtype=float)
            if log:
                cost = np.where(cost > 0.0, cost, np.nan)
            plot_cost.bar(
                position + (ks - (n_sets - 1) / 2) * height,
                cost,
                horizontal=True,
                bar_width=height,
                color=self.color(pset),
                fill_opacity=0.8,
                label=pset.sid,
                legend=pset.sid,
            )
        plot_weight = chart.plot(
            0,
            2,
            x=ChartAxis(label="Weight curve w_k", grid=True),
            y=ChartAxis(
                scale="category", categories=labels, reverse=True, ticks_visible=False
            ),
            legend=False,
        )
        plot_weight.bar(
            position,
            np.asarray(costs_ref.weight_curve, dtype=float),
            horizontal=True,
            color=normalize_color("tab:blue"),
            fill_opacity=0.8,
        )
        self._save_chart(chart, path)
        return chart

    @staticmethod
    def _rows_size(
        labels: Sequence[str], width: float, panels: float
    ) -> tuple[float, float]:
        """Get the size of a figure with a row per curve and its label.

        The figure grows with the number of curves and the length of their
        labels of `ROW_FONT` points, so that every label is readable, e.g.
        the 164 curves of a large fit.

        Args:
            labels: the labels of the rows.
            width: the smallest width in pixels, the height is at least 432.
            panels: the width in pixels of the panels next to the labels.

        Returns:
            The width and the height in pixels.
        """
        longest = max((len(label) for label in labels), default=0)
        return (
            max(width, panels + ROW_LABEL_CHAR * longest),
            max(432.0, 108.0 + ROW_HEIGHT * len(labels)),
        )

    def plot_residual_boxplot(self, path: Path) -> Chart:
        """Plot the distribution of the squared weighted residuals per curve.

        A curve is a row, in the order of the fit mappings from the top.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        labels = self._row_labels()
        n_sets = len(self.parameter_sets)
        residuals = {
            pset.sid: [
                np.power(np.asarray(values, dtype=float), 2)
                for values in self.residual_data(pset)["residuals_weighted"]
            ]
            for pset in self.parameter_sets
        }
        log = self._has_positive(*(r for rs in residuals.values() for r in rs))
        chart = Chart(
            *self._rows_size(labels, width=432.0, panels=252.0),
            title="Residual contribution" if self.show_titles else None,
            fonts={"tick": ROW_FONT},
            legend_position="outside",
        )
        plot = chart.plot(
            x=ChartAxis(
                label="Weighted residuals² (w_k · w_ik · (f(x_ik) - y_ik))²",
                scale="log" if log else "linear",
                grid=True,
            ),
            y=ChartAxis(scale="category", categories=labels, reverse=True),
        )
        for ks, pset in enumerate(self.parameter_sets):
            offset = (ks - (n_sets - 1) / 2) * 0.25
            x = np.concatenate(residuals[pset.sid]) if residuals[pset.sid] else []
            y = np.concatenate(
                [
                    np.full(len(values), k + offset)
                    for k, values in enumerate(residuals[pset.sid])
                ]
            )
            if log:
                x = np.where(np.asarray(x) > 0.0, x, np.nan)
            plot.curve(
                x,
                y,
                line="none",
                marker="square",
                size=4,
                marker_edge="#000000",
                marker_edge_width=0.5,
                color=self.color(pset),
                label=pset.sid,
                legend=pset.sid,
            )
        self._save_chart(chart, path)
        return chart

    def plot_cost_scatter(self, path: Path) -> Chart:
        """Plot the cost of every curve against the cost of the reference set.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        reference = self.reference_set
        costs_ref: pd.DataFrame = self._cost_df(reference)
        other_sets = list(self.parameter_sets)[1:]
        all_costs = [costs_ref.cost, *[self._cost_df(p).cost for p in other_sets]]
        min_cost, max_cost = self._log_limits(*all_costs, factor=2.0)
        chart = Chart(
            400.0,
            400.0,
            title=f"Cost against '{reference.sid}'" if self.show_titles else None,
        )
        plot = chart.plot(
            x=ChartAxis(
                label=f"Cost '{reference.sid}'",
                scale="log",
                min=min_cost,
                max=max_cost,
                grid=True,
            ),
            y=ChartAxis(
                label="Cost", scale="log", min=min_cost, max=max_cost, grid=True
            ),
        )
        plot.curve(
            [min_cost, max_cost],
            [min_cost, max_cost],
            line="dashed",
            width=1.5,
            color="#000000",
            label="no change",
            legend="no change",
        )
        x_ref = np.asarray(costs_ref.cost, dtype=float)
        for pset in other_sets:
            costs_x = np.asarray(self._cost_df(pset).cost, dtype=float)
            plot.curve(
                x_ref,
                costs_x,
                line="none",
                marker="circle",
                size=10,
                opacity=0.8,
                color=self.color(pset),
                label=pset.sid,
                legend=pset.sid,
            )
            for k, exp_key in enumerate(self.problem.experiment_keys):
                if x_ref[k] > 0 and costs_x[k] > 0:
                    plot.text(
                        x_ref[k], costs_x[k], exp_key, color="#000000", opacity=0.7
                    )
        self._save_chart(chart, path)
        return chart

    # --------------------------------------------------------------------
    # the optimization runs
    # --------------------------------------------------------------------
    @property
    def opt_result_required(self) -> OptimizationResult:
        """Result of the optimization, required for the plots of the runs.

        Raises:
            ValueError: if the report was created without a result.
        """
        if self.opt_result is None:
            raise ValueError(
                "The plots of the optimization runs require the "
                "'OptimizationResult' of a fit."
            )
        return self.opt_result

    def plot_waterfall(self, path: Path) -> Chart:
        """Create waterfall plot for the fit results.

        Plots the optimization runs sorted by cost.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        optres = self.opt_result_required
        chart = Chart(400.0, 400.0)
        plot = chart.plot(
            title="Waterfall plot" if self.show_titles else None,
            x=ChartAxis(label="Index (ordered optimizer run)"),
            y=ChartAxis(
                label="Offset cost value (relative to best start)", scale="log"
            ),
            legend=False,
        )
        costs = np.asarray(optres.df_fits.cost.values, dtype=float)
        plot.curve(
            np.arange(optres.size),
            1 + (costs - costs[0]),
            marker="circle",
            width=1.5,
            color="#000000",
        )
        self._save_chart(chart, path)
        return chart

    def plot_traces(self, path: Path) -> Chart:
        """Plot optimization traces.

        Optimization time course of costs, the final cost of every run as a
        black point.

        Args:
            path: the file of the figure, see `_save_chart`.

        Returns:
            The figure.
        """
        optres = self.opt_result_required
        traces = [
            np.asarray(optres.df_traces[optres.df_traces.run == run].cost, dtype=float)
            for run in range(optres.size)
        ]
        traces = [trace for trace in traces if trace.size]
        log = self._has_positive(*traces)
        chart = Chart(400.0, 400.0)
        plot = chart.plot(
            title="Optimization traces" if self.show_titles else None,
            x=ChartAxis(label="Optimization step"),
            y=ChartAxis(label="Cost", scale="log" if log else "linear"),
            legend=False,
        )
        for trace in traces:
            shown = np.where(trace > 0.0, trace, np.nan) if log else trace
            plot.curve(np.arange(trace.size), shown, width=1.5, opacity=0.8)
            plot.curve(
                [trace.size - 1],
                [shown[-1]],
                line="none",
                marker="circle",
                color="#000000",
                opacity=0.8,
            )
        self._save_chart(chart, path)
        return chart
