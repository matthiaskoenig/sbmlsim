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
"""

from __future__ import annotations

import datetime
import json
import logging
import webbrowser
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sbmlsim import __version__
from sbmlsim.fit import display
from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, plot_all
from sbmlsim.fit.metrics import FitMetrics
from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.plot.serialization_matplotlib import plt
from sbmlsim.report.templates import template_environment

logger = logging.getLogger(__name__)

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

#: markers of the parameter sets in the figures which color by study, so that
#: the sets are told apart where the color is taken
SET_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")

#: styles of the agreement band. The goodness of fit and the Bland-Altman plot
#: draw the same band, once as lines parallel to the identity and once as
#: horizontal lines, so the two figures are read the same way
IDENTITY_STYLE: dict[str, Any] = {"linestyle": "-", "linewidth": 1.5}
BIAS_STYLE: dict[str, Any] = {"linestyle": "-.", "linewidth": 1.2}
LIMITS_STYLE: dict[str, Any] = {"linestyle": "--", "linewidth": 1.2}

#: opacity of the filled area between the limits of agreement
BAND_ALPHA: float = 0.12

#: the figures which cover a full row of the report, i.e. the goodness of fit
#: and the Bland-Altman plot with their panel per kind and their metrics
WIDE_PLOTS: frozenset[str] = frozenset({"goodness_of_fit", "bland_altman"})

#: style of the box with the key metrics of a panel
METRICS_BOX: dict[str, Any] = {
    "boxstyle": "round,pad=0.4",
    "facecolor": "white",
    "edgecolor": "0.6",
    "alpha": 0.9,
}

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
        show_titles: bool = True,
        image_format: str = "svg",
    ) -> None:
        """Construct the report.

        The problem is initialized with the settings, which resolves the data of
        the fit mappings. A problem which is already initialized with the same
        settings is left alone.

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
            show_titles: add titles to the panels.
            image_format: format of the figures.
        """
        self.problem = problem
        self.settings = settings
        self.parameter_sets = ParameterSets.of(parameter_sets)
        self.opt_result = opt_result
        self.identifiability = identifiability
        self.fisher = fisher
        self.show_titles = show_titles
        self.image_format = image_format

        # resolves the data, a no-op if the problem is already initialized
        problem.initialize(settings)

        # residual data of the mappings, by parameter set
        self._res_data: dict[str, dict[str, list[Any]]] = {}

        # data points with their kind and predictions, by parameter set
        self._points: dict[str, pd.DataFrame] = {}

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
        problem.initialize(settings)

        sets: list[ParameterSet] = []
        if with_model:
            sets.append(problem.parameter_set_model())
        sets.extend(opt_result.parameter_sets(size=size))

        return FitReport(
            problem=problem,
            settings=settings,
            parameter_sets=ParameterSets(sets),
            opt_result=opt_result,
            **kwargs,
        )

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

    def mapping_title(self, k: int) -> str:
        """Get the title of the plots of a fit mapping.

        Data which is not fitted is marked, so that it is visible in the
        figures which curves the parameters were fitted on.
        """
        title = f"{self.problem.experiment_keys[k]} {self.problem.mapping_keys[k]}"
        kind = self.problem.mapping_kinds[k]
        if kind is not MappingKind.TRAINING:
            title = f"{title} [{kind.value}]"
        return title

    def color(self, pset: ParameterSet) -> str:
        """Get the color of a parameter set."""
        index = [p.sid for p in self.parameter_sets].index(pset.sid)
        return SET_COLORS[index % len(SET_COLORS)]

    def studies(self) -> list[str]:
        """Get the studies of the problem, i.e. its simulation experiments.

        In the order the fit mappings of the problem name them, so the color of
        a study does not depend on which mappings a figure shows.
        """
        return list(dict.fromkeys(self.problem.experiment_keys))

    def study_color(self, study: str) -> str:
        """Get the color of a study, the same in every figure of the report."""
        return STUDY_COLORS[self.studies().index(study) % len(STUDY_COLORS)]

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
        return pset.x(self.problem.pids)

    def metrics(self, pset: ParameterSet) -> FitMetrics:
        """Get the metrics of a parameter set on the problem.

        Args:
            pset: parameter set of the report.

        Returns:
            The metrics of the set, see `sbmlsim.fit.metrics`.
        """
        return FitMetrics(problem=self.problem, parameter_set=pset)

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

        Every evaluation simulates all fit mappings, so the results are cached
        for the plots and tables which use the same set.
        """
        if pset.sid not in self._res_data:
            self._res_data[pset.sid] = self.problem.residuals(  # ty: ignore[invalid-assignment]
                xlog=self.problem.to_scale(self.x(pset)), complete_data=True
            )
        return self._res_data[pset.sid]

    def points(self, pset: ParameterSet) -> pd.DataFrame:
        """Get the data points of a parameter set with their kind.

        The table of `FitMetrics.datapoints_df`, i.e. one row per data point
        with the measurement `DV`, the prediction `IPRED` and the `kind` of the
        fit mapping it belongs to. It is the source of the goodness of fit and
        the Bland-Altman plots and is cached, a data point costs a simulation.
        """
        if pset.sid not in self._points:
            self._points[pset.sid] = self.metrics(pset).datapoints_df()
        return self._points[pset.sid]

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
        mpl_parameters: dict[str, Any] | None = None,
    ) -> Path:
        """Create the complete report.

        Writes the figures, the text report, the parameter sets and the HTML
        report into `output_dir / name`.

        Args:
            output_dir: base directory of the reports.
            name: name of this report, the id of the optimization by default.
            show_report: open the HTML report in a web browser.
            mpl_parameters: additional matplotlib rc parameters of the figures.

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

        # the parameters are the input of a report, they are stored with it
        self.parameter_sets.to_json(path=results_dir / "parameters.json")

        # metrics of the parameter sets
        for df, name in [
            (self.metrics_df(), "metrics.tsv"),
            (self.metrics_mappings_df(), "metrics_mappings.tsv"),
            (self.datapoints_df(), "datapoints.tsv"),
        ]:
            df.to_csv(results_dir / name, sep="\t", index=False)
        if self.opt_result:
            self.opt_result.to_json(path=results_dir / "optimization_result.json")
            self.opt_result.to_tsv(path=results_dir / "optimization_result.tsv")
        if self.fisher:
            fisher_json = results_dir / "fisher.json"
            fisher_json.write_text(json.dumps(self.fisher.to_dict(), indent=2))
            self.fisher.summary_df.to_csv(
                results_dir / "fisher.tsv", sep="\t", index=False
            )
        if self.identifiability:
            self.identifiability.to_json(path=results_dir / "identifiability.json")
            self.identifiability.summary_df().to_csv(
                results_dir / "identifiability.tsv", sep="\t", index=False
            )

        self._write_text_report(path=results_dir / "report.txt")
        self._create_figures(plots_dir=plots_dir, mpl_parameters=mpl_parameters)
        self.html_report(path=results_dir / "index.html")

        report_path = results_dir / "index.html"
        display.link("report", report_path)
        if show_report:
            webbrowser.open(report_path.resolve().as_uri(), new=2)

        return results_dir

    def _write_text_report(self, path: Path) -> None:
        """Write the text report of the problem, the parameters and the runs."""
        info = [self.problem.report(path=None, print_output=False)]
        info.append(self.parameters_report())
        info.extend(self.metrics(pset).report() for pset in self.parameter_sets)
        if self.opt_result:
            info.append(self.opt_result.report(path=None, print_output=False))
        if self.identifiability:
            info.append(self.identifiability.report())

        with open(path, "w", encoding="utf-8") as f_report:
            f_report.write("\n".join(info))

    def parameters_report(self) -> str:
        """Get the report of the parameter sets.

        Reports the values of every set and the parameters which ended up close
        to one of their bounds.
        """
        info = [
            "-" * 80,
            f"Parameter sets: {[pset.sid for pset in self.parameter_sets]}",
            "-" * 80,
            self.parameter_sets.to_df().to_string(index=False),
            "",
        ]
        for pset in self.parameter_sets:
            if pset.cost is not None:
                info.append(f"{pset.sid}: cost = {pset.cost:.6g}")
            for msg in bound_warnings(self.problem.parameters, self.x(pset)):
                info.append(f"\t>>> {pset.sid}: {msg} <<<")
        info.append("-" * 80)
        return "\n".join(info)

    #: caption of every plot which describes the whole fit
    #: what a value of the report means, shown as a tooltip on its column.
    #: The keys are the column names of the tables of the report
    HINTS: ClassVar[dict[str, str]] = {
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
        "value": "Value of the parameter in the units of the model.",
        "se": (
            "Standard error of the parameter, the square root of the diagonal "
            "of the covariance, in the space the optimizer searches."
        ),
        "cv": (
            "The standard error relative to the value, in percent, in the "
            "space the optimizer searches."
        ),
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
            "mapping, with the agreement of the training data and the metrics "
            "of the panel"
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
            "The validation panel shows how the fit describes data it was not "
            "fitted on, the outlier panel the data the fit dropped. The box "
            "of a panel carries its R², its normalized RMSE and the RMSE of "
            "the residuals of the cost."
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

    def _plots(self, plots_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
        """Get the plots of the given names which were created.

        A plot of `WIDE_PLOTS` is `wide`, i.e. it covers a full row.
        """
        return [
            {
                "src": f"plots/{name}.{self.image_format}",
                "caption": self.PLOT_CAPTIONS.get(name, name),
                "hint": self.PLOT_HINTS.get(name, ""),
                "wide": name in WIDE_PLOTS,
            }
            for name in names
            if (plots_dir / f"{name}.{self.image_format}").exists()
        ]

    def html_context(self, results_dir: Path, name: str) -> dict[str, Any]:
        """Collect everything the HTML report shows.

        Args:
            results_dir: directory of the report, the files are relative to it.
            name: name of the report.

        Returns:
            The context of the `fit_report.html` template.
        """
        plots_dir = results_dir / "plots"
        counts = self.problem.mapping_counts()
        # the data the model does not describe is not resolved, so it has no
        # rows, no metrics and no figures: it is not part of the report
        kinds = [kind.value for kind in EVALUATED_KINDS]
        metrics = self.metrics_df()
        mapping_metrics = self.metrics_mappings_df()

        # the parameters with one column per set
        psets = list(self.parameter_sets)
        parameters = [
            {
                "pid": p.pid,
                # not `values`, jinja resolves that to `dict.values`
                "set_values": [
                    f"{pset.values.get(p.pid, float('nan')):.5g}" for pset in psets
                ],
                "lower": f"{p.lower_bound:.4g}",
                "upper": f"{p.upper_bound:.4g}",
                "unit": p.unit or "model",
            }
            for p in self.problem.parameters
        ]
        warnings: list[str] = []
        for pset in psets:
            warnings.extend(
                f"{pset.sid}: {message}"
                for message in bound_warnings(self.problem.parameters, self.x(pset))
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
            (row["parameter_set"], row["mapping"]): row for row in mapping_rows
        }
        reference = psets[-1].sid
        captions = ["Data and simulation", "Residuals and weighted residuals"]
        mappings: list[dict[str, Any]] = []
        for k, mapping_id in enumerate(self.problem.mapping_keys):
            sid = self.problem.experiment_keys[k]
            row = metrics_by_mapping.get((reference, mapping_id))
            plots = self._plots(
                plots_dir, [f"{sid}_{mapping_id}", f"fit_{sid}_{mapping_id}"]
            )
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
                        "RMSE": f"{row['RMSE']:.4g}" if row else "-",
                        "R²": f"{row['R2']:.4g}" if row else "-",
                    },
                    "plots": plots,
                }
            )

        files = [
            {"href": "report.txt", "label": "report.txt"},
            {"href": "parameters.json", "label": "parameters.json"},
            {"href": "metrics.tsv", "label": "metrics.tsv"},
            {"href": "metrics_mappings.tsv", "label": "metrics_mappings.tsv"},
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
            "title": f"{name} | sbmlsim fit report",
            "fit_id": name,
            "opid": self.problem.opid,
            "version": __version__,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "badges": badges,
            "kinds": kinds,
            "fit_info": self.fit_info(),
            "parameters": parameters,
            "parameter_set_ids": [pset.sid for pset in psets],
            "bound_warnings": warnings,
            "settings": {
                key.replace("_", " "): value
                for key, value in self.settings.to_dict().items()
            },
            "data_summary": data_summary,
            "data_total": {"counts": totals, "total": sum(totals)},
            "hints": dict(self.HINTS),
            "metrics_columns": [
                {"name": column, "hint": self.HINTS.get(column)}
                for column in metrics.columns
            ],
            "metrics": [
                [
                    f"{value:.6g}" if isinstance(value, float) else str(value)
                    for value in row.values()
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
            ],
            "numeric_columns": ["n", "MSE", "RMSE", "NRMSE", "RMSE_w", "R2"],
            "mapping_metrics": [
                {
                    "parameter_set": row["parameter_set"],
                    "experiment": row["experiment"],
                    "mapping": row["mapping"],
                    "kind": row["kind"],
                    "n": int(row["n"]),
                    "mse": f"{row['MSE']:.4g}",
                    "rmse": f"{row['RMSE']:.4g}",
                    "nrmse": f"{row['NRMSE']:.4g}",
                    "rmse_w": f"{row['RMSE_w']:.4g}",
                    "r2": f"{row['R2']:.4g}",
                }
                for row in mapping_rows
            ],
            "run_plots": self._plots(plots_dir, ["traces", "waterfall"]),
            "result_plots": self._plots(
                plots_dir,
                [
                    "goodness_of_fit",
                    "bland_altman",
                    "cost_bar",
                    "residual_boxplot",
                    "cost_scatter",
                ],
            ),
            "mappings": mappings,
            "run_columns": run_columns,
            "runs": runs,
            "identifiability": self._identifiability_context(plots_dir),
            "fisher": self._fisher_context(),
            "files": files,
        }

    def _fisher_context(self) -> dict[str, Any] | None:
        """Collect what the Fisher information of the report shows."""
        fim = self.fisher
        if fim is None:
            return None

        df = fim.summary_df
        eigenvalues = fim.eigenvalues
        info: dict[str, Any] = {
            "parameter set": fim.sid,
            "parameter scale": fim.scale.name,
            "data points": str(fim.n),
            "rank": f"{fim.rank} of {fim.k}",
            "condition number": f"{fim.condition_number:.4g}",
            "confidence level": f"{fim.alpha:.0%}",
        }
        correlation = fim.correlation
        return {
            "info": info,
            "identifiable": fim.is_identifiable,
            "columns": [
                {"name": column, "hint": self.HINTS.get(column)}
                for column in df.columns
            ],
            "rows": [
                [
                    value if isinstance(value, str) else f"{value:.5g}"
                    for value in row.values()
                ]
                for row in df.to_dict(orient="records")
            ],
            "eigenvalues": [f"{value:.4g}" for value in eigenvalues],
            "pids": list(fim.pids),
            "correlation": [
                [f"{correlation.iloc[i, j]:.3f}" for j in range(fim.k)]
                for i in range(fim.k)
            ],
        }

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
        }

    def fit_info(self) -> dict[str, str]:
        """Get the key facts of the fit, the same the console reports."""
        info = {
            "problem": self.problem.opid,
            "parameter sets": ", ".join(pset.sid for pset in self.parameter_sets),
            "fit mappings": ", ".join(
                f"{count} {kind.value}"
                for kind, count in self.problem.mapping_counts().items()
            ),
            "experiments": ", ".join(
                collection.experiment_class.__name__
                for collection in self.problem.mapping_collections
            ),
            "base path": str(self.problem.base_path),
            "data path": str(self.problem.data_path),
        }
        if self.opt_result:
            info["optimization"] = f"{self.opt_result.size} runs"
        return info

    def html_report(self, path: Path, name: str | None = None) -> None:
        """Create the interactive HTML report of the fit.

        The report is a single page with three sections: the overview of the
        fit, its results and the single fit mappings. It is rendered from the
        `fit_report.html` template and needs no network access.

        Args:
            path: file to write.
            name: name of the report, the directory of `path` by default.
        """
        context = self.html_context(
            results_dir=path.parent, name=name if name else path.parent.name
        )
        template = template_environment().get_template("fit_report.html")
        with open(path, "w", encoding="utf-8") as f_html:
            f_html.write(template.render(context))

    # --------------------------------------------------------------------
    # figures
    # --------------------------------------------------------------------
    def _create_figures(
        self, plots_dir: Path, mpl_parameters: dict[str, Any] | None = None
    ) -> None:
        """Create all figures of the report."""
        rc_params_copy = {**plt.rcParams}
        matplotlib.rcdefaults()

        parameters: dict[str, Any] = {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
        }
        parameters.update(mpl_parameters or {})
        # the keys of RcParams are typed as literals since matplotlib 3.11
        plt.rcParams.update(parameters)  # ty: ignore[no-matching-overload]

        try:
            if self.opt_result:
                self.plot_traces(path=plots_dir / f"traces.{self.image_format}")
                if self.opt_result.size > 1:
                    self.plot_waterfall(
                        path=plots_dir / f"waterfall.{self.image_format}"
                    )

            self.plot_goodness_of_fit(
                path=plots_dir / f"goodness_of_fit.{self.image_format}"
            )
            self.plot_bland_altman(path=plots_dir / f"bland_altman.{self.image_format}")
            self.plot_cost_bar(path=plots_dir / f"cost_bar.{self.image_format}")
            self.plot_residual_boxplot(
                path=plots_dir / f"residual_boxplot.{self.image_format}"
            )
            if len(self.parameter_sets) > 1:
                self.plot_cost_scatter(
                    path=plots_dir / f"cost_scatter.{self.image_format}"
                )

            self.plot_fit(output_dir=plots_dir)
            self.plot_fit_residual(output_dir=plots_dir)
            if self.identifiability:
                plot_all(self.identifiability, plots_dir, self.image_format)
        finally:
            # restore parameters
            plt.rcParams.update(rc_params_copy)

    def _create_mpl_figure(
        self, width: float = 5.0, height: float = 5.0, layout: str = "constrained"
    ) -> tuple[Figure, Axes]:
        """Create matplotlib figure."""
        return plt.subplots(nrows=1, ncols=1, figsize=(width, height), layout=layout)

    def _save_mpl_figure(self, fig: Figure, path: Path) -> None:
        """Save matplotlib figure to path."""
        fig.savefig(path)
        plt.close(fig)

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

    def _set_legend(self, ax: Axes) -> None:
        """Add a legend without duplicate entries."""
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        if unique:
            ax.legend(unique.values(), unique.keys())

    @staticmethod
    def _set_figure_legend(fig: Figure, axes: Sequence[Axes]) -> None:
        """Add one legend for all panels of a figure, without duplicates.

        A study is not in every panel, e.g. only one study has outliers, so the
        entries are collected over the panels and shown once next to them.
        """
        handles: list[Any] = []
        labels: list[str] = []
        for ax in axes:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        unique = dict(zip(labels, handles, strict=True))
        if unique:
            fig.legend(
                unique.values(),
                unique.keys(),
                loc="outside right upper",
                fontsize="small",
            )

    def plot_fit(self, output_dir: Path) -> None:
        """Plot the data and the simulation of every parameter set per mapping."""
        res_data = {pset.sid: self.residual_data(pset) for pset in self.parameter_sets}

        for k, mapping_id in enumerate(self.problem.mapping_keys):
            fig, [ax1, ax2] = plt.subplots(
                nrows=1, ncols=2, figsize=(10, 5), layout="constrained"
            )

            sid = self.problem.experiment_keys[k]
            x_ref = self.problem.x_references[k]
            y_ref = self.problem.y_references[k]
            y_ref_err = self.problem.y_errors[k]
            y_ref_err_type = self.problem.y_errors_type[k]
            x_id = self.problem.xid_observable[k]
            y_id = self.problem.yid_observable[k]

            for ax in [ax1, ax2]:
                if self.show_titles:
                    ax.set_title(self.mapping_title(k))
                ax.set_ylabel(y_id)
                ax.set_xlabel(x_id)

                # reference data, the same for all parameter sets
                if y_ref_err is None:
                    ax.plot(
                        x_ref,
                        y_ref,
                        "s",
                        color="black",
                        label="reference_data",
                        markersize=10,
                    )
                else:
                    ax.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label=f"reference_data ± {y_ref_err_type}",
                        markersize=10,
                    )

                # simulation of every parameter set
                for pset in self.parameter_sets:
                    data = res_data[pset.sid]
                    ax.plot(
                        data["x_obs"][k].values,
                        data["y_obs"][k].values,
                        "-",
                        color=self.color(pset),
                        label=pset.sid,
                    )

                xdelta = np.max(x_ref) - np.min(x_ref)
                ax.set_xlim(
                    left=np.min(x_ref) - 0.1 * xdelta,
                    right=np.max(x_ref) + 0.1 * xdelta,
                )
                self._set_legend(ax)

            ax2.set_yscale("log")
            ax2.set_ylim(bottom=self._log_limits(y_ref, factor=1.0 / 0.3)[0])

            self._save_mpl_figure(
                fig, path=output_dir / f"{sid}_{mapping_id}.{self.image_format}"
            )

    def plot_fit_residual(self, output_dir: Path) -> None:
        """Plot data, prediction and residuals of every mapping.

        The upper panels show the data, the interpolated prediction and the
        residuals, the lower panels the squared weighted residuals; the right
        panels are logarithmic.
        """
        res_data = {pset.sid: self.residual_data(pset) for pset in self.parameter_sets}

        for k, mapping_id in enumerate(self.problem.mapping_keys):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                nrows=2, ncols=2, figsize=(10, 10), layout="constrained"
            )

            sid = self.problem.experiment_keys[k]
            x_ref = self.problem.x_references[k]
            y_ref = self.problem.y_references[k]
            y_ref_err = self.problem.y_errors[k]
            x_id = self.problem.xid_observable[k]
            y_id = self.problem.yid_observable[k]

            for ax in (ax1, ax3):
                ax.axhline(y=0, color="black")
                ax.set_ylabel(y_id)
            for ax in (ax3, ax4):
                ax.set_xlabel(x_id)

            for ax in (ax1, ax2):
                if self.show_titles:
                    ax.set_title(self.mapping_title(k))
                plt.setp(ax.get_xticklabels(), visible=False)

                # reference data
                if y_ref_err is None:
                    ax.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label="reference_data",
                    )

                for pset in self.parameter_sets:
                    data = res_data[pset.sid]
                    color = self.color(pset)
                    ax.plot(
                        data["x_obs"][k].values,
                        data["y_obs"][k].values,
                        "-",
                        color=color,
                        label=pset.sid,
                    )
                    ax.plot(x_ref, data["y_obsip"][k], "o", color=color, alpha=0.6)
                    ax.plot(
                        x_ref,
                        data["residuals"][k],
                        "v",
                        color=color,
                        alpha=0.6,
                        label=f"{pset.sid} residuals",
                    )

            for ax in (ax3, ax4):
                for pset in self.parameter_sets:
                    res_weighted2 = np.power(
                        res_data[pset.sid]["residuals_weighted"][k], 2
                    )
                    ax.plot(
                        x_ref,
                        res_weighted2,
                        "o",
                        color=self.color(pset),
                        label=f"{pset.sid} $(w \\cdot r)^2$",
                    )
                ax.set_xlim(ax1.get_xlim())

            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xlim(right=1.1 * np.max(x_ref))
                self._set_legend(ax)

            for ax in (ax2, ax4):
                ax.set_yscale("log")

            self._save_mpl_figure(
                fig=fig,
                path=output_dir / f"fit_{sid}_{mapping_id}.{self.image_format}",
            )

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

    kwargs_scatter: ClassVar[dict[str, Any]] = {
        "markersize": "10",
        "markeredgecolor": "black",
        "alpha": 0.7,
        "linestyle": "",
        "marker": "o",
    }

    # --------------------------------------------------------------------
    # comparison of the parameter sets
    # --------------------------------------------------------------------
    def _panels(self, height: float = 5.5) -> tuple[Figure, list[Axes], list[str]]:
        """Create a figure with a panel per subset of the data points.

        The figure covers a full row of the report, so the panels are large
        enough for the points, the band and the box with the metrics.
        """
        kinds = self.point_kinds()
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(kinds),
            figsize=(height * len(kinds), height),
            layout="constrained",
            squeeze=False,
        )
        return fig, list(axes[0]), kinds

    def _band_color(self, pset: ParameterSet) -> str:
        """Get the color of the agreement band of a parameter set.

        The points carry the color of their study, so the band is neutral and
        only takes a color when several sets are compared.
        """
        return "black" if len(self.parameter_sets) == 1 else self.color(pset)

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
        of `FitMetrics.summary`; the absolute RMSE is not shown, the data
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
        lines: list[str] = []
        for pset in self.parameter_sets:
            prefix = f"{pset.sid}: " if len(self.parameter_sets) > 1 else ""
            if plot == "goodness_of_fit":
                summary = self.metrics(pset).summary(kind=MappingKind(kind))
                lines.append(
                    f"{prefix}R² = {summary['R2']:.3f}, "
                    f"NRMSE = {summary['NRMSE']:.3g}, "
                    f"RMSE$_w$ = {summary['RMSE_w']:.3g}"
                )
            elif plot == "bland_altman":
                dp = self._of_kind(self.points(pset), kind)
                _mean, difference, _mask = self._log_ratio(dp)
                if difference.size == 0:
                    lines.append(f"{prefix}no ratio")
                    continue
                bias_training, half = self.agreement(pset)
                inside = np.mean(np.abs(difference - bias_training) <= half + 1e-12)
                bias = float(np.mean(difference))
                sd = float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0
                lines.append(
                    f"{prefix}bias = {10**bias:.2f}x, SD = {10**sd:.2f}x, "
                    f"in LoA = {inside:.0%}"
                )
            else:
                raise ValueError(f"Unknown plot '{plot}'.")
        return "\n".join(lines)

    def _add_metrics_box(self, ax: Axes, kind: str, plot: str) -> None:
        """Draw the box with the key metrics into the corner of a panel."""
        ax.text(
            0.03,
            0.97,
            self.panel_metrics(kind, plot),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox=METRICS_BOX,
            zorder=10,
        )

    def plot_goodness_of_fit(self, path: Path) -> None:
        """Plot the predicted against the measured data points, per kind.

        One panel per kind of fit mapping, i.e. the training data, the
        validation data and the outliers separately. The points scatter around
        the identity line when the model describes the data.

        The band is the agreement of `agreement`, i.e. the bias and the limits
        `bias ± 1.96 SD` of the training data, which on logarithmic axes are
        lines parallel to the identity: a point inside the band is a prediction
        the fit agrees to. It is the same band the Bland-Altman plot draws, in
        the same styles, so the two figures are read the same way.
        """
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        fig, axes, kinds = self._panels()

        min_dp, max_dp = self._log_limits(
            *[dp.DV for dp in points.values()], *[dp.IPRED for dp in points.values()]
        )
        edge = np.array([min_dp, max_dp])

        for ax, kind in zip(axes, kinds, strict=True):
            # the band, labelled on the last panel so that the legend lists the
            # studies first and the band, which is the same everywhere, last
            is_last = ax is axes[-1]
            for pset in self.parameter_sets:
                bias, half = self.agreement(pset)
                color = self._band_color(pset)
                _identity, bias_label, limits_label = self._band_labels(pset)
                ax.fill_between(
                    edge,
                    edge * 10 ** (bias - half),
                    edge * 10 ** (bias + half),
                    color=color,
                    alpha=BAND_ALPHA,
                    zorder=0,
                )
                ax.plot(
                    edge,
                    edge * 10**bias,
                    color=color,
                    label=bias_label if is_last else None,
                    **BIAS_STYLE,
                )
                for limit in (bias - half, bias + half):
                    ax.plot(
                        edge,
                        edge * 10**limit,
                        color=color,
                        label=limits_label if is_last else None,
                        **LIMITS_STYLE,
                    )
            ax.plot(
                edge,
                edge,
                color="black",
                label=self._band_labels(self.reference_set)[0] if is_last else None,
                **IDENTITY_STYLE,
            )

            for pset in self.parameter_sets:
                dp = self._of_kind(points[pset.sid], kind)
                kwargs: dict[str, Any] = {
                    **self.kwargs_scatter,
                    "marker": self.set_marker(pset),
                }
                for study, of_study in dp.groupby("experiment", sort=False):
                    ax.plot(
                        of_study.DV.values,
                        of_study.IPRED.values,
                        label=self._point_label(study, pset),
                        color=self.study_color(str(study)),
                        **kwargs,
                    )

            n_points = len(self._of_kind(points[self.reference_set.sid], kind))
            ax.set_title(f"{kind} (n={n_points})")
            ax.set_xlabel("Experiment $y_{i,k}$", fontweight="bold")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min_dp, max_dp)
            ax.set_ylim(min_dp, max_dp)
            ax.grid()
            self._add_metrics_box(ax, kind, "goodness_of_fit")

        axes[0].set_ylabel("Prediction $f(x_{i,k})$", fontweight="bold")
        self._set_figure_legend(fig, axes)
        if self.show_titles:
            fig.suptitle("Goodness of fit")
        self._save_mpl_figure(fig=fig, path=path)

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

    @staticmethod
    def _include_limits(ax: Axes, agreement: Iterable[tuple[float, float]]) -> None:
        """Widen the y axis of a panel so that the limits of agreement fit.

        The limits are those of the training data and are drawn in every panel,
        so a panel whose own points stay well inside them must not crop them.
        """
        bounds = [bias + sign * half for bias, half in agreement for sign in (-1, 1)]
        if not bounds:
            return
        low, high = ax.get_ylim()
        margin = 0.05 * max(high - low, max(bounds) - min(bounds), 1e-6)
        ax.set_ylim(
            min(low, min(bounds) - margin),
            max(high, max(bounds) + margin),
        )

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

    def plot_bland_altman(self, path: Path) -> None:
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
        """
        points = {pset.sid: self.points(pset) for pset in self.parameter_sets}
        agreement = {pset.sid: self.agreement(pset) for pset in self.parameter_sets}
        fig, axes, kinds = self._panels()

        for ax, kind in zip(axes, kinds, strict=True):
            n_shown = 0
            for pset in self.parameter_sets:
                dp = self._of_kind(points[pset.sid], kind)
                mean, difference, mask = self._log_ratio(dp)
                n_shown = max(n_shown, difference.size)
                if difference.size == 0:
                    continue

                studies = np.asarray(dp.experiment, dtype=object)[mask]
                kwargs: dict[str, Any] = {
                    **self.kwargs_scatter,
                    "marker": self.set_marker(pset),
                }
                for study in dict.fromkeys(studies):
                    of_study = studies == study
                    ax.plot(
                        mean[of_study],
                        difference[of_study],
                        label=self._point_label(study, pset),
                        color=self.study_color(str(study)),
                        **kwargs,
                    )

            # the agreement of the training data, the same lines in every
            # panel. The points carry the color of their study, so the lines
            # are neutral and only take a color when several sets are compared
            # the same band as the goodness of fit, here as horizontal lines.
            # Only the last panel labels it, so that the legend lists the
            # studies first and the band, which is the same everywhere, last
            is_last = ax is axes[-1]
            for pset in self.parameter_sets:
                bias, half = agreement[pset.sid]
                color = self._band_color(pset)
                _identity, bias_label, limits_label = self._band_labels(pset)
                ax.axhspan(
                    bias - half,
                    bias + half,
                    color=color,
                    alpha=BAND_ALPHA,
                    zorder=0,
                )
                ax.axhline(
                    bias,
                    color=color,
                    label=bias_label if is_last else None,
                    **BIAS_STYLE,
                )
                for limit in (bias - half, bias + half):
                    ax.axhline(
                        limit,
                        color=color,
                        label=limits_label if is_last else None,
                        **LIMITS_STYLE,
                    )

            # the identity of the goodness of fit is no difference here
            ax.axhline(
                0.0,
                color="black",
                label=self._band_labels(self.reference_set)[0] if is_last else None,
                **IDENTITY_STYLE,
            )
            # a data point which is zero or negative has no ratio, so a panel
            # can show fewer points than the metrics of the kind count
            n_points = len(self._of_kind(points[self.reference_set.sid], kind))
            label = (
                f"n={n_shown}" if n_shown == n_points else f"n={n_shown} of {n_points}"
            )
            ax.set_title(f"{kind} ({label})")
            ax.set_xlabel("Geometric mean $\\sqrt{f(x_{i,k}) y_{i,k}}$")
            ax.set_xscale("log")
            ax.grid()
            self._include_limits(ax, agreement.values())
            self._add_metrics_box(ax, kind, "bland_altman")

        axes[0].set_ylabel("$\\log_{10}\\frac{f(x_{i,k})}{y_{i,k}}$", fontweight="bold")
        self._set_figure_legend(fig, axes)
        if self.show_titles:
            fig.suptitle("Bland-Altman")
        self._save_mpl_figure(fig=fig, path=path)

    def plot_cost_bar(self, path: Path) -> None:
        """Plot the cost and the weight of every curve, per set."""
        costs = {pset.sid: self._cost_df(pset) for pset in self.parameter_sets}
        costs_ref = costs[self.reference_set.sid]

        fig: Figure
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(8, 6), layout="constrained"
        )
        if self.show_titles:
            fig.suptitle("Curve costs and weights")

        n_sets = len(self.parameter_sets)
        height = 0.8 / n_sets
        position = np.arange(len(costs_ref))
        ticklabels = [
            f"{costs_ref.experiment[k]}|{costs_ref.mapping[k]}"
            for k in range(len(costs_ref))
        ]

        for ks, pset in enumerate(self.parameter_sets):
            ax1.barh(
                position + ks * height,
                costs[pset.sid].cost,
                height=height,
                color=self.color(pset),
                alpha=0.8,
                label=pset.sid,
            )

        ax1.set_yticks(position + 0.4 - height / 2)
        ax1.set_yticklabels(ticklabels, ha="right", fontdict={"fontsize": 8})
        ax1.grid(True, axis="x")
        ax1.set_xlabel("Cost")
        ax1.set_xscale("log")
        self._set_legend(ax1)

        # the weights of the curves do not depend on the parameters
        ax2.barh(position, costs_ref.weight_curve, color="tab:blue", alpha=0.8)
        ax2.set_yticks(position)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.grid(True, axis="x")
        ax2.set_xlabel("Weight curve: $w_{k}$")

        self._save_mpl_figure(fig=fig, path=path)

    def plot_residual_boxplot(self, path: Path) -> None:
        """Plot the distribution of the squared weighted residuals per curve."""
        costs_ref = self._cost_df(self.reference_set)

        fig, ax = self._create_mpl_figure(width=6.0, height=6.0)
        if self.show_titles:
            ax.set_title("Residual contribution")

        n_mappings = len(self.problem.mapping_keys)
        ticklabels = [
            f"{costs_ref.experiment[k]}|{costs_ref.mapping[k]}"
            for k in range(len(costs_ref))
        ]

        n_sets = len(self.parameter_sets)
        for ks, pset in enumerate(self.parameter_sets):
            res_data = self.residual_data(pset)
            offset = (ks - (n_sets - 1) / 2) * 0.25
            for k in range(n_mappings):
                res_weighted2 = np.power(res_data["residuals_weighted"][k], 2)
                ax.plot(
                    res_weighted2,
                    (k + 1 + offset) * np.ones_like(res_weighted2),
                    linestyle="",
                    marker="s",
                    markeredgecolor="black",
                    color=self.color(pset),
                    markersize=3,
                    label=pset.sid if k == 0 else None,
                )

        ax.set_yticks(list(range(1, n_mappings + 1)))
        ax.set_yticklabels(ticklabels, ha="right", fontdict={"fontsize": 8})
        ax.grid(True, axis="x")
        ax.set_xlabel(
            "Weighted residuals^2\n$(w_{k} \\cdot w_{i,k} (f(x_{i,k}) - y_{i,k}))^2$"
        )
        ax.set_xscale("log")
        self._set_legend(ax)
        self._save_mpl_figure(fig=fig, path=path)

    def plot_cost_scatter(self, path: Path) -> None:
        """Plot the cost of every curve against the cost of the reference set."""
        reference = self.reference_set
        costs_ref: pd.DataFrame = self._cost_df(reference)

        fig, ax = self._create_mpl_figure()
        other_sets = list(self.parameter_sets)[1:]
        all_costs = [costs_ref.cost, *[self._cost_df(p).cost for p in other_sets]]
        min_cost, max_cost = self._log_limits(*all_costs, factor=2.0)

        ax.plot([min_cost, max_cost], [min_cost, max_cost], "--", color="black")
        if self.show_titles:
            ax.set_title(f"Cost against '{reference.sid}'")

        for pset in other_sets:
            costs_x = self._cost_df(pset)
            ax.plot(
                costs_ref.cost,
                costs_x.cost,
                linestyle="",
                marker="o",
                color=self.color(pset),
                markersize=10,
                alpha=0.8,
                label=pset.sid,
            )
            for k, exp_key in enumerate(self.problem.experiment_keys):
                ax.annotate(
                    exp_key,
                    xy=(costs_ref.cost[k], costs_x.cost[k]),
                    fontsize="x-small",
                    alpha=0.7,
                )

        ax.set_xlabel(f"Cost '{reference.sid}'")
        ax.set_ylabel("Cost")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()

        legend_elements = [
            Line2D([0], [0], linestyle="--", color="black", label="no change"),
        ]
        handles, _labels = ax.get_legend_handles_labels()
        ax.legend(handles=[*handles, *legend_elements])
        self._save_mpl_figure(fig=fig, path=path)

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

    def plot_waterfall(self, path: Path) -> None:
        """Create waterfall plot for the fit results.

        Plots the optimization runs sorted by cost.
        """
        optres = self.opt_result_required
        fig, ax = self._create_mpl_figure()
        if self.show_titles:
            ax.set_title("Waterfall plot")
        ax.plot(
            range(optres.size),
            1 + (optres.df_fits.cost.values - optres.df_fits.cost.values[0]),
            "-o",
            color="black",
        )
        ax.set_xlabel("Index (ordered optimizer run)")
        ax.set_ylabel("Offset cost value (relative to best start)")
        ax.set_yscale("log")
        self._save_mpl_figure(fig, path=path)

    def plot_traces(self, path: Path) -> None:
        """Plot optimization traces.

        Optimization time course of costs.
        """
        optres = self.opt_result_required
        fig, ax = self._create_mpl_figure()
        if self.show_titles:
            ax.set_title("Optimization traces")
        for run in range(optres.size):
            df_run = optres.df_traces[optres.df_traces.run == run]
            if len(df_run) == 0:
                continue
            ax.plot(range(len(df_run)), df_run.cost.values, "-", alpha=0.8)
            # final cost of the trace
            ax.plot(
                len(df_run) - 1, df_run.cost.values[-1], "o", color="black", alpha=0.8
            )

        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Cost")
        ax.set_yscale("log")

        self._save_mpl_figure(fig, path=path)
