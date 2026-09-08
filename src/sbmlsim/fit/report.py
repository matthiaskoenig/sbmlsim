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

import logging
import webbrowser
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sbmlsim.console import console
from sbmlsim.fit.metrics import FitMetrics
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.plot.serialization_matplotlib import plt

logger = logging.getLogger(__name__)

#: colors of the parameter sets, in the order of the sets
SET_COLORS: tuple[str, ...] = (
    "black",
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
            show_titles: add titles to the panels.
            image_format: format of the figures.
        """
        self.problem = problem
        self.settings = settings
        self.parameter_sets = ParameterSets.of(parameter_sets)
        self.opt_result = opt_result
        self.show_titles = show_titles
        self.image_format = image_format

        # resolves the data, a no-op if the problem is already initialized
        problem.initialize(settings)

        # residual data of the mappings, by parameter set
        self._res_data: dict[str, dict[str, list[Any]]] = {}

    @staticmethod
    def from_optimization_result(
        problem: OptimizationProblem,
        opt_result: OptimizationResult,
        size: int = 1,
        with_model: bool = True,
        **kwargs: Any,
    ) -> FitReport:
        """Create the report of an optimization.

        Args:
            problem: definition of the optimization problem.
            opt_result: result of the optimization, it carries the settings.
            size: number of fitted parameter sets to report, the best first.
            with_model: report the initial values of the model as the reference
                set, so that the plots compare the fit against them.
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
                xlog=np.log10(self.x(pset)), complete_data=True
            )
        return self._res_data[pset.sid]

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

        console.rule(f"Report '{name}'", align="left", style="white")
        console.print(f"{'sets':<12}: {[pset.sid for pset in self.parameter_sets]}")

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

        self._write_text_report(path=results_dir / "report.txt")
        self._create_figures(plots_dir=plots_dir, mpl_parameters=mpl_parameters)
        self.html_report(path=results_dir / "index.html")

        report_path = results_dir / "index.html"
        console.print(
            f"{'report':<12}: file://{report_path.resolve()}",
            style="success",
            soft_wrap=True,
        )
        if show_report:
            webbrowser.open(f"file://{report_path.resolve()!s}", new=2)

        return results_dir

    def _write_text_report(self, path: Path) -> None:
        """Write the text report of the problem, the parameters and the runs."""
        info = [self.problem.report(path=None, print_output=False)]
        info.append(self.parameters_report())
        info.extend(self.metrics(pset).report() for pset in self.parameter_sets)
        if self.opt_result:
            info.append(self.opt_result.report(path=None, print_output=False))

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

    def html_report(self, path: Path) -> None:
        """Create the HTML report of the fit."""
        title = f"{self.problem.opid}"
        if self.opt_result:
            title = f"{title} [{self.opt_result.sid}]"

        parameters_html = self.parameter_sets.to_df().to_html(index=False)
        metrics_html = self.metrics_df().to_html(index=False)
        metrics_mappings_html = self.metrics_mappings_df().to_html(index=False)

        warnings_info: list[str] = []
        for pset in self.parameter_sets:
            for msg in bound_warnings(self.problem.parameters, self.x(pset)):
                warnings_info.append(f"<li>{pset.sid}: {msg}</li>")
        warnings_html = (
            f"<ul>{''.join(warnings_info)}</ul>" if warnings_info else "<p>none</p>"
        )

        fit_images = "\n".join(
            f'<p><img src="plots/{self.problem.experiment_keys[k]}_{mapping_id}'
            f'.{self.image_format}"></p>'
            for k, mapping_id in enumerate(self.problem.mapping_keys)
        )

        run_images = ""
        runs_html = ""
        if self.opt_result:
            run_images = (
                f"<h2>Optimization performance</h2>\n<p>\n"
                f'<img src="./plots/traces.{self.image_format}">\n'
                f'<img src="./plots/waterfall.{self.image_format}">\n</p>'
            )
            runs_html = (
                f"<h2>Optimization runs</h2>\n{self.opt_result.df_fits.to_html()}"
            )

        cost_scatter = ""
        if len(self.parameter_sets) > 1:
            cost_scatter = f'<img src="./plots/cost_scatter.{self.image_format}">'

        html = f"""
        <html>

        <body>
        <h1>Parameter fitting: {title}</h1>

        <h2>Parameters</h2>
        {parameters_html}

        <h3>Parameters on their bounds</h3>
        {warnings_html}

        <h2>Metrics</h2>
        {metrics_html}

        <h3>Metrics of the fit mappings</h3>
        {metrics_mappings_html}

        <h2>Settings</h2>
        <pre>{self.settings}</pre>

        <p>
        <ul>
            <li><a target="_blank" href="report.txt">report.txt</a></li>
            <li><a target="_blank" href="parameters.json">parameters.json</a></li>
            <li><a target="_blank" href="metrics.tsv">metrics.tsv</a></li>
            <li><a target="_blank" href="metrics_mappings.tsv">metrics_mappings.tsv</a></li>
            <li><a target="_blank" href="datapoints.tsv">datapoints.tsv</a></li>
        </ul>
        </p>

        {run_images}

        <h2>Data point prediction</h2>
        <p>
        <img src="./plots/datapoint_scatter.{self.image_format}">
        <img src="./plots/residual_scatter.{self.image_format}">
        </p>

        <p>
        <img src="./plots/residual_boxplot.{self.image_format}">
        <img src="./plots/cost_bar.{self.image_format}">
        {cost_scatter}
        </p>

        <h2>Fits</h2>
        <p>
        {fit_images}
        </p>

        {runs_html}
        </body>
        </html>
        """
        with open(path, "w", encoding="utf-8") as f_out:
            f_out.write(html)

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

            self.plot_datapoint_scatter(
                path=plots_dir / f"datapoint_scatter.{self.image_format}"
            )
            self.plot_residual_scatter(
                path=plots_dir / f"residual_scatter.{self.image_format}"
            )
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
    def plot_datapoint_scatter(self, path: Path) -> None:
        """Plot the predicted against the measured data points, per set."""
        fig, ax = self._create_mpl_figure()
        dps = {pset.sid: self._datapoints_df(pset) for pset in self.parameter_sets}

        min_dp, max_dp = self._log_limits(
            *[dp.y_ref for dp in dps.values()], *[dp.y_obs for dp in dps.values()]
        )

        ax.fill_between(
            [min_dp, max_dp, max_dp, min_dp],
            [min_dp / 10, max_dp / 10, max_dp * 10, min_dp * 10],
            color="lightgray",
        )
        ax.plot([min_dp, max_dp], [min_dp, max_dp], color="black")
        for bfactor in [1 / 10, 10]:
            ax.plot(
                [min_dp, max_dp],
                [min_dp * bfactor, max_dp * bfactor],
                "--",
                color="black",
            )

        for pset in self.parameter_sets:
            dp = dps[pset.sid]
            ax.plot(
                dp.y_ref.values,
                dp.y_obs.values,
                label=pset.sid,
                color=self.color(pset),
                **self.kwargs_scatter,
            )

        ax.set_xlabel("Experiment $y_{i,k}$", fontweight="bold")
        ax.set_ylabel("Prediction $f(x_{i,k})$", fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_dp, max_dp)
        ax.grid()
        self._set_legend(ax)
        if self.show_titles:
            ax.set_title("Data points")
        self._save_mpl_figure(fig=fig, path=path)

    def plot_residual_scatter(self, path: Path) -> None:
        """Plot the relative residuals against the data, per set."""
        fig, ax = self._create_mpl_figure()

        for pset in self.parameter_sets:
            dp = self._datapoints_df(pset)
            with np.errstate(divide="ignore", invalid="ignore"):
                ydata = dp.residual.values / dp.y_ref.values
            ax.plot(
                dp.y_ref.values,
                ydata,
                label=pset.sid,
                color=self.color(pset),
                **self.kwargs_scatter,
            )

        ax.axhline(y=0.0, linestyle="--", color="black")
        ax.axhspan(-0.5, 0.5, color="lightgray", zorder=0)
        ax.set_xlabel("Experiment $y_{i,k}$", fontweight="bold")
        ax.set_ylabel(
            "Relative residual $\\frac{f(x_{i,k})-y_{i,k}}{y_{i,k}}$", fontweight="bold"
        )
        ax.set_xscale("log")
        ax.grid()
        self._set_legend(ax)
        if self.show_titles:
            ax.set_title("Residuals")
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
