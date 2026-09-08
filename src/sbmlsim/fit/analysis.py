"""Analysis of fitting results."""

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

from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    LossFunctionType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.plot.serialization_matplotlib import plt
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


class OptimizationAnalysis:
    """Class for analyzing optimization results.

    Creates all plots and results.
    """

    def __init__(
        self,
        opt_result: OptimizationResult,
        output_name: str,
        output_dir: Path,
        op: OptimizationProblem | None = None,
        show_plots: bool = False,
        show_report: bool = False,
        show_titles: bool = True,
        residual: ResidualType | None = None,
        loss_function: LossFunctionType | None = None,
        weighting_curves: list[WeightingCurvesType] | None = None,
        weighting_points: WeightingPointsType | None = None,
        variable_step_size: bool = True,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        image_format: str = "svg",
        **kwargs,
    ) -> None:
        """Construct Optimization analysis.

        :param output_name: name of the optimization
        :param output_dir: base path for output
        :param show_plots: boolean flag to display plots, i.e., call plt.show()
        :param show_report: boolean flag to open the HTML report in a web browser
        :param show_titles: boolean flag to add titles to the panels
        :param residual: handling of residuals
        :param loss_function: loss function for handling outliers/residual transformation
        :param weighting_curves: list of options for weighting curves (fit mappings)
        :param weighting_points: weighting of points
        :param seed: integer random seed (for sampling of parameters)
        :param absolute_tolerance: absolute tolerance of simulator
        :param relative_tolerance: relative tolerance of simulator
        :param variable_step_size: use variable step size in solver

        """
        self.sid = opt_result.sid
        self.optres: OptimizationResult = opt_result
        # the results directory uses the hash of the OptimizationResult
        results_dir: Path = output_dir / self.sid / output_name
        if not results_dir.exists():
            logger.warning("create output directory: '%s'", results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir

        self.image_format = image_format
        self.show_plots = show_plots
        self.show_report = show_report
        self.show_titles = show_titles
        # residual data of the mappings, by parameter vector
        self._res_data: dict[bytes, dict[str, list[Any]]] = {}

        if kwargs:
            for key, value in kwargs.items():
                logger.warning(
                    "Unsupported argument to OptimizationAnalysis '%s: %s'.", key, value
                )

        if op:
            if residual is None or weighting_points is None:
                raise ValueError(
                    "'residual' and 'weighting_points' are required to initialize "
                    "the OptimizationProblem of an OptimizationAnalysis."
                )
            op.initialize(
                residual=residual,
                loss_function=loss_function
                if loss_function is not None
                else LossFunctionType.LINEAR,
                weighting_curves=weighting_curves if weighting_curves else [],
                weighting_points=weighting_points,
                variable_step_size=variable_step_size,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )

        self._op: OptimizationProblem | None = op

    def residual_data(self, x: np.ndarray) -> dict[str, list[Any]]:
        """Get the complete residual data of the mappings for the parameters x.

        Every evaluation simulates all fit mappings, so the results are cached for
        the plots and tables which use the same parameter vector.
        """
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self._res_data:
            self._res_data[key] = self.op.residuals(  # ty: ignore[invalid-assignment]
                xlog=np.log10(x), complete_data=True
            )
        return self._res_data[key]

    @property
    def op(self) -> OptimizationProblem:
        """Optimization problem of the analysis, required for the fit plots."""
        if self._op is None:
            raise ValueError("OptimizationAnalysis requires the OptimizationProblem.")
        return self._op

    def run(self, mpl_parameters: dict[str, Any] | None = None) -> None:
        """Execute complete analysis.

        This creates all plots and reports.
        """
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------
        # Create HTML report
        # ----------------------
        self.html_report(path=self.results_dir / "index.html")

        # ----------------------
        # Write text report
        # ----------------------
        problem_info: str = ""
        if self._op:
            problem_info = self.op.report(
                path=None,
                print_output=False,
            )
        result_info = self.optres.report(
            path=None,
            print_output=True,
        )
        info = problem_info + result_info
        with open(self.results_dir / "report.txt", "w", encoding="utf-8") as f_report:
            f_report.write(info)

        # FIXME: create JSON information for problem
        self.optres.to_json(path=self.results_dir / "optimization_result.json")
        self.optres.to_tsv(path=self.results_dir / "optimization_result.tsv")

        # ----------------------
        # Create figures
        # ----------------------
        rc_params_copy = {**plt.rcParams}
        # reset matplotlib parameters
        matplotlib.rcdefaults()
        if mpl_parameters is None:
            mpl_parameters = {}

        parameters: dict[str, Any] = {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
        }
        parameters.update(mpl_parameters)
        # the keys of RcParams are typed as literals since matplotlib 3.11
        plt.rcParams.update(parameters)  # ty: ignore[no-matching-overload]

        # optimization traces
        self.plot_traces(
            path=plots_dir / f"traces.{self.image_format}",
        )
        # waterfall plot
        if self.optres.size > 1:
            self.plot_waterfall(
                path=plots_dir / f"waterfall.{self.image_format}",
            )

        # plot fit results for optimal parameters
        if self._op:
            xopt = self.optres.xopt

            self.plot_datapoint_scatter(
                x=xopt,
                path=plots_dir / f"datapoint_scatter.{self.image_format}",
            )
            self.plot_residual_scatter(
                x=xopt,
                path=plots_dir / f"residual_scatter.{self.image_format}",
            )

            self.plot_cost_scatter(
                x=xopt,
                path=plots_dir / f"cost_scatter.{self.image_format}",
            )
            self.plot_cost_bar(
                x=xopt,
                path=plots_dir / f"cost_bar.{self.image_format}",
            )

            self.plot_residual_boxplot(
                x=xopt,
                path=plots_dir / f"residual_boxplot.{self.image_format}",
            )

            # plot individual fit mappings
            self.plot_fit(output_dir=plots_dir, x=xopt)
            self.plot_fit_residual(output_dir=plots_dir, x=xopt)

        # restore parameters
        plt.rcParams.update(rc_params_copy)

        report_path = self.results_dir / "index.html"
        logger.info("Analysis finished: file://%s", report_path)
        if self.show_report:
            webbrowser.open(f"file://{report_path!s}", new=2)

    def html_report(self, path: Path) -> None:
        """Create HTML report of the fit."""
        title = f"{self.op.opid} [{self.sid}]"

        xopt = self.optres.xopt
        parameter_info = [
            f"&gt;&gt;&gt; {msg} &lt;&lt;&lt;"
            for msg in bound_warnings(self.optres.parameters, xopt)
        ]
        fitted_pars = {
            p.pid: (xopt[k], p.unit, p.lower_bound, p.upper_bound)
            for k, p in enumerate(self.optres.parameters)
        }

        for key, value in fitted_pars.items():
            parameter_info.append(
                f"<strong>{key}</strong>: {value[0]} {value[1]}, [{value[2]} - {value[3]}]"
            )
        parameters = "<br/>".join(parameter_info)

        fit_images_info = []
        for k, mapping_id in enumerate(self.op.mapping_keys):
            sid = self.op.experiment_keys[k]

            fit_images_info.append(
                f'<p><img src="plots/{sid}_{mapping_id}.{self.image_format}"></p>'
            )
        fit_images = "\n".join(fit_images_info)

        html = f"""
        <html>

        <body>
        <h1>Parameter fitting: {title}</h1>

        <h2>Optimal parameters</h2>
        <p>
        {parameters}
        </p>
        <ul>
            <li><a target="_blank" href="report.txt">report.txt</a></li>
            <li><a target="_blank" href="optimization_result.json">optimization_result.json</a></li>
            <li><a target="_blank" href="optimization_result.tsv">optimization_result.tsv</a></li>
        </ul>

        <h2>Optimization Performance</h2>
        <p>
        <img src="./plots/traces.{self.image_format}">
        <img src="./plots/waterfall.{self.image_format}">
        </p>

        <h2>Data point prediction</h2>
        <p>
        <img src="./plots/datapoint_scatter.{self.image_format}">
        <img src="./plots/residual_scatter.{self.image_format}">
        </p>

        <p>
        <img src="./plots/residual_boxplot.{self.image_format}">
        <img src="./plots/cost_bar.{self.image_format}">
        </p>

        <h2>Fits</h2>
        <p>
        {fit_images}
        </p>

        <h2>Optimization results</h2>
        {self.optres.df_fits.to_html()}
        </body>
        </html>
        """
        with open(path, "w", encoding="utf-8") as f_out:
            f_out.write(html)

    def _create_mpl_figure(
        self, width: float = 5.0, height: float = 5.0, layout: str = "constrained"
    ) -> tuple[Figure, Axes]:
        """Create matplotlib figure."""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height), layout=layout)
        # fig.subplots_adjust(left=0.2, bottom=0.1)

        return fig, ax

    def _save_mpl_figure(self, fig: Figure, path: Path) -> None:
        """Save matplotlib figure to path."""
        if self.show_plots:
            plt.show()
        if path is not None:
            # fig.savefig(path, bbox_inches="tight")
            fig.savefig(path)
        plt.close(fig)

    @timeit
    def plot_fit(self, output_dir: Path, x: np.ndarray) -> None:
        """Plot fitted curves with experimental data for given parameter set x.

        Creates an overview of all fit mappings.

        :param output_dir: path to figures
        :param x: parameters to evaluate

        :return: None
        """
        # residual data and simulations of optimal parameters
        res_data = self.residual_data(x)

        for k, mapping_id in enumerate(self.op.mapping_keys):
            fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            # global reference data
            sid = self.op.experiment_keys[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            y_ref_err_type = self.op.y_errors_type[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

            for ax in [ax1, ax2]:
                ax.set_title(f"{sid} {mapping_id}")
                ax.set_ylabel(y_id)
                ax.set_xlabel(x_id)

                # calculated data in residuals
                x_obs = res_data["x_obs"][k]
                y_obs = res_data["y_obs"][k]

                # FIXME: add residuals

                # plot data
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
                # plot simulation
                ax.plot(
                    x_obs.values, y_obs.values, "-", color="blue", label="observable"
                )

                xdelta = np.max(x_ref) - np.min(x_ref)
                ax.set_xlim(
                    left=np.min(x_ref) - 0.1 * xdelta,
                    right=np.max(x_ref) + 0.1 * xdelta,
                )
                ax.legend()

            ax2.set_yscale("log")
            ax2.set_ylim(bottom=self._log_limits(y_ref, factor=1.0 / 0.3)[0])

            self._save_mpl_figure(
                fig, path=output_dir / f"{sid}_{mapping_id}.{self.image_format}"
            )

    @timeit
    def plot_fit_residual(self, output_dir: Path, x: np.ndarray) -> None:
        """Plot resulting fit for all individual fit mappings.

        This consists of
        - data
        - prediction
        - residuals
        - weighed residuals squared

        For better analysis log and linear results are depicted.
        :param x: parameters to evaluate
        """
        res_data = self.residual_data(x)

        for k, mapping_id in enumerate(self.op.mapping_keys):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                nrows=2, ncols=2, figsize=(10, 10)
            )

            # global reference data
            sid = self.op.experiment_keys[k]
            # weights = self.op.weights_points[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

            # calculated data in residuals
            x_obs = res_data["x_obs"][k]
            y_obs = res_data["y_obs"][k]
            y_obsip = res_data["y_obsip"][k]

            res = res_data["residuals"][k]
            res_weighted = res_data["residuals_weighted"][k]
            res_weighted2 = np.power(res_weighted, 2)

            for ax in (ax1, ax3):
                ax.axhline(y=0, color="black")
                ax.set_ylabel(y_id)
            for ax in (ax3, ax4):
                ax.set_xlabel(x_id)

            for ax in (ax1, ax2):
                ax.set_title(f"{sid}.{mapping_id}")
                plt.setp(ax.get_xticklabels(), visible=False)

                # residuals
                ax.plot(x_ref, res, "o", color="darkorange", label="residuals")
                ax.fill_between(
                    x_ref,
                    res,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkorange",
                    label="__nolabel__",
                )

                # prediction
                ax.plot(
                    x_obs.values, y_obs.values, "-", color="blue", label="observable"
                )
                ax.plot(x_ref, y_obsip, "o", color="blue", label="interpolation")

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

            for ax in (ax3, ax4):
                ax.plot(
                    x_ref,
                    res_weighted2,
                    "o",
                    color="red",
                    label="(weighted residuals)^2",
                )
                ax.fill_between(
                    x_ref,
                    res_weighted2,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkred",
                    label="__nolabel__",
                )

                ax.set_xlim(ax1.get_xlim())

            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xlim(right=1.1 * np.max(x_ref))
                ax.legend()

            for ax in (ax2, ax4):
                ax.set_yscale("log")

            self._save_mpl_figure(
                fig=fig,
                path=output_dir / f"fit_{sid}_{mapping_id}.{self.image_format}",
            )

    def _cost_df(self, x: np.ndarray) -> pd.DataFrame:
        """Calculate cost dataframe for given parameter set."""
        res_data = self.residual_data(x)
        data = []
        for k, _ in enumerate(self.op.mapping_keys):
            data.append(
                {
                    "id": f"{self.op.experiment_keys[k]}_{self.op.mapping_keys[k]}",
                    "experiment": self.op.experiment_keys[k],
                    "mapping": self.op.mapping_keys[k],
                    "cost": res_data["cost"][k],
                    "weight_curve": res_data["weights_curve"][k],
                }
            )

        return pd.DataFrame(
            data,
            columns=pd.Index(["id", "experiment", "mapping", "cost", "weight_curve"]),
        )

    def _datapoints_df(self, x: np.ndarray) -> pd.DataFrame:
        """Calculate data point dataframe for given parameter set."""
        res_data = self.residual_data(x)

        data = []
        for k, _ in enumerate(self.op.mapping_keys):
            experiment = self.op.experiment_keys[k]
            mapping = self.op.mapping_keys[k]

            x_ref = self.op.x_references[k]
            y_ref_err = self.op.y_errors[k]
            y_ref_err_type = self.op.y_errors_type[k]
            y_ref = self.op.y_references[k]
            y_obs = res_data["y_obsip"][k]
            residuals = res_data["residuals"][k]
            for ix in range(len(y_obs)):
                y_err = np.nan if not y_ref_err_type else y_ref_err[ix]

                data.append(
                    {
                        "experiment": experiment,
                        "mapping": mapping,
                        "x_ref": x_ref[ix],
                        "y_ref": y_ref[ix],
                        "y_ref_err": y_err,
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

    kwargs_scatter: ClassVar[dict[str, Any]] = {
        "markersize": "10",
        "markeredgecolor": "black",
        "alpha": 0.7,
        "linestyle": "",
        "marker": "o",
    }

    @timeit
    def plot_datapoint_scatter(self, x: np.ndarray, path: Path) -> None:
        """Plot cost scatter plot.

        Compares cost of model parameters to the given parameter set.
        """
        fig, ax = self._create_mpl_figure()
        dp: pd.DataFrame = self._datapoints_df(x=x)

        # FIXME: plot error bars
        # plot lines
        min_dp, max_dp = self._log_limits(dp.y_ref, dp.y_obs)

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

        # plot data
        for experiment in sorted(dp.experiment.unique()):
            ax.plot(
                dp.y_ref[dp.experiment == experiment].values,
                dp.y_obs[dp.experiment == experiment].values,
                # yerr=dp.y_ref_err,
                **self.kwargs_scatter,
            )

        # annotations
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.asarray(dp.y_ref.values, dtype=float) / np.asarray(
                dp.y_obs.values, dtype=float
            )
        for k in range(len(dp)):
            # plot labels for datapoints far away
            ratio = ratios[k]
            if not np.isfinite(ratio):
                continue
            if ratio > 10 or ratio < 1 / 10:
                ax.annotate(
                    dp.experiment.values[k],
                    xy=(
                        dp.y_ref.values[k],
                        dp.y_obs.values[k],
                    ),
                    fontsize="x-small",
                    alpha=0.9,
                    # textcoords="offset fontsize"
                )
        ax.set_xlabel("Experiment $y_{i,k}$", fontweight="bold")
        ax.set_ylabel("Prediction $f(x_{i,k})$", fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_dp, max_dp)
        ax.grid()
        if self.show_titles:
            ax.set_title("Data points")
        self._save_mpl_figure(fig=fig, path=path)

    @timeit
    def plot_residual_scatter(self, x: np.ndarray, path: Path) -> None:
        """Plot residual plot."""
        fig, ax = self._create_mpl_figure()
        dp: pd.DataFrame = self._datapoints_df(x=x)

        xdata = dp.y_ref
        ydata = dp.residual / dp.y_ref

        for experiment in sorted(dp.experiment.unique()):
            ax.plot(
                xdata[dp.experiment == experiment].values,
                ydata[dp.experiment == experiment].values,
                **self.kwargs_scatter,
            )

        min_res = np.min(ydata)
        max_res = np.max(ydata)

        ax.fill_between(
            [min_res * 0.5, max_res * 2, max_res * 2, min_res * 0.5],
            [-0.5, -0.5, 0.5, 0.5],
            color="lightgray",
        )
        ax.plot(
            [min_res * 0.5, max_res * 2],
            [0, 0],
            "--",
            color="black",
        )

        for k in range(len(dp)):
            # # errorbars
            # if dp.y_ref_err is not None:
            #     ax.errorbar(
            #         xdata[k], ydata[k],
            #         yerr=dp.y_ref_err[k]/ydata[k],
            #         linestyle="",
            #         marker="",
            #         # label="model",
            #         color="black",
            #         markersize="1",
            #         alpha=0.9,
            #     )

            if np.abs(ydata[k]) > 0.5:
                ax.annotate(
                    dp.experiment.values[k],
                    xy=(
                        xdata[k],
                        ydata[k],
                    ),
                    fontsize="x-small",
                    alpha=0.7,
                )
        ax.set_xlabel("Experiment $y_{i,k}$", fontweight="bold")
        ax.set_ylabel(
            "Relative residual $\\frac{f(x_{i,k})-y_{i,k}}{y_{i,k}}$", fontweight="bold"
        )
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.grid()
        if self.show_titles:
            ax.set_title("Residuals")
        self._save_mpl_figure(fig=fig, path=path)

    @timeit
    def plot_cost_bar(self, x: np.ndarray, path: Path) -> None:
        """Plot cost bar plot.

        Compare costs of all curves.
        """
        costs_x: pd.DataFrame = self._cost_df(x=x)

        fig: Figure
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
        fig.subplots_adjust(left=0.3)

        if self.show_titles:
            fig.suptitle("Curve costs and weights")

        position = list(range(len(costs_x)))
        ticklabels = [
            f"{costs_x.experiment[k]}|{costs_x.mapping[k]}" for k in range(len(costs_x))
        ]

        ax1.barh(position, costs_x.cost, color="black", alpha=0.8)

        ax1.set_yticks(position)
        ax1.set_yticklabels(
            ticklabels,
            # rotation=60,
            ha="right",
            fontdict={"fontsize": 8},
        )
        ax1.grid(True, axis="x")
        ax1.set_xlabel("Cost")
        ax1.set_xscale("log")

        ax2.barh(position, costs_x.weight_curve, color="tab:blue", alpha=0.8)

        ax2.set_yticks(position)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.grid(True, axis="x")
        ax2.set_xlabel("Weight curve: $w_{k}$")
        # ax1.set_xscale("log")

        self._save_mpl_figure(fig=fig, path=path)

    def plot_residual_boxplot(self, x: np.ndarray, path: Path) -> None:
        """Plot residual boxplot.

        Compare costs of all curves.
        """
        costs_x: pd.DataFrame = self._cost_df(x=x)

        fig, ax = self._create_mpl_figure()
        if self.show_titles:
            ax.set_title("Residual contribution")

        position = list(range(1, len(costs_x) + 1))
        ticklabels = [
            f"{costs_x.experiment[k]}|{costs_x.mapping[k]}" for k in range(len(costs_x))
        ]

        res_data = self.residual_data(x)

        box_data = []
        for k, _ in enumerate(self.op.mapping_keys):
            res_weighted = res_data["residuals_weighted"][k]
            res_weighted2 = np.power(res_weighted, 2)
            box_data.append(res_weighted2)
            ax.plot(
                res_weighted2,
                (k + 0.7) * np.ones_like(res_weighted2),
                linestyle="",
                marker="s",
                markeredgecolor="black",
                color="tab:blue",
                markersize=3,
            )

        ax.boxplot(box_data, orientation="horizontal")

        ax.set_yticks(position)
        ax.set_yticklabels(
            ticklabels,
            # rotation=60,
            ha="right",
            fontdict={"fontsize": 8},
        )
        ax.grid(True, axis="x")
        ax.set_xlabel(
            "Weighted residuals^2\n$(w_{k} \\cdot w_{i,k} (f(x_{i,k}) - y_{i,k}))^2$"
        )
        ax.set_xscale("log")
        self._save_mpl_figure(fig=fig, path=path)

    @timeit
    def plot_cost_scatter(self, x: np.ndarray, path: Path) -> None:
        """Plot cost scatter plot.

        Compares cost of model parameters to the given parameter set.
        """
        costs_xmodel: pd.DataFrame = self._cost_df(x=self.op.xmodel)
        costs_x: pd.DataFrame = self._cost_df(x=x)

        min_cost = np.min(
            [
                np.min(costs_xmodel.cost),
                np.min(costs_x.cost),
            ]
        )
        max_cost = np.max(
            [
                np.max(costs_xmodel.cost),
                np.max(costs_x.cost),
            ]
        )

        fig, ax = self._create_mpl_figure()
        ax.plot(
            [min_cost * 0.5, max_cost * 2],
            [min_cost * 0.5, max_cost * 2],
            "--",
            color="black",
        )
        if self.show_titles:
            ax.set_title("Cost improvement")

        for k, _ in enumerate(self.op.experiment_keys):
            ax.plot(
                costs_xmodel.cost[k],
                costs_x.cost[k],
                linestyle="",
                marker="o",
                # label="model",
                color="tab:red"
                if costs_xmodel.cost[k] < costs_x.cost[k]
                else "tab:blue",
                markersize="10",
                alpha=0.8,
            )

        for k, exp_key in enumerate(self.op.experiment_keys):
            ax.annotate(
                exp_key,
                xy=(
                    costs_xmodel.cost[k],
                    costs_x.cost[k],
                ),
                fontsize="x-small",
                alpha=0.7,
            )
        ax.set_xlabel("Initial cost")
        ax.set_ylabel("Cost after fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="tab:red",
                label="Increased cost",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="tab:blue",
                label="Decreased cost",
                markersize=10,
            ),
        ]

        ax.legend(handles=legend_elements)
        self._save_mpl_figure(fig=fig, path=path)

    @timeit
    def plot_waterfall(self, path: Path) -> None:
        """Create waterfall plot for the fit results.

        Plots the optimization runs sorted by cost.
        """
        fig, ax = self._create_mpl_figure()
        if self.show_titles:
            ax.set_title("Waterfall plot")
        ax.plot(
            range(self.optres.size),
            1 + (self.optres.df_fits.cost.values - self.optres.df_fits.cost.values[0]),
            "-o",
            color="black",
        )
        ax.set_xlabel("Index (ordered optimizer run)")
        ax.set_ylabel("Offset cost value (relative to best start)")
        ax.set_yscale("log")
        self._save_mpl_figure(fig, path=path)

    @timeit
    def plot_traces(self, path: Path) -> None:
        """Plot optimization traces.

        Optimization time course of costs.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        if self.show_titles:
            ax.set_title("Optimization traces")
        for run in range(self.optres.size):
            df_run = self.optres.df_traces[self.optres.df_traces.run == run]
            ax.plot(range(len(df_run)), df_run.cost.values, "-", alpha=0.8)

        for run in range(self.optres.size):
            df_run = self.optres.df_traces[self.optres.df_traces.run == run]
            # plot final optimization cost of trace
            if len(df_run) > 0:
                ax.plot(
                    len(df_run) - 1,
                    df_run.cost.values[-1],
                    "o",
                    color="black",
                    alpha=0.8,
                )

        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Cost")
        ax.set_yscale("log")

        self._save_mpl_figure(fig, path=path)
