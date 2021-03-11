"""Analysis of fitting results."""

import datetime
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.optimize import OptimizeResult

from sbmlsim.fit.fit import logger
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem, OptimizationAnalysis
from sbmlsim.plot.plotting_matplotlib import plt
from sbmlsim.serialization import ObjectJSONEncoder, from_json, to_json
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit

import datetime
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.optimize import OptimizeResult

from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.plot.plotting_matplotlib import plt
from sbmlsim.serialization import ObjectJSONEncoder, from_json, to_json
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


class OptimizationResultAnalysis:

    @staticmethod
    def _save_fig(fig: Figure, path: Path, show_plots: bool = True):
        if show_plots:
            plt.show()
        if path:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @timeit
    def plot_waterfall(self, path: Path = None, show_plots: bool = True):
        """Create waterfall plot for the fit results.

        Plots the optimization runs sorted by cost.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(
            range(self.size),
            1 + (self.df_fits.cost - self.df_fits.cost.iloc[0]),
            "-o",
            color="black",
        )
        ax.set_xlabel("index (Ordered optimizer run)")
        ax.set_ylabel("Offset cost value (relative to best start)")
        ax.set_yscale("log")
        ax.set_title("Waterfall plot")
        self._save_fig(fig, path=path, show_plots=show_plots)

    @timeit
    def plot_traces(self, path: Path = None, show_plots: bool = True) -> None:
        """Plot optimization traces.

        Optimization time course of costs.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        for run in range(self.size):
            df_run = self.df_traces[self.df_traces.run == run]
            ax.plot(range(len(df_run)), df_run.cost, "-", alpha=0.8)
        for run in range(self.size):
            df_run = self.df_traces[self.df_traces.run == run]
            ax.plot(
                len(df_run) - 1, df_run.cost.iloc[-1], "o", color="black", alpha=0.8
            )

        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.set_yscale("log")
        ax.set_title("Traces")

        self._save_fig(fig, path=path, show_plots=show_plots)

    @timeit
    def plot_correlation(
        self,
        path: Path = None,
        show_plots: bool = True,
    ):
        """Plot correlation of parameters for analysis."""
        df = self.df_fits

        pids = [p.pid for p in self.parameters]
        sns.set(style="ticks", color_codes=True)
        # sns_plot = sns.pairplot(data=df[pids + ["cost"]], hue="cost", corner=True)

        npars = len(pids)
        # x0 =
        fig, axes = plt.subplots(
            nrows=npars, ncols=npars, figsize=(5 * npars, 5 * npars)
        )
        cost_normed = df.cost - df.cost.min()
        cost_normed = 1 - cost_normed / cost_normed.max()
        # print("cost_normed", cost_normed)
        size = np.power(15 * cost_normed, 2)

        bound_kwargs = {"color": "darkgrey", "linestyle": "--", "alpha": 1.0}

        for kx, pidx in enumerate(pids):
            for ky, pidy in enumerate(pids):
                if npars == 1:
                    ax = axes
                else:
                    ax = axes[ky][kx]

                # optimal values
                if kx > ky:
                    ax.set_xlabel(pidx)
                    # ax.set_xlim(self.parameters[kx].lower_bound, self.parameters[kx].upper_bound)
                    ax.axvline(x=self.parameters[kx].lower_bound, **bound_kwargs)
                    ax.axvline(x=self.parameters[kx].upper_bound, **bound_kwargs)
                    ax.set_ylabel(pidy)
                    # ax.set_ylim(self.parameters[ky].lower_bound, self.parameters[ky].upper_bound)
                    ax.axhline(y=self.parameters[ky].lower_bound, **bound_kwargs)
                    ax.axhline(y=self.parameters[ky].upper_bound, **bound_kwargs)

                    # start values
                    xall = []
                    yall = []
                    xstart_all = []
                    ystart_all = []
                    for ks in range(len(size)):
                        x = df.x[ks][kx]
                        y = df.x[ks][ky]
                        xall.append(x)
                        yall.append(y)
                        if "x0" in df.columns:
                            xstart = df.x0[ks][kx]
                            ystart = df.x0[ks][ky]
                            xstart_all.append(xstart)
                            ystart_all.append(ystart)

                    # start point
                    ax.plot(
                        xstart_all,
                        ystart_all,
                        "^",
                        color="black",
                        markersize=2,
                        alpha=0.5,
                    )
                    # optimal values
                    ax.scatter(
                        df[pidx], df[pidy], c=df.cost, s=size, alpha=0.9, cmap="jet"
                    ),

                    ax.plot(
                        self.xopt[kx],
                        self.xopt[ky],
                        "s",
                        color="darkgreen",
                        markersize=30,
                        alpha=0.7,
                    )

                if kx == ky:
                    ax.set_xlabel(pidx)
                    ax.axvline(x=self.parameters[kx].lower_bound, **bound_kwargs)
                    ax.axvline(x=self.parameters[kx].upper_bound, **bound_kwargs)
                    # ax.set_xlim(self.parameters[kx].lower_bound,
                    #            self.parameters[kx].upper_bound)
                    ax.set_ylabel("cost")
                    ax.plot(
                        df[pidx],
                        df.cost,
                        color="black",
                        marker="s",
                        linestyle="None",
                        alpha=1.0,
                    )

                # traces (walk through cost function)
                if kx < ky:
                    ax.set_xlabel(pidy)
                    # ax.set_xlim(self.parameters[ky].lower_bound, self.parameters[ky].upper_bound)
                    ax.axvline(x=self.parameters[ky].lower_bound, **bound_kwargs)
                    ax.axvline(x=self.parameters[ky].upper_bound, **bound_kwargs)
                    ax.set_ylabel(pidx)
                    # ax.set_ylim(self.parameters[kx].lower_bound, self.parameters[kx].upper_bound)
                    ax.axhline(y=self.parameters[kx].lower_bound, **bound_kwargs)
                    ax.axhline(y=self.parameters[kx].upper_bound, **bound_kwargs)

                    # ax.plot([ystart, y], [xstart, x], "-", color="black", alpha=0.7)

                    for run in range(self.size):
                        df_run = self.df_traces[self.df_traces.run == run]
                        # ax.plot(df_run[pidy], df_run[pidx], '-', color="black", alpha=0.3)
                        ax.scatter(
                            df_run[pidy],
                            df_run[pidx],
                            c=df_run.cost,
                            cmap="jet",
                            marker="s",
                            alpha=0.8,
                        )

                    # end point
                    # ax.plot(yall, xall, "o", color="black", markersize=10, alpha=0.9)
                    ax.plot(
                        self.xopt[ky],
                        self.xopt[kx],
                        "s",
                        color="darkgreen",
                        markersize=30,
                        alpha=0.7,
                    )

                ax.set_xscale("log")
                if kx != ky:
                    ax.set_yscale("log")
                if kx == ky:
                    ax.set_yscale("log")

        # correct scatter limits
        for kx, _pidx in enumerate(pids):
            for ky, _pidy in enumerate(pids):
                if kx < ky:
                    axes[ky][kx].set_xlim(axes[kx][ky].get_xlim())
                    axes[ky][kx].set_ylim(axes[kx][ky].get_ylim())

        self._save_fig(fig=fig, path=path, show_plots=show_plots)


def process_optimization_result(
    opt_result: OptimizationResult,
    output_path: Path,
    problem: OptimizationProblem = None,
    show_plots=True,
    fitting_type=None,
    weighting_local=None,
    residual_type=None,
    variable_step_size=True,
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
):
    """Process the optimization results.

    Creates reports and stores figures and results.
    """
    # the results directory uses the hash of the OptimizationResult
    results_dir: Path = output_path / opt_result.sid
    if not results_dir.exists():
        logger.warning(f"create output directory: '{results_dir}'")
        results_dir.mkdir(parents=True, exist_ok=True)

    problem_info = ""
    if problem:
        # FIXME: problem not initialized on multi-core and no simulator is assigned.
        # This should happen automatically, to ensure correct behavior
        problem.initialize(
            fitting_strategy=fitting_type,
            weighting_points=weighting_local,
            residual_type=residual_type,
        )
        problem.set_simulator(simulator=SimulatorSerial())
        problem.variable_step_size = variable_step_size
        problem.absolute_tolerance = absolute_tolerance
        problem.relative_tolerance = relative_tolerance

        problem_info = problem.report(
            path=None,
            print_output=False,
        )

    # write report
    result_info = opt_result.report(
        path=None,
        print_output=True,
    )
    info = problem_info + result_info
    with open(results_dir / "00_report.txt", "w") as f_report:
        f_report.write(info)

    opt_result.to_json(path=results_dir / "01_optimization_result.json")
    opt_result.to_tsv(path=results_dir / "01_optimization_result.tsv")

    if opt_result.size > 1:
        opt_result.plot_waterfall(
            path=results_dir / "02_waterfall.svg", show_plots=show_plots
        )
    opt_result.plot_traces(path=results_dir / "02_traces.svg", show_plots=show_plots)

    # plot top fit
    if problem:
        xopt = opt_result.xopt
        optimization_analyzer = OptimizationAnalysis(optimization_problem=problem)

        df_costs = optimization_analyzer.plot_costs(
            x=xopt, path=results_dir / "03_cost_improvement.svg", show_plots=show_plots
        )
        df_costs.to_csv(results_dir / "03_cost_improvement.tsv", sep="\t", index=False)

        optimization_analyzer.plot_fits(
            x=xopt, path=results_dir / "05_fits.svg", show_plots=show_plots
        )
        optimization_analyzer.plot_residuals(
            x=xopt, output_path=results_dir, show_plots=show_plots
        )

    if opt_result.size > 1:
        opt_result.plot_correlation(
            path=results_dir / "04_parameter_correlation", show_plots=show_plots
        )

    # TODO: overall HTML report for simple overview



class OptimizationAnalysis:
    """Class for creating plots and results."""

    def __init__(self, optimization_problem: OptimizationProblem):
        self.op = optimization_problem

    @staticmethod
    def _save_fig(fig, path: Path, show_plots: bool = True):
        if show_plots:
            plt.show()
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @timeit
    def plot_fits(self, x, path: Path = None, show_plots: bool = True):
        """Plot fitted curves with experimental data.

        Overview of all fit mappings.

        :param x: parameters to evaluate
        :return:
        """
        n_plots = len(self.op.mapping_keys)
        fig, axes = plt.subplots(
            nrows=n_plots, ncols=2, figsize=(10, 5 * n_plots), squeeze=False
        )

        # residual data and simulations of optimal paraemters
        res_data = self.op.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.op.mapping_keys):

            # global reference data
            sid = self.op.experiment_keys[k]
            mapping_id = self.op.mapping_keys[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            y_ref_err_type = self.op.y_errors_type[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

            for ax in axes[k]:
                ax.set_title(f"{sid} {mapping_id}")
                ax.set_xlabel(x_id)
                ax.set_ylabel(y_id)

                # calculated data in residuals
                x_obs = res_data["x_obs"][k]
                y_obs = res_data["y_obs"][k]

                # FIXME: add residuals

                # plot data
                if y_ref_err is None:
                    ax.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label=f"reference_data ± {y_ref_err_type}",
                    )
                # plot simulation
                ax.plot(x_obs, y_obs, "-", color="blue", label="observable")
                ax.legend()

            axes[k][1].set_yscale("log")
            axes[k][1].set_ylim(bottom=0.3 * np.nanmin(y_ref))

        self._save_fig(fig, path=path, show_plots=show_plots)

    @timeit
    def plot_residuals(self, x, output_path: Path = None, show_plots: bool = True):
        """Plot residual data.

        :param res_data_start: initial residual data
        :return:
        """
        titles = ["model", "fit"]
        res_data_start = self.op.residuals(
            xlog=np.log10(self.op.xmodel), complete_data=True
        )
        res_data_fit = self.op.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.op.mapping_keys):
            fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(
                nrows=3, ncols=2, figsize=(10, 10)
            )

            axes = [(a1, a3, a5), (a2, a4, a6)]
            if titles is None:
                titles = ["Model", "Fit"]

            # global reference data
            sid = self.op.experiment_keys[k]
            mapping_id = self.op.mapping_keys[k]
            # weights = self.op.weights_points[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

            for kdata, res_data in enumerate([res_data_start, res_data_fit]):
                ax1, ax2, ax3 = axes[kdata]
                title = titles[kdata]

                # calculated data in residuals

                x_obs = res_data["x_obs"][k]
                y_obs = res_data["y_obs"][k]
                y_obsip = res_data["y_obsip"][k]

                res = res_data["residuals"][k]
                res_weighted = res_data["residuals_weighted"][k]
                res_abs = res_data["res_abs"][k]
                # res_rel = res_data["res_rel"][k]

                cost = res_data["cost"][k]

                for ax in (ax1, ax2, ax3):
                    ax.axhline(y=0, color="black")
                    ax.set_ylabel(y_id)
                ax3.set_xlabel(x_id)

                if y_ref_err is None:
                    ax1.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax1.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label="reference_data",
                    )

                ax1.plot(x_obs, y_obs, "-", color="blue", label="observable")
                ax1.plot(x_ref, y_obsip, "o", color="blue", label="interpolation")
                for ax in (ax1, ax2):
                    ax.plot(x_ref, res_abs, "o", color="darkorange", label="obs-ref")
                ax1.fill_between(
                    x_ref,
                    res_abs,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkorange",
                    label="__nolabel__",
                )

                ax2.plot(
                    x_ref,
                    res_weighted,
                    "o",
                    color="darkgreen",
                    label="weighted residuals",
                )
                ax2.fill_between(
                    x_ref,
                    res_weighted,
                    np.zeros_like(res_weighted),
                    alpha=0.4,
                    color="darkgreen",
                    label="__nolabel__",
                )

                res_weighted2 = np.power(res_weighted, 2)
                ax3.plot(
                    x_ref,
                    res_weighted2,
                    "o",
                    color="darkred",
                    label="(weighted residuals)^2",
                )
                ax3.fill_between(
                    x_ref,
                    res_weighted2,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkred",
                    label="__nolabel__",
                )

                for ax in (ax1, ax2):
                    plt.setp(ax.get_xticklabels(), visible=False)

                # ax3.set_xlabel("x")
                for ax in (ax2, ax3):
                    ax.set_xlim(ax1.get_xlim())

                if title:
                    full_title = "{}_{}: {} (cost={:.3e})".format(
                        sid, mapping_id, title, cost
                    )
                    ax1.set_title(full_title)
                for ax in (ax1, ax2, ax3):
                    # plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.set_ylabel("y")
                    # ax.set_yscale("log")
                    ax.legend()

            # adapt axes
            if res_data_fit is not None:
                for axes in [(a1, a2), (a3, a4), (a5, a6)]:
                    ax1, ax2 = axes
                    # ylim1 = ax1.get_ylim()
                    # ylim2 = ax2.get_ylim()
                    # # for ax in axes:
                    # #    ax.set_ylim([min(ylim1[0], ylim2[0]), max(ylim1[1],ylim2[1])])

            if show_plots:
                plt.show()
            if output_path is not None:
                fig.savefig(
                    output_path / f"06_residuals_{sid}_{mapping_id}.svg",
                    bbox_inches="tight",
                )

    @timeit
    def plot_costs(self, x, path: Path = None, show_plots: bool = True) -> pd.DataFrame:
        """Plot cost function comparison.

        # FIXME: separate calculation of cost DataFrame
        """
        xparameters = {
            # model parameters
            "model": self.op.xmodel,
            # initial values of fit parameter
            "initial": self.op.x0,
            # provided parameters
            "fit": x,
        }
        data = []
        costs = {}
        for key, xpar in xparameters.items():
            res_data = self.op.residuals(xlog=np.log10(xpar), complete_data=True)
            costs[key] = res_data["cost"]
            for k, _ in enumerate(self.op.mapping_keys):
                data.append(
                    {
                        "id": f"{self.op.experiment_keys[k]}_{self.op.mapping_keys[k]}",
                        "experiment": self.op.experiment_keys[k],
                        "mapping": self.op.mapping_keys[k],
                        "cost": res_data["cost"][k],
                        "type": key,
                    }
                )

        cost_df = pd.DataFrame(
            data, columns=["id", "experiment", "mapping", "cost", "type"]
        )

        min_cost = np.min(
            [
                np.min(costs["fit"]),
                np.min(costs["model"]),
                np.min(costs["initial"]),
            ]
        )
        max_cost = np.max(
            [
                np.max(costs["fit"]),
                np.max(costs["model"]),
                np.max(costs["initial"]),
            ]
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        ax.plot(
            [min_cost * 0.5, max_cost * 2],
            [min_cost * 0.5, max_cost * 2],
            "--",
            color="black",
        )
        # ax.plot(costs["initial"], costs["fit"], linestyle="", marker="s", label="initial")
        ax.plot(
            costs["model"],
            costs["fit"],
            linestyle="",
            marker="o",
            label="model",
            color="black",
            markersize="10",
            alpha=0.8,
        )

        for k, exp_key in enumerate(self.op.experiment_keys):
            ax.annotate(
                exp_key,
                xy=(
                    costs["model"][k],
                    costs["fit"][k],
                ),
                fontsize="x-small",
                alpha=0.7,
            )

        ax.set_xlabel("initial cost")
        ax.set_ylabel("fit cost")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.set_xlim(min_cost * 0.5, max_cost * 2)
        ax.set_ylim(min_cost * 0.5, max_cost * 2)
        ax.legend()

        # sns.set_color_codes("pastel")
        # sns.barplot(ax=ax, x="cost", y="id", hue="type", data=cost_df)
        # ax.set_xscale("log")
        if show_plots:
            plt.show()
        if path:
            fig.savefig(path, bbox_inches="tight")

        return cost_df
