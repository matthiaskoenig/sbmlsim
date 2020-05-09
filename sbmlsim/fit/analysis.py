from typing import List, Dict, Iterable, Set
import numpy as np
import scipy
from scipy import optimize
from scipy import interpolate
from collections import defaultdict
from pathlib import Path
from enum import Enum


import time
import pandas as pd
import logging
from dataclasses import dataclass

from sbmlsim.data import Data
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.utils import timeit
from sbmlsim.plot.plotting_matplotlib import plt  # , GridSpec
import seaborn as sns
from scipy.optimize import OptimizeResult

from .objects import FitParameter


logger = logging.getLogger(__name__)


class OptimizeTrajectory(object):
    pass


class OptimizationResult(object):
    """Result of optimization problem"""

    def __init__(self, parameters: List[FitParameter], fits: List[OptimizeResult],
                 trajectories: List[OptimizeTrajectory]):
        """

        :param problem: definiition of optimization problem
        :param fits:
        :param trajectories:
        """
        self.parameters = parameters
        self.fits = fits
        self.trajectories = trajectories

        # create data frame from results
        self.df = OptimizationResult.process_fits(self.parameters, self.fits)

    @property
    def size(self):
        """Number of optimizations in result."""
        return len(self.df)

    @property
    def xopt(self):
        """Optimal parameters"""
        return self.df.x.iloc[0]

    @staticmethod
    def process_fits(parameters: List[FitParameter], fits: List[OptimizeResult]):
        """Process the optimization results."""
        results = []
        pids = [p.pid for p in parameters]
        # print(fits)
        for k, fit in enumerate(fits):
            res = {
                'index': k,
                # 'status': fit.status,
                'success': fit.success,
                'duration': fit.duration,
                'cost': fit.cost,
                 # 'optimality': fit.optimality,
                'message': fit.message if hasattr(fit, "message") else None
            }
            # add parameter columns
            for k, pid in enumerate(pids):
                res[pid] = fit.x[k]
            res['x'] = fit.x
            res['x0'] = fit.x0

            results.append(res)
        df = pd.DataFrame(results)
        df.sort_values(by=["cost"], inplace=True)
        # reindex
        df = df.reindex(range(len(df)))

        return df

    def report(self, output_path: Path=None):
        """ Readable report of optimization. """
        pd.set_option('display.max_columns', None)
        info = [
            "-" * 80,
            str(self.df),
            "-" * 80,
            "Optimal parameters:",
        ]

        xopt = self.xopt
        fitted_pars = {}
        for k, p in enumerate(self.parameters):
            fitted_pars[p.pid] = (xopt[k], p.unit)

        for key, value in fitted_pars.items():
            info.append(
                "\t'{}': Q_({}, '{}'),".format(key, value[0], value[1])
            )
        info.append("-" * 80)
        info = "\n".join(info)
        print(info)
        if output_path is not None:
            filepath = output_path / "fit_report.txt"
            with open(filepath, "w") as fout:
                fout.write(info)

    def plot_waterfall(self, output_path=None):
        """Creates waterfall plot for the fit results.

        Plots the optimization runs sorted by cost.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.plot(range(self.size), 1 + (self.df.cost-self.cost.iloc[0]), '-o', color="black")
        ax.set_xlabel("index (Ordered optimizer run)")
        ax.set_ylabel("Offsetted cost value (relative to best start)")
        # ax.set_yscale("log")
        ax.set_title("Waterfall plot")
        plt.show()

        if output_path is not None:
            filepath = output_path / "01_waterfall.svg"
            fig.savefig(filepath, bbox_inches="tight")

    def plot_correlation(self, output_path=None):
        """Process the optimization results."""
        df = self.fit_results

        pids = [p.pid for p in self.parameters]
        sns.set(style="ticks", color_codes=True)
        # sns_plot = sns.pairplot(data=df[pids + ["cost"]], hue="cost", corner=True)

        npars = len(pids)
        # x0 =
        fig, axes = plt.subplots(nrows=npars, ncols=npars, figsize=(5*npars, 5*npars))

        cost_normed = (df.cost - df.cost.min())
        cost_normed = 1 - cost_normed/cost_normed.max()
        print("cost_normed", cost_normed)
        size = np.power(15*cost_normed, 2)

        # FIXME: plot bounds
        # FIXME: plot cost of initial guesses (both should be in data frame)

        for kx, pidx in enumerate(pids):
            for ky, pidy in enumerate(pids):
                if npars == 1:
                    ax = axes
                else:
                    ax = axes[ky][kx]
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                # optimal values
                if kx > ky:
                    ax.set_xlabel(pidx)
                    ax.set_ylabel(pidy)
                    # optimal values
                    ax.scatter(df[pidx], df[pidy], c=df.cost, s=size, alpha=0.75, cmap="jet")

                    ax.plot(self.xopt[kx], self.xopt[ky], "s",
                                      color="darkgreen", markersize=30,
                                      alpha=0.7)

                if kx == ky:
                    ax.set_ylabel(pidy)
                    ax.set_ylabel("cost")
                    ax.scatter(df[pidx], df.cost, color="black", marker="_", alpha=1.0)

                # trajectory
                if kx < ky:
                    ax.set_xlabel(pidy)
                    ax.set_ylabel(pidx)
                    # start values
                    xall = []
                    yall = []
                    xstart_all = []
                    ystart_all = []
                    for ks in range(len(size)):
                        # FIXME: use iloc!
                        x = df.x[ks][kx]
                        y = df.x[ks][ky]
                        xall.append(x)
                        yall.append(y)
                        if 'x0' in df.columns:
                            xstart = df.x0[ks][kx]
                            ystart = df.x0[ks][ky]
                            xstart_all.append(xstart)
                            ystart_all.append(ystart)

                            ax.plot([ystart, y], [xstart, x], "-", color="darkblue", alpha=0.7)

                    # start point
                    ax.plot(ystart_all, xstart_all, "^", color="black", markersize=10, alpha=0.9)
                    # end point
                    ax.plot(yall, xall, "o", color="black", markersize=10, alpha=0.9)


        plt.show()
        if output_path is not None:
            filepath = output_path / "03_parameter_correlation.svg"
            fig.savefig(filepath, bbox_inches="tight")


