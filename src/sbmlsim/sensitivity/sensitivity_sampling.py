"""
Sampling-based sensitivity and uncertainty analysis.

This module implements a sampling-based sensitivity analysis in which model
parameters are varied simultaneously according to predefined bounds and
probability assumptions, and the resulting distribution of model outputs is
analyzed statistically.

The primary purpose of this approach is to quantify output uncertainty and
variability induced by parameter uncertainty, rather than to compute
variance-decomposition sensitivity indices. It is therefore complementary to
local (derivative-based) and global (variance-based) sensitivity analyses.

Parameter samples are generated using Latin Hypercube Sampling (LHS), a
stratified Monte Carlo method that ensures efficient coverage of the
multidimensional parameter space. In the current implementation, parameters
are sampled independently assuming uniform distributions within their bounds.

For each analysis group and output variable, descriptive statistics are
computed from the simulated sample ensemble, including:
- mean and median
- standard deviation and coefficient of variation
- minimum and maximum
- selected quantiles (5% and 95%)

These statistics provide a compact summary of output uncertainty and enable
comparisons across model outputs and experimental or physiological conditions.

The module is intended for:
- uncertainty propagation analyses
- robustness and variability assessments
- exploratory model analysis and screening
- reporting uncertainty ranges in computational modeling studies

It integrates with the sbmlsim sensitivity framework and supports result
caching, tabular export, and visualization of output distributions.

Notes
-----
This method does not compute sensitivity indices in the strict variance-based
sense (e.g., Sobol indices). Instead, it characterizes how uncertainty in
parameters propagates to uncertainty in model outputs via sampling.

Typical workflows combine this approach with local or global sensitivity
analysis to obtain both quantitative sensitivity measures and uncertainty
estimates.
"""

from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import xarray as xr
from pymetadata.console import console
from scipy.stats import qmc
from matplotlib import pyplot as plt

from sbmlsim.sensitivity.analysis import SensitivitySimulation, AnalysisGroup, \
    SensitivityAnalysis
from sbmlsim.sensitivity.parameters import SensitivityParameter


class SamplingSensitivityAnalysis(SensitivityAnalysis):
    """Sensitivity/uncertainty analysis based on sampling.

    FIXME: more control on sampling
        cv: float = 0.1,
        distribution: DistributionType = DistributionType.NORMAL_DISTRIBUTION,

    """

    sensitivity_keys = [
        "mean",
        "median",
        "std",
        "cv",
        "min",
        "q005",
        "q095",
        "max"
    ]

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        N: int,
        seed: Optional[int] = None,
        n_cores: Optional[int] = None,
        cache_results: bool = False,
    ):

        super().__init__(
            sensitivity_simulation=sensitivity_simulation,
            parameters=parameters,
            groups=groups,
            results_path=results_path,
            seed=seed,
            n_cores=n_cores,
            cache_results=cache_results,
        )
        self.N: int = N
        self.prefix = f"sampling_N{self.N}"

    def create_samples(self) -> None:
        """Create LHS samples.

        Latin hypercube sampling (LHS) is a stratified sampling method used to
        generate near‑random samples from a multidimensional distribution for Monte
        Carlo simulations and computer experiments.

        Assuming uniform distributions within the provided bounds.

        Use LHS sampling of parameters.
        """
        # LHS sampling (uniform distributed in bounds)
        sampler = qmc.LatinHypercube(d=self.num_parameters)  # number of dimensions
        lower = np.array([p.lower_bound for p in self.parameters])
        upper = np.array([p.upper_bound for p in self.parameters])

        for gid in self.group_ids:
            u = sampler.random(n=self.N)  # shape (n, d), in [0, 1], number of samples
            self.samples[gid] = xr.DataArray(
                qmc.scale(u, lower, upper),  # scale to parameter bounds
                dims=["sample", "parameter"],
                coords={"sample": range(self.N),
                        "parameter": self.parameter_ids},
                name="samples"
            )

    def calculate_sensitivity(self, cache_filename: Optional[str] = None,
                              cache: bool = False) -> None:
        """Calculate the sensitivity matrices for sampling sensitivity."""

        data = self.read_cache(cache_filename, cache)
        if data:
            self.sensitivity = data
            return

        for gid in self.group_ids:
            # calculate readouts
            for key in self.sensitivity_keys:
                self.sensitivity[gid][key] = xr.DataArray(
                    np.full(self.num_outputs, np.nan),
                    dims=["output"],
                    coords={
                        "output": self.output_ids},
                    name=key
                )

            for ko, oid in enumerate(self.outputs):
                # num_samples x num_outputs
                data = self.results[gid].values[:, ko]
                for key in self.sensitivity_keys:
                    if key == "mean":
                        value = np.mean(data)
                    elif key == "median":
                        value = np.median(data)
                    elif key == "std":
                        value = np.std(data)
                    elif key == "cv":
                        value = np.std(data) / np.mean(data)
                    elif key == "min":
                        value = np.min(data)
                    elif key == "q005":
                        value = np.quantile(data, q=0.05)
                    elif key == "q095":
                        value = np.quantile(data, q=0.95)
                    elif key == "max":
                        value = np.max(data)
                    else:
                        raise KeyError(key)

                    self.sensitivity[gid][key][ko] = value

        self.df_sampling_sensitivity(self.results_path / f"{self.prefix}.tsv")

        # write to cache
        self.write_cache(data=self.sensitivity, cache_filename=cache_filename,
                         cache=cache)

    def df_sampling_sensitivity(
        self,
        df_path: Path,
    ):
        # dataframe with the values
        items = []
        for group in self.groups:
            for ko, output in enumerate(self.outputs):
                item: dict[str, Any] = {
                    "gid": group.uid,
                    "gname": group.name,
                    "uid": output.uid,
                    "name": output.name,
                    "N": self.N,
                }
                for key in self.sensitivity_keys:
                    item[key] = self.sensitivity[group.uid][key].values[ko]
                item["unit"] = output.unit

                items.append(item)

        df = pd.DataFrame(items)
        console.print(df)

        # create compact DataFrame
        items_compact = []
        for ko, output in enumerate(self.outputs):
            item: dict[str, Any] = {
                "output": output.name,
            }
            for group in self.groups:
                m = self.sensitivity[group.uid]["mean"].values[ko]
                # std = self.sensitivity[group.uid]["std"].values[ko]
                cv = self.sensitivity[group.uid]["cv"].values[ko]
                # q005 = self.sensitivity[group.uid]["q005"].values[ko]
                # q095 = self.sensitivity[group.uid]["q095"].values[ko]

                item[group.uid] = f"{m:.3g} ({cv * 100:.1f})"
            item["unit"] = output.unit

            items_compact.append(item)

        df_compact = pd.DataFrame(items_compact)
        console.print(df_compact)

        if df_path:
            df.to_csv(df_path, index=False, sep="\t")
            df_compact.to_csv(df_path.parent / f"{df_path.stem}_compact.tsv",
                              index=False, sep="\t")

            # latex table
            latex_path = df_path.parent / f"{df_path.stem}.tex"
            df_latex: pd.DataFrame = df_compact.copy()
            # df_latex.drop(['gid', 'uid', 'N', "min", "max", "q005", "q095"], axis=1, inplace=True)
            latex_str = df_latex.to_latex(None, index=False)
            latex_str = latex_str.replace("∞", r"$\infty$")
            latex_str = latex_str.replace("*", r"$\cdot$")

            with open(latex_path, "w") as f:
                f.write(latex_str)

        return df


    @staticmethod
    def _figshape(n: int) -> tuple[int, int]:
        """Calculates a reasonable figure shape for a number of panels n.

        returns: (nrows, ncols)
        """
        if n <= 4:
            return 1, n

        ncols = np.ceil(np.sqrt(n))
        n_empty = ncols * ncols - n
        n_empty_rows = np.floor(n_empty / ncols)
        nrows = ncols - n_empty_rows
        return int(nrows), int(ncols)

    def plot_data(self, type: str, show_jitter: bool = True, show_violin: bool = True, **kwargs):
        """Boxplots for the sampled output."""
        super().plot(**kwargs)

        # calculate number of rows and columns
        if type == "samples":
            n = self.num_parameters
        elif type == "outputs":
            n = self.num_outputs

        nrows, ncols = self._figshape(n=n)
        label_fontsize = 13

        f, axes = plt.subplots(figsize=(4 * ncols, 4 * nrows),
                               nrows=int(nrows), ncols=int(ncols),
                               layout="constrained")
        for ka, ax in enumerate(axes.flat):
            if ka > n - 1:
                ax.axis('off')
            else:

                if type == "samples":
                    data = [self.samples[g.uid].values[:, ka] for g in self.groups]
                elif type == "outputs":
                    data = [self.results[g.uid].values[:, ka] for g in self.groups]
                colors = [g.color for g in self.groups]
                labels = [g.uid for g in self.groups]
                # outliers for scatter
                # Q1 = np.percentile(data, 25)
                # Q3 = np.percentile(data, 75)
                # IQR = Q3 - Q1
                # lower_fence = Q1 - 1.5 * IQR
                # upper_fence = Q3 + 1.5 * IQR
                # data_no_outliers = data[(data > lower_fence) & (data < upper_fence)]
                # data_no_outliers = data

                bp = ax.boxplot(
                    data,
                    positions=range(self.num_groups),
                    labels=labels,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    boxprops=dict(
                        # facecolor=colors,  #'lightblue',
                        # alpha=0.7
                    )
                )
                for box, color in zip(bp["boxes"], colors):
                    box.set_facecolor(color)

                # violin
                if show_violin:
                    violin_offset = 0.3
                    vp = ax.violinplot(
                        data,
                        positions=[k + violin_offset for k in range(self.num_groups)],
                        showmeans=True,
                        showmedians=True,
                        showextrema=False,
                    )

                    for body, color in zip(vp["bodies"], colors):
                        body.set_facecolor(color)

                # jitter
                if show_jitter:
                    jitter_offset = 0.3
                    jitter_width = 0.02  # Adjust for spacing
                    for kg, g in enumerate(self.groups):
                        data_g = data[kg]
                        x_jitter = np.random.normal(kg + jitter_offset, jitter_width, len(data_g))
                        ax.scatter(x_jitter, data_g, alpha=0.7, s=30, color='white',
                                   edgecolors='black'
                    )

                # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
                # ax.set_ylim(bottom=0)
                # ax.set_title(output.name, fontsize=15, fontweight="bold")

                if type == "samples":
                    parameter = self.parameters[ka]
                    ylabel = f"{parameter.uid}: {parameter.name} [{parameter.unit if parameter.unit else 'AU'}]"
                    ax.set_ylabel(ylabel, fontsize=label_fontsize,
                                  fontweight="bold")
                elif type == "outputs":
                    output = self.outputs[ka]
                    ylabel = f"{output.name} [{output.unit if output.unit else 'AU'}]"
                    ax.set_ylabel(ylabel,
                                  fontsize=label_fontsize,
                                  fontweight="bold")

                # Make x and y tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.tick_params(axis='x', labelrotation=90)
                # ax.legend(True)


        # if title:
        #     plt.suptitle(title, fontsize=20, fontweight="bold")

        plt.savefig(
            self.results_path / f"{self.prefix}_sensitivity_{type}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()



    def plot(self, **kwargs):
        """Boxplots for the Sampling sensitivity."""
        self.plot_data(type="samples", **kwargs)
        self.plot_data(type="outputs", **kwargs)
