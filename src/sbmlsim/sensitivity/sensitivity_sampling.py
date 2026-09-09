"""Sampling-based sensitivity and uncertainty analysis.

This module implements a sampling-based sensitivity and uncertainty analysis
approach. Model parameters are varied simultaneously within their bounds, and
the resulting distribution of model outputs is analyzed statistically.

Parameter samples are generated using Latin Hypercube Sampling (LHS), assuming
independent and uniformly distributed parameters.

For each analysis group and output variable, descriptive statistics are
computed, including:

- mean and median
- standard deviation and coefficient of variation
- minimum and maximum
- lower and upper quantiles (5% and 95%)

Uncertainty is calculated as Ui,j = (Percentile97.5(i,j) - Percentile2.5(i,j)) / Percentile50(i,j)

This approach focuses on uncertainty propagation rather than variance-based
sensitivity indices and is therefore complementary to local and Sobol-based
methods.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.stats import qmc

from sbmlsim.console import console
from sbmlsim.sensitivity.analysis import (
    AnalysisGroup,
    SensitivityAnalysis,
    SensitivitySimulation,
)
from sbmlsim.sensitivity.classification import (
    uncertainty_classification,
    uncertainty_classification_symbol,
)
from sbmlsim.sensitivity.parameters import SensitivityParameter


class SamplingSensitivityAnalysis(SensitivityAnalysis):
    """Sensitivity/uncertainty analysis based on sampling."""

    sensitivity_keys: ClassVar[list[str]] = [
        "mean",
        "median",
        "std",
        "cv",
        "min",
        "q005",
        "q095",
        "max",
        "U",
    ]

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        N: int,
        seed: int | None = None,
        n_cores: int | None = None,
        cache_results: bool = False,
    ):
        """Initialize the sampling analysis with N samples per group."""
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
                coords={"sample": range(self.N), "parameter": self.parameter_ids},
                name="samples",
            )

    def calculate_sensitivity(
        self, cache_filename: str | None = None, cache: bool = False
    ) -> None:
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
                    coords={"output": self.output_ids},
                    name=key,
                )

            for ko, _oid in enumerate(self.outputs):
                # num_samples x num_outputs
                data = self.results_required(gid).values[:, ko]
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
                    elif key == "U":
                        value = (
                            np.percentile(data, 97.5) - np.percentile(data, 2.5)
                        ) / np.percentile(data, 50)
                    else:
                        raise KeyError(key)

                    self.sensitivity[gid][key][ko] = value

        self.df_sampling_sensitivity(self.results_path / f"{self.prefix}.tsv")

        # write to cache
        self.write_cache(
            data=self.sensitivity, cache_filename=cache_filename, cache=cache
        )

    def df_sampling_sensitivity(
        self,
        df_path: Path,
    ):
        """Write the sampling sensitivities as a table to the given path."""
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

                item["symbol"] = uncertainty_classification_symbol(item["U"])
                item["classification"] = uncertainty_classification(item["U"]).value
                item["unit"] = output.unit

                items.append(item)

        df = pd.DataFrame(items)
        console.print(df)
        console.print()

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
            df_compact.to_csv(
                df_path.parent / f"{df_path.stem}_compact.tsv", index=False, sep="\t"
            )

            # latex table
            latex_path = df_path.parent / f"{df_path.stem}.tex"
            df_latex: pd.DataFrame = df_compact.copy()
            # df_latex.drop(['gid', 'uid', 'N', "min", "max", "q005", "q095"], axis=1, inplace=True)
            latex_str = df_latex.to_latex(None, index=False)
            latex_str = latex_str.replace("∞", r"$\infty$")
            latex_str = latex_str.replace("*", r"$\cdot$")

            with open(latex_path, "w", encoding="utf-8") as f:
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

    def plot_data(
        self, type: str, show_jitter: bool = True, show_violin: bool = True, **kwargs
    ):
        """Boxplots for the sampled output."""
        super().plot(**kwargs)

        # calculate number of rows and columns
        if type == "samples":
            n = self.num_parameters
        elif type == "outputs":
            n = self.num_outputs

        nrows, ncols = self._figshape(n=n)
        label_fontsize = 13

        f, axes = plt.subplots(
            figsize=(4 * ncols, 4 * nrows),
            nrows=int(nrows),
            ncols=int(ncols),
            layout="constrained",
        )
        for ka, ax in enumerate(axes.flat):
            if ka > n - 1:
                ax.axis("off")
            else:
                if type == "samples":
                    data = [
                        self.samples_required(g.uid).values[:, ka] for g in self.groups
                    ]
                elif type == "outputs":
                    data = [
                        self.results_required(g.uid).values[:, ka] for g in self.groups
                    ]
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
                    tick_labels=labels,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black"},
                    whiskerprops={"color": "black"},
                    capprops={"color": "black"},
                    boxprops={
                        # facecolor=colors,  #'lightblue',
                        # alpha=0.7
                    },
                )
                for box, color in zip(bp["boxes"], colors, strict=False):
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

                    for body, color in zip(vp["bodies"], colors, strict=False):
                        body.set_facecolor(color)

                # jitter
                if show_jitter:
                    jitter_offset = 0.3
                    jitter_width = 0.02  # Adjust for spacing
                    for kg, _g in enumerate(self.groups):
                        data_g = data[kg]
                        x_jitter = np.random.normal(
                            kg + jitter_offset, jitter_width, len(data_g)
                        )
                        ax.scatter(
                            x_jitter,
                            data_g,
                            alpha=0.7,
                            s=30,
                            color="white",
                            edgecolors="black",
                        )

                # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
                # ax.set_ylim(bottom=0)
                # ax.set_title(output.name, fontsize=15, fontweight="bold")

                if type == "samples":
                    parameter = self.parameters[ka]
                    ylabel = f"{parameter.uid}: {parameter.name} [{parameter.unit if parameter.unit else 'AU'}]"
                    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")
                elif type == "outputs":
                    output = self.outputs[ka]
                    ylabel = f"{output.name} [{output.unit if output.unit else 'AU'}]"
                    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

                # Make x and y tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight("bold")
                ax.tick_params(axis="x", labelrotation=90)
                # ax.legend(True)

        # if title:
        #     plt.suptitle(title, fontsize=20, fontweight="bold")

        plt.savefig(
            self.results_path / f"{self.prefix}_sensitivity_{type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(f)

    def plot(self, **kwargs):
        """Boxplots for the Sampling sensitivity."""
        self.plot_data(type="samples", **kwargs)
        self.plot_data(type="outputs", **kwargs)
