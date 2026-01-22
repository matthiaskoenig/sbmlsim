from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import xarray as xr
from pymetadata.console import console
from scipy.stats import qmc

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

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 N: int,
                 **kwargs,
                 ):

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)
        self.N: int = N

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

    def calculate_sensitivity(self, cache_filename: Optional[str] = None, cache: bool = False) -> None:
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
                        value = np.std(data)/np.mean(data)
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

        # write to cache
        self.write_cache(data=self.sensitivity, cache_filename=cache_filename, cache=cache)


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
                std = self.sensitivity[group.uid]["std"].values[ko]
                cv = self.sensitivity[group.uid]["cv"].values[ko]
                q005 = self.sensitivity[group.uid]["q005"].values[ko]
                q095 = self.sensitivity[group.uid]["q095"].values[ko]

                item[group.uid] = f"{m:.3g} ({cv*100:.1f})"
            item["unit"] = output.unit

            items_compact.append(item)

        df_compact = pd.DataFrame(items_compact)
        console.print(df_compact)

        if df_path:
            df.to_csv(df_path, index=False, sep="\t")
            df_compact.to_csv(df_path.parent / f"{df_path.stem}_compact.tsv", index=False, sep="\t")

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

    def plot_sampling_sensitivity(
        self,
        fig_path: Path,
        **kwargs
        ):
        """Boxplots for the Sampling sensitivity."""

        # width
        figsize = (15, 15)
        label_fontsize = 15
        from matplotlib import pyplot as plt
        ncols = np.ceil(np.sqrt(self.num_outputs))
        n_empty = ncols*ncols - self.num_outputs
        n_empty_rows = np.floor(n_empty/ncols)

        nrows = ncols-n_empty_rows

        f, axes = plt.subplots(figsize=figsize, nrows=int(nrows), ncols=int(ncols), layout="constrained")
        for ko, ax in enumerate(axes.flat):
            if ko > self.num_outputs-1:
                ax.axis('off')
            else:

                output = self.outputs[ko]
                data = [self.results[g.uid].values[:, ko] for g in self.groups]
                colors = [g.color for g in self.groups]
                labels = [g.uid for g in self.groups]
                # outliers for scatter
                # Q1 = np.percentile(data, 25)
                # Q3 = np.percentile(data, 75)
                # IQR = Q3 - Q1
                # lower_fence = Q1 - 1.5 * IQR
                # upper_fence = Q3 + 1.5 * IQR
                # data_no_outliers = data[(data > lower_fence) & (data < upper_fence)]
                data_no_outliers = data

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

                # ax.violinplot(data, positions=[0.8], showmeans=True,
                #                showmedians=True,
                #                showextrema = False
                #               )
                # jitter_width = 0.05  # Adjust for spacing
                # x_jitter = np.random.normal(0.8, jitter_width, len(data_no_outliers))
                # ax.scatter(x_jitter, data_no_outliers, alpha=0.7, s=30, color='darkgrey',
                #                edgecolors='black'
                # )

                # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
                # ax.set_ylim(bottom=0)
                # ax.set_title(output.name, fontsize=15, fontweight="bold")
                ax.set_ylabel(f"{output.name} [{output.unit}]", fontsize=label_fontsize, fontweight="bold")
                # ax.tick_params(axis='x', which='both', labelbottom=False)
                # ax.grid(True, axis="y")
                ax.tick_params(axis='x', labelrotation=90)

        # if title:
        #     plt.suptitle(title, fontsize=20, fontweight="bold")
        if fig_path:
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def run_sensitivity_analysis(
            results_path: Path,
            sensitivity_simulation: SensitivitySimulation,
            parameters: list[SensitivityParameter],
            groups: list[AnalysisGroup],
            N: int,
            seed: int,
            cache_results: bool = False,
            cache_sensitivity: bool = False,
    ) -> None:
        """Sampling sensitivity/uncertainty analysis.

        :param sensitivity_simulation: Sensitivity simulation.
        :param parameters: Sensitivity parameters.
        :param groups: Sensitivity groups.
        :param N: Number of samples.
        :param seed: Random seed.
        """
        console.rule("SAMPLING SENSITIVITY ANALYSIS", style="blue bold", align="center")
        if cache_sensitivity and not cache_results:
            # sensitivity must be recalculated for new results
            cache_sensitivity = False

        sa = SamplingSensitivityAnalysis(
            sensitivity_simulation=sensitivity_simulation,
            parameters=parameters,
            results_path=results_path,
            N=N,
            seed=seed,
            groups=groups,
        )
        console.rule("Samples", style="white")
        sa.create_samples()
        console.print(sa.samples_table())

        console.rule("Results", style="white")
        sa.simulate_samples(cache_filename=f"sampling_results_N{sa.N}.pkl", cache=cache_results)
        console.print(sa.results_table())

        console.rule("Sensitivity", style="white")
        sa.calculate_sensitivity(cache_filename=f"sampling_sensitivity_N{sa.N}.pkl", cache=cache_sensitivity)
        sa.df_sampling_sensitivity(
            df_path=sa.results_path / f"sampling_statistics_N{sa.N}.tsv"
        )
        # console.print(sa.sensitivity_tables())

        console.rule("Plotting", style="white")
        sa.plot_sampling_sensitivity(
            fig_path=sa.results_path / f"sampling_sensitivity_N{sa.N}.png",
        )
