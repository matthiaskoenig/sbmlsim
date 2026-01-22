from pathlib import Path
from typing import Optional

from pymetadata.console import console


import numpy as np
import xarray as xr


import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol

from sbmlsim.sensitivity.analysis import SensitivityAnalysis, SensitivitySimulation, \
    AnalysisGroup
from sbmlsim.sensitivity.parameters import SensitivityParameter


class SobolSensitivityAnalysis(SensitivityAnalysis):
    """Global sensitivity analysis based on Sobol method.

    - [ ] SOBOL indices Sobol Sensitivity Analysis (Sobol 2001, Saltelli 2002, Saltelli et al. 2010)
      http://www.sciencedirect.com/science/article/pii/S0378475400002706
      https://www.sciencedirect.com/science/article/pii/S0010465502002801
      https://www.sciencedirect.com/science/article/pii/S0010465509003087
    """

    sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 N: int,
                 **kwargs,
                 ):
        """
        N: length of chain (Sobol' sequence), must be power of 2, i.e. 2^m e.g. 4096

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)
        self.N: int = N

        # define the problem specification
        self.ssa_problems: dict[str, ProblemSpec] = {}
        for group in self.groups:
            self.ssa_problems[group.uid] = ProblemSpec({
                'num_vars': self.num_parameters,
                'names': self.parameter_ids,
                'bounds': [ [p.lower_bound, p.upper_bound] for p in self.parameters],
                "outputs": self.output_ids,
            })

    def create_samples(self) -> None:
        """Create samples for sobol.

        Generates model inputs using Saltelli's extension of the Sobol' sequence

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """
        console.rule("Samples", style="white")
        # (num_samples x num_outputs)
        #  total model evaluations are (2d+2) * N for d input factors
        num_samples = (2 * self.num_parameters + 2) * self.N

        for gid in self.group_ids:
            # libsa samples based on definition
            ssa_samples = saltelli.sample(self.ssa_problems[gid], N=self.N, calc_second_order=True)
            self.ssa_problems[gid].set_samples(ssa_samples)

            self.samples[gid] = xr.DataArray(
                ssa_samples,
                dims=["sample", "parameter"],
                coords={"sample": range(num_samples),
                        "parameter": self.parameter_ids},
                name="samples"
            )
        console.print(self.samples)


    def calculate_sensitivity(self, cache_filename: Optional[str] = None, cache: bool = False):
        """Calculate the sensitivity matrices for SOBOL analysis."""

        data = self.read_cache(cache_filename, cache)
        if data:
            self.sensitivity = data
            return

        for gid in self.group_ids:
            Y = self.results[gid].values
            self.ssa_problems[gid].set_results(Y)

            # num_parameters x num_outputs

            for key in self.sensitivity_keys:
                self.sensitivity[gid][key] = xr.DataArray(
                    np.full((self.num_parameters, self.num_outputs), np.nan),
                    dims=["parameter", "output"],
                    coords={"parameter": self.parameter_ids,
                            "output": self.output_ids},
                    name=key
                )

            # Perform Analysis
            # Si is a Python dict-like with the keys "S1", "S2", "ST",
            # "S1_conf", "S2_conf", and "ST_conf".
            # The _conf keys store the corresponding confidence intervals,
            # typically with a confidence level of 95%.

            # Calculate Sobol indices for every output
            for ko in range(self.num_outputs):
                Yo = Y[:, ko]
                Si = SALib.analyze.sobol.analyze(
                    self.ssa_problems[gid], Yo,
                    calc_second_order=True,
                    print_to_console=False,
                    n_processors=4,
                )
                for key in self.sensitivity_keys:
                    self.sensitivity[gid][key][:, ko] = Si[key]

        # write to cache
        self.write_cache(data=self.sensitivity, cache_filename=cache_filename, cache=cache)

    def plot_sobol_indices(
        self,
        fig_path: Path,
        ):
        """Barplots for the Sobol indices."""
        # parameter_labels: dict[str, str] = {p.uid: f"{p.uid}: {p.name}" for p in self.parameters}
        parameter_labels: dict[str, str] = {p.uid: p.uid for p in self.parameters}
        output_labels: dict[str, str] = {q.uid: q.name for q in self.outputs}

        for group in self.groups:
            gid = group.uid
            ymax = self.sensitivity[gid]["ST"].max(dim=None)
            ymin = self.sensitivity[gid]["S1"].min(dim=None)

            for ko, output in enumerate(self.outputs):
                # f_path = fig_path.parent / f"FigS{ko+22}_{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"
                f_path = fig_path.parent / f"{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"

                S1 = self.sensitivity[gid]["S1"][:, ko]
                ST = self.sensitivity[gid]["ST"][:, ko]
                S1_conf = self.sensitivity[gid]["S1_conf"][:, ko]
                ST_conf = self.sensitivity[gid]["ST_conf"][:, ko]
                sobol_barplot(
                    S1=S1,
                    ST=ST,
                    S1_conf=S1_conf,
                    ST_conf=ST_conf,
                    title=f"{output_labels[output.uid]} ({group.name})",
                    fig_path=f_path,
                    parameter_labels=parameter_labels,
                    ymax=np.max([1.05, ymax]),
                    ymin=np.min([-0.05, ymin]),
                )

def sobol_barplot(
    S1, ST, S1_conf, ST_conf,
    parameter_labels: dict[str, str],
    fig_path: Optional[Path] = None,
    title: Optional[str] = None,
    ymax: float = 1.1,
    ymin: float = -0.1,
):
    # width
    figsize = (15, 3)
    label_fontsize = 15

    categories: list[str] = list(parameter_labels.values())
    f, ax = plt.subplots(figsize=figsize)

    ax.bar(categories, ST, label='ST',
           color="tab:orange",
           alpha=1.0,
           edgecolor="black",
           yerr=ST_conf, capsize=5
           )

    ax.bar(categories, S1, label='S1', color="tab:blue",
           edgecolor="black", yerr=S1_conf, capsize=5)


    # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel('Sobol Index', fontsize=label_fontsize, fontweight="bold")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.grid(True, axis="y")
    ax.tick_params(axis='x', labelrotation=90)
    # ax.tick_params(axis='x', labelweight='bold')
    ax.legend()

    if title:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

