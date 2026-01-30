"""
Global sensitivity analysis using Sobol indices.

This module provides routines to perform variance-based global sensitivity
analysis based on Sobol indices. Sobol sensitivity analysis quantifies how
uncertainty in model parameters contributes to the variance of one or more
model outputs, allowing a decomposition into main effects, interaction
effects, and total effects.

The implemented methodology follows the classical Sobol framework and its
later refinements, including Monte Carlo–based estimators for first-order,
higher-order, and total-effect sensitivity indices. The approach is fully
global, meaning that parameters are varied simultaneously over their entire
admissible ranges according to prescribed probability distributions.

Sobol indices are defined as:
- First-order indices (S_i), measuring the contribution of a single parameter
  to the output variance, ignoring interactions.
- Higher-order indices (S_ij, S_ijk, ...), measuring interaction effects
  between parameters.
- Total-effect indices (S_Ti), measuring the total contribution of a parameter
  to the output variance, including all interactions.

The analysis requires:
- A deterministic model or simulation function creating scalar outputs.
- A set of input parameters with specified bounds.
- A sampling scheme based on quasi-random or Monte Carlo methods.

References
----------
Sobol, I. M. (2001).
Global sensitivity indices for nonlinear mathematical models and their
Monte Carlo estimates.
Mathematics and Computers in Simulation, 55(1–3), 271–280.
https://www.sciencedirect.com/science/article/pii/S0378475400002706

Saltelli, A. (2002).
Making best use of model evaluations to compute sensitivity indices.
Computer Physics Communications, 145(2), 280–297.
https://www.sciencedirect.com/science/article/pii/S0010465502002801

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010).
Variance based sensitivity analysis of model output. Design and estimator
for the total sensitivity index.
Computer Physics Communications, 181(2), 259–270.
https://www.sciencedirect.com/science/article/pii/S0010465509003087
"""

from pathlib import Path
from typing import Optional

import SALib
import numpy as np
import xarray as xr
from SALib import ProblemSpec
from SALib.sample import saltelli

from sbmlsim.sensitivity import (
    SensitivityAnalysis,
    SensitivitySimulation,
    SensitivityParameter,
    AnalysisGroup,
)
from sbmlsim.sensitivity.plots import plot_S1_ST_indices


class SobolSensitivityAnalysis(SensitivityAnalysis):
    """Global sensitivity analysis based on Sobol method."""

    sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]

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
        **kwargs,
    ):
        """
        N: length of chain (Sobol' sequence), must be power of 2, i.e. 2^m e.g. 4096

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """
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
        self.prefix = f"sobol_N{self.N}"

        # define the problem specification
        self.ssa_problems: dict[str, ProblemSpec] = {}
        for group in self.groups:
            self.ssa_problems[group.uid] = ProblemSpec(
                {
                    "num_vars": self.num_parameters,
                    "names": self.parameter_ids,
                    "bounds": [[p.lower_bound, p.upper_bound] for p in self.parameters],
                    "outputs": self.output_ids,
                }
            )

    def create_samples(self) -> None:
        """Create samples for sobol.

        Generates model inputs using Saltelli's extension of the Sobol' sequence

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """
        # (num_samples x num_outputs)
        #  total model evaluations are (2d+2) * N for d input factors
        num_samples = (2 * self.num_parameters + 2) * self.N

        for gid in self.group_ids:
            # libsa samples based on definition
            ssa_samples = saltelli.sample(
                self.ssa_problems[gid], N=self.N, calc_second_order=True
            )
            self.ssa_problems[gid].set_samples(ssa_samples)

            self.samples[gid] = xr.DataArray(
                ssa_samples,
                dims=["sample", "parameter"],
                coords={"sample": range(num_samples), "parameter": self.parameter_ids},
                name="samples",
            )

    def calculate_sensitivity(
        self, cache_filename: Optional[str] = None, cache: bool = False
    ):
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
                    coords={"parameter": self.parameter_ids, "output": self.output_ids},
                    name=key,
                )

            # Calculate Sobol indices for every output, typically with a confidence
            # level of 95%.
            for ko in range(self.num_outputs):
                Yo = Y[:, ko]
                Si = SALib.analyze.sobol.analyze(
                    self.ssa_problems[gid],
                    Yo,
                    calc_second_order=True,
                    num_resamples=100,
                    conf_level=0.95,
                    print_to_console=False,
                    n_processors=4,
                )
                for key in self.sensitivity_keys:
                    self.sensitivity[gid][key][:, ko] = Si[key]

        # write to cache
        self.write_cache(
            data=self.sensitivity, cache_filename=cache_filename, cache=cache
        )

    def plot(self):
        super().plot()
        for kg, group in enumerate(self.groups):
            # heatmaps
            for key in ["ST", "S1"]:
                self.plot_sensitivity(
                    group_id=group.uid,
                    sensitivity_key=key,
                    # title=f"{key} {group.name}",
                    cutoff=0.05,
                    cluster_rows=False,
                    cmap="viridis",
                    vcenter=0.5,
                    vmin=0.0,
                    vmax=1.0,
                    fig_path=self.results_path
                    / f"{self.prefix}_sensitivity_{kg:>02}_{group.uid}_{key}.png",
                )

            # barplots
            plot_S1_ST_indices(
                sa=self,
                fig_path=self.results_path
                / f"{self.prefix}_sensitivity_{kg:>02}_{group.uid}.png",
            )
