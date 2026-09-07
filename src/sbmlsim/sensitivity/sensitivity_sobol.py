"""Global sensitivity analysis using Sobol indices.

This module provides routines for variance-based global sensitivity analysis
using Sobol indices. Sobol analysis decomposes the variance of model outputs
into contributions from individual parameters and their interactions.

The following indices are computed:

- First-order indices (S1)
- Total-effect indices (ST)
- Associated confidence intervals

Sampling is based on Saltelli's extension of the Sobol sequence and requires
(2D + 2) * N model evaluations for D parameters.

References:
    - Sobol, I. M. (2001). Math. Comput. Simul., 55, 271–280.
    - Saltelli, A. (2002). Comput. Phys. Commun., 145, 280–297.
    - Saltelli et al. (2010). Comput. Phys. Commun., 181, 259–270.
"""

from pathlib import Path

import numpy as np
import SALib
import xarray as xr
from SALib import ProblemSpec
from SALib.sample import saltelli

from sbmlsim.sensitivity import (
    AnalysisGroup,
    SensitivityAnalysis,
    SensitivityParameter,
    SensitivitySimulation,
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
        seed: int | None = None,
        n_cores: int | None = None,
        cache_results: bool = False,
        **kwargs,
    ):
        """N: length of chain (Sobol' sequence), must be power of 2, i.e. 2^m e.g. 4096.

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
        self, cache_filename: str | None = None, cache: bool = False
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
