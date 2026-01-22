"""
Global sensitivity analysis using FAST (Fourier Amplitude Sensitivity Test).

This module implements variance-based global sensitivity analysis using the
Fourier Amplitude Sensitivity Test (FAST). FAST quantifies the contribution of
individual model parameters to the variance of model outputs by mapping
parameter variations onto periodic functions and analyzing the resulting
output spectrum in the frequency domain.

The method provides efficient estimation of first-order (main-effect)
sensitivity indices and, in extended variants (eFAST), total-effect indices.
Compared to Monte Carlo–based Sobol methods, FAST offers favorable scaling with
the number of parameters and is well suited for medium- to large-scale
deterministic models.

The implementation is intended for use in computational modeling workflows,
including systems biology, pharmacokinetics/pharmacodynamics, and digital twin
applications, where robust global assessment of parameter influence is required.

References
----------
Cukier, R. I., Fortuin, C. M., Shuler, K. E., Petschek, A. G., & Schaibly, J. H. (1973).
Study of the sensitivity of coupled reaction systems to uncertainties in rate
coefficients. I. Theory.
Journal of Chemical Physics, 59, 3873–3878.
https://doi.org/10.1063/1.1680571

Saltelli, A., Tarantola, S., & Chan, K. P.-S. (1999).
A quantitative model-independent method for global sensitivity analysis of
model output.
Technometrics, 41(1), 39–56.
https://doi.org/10.1080/00401706.1999.10485594
"""

from pathlib import Path
from typing import Optional

import SALib
import numpy as np
import xarray as xr
from SALib import ProblemSpec
from SALib.analyze import fast
from SALib.sample import fast_sampler
from pymetadata.console import console

from sbmlsim.sensitivity import (
    SensitivityAnalysis,
    SensitivitySimulation,
    SensitivityParameter,
    AnalysisGroup,
)
from sbmlsim.sensitivity.plots import plot_S1_ST_indices


class FASTSensitivityAnalysis(SensitivityAnalysis):
    """Global sensitivity analysis based Fourier Amplitude Sensitivity Test (FAST)
    (Cukier et al. 1973, Saltelli et al. 1999)."""

    sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        N: int,
        M: int = 4,
        **kwargs,
    ):
        """
        N (int) – The number of samples to generate
        M (int) – The interference parameter, i.e., the number of harmonics to sum
        in the Fourier series decomposition (default 4)

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """

        super().__init__(sensitivity_simulation, parameters, groups, results_path,
                         **kwargs)
        self.N: int = N
        self.M: int = M

        # define the problem specification
        self.ssa_problems: dict[str, ProblemSpec] = {}
        for group in self.groups:
            self.ssa_problems[group.uid] = ProblemSpec({
                'num_vars': self.num_parameters,
                'names': self.parameter_ids,
                'bounds': [[p.lower_bound, p.upper_bound] for p in self.parameters],
                "outputs": self.output_ids,
            })

    def create_samples(self) -> None:
        """Create samples for FAST."""
        # (num_samples x num_outputs)
        #  total model evaluations are N * num_parameters
        num_samples = self.N * self.num_parameters

        for gid in self.group_ids:
            # libssa samples based on definition
            ssa_samples = fast_sampler.sample(
                self.ssa_problems[gid], N=self.N, M=self.M,
            )
            self.ssa_problems[gid].set_samples(ssa_samples)

            self.samples[gid] = xr.DataArray(
                ssa_samples,
                dims=["sample", "parameter"],
                coords={"sample": range(num_samples),
                        "parameter": self.parameter_ids},
                name="samples"
            )

    def calculate_sensitivity(self, cache_filename: Optional[str] = None,
                              cache: bool = False):
        """ Perform extended Fourier Amplitude Sensitivity Test on model outputs.

        Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
        size D (the number of parameters) containing the indices in the same order
        as the parameter file.
        """

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

            # Calculate FAST indices
            for ko in range(self.num_outputs):
                Yo = Y[:, ko]
                Si = SALib.analyze.fast.analyze(
                    self.ssa_problems[gid], Yo,
                    M=self.M,
                    num_resamples=100,
                    conf_level=0.95,
                    print_to_console=False,
                )
                for key in self.sensitivity_keys:
                    self.sensitivity[gid][key][:, ko] = Si[key]

        # write to cache
        self.write_cache(data=self.sensitivity, cache_filename=cache_filename,
                         cache=cache)

    @staticmethod
    def run_sensitivity_analysis(
        results_path: Path,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        N: int,
        seed: int,
        M: int = 4,
        cache_results: bool = False,
        cache_sensitivity: bool = False,
    ) -> None:
        """FAST sensitivity analysis.

        First-order FAST (main effects only):
        100 × num_pars samples is usually sufficient

        Extended FAST (eFAST, total effects):
        200–500 × k samples recommended
        (higher frequencies needed to separate interactions)

        :param sensitivity_simulation: Sensitivity simulation.
        :param parameters: Sensitivity parameters.
        :param groups: Sensitivity groups.
        N (int) – The number of samples to generate
        M (int) – The interference parameter, i.e., the number of harmonics to sum
        :param seed: Random seed.
        """
        prefix = "fast"
        console.rule(f"{prefix.upper()} SENSITIVITY ANALYSIS", style="blue bold", align="center")
        if cache_sensitivity and not cache_results:
            # sensitivity must be recalculated for new results
            cache_sensitivity = False

        sa = FASTSensitivityAnalysis(
            sensitivity_simulation=sensitivity_simulation,
            parameters=parameters,
            groups=groups,
            results_path=results_path,
            N=N,
            M=M,
            seed=seed,
        )

        console.rule("Samples", style="white")
        sa.create_samples()
        console.print(sa.samples_table())

        console.rule("Results", style="white")
        sa.simulate_samples(cache_filename=f"{prefix}_results_N{sa.N}.pkl",
                            cache=cache_results)
        console.print(sa.results_table())

        console.rule("Sensitivity", style="white")
        sa.calculate_sensitivity(cache_filename=f"{prefix}_sensitivity_N{sa.N}.pkl",
                                 cache=cache_sensitivity)

        console.rule("Plotting", style="white")
        for kg, group in enumerate(sa.groups):
            # heatmaps
            for key in ["ST", "S1"]:
                sa.plot_sensitivity(
                    group_id=group.uid,
                    sensitivity_key=key,
                    # title=f"{key} {group.name}",
                    cutoff=0.05,
                    cluster_rows=False,
                    cmap="viridis",
                    vcenter=0.5,
                    vmin=0.0,
                    vmax=1.0,
                    fig_path=sa.results_path / f"{prefix}_sensitivity_N{sa.N}_{kg:>02}_{group.uid}_{key}.png"
                )

            # barplots
            plot_S1_ST_indices(
                sa=sa,
                fig_path=sa.results_path / f"{prefix}_sensitivity_N{sa.N}_{kg:>02}_{group.uid}.png",
            )
