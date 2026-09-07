"""Global sensitivity analysis using FAST (Fourier Amplitude Sensitivity Test).

This module implements variance-based global sensitivity analysis using the
Fourier Amplitude Sensitivity Test (FAST). FAST quantifies the contribution of
individual model parameters to the variance of model outputs by mapping
parameter variations onto periodic functions and analyzing the resulting output
spectrum in the frequency domain.

The method provides efficient estimation of first-order (main-effect)
sensitivity indices and, in extended variants (eFAST), total-effect indices.
Compared to Monte Carlo–based Sobol methods, FAST offers favorable scaling with
the number of parameters and is well suited for medium- to large-scale
deterministic models.

This implementation is intended for use in computational modeling workflows,
including systems biology, pharmacokinetics/pharmacodynamics, and digital twin
applications, where a robust global assessment of parameter influence is
required.

References:
    Cukier, R. I., Fortuin, C. M., Shuler, K. E., Petschek, A. G., & Schaibly, J. H.
    (1973). Study of the sensitivity of coupled reaction systems to uncertainties
    in rate coefficients. I. Theory. Journal of Chemical Physics, 59, 3873–3878.
    https://doi.org/10.1063/1.1680571

    Saltelli, A., Tarantola, S., & Chan, K. P.-S. (1999). A quantitative
    model-independent method for global sensitivity analysis of model output.
    Technometrics, 41(1), 39–56.
    https://doi.org/10.1080/00401706.1999.10485594
"""

from pathlib import Path

import numpy as np
import SALib
import xarray as xr
from SALib import ProblemSpec
from SALib.sample import fast_sampler

from sbmlsim.sensitivity import (
    AnalysisGroup,
    SensitivityAnalysis,
    SensitivityParameter,
    SensitivitySimulation,
)
from sbmlsim.sensitivity.plots import plot_S1_ST_indices


class FASTSensitivityAnalysis(SensitivityAnalysis):
    """Global sensitivity analysis using the Fourier Amplitude Sensitivity Test.

    This class implements the FAST methodology for
    estimating first-order and total-effect sensitivity indices. It integrates
    with the common sensitivity analysis infrastructure provided by
    `SensitivityAnalysis` and supports grouped analyses and multiple model
    outputs.

    References:
        Cukier et al. (1973); Saltelli et al. (1999)
    """

    sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        N: int,
        M: int = 4,
        seed: int | None = None,
        n_cores: int | None = None,
        cache_results: bool = False,
        **kwargs,
    ):
        """Initialize a FAST sensitivity analysis.

        Args:
            sensitivity_simulation: Simulation object defining how model
                evaluations are performed.
            parameters: List of parameters included in the sensitivity analysis.
            groups: Analysis groups defining parameter perturbations or scenarios.
            results_path: Directory where results and figures are stored.
            N: Base sample size controlling the total number of model evaluations.
                The total number of samples scales with `N * num_parameters`.
            M: Interference parameter defining the number of harmonics used in
                the Fourier series decomposition (default: 4).
            seed: Optional random seed for reproducibility.
            n_cores: Optional number of CPU cores used for parallel execution.
            cache_results: Whether to cache sensitivity results to disk.
            **kwargs: Additional keyword arguments passed to the base class.
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
        self.M: int = M
        self.prefix = f"fast_M{self.M}_N{self.N}"

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
        """Create parameter samples for FAST analysis.

        This method generates FAST samples for each analysis group using the
        SALib FAST sampler. The resulting samples are stored as xarray objects
        and later used for model evaluation.
        """
        # (num_samples x num_outputs)
        #  total model evaluations are N * num_parameters
        num_samples = self.N * self.num_parameters

        for gid in self.group_ids:
            # libssa samples based on definition
            ssa_samples = fast_sampler.sample(
                self.ssa_problems[gid],
                N=self.N,
                M=self.M,
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
        """Compute FAST sensitivity indices for all model outputs.

        This method performs the extended Fourier Amplitude Sensitivity Test
        (eFAST) to estimate first-order (S1) and total-effect (ST) sensitivity
        indices, along with corresponding confidence intervals, for each model
        output and parameter.

        Args:
            cache_filename: Optional filename for reading or writing cached
                sensitivity results.
            cache: Whether to read from or write results to cache.

        Notes:
            The sensitivity indices are computed independently for each output
            variable and stored in a structured xarray-based format.
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
                    coords={"parameter": self.parameter_ids, "output": self.output_ids},
                    name=key,
                )

            # Calculate FAST indices
            for ko in range(self.num_outputs):
                Yo = Y[:, ko]
                Si = SALib.analyze.fast.analyze(
                    self.ssa_problems[gid],
                    Yo,
                    M=self.M,
                    num_resamples=100,
                    conf_level=0.95,
                    print_to_console=False,
                )
                for key in self.sensitivity_keys:
                    self.sensitivity[gid][key][:, ko] = Si[key]

        # write to cache
        self.write_cache(
            data=self.sensitivity, cache_filename=cache_filename, cache=cache
        )

    def plot(self):
        """Generate standard FAST sensitivity plots.

        This method creates heatmaps and bar plots for first-order (S1) and
        total-effect (ST) sensitivity indices for each analysis group and stores
        the resulting figures in the configured results directory.
        """
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
