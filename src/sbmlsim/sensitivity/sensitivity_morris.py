"""
Morris sensitivity analysis.

Method of Morris, including groups and optimal trajectories (Morris 1991, Campolongo et al. 2007)
"""

from pathlib import Path
from typing import Optional

import SALib
import numpy as np
import xarray as xr
from SALib import ProblemSpec
from SALib.sample.morris import sample as morris_sample

from sbmlsim.sensitivity import (
    SensitivityAnalysis,
    SensitivitySimulation,
    SensitivityParameter,
    AnalysisGroup,
)


class MorrisSensitivityAnalysis(SensitivityAnalysis):
    """Morris sensitivity analysis.

    Campolongo et al., [2] introduces an optimal trajectories approach which attempts to maximize the parameter space scanned for a given number of trajectories (where optimal_trajectories). The approach accomplishes this aim by randomly generating a high number of possible trajectories (500 to 1000 in [2]) and selecting a subset of r trajectories which have the highest spread in parameter space. The r variable in [2] corresponds to the optimal_trajectories parameter here.

    Calculating all possible combinations of trajectories can be computationally expensive. The number of factors makes little difference, but the ratio between number of optimal trajectories and the sample size results in an exponentially increasing number of scores that must be computed to find the optimal combination of trajectories. We suggest going no higher than 4 levels from a pool of 100 samples with this “brute force” approach.

    Ruano et al., [3] proposed an alternative approach with an iterative process that maximizes the distance between subgroups of generated trajectories, from which the final set of trajectories are selected, again maximizing the distance between each. The approach is not guaranteed to produce the most optimal spread of trajectories, but are at least locally maximized and significantly reduce the time taken to select trajectories. With local_optimization = True (which is default), it is possible to go higher than the previously suggested 4 levels from a pool of 100 samples.

    [1] Morris, M.D., 1991. Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics 33, 161-174. https://doi.org/10.1080/00401706.1991.10484804
    [2] Campolongo, F., Cariboni, J., & Saltelli, A. 2007. An effective screening design for sensitivity analysis of large models. Environmental Modelling & Software, 22(10), 1509-1518. https://doi.org/10.1016/j.envsoft.2006.10.004
    [3] Ruano, M.V., Ribes, J., Seco, A., Ferrer, J., 2012. An improved sampling strategy based on trajectory design for application of the Morris method to systems with many input factors. Environmental Modelling & Software 37, 103-109. https://doi.org/10.1016/j.envsoft.2012.03.008

    """

    sensitivity_keys = ["mu", "mu_star", "sigma", "mu_star_conf"]

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        N: int,
        optimal_trajectories: int,
        num_levels: int = 4,
        local_optimization: bool = True,
        seed: Optional[int] = None,
        n_cores: Optional[int] = None,
        cache_results: bool = False,
        **kwargs,
    ):
        """
        Resulting simulations are (D+1) * N/T with D number of parameters.


        N (int) – The number of trajectories to generate
        optimal_trajectories - The number of optimal trajectories to sample (between 2 and N)
        num_levels - The number of grid levels to use (should be even)
        local_optimization - Flag whether to use local optimization according to Ruano et al. (2012) Speeds up the process tremendously for bigger N and num_levels. If set to False brute force method is used
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
        self.optimal_trajectories: int = optimal_trajectories
        self.num_levels: int = num_levels
        self.local_optimization: bool = local_optimization
        self.prefix = f"morris_levels{self.num_levels}_N{self.N}"

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
        """Create samples using the Method of Morris.
         Three variants of Morris' sampling for elementary effects is supported:

        - Vanilla Morris (see [1])
          when ``optimal_trajectories`` is ``None``/``False`` and
          ``local_optimization`` is ``False``
        - Optimised trajectories when ``optimal_trajectories=True`` using
            Campolongo's enhancements (see [2]) and optionally Ruano's enhancement
            (see [3]) when ``local_optimization=True``
        - Morris with groups when the problem definition specifies groups of
          parameters
        """
        # (num_samples x num_outputs)
        for gid in self.group_ids:
            # libssa samples based on definition
            morris_samples = morris_sample(
                self.ssa_problems[gid],
                N=self.N,
                num_levels=self.num_levels,
                optimal_trajectories=self.optimal_trajectories,
                local_optimization=self.local_optimization,
            )
            self.ssa_problems[gid].set_samples(morris_samples)

            self.samples[gid] = xr.DataArray(
                morris_samples,
                dims=["sample", "parameter"],
                coords={
                    "sample": range(morris_samples.shape[0]),
                    "parameter": self.parameter_ids,
                },
                name="samples",
            )

    def calculate_sensitivity(
        self, cache_filename: Optional[str] = None, cache: bool = False
    ):
        """Perform extended Fourier Amplitude Sensitivity Test on model outputs.

        Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
        size D (the number of parameters) containing the indices in the same order
        as the parameter file.

        Returns a result set with keys mu, mu_star, sigma, and mu_star_conf, where each entry corresponds to the parameters defined in the problem spec or parameter file.

        mu metric indicates the mean of the distribution
        mu_star metric indicates the mean of the distribution of absolute values
        sigma is the standard deviation of the distribution
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

            # Calculate Morris indices
            for ko in range(self.num_outputs):
                Yo = Y[:, ko]
                Si = SALib.analyze.morris.analyze(
                    self.ssa_problems[gid],
                    Yo,
                    scaled=False,
                    num_levels=self.num_levels,
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
        super().plot()
        for kg, group in enumerate(self.groups):
            # morris plots
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
