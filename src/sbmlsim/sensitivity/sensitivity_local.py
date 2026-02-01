"""Local sensitivity analysis using finite differences.

This module implements a local, derivative-based sensitivity analysis using
symmetric finite differences around a reference parameter set. Each model
parameter is perturbed individually while all other parameters are kept
constant.

The method is intended for deterministic simulation models and is useful for:
- Identifying locally influential parameters
- Debugging and inspecting model behavior
- Screening parameters prior to optimization or uncertainty analysis
- Complementing global sensitivity analysis methods

Sensitivities are computed per analysis group and output variable and are
reported as both raw and normalized (dimensionless) sensitivities.

Notes:
    For a parameter p with reference value p0, sensitivities are computed as:

        p_plus  = p0 * (1 + difference)
        p_minus = p0 * (1 - difference)

        S = (q(p_plus) - q(p_minus)) / (p_plus - p_minus)

    Normalized sensitivities are defined as:

        S_norm = S * (p0 / q(p0))

    Here a multistep method is implemented following Najjar et al.

References:

    - Najjar A, Hamadeh A, Krause S, Schepky A, Edginton A. Global sensitivity analysis of Open Systems Pharmacology Suite physiologically based pharmacokinetic models. CPT Pharmacometrics Syst Pharmacol. 2024 Dec;13(12):2052-2067. doi: 10.1002/psp4.13256. Epub 2024 Nov 5. PMID: 39498820; PMCID: PMC11646943.

"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from pymetadata.console import console
from pymetadata.log import get_logger

from sbmlsim.sensitivity.analysis import (
    SensitivitySimulation,
    AnalysisGroup,
    SensitivityAnalysis,
)
from sbmlsim.sensitivity.classification import (
    sensitivity_classification,
    sensitivity_classification_symbol,
)
from sbmlsim.sensitivity.parameters import SensitivityParameter

logger = get_logger(__name__)


class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on symmetric finite differences.

    Each model parameter is perturbed individually by a small relative amount
    around a reference parameter set, while all other parameters are held
    constant. For each parameter, two perturbed simulations (increase and
    decrease) are evaluated in addition to a reference simulation.

    Attributes:
        difference (float): Relative parameter perturbation used for the
            finite-difference approximation (e.g., 0.01 corresponds to ±1%).
        prefix (str): Prefix used for naming result files.
    """

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        seed: Optional[int] = None,
        n_cores: Optional[int] = None,
        cache_results: bool = False,
        difference: float = 0.01,
        n_var: int = 3,
    ) -> None:
        """Initialize the local sensitivity analysis.

        Args:
            sensitivity_simulation (SensitivitySimulation):
                Simulation wrapper providing model execution and result handling.
            parameters (list[SensitivityParameter]):
                List of model parameters to perturb.
            groups (list[AnalysisGroup]):
                Analysis groups defining parameter modifications and conditions.
            results_path (Path):
                Directory where results and plots will be stored.
            seed (int, optional):
                Random seed for reproducibility.
            n_cores (int, optional):
                Number of CPU cores used for parallel simulations.
            cache_results (bool, optional):
                Whether simulation and sensitivity results should be cached.
            difference (float, optional):
                Relative perturbation size used for finite differences.
                Defaults to 0.01 (±1%).
            n_var (int, optional):
                Represents the number of steps at which sensitivity is to be evaluated
                within the variation fold change
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

        self.difference: float = difference
        self.n_var: int = n_var
        self.prefix = f"local_d{self.difference}_nvar{self.n_var}"

    @property
    def num_samples(self) -> int:
        """Return the total number of required simulation samples.

        The local sensitivity analysis requires:
        - 2 + n_var simulations per parameter (positive and negative perturbations)
        - One reference simulation

        Returns:
            int: Total number of parameter samples.
        """
        return 2 * self.n_var * self.num_parameters + 1

    def create_samples(self) -> None:
        """Create parameter samples for local sensitivity analysis.

        For each analysis group, this method constructs a sample matrix
        containing:
        - One reference parameter vector
        - n_var perturbed parameter vectors per parameter (+difference)
        - n_var perturbed parameter vectors per parameter (-difference)

        Samples are stored as an ``xarray.DataArray`` indexed by sample and
        parameter identifiers.
        """
        for group in self.groups:
            # Load reference model state
            r = self.sensitivity_simulation.load_model(
                self.sensitivity_simulation.model_path,
                selections=self.sensitivity_simulation.selections,
            )

            # Compute reference parameter values with all changes applied
            parameter_values: dict[str, float] = (
                self.sensitivity_simulation.parameter_values(
                    r=r,
                    parameters=self.parameters,
                    changes={
                        **self.sensitivity_simulation.changes_simulation,
                        **group.changes,
                    },
                )
            )

            num_samples = 2 * self.n_var * self.num_parameters + 1
            samples = xr.DataArray(
                np.full((num_samples, self.num_parameters), np.nan),
                dims=["sample", "parameter"],
                coords={
                    "sample": range(num_samples),
                    "parameter": [p.uid for p in self.parameters],
                },
                name="samples",
            )

            reference_values = np.array(list(parameter_values.values()))
            for kp, pid in enumerate(parameter_values):
                value = parameter_values[pid]

                for kv in range(self.n_var):
                    samples[2 * self.n_var * kp + kv, :] = reference_values
                    samples[2 * self.n_var * kp + kv, kp] = value * (
                        1.0 + (kv + 1) / self.n_var * self.difference
                    )

                for kv in range(self.n_var):
                    samples[2 * self.n_var * kp + self.n_var + kv, :] = reference_values
                    samples[2 * self.n_var * kp + self.n_var + kv, kp] = value * (
                        1.0 - (kv + 1) / self.n_var * self.difference
                    )

            samples[-1, :] = reference_values
            self.samples[group.uid] = samples

    def calculate_sensitivity(
        self,
        cache_filename: Optional[str] = None,
        cache: bool = False,
    ) -> None:
        """Compute raw and normalized local sensitivities.

        Sensitivities are calculated using a symmetric finite-difference scheme
        for each parameter–output combination.

        Args:
            cache_filename (str, optional):
                Filename used to read/write cached sensitivity results.
            cache (bool, optional):
                Whether cached results should be used.
        """
        data = self.read_cache(cache_filename, cache)
        if data:
            self.sensitivity = data
            return

        for gid in self.group_ids:
            for key in ["raw", "normalized"]:
                self.sensitivity[gid][key] = xr.DataArray(
                    np.full((self.num_parameters, self.num_outputs), np.nan),
                    dims=["parameter", "output"],
                    coords={
                        "parameter": self.parameter_ids,
                        "output": self.output_ids,
                    },
                    name=key,
                )

            sensitivity_raw = self.sensitivity[gid]["raw"]
            sensitivity_normalized = self.sensitivity[gid]["normalized"]

            samples = self.samples[gid]
            results = self.results[gid]

            for kp, _ in enumerate(self.parameters):
                p_ref = samples[-1, kp]
                p_up = samples[
                    (2 * self.n_var * kp) : (2 * self.n_var * kp + self.n_var), kp
                ].values
                p_down = samples[
                    (2 * self.n_var * kp + self.n_var) : (
                        2 * self.n_var * kp + 2 * self.n_var
                    ),
                    kp,
                ].values

                for ko, _ in enumerate(self.outputs):
                    q_ref = results[-1, ko].values
                    q_up = results[
                        (2 * self.n_var * kp) : (2 * self.n_var * kp + self.n_var), ko
                    ].values
                    q_down = results[
                        (2 * self.n_var * kp + self.n_var) : (
                            2 * self.n_var * kp + 2 * self.n_var
                        ),
                        ko,
                    ].values

                    # console.print(f"{q_up=}")
                    # console.print(f"{q_down=}")
                    # console.print(f"{p_up=}")
                    # console.print(f"{p_down=}")
                    delta = (q_up - q_down) / (p_up - p_down)
                    delta_mean = delta.mean()
                    # check linearity within range
                    if not np.isclose(delta_mean, 0.0):
                        max_diff = (delta.max() - delta.min()) / delta_mean
                        if max_diff > 0.10:
                            # this happens if the output is highly nonlinear in the scanned range,
                            # or if large numerical differences exist in the solution (e.g. incorrect discretization)
                            # This can also be due to problems in calculating the respective output (e.g. highly
                            # variable due to numerical fluctuations).
                            # This warning should be taken seriously and be investigated.
                            logger.error(
                                f"Large delta difference: {max_diff*100:.1f} % for {delta}. "
                                f"Parameter {self.parameter_ids[kp]} on output {self.output_ids[ko]}."
                            )
                    sensitivity_raw[kp, ko] = np.sum(delta) / self.n_var

                    sensitivity_normalized[kp, ko] = (
                        sensitivity_raw[kp, ko] * p_ref / q_ref
                    )
                    # console.print(f"{sensitivity_raw=}")

        # create tables
        dfs = self.dfs_sensitivity()
        for kg, gid in enumerate(self.group_ids):
            df = dfs[gid]
            df.to_csv(
                self.results_path / f"{self.prefix}_{kg:>02}_{gid}.tsv",
                sep="\t",
                index=False,
            )

        self.write_cache(
            data=self.sensitivity,
            cache_filename=cache_filename,
            cache=cache,
        )

    def dfs_sensitivity(self) -> dict[str, pd.DataFrame]:
        """Return sensitivity dataframe."""
        dfs: dict[str, pd.DataFrame] = {}
        for gid in self.group_ids:
            items = []
            sensitivity = self.sensitivity[gid]["normalized"].values
            for kp, pid in enumerate(self.parameter_ids):
                for ko, oid in enumerate(self.output_ids):
                    s = sensitivity[kp, ko]
                    classification = sensitivity_classification(s)

                    items.append(
                        {
                            "parameter": pid,
                            "output": oid,
                            "effect": sensitivity_classification_symbol(s),
                            "Sij": s,
                            "|Sij|": np.abs(s),
                            "classification": classification.value,
                            "method": "LocalSensitivity",
                            "difference": self.difference,
                            "n_var": self.n_var,
                        }
                    )
            df = pd.DataFrame(items)
            df.sort_values(
                inplace=True,
                by="|Sij|",
                ascending=False,
                na_position="last",
                ignore_index=True,
            )
            dfs[gid] = df
            console.print(df)
            console.print()

        return dfs

    def plot(self) -> None:
        """Generate plots for normalized local sensitivities.

        Produces heatmaps of normalized sensitivities for each analysis group
        and saves the figures to the results directory.

        Using default cutoff of 0.1 for negligible.
        """
        super().plot()
        console.rule("Plotting", style="white")
        for kg, group in enumerate(self.groups):
            self.plot_sensitivity(
                group_id=group.uid,
                sensitivity_key="normalized",
                cutoff=0.1,
                cluster_rows=False,
                cmap="seismic",
                vcenter=0.0,
                vmin=-2.0,
                vmax=2.0,
                fig_path=(
                    self.results_path / f"{self.prefix}_{kg:>02}_{group.uid}.png"
                ),
            )
