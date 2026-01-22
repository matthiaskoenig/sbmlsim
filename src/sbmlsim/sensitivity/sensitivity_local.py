"""
Local sensitivity analysis based on finite differences.

This module implements a local (derivative-based) sensitivity analysis using
two-sided finite differences around a reference parameter set. Each model
parameter is perturbed individually by a small relative amount, and the
resulting change in model outputs is used to approximate local sensitivities.

The analysis is designed for deterministic simulation models and is
particularly suited for:
- Identifying locally influential parameters
- Debugging and model inspection
- Complementing global sensitivity analyses
- Supporting parameter screening prior to optimization or uncertainty analysis

Sensitivities are computed for each analysis group and output variable, and
can be reported both as raw sensitivities and as normalized, dimensionless
sensitivities.

The implementation builds on the sbmlsim sensitivity framework and integrates
with existing simulation, caching, and plotting utilities.

Notes
-----
For each parameter p_i with reference value p_{i,0}, two perturbed simulations
are generated:
    p_i_plus  = p_{i,0} * (1 + difference)
    p_i_minus = p_{i,0} * (1 - difference)

Raw sensitivities are computed using a symmetric finite-difference scheme:
    S(q_k, p_i) = (q_k(p_i_plus) - q_k(p_i_minus)) / (p_i_plus - p_i_minus)

Normalized sensitivities represent the relative change in output per relative
change in parameter:
    S_norm(q_k, p_i) = S(q_k, p_i) * (p_{i,0} / q_k(p_{i,0}))
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from pymetadata.console import console

from sbmlsim.sensitivity.analysis import (
    SensitivitySimulation,
    AnalysisGroup,
    SensitivityAnalysis,
)
from sbmlsim.sensitivity.parameters import SensitivityParameter


class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on symmetric finite differences.

    This class implements a local sensitivity analysis in which each model
    parameter is perturbed individually by a small relative amount while all
    other parameters are kept at their reference values.

    For each parameter, two simulations are performed (increase and decrease),
    in addition to a reference simulation. Sensitivities are computed for each
    output variable and analysis group.

    Attributes
    ----------
    difference : float
        Relative parameter perturbation used for the finite-difference
        approximation (e.g., 0.01 corresponds to ±1% changes).
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
    ) -> None:
        """Initialize the local sensitivity analysis."""

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
        self.prefix = f"local_d{self.difference}"

    @property
    def num_samples(self) -> int:
        """Return the number of samples required for the analysis.

        The local sensitivity analysis requires:
        - Two simulations per parameter (increase and decrease)
        - One reference simulation

        Returns
        -------
        int
            Total number of parameter samples.
        """
        return 2 * self.num_parameters + 1

    def create_samples(self) -> None:
        """Create parameter samples for the local sensitivity analysis.

        For each analysis group, this method constructs a sample matrix
        containing:
        - One reference parameter vector
        - Two perturbed parameter vectors per parameter (±difference)

        The samples are stored as xarray.DataArray objects and indexed by
        sample and parameter identifiers.
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

            num_samples = 2 * self.num_parameters + 1
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

                samples[2 * kp, :] = reference_values
                samples[2 * kp, kp] = value * (1.0 + self.difference)

                samples[2 * kp + 1, :] = reference_values
                samples[2 * kp + 1, kp] = value * (1.0 - self.difference)

            samples[-1, :] = reference_values
            self.samples[group.uid] = samples

    def calculate_sensitivity(
        self,
        cache_filename: Optional[str] = None,
        cache: bool = False,
    ) -> None:
        """Compute raw and normalized local sensitivity matrices.

        This method calculates two-sided finite-difference sensitivities for
        each parameter–output combination and stores both raw and normalized
        sensitivity matrices.

        Optionally, results can be read from or written to a cache file.

        Parameters
        ----------
        cache_filename : str, optional
            Filename for caching sensitivity results.
        cache : bool, optional
            Whether to read from and/or write to the cache.
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

            for kp, p in enumerate(self.parameters):
                p_ref = samples[-1, kp]
                p_up = samples[2 * kp, kp]
                p_down = samples[2 * kp + 1, kp]

                for ko, oid in enumerate(self.outputs):
                    q_ref = results[-1, ko]
                    q_up = results[2 * kp, ko]
                    q_down = results[2 * kp + 1, ko]

                    sensitivity_raw[kp, ko] = (q_up - q_down) / (p_up - p_down)
                    sensitivity_normalized[kp, ko] = (
                        sensitivity_raw[kp, ko] * p_ref / q_ref
                    )

        self.write_cache(
            data=self.sensitivity,
            cache_filename=cache_filename,
            cache=cache,
        )

    def plot(self):
        super().plot()
        console.rule("Plotting", style="white")
        for kg, group in enumerate(self.groups):
            self.plot_sensitivity(
                group_id=group.uid,
                sensitivity_key="normalized",
                cutoff=0.05,
                cluster_rows=False,
                cmap="seismic",
                vcenter=0.0,
                vmin=-2.0,
                vmax=2.0,
                fig_path=(
                    self.results_path
                    / f"{self.prefix}_{kg:>02}_{group.uid}.png"
                ),
            )
