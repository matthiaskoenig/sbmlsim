from pathlib import Path

import xarray as xr

from sbmlsim.sensitivity.analysis import SensitivitySimulation, AnalysisGroup, \
    SensitivityAnalysis
from sbmlsim.sensitivity.parameters import SensitivityParameter


class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on local differences.

    Each model parameter p_i is perturbed individually by ±1% relative to its
    reference value p_{i,0}. Local sensitivities are computed using a symmetric
    midpoint finite-difference approximation:

        S(q_k, p_i) =
            (q_k(p_i_plus) - q_k(p_i_minus)) / (p_i_plus - p_i_minus),

    where:
        p_i_plus  = p_{i,0} * (1 + 0.01)
        p_i_minus = p_{i,0} * (1 - 0.01)

    Sensitivities are normalized to obtain dimensionless measures representing
    the relative change in model output per relative change in the parameter:

        S_norm(q_k, p_i) =
            ((q_k(p_i_plus) - q_k(p_i_minus)) / (p_i_plus - p_i_minus))
            * (p_{i,0} / q_k(p_{i,0}))

    param difference: change for calculation of local sensitivity (0.01 = 1% change)
    """

    def __init__(self, sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 difference: float = 0.01,
                 **kwargs) -> None:

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)

        self.difference: float = difference

    @property
    def num_samples(self) -> int:
        """Number of parameter samples to simulate."""

        return 2 * self.num_parameters + 1

    def create_samples(self) -> None:
        """Create samples for the local sensitivity analysis.

        This requires a reference simulation and 2 simulations per parameter
        with increase and decrease of the respective parameter.
        """
        for group in self.groups:

            # Calculate the parameter values in the reference state
            r = self.sensitivity_simulation.load_model(self.sensitivity_simulation.model_path, selections=self.sensitivity_simulation.selections)

            # parameter values require simulation and group changes to be applied
            parameter_values: dict[str, float] = self.sensitivity_simulation.parameter_values(
                r=r,
                parameters=self.parameters,
                changes={
                    **self.sensitivity_simulation.changes_simulation,
                    **group.changes,
                }
            )

            # (num_samples x num_outputs)
            num_samples = 2 * self.num_parameters + 1
            samples = xr.DataArray(
                np.full((num_samples, self.num_parameters), np.nan),
                dims=["sample", "parameter"],
                coords={"sample": range(num_samples), "parameter": [p.uid for p in self.parameters]},
                name="samples"
            )

            reference_values = np.array(list(parameter_values.values()))
            for kp, pid in enumerate(parameter_values):
                value = parameter_values[pid]

                # right sided changes
                samples[2*kp, :] = reference_values
                samples[2*kp, kp] = value * (1.0 + self.difference)  # up
                samples[2 * kp + 1 , :] = reference_values
                samples[2 * kp + 1, kp] = value * (1.0 - self.difference) # down

            # reference values
            samples[-1, :] = reference_values # reference

            self.samples[group.uid] = samples

    def calculate_sensitivity(self, cache_filename: Optional[str] = None, cache: bool = False) -> None:
        """Calculate the two-sided local sensitivity matrix."""
        data = self.read_cache(cache_filename, cache)
        if data:
            self.sensitivity = data
            return

        for gid in self.group_ids:
            # num_parameters x num_outputs
            for key in ["raw", "normalized"]:
                self.sensitivity[gid][key] = xr.DataArray(
                np.full((self.num_parameters, self.num_outputs), np.nan),
                dims=["parameter", "output"],
                coords={"parameter": self.parameter_ids,
                        "output": self.output_ids},
                name=key
            )

            sensitivity_raw = self.sensitivity[gid]["raw"]
            sensitivity_normalized = self.sensitivity[gid]["normalized"]

            samples = self.samples[gid]
            results = self.results[gid]

            for kp, p in enumerate(self.parameters):
                p_ref = samples[-1, kp]
                p_up = samples[2*kp, kp]
                p_down = samples[2 * kp + 1, kp]

                for ko, oid in enumerate(self.outputs):
                    # num_samples x num_outputs
                    q_ref = results[-1, ko]
                    q_up = results[2*kp, ko]
                    q_down = results[2 * kp + 1, ko]

                    # two-sided sensitivity
                    sensitivity_raw[kp, ko] = (q_up - q_down) / (p_up - p_down)
                    # normalized: relative change in output per relative change in parameter
                    sensitivity_normalized[kp, ko] = sensitivity_raw[kp, ko] * p_ref/q_ref

        # write to cache
        self.write_cache(data=self.sensitivity, cache_filename=cache_filename, cache=cache)


def local_sensitivity_analysis():
    """Local sensitivity analysis"""
    console.rule("LOCAL SENSITIVITY ANALYSIS", style="blue bold", align="center")

    sensitivity_simulation = CanagliflozinSensitivitySimulation.sensitivity_simulation()
    parameters = sensitivity_simulation.sensitivity_parameters()

    sa = LocalSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        results_path=RESULTS_PATH / "sensitivity",
        seed=1234,
        difference=0.01,  # 1% change
    )

    console.rule("Samples", style="white")
    sa.create_samples()

    console.rule("Results", style="white")
    sa.simulate_samples()
    console.print(sa.results)

    console.rule("Sensitivity", style="white")
    sa.calculate_sensitivity()
    console.print(sa.sensitivity)

    console.rule("Plotting", style="white")
    for kg, group in enumerate(sa.groups):
        sa.plot_sensitivity(
            group_id=group.uid,
            sensitivity_key="normalized",
            # title=f"{group.name}",
            cutoff=0.05,
            cluster_rows = False,
            cmap = "seismic",
            vcenter=0.0,
            vmin=-2.0,
            vmax=2.0,
            fig_path=sa.results_path / f"local_sensitivity_{kg:>02}_{group.uid}_{sa.difference}.png",
        )
