"""Example for sensitivity analysis."""
from pathlib import Path

import numpy as np
import roadrunner
from pymetadata.console import console

from sbmlsim.sensitivity import (
    SensitivityParameter,
    SensitivityOutput,
    AnalysisGroup,
    SensitivitySimulation,
)

# model
model_path: Path = Path(__file__).parent / "simple_chain.xml"

# subgroups to perform sensitivity analysis on
sensitivity_groups: list[AnalysisGroup] = [
    AnalysisGroup(
        uid="lowS1",
        name="Low S1",
        changes={"[S1]": 0.1},
        color="tab:red",
    ),
    AnalysisGroup(
        uid="refS1",
        name="Reference S1",
        changes={"[S1]": 1},
        color="dimgrey",
    ),
    AnalysisGroup(
        uid="highS1",
        name="High S1",
        changes={"[S1]": 10},
        color="tab:blue",
    ),
]


class ExampleSensitivitySimulation(SensitivitySimulation):
    """Simulation for sensitivity calculation."""
    tend = 1000
    steps = 1000

    def simulate(self, r: roadrunner.RoadRunner, changes: dict[str, float]) -> dict[
        str, float]:

        # apply changes and simulate
        all_changes = {
            **self.changes_simulation,  # model
            **changes  # sensitivity
        }
        self.apply_changes(r, all_changes, reset_all=True)

        # ensure identical tolerances on all simulations
        r.integrator.setValue("absolute_tolerance", self.init_tolerances)
        s = r.simulate(start=0, end=self.tend, steps=self.steps)

        # calculate outputs y (custom functions)
        # this can be registered functions calculating scalars based on subsets of the
        # timecourse vectors
        y: dict[str, float] = {}
        t = s["time"]
        for key in "S1", "S2", "S3":
            rr_key = f"[{key}]"
            v = s[rr_key]
            t_idx = np.argmax(v)
            if key in ["S2", "S3"]:
                y[f"{rr_key}_tmax"] = t[t_idx]
                y[f"{rr_key}_max"] = v[t_idx]
            y[f"{rr_key}_auc"] = np.trapezoid(y=v, x=t)

        return y


sensitivity_simulation = ExampleSensitivitySimulation(
    model_path=model_path,
    selections=["time", "[S1]", "[S2]", "[S3]"],
    changes_simulation={},
    outputs=[
        SensitivityOutput(uid='[S1]_auc', name='[S1] AUC', unit=None),
        SensitivityOutput(uid='[S2]_tmax', name='[S2] time maximum', unit=None),
        SensitivityOutput(uid='[S2]_max', name='[S2] maximum', unit=None),
        SensitivityOutput(uid='[S2]_auc', name='[S2] AUC', unit=None),
        SensitivityOutput(uid='[S3]_tmax', name='[S3] time maximum', unit=None),
        SensitivityOutput(uid='[S3]_max', name='[S3] maximum', unit=None),
        SensitivityOutput(uid='[S3]_auc', name='[S3] AUC', unit=None),
    ]
)


def _sensitivity_parameters() -> list[SensitivityParameter]:
    """Definition of parameters and bounds for sensitivity analysis."""
    console.rule("Parameters", style="white")
    parameters: list[SensitivityParameter] = SensitivityParameter.parameters_from_sbml(
        sbml_path=model_path,
        exclude_ids=None,
        exclude_na=True,
        exclude_zero=True,
    )
    # setting bounds;
    bounds_fraction = 0.15  # fraction of bounds relative to value
    for p in parameters:
        if np.isnan(p.lower_bound) and np.isnan(p.upper_bound):
            p.lower_bound = p.value * (1 - bounds_fraction)
            p.upper_bound = p.value * (1 + bounds_fraction)

    return parameters


sensitivity_parameters = _sensitivity_parameters()

if __name__ == "__main__":
    import multiprocessing
    from sbmlsim.sensitivity import (
        SobolSensitivityAnalysis,
        LocalSensitivityAnalysis,
        SamplingSensitivityAnalysis,
        FASTSensitivityAnalysis,
    )

    sensitivity_path = Path(__file__).parent / "results"
    df = SensitivityParameter.parameters_to_df(sensitivity_parameters)
    df.to_csv(sensitivity_path / "parameters.tsv", sep="\t", index=False)
    console.print(df)

    settings = {
        "cache_results": True,
        "n_cores": int(round(0.9 * multiprocessing.cpu_count())),
        "seed": 1234
    }

    sa_sampling = SamplingSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        results_path=sensitivity_path / "sampling",
        N=1000,
        **settings,
    )

    sa_local = LocalSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        results_path=sensitivity_path / "local",
        difference=0.01,
        **settings,
    )

    sa_sobol = SobolSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[1]],
        results_path=sensitivity_path / "sobol",
        N=4096,
        **settings,
    )

    sa_fast = FASTSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        results_path=sensitivity_path / "fast",
        N=1000,
        **settings,
    )

    sas = [
        sa_local,
        # sa_sampling,
        # sa_sobol,
        # sa_fast,
    ]
    for sa in sas:
        sa.execute()
        sa.plot()
