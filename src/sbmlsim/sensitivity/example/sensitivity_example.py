"""Example for sensitivity analysis."""
from pathlib import Path

import numpy as np
import roadrunner
from roadrunner._roadrunner import NamedArray
from pymetadata.console import console

from sbmlsim.sensitivity.analysis import (
    SensitivitySimulation,
    SensitivityOutput,
    AnalysisGroup,
)
from sbmlsim.sensitivity.parameters import (
    SensitivityParameter,
    ParameterType,
    parameters_for_sensitivity_analysis,
)

model_path: Path = Path(__file__).parent / "simple_chain.xml"


# Subgroups to perform sensitivity analysis on
sensitivity_groups: list[AnalysisGroup] = [
    AnalysisGroup(
        uid="low S1",
        name="Low S1",
        changes={"[S1]": 0.1},
        color="tab:red",
    ),
    AnalysisGroup(
        uid="reference S1",
        name="Reference S1",
        changes={"[S1]": 1},
        color="dimgrey",
    ),
    AnalysisGroup(
        uid="high S1",
        name="High S1",
        changes={"[S1]": 10},
        color="tab:blue",
    ),
]


class ExampleSensitivitySimulation(SensitivitySimulation):
    """Simulation for sensitivity calculation."""
    tend = 1000  #
    steps = 1000

    def simulate(self, r: roadrunner.RoadRunner, changes: dict[str, float]) -> dict[str, float]:

        # apply changes and simulate
        all_changes = {
            **self.changes_simulation,  # model
            **changes  # sensitivity
        }
        self.apply_changes(r, all_changes, reset_all=True)
        # ensure tolerances
        r.integrator.setValue("absolute_tolerance", self.init_tolerances)
        s: NamedArray = r.simulate(start=0, end=self.tend, steps=self.steps)

        # pharmacokinetic parameters
        y: dict[str, float] = {}

        # calculate outputs (custom functions)
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
    selections=[
        "time",
        "[S1]",
        "[S2]",
        "[S3]",
    ],
    changes_simulation = {},
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
    parameters: list[SensitivityParameter] = parameters_for_sensitivity_analysis(
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

    from sbmlsim.sensitivity import (
        LocalSensitivityAnalysis,
        SobolSensitivityAnalysis,
        SamplingSensitivityAnalysis,
    )

    sensitivity_path = Path(__file__).parent / "results"
    console.print(SensitivityParameter.parameters_to_df(sensitivity_parameters))

    SamplingSensitivityAnalysis.run_sensitivity_analysis(
        results_path=sensitivity_path / "sampling",
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        # cache_results=False,
        # cache_sensitivity=False,
        N=200,
        seed=1234,
    )

    SobolSensitivityAnalysis.run_sensitivity_analysis(
        results_path=sensitivity_path / "sobol",
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=sensitivity_groups,
        # cache_results=False,
        # cache_sensitivity=False,
        # N=2048
        N=8,
        seed=1234,
    )


