# Sensitivity analysis

Sensitivity analysis quantifies how the outputs of a model depend on its parameters. `sbmlsim.sensitivity` implements local sensitivities by finite differences and the global methods of [SALib](https://salib.readthedocs.io): sampling based uncertainty analysis, the Morris screening method, Sobol indices and the Fourier amplitude sensitivity test (FAST), see [References](references.md#sensitivity-analysis). All methods share the same description of the problem, i.e., the simulation which computes the outputs, the parameters with their bounds, and the groups of model conditions under which the analysis is run.

## The sensitivity simulation

A `SensitivitySimulation` defines what is simulated and which scalar outputs are computed from the timecourse. It is a subclass with a `simulate` method which receives the roadrunner instance and the parameter changes of one sample and returns the outputs. The samples are simulated in worker processes, so the class has to live in an importable module, here `examples/sensitivity/sensitivity_example.py`:

```py
import numpy as np
import roadrunner

from sbmlsim.sensitivity import SensitivityOutput, SensitivitySimulation


class ChainSimulation(SensitivitySimulation):
    """Simulation of the simple chain model with its outputs."""

    def simulate(
        self, r: roadrunner.RoadRunner, changes: dict[str, float]
    ) -> dict[str, float]:
        self.apply_changes(r, {**self.changes_simulation, **changes}, reset_all=True)
        s = r.simulate(start=0, end=1000, steps=1000)
        t = s["time"]
        y: dict[str, float] = {}
        for key in ["S1", "S2", "S3"]:
            v = s[f"[{key}]"]
            y[f"[{key}]_auc"] = np.trapezoid(y=v, x=t)
        y["[S2]_max"] = np.max(s["[S2]"])
        return y
```

The outputs are declared as `SensitivityOutput` objects, the changes of the simulation (e.g., a dose) as `changes_simulation`:

```python
from examples.sensitivity.sensitivity_example import (
    ExampleSensitivitySimulation,
    model_path,
)
from sbmlsim.sensitivity import SensitivityOutput

simulation = ExampleSensitivitySimulation(
    model_path=model_path,
    selections=["time", "[S1]", "[S2]", "[S3]"],
    changes_simulation={},
    outputs=[
        SensitivityOutput(uid="[S1]_auc", name="[S1] AUC", unit=None),
        SensitivityOutput(uid="[S2]_auc", name="[S2] AUC", unit=None),
        SensitivityOutput(uid="[S3]_auc", name="[S3] AUC", unit=None),
        SensitivityOutput(uid="[S2]_max", name="[S2] maximum", unit=None),
        SensitivityOutput(uid="[S2]_tmax", name="[S2] time maximum", unit=None),
    ],
)
```

## Parameters and groups

`SensitivityParameter.parameters_from_sbml` reads the constant parameters of the model with their values and units; the bounds of the analysis are set relative to the values or from data:

```python
from sbmlsim.sensitivity import AnalysisGroup, SensitivityParameter

parameters = SensitivityParameter.parameters_from_sbml(
    sbml_path=model_path, exclude_ids=None, exclude_na=True, exclude_zero=True
)
for p in parameters:
    p.lower_bound = p.value * 0.85
    p.upper_bound = p.value * 1.15
print(SensitivityParameter.parameters_to_df(parameters))
```

An `AnalysisGroup` is a condition of the model under which the sensitivities are computed, e.g., a low and a high initial concentration. Every group is analysed separately and the results are compared across groups:

```python
groups = [
    AnalysisGroup(uid="lowS1", name="Low S1", changes={"[S1]": 0.1}, color="tab:red"),
    AnalysisGroup(uid="highS1", name="High S1", changes={"[S1]": 10}, color="tab:blue"),
]
```

## Running an analysis

Every analysis is created with the simulation, the parameters, the groups and a `results_path`; `execute` creates the samples, simulates them (in parallel on `n_cores`, cached with `cache_results=True`) and computes the sensitivities, `plot` writes the figures into the results path:

```python
from pathlib import Path

from sbmlsim.sensitivity import LocalSensitivityAnalysis

sa_local = LocalSensitivityAnalysis(
    sensitivity_simulation=simulation,
    parameters=parameters,
    groups=groups,
    results_path=Path.cwd() / "sensitivity" / "local",
    difference=0.01,
    n_cores=1,
    seed=1234,
)
sa_local.execute()
print(sa_local.sensitivity_df(group_id="lowS1", key="normalized"))
```

The local analysis varies every parameter by the relative `difference` around its value and reports the raw and the normalized sensitivities, i.e., the relative change of the output per relative change of the parameter.

The global methods sample the parameter space within the bounds:

```py
from sbmlsim.sensitivity import (
    FASTSensitivityAnalysis,
    MorrisSensitivityAnalysis,
    SamplingSensitivityAnalysis,
    SobolSensitivityAnalysis,
)

kwargs = dict(
    sensitivity_simulation=simulation, parameters=parameters, groups=groups, n_cores=4
)
sa_sampling = SamplingSensitivityAnalysis(results_path=Path("sampling"), N=1000, **kwargs)
sa_sobol = SobolSensitivityAnalysis(results_path=Path("sobol"), N=4096, **kwargs)
sa_fast = FASTSensitivityAnalysis(results_path=Path("fast"), N=1000, **kwargs)
sa_morris = MorrisSensitivityAnalysis(
    results_path=Path("morris"), N=100, num_levels=4, optimal_trajectories=25, **kwargs
)
for sa in [sa_sampling, sa_sobol, sa_fast, sa_morris]:
    sa.execute()
    sa.plot()
```

- **Sampling** draws `N` parameter sets and reports the distribution of every output (mean, median, standard deviation, coefficient of variation, quantiles), i.e., an uncertainty analysis.
- **Sobol** computes the first order (`S1`) and total (`ST`) variance based indices with the Saltelli sampling scheme; `N` samples per parameter.
- **FAST** computes first order and total indices with the extended Fourier amplitude sensitivity test.
- **Morris** computes the elementary effects `mu`, `mu_star` and `sigma` of the screening method, with `num_levels` grid levels and `optimal_trajectories` trajectories.

The sensitivities are stored as `xarray.DataArray` objects per group and key (`sa.sensitivity[group_id][key]`), returned as data frames with `sensitivity_df`, and written as tables and figures into the results path. `sbmlsim.sensitivity.classification` groups parameters by their sensitivities across outputs.

The complete example with all methods is `examples/sensitivity/sensitivity_example.py`.
