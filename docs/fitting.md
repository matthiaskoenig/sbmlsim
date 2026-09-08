# Parameter fitting

Parameter fitting adjusts model parameters so that the simulations of experiments match the experimental data. In `sbmlsim` a fit is an `OptimizationProblem` built from `FitExperiment` objects, which name the simulation experiments and their fit mappings, and `FitParameter` objects with the bounds of the parameters. The problem is run with local or global optimizers of scipy and analysed with `OptimizationAnalysis`.

## Fit mappings

A fit mapping pairs a reference, the experimental data, with an observable, the simulated variable. It is defined in the `fit_mappings()` of a `SimulationExperiment` with `FitData` objects, which are `Data` references (see [Data](data.md)) with their errors and counts:

```python
from sbmlsim.fit import FitData, FitMapping

# inside SimulationExperiment.fit_mappings()
#   reference: the dataset column and its error, observable: the task variable
mapping_code = """
def fit_mappings(self) -> dict[str, FitMapping]:
    return {
        "fm_mid_iv": FitMapping(
            self,
            reference=FitData(
                self, dataset="Fig1_midazolam_iv", xid="time", yid="mean", yid_sd="mean_sd"
            ),
            observable=FitData(self, task="task_mid_iv", xid="time", yid="[Cve_mid]"),
            metadata=None,
        ),
    }
"""
print(mapping_code)
```

The units of the reference and the observable are compared and the reference is converted to the units of the model. A `MappingMetaData` on a mapping carries application specific information such as the tissue or the dosing and an `outlier` flag which excludes the mapping from the fit; `sbmlsim.fit.helpers` collects this metadata into a table.

## Fit parameters and experiments

`FitParameter` names a parameter of the model with its start value, bounds and unit, `FitExperiment` names an experiment class and the mappings of it which enter the fit, with optional weights:

```python
from sbmlsim.fit import FitExperiment, FitParameter

from examples.midazolam.experiments.kupferschmidt1995 import Kupferschmidt1995

fit_experiments = [
    FitExperiment(experiment=Kupferschmidt1995, mappings=["fm_mid_iv", "fm_mid1oh_iv"]),
]
fit_parameters = [
    FitParameter(
        pid="LI__MIDIM_Vmax",
        start_value=0.1,
        lower_bound=1e-5,
        upper_bound=1e3,
        unit="mmole_per_min",
    ),
    FitParameter(
        pid="KI__MID1OHEX_Km",
        start_value=100,
        lower_bound=1e-5,
        upper_bound=1e-1,
        unit="mM",
    ),
]
print(fit_experiments[0])
print(FitParameter.parameters_to_df(fit_parameters))
```

`FitExperiment(use_mapping_weights=True)` weights the mappings by the weights of the `FitMapping` objects, e.g., the counts of the data, instead of the weights given here.

## The optimization problem

The `OptimizationProblem` collects the fit experiments and parameters with the `base_path` and `data_path` of the experiments:

```python
from examples.midazolam import MIDAZOLAM_PATH
from sbmlsim.fit.optimization import OptimizationProblem

op = OptimizationProblem(
    opid="mid_iv",
    fit_experiments=fit_experiments,
    fit_parameters=fit_parameters,
    base_path=MIDAZOLAM_PATH,
    data_path=MIDAZOLAM_PATH / "data",
)
print(op)
```

The problem is picklable, so it is distributed to worker processes; `initialize` then creates the runner, loads the models, resolves the data and calculates the weights. The options of the initialization define how the residuals are computed:

- `ResidualType`: `ABSOLUTE` residuals or `NORMALIZED` residuals, i.e., relative to the data,
- `LossFunctionType`: `LINEAR`, `SOFT_L1`, `CAUCHY` or `ARCTAN` as in `scipy.optimize.least_squares`,
- `WeightingCurvesType`: weighting of the curves by their `MAPPING` weight and by the number of `POINTS`,
- `WeightingPointsType`: `NO_WEIGHTING` or `ERROR_WEIGHTING` of the points by their errors.

## Running the optimization

`run_optimization` samples `size` start points within the bounds (see `sbmlsim.fit.sampling`), runs the optimizer from every start point, in parallel on `n_cores`, and returns an `OptimizationResult`:

```py
from sbmlsim.fit.options import (
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.runner import run_optimization

opt_result = run_optimization(
    problem=op,
    size=10,
    n_cores=4,
    seed=1234,
    algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
    residual=ResidualType.NORMALIZED,
    weighting_curves=[WeightingCurvesType.POINTS],
    weighting_points=WeightingPointsType.ERROR_WEIGHTING,
)
```

`OptimizationAlgorithmType.LEAST_SQUARE` is the local least squares optimizer, `DIFFERENTIAL_EVOLUTION` the global one. The `OptimizationResult` holds the fits of all start points with their costs, the optimal parameters `xopt`, and the trajectories of the optimizer; it is stored as JSON and TSV with `to_json` and `to_tsv`, and results of several runs are combined with `OptimizationResult.combine`.

## Analysing the fit

`OptimizationAnalysis` writes the report of a fit: the parameter table with the bounds, the correlation of the parameters over the fits, waterfall and trajectory plots of the optimizations, and, when the problem is passed, the fitted curves against the data with the residuals for every mapping:

```py
from sbmlsim.fit.analysis import OptimizationAnalysis

analysis = OptimizationAnalysis(
    opt_result=opt_result,
    output_name="mid_iv",
    output_dir=Path("results"),
    op=op,
    residual=ResidualType.NORMALIZED,
    weighting_curves=[WeightingCurvesType.POINTS],
    weighting_points=WeightingPointsType.ERROR_WEIGHTING,
)
analysis.run()
```

The complete fitting problems of the midazolam model are in `examples/midazolam/fitting_problems.py` and run by `examples/midazolam/fitting_example.py`.

## PEtab

`sbmlsim.fit.petab_omex` packages a [PEtab](https://petab.readthedocs.io) parameter estimation problem, i.e., the model, the condition, observable, measurement and parameter tables and the PEtab YAML, as a COMBINE archive with `create_petab_omex`, so that the problem is exchanged with other tools. `examples/petab/` shows PEtab problems solved with pypesto and AMICI, which are not dependencies of sbmlsim.
