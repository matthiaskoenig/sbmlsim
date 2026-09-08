# Parameter fitting

Parameter fitting adjusts model parameters so that the simulations of experiments match the experimental data. In `sbmlsim` a fit is an `OptimizationProblem` built from `FitExperiment` objects, which name the simulation experiments and their fit mappings, and `FitParameter` objects with the bounds of the parameters. The problem is run with local or global optimizers of scipy and analysed with `OptimizationAnalysis`.

The example throughout this page is `examples/hctz/`, a whole body model of hydrochlorothiazide with the simulation experiments of two studies and the fit problem built on them.

## Fit mappings

A fit mapping pairs a reference, the experimental data, with an observable, the simulated variable. It is defined in the `fit_mappings()` of a `SimulationExperiment` with `FitData` objects, which are `Data` references (see [Data](data.md)) with their errors and counts:

```python
mapping_code = '''
def fit_mappings(self) -> dict[str, FitMapping]:
    return {
        "fm_hctz5po_4": FitMapping(
            self,
            reference=FitData(
                self, dataset="hctz5po_4", xid="time", yid="value", count="count"
            ),
            observable=FitData(
                self, task="task_hctz_po5", xid="time", yid="[Cve_hctz]"
            ),
            metadata=HCTZMappingMetaData(
                tissue=Tissue.PLASMA,
                route=Route.PO,
                application_form=ApplicationForm.SUSPENSION,
                dosing=Dosing.SINGLE,
                health=Health.HEALTHY,
                fasting=Fasting.FASTED,
            ),
        ),
    }
'''
print(mapping_code)
```

The units of the reference and the observable are compared and the reference is converted to the units of the model. A `MappingMetaData` on a mapping carries application specific information such as the tissue or the dosing and an `outlier` flag which excludes the mapping from the fit. Its fields are keyword only, so that a subclass can add fields without a default; `examples/hctz/experiments/metadata.py` is such a subclass. `sbmlsim.fit.helpers` filters the mappings by their metadata and collects it into a table:

```python
from examples.hctz.fitting.fit_experiments import f_fitexp_pkiv

fit_experiments = f_fitexp_pkiv()
print(fit_experiments)
```

## Fit parameters and experiments

`FitParameter` names a parameter of the model with its start value, bounds and unit, `FitExperiment` names an experiment class and the mappings of it which enter the fit, with optional weights:

```python
from examples.hctz.experiments.studies import Beermann1976
from sbmlsim.fit import FitExperiment, FitParameter

fit_experiments = [
    FitExperiment(
        experiment=Beermann1976,
        mappings=["fm_hctz_iv1_5_urine", "fm_hctz_iv35_4_urine"],
    ),
]
fit_parameters = [
    FitParameter(
        pid="Ka_dis_hctz",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="1/hr",
    ),
    FitParameter(
        pid="KI__HCTZEX_k",
        start_value=1e-6,
        lower_bound=1e-10,
        upper_bound=1,
        unit="1/ml",
    ),
]
print(fit_experiments[0])
print(FitParameter.parameters_to_df(fit_parameters))
```

A `FitExperiment` without mappings uses all fit mappings of its experiment, they are resolved when the problem is initialized. `FitExperiment(use_mapping_weights=True)` weights the mappings by the weights of the `FitMapping` objects, e.g., the counts of the data, instead of the weights given here; setting both is an error.

The optimization runs in logarithmic parameter space, so every parameter needs finite positive bounds and, if it is given, a positive start value.

## The optimization problem

The `OptimizationProblem` collects the fit experiments and parameters with the `base_path` and `data_path` of the experiments:

```python
from examples.hctz import DATA_PATH, HCTZ_PATH
from sbmlsim.fit.optimization import OptimizationProblem

op = OptimizationProblem(
    opid="hctz_iv",
    fit_experiments=fit_experiments,
    fit_parameters=fit_parameters,
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
)
print(op)
```

The problem is picklable, so it is distributed to worker processes; `initialize` then creates the runner, loads the models, resolves the data and calculates the weights. It can be called more than once, e.g., to run a fit and to analyse it afterwards. The options of the initialization define how the residuals are computed:

- `ResidualType`: `ABSOLUTE` residuals or `NORMALIZED` residuals, i.e., relative to the data,
- `LossFunctionType`: `LINEAR`, `SOFT_L1`, `CAUCHY` or `ARCTAN` as in `scipy.optimize.least_squares`, applied to the squared residuals so that the cost is `0.5 * sum(rho(r**2))`,
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

`OptimizationAnalysis` writes the report of a fit: the parameter table with the bounds, waterfall and trajectory plots of the optimizations, the predicted against the measured data points and, when the problem is passed, the fitted curves against the data with the residuals for every mapping. The report is an `index.html` in the output directory; `show_report=True` opens it in a browser and `show_plots=True` shows the figures.

```py
from pathlib import Path

from sbmlsim.fit.analysis import OptimizationAnalysis

analysis = OptimizationAnalysis(
    opt_result=opt_result,
    output_name="hctz_iv",
    output_dir=Path("results"),
    op=op,
    residual=ResidualType.NORMALIZED,
    weighting_curves=[WeightingCurvesType.POINTS],
    weighting_points=WeightingPointsType.ERROR_WEIGHTING,
)
analysis.run()
```

The complete fit problems of the HCTZ model are in `examples/hctz/fitting/`, `fit_experiments.py` builds the subsets of the data and `fitting.py` runs them:

```bash
python -m examples.hctz.fitting.fitting --runs=10 --cores=4 --seed=1234 \
    --method=LSQ --strategy=ALL --subset=PK --name=PK_LSQ_ALL
```

## PEtab

`sbmlsim.fit.petab_omex` packages a [PEtab](https://petab.readthedocs.io) parameter estimation problem, i.e., the model, the condition, observable, measurement and parameter tables and the PEtab YAML, as a COMBINE archive with `create_petab_omex`, so that the problem is exchanged with other tools. `examples/petab/` shows PEtab problems solved with pypesto and AMICI, which are not dependencies of sbmlsim.
