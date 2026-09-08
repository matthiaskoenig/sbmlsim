# Parameter fitting

Parameter fitting adjusts model parameters so that the simulations of experiments match the experimental data. In `sbmlsim` a fit is an `OptimizationProblem` built from `FitExperiment` objects, which name the simulation experiments and their fit mappings, and `FitParameter` objects with the bounds of the parameters. The problem is run with local or global optimizers of scipy and analysed with `OptimizationAnalysis`.

The example throughout this page is `examples/hctz/`, a whole body model of hydrochlorothiazide with the simulation experiments of two studies and the fit problem built on them.

## Fit mappings

A fit mapping pairs a reference, the experimental data, with an observable, the simulated variable. It is defined in the `fit_mappings()` of a `SimulationExperiment` with `FitData` objects, which are `Data` references (see [Data](data.md)) with their errors and counts:

```python
mapping_code = """
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
"""
print(mapping_code)
```

The units of the reference and the observable are compared and the reference is converted to the units of the model. A `MappingMetaData` on a mapping describes its curve with application specific information such as the tissue, the route or the dosing. It describes the data, not what a fit does with the data. Its fields are keyword only, so that a subclass can add fields without a default; `examples/hctz/experiments/metadata.py` is such a subclass.

## Training, validation and outlier data

What a fit does with a curve is decided when the data of the fit is selected, not on the fit mapping: the same curve is training data of one fit and validation data of another. Every `FitExperiment` therefore carries a `MappingKind` for the mappings it selects:

- `MappingKind.TRAINING` (the default): the mappings are fitted, i.e., their residuals enter the cost of the optimization,
- `MappingKind.VALIDATION`: the mappings are not fitted. They are simulated and evaluated together with the training data when the fit is reported, which shows how the fitted parameters describe data they were not fitted on,
- `MappingKind.OUTLIER`: the mappings are not used at all. An optimization problem skips its outliers, they stay in the overview of the data so that it is visible which curves were dropped.

`fit_experiments_by_kind` selects and classifies the data in one step: every kind gets its own filters and the mappings of all kinds are listed in a single overview.

```python
from examples.hctz import DATA_PATH, HCTZ_PATH
from examples.hctz.experiments.studies import Beermann1976
from sbmlsim.fit import MappingKind
from sbmlsim.fit.helpers import (
    filter_empty,
    filter_keys,
    filter_not_keys,
    fit_experiments_by_kind,
)

validation = {"fm_hctz_iv35_4_urine"}
fit_experiments_kinds = fit_experiments_by_kind(
    experiment_classes=[Beermann1976],
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
    filters_by_kind={
        MappingKind.TRAINING: [filter_empty, filter_not_keys(validation)],
        MappingKind.VALIDATION: [filter_keys(validation)],
    },
)
```

The overview ends in a line such as `mappings : 32 (28 training, 2 validation, 2 outlier)`. On the problem, `mapping_counts()` reports how many mappings of each kind it has and `training_indices` and `validation_indices` are their positions in the resolved data.

`sbmlsim.fit.helpers` filters the mappings by their metadata and collects it into a table:

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

The problem is picklable, so it is distributed to worker processes; `initialize` then creates the runner, loads the models, resolves the data and calculates the weights. It takes the `FitSettings`, which decide how the residuals are computed:

```python
from sbmlsim.fit import FitSettings
from sbmlsim.fit.options import (
    LossFunctionType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)

settings = FitSettings(
    residual=ResidualType.NORMALIZED,
    loss_function=LossFunctionType.LINEAR,
    weighting_curves=(WeightingCurvesType.POINTS,),
    weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    relative_tolerance=1e-6,
    absolute_tolerance=1e-6,
)
print(settings)
```

- `ResidualType`: `ABSOLUTE` residuals or `NORMALIZED` residuals, i.e., relative to the data,
- `LossFunctionType`: `LINEAR`, `SOFT_L1`, `CAUCHY` or `ARCTAN` as in `scipy.optimize.least_squares`, applied to the squared residuals so that the cost is `0.5 * sum(rho(r**2))`,
- `WeightingCurvesType`: weighting of the curves by their `MAPPING` weight and by the number of `POINTS`,
- `WeightingPointsType`: `NO_WEIGHTING` or `ERROR_WEIGHTING` of the points by their errors.

The same settings are needed to report a fit, so they are stored with its result. Initializing a problem again with the settings it already has does nothing, i.e., a fit and its report resolve the data once.

## Running the optimization

`run_optimization` samples `size` start points within the bounds (see `sbmlsim.fit.sampling`), runs the optimizer from every start point, in parallel on `n_cores`, and returns an `OptimizationResult`. The progress of the runs is shown on the console:

```py
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.runner import run_optimization

opt_result = run_optimization(
    problem=op,
    settings=settings,
    size=10,
    n_cores=4,
    seed=1234,
    algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
)
```

`OptimizationAlgorithmType.LEAST_SQUARE` is the local least squares optimizer, `DIFFERENTIAL_EVOLUTION` the global one. The `OptimizationResult` holds the fits of all start points with their costs, the optimal parameters `xopt`, the trajectories of the optimizer and the settings the fit was run with; it is stored as JSON and TSV with `to_json` and `to_tsv`, and results of several runs are combined with `OptimizationResult.combine`.

## Parameter sets

A fit produces parameter values, and these values are what a report is made from. `OptimizationResult.parameter_sets` returns the parameters of the best runs as `ParameterSets`, which are stored as JSON:

```py
from pathlib import Path

parameter_sets = opt_result.parameter_sets(size=1)
parameter_sets.to_json(path=Path("parameters.json"))
print(parameter_sets.to_df())
```

A set carries the values, their units, the cost and where it comes from. `OptimizationProblem.parameter_set_model()` gives the values the models start from, which is the natural reference to compare a fit against.

## Reporting the fit

Reporting is separate from optimizing. A `FitReport` is created from the definition of the problem, the settings and one or more parameter sets; it does not need a fit to have been run in the same session:

```py
from sbmlsim.fit import ParameterSets
from sbmlsim.fit.report import FitReport

report = FitReport(
    problem=op,
    settings=settings,
    parameter_sets=ParameterSets.from_json(Path("parameters.json")),
)
report.create(output_dir=Path("results"), name="hctz_iv")
```

The report writes `index.html`, `report.txt`, the `parameters.json` it was made from, the metrics as TSV and the figures: the parameter table with the bounds, the predicted against the measured data points, the costs of the curves and the fitted curves against the data with the residuals for every mapping. `show_report=True` opens the HTML in a browser.

Every parameter set becomes a column of the parameter table and a curve in the plots, so several sets are compared in a single report, e.g., two fits against each other. The first set is the reference the others are compared against.

`FitReport.from_optimization_result` is the shortcut for the report of a fit. It reads the settings from the result, uses the initial values of the model as the reference set, and adds the plots which describe the runs rather than a parameter set, i.e., the optimization traces and the waterfall plot:

```py
report = FitReport.from_optimization_result(problem=op, opt_result=opt_result)
report.create(output_dir=Path("results"), name="hctz_iv")
```

## Metrics

`FitMetrics` calculates the metrics of a parameter set on an initialized problem, i.e., from the data of the fit mappings and the predictions of the model. The column names follow the convention of population pharmacokinetics: `DV` is the measured value, `PRED` the prediction of the population parameters, `IPRED` the prediction of the individual parameters, `RES` and `IRES` the residuals `DV - PRED` and `DV - IPRED`, and `IWRES` the residual weighted with the weights of the problem. A deterministic fit has a single parameter set, so `PRED` is `IPRED` unless a `population_parameter_set` is given:

```py
from sbmlsim.fit import FitMetrics

metrics = FitMetrics(problem=op, parameter_set=parameter_sets[0])
print(metrics.datapoints_df().head())
print(metrics.mappings_df())
print(metrics.summary())
```

`summary()` gives the metrics over all data points: the number of data points `n`, the number of fitted parameters `k`, the `cost`, `MSE`, `RMSE`, `RMSE_w`, `R2` and `AIC`; `mappings_df()` gives them per fit mapping. `MSE`, `RMSE`, `R2` and `AIC` are unweighted metrics of the data and the predictions, so they are dominated by the mappings with the largest values, while `RMSE_w` uses the weighting of the settings. A parameter set can therefore have a larger RMSE and smaller weighted residuals than another one, which is what the weighting is for.

A fit is evaluated on the data it was fitted on and on the data it was not, so `summary(kind=...)` restricts the metrics to a kind of mapping and `summary_df()` has a row for the training data, a row for the validation data and a row over all of them. The `cost` is the objective of the optimization, which is defined on the training data alone, so it is only reported there:

```py
metrics.summary_df()
```

The functions of `sbmlsim.fit.metrics` are used on their own as well: `sse`, `mse`, `rmse`, `aic` and `r_squared` take arrays of residuals or of data and predictions. R² is not the square of a correlation for a non-linear model and is negative when a prediction is worse than the mean of the data.

A report calculates the metrics for every one of its parameter sets and writes them as `metrics.tsv`, `metrics_mappings.tsv` and `datapoints.tsv`, so several sets are compared by their AIC, RMSE and R².

## Running a fit from the command line

Creating the optimization problems, running the optimizations and reporting them is the same for every model, so it lives in `sbmlsim.fit.cli` and a model only defines its fits. A `FitDefinition` is what enters a fit: the fit experiments, the parameters which are adjusted, where the experiments and their data are, and the settings.

```python
from sbmlsim.fit.cli import FitDefinition

definition = FitDefinition(
    fit_experiments=f_fitexp_pkiv,  # called when the fit runs
    parameters=fit_parameters,
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
    settings=settings,
)
print(definition.problem(opid="hctz_iv"))
```

`run_fit` builds the problems for a strategy and runs them: `OptimizationStrategy.ALL` fits all experiments together, i.e., one parameter set describes every experiment, `SINGLE` fits every experiment on its own, which gives the individual parameters of the metrics. It returns a `FitRun` per optimization, which carries the problem and the result and creates the report.

`fit_cli` and `report_cli` are the command line tools around this. A model passes its definitions by name and gets the whole tool:

```python
from sbmlsim.fit.cli import fit_cli

FIT_DEFINITIONS = {"PKIV": definition}


def main() -> None:
    fit_cli(FIT_DEFINITIONS)


print(FIT_DEFINITIONS)
```

Every fit gets an id when it starts, `<problem>_<date>_<time>__<hash>`, e.g. `PK_20260908_144538__ea1ff`. It is the id of the optimization problem, of its result and of the directory of its report, so everything a fit produces carries the same key and sorts by time. The output of a fit is a sequence of sections, each with its own icon: the fit with its strategy, algorithm and paths, the parameters which are optimized with their bounds and units, the settings, the data with the number of fit mappings per experiment and kind, the optimization with its progress, and the report. `sbmlsim.fit.display` renders them and is used on its own as well.

The fit problems of the HCTZ model are in `examples/hctz/fitting/`: `fit_experiments.py` builds the subsets of the data, `parameters.py` holds the fit parameters and `fitting.py` is the definitions plus the four lines above:

```bash
python -m examples.hctz.fitting.fitting --subset=PK --runs=10 --cores=4 \
    --seed=1234 --method=LSQ --strategy=ALL --name=PK_LSQ_ALL
```

`report.py` is `report_cli` on the same definitions. It creates a report from the `parameters.json` of a finished fit without optimizing again, and compares the parameters of several fits:

```bash
python -m examples.hctz.fitting.report results/fit/PK_LSQ_ALL/parameters.json
python -m examples.hctz.fitting.report run1/parameters.json run2/parameters.json
```

## PEtab

`sbmlsim.fit.petab_omex` packages a [PEtab](https://petab.readthedocs.io) parameter estimation problem, i.e., the model, the condition, observable, measurement and parameter tables and the PEtab YAML, as a COMBINE archive with `create_petab_omex`, so that the problem is exchanged with other tools. `examples/petab/` shows PEtab problems solved with pypesto and AMICI, which are not dependencies of sbmlsim.
