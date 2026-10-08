# Parameter fitting

Parameter fitting adjusts model parameters so that the simulations of experiments match the experimental data. In `sbmlsim` a fit is an `OptimizationProblem` built from `FitMappingCollection` objects, which name the simulation experiments and their fit mappings, and `FitParameter` objects with the bounds of the parameters. The problem is run with local or global optimizers of scipy, reported with `FitReport`, and the identifiability of the fitted parameters is analysed with the profile likelihood and the Fisher information, see [Identifiability](identifiability.md).

The example throughout this page is `examples/hctz_fitting/`, a whole body model of hydrochlorothiazide with the simulation experiments of two studies and the fit problem built on them.

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

The units of the reference and the observable are compared and the reference is converted to the units of the model. A `MappingMetaData` on a mapping describes its curve with application specific information such as the tissue, the route or the dosing. It describes the data, not what a fit does with the data. Its fields are keyword only, so that a subclass can add fields without a default; `examples/hctz_fitting/experiments/metadata.py` is such a subclass.

## Training, validation and outlier data

What a fit does with a curve is decided when the data of the fit is selected, not on the fit mapping: the same curve is training data of one fit and validation data of another. Every `FitMappingCollection` therefore carries a `MappingKind` for the mappings it selects: `TRAINING` enters the cost, `VALIDATION` and `OUTLIER` are evaluated but not fitted, and `EXCLUDED` is not used at all. Outlier and excluded say different things: an outlier is a decision about the data, i.e. it is not usable, and an exclusion is a decision about the fit, i.e. the data is not what the fit is about or the model does not describe what was measured, e.g. an arm of a study with a coadministration the model has no interaction for.

- `MappingKind.TRAINING` (the default): the mappings are fitted, i.e., their residuals enter the cost of the optimization,
- `MappingKind.VALIDATION`: the mappings are not fitted. They are simulated and evaluated together with the training data when the fit is reported, which shows how the fitted parameters describe data they were not fitted on,
- `MappingKind.OUTLIER`: the mappings are not fitted, because the data is not usable. They are resolved and evaluated like the validation data, so a report has their metrics and their figures and the decision to drop a curve can be checked against the model,
- `MappingKind.EXCLUDED`: the mappings are not used at all. An optimization problem does not resolve them, so they have no metrics and are not part of a report; they stay in the overview of the data of a `FitDefinition` so that it is visible which curves were dropped.

`sbmlsim.fit.helpers` selects the data of a fit from the complete list of fit mappings of its simulation experiments. `FitMappings` instantiates the experiments once, which loads the models and the datasets, and `FitMappings.select` sets the kind of every mapping in three ordered steps:

1. **The filters select the training data.** A filter is a callable of the key of a fit mapping and the `FitMapping`, e.g. a test on its `MappingMetaData`. A mapping which passes every filter is training data, a mapping which fails one is `EXCLUDED`. No filters select every mapping.
2. **The outliers are named by their keys.** An outlier is training data whose values are not usable, so it is tagged once for the complete list of mappings and not per fit: it is an `OUTLIER` in every fit whose filters select it. An outlier the filters exclude stays excluded.
3. **Part of the training data is the validation data.** It is selected from what is left, by the keys of the mappings or by a filter, and is `VALIDATION`. Everything else is `TRAINING`.

The steps are ordered, so a mapping which hits several has one kind: excluded beats outlier, outlier beats validation and validation beats training. The result is a `MappingSelection` with the kind of every mapping, the overview table `df` and the `collections` a `FitDefinition` is defined with, one `FitMappingCollection` per experiment and kind; `select_mapping_collections` does all of it in one call.

```python
from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.studies import Beermann1976, Patel1984
from sbmlsim.fit.helpers import FitMappings

fit_mappings = FitMappings(
    experiment_classes=[Beermann1976, Patel1984],
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
)
selection = fit_mappings.select(
    filters=[lambda key, fm: "urine" in key],  # the training data
    outliers={"fm_hctz_iv1_5_urine"},  # not usable, tagged once for all fits
    validation={"fm_200_tab_urine", "fm_200_sus_urine"},  # evaluated, not fitted
)
mapping_collections = selection.collections
```

The overview ends in a line such as `mappings : 32 (28 training, 2 validation, 2 outlier)`. On the problem, `mapping_counts()` reports how many mappings of each kind it has and `training_indices`, `validation_indices` and `outlier_indices` are their positions in the resolved data.

`examples/hctz_fitting/fitting/mapping_collections.py` is the selection of the reference problem: the outliers and the validation data are named once, and every fit problem is a list of filters on the metadata of the mappings.

```python
from examples.hctz_fitting.fitting.mapping_collections import f_collections_pk

mapping_collections = f_collections_pk()
print(mapping_collections)
```

## Fit parameters and experiments

`FitParameter` names a parameter of the model with its start value, bounds and unit, `FitMappingCollection` names a simulation experiment class and the mappings of it which enter the fit together, with optional weights:

```python
from examples.hctz_fitting.experiments.studies import Beermann1976
from sbmlsim.fit import FitMappingCollection, FitParameter

mapping_collections = [
    FitMappingCollection(
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
print(mapping_collections[0])
print(FitParameter.parameters_to_df(fit_parameters))
```

A `FitMappingCollection` without mappings uses all fit mappings of its experiment, they are resolved when the problem is initialized. `FitMappingCollection(use_mapping_weights=True)` weights the mappings by the weights of the `FitMapping` objects, e.g., the counts of the data, instead of the weights given here; setting both is an error.

`FitSettings.parameter_scale` is the space the optimizer searches the parameters in: `LOG10` by default, because a rate constant spans orders of magnitude and an optimizer on the linear scale spends its steps on the largest parameters, and `LOG` or `LINEAR` if a problem wants them. The bounds, the start values and the fitted parameters are always on the linear scale, i.e. in the units of the model, only the search happens in the scaled space. A logarithm needs finite positive bounds and a positive start value. A parameter on the linear scale may have infinite bounds, which the local optimizer takes; the global optimizer samples a finite box and rejects them. `FitParameter.scale` gives one parameter a scale of its own, e.g. the linear scale for a parameter which is negative or zero while the others are searched on the logarithmic scale of the settings.

The scale is a property of the optimization and not of the model or of the data, which is why it is part of the settings; PEtab v2 removed the `parameterScale` of its parameter table for the same reason.

### One parameter per subset of the data

A `FitParameter` can be estimated for one part of the data only, leaving the model's own value for the rest, e.g. a dissolution rate estimated from the oral data and left untouched for the intravenous data, which does not depend on it: `target` says which entity of the model the value is written to (`pid` by default) and `mappings` is a filter, or an iterable of filters, which selects the fit mappings the parameter applies to; several parameters share one `target` when each of them covers a different part of the data, and `FitParameter.is_versioned`/`target_id` are the two accessors a fit, a report and the PEtab layer read instead of `mappings`/`target` directly.

```python
from examples.hctz_fitting.experiments.metadata import Route
from examples.hctz_fitting.fitting.mapping_collections import _metadata
from sbmlsim.fit import FitMapping, FitParameter


def is_oral(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the oral data, which is where a dissolution rate applies."""
    return _metadata(fit_mapping).route == Route.PO


#: the dissolution estimated for the oral data only. A selector must be a
#: module level function: the workers of a parallel fit unpickle it. There is
#: no intravenous version: an intravenous dose has nothing to dissolve, so
#: `Ka_dis_hctz` has no effect on the intravenous curves at all, and versioning
#: it there too would add a parameter no curve constrains. Leaving the
#: intravenous mappings unversioned keeps them on the model's shared value,
#: which is the correct value for them and exactly what
#: `problem.parameter_mapping.coverage()` reports as uncovered below.
PARAMETERS_BY_ROUTE = [
    FitParameter(
        pid="Ka_dis_hctz_po",
        start_value=0.35,
        lower_bound=0.01,
        upper_bound=10.0,
        unit="1/hr",
        target="Ka_dis_hctz",
        mappings=is_oral,
    ),
    # ... the rest of the parameters, unversioned
]
```

A selector must be a module level function, not a lambda and not a closure: `OptimizationProblem.__getstate__` reduces a problem to its uninitialized definition for the workers of a parallel fit, and the fit parameters, selectors included, travel with it; a lambda does not pickle and a parallel fit fails when the workers start, not when the fit is defined, which is why `examples/hctz_fitting/fitting/parameters.py` (`PARAMETERS_BY_ROUTE`) defines `is_oral` next to the parameters rather than inline.

`sbmlsim.fit.parameter_mapping.ParameterMapping` resolves every selector to the simulation groups of the initialized problem and validates the binding: two parameters must not write one target in one simulation, a selector must not split a simulation, i.e. select some but not all of the fit mappings which share a simulation, and the versions of a target must agree on their unit, since the unit is how the value reaches the model. A version whose selector matches no fit mapping only warns, because it is usually a mistyped filter rather than an intended gap, and the parameter would otherwise sit in the parameter vector without ever changing the model.

`problem.parameter_mapping.coverage()` reports what every parameter reaches, and the console and the HTML report of a fit show it as a table once a parameter is versioned: the parameter, the target, how many of the problem's simulations it covers, and the simulations it does not reach. An uncovered simulation keeps the value the model has for the target rather than being an error, e.g. an oral dissolution rate does nothing on the intravenous data, and the coverage table makes that a fact which is read rather than one which is discovered later.

The PEtab v2 export writes a version as a condition: the condition assigns the target the value of the estimated parameter, and every experiment the version covers references it on period 0. The selector itself is a python callable and does not round trip; PEtab stores the resolution, so a problem which is read back selects the same fit mappings by their id (`sbmlsim.fit.helpers.filter_keys`) rather than by the original rule. The fit, its cost and its parameters are the same, i.e. the round trip is exact in effect and not in source form. An experiment whose fit mappings span several `MappingKind`s is written as one PEtab experiment per kind, so a problem read back can report a higher coverage count than the fit which was written even though the binding and the simulations themselves are unchanged; this is the `experiment-split` gap of `sbmlsim.fit.petab_v2.gaps`.

## The optimization problem

The `OptimizationProblem` collects the fit mapping collections and parameters with the `base_path` and `data_path` of the experiments:

```python
from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from sbmlsim.fit.optimization import OptimizationProblem

op = OptimizationProblem(
    opid="hctz_iv",
    mapping_collections=mapping_collections,
    fit_parameters=fit_parameters,
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
)
print(op)
```

The problem is picklable, so it is distributed to worker processes; `initialize` then creates the runner, loads the models, resolves the data and calculates the weights, and groups the fit mappings which share a simulation. Several mappings read different observables of the same simulation, e.g. the plasma concentration and the amount in urine of one dosing, and such a group is simulated once per evaluation of the residuals with the selections of all of its mappings, which is where the time of a fit goes. It takes the `FitSettings`, which decide how the residuals are computed:

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

`variable_step_size`, `relative_tolerance`, `absolute_tolerance` and `initial_time_step` are the settings of the integrator of the simulations of the fit, see [Simulations](simulation.md#selections-and-integrator-settings). The same settings are needed to report a fit, so they are stored with its result. Initializing a problem again with the settings it already has does nothing, i.e., a fit and its report resolve the data once.

## Running the optimization

`run_optimization` samples `size` start points within the bounds (see `sbmlsim.fit.sampling`; a parameter with an infinite bound is not sampled and starts from its start value, and a parameter with the linear `FitParameter.scale` is sampled uniformly also under a logarithmic sampling), runs the optimizer from every start point, in parallel on `n_cores` workers (one worker runs them serially in the calling process), and returns an `OptimizationResult`. `sampling=SamplingType.START` does not sample: every run starts from the start values of the parameters, which is what a fit needs to improve a solution, e.g. the nominal values of a model or a trained network; the runs are identical then, so `size=1` is enough for the local optimizer. The progress of the runs is shown on the console, with the runs which are done, the elapsed time and an estimate of the total runtime, e.g. `~ 0:12:30 total`; the estimate is the time per batch of `n_cores` runs times the number of batches (a batch is one run when `n_cores=1`), so it is there as soon as the first run is done and settles as more runs come back:

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

A parallel fit (`n_cores` of two or more) starts worker processes which import the script again, so the fit must run behind a guard, otherwise the workers fit again and the fit does not end:

```py
def main() -> None:
    run_optimization(problem=op, settings=settings, size=10, n_cores=4)


if __name__ == "__main__":
    main()
```

Every repeat is a task of the pool, which hands the next repeat to the worker which is free, so repeats of different duration do not leave workers idle, and every worker resolves the data of the problem once. The start points are sampled by the runner and not by the workers, so a fit with a seed gives the same start points for any number of workers. A fit with one worker, `n_cores=1` which is the default, runs the repeats in the process of the caller without starting a pool and gives the result of `serial=True` for the same seed; `serial=True` does so whatever `n_cores` says, which is what a debugger needs.

A fit keeps what it has. A single optimization which fails, with an error of the integrator or any other error of the objective, is a result which carries its message and the other repeats are unaffected; `timeout` gives every repeat a budget in seconds and a repeat which runs out of it keeps the best parameters it reached. In a parallel fit a worker which dies loses the one repeat it was running. With `runs_dir` every repeat is written as JSON the moment it finishes, so a fit which is interrupted or crashes leaves the repeats which are done and `OptimizationResult.from_directory` reads them back:

```py
from pathlib import Path

from sbmlsim.fit.result import OptimizationResult

opt_result = run_optimization(
    problem=op,
    settings=settings,
    size=10,
    n_cores=4,
    timeout=600,
    runs_dir=Path("results") / "runs",
)
# the same result, from the files alone
recovered = OptimizationResult.from_directory(Path("results") / "runs")
```

`OptimizationAlgorithmType.LEAST_SQUARE` is the local least squares optimizer, `DIFFERENTIAL_EVOLUTION` the global one. The `OptimizationResult` holds the fits of all start points with their costs, the optimal parameters `xopt`, the cost trajectories of the optimizer and the settings the fit was run with; it is stored as JSON and TSV with `to_json` and `to_tsv`, and results of several runs are combined with `OptimizationResult.combine`.

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

The report writes `index.html`, `report.txt`, the `parameters.json` it was made from, the metrics as TSV and the figures. `show_report=True` opens the HTML in a browser.

Two figures are drawn per fit mapping, the data with the simulation and the residuals, and they are almost the whole cost of a report: for a problem with 35 mappings they are 70 of the 76 figures and 88% of the time. `mapping_figures=False` leaves them out, which is what a report read for its tables and its overview figures wants; the cards of the mappings then carry their metrics alone. The command line tools take `--no-mapping-figures` for the same reason.

`index.html` is an interactive page with three sections: **Overview** repeats what the console reports, i.e., the fit, the parameters with their bounds and units, the settings and the data per experiment and kind; **Results** has the metrics per parameter set and kind, the plots of the optimization runs, the goodness of fit and the Bland-Altman plot with a panel per kind of fit mapping, and the contribution of every fit mapping to the cost; **Fit mappings** is one card per mapping with its figures and its metrics. A search box filters the mappings and the tables, the chips filter by training, validation and outlier data, which are the kinds a fit evaluates and therefore the kinds a report shows, the tables sort by any column and a figure opens full size when it is clicked. The page carries its own style and script, so it works from a file and can be archived or sent as it is.

Every parameter set becomes a column of the parameter table and a curve in the plots, so several sets are compared in a single report, e.g., two fits against each other. The first set is the reference the others are compared against.

A problem with neural networks (see [PEtab SciML](petab_sciml.md#the-report-of-a-hybrid-fit)) shows every network in the overview of the report with its pattern, its layers and its targets, and its arrays as one row each with the number of elements, the estimated ones, their bounds and the range and the norm of their values; the parameters of the model keep their rows and `parameters.json` keeps every element.

Two figures show the data points of the fit rather than the curves, each with one panel per kind of fit mapping, i.e. the training data, the validation data and the outliers. The points are colored by study, i.e. by the simulation experiment a fit mapping belongs to, and a study keeps its color in both figures:

- **goodness of fit** (`goodness_of_fit`) plots the prediction against the measurement on logarithmic axes, with the identity line `prediction = measurement`. The points scatter around it when the model describes the data, and a systematic deviation from it is a systematic error of the model.
- **Bland-Altman** (`bland_altman`) plots the agreement of the two as a ratio, `log10(f(x)/y)` over the geometric mean of prediction and measurement, with the bias and the limits of agreement `bias ± 1.96 SD` written as fold factors. The data of a fit spans orders of magnitude, so the agreement is multiplicative: a bias of `1.02x` is a fit which is unbiased and limits of `0.51-2.04x` say that a prediction is within a factor of two of the measurement. A trend over the mean is a model which describes the large or the small values better. A data point which is zero or negative has no ratio and is left out, which is why a panel can show fewer points than the metrics of its kind count.

  `FitReport.agreement` calculates the bias and the limits on the **training data** alone and both figures draw the same lines in every panel, so the validation data and the outliers are read against what the fit agrees to. Per panel they would give every subset its own reference and the panels could not be compared, and over all data points the outliers, which are far away by definition, would widen them. A panel whose points stay inside the limits still shows them.

  The band is the same in the two figures and drawn in the same styles, so they are read the same way: a solid line for `prediction = measurement`, a dash-dotted line for the bias, dashed lines for the limits and the area between them filled. A ratio is a horizontal line in the Bland-Altman plot and a line parallel to the diagonal in the goodness of fit, which on logarithmic axes is the same thing, so a point inside the band is a prediction the fit agrees to in either figure.

The panels next to each other are what the kinds are for: the training panel is how well the fit describes the data it was fitted on, the validation panel how it describes data it was not, and the outlier panel where the curves a fit dropped sit relative to the model. There is no panel over all data points, which would pool the data a fit was fitted on with the data it dropped. Several parameter sets in one report are told apart by their marker, since the color says which study a point comes from.

`FitReport.from_optimization_result` is the shortcut for the report of a fit. It reads the settings from the result and adds the plots which describe the runs rather than a parameter set, i.e., the optimization traces and the waterfall plot. The report shows the fitted parameters alone; `with_model=True` reports the values the model started from as the reference set as well, so that the figures and the tables show what the fit changed:

```py
report = FitReport.from_optimization_result(problem=op, opt_result=opt_result)
report.create(output_dir=Path("results"), name="hctz_iv")

# the fit against the model it started from
report = FitReport.from_optimization_result(
    problem=op, opt_result=opt_result, with_model=True
)
```

## Metrics

`FitMetrics` calculates the metrics of a parameter set on an initialized problem, i.e., from the data of the fit mappings and the predictions of the model. The column names follow the convention of population pharmacokinetics: `DV` is the measured value, `PRED` the prediction of the population parameters, `IPRED` the prediction of the individual parameters, `RES` and `IRES` the residuals `DV - PRED` and `DV - IPRED`, `NRES` the residual normalized by the mean of its curve, `IRES / mean(DV)`, and `IWRES` the residual of the cost, i.e., the residual of the residual type of the settings, weighted and with the loss function applied. A deterministic fit has a single parameter set, so `PRED` is `IPRED` unless a `population_parameter_set` is given:

```py
from sbmlsim.fit import FitMetrics

metrics = FitMetrics(problem=op, parameter_set=parameter_sets[0])
print(metrics.datapoints_df().head())
print(metrics.mappings_df())
print(metrics.summary())
```

`summary()` gives the metrics over all data points: the number of data points `n`, the number of fitted parameters `k`, the `cost`, `MSE`, `RMSE`, `NRMSE`, `RMSE_w`, `R2`, `AIC` and `BIC`; `mappings_df()` gives them per fit mapping. `MSE`, `RMSE`, `R2`, `AIC` and `BIC` are unweighted metrics of the data and the predictions, so they are dominated by the mappings with the largest values: the data of a fit spans orders of magnitude, and a curve of small values which is missed by a factor of 20 still has a tiny absolute error. `NRMSE` is the root mean square of `NRES`, so every curve counts the same whatever its magnitude, and `RMSE_w` is the root mean square of `IWRES`, i.e., what the fit minimizes; the cost of the training data is `0.5 * n * RMSE_w²`. A parameter set can therefore have a larger RMSE and smaller weighted residuals than another one, which is what the weighting is for. The boxes of the goodness of fit show `R²`, `NRMSE` and `RMSE_w` per kind, not the absolute RMSE.

A fit is evaluated on the data it was fitted on and on the data it was not, so `summary(kind=...)` restricts the metrics to a kind of mapping and `summary_df()` has one row per kind: the training data, the validation data and the outliers. The outliers are in there because a curve which a fit drops is a claim about the data which the metrics make checkable: an outlier with an R² as good as the training data is a curve which was dropped without reason. There is no row over all data points, which would pool the data a fit was fitted on with the data it dropped; `summary()` without a kind gives that number where it is wanted. The `cost` is the objective of the optimization, which is defined on the training data alone, so it is only reported there:

```py
metrics.summary_df()
```

Both information criteria are calculated for a least squares fit with normally distributed residuals, i.e. `n * ln(MSE)` plus a penalty per parameter, up to the same additive constant: only differences between models fitted on the same data are meaningful. They differ in the penalty, `2 * k` for the AIC and `k * ln(n)` for the BIC, so the BIC charges a parameter more as soon as there are more than seven data points and increasingly so with more of them. A model which the AIC prefers and the BIC does not is a model whose extra parameter buys a little fit on a lot of data.

The functions of `sbmlsim.fit.metrics` are used on their own as well: `sse`, `mse`, `rmse`, `aic`, `bic` and `r_squared` take arrays of residuals or of data and predictions. R² is not the square of a correlation for a non-linear model and is negative when a prediction is worse than the mean of the data.

A report calculates the metrics for every one of its parameter sets and writes them as `metrics.tsv`, `metrics_mappings.tsv` and `datapoints.tsv`, so several sets are compared by their AIC, RMSE and R².

## Identifiability

Whether the data determines the fitted parameters is answered by the profile likelihood and the Fisher information, see [Identifiability](identifiability.md).

## Running a fit from the command line

Creating the optimization problems, running the optimizations and reporting them is the same for every model, so it lives in `sbmlsim.fit.cli` and a model only defines its fits. A `FitDefinition` is what enters a fit: the fit mapping collections, the parameters which are adjusted, where the experiments and their data are, and the settings.

```python
from sbmlsim.fit.cli import FitDefinition

definition = FitDefinition(
    mapping_collections=f_collections_pk,  # called when the fit runs
    parameters=fit_parameters,
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
    settings=settings,
)
print(definition.problem(opid="hctz_pk"))
```

`run_fit` builds the problems for a strategy and runs them: `OptimizationStrategy.ALL` fits all experiments together, i.e., one parameter set describes every experiment, `SINGLE` fits every experiment on its own, which gives the individual parameters of the metrics. It returns a `FitRun` per optimization, which carries the problem and the result and creates the report.

`fit_cli` and `report_cli` are the command line tools around this. A model passes its definitions by name and gets the whole tool:

```python
from sbmlsim.fit.cli import fit_cli

FIT_DEFINITIONS = {"PK": definition}


def main() -> None:
    fit_cli(FIT_DEFINITIONS)


print(FIT_DEFINITIONS)
```

Every fit gets an id when it starts, `<problem>_<date>_<time>__<hash>`, e.g. `PK_20260908_144538__ea1ff`. It is the id of the optimization problem, of its result and of the directory of its report, so everything a fit produces carries the same key and sorts by time. The output of a fit is a sequence of sections, each with its own icon: the fit with its strategy, algorithm and paths, the parameters which are optimized with their bounds and units, the settings, the data with the number of fit mappings per experiment and kind, the optimization with its progress, and the report. `sbmlsim.fit.display` renders them and is used on its own as well.

The fit problems of the HCTZ model are in `examples/hctz_fitting/fitting/`: `mapping_collections.py` builds the subsets of the data, `parameters.py` holds the fit parameters and `fitting.py` is the definitions plus the four lines above:

```bash
python -m examples.hctz_fitting.fitting.fitting --subset=PK --runs=10 --cores=4 \
    --seed=1234 --method=LSQ --strategy=ALL --name=PK_LSQ_ALL
```

`report.py` is `report_cli` on the same definitions. It creates a report from the `parameters.json` of a finished fit without optimizing again, and compares the parameters of several fits:

```bash
python -m examples.hctz_fitting.fitting.run_report results/fit/PK_LSQ_ALL/parameters.json
python -m examples.hctz_fitting.fitting.run_report run1/parameters.json run2/parameters.json
```

## PEtab

A fit is exchanged with the tools of the PEtab ecosystem as a PEtab v2 problem, which `sbmlsim.fit.petab_v2` writes and reads, see [PEtab](petab.md); a fit of a model with neural networks is a problem of PEtab SciML, see [PEtab SciML](petab_sciml.md). `sbmlsim.fit.petab_omex` packages a problem as a COMBINE archive.
