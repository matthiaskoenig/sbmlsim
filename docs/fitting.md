# Parameter fitting

Parameter fitting adjusts model parameters so that the simulations of experiments match the experimental data. In `sbmlsim` a fit is an `OptimizationProblem` built from `FitMappingCollection` objects, which name the simulation experiments and their fit mappings, and `FitParameter` objects with the bounds of the parameters. The problem is run with local or global optimizers of scipy, reported with `FitReport`, and the identifiability of the fitted parameters is analysed with the profile likelihood.

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

What a fit does with a curve is decided when the data of the fit is selected, not on the fit mapping: the same curve is training data of one fit and validation data of another. Every `FitMappingCollection` therefore carries a `MappingKind` for the mappings it selects: `TRAINING` enters the cost, `VALIDATION` is evaluated but not fitted, and `OUTLIER` and `EXCLUDED` are not used at all. The two say different things: an outlier is a decision about the data, i.e. it is not usable, and an exclusion is a decision about the model, i.e. the model does not describe what was measured, e.g. an arm of a study with a coadministration the model has no interaction for.

- `MappingKind.TRAINING` (the default): the mappings are fitted, i.e., their residuals enter the cost of the optimization,
- `MappingKind.VALIDATION`: the mappings are not fitted. They are simulated and evaluated together with the training data when the fit is reported, which shows how the fitted parameters describe data they were not fitted on,
- `MappingKind.OUTLIER`: the mappings are not used at all. An optimization problem skips its outliers, they stay in the overview of the data so that it is visible which curves were dropped.

`mapping_collections_by_kind` selects and classifies the data in one step: every kind gets its own filters and the mappings of all kinds are listed in a single overview.

```python
from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.studies import Beermann1976
from sbmlsim.fit import MappingKind
from sbmlsim.fit.helpers import (
    filter_empty,
    filter_keys,
    filter_not_keys,
    mapping_collections_by_kind,
)

validation = {"fm_hctz_iv35_4_urine"}
mapping_collections_kinds = mapping_collections_by_kind(
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
from examples.hctz_fitting.fitting.mapping_collections import f_collections_pkiv

mapping_collections = f_collections_pkiv()
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

`FitSettings.parameter_scale` is the space the optimizer searches the parameters in: `LOG10` by default, because a rate constant spans orders of magnitude and an optimizer on the linear scale spends its steps on the largest parameters, and `LOG` or `LINEAR` if a problem wants them. The bounds, the start values and the fitted parameters are always on the linear scale, i.e. in the units of the model, only the search happens in the scaled space. A logarithm needs finite positive bounds and a positive start value, the linear scale only needs finite bounds.

The scale is a property of the optimization and not of the model or of the data, which is why it is part of the settings; PEtab v2 removed the `parameterScale` of its parameter table for the same reason.

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

A parallel fit starts worker processes which import the script again, so the fit must run behind a guard, otherwise the workers fit again and the fit does not end:

```py
def main() -> None:
    run_optimization(problem=op, settings=settings, size=10, n_cores=4)


if __name__ == "__main__":
    main()
```

Every repeat is a task of the pool, which hands the next repeat to the worker which is free, so repeats of different duration do not leave workers idle, and every worker resolves the data of the problem once. The start points are sampled by the runner and not by the workers, so a fit with a seed gives the same start points for any number of workers. `serial=True` runs the repeats in the process of the caller, which is what a debugger needs.

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

`index.html` is an interactive page with three sections: **Overview** repeats what the console reports, i.e., the fit, the parameters with their bounds and units, the settings and the data per experiment and kind; **Results** has the metrics per parameter set and kind, the plots of the optimization runs and of the predictions, and the contribution of every fit mapping to the cost; **Fit mappings** is one card per mapping with its figures and its metrics. A search box filters the mappings and the tables, the chips filter by training, validation and outlier data, the tables sort by any column and a figure opens full size when it is clicked. The page carries its own style and script, so it works from a file and can be archived or sent as it is.

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

`summary()` gives the metrics over all data points: the number of data points `n`, the number of fitted parameters `k`, the `cost`, `MSE`, `RMSE`, `RMSE_w`, `R2`, `AIC` and `BIC`; `mappings_df()` gives them per fit mapping. `MSE`, `RMSE`, `R2`, `AIC` and `BIC` are unweighted metrics of the data and the predictions, so they are dominated by the mappings with the largest values, while `RMSE_w` uses the weighting of the settings. A parameter set can therefore have a larger RMSE and smaller weighted residuals than another one, which is what the weighting is for.

A fit is evaluated on the data it was fitted on and on the data it was not, so `summary(kind=...)` restricts the metrics to a kind of mapping and `summary_df()` has a row for the training data, a row for the validation data and a row over all of them. The `cost` is the objective of the optimization, which is defined on the training data alone, so it is only reported there:

```py
metrics.summary_df()
```

Both information criteria are calculated for a least squares fit with normally distributed residuals, i.e. `n * ln(MSE)` plus a penalty per parameter, up to the same additive constant: only differences between models fitted on the same data are meaningful. They differ in the penalty, `2 * k` for the AIC and `k * ln(n)` for the BIC, so the BIC charges a parameter more as soon as there are more than seven data points and increasingly so with more of them. A model which the AIC prefers and the BIC does not is a model whose extra parameter buys a little fit on a lot of data.

The functions of `sbmlsim.fit.metrics` are used on their own as well: `sse`, `mse`, `rmse`, `aic`, `bic` and `r_squared` take arrays of residuals or of data and predictions. R² is not the square of a correlation for a non-linear model and is negative when a prediction is worse than the mean of the data.

A report calculates the metrics for every one of its parameter sets and writes them as `metrics.tsv`, `metrics_mappings.tsv` and `datapoints.tsv`, so several sets are compared by their AIC, RMSE and R².

## Identifiability

A fit always returns numbers. Whether the data determines those numbers is a separate question, and it is the one that decides whether a parameter may be interpreted, compared between conditions or extrapolated from. A parameter which the data does not determine takes whatever value the optimizer happened to stop at.

### What identifiability means

A parameter is **identifiable** when the data could not have been produced by a different value of it. Two things can go wrong, and they are usually distinguished (Raue et al. 2014, Wieland et al. 2021):

- **Structural non-identifiability** is a property of the model and its observables, not of the data. The parameter enters the observables only in combination with others, so no amount of data of that kind determines it: only a product, a ratio or a sum is determined. A model with a concentration `k * S` observed alone can never separate `k` from `S`. This is decided on the equations, with differential algebra or Lie derivatives (Bellman & Åström 1970, Chiş et al. 2011, Villaverde et al. 2016); `sbmlsim` does not analyse it, but a structural non-identifiability shows up in the analyses below as a perfectly flat direction.
- **Practical non-identifiability** is a property of the model *and the data at hand*. The parameter is determined in principle, but the data is too sparse, too noisy or measured in the wrong place to pin it down, so its confidence interval extends to infinity in one or both directions. More or better data fixes this; a different model is needed for the structural case.

Between the two extremes, most models of systems biology are **sloppy**: a few combinations of parameters are determined well and many are determined poorly, with the sensitivity spread over orders of magnitude (Gutenkunst et al. 2007, Transtrum et al. 2015). A sloppy model still predicts well along the directions the data constrains, which is why a poorly determined parameter is not by itself a reason to distrust a model — it is a reason not to interpret that parameter.

### The two analyses

`sbmlsim` implements the two analyses which work on a fitted model and its data:

| | [Profile likelihood](#profile-likelihood) | [Fisher information](#fisher-information) |
| --- | --- | --- |
| what it looks at | the cost along the parameter, re-optimizing the others | the curvature of the cost at the optimum |
| cost | a re-optimization per point, hundreds of simulations | one jacobian, `2k` simulations |
| intervals | asymmetric, invariant under a transformation of the parameters | symmetric in the scaled space, exact only for a quadratic cost |
| finds | structural and practical non-identifiability | structural non-identifiability, sloppiness, correlations |

The Fisher information is the cheap local answer and the profile likelihood the expensive exact one. They agree where the cost is a quadratic around the optimum and disagree where it is not, which is the normal case for a non-linear model: the profile is then the one to trust (Raue et al. 2009, Wieland et al. 2021).

Neither is a substitute for the other question worth asking, which is whether the *prediction* is determined even when the parameters are not (Simpson & Maclaren 2023).

### What to do about it

A parameter which comes back non-identifiable leaves four options: measure something else, fix the parameter to a literature value, reduce the model so that the parameter and the ones it is coupled to become one, or report it as undetermined and refrain from interpreting it. The paths of the other parameters along a profile say which parameters are coupled to it and therefore which reduction is the right one (Maiwald et al. 2016).

### Profile likelihood

A fit gives the parameters which describe the data best, the profile likelihood says how well the data determines every one of them. `sbmlsim.fit.identifiability` implements the method of Raue et al. 2009: a parameter is fixed at values around the optimum, all other parameters are optimized again for every value, and the resulting profile, the best cost as a function of the parameter, is compared with a threshold. The cost of a fit is the cost of `scipy.optimize.least_squares`, `cost = 0.5 * Σ r²` with the weighted residuals `r`. With residuals which are standardized by the errors of the data, `2 * cost` is the negative log-likelihood up to a constant, and the threshold of the likelihood ratio test on the cost is

```
cost_threshold = cost_min + chi2.ppf(alpha, df) / 2
```

with the confidence level `alpha` and `df = 1` for the pointwise confidence intervals of single parameters, i.e., `1.92` above the minimal cost at 95%; `df` equal to the number of parameters gives simultaneous intervals. The values at which the profile crosses the threshold are the bounds of the confidence interval of the parameter. These intervals are invariant under a transformation of the parameters and may be asymmetric, which is where the intervals of the [Fisher information](#fisher-information) fail for non-linear models (Wieland et al. 2021). With another weighting of the residuals the threshold is a heuristic on the same scale.

The shape of the profile classifies the parameter (`Identifiability`):

- `IDENTIFIABLE`: the profile crosses the threshold on both sides of the optimum, the confidence interval is finite,
- `NON_IDENTIFIABLE_LOWER`, `NON_IDENTIFIABLE_UPPER`, `NON_IDENTIFIABLE`: the profile has a minimum but stays below the threshold up to the lower bound, the upper bound or both bounds of the parameter, i.e., the parameter is practically non-identifiable, the data does not determine it towards small and/or large values,
- `STRUCTURAL`: the profile is flat over the scanned range, the parameter is compensated by the other parameters and the data carries no information about it.

The scans run in the space the fit searches, i.e. the `parameter_scale` of its settings, with adaptive steps: a step which raises the cost by more than `max_cost_fraction` of the distance to the threshold is reduced and repeated, a step which raises it by little is enlarged, so the profile is resolved where it changes. The other parameters start from the previous point of the profile and their paths are stored, so a parameter which is coupled to the scanned one is seen in its path (Maiwald et al. 2016). A scan stops when the profile crosses the threshold, at the bound of the parameter or after `max_points`. A scan which finds a lower cost than the parameter set reports that the fit did not converge, and the threshold is taken relative to the lowest cost of all profiles.

```py
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood

result = profile_likelihood(
    problem=op,
    settings=settings,
    parameter_set=parameter_sets[0],
    profile_settings=ProfileSettings(alpha=0.95, max_points=30),
    n_cores=4,
)
print(result.summary_df())
result.to_json(Path("identifiability.json"))
```

`ProfileSettings` holds the confidence level, the degrees of freedom, the steps in decades of the parameter and `reoptimize`: without the re-optimization the other parameters stay at the optimum, which is a plain scan of the cost and a lower bound of the profile, fast and sufficient to find the non-identifiable parameters, but with intervals which are too narrow for coupled parameters. The scans are independent and run in the worker pool of the fit runner, two per parameter, so a parallel analysis needs the `if __name__ == "__main__":` guard like a parallel fit. `pids` restricts the analysis to some of the parameters.

The `IdentifiabilityResult` carries a `ParameterProfile` per parameter with the values, the costs, the paths of all parameters and the confidence interval, `summary_df()` is the table of the parameters with their intervals and classification, `report()` the text and `to_json`/`from_json` the storage. `plot_profiles` draws the overview of all profiles, `plot_profile` the profile of one parameter with the paths of the other parameters along it.

A report shows the analysis: `FitReport(..., identifiability=result)` adds the section **Identifiability** with the table, the overview and one figure per parameter, and writes `identifiability.json` and `identifiability.tsv`. `FitReport(..., fisher=fim)` adds the Fisher information to the same section, i.e. its table of errors and intervals, the eigenvalues and the correlation of the parameters, and writes `fisher.json` and `fisher.tsv`; a report of an information which is rank deficient says that its errors cannot be read. Every metric and every figure of a report carries a `?` which explains what the value is and what to look for in the figure. `FitRun.identifiability()` computes the profiles of the best parameter set of a finished fit, so a global optimization followed by the identifiability of its result is

```py
runs = run_fit(
    definition,
    algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
    size=2,
    n_cores=4,
)
run = runs[opid]
identifiability = run.identifiability(n_cores=4)
fisher = run.fisher()
run.report(
    output_dir=Path("results"),
    identifiability=identifiability,
    fisher=fisher,
)
```

which is `examples/hctz_fitting/fitting/identifiability.py`. `identifiability_cli` is the command line tool for stored parameters, `examples/hctz_fitting/fitting/identifiability_report.py` on the HCTZ definitions; it computes both analyses, `--no-fisher` reports the profiles alone:

```bash
python -m examples.hctz_fitting.fitting.identifiability --subset=PK --runs=2 --cores=4
python -m examples.hctz_fitting.fitting.identifiability_report results/fit/PK/parameters.json --cores=4
```

### Fisher information

The profile likelihood follows the cost and costs a re-optimization per point. The Fisher information is the local alternative: the curvature of the cost at the optimum, from one jacobian. For a fit which minimizes `cost = 0.5 * Σ r²` with the weighted residuals `r`, the Gauss-Newton approximation of the Hessian is the Fisher information matrix, and its inverse, scaled by the variance of the residuals, is the covariance of the parameters:

```
FIM = J' J,   cov = σ² (J' J)⁻¹,   σ² = 2 cost / (n - k)
```

with the jacobian `J = ∂r/∂θ` in the space the optimizer searches, see `FitSettings.parameter_scale`:

```py
from sbmlsim.fit.fisher import fisher_information

fim = fisher_information(problem=op, settings=settings, parameter_set=parameter_sets[0])
print(fim.summary_df)
print(fim.correlation)
```

`summary_df` is the table of the parameters with their standard error, the coefficient of variation and the confidence interval `θ ± t · SE`, computed in the scaled space and transformed back, so on a logarithmic scale the interval is not symmetric around the value. `correlation` is the correlation of the parameters: a pair at `±1` is a pair the data only determines together, i.e. the fit trades one against the other.

The eigenvalues say what the data constrains. A direction with a small eigenvalue is a combination of parameters the data does not determine, `rank` counts the ones it does, and `is_identifiable` is whether that is all of them. `condition_number`, the ratio of the largest to the smallest eigenvalue, is how sloppy the problem is.

A parameter which no data informs is exactly zero in the jacobian and gives a zero eigenvalue, i.e. the analysis finds a structural non-identifiability without scanning: the intravenous problem of the HCTZ example determines the renal excretion and carries no information about the absorption of an oral dose, so its Fisher information has rank 1 of 3.

The two analyses answer different questions and disagree where the cost is not a quadratic. The Fisher information is a local statement about the optimum, the profile likelihood follows the cost until it rises by the threshold, so the profile is the one to trust for a non-linear model and the Fisher information is the one to compute when a profile is too expensive. A parameter which the Fisher information calls determined and the profile calls non-identifiable has a cost which is curved at the optimum and flat away from it, which is what the profile is for; in the `Perelson_Science1996` example of the benchmark collection the loss rate of the infected cells is such a parameter.

The publications behind the methods are listed under [References](references.md#parameter-fitting).

## Running a fit from the command line

Creating the optimization problems, running the optimizations and reporting them is the same for every model, so it lives in `sbmlsim.fit.cli` and a model only defines its fits. A `FitDefinition` is what enters a fit: the fit mapping collections, the parameters which are adjusted, where the experiments and their data are, and the settings.

```python
from sbmlsim.fit.cli import FitDefinition

definition = FitDefinition(
    mapping_collections=f_collections_pkiv,  # called when the fit runs
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

`sbmlsim.fit.petab_omex` packages a [PEtab](https://petab.readthedocs.io) parameter estimation problem, i.e., the model, the condition, observable, measurement and parameter tables and the PEtab YAML, as a COMBINE archive with `create_petab_omex`, so that the problem is exchanged with other tools. `examples/petab/` shows PEtab problems solved with pypesto and AMICI, which are not dependencies of sbmlsim.
