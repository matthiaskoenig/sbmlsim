# PEtab

[PEtab](https://petab.readthedocs.io) is a table based format for parameter estimation problems of systems biology models. `sbmlsim.fit.petab_v2` writes a fit of `sbmlsim` as a PEtab v2 problem and reads a PEtab v2 problem into a fit. With it you can hand a fit to other tools of the PEtab ecosystem, and fit problems from others, such as those of the [benchmark collection](petab_benchmark.md), with `sbmlsim`.

Problems that combine a model with neural networks use the extension PEtab SciML, see [PEtab SciML](petab_sciml.md). The [PEtab Test Suite](petab_testsuite.md) checks that the reader follows the standard.

## Writing a fit as PEtab

`to_petab` writes the YAML file of the problem, its tables and its models into a directory and returns the path of the YAML file:

```py
from pathlib import Path

from sbmlsim.fit.petab_v2 import to_petab

yaml_file = to_petab(problem, Path("results") / "petab", settings=settings)
```

`problem` is an `OptimizationProblem` and `settings` its `FitSettings`, as in [Parameter fitting](fitting.md). By default the parameters are written with their start values. To write the result of a fit instead, pass `parameter_set=` with a `ParameterSet`. A problem read back from the files then starts from the fitted values. The problem you export is not changed.

This is how the parts of a fit end up in PEtab:

| `sbmlsim` | PEtab v2 |
| --- | --- |
| the model of a task | a model in `model_files`, each SBML file once |
| a `FitMappingCollection` whose mappings share a simulation | one experiment, with the id of the collection |
| a collection over several simulations, e.g. the doses of a study | one experiment per simulation, numbered after the collection |
| the pre-initialization changes of a `Simulation` | the condition of the first period of the experiment |
| a time of a `Change` | a period of the experiment, with the changes as its condition |
| a `SteadyState` | the period at `time=-inf` |
| a fit mapping | an observable; a selection like `[S1]` becomes math of the model |
| the reference data of a mapping | the measurements of its observable; their standard deviation is the noise parameter |
| a `FitParameter` | an estimated parameter, with its bounds and start value |

The training data, the validation data and the outliers are all written, and the kind of each mapping goes into the `sbmlsim` extension. Data marked `excluded` is left out.

Selections need translating, because PEtab has none. In `sbmlsim`, `S1` is the amount of a species and `[S1]` its concentration. In PEtab math, the identifier of a species means its amount if `hasOnlySubstanceUnits=true` and its concentration otherwise. The exporter converts between the two, e.g. the concentration of an amount based species becomes `S1 / compartment`, and a formula value like `[S] + 5` is written as `S + 5`. A condition can only assign an identifier, so a change that sets the concentration of an amount based species (or the amount of a concentration based one) cannot be written and the export raises.

A parameter that is estimated separately for parts of the data (see [Parameter fitting](fitting.md)) is written as a condition that assigns the model entity the value of the estimated parameter. The python selector that chose the parts is not written; the problem stores which fit mappings it chose, and reading it back selects the same mappings by id.

## Reading a PEtab problem

`from_petab` reads a problem and returns an `OptimizationProblem` and the `FitSettings` to run it with:

```py
from pathlib import Path

from sbmlsim.fit.petab_v2 import from_petab
from sbmlsim.fit.runner import run_optimization

problem, settings = from_petab(Path("results") / "petab" / "problem.yaml")
opt_result = run_optimization(problem=problem, settings=settings, size=5, n_cores=4)
```

The reader builds a `SimulationExperiment` from the tables: the models of the problem are its models, each experiment is a timecourse simulation, and each observable is a fit mapping with its measurements as data. You get one `FitMappingCollection` per experiment. A fit mapping is named after its observable, or `<observable>_<experiment>` when the observable is measured in several experiments. `PetabReader` (with `PetabReader.from_yaml`) gives access to all parts.

Some things to know about reading:

- A problem with the `sbmlsim` extension gets the settings it was written with. A problem without it gets the default `FitSettings`, except that a measurement at the time of a change is compared with the state after the change, as PEtab defines it (`data_at_change=DataAtChangeType.AFTER`, see [Data at the time of a change](fitting.md#data-at-the-time-of-a-change)).
- An observable that is the identifier of a model entity becomes the selection of that entity, with its unit. Any other observable is read as an `ObservableModel` (`sbmlsim.fit.objects`): the formula over roadrunner selections, evaluated on the simulation at the measurements. `log` is the natural logarithm and a concentration based species is its concentration `[S]`. The `observableParameters` of a measurement fill the placeholders. The model itself is not changed.
- Models the reader generates, e.g. a model with neural networks, are written next to the source model. Pass `derived_dir=` to `from_petab` or `PetabReader.from_yaml` to write them to another directory, e.g. when the source directory is read-only.
- The experiment class the reader builds can be pickled, so a problem you read runs parallel fits like any other.
- A fit written and read back comes back as one collection per simulation rather than the collections it was defined with. It is the same fit of the same data.
- When the `sbmlsim` extension carries a simulation, the reader keeps its units and output grid and compares it with the periods and conditions of the tables. If they disagree, it warns, names the experiment, the conditions and what differs, and uses the extension. After editing the tables by hand, update the extension too, or remove the `simulation` of that experiment from it.

## What PEtab does not express

An `sbmlsim` fit carries more than the PEtab tables: the units of everything, what the fit does with each part of the data, how residuals are weighted. Whatever the tables cannot hold goes into the `sbmlsim` extension, a block in the YAML file that other tools ignore. That is why a problem written and read again by `sbmlsim` is the fit it started from.

The extension is marked `required`, because its settings define the objective `sbmlsim` optimizes. A tool that does not know `sbmlsim` therefore rejects the problem instead of quietly fitting the same data with another objective. If you want other tools to fit the problem, write it with `to_petab(..., required_extension=False)`. They then optimize the objective PEtab defines, which is a different fit of the same data.

To see what a problem loses before you write it, list its gaps:

```py
from sbmlsim.console import console
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table

problem.initialize(settings)
console.print(gaps_table(gaps_of_problem(problem)))
```

The full catalogue is `sbmlsim.fit.petab_v2.gaps`. Each gap has one of three kinds:

- **extension**: PEtab has no place for it and the extension carries it, e.g. the units, the fit settings, the kind of every mapping, the output grid, the metadata of a curve and the integrator settings. This includes the scale the optimizer searches in: PEtab v2 removed the `parameterScale` column, so bounds and start values are written on the linear scale. Another example is `measurement-at-change`: by default `sbmlsim` compares a measurement at the time of a change with the state before the change, while other tools compare it with the state after. The export logs these measurements, and `gaps.measurements_at_changes` lists them per mapping.
- **lossy**: the information is transformed. The noise model and the priors of PEtab are only evaluated by `sbmlsim`, not fitted. An estimated noise parameter does not change the weighted least squares. The weights of `sbmlsim` are not the standard deviation PEtab uses. The reader builds one simulation experiment for the whole problem, so a task selects the observables of the whole problem (gap `selections`), which changes the cost of the HCTZ example by `7e-6` after a round trip.
- **unsupported**: the export raises and names the experiments and mappings. This covers an observable that is a python function, a mapping whose x is not the time, a mapping that compares values per simulation or over a scan dimension (`scalar-observable`), and a task that is a scan (`scan-task`).

## The log-likelihood

PEtab defines the objective as the likelihood of the measurements under a noise model, while `sbmlsim` fits by weighted least squares. `sbmlsim.fit.petab_v2` calculates the log-likelihood and related values so you can evaluate a problem and compare a parameter set with the results of other tools. The optimizer does not use them.

```py
from dataclasses import replace
from pathlib import Path

from sbmlsim.fit.petab_v2 import (
    chi2,
    from_petab,
    gradient,
    log_likelihood,
    log_prior,
    unnorm_log_posterior,
)

problem, settings = from_petab(Path("results") / "petab" / "problem.yaml")
problem.initialize(settings)

llh = log_likelihood(problem)  # at the nominal values
llh_fit = log_likelihood(problem, parameters=opt_result.parameter_set())
chi2_fit = chi2(problem, parameters=opt_result.parameter_set())
priors = log_prior(problem)  # by parameter id
posterior = unnorm_log_posterior(problem)

# a gradient needs a fixed grid and tight tolerances
problem.initialize(
    replace(
        settings,
        variable_step_size=False,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
)
grad = gradient(problem)
```

`log_likelihood` simulates the problem and sums the log density of every measurement of the training data. Validation data and outliers do not enter, and neither do the residual type, weights and loss function of the `FitSettings`. Without `parameters` it uses the nominal values, i.e. the start values of the parameters. For a problem that was read these are the `nominalValue` of the parameter table, except that a nominal value outside the bounds is clipped into them, with a warning. The reader's `nominal_parameters(problem)` gives the unclipped values, which is where PEtab evaluates a problem.

The simulation `y` is the median of the distribution of a measurement `m`, and the noise formula gives its scale `σ`, as PEtab v2 defines them:

| `noiseDistribution` | log density of a measurement |
| --- | --- |
| `normal` | `-0.5 log(2π σ²) - 0.5 ((m - y) / σ)²` |
| `log-normal` | `-0.5 log(2π σ² m²) - 0.5 ((log m - log y) / σ)²` |
| `laplace` | `-log(2σ) - abs(m - y) / σ` |
| `log-laplace` | `-log(2σ m) - abs(log m - log y) / σ` |

The logarithmic distributions need positive measurements and simulations. A problem whose residuals are relative to the baseline of a curve (`ABSOLUTE_TO_BASELINE`, `NORMALIZED_TO_BASELINE`) has no log-likelihood, and `log_likelihood` raises.

The reader keeps the noise formula and distribution of every observable as the `NoiseModel` of its fit mapping (`problem.noise_models` after `initialize`), and the export writes them back. The symbols of a noise formula take these values:

| symbol | value |
| --- | --- |
| a placeholder of `noisePlaceholders` | the `noiseParameters` of the measurement, a number or a formula |
| the id of the observable | the simulation at the measurement |
| a model entity, e.g. a parameter of the parameter table or a species | its value in the simulation at the measurement |
| a noise parameter the model does not have | the value of the parameter set if it has one, the nominal value otherwise |

A fit mapping without a noise model, which is every mapping of a problem defined in python, has normal noise whose standard deviation is the error of its data: the standard deviations, or the standard errors when there are none. A point whose error is zero or missing gets the largest error of its curve, and data without any errors has the scale `1.0`. The export writes these errors, so a problem and its PEtab version have the same log-likelihood. To give a mapping another noise model, pass a `NoiseModel` as the `noise` argument of `FitMapping`.

The other functions:

- `chi2` is the sum of the squared residuals in units of the noise scale, `(m - y) / σ`, or `(log m - log y) / σ` for the logarithmic distributions.
- `log_prior` gives the log density of the prior of every parameter of the fit: the `Prior` of the `FitParameter`, truncated at its bounds, or a uniform prior over the bounds when there is none. The reader keeps `priorDistribution` and `priorParameters` as the prior, and the export writes them.
- `unnorm_log_posterior` is the log-likelihood plus the log priors.
- `gradient` is the central finite difference of the log-likelihood on the linear scale, with the step `step * max(|x|, 1)`, as a `pandas.Series` indexed by parameter id. It uses three points by default and five with `order=4`, whose error falls faster with the step. It never simulates outside the bounds: next to a bound the difference is one sided, and a warning names the parameters where it had to fall back to fewer points. Initialize the problem with `variable_step_size=False` first, as above: with a variable step size the simulations differ by about `1e-6` between runs and the gradient is noise, which is logged as a warning.

## Extensions of other tools

A problem can carry extensions of any tool in the `extensions` block of its YAML, each marked as `required` or not. The reader knows the `sbmlsim` extension and, with the `sciml` extra installed, the `sciml` extension of [PEtab SciML](petab_sciml.md). An extension that is not required is ignored, with a message in the log. For a problem that requires an unknown extension, `from_petab` raises a `ValueError` naming it. A problem that requires `sciml` without the extra raises an `ImportError` that tells you to `pip install sbmlsim[sciml]`.

## The example

`examples/hctz_fitting/fitting/petab_problem.py` runs all of this on the HCTZ fit. It reports the gaps of the fit, writes it as PEtab, validates the files with `petab`, shows which collection every experiment came from, reads the problem back and compares the cost and the log-likelihood of the two:

```bash
python -m examples.hctz_fitting.fitting.petab_problem
python -m examples.hctz_fitting.fitting.petab_problem --subset=PK --portable
```

`--portable` writes the extension as not required, i.e. a problem for other tools. The problem is written into `results/petab/<subset>`.

## COMBINE archives

`sbmlsim.fit.petab_omex.create_petab_omex(omex_file, yaml_file)` packages a PEtab problem as a COMBINE archive, with the YAML file as its master entry.
