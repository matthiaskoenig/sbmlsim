# PEtab

[PEtab](https://petab.readthedocs.io) is the table based format for parameter estimation problems of systems biology models. `sbmlsim.fit.petab_v2` writes an optimization problem as a PEtab v2 problem and reads a PEtab v2 problem into an optimization problem, so a fit is handed to the tools of the PEtab ecosystem and a problem of the [benchmark collection](petab_benchmark.md) is fitted with `sbmlsim`. Hybrid problems of a model and neural networks are the extension PEtab SciML, see [PEtab SciML](petab_sciml.md), and the conformance of the reader is checked with the [PEtab Test Suite](petab_testsuite.md).

## Writing a fit as PEtab

`to_petab` writes the YAML of the problem, its tables and the models of the fit into a directory:

```py
from pathlib import Path

from sbmlsim.fit.petab_v2 import to_petab

yaml_file = to_petab(problem, Path("results") / "petab", settings=settings)
```

`to_petab(..., parameter_set=fitted)` writes the values of a `ParameterSet`, e.g. the result of a fit, instead of the start values: they are the `nominalValue` of the parameter table, the start values of the `sbmlsim` block and the values of the arrays of the networks, so the problem which is read again starts from the fit. The problem which is exported is not changed.

The problem is translated as follows:

| `sbmlsim` | PEtab v2 |
| --- | --- |
| the model of a task | a model of `model_files`, every SBML file once |
| a `FitMappingCollection` whose mappings share a simulation | one experiment, with the id of the collection |
| a collection over several simulations, e.g. the doses of a study | one experiment per simulation, numbered after the collection |
| the changes before the initialization of a `Simulation` | the condition of the first period of the experiment |
| a time of a `Change` of the `Simulation` | a period of the experiment, its changes are its condition |
| a `SteadyState` | the period at `time=-inf` |
| a fit mapping | an observable, the selection `[S1]` becomes the math of the model |
| the reference data of a mapping | the measurements of its observable, its error is the noise parameter |
| a `FitParameter` | a parameter which is estimated, with its bounds and its start value |

Everything the fit evaluates is written, i.e. the training data, the validation data and the outliers, so that a round trip keeps the fit; the kind of every mapping is in the extension. A tool which reads the problem without the extension fits every measurement it finds, which is why the extension is required. The data the model does not describe is `excluded` and is not part of the problem at all.

`sbmlsim` names an observable and the target of a change with a selection of roadrunner, where `S1` is the amount of a species and `[S1]` its concentration, and PEtab has no selections: in the math of a model the identifier of a species is its amount if `hasOnlySubstanceUnits=true` and its concentration if it is `false`. `sbmlsim.fit.petab_v2.symbols` converts between the two rather than dropping the brackets, i.e. the concentration of an amount based species is the formula `S1 / compartment`. A condition assigns an identifier and not an expression, so a change which sets the concentration of an amount based species has no PEtab representation and the export raises.

The standard deviation of the data is the noise of a measurement, through the `sd` placeholder the observable declares in `noisePlaceholders`; the `noiseParameter${n}_${observableId}` names of PEtab v1 are gone in v2.

## Reading a PEtab problem

`from_petab` reads a problem as an `OptimizationProblem` and the settings to run it with:

```py
from pathlib import Path

from sbmlsim.fit.petab_v2 import from_petab
from sbmlsim.fit.runner import run_optimization

problem, settings = from_petab(Path("results") / "petab" / "problem.yaml")
opt_result = run_optimization(problem=problem, settings=settings, size=5, n_cores=4)
```

The reader builds a `SimulationExperiment` from the tables, in the same way the SED-ML parser builds one from a document: the models of the problem are its models, its experiments are the timecourse simulations, its observables are the fit mappings and the measurements of an observable are its dataset. `PetabReader` gives access to the parts.

An experiment of PEtab is a simulation with its conditions and the observables which are measured in it, which is what a `FitMappingCollection` is, so the reader gives one collection back per experiment of the problem. A fit mapping is an observable in an experiment: it is named after its observable, and `<observable>_<experiment>` when the observable is measured in several experiments. The math of an observable formula is translated into the math of SBML, e.g. `log` of PEtab is the natural logarithm and `log` of a formula of SBML is the one to the base 10. A fit which is written and read again therefore comes back as one collection per simulation rather than as the collections it was defined with, which is the same fit of the same data.

## What PEtab does not express

PEtab describes a problem as tables and `sbmlsim` describes more than that, i.e. the units of everything, what a fit does with a subset of the data and how the residuals are weighted. What the tables do not hold goes into the `sbmlsim` extension of the problem, a block in its YAML which other tools ignore, so a problem which is written and read again is the fit it started from.

PEtab says that an extension which changes the mathematical interpretation of a problem must be `required`, and that a tool must reject a problem which requires an extension it does not know. The settings the extension carries are the objective `sbmlsim` optimizes, so it is required: a tool which does not know `sbmlsim` says so instead of fitting the same data with another objective without telling anyone.

A problem which is meant to be fitted by other tools is written with `to_petab(..., required_extension=False)`. They then read the tables and optimize the objective PEtab defines, which is a different fit of the same data.

`sbmlsim.fit.petab_v2.gaps` is the catalogue of the differences and `gaps_of_problem` reports the ones a problem runs into, before it is written:

```py
from sbmlsim.console import console
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table

problem.initialize(settings)
console.print(gaps_table(gaps_of_problem(problem)))
```

A gap is of one of three kinds:

- **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The scale the optimizer searches in is part of the settings, which is where PEtab v2 puts it as well: it removed the `parameterScale` of its parameter table because the scale is a property of the optimization and not of the problem, so the bounds and the start values are written on the linear scale. A parameter with a scale of its own (`FitParameter.scale`), e.g. an element of a neural network on the linear scale, is the `sciml-parameter-scale` gap. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
- **lossy**: the information is transformed. The noise model of PEtab is its objective and is only evaluated by `sbmlsim`, the weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
- **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation. The reader raises for a problem which requires the extension of another tool.

The round trip of the HCTZ example keeps the settings, the parameters with their units, the mappings with their kinds and the reference data of every mapping, and its cost agrees to `7e-6`, which is the `selections` gap above.

A parameter which is estimated separately for parts of the data is a condition of PEtab: the condition assigns the entity of the model the value of the estimated parameter, and the experiments of the subset reference it. The selector which chose the subset is a python callable and is not written; PEtab stores the resolution, so a problem which is read back selects the same fit mappings by their id. The fit, its cost and its parameters are the same, i.e. the round trip is exact in effect and not in source form.

## The log-likelihood

PEtab defines the objective of a problem as the likelihood of its measurements under a noise model, and `sbmlsim` fits by weighted least squares. `sbmlsim.fit.petab_v2.likelihood` calculates the log-likelihood of a problem for its evaluation, e.g. to compare a parameter set with the result of another tool. The optimizer does not use it.

```py
from dataclasses import replace
from pathlib import Path

from sbmlsim.fit.petab_v2 import from_petab, gradient, log_likelihood

problem, settings = from_petab(Path("results") / "petab" / "problem.yaml")
problem.initialize(settings)

llh = log_likelihood(problem)
llh_fit = log_likelihood(problem, parameters=opt_result.parameter_set())

# a gradient needs an integrator which is more exact than its step
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

`log_likelihood` simulates the problem at the parameters and sums the log density of every measurement of the training data; the validation data and the outliers do not enter, and neither do the residual, the weights and the loss function of the `FitSettings`. Without parameters it is evaluated at the nominal values, i.e. the start values of the parameters, which are the `nominalValue` of the parameter table of a problem which was read. The simulation `y` is the median of the distribution of the measurement `m` and the noise formula gives its scale `σ`, which are the definitions of PEtab v2:

| `noiseDistribution` | log density of a measurement |
| --- | --- |
| `normal` | `-0.5 log(2π σ²) - 0.5 ((m - y) / σ)²` |
| `log-normal` | `-0.5 log(2π σ² m²) - 0.5 ((log m - log y) / σ)²` |
| `laplace` | `-log(2σ) - abs(m - y) / σ` |
| `log-laplace` | `-log(2σ m) - abs(log m - log y) / σ` |

The reader keeps the noise formula and the noise distribution of every observable as the `NoiseModel` of its fit mapping, `problem.noise_models` holds them after `initialize`, and the export writes them back, so a round trip keeps the noise. The symbols of a noise formula are resolved as follows:

| symbol | value |
| --- | --- |
| a placeholder of `noisePlaceholders` | the `noiseParameters` of the measurement, a number or a formula of parameters |
| the id of the observable | the simulation at the measurement |
| a parameter of the fit | the value of the parameter set |
| another parameter of the parameter table | the value of the parameter set if it has one, the `nominalValue` otherwise |

A parameter of the noise which the problem estimates is therefore evaluated and not estimated, which is the `noise-parameters` gap: `log_likelihood(problem, ParameterSet(sid="fit", values={..., "sd_obs": 0.1}))` gives the log-likelihood another tool reports for its estimate. A noise formula over anything else, e.g. a species of the model, is read with a warning and `log_likelihood` raises for it. A fit mapping which has no noise model, i.e. every mapping of a problem which is defined in python, has a normal noise whose standard deviation is the error of its reference data as the problem resolves it: the standard deviations of the data, the standard errors when the data has none, and for a point whose error is zero or missing the largest error of its curve; data without errors, or whose errors are all zero or missing, has the scale `1.0`. The log-likelihood uses these errors and the export writes them, so a problem and its PEtab problem have the same log-likelihood. A `NoiseModel` is given to a `FitMapping` with its `noise` argument.

The log-likelihood is the one of the measurements, so a problem whose residuals are relative to the baseline of a curve (`ABSOLUTE_TO_BASELINE`, `NORMALIZED_TO_BASELINE`) has none and `log_likelihood` raises. The logarithmic distributions require positive measurements and simulations.

`gradient` is the central finite difference of the log-likelihood on the linear scale, with the step `step * max(|x|, 1)` for every parameter of the fit, and returns a `pandas.Series` indexed by the ids of the parameters. The difference is of three points by default and of five points with `order=4`, whose error grows with the fourth power of the step instead of the square: the log-likelihood of a model which oscillates is strongly curved in the parameters of a neural network, and the reference values of the PEtab SciML test suite are differences of five points. The model is not simulated outside the bounds of a parameter: next to a bound the difference is one sided with the full step (three points, five points with `order=4`), because a step which is shrunk to the distance to the bound divides the error of a simulation by a vanishing step; a parameter outside its bounds is an error. Two fall backs are logged, one warning per call which names the parameters: the difference of three points where the bounds leave no room for the five points of `order=4`, and the secant of the bounds where the bounds are closer than the steps of a difference. A difference divides the error of a simulation by the step. With `variable_step_size=True` the data is interpolated on the steps of the integrator, which differ between two simulations, so the simulations of one problem differ by `1e-6` however tight the tolerances are and the gradient is noise; `gradient` logs a warning in this case.

## Extensions of other tools

A problem carries the extensions of any tool in the `extensions` block of its YAML, and `required` says whether the problem can be interpreted without one of them. The reader knows the `sbmlsim` extension and, with the extra `sciml` installed, the `sciml` extension of [PEtab SciML](petab_sciml.md). A problem which requires another extension is not read: `from_petab` raises a `ValueError` which names the extension, before `petab` reads the files of the problem; a problem which requires `sciml` without the extra raises an `ImportError` which names `pip install sbmlsim[sciml]`. An extension which is not required is ignored and the log says so.

## The example

`examples/hctz_fitting/fitting/petab_problem.py` runs the layer on the reference problem, i.e. it reports the gaps of the fit, writes it, validates the problem with `petab`, lists which collection every experiment came from, reads the fit back and compares the cost and the log-likelihood of the two:

```bash
python -m examples.hctz_fitting.fitting.petab_problem
python -m examples.hctz_fitting.fitting.petab_problem --subset=PK --portable
```

The `PK` problem is several simulation experiments, which is the `selections` gap, so its cost comes back with a relative difference of the order of `1e-5`; a problem of a single experiment comes back exactly.

## COMBINE archives

`sbmlsim.fit.petab_omex.create_petab_omex` packages a PEtab problem as a COMBINE archive, with the YAML as its master entry.
