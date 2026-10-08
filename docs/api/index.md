# API reference

The API reference is generated from the docstrings of the package.

## sbmlsim

The top level modules: data, units and the shared output.

| module | description |
| --- | --- |
| [data](data.md) | `Data` objects referencing simulation results and datasets, the input of plots and calculations |
| [units](units.md) | unit registry of a model and unit conversions with pint |
| [serialization](serialization.md) | JSON serialization of experiments |
| [utils](utils.md) | timing and other helpers |
| [console](console.md) | shared rich console |
| [display](display.md) | the output of scripts: sections, key/value blocks and links which the terminal opens with a click |
| [log](log.md) | logging of the package |

## sbmlsim.model

Models and model changes, see [Models](../models.md).

| module | description |
| --- | --- |
| [model.model](model.model.md) | `AbstractModel`, the model of a simulation experiment with its changes and selections |
| [model.model_roadrunner](model.model_roadrunner.md) | `RoadrunnerSBMLModel`, the roadrunner instance of an SBML model with its units and parameter changes |
| [model.model_change](model.model_change.md) | `ModelChange`, clamping species and other structural changes |
| [model.model_resources](model.model_resources.md) | resolving model sources, i.e., files, URNs and URLs |
| [model.symbols](model.symbols.md) | `ModelSymbols`, the entities of a model a simulation changes and its initial assignments |
| [model.provenance](model.provenance.md) | the record of what was added to a derived model, i.e. the compiled networks, and its inverse |

## sbmlsim.simulation

Definition of simulations, see [Simulations](../simulation.md) and [Parameter scans](../scans.md).

| module | description |
| --- | --- |
| [simulation.definition](simulation.definition.md) | `Simulation`, `Change` and `SteadyState`, a simulation with its changes, with units |
| [simulation.scan](simulation.scan.md) | `ScanSim`, a simulation over the dimensions of parameter changes |
| [simulation.sensitivity](simulation.sensitivity.md) | `ModelSensitivity`, sensitivity scans of parameters and initial conditions |
| [simulation.range](simulation.range.md) | `Dimension`, a dimension of a scan |

## sbmlsim.simulator, sbmlsim.task

Execution of simulations.

| module | description |
| --- | --- |
| [simulator.simulation_serial](simulator.simulation_serial.md) | `SimulatorSerial`, running simulations and scans on a roadrunner model |
| [simulator.plan](simulator.plan.md) | `Plan`, a simulation compiled against a model, without units |
| [simulator.executor](simulator.executor.md) | `execute`, running a plan on roadrunner with the semantics of PEtab v2 |
| [simulator.formula](simulator.formula.md) | the formulas of the changes of a simulation |
| [task.task](task.task.md) | `Task`, a simulation applied to a model |

## sbmlsim.experiment, sbmlsim.result

Simulation experiments and their results, see [Simulation experiments](../experiments.md).

| module | description |
| --- | --- |
| [experiment.experiment](experiment.experiment.md) | `SimulationExperiment`, models, datasets, simulations, tasks, data and figures of an experiment |
| [experiment.runner](experiment.runner.md) | `ExperimentRunner`, executing experiments and writing their results |
| [result.timecourse](result.timecourse.md) | `TimecourseResult`, the array of the selections of a single timecourse simulation |
| [result.xresult](result.xresult.md) | `XResult`, simulation results as an xarray dataset with units |

## sbmlsim.plot, sbmlsim.report

Figures and reports, see [Plots and reports](../plotting.md).

| module | description |
| --- | --- |
| [plot.plotting](plot.plotting.md) | `Figure`, `Plot`, `Axis`, `Curve` and their styles, the plot description independent of the backend |
| [plot.serialization_matplotlib](plot.serialization_matplotlib.md) | rendering of the figures with matplotlib |
| [report.experiment_report](report.experiment_report.md) | HTML and markdown reports of simulation experiments |

## sbmlsim.fit

Parameter fitting, see [Parameter fitting](../fitting.md).

| module | description |
| --- | --- |
| [fit.objects](fit.objects.md) | `FitParameter`, `FitMapping`, `FitData` and `FitMappingCollection`, the objects of a fit problem |
| [fit.optimization](fit.optimization.md) | `OptimizationProblem`, the residuals and cost of a fit problem |
| [fit.options](fit.options.md) | `FitSettings` and the options of the optimization, i.e., algorithms, residuals, weighting and loss functions |
| [fit.parameters](fit.parameters.md) | `ParameterSet` and `ParameterSets`, the fitted parameters a report is created from |
| [fit.parameter_mapping](fit.parameter_mapping.md) | which parameter writes which entity in which simulation, the coverage of versioned parameters |
| [fit.derived](fit.derived.md) | derived changes of a simulation, the protocol a network before the simulation implements, and the summary of a hook for the console and the report |
| [fit.result](fit.result.md) | `OptimizationResult`, the result of an optimization |
| [fit.runner](fit.runner.md) | running optimizations serially or in parallel |
| [fit.report](fit.report.md) | `FitReport`, the figures and reports of one or more parameter sets |
| [fit.cli](fit.cli.md) | `FitDefinition` and the general command line tools which run and report a fit |
| [fit.display](fit.display.md) | the sections of the console output of a fit: the problem, the parameters, the settings and the data |
| [fit.sampling](fit.sampling.md) | sampling of initial parameter values |
| [fit.metrics](fit.metrics.md) | `FitMetrics` and the metrics of a fit: PRED, IPRED, residuals, MSE, RMSE, R² and AIC |
| [fit.identifiability](fit.identifiability.md) | `profile_likelihood`, the confidence intervals and the classification of the parameters from the profiles |
| [fit.fisher](fit.fisher.md) | `fisher_information`, the standard errors, correlations and constrained directions from one jacobian |
| [fit.helpers](fit.helpers.md) | helpers for fitting |
| [fit.petab_omex](fit.petab_omex.md) | COMBINE archives of PEtab problems |

PEtab v2, see [PEtab](../petab.md).

| module | description |
| --- | --- |
| [fit.petab_v2.export](fit.petab_v2.export.md) | writing an `OptimizationProblem` as a PEtab v2 problem |
| [fit.petab_v2.reader](fit.petab_v2.reader.md) | reading a PEtab v2 problem back into a fit problem |
| [fit.petab_v2.extension](fit.petab_v2.extension.md) | the `sbmlsim` block of the problem, i.e., what PEtab does not express |
| [fit.petab_v2.symbols](fit.petab_v2.symbols.md) | the identifiers and symbols shared by the PEtab layer |
| [fit.petab_v2.gaps](fit.petab_v2.gaps.md) | the catalogue of the differences between a fit and its PEtab problem |
| [fit.petab_v2.likelihood](fit.petab_v2.likelihood.md) | the log-likelihood and its gradient, chi2 and the priors, the noise models of PEtab v2 |
| [fit.petab_v2.testsuite](fit.petab_v2.testsuite.md) | the [PEtab Test Suite](../petab_testsuite.md): its cases of models and of math, and their comparison |
| [fit.petab_v2.benchmark](fit.petab_v2.benchmark.md) | the [PEtab benchmark collection](../petab_benchmark.md): its problems converted to PEtab v2, simulated and compared with the collection and with AMICI |
| [fit.petab_v2.sciml](fit.petab_v2.sciml.md) | the networks of a PEtab SciML problem read into `Hybridization` objects |
| [fit.petab_v2.sciml_export](fit.petab_v2.sciml_export.md) | the hybridizations of a problem written as PEtab SciML |

## sbmlsim.sensitivity

Local and global sensitivity analysis, see [Sensitivity analysis](../sensitivity.md).

| module | description |
| --- | --- |
| [sensitivity.analysis](sensitivity.analysis.md) | the common analysis of a model, i.e., outputs, observables and the simulation of parameter samples |
| [sensitivity.parameters](sensitivity.parameters.md) | selection, bounds and distributions of the analysed parameters |
| [sensitivity.sensitivity_local](sensitivity.sensitivity_local.md) | local sensitivities by finite differences |
| [sensitivity.sensitivity_sampling](sensitivity.sensitivity_sampling.md) | sampling based sensitivity and uncertainty analysis |
| [sensitivity.sensitivity_morris](sensitivity.sensitivity_morris.md) | Morris elementary effects screening |
| [sensitivity.sensitivity_sobol](sensitivity.sensitivity_sobol.md) | variance based Sobol indices |
| [sensitivity.sensitivity_fast](sensitivity.sensitivity_fast.md) | Fourier amplitude sensitivity test (FAST) |
| [sensitivity.classification](sensitivity.classification.md) | classification of sensitivities and uncertainties |
| [sensitivity.plots](sensitivity.plots.md) | plots of the sensitivity results |

## sbmlsim.sciml

The neural networks of hybrid problems, see [PEtab SciML](../petab_sciml.md). The package needs the extra `sciml`.

| module | description |
| --- | --- |
| [sciml](sciml.md) | the package: `Network`, `Hybridization`, `NetworkInput`, `NetworkPattern`, `compile_network`, `network_fit_parameters`, `nominal_parameters`, the id functions and the errors |
| [sciml.network](sciml.network.md) | `Network`, the architecture and the arrays of a network, its forward pass and the ids of its elements, inputs and outputs |
| [sciml.hybridization](sciml.hybridization.md) | `Hybridization`, where a network sits, its inputs and outputs, its fit parameters and the derived changes of a network before the simulation |
| [sciml.compiler](sciml.compiler.md) | `compile_network`, a network in the right hand side or in an observable written into the model as assignment rules |
| [sciml.formula](sciml.formula.md) | the math of SBML as sympy expressions, read with libsbml and sbmlmath, and written back, for the compiler and the hybridization |
| [sciml.parameters](sciml.parameters.md) | the nominal values and the fit parameters of a network per network, layer or array |
| [sciml.interpreter](sciml.interpreter.md) | the walk over the forward pass of the NN YAML with a backend |
| [sciml.backend](sciml.backend.md) | the numpy backend of the forward pass and the sympy backend of the compiler |
| [sciml.layers](sciml.layers.md) | the layers and functions of PEtab SciML with the backends they support |
| [sciml.errors](sciml.errors.md) | the errors of the package |
| [sciml.testsuite](sciml.testsuite.md) | the [PEtab SciML Test Suite](../sciml_testsuite.md): its cases, their comparison and the round trip |

## sbmlsim.testsuite

The semantic cases of the SBML Test Suite, see [SBML Test Suite](../testsuite.md).

| module | description |
| --- | --- |
| [testsuite.cases](testsuite.cases.md) | `SemanticCase` and `SemanticSuite`, the cases of a release on disk and its download |
| [testsuite.runner](testsuite.runner.md) | `run_case`, simulating a case, and `CaseStatus`, the outcomes a case can have |
| [testsuite.comparison](testsuite.comparison.md) | comparing the results of a case within its tolerances |
| [testsuite.report](testsuite.report.md) | `TestSuiteReport`, the interactive report of a run |
| [testsuite.cache](testsuite.cache.md) | the download and the cache of a test suite, shared by the SBML Test Suite, the PEtab test suite and the PEtab SciML test suite |
| [testsuite.baseline](testsuite.baseline.md) | the baseline of the PEtab test suites and of the benchmark collection, i.e. the cases which do not pass with their reasons |

## sbmlsim.comparison

| module | description |
| --- | --- |
| [comparison.diff](comparison.diff.md) | numerical comparison of simulation results from different simulators |
