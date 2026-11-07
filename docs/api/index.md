# API reference

The API reference is generated from the docstrings of the package.

## sbmlsim

The top level modules: data, units and the shared output.

| module | description |
| --- | --- |
| [data](data.md) | `Data` objects referencing simulation results and datasets, the input of plots and calculations |
| [units](units.md) | unit registry of a model and unit conversions with pint |
| [distrib](distrib.md) | the grammar of the distrib formulas of the fitted parameters (`normal`, `lognormal`, `uniform`, truncated), without a simulator |
| [licenses](licenses.md) | the license texts of a model (CC BY-SA 4.0, MIT) and `write_license`, which writes them byte for byte |
| [engine](engine.md) | `Engine`, the engines which simulate a model, roadrunner, COPASI and AMICI, the imports of the optional COPASI and AMICI and the directory of the compiled models of AMICI the workers of a pool load from |
| [sbml_math](sbml_math.md) | the math of SBML as sympy expressions and numpy functions, the math of the events of a model |
| [serialization](serialization.md) | JSON serialization of experiments |
| [utils](utils.md) | timing and other helpers |
| [parallel](parallel.md) | the process pools of sbmlsim: the start method, the kept pools and the caches of the workers |
| [project](project.md) | `Project`, a model package: its model variants and factory, experiments, fits and their records, scans and uncertainty analyses, which the run layer resolves, with the settings of its simulator, figures, checks and release |
| [project_checks](project_checks.md) | the checks of a model package as functions over a `Project` |
| [events](events.md) | the events of a run as JSON lines |
| [atomic](atomic.md) | atomic writes of files and directories |
| [cgroup](cgroup.md) | the CPU quota of the cgroup of the process |
| [console](console.md) | shared rich console |
| [display](display.md) | the output of scripts: sections, key/value blocks and links which the terminal opens with a click |
| [log](log.md) | logging of the package |

## sbmlsim.model

Models and the changes of a model, see [Models](../models.md).

| module | description |
| --- | --- |
| [model.model](model.model.md) | `AbstractModel`, the model of a simulation experiment with its changes and selections |
| [model.model_sbml](model.model_sbml.md) | `SBMLModel`, the SBML model independent of its engine, and `load_model`, which loads a model in an engine |
| [model.model_roadrunner](model.model_roadrunner.md) | `RoadrunnerSBMLModel`, the roadrunner instance of an SBML model with its units and parameter changes |
| [model.model_copasi](model.model_copasi.md) | `CopasiSBMLModel`, an SBML model loaded in COPASI, with the values and selections of roadrunner |
| [model.model_amici](model.model_amici.md) | `AmiciSBMLModel`, an SBML model compiled with AMICI, with the values and selections of roadrunner, the cache of the compiled models and the refused features |
| [model.comparisons](model.comparisons.md) | the comparisons of the maths of a model, which COPASI and AMICI shift at their threshold |
| [model.model_resources](model.model_resources.md) | `Source`, resolving the source of a model, i.e., a file or the SBML itself |
| [model.event_rows](model.event_rows.md) | `with_event_markers`, the markers which tell the rows of the events of the model, and `at_event_times`, which gives the two rows of an event in the steps of the integrator the time of the event |
| [model.symbols](model.symbols.md) | `ModelSymbols`, the entities of a model a simulation changes and its initial assignments |
| [model.provenance](model.provenance.md) | the record of what was added to a derived model, i.e. the compiled networks, and its inverse |

## sbmlsim.simulation

Definition of simulations, see [Simulations](../simulation.md) and [Parameter scans](../scans.md) and [Observables](../observables.md).

| module | description |
| --- | --- |
| [simulation.definition](simulation.definition.md) | `Simulation`, `Change` and `SteadyState`, a simulation with its changes, with units |
| [simulation.scan](simulation.scan.md) | `Scan` and `Dimension`, a simulation over dimensions of values, simulations and models |
| [simulation.observables](simulation.observables.md) | `Formula`, `PK` and `Custom`, what a scan computes from every simulation |
| [simulation.sampling](simulation.sampling.md) | the sampler: distributions, references and the designs of a scan, of the sensitivity analyses, of a fit and of a population |

## sbmlsim.simulator, sbmlsim.task

Execution of simulations.

| module | description |
| --- | --- |
| [simulator.simulator](simulator.simulator.md) | `Simulator`, simulations and scans of models, serially or in a pool of processes |
| [simulator.plan](simulator.plan.md) | `Plan`, a simulation compiled against a model, without units |
| [simulator.executor](simulator.executor.md) | `execute`, running a plan on roadrunner with the semantics of PEtab v2 |
| [simulator.executor_copasi](simulator.executor_copasi.md) | `execute`, running a plan on COPASI with the semantics of the roadrunner executor, and the known differences |
| [simulator.executor_amici](simulator.executor_amici.md) | `execute`, running a plan on AMICI with the semantics of the roadrunner executor, the executor firing the events, and the known differences |
| [simulator.segments](simulator.segments.md) | `simulate_segments`, the segments of a plan between its changes, shared by the executors of COPASI and AMICI |
| [simulator.engines](simulator.engines.md) | `execute_plan`, running a plan with the executor of the engine of a model |
| [simulator.changes](simulator.changes.md) | the changes of a plan and the events a change fires, shared by the engines |
| [simulator.formula](simulator.formula.md) | the formulas of the changes of a simulation |
| [simulator.worker](simulator.worker.md) | the chunks of a scan, run on one plan and one model in a worker |
| [simulator.observables](simulator.observables.md) | the compiled observables of a scan and their evaluation on the native solutions |
| [simulator.pk](simulator.pk.md) | the non-compartmental analysis of a `PK` observable with pkpdutils |
| [task.task](task.task.md) | `Task`, a simulation applied to a model |

## sbmlsim.experiment, sbmlsim.result

Simulation experiments and their results, see [Simulation experiments](../experiments.md).

| module | description |
| --- | --- |
| [experiment.experiment](experiment.experiment.md) | `SimulationExperiment`, models, datasets, simulations, tasks, data and figures of an experiment |
| [experiment.runner](experiment.runner.md) | `ExperimentRunner`, executing experiments and writing their results |
| [experiment.progress](experiment.progress.md) | `RunProgress` and `PhaseClock`, the progress of a run of experiments and the seconds of their phases |
| [experiment.draws](experiment.draws.md) | `UncertaintySettings`, `UncertaintyRun` and `apply_uncertainty`, the experiments over draws of the fitted parameters with bands; `check_fit_parameters` of the variants of a project |
| [experiment.sids](experiment.sids.md) | `to_sid`, the SId of a text such as a label of the data, a key of an experiment |
| [result.timecourse](result.timecourse.md) | `TimecourseResult`, the array of the selections of a single timecourse simulation |
| [result.scan](result.scan.md) | `ScanResult`, the result of a scan as an xarray dataset with units, and the functions of a point: its timecourse with the state after a change, the start and end values, the band of the draws |
| [result.netcdf](result.netcdf.md) | the compressed netCDF files of `ScanResult` and `SensitivityResult` |

## sbmlsim.plot, sbmlsim.report

Figures and reports, see [Plots and reports](../plotting.md).

| module | description |
| --- | --- |
| [plot.plotting](plot.plotting.md) | `Figure`, `Plot`, `Axis`, `Curve` and their styles, the plot description independent of the backend |
| [plot.points](plot.points.md) | the lines, colours, legend labels and colour bars of a curve over the points of a scan, and the quantiles of a band |
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
| [fit.variability](fit.variability.md) | `Variability`, the relative distributions of the subjects of a fit problem; `subject_predictions` and `predictive_check`, the visual predictive check of the mappings with virtual groups |
| [fit.helpers](fit.helpers.md) | helpers for fitting |
| [fit.petab_omex](fit.petab_omex.md) | COMBINE archives of PEtab v2 problems |

PEtab v2, see [PEtab](../petab.md).

| module | description |
| --- | --- |
| [fit.petab_v2.export](fit.petab_v2.export.md) | writing an `OptimizationProblem` as a PEtab v2 problem |
| [fit.petab_v2.reader](fit.petab_v2.reader.md) | reading a PEtab v2 problem back into a fit problem |
| [fit.petab_v2.extension](fit.petab_v2.extension.md) | the `sbmlsim` block of the problem, i.e., what PEtab does not express |
| [fit.petab_v2.symbols](fit.petab_v2.symbols.md) | the identifiers and symbols shared by the PEtab layer |
| [fit.petab_v2.gaps](fit.petab_v2.gaps.md) | the catalogue of the differences between a fit and its PEtab problem |
| [fit.petab_v2.conversion](fit.petab_v2.conversion.md) | the conversion of a problem of PEtab v1 and its simulations to PEtab v2 |
| [fit.petab_v2.likelihood](fit.petab_v2.likelihood.md) | the log-likelihood and its gradient, chi2 and the priors, the noise models of PEtab v2 |
| [fit.petab_v2.testsuite](fit.petab_v2.testsuite.md) | the [PEtab Test Suite](../petab_testsuite.md): its cases of models and of math, and their comparison |
| [fit.petab_v2.benchmark](fit.petab_v2.benchmark.md) | the [PEtab benchmark collection](../petab_benchmark.md): its problems converted to PEtab v2, simulated and compared with the collection and with AMICI |
| [fit.petab_v2.sciml](fit.petab_v2.sciml.md) | the networks of a PEtab SciML problem read into `Hybridization` objects |
| [fit.petab_v2.sciml_export](fit.petab_v2.sciml_export.md) | the hybridizations of a problem written as PEtab SciML |

## sbmlsim.sensitivity

Local and global sensitivity analysis, see [Sensitivity analysis](../sensitivity.md).

| module | description |
| --- | --- |
| [sensitivity.indices](sensitivity.indices.md) | the local, Sobol, FAST and Morris indices of the result of a scan |
| [sensitivity.result](sensitivity.result.md) | the `SensitivityResult` with the indices, `classify` and netCDF |
| [sensitivity.classification](sensitivity.classification.md) | classification of sensitivities and uncertainties |
| [sensitivity.plots](sensitivity.plots.md) | heatmap, bars of the indices and the plane of Morris |
| [sensitivity.uncertainty](sensitivity.uncertainty.md) | bands and distributions of a scan over draws |
| [sensitivity.parameters](sensitivity.parameters.md) | the table of the parameters of an analysis, as sbmlsim 0.8 wrote it |
| [sensitivity.runner](sensitivity.runner.md) | the sensitivity analyses of a project over the groups of its simulations, with the files of sbmlsim 0.8.5 |

## sbmlsim.release

The release of a project as a repository of its own, see [Release](../release.md).

| module | description |
| --- | --- |
| [release.settings](release.settings.md) | `ReleaseSettings` and `Author`, the release metadata of a project, standard library only |
| [release.archive](release.archive.md) | the COMBINE archive of the models of a release: the SBML, its ODEs and antimony, the licenses |
| [release.workflow](release.workflow.md) | the build of a release into an output and its publishing, `build_release` and `publish_release` |
| [release.source](release.source.md) | the source repository and the release repository of a model, the checks before a build |
| [release.project](release.project.md) | the project files of a release repository written from templates |
| [release.build](release.build.md) | the build of a release repository: the analyses of the release in its own environment |
| [release.notes](release.notes.md) | the release notes of a release repository |
| [release.docs](release.docs.md) | the documentation site and the README of a release repository |
| [release.manifest](release.manifest.md) | the manifest `release.json` of a release and the removal of what the release did not build |
| [release.publish](release.publish.md) | the publishing of a built release: commit, tag, Pages, Docker image and GitHub release |
| [release.hooks](release.hooks.md) | `ReleaseHook`, what a framework adds to the release of a model package |
| [release.cli](release.cli.md) | the command line `python -m sbmlsim.release` |

## sbmlsim.credibility

The credibility of a model for its contexts of use, see [Credibility](../credibility.md).

| module | description |
| --- | --- |
| [credibility.declaration](credibility.declaration.md) | `Credibility`, `ContextOfUse`, `Domain`, `Evidence`, `QuantityOfInterest` and the acceptance rules, the declaration of the contexts of use of a model, and `metadata_fields`, the fields of the metadata a domain claims |
| [credibility.risk](credibility.risk.md) | the model risk of the influence and the consequence, the default acceptance and the default minimum of points |
| [credibility.factors](credibility.factors.md) | `FACTORS`, the credibility factors of ASME V&V 40 and their status |
| [credibility.extensions](credibility.extensions.md) | `EvidenceExtension`, the evidence a project adds to the credibility, and the sections of the extensions in the records |
| [credibility.selection](credibility.selection.md) | `select`, the data points of the evidence of the contexts of use |
| [credibility.records](credibility.records.md) | `evaluate_fits` and `mapping_table`, the data points of the adopted fits with the recorded kinds of their mappings |
| [credibility.lock](credibility.lock.md) | the lock of the evidence, written into an out and compared with the one of the project |
| [credibility.check](credibility.check.md) | `check_declaration`, the static check of a declaration against the adopted fits |
| [credibility.uncertainty](credibility.uncertainty.md) | `propagate_fits`, the parameter uncertainty of the adopted fits propagated to the decisive predictions |
| [credibility.variability](credibility.variability.md) | `predictive_checks`, the visual predictive check of the decisive points with the variability of the subjects |
| [credibility.cross_validation](credibility.cross_validation.md) | `cross_validate`, the leave-one-study-out cross-validation of an adopted fit |
| [credibility.verification](credibility.verification.md) | `verify`, the decisive predictions with tightened tolerances and a second integration method, and the open checks, `Check` and `register_check` |
| [credibility.battery](credibility.battery.md) | `run_battery`, the checks of the variants of a model, and `BatteryHook`, the simulations and checks a project adds |
| [credibility.assessment](credibility.assessment.md) | `assess`, the assessment of the contexts of use on the adopted fits, `CredibilityAssessment` and its record |
| [credibility.report](credibility.report.md) | `write_outputs`, the record, the tables and the report of an assessment |
| [credibility.scorecard](credibility.scorecard.md) | `scorecard` and `compare`, the assessments of many models and their regressions against a baseline |

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
| [testsuite.submission](testsuite.submission.md) | `write_submission`, the archive with the results of a run for the SBML Test Suite Database |
| [testsuite.cache](testsuite.cache.md) | the download and the cache of a test suite, shared by the SBML Test Suite, the PEtab test suite and the PEtab SciML test suite |
| [testsuite.baseline](testsuite.baseline.md) | the baseline of the PEtab test suites and of the benchmark collection, i.e. the cases which do not pass with their reasons |

## sbmlsim.comparison

| module | description |
| --- | --- |
| [comparison.diff](comparison.diff.md) | numerical comparison of simulation results from different simulators |
