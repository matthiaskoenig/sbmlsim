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
| [log](log.md) | logging of the package |

## sbmlsim.model

Models and model changes, see [Models](../models.md).

| module | description |
| --- | --- |
| [model.model](model.model.md) | `AbstractModel`, the model of a simulation experiment with its changes and selections |
| [model.model_roadrunner](model.model_roadrunner.md) | `RoadrunnerSBMLModel`, the roadrunner instance of an SBML model with its units and parameter changes |
| [model.model_change](model.model_change.md) | `ModelChange`, clamping species and other structural changes |
| [model.model_resources](model.model_resources.md) | resolving model sources, i.e., files, URNs and URLs |

## sbmlsim.simulation

Definition of simulations, see [Timecourse simulations](../simulation.md) and [Parameter scans](../scans.md).

| module | description |
| --- | --- |
| [simulation.timecourse](simulation.timecourse.md) | `Timecourse` and `TimecourseSim`, concatenated timecourses with changes |
| [simulation.scan](simulation.scan.md) | `ScanSim`, a simulation over the dimensions of parameter changes |
| [simulation.sensitivity](simulation.sensitivity.md) | `ModelSensitivity`, sensitivity scans of parameters and initial conditions |
| [simulation.range](simulation.range.md) | ranges of values for scans |
| [simulation.change](simulation.change.md) | changes applied to a model before a simulation |
| [simulation.algorithm](simulation.algorithm.md) | `Algorithm` and `AlgorithmParameter`, the KISAO description of an integrator |
| [simulation.kisaos](simulation.kisaos.md) | the KISAO terms of the supported algorithms and parameters |
| [simulation.calculation](simulation.calculation.md) | calculations on simulation results |
| [simulation.base](simulation.base.md) | base classes shared by the simulation objects and SED-ML |
| [simulation.simulation](simulation.simulation.md) | `AbstractSim`, the base of all simulations |

## sbmlsim.simulator, sbmlsim.task

Execution of simulations.

| module | description |
| --- | --- |
| [simulator.simulation_serial](simulator.simulation_serial.md) | `SimulatorSerial`, running timecourses and scans on a roadrunner model |
| [task.task](task.task.md) | `Task`, a simulation applied to a model |

## sbmlsim.experiment, sbmlsim.result

Simulation experiments and their results, see [Simulation experiments](../experiments.md).

| module | description |
| --- | --- |
| [experiment.experiment](experiment.experiment.md) | `SimulationExperiment`, models, datasets, simulations, tasks, data and figures of an experiment |
| [experiment.runner](experiment.runner.md) | `ExperimentRunner`, executing experiments and writing their results |
| [result.xresult](result.xresult.md) | `XResult`, simulation results as an xarray dataset with units |
| [result.datagenerator](result.datagenerator.md) | data generators processing results |
| [result.report](result.report.md) | reports of results |

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
| [fit.objects](fit.objects.md) | `FitParameter`, `FitMapping`, `FitData` and `FitExperiment`, the objects of a fit problem |
| [fit.optimization](fit.optimization.md) | `OptimizationProblem`, the residuals and cost of a fit problem |
| [fit.options](fit.options.md) | `FitSettings` and the options of the optimization, i.e., algorithms, residuals, weighting and loss functions |
| [fit.parameters](fit.parameters.md) | `ParameterSet` and `ParameterSets`, the fitted parameters a report is created from |
| [fit.result](fit.result.md) | `OptimizationResult`, the result of an optimization |
| [fit.runner](fit.runner.md) | running optimizations serially or in parallel |
| [fit.report](fit.report.md) | `FitReport`, the figures and reports of one or more parameter sets |
| [fit.cli](fit.cli.md) | `FitDefinition` and the general command line tools which run and report a fit |
| [fit.display](fit.display.md) | the sections of the console output of a fit: the problem, the parameters, the settings and the data |
| [fit.sampling](fit.sampling.md) | sampling of initial parameter values |
| [fit.metrics](fit.metrics.md) | `FitMetrics` and the metrics of a fit: PRED, IPRED, residuals, MSE, RMSE, R² and AIC |
| [fit.helpers](fit.helpers.md) | helpers for fitting |
| [fit.petab_omex](fit.petab_omex.md) | COMBINE archives of PEtab problems |

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

## sbmlsim.combine

SED-ML, NuML and COMBINE archives, see [SED-ML and COMBINE archives](../sedml.md).

| module | description |
| --- | --- |
| [combine.sedml.parser](combine.sedml.parser.md) | `SEDMLParser`, a SED-ML document into a simulation experiment |
| [combine.sedml.runner](combine.sedml.runner.md) | executing SED-ML files and COMBINE archives |
| [combine.sedml.task](combine.sedml.task.md) | tasks and repeated tasks of SED-ML |
| [combine.sedml.data](combine.sedml.data.md) | data descriptions, i.e., NuML, CSV and TSV data |
| [combine.sedml.numl](combine.sedml.numl.md) | parser for NuML data |
| [combine.sedml.report](combine.sedml.report.md) | SED-ML reports |
| [combine.sedml.io](combine.sedml.io.md) | reading and writing SED-ML documents |
| [combine.datagenerator](combine.datagenerator.md) | data generators of SED-ML |
| [combine.mathml](combine.mathml.md) | evaluation of MathML expressions |

## sbmlsim.interpolation, sbmlsim.comparison

| module | description |
| --- | --- |
| [interpolation.interpolation](interpolation.interpolation.md) | interpolation of datasets as SBML models |
| [comparison.diff](comparison.diff.md) | numerical comparison of simulation results from different simulators |
