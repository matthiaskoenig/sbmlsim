# PEtab

[PEtab](https://petab.readthedocs.io) is the table based format for parameter estimation problems of systems biology models. `sbmlsim.fit.petab_v2` writes an optimization problem as a PEtab v2 problem and reads a PEtab v2 problem into an optimization problem, so a fit is handed to the tools of the PEtab ecosystem and a problem of the [benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab) is fitted with `sbmlsim`.

## Writing a fit as PEtab

`to_petab` writes the YAML of the problem, its tables and the models of the fit into a directory:

```py
from pathlib import Path

from sbmlsim.fit.petab_v2 import to_petab

yaml_file = to_petab(problem, Path("results") / "petab", settings=settings)
```

The problem is translated as follows:

| `sbmlsim` | PEtab v2 |
| --- | --- |
| the model of a task | a model of `model_files`, every SBML file once |
| a `FitMappingCollection` whose mappings share a simulation | one experiment, with the id of the collection |
| a collection over several simulations, e.g. the doses of a study | one experiment per simulation, numbered after the collection |
| a `Timecourse` of the `TimecourseSim` | a period of the experiment, its changes are a condition |
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

An experiment of PEtab is a simulation with its conditions and the observables which are measured in it, which is what a `FitMappingCollection` is, so the reader gives one collection back per experiment of the problem. A fit which is written and read again therefore comes back as one collection per simulation rather than as the collections it was defined with, which is the same fit of the same data.

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

- **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The scale the optimizer searches in is part of the settings, which is where PEtab v2 puts it as well: it removed the `parameterScale` of its parameter table because the scale is a property of the optimization and not of the problem, so the bounds and the start values are written on the linear scale. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
- **lossy**: the information is transformed. The weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
- **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation.

The round trip of the HCTZ example keeps the settings, the parameters with their units, the mappings with their kinds and the reference data of every mapping, and its cost agrees to `7e-6`, which is the `selections` gap above.

## The example

`examples/hctz_fitting/fitting/petab_problem.py` runs the layer on the reference problem, i.e. it reports the gaps of the fit, writes it, validates the problem with `petab`, lists which collection every experiment came from and reads the fit back:

```bash
python -m examples.hctz_fitting.fitting.petab_problem
python -m examples.hctz_fitting.fitting.petab_problem --subset=PKIV --portable
```

The `PKIV` problem is one simulation experiment, so it comes back exactly (a relative difference of `5e-16` in the cost); `PK` is two, which is the `selections` gap and a difference of `7e-6`.

## A problem of the benchmark collection

`examples/petab/benchmark.py` fits problems which `sbmlsim` did not write, from the [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab): `Perelson_Science1996`, the viral dynamics of HIV-1 after the start of a protease inhibitor, and `Boehm_JProteomeRes2014`, the dimerization of STAT5A and STAT5B. The collection is PEtab 1.0, so the example converts a problem with the converter of the library first:

```py
from petab.v2.petab1to2 import petab1to2

petab1to2(problem_dir / "Perelson_Science1996.yaml", output_dir=petab2_dir)
```

```bash
python -m examples.petab.benchmark
python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014
python -m examples.petab.benchmark --runs=8 --no-identifiability
```

The example reads the converted problem, reports what PEtab cannot express about it, fits it, analyses the identifiability of the fitted parameters and writes the report of the fit. For `Perelson_Science1996` the clearance rate `c` of the virions is identifiable and the loss rate `delta` of the infected cells is not identifiable towards zero.

The observables of `Boehm_JProteomeRes2014` are formulas over several species, e.g. `(100 * pApB + 200 * pApA * specC17) / (...)`, which is not what roadrunner selects. `sbmlsim.fit.petab_v2.observables` therefore writes a copy of the model in which every such observable is a parameter with an assignment rule, and the fit selects that parameter: the math of PEtab is the math of the model, so the formula is the rule. Simulated at the nominal parameters of the problem, the observables agree with the `simulatedData` of the collection to `2e-4` at a relative tolerance of `1e-9` of the integrator.

The parameters of a problem which are not estimated are applied to the model with the nominal values of the parameter table, which is what PEtab prescribes and which the model does not have to agree with.

A problem of the collection is not the same fit for `sbmlsim` as it is for PEtab: `Boehm_JProteomeRes2014` estimates the standard deviation of each of its observables, i.e. a Gaussian likelihood over the noise, while `sbmlsim` weights the data and fits the parameters of the model, so the two objectives have different optima. The profile likelihood of the example says so: it finds a lower cost than the parameters it starts from and reports that the fit did not converge.

A simulation experiment which is read from a PEtab problem is created when the problem is read, and a class which is created cannot be pickled, so such a fit runs in one process: `n_cores=1`, which is what the example uses.

Two things of the collection do not survive the conversion, and the example shows both. The parameter table of v1 has a `parameterScale`, which v2 removed and which is `FitSettings.parameter_scale` here, and the observable of the problem has a `log10-normal` noise distribution which the converter maps to `log-normal`; `sbmlsim` has no log noise, so the example fits the relative residuals (`ResidualType.NORMALIZED`) which describe a viral load over orders of magnitude in the same spirit. The problem also estimates the standard deviation `sd_task0_model0_perelson1_V` of its observable, which is not an entity of the model: `sbmlsim` fits the parameters of a model and weights the data, so it is not fitted and the reader says so.

## COMBINE archives

`sbmlsim.fit.petab_omex.create_petab_omex` packages a PEtab problem as a COMBINE archive, with the YAML as its master entry.
