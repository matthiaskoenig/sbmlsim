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

The training and the validation data are written, the outliers of a fit are not: a tool which reads the problem fits every measurement it finds.

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

- **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
- **lossy**: the information is transformed. The weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
- **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation.

The round trip of the HCTZ example keeps the settings, the parameters with their units, the mappings with their kinds and the reference data of every mapping, and its cost agrees to `7e-6`, which is the `selections` gap above.

## The example

`examples/hctz/fitting/petab_problem.py` runs the layer on the reference problem, i.e. it reports the gaps of the fit, writes it, validates the problem with `petab`, lists which collection every experiment came from and reads the fit back:

```bash
python -m examples.hctz.fitting.petab_problem
python -m examples.hctz.fitting.petab_problem --subset=PKIV --portable
```

The `PKIV` problem is one simulation experiment, so it comes back exactly (a relative difference of `5e-16` in the cost); `PK` is two, which is the `selections` gap and a difference of `7e-6`.

## COMBINE archives

`sbmlsim.fit.petab_omex.create_petab_omex` packages a PEtab problem as a COMBINE archive, with the YAML as its master entry.
