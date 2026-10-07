# PEtab Benchmark Problems

The [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab) is a set of parameter estimation problems of published models of systems biology, each with its model, its data and the simulations at the nominal parameters (`simulations.tsv`). The problems were not written by `sbmlsim`, so they test the reader of [PEtab](petab.md) on what other tools produce, and the simulations of the collection are the reference a simulation of a problem is compared with. The collection is PEtab 1.0, a problem is converted to PEtab v2 before it is read. Not every problem of the collection is read and simulated yet.

## Fitting a problem of the collection

`examples/petab/benchmark.py` fits problems which `sbmlsim` did not write, from the collection: `Perelson_Science1996`, the viral dynamics of HIV-1 after the start of a protease inhibitor, and `Boehm_JProteomeRes2014`, the dimerization of STAT5A and STAT5B. The collection is PEtab 1.0, so the example converts a problem with the converter of the library first:

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

The observables of `Boehm_JProteomeRes2014` are formulas over several species, e.g. `(100 * pApB + 200 * pApA * specC17) / (...)`, which is not what roadrunner selects. `sbmlsim.fit.petab_v2.observables` therefore writes a copy of the model in which every such observable is a parameter with an assignment rule, and the fit selects that parameter: the identifiers of the math of PEtab are the ones of the model, and the formula is translated into the math of SBML for the rule. Simulated at the nominal parameters of the problem, the observables agree with the `simulatedData` of the collection to `2e-4` at a relative tolerance of `1e-9` of the integrator.

The parameters of a problem which are not estimated are applied to the model with the nominal values of the parameter table, which is what PEtab prescribes and which the model does not have to agree with.

A problem of the collection is not the same fit for `sbmlsim` as it is for PEtab: `Boehm_JProteomeRes2014` estimates the standard deviation of each of its observables, i.e. a Gaussian likelihood over the noise, while `sbmlsim` weights the data and fits the parameters of the model, so the two objectives have different optima. The profile likelihood of the example says so: it finds a lower cost than the parameters it starts from and reports that the fit did not converge.

A simulation experiment which is read from a PEtab problem is created when the problem is read, and a class which is created cannot be pickled, so such a fit runs in one process: `n_cores=1`, which is what the example uses.

Two things of the collection do not survive the conversion, and the example shows both. The parameter table of v1 has a `parameterScale`, which v2 removed and which is `FitSettings.parameter_scale` here, and the observable of the problem has a `log10-normal` noise distribution which the converter maps to `log-normal`; the noise of an observable is what the log-likelihood is calculated with and not what `sbmlsim` fits, so the example fits the relative residuals (`ResidualType.NORMALIZED`) which describe a viral load over orders of magnitude in the same spirit. The problem also estimates the standard deviation `sd_task0_model0_perelson1_V` of its observable, which is not an entity of the model: `sbmlsim` fits the parameters of a model and weights the data, so it is not fitted and the reader says so.
