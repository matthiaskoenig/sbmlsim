# PEtab Benchmark Problems

The [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab) is a set of parameter estimation problems of published models of systems biology, each with its model, its data and the simulations at the nominal parameters (`simulations.tsv`). The problems were not written by `sbmlsim`, so they test the reader of [PEtab](petab.md) on what other tools produce, and the simulations of the collection are the reference a simulation of a problem is compared with. The collection is PEtab 1.0, a problem is converted to PEtab v2 before it is read. All 35 problems of the pinned commit are converted and read; 28 agree with the simulations of the collection and with the log-likelihoods of AMICI, see [Running the collection](#running-the-collection).

## Running the collection

`sbmlsim.fit.petab_v2.benchmark` reads and simulates every problem of a pinned commit of the collection. `BenchmarkCollection.load()` downloads the commit into `~/.cache/sbmlsim/petab-benchmark/<commit>/` (`SBMLSIM_BENCHMARK_PATH` points elsewhere), converts every problem with `petab.v2.petab1to2` once and fetches the log-likelihoods which AMICI states for the problems (`tests/benchmark_models/benchmark_models.yaml` of a pinned commit of AMICI). `BenchmarkProblem.run()` reads a converted problem, simulates it once at the nominal values of its parameter table and compares:

| value | reference | tolerance |
| --- | --- | --- |
| the simulation of every measurement | `simulations.tsv` of the collection | `1e-3` absolute and relative |
| the log-likelihood | the `llh` of AMICI, where it has one and no observable is on the scale `log10`, whose density the conversion changes | `1e-3` absolute and `1e-6` relative |

A simulation of the collection is the one of its measurement row by row when the observables, the conditions and the times agree, up to ids which were renamed after the simulations were written, and is found by the observable, the conditions and the time of the measurement otherwise. The outcome of a problem is `pass`, `tolerance`, `conversion` (`petab1to2` fails) or `error`, and the result carries the time it took to read, to initialize, to simulate and to calculate the log-likelihood.

```bash
uv run python scripts/petab_benchmark.py download                      # fetch, convert and cache
uv run python scripts/petab_benchmark.py run --processes 8 --output results/benchmark
uv run python scripts/petab_benchmark.py report --output results/benchmark  # benchmark.md
uv run python scripts/petab_benchmark.py baseline --processes 8        # refresh the baseline
uv run pytest -m petab_benchmark tests/fit                              # every problem is a test
tox r -e benchmark                                                      # download and run
```

Of the seven which do not pass, five disagree with tables of the collection which are off, `Chen_MSB2009` stops in roadrunner at the steps of its rules, and `Froehlich_CellSystems2018`, 9169 experiments with a pre-equilibration each, takes longer than the `600` seconds a problem gets in `scripts/petab_benchmark.py`. The problems which do not pass are recorded in `tests/data/benchmark_baseline.json` with their status and the reason, which the tests and `run` compare a run with in both directions. Where the simulations of the collection disagree with `sbmlsim` while the log-likelihood agrees with the one of AMICI, it is the table of the collection which is off: the simulations of `Zheng_PNAS2012` are its measurements, the ones of `Perelson_Science1996` are not at the nominal parameters, which roadrunner without `sbmlsim` confirms.

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

The observables of `Boehm_JProteomeRes2014` are formulas over several species, e.g. `(100 * pApB + 200 * pApA * specC17) / (...)`, which is not what roadrunner selects. `sbmlsim` therefore reads every such observable as an `ObservableModel` (`sbmlsim.fit.objects`), i.e. the formula in the selections of roadrunner, which the fit evaluates on the simulation at the measurements: the identifiers of the math of PEtab are the ones of the model, a concentration based species is its concentration `[S]`. Simulated at the nominal parameters of the problem, the observables agree with the `simulatedData` of the collection to `2e-4` at a relative tolerance of `1e-9` of the integrator.

The parameters of a problem which are not estimated are applied to the model with the nominal values of the parameter table, which is what PEtab prescribes and which the model does not have to agree with.

A problem of the collection is not the same fit for `sbmlsim` as it is for PEtab: `Boehm_JProteomeRes2014` estimates the standard deviation of each of its observables, i.e. a Gaussian likelihood over the noise, while `sbmlsim` weights the data and fits the parameters of the model, so the two objectives have different optima. The profile likelihood of the example says so: it finds a lower cost than the parameters it starts from and reports that the fit did not converge.

A simulation experiment which is read from a PEtab problem is created when the problem is read, and a class which is created cannot be pickled, so such a fit runs in one process: `n_cores=1`, which is what the example uses.

Two things of the collection do not survive the conversion, and the example shows both. The parameter table of v1 has a `parameterScale`, which v2 removed and which is `FitSettings.parameter_scale` here, and the observable of the problem has a `log10-normal` noise distribution which the converter maps to `log-normal`; the noise of an observable is what the log-likelihood is calculated with and not what `sbmlsim` fits, so the example fits the relative residuals (`ResidualType.NORMALIZED`) which describe a viral load over orders of magnitude in the same spirit. The problem also estimates the standard deviation `sd_task0_model0_perelson1_V` of its observable, which is not an entity of the model: `sbmlsim` fits the parameters of a model and weights the data, so it is not fitted and the reader says so.
