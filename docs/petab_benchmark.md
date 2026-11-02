# PEtab Benchmark Problems

The [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab) is a set of parameter estimation problems of published systems biology models. Each problem has its model, its data and the simulations at the nominal parameters (`simulations.tsv`). Other people wrote these problems, so they test the [PEtab](petab.md) reader of `sbmlsim` on real files, and the simulations of the collection are the reference to compare with.

`sbmlsim` reads PEtab v2 only. The collection has every problem in PEtab v1 and some also in PEtab v2 (`Boehm_JProteomeRes2014` and `Schwen_PONE2014` at the pinned commit): a problem is run in its encoding of PEtab v2 where the collection has one, and converted from PEtab v1 otherwise. The pinned commit is the newest of the collection and the log-likelihoods of AMICI are the ones of its release v1.2.0. Of the 35 problems, `Froehlich_CellSystems2018` is left out, its 9169 experiments with a pre-equilibration each take longer than the 600 seconds a problem gets (`EXCLUDED_PROBLEMS`). The other 34 are read, and 28 agree with the collection and with AMICI. The other six are explained [below](#problems-that-do-not-pass).

## Running the collection

```bash
uv run python scripts/petab_benchmark.py download                                     # download, convert and cache
uv run python scripts/petab_benchmark.py run --processes 8 --output results/benchmark  # run every problem
uv run python scripts/petab_benchmark.py report --output results/benchmark             # write benchmark.md
uv run python scripts/petab_benchmark.py baseline --processes 8                       # refresh the baseline
uv run pytest -m petab_benchmark tests/fit                                             # one test per problem
tox r -e benchmark                                                                     # download, then run the tests
```

`--problems` runs only the named problems, also an excluded one. Every problem gets at most 600 seconds.

`download` fetches the pinned commit into `~/.cache/sbmlsim/petab-benchmark/<commit>/` (`SBMLSIM_BENCHMARK_PATH` points elsewhere), writes every problem in PEtab v2 into `v2/<problem>` (its encoding of PEtab v2, or its encoding of PEtab v1 converted with `petab.v2.petab1to2`) and fetches the log-likelihoods that AMICI reports for the problems. The simulations of the collection are tables of PEtab v1, they are converted with the problem: the problem with the simulations in place of its measurements is converted as well, so the simulations get the experiments of PEtab v2 that the conversion gives the measurements (`sbmlsim.fit.petab_v2.conversion`). After the download nothing reads PEtab v1. A run then reads each problem, simulates it once at the nominal values of its parameter table and compares:

| value | reference | tolerance |
| --- | --- | --- |
| the simulation at every measurement | `simulations.tsv` of the collection | `1e-3` absolute and relative |
| the log-likelihood | the `llh` of AMICI, where AMICI has one | `1e-3` absolute, `1e-6` relative |

Simulations are matched to the collection row by row when the observables, experiments and times agree, also when ids were renamed after the collection wrote its simulations, or by an encoding of PEtab v2 of the collection. Otherwise they are matched by observable, experiment and time. Simulations of conditions a problem renamed afterwards (`Fujita_SciSignal2010`) get the conditions of the measurements before they are converted. The outcome of a problem is `pass`, `tolerance`, `conversion` (`petab1to2` fails) or `error`, and the result records how long reading, initializing, simulating and the log-likelihood took.

The code is `sbmlsim.fit.petab_v2.benchmark`. The tests are marked `petab_benchmark` and run only on demand, because the largest problems take minutes.

## Problems that do not pass

`tests/data/benchmark_baseline.json` lists the six problems that do not pass, each with the reason. The tests and `run` compare with it in both directions.

In four of them the simulations of the collection are wrong, while `sbmlsim` agrees with AMICI or with an independent integrator; they are reported upstream ([Benchmark-Models-PEtab#278](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/issues/278), [#181](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/issues/181)):

- `Fujita_SciSignal2010` and `Sneyd_PNAS2002`: the simulations of the collection differ, while the log-likelihood agrees with AMICI; the log-likelihood of the simulations of the collection does not.
- `Zheng_PNAS2012`: the simulations of the collection are its measurements. The log-likelihood agrees with AMICI.
- `Perelson_Science1996`: the simulations of the collection are not at the nominal parameters. roadrunner without `sbmlsim` and SciPy give the same values as `sbmlsim`.

`Crauste_CellSystems2017` needs a tighter integrator (`PROBLEM_SETTINGS`, `EarlyEffector` grows towards a blow up in finite time), with which its simulations agree with the collection; its log-likelihood, which the simulations of the collection give as well, differs from the one of AMICI by `1.2e-3`, just above the tolerance.

The other one:

- `Chen_MSB2009`: a few simulations differ by up to 0.28 % and the log-likelihood differs from AMICI. Tighter tolerances do not change it.

## Fitting a problem of the collection

`examples/petab/benchmark.py` fits two problems of the collection, which are copied next to it in PEtab v2: the encoding of PEtab v2 of the collection of `Boehm_JProteomeRes2014`, and `Perelson_Science1996` converted from PEtab v1:

- `Perelson_Science1996`: the viral dynamics of HIV-1 after the start of a protease inhibitor, with one observable that is a species of the model.
- `Boehm_JProteomeRes2014`: the dimerization of STAT5A and STAT5B, with three observables that are formulas over several species.

```bash
python -m examples.petab.benchmark
python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014
python -m examples.petab.benchmark --runs=8 --no-identifiability
python -m examples.petab.benchmark --runs=8 --cores=4
```

The example reads the problem, prints what PEtab cannot express about it, fits it, analyses the [identifiability](identifiability.md) of the fitted parameters and writes the report of the fit into `results/<problem>`. For `Perelson_Science1996` the clearance rate `c` of the virions is identifiable, while the loss rate `delta` of the infected cells is not identifiable towards zero. `--cores` sets the number of worker processes; the default of one fits in the calling process.

A few things to know when you fit a problem of the collection with `sbmlsim`:

- An observable that is a formula, such as `(100 * pApB + 200 * pApA * specC17) / (...)` in `Boehm_JProteomeRes2014`, is read as an `ObservableModel` and evaluated on the simulation at the measurements. The model is not changed.
- Parameters that are not estimated are set to the nominal values of the parameter table, as PEtab prescribes, even where the model has other values.
- `sbmlsim` fits weighted least squares of the model parameters. PEtab defines the objective as a likelihood whose noise parameters can be estimated too. `Boehm_JProteomeRes2014` estimates the standard deviation of each observable, so the two objectives have different optima. The same holds for the standard deviation `sd_task0_model0_perelson1_V` of `Perelson_Science1996`: it is a parameter of the fit, but the least squares cost does not depend on it, so the fit cannot determine it and its profile is flat. See [What PEtab does not express](petab.md#what-petab-does-not-express).
- PEtab v2 has no `parameterScale` per parameter. In `sbmlsim` the scale is `FitSettings.parameter_scale`, except for a parameter whose lower bound is not positive, which is searched on the linear scale: the encoding of PEtab v2 of `Boehm_JProteomeRes2014` estimates the logarithms of its rates, `log10_k_phos`, with the bounds `-5` and `5`.
- The observable of `Perelson_Science1996` has `log10-normal` noise, which the converter maps to `log-normal`. Since the noise does not enter the least squares fit, the example fits relative residuals (`ResidualType.NORMALIZED`), which suit a viral load over orders of magnitude.
