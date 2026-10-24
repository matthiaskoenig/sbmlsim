# PEtab Benchmark Problems

The [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab) is a set of parameter estimation problems of published systems biology models. Each problem has its model, its data and the simulations at the nominal parameters (`simulations.tsv`). Other people wrote these problems, so they test the [PEtab](petab.md) reader of `sbmlsim` on real files, and the simulations of the collection are the reference to compare with.

The collection is in PEtab 1.0. `sbmlsim` converts every problem to PEtab v2 before reading it. Of the 35 problems of the pinned commit, all are converted and read, and 28 agree with the collection and with AMICI. The other seven are explained [below](#problems-that-do-not-pass).

## Running the collection

```bash
uv run python scripts/petab_benchmark.py download                                     # download, convert and cache
uv run python scripts/petab_benchmark.py run --processes 8 --output results/benchmark  # run every problem
uv run python scripts/petab_benchmark.py report --output results/benchmark             # write benchmark.md
uv run python scripts/petab_benchmark.py baseline --processes 8                       # refresh the baseline
uv run pytest -m petab_benchmark tests/fit                                             # one test per problem
tox r -e benchmark                                                                     # download, then run the tests
```

`--problems` runs only the named problems. Every problem gets at most 600 seconds.

`download` fetches the pinned commit into `~/.cache/sbmlsim/petab-benchmark/<commit>/` (`SBMLSIM_BENCHMARK_PATH` points elsewhere), converts every problem with `petab.v2.petab1to2` and fetches the log-likelihoods that AMICI reports for the problems. A run then reads each converted problem, simulates it once at the nominal values of its parameter table and compares:

| value | reference | tolerance |
| --- | --- | --- |
| the simulation at every measurement | `simulations.tsv` of the collection | `1e-3` absolute and relative |
| the log-likelihood | the `llh` of AMICI, where AMICI has one and no observable is on the `log10` scale (the conversion changes its density) | `1e-3` absolute, `1e-6` relative |

Simulations are matched to the collection row by row when the observables, conditions and times agree, also when ids were renamed after the collection wrote its simulations. Otherwise they are matched by observable, conditions and time. The outcome of a problem is `pass`, `tolerance`, `conversion` (`petab1to2` fails) or `error`, and the result records how long reading, initializing, simulating and the log-likelihood took.

The code is `sbmlsim.fit.petab_v2.benchmark`. The tests are marked `petab_benchmark` and run only on demand, because the largest problems take minutes.

## Problems that do not pass

`tests/data/benchmark_baseline.json` lists the seven problems that do not pass, each with the reason. The tests and `run` compare with it in both directions.

In five of them the simulations of the collection are wrong, while `sbmlsim` agrees with AMICI or with plain roadrunner:

- `Crauste_CellSystems2017`, `Fujita_SciSignal2010` and `Sneyd_PNAS2002`: the simulations of the collection differ, while the log-likelihood agrees with AMICI.
- `Zheng_PNAS2012`: the simulations of the collection are its measurements. The log-likelihood agrees with AMICI.
- `Perelson_Science1996`: the simulations of the collection are not at the nominal parameters. roadrunner without `sbmlsim` gives the same values as `sbmlsim`.

The other two:

- `Chen_MSB2009`: a few simulations differ by up to 0.28 % and the log-likelihood differs from AMICI. Tighter tolerances do not change it.
- `Froehlich_CellSystems2018`: 9169 experiments, each with a pre-equilibration, take longer than the 600 seconds a problem gets.

## Fitting a problem of the collection

`examples/petab/benchmark.py` fits two problems of the collection, which are copied next to it:

- `Perelson_Science1996`: the viral dynamics of HIV-1 after the start of a protease inhibitor, with one observable that is a species of the model.
- `Boehm_JProteomeRes2014`: the dimerization of STAT5A and STAT5B, with three observables that are formulas over several species.

```bash
python -m examples.petab.benchmark
python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014
python -m examples.petab.benchmark --runs=8 --no-identifiability
python -m examples.petab.benchmark --runs=8 --cores=4
```

The example converts the problem with `petab1to2`, reads it, prints what PEtab cannot express about it, fits it, analyses the [identifiability](identifiability.md) of the fitted parameters and writes the report of the fit into `results/<problem>`. For `Perelson_Science1996` the clearance rate `c` of the virions is identifiable, while the loss rate `delta` of the infected cells is not identifiable towards zero. `--cores` sets the number of worker processes; the default of one fits in the calling process.

A few things to know when you fit a problem of the collection with `sbmlsim`:

- An observable that is a formula, such as `(100 * pApB + 200 * pApA * specC17) / (...)` in `Boehm_JProteomeRes2014`, is read as an `ObservableModel` and evaluated on the simulation at the measurements. The model is not changed.
- Parameters that are not estimated are set to the nominal values of the parameter table, as PEtab prescribes, even where the model has other values.
- `sbmlsim` fits weighted least squares of the model parameters. PEtab defines the objective as a likelihood whose noise parameters can be estimated too. `Boehm_JProteomeRes2014` estimates the standard deviation of each observable, so the two objectives have different optima. The same holds for the standard deviation `sd_task0_model0_perelson1_V` of `Perelson_Science1996`: it is a parameter of the fit, but the least squares cost does not depend on it, so the fit cannot determine it and its profile is flat. See [What PEtab does not express](petab.md#what-petab-does-not-express).
- PEtab 1.0 has a `parameterScale` per parameter, which v2 removed. In `sbmlsim` the scale is `FitSettings.parameter_scale`.
- The observable of `Perelson_Science1996` has `log10-normal` noise, which the converter maps to `log-normal`. Since the noise does not enter the least squares fit, the example fits relative residuals (`ResidualType.NORMALIZED`), which suit a viral load over orders of magnitude.
