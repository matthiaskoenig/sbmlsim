# SBML Test Suite

The [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite) is the conformance suite of SBML. Each semantic case is a small model, a settings file that says how to simulate it, and a CSV file with the results a correct simulator produces. The tags of a case say which parts of SBML it uses. Running the suite tells you which parts of SBML `sbmlsim` simulates correctly with the engine it runs them on.

[The report of the current run](testsuite/report/index.html) is built with the documentation from the `develop` branch; you can filter it by outcome and by tag.

## Run it

```bash
# download the pinned commit into the cache
uv run python scripts/testsuite.py download

# run every case and print how many pass
uv run python scripts/testsuite.py run

# run every case and write the interactive report
uv run python scripts/testsuite.py report --output site/testsuite/report
```

The cases run in parallel, one process per core. `--workers` sets the number of processes, and `--workers 1` runs everything in one process. `--commit latest` runs the head of the branch `develop` of the suite instead of the pinned commit.

`--engine` picks the engine. The cases run on sbmlsim-core, the default engine of the package, and roadrunner is the reference they are measured against:

```bash
uv run python scripts/testsuite.py run --engine roadrunner
```

The cases are cached in `~/.cache/sbmlsim/test-suite/<commit>/` (or under `XDG_CACHE_HOME` when it is set). If you keep the cases somewhere else, for example on a machine without network access, point `SBMLSIM_TEST_SUITE_PATH` at that directory.

## What is run

The cases come from the branch `develop` of the suite, pinned to a commit (`SUITE_COMMIT` in `sbmlsim.testsuite.cases`), and not from a release: the releases of the suite are rare, and a case fixed upstream would wait for the next one. Case 01511, whose output grid stepped across the window of its trigger in release 3.5.0, was fixed in [d6dcd2c](https://github.com/sbmlteam/sbml-test-suite/commit/d6dcd2cc6408d5eb881638ac6ce7a05fd7d2e4d1) and passes since. The pinned commit has 1789 timecourse cases, and **1754 of them pass on sbmlsim-core**. The 35 that do not are the 35 cases of a fast reaction, which the compiler of the engine does not support: 33 are refused when the model is read and 2 when it is converted (see [cases which are refused or cannot be integrated](#cases-which-are-refused-or-cannot-be-integrated)). There is no case the engine reads and then simulates outside the tolerances of the suite.

How the engines compare on the same 1789 cases:

| engine | pass | not read | integration failed | not converted |
|---|---|---|---|---|
| core (the default) | **1754** | 33 | 0 | 2 |
| roadrunner (the reference) | 1576 | 206 | 5 | 2 |

The 35 cases sbmlsim-core does not pass are **a strict subset of the 213 roadrunner does not**: there is no case which passes on roadrunner and fails on sbmlsim-core. The 178 cases sbmlsim-core passes and roadrunner does not are

| why roadrunner refuses them | cases |
|---|---|
| `Unable to support algebraic rules` (sbmlsim-core simulates them with SUNDIALS IDA since sbmlsim-core 0.10.0) | 123 |
| `Unable to support delay differential equations` (an integrator of its own for the delays since sbmlsim-core 0.11.0) | 50 |
| `CV_CONV_FAILURE` on a `stoichiometryMath` (see [roadrunner and the stoichiometry math](#roadrunner-and-the-stoichiometry-math)) | 5 |

Only the timecourse cases run, each in SBML level 3, which is the only level `sbmlsim` simulates:

- The stochastic cases need many replicates and a comparison of distributions, and the flux balance cases are no timecourses (they have no duration). Neither is run.
- The older encodings of a case (SBML L1V2 up to L3V2) contain the same model. They test the SBML reader, which [sbmlutils](https://github.com/matthiaskoenig/sbmlutils) covers, so only one is simulated: the newest encoding the case ships. When that is level 3 (`l3v2`, or `l3v1` for 46 cases) it runs as shipped. A level 2 encoding is converted to level 3 version 2 with libsbml.
- A conversion is written once into the directory of the case as `NNNNN-sbml-l3v2-converted.xml`, next to a sidecar `NNNNN-sbml-l3v2-converted.json` that names the version of libsbml which wrote it, the format of the conversion (`CONVERSION_FORMAT` in `sbmlsim.testsuite.cases`, raised whenever the conversion changes what it produces) and the encoding it was converted from. A pair is reused only when its libsbml and its format are the running ones, else it is written again, so a cache of an older conversion is never served. A level 3 case has no converted file.
- A case is simulated from its `NNNNN-settings.txt` file, which has the start, the duration, the number of steps, the variables and the tolerances. The SED-ML file that also ships with a case is not used.

A case passes when every point of every compared variable is within the tolerances of the case:

```
|c - u| <= abs_tol + rel_tol * |c|
```

Here `c` is the expected value and `u` the simulated one. The integrator runs with tolerances of `1e-12`, an order of magnitude tighter than the tightest tolerance a case compares with. A simulator that integrates only as accurately as the comparison demands can pass cases by luck.

## The outcome of a case

A case that does not pass gets one of six outcomes. The report shows them separately, because they mean different things:

| outcome | meaning |
| --- | --- |
| `tolerance` | the case was simulated and the results are wrong |
| `not_read` | the model could not be loaded, e.g. because it has an algebraic rule |
| `not_converted` | the newest encoding of the case is below level 3 and libsbml could not convert it to level 3 version 2 without dropping something, so `sbmlsim` refuses it instead of simulating a different model |
| `simulation_error` | the model was loaded and the integration failed |
| `missing_variable` | the simulation ran but did not produce a variable the case compares |
| `repeat` | the results are right, but simulating the same loaded model again gives different results |

`repeat` comes from an extra check. `sbmlsim` simulates a loaded model many times, for example once for every point of a scan, so each simulation must be independent of the ones before it. A case that passes is therefore simulated again on its loaded model, and each of these runs must match the first one exactly, bit for bit:

1. the same simulation again,
2. a simulation of half the duration, followed by the full simulation,
3. the simulation with a steady state before it, compared with the same on a freshly loaded model.

libroadrunner runs simultaneous events of the same priority in a random order. The runner seeds the random generator of the model before every simulation, so these cases are reproducible; another engine orders them itself and draws nothing, so the seeding does nothing there. Regenerating a model, which checks 2 and 3 can do, also draws from that generator, so for the cases tagged `RandomEventExecution`, checks 2 and 3 compare with the expected results within the tolerances of the case instead of bit for bit. Every case that passes also passes this check, on either engine.

## Cases which are refused or cannot be integrated

`not_converted` is a refusal and no failure of the simulation. libsbml records what it cannot convert in its error log and still returns success, so `sbmlsim` refuses a conversion for which libsbml logged an error. The two cases are 01397 and 01572, whose only encodings are SBML level 2 and whose reaction is a fast reaction: level 3 version 2 has no `fast` attribute, so the converted model would simulate the reaction as a slow one and give wrong results at the start. The other 33 of the 35 cases with a fast reaction have `l3v1` as their newest encoding, which is level 3 and so is not converted and keeps the attribute.

### roadrunner and the stoichiometry math

The cases 01435, 01437, 01439, 01441 and 01443 end as `simulation_error`. They are level 2 models with a `stoichiometryMath`, which level 3 does not have. libsbml converts it into an assignment rule of the species reference, marks the reference as non-constant and leaves its stoichiometry unset. That is a correct level 3 translation, and it logs nothing, so the conversion is not refused. libroadrunner cannot integrate such a model: CVODE stops at the start of the integration with `CV_CONV_FAILURE`. These five cases passed before `sbmlsim` converted every model to level 3. This is a limitation of libroadrunner and `sbmlsim` does not work around it; **sbmlsim-core integrates all five**, so they pass on the default engine and this section is about the reference alone.

## Part of the tests of sbmlsim

`tests/testsuite/test_semantic.py` has one test per case, so a failure names the case. The tests carry the marker `testsuite`, and a normal `pytest` skips them: they mostly test the engine, which a change to `sbmlsim` rarely affects, and they take longer than all other tests together. Run them with

```bash
pytest -m testsuite          # the cases must be downloaded first
tox r -e testsuite           # downloads them, then runs the tests
```

Without a download the tests are skipped, so a normal test run never needs the network.

`tests/data/testsuite_baseline.json` lists every case that does not pass, with its outcome. The tests fail when a listed case passes and when an unlisted case fails. After a change that fixes or breaks cases, refresh the list with

```bash
uv run python scripts/testsuite.py baseline
```

and the diff of the file shows what changed.

## Which commit is run

The tests, the documentation and the release run against the pinned commit, so a change upstream cannot break the build by itself. A scheduled workflow (`.github/workflows/testsuite-pin.yml`) checks every week whether the semantic cases of the branch `develop` changed since the pinned commit and, if they did, opens a pull request that bumps the pin to the head of the branch and refreshes the baseline. A commit that changes only the test runner of the suite is no reason for a pull request. That pull request is where the results of new or changed cases show up first.

The release workflow runs the tests of the suite before every release of `sbmlsim`.

## Submission

The results of a run can be submitted to the SBML Test Suite Database:

```bash
uv run python scripts/testsuite.py submission --output dist
```

This writes a zip archive with one CSV file per case that could be simulated, named after the case, and a `manifest.json` with the engine, the versions of `sbmlsim`, libroadrunner and libsbml and the commit of the suite. A case that cannot be read or integrated has no CSV file, which is how a submission says it is not supported. The release workflow builds this archive against the pinned commit and attaches it to the GitHub release. Uploading it to the database is done by hand.
