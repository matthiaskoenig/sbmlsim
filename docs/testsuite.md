# SBML Test Suite

The [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite) is the conformance suite of SBML. Each semantic case is a small model, a settings file that says how to simulate it, and a CSV file with the results a correct simulator produces. The tags of a case say which parts of SBML it uses. Running the suite tells you which parts of SBML `sbmlsim`, and with it libroadrunner, simulates correctly.

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

The cases are cached in `~/.cache/sbmlsim/test-suite/<commit>/` (or under `XDG_CACHE_HOME` when it is set). If you keep the cases somewhere else, for example on a machine without network access, point `SBMLSIM_TEST_SUITE_PATH` at that directory.

## What is run

The cases come from the branch `develop` of the suite, pinned to a commit (`SUITE_COMMIT` in `sbmlsim.testsuite.cases`), and not from a release: the releases of the suite are rare, and a case fixed upstream would wait for the next one. Case 01511, whose output grid stepped across the window of its trigger in release 3.5.0, was fixed in [d6dcd2c](https://github.com/sbmlteam/sbml-test-suite/commit/d6dcd2cc6408d5eb881638ac6ce7a05fd7d2e4d1) and passes since. The pinned commit has 1789 timecourse cases, and 1581 of them pass. The 208 that do not cannot be read.

Only the timecourse cases run, each in the newest SBML encoding it ships:

- The stochastic cases need many replicates and a comparison of distributions, and the flux balance cases are no timecourses (they have no duration). Neither is run.
- The older encodings of a case (SBML L1V2 up to L3V2) contain the same model. They test the SBML reader, which [sbmlutils](https://github.com/matthiaskoenig/sbmlutils) covers, so only the newest one is simulated.
- A case is simulated from its `NNNNN-settings.txt` file, which has the start, the duration, the number of steps, the variables and the tolerances. The SED-ML file that also ships with a case is not used.

A case passes when every point of every compared variable is within the tolerances of the case:

```
|c - u| <= abs_tol + rel_tol * |c|
```

Here `c` is the expected value and `u` the simulated one. The integrator runs with tolerances of `1e-12`, an order of magnitude tighter than the tightest tolerance a case compares with. A simulator that integrates only as accurately as the comparison demands can pass cases by luck.

## The outcome of a case

A case that does not pass gets one of five outcomes. The report shows them separately, because they mean different things:

| outcome | meaning |
| --- | --- |
| `tolerance` | the case was simulated and the results are wrong |
| `not_read` | the model could not be loaded, e.g. because it has an algebraic rule |
| `simulation_error` | the model was loaded and the integration failed |
| `missing_variable` | the simulation ran but did not produce a variable the case compares |
| `repeat` | the results are right, but simulating the same loaded model again gives different results |

`repeat` comes from an extra check. `sbmlsim` simulates a loaded model many times, for example once for every point of a scan, so each simulation must be independent of the ones before it. A case that passes is therefore simulated again on its loaded model, and each of these runs must match the first one exactly, bit for bit:

1. the same simulation again,
2. a simulation of half the duration, followed by the full simulation,
3. the simulation with a steady state before it, compared with the same on a freshly loaded model.

libroadrunner runs simultaneous events of the same priority in a random order. The runner seeds the random generator of the model before every simulation, so these cases are reproducible. Regenerating a model, which checks 2 and 3 can do, also draws from that generator, so for the cases tagged `RandomEventExecution`, checks 2 and 3 compare with the expected results within the tolerances of the case instead of bit for bit. All 1581 cases that pass also pass this check.

## Part of the tests of sbmlsim

`tests/testsuite/test_semantic.py` has one test per case, so a failure names the case. The tests carry the marker `testsuite`, and a normal `pytest` skips them: they mostly test libroadrunner, which a change to `sbmlsim` rarely affects, and they take longer than all other tests together. Run them with

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

This writes a zip archive with one CSV file per case that could be simulated, named after the case, and a `manifest.json` with the versions of `sbmlsim`, libroadrunner and libsbml and the commit of the suite. A case that cannot be read or integrated has no CSV file, which is how a submission says it is not supported. The release workflow builds this archive against the pinned commit and attaches it to the GitHub release. Uploading it to the database is done by hand.
