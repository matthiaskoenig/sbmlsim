# SBML Test Suite

The [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite) is the conformance suite of SBML. Every semantic case is a model, a settings file which says how to simulate it and a CSV with the results a correct simulator produces, and the tags of a case say which parts of SBML it exercises. Running it answers what `sbmlsim`, i.e. libroadrunner, supports, and the results are what a submission to the SBML Test Suite Database is made of.

[**The report of the current run**](testsuite/report/index.html) is generated with the documentation, so it describes the code of the `develop` branch.

## What is run

The semantic cases which are timecourse simulations, in the newest SBML encoding it provides:

- the **stochastic** cases need many replicates and a comparison of distributions, and the **flux balance** cases need a steady state rather than a simulation, so neither is run,
- the other encodings of a case, i.e. the same model as SBML L1V2 through L3V2, test the reader and not the simulation; reading is covered by [sbmlutils](https://github.com/matthiaskoenig/sbmlutils),
- a case is simulated from its `NNNNN-settings.txt`, not from the SED-ML it also ships: the settings carry the start, the duration, the steps, the variables and the tolerances, which is everything the simulation needs.

A case passes when every point of every compared variable is within the tolerances of the case,

```
|c - u| <= abs_tol + rel_tol * |c|
```

with the expected value `c` and the simulated value `u`. The integration itself runs an order of magnitude tighter than the tolerances a case compares with: integrating no more accurately than the comparison demands is how a suite runner passes cases it should not.

A case which passes is simulated again on its loaded model, as every point of a scan after the first of its chunk and every simulation of an experiment after the first are, and each of these simulations must equal the first, bitwise (the repeat check, #305): (1) the same simulation again; (2) a simulation of half the duration, then the simulation of the case, which integrates beyond it, so a model with an event with a delay is generated again to discard the pending events of the shorter one; (3) the simulation with a steady state before it, which integrates without end, compared with that simulation of a fresh model (the steady state for at most the duration of the case; both may fail to reach it, then with the same error). libroadrunner executes simultaneous events of the same priority in a random order drawn with `std::rand` of the C library, whose state belongs to the process, so the runner seeds that generator before every simulation of a case (`seed_event_order`): without it the 13 cases with such events (e.g. 00952, 01588) differ in the last bits between the first and the second simulation, and between processes. A regeneration of a model draws from the generator as well, so in (2) and (3) a case with the test tag `RandomEventExecution` is compared with its expected results and within the tolerances of the case instead: 00952, 00962, 00964, 00965, 00966, 01466, 01590, 01591, 01599, 01605, 01626 and 01627 (01589, 01592 and 01593 are not read). Only 01590 needs it: it has an event with a delay, its model is generated again, and it differs by `7.1e-14`. Over the 1789 timecourse cases of 3.5.0, all 1580 which pass pass the repeat check.

A case which does not pass does so in one of five ways, and the report separates them because they say different things: `tolerance` (it was simulated and the results are wrong), `not_read` (the model could not be loaded, e.g. an algebraic rule), `simulation_error` (the integration failed), `missing_variable` (a compared variable was not produced) and `repeat` (the results are right, and a simulation of the loaded model after the first differs from it or fails).

## Running it

```bash
# fetch the pinned release into the cache, which the tests need
uv run python scripts/testsuite.py download

# run the cases and write the interactive report
uv run python scripts/testsuite.py report --output site/testsuite/report

# the archive which is submitted to the SBML Test Suite Database
uv run python scripts/testsuite.py submission --output dist
```

`--version latest` resolves the newest release of the suite instead of the pinned one. The cases run in one process per core, `--workers` sets the number of processes and `--workers 1` runs them in the process of the script. The cases are cached under `~/.cache/sbmlsim/test-suite/<version>/`, and `SBMLSIM_TEST_SUITE_PATH` points at a directory of cases somewhere else.

## Part of the test suite of sbmlsim

`tests/testsuite/test_semantic.py` is one test per case, so a failure names the case and the cases distribute over the workers of pytest-xdist. They carry the `testsuite` marker and a normal `pytest` deselects them: the cases answer what libroadrunner supports, which a change to `sbmlsim` rarely moves, and they take longer than the rest of the tests together. They run before a release, i.e. the `testsuite` job of the CI is what the release job waits for, and by hand with

```bash
pytest -m testsuite          # the cases must be downloaded
tox r -e testsuite           # downloads them first
```

The suite is not fully green and never will be, so a test cannot demand that every case passes: `tests/data/testsuite_baseline.json` records the outcome of every case which does not pass, and a run is compared with it.

The baseline fails in both directions. A case which passed and now fails is a regression. A case which is in the baseline and passes is a baseline which is out of date, and the run says so rather than hiding the improvement; `scripts/testsuite.py baseline` refreshes it, and the diff of that file is the record of what a change fixed or broke.

The tests skip when the suite was not downloaded: a test run must not depend on the network, and 13 MB is not something a test fetches unasked.

## Which release is run

`SUITE_VERSION` in `sbmlsim.testsuite.cases` pins the release the local test suite and the pull requests run against, so a run is reproducible and a new upstream release cannot turn a green build red on its own. A release of `sbmlsim` resolves the newest release of the suite instead and submits against that, and a scheduled job opens a pull request bumping the pin when upstream is newer, which is where the failures of new cases are seen.

## Submission

`scripts/testsuite.py submission` writes one CSV per case which could be simulated, named after the case, and a `manifest.json` with the versions of `sbmlsim`, libroadrunner and libsbml and the release of the suite. A case which cannot be read or integrated has no results and is left out, which is how a submission says that it is not supported. The release workflow attaches the archive to the GitHub release; uploading it to the SBML Test Suite Database is a deliberate step and is not automated.
