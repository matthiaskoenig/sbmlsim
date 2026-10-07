# PEtab SciML Test Suite

The [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) is the conformance suite of [PEtab SciML](petab_sciml.md), the extension of PEtab v2 for hybrid problems of a model and neural networks. Its cases are the reference which `sbmlsim.sciml` and the reader of `sbmlsim.fit.petab_v2` are measured against: a network is read and evaluated like PyTorch evaluates it, and a hybrid problem has the log-likelihood, the simulations and the gradient the reference tools compute.

## What is run

The suite has three groups of cases, every case is a directory named by its number with a `solutions.yaml` which holds the reference values and their tolerances. `sbmlsim.sciml.testsuite` reads a case of every group and compares it:

| group | case | compared |
| --- | --- | --- |
| `ml_model_import` | `ModelImportCase` | the outputs of the forward pass of a network |
| `initialization` | `InitializationCase` | the nominal values of the arrays of a network |
| `sciml_problem_import` | `ProblemImportCase` | the log-likelihood, the simulations at the measurements and the gradient |

A problem is simulated on a fixed grid with the tolerances `1e-13`, because the reference values are simulated with `1e-12` and the error of a simulation enters the gradient multiplied by the sensitivities of the log-likelihood. The gradient is the central difference of five points, which is what the reference values are. Every problem which is read is also written with `to_petab` and read back, and the round trip must give the same parameters, hybridizations, data and log-likelihood.

A case which does not pass is a `tolerance` (it was evaluated and the values are wrong), `shape` (the arrays do not have the shape of the reference), `unsupported` (it runs into a gap of `sbmlsim`, see [What is not supported](petab_sciml.md#what-is-not-supported)) or `error`.

## Running it

```bash
# fetch the pinned commit into the cache, which the tests need
uv run python scripts/sciml_testsuite.py download

# run the cases and report the outcome
uv run python scripts/sciml_testsuite.py run

# refresh the expected outcomes after a change which fixes or breaks cases
uv run python scripts/sciml_testsuite.py baseline
```

The suite has no releases, so `SCIML_SUITE_COMMIT` in `sbmlsim.sciml.testsuite` pins it by a commit. The cases are cached under `sbmlsim/petab-sciml-testsuite/<commit>` in the user cache, and `SBMLSIM_SCIML_SUITE_PATH` points at the directory `test_cases` of a checkout somewhere else.

## Part of the test suite of sbmlsim

`tests/sciml/test_testsuite.py` is one test per case and group (`test_model_import`, `test_initialization`, `test_problem_import`, `test_problem_round_trip`), so a failure names the case. They carry the `sciml_testsuite` marker and a normal `pytest` deselects them, because they need the download of the suite. They run before a release, in the `sciml` job of the CI, and by hand with

```bash
pytest -m sciml_testsuite tests/sciml   # the cases must be downloaded
tox r -e sciml                          # downloads them first
```

`tests/data/sciml_baseline.json` records the cases which do not pass, each with its outcome and its reason, and a run is compared with it. As for the [SBML Test Suite](testsuite.md), the baseline fails in both directions: a case which passed and now fails is a regression, and a case which is in the baseline and passes is a baseline which is out of date. A case is not listed without a reason, the tests reject it.
