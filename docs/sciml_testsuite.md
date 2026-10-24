# PEtab SciML Test Suite

The [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) is the conformance suite of [PEtab SciML](petab_sciml.md), the PEtab extension for hybrid problems that combine a model with neural networks. It checks two things: that `sbmlsim.sciml` evaluates a network the way PyTorch does, and that a hybrid problem gives the same log-likelihood, simulations and gradient as the reference tools.

Of the 96 cases of the pinned commit, 92 pass. The four that do not run into known gaps (see [below](#cases-that-do-not-pass)).

## Run it

The suite needs the `sciml` extra (`pip install sbmlsim[sciml]`, included in `dev`).

```bash
uv run python scripts/sciml_testsuite.py download    # download the pinned commit
uv run python scripts/sciml_testsuite.py run         # run every case, compare with the baseline
uv run python scripts/sciml_testsuite.py baseline    # refresh the baseline after a change
pytest -m sciml_testsuite tests/sciml                # one test per case, after the download
tox r -e sciml                                       # download, then run the tests
```

The suite has no releases, so it is pinned by a commit (`SCIML_SUITE_COMMIT` in `sbmlsim.sciml.testsuite`). The cases are cached in `~/.cache/sbmlsim/petab-sciml-testsuite/<commit>/`. To use a checkout somewhere else, point `SBMLSIM_SCIML_SUITE_PATH` at its `test_cases` directory.

`tests/sciml/test_testsuite.py` has one test per case and group (`test_model_import`, `test_initialization`, `test_problem_import` and `test_problem_round_trip`), so a failure names the case. A normal `pytest` skips them because they need the download. They run before every release in the `sciml` job of the CI.

## What is run

The suite has three groups of cases. Each case is a directory named by its number, with a `solutions.yaml` that holds the reference values and their tolerances. `sbmlsim.sciml.testsuite` runs all three:

| group | case class | compared |
| --- | --- | --- |
| `ml_model_import` | `ModelImportCase` | the output of a forward pass of a network |
| `initialization` | `InitializationCase` | the nominal values of the arrays of a network |
| `sciml_problem_import` | `ProblemImportCase` | the log-likelihood, the simulations at the measurements and the gradient |

A problem is simulated on a fixed grid with integrator tolerances of `1e-13`. The reference values were simulated with `1e-12`, and any simulation error enters the gradient amplified by the sensitivities of the log-likelihood. The gradient is a central difference of five points, like the reference values.

Every problem that is read is also written back with `to_petab` and read again. The round trip must give the same parameters, hybridizations, data and log-likelihood.

The outcome of a case that does not pass is `tolerance` (the values are wrong), `shape` (the arrays do not have the shape of the reference), `unsupported` (the case needs something `sbmlsim` does not support, see [What is not supported](petab_sciml.md#what-is-not-supported)) or `error`.

## Cases that do not pass

`tests/data/sciml_baseline.json` lists the cases that do not pass, each with its outcome and reason; a case without a reason fails the tests. As for the [SBML Test Suite](testsuite.md), the baseline is checked in both directions: a case that breaks is a regression, and a listed case that passes means the baseline is out of date.

| case | outcome | reason |
| --- | --- | --- |
| `ml_model_import/020` | `tolerance` | `AlphaDropout`: the reference values are the mean of many forward passes in training mode, while `sbmlsim` evaluates dropout in evaluation mode, where it is the identity (gap `sciml-training-mode`) |
| `sciml_problem_import/032` to `034` | `unsupported` | priors on the parameters: the cases state a log-posterior, and priors on network parameters are not supported yet (gap `sciml-priors`, issue #190) |
