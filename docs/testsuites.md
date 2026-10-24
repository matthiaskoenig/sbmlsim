# Test suites

`sbmlsim` is checked against the conformance suites of the standards it implements and against a collection of published problems. Each one answers a different question:

| suite | what it checks | run with |
| --- | --- | --- |
| [SBML Test Suite](testsuite.md) | how SBML models are simulated, i.e. what libroadrunner supports | `pytest -m testsuite` |
| [PEtab Test Suite](petab_testsuite.md) | the meaning of PEtab problems: initialization, conditions, observables, noise and priors | `pytest -m petab_testsuite tests/fit` |
| [PEtab Benchmark Problems](petab_benchmark.md) | reading and simulating published PEtab problems, compared with the collection and with AMICI | `pytest -m petab_benchmark tests/fit` |
| [PEtab SciML Test Suite](sciml_testsuite.md) | neural networks and hybrid problems of PEtab SciML | `pytest -m sciml_testsuite tests/sciml` |

Each suite is pinned to a release or a commit, so a run gives the same result tomorrow as today. A normal `pytest` skips all of them, because they need a download first. The `tox` environments `testsuite`, `petab`, `benchmark` and `sciml` download the cases and then run them.

Not every case passes. The ones that do not are listed with their outcome, and for the PEtab suites with the reason, in a baseline file under `tests/data/`: `testsuite_baseline.json`, `petab_baseline.json`, `benchmark_baseline.json` and `sciml_baseline.json`. A run is compared with its baseline in both directions. A case that passed before and fails now is a regression. A case that is listed and now passes means the baseline is out of date, and the run fails until the baseline is refreshed.

The three test suites run before every release. The benchmark collection runs on demand, because its largest problems take minutes.
