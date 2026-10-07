# Test suites

`sbmlsim` is measured against the conformance suites of the standards it implements and against a collection of published problems. Each answers a different question:

| suite | what it checks | in sbmlsim |
| --- | --- | --- |
| [SBML Test Suite](testsuite.md) | the simulation of SBML models, i.e. what libroadrunner supports | `pytest -m testsuite`, a report with the documentation and a submission with every release |
| [PEtab Test Suite](petab_testsuite.md) | the semantics of PEtab problems: initialization, conditions, observables, noise and priors | `pytest -m petab_testsuite` |
| [PEtab Benchmark Problems](petab_benchmark.md) | the reading and the simulation of published PEtab problems, compared with the collection and with AMICI | `pytest -m petab_benchmark`, `scripts/petab_benchmark.py` |
| [PEtab SciML Test Suite](sciml_testsuite.md) | the neural networks and hybrid problems of PEtab SciML | `pytest -m sciml_testsuite` |

The SBML Test Suite, the PEtab Test Suite, the PEtab benchmark collection and the PEtab SciML Test Suite are pinned to a release and a commit, so a run is reproducible, and the cases which do not pass are recorded in a baseline with their reason (`tests/data/testsuite_baseline.json`, `tests/data/petab_baseline.json`, `tests/data/benchmark_baseline.json`, `tests/data/sciml_baseline.json`). A normal `pytest` deselects their cases, because they need the download of the suite; the test suites run before every release, the benchmark collection, whose largest problems take minutes, on demand.
