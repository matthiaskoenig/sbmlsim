"""Running the semantic cases of the SBML Test Suite.

The [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite) is the
conformance suite of SBML: every semantic case is a model with a settings file
which says how to simulate it and a CSV of the results a correct simulator
produces. Running it says which parts of SBML the simulation of `sbmlsim`, i.e.
libroadrunner, gets right, and the results are what a submission to the SBML
Test Suite Database is made of.

This package is the library half of that: `SemanticCase` is a case on disk,
`SemanticSuite` is a directory of them with the download and the cache of a
release, `run_case` simulates one and `compare_case` decides whether its
results are within the tolerances of the case. Writing the report and the
submission archive is `scripts/testsuite_report.py`.

Only the semantic cases which are timecourse simulations are run: the
stochastic and the flux balance cases need a different kind of simulation, and
the encodings of a case in the other SBML levels and versions test reading,
which is covered by `sbmlutils`.
"""

from sbmlsim.testsuite.cases import SemanticCase, SemanticSuite
from sbmlsim.testsuite.comparison import CaseComparison, compare_case
from sbmlsim.testsuite.runner import CaseResult, CaseStatus, run_case, run_suite

__all__ = [
    "CaseComparison",
    "CaseResult",
    "CaseStatus",
    "SemanticCase",
    "SemanticSuite",
    "compare_case",
    "run_case",
    "run_suite",
]
