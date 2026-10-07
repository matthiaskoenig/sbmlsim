"""The baseline of a test suite: the cases which do not pass, with their reasons.

The PEtab SciML test suite and the PEtab v2 test suite record the outcome of
their cases in a JSON file, `tests/data/<suite>_baseline.json`:

    {
      "suite_commit": "<commit>",
      "n_cases": 129,
      "n_passed": 128,
      "expected_failures": {
        "sbml/0001": {"status": "tolerance", "reason": "why it does not pass"}
      }
    }

A run is compared with it in both directions: a case which passed and fails is
a regression, a case which is listed and passes is a baseline which is out of
date. `scripts/sciml_testsuite.py` and `scripts/petab_testsuite.py` are the
command lines which write and check it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

#: the reason of a case which is new in the baseline. The tests reject it, so
#: a case is not listed until somebody wrote down why it does not pass
MISSING_REASON = "reason missing"

#: the status of a case which passes
PASS = "pass"


class Outcome(Protocol):
    """The outcome of a case, see the `CaseResult` of a suite."""

    @property
    def key(self) -> str:
        """Get the key of the case in the baseline."""
        ...

    @property
    def passed(self) -> bool:
        """Check whether the case passes."""
        ...

    @property
    def status(self) -> Any:
        """Get the status, an enum whose value is the name in the baseline."""
        ...


def unexpected_outcomes(
    results: Sequence[Outcome], baseline: dict[str, Any]
) -> list[str]:
    """Get the cases which do not have the outcome the baseline records.

    Args:
        results: the results of a run.
        baseline: the content of the baseline.

    Returns:
        One line per case, `<key>: '<expected>' -> '<observed>'`, and one per
        case of the baseline which was not run.
    """
    expected_failures = baseline["expected_failures"]
    lines: list[str] = []
    for result in results:
        recorded = expected_failures.get(result.key)
        expected = PASS if recorded is None else recorded["status"]
        if result.status.value != expected:
            lines.append(f"{result.key}: '{expected}' -> '{result.status.value}'")
    keys = {result.key for result in results}
    for key, recorded in expected_failures.items():
        if key not in keys:
            lines.append(f"{key}: '{recorded['status']}' -> not run")
    return lines


def write_baseline(results: Sequence[Outcome], commit: str, path: Path) -> list[str]:
    """Write the cases which do not pass, keeping the reasons which are recorded.

    A reason is kept while the case fails with the recorded status. A case
    whose status changed gets `MISSING_REASON`, the recorded reason describes
    the old status.

    Args:
        results: the results of a run.
        commit: the commit of the suite which was run.
        path: the baseline.

    Returns:
        A note per case whose status changed, `<key>: '<old>' -> '<new>'`.
    """
    recorded: dict[str, dict[str, str]] = {}
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))["expected_failures"]
    notes: list[str] = []
    failures: dict[str, dict[str, str]] = {}
    for result in results:
        if result.passed:
            continue
        reason = MISSING_REASON
        previous = recorded.get(result.key)
        if previous is not None:
            if previous["status"] == result.status.value:
                reason = previous["reason"]
            else:
                notes.append(
                    f"{result.key}: the status changed '{previous['status']}' -> "
                    f"'{result.status.value}', the recorded reason was dropped"
                )
        failures[result.key] = {"status": result.status.value, "reason": reason}
    baseline = {
        "suite_commit": commit,
        "n_cases": len(results),
        "n_passed": len(results) - len(failures),
        "expected_failures": failures,
    }
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return notes
