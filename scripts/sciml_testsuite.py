"""Download and run the PEtab SciML test suite.

The cases of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite)
say which networks and which hybrid problems `sbmlsim` supports. This script
is the command line around `sbmlsim.sciml.testsuite`:

```bash
# fetch the pinned commit into the cache, which the tests need
uv run python scripts/sciml_testsuite.py download

# run the cases and report the outcome, fails when a case does not have the
# outcome of the baseline
uv run python scripts/sciml_testsuite.py run

# refresh the expected outcomes after a change which fixes or breaks cases
uv run python scripts/sciml_testsuite.py baseline
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.sciml.testsuite import (
    SCIML_SUITE_COMMIT,
    CaseResult,
    CaseStatus,
    SciMLSuite,
)

#: the expected outcomes the tests compare a run with
BASELINE_PATH = Path(__file__).parent.parent / "tests" / "data" / "sciml_baseline.json"

#: the reason of a case which is new in the baseline. The tests reject it, so
#: a case is not listed until somebody wrote down why it does not pass
MISSING_REASON = "reason missing"


def run(suite: SciMLSuite) -> list[CaseResult]:
    """Run the cases of the suite and report the outcome."""
    console.print(
        f"Running the PEtab SciML test suite '{suite.commit}' from {suite.path}"
    )
    results = suite.run()
    counts = Counter(r.status for r in results)
    console.print(f"[bold]{counts[CaseStatus.PASS]}/{len(results)} cases pass[/bold]")
    for result in results:
        if not result.passed:
            console.print(f"  {result.key}  {result.status.value}  {result.message}")
    return results


def unexpected_outcomes(results: list[CaseResult], baseline: dict) -> list[str]:
    """Get the cases which do not have the outcome the baseline records.

    Args:
        results: the results of a run.
        baseline: the content of the baseline.

    Returns:
        One line per case, `<key>: '<expected>' -> '<observed>'`.
    """
    expected_failures = baseline["expected_failures"]
    lines: list[str] = []
    for result in results:
        recorded = expected_failures.get(result.key)
        expected = CaseStatus.PASS.value if recorded is None else recorded["status"]
        if result.status.value != expected:
            lines.append(f"{result.key}: '{expected}' -> '{result.status.value}'")
    return lines


def write_baseline(results: list[CaseResult], suite: SciMLSuite, path: Path) -> None:
    """Write the cases which do not pass, keeping the reasons which are recorded.

    A reason is kept while the case fails with the recorded status. A case
    whose status changed gets `MISSING_REASON`, the recorded reason describes
    the old status, and is reported.

    Args:
        results: the results of a run.
        suite: the suite which was run.
        path: the baseline.
    """
    recorded: dict[str, dict[str, str]] = {}
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))["expected_failures"]
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
                console.print(
                    f"  [yellow]{result.key}: the status changed "
                    f"'{previous['status']}' -> '{result.status.value}', the "
                    f"recorded reason was dropped[/yellow]"
                )
        failures[result.key] = {"status": result.status.value, "reason": reason}
    baseline = {
        "suite_commit": suite.commit,
        "n_cases": len(results),
        "n_passed": len(results) - len(failures),
        "expected_failures": failures,
    }
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    console.print(f"Baseline: {path}")
    for key, expected in failures.items():
        if expected["reason"] == MISSING_REASON:
            console.print(f"  [red]{key}: write the reason into the baseline[/red]")


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: command line arguments, `sys.argv` by default.

    Returns:
        The exit code: `1` when `run` finds a case without the outcome of the
        baseline, `0` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "run", "baseline"])
    parser.add_argument("--commit", default=SCIML_SUITE_COMMIT)
    args = parser.parse_args(argv)

    log.enable_rich_logging()
    suite = SciMLSuite.load(args.commit)
    if args.command == "download":
        console.print(f"PEtab SciML test suite '{suite.commit}': {suite.path}")
        return 0

    results = run(suite)
    if args.command == "baseline":
        write_baseline(results, suite, BASELINE_PATH)
        return 0
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    unexpected = unexpected_outcomes(results, baseline)
    if unexpected:
        console.print(
            f"[red]{len(unexpected)} cases do not have the outcome of the "
            f"baseline {BASELINE_PATH}[/red]"
        )
        for line in unexpected:
            console.print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
