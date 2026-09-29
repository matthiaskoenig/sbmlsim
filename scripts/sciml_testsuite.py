"""Download and run the PEtab SciML test suite.

The cases of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite)
say which networks and which hybrid problems `sbmlsim` supports. This script
is the command line around `sbmlsim.sciml.testsuite`:

```bash
# fetch the pinned commit into the cache, which the tests need
uv run python scripts/sciml_testsuite.py download

# run the cases and report the outcome
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


def write_baseline(results: list[CaseResult], suite: SciMLSuite, path: Path) -> None:
    """Write the cases which do not pass, keeping the reasons which are recorded.

    Args:
        results: the results of a run.
        suite: the suite which was run.
        path: the baseline.
    """
    reasons: dict[str, str] = {}
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        reasons = {
            key: expected["reason"]
            for key, expected in recorded["expected_failures"].items()
        }
    failures = {
        r.key: {"status": r.status.value, "reason": reasons.get(r.key, MISSING_REASON)}
        for r in results
        if not r.passed
    }
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
    """Run the command line."""
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


if __name__ == "__main__":
    sys.exit(main())
