"""Download and run the PEtab v2 test suite.

The cases of the [PEtab test suite](https://github.com/PEtab-dev/petab_test_suite)
say which parts of PEtab v2 `sbmlsim` supports. This script is the command line
around `sbmlsim.fit.petab_v2.testsuite`:

```bash
# fetch the pinned commit into the cache, which the tests need
uv run python scripts/petab_testsuite.py download

# run the cases and report the outcome, fails when a case does not have the
# outcome of the baseline
uv run python scripts/petab_testsuite.py run

# refresh the expected outcomes after a change which fixes or breaks cases
uv run python scripts/petab_testsuite.py baseline
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
from sbmlsim.fit.petab_v2.testsuite import (
    PETAB_SUITE_COMMIT,
    CaseResult,
    CaseStatus,
    PetabSuite,
)
from sbmlsim.testsuite import baseline
from sbmlsim.testsuite.baseline import MISSING_REASON, unexpected_outcomes

#: the expected outcomes the tests compare a run with
BASELINE_PATH = Path(__file__).parent.parent / "tests" / "data" / "petab_baseline.json"


def run(suite: PetabSuite) -> list[CaseResult]:
    """Run the cases of the suite and report the outcome."""
    console.print(f"Running the PEtab test suite '{suite.commit}' from {suite.path}")
    results = suite.run()
    counts = Counter(r.status for r in results)
    console.print(f"[bold]{counts[CaseStatus.PASS]}/{len(results)} cases pass[/bold]")
    for result in results:
        if not result.passed:
            console.print(f"  {result.key}  {result.status.value}  {result.message}")
    return results


def write_baseline(results: list[CaseResult], suite: PetabSuite, path: Path) -> None:
    """Write the cases which do not pass, keeping the reasons which are recorded.

    See `sbmlsim.testsuite.baseline.write_baseline`, a case whose status
    changed is reported.

    Args:
        results: the results of a run.
        suite: the suite which was run.
        path: the baseline.
    """
    for note in baseline.write_baseline(results, suite.commit, path):
        console.print(f"  [yellow]{note}[/yellow]")
    console.print(f"Baseline: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))["expected_failures"]
    for key, expected in recorded.items():
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
    parser.add_argument("--commit", default=PETAB_SUITE_COMMIT)
    args = parser.parse_args(argv)

    log.enable_rich_logging()
    suite = PetabSuite.load(args.commit)
    if args.command == "download":
        console.print(f"PEtab test suite '{suite.commit}': {suite.path}")
        return 0

    if args.command == "run" and not BASELINE_PATH.is_file():
        console.print(
            f"[red]The baseline {BASELINE_PATH} does not exist, write it with "
            f"`baseline` first[/red]"
        )
        return 1
    results = run(suite)
    if args.command == "baseline":
        write_baseline(results, suite, BASELINE_PATH)
        return 0
    recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    unexpected = unexpected_outcomes(results, recorded)
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
