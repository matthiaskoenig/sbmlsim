"""Run the SBML Test Suite and produce its report and its submission.

The semantic cases of the [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite)
say which parts of SBML the simulation of `sbmlsim` supports. This script is
the command line around `sbmlsim.testsuite`:

```bash
# fetch the pinned release into the cache, which the tests need
uv run python scripts/testsuite.py download

# run the cases and write the interactive report
uv run python scripts/testsuite.py report --output site/testsuite/report

# refresh the expected outcomes after a change which fixes or breaks cases
uv run python scripts/testsuite.py baseline

# the archive which is submitted to the SBML Test Suite Database
uv run python scripts/testsuite.py submission --output dist
```

`--version latest` resolves the newest release instead of the pinned one,
which is what the release workflow submits against.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.testsuite import SemanticSuite, run_suite
from sbmlsim.testsuite.cases import SUITE_VERSION
from sbmlsim.testsuite.report import TestSuiteReport, versions
from sbmlsim.testsuite.runner import CaseResult, CaseStatus

#: the expected outcomes the tests compare a run with
BASELINE_PATH = (
    Path(__file__).parent.parent / "tests" / "data" / "testsuite_baseline.json"
)


def resolve_version(version: str) -> str:
    """Resolve the release of the suite to run.

    Args:
        version: a release such as `3.5.0`, or `latest` for the newest one.

    Returns:
        The release.
    """
    return SemanticSuite.latest_version() if version == "latest" else version


def load(version: str) -> SemanticSuite:
    """Get the suite, downloading it if it is not cached."""
    return SemanticSuite.load(resolve_version(version))


def run(suite: SemanticSuite) -> list[CaseResult]:
    """Run every timecourse case of the suite and report the outcome."""
    console.print(f"Running the SBML Test Suite '{suite.version}' from {suite.path}")
    results = run_suite(suite)
    counts = Counter(r.status for r in results)
    n = len(results)
    console.print(f"[bold]{counts[CaseStatus.PASS]}/{n} cases pass[/bold]")
    for status, count in counts.most_common():
        console.print(f"  {status.value:18s} {count:5d}  {100 * count / n:5.1f}%")
    return results


def write_submission(suite: SemanticSuite, output_dir: Path) -> Path:
    """Write the archive which is submitted to the SBML Test Suite Database.

    The archive holds the results of every case which could be simulated, one
    CSV per case named after it, and a manifest with the versions the results
    were produced with. A case which the simulator cannot read or integrate has
    no results and is left out, which is how a submission says that it does not
    support the case. Uploading the archive is a deliberate step and is not
    done here.

    Args:
        suite: the suite to run.
        output_dir: directory the archive is written into.

    Returns:
        Path of the archive.
    """
    from sbmlsim import __version__
    from sbmlsim.model import AbstractModel
    from sbmlsim.simulator.simulation_serial import SimulatorSerial
    from sbmlsim.testsuite.runner import (
        INTEGRATOR_ABSOLUTE_TOLERANCE,
        INTEGRATOR_RELATIVE_TOLERANCE,
        simulate_case,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sbmlsim-{__version__}-sbml-test-suite-{suite.version}.zip"

    n_cases = 0
    submitted = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for case in suite.cases():
            n_cases += 1
            simulator = SimulatorSerial(
                absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
                relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
                variable_step_size=False,
            )
            try:
                simulator.set_model(model=AbstractModel(source=case.model_path))
                df = simulate_case(case, simulator)
            except Exception as err:
                console.print(f"  [dim]{case.cid}: no results ({err})[/dim]")
                continue
            # the submission names the columns as the case names its variables,
            # i.e. `S1` and not the selection `[S1]`
            df = df.rename(
                columns=dict(
                    zip(case.selections, ["time", *case.variables], strict=True)
                )
            )
            zf.writestr(f"{case.cid}.csv", df.to_csv(index=False))
            submitted += 1

        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "simulator": "sbmlsim",
                    "versions": versions(),
                    "suite_version": suite.version,
                    "n_cases": n_cases,
                    "n_submitted": submitted,
                },
                indent=2,
            ),
        )

    console.print(f"Submission archive with {submitted} of {n_cases} cases: {path}")
    return path


def write_baseline(results: list[CaseResult], suite: SemanticSuite) -> Path:
    """Write the expected outcomes the tests compare a run with."""
    failures = {r.cid: r.status.value for r in results if not r.passed}
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "suite_version": suite.version,
                "n_cases": len(results),
                "n_passed": len(results) - len(failures),
                "expected_failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"Baseline with {len(failures)} expected failures: {BASELINE_PATH}")
    return BASELINE_PATH


def main(argv: list[str] | None = None) -> int:
    """Run the command line tool.

    Args:
        argv: command line arguments, `sys.argv` by default.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command",
        choices=["download", "run", "report", "baseline", "submission"],
        help="what to do",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=SUITE_VERSION,
        help=f"release of the suite, or 'latest'; '{SUITE_VERSION}' by default",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("site") / "testsuite" / "report",
        help="directory of the report or of the submission archive",
    )
    options = parser.parse_args(argv)
    log.enable_rich_logging()

    suite = load(options.version)
    if options.command == "download":
        console.print(f"SBML Test Suite '{suite.version}': {suite.path}")
        return 0

    if options.command == "submission":
        # the submission needs the results of every case and not their outcome,
        # so it simulates in one pass of its own
        write_submission(suite, options.output)
        return 0

    results = run(suite)
    if options.command == "report":
        TestSuiteReport(results, suite_version=suite.version).create(options.output)
    elif options.command == "baseline":
        write_baseline(results, suite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
