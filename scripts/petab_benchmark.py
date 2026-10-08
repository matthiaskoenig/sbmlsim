"""Download and run the problems of the PEtab benchmark collection.

The problems of the [benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)
say which published problems `sbmlsim` reads and simulates as the collection
does. This script is the command line around `sbmlsim.fit.petab_v2.benchmark`:

```bash
# fetch the pinned commit and the references of AMICI, convert the problems
uv run python scripts/petab_benchmark.py download

# run the problems in parallel, write the results and fail when a problem
# does not have the outcome of the baseline
uv run python scripts/petab_benchmark.py run --processes 8 --output results/benchmark

# the table of the results of a run
uv run python scripts/petab_benchmark.py report --output results/benchmark

# refresh the expected outcomes after a change which fixes or breaks problems
uv run python scripts/petab_benchmark.py baseline --processes 8
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
from sbmlsim.fit.petab_v2.benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_TIMEOUT,
    BenchmarkCollection,
    BenchmarkResult,
    BenchmarkStatus,
)
from sbmlsim.parallel import process_context
from sbmlsim.testsuite import baseline
from sbmlsim.testsuite.baseline import MISSING_REASON, unexpected_outcomes

#: the expected outcomes the tests compare a run with
BASELINE_PATH = (
    Path(__file__).parent.parent / "tests" / "data" / "benchmark_baseline.json"
)

#: the file of the results of a run in the output directory
RESULTS_FILE = "benchmark.json"


def _run_problem(
    task: tuple[BenchmarkCollection, str, float | None],
) -> BenchmarkResult:
    """Run a problem in a worker, with the time limit `BENCHMARK_TIMEOUT`."""
    collection, name, reference = task
    return collection.problem(name).run(
        llh_reference=reference, timeout=BENCHMARK_TIMEOUT
    )


def run(
    collection: BenchmarkCollection, names: list[str] | None, processes: int
) -> list[BenchmarkResult]:
    """Run the problems of the collection and report the outcome."""
    console.print(
        f"Running the PEtab benchmark collection '{collection.commit}' from "
        f"{collection.path}"
    )
    names = names or collection.problem_names()
    references = collection.references()
    tasks = [(collection, name, references.get(name)) for name in names]
    results: dict[str, BenchmarkResult] = {}
    with process_context().Pool(processes, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(_run_problem, tasks):
            results[result.name] = result
            console.print(
                f"  {result.name}  {result.status.value}  "
                f"{sum(result.timings.values()):.1f} s  {result.message[:200]}"
            )
    ordered = [results[name] for name in names]
    counts = Counter(r.status for r in ordered)
    console.print(
        f"[bold]{counts[BenchmarkStatus.PASS]}/{len(ordered)} problems pass[/bold]"
    )
    return ordered


def write_results(results: list[BenchmarkResult], output: Path) -> Path:
    """Write the results of a run as JSON."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / RESULTS_FILE
    path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2) + "\n", encoding="utf-8"
    )
    return path


def report(output: Path) -> Path:
    """Write the table of the results of a run as markdown."""
    results = json.loads((output / RESULTS_FILE).read_text(encoding="utf-8"))
    lines = [
        "| problem | status | simulations | max difference | llh | llh AMICI "
        "| read [s] | initialize [s] | simulate [s] | llh [s] | message |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value: float | None, digits: str = ".4g") -> str:
        return "" if value is None else format(value, digits)

    for r in results:
        timings = r["timings"]
        message = r["message"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['name']} | {r['status']} | {r['n_simulations']} "
            f"| {number(r['max_difference'], '.2e')} | {number(r['llh'], '.10g')} "
            f"| {number(r['llh_reference'], '.10g')} "
            f"| {number(timings.get('read'), '.2f')} "
            f"| {number(timings.get('initialize'), '.2f')} "
            f"| {number(timings.get('simulate'), '.3f')} "
            f"| {number(timings.get('llh'), '.3f')} | {message[:300]} |"
        )
    path = output / "benchmark.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_baseline(
    results: list[BenchmarkResult], collection: BenchmarkCollection, path: Path
) -> None:
    """Write the problems which do not pass, keeping the reasons which are recorded.

    See `sbmlsim.testsuite.baseline.write_baseline`, a problem whose status
    changed is reported.
    """
    for note in baseline.write_baseline(results, collection.commit, path):
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
        The exit code: `1` when `run` finds a problem without the outcome of
        the baseline, `0` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "run", "report", "baseline"])
    parser.add_argument("--commit", default=BENCHMARK_COMMIT)
    parser.add_argument("--problems", nargs="*", help="the problems, all by default")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results") / "benchmark")
    args = parser.parse_args(argv)

    log.enable_rich_logging()
    if args.command == "report":
        console.print(f"Report: {report(args.output)}")
        return 0
    collection = BenchmarkCollection.load(args.commit)
    if args.command == "download":
        console.print(
            f"PEtab benchmark collection '{collection.commit}': {collection.path}"
        )
        return 0

    if args.command == "run" and not BASELINE_PATH.is_file():
        console.print(
            f"[red]The baseline {BASELINE_PATH} does not exist, write it with "
            f"`baseline` first[/red]"
        )
        return 1
    results = run(collection, args.problems, args.processes)
    console.print(f"Results: {write_results(results, args.output)}")
    if args.command == "baseline":
        write_baseline(results, collection, BASELINE_PATH)
        return 0
    recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if args.problems:
        # the baseline of the problems which were run
        recorded = {
            **recorded,
            "expected_failures": {
                key: value
                for key, value in recorded["expected_failures"].items()
                if key in args.problems
            },
        }
    unexpected = unexpected_outcomes(results, recorded)
    if unexpected:
        console.print(
            f"[red]{len(unexpected)} problems do not have the outcome of the "
            f"baseline {BASELINE_PATH}[/red]"
        )
        for line in unexpected:
            console.print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
