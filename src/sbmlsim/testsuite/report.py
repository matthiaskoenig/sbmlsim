"""The report of a run of the SBML Test Suite.

`TestSuiteReport` renders the results of a run as a single HTML page which
extends `report_base.html`, i.e. it has the search, the filter chips and the
sortable tables of the other reports of `sbmlsim` and needs no network. The
page answers which parts of SBML the simulation supports: the cases are
aggregated over their `componentTags` and their `testTags`, and the outcome of
a case is a filter over the case table.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sbmlsim.report.templates import template_environment
from sbmlsim.testsuite.runner import CaseResult, CaseStatus

logger = logging.getLogger(__name__)

#: template of the report
TEMPLATE = "testsuite_report.html"

#: what an outcome says, shown in the overview of the report
STATUS_DESCRIPTION: dict[CaseStatus, str] = {
    CaseStatus.PASS: "the results are within the tolerances of the case",
    CaseStatus.TOLERANCE: "the case was simulated and its results are wrong",
    CaseStatus.NOT_READ: "the model could not be loaded, i.e. SBML which is not supported",
    CaseStatus.SIMULATION_ERROR: "the model was loaded and the integration failed",
    CaseStatus.MISSING_VARIABLE: "a compared variable was not produced by the simulation",
}


def versions() -> dict[str, str]:
    """Get the versions the results were produced with.

    Returns:
        The packages which decide the results, by name.
    """
    import libsbml
    import roadrunner

    from sbmlsim import __version__

    return {
        "sbmlsim": __version__,
        "libroadrunner": roadrunner.__version__.split(";")[0].strip(),
        "libsbml": libsbml.getLibSBMLDottedVersion(),
    }


@dataclass(frozen=True)
class TagSummary:
    """The cases of one tag of the suite.

    Attributes:
        tag: the component or test tag.
        n: number of cases which carry it.
        passed: number of them which passed.
    """

    tag: str
    n: int
    passed: int

    @property
    def failed(self) -> int:
        """Get the number of cases of the tag which did not pass."""
        return self.n - self.passed

    @property
    def rate(self) -> float:
        """Get the fraction of the cases of the tag which passed."""
        return self.passed / self.n if self.n else 0.0


class TestSuiteReport:
    """Report of a run of the semantic cases of the SBML Test Suite."""

    def __init__(self, results: Sequence[CaseResult], suite_version: str) -> None:
        """Construct the report.

        Args:
            results: the results of the run, one per case.
            suite_version: release of the suite which was run.
        """
        self.results = list(results)
        self.suite_version = suite_version

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{self.suite_version}: "
            f"{self.n_passed}/{len(self.results)}>"
        )

    @property
    def n_passed(self) -> int:
        """Get the number of cases which passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        """Get the fraction of the cases which passed."""
        return self.n_passed / len(self.results) if self.results else 0.0

    def counts(self) -> dict[CaseStatus, int]:
        """Get the number of cases per outcome, the largest first."""
        counter = Counter(r.status for r in self.results)
        return dict(counter.most_common())

    def tag_summaries(self, test_tags: bool = False) -> list[TagSummary]:
        """Aggregate the cases over their tags.

        A case carries several tags and counts for each of them, so the
        numbers of the tags do not add up to the number of cases.

        Args:
            test_tags: aggregate the `testTags` instead of the `componentTags`.

        Returns:
            One summary per tag, the worst pass rate first. Sorting by the
            number of failures instead would lead with the tags almost every
            case carries, i.e. `Parameter` and `Species`, which only restate
            the overall rate; a tag which fails everywhere is the finding.
        """
        total: Counter[str] = Counter()
        passed: Counter[str] = Counter()
        for result in self.results:
            tags = result.test_tags if test_tags else result.component_tags
            for tag in tags:
                total[tag] += 1
                if result.passed:
                    passed[tag] += 1

        summaries = [
            TagSummary(tag=tag, n=n, passed=passed[tag]) for tag, n in total.items()
        ]
        return sorted(summaries, key=lambda s: (s.rate, -s.failed, s.tag))

    def to_df(self) -> pd.DataFrame:
        """Get the results as a table, one row per case."""
        return pd.DataFrame(
            [
                {
                    "case": r.cid,
                    "status": r.status.value,
                    "encoding": r.encoding,
                    "component_tags": " ".join(sorted(r.component_tags)),
                    "test_tags": " ".join(sorted(r.test_tags)),
                    "duration": round(r.duration, 4),
                    "message": r.message,
                }
                for r in self.results
            ]
        )

    def context(self) -> dict[str, Any]:
        """Collect everything the report shows.

        Returns:
            The context of the `testsuite_report.html` template.
        """
        counts = self.counts()
        n = len(self.results)

        def _tag_rows(summaries: list[TagSummary]) -> list[dict[str, Any]]:
            """Render the aggregation of a set of tags."""
            return [
                {
                    "tag": s.tag,
                    "n": s.n,
                    "passed": s.passed,
                    "failed": s.failed,
                    "rate": f"{100 * s.rate:.0f}%",
                    "percent": f"{100 * s.rate:.0f}",
                }
                for s in summaries
            ]

        return {
            "suite_version": self.suite_version,
            "statuses": [status.value for status in CaseStatus],
            "badges": [
                {"label": "cases", "value": n},
                {"label": "passed", "value": self.n_passed},
                {"label": "pass rate", "value": f"{100 * self.pass_rate:.1f}%"},
                {"label": "suite", "value": self.suite_version},
            ],
            "outcomes": [
                {
                    "status": status.value,
                    "n": count,
                    "share": f"{100 * count / n:.1f}%" if n else "0%",
                    "description": STATUS_DESCRIPTION[status],
                }
                for status, count in counts.items()
            ],
            "environment": versions(),
            "components": _tag_rows(self.tag_summaries()),
            "tests": _tag_rows(self.tag_summaries(test_tags=True)),
            "cases": [
                {
                    "cid": r.cid,
                    "status": r.status.value,
                    "encoding": r.encoding,
                    "components": " ".join(sorted(r.component_tags)),
                    "tests": " ".join(sorted(r.test_tags)),
                    "message": r.message,
                }
                for r in self.results
            ],
            "files": [
                {"href": "results.tsv", "label": "results.tsv"},
                {"href": "results.json", "label": "results.json"},
            ],
        }

    def create(self, output_dir: Path) -> Path:
        """Write the report and the results it is made from.

        Args:
            output_dir: directory of the report, created if it is missing.

        Returns:
            Path of `index.html`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.to_df().to_csv(output_dir / "results.tsv", sep="\t", index=False)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "suite_version": self.suite_version,
                    "versions": versions(),
                    "n_cases": len(self.results),
                    "n_passed": self.n_passed,
                    "results": [r.to_dict() for r in self.results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        template = template_environment().get_template(TEMPLATE)
        path = output_dir / "index.html"
        path.write_text(template.render(**self.context()), encoding="utf-8")
        logger.info("SBML Test Suite report: file://%s", path)
        return path
