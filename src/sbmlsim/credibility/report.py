"""The outputs of a credibility assessment: the record, the tables and the report.

`write_outputs` writes an assessment into a directory of its own:

- `credibility.json`: the assessment, see `CredibilityAssessment.to_json`.
- `contexts.tsv`: one row per context of use with its level, influence,
  consequence, risk, verdict, the decisive and the evaluable points, the
  minimum of evaluable points, the uncovered quantities, whether it is
  verified and per acceptance rule, of the context and of the evidence
  extensions, its value (`<name>`) and limit (`<name>_limit`).
- `evidence.tsv`: one row per context of use and evidence and per limitation
  (the limitations of the model with the context `""`): the role, the points
  (`n`), mappings and studies, GMFE, AFE and the criteria.
- `domain.tsv`: one row per context of use, field and value of the domain:
  whether it is claimed, its coverage and its points (`n`).
- `quantities.tsv`: one row per context of use and quantity of interest: its
  description, whether it is a ratio, its coverage and its points.
- `verification.tsv`: one row per context of use and check of the
  verification, empty if the assessment was not verified: the factor, the
  compared points (`n`), the largest relative difference and its point,
  whether the acceptance changed, whether the check passed and its problems;
  the checks of the battery are rows of every context of use.
- `uncertainty.tsv`: one row per context of use and acceptance rule over the
  parameter draws of the adopted fits, empty without the propagation: the
  probability that the rule is met with its anytime-valid sequence, the
  decision of the statement at the threshold, the draws and the failed and
  needed draws.
- `uncertainty_quantities.tsv`: one row per context of use and quantity of
  interest which is no ratio: the points, the median fold width of the
  prediction bands and their coverage of the observations.
- `uncertainty_sensitivity.tsv`: one row per context of use, target (a
  quantity or a rule) and fitted parameter: the first-order index of the
  target over the parameter draws.
- `variability.tsv`: one row per context of use with the predictive check of
  its decisive points with the declared variability of the subjects, empty
  without: the subjects, the parameter vectors, the points and the means of
  groups, the coverage with its Wilson interval, the expected coverage and the
  agreement.
- `variability_quantities.tsv`: one row per context of use and quantity of
  interest which is no ratio: the points with a band and their coverage.
- `vpc.tsv`: one row per data point of the adopted fits with the band of its
  predictive check, empty without: the subjects of the observation, the
  statistic, the typical prediction, the quantiles and whether it is inside.
- `cross_validation.tsv`: one row per cross-validated adopted fit and held-out
  study: its points, the GMFE and the share within 2-fold of the held-out and
  of the fitted predictions, the cost of the refit.
- `battery.tsv`: one row per check of the battery of the variants, empty if it
  did not run: the factor, the status, the cases, the largest deviation and
  where, the statement and the problems.
- the tables of the evidence extensions (`EvidenceExtension.report_tables`)
  and of the battery hook (`BatteryHook.report_tables`), `<name>.tsv`.
- `index.html`: the report (`render_html`), which extends `report_base.html`
  of `sbmlsim.report.templates` and loads nothing from the network.

`print_summary` prints the table of the contexts of use, their verdicts, the
role of the evidence which decides them and the evaluable points with their
minimum, and how many of them are verified.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import pandas as pd
from markupsafe import Markup
from rich import box
from rich.markup import escape
from rich.table import Table

from sbmlsim.console import console
from sbmlsim.credibility.assessment import (
    ContextResult,
    CredibilityAssessment,
    EvidenceResult,
    Verdict,
)
from sbmlsim.credibility.extensions import CONTEXT_BLOCKS, ReportSection
from sbmlsim.credibility.risk import DEFAULT_MIN_POINTS
from sbmlsim.credibility.variability import VPC_COLUMNS
from sbmlsim.credibility.verification import CHECKS, PREDICTION_CHECKS, check_of
from sbmlsim.report.templates import template_environment

if TYPE_CHECKING:
    from pathlib import Path

    from sbmlsim.credibility.battery import BatteryHook
    from sbmlsim.credibility.extensions import EvidenceExtension

__all__ = [
    "BATTERY_COLUMNS",
    "CONTEXT_COLUMNS",
    "CROSS_VALIDATION_COLUMNS",
    "DOMAIN_COLUMNS",
    "EVIDENCE_COLUMNS",
    "QUANTITY_COLUMNS",
    "RECORD",
    "SENSITIVITY_COLUMNS",
    "TEMPLATE",
    "UNCERTAINTY_COLUMNS",
    "UNCERTAINTY_QUANTITY_COLUMNS",
    "VARIABILITY_COLUMNS",
    "VERIFICATION_COLUMNS",
    "battery_table",
    "contexts_table",
    "cross_validation_table",
    "domain_table",
    "evidence_table",
    "print_summary",
    "quantities_table",
    "render_html",
    "sensitivity_table",
    "uncertainty_quantities_table",
    "uncertainty_table",
    "variability_quantities_table",
    "variability_table",
    "verification_table",
    "vpc_table",
    "write_outputs",
]

logger = logging.getLogger(__name__)

#: the template of the report, in the templates of sbmlsim
TEMPLATE: Final = "credibility.html"
#: the record of the assessment, which marks a directory of the outputs
RECORD: Final = "credibility.json"

#: the style of a verdict in the console
_VERDICT_STYLES: Final = {
    Verdict.SUPPORTED: "green",
    Verdict.SUPPORTED_WITH_EXTRAPOLATION: "blue",
    Verdict.NOT_SUPPORTED: "red",
    Verdict.INSUFFICIENT_EVIDENCE: "orange3",
    Verdict.NOT_VERIFIED: "magenta",
}

#: the leading columns of `contexts_table`
CONTEXT_COLUMNS: Final = (
    "id",
    "title",
    "level",
    "influence",
    "consequence",
    "risk",
    "verdict",
    "n_decisive",
    "n_evaluable",
    "minimum_points",
    "uncovered_quantities",
    "verified",
)
#: the leading columns of `evidence_table`, the metrics follow
EVIDENCE_COLUMNS: Final = (
    "context",
    "name",
    "role",
    "selection",
    "n",
    "n_mappings",
    "n_studies",
    "outliers_dropped",
    "limitation_subtracted",
)
#: the columns of `domain_table`
DOMAIN_COLUMNS: Final = ("context", "field", "value", "claimed", "coverage", "n")
#: the columns of `quantities_table`
QUANTITY_COLUMNS: Final = (
    "context",
    "quantity",
    "description",
    "ratio",
    "coverage",
    "n_points",
)
#: the columns of `verification_table`
VERIFICATION_COLUMNS: Final = (
    "context",
    "check",
    "factor",
    "n",
    "max_difference",
    "at",
    "acceptance_changed",
    "passed",
    "problems",
)
#: the columns of `cross_validation_table`
CROSS_VALIDATION_COLUMNS: Final = (
    "fit",
    "subset",
    "study",
    "n_points",
    "gmfe_held_out",
    "gmfe_fitted",
    "within_2fold_held_out",
    "within_2fold_fitted",
    "cost",
)
#: the columns of `battery_table`
BATTERY_COLUMNS: Final = (
    "check",
    "factor",
    "status",
    "cases",
    "largest",
    "at",
    "statement",
    "problems",
)
#: the columns of `uncertainty_table`
UNCERTAINTY_COLUMNS: Final = (
    "context",
    "rule",
    "limit",
    "estimate",
    "low",
    "high",
    "threshold",
    "decision",
    "n",
    "n_failed",
    "n_needed",
)
#: the columns of `uncertainty_quantities_table`
UNCERTAINTY_QUANTITY_COLUMNS: Final = (
    "context",
    "quantity",
    "n_points",
    "band_fold",
    "coverage",
)
#: the columns of `sensitivity_table`
SENSITIVITY_COLUMNS: Final = ("context", "target", "kind", "fit", "parameter", "index")
#: the columns of `variability_table`
VARIABILITY_COLUMNS: Final = (
    "context",
    "subjects",
    "studies",
    "vectors",
    "groups",
    "seed",
    "targets",
    "between",
    "n_points",
    "n_groups",
    "coverage",
    "low",
    "high",
    "level",
    "expected",
    "agreement",
    "failed",
)


def _columns(rows: Iterable[Mapping[str, object]], leading: Sequence[str]) -> list[str]:
    """Get the leading columns and the other keys of rows in their first order."""
    columns = list(leading)
    for row in rows:
        columns += [key for key in row if key not in columns]
    return columns


def contexts_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use with the values and limits of its rules.

    Args:
        assessment: the assessment.

    Returns:
        The columns `id`, `title`, `level`, `influence`, `consequence`,
        `risk`, `verdict`, `n_decisive`, `n_evaluable`, `minimum_points`,
        `uncovered_quantities` (the ids, comma separated, empty without),
        `verified` (empty if the assessment was not verified) and per
        acceptance rule, the rules of the extensions included, `<name>` and
        `<name>_limit`, empty for a context without the rule.
    """
    rows: list[dict[str, object]] = []
    for c in assessment.contexts:
        row: dict[str, object] = {
            "id": c.id,
            "title": c.title,
            "level": c.level.value,
            "influence": c.influence.value,
            "consequence": c.consequence.value,
            "risk": c.risk.value,
            "verdict": c.verdict.value,
            "n_decisive": c.n_decisive,
            "n_evaluable": c.n_evaluable,
            "minimum_points": c.minimum_points,
            "uncovered_quantities": ",".join(c.uncovered_quantities),
            "verified": c.verified,
        }
        for result in c.acceptance:
            row[result["name"]] = result["value"]
            row[f"{result['name']}_limit"] = result["limit"]
        rows.append(row)
    return pd.DataFrame(rows, columns=_columns(rows, CONTEXT_COLUMNS))


def evidence_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per evidence of a context of use and per limitation.

    Args:
        assessment: the assessment.

    Returns:
        The columns `context` (`""` for a limitation of the model), `name`,
        `role`, `selection`, `n` (the points), `n_mappings`, `n_studies`,
        `outliers_dropped`, `limitation_subtracted` and the metrics: GMFE, AFE,
        the criteria and the values of the acceptance rules of the context,
        empty if not evaluable.
    """
    rows = [
        {
            "context": e.context,
            "name": e.name,
            "role": e.role.value,
            "selection": e.selection,
            "n": e.n_points,
            "n_mappings": e.n_mappings,
            "n_studies": e.n_studies,
            "outliers_dropped": e.outliers_dropped,
            "limitation_subtracted": e.limitation_subtracted,
            **e.metrics,
        }
        for e in (*assessment.evidence, *assessment.limitations)
    ]
    return pd.DataFrame(rows, columns=_columns(rows, EVIDENCE_COLUMNS))


def domain_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use, field and value of its domain.

    Args:
        assessment: the assessment.

    Returns:
        The columns `context`, `field`, `value`, `claimed`, `coverage` (empty
        for a value outside the claim) and `n`, the points, see
        `CoverageEntry`.
    """
    rows = [
        {
            "context": e.context,
            "field": e.field,
            "value": e.value,
            "claimed": e.claimed,
            "coverage": None if e.coverage is None else e.coverage.value,
            "n": e.n_points,
        }
        for e in assessment.coverage
    ]
    return pd.DataFrame(rows, columns=list(DOMAIN_COLUMNS))


def quantities_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use and quantity of interest.

    Args:
        assessment: the assessment.

    Returns:
        The columns `context`, `quantity`, `description`, `ratio`, `coverage`
        (empty for a ratio without a pairing) and `n_points`, see
        `QuantityRecord`.
    """
    rows = [
        {
            "context": c.id,
            "quantity": q["id"],
            "description": q["description"],
            "ratio": q["ratio"],
            "coverage": q["coverage"],
            "n_points": q["n_points"],
        }
        for c in assessment.contexts
        for q in c.quantities
    ]
    return pd.DataFrame(rows, columns=list(QUANTITY_COLUMNS))


def verification_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use and check of the verification.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `VERIFICATION_COLUMNS`, see `CheckRecord`, the problems
        joined by `; `; no rows if the assessment was not verified.
    """
    rows = [
        {
            "context": c.id,
            "check": r["check"],
            "factor": r["factor"],
            "n": r["n_points"],
            "max_difference": r["max_difference"],
            "at": r["at"],
            "acceptance_changed": r["acceptance_changed"],
            "passed": r["passed"],
            "problems": "; ".join(r["problems"]),
        }
        for c in assessment.contexts
        for r in c.verification
    ]
    return pd.DataFrame(rows, columns=list(VERIFICATION_COLUMNS))


def uncertainty_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use and rule over the parameter draws.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `UNCERTAINTY_COLUMNS`, see `RuleProbability`; no rows
        without the propagation.
    """
    rows = [
        {
            "context": c.id,
            "rule": r["name"],
            "threshold": c.uncertainty["threshold"],
            **{k: v for k, v in r.items() if k != "name"},
        }
        for c in assessment.contexts
        if c.uncertainty is not None
        for r in c.uncertainty["rules"]
    ]
    return pd.DataFrame(rows, columns=list(UNCERTAINTY_COLUMNS))


def uncertainty_quantities_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use and quantity with its prediction band.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `UNCERTAINTY_QUANTITY_COLUMNS`, see `QuantityBand`; no
        rows without the propagation.
    """
    rows = [
        {"context": c.id, **q}
        for c in assessment.contexts
        if c.uncertainty is not None
        for q in c.uncertainty["quantities"]
    ]
    return pd.DataFrame(rows, columns=list(UNCERTAINTY_QUANTITY_COLUMNS))


def sensitivity_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use, target and parameter of the draws.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `SENSITIVITY_COLUMNS`, see `SensitivityIndex`; no rows
        without the propagation.
    """
    rows = [
        {"context": c.id, **i}
        for c in assessment.contexts
        if c.uncertainty is not None
        for i in c.uncertainty["sensitivity"]
    ]
    return pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS))


def variability_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use with its predictive check.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `VARIABILITY_COLUMNS`, see `VariabilityRecord`, the
        targets of the subjects and between the studies as `target
        distribution` joined by `;`; no rows without the predictive check.
    """
    rows = [
        {
            "context": c.id,
            **{k: v for k, v in c.variability.items() if k in VARIABILITY_COLUMNS},
            "targets": ";".join(
                f"{t} {d}" for t, d in c.variability["targets"].items()
            ),
            "between": ";".join(
                f"{t} {d}" for t, d in c.variability["between"].items()
            ),
        }
        for c in assessment.contexts
        if c.variability is not None
    ]
    return pd.DataFrame(rows, columns=list(VARIABILITY_COLUMNS))


def variability_quantities_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per context of use and quantity of its predictive check.

    Args:
        assessment: the assessment.

    Returns:
        The columns `context`, `quantity`, `n_points` and `coverage`; no rows
        without the predictive check.
    """
    rows = [
        {"context": c.id, **q}
        for c in assessment.contexts
        if c.variability is not None
        for q in c.variability["quantities"]
    ]
    return pd.DataFrame(rows, columns=["context", "quantity", "n_points", "coverage"])


def vpc_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per data point of the adopted fits with its band.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `VPC_COLUMNS`, see `VpcPoint`; no rows without the
        predictive check.
    """
    return pd.DataFrame(
        [dict(point) for point in assessment.vpc], columns=list(VPC_COLUMNS)
    )


def cross_validation_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per cross-validated fit and held-out study.

    Args:
        assessment: the assessment.

    Returns:
        The columns `fit`, `subset` and the fields of `HeldOutStudy`; no rows
        without a cross-validation.
    """
    rows = [
        {"fit": cv.fit, "subset": cv.subset, **study.to_dict()}
        for cv in assessment.cross_validations
        for study in cv.studies
    ]
    return pd.DataFrame(rows, columns=list(CROSS_VALIDATION_COLUMNS))


def battery_table(assessment: CredibilityAssessment) -> pd.DataFrame:
    """Get one row per check of the battery of the variants of the model.

    Args:
        assessment: the assessment.

    Returns:
        The columns of `BATTERY_COLUMNS`, see `BatteryResult`, the problems
        joined by `; `; no rows if the battery did not run.
    """
    battery = (
        None if assessment.verification is None else assessment.verification.battery
    )
    rows = [
        {
            "check": r.check.id,
            "factor": r.check.factor,
            "status": r.status.value,
            "cases": r.cases,
            "largest": r.largest,
            "at": r.at,
            "statement": r.statement,
            "problems": "; ".join(r.problems),
        }
        for r in ([] if battery is None else battery.results)
    ]
    return pd.DataFrame(rows, columns=list(BATTERY_COLUMNS))


def _number(value: float | None) -> str:
    """Show a metric with three significant digits, `n/a` if not evaluable."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}"


def _metric_names(results: Sequence[EvidenceResult]) -> list[str]:
    """Get the names of the metrics of evidence in their first order."""
    names: list[str] = []
    for result in results:
        names += [name for name in result.metrics if name not in names]
    return names


def _blocks(
    context: ContextResult, extensions: Sequence[EvidenceExtension]
) -> dict[str, Markup]:
    """Get the blocks of the extensions in the card of a context, by the block they follow."""
    record = context.to_dict()
    html: dict[str, list[str]] = {after: [] for after in CONTEXT_BLOCKS}
    for extension in extensions:
        for block in extension.context_report_blocks(
            record, context.section(extension.name)
        ):
            html[block.after].append(block.html)
    return {after: Markup("\n".join(parts)) for after, parts in html.items()}


def _context_view(
    context: ContextResult,
    assessment: CredibilityAssessment,
    extensions: Sequence[EvidenceExtension],
) -> dict[str, object]:
    """Get what the report shows of a context of use."""
    evidence = [e for e in assessment.evidence if e.context == context.id]
    return {
        "result": context,
        "evidence": evidence,
        "metrics": _metric_names(evidence),
        "coverage": [c for c in assessment.coverage if c.context == context.id],
        "blocks": _blocks(context, extensions),
    }


def _sections(
    assessment: CredibilityAssessment, extensions: Sequence[EvidenceExtension]
) -> list[ReportSection]:
    """Get the sections of the extensions in the report.

    Raises:
        ValueError: if a section takes the anchor of a section of the core or
            of another extension.
    """
    taken = {
        "overview",
        "contexts",
        "limitations",
        "unclaimed",
        "verification",
        "battery",
        "cross-validation",
        "provenance",
    }
    record = assessment.to_dict()
    sections: list[ReportSection] = []
    for extension in extensions:
        section = extension.report_section(record)
        if section is None:
            continue
        if section.id in taken or section.id.startswith("cou-"):
            raise ValueError(
                f"The section '{section.id}' of the extension '{extension.name}' "
                "takes the anchor of another section of the report."
            )
        taken.add(section.id)
        sections.append(
            ReportSection(id=section.id, title=section.title, html=Markup(section.html))
        )
    return sections


def render_html(
    assessment: CredibilityAssessment,
    extensions: Sequence[EvidenceExtension] = (),
    hook: BatteryHook | None = None,
) -> str:
    """Render the report of an assessment.

    The template `credibility.html` of sbmlsim extends `report_base.html`. The
    sections: the overview (the contexts of use and the limitations of the
    model), one card per context of use, the limitations, the unclaimed
    validation data, the verification, the cross-validation, the sections of
    the extensions and the provenance. The blocks of an extension in the card
    of a context follow the block of the core they name
    (`EvidenceExtension.context_report_blocks`); without extensions the report
    shows the core alone.

    Args:
        assessment: the assessment.
        extensions: the evidence extensions of the project, in their order.
        hook: the battery hook of the project, whose HTML follows the table of
            the battery.

    Returns:
        The HTML of the report.
    """
    env = template_environment()
    env.filters["number"] = _number
    verdicts = [v for v in Verdict if any(c.verdict is v for c in assessment.contexts)]
    battery = (
        None if assessment.verification is None else assessment.verification.battery
    )
    check_labels = {check.id: check.label for check in CHECKS.values()}
    for c in assessment.contexts:
        for r in c.verification:
            check_labels.setdefault(r["check"], check_of(r["check"], r["factor"]).label)
    return env.get_template(TEMPLATE).render(
        assessment=assessment,
        provenance=assessment.provenance,
        contexts=[
            _context_view(c, assessment, extensions) for c in assessment.contexts
        ],
        verdicts=verdicts,
        verdict_counts={
            v: sum(c.verdict is v for c in assessment.contexts) for v in verdicts
        },
        model_limitations=[e for e in assessment.limitations if not e.context],
        default_min_points=[(risk.value, n) for risk, n in DEFAULT_MIN_POINTS.items()],
        limitation_metrics=_metric_names(assessment.limitations),
        verification=assessment.verification,
        check_labels=check_labels,
        prediction_checks=[c.id for c in PREDICTION_CHECKS],
        extension_labels={e.name: e.label or e.name for e in extensions},
        extension_sections=_sections(assessment, extensions),
        battery_html=Markup(
            hook.report_html(battery.sections)
            if hook is not None and battery is not None
            else ""
        ),
    )


def _tsv(table: pd.DataFrame) -> str:
    """Get a table as TSV, missing values empty."""
    return table.to_csv(sep="\t", index=False, lineterminator="\n")


def _files(
    assessment: CredibilityAssessment,
    extensions: Sequence[EvidenceExtension],
    hook: BatteryHook | None,
) -> dict[str, str]:
    """Get the tables and the report of an assessment by their file.

    Raises:
        ValueError: if a table of an extension or of the hook takes the name
            of another file.
    """
    files = {
        "contexts.tsv": _tsv(contexts_table(assessment)),
        "evidence.tsv": _tsv(evidence_table(assessment)),
        "domain.tsv": _tsv(domain_table(assessment)),
        "quantities.tsv": _tsv(quantities_table(assessment)),
        "variability.tsv": _tsv(variability_table(assessment)),
        "variability_quantities.tsv": _tsv(variability_quantities_table(assessment)),
        "vpc.tsv": _tsv(vpc_table(assessment)),
        "uncertainty.tsv": _tsv(uncertainty_table(assessment)),
        "uncertainty_quantities.tsv": _tsv(uncertainty_quantities_table(assessment)),
        "uncertainty_sensitivity.tsv": _tsv(sensitivity_table(assessment)),
        "cross_validation.tsv": _tsv(cross_validation_table(assessment)),
        "verification.tsv": _tsv(verification_table(assessment)),
        "battery.tsv": _tsv(battery_table(assessment)),
    }
    record = assessment.to_dict()
    tables: list[tuple[str, Mapping[str, pd.DataFrame]]] = [
        (f"the extension '{e.name}'", e.report_tables(record)) for e in extensions
    ]
    battery = (
        None if assessment.verification is None else assessment.verification.battery
    )
    if hook is not None:
        sections = {} if battery is None else battery.sections
        tables.append((f"the battery hook '{hook.name}'", hook.report_tables(sections)))
    for owner, named in tables:
        for stem, table in named.items():
            name = f"{stem}.tsv"
            if name in files or name in (RECORD, "index.html"):
                raise ValueError(
                    f"The table '{name}' of {owner} takes the name of another output."
                )
            files[name] = _tsv(table)
    files["index.html"] = render_html(assessment, extensions, hook)
    return files


def _clear(directory: Path) -> None:
    """Remove every entry of a directory, the directory itself is kept."""
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def write_outputs(
    assessment: CredibilityAssessment,
    output_dir: Path,
    extensions: Sequence[EvidenceExtension] = (),
    hook: BatteryHook | None = None,
) -> Path:
    """Write the record, the tables and the report of an assessment.

    A directory of the outputs is replaced as a whole, so no file of an earlier
    assessment is left; a directory is one when it holds `credibility.json`.
    The outputs are rendered before anything is removed. The entries of the
    directory are removed and the directory itself is kept: windows refuses
    to remove or rename a directory which a process has open, e.g. as its
    working directory or in an explorer.

    Args:
        assessment: the assessment.
        output_dir: the directory, created if missing.
        extensions: the evidence extensions of the project, whose tables and
            blocks the outputs have.
        hook: the battery hook of the project.

    Returns:
        The directory.

    Raises:
        FileExistsError: if `output_dir` is a file or a non-empty directory
            without `credibility.json`.
        ValueError: if a table of an extension takes the name of another
            output.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"The output directory '{output_dir}' is a file.")
    if (
        output_dir.is_dir()
        and any(output_dir.iterdir())
        and not (output_dir / RECORD).is_file()
    ):
        raise FileExistsError(
            f"The output directory '{output_dir}' is not empty and holds no "
            f"{RECORD}; it is not replaced."
        )
    files = _files(assessment, extensions, hook)
    if output_dir.is_dir():
        logger.info("Replace the credibility outputs in '%s'", output_dir)
        _clear(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assessment.to_json(output_dir / RECORD)
    for name, text in files.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    return output_dir


def _verification_line(assessment: CredibilityAssessment) -> str:
    """Get the line of the summary which tells whether the contexts are verified."""
    if assessment.verification is None:
        return "The predictions are not verified."
    verified = sum(c.verified is True for c in assessment.contexts)
    n = len(assessment.contexts)
    style = "green" if verified == n else "magenta"
    checks = ", ".join(check.id for check in PREDICTION_CHECKS)
    line = f"[{style}]{verified} of {n} contexts of use verified[/] ({checks}"
    battery = assessment.verification.battery
    if battery is None:
        return f"{line}; the battery of the variants did not run)."
    applied = [r for r in battery.results if r.passed is not None]
    passed = sum(r.passed is True for r in applied)
    failed = ", ".join(r.check.id for r in applied if r.passed is False)
    return (
        f"{line}; battery of the variants: {passed} of {len(applied)} applicable "
        f"checks passed{f', failed: {failed}' if failed else ''})."
    )


def print_summary(assessment: CredibilityAssessment) -> None:
    """Print the contexts of use and their verdicts on the console of sbmlsim.

    Args:
        assessment: the assessment.
    """
    table = Table(
        title=f"Credibility of {escape(assessment.provenance.title)}",
        box=box.SIMPLE_HEAD,
        # one space between the columns and no edges, the table fits 79
        # columns, the width of a console of windows
        padding=0,
        pad_edge=False,
        show_edge=False,
    )
    # an id is not cut, a column which does not fit folds its text; a verdict
    # wraps between its words (`supported with extrapolation` beside a long id)
    table.add_column(
        "context",
        overflow="fold",
        min_width=max((len(c.id) for c in assessment.contexts), default=0),
    )
    table.add_column("level", no_wrap=True)
    table.add_column("risk", no_wrap=True)
    table.add_column(
        "verdict",
        min_width=max(len(word) for v in Verdict for word in v.label.split()),
    )
    # a description (C1) is decided on the data it was fitted to
    table.add_column("decided on", no_wrap=True)
    table.add_column("points", justify="right", no_wrap=True)
    # the evaluable points and their minimum
    table.add_column("evaluable", justify="right", no_wrap=True)
    for c in assessment.contexts:
        table.add_row(
            escape(c.id),
            c.level.value,
            c.risk.value,
            f"[{_VERDICT_STYLES[c.verdict]}]{c.verdict.label}[/]",
            c.decisive_role.value,
            str(c.n_decisive),
            # a record of pkpdsim 0.17 has no minimum
            f"{c.n_evaluable}/{c.minimum_points}"
            if c.minimum_points
            else str(c.n_evaluable),
        )
    # at most the 79 columns of a console of windows
    console.print(table, width=min(console.width, 79))
    console.print(_verification_line(assessment))
