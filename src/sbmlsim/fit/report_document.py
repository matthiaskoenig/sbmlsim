"""Report document of a parameter fit: its methods, its results and references.

`results_section` writes the results of a `FitReport` as blocks of a report
document: the table of the fitted parameters with their confidence intervals
and identifiability, the table of the data per study and kind, the table of
the metrics per kind, the figures of the fit which were created and the table
of the provenance. `fit_document` puts the methods of `methods_section` and
the results into one document with the references they cite.

Only what the fit has is reported and nothing is invented: a figure whose
file does not exist is left out, so a document references no table or figure
it does not contain, the provenance of a result which has none reads "not
recorded", and the end of a profile interval which the profile does not prove
reads "not determined".
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import Identifiability, ParameterProfile
from sbmlsim.fit.methods import (
    ReportContext,
    _created,
    _label,
    describe_parameter,
    methods_section,
)
from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.report.document import (
    Block,
    Code,
    Document,
    Figure,
    Inline,
    Math,
    Ref,
    Section,
    Table,
    Text,
    p,
)
from sbmlsim.report.references import bibliography

if TYPE_CHECKING:
    # the report builds its document, it imports this module
    from sbmlsim.fit.report import FitReport

__all__ = ["FIGURES", "fit_document", "results_section"]

#: the figures of the results, in their order; a figure is shown if its file exists
FIGURES: tuple[str, ...] = (
    "traces",
    "waterfall",
    "goodness_of_fit",
    "bland_altman",
    "profiles",
)

#: the identifiability of a parameter in the table of the parameters
IDENTIFIABLE: dict[Identifiability, str] = {
    Identifiability.IDENTIFIABLE: "yes",
    Identifiability.NON_IDENTIFIABLE_LOWER: "no (LB)",
    Identifiability.NON_IDENTIFIABLE_UPPER: "no (UB)",
    Identifiability.NON_IDENTIFIABLE: "no",
    Identifiability.STRUCTURAL: "structural",
}

#: the side of a profile interval which the profile does not prove
NOT_DETERMINED = "not determined"

#: the item of the table of the provenance which a result has not recorded
NOT_RECORDED = "not recorded"

#: the kinds of data of the table of the data, its columns
DATA_KINDS: tuple[MappingKind, ...] = EVALUATED_KINDS

#: the names of the packages of the provenance in the table of the provenance
PACKAGE_NAMES: dict[str, str] = {
    "python": "Python",
    "sbmlsim": "sbmlsim",
    "libroadrunner": "libRoadRunner",
    "scipy": "SciPy",
    "numpy": "NumPy",
}


def _g(value: Any) -> str:
    """Format a number with three significant digits, `-` for a missing one."""
    if value is None:
        return "-"
    x = float(value)
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:.3g}"


def _unit(unit: str | None) -> str:
    """Format a unit of pint, without the parentheses around a single unit."""
    if not unit:
        return "model unit"
    return re.sub(r"\(([A-Za-z_]\w*)\)", r"\1", unit)


def _fit_id(report: FitReport) -> str:
    """Get the id of the fit of a report, the id of its result or its problem."""
    return report.opt_result.sid if report.opt_result else report.problem.opid


def _best_set(report: FitReport) -> ParameterSet:
    """Get the parameter set of the report with the lowest cost.

    The reference set if no set has a cost, e.g. a report of the values of the
    model alone.
    """
    costs = {
        k: float(pset.cost)
        for k, pset in enumerate(report.parameter_sets)
        if pset.cost is not None
    }
    if not costs:
        return report.reference_set
    return report.parameter_sets[min(costs, key=costs.__getitem__)]


def _section(title: str, key: str, blocks: list[Block]) -> Section:
    """Get a sub-section of the results."""
    return Section(title, tuple(blocks), level=2, key=key)


def _caption(*parts: str | Inline) -> tuple[Inline, ...]:
    """Get the spans of a caption."""
    return p(*parts).content


def _markup(text: str) -> list[Inline]:
    """Get the spans of a text whose parts in backticks are code."""
    return [
        Code(part) if k % 2 else Text(part)
        for k, part in enumerate(re.split(r"`", text))
        if part
    ]


def results_section(
    report: FitReport, context: ReportContext, plots_dir: Path
) -> Section:
    """Write the results of a fit.

    Args:
        report: the report of the fit; its parameter set with the lowest cost
            is the result which is reported.
        context: what the report says about the fit beyond the fit.
        plots_dir: the folder of the plots of the report, `plots` next to the
            document; a figure is shown if its file is in the folder.

    Returns:
        The section of the results with the sub-sections `results:parameters`,
        `results:data`, `results:metrics`, `results:figures` (if a figure
        exists) and `results:provenance`, whose items read "not recorded" for
        a result without a provenance.
    """
    fit_id = _fit_id(report)
    best = _best_set(report)
    blocks: list[Block] = [
        _parameters(report, context, best, fit_id),
        _data(report, best, fit_id),
        _metrics(report, best, fit_id),
    ]
    figures = _figures(report, plots_dir, fit_id)
    if figures is not None:
        blocks.append(figures)
    blocks.append(_provenance(report, fit_id))
    title = f"Results: {context.title}" if context.title else "Results"
    return Section(title, tuple(blocks), level=1, key="results")


def fit_document(
    report: FitReport, context: ReportContext, plots_dir: Path
) -> Document:
    """Write the report document of a fit, its methods and its results.

    Args:
        report: the report of the fit.
        context: what the report says about the fit beyond the fit.
        plots_dir: the folder of the plots of the report, see `results_section`.

    Returns:
        The document with the title of the context or the id of the fit, the
        sections `methods` and `results` and the cited references, a software
        with the version of the provenance of the fit.
    """
    result = report.opt_result
    methods = methods_section(
        problem=report.problem,
        settings=report.settings,
        result=result,
        identifiability=report.identifiability,
        fisher=report.fisher,
        context=context,
    )
    results = results_section(report, context, plots_dir)
    document = Document(
        title=context.title or _fit_id(report), blocks=(methods, results)
    )
    provenance = result.provenance if result is not None else None
    versions = provenance.versions if provenance is not None else None
    return Document(
        title=document.title,
        blocks=document.blocks,
        references=bibliography(document, versions=versions),
    )


def _interval(
    profile: ParameterProfile | None, fisher: pd.DataFrame | None, pid: str
) -> tuple[str, str] | None:
    """Get the confidence interval of a parameter as text.

    The interval of the profile likelihood, an open side of a classified
    profile as the bound `LB` or `UB`; a missing side of a profile without a
    classification (a point failed or did not converge) is not proven open and
    reads `not determined`. Without a profile the Wald interval of the Fisher
    information; else `None`.
    """
    if profile is not None:
        open_lower, open_upper = (
            ("LB", "UB")
            if profile.identifiability is not None
            else (NOT_DETERMINED, NOT_DETERMINED)
        )
        lower = open_lower if profile.ci_lower is None else _g(profile.ci_lower)
        upper = open_upper if profile.ci_upper is None else _g(profile.ci_upper)
        return lower, upper
    if fisher is not None and pid in fisher.index:
        lower, upper = fisher.loc[pid, "ci_lower"], fisher.loc[pid, "ci_upper"]
        if math.isfinite(float(lower)) or math.isfinite(float(upper)):
            return _g(lower), _g(upper)
    return None


def _interval_text(
    level: float,
    profiles: bool,
    fisher: FisherInformation | None,
    undetermined: bool = False,
) -> str:
    """Write where the confidence intervals and the identifiability come from.

    `undetermined` explains `not determined`, a side of a profile interval
    which is not proven.
    """
    wald = "the Wald intervals of the Fisher information"
    if fisher is not None and not math.isclose(fisher.alpha, level):
        wald += f" at the confidence level {fisher.alpha:.0%}"
    if profiles:
        text = (
            f"The {level:.0%} confidence intervals are those of the profile "
            "likelihood, LB and UB an interval which is open towards the lower or "
            "the upper bound"
        )
        if fisher is not None:
            text += f", and {wald} for a parameter without a profile"
        text = (
            f"{text}. Identifiable is the classification of the profile: yes, no "
            "(LB) or no (UB) if the profile does not reach the threshold towards "
            "the lower or the upper bound, no if it reaches it on neither side, "
            "structural if it is flat."
        )
        if undetermined:
            text += (
                f" A side is {NOT_DETERMINED} if the profile does not reach the "
                "threshold on it and has points which failed or did not converge, "
                "so its end is not proven; such a profile is not classified (-)."
            )
        return text
    if fisher is not None:
        return (
            f"The {level:.0%} confidence intervals are {wald}; the identifiability "
            "was not analysed with the profile likelihood."
        )
    return (
        "No confidence intervals were computed and the identifiability was not "
        "analysed."
    )


def _parameters(
    report: FitReport, context: ReportContext, best: ParameterSet, fit_id: str
) -> Section:
    """Write the table of the fitted parameters."""
    identifiability = report.identifiability
    fisher = report.fisher
    profiles = identifiability.profiles if identifiability is not None else {}
    fisher_df = fisher.summary_df.set_index("parameter") if fisher is not None else None
    if identifiability is not None:
        level = identifiability.settings.alpha
    elif fisher is not None:
        level = fisher.alpha
    else:
        level = 0.95
    units = report.problem.parameter_units
    rows = []
    for parameter in report.problem.parameters:
        pid = parameter.pid
        value = _g(best.values.get(pid))
        interval = _interval(profiles.get(pid), fisher_df, pid)
        if interval is not None:
            value = f"{value} [{interval[0]}, {interval[1]}]"
        profile = profiles.get(pid)
        identifiable = (
            IDENTIFIABLE[profile.identifiability]
            if profile is not None and profile.identifiability is not None
            else "-"
        )
        unit = units.get(pid)
        rows.append(
            (
                Code(pid),
                describe_parameter(report.problem, pid, context),
                value,
                _unit(unit),
                f"[{_g(parameter.lower_bound)}, {_g(parameter.upper_bound)}]",
                identifiable,
            )
        )

    undetermined = any(
        profile.identifiability is None
        and (profile.ci_lower is None or profile.ci_upper is None)
        for profile in profiles.values()
    )
    ci = _interval_text(level, bool(profiles), fisher, undetermined)
    cost = f", cost {_g(best.cost)}" if best.cost is not None else ""
    table = Table(
        caption=_caption(
            f"Parameters of the fit {fit_id}, the parameter set {best.sid}{cost}, "
            f"with their bounds in the optimization. {ci}"
        ),
        header=(
            "Parameter",
            "Description",
            f"Value [{level:.0%} CI]",
            "Unit",
            "Bounds",
            "Identifiable",
        ),
        rows=tuple(rows),
        label=_label(f"tab:parameters:{fit_id}"),
        align="llrlrl",
    )
    return _section("Parameters", "results:parameters", [table])


def _data(report: FitReport, best: ParameterSet, fit_id: str) -> Section:
    """Write the table of the fit mappings and data points per study and kind."""
    df = report.metrics_mappings_df()
    df = df[df.parameter_set == best.sid]

    def cell(of: pd.DataFrame) -> str:
        return f"{len(of)} ({int(of.n.sum())})" if len(of) else "-"

    rows: list[tuple[str | Code | Math, ...]] = []
    for experiment, of_experiment in df.groupby("experiment", sort=False):
        rows.append(
            (
                str(experiment),
                *(
                    cell(of_experiment[of_experiment.kind == k.value])
                    for k in DATA_KINDS
                ),
                cell(of_experiment),
            )
        )
    rows.append(
        ("Total", *(cell(df[df.kind == k.value]) for k in DATA_KINDS), cell(df))
    )
    table = Table(
        caption=_caption(
            f"Data of the fit {fit_id}: the number of fit mappings and, in "
            "parentheses, of data points per study and kind. The training data "
            "enters the cost, the validation data and the outliers are evaluated "
            "only."
        ),
        header=("Study", *(k.value.capitalize() for k in DATA_KINDS), "Total"),
        rows=tuple(rows),
        label=_label(f"tab:data:{fit_id}"),
        align="lrrrr",
    )
    return _section("Data", "results:data", [table])


#: the metrics of the table of the metrics, column of `metrics_df` and header
METRICS: tuple[tuple[str, str], ...] = (
    ("n", "n"),
    ("cost", "cost"),
    ("MSE", "MSE"),
    ("RMSE", "RMSE"),
    ("NRMSE", "NRMSE"),
    ("RMSE_w", "RMSE_w"),
    ("R2", "R²"),
    ("within_2fold", "2-fold [%]"),
    ("AIC", "AIC"),
    ("BIC", "BIC"),
)


def _metrics(report: FitReport, best: ParameterSet, fit_id: str) -> Section:
    """Write the table of the metrics per kind of data."""
    df = report.metrics_df()
    df = df[df.parameter_set == best.sid]
    rows: list[tuple[str | Code | Math, ...]] = []
    for _, row in df.iterrows():
        cells: list[str | Code | Math] = [str(row["kind"])]
        for column, _ in METRICS:
            if column == "n":
                cells.append(str(int(row["n"])))
            elif column in ("AIC", "BIC") and math.isfinite(float(row[column])):
                # only their differences are read, to one decimal
                cells.append(f"{float(row[column]):.1f}")
            elif column == "within_2fold" and math.isfinite(float(row[column])):
                # a percentage, to one decimal
                cells.append(f"{float(row[column]):.1f}")
            else:
                cells.append(_g(row[column]))
        rows.append(tuple(cells))
    table = Table(
        caption=_caption(
            f"Goodness of fit of the fit {fit_id} per kind of data, the parameter "
            f"set {best.sid}. The cost is defined on the training data only, "
            "2-fold is the percentage of the data points with a positive value "
            "and prediction whose prediction is within twofold of the data."
        ),
        header=("Kind", *(header for _, header in METRICS)),
        rows=tuple(rows),
        label=_label(f"tab:metrics:{fit_id}"),
        align="l" + "r" * len(METRICS),
    )
    return _section("Goodness of fit", "results:metrics", [table])


def _figures(report: FitReport, plots_dir: Path, fit_id: str) -> Section | None:
    """Write the figures of the fit whose files exist, `None` if none exists."""
    blocks: list[Block] = []
    for key in FIGURES:
        name = f"{key}.{report.image_format}"
        if not (Path(plots_dir) / name).exists():
            continue
        caption: list[Inline] = [Text(f"{report.PLOT_CAPTIONS[key]}, fit {fit_id}.")]
        if key == "goodness_of_fit":
            statement = report.twofold_statement()
            if statement:
                caption.append(Text(f" {statement}."))
        hint = report.PLOT_HINTS.get(key, "")
        if hint:
            caption.extend([Text(" "), *_markup(hint)])
        blocks.append(
            Figure(
                path=f"plots/{name}",
                caption=tuple(caption),
                label=_label(f"fig:{key}:{fit_id}"),
            )
        )
    if not blocks:
        return None
    return _section("Figures", "results:figures", blocks)


def _provenance(report: FitReport, fit_id: str) -> Section:
    """Write the table of the provenance, "not recorded" without a provenance."""
    result = report.opt_result
    provenance = result.provenance if result is not None else None
    if provenance is None:
        # a fit before the provenance was stored: nothing is invented
        table = Table(
            caption=_caption(
                f"Provenance of the fit {fit_id}: it was not recorded with the fit."
            ),
            header=("Item", "Value"),
            rows=(
                ("Fit", Code(fit_id)),
                ("Created", NOT_RECORDED),
                ("Seed", NOT_RECORDED),
                ("Versions", NOT_RECORDED),
                ("Repository", NOT_RECORDED),
            ),
            label=_label(f"tab:provenance:{fit_id}"),
            align="ll",
        )
        return _section("Provenance", "results:provenance", [table])
    rows: list[tuple[str | Code | Math, ...]] = [
        ("Fit", Code(fit_id)),
        ("Created", _created(provenance.created)),
        ("Seed", str(provenance.seed) if provenance.seed is not None else "none"),
    ]
    rows.extend(
        (PACKAGE_NAMES.get(package, package), version)
        for package, version in provenance.versions.items()
    )
    git = provenance.git
    if git is None:
        rows.append(("Repository", "not under version control"))
    else:
        rows.extend(
            [
                ("Repository", Code(git.root)),
                ("Commit", Code(git.sha)),
                ("Uncommitted changes", "yes" if git.dirty else "no"),
            ]
        )
    caption: list[str | Inline] = [
        f"Provenance of the fit {fit_id}: the versions it ran with and the "
        "revision of its models and data"
    ]
    if provenance.files:
        # the checksums are in the table of the methods, in full
        caption.extend(
            [
                "; the SHA-256 checksums of the models and data are listed in ",
                Ref(_label(f"tab:checksums:{fit_id}")),
            ]
        )
    caption.append(".")
    table = Table(
        caption=_caption(*caption),
        header=("Item", "Value"),
        rows=tuple(rows),
        label=_label(f"tab:provenance:{fit_id}"),
        align="ll",
    )
    return _section("Provenance", "results:provenance", [table])
