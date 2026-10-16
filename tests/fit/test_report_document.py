"""Tests of the report document of a fit, its methods and its results."""

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import (
    Identifiability,
    IdentifiabilityResult,
    ParameterProfile,
    ProfileSettings,
)
from sbmlsim.fit.methods import ReportContext
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.report_document import fit_document, results_section
from sbmlsim.report.document import Block, Code, Figure, Ref, Section, Table
from sbmlsim.report.render import HTMLRenderer, LaTeXRenderer, TypstRenderer

Reported = tuple[FitReport, Path]


def _blocks(blocks: tuple[Block, ...]) -> list[Block]:
    """Get all blocks of a tree of blocks."""
    out: list[Block] = []
    for block in blocks:
        out.append(block)
        if isinstance(block, Section):
            out.extend(_blocks(block.blocks))
    return out


def _refs(blocks: tuple[Block, ...]) -> list[str]:
    """Get the labels every inline reference of the blocks points to."""
    labels = []
    for block in _blocks(blocks):
        content = getattr(block, "content", ()) or getattr(block, "caption", ())
        labels.extend(span.label for span in content if isinstance(span, Ref))
    return labels


def _table(section: Section, key: str) -> Table:
    """Get the first table of the sub-section with the key."""
    (sub,) = [b for b in section.blocks if isinstance(b, Section) and b.key == key]
    return next(b for b in sub.blocks if isinstance(b, Table))


def _copy(report: FitReport) -> FitReport:
    """Get a copy of the report which shares its problem, with its own result."""
    other = copy.copy(report)
    other.opt_result = copy.deepcopy(report.opt_result)
    return other


def _render_all(report: FitReport, plots_dir: Path) -> str:
    doc = fit_document(report, ReportContext(title="Toy"), plots_dir)
    TypstRenderer().render(doc)
    LaTeXRenderer().render(doc)
    return HTMLRenderer().render(doc)


def test_results_tables_and_missing_figures(
    fitted_report, tmp_path
) -> None:  # FitReport of a size=1 fit, no identifiability, created into tmp_path
    report, plots_dir = fitted_report
    doc = fit_document(report, ReportContext(title="Toy"), plots_dir)
    html = HTMLRenderer().render(doc)
    assert "Value [95% CI]" in html
    assert "waterfall" not in html  # size=1 has no waterfall plot
    assert "tab:provenance" in html


def test_the_sections_of_the_results(fitted_report: Reported) -> None:
    report, plots_dir = fitted_report
    section = results_section(report, ReportContext(), plots_dir)
    assert [b.key for b in section.blocks if isinstance(b, Section)] == [
        "results:parameters",
        "results:data",
        "results:metrics",
        "results:figures",
        "results:provenance",
    ]
    assert report.opt_result is not None
    fit_id = report.opt_result.sid
    labels = [b.label for b in _blocks(section.blocks) if isinstance(b, Table)]
    assert labels == [
        f"tab:parameters:{fit_id}",
        f"tab:data:{fit_id}",
        f"tab:metrics:{fit_id}",
        f"tab:provenance:{fit_id}",
    ]
    figures = [b for b in _blocks(section.blocks) if isinstance(b, Figure)]
    assert [f.label for f in figures] == [
        f"fig:traces:{fit_id}",
        f"fig:goodness_of_fit:{fit_id}",
        f"fig:bland_altman:{fit_id}",
    ]
    for figure in figures:
        assert (plots_dir.parent / figure.path).exists()
    # the goodness of fit states the percentage of the points within twofold
    (gof,) = [f for f in figures if f.label == f"fig:goodness_of_fit:{fit_id}"]
    text = "".join(getattr(span, "text", "") for span in gof.caption)
    training = report.metrics(report.reference_set).summary(kind=MappingKind.TRAINING)
    assert f"{training['within_2fold']:.1f}% of the training" in text

    doc = fit_document(report, ReportContext(), plots_dir)
    assert doc.title == fit_id
    assert [b.key for b in doc.blocks] == ["methods", "results"]
    keys = [r.key for r in doc.references]
    assert "Koenig2026_sbmlsim" in keys
    sbmlsim = next(r for r in doc.references if r.key == "Koenig2026_sbmlsim")
    assert report.opt_result.provenance is not None
    version = report.opt_result.provenance.versions["sbmlsim"]
    assert sbmlsim.fields["note"] == f"Version {version}"


def test_the_tables(fitted_report: Reported) -> None:
    report, plots_dir = fitted_report
    section = results_section(report, ReportContext(), plots_dir)

    parameters = _table(section, "results:parameters")
    assert parameters.header == (
        "Parameter",
        "Description",
        "Value [95% CI]",
        "Unit",
        "Bounds",
        "Identifiable",
    )
    assert len(parameters.rows) == len(report.problem.pids)
    best = report.parameter_sets[0]
    for row, pid in zip(parameters.rows, report.problem.pids, strict=True):
        # without a profile or a Fisher information the value alone
        assert row[2] == f"{best.values[pid]:.3g}"
        assert row[5] == "-"

    data = _table(section, "results:data")
    assert data.header == ("Study", "Training", "Validation", "Outlier", "Total")
    assert data.rows[-1][0] == "Total"
    df = report.metrics_mappings_df()
    df = df[df.parameter_set == best.sid]
    total = f"{len(df)} ({int(df.n.sum())})"
    assert data.rows[-1][-1] == total
    assert all(len(row) == 5 for row in data.rows)

    metrics = _table(section, "results:metrics")
    assert metrics.header == (
        "Kind",
        "n",
        "cost",
        "MSE",
        "RMSE",
        "NRMSE",
        "RMSE_w",
        "R²",
        "2-fold [%]",
        "AIC",
        "BIC",
    )
    kinds = [row[0] for row in metrics.rows]
    assert kinds[0] == "training"
    # the percentage of the data points within twofold, to one decimal
    within = report.metrics_df()
    within = within[within.parameter_set == best.sid].set_index("kind")
    column = metrics.header.index("2-fold [%]")
    assert metrics.rows[0][column] == f"{within.loc['training', 'within_2fold']:.1f}"
    # the cost is defined on the training data only
    assert all(row[2] == "-" for row in metrics.rows[1:])
    assert metrics.rows[0][2] != "-"

    provenance = _table(section, "results:provenance")
    items = [row[0] for row in provenance.rows]
    assert items[:3] == ["Fit", "Created", "Seed"]
    assert provenance.rows[2][1] == "1234"
    assert "sbmlsim" in items


def test_a_missing_figure_is_left_out(fitted_report: Reported, tmp_path: Path) -> None:
    report, plots_dir = fitted_report
    plots = tmp_path / "plots"
    shutil.copytree(plots_dir, plots)
    (plots / f"goodness_of_fit.{report.image_format}").unlink()
    html = _render_all(report, plots)
    assert "fig:goodness_of_fit" not in html
    assert "fig:bland_altman" in html

    # no figure at all: no section of the figures
    shutil.rmtree(plots)
    plots.mkdir()
    section = results_section(report, ReportContext(), plots)
    keys = [b.key for b in section.blocks if isinstance(b, Section)]
    assert "results:figures" not in keys
    assert "<figure" not in _render_all(report, plots)


def test_every_reference_resolves(fitted_report: Reported) -> None:
    """A fit of one run without identifiability references what it has."""
    report, plots_dir = fitted_report
    doc = fit_document(report, ReportContext(), plots_dir)
    labels = {getattr(b, "label", "") for b in _blocks(doc.blocks)} - {""}
    refs = _refs(doc.blocks)
    assert refs, "the methods reference the checksums of the provenance"
    assert set(refs) <= labels
    for figure in (b for b in _blocks(doc.blocks) if isinstance(b, Figure)):
        assert (plots_dir.parent / figure.path).exists()
    html = _render_all(report, plots_dir)
    # a checksum is shortened to fit on a page
    assert report.opt_result is not None and report.opt_result.provenance is not None
    digest = next(iter(report.opt_result.provenance.files.values()))
    assert f"{digest[:12]}..." in html
    assert digest not in html


def test_without_provenance(fitted_report: Reported) -> None:
    """A record without provenance has a table which invents nothing."""
    report, plots_dir = fitted_report
    report = _copy(report)
    assert report.opt_result is not None
    fit_id = report.opt_result.sid
    report.opt_result.provenance = None
    section = results_section(report, ReportContext(), plots_dir)
    table = _table(section, "results:provenance")
    assert table.label == f"tab:provenance:{fit_id}"
    assert table.rows == (
        ("Fit", Code(fit_id)),
        ("Created", "not recorded"),
        ("Seed", "not recorded"),
        ("Versions", "not recorded"),
        ("Repository", "not recorded"),
    )
    assert not any(isinstance(span, Ref) for span in table.caption)

    doc = fit_document(report, ReportContext(), plots_dir)
    labels = {getattr(b, "label", "") for b in _blocks(doc.blocks)} - {""}
    assert set(_refs(doc.blocks)) <= labels
    assert not any(label.startswith("tab:checksums") for label in labels)
    html = _render_all(report, plots_dir)
    assert "was not recorded" in html
    assert f"tab:provenance:{fit_id}" in html
    assert "tab:checksums" not in html
    # no seed sentence in the methods
    assert "(seed " not in html
    sbmlsim = next(r for r in doc.references if r.key == "Koenig2026_sbmlsim")
    assert "note" not in sbmlsim.fields


def _profile(pid: str, k: int) -> ParameterProfile:
    """Get a profile of three points of a problem of k parameters."""
    return ParameterProfile(
        pid=pid,
        values=np.array([0.5, 1.0, 2.0]),
        costs=np.array([2.0, 1.0, 2.0]),
        paths=np.ones((3, k)),
        converged=np.ones(3, dtype=bool),
        index_optimum=1,
    )


@pytest.mark.parametrize(
    ("ci", "identifiability", "interval", "identifiable"),
    [
        ((0.6, 1.8), Identifiability.IDENTIFIABLE, "[0.6, 1.8]", "yes"),
        ((None, 1.8), Identifiability.NON_IDENTIFIABLE_LOWER, "[LB, 1.8]", "no (LB)"),
        ((0.6, None), Identifiability.NON_IDENTIFIABLE_UPPER, "[0.6, UB]", "no (UB)"),
        ((None, None), Identifiability.NON_IDENTIFIABLE, "[LB, UB]", "no"),
        ((None, None), Identifiability.STRUCTURAL, "[LB, UB]", "structural"),
    ],
)
def test_intervals_from_the_profiles_and_the_fisher_information(
    fitted_report: Reported,
    ci: tuple[float | None, float | None],
    identifiability: Identifiability,
    interval: str,
    identifiable: str,
) -> None:
    report, plots_dir = fitted_report
    report = _copy(report)
    problem = report.problem
    pids = list(problem.pids)
    k = len(pids)
    best = report.parameter_sets[0]

    def value(pid: str) -> str:
        return f"{best.values[pid]:.3g}"

    report.fisher = FisherInformation(
        opid=problem.opid,
        sid=best.sid,
        pids=pids,
        values=np.array([best.values[pid] for pid in pids]),
        scale=ParameterScaleType.LOG10,
        matrix=np.eye(k) * 100.0,
        cost=1.0,
        n=40,
    )
    fisher = report.fisher.summary_df.set_index("parameter")

    # the Fisher information alone: the Wald intervals
    table = _table(
        results_section(report, ReportContext(), plots_dir), "results:parameters"
    )
    assert table.header[2] == "Value [95% CI]"
    for pid, row in zip(pids, table.rows, strict=True):
        lower, upper = fisher.loc[pid, "ci_lower"], fisher.loc[pid, "ci_upper"]
        assert row[2] == f"{value(pid)} [{lower:.3g}, {upper:.3g}]"
        assert row[5] == "-"

    # a profile of the first parameter, the others keep their Wald intervals
    report.identifiability = IdentifiabilityResult(
        opid=problem.opid,
        parameter_set=best,
        parameters=problem.parameters,
        settings=ProfileSettings(alpha=0.9),
        fit_settings=FitSettings(),
        cost=1.0,
        profiles={pids[0]: _profile(pids[0], k)},
    )
    # the analysis evaluates its profiles, the classification is set afterwards
    profile = report.identifiability.profiles[pids[0]]
    profile.ci_lower, profile.ci_upper = ci
    profile.identifiability = identifiability
    table = _table(
        results_section(report, ReportContext(), plots_dir), "results:parameters"
    )
    assert table.header[2] == "Value [90% CI]"
    assert table.rows[0][2] == f"{value(pids[0])} {interval}"
    assert table.rows[0][5] == identifiable
    last = pids[-1]
    lower, upper = fisher.loc[last, "ci_lower"], fisher.loc[last, "ci_upper"]
    assert table.rows[-1][2] == f"{value(last)} [{lower:.3g}, {upper:.3g}]"
    assert table.rows[-1][5] == "-"
    html = _render_all(report, plots_dir)
    assert "at the confidence level 95%" in html


def test_the_document_of_a_fit_of_two_runs(fitted, tmp_path: Path) -> None:
    problem, result = fitted
    report = FitReport.from_optimization_result(
        problem=problem, opt_result=result, size=2, mapping_figures=False
    )
    plots_dir = report.create(output_dir=tmp_path) / "plots"
    html = _render_all(report, plots_dir)
    assert f"fig:waterfall:{result.sid}" in html
    assert "waterfall plot" in html


def test_an_unproven_end_of_a_profile_is_not_determined(
    fitted_report: Reported,
) -> None:
    """A profile with failed points has no class, its open side is not proven."""
    report, plots_dir = fitted_report
    report = _copy(report)
    problem = report.problem
    pids = list(problem.pids)
    k = len(pids)
    best = report.parameter_sets[0]
    report.fisher = FisherInformation(
        opid=problem.opid,
        sid=best.sid,
        pids=pids,
        values=np.array([best.values[pid] for pid in pids]),
        scale=ParameterScaleType.LOG10,
        matrix=np.eye(k) * 100.0,
        cost=1.0,
        n=40,
    )
    profile = _profile(pids[0], k)
    profile.converged = np.array([False, True, True])
    report.identifiability = IdentifiabilityResult(
        opid=problem.opid,
        parameter_set=best,
        parameters=problem.parameters,
        settings=ProfileSettings(),
        fit_settings=FitSettings(),
        cost=1.0,
        profiles={pids[0]: profile},
    )
    # the evaluation keeps a proven end and gives no class
    profile = report.identifiability.profiles[pids[0]]
    assert profile.identifiability is None
    profile.ci_lower, profile.ci_upper = None, 1.8

    table = _table(
        results_section(report, ReportContext(), plots_dir), "results:parameters"
    )
    value = f"{best.values[pids[0]]:.3g}"
    # no Fisher interval for the side the profile does not prove
    assert table.rows[0][2] == f"{value} [not determined, 1.8]"
    assert table.rows[0][5] == "-"
    caption = " ".join(getattr(span, "text", "") for span in table.caption)
    assert "is not determined if" in caption
    html = _render_all(report, plots_dir)
    assert "not determined" in html

    # a classified profile has no such sentence
    profile.identifiability = Identifiability.NON_IDENTIFIABLE_LOWER
    table = _table(
        results_section(report, ReportContext(), plots_dir), "results:parameters"
    )
    assert table.rows[0][2] == f"{value} [LB, 1.8]"
    caption = " ".join(getattr(span, "text", "") for span in table.caption)
    assert "not determined" not in caption
