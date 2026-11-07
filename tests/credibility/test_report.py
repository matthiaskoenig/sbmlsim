"""The outputs and the report of the credibility assessment (`sbmlsim.credibility.report`)."""

import dataclasses
import re
from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.credibility import (
    CredibilityAssessment,
    Provenance,
    QuantityRecord,
    Verdict,
    assess_table,
)
from sbmlsim.credibility.extensions import CONTEXT_BLOCKS, ContextBlock
from sbmlsim.credibility.report import print_summary, render_html, write_outputs
from tests.credibility.tables import fit_table
from tests.credibility_toy.credibility import CREDIBILITY

DATA = Path(__file__).parents[1] / "data" / "credibility"

FILES = {
    "credibility.json",
    "contexts.tsv",
    "evidence.tsv",
    "domain.tsv",
    "quantities.tsv",
    "verification.tsv",
    "cross_validation.tsv",
    "variability.tsv",
    "variability_quantities.tsv",
    "vpc.tsv",
    "uncertainty.tsv",
    "uncertainty_quantities.tsv",
    "uncertainty_sensitivity.tsv",
    "battery.tsv",
    "index.html",
}

PROVENANCE = Provenance(
    model="credibility",
    title="Toy model with every kind of evidence",
    fits=["DECAY_fit"],
    versions={"sbmlsim": "0.0.0"},
    commit="0123abc",
    dirty=False,
    date="2026-11-03T12:00:00+01:00",
)


@pytest.fixture(scope="module")
def assessment() -> CredibilityAssessment:
    """Get the assessment of the toy package with a context of every verdict."""
    assessed = assess_table(CREDIBILITY, fit_table(), PROVENANCE)
    verdicts = {c.verdict for c in assessed.contexts}
    extra = [
        dataclasses.replace(assessed.contexts[1], id=f"other{index}", verdict=verdict)
        for index, verdict in enumerate(v for v in Verdict if v not in verdicts)
    ]
    return dataclasses.replace(assessed, contexts=[*assessed.contexts, *extra])


def _tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def test_write_outputs(assessment: CredibilityAssessment, tmp_path: Path) -> None:
    output_dir = tmp_path / "credibility"
    assert write_outputs(assessment, output_dir) == output_dir
    assert {p.name for p in output_dir.iterdir()} == FILES
    assert (
        CredibilityAssessment.from_json(output_dir / "credibility.json").to_dict()
        == assessment.to_dict()
    )

    contexts = _tsv(output_dir / "contexts.tsv")
    assert list(contexts["id"]) == [c.id for c in assessment.contexts]
    assert list(contexts["verdict"]) == [c.verdict.value for c in assessment.contexts]
    assert list(contexts.columns[:11]) == [
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
    ]
    assert list(contexts["minimum_points"]) == [
        c.minimum_points for c in assessment.contexts
    ]
    assert set(contexts["uncovered_quantities"]) == {""}
    dose = contexts.set_index("id").loc["dose"]
    assert dose["within_2fold_limit"] == ">= 80"
    assert dose["GMFE_limit"] == "<= 1.5"
    assert float(dose["GMFE"]) == pytest.approx(1.0)

    evidence = _tsv(output_dir / "evidence.tsv")
    n_evidence = len(assessment.evidence) + len(assessment.limitations)
    assert len(evidence) == n_evidence
    assert {"context", "name", "role", "n", "GMFE", "AFE", "within_2fold"} <= set(
        evidence.columns
    )
    (limitation,) = evidence[evidence["role"] == "limitation"].itertuples()
    assert limitation.context == ""

    domain = _tsv(output_dir / "domain.tsv")
    assert list(domain.columns) == [
        "context",
        "field",
        "value",
        "claimed",
        "coverage",
        "n",
    ]
    assert len(domain) == len(assessment.coverage)

    quantities = _tsv(output_dir / "quantities.tsv")
    assert list(quantities.columns) == [
        "context",
        "quantity",
        "description",
        "ratio",
        "coverage",
        "n_points",
    ]
    assert len(quantities) == sum(len(c.quantities) for c in assessment.contexts)
    (ratio,) = quantities[quantities["quantity"] == "A_ratio"].itertuples()
    assert (ratio.context, ratio.ratio, ratio.coverage, ratio.n_points) == (
        "renal",
        True,
        "",
        0,
    )
    renal = quantities[(quantities["context"] == "renal") & ~quantities["ratio"]]
    assert list(renal["coverage"]) == ["covered"]
    assert list(renal["n_points"]) == [5]

    html = (output_dir / "index.html").read_text(encoding="utf-8")
    for context in assessment.contexts:
        assert context.id in html
    for verdict in Verdict:
        assert f">{verdict.label}<" in html
    assert "CredB" in html and "fm_zero" in html  # the unclaimed validation data
    assert "0123abc" in html  # the provenance
    external = re.compile(r"<(?:script|link)\b[^>]*(?:src|href)=\"?https?://", re.I)
    assert not external.search(html)


def test_the_report_shows_what_is_not_assessed(
    assessment: CredibilityAssessment,
) -> None:
    html = render_html(assessment)
    # the ratio quantity of renal and the domain statement of dose
    assert "A_ratio" in html
    assert "ratio, no pairing declared" in html
    assert "the single doses of the data" in html
    assert "statement, not assessed" in html


def test_the_report_shows_what_decides(assessment: CredibilityAssessment) -> None:
    html = render_html(assessment)
    overview = html[html.index('id="overview"') : html.index('id="contexts"')]
    assert "decided on" in overview
    rows = overview.split("<tr")
    for context, role in (("describe", "calibration"), ("dose", "validation")):
        (row,) = (r for r in rows if f'href="#cou-{context}"' in r)
        assert f'class="decided">{role}<' in row


def test_the_report_shows_the_quantity_coverage_and_the_minimum(
    assessment: CredibilityAssessment,
) -> None:
    uncovered = dataclasses.replace(
        assessment.contexts[1],
        id="uncovered",
        quantities=[
            QuantityRecord(
                id="[B]",
                description="the concentration of B",
                ratio=False,
                coverage="uncovered",
                n_points=0,
            )
        ],
    )
    html = render_html(dataclasses.replace(assessment, contexts=[uncovered]))
    assert 'class="state uncovered">uncovered<' in html
    html = render_html(assessment)
    assert 'class="state covered">covered<' in html
    renal = html[html.index('id="cou-renal"') :]
    assert "<dd>5 (minimum 20)" in renal
    assert 'class="state below">below the minimum<' in renal
    dose = html[html.index('id="cou-dose"') : html.index('id="cou-renal"')]
    assert "<dd>10 (minimum 10)" in dose
    assert "below the minimum" not in dose
    overview = html[html.index('id="overview"') : html.index('id="contexts"')]
    assert "minimum points" in overview
    assert (
        "by default 5 for a low risk, 10 for a medium risk and 20 for a high risk."
        in overview
    )
    (row,) = (r for r in overview.split("<tr") if 'href="#cou-renal"' in r)
    assert '<td class="num">20</td>' in row


def test_the_report_of_a_record_of_0_17(assessment: CredibilityAssessment) -> None:
    """A record without the minimum and the coverage of the quantities renders."""
    old = dataclasses.replace(
        assessment,
        contexts=[
            dataclasses.replace(
                c,
                minimum_points=0,
                quantities=[
                    q | {"coverage": None, "n_points": 0} for q in c.quantities
                ],
            )
            for c in assessment.contexts
        ],
    )
    html = render_html(old)
    dose = html[html.index('id="cou-dose"') : html.index('id="cou-renal"')]
    assert "<dt>evaluable points</dt><dd>10</dd>" in dose
    assert "below the minimum" not in html
    assert 'class="state covered"' not in html


def test_the_verdict_is_a_word_with_a_colour(assessment: CredibilityAssessment) -> None:
    html = render_html(assessment)
    for verdict in Verdict:
        assert f'class="verdict {verdict.value}">{verdict.label}<' in html


def test_write_outputs_replaces_its_own_directory(
    assessment: CredibilityAssessment, tmp_path: Path
) -> None:
    write_outputs(assessment, tmp_path)
    (tmp_path / "stale.txt").write_text("of the first run", encoding="utf-8")
    (tmp_path / "figures").mkdir()
    fewer = dataclasses.replace(assessment, contexts=assessment.contexts[:-1])
    write_outputs(fewer, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == FILES
    assert len(_tsv(tmp_path / "contexts.tsv")) == len(assessment.contexts) - 1


def test_write_outputs_refuses_a_foreign_directory(
    assessment: CredibilityAssessment, tmp_path: Path
) -> None:
    (tmp_path / "other.txt").write_text("not an assessment", encoding="utf-8")
    with pytest.raises(FileExistsError, match=r"holds no credibility\.json"):
        write_outputs(assessment, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"other.txt"}
    file = tmp_path / "other.txt"
    with pytest.raises(FileExistsError):
        write_outputs(assessment, file)


def test_write_outputs_into_an_empty_directory(
    assessment: CredibilityAssessment, tmp_path: Path
) -> None:
    write_outputs(assessment, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == FILES


def test_print_summary(
    assessment: CredibilityAssessment, capsys: pytest.CaptureFixture[str]
) -> None:
    print_summary(assessment)
    out = capsys.readouterr().out
    assert "decided on" in out
    for context in assessment.contexts:
        assert context.id in out
        assert context.verdict.label in out
    # the table fits a console of windows, 79 columns
    assert max(len(line) for line in out.splitlines()) <= 79
    (describe,) = (line for line in out.splitlines() if "describe " in line)
    assert "calibration" in describe
    (dose,) = (line for line in out.splitlines() if line.startswith("dose "))
    assert "validation" in dose
    assert dose.endswith(" 10/10")
    (renal,) = (line for line in out.splitlines() if "renal " in line)
    assert renal.endswith(" 5/20")


def test_print_summary_of_a_record_of_0_17(
    assessment: CredibilityAssessment, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record without the minimum shows the evaluable points alone."""
    old = dataclasses.replace(
        assessment,
        contexts=[
            dataclasses.replace(c, minimum_points=0) for c in assessment.contexts
        ],
    )
    print_summary(old)
    out = capsys.readouterr().out
    assert "/" not in out.split("evaluable", 1)[1]
    (dose,) = (line for line in out.splitlines() if line.startswith("dose "))
    assert dose.endswith(" 10")


def test_print_summary_escapes_the_id(
    assessment: CredibilityAssessment, capsys: pytest.CaptureFixture[str]
) -> None:
    """An id of a record is text, not the markup of the console."""
    marked = dataclasses.replace(assessment.contexts[0], id="[bold]x")
    print_summary(dataclasses.replace(assessment, contexts=[marked]))
    assert "[bold]x" in capsys.readouterr().out


def _arms() -> CredibilityAssessment:
    """Get the assessment of the declaration with the arms of the toy extension."""
    from tests.credibility.test_assessment import _with_arms
    from tests.credibility_toy.extensions import ARMS

    return assess_table(
        _with_arms(),
        fit_table(),
        PROVENANCE,
        extensions=[ARMS],
        prepared={"arms": ["CredA", "CredB"]},
    )


def test_the_blocks_of_an_extension_follow_their_block_of_the_core(
    tmp_path: Path,
) -> None:
    from tests.credibility_toy.extensions import ARMS

    assessment = _arms()
    html = render_html(assessment, [ARMS])
    renal = html[html.index('id="cou-renal"') : html.index('id="limitations"')]
    acceptance = renal.index("<h4>Acceptance</h4>")
    rule = renal.index('arms_within <span class="soft">arms of a ratio</span>')
    pairs = renal.index('<table class="arms-pairs">')
    verification = renal.index("<h4>Verification</h4>")
    factors = renal.index("<h4>Factors of V&amp;V 40</h4>")
    note = renal.index('<p class="arms-note">1 pairs of arms.</p>')
    extrapolation = renal.index("<h4>Extrapolation</h4>")
    # the rule of the extension is a row of the acceptance table, its pairs
    # follow the table, its note follows the factors
    assert acceptance < rule < pairs < verification < factors < note < extrapolation
    assert renal.count("arms-pairs") == 1
    # the cards of the contexts without a section have no block
    dose = html[html.index('id="cou-dose"') : html.index('id="cou-renal"')]
    assert "arms-pairs" not in dose and "arms-note" not in dose
    # the section of the extension has its entry in the navigation and lies
    # after the sections of the core and before the provenance
    nav = html[: html.index('id="overview"')]
    assert nav.index('href="#verification"') < nav.index(
        'href="#arms">Arms of the ratios<'
    )
    assert nav.index('href="#arms"') < nav.index('href="#provenance"')
    section = html.index('<section id="arms">')
    assert html.index('<section id="verification">') < section
    assert section < html.index('<section id="provenance">')
    assert '<p class="arms-studies">CredA, CredB</p>' in html
    # the HTML of an extension is not escaped, the text of the core is
    output = write_outputs(assessment, tmp_path / "out", [ARMS])
    arms = pd.read_csv(output / "arms.tsv", sep="\t")
    assert list(arms["numerator"]) == ["fm_renal"]
    assert {p.name for p in output.iterdir()} == FILES | {"arms.tsv"}


def test_the_report_without_the_extension_is_complete(tmp_path: Path) -> None:
    """The report of an assessment with sections of extensions renders without them."""
    assessment = _arms()
    html = render_html(assessment)
    assert "arms-pairs" not in html and 'id="arms"' not in html
    # the rule of the extension is still a row of its context, named by the
    # extension
    renal = html[html.index('id="cou-renal"') : html.index('id="limitations"')]
    assert 'arms_within <span class="soft">arms</span>' in renal
    for block in ("Acceptance", "Verification", "Factors of V&amp;V 40"):
        assert f"<h4>{block}</h4>" in renal
    output = write_outputs(assessment, tmp_path / "out")
    assert {p.name for p in output.iterdir()} == FILES
    contexts = pd.read_csv(output / "contexts.tsv", sep="\t")
    assert "arms_within" in contexts.columns


def test_a_block_follows_a_block_of_the_core() -> None:
    assert CONTEXT_BLOCKS == (
        "acceptance",
        "uncertainty",
        "variability",
        "verification",
        "factors",
    )
    with pytest.raises(ValueError, match="follows one of"):
        ContextBlock("<p/>", after="quantities")


def test_the_report_of_a_record_of_pkpdsim_0_19(tmp_path: Path) -> None:
    record = CredibilityAssessment.from_json(DATA / "assessment-0.19.json")
    output = write_outputs(record, tmp_path / "out")
    html = (output / "index.html").read_text(encoding="utf-8")
    # the checks of the PK battery of pkpdsim are shown with their labels
    assert "mass balance" in html
    assert "superposition" in html
    battery = pd.read_csv(output / "battery.tsv", sep="\t", keep_default_na=False)
    assert battery.set_index("check").loc["zero_dose", "factor"] == (
        "numerical_code_verification"
    )
    assert {p.name for p in output.iterdir()} == FILES
