"""The scorecard across models (`sbmlsim.credibility.scorecard`).

The models are assessments of the declaration of the toy package whose renal
context pairs its arms (the toy extension `arms`), with the renal arm
predicted by a factor.
"""

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.credibility import CredibilityAssessment, Verdict, assess_table
from sbmlsim.credibility.scorecard import (
    SCORECARD_COLUMNS,
    Regression,
    compare,
    load_records,
    pool,
    scorecard,
)
from tests.credibility.tables import PROVENANCE, fit_table
from tests.credibility.test_assessment import _with_arms
from tests.credibility_toy.extensions import ARMS

DATA = Path(__file__).parents[1] / "data" / "credibility"


def _assessment(model: str, renal_factor: float = 1.1) -> CredibilityAssessment:
    """Get an assessment of the toy package as the one of a model."""
    fits = fit_table(factors={("CredA", "fm_renal"): renal_factor}, times=True)
    provenance = dataclasses.replace(PROVENANCE, model=model)
    return assess_table(_with_arms(), fits, provenance, extensions=[ARMS])


def _write(directory: Path, *assessments: CredibilityAssessment) -> Path:
    for assessment in assessments:
        folder = directory / assessment.provenance.model / "credibility"
        folder.mkdir(parents=True)
        assessment.to_json(folder / "credibility.json")
    return directory


def test_load_records(tmp_path: Path) -> None:
    root = _write(tmp_path, _assessment("b"), _assessment("a"))
    records = load_records([root])
    assert list(records) == ["a", "b"]
    single = load_records([root / "a" / "credibility" / "credibility.json"])
    assert list(single) == ["a"]


def test_load_records_refuses_two_records_of_a_model(tmp_path: Path) -> None:
    _write(tmp_path / "one", _assessment("a"))
    _write(tmp_path / "two", _assessment("a"))
    with pytest.raises(ValueError, match="both of the model 'a'"):
        load_records([tmp_path])
    with pytest.raises(FileNotFoundError, match="No record"):
        load_records([])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_records([tmp_path / "nothing"])


def test_the_scorecard() -> None:
    table = scorecard(
        {"a": _assessment("a"), "b": _assessment("b", 2.0)}, extensions=[ARMS]
    )
    assert list(table.columns[: len(SCORECARD_COLUMNS)]) == list(SCORECARD_COLUMNS)
    # the columns of the extension follow the core, the rules follow them
    assert table.columns[len(SCORECARD_COLUMNS)] == "arms"
    assert len(table) == 6
    renal = table[table["context"] == "renal"].set_index("model")
    assert renal.loc["a", "verdict"] == Verdict.SUPPORTED.value
    assert renal.loc["b", "verdict"] == Verdict.NOT_SUPPORTED.value
    assert renal.loc["a", "arms_within"] == 100.0
    assert renal.loc["b", "arms_within"] == 0.0
    assert renal.loc["a", "arms"] == 1
    assert "within_2fold" in table.columns
    # without the extension the scorecard has the rule of the extension, not
    # its column
    plain = scorecard({"a": _assessment("a")})
    assert "arms" not in plain.columns
    assert "arms_within" in plain.columns


def test_compare_finds_the_regressions() -> None:
    baseline = scorecard({"a": _assessment("a"), "b": _assessment("b")})
    current = scorecard({"a": _assessment("a", 2.0)})
    regressions = compare(baseline, current)
    assert (
        Regression(
            model="a",
            context="renal",
            what="verdict",
            baseline="supported",
            current="not_supported",
        )
        in regressions
    )
    assert (
        Regression(
            model="a",
            context="renal",
            what="arms_within",
            baseline="100",
            current="0",
        )
        in regressions
    )
    missing = [r for r in regressions if r.what == "missing"]
    assert {(r.model, r.context) for r in missing} == {
        ("b", "describe"),
        ("b", "dose"),
        ("b", "renal"),
    }
    assert str(missing[0]) == "b/describe: missing present -> missing"
    assert compare(baseline, baseline) == []


def test_an_extension_judges_its_columns(tmp_path: Path) -> None:
    """The column of the extension is no rule, its regressions judge it, also from a TSV."""
    baseline = scorecard({"a": _assessment("a")}, extensions=[ARMS])
    path = tmp_path / "scorecard.tsv"
    baseline.to_csv(path, sep="\t", index=False)
    read = pd.read_csv(path, sep="\t", keep_default_na=False)
    fewer = scorecard({"a": _assessment("a")}, extensions=[ARMS])
    fewer.loc[fewer["context"] == "renal", "arms"] = 0
    for before in (baseline, read):
        (regression,) = compare(before, fewer, extensions=[ARMS])
        assert (regression.what, regression.baseline, regression.current) == (
            "arms",
            "1",
            "0",
        )
    assert compare(read, baseline, extensions=[ARMS]) == []


@pytest.mark.parametrize(
    ("rule", "before", "after", "worse"),
    [
        ("within_2fold", 90.0, 86.0, False),
        ("within_2fold", 90.0, 84.0, True),
        ("GMFE", 1.3, 1.36, False),
        ("GMFE", 1.3, 1.4, True),
        ("AFE", 1.0, 1.04, False),
        ("AFE", 1.0, 0.9, True),
        ("arms_within", 80.0, float("nan"), True),
        ("arms_within", float("nan"), 0.0, False),
    ],
)
def test_the_tolerance_of_a_rule(
    rule: str, before: float, after: float, worse: bool
) -> None:
    row = dict.fromkeys(SCORECARD_COLUMNS, "") | {
        "model": "a",
        "context": "c",
        "verdict": "supported",
        "verified": "",
    }
    baseline = pd.DataFrame([row | {rule: before}])
    current = pd.DataFrame([row | {rule: after}])
    assert bool(compare(baseline, current)) is worse


def test_a_context_which_is_no_longer_verified() -> None:
    row = dict.fromkeys(SCORECARD_COLUMNS, "") | {
        "model": "a",
        "context": "c",
        "verdict": "supported",
    }
    baseline = pd.DataFrame([row | {"verified": "True"}])
    current = pd.DataFrame([row | {"verified": "False"}])
    (regression,) = compare(baseline, current)
    assert regression.what == "verified"


def test_a_baseline_of_pkpdsim_0_19() -> None:
    """The columns of the extensions of a baseline of pkpdsim are no rules."""
    data = json.loads((DATA / "assessment-0.19.json").read_text(encoding="utf-8"))
    record = CredibilityAssessment.from_dict(data)
    current = scorecard({"credibility": record})
    baseline = current.assign(temporal_met="True", temporal_points=5, inputs_differ=0)
    assert compare(baseline, current) == []


def test_the_pooling_of_an_extension() -> None:
    records = {"a": _assessment("a"), "b": _assessment("b", 2.0)}
    pooled = pool(records, ARMS, {"field": "health"})
    assert pooled is not None
    assert list(pooled["model"]) == ["a", "b"]
    assert list(pooled["n_pairs"]) == [1, 1]
