"""`verify scorecard`: the credibility records of many models in one table."""

from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.credibility.scorecard import scorecard
from sbmlsim.verify.cli import verify
from tests.credibility.test_scorecard import _assessment, _write
from tests.credibility_toy.extensions import ARMS

EXTENSIONS = ["--extensions", "tests.credibility_toy.extensions:ARMS"]


def test_the_scorecard_of_many_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _write(tmp_path / "records", _assessment("b"), _assessment("a"))
    out, qualification = tmp_path / "scorecard.tsv", tmp_path / "qualification.tsv"
    code = verify(
        [
            "scorecard",
            str(records),
            *EXTENSIONS,
            "--pool",
            "arms",
            "--out",
            str(out),
            "--qualification",
            str(qualification),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "2 models" in text
    assert "a/renal:" in text
    table = pd.read_csv(out, sep="\t", keep_default_na=False)
    assert set(table["model"]) == {"a", "b"}
    assert "arms" in table.columns
    assert len(pd.read_csv(qualification, sep="\t")) > 0


def test_the_scorecard_without_extensions_has_the_columns_of_the_core(
    tmp_path: Path,
) -> None:
    records = _write(tmp_path / "records", _assessment("a"))
    out = tmp_path / "scorecard.tsv"
    assert verify(["scorecard", str(records), "--out", str(out)]) == 0
    assert "arms" not in pd.read_csv(out, sep="\t").columns


def test_a_regression_against_a_baseline_exits_with_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _write(tmp_path / "records", _assessment("a"))
    baseline = scorecard({"a": _assessment("a")}, extensions=[ARMS])
    path = tmp_path / "baseline.tsv"
    baseline.to_csv(path, sep="\t", index=False)
    args = ["scorecard", str(records), *EXTENSIONS, "--baseline", str(path)]
    assert verify(args) == 0
    assert "0 regressions" in capsys.readouterr().out
    baseline.loc[baseline["context"] == "renal", "arms"] = 5
    baseline.to_csv(path, sep="\t", index=False)
    assert verify(args) == 1
    assert "1 regressions" in capsys.readouterr().out


def test_the_extensions_of_a_project(tmp_path: Path) -> None:
    records = _write(tmp_path / "records", _assessment("a"))
    out = tmp_path / "scorecard.tsv"
    code = verify(
        [
            "scorecard",
            str(records),
            "--project",
            "tests.verify.projects:WITH_ARMS",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "arms" in pd.read_csv(out, sep="\t").columns


@pytest.mark.parametrize(
    "extra",
    [
        ["--extensions", "tests.credibility_toy.extensions:MISSING"],
        [*EXTENSIONS, "--pool", "other"],
        ["--qualification", "q.tsv"],
        [*EXTENSIONS, *EXTENSIONS],
    ],
)
def test_bad_arguments_exit_with_2(tmp_path: Path, extra: list[str]) -> None:
    records = _write(tmp_path / "records", _assessment("a"))
    assert verify(["scorecard", str(records), *extra]) == 2


def test_records_which_cannot_be_read_exit_with_2(tmp_path: Path) -> None:
    assert verify(["scorecard", str(tmp_path / "missing")]) == 2
    assert verify(["scorecard", str(tmp_path), "--baseline", "none.tsv"]) == 2


def test_the_pool_is_worded_by_the_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default prints every row as columns and values, an extension words them."""
    records = _write(tmp_path / "records", _assessment("a"))
    args = ["scorecard", str(records), *EXTENSIONS, "--pool", "arms"]
    assert verify(args) == 0
    out = capsys.readouterr().out
    assert "qualification by the default:" in out
    assert "model=a" in out or "model=all" in out

    table = pd.DataFrame({"model": ["a", "all"], "value": ["x", "x"], "n": [1, 2]})
    monkeypatch.setattr(type(ARMS), "pool", lambda self, contexts, options: table)
    assert verify(args) == 0
    assert "  model=all, value=x, n=2" in capsys.readouterr().out
    monkeypatch.setattr(
        type(ARMS),
        "pool_lines",
        lambda self, table, options: [
            f"  {r['value']}: {r['n']}"
            for r in table.to_dict("records")
            if r["model"] == "all"
        ],
    )
    assert verify([*args, "--field", "health"]) == 0
    out = capsys.readouterr().out
    assert "qualification by health:" in out
    assert "  x: 2" in out

    empty = table[table["model"] == "none"]
    monkeypatch.setattr(type(ARMS), "pool", lambda self, contexts, options: empty)
    assert verify(args) == 0
    assert "qualification by" not in capsys.readouterr().out


def test_a_bad_extension_object_exits_with_2(tmp_path: Path) -> None:
    records = _write(tmp_path / "records", _assessment("a"))
    bad = ["--extensions", "tests.verify.projects:BROKEN"]
    assert verify(["scorecard", str(records), *bad]) == 2
