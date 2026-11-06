"""The fit and the report of a fit of a project: start, fixed adopted fits, adoption.

The fit of the toy project (`tests.project_toy`), whose problem `DECAY` fits
the rate `k` of the decay in about a second, with its profiles, report and
adoption; the adopted fits of a test are given by replacing the project of
the package for the test.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import libsbml
import pytest

from sbmlsim import run
from sbmlsim.fit import FitParameter
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.result import FIT_ID, OptimizationResult
from sbmlsim.project import Project, write_adopted
from sbmlsim.run import ConfigurationError
from sbmlsim.run.fit_kinds import FitSpec, ReportSpec
from tests.project_toy import PROJECT, TOY_PATH

ROOT = Path(__file__).parents[2]
TOY = "tests.project_toy:PROJECT"

#: one repeat of the fit of the toy project
FIT: dict[str, Any] = {"project": TOY, "subset": "DECAY", "repeats": 1, "seed": 1}

#: the fit adopted with its profiles and its report
ADOPT: dict[str, Any] = {
    **FIT,
    "identifiability": True,
    "profile": {"max_points": 3},
    "report": True,
    "mapping_figures": False,
    "adopt": True,
}

#: the id of a hand-made adopted fit of another subset
OTHER = "OTHER_20260101_000000__abcde"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _status() -> str:
    """Get the changes of the toy project in the checkout."""
    return subprocess.run(
        ["git", "status", "--porcelain", "tests/project_toy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def adopted_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of the adopted fit of the toy project."""
    out = tmp_path_factory.mktemp("adopt") / "out"
    before = _status()
    run.execute("fit", ADOPT, out, workers=1)
    assert _status() == before
    return out


def _with_project(monkeypatch: pytest.MonkeyPatch, project: Project) -> None:
    """Replace the project of the toy package for a test."""
    monkeypatch.setattr("tests.project_toy.PROJECT", project)


def _adopted_project(out: Path) -> Project:
    """Get the toy project whose adopted fits are the ones a fit adopted in `out`."""
    return PROJECT.replace(fits_dir=out / "fits", adopted=out / "adopted.json")


# ----------------------------------------------------------------------------
# adopt
# ----------------------------------------------------------------------------
def test_an_adopted_fit_has_the_id_of_a_fit(adopted_out: Path) -> None:
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    assert _json(adopted_out / "adopted.json") == {"fits": {"DECAY": fit_id}}
    assert FIT_ID.match(fit_id)
    planned = FitSpec.model_validate(_json(adopted_out / "spec.json"))
    assert planned.opid == fit_id
    summary = _json(adopted_out / "summary.json")
    assert summary["adopted"] == f"fits/{fit_id}"
    assert summary["n_parameters"] == 1


def test_the_adopted_record_is_in_the_out(adopted_out: Path) -> None:
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    records = adopted_out / fit_id
    for name in ("identifiability.json", "fisher.json", "report.pdf", "index.html"):
        assert (records / name).exists(), name
    record = adopted_out / "fits" / fit_id
    for name in ("parameters.json", "identifiability.json", "fisher.json"):
        assert (record / name).exists(), name
    assert (record / "report.typ").exists()
    assert not (record / "index.html").exists()
    # the project reads the record once it is committed
    project = _adopted_project(adopted_out)
    assert project.adopted_record(fit_id) == record
    assert project.fit_records()["k"].fit_id == fit_id


def test_the_report_of_the_fit_has_the_context_of_the_project(
    adopted_out: Path,
) -> None:
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    typst = (adopted_out / fit_id / "report.typ").read_text(encoding="utf-8")
    assert "toy: fit DECAY" in typst


def test_an_adopted_fit_of_another_id_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="is a fit of 'PK'"):
        run.plan("fit", {**ADOPT, "opid": "PK_20261106_120000__abcde"})
    with pytest.raises(ConfigurationError, match="id of a fit"):
        run.plan("fit", {**ADOPT, "opid": "DECAY"})
    with pytest.raises(ConfigurationError, match="identifiability"):
        run.plan("fit", {**FIT, "adopt": True})
    with pytest.raises(ConfigurationError, match="profile settings"):
        run.plan("fit", {**FIT, "identifiability": True, "profile": {"x": 1}})


def test_a_fit_without_adopt_keeps_the_subset_as_its_id() -> None:
    planned = run.plan("fit", FIT)
    assert isinstance(planned, FitSpec)
    assert planned.opid == "DECAY"


# ----------------------------------------------------------------------------
# report of the adopted fits
# ----------------------------------------------------------------------------
def test_the_report_of_every_adopted_fit(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_project(monkeypatch, _adopted_project(adopted_out))
    out = tmp_path / "out"
    summary = run.execute(
        "report",
        {"project": TOY, "subset": "all", "adopted": True, "mapping_figures": False},
        out,
    )
    assert summary == {"reports": {"DECAY": "DECAY"}}
    typst = (out / "DECAY" / "report.typ").read_text(encoding="utf-8")
    assert "toy: fit DECAY" in typst
    assert (out / "DECAY" / "report.pdf").read_bytes().startswith(b"%PDF")
    # the report of the subset alone
    single = tmp_path / "single"
    run.execute(
        "report",
        {"project": TOY, "subset": "DECAY", "adopted": True, "mapping_figures": False},
        single,
    )
    assert (single / "DECAY" / "index.html").exists()


def test_the_report_of_an_adopted_fit_is_the_report_of_its_record(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adopted fit is reported as its record with the input `fit` is."""
    project = _adopted_project(adopted_out)
    _with_project(monkeypatch, project)
    (fit_id,) = project.adopted_fits().values()
    adopted = tmp_path / "adopted"
    run.execute(
        "report",
        {"project": TOY, "subset": "all", "adopted": True, "mapping_figures": False},
        adopted,
    )
    single = tmp_path / "single"
    run.execute(
        "report",
        {"project": TOY, "subset": "DECAY", "mapping_figures": False},
        single,
        inputs={"fit": project.adopted_record(fit_id)},
    )
    assert (adopted / "DECAY" / "report.typ").read_text(encoding="utf-8") == (
        single / "report" / "report.typ"
    ).read_text(encoding="utf-8")


def test_the_report_of_a_project_without_adopted_fits(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="no adopted fit"):
        run.execute(
            "report", {"project": TOY, "subset": "all", "adopted": True}, tmp_path
        )
    with pytest.raises(ValueError, match="input 'fit'"):
        ReportSpec(project=TOY, subset="all", adopted=True, fit=tmp_path)


def test_an_adopted_fit_without_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits = tmp_path / "fits"
    fits.mkdir()
    write_adopted(fits / "adopted.json", {"DECAY": "DECAY_20260101_000000__abcde"})
    _with_project(monkeypatch, PROJECT.replace(fits_dir=fits))
    with pytest.raises(ConfigurationError, match=r"no .*parameters\.json"):
        run.execute(
            "report",
            {"project": TOY, "subset": "all", "adopted": True},
            tmp_path / "out",
        )


# ----------------------------------------------------------------------------
# the fixed adopted fits
# ----------------------------------------------------------------------------
def test_a_fit_records_the_fixed_adopted_fits(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adopted fit of another subset is fixed and named in the report."""
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    fits = tmp_path / "fits"
    shutil.copytree(adopted_out / "fits" / fit_id, fits / OTHER)
    write_adopted(fits / "adopted.json", {"OTHER": OTHER})
    _with_project(monkeypatch, PROJECT.replace(fits_dir=fits))

    out = tmp_path / "out"
    run.execute(
        "fit",
        {**FIT, "fix_adopted": True, "report": True, "mapping_figures": False},
        out,
        workers=1,
    )
    provenance = _json(out / "DECAY" / "optimization_result.json")["provenance"]
    assert provenance["arguments"]["fixed_fits"] == [OTHER]
    # the fit optimizes `k` itself, the value of the adopted fit is not fixed
    assert provenance["fixed_values"] == {}
    typst = (out / "DECAY" / "report.typ").read_text(encoding="utf-8")
    assert "were fixed at their fitted values" in typst

    # without fix_adopted nothing is recorded
    plain = tmp_path / "plain"
    run.execute("fit", FIT, plain, workers=1)
    provenance = _json(plain / "DECAY" / "optimization_result.json")["provenance"]
    assert "fixed_fits" not in provenance["arguments"]


# ----------------------------------------------------------------------------
# start
# ----------------------------------------------------------------------------
def test_a_fit_starts_from_a_point(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute("fit", {**FIT, "start": {"k": 0.3}}, out, workers=1)
    result = _json(out / "DECAY" / "optimization_result.json")
    assert result["provenance"]["start_values"] == {"k": 0.3}
    (repeat,) = result["fits"]
    assert repeat["x0"] == [0.3]
    assert result["provenance"]["arguments"]["start"] == {"k": 0.3}
    # the definition of the project keeps its own start value
    (parameter,) = PROJECT.fit("DECAY").parameters
    assert parameter.start_value == 0.5


def test_a_fit_starts_from_another_fit(adopted_out: Path, tmp_path: Path) -> None:
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    best = _json(adopted_out / fit_id / "parameters.json")["sets"][0]["values"]["k"]
    out = tmp_path / "out"
    run.execute("fit", {**FIT, "start": str(adopted_out / fit_id)}, out, workers=1)
    result = OptimizationResult.from_json(out / "DECAY" / "optimization_result.json")
    assert result.provenance is not None
    assert result.provenance.start_values == {"k": pytest.approx(best, rel=1e-12)}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start": {"x": 0.3}}, "not fitted"),
        ({"start": {"k": 2.0}}, "outside of its bounds"),
        ({"start": "DECAY_nofit"}, "neither a json file nor a fit"),
        ({"start": {"k": 0.3}, "repeats": 2}, "identical"),
        ({"start": {"k": 0.3}, "algorithm": "DE"}, "least squares"),
        ({"start": {"k": 0.3}, "optimizer": {"sampling": "UNIFORM"}}, "sampling"),
    ],
)
def test_a_start_which_cannot_be_used_is_refused(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        run.plan("fit", {**FIT, **changes})


# ----------------------------------------------------------------------------
# remainders
# ----------------------------------------------------------------------------
def _remainder_project(directory: Path) -> Project:
    """Get a project whose model has the remainder `f_c = 1 - f_a - f_b`."""
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    for pid, value in (("f_a", 0.2), ("f_b", 0.3), ("f_c", 0.5)):
        parameter = model.createParameter()
        parameter.setId(pid)
        parameter.setValue(value)
        parameter.setConstant(True)
    assignment = model.createInitialAssignment()
    assignment.setSymbol("f_c")
    assignment.setMath(libsbml.parseL3Formula("1 - f_a - f_b"))
    libsbml.writeSBMLToFile(doc, str(directory / "model.xml"))
    return Project(
        name="remainder",
        root=directory,
        variants={"default": Path("model.xml")},
        fits={
            "F": FitDefinition(
                mapping_collections=dict,
                parameters=[
                    FitParameter(
                        pid="f_a", start_value=0.1, lower_bound=0.01, upper_bound=0.8
                    )
                ],
                base_path=TOY_PATH,
                data_path=TOY_PATH,
            )
        },
    )


def test_the_plan_refuses_a_remainder_which_can_become_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_project(monkeypatch, _remainder_project(tmp_path))
    with pytest.raises(ConfigurationError, match="remainder 'f_c'"):
        run.plan("fit", {"project": TOY, "subset": "F"})


def test_the_report_of_a_record_without_provenance(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report states that the provenance was not recorded and names no fixed fits."""
    (fit_id,) = _json(adopted_out / "adopted.json")["fits"].values()
    fits = tmp_path / "fits"
    shutil.copytree(adopted_out / "fits" / fit_id, fits / fit_id)
    path = fits / fit_id / "optimization_result.json"
    data = _json(path)
    assert data.pop("provenance") is not None
    path.write_text(json.dumps(data), encoding="utf-8")
    write_adopted(fits / "adopted.json", {"DECAY": fit_id})
    _with_project(monkeypatch, PROJECT.replace(fits_dir=fits))
    out = tmp_path / "out"
    run.execute(
        "report",
        {"project": TOY, "subset": "all", "adopted": True, "mapping_figures": False},
        out,
    )
    typst = (out / "DECAY" / "report.typ").read_text(encoding="utf-8")
    assert "not recorded" in typst
    assert "were fixed" not in typst
