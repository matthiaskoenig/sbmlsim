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
from sbmlsim.run.context import load_spec
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
    with pytest.raises(ConfigurationError, match="profile"):
        run.plan("fit", {**FIT, "identifiability": True, "profile": {"x": 1}})
    with pytest.raises(ConfigurationError, match="profile settings"):
        run.plan("fit", {**FIT, "identifiability": True, "profile": {"max_points": 0}})


def test_the_fit_profiles_the_given_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbmlsim.fit import identifiability

    with pytest.raises(ConfigurationError, match="not distinct parameters"):
        run.plan("fit", {**FIT, "identifiability": True, "profile": {"pids": ["x"]}})
    given: list[object] = []
    profile = identifiability.profile_likelihood

    def recording(*args: Any, **kwargs: Any) -> Any:
        given.append(kwargs.get("pids"))
        return profile(*args, **kwargs)

    monkeypatch.setattr(identifiability, "profile_likelihood", recording)
    spec = {
        **FIT,
        "identifiability": True,
        "fisher": False,
        "profile": {"max_points": 3, "pids": ["k"]},
    }
    run.execute("fit", spec, tmp_path / "out", workers=1)
    assert given == [["k"]]
    result = _json(tmp_path / "out" / "DECAY" / "identifiability.json")
    assert list(result["profiles"]) == ["k"]


def test_an_adopted_fit_is_reported_by_default() -> None:
    adopt = {**FIT, "identifiability": True, "adopt": True}
    assert FitSpec.model_validate(adopt).report
    assert not FitSpec.model_validate({**adopt, "report": False}).report
    assert not FitSpec.model_validate(FIT).report


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
    (fit_id,) = _adopted_project(adopted_out).adopted_fits().values()
    assert summary == {"reports": {"DECAY": fit_id}}
    typst = (out / fit_id / "report.typ").read_text(encoding="utf-8")
    assert "toy: fit DECAY" in typst
    assert (out / fit_id / "report.pdf").read_bytes().startswith(b"%PDF")
    # the report of the subset alone
    single = tmp_path / "single"
    run.execute(
        "report",
        {"project": TOY, "subset": "DECAY", "adopted": True, "mapping_figures": False},
        single,
    )
    assert (single / fit_id / "index.html").exists()


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
    assert (adopted / fit_id / "report.typ").read_text(encoding="utf-8") == (
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
        {**FIT, "report": True, "mapping_figures": False},
        out,
        workers=1,
    )
    provenance = _json(out / "DECAY" / "optimization_result.json")["provenance"]
    assert provenance["arguments"]["fixed_fits"] == [OTHER]
    # the fit optimizes `k` itself, the value of the adopted fit is not fixed
    assert provenance["fixed_values"] == {}
    typst = (out / "DECAY" / "report.typ").read_text(encoding="utf-8")
    assert "were fixed at their fitted values" in typst
    planned = FitSpec.model_validate(_json(out / "spec.json"))
    assert (planned.fixed_fits, planned.fixed_values) == ([OTHER], {})

    # a repeat run again records the fixed fits of the plan, although the
    # adopted fits of the project changed
    _with_project(monkeypatch, PROJECT.replace(fits_dir=tmp_path / "empty"))
    (out / "DECAY" / "runs" / "DECAY_run_0.json").unlink()
    run.execute(
        "fit",
        {**FIT, "report": True, "mapping_figures": False},
        out,
        workers=1,
        resume=True,
    )
    provenance = _json(out / "DECAY" / "optimization_result.json")["provenance"]
    assert provenance["arguments"]["fixed_fits"] == [OTHER]

    # without record_adopted nothing is recorded
    plain = tmp_path / "plain"
    run.execute("fit", {**FIT, "record_adopted": False}, plain, workers=1)
    provenance = _json(plain / "DECAY" / "optimization_result.json")["provenance"]
    assert "fixed_fits" not in provenance["arguments"]
    assert provenance["fixed_values"] == {}


def test_a_project_without_adopted_fits_records_no_fixed_fits(tmp_path: Path) -> None:
    """The default `record_adopted` changes nothing for a project without adopted fits."""
    out = tmp_path / "out"
    run.execute(
        "fit", {**FIT, "report": True, "mapping_figures": False}, out, workers=1
    )
    provenance = _json(out / "DECAY" / "optimization_result.json")["provenance"]
    assert provenance["arguments"]["fixed_fits"] == []
    assert provenance["fixed_values"] == {}
    typst = (out / "DECAY" / "report.typ").read_text(encoding="utf-8")
    assert "were fixed" not in typst


# ----------------------------------------------------------------------------
# start
# ----------------------------------------------------------------------------
def test_a_fit_starts_from_a_point(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute("fit", {**FIT, "start": {"k": 0.3}}, out, workers=1)
    planned = FitSpec.model_validate(_json(out / "spec.json"))
    assert (planned.start, planned.start_from) == ({"k": 0.3}, None)
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
    start = str(adopted_out / fit_id)
    run.execute("fit", {**FIT, "start": start}, out, workers=1)
    planned = FitSpec.model_validate(_json(out / "spec.json"))
    assert planned.start == {"k": pytest.approx(best, rel=1e-12)}
    assert planned.start_from == start
    result = OptimizationResult.from_json(out / "DECAY" / "optimization_result.json")
    assert result.provenance is not None
    assert result.provenance.start_values == {"k": pytest.approx(best, rel=1e-12)}
    assert result.provenance.arguments["start"] == start


def test_a_relative_start_of_a_spec_file_is_relative_to_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "start.json").write_text(json.dumps({"k": 0.3}), encoding="utf-8")
    spec = specs / "fit.json"
    spec.write_text(json.dumps({**FIT, "start": "start.json"}), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = tmp_path / "out"
    run.execute("fit", spec, out, workers=1)
    planned = FitSpec.model_validate(_json(out / "spec.json"))
    assert planned.start == {"k": 0.3}
    assert planned.start_from == str(specs / "start.json")

    # a resume after the start file was deleted is a resume of the same spec,
    # which starts from the planned point
    (specs / "start.json").unlink()
    (out / "DECAY" / "runs" / "DECAY_run_0.json").unlink()
    run.execute("fit", spec, out, workers=1, resume=True)
    result = _json(out / "DECAY" / "optimization_result.json")
    assert result["provenance"]["start_values"] == {"k": 0.3}


def test_a_fit_id_of_a_spec_file_stays_an_id(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start of the form of a fit id is found in the fits of the project."""
    project = _adopted_project(adopted_out)
    (fit_id,) = project.adopted_fits().values()
    _with_project(monkeypatch, project)
    spec = tmp_path / "fit.json"
    spec.write_text(json.dumps({**FIT, "start": fit_id}), encoding="utf-8")
    # a file of that name next to the spec does not make it a path
    (tmp_path / fit_id).mkdir()
    loaded = load_spec(run.get_kind("fit"), spec)
    assert isinstance(loaded, FitSpec)
    assert loaded.start == fit_id
    # another name is a path relative to the spec file
    spec.write_text(json.dumps({**FIT, "start": "point.json"}), encoding="utf-8")
    relative = load_spec(run.get_kind("fit"), spec)
    assert isinstance(relative, FitSpec)
    assert relative.start == str(tmp_path / "point.json")


def test_a_resume_starts_from_the_planned_point(
    adopted_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start an adopted fit gave is kept although the adopted fits changed."""
    project = _adopted_project(adopted_out)
    (fit_id,) = project.adopted_fits().values()
    _with_project(monkeypatch, project)
    out = tmp_path / "out"
    run.execute("fit", {**FIT, "start": fit_id}, out, workers=1)
    planned = FitSpec.model_validate(_json(out / "spec.json"))
    assert planned.start_from == fit_id
    # the adopted fit is gone, the lost repeat runs again from the planned point
    _with_project(monkeypatch, PROJECT.replace(fits_dir=tmp_path / "empty"))
    (out / "DECAY" / "runs" / "DECAY_run_0.json").unlink()
    run.execute("fit", {**FIT, "start": fit_id}, out, workers=1, resume=True)
    result = OptimizationResult.from_json(out / "DECAY" / "optimization_result.json")
    assert result.provenance is not None
    assert result.provenance.start_values == planned.start


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
    typst = (out / fit_id / "report.typ").read_text(encoding="utf-8")
    assert "not recorded" in typst
    assert "were fixed" not in typst


# ----------------------------------------------------------------------------
# the adopt kind
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def finished_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of a finished fit with its profiles, not adopted."""
    out = tmp_path_factory.mktemp("finished") / "out"
    run.execute(
        "fit",
        {
            **FIT,
            "opid": "DECAY_20261106_120000__abcde",
            "identifiability": True,
            "profile": {"max_points": 3},
            "report": True,
            "mapping_figures": False,
        },
        out,
        workers=1,
    )
    return out


def test_the_adopt_kind_adopts_a_finished_fit(
    finished_out: Path, tmp_path: Path
) -> None:
    fit_id = "DECAY_20261106_120000__abcde"
    out = tmp_path / "out"
    before = _status()
    summary = run.execute("adopt", {"project": TOY}, out, inputs={"fit": finished_out})
    assert _status() == before
    assert summary == {"subset": "DECAY", "opid": fit_id, "adopted": f"fits/{fit_id}"}
    assert _json(out / "adopted.json") == {"fits": {"DECAY": fit_id}}
    record = out / "fits" / fit_id
    for name in ("parameters.json", "identifiability.json", "report.typ"):
        assert (record / name).exists(), name
    # nothing temporary is left next to the record
    assert [p.name for p in (out / "fits").iterdir()] == [fit_id]
    # the records directory itself is an input as well
    direct = tmp_path / "direct"
    run.execute(
        "adopt", {"project": TOY}, direct, inputs={"fit": finished_out / fit_id}
    )
    assert (direct / "fits" / fit_id / "parameters.json").exists()


def test_the_adopt_kind_refuses_an_adopted_fit_unless_forced(
    finished_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_id = "DECAY_20261106_120000__abcde"
    first = tmp_path / "first"
    run.execute("adopt", {"project": TOY}, first, inputs={"fit": finished_out})
    _with_project(monkeypatch, _adopted_project(first))
    with pytest.raises(ConfigurationError, match="adopted already"):
        run.execute(
            "adopt", {"project": TOY}, tmp_path / "again", inputs={"fit": finished_out}
        )
    run.execute(
        "adopt",
        {"project": TOY, "force": True},
        tmp_path / "forced",
        inputs={"fit": finished_out},
    )
    assert (tmp_path / "forced" / "fits" / fit_id / "parameters.json").exists()


def test_the_adopt_kind_needs_a_fit_with_uncertainty(tmp_path: Path) -> None:
    out = tmp_path / "fit"
    run.execute("fit", {**FIT, "opid": "DECAY_20261106_120000__abcde"}, out, workers=1)
    with pytest.raises(ConfigurationError, match="neither a profile"):
        run.execute("adopt", {"project": TOY}, tmp_path / "out", inputs={"fit": out})
    with pytest.raises(ConfigurationError, match="needs the records"):
        run.execute("adopt", {"project": TOY}, tmp_path / "none")


def test_the_adopt_kind_needs_the_id_of_a_fit(tmp_path: Path) -> None:
    out = tmp_path / "fit"
    run.execute("fit", FIT, out, workers=1)
    with pytest.raises(ConfigurationError, match="not the id of a fit"):
        run.execute("adopt", {"project": TOY}, tmp_path / "out", inputs={"fit": out})


# ----------------------------------------------------------------------------
# the subset all, a project without variants
# ----------------------------------------------------------------------------
def test_a_subset_named_all_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = PROJECT.replace(fits={"all": PROJECT.fit("DECAY")})
    with pytest.raises(ValueError, match="rename the subset"):
        project.load_fit_definitions()
    _with_project(monkeypatch, project)
    with pytest.raises(ConfigurationError, match="rename the subset"):
        run.plan("fit", {"project": TOY, "subset": "all"})


def test_the_adopt_kind_reports_a_project_whose_fits_are_not_read(
    finished_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_project(monkeypatch, PROJECT.replace(fits={"all": PROJECT.fit("DECAY")}))
    with pytest.raises(ConfigurationError, match="rename the subset"):
        run.execute(
            "adopt", {"project": TOY}, tmp_path / "out", inputs={"fit": finished_out}
        )


def test_a_project_without_variants_checks_no_remainders(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _with_project(monkeypatch, PROJECT.replace(variants={}))
    with caplog.at_level("INFO", logger="sbmlsim.run.fit_kinds"):
        run.plan("fit", FIT)
    assert "remainders of its models are not checked" in caplog.text
