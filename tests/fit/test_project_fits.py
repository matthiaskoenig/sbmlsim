"""A fit of a project: its id, its start, the fixed adopted fits, its context, its remainders."""

import copy
import dataclasses
import json
import shutil
from datetime import date
from pathlib import Path

import libsbml
import pytest

from sbmlsim import run
from sbmlsim.fit import FitParameter
from sbmlsim.fit.cli import load_fit_records, records_context, report_context
from sbmlsim.fit.project_fits import (
    current_fixed_fits,
    fixed_values,
    recorded_fixed_fits,
    remainder_problems,
    start_point,
    with_start,
)
from sbmlsim.fit.result import fit_date
from sbmlsim.project import Project, write_adopted
from tests.project_toy import PROJECT

#: the id of a hand-made adopted fit of another subset
OTHER = "OTHER_20260101_000000__abcde"


@pytest.fixture(scope="module")
def toy_fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the records of a fit of the toy project with its profiles."""
    out = tmp_path_factory.mktemp("fit") / "out"
    run.execute(
        "fit",
        {
            "project": "tests.project_toy:PROJECT",
            "subset": "DECAY",
            "opid": "DECAY_20261106_120000__abcde",
            "repeats": 1,
            "seed": 1,
            "identifiability": True,
            "profile": {"max_points": 3},
            "fisher": False,
        },
        out,
        workers=1,
    )
    return out / "DECAY_20261106_120000__abcde"


def _rename(value: object, old: str, new: str) -> object:
    """Rename a string and a key of a JSON document."""
    if isinstance(value, dict):
        return {
            (new if k == old else k): _rename(v, old, new) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_rename(v, old, new) for v in value]
    return new if value == old else value


def adopted_other(toy_fit: Path, directory: Path, pid: str = "k") -> Project:
    """Get the toy project with an adopted fit of the subset OTHER.

    The record is the one of the toy fit under the id of a fit of OTHER, whose
    parameter `k` is named `pid`.
    """
    fits = directory / "fits"
    record = fits / OTHER
    shutil.copytree(toy_fit, record, ignore=shutil.ignore_patterns("runs"))
    if pid != "k":
        for name in (
            "parameters.json",
            "optimization_result.json",
            "identifiability.json",
        ):
            path = record / name
            data = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(_rename(data, "k", pid)), encoding="utf-8")
    write_adopted(fits / "adopted.json", {"OTHER": OTHER})
    return PROJECT.replace(fits_dir=fits)


# ----------------------------------------------------------------------------
# the ids of the fits
# ----------------------------------------------------------------------------
def test_the_date_of_a_fit_id() -> None:
    assert fit_date("PK_X_20261016_185018__3efe3") == date(2026, 10, 16)
    with pytest.raises(ValueError, match="is not the id of a fit"):
        fit_date("PK")
    with pytest.raises(ValueError, match="is no date"):
        fit_date("PK_20261399_185018__3efe3")


# ----------------------------------------------------------------------------
# the start of a fit
# ----------------------------------------------------------------------------
def test_a_start_point_of_values(tmp_path: Path) -> None:
    definition = PROJECT.fit("DECAY")
    assert start_point(PROJECT, definition, {"k": 0.3}) == {"k": 0.3}
    start = tmp_path / "start.json"
    start.write_text(json.dumps({"k": 0.3}), encoding="utf-8")
    assert start_point(PROJECT, definition, start) == {"k": 0.3}
    assert start_point(PROJECT, definition, str(start)) == {"k": 0.3}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"x": 0.3}, "not fitted"),
        ({"k": 2.0}, "outside of its bounds"),
        ({"k": "a"}, "not a number"),
        ({"k": True}, "not a number"),
    ],
)
def test_a_start_which_cannot_be_used_is_refused(values: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        start_point(PROJECT, PROJECT.fit("DECAY"), values)


def test_an_unknown_start_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither a json file nor a fit"):
        start_point(PROJECT, PROJECT.fit("DECAY"), "DECAY_nofit", tmp_path)


def test_a_fit_starts_from_another_fit(toy_fit: Path) -> None:
    """A finished fit is its directory or its id in the directory of the fits."""
    best = json.loads((toy_fit / "parameters.json").read_text(encoding="utf-8"))
    k = best["sets"][0]["values"]["k"]
    definition = PROJECT.fit("DECAY")
    assert start_point(PROJECT, definition, toy_fit) == {
        "k": pytest.approx(k, rel=1e-12)
    }
    assert start_point(PROJECT, definition, toy_fit.name, toy_fit.parent) == {
        "k": pytest.approx(k, rel=1e-12)
    }


def test_a_fit_starts_from_an_adopted_fit(toy_fit: Path, tmp_path: Path) -> None:
    fits = tmp_path / "fits"
    shutil.copytree(toy_fit, fits / toy_fit.name)
    project = PROJECT.replace(fits_dir=fits)
    best = json.loads((toy_fit / "parameters.json").read_text(encoding="utf-8"))
    assert start_point(project, project.fit("DECAY"), toy_fit.name) == {
        "k": pytest.approx(best["sets"][0]["values"]["k"], rel=1e-12)
    }


def test_a_start_without_the_unit_of_a_parameter_is_refused(tmp_path: Path) -> None:
    """A value of a fit without its unit cannot be converted into the unit of the fit."""
    definition = PROJECT.fit("DECAY")
    fit = tmp_path / "DECAY_20261030_120000__abcde"
    fit.mkdir()
    best: dict[str, object] = {"values": {"k": 12.0}}
    (fit / "parameters.json").write_text(json.dumps({"sets": [best]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"has no unit of 'k'.*1/min"):
        start_point(PROJECT, definition, fit.name, tmp_path)
    best["units"] = {"k": "1/hr"}
    (fit / "parameters.json").write_text(json.dumps({"sets": [best]}), encoding="utf-8")
    assert start_point(PROJECT, definition, fit.name, tmp_path) == {
        "k": pytest.approx(0.2)
    }


def test_with_start_keeps_the_definition() -> None:
    definition = PROJECT.fit("DECAY")
    started = with_start(definition, {"k": 0.3})
    assert [p.start_value for p in started.parameters] == [0.3]
    assert [p.start_value for p in definition.parameters] == [0.5]


# ----------------------------------------------------------------------------
# the fixed adopted fits
# ----------------------------------------------------------------------------
def test_the_fixed_fits_are_the_adopted_fits_of_the_other_subsets(
    toy_fit: Path, tmp_path: Path
) -> None:
    project = adopted_other(toy_fit, tmp_path)
    assert current_fixed_fits(project, "DECAY") == [OTHER]
    assert current_fixed_fits(project, "OTHER") == []
    assert current_fixed_fits(PROJECT, "DECAY") == []
    assert fixed_values(PROJECT, "DECAY", PROJECT.fit("DECAY")) == ([], {})


def test_the_fixed_values_leave_out_the_parameters_of_the_fit(
    toy_fit: Path, tmp_path: Path
) -> None:
    project = adopted_other(toy_fit, tmp_path / "a")
    # the fit optimizes `k` itself, the value of the adopted fit is not fixed
    assert fixed_values(project, "DECAY", project.fit("DECAY")) == ([OTHER], {})
    other = adopted_other(toy_fit, tmp_path / "b", pid="k_other")
    value = other.fit_records().records[OTHER].parameters["k_other"].value
    assert fixed_values(other, "DECAY", other.fit("DECAY")) == (
        [OTHER],
        {"k_other": value},
    )


def test_the_recorded_fixed_fits(toy_fit: Path) -> None:
    records = load_fit_records(toy_fit)
    assert recorded_fixed_fits(records) == ()
    assert records.opt_result is not None
    opt_result = copy.deepcopy(records.opt_result)
    assert opt_result.provenance is not None
    opt_result.provenance.arguments["fixed_fits"] = [OTHER]
    assert recorded_fixed_fits(dataclasses.replace(records, opt_result=opt_result)) == (
        OTHER,
    )
    opt_result.provenance = None
    assert (
        recorded_fixed_fits(dataclasses.replace(records, opt_result=opt_result)) == ()
    )


# ----------------------------------------------------------------------------
# the context of the report
# ----------------------------------------------------------------------------
def test_the_report_context_names_the_fixed_adopted_fits() -> None:
    definition = PROJECT.fit("DECAY")
    context = report_context(PROJECT, "DECAY", definition, fixed_fits=[OTHER])
    assert context.title == "toy: fit DECAY"
    assert context.description == "the rate of the decay"
    assert context.notes == (
        f"The parameters of the adopted fit {OTHER} were fixed at their fitted values.",
    )
    assert report_context(
        PROJECT, "DECAY", definition, fixed_fits=["A_1", "B_2"]
    ).notes == (
        "The parameters of the adopted fits A_1, B_2 were fixed at their fitted "
        "values.",
    )
    assert report_context(PROJECT, "DECAY", definition, fixed_fits=()).notes == ()
    titled = PROJECT.replace(title="Toy model")
    assert (
        report_context(titled, "DECAY", definition, fixed_fits=()).title
        == "Toy model: fit DECAY"
    )


def test_the_records_context_names_the_recorded_fixed_fits(toy_fit: Path) -> None:
    definition = PROJECT.fit("DECAY")
    records = load_fit_records(toy_fit)
    assert records_context(PROJECT)("DECAY", definition, records).notes == ()
    assert records.opt_result is not None
    opt_result = copy.deepcopy(records.opt_result)
    assert opt_result.provenance is not None
    opt_result.provenance.arguments["fixed_fits"] = [OTHER]
    context = records_context(PROJECT)(
        "DECAY", definition, dataclasses.replace(records, opt_result=opt_result)
    )
    assert OTHER in context.notes[0]


# ----------------------------------------------------------------------------
# the remainders of a model
# ----------------------------------------------------------------------------
def remainder_model(path: Path, f_b: float = 0.3) -> Path:
    """Write a model with the remainder `f_c = 1 - f_a - f_b`."""
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    for pid, value in (("f_a", 0.2), ("f_b", f_b), ("f_c", 0.5), ("x", 1.0)):
        parameter = model.createParameter()
        parameter.setId(pid)
        parameter.setValue(value)
        parameter.setConstant(True)
    for symbol, formula in (("f_c", "1 - f_a - f_b"), ("x", "2 * f_a")):
        assignment = model.createInitialAssignment()
        assignment.setSymbol(symbol)
        assignment.setMath(libsbml.parseL3Formula(formula))
    libsbml.writeSBMLToFile(doc, str(path))
    return path


def fraction(pid: str, upper: float) -> FitParameter:
    """Get a fitted fraction."""
    return FitParameter(pid=pid, start_value=0.1, lower_bound=0.01, upper_bound=upper)


def test_a_remainder_which_can_become_negative(tmp_path: Path) -> None:
    path = remainder_model(tmp_path / "model.xml")
    assert remainder_problems(path, [fraction("f_a", 0.7)]) == []
    (problem,) = remainder_problems(path, [fraction("f_a", 0.8)])
    assert "'f_c' = 1 - f_a - f_b" in problem
    assert "f_a <= 0.8" in problem
    assert "1.1 > 1" in problem
    # a fixed value replaces the value of the model
    assert remainder_problems(path, [fraction("f_a", 0.8)], {"f_b": 0.1}) == []
    # both fitted
    (both,) = remainder_problems(path, [fraction("f_a", 0.6), fraction("f_b", 0.6)])
    assert "f_a <= 0.6, f_b <= 0.6" in both
    # a parameter of no remainder
    assert remainder_problems(path, [fraction("x", 10.0)]) == []
