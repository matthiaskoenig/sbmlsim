"""The kind `sensitivity` of a sensitivity analysis of a project over groups.

On the toy project, whose sensitivity analysis `decay` is a reference to a
runner of the project, and on the project `PROJECT` of this module, whose
analysis `subfolders` is two runners by subfolder without a project of their
own, which the project binds.
"""

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sbmlsim import run
from sbmlsim.project import Project
from sbmlsim.run import ConfigurationError, PlanMismatch
from sbmlsim.run.scan_kinds import PLAN_MARKER, SensitivitySpec
from sbmlsim.run.units import MissingUnits
from sbmlsim.sensitivity.runner import (
    LocalAnalysis,
    SamplingAnalysis,
    SensitivityRunner,
)
from tests.project_toy import PROJECT as TOY
from tests.project_toy.sensitivity import CONTROL, SENSITIVITY, SLOW

DECAY: dict[str, Any] = {"project": "tests.project_toy:PROJECT", "analysis": "decay"}

#: a project with the runners of two subfolders, bound to it
PROJECT = TOY.replace(
    name="subfolders",
    sensitivities={
        "subfolders": {
            "single": SensitivityRunner(
                parameters=SENSITIVITY.parameters,
                outputs=SENSITIVITY.outputs,
                observables=SENSITIVITY.observables,
                analyses={"local": LocalAnalysis(groups=[CONTROL])},
            ),
            "multi": SensitivityRunner(
                parameters=SENSITIVITY.parameters,
                outputs=SENSITIVITY.outputs[:1],
                observables=SENSITIVITY.observables,
                analyses={
                    "local": LocalAnalysis(groups=[SLOW]),
                    "sampling": SamplingAnalysis(groups=[CONTROL, SLOW], n=4),
                },
                default=("local", "sampling"),
            ),
        }
    },
)
#: the files of a run beside its results
RUN_FILES = {"run.json", "spec.json", "summary.json", "progress.jsonl", "progress"}

SUBFOLDERS: dict[str, Any] = {
    "project": f"{__name__}:PROJECT",
    "analysis": "subfolders",
}


def plan(spec: dict[str, Any]) -> SensitivitySpec:
    planned = run.plan("sensitivity", spec)
    assert isinstance(planned, SensitivitySpec)
    return planned


def test_the_plan_sets_the_analyses_and_the_units() -> None:
    planned = plan(DECAY)
    assert planned.analyses == ["local"]
    assert planned.units == 1
    assert planned.scan_hash is None
    planned = plan({**DECAY, "analyses": ["sampling", "local"]})
    assert planned.analyses == ["sampling", "local"]
    assert planned.units == 2
    planned = plan(SUBFOLDERS)
    assert planned.subfolders == ["single", "multi"]
    assert planned.analyses is None
    assert planned.units == 3
    planned = plan({**SUBFOLDERS, "analyses": ["local"]})
    assert planned.units == 2
    planned = plan({**SUBFOLDERS, "subfolders": ["multi"]})
    assert planned.units == 2


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({**DECAY, "analyses": ["morris"]}, r"Unknown analyses \['morris'\]"),
        ({**DECAY, "subfolders": ["single"]}, "no subfolders"),
        ({**DECAY, "scan_hash": "abc"}, "no scan"),
        ({**SUBFOLDERS, "subfolders": ["other"]}, r"\['other'\] have no runner"),
        (
            {"project": "tests.project_toy:PROJECT", "analysis": "nothing"},
            r"no scan 'nothing'.*sensitivity analyses over groups are \['decay'\]",
        ),
        (
            {
                "project": "tests.project_toy:PROJECT",
                "analysis": "decay2",
                "analyses": ["local"],
            },
            "neither analyses nor subfolders",
        ),
        (
            {
                "project": "tests.project_toy:PROJECT",
                "analysis": "decay2",
                "run_plan": {"units": []},
            },
            "no run_plan",
        ),
    ],
    ids=["analysis", "subfolders", "hash", "subfolder", "unknown", "scan", "run_plan"],
)
def test_the_plan_refuses_a_spec(spec: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        run.plan("sensitivity", spec)


def test_a_whole_run_writes_the_files_of_the_runner(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(
        "sensitivity", {**DECAY, "analyses": ["local", "sampling"]}, out, workers=1
    )
    assert summary["analyses"] == ["local", "sampling"]
    assert summary["parameters"] == 2
    assert sorted(p.name for p in (out / "local").iterdir()) == [
        PLAN_MARKER,
        "local_d0.01_nvar3_00_control.png",
        "local_d0.01_nvar3_00_control.tsv",
        "local_d0.01_nvar3_01_slow.png",
        "local_d0.01_nvar3_01_slow.tsv",
    ]
    assert (out / "sampling" / "sampling_N8.tsv").is_file()
    table = pd.read_csv(out / "parameters.tsv", sep="\t").set_index("uid")
    assert table.loc["k", "value_slow"] == pytest.approx(0.1)
    # no temporary folder is left
    assert not [p for p in out.iterdir() if p.name.startswith(".")]


def test_units_and_merge_give_the_whole_run(tmp_path: Path) -> None:
    planned = run.plan("sensitivity", SUBFOLDERS)
    out = tmp_path / "split"
    run.compute_unit("sensitivity", planned, 1, out)
    with pytest.raises(MissingUnits) as info:
        run.merge("sensitivity", planned, out)
    assert list(info.value.missing) == [0, 2]
    run.compute_unit("sensitivity", planned, 2, out)
    run.compute_unit("sensitivity", planned, 0, out)
    summary = run.merge("sensitivity", planned, out)
    assert summary["analyses"] == ["single/local", "multi/local", "multi/sampling"]
    assert (out / "parameters.tsv").is_file()

    whole = tmp_path / "whole"
    assert run.execute("sensitivity", SUBFOLDERS, whole, workers=1) == summary
    # every file of the analyses and the table, byte by byte
    names = sorted(
        p.relative_to(whole).as_posix()
        for p in whole.rglob("*")
        if p.is_file() and p.parts[len(whole.parts)] not in RUN_FILES
    )
    assert "parameters.tsv" in names
    assert "multi/sampling/sampling_N4_sensitivity_outputs.png" in names
    assert names == sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.parts[len(out.parts)] not in RUN_FILES
    )
    for name in names:
        assert (out / name).read_bytes() == (whole / name).read_bytes(), name


def test_a_unit_whose_folder_exists_is_not_computed_again(tmp_path: Path) -> None:
    planned = run.plan("sensitivity", DECAY)
    out = tmp_path / "out"
    run.compute_unit("sensitivity", planned, 0, out)
    table = out / "local" / "local_d0.01_nvar3_00_control.tsv"
    table.write_text("kept", encoding="utf-8")
    run.compute_unit("sensitivity", planned, 0, out)
    assert table.read_text(encoding="utf-8") == "kept"


def test_a_unit_of_another_plan_is_computed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smoke run of 2 samples leaves no folder a real run takes as its own."""
    spec = {**DECAY, "analyses": ["sampling"]}
    out = tmp_path / "out"
    monkeypatch.setenv("SBMLSIM_SENSITIVITY_N", "2")
    smoke = plan(spec)
    assert smoke.run_plan is not None
    assert smoke.run_plan["units"][0]["n"] == 2
    run.compute_unit("sensitivity", smoke, 0, out)
    assert (out / "sampling" / "sampling_N2.tsv").is_file()
    # the plan fixes the number of samples: a unit of another one is refused
    monkeypatch.delenv("SBMLSIM_SENSITIVITY_N")
    with pytest.raises(PlanMismatch, match=r"\['sampling'\] differ"):
        run.compute_unit("sensitivity", smoke, 0, out)
    planned = plan(spec)
    assert planned.run_plan is not None
    assert planned.run_plan["units"][0]["n"] == 8
    # the folder of the smoke run, e.g. copied into the output of the real
    # run, is no unit of it
    real = tmp_path / "real"
    real.mkdir()
    shutil.copytree(out / "sampling", real / "sampling")
    with pytest.raises(MissingUnits):
        run.merge("sensitivity", planned, real)
    out = real
    run.compute_unit("sensitivity", planned, 0, out)
    files = sorted(p.name for p in (out / "sampling").iterdir() if p.is_file())
    assert "sampling_N8.tsv" in files
    assert "sampling_N2.tsv" not in files
    run.merge("sensitivity", planned, out)


def test_a_resume_with_another_number_of_samples_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {**DECAY, "analyses": ["sampling"]}
    out = tmp_path / "out"
    monkeypatch.setenv("SBMLSIM_SENSITIVITY_N", "2")
    run.execute("sensitivity", spec, out, workers=1)
    monkeypatch.delenv("SBMLSIM_SENSITIVITY_N")
    with pytest.raises(PlanMismatch, match="SBMLSIM_SENSITIVITY_N"):
        run.execute("sensitivity", spec, out, workers=1, resume=True)


def test_a_changed_table_is_another_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = plan(DECAY)
    assert planned.run_plan is not None
    changed = planned.model_copy(
        update={"run_plan": {**planned.run_plan, "table": "0" * 64}}
    )
    with pytest.raises(PlanMismatch, match="the table of the parameters"):
        run.compute_unit("sensitivity", changed, 0, tmp_path / "out")


def test_a_resumed_run_keeps_the_analyses_it_has(tmp_path: Path) -> None:
    out = tmp_path / "out"
    spec = {**DECAY, "analyses": ["local"]}
    run.execute("sensitivity", spec, out, workers=1)
    table = out / "local" / "local_d0.01_nvar3_00_control.tsv"
    table.write_text("kept", encoding="utf-8")
    run.execute("sensitivity", spec, out, workers=1, resume=True)
    assert table.read_text(encoding="utf-8") == "kept"


def test_a_plan_of_other_units_is_refused(tmp_path: Path) -> None:
    planned = run.plan("sensitivity", DECAY)
    other = planned.model_copy(update={"units": 3})
    with pytest.raises(PlanMismatch, match="3 units"):
        run.compute_unit("sensitivity", other, 0, tmp_path / "out")


def test_the_project_lists_its_sensitivity_analyses() -> None:
    pairs = {(a["kind"], a["name"]): a["description"] for a in PROJECT.analyses()}
    assert pairs[("sensitivity", "subfolders")] == (
        "the sensitivity analyses 'subfolders' of the subfolders single, multi"
    )
    toy = {(a["kind"], a["name"]): a["description"] for a in TOY.analyses()}
    assert toy[("sensitivity", "decay")] == "the sensitivity analyses 'decay'"


def test_the_runners_are_bound_to_the_project() -> None:
    runners = PROJECT.sensitivity("subfolders")
    assert {key: r.project for key, r in runners.items()} == {
        "single": PROJECT,
        "multi": PROJECT,
    }
    assert TOY.sensitivity("decay") is SENSITIVITY
    with pytest.raises(KeyError, match="decay"):
        TOY.sensitivity("other")


def test_a_runner_of_another_project_is_refused() -> None:
    other = Project(
        name="other",
        root=TOY.root,
        variants=TOY.variants,
        sensitivities={"decay": SENSITIVITY},
    )
    with pytest.raises(ValueError, match="runs the project 'toy'"):
        other.sensitivity("decay")


@pytest.mark.parametrize(
    ("sensitivities", "match"),
    [
        ({"x": 1}, "SensitivityRunner"),
        ({"x": {}}, "SensitivityRunner"),
        ({"x": "no reference"}, "module:attribute"),
        ({"x": ".relative:R"}, "needs the package"),
        ({"decay": SENSITIVITY}, r"\['decay'\] are scans with indices"),
    ],
    ids=["object", "empty", "reference", "relative", "clash"],
)
def test_the_sensitivities_of_a_project_are_checked(
    sensitivities: dict[str, Any], match: str, tmp_path: Path
) -> None:
    from sbmlsim.project import ScanAnalysis
    from tests.project_toy import decay_scan

    scans = {
        "decay": ScanAnalysis(model=Path("m.xml"), scan=decay_scan, indices="local")
    }
    with pytest.raises(ValueError, match=match):
        Project(name="p", root=tmp_path, scans=scans, sensitivities=sensitivities)
