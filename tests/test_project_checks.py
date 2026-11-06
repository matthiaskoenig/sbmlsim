"""The checks of a project as functions, run on the toy project."""

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import pytest

from sbmlsim import project_checks
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.project import CheckSettings, Project, SimulatorSettings
from tests.project_toy import PROJECT, TOY_PATH
from tests.project_toy.broken import ToyCannotInitialize, ToyFigureFails
from tests.project_toy.experiments import (
    ToyDecay,
    ToyDecayHalf,
    ToyDecaySlow,
    ToyUnregistered,
)

PARTIAL = PROJECT.replace(experiments=".broken:PARTIAL")
BROKEN = PROJECT.replace(experiments=".broken:BROKEN")


def test_the_modules_of_the_experiments() -> None:
    modules, errors = project_checks.experiment_modules(PROJECT)
    assert errors == {}
    assert [module.__name__ for module in modules] == ["tests.project_toy.experiments"]


def test_the_defined_experiments_leave_out_the_base() -> None:
    modules, _ = project_checks.experiment_modules(PROJECT)
    assert project_checks.defined_experiments(modules) == {
        ToyDecay,
        ToyDecaySlow,
        ToyDecayHalf,
        ToyUnregistered,
    }


def test_the_modules_of_a_package_of_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package of experiments is imported with its modules, a skipped one not."""
    package = tmp_path / "checked_toy" / "studies"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "fine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "broken.py").write_text(
        "import not_a_module_of_any_name\n", encoding="utf-8"
    )
    (package / "optional.py").write_text(
        "import not_a_module_of_any_name\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    project = Project(
        name="toy",
        root=tmp_path / "checked_toy",
        package="checked_toy",
        checks=CheckSettings(
            experiment_modules="studies",
            not_importable={"studies.optional": "an optional dependency"},
        ),
    )
    modules, errors = project_checks.experiment_modules(project)
    assert [module.__name__ for module in modules] == [
        "checked_toy.studies",
        "checked_toy.studies.fine",
    ]
    assert list(errors) == ["studies.broken"]
    assert "not_a_module_of_any_name" in errors["studies.broken"]


def test_a_module_of_experiments_which_does_not_import() -> None:
    project = PROJECT.replace(checks=CheckSettings(experiment_modules="missing"))
    modules, errors = project_checks.experiment_modules(project)
    assert modules == []
    assert list(errors) == ["missing"]
    with pytest.raises(AssertionError, match="missing does not import"):
        project_checks.check_registry(
            project.replace(checks=CheckSettings(experiment_modules="missing"))
        )


def test_the_checks_need_a_package() -> None:
    with pytest.raises(AssertionError, match="no package"):
        project_checks.experiment_modules(
            Project(name="toy", root=TOY_PATH, experiments=[ToyDecay])
        )


def test_a_complete_registry_passes() -> None:
    project_checks.check_registry(PROJECT)


def test_an_unregistered_experiment_fails() -> None:
    with pytest.raises(AssertionError, match=r"not in the group 'all'.*ToyDecayHalf"):
        project_checks.check_registry(PARTIAL)


def test_a_known_exception_passes() -> None:
    known = PARTIAL.replace(
        checks=CheckSettings(
            not_registered={
                "ToyDecaySlow": "a test",
                "ToyDecayHalf": "a test",
                "ToyUnregistered": "a test",
            }
        )
    )
    project_checks.check_registry(known)


def test_a_stale_exception_fails() -> None:
    stale = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyDecay": "a test", "ToyUnregistered": "a test"}
        )
    )
    with pytest.raises(AssertionError, match="listed as not registered"):
        project_checks.check_registry(stale)


def test_a_listed_class_which_is_not_defined_fails() -> None:
    gone = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyGone": "a test", "ToyUnregistered": "a test"}
        )
    )
    with pytest.raises(AssertionError, match=r"not defined.*ToyGone"):
        project_checks.check_registry(gone)


def test_a_listed_module_which_does_not_exist_fails() -> None:
    gone = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyUnregistered": "a test"},
            not_importable={"experiments_gone": "a test"},
        )
    )
    with pytest.raises(AssertionError, match=r"no module.*experiments_gone"):
        project_checks.check_registry(gone)


def test_the_experiments_initialize() -> None:
    assert set(project_checks.initialize_experiments(PROJECT)) == {
        "ToyDecay",
        "ToyDecaySlow",
        "ToyDecayHalf",
    }


def test_the_initialization_compiles_at_the_level_of_the_project() -> None:
    project = PROJECT.replace(simulator=SimulatorSettings(jit_optimization="O1"))
    assert str(project_checks._runner(project).jit_optimization) == "O1"


@pytest.mark.parametrize("one_at_a_time", [None, "a test"])
def test_all_experiments_run(tmp_path: Path, one_at_a_time: str | None) -> None:
    project = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyUnregistered": "a test"}, one_at_a_time=one_at_a_time
        )
    )
    project_checks.run_all_experiments(project, tmp_path)
    if one_at_a_time:
        for name in ("ToyDecay", "ToyDecaySlow", "ToyDecayHalf"):
            assert (tmp_path / name / "index.html").exists()
    else:
        assert (tmp_path / "all" / "_figures").is_dir()


def test_the_fits_start(tmp_path: Path) -> None:
    runs = project_checks.start_fits(PROJECT, tmp_path)
    assert set(runs) == {"DECAY"}


def test_the_fits_start_at_the_level_of_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbmlsim.fit import cli

    seen: list[str] = []

    def recording_run_fit(definition: FitDefinition, **kwargs: object) -> dict:
        seen.append(str(definition.problem(opid="level").jit_optimization))
        return {"run": object()}

    monkeypatch.setattr(cli, "run_fit", recording_run_fit)
    project = PROJECT.replace(simulator=SimulatorSettings(jit_optimization="O1"))
    project_checks.start_fits(project, tmp_path)
    assert seen == ["O1"]


def test_a_project_of_its_package_passes_the_check() -> None:
    project_checks.check_model(PROJECT)


def test_a_project_without_a_package_checks_its_paths() -> None:
    project = Project(name="toy", root=TOY_PATH, variants=PROJECT.variants)
    project_checks.check_model(project)


def test_a_broken_project_names_every_problem() -> None:
    broken = PROJECT.replace(
        root=TOY_PATH.parent,
        factory=".factory:missing",
        experiments=".broken:NOT_EXPERIMENTS",
        fits=".broken:NO_FITS",
        actions={"gone": ".actions:gone", "rates": ".actions:write_rates"},
    )
    with pytest.raises(AssertionError) as info:
        project_checks.check_model(broken)
    message = str(info.value)
    for expected in [
        "no attribute of the package 'tests.project_toy' is this project",
        "is not the directory",
        "factory '.factory:missing' does not resolve",
        "the experiments do not load",
        "the fit definitions are not a non-empty mapping",
        "action 'gone' '.actions:gone' does not resolve",
    ]:
        assert expected in message, expected
    assert "action 'rates'" not in message


def without_workers(project: Project, out: Path) -> None:
    """An action which does not take the workers of its call."""


def test_an_action_which_is_not_called_as_an_action_fails() -> None:
    project = PROJECT.replace(
        actions={"rates": ".actions:write_rates", "wrong": without_workers}
    )
    with pytest.raises(AssertionError) as info:
        project_checks.check_model(project)
    message = str(info.value)
    assert "the action 'wrong' is not called as" in message
    assert "'rates'" not in message


def test_a_package_which_does_not_import_fails() -> None:
    with pytest.raises(AssertionError, match="does not import"):
        project_checks.check_model(PROJECT.replace(package="tests.project_missing"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "other"}, "is not the name of the package"),
        (
            {"variants": {"default": Path("/elsewhere/model.xml")}},
            r"variant 'default' .*model\.xml is not below",
        ),
        ({"variants": {"default": Path("models/gone.xml")}}, "does not exist"),
        ({"data_path": [Path("/does/not/exist")]}, "does not exist"),
        ({"fits_dir": Path("/elsewhere/fits")}, "fit records"),
    ],
)
def test_check_model_finds_a_slip(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        project_checks.check_model(PROJECT.replace(**changes))


def test_every_failure_of_the_initialization_is_named() -> None:
    with pytest.raises(AssertionError) as info:
        project_checks.initialize_experiments(BROKEN)
    message = str(info.value)
    assert "1 of 3 experiments cannot be initialized" in message
    assert (
        "ToyCannotInitialize: RuntimeError: the simulations of ToyCannotInitialize fail"
        in message
    )


def test_every_failure_of_a_run_is_named(tmp_path: Path) -> None:
    with pytest.raises(AssertionError) as info:
        project_checks.run_all_experiments(BROKEN, tmp_path)
    message = str(info.value)
    assert ToyCannotInitialize.__name__ in message
    assert ToyFigureFails.__name__ in message
    assert "figures_mpl()" in message


def test_every_failure_of_a_run_one_at_a_time_is_named(tmp_path: Path) -> None:
    one_by_one = BROKEN.replace(
        checks=CheckSettings(one_at_a_time="a test of the failures")
    )
    with pytest.raises(AssertionError) as info:
        project_checks.run_all_experiments(one_by_one, tmp_path)
    message = str(info.value)
    assert ToyCannotInitialize.__name__ in message
    assert ToyFigureFails.__name__ in message
    # the experiment after a failing one (ToyFigureFails) ran and wrote its report
    assert list((tmp_path / "ToyCannotInitialize").glob("**/*.html"))


class Check(StrEnum):
    """The checks of a battery of the tests."""

    STEADY_STATE = "steady_state"
    REBUILD = "rebuild"


class Status(StrEnum):
    """The status of a check of the tests."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Result:
    """The result of a check of the tests."""

    check: Check
    passed: bool | None
    status: Status
    problems: list[str] = field(default_factory=list)


@dataclass
class Battery:
    """A battery of the tests."""

    results: list[Result]


def _battery(*results: Result) -> Any:
    """Get the hook which runs a battery of these results."""

    def run(project: Project) -> Battery:
        assert project is PROJECT or project.name == PROJECT.name
        return Battery(list(results))

    return run


PASSED = Result(Check.REBUILD, True, Status.PASSED)
FAILED = Result(Check.STEADY_STATE, False, Status.FAILED, ["not steady"])
NOT_APPLICABLE = Result(Check.STEADY_STATE, None, Status.NOT_APPLICABLE)


def test_a_battery_which_passes() -> None:
    battery = project_checks.check_battery(PROJECT, _battery(PASSED, NOT_APPLICABLE))
    assert battery.results == [PASSED, NOT_APPLICABLE]


def test_a_failed_check_of_a_battery_is_a_problem() -> None:
    with pytest.raises(AssertionError, match=r"'steady_state' failed:\n  not steady"):
        project_checks.check_battery(PROJECT, _battery(PASSED, FAILED))


class PlainCheck(Enum):
    """Checks whose `str` is not their value."""

    STEADY_STATE = "steady_state"


def test_the_checks_of_a_battery_are_named_by_their_value() -> None:
    plain = Result(PlainCheck.STEADY_STATE, False, Status.FAILED, ["x"])  # ty: ignore[invalid-argument-type]
    known = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyUnregistered": "a test"},
            battery={"steady_state": "a test"},
        )
    )
    project_checks.check_battery(known, _battery(plain))
    with pytest.raises(AssertionError, match="'steady_state' failed"):
        project_checks.check_battery(PROJECT, _battery(plain))


def test_a_known_failure_of_a_battery() -> None:
    known = PROJECT.replace(
        checks=CheckSettings(
            not_registered={"ToyUnregistered": "a test"},
            battery={"steady_state": "a test"},
        )
    )
    project_checks.check_battery(known, _battery(PASSED, FAILED))
    with pytest.raises(AssertionError, match="no longer applies, it not applicable"):
        project_checks.check_battery(known, _battery(PASSED, NOT_APPLICABLE))
