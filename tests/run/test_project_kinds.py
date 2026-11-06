"""The kinds `factory` and `action` of a project, run on the toy project."""

import shutil
import sys
from pathlib import Path

import pytest

from sbmlsim import run
from sbmlsim.project import Project
from sbmlsim.run import ConfigurationError
from sbmlsim.run.project_kinds import ActionSpec, FactorySpec
from tests.project_toy import PROJECT

TOY = "tests.project_toy:PROJECT"
#: a project of the module, set by a test
CHANGED: Project | None = None
#: a project without a factory
NO_FACTORY = PROJECT.replace(factory=None)


def test_the_kinds_are_registered() -> None:
    assert run.get_kind("factory").spec is FactorySpec
    assert run.get_kind("action").spec is ActionSpec
    assert not run.get_kind("factory").uses_cores
    assert not run.get_kind("factory").splittable
    assert not run.get_kind("action").splittable


def test_the_factory_builds_the_models_into_the_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute("factory", {"project": TOY}, out)
    assert summary == {
        "directory": "models",
        "outputs": 2,
        "declared": True,
        "changed": [],
    }
    for rebuilt, committed in zip(
        PROJECT.factory_paths(out), PROJECT.factory_paths(), strict=True
    ):
        assert rebuilt.read_bytes() == committed.read_bytes()
    assert (out / "run.json").exists()


def test_the_factory_names_the_changed_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "package"
    shutil.copytree(PROJECT.root / "models", root / "models")
    slow = root / "models" / "toy_slow.xml"
    slow.write_text(
        slow.read_text(encoding="utf-8").replace("0.1", "0.3"), encoding="utf-8"
    )
    (root / "models" / "toy.xml").unlink()
    project = PROJECT.replace(
        root=root,
        variants={
            "default": Path("models/toy.xml"),
            "slow": Path("models/toy_slow.xml"),
        },
        factory_dir=None,
    )
    monkeypatch.setattr(sys.modules[__name__], "CHANGED", project)
    summary = run.execute(
        "factory", {"project": f"{__name__}:CHANGED"}, tmp_path / "out"
    )
    assert summary["changed"] == ["models/toy.xml", "models/toy_slow.xml"]
    # the package is not written
    assert not (root / "models" / "toy.xml").exists()


UNDECLARED: Project | None = None


def test_the_factory_without_declared_outputs_compares_what_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The variants and the written files the package has, never nothing."""
    root = tmp_path / "package"
    shutil.copytree(PROJECT.root / "models", root / "models")
    (root / "models" / "notes.txt").write_text("a note\n", encoding="utf-8")
    project = PROJECT.replace(
        root=root,
        variants={"default": Path("models/toy.xml")},
        factory_dir=None,
        factory_outputs=[],
    )
    monkeypatch.setattr(sys.modules[__name__], "UNDECLARED", project)
    spec = {"project": f"{__name__}:UNDECLARED"}
    summary = run.execute("factory", spec, tmp_path / "same")
    assert summary == {
        "directory": "models",
        "outputs": 2,
        "declared": False,
        "changed": [],
    }
    assert "declares no factory_outputs" in caplog.text
    slow = root / "models" / "toy_slow.xml"
    slow.write_text(
        slow.read_text(encoding="utf-8").replace("0.1", "0.3"), encoding="utf-8"
    )
    summary = run.execute("factory", spec, tmp_path / "changed")
    assert summary["changed"] == ["models/toy_slow.xml"]
    # nothing of the package written: nothing to compare
    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = project.replace(root=empty, variants={}, factory_dir=Path("models"))
    monkeypatch.setattr(sys.modules[__name__], "UNDECLARED", nothing)
    with pytest.raises(RuntimeError, match="nothing to compare"):
        run.execute("factory", spec, tmp_path / "nothing")


def test_the_factory_of_a_project_without_one() -> None:
    with pytest.raises(ConfigurationError, match="no factory"):
        run.plan("factory", {"project": f"{__name__}:NO_FACTORY"})
    with pytest.raises(ConfigurationError, match="not loaded"):
        run.plan("factory", {"project": "tests.project_toy:MISSING"})


def test_an_action_runs_into_the_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    spec = {"project": TOY, "name": "rates", "arguments": {"factor": 2}}
    summary = run.execute("action", spec, out, workers=3)
    assert list(summary.items()) == [("rows", 2), ("workers", 3), ("action", "rates")]
    assert (out / "rates.tsv").read_text(encoding="utf-8") == (
        "variant\trate\ndefault\t0.4\nslow\t0.2\n"
    )


def test_an_action_without_a_summary(tmp_path: Path) -> None:
    summary = run.execute(
        "action", {"project": TOY, "name": "nothing"}, tmp_path / "out"
    )
    assert summary == {"action": "nothing"}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "other"}, r"no action 'other'"),
        ({"name": "rates", "arguments": {"facter": 2}}, r"\['facter'\]"),
        ({"name": "nothing", "arguments": {"workers": 2}}, "workers"),
    ],
)
def test_the_plan_checks_the_action(spec: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        run.plan("action", {"project": TOY, **spec})


def test_an_action_which_is_not_resolved() -> None:
    with pytest.raises(ConfigurationError, match="not resolved"):
        run.plan("action", {"project": f"{__name__}:UNRESOLVED", "name": "gone"})


def not_a_summary(project: Project, out: Path, /, *, workers: int) -> int:
    """Return no summary."""
    return 1


UNRESOLVED = PROJECT.replace(actions={"gone": ".actions:gone"})
WRONG = PROJECT.replace(actions={"wrong": not_a_summary})


def test_an_action_returns_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="returned a int"):
        run.execute(
            "action", {"project": f"{__name__}:WRONG", "name": "wrong"}, tmp_path / "o"
        )


def test_list_names_the_actions_and_the_factory() -> None:
    entries = run.list_analyses(TOY)
    pairs = [(e["kind"], e["name"]) for e in entries]
    assert ("action", "rates") in pairs
    assert ("factory", "models") in pairs
    assert ("simulate", "ToyDecay") not in pairs
    experiments = run.list_analyses(TOY, experiments=True)
    assert ("simulate", "ToyDecay") in [(e["kind"], e["name"]) for e in experiments]


def test_simulate_runs_a_group_of_experiments(tmp_path: Path) -> None:
    planned = run.plan("simulate", {"project": TOY, "experiments": ["variants"]})
    assert getattr(planned, "experiments", None) == ["ToyDecaySlow", "ToyDecayHalf"]
    mixed = run.plan(
        "simulate", {"project": TOY, "experiments": ["ToyDecayHalf", "variants"]}
    )
    assert getattr(mixed, "experiments", None) == ["ToyDecayHalf", "ToyDecaySlow"]
    with pytest.raises(ConfigurationError, match="names no experiments"):
        run.plan("simulate", {"project": TOY, "experiments": []})
    summary = run.execute(
        "simulate", {"project": TOY, "experiments": ["studies"]}, tmp_path / "out"
    )
    assert summary == {"experiments": 1, "failed": 0}


def test_simulate_draws_the_figures_in_the_given_formats(tmp_path: Path) -> None:
    spec = {"project": TOY, "experiments": ["ToyDecay"], "figure_formats": [" PNG "]}
    assert getattr(run.plan("simulate", spec), "figure_formats", None) == ["png"]
    with pytest.raises(ConfigurationError, match="figure formats"):
        run.plan("simulate", {**spec, "figure_formats": [" "]})
    run.execute("simulate", spec, tmp_path / "out")
    figures = {p.suffix for p in (tmp_path / "out").rglob("*_fig_decay.*")}
    assert figures == {".png"}
