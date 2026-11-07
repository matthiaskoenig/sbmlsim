"""The kind `credibility` of the run layer (`sbmlsim.run.credibility_kinds`).

The project of a spec is a reference; the tests set the toy project with its
adopted fit of the session fixture as the attribute `PROJECT` of this module.
"""

import json
import shutil
from pathlib import Path

import pytest

from sbmlsim import run
from sbmlsim.credibility import CredibilityAssessment
from sbmlsim.credibility.lock import LOCK_FILE
from sbmlsim.project import Project
from sbmlsim.run.credibility_kinds import CREDIBILITY_DIR, CredibilitySpec
from sbmlsim.run.kinds import ConfigurationError
from tests.credibility_toy import PROJECT as TOY

DATA = Path(__file__).parents[1] / "data" / "credibility"

#: the project of the specs, set by the fixture `project`
PROJECT: Project = TOY
#: the toy project of `tests.project_toy`, which has no adopted fits
NO_FITS = TOY.replace(adopted=None)

REF = f"{__name__}:PROJECT"


@pytest.fixture
def project(
    credibility_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Project:
    """Set the toy project with its adopted fit and a root of its own as `PROJECT`."""
    root = tmp_path / "root"
    root.mkdir()
    rooted = credibility_project.replace(root=root)
    monkeypatch.setattr(f"{__name__}.PROJECT", rooted)
    return rooted


def test_the_kind() -> None:
    kind = run.get_kind("credibility")
    assert kind.spec is CredibilitySpec
    assert kind.inputs == ("cross_validation", "record")
    assert not kind.uses_cores
    assert not kind.splittable


@pytest.mark.usefixtures("project")
def test_an_assessment_into_the_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute("credibility", {"project": REF, "lock": True}, out)
    assert summary["directory"] == CREDIBILITY_DIR
    assert summary["verdicts"] == {
        "describe": "supported",
        "dose": "supported",
        "renal": "insufficient_evidence",
    }
    assert summary["verified"] == 3
    # the lock is written into the out, never into the root of the project
    assert summary["lock"] == "written"
    assert (out / LOCK_FILE).is_file()
    assert not (PROJECT.root / LOCK_FILE).exists()
    directory = out / CREDIBILITY_DIR
    record = CredibilityAssessment.from_json(directory / "credibility.json")
    assert record.verification is not None
    assert record.verification.battery is not None
    # the battery hook of the toy package writes its table
    assert (directory / "doses.tsv").is_file()
    assert "decay-doses" in (directory / "index.html").read_text(encoding="utf-8")

    # the outputs of the record are written again without an assessment
    again = tmp_path / "again"
    summary = run.execute(
        "credibility", {"project": REF}, again, inputs={"record": out}
    )
    assert summary["verdicts"]["dose"] == "supported"
    rewritten = CredibilityAssessment.from_json(
        again / CREDIBILITY_DIR / "credibility.json"
    )
    assert rewritten.to_dict() == record.to_dict()


@pytest.mark.usefixtures("project")
def test_the_options_of_an_assessment(tmp_path: Path) -> None:
    for settings, message in (
        ({"draws": 8}, "settings of the uncertainty"),
        ({"subjects": 8}, "settings of the variability"),
    ):
        with pytest.raises(ConfigurationError, match=message):
            run.plan("credibility", {"project": REF, **settings})
    with pytest.raises(ConfigurationError, match="is an option of an assessment"):
        run.execute(
            "credibility",
            {"project": REF, "lock": True},
            tmp_path / "out",
            inputs={"record": tmp_path},
        )
    with pytest.raises(
        ConfigurationError,
        match="verification: false, battery: false are options of an assessment",
    ):
        run.execute(
            "credibility",
            {"project": REF, "verification": False, "battery": False},
            tmp_path / "out",
            inputs={"record": tmp_path},
        )
    with pytest.raises(ConfigurationError, match="cross-validation of no adopted fit"):
        run.execute(
            "credibility",
            {"project": REF, "verification": False},
            tmp_path / "empty",
            inputs={"cross_validation": tmp_path},
        )
    with pytest.raises(ConfigurationError, match="no credibility declaration"):
        run.plan("credibility", {"project": f"{__name__}:NO_DECLARATION"})
    with pytest.raises(ConfigurationError, match=r"neither a credibility\.json"):
        run.execute(
            "credibility", {"project": REF}, tmp_path / "x", inputs={"record": tmp_path}
        )


NO_DECLARATION = TOY.replace(credibility=None)


def test_the_uncertainty_and_the_cross_validations(
    project: Project, tmp_path: Path
) -> None:
    (fit_id,) = project.adopted_fits().values()
    validations = tmp_path / "validations"
    folder = validations / "fits" / fit_id
    folder.mkdir(parents=True)
    shutil.copy(DATA / "cross_validation-0.19.json", folder / "cross_validation.json")
    out = tmp_path / "out"
    run.execute(
        "credibility",
        {
            "project": REF,
            "verification": False,
            "uncertainty": True,
            "draws": 4,
            "seed": 2,
        },
        out,
        inputs={"cross_validation": validations},
    )
    data = json.loads(
        (out / CREDIBILITY_DIR / "credibility.json").read_text(encoding="utf-8")
    )
    assert data["verification"] is None
    assert [c["subset"] for c in data["cross_validations"]] == ["DECAY"]
    dose = next(c for c in data["contexts"] if c["id"] == "dose")
    assert dose["uncertainty"]["n_draws"] == 4
    factors = {f["id"]: f for f in dose["factors"]}
    assert factors["model_form"]["status"] == "assessed"


@pytest.mark.usefixtures("project")
def test_the_second_simulator_switched_off(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute("credibility", {"project": REF, "cross_simulator": False}, out)
    record = CredibilityAssessment.from_json(out / CREDIBILITY_DIR / "credibility.json")
    assert record.verification is not None
    assert record.verification.battery is not None
    cross = record.verification.battery.result("cross_simulator")
    assert (cross.passed, cross.statement) == (None, "Not run.")


def test_an_invalid_declaration_is_a_configuration_error(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan checks the declaration, so an invalid one is exit code 4 at the plan."""
    from sbmlsim.credibility.declaration import DeclarationError

    def invalid(self: Project) -> None:
        raise DeclarationError(["the domain 'x' names no field"])

    monkeypatch.setattr(Project, "load_credibility", invalid)
    with pytest.raises(ConfigurationError, match="declaration is invalid"):
        run.plan("credibility", {"project": REF})
