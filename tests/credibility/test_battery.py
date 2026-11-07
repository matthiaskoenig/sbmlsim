"""The generic battery of the checks of the variants of a model (`sbmlsim.credibility.battery`).

The checks run on small antimony models and on the toy credibility package,
whose battery hook (`tests.credibility_toy.battery`) doses A; the rebuild runs
on the factory of the toy project of `tests.project_toy`.
"""

import dataclasses
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, override

import antimony
import libsbml
import pytest

from sbmlsim.credibility import (
    BATTERY_CHECKS,
    Battery,
    BatteryHook,
    BatteryResult,
    BatteryRun,
    BatteryStatus,
    Check,
    CredibilityAssessment,
    FactorStatus,
    Verdict,
    assess_table,
    run_battery,
    write_outputs,
)
from sbmlsim.credibility.battery import (
    BatteryCollector,
    BatteryContribution,
    BatterySettings,
    accumulating_states,
    read_build,
)
from sbmlsim.credibility.report import battery_table
from sbmlsim.credibility.verification import (
    CROSS_SIMULATOR,
    NON_NEGATIVITY,
    PLAUSIBILITY,
    REBUILD,
    STEADY_STATE,
)
from sbmlsim.project import CheckSettings, CredibilitySettings, Project
from sbmlsim.project_checks import check_battery
from tests.credibility.tables import PROVENANCE, battery_of, fit_table, verification_of
from tests.credibility_toy import PROJECT
from tests.credibility_toy.battery import BATTERY, DECAY
from tests.credibility_toy.credibility import CREDIBILITY
from tests.project_toy import PROJECT as TOY_PROJECT

DATA = Path(__file__).parents[1] / "data" / "credibility"

#: a source of A at its steady state which flows into the sink B
STEADY = """
model steady
  compartment V = 1;
  species A in V = 2; species B in V = 0;
  k0 = 1; k = 0.5;
  SOURCE: => A; k0;
  FLOW: A => B; k * A;
  {extra}
end
"""


def _sbml(path: Path, text: str) -> Path:
    """Write the SBML of an antimony model."""
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(text) < 0:
        raise ValueError(antimony.getLastError())
    path.write_text(
        antimony.getSBMLString(antimony.getMainModuleName()), encoding="utf-8"
    )
    return path


def _project(tmp_path: Path, extra: str = "") -> Project:
    """Get a project of the steady model without a hook, fits and factory."""
    path = _sbml(tmp_path / "steady.xml", STEADY.replace("{extra}", extra))
    return Project(name="steady", root=tmp_path, variants={"default": path})


def _battery(project: Project, **kwargs: object) -> Battery:
    return run_battery(
        project,
        rebuild=False,
        cross_simulator=False,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def test_a_sound_model_passes(tmp_path: Path) -> None:
    battery = _battery(_project(tmp_path))
    assert [r.check for r in battery.results] == list(BATTERY_CHECKS)
    statuses = {r.check.id: r.status for r in battery.results}
    assert statuses == {
        "non_negativity": BatteryStatus.PASSED,
        "steady_state": BatteryStatus.PASSED,
        "cross_simulator": BatteryStatus.NOT_APPLICABLE,
        "plausibility": BatteryStatus.PASSED,
        "rebuild": BatteryStatus.NOT_APPLICABLE,
    }
    # the sink B accumulates, A stays at its start
    assert battery.result(STEADY_STATE).cases == 1
    assert battery.result(NON_NEGATIVITY).cases == 2
    assert battery.sections == {}
    assert battery.settings.startswith("roadrunner CVODE BDF, rtol 1e-10")


def test_the_accumulating_states(tmp_path: Path) -> None:
    path = _sbml(tmp_path / "steady.xml", STEADY.replace("{extra}", ""))
    doc = libsbml.readSBMLFromFile(str(path))
    assert accumulating_states(doc.getModel()) == ("B",)
    build = read_build("default", path)
    assert (build.species, build.states) == (("A", "B"), ())
    assert build.accumulating == frozenset({"B"})


def test_a_state_which_moves_fails_the_steady_state(tmp_path: Path) -> None:
    result = _battery(_project(tmp_path, "A = 3;")).result(STEADY_STATE)
    assert result.status is BatteryStatus.FAILED
    assert result.problems[0].startswith("default: 'A' changes by ")
    assert result.at == "default: A"


def test_a_negative_species_fails_the_non_negativity(tmp_path: Path) -> None:
    project = _project(tmp_path, "species C in V = 1; DRAIN: C => ; 0.1;")
    result = _battery(project).result(NON_NEGATIVITY)
    assert result.status is BatteryStatus.FAILED
    assert result.problems[0].startswith("default without a change: 'C' reaches ")


def test_a_compartment_without_a_volume_is_implausible(tmp_path: Path) -> None:
    result = _battery(_project(tmp_path, "compartment Vx = 0;")).result(PLAUSIBILITY)
    assert result.status is BatteryStatus.FAILED
    assert result.problems == ("default: the compartment 'Vx' has the size 0",)


def test_a_fitted_value_at_its_bound_is_implausible(
    credibility_project: Project, tmp_path: Path
) -> None:
    (fit_id,) = credibility_project.adopted_fits().values()
    fits = tmp_path / "fits"
    record = credibility_project.adopted_record(fit_id)
    shutil.copytree(record, fits / fit_id)
    parameters = fits / fit_id / "parameters.json"
    data = json.loads(parameters.read_text(encoding="utf-8"))
    project = credibility_project.replace(fits_dir=fits)
    pid, parameter = next(iter(project.fit_records().parameters.items()))
    assert parameter.upper_bound == 1.0
    _set_value(data, pid, parameter.upper_bound)
    parameters.write_text(json.dumps(data), encoding="utf-8")
    result = _battery(project).result(PLAUSIBILITY)
    assert result.status is BatteryStatus.FAILED
    assert result.problems == (
        f"the value 1 of '{pid}' of the fit '{fit_id}' lies at its upper bound 1",
    )


def _set_value(data: dict, pid: str, value: float) -> None:
    """Set the fitted value of a parameter in the JSON of the parameters of a fit."""
    for parameter_set in data["sets"]:
        parameter_set["values"][pid] = value


def _skip_without_copasi() -> None:
    """Skip a test without COPASI, imported through sbmlsim, which keeps the locale."""
    from sbmlsim.engine import import_copasi

    try:
        import_copasi()
    except ImportError:
        pytest.skip("COPASI is not installed")


def test_the_hook_of_the_toy_package(credibility_project: Project) -> None:
    battery = run_battery(credibility_project)
    assert [r.check.id for r in battery.results] == list(BATTERY.order)
    assert all(r.passed is not False for r in battery.results), battery.results
    decay = battery.result(DECAY)
    assert (decay.status, decay.cases) == (BatteryStatus.PASSED, 2)
    assert [d["case"] for d in battery.sections["doses"]] == ["dose", "late dose"]
    # the hook simulates A from 0, at its steady state
    assert battery.result(STEADY_STATE).status is BatteryStatus.PASSED
    assert battery.settings.endswith("A dose of 1 of A at 0 and of 0.5 at 30.")
    assert "from 0 to 60 on a grid of 60 intervals" in battery.settings
    # COPASI repeats the case without a change and the dose, not the late dose
    cross = battery.result(CROSS_SIMULATOR)
    _skip_without_copasi()
    assert (cross.status, cross.cases) == (BatteryStatus.PASSED, 2)
    assert "COPASI" in cross.statement
    # the toy package has no factory
    assert battery.result(REBUILD).status is BatteryStatus.NOT_APPLICABLE


def test_without_a_hook_the_decay_fails_the_steady_state(
    credibility_project: Project,
) -> None:
    project = credibility_project.replace(
        credibility_settings=CredibilitySettings(
            metadata_type=credibility_project.credibility_settings.metadata_type
        )
    )
    assert project.battery_hook() is None
    battery = _battery(project)
    assert [r.check for r in battery.results] == list(BATTERY_CHECKS)
    assert battery.result(STEADY_STATE).status is BatteryStatus.FAILED


class _WrongHook(BatteryHook):
    """A hook which gives the result of a check it does not add."""

    name: ClassVar[str] = "wrong"
    checks: ClassVar[tuple[Check, ...]] = (DECAY,)

    @override
    def contribute(self, runs: Sequence[BatteryRun]) -> BatteryContribution:
        return BatteryContribution(
            results=(BatteryCollector().result(STEADY_STATE, "wrong"),)
        )


def test_a_hook_gives_the_results_of_its_checks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gives the results of"):
        _battery(_project(tmp_path), hook=_WrongHook())


def test_the_rebuild(tmp_path: Path) -> None:
    passed = run_battery(TOY_PROJECT, cross_simulator=False).result(REBUILD)
    assert passed.status is BatteryStatus.PASSED
    assert passed.cases == len(TOY_PROJECT.compared_outputs(tmp_path / "none"))
    root = tmp_path / "root"
    models = root / "models"
    models.mkdir(parents=True)
    for path in TOY_PROJECT.variants.values():
        text = path.read_text(encoding="utf-8")
        if path.name == "toy.xml":
            text = text.replace('name="toy decay model"', 'name="the toy decay model"')
        (models / path.name).write_text(text, encoding="utf-8")
    changed = TOY_PROJECT.replace(
        root=root,
        variants={
            name: models / path.name for name, path in TOY_PROJECT.variants.items()
        },
        factory_dir=models,
    )
    failed = run_battery(changed, cross_simulator=False).result(REBUILD)
    assert failed.status is BatteryStatus.FAILED
    assert failed.problems == ("'models/toy.xml' differs from its rebuild",)


def _broken_factory(directory: Path) -> None:
    raise RuntimeError("broken")


def test_a_factory_which_fails(tmp_path: Path) -> None:
    project = _project(tmp_path).replace(factory=_broken_factory, factory_dir=tmp_path)
    result = run_battery(project, cross_simulator=False).result(REBUILD)
    assert result.status is BatteryStatus.FAILED
    assert result.problems == ("the factory failed: RuntimeError: broken",)


def test_without_a_second_engine_the_cross_simulator_is_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sbmlsim.engine as engine

    def missing() -> None:
        raise ImportError("not installed")

    monkeypatch.setattr(engine, "import_amici", missing)
    monkeypatch.setattr(engine, "import_copasi", missing)
    battery = run_battery(_project(tmp_path), rebuild=False)
    result = battery.result(CROSS_SIMULATOR)
    assert result.status is BatteryStatus.NOT_APPLICABLE
    assert result.statement == (
        "Neither AMICI nor COPASI is installed, install sbmlsim with the extra "
        "'amici' or 'copasi'."
    )


def test_the_second_simulator(tmp_path: Path) -> None:
    _skip_without_copasi()
    result = run_battery(_project(tmp_path), rebuild=False).result(CROSS_SIMULATOR)
    assert result.status is BatteryStatus.PASSED, result.problems
    assert result.largest < 1e-6


def test_the_settings_of_the_battery() -> None:
    with pytest.raises(ValueError, match="horizon of the battery is positive"):
        BatterySettings(horizon=0.0)
    with pytest.raises(ValueError, match="needs a step"):
        BatterySettings(steps=0)
    with pytest.raises(ValueError, match="second engines of the battery"):
        BatterySettings(engines=("roadrunner",))


def test_the_battery_round_trip(tmp_path: Path) -> None:
    battery = run_battery(PROJECT.replace(root=tmp_path), cross_simulator=False)
    read = Battery.from_dict(json.loads(json.dumps(battery.to_dict())))
    assert read.to_dict() == battery.to_dict()
    assert read.sections == battery.sections
    assert math.isnan(read.result(REBUILD).largest)
    assert read.result("decay").check == DECAY


def test_the_battery_of_pkpdsim_0_19() -> None:
    """The battery of a record of pkpdsim 0.19: its PK checks and its fates."""
    data = json.loads((DATA / "assessment-0.19.json").read_text(encoding="utf-8"))
    battery = Battery.from_dict(data["verification"]["battery"])
    assert [r.check.id for r in battery.results] == [
        "mass_balance",
        "non_negativity",
        "zero_dose",
        "steady_state",
        "superposition",
        "cross_simulator",
        "plausibility",
        "rebuild",
    ]
    assert battery.sections == {"fates": []}
    assert battery.result("rebuild").passed is True


def _with_battery(battery: Battery) -> CredibilityAssessment:
    fits = fit_table()
    verification = verification_of(fits, battery=battery)
    return assess_table(CREDIBILITY, fits, PROVENANCE, verification)


def _failed(battery: Battery, check: Check) -> Battery:
    return dataclasses.replace(
        battery,
        results=tuple(
            dataclasses.replace(r, status=BatteryStatus.FAILED, problems=("broken",))
            if r.check == check
            else r
            for r in battery.results
        ),
    )


def test_the_battery_in_an_assessment(tmp_path: Path) -> None:
    battery = battery_of()
    assessment = _with_battery(battery)
    applicable = [r.check.id for r in battery.results if r.passed is not None]
    for context in assessment.contexts:
        checks = [r["check"] for r in context.verification]
        assert checks[3:] == applicable
        factors = {f["id"]: f for f in context.factors}
        code = factors["numerical_code_verification"]
        assert code["status"] == FactorStatus.ASSESSED.value
        assert code["statement"].startswith("sbmlsim runs the SBML Test Suite")
        assert "Non-negativity of 4 cases" in code["statement"]
        assert factors["software_quality_assurance"]["status"] == "assessed"
    assert assessment.context("dose").verdict is Verdict.SUPPORTED
    broken = _with_battery(_failed(battery, STEADY_STATE))
    for context in broken.contexts:
        if context.n_decisive:
            assert context.verdict is Verdict.NOT_VERIFIED, context.id
    record = tmp_path / "credibility.json"
    broken.to_json(record)
    read = CredibilityAssessment.from_json(record).verification
    assert read is not None
    assert broken.verification is not None
    assert read.to_dict() == broken.verification.to_dict()


def test_the_outputs_of_the_battery(tmp_path: Path) -> None:
    assessment = _with_battery(battery_of())
    table = battery_table(assessment)
    assert list(table["check"]) == [c.id for c in BATTERY_CHECKS]
    assert set(table["status"]) == {"passed", "not_applicable"}
    output = write_outputs(assessment, tmp_path / "credibility", hook=BATTERY)
    assert (
        (output / "battery.tsv")
        .read_text(encoding="utf-8")
        .startswith("check\tfactor\tstatus")
    )
    assert (output / "doses.tsv").read_text(encoding="utf-8").splitlines() == [
        "variant\tcase\tend",
        "default\tdose\t0.1",
    ]
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "Battery of the variants" in html
    assert '<table class="decay-doses">' in html
    assert "battery of the variants</a>, 4 checks" in html
    # without the hook the battery shows its checks alone
    plain = write_outputs(assessment, tmp_path / "plain")
    assert not (plain / "doses.tsv").exists()
    assert "decay-doses" not in (plain / "index.html").read_text(encoding="utf-8")


def test_the_known_failures_of_the_battery(credibility_project: Project) -> None:
    def run(project: Project) -> Battery:
        return _failed(battery_of(), STEADY_STATE)

    with pytest.raises(AssertionError, match="the check 'steady_state' failed"):
        check_battery(credibility_project, run)
    known = credibility_project.replace(
        checks=CheckSettings(battery={"steady_state": "the decay"})
    )
    check_battery(known, run)


def test_a_known_failure_is_shown_in_the_verification(
    credibility_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sbmlsim.credibility.battery as module
    from sbmlsim.credibility import evaluate_fits, verify

    monkeypatch.setattr(
        module, "run_battery", lambda project: _failed(battery_of(), STEADY_STATE)
    )
    known = credibility_project.replace(
        checks=CheckSettings(battery={"steady_state": "the decay"})
    )
    verification = verify(known, evaluate_fits(known))
    assert verification.summary.battery is not None
    result: BatteryResult = verification.summary.battery.result(STEADY_STATE)
    assert result.passed is False
    assert result.problems == ("broken", "known failure: the decay")
