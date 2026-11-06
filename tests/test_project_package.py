"""The project of a model package: references, variants, factory and settings.

The tests run the project of the toy package `tests.project_toy`.
"""

import dataclasses
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from sbmlsim.experiment import ExperimentRunError, SimulationExperiment
from sbmlsim.fit.result import fit_subset
from sbmlsim.model.model_roadrunner import JitOptimization
from sbmlsim.project import (
    ADOPTED_FILE,
    CheckSettings,
    FigureOutputs,
    Project,
    SimulatorSettings,
    check_reference,
    resolve,
    write_adopted,
)
from tests.project_toy import PROJECT, TOY_PATH

REPOSITORY = Path(__file__).parents[1]

#: a fit id of the subset of the toy fit
FIT = "DECAY_20261106_120000__abcde"


def with_fields(project: Project, **fields: Any) -> Project:
    """Get a project with other fields, validated as a new project."""
    return Project(**{**dict(project), **fields})


def test_importing_a_project_imports_none_of_its_references() -> None:
    code = (
        "import sys, tests.project_toy; "
        "print(sorted(m for m in sys.modules if m.startswith('tests.project_toy.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        cwd=REPOSITORY,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ----------------------------------------------------------------------------
# references
# ----------------------------------------------------------------------------
def test_resolve_an_absolute_reference() -> None:
    assert resolve("tests.project_toy:TOY_PATH") == TOY_PATH


def test_resolve_a_relative_reference() -> None:
    groups = resolve(".experiments:EXPERIMENTS", "tests.project_toy")
    assert set(groups) == {"studies", "variants"}


def test_resolve_a_dotted_attribute() -> None:
    assert resolve("tests.project_toy:PROJECT.name") == "toy"


@pytest.mark.parametrize(
    "reference",
    ["experiments.EXPERIMENTS", ":EXPERIMENTS", ".experiments:", "a:b-c", "a b:c"],
)
def test_a_malformed_reference_is_refused(reference: str) -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        check_reference(reference)
    with pytest.raises(ValidationError, match="module:attribute"):
        with_fields(PROJECT, experiments=reference)


def test_a_relative_reference_needs_a_package() -> None:
    with pytest.raises(ValueError, match="needs a package"):
        resolve(".experiments:EXPERIMENTS")
    with pytest.raises(ValidationError, match="needs the package"):
        with_fields(PROJECT, package=None)


def test_a_missing_attribute_names_the_reference() -> None:
    project = with_fields(PROJECT, credibility=".credibility:MISSING")
    with pytest.raises(AttributeError, match=r"\.credibility:MISSING"):
        project.load_credibility()


def test_a_reference_is_resolved_once_and_is_no_part_of_the_value() -> None:
    project = with_fields(PROJECT)
    assert project == PROJECT
    first = project.load_credibility()
    assert first == {"context": "the decay of A"}
    assert project.load_credibility() is first
    assert project == with_fields(PROJECT)
    # a copy shares the resolved references, which are keyed by the reference
    copy = project.model_copy(update={"credibility": ".experiments:EXPERIMENTS"})
    assert copy.load_credibility() is resolve(
        ".experiments:EXPERIMENTS", "tests.project_toy"
    )
    assert project.load_credibility() is first


def test_a_pickled_project_drops_its_resolved_references() -> None:
    project = with_fields(PROJECT)
    project.load_experiments()
    restored = pickle.loads(pickle.dumps(project))
    assert restored == project
    assert dict(restored._resolved) == {}
    assert restored.experiment_classes() == project.experiment_classes()


def test_objects_instead_of_references() -> None:
    from tests.project_toy.credibility import CREDIBILITY
    from tests.project_toy.experiments import EXPERIMENTS
    from tests.project_toy.factory import create_toy_models
    from tests.project_toy.fitting import FIT_DEFINITIONS

    project = with_fields(
        PROJECT,
        experiments=EXPERIMENTS,
        fits=FIT_DEFINITIONS,
        factory=create_toy_models,
        credibility=CREDIBILITY,
    )
    assert project.load_credibility() is CREDIBILITY
    assert project.load_factory() is create_toy_models
    assert project.load_fit_definitions() == PROJECT.load_fit_definitions()
    assert project.load_experiments() == PROJECT.load_experiments()


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def test_the_groups_of_the_experiments() -> None:
    from tests.project_toy.experiments import ToyDecay, ToyDecayHalf, ToyDecaySlow

    groups = PROJECT.load_experiments()
    assert groups == {
        "studies": [ToyDecay],
        "variants": [ToyDecaySlow, ToyDecayHalf],
        "all": [ToyDecay, ToyDecaySlow, ToyDecayHalf],
    }
    assert PROJECT.experiment_classes() == [ToyDecay, ToyDecaySlow, ToyDecayHalf]
    assert PROJECT.experiment_classes(["variants"]) == [ToyDecaySlow, ToyDecayHalf]
    assert PROJECT.experiment_classes(["ToyDecayHalf", "all"]) == [
        ToyDecayHalf,
        ToyDecay,
        ToyDecaySlow,
    ]
    with pytest.raises(KeyError, match="studies"):
        PROJECT.experiment_classes(["other"])


def test_experiments_as_classes_or_a_callable() -> None:
    from tests.project_toy.experiments import ToyDecay, ToyDecaySlow

    def experiments() -> list[type[SimulationExperiment]]:
        return [ToyDecay, ToyDecaySlow]

    for value in [experiments, [ToyDecay, ToyDecaySlow]]:
        project = with_fields(PROJECT, experiments=value)
        assert project.load_experiments() == {"all": [ToyDecay, ToyDecaySlow]}


def test_experiments_which_are_no_experiments() -> None:
    project = with_fields(PROJECT, experiments=".credibility:CREDIBILITY")
    with pytest.raises(TypeError, match="no SimulationExperiment"):
        project.load_experiments()
    project = with_fields(PROJECT, experiments=None)
    with pytest.raises(ValueError, match="no simulation experiments"):
        project.load_experiments()


# ----------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------
def test_the_variants_are_relative_to_the_root() -> None:
    assert PROJECT.variants == {
        "default": TOY_PATH / "models" / "toy.xml",
        "slow": TOY_PATH / "models" / "toy_slow.xml",
    }
    assert PROJECT.model_path() == TOY_PATH / "models" / "toy.xml"
    assert PROJECT.model_path("slow") == TOY_PATH / "models" / "toy_slow.xml"
    with pytest.raises(KeyError, match="slow"):
        PROJECT.model_path("female")
    with pytest.raises(ValueError, match="no model variants"):
        with_fields(PROJECT, variants={}).model_path()


def test_every_variant_is_a_file_of_its_own() -> None:
    with pytest.raises(ValidationError, match="file of its own"):
        with_fields(PROJECT, variants={"a": Path("toy.xml"), "b": TOY_PATH / "toy.xml"})


def test_load_roadrunner_at_the_level_of_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sbmlsim.model.model_roadrunner as model_roadrunner

    calls: list[tuple[object, object]] = []
    load = model_roadrunner.load_roadrunner

    def recording(source: Any, optimization: Any = None) -> Any:
        calls.append((source, optimization))
        return load(source, optimization)

    monkeypatch.setattr(model_roadrunner, "load_roadrunner", recording)
    project = with_fields(PROJECT, simulator=SimulatorSettings(jit_optimization="O0"))
    r = project.load_roadrunner("slow")
    assert calls == [(PROJECT.model_path("slow"), JitOptimization.O0)]
    assert r.getValue("k") == pytest.approx(0.1)
    assert PROJECT.load_roadrunner().getValue("k") == pytest.approx(0.2)


# ----------------------------------------------------------------------------
# factory and credibility
# ----------------------------------------------------------------------------
def test_the_factory_builds_its_outputs(tmp_path: Path) -> None:
    assert PROJECT.create_models(tmp_path / "models") == tmp_path / "models"
    assert PROJECT.factory_paths() == [
        TOY_PATH / "models" / "toy.xml",
        TOY_PATH / "models" / "toy_slow.xml",
    ]
    for output, path in zip(
        PROJECT.factory_outputs, PROJECT.factory_paths(), strict=True
    ):
        assert (tmp_path / output).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("output", [TOY_PATH / "toy.xml", Path("../toy.xml")])
def test_the_outputs_of_the_factory_are_inside_the_root(output: Path) -> None:
    with pytest.raises(ValidationError, match="relative to the root"):
        with_fields(PROJECT, factory_outputs=[output])


def test_a_factory_which_is_missing_or_no_function() -> None:
    with pytest.raises(ValueError, match="no factory"):
        with_fields(PROJECT, factory=None).load_factory()
    with pytest.raises(TypeError, match="not a function"):
        with_fields(PROJECT, factory="tests.project_toy:TOY_PATH").load_factory()


def test_a_project_without_credibility() -> None:
    with pytest.raises(ValueError, match="no credibility"):
        with_fields(PROJECT, credibility=None).load_credibility()


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
def test_the_simulator_has_the_tolerances() -> None:
    settings = SimulatorSettings(
        absolute_tolerance=1e-8,
        relative_tolerance=1e-7,
        jit_optimization="O1",
        integrator={"variable_step_size": False},
    )
    assert settings.jit_optimization is JitOptimization.O1
    simulator = settings.simulator(n_workers=2)
    assert simulator.n_workers == 2
    assert simulator.integrator_settings["absolute_tolerance"] == 1e-8
    assert simulator.integrator_settings["relative_tolerance"] == 1e-7
    assert simulator.integrator_settings["variable_step_size"] is False
    assert SimulatorSettings().simulator(1).integrator_settings[
        "relative_tolerance"
    ] == (1e-10)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"jit_optimization": "O9"}, "jit_optimization"),
        (
            {"absolute_tolerance": 1e-8, "integrator": {"absolute_tolerance": 1e-9}},
            "twice",
        ),
        ({"integrator": {"jit_optimization": "O0"}}, "no settings of the integrator"),
        ({"integrator": {"n_workers": 2}}, "no settings of the integrator"),
    ],
)
def test_the_simulator_settings_refuse(fields: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        SimulatorSettings(**fields)


def test_the_figure_outputs() -> None:
    assert FigureOutputs().formats == ("svg",)
    assert FigureOutputs(formats=["SVG", "Png"]).formats == ("svg", "png")
    with pytest.raises(TypeError, match="not 'svg'"):
        FigureOutputs(formats="svg")
    with pytest.raises(ValidationError, match="figure format"):
        FigureOutputs(formats=())
    assert FigureOutputs().rc_params() == {}
    outputs = FigureOutputs(savefig_dpi=17, rc={"font.size": 9})
    assert outputs.rc_params() == {"font.size": 9, "savefig.dpi": 17}
    assert PROJECT.figures.formats == ("svg", "png")


def test_the_checks() -> None:
    reasons = {"ToyUnregistered": "a test"}
    checks = CheckSettings(not_registered=reasons, battery={"steady_state": "slow"})
    reasons["other"] = "changed"
    assert dict(checks.not_registered) == {"ToyUnregistered": "a test"}
    assert CheckSettings() == CheckSettings(not_registered={})
    assert PROJECT.checks.not_registered == {"ToyUnregistered": "a test"}
    refused: list[dict[str, Any]] = [
        {"one_at_a_time": " "},
        {"battery": {"steady_state": ""}},
    ]
    for fields in refused:
        with pytest.raises(ValidationError, match="reason"):
            CheckSettings(**fields)


def test_a_project_without_a_release() -> None:
    assert Project(name="p", root=TOY_PATH).release is None


# ----------------------------------------------------------------------------
# fits
# ----------------------------------------------------------------------------
def test_the_fits_get_the_level_of_the_project() -> None:
    from tests.project_toy.fitting import FIT_DEFINITIONS

    assert PROJECT.fit("DECAY") is FIT_DEFINITIONS["DECAY"]
    project = with_fields(PROJECT, simulator=SimulatorSettings(jit_optimization="O0"))
    definition = project.fit("DECAY")
    assert definition.jit_optimization is JitOptimization.O0
    assert definition.parameters is FIT_DEFINITIONS["DECAY"].parameters
    assert definition.problem("p").jit_optimization is JitOptimization.O0
    own = dataclasses.replace(FIT_DEFINITIONS["DECAY"], jit_optimization="O3")
    leveled = with_fields(project, fits={"DECAY": own})
    assert leveled.fit("DECAY").jit_optimization is JitOptimization.O3


def test_a_fit_definition_checks_its_level() -> None:
    from tests.project_toy.fitting import FIT_DEFINITIONS

    definition = FIT_DEFINITIONS["DECAY"]
    assert definition.jit_optimization is None
    with pytest.raises(ValueError, match="optimization level"):
        dataclasses.replace(definition, jit_optimization="O9")


def test_fits_which_are_no_fit_definitions() -> None:
    project = with_fields(PROJECT, fits=".credibility:CREDIBILITY")
    with pytest.raises(TypeError, match="no fit definitions"):
        project.load_fit_definitions()


def test_the_adopted_fits_are_in_the_fits_dir(tmp_path: Path) -> None:
    assert PROJECT.fits_dir == TOY_PATH / "fits"
    assert PROJECT.adopted_file == TOY_PATH / "fits" / ADOPTED_FILE
    project = with_fields(PROJECT, fits_dir=tmp_path)
    assert project.adopted_fits() == {}
    write_adopted(tmp_path / ADOPTED_FILE, {"DECAY": FIT})
    assert project.adopted_fits() == {"DECAY": FIT}
    other = with_fields(project, adopted=Path("adopted.json"))
    assert other.adopted_file == TOY_PATH / "adopted.json"
    assert with_fields(project, fits_dir=None).adopted_file is None


def test_the_record_of_an_adopted_fit(tmp_path: Path) -> None:
    project = with_fields(PROJECT, fits_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="parameters"):
        project.adopted_record(FIT)
    (tmp_path / FIT).mkdir()
    (tmp_path / FIT / "parameters.json").write_text("{}", encoding="utf-8")
    assert project.adopted_record(FIT) == tmp_path / FIT
    with pytest.raises(ValueError, match="no fit subset"):
        project.adopted_record("PK_20261106_120000__abcde")
    with pytest.raises(ValueError, match="not the id of a fit"):
        project.adopted_record("../DECAY_20261106_120000__abcde")
    with pytest.raises(ValueError, match="fits_dir"):
        with_fields(project, fits_dir=None).adopted_record(FIT)


def test_the_subset_of_a_fit_id() -> None:
    assert fit_subset(FIT) == "DECAY"
    assert fit_subset("PK_X_20261106_120000__abcde") == "PK_X"
    for bad in ["DECAY", "DECAY_2026_120000__abcde", "a/b_20261106_120000__abcde"]:
        with pytest.raises(ValueError, match="not the id of a fit"):
            fit_subset(bad)


# ----------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------
def test_the_analyses_of_the_toy() -> None:
    pairs = [(a["kind"], a["name"]) for a in PROJECT.analyses()]
    assert pairs == [
        ("simulate", "all"),
        *[
            (kind, "DECAY")
            for kind in ["fit", "identifiability", "fisher", "report", "petab"]
        ],
    ]


def test_run_experiments_writes_into_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    results = PROJECT.run_experiments(
        ["studies"], "results", n_workers=1, progress=False
    )
    assert [r.experiment.sid for r in results] == ["ToyDecay"]
    assert not any(r.failed for r in results)
    figures = {p.name for p in (tmp_path / "results").rglob("*_fig_decay.*")}
    assert figures == {"ToyDecay_fig_decay.svg", "ToyDecay_fig_decay.png"}
    # the static images have the resolution of the project, 30 dpi, and not
    # the 72 dpi of the figure settings
    png = tmp_path / "results" / "ToyDecay" / "ToyDecay_fig_decay.png"
    with Image.open(png) as image:
        assert image.size[0] < 7 * 40
    assert not (TOY_PATH / "results").exists()


def test_run_experiments_with_classes_and_formats(tmp_path: Path) -> None:
    from tests.project_toy.experiments import ToyDecaySlow

    results = PROJECT.run_experiments(
        ToyDecaySlow, tmp_path, n_workers=1, progress=False, figure_formats=["SVG"]
    )
    assert [r.experiment.sid for r in results] == ["ToyDecaySlow"]
    assert {p.suffix for p in tmp_path.rglob("*_fig_decay.*")} == {".svg"}
    with pytest.raises(TypeError, match="names of groups"):
        PROJECT.run_experiments(["ToyDecay", ToyDecaySlow], tmp_path)  # ty: ignore[invalid-argument-type]


def test_run_experiments_raises_after_the_report(tmp_path: Path) -> None:
    from tests.project_toy.experiments import ToyDecay

    class ToyBroken(ToyDecay):
        variant = "missing"

    with pytest.raises(ExperimentRunError):
        PROJECT.run_experiments([ToyBroken], tmp_path, n_workers=1, progress=False)
    assert (tmp_path / "index.html").exists()
