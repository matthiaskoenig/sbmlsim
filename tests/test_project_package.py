"""The project of a model package: references, variants, factory and settings.

The tests run the project of the toy package `tests.project_toy`.
"""

import ast
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


#: the modules which importing a project must not import
HEAVY = ("roadrunner", "sbmlsim.experiment", "sbmlsim.fit", "sbmlsim.simulator")


def imported(module: str, prefix: str) -> list[str]:
    """Get the modules of the prefixes an import imports, in a fresh process."""
    code = (
        f"import sys, {module}; "
        f"print(sorted(m for m in sys.modules if m.startswith({prefix!r})))"
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
    return ast.literal_eval(result.stdout.strip())


def heavy(module: str) -> list[str]:
    """Get the heavy modules an import imports, see `HEAVY`."""
    return [m for m in imported(module, "") if m in HEAVY]


def test_importing_a_project_imports_none_of_its_references() -> None:
    assert imported("tests.project_toy", "tests.project_toy.") == []
    assert heavy("tests.project_toy") == []


def test_the_project_module_is_light() -> None:
    assert heavy("sbmlsim.project") == []


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
        PROJECT.replace(experiments=reference)


def test_a_relative_reference_needs_a_package() -> None:
    with pytest.raises(ValueError, match="needs a package"):
        resolve(".experiments:EXPERIMENTS")
    with pytest.raises(ValidationError, match="needs the package"):
        PROJECT.replace(package=None)


def test_a_missing_attribute_names_the_reference() -> None:
    project = PROJECT.replace(credibility=".credibility:MISSING")
    with pytest.raises(AttributeError, match=r"\.credibility:MISSING"):
        project.load_credibility()


def test_a_reference_is_resolved_once_and_is_no_part_of_the_value() -> None:
    project = PROJECT.replace()
    assert project == PROJECT
    first = project.load_credibility()
    assert first == {"context": "the decay of A"}
    assert project.load_credibility() is first
    assert project == PROJECT.replace()
    # a copy of pydantic shares the resolved references, which are keyed by
    # the reference
    copy = project.model_copy(update={"credibility": ".experiments:EXPERIMENTS"})
    assert copy.load_credibility() is resolve(
        ".experiments:EXPERIMENTS", "tests.project_toy"
    )
    assert project.load_credibility() is first


def test_replace_validates_the_changes() -> None:
    project = PROJECT.replace(name="other", fits_dir=Path("records"))
    assert project.name == "other"
    assert project.fits_dir == TOY_PATH / "records"
    assert project.variants == PROJECT.variants
    with pytest.raises(ValidationError, match="module:attribute"):
        PROJECT.replace(experiments="no reference")
    # the paths are stored relative to the root, a new root does not move them
    moved = PROJECT.replace(root=Path("/elsewhere"), factory=None)
    assert moved.variants == PROJECT.variants


def test_a_pickled_project_drops_its_resolved_references() -> None:
    project = PROJECT.replace()
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

    project = PROJECT.replace(
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
        project = PROJECT.replace(experiments=value)
        assert project.load_experiments() == {"all": [ToyDecay, ToyDecaySlow]}


def test_two_experiments_of_one_name_are_refused() -> None:
    from tests.project_toy.experiments import ToyDecay

    other = type("ToyDecay", (SimulationExperiment,), {})
    project = PROJECT.replace(experiments={"a": [ToyDecay], "b": [other]})
    with pytest.raises(ValueError, match="two classes named 'ToyDecay'") as info:
        project.experiment_classes(["ToyDecay"])
    assert "tests.project_toy.experiments.ToyDecay" in str(info.value)
    assert f"{__name__}.ToyDecay" in str(info.value)


def test_experiments_which_are_no_experiments() -> None:
    project = PROJECT.replace(experiments=".credibility:CREDIBILITY")
    with pytest.raises(TypeError, match="no SimulationExperiment"):
        project.load_experiments()
    project = PROJECT.replace(experiments=None)
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
    assert PROJECT.variant_path() == TOY_PATH / "models" / "toy.xml"
    assert PROJECT.variant_path("slow") == TOY_PATH / "models" / "toy_slow.xml"
    with pytest.raises(KeyError, match="slow"):
        PROJECT.variant_path("female")
    with pytest.raises(ValueError, match="no model variants"):
        PROJECT.replace(variants={}).variant_path()


def test_the_model_path_is_the_default_variant() -> None:
    assert PROJECT.model_path == TOY_PATH / "models" / "toy.xml"
    assert str(PROJECT.model_path) == str(TOY_PATH / "models" / "toy.xml")
    with pytest.raises(ValueError, match="no model variants"):
        _ = PROJECT.replace(variants={}).model_path


def test_every_variant_is_a_file_of_its_own() -> None:
    with pytest.raises(ValidationError, match="file of its own"):
        PROJECT.replace(variants={"a": Path("toy.xml"), "b": TOY_PATH / "toy.xml"})


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
    project = PROJECT.replace(simulator=SimulatorSettings(jit_optimization="O0"))
    r = project.load_roadrunner("slow")
    assert calls == [(PROJECT.variant_path("slow"), JitOptimization.O0)]
    assert r.getValue("k") == pytest.approx(0.1)
    assert PROJECT.load_roadrunner().getValue("k") == pytest.approx(0.2)


# ----------------------------------------------------------------------------
# factory and credibility
# ----------------------------------------------------------------------------
def test_the_factory_directory_is_the_one_of_the_first_variant() -> None:
    assert PROJECT.factory_dir == TOY_PATH / "models"
    assert PROJECT.factory_paths() == [
        TOY_PATH / "models" / "toy.xml",
        TOY_PATH / "models" / "toy_slow.xml",
    ]
    project = PROJECT.replace(factory_dir=Path("build"))
    assert project.factory_dir == TOY_PATH / "build"


def test_the_factory_builds_in_place(tmp_path: Path) -> None:
    project = PROJECT.replace(
        root=tmp_path,
        variants={
            "default": Path("models/toy.xml"),
            "slow": Path("models/toy_slow.xml"),
        },
        factory_dir=None,
    )
    assert project.factory_dir == tmp_path / "models"
    assert project.create_models() == tmp_path / "models"
    for built, path in zip(
        project.factory_paths(), PROJECT.factory_paths(), strict=True
    ):
        assert built.read_bytes() == path.read_bytes()


def test_a_rebuild_has_the_layout_of_the_package(tmp_path: Path) -> None:
    factory_dir = PROJECT.factory_dir
    assert factory_dir is not None
    out = tmp_path / "rebuild"
    built = PROJECT.create_models(out / factory_dir.relative_to(PROJECT.root))
    assert built == out / "models"
    rebuilt = PROJECT.factory_paths(out)
    assert rebuilt == [out / o for o in PROJECT.factory_outputs]
    for new, old in zip(rebuilt, PROJECT.factory_paths(), strict=True):
        assert new.read_bytes() == old.read_bytes()


def test_the_factory_directory_is_inside_the_root(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="outside of the root"):
        PROJECT.replace(factory_dir=tmp_path)
    project = PROJECT.replace(variants={}, factory_dir=None)
    assert project.factory_dir is None
    with pytest.raises(ValueError, match="factory_dir"):
        project.create_models()


@pytest.mark.parametrize("output", [TOY_PATH / "toy.xml", Path("../toy.xml")])
def test_the_outputs_of_the_factory_are_inside_the_root(output: Path) -> None:
    with pytest.raises(ValidationError, match="relative to the root"):
        PROJECT.replace(factory_outputs=[output])


def test_a_factory_which_is_missing_or_no_function() -> None:
    with pytest.raises(ValueError, match="no factory"):
        PROJECT.replace(factory=None).load_factory()
    with pytest.raises(TypeError, match="not a function"):
        PROJECT.replace(factory="tests.project_toy:TOY_PATH").load_factory()


def test_a_project_without_credibility() -> None:
    with pytest.raises(ValueError, match="no credibility"):
        PROJECT.replace(credibility=None).load_credibility()


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

    without = PROJECT.replace(simulator=SimulatorSettings())
    assert without.fit("DECAY") is FIT_DEFINITIONS["DECAY"]
    project = PROJECT.replace(simulator=SimulatorSettings(jit_optimization="O0"))
    definition = project.fit("DECAY")
    assert definition.jit_optimization is JitOptimization.O0
    assert definition.parameters is FIT_DEFINITIONS["DECAY"].parameters
    assert definition.problem("p").jit_optimization is JitOptimization.O0
    own = dataclasses.replace(FIT_DEFINITIONS["DECAY"], jit_optimization="O3")
    leveled = project.replace(fits={"DECAY": own})
    assert leveled.fit("DECAY").jit_optimization is JitOptimization.O3


def test_a_fit_definition_checks_its_level() -> None:
    from tests.project_toy.fitting import FIT_DEFINITIONS

    definition = FIT_DEFINITIONS["DECAY"]
    assert definition.jit_optimization is None
    with pytest.raises(ValueError, match="optimization level"):
        dataclasses.replace(definition, jit_optimization="O9")


def test_fits_which_are_no_fit_definitions() -> None:
    project = PROJECT.replace(fits=".credibility:CREDIBILITY")
    with pytest.raises(TypeError, match="no fit definitions"):
        project.load_fit_definitions()


def test_the_adopted_fits_are_in_the_fits_dir(tmp_path: Path) -> None:
    assert PROJECT.fits_dir == TOY_PATH / "fits"
    assert PROJECT.adopted_file == TOY_PATH / "fits" / ADOPTED_FILE
    project = PROJECT.replace(fits_dir=tmp_path)
    assert project.adopted_fits() == {}
    write_adopted(tmp_path / ADOPTED_FILE, {"DECAY": FIT})
    assert project.adopted_fits() == {"DECAY": FIT}
    other = project.replace(adopted=Path("adopted.json"))
    assert other.adopted_file == TOY_PATH / "adopted.json"
    assert project.replace(fits_dir=None).adopted_file is None


def test_the_record_of_an_adopted_fit(tmp_path: Path) -> None:
    project = PROJECT.replace(fits_dir=tmp_path)
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
        project.replace(fits_dir=None).adopted_record(FIT)


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
        ("scan", "decay"),
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


def test_the_figure_outputs_reach_the_workers_of_a_pool(tmp_path: Path) -> None:
    results = PROJECT.run_experiments(None, tmp_path, n_workers=2, progress=False)
    assert len(results) == 3
    pngs = sorted(tmp_path.rglob("*_fig_decay.png"))
    assert len(pngs) == 3
    for png in pngs:
        with Image.open(png) as image:
            assert image.size[0] < 7 * 40, png


def test_run_experiments_compiles_at_the_level_of_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbmlsim.experiment import runner

    levels: list[JitOptimization | None] = []

    class Recording(runner.ExperimentRunner):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            levels.append(self.jit_optimization)
            levels.extend(
                getattr(model, "jit_optimization", None)
                for model in self.models.values()
            )

    monkeypatch.setattr(runner, "ExperimentRunner", Recording)
    PROJECT.run_experiments(["studies"], tmp_path, n_workers=1, progress=False)
    assert levels == [JitOptimization.O0, JitOptimization.O0]


def test_a_scan_compiles_at_the_level_of_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbmlsim import run
    from sbmlsim.model import model_sbml
    from sbmlsim.result import ScanResult

    loaded: list[Any] = []
    load_model = model_sbml.load_model

    def recording(*args: Any, **kwargs: Any) -> Any:
        model = load_model(*args, **kwargs)
        level = getattr(model, "jit_optimization", None)
        loaded.append((kwargs.get("jit_optimization"), level))
        return model

    monkeypatch.setattr(model_sbml, "load_model", recording)
    spec = {"project": "tests.project_toy:PROJECT", "analysis": "decay"}
    run.execute("scan", spec, tmp_path / "out", workers=1)
    assert loaded == [(JitOptimization.O0, JitOptimization.O0)]
    result = ScanResult.from_netcdf(tmp_path / "out" / "result.nc")
    assert result.dims == ("rate",)


def test_the_simulate_kind_runs_a_subclass_which_overrides_run_experiments(
    tmp_path: Path,
) -> None:
    from sbmlsim import run

    summary = run.execute(
        "simulate",
        {"project": "tests.project_toy.legacy:LEGACY", "experiments": ["ToyDecay"]},
        tmp_path / "out",
    )
    assert summary == {"experiments": 1, "failed": 0}
