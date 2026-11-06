"""The project descriptor: the analyses of a model package."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sbmlsim.engine import Engine
from sbmlsim.project import (
    ADOPTED_FILE,
    Project,
    SimulatorSettings,
    load_project,
    write_adopted,
)
from sbmlsim.simulation import Scan

HCTZ_PROJECT = "examples.hctz_fitting.project:PROJECT"


@pytest.fixture(scope="module")
def project() -> Project:
    return load_project(HCTZ_PROJECT)


def test_load_project(project: Project) -> None:
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    assert isinstance(project, Project)
    assert project.name == "hctz"
    assert list(project.fits) == ["PK", "IV"]
    assert project.fit("PK") is FIT_DEFINITIONS["PK"]
    assert set(project.scans) == {"dose", "sobol"}
    assert set(project.uncertainties) == {"cl"}
    assert project.experiments is not None
    assert [e.__name__ for e in project.experiments()] == [
        "Beermann1976",
        "Patel1984",
        "Weir1998",
    ]


def test_load_project_refuses_a_ref_without_colon() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        load_project("examples.hctz_fitting.project")


def test_load_project_refuses_no_project() -> None:
    with pytest.raises(ValueError, match="no Project"):
        load_project("examples.hctz_fitting:HCTZ_PATH")
    with pytest.raises(ValueError, match="no attribute"):
        load_project("examples.hctz_fitting.project:MISSING")


def test_load_project_passes_on_an_import_error() -> None:
    with pytest.raises(ImportError):
        load_project("examples.no_such_module:PROJECT")


def test_unknown_names_are_named(project: Project) -> None:
    with pytest.raises(KeyError) as info:
        project.fit("PD")
    assert "PK" in str(info.value)
    assert "IV" in str(info.value)
    with pytest.raises(KeyError, match="dose"):
        project.scan("other")
    with pytest.raises(KeyError, match="cl"):
        project.uncertainty("other")


def test_analyses_lists_every_kind(project: Project) -> None:
    analyses = project.analyses()
    assert all(set(a) == {"kind", "name", "description"} for a in analyses)
    assert all(a["description"] for a in analyses)
    pairs = [(a["kind"], a["name"]) for a in analyses]
    assert pairs == [
        ("simulate", "all"),
        *[
            (kind, subset)
            for subset in ["PK", "IV"]
            for kind in ["fit", "identifiability", "fisher", "report", "petab"]
        ],
        ("scan", "dose"),
        ("scan", "sobol"),
        ("sensitivity", "sobol"),
        ("uncertainty", "cl"),
    ]
    assert ("sensitivity", "dose") not in pairs


def test_analyses_of_a_project_without_analyses(tmp_path: Path) -> None:
    assert Project(name="empty", root=tmp_path).analyses() == []


def test_adopted_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ADOPTED_FILE
    project = Project(name="p", root=tmp_path, adopted=path)
    assert project.adopted_fits() == {}
    write_adopted(path, {"PK": "fit_b", "IV": "fit_a"})
    assert project.adopted_fits() == {"IV": "fit_a", "PK": "fit_b"}
    text = path.read_text(encoding="utf-8")
    assert list(json.loads(text)["fits"]) == ["IV", "PK"]
    assert Project(name="p", root=tmp_path).adopted_fits() == {}


def test_adopted_refuses_a_file_of_another_format(tmp_path: Path) -> None:
    path = tmp_path / ADOPTED_FILE
    path.write_text(json.dumps({"fits": {"PK": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="adopted"):
        Project(name="p", root=tmp_path, adopted=path).adopted_fits()


def test_a_project_is_frozen_and_forbids_extra_fields(tmp_path: Path) -> None:
    project = Project(name="p", root=tmp_path)
    with pytest.raises(ValidationError):
        project.name = "q"
    with pytest.raises(ValidationError):
        Project(name="p", root=tmp_path, other=1)


def test_simulator_settings() -> None:
    settings = SimulatorSettings(
        engine="roadrunner", integrator={"relative_tolerance": 1e-8}
    )
    assert settings.engine is Engine.ROADRUNNER
    simulator = settings.simulator(n_workers=1)
    assert simulator.engine is Engine.ROADRUNNER
    assert simulator.n_workers == 1
    with pytest.raises(ValidationError):
        SimulatorSettings(engine="none")


def test_the_callables_build_deterministic_objects(project: Project) -> None:
    for name in project.scans:
        first, second = project.scan(name).scan(), project.scan(name).scan()
        assert isinstance(first, Scan)
        assert isinstance(second, Scan)
        assert first.dims == second.dims
        for d1, d2 in zip(first.dimensions, second.dimensions, strict=True):
            assert d1.values.keys() == d2.values.keys()
            for target, v1 in d1.values.items():
                v2 = d2.values[target]
                assert str(getattr(v1, "units", "")) == str(getattr(v2, "units", ""))
                np.testing.assert_array_equal(
                    getattr(v1, "magnitude", v1), getattr(v2, "magnitude", v2)
                )
    sobol = project.scan("sobol")
    assert sobol.indices == "sobol"
    times = project.scan("dose").time
    assert times is not None
    assert len(times) == 49
    cl = project.uncertainty("cl")
    assert cl.n == 32
    assert cl.seed == 1


def test_import_sbmlsim_does_not_import_project() -> None:
    code = "import sys, sbmlsim; print('sbmlsim.project' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_the_scans_of_the_example_run(project: Project) -> None:
    from sbmlsim import sensitivity

    results = {}
    for name in ["dose", "sobol"]:
        analysis = project.scan(name)
        observables = analysis.observables() if analysis.observables else None
        simulator = project.simulator.simulator(n_workers=1)
        results[name] = simulator.run(
            analysis.model,
            analysis.scan(),
            observables,
            time=analysis.time,
            keep=analysis.keep,
            on_error=analysis.on_error,
        )
        assert np.isfinite(results[name].ds["pk.cmax"]).all()
    assert results["dose"].ds.sizes["time"] == 49
    assert results["dose"].ds.sizes["dose"] == 3
    indices = sensitivity.sobol(results["sobol"], **project.scan("sobol").index_options)
    assert "pk.auc_inf_obs.ST" in indices.ds


def test_the_uncertainty_of_the_example_propagates(project: Project) -> None:
    from sbmlsim.uncertainty import propagate

    analysis = project.uncertainty("cl")
    result = propagate(
        analysis.model,
        analysis.simulation(),
        analysis.uq(),
        analysis.observables() if analysis.observables else (),
        outputs=analysis.outputs,
        n=analysis.n,
        seed=analysis.seed,
        keep=analysis.keep,
        time=analysis.time,
        on_error=analysis.on_error,
    )
    summary = result.summary(list(analysis.outputs or []))
    assert np.isfinite(summary["mean"]).all()
