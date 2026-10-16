"""Tests of the provenance of a fit."""

import subprocess
from pathlib import Path

import pytest

from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.provenance import (
    FitProvenance,
    GitInfo,
    collect_versions,
    create_provenance,
    file_checksums,
    git_info,
)
from sbmlsim.fit.result import OptimizationResult


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.org")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "model.xml").write_text("<sbml/>")
    _git(tmp_path, "add", "model.xml")
    _git(tmp_path, "commit", "-q", "-m", "model")
    return tmp_path


def test_versions_name_the_simulation_stack() -> None:
    versions = collect_versions()
    assert {"python", "sbmlsim", "libroadrunner", "scipy", "numpy"} <= set(versions)


def test_git_info_of_a_clean_and_a_dirty_tree(repo: Path) -> None:
    clean = git_info(repo / "model.xml")
    assert clean is not None and len(clean.sha) == 40 and not clean.dirty
    (repo / "model.xml").write_text("<sbml>changed</sbml>")
    dirty = git_info(repo)
    assert dirty is not None and dirty.dirty


def test_git_info_outside_a_repository(tmp_path: Path) -> None:
    assert git_info(tmp_path) is None


def test_checksums_are_relative_to_the_root(repo: Path) -> None:
    sums = file_checksums([repo / "model.xml"], root=repo)
    assert list(sums) == ["model.xml"]
    assert len(sums["model.xml"]) == 64


def test_round_trip(repo: Path) -> None:
    provenance = create_provenance(
        arguments={"runs": 3, "method": "LSQ"},
        seed=1234,
        run_seeds=[None, None, None],
        base_path=repo,
        files=[repo / "model.xml"],
        start_values={"k": 1.0},
        fixed_values={"f": 2.0},
    )
    assert isinstance(provenance.git, GitInfo)
    assert FitProvenance.from_dict(provenance.to_dict()) == provenance


def test_a_fit_stores_its_provenance(
    definition_hctz_pk: FitDefinition, tmp_path: Path
) -> None:
    runs = run_fit(
        definition_hctz_pk,
        size=2,
        n_cores=1,
        seed=1234,
        output_dir=tmp_path,
        fixed_values={"x": 1.5},
        max_nfev=1,
    )
    (run,) = runs.values()
    provenance = run.result.provenance
    assert provenance is not None
    assert provenance.seed == 1234
    assert provenance.arguments["size"] == 2
    assert len(provenance.run_seeds) == 2
    assert provenance.files, "the models and the data have checksums"
    assert provenance.start_values
    assert provenance.fixed_values == {"x": 1.5}
    paths = sorted((tmp_path / run.problem.opid / "runs").glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        stored = OptimizationResult.from_json(path)
        assert stored.provenance == provenance
    combined = OptimizationResult.from_directory(tmp_path / run.problem.opid / "runs")
    assert combined.provenance == provenance
