"""Tests of the provenance of a fit."""

import shutil
import subprocess
from pathlib import Path, PureWindowsPath

import pytest

from examples.hctz_fitting import DATA_PATH
from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.provenance import (
    FitProvenance,
    GitInfo,
    checksum_key,
    collect_versions,
    create_provenance,
    data_files,
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


def test_checksum_keys_are_posix_paths() -> None:
    root = PureWindowsPath(r"C:\repo")
    path = PureWindowsPath(r"C:\repo\data\Beermann1976\.Beermann1976_Fig1.tsv")
    assert checksum_key(path, root) == "data/Beermann1976/.Beermann1976_Fig1.tsv"
    assert (
        checksum_key(path, None) == "C:/repo/data/Beermann1976/.Beermann1976_Fig1.tsv"
    )
    other = PureWindowsPath(r"D:\elsewhere\model.xml")
    assert checksum_key(other, root) == "D:/elsewhere/model.xml"


def test_nested_checksum_keys_use_slashes(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "data" / ".study.tsv").write_text("a\tb\n")
    sums = file_checksums([repo / "data" / ".study.tsv"], root=repo)
    assert list(sums) == ["data/.study.tsv"]


def _data_with_clutter(tmp_path: Path) -> Path:
    """Copy the HCTZ data and add files no experiment reads."""
    data = tmp_path / "data"
    shutil.copytree(DATA_PATH, data)
    (data / "__init__.py").write_text("")
    (data / "collect_data.py").write_text("")
    (data / "__pycache__").mkdir()
    (data / "__pycache__" / "collect_data.cpython-314.pyc").write_bytes(b"\0")
    (data / ".ipynb_checkpoints").mkdir()
    (data / ".ipynb_checkpoints" / ".Beermann1976_Fig1-checkpoint.tsv").write_text("")
    (data / ".DS_Store").write_bytes(b"\0")
    (data / "Beermann1976" / ".gitignore").write_text("*.png\n")
    return data


def test_data_files_keep_the_study_data(tmp_path: Path) -> None:
    data = _data_with_clutter(tmp_path)
    study_data = sorted(
        p.resolve() for p in data.rglob("*.tsv") if ".ipynb" not in str(p)
    )
    assert len(study_data) == len(list(DATA_PATH.rglob("*.tsv")))
    assert all(p.name.startswith(".") for p in study_data), "PK-DB data are hidden"
    assert data_files([data]) == study_data


def test_a_problem_hashes_its_models_and_its_data(
    op_hctz_pk: OptimizationProblem, tmp_path: Path
) -> None:
    data = _data_with_clutter(tmp_path)
    op_hctz_pk.data_path = data
    names = [p.name for p in op_hctz_pk.files()]
    assert sum(name.endswith(".xml") for name in names) >= 1
    tsv = [name for name in names if name.endswith(".tsv")]
    assert len(tsv) == len(list(DATA_PATH.rglob("*.tsv")))
    assert all(name.endswith((".xml", ".tsv")) for name in names), names


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
