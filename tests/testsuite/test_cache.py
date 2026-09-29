"""Tests of the download and the cache of a test suite."""

import zipfile
from pathlib import Path

import pytest

from sbmlsim.testsuite import cache
from sbmlsim.testsuite.cases import SemanticSuite


def _archive(path: Path, files: dict[str, str]) -> str:
    """Write a zip archive and return its URL."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path.as_uri()


def test_the_cache_is_in_the_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache follows `XDG_CACHE_HOME`, a variable points elsewhere."""
    monkeypatch.delenv("SBMLSIM_SOME_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert cache.cache_root() == Path("/tmp/cache/sbmlsim")
    assert cache.cache_path("SBMLSIM_SOME_SUITE_PATH", "suite", "1.0") == Path(
        "/tmp/cache/sbmlsim/suite/1.0"
    )

    monkeypatch.setenv("SBMLSIM_SOME_SUITE_PATH", "/elsewhere")
    assert cache.cache_path("SBMLSIM_SOME_SUITE_PATH", "suite", "1.0") == Path(
        "/elsewhere"
    )


def test_the_cache_without_xdg_is_in_the_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """`~/.cache` is the user cache when `XDG_CACHE_HOME` is not set or empty."""
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert cache.cache_root() == Path.home() / ".cache" / "sbmlsim"


def test_an_archive_is_unpacked_into_place(tmp_path: Path) -> None:
    """The selected directory of the archive becomes the target."""
    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
    target = tmp_path / "cache" / "suite" / "1.0"

    path = cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")

    assert path == target
    assert (target / "001" / "a.txt").read_text() == "a"
    assert [p.name for p in target.parent.iterdir()] == ["1.0"]


def test_a_failed_download_leaves_nothing(tmp_path: Path) -> None:
    """Neither the target nor the staging directory exist after a failure."""
    target = tmp_path / "cache" / "suite" / "1.0"
    with pytest.raises(OSError):
        cache.fetch((tmp_path / "missing.zip").as_uri(), target, select=lambda s: s)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_a_file_which_is_not_an_archive(tmp_path: Path) -> None:
    """A download which is not a zip archive is an error and leaves nothing."""
    (tmp_path / "page.zip").write_text("<html>not found</html>")
    target = tmp_path / "cache" / "suite" / "1.0"
    with pytest.raises(zipfile.BadZipFile):
        cache.fetch((tmp_path / "page.zip").as_uri(), target, select=lambda s: s)
    assert list(target.parent.iterdir()) == []


def test_an_archive_without_cases(tmp_path: Path) -> None:
    """The error of the selection is raised and the staging is removed."""
    url = _archive(tmp_path / "suite.zip", {"readme.txt": "no cases"})
    target = tmp_path / "cache" / "semantic"
    with pytest.raises(OSError, match="No case directories"):
        cache.fetch(url, target, select=SemanticSuite._cases_dir)
    assert list(target.parent.iterdir()) == []


def test_the_semantic_suite_is_loaded_through_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SemanticSuite.load` downloads into the path of its release."""
    url = _archive(
        tmp_path / "semantic.zip", {"semantic/00001/00001-settings.txt": "start: 0"}
    )
    monkeypatch.delenv("SBMLSIM_TEST_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.testsuite.cases.SUITE_URL", url)

    suite = SemanticSuite.load("9.9.9")

    assert suite.version == "9.9.9"
    assert suite.path == tmp_path / "cache/sbmlsim/test-suite/9.9.9/semantic"
    assert (suite.path / "00001" / "00001-settings.txt").is_file()
    assert SemanticSuite.cached("9.9.9") == suite
