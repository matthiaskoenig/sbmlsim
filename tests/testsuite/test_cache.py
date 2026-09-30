"""Tests of the download and the cache of a test suite."""

import logging
import os
import time
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


def test_two_fetches_of_one_target(tmp_path: Path) -> None:
    """A fetch which finds the target in place uses it.

    The second fetch runs while the first one selects its cases, as a second
    process would. Each fetch has its own staging directory.
    """
    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
    target = tmp_path / "cache" / "suite" / "1.0"
    stagings: list[Path] = []

    def select(staging: Path) -> Path:
        stagings.append(staging)
        if len(stagings) == 1:
            cache.fetch(url, target, select=select)
        return staging / "suite-1.0/cases"

    assert cache.fetch(url, target, select=select) == target

    assert len(set(stagings)) == 2
    assert (target / "001" / "a.txt").read_text() == "a"
    assert [p.name for p in target.parent.iterdir()] == ["1.0"]


def test_the_download_is_logged_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The suite says what it downloads, `fetch` adds the URL at DEBUG."""
    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
    target = tmp_path / "cache" / "suite" / "1.0"
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.testsuite.cache"):
        cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")
    records = [r for r in caplog.records if r.name == "sbmlsim.testsuite.cache"]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
    assert any(url in r.getMessage() for r in records)


@pytest.mark.parametrize("member", ["../evil.txt", "../../evil.txt", "ABSOLUTE"])
def test_a_member_outside_of_the_archive_stays_inside(
    tmp_path: Path, member: str
) -> None:
    """A member with `..` or an absolute path is unpacked below the staging."""
    outside = tmp_path / "evil.txt"
    if member == "ABSOLUTE":
        member = str(outside)
    url = _archive(
        tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a", member: "evil"}
    )
    target = tmp_path / "cache" / "suite" / "1.0"

    cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")

    assert (target / "001" / "a.txt").read_text() == "a"
    assert list(tmp_path.rglob("evil.txt")) == []


def test_a_stale_staging_directory_is_removed(tmp_path: Path) -> None:
    """A fetch removes what a killed fetch of its target left behind."""
    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
    target = tmp_path / "cache" / "suite" / "1.0"
    target.parent.mkdir(parents=True)
    stale = target.parent / ".1.0.abc.incomplete"
    stale.mkdir()
    (stale / "archive.zip").write_text("x")
    old = time.time() - 2 * cache.STALE_AFTER
    os.utime(stale, (old, old))
    fresh = target.parent / ".1.0.def.incomplete"
    fresh.mkdir()
    other = target.parent / ".2.0.abc.incomplete"
    other.mkdir()
    os.utime(other, (old, old))

    cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")

    assert sorted(p.name for p in target.parent.iterdir()) == [
        ".1.0.def.incomplete",
        ".2.0.abc.incomplete",
        "1.0",
    ]
    assert cache.remove_stale(target / "x", stale_after=0.0) == []
