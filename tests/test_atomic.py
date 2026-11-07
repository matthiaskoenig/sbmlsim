"""Atomic writes keep the mode of a new file and leave no partial file."""

import os
import stat
from pathlib import Path

import pytest

from sbmlsim.atomic import atomic_directory, atomic_path, write_text_atomic


def _mode(path: Path) -> int:
    """Get the permission bits of a file."""
    return stat.S_IMODE(path.stat().st_mode)


def test_mode_is_the_one_of_a_new_file(tmp_path: Path) -> None:
    """The file gets the mode of a file written with `Path.write_text`."""
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    path = tmp_path / "run.json"
    write_text_atomic(path, "{}")
    assert _mode(path) == _mode(reference)


def test_mode_follows_the_umask(tmp_path: Path) -> None:
    """A file written under another umask has the mode of that umask."""
    previous = os.umask(0o027)
    try:
        reference = tmp_path / "reference.json"
        reference.write_text("{}", encoding="utf-8")
        path = tmp_path / "run.json"
        write_text_atomic(path, "{}")
    finally:
        os.umask(previous)
    assert _mode(path) == _mode(reference)


def test_directory_holds_only_the_target(tmp_path: Path) -> None:
    """Nothing but the written file is left, also when it is written again."""
    path = tmp_path / "run.json"
    write_text_atomic(path, "first")
    write_text_atomic(path, "second")
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    assert path.read_text(encoding="utf-8") == "second"


def test_failing_writer_keeps_the_old_file(tmp_path: Path) -> None:
    """A writer which raises leaves the old content and no temporary file."""
    path = tmp_path / "run.json"
    write_text_atomic(path, "old")
    with pytest.raises(RuntimeError, match="broken"), atomic_path(path) as temporary:
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("broken")
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_writer_which_replaces_the_temporary_file(tmp_path: Path) -> None:
    """A writer may replace the temporary file, e.g. a library which removes it."""
    reference = tmp_path / "reference.nc"
    reference.write_bytes(b"\x00")
    path = tmp_path / "result.nc"
    with atomic_path(path) as temporary:
        temporary.unlink()
        temporary.write_bytes(b"\x01")
        temporary.chmod(0o600)
    assert path.read_bytes() == b"\x01"
    assert _mode(path) == _mode(reference)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reference.nc", "result.nc"]


def test_a_directory_is_replaced_once_it_is_complete(tmp_path: Path) -> None:
    target = tmp_path / "local"
    with atomic_directory(target) as temporary:
        assert not target.exists()
        (temporary / "a.tsv").write_text("first", encoding="utf-8")
    assert (target / "a.tsv").read_text(encoding="utf-8") == "first"
    with atomic_directory(target) as temporary:
        (temporary / "b.tsv").write_text("second", encoding="utf-8")
        # the old directory stays until the new one is complete
        assert (target / "a.tsv").exists()
    assert sorted(p.name for p in target.iterdir()) == ["b.tsv"]
    assert [p.name for p in tmp_path.iterdir()] == ["local"]


def test_a_failed_directory_leaves_the_old_one(tmp_path: Path) -> None:
    target = tmp_path / "local"
    target.mkdir()
    (target / "a.tsv").write_text("kept", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed"), atomic_directory(target) as tmp:
        (tmp / "b.tsv").write_text("lost", encoding="utf-8")
        raise RuntimeError("failed")
    assert sorted(p.name for p in target.iterdir()) == ["a.tsv"]
    assert [p.name for p in tmp_path.iterdir()] == ["local"]


def test_a_directory_has_the_mode_of_a_new_directory(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    with atomic_directory(tmp_path / "local") as temporary:
        (temporary / "a.tsv").write_text("x", encoding="utf-8")
    assert _mode(tmp_path / "local") == _mode(reference)


def test_a_failed_replacement_restores_the_old_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old directory is moved aside; it is moved back when the new one fails."""
    target = tmp_path / "local"
    target.mkdir()
    (target / "a.tsv").write_text("kept", encoding="utf-8")
    rename = Path.rename

    def failing(self: Path, other: Path) -> Path:
        if self.name.endswith(".tmp"):
            raise OSError("rename failed")
        return rename(self, other)

    monkeypatch.setattr(Path, "rename", failing)
    with (
        pytest.raises(OSError, match="rename failed"),
        atomic_directory(target) as temporary,
    ):
        (temporary / "b.tsv").write_text("lost", encoding="utf-8")
    assert sorted(p.name for p in target.iterdir()) == ["a.tsv"]
    assert [p.name for p in tmp_path.iterdir()] == ["local"]


def test_an_old_directory_which_is_not_removed_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import shutil

    target = tmp_path / "local"
    target.mkdir()

    def failing(path: Path, *args: object, **kwargs: object) -> None:
        raise OSError("busy")

    monkeypatch.setattr(shutil, "rmtree", failing)
    with atomic_directory(target) as temporary:
        (temporary / "b.tsv").write_text("new", encoding="utf-8")
    assert (target / "b.tsv").exists()
    assert "is not removed: busy" in caplog.text
