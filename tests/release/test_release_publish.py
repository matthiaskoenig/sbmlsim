"""Tests of the publishing of a release."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from sbmlsim.release.publish import (
    check_lock,
    check_tools,
    publish,
)
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError
from tests.project_toy import PROJECT

SETTINGS = cast(ReleaseSettings, PROJECT.release)
VERSION = "1.0.0"
IMAGE = SETTINGS.docker_image
REPO = SETTINGS.repository


class FakeRunner:
    """Keeps the state of git, GitHub and the registry between calls."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.dirty = True
        self.head = "c0"
        self.head_version: str | None = None
        self.tags: dict[str, str] = {}
        self.remote: dict[str, str] = {}
        self.pages: str | None = None
        self.images: dict[str, str] = {}
        self.local_images: set[str] = set()
        self.releases: dict[str, list[str]] = {}
        self.fail_on: str | None = None
        self.missing: set[str] = set()
        self.streamed: list[list[str]] = []

    def __call__(self, args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        cmd = [str(a) for a in args]
        self.commands.append(cmd)
        if stream:
            self.streamed.append(cmd)
        joined = " ".join(cmd)
        if self.fail_on and self.fail_on in joined:
            self.fail_on = None
            raise ReleaseError(f"'{joined}' failed")
        if joined in self.missing:
            raise ReleaseError(f"'{joined}' failed")
        return self._run(cmd, joined)

    def _run(self, cmd: list[str], joined: str) -> str:
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return " M x\n" if self.dirty else ""
        if cmd[:2] == ["git", "show"]:
            if self.head_version is None:
                raise ReleaseError("no manifest")
            return json.dumps({"version": self.head_version})
        if cmd[:2] == ["git", "add"]:
            return ""
        if cmd[:2] == ["git", "commit"]:
            self.dirty = False
            self.head = "c1"
            self.head_version = VERSION
            return ""
        if cmd[:3] == ["git", "tag", "--list"]:
            return cmd[3] if cmd[3] in self.tags else ""
        if cmd[:2] == ["git", "rev-list"]:
            # a tag is named by its full ref, never a branch of the same name
            assert cmd[-1].startswith("refs/tags/")
            return self.tags[cmd[-1].removeprefix("refs/tags/")]
        if cmd[:2] == ["git", "rev-parse"]:
            return self.head
        if cmd[:2] == ["git", "tag"]:
            self.tags[cmd[2]] = self.head
            return ""
        if cmd[:2] == ["git", "ls-remote"]:
            return "".join(f"{sha}\t{ref}\n" for ref, sha in self.remote.items())
        if cmd[:2] == ["git", "push"]:
            self.remote["refs/heads/main"] = self.head
            self.remote[f"refs/tags/{cmd[-1]}"] = self.tags[cmd[-1]]
            return ""
        if cmd[:2] == ["gh", "api"]:
            return self._pages(cmd)
        if cmd[:3] == ["docker", "manifest", "inspect"]:
            if cmd[3] not in self.images:
                raise ReleaseError("no such manifest")
            return self.images[cmd[3]]
        if cmd[:4] == ["docker", "buildx", "imagetools", "create"]:
            self.images[cmd[5]] = self.images[cmd[6]]
            return ""
        if cmd[:2] == ["docker", "build"]:
            self.local_images.update(cmd[5::2][:2])
            return ""
        if cmd[:2] == ["docker", "push"]:
            self.images[cmd[2]] = "digest-1"
            return ""
        if cmd[:3] == ["gh", "release", "view"]:
            if cmd[3] not in self.releases:
                raise ReleaseError("release not found")
            assets = [{"name": n} for n in self.releases[cmd[3]]]
            return json.dumps({"assets": assets})
        if cmd[:3] == ["gh", "release", "create"]:
            self.releases[cmd[3]] = [Path(cmd[-1]).name]
            return ""
        if cmd[:3] == ["gh", "release", "upload"]:
            self.releases[cmd[3]].append(Path(cmd[4]).name)
            return ""
        return ""

    def _pages(self, cmd: list[str]) -> str:
        if cmd[2] == "-X":
            if (cmd[3] == "POST") == (self.pages is not None):
                raise ReleaseError(f"gh: {cmd[3]} of the Pages site failed")
            self.pages = cmd[-1].removeprefix("build_type=")
            return ""
        if self.pages is None:
            raise ReleaseError("gh: Not Found (HTTP 404)")
        return json.dumps({"build_type": self.pages, "status": "built"})


ARCHIVE = Path("toy.zip")
BUILD = [
    "docker", "build", "--platform", "linux/amd64",
    "-t", f"{IMAGE}:{VERSION}", "-t", f"{IMAGE}:latest", ".",
]  # fmt: skip


def _publish(runner: FakeRunner) -> list[str]:
    return publish(SETTINGS, Path("/release"), VERSION, ARCHIVE, runner)


def test_publish_all_steps() -> None:
    runner = FakeRunner()
    steps = _publish(runner)
    assert steps == [
        "image", "commit", "tag", "pages", "push", "docker", "github_release"
    ]  # fmt: skip
    mutating = [
        c
        for c in runner.commands
        if c[:2] in (["git", "add"], ["git", "commit"], ["git", "push"])
        or c[:2] in (["docker", "build"], ["docker", "push"])
        or c[:3] == ["gh", "release", "create"]
        or c == ["git", "tag", VERSION]
        or c[:4] == ["gh", "api", "-X", "POST"]
    ]
    assert mutating == [
        BUILD,
        ["git", "add", "-A"],
        ["git", "commit", "-m", f"Release {VERSION}"],
        ["git", "tag", VERSION],
        ["gh", "api", "-X", "POST", f"repos/{REPO}/pages", "-f", "build_type=workflow"],
        ["git", "push", "origin", "main", VERSION],
        ["docker", "push", f"{IMAGE}:{VERSION}"],
        ["docker", "push", f"{IMAGE}:latest"],
        [
            "gh", "release", "create", VERSION,
            "--repo", REPO,
            "--title", f"{SETTINGS.repo_name} {VERSION}",
            "--notes-file", f"release-notes/{VERSION}.md",
            str(ARCHIVE),
        ],
    ]  # fmt: skip


def test_publish_logs_steps_and_streams(caplog: pytest.LogCaptureFixture) -> None:
    runner = FakeRunner()
    with caplog.at_level(logging.INFO, logger="sbmlsim.release"):
        _publish(runner)
    for line in (
        "Commit",
        "Tag",
        "GitHub Pages",
        "Push",
        "Docker image build",
        "Docker image push",
        "GitHub release",
    ):
        assert line in caplog.messages
    assert runner.streamed == [
        BUILD,
        ["git", "push", "origin", "main", VERSION],
        ["docker", "push", f"{IMAGE}:{VERSION}"],
        ["docker", "push", f"{IMAGE}:latest"],
    ]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="sbmlsim.release"):
        _publish(runner)
    assert "Commit: done already" in caplog.messages


def test_publish_resume_after_docker_failure() -> None:
    runner = FakeRunner()
    runner.fail_on = f"docker push {IMAGE}:{VERSION}"
    with pytest.raises(ReleaseError):
        _publish(runner)
    assert _publish(runner) == ["image", "docker", "github_release"]
    assert VERSION in runner.releases


def test_publish_nothing_before_the_image_builds() -> None:
    """A tag cannot be moved, an image which does not build stops the release first."""
    runner = FakeRunner()
    runner.fail_on = "docker build"
    with pytest.raises(ReleaseError, match="docker build"):
        _publish(runner)
    assert not runner.tags
    assert not runner.remote
    assert runner.dirty
    assert _publish(runner)[0] == "image"
    assert VERSION in runner.releases


def test_publish_idempotent() -> None:
    runner = FakeRunner()
    _publish(runner)
    assert _publish(runner) == []
    assert _publish(runner) == []


def test_tag_elsewhere_refused() -> None:
    runner = FakeRunner()
    runner.tags[VERSION] = "old"
    with pytest.raises(ReleaseError, match="not at HEAD"):
        _publish(runner)


def test_pages_other_failure_raised() -> None:
    runner = FakeRunner()
    runner.fail_on = "gh api"
    with pytest.raises(ReleaseError, match="gh api"):
        _publish(runner)


def _lock(release: Path, source: str) -> None:
    (release / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "toy-release"\nversion = "1.0.0"\n'
        'source = { editable = "." }\n\n'
        f'[[package]]\nname = "pkpdsim"\nversion = "0.7.1"\nsource = {source}\n',
        encoding="utf-8",
    )


def test_check_lock_and_tools_ok(tmp_path: Path) -> None:
    _lock(tmp_path, '{ registry = "https://pypi.org/simple" }')
    check_lock(tmp_path)
    check_tools(tmp_path, FakeRunner())


def test_check_lock_local_source(tmp_path: Path) -> None:
    _lock(tmp_path, '{ directory = "../pkpdsim" }')
    with pytest.raises(ReleaseError, match="pkpdsim"):
        check_lock(tmp_path)


#: an explicit index of a release project and a package pinned to it
INDEX_PROJECT = """\
[tool.uv.sources]
pkpdsim = { index = "dev" }

[[tool.uv.index]]
name = "dev"
url = "https://example.org/simple/"
explicit = true
"""


def test_check_lock_explicit_index(tmp_path: Path) -> None:
    """A package of an explicit index of the project is a package of a registry."""
    _lock(tmp_path, '{ registry = "https://example.org/simple" }')
    (tmp_path / "pyproject.toml").write_text(INDEX_PROJECT, encoding="utf-8")
    check_lock(tmp_path)


def test_check_lock_undeclared_index(tmp_path: Path) -> None:
    _lock(tmp_path, '{ registry = "https://example.org/simple" }')
    with pytest.raises(ReleaseError, match="pkpdsim"):
        check_lock(tmp_path)


def test_check_lock_source_of_an_undeclared_index(tmp_path: Path) -> None:
    _lock(tmp_path, '{ registry = "https://example.org/simple" }')
    (tmp_path / "pyproject.toml").write_text(
        INDEX_PROJECT.replace('index = "dev" }', 'index = "other" }'), encoding="utf-8"
    )
    with pytest.raises(ReleaseError, match=r"tool\.uv\.sources: pkpdsim"):
        check_lock(tmp_path)


def test_check_lock_uv_sources(tmp_path: Path) -> None:
    _lock(tmp_path, '{ registry = "https://pypi.org/simple" }')
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.sources]\npkpdsim = { path = "../pkpdsim" }\n', encoding="utf-8"
    )
    with pytest.raises(ReleaseError, match=r"tool\.uv\.sources"):
        check_lock(tmp_path)


def test_check_tools_lists_missing_tools(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.missing = {"docker info", "docker buildx version", "gh auth status"}
    with pytest.raises(ReleaseError) as err:
        check_tools(tmp_path, runner)
    assert "docker info" in str(err.value)
    assert "docker buildx version" in str(err.value)
    assert "gh auth status" in str(err.value)


LATEST = f"{IMAGE}:latest"
RETAG = ["docker", "buildx", "imagetools", "create", "-t", LATEST, f"{IMAGE}:{VERSION}"]


def _docker_commands(runner: FakeRunner) -> list[list[str]]:
    return [
        c
        for c in runner.commands
        if c[:2] in (["docker", "build"], ["docker", "push"]) or c == RETAG
    ]


def test_publish_resume_after_latest_push_failure() -> None:
    runner = FakeRunner()
    runner.fail_on = f"docker push {LATEST}"
    with pytest.raises(ReleaseError):
        _publish(runner)
    runner.commands.clear()
    assert _publish(runner) == ["docker", "github_release"]
    assert runner.images[LATEST] == runner.images[f"{IMAGE}:{VERSION}"]
    assert _docker_commands(runner) == [RETAG]


def test_docker_retags_a_differing_latest() -> None:
    runner = FakeRunner()
    _publish(runner)
    runner.images[LATEST] = "other"
    runner.commands.clear()
    assert _publish(runner) == ["docker"]
    assert runner.images[LATEST] == "digest-1"
    assert _docker_commands(runner) == [RETAG]


def test_docker_pushed_version_is_never_rebuilt() -> None:
    """The version on the registry is the release, a local image may differ."""
    runner = FakeRunner()
    runner.images[f"{IMAGE}:{VERSION}"] = "digest-0"
    _publish(runner)
    assert runner.images[LATEST] == "digest-0"
    assert _docker_commands(runner) == [RETAG]


def test_docker_builds_over_a_stale_local_image() -> None:
    """A local image of the version may be of an earlier build, it is rebuilt."""
    runner = FakeRunner()
    runner.local_images = {f"{IMAGE}:{VERSION}", LATEST}
    _publish(runner)
    assert _docker_commands(runner) == [
        BUILD,
        ["docker", "push", f"{IMAGE}:{VERSION}"],
        ["docker", "push", LATEST],
    ]


def test_pages_switched_to_workflow() -> None:
    runner = FakeRunner()
    runner.pages = "legacy"
    assert "pages" in _publish(runner)
    assert runner.pages == "workflow"
    assert [
        "gh", "api", "-X", "PUT", f"repos/{REPO}/pages", "-f", "build_type=workflow"
    ] in runner.commands  # fmt: skip
    assert "pages" not in _publish(runner)


def test_release_without_archive_is_completed() -> None:
    runner = FakeRunner()
    _publish(runner)
    runner.releases[VERSION] = []
    assert _publish(runner) == ["github_release"]
    assert runner.releases[VERSION] == [ARCHIVE.name]
    upload = [c for c in runner.commands if c[:3] == ["gh", "release", "upload"]]
    assert upload[-1][-1] == "--clobber"


def test_check_lock_missing_lock(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match=r"no uv\.lock"):
        check_lock(tmp_path)


def test_check_tools_without_lock(tmp_path: Path) -> None:
    check_tools(tmp_path, FakeRunner())
    runner = FakeRunner()
    runner.missing = {"docker info"}
    with pytest.raises(ReleaseError, match="docker") as err:
        check_tools(tmp_path, runner)
    assert "uv.lock" not in str(err.value)


def test_check_lock_without_tools(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.missing = {"docker info", "gh auth status"}
    _lock(tmp_path, '{ registry = "https://pypi.org/simple" }')
    check_lock(tmp_path)
    _lock(tmp_path, '{ directory = "../pkpdsim" }')
    with pytest.raises(ReleaseError, match="pkpdsim") as err:
        check_lock(tmp_path)
    assert "docker" not in str(err.value)
