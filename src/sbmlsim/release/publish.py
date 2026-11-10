"""Publish a built release: repository, tag, Pages, Docker image, GitHub release.

Every step checks whether it is already done and is skipped then, so a publish
which failed half way is finished by running the command again. The tools are
the real `git`, `gh` and `docker` command line programs, called through a
`Runner`.
"""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path

from sbmlsim.release.manifest import MANIFEST_FILE
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError, Runner, run, stage

__all__ = [
    "DOCKER_PLATFORM",
    "check_lock",
    "check_tools",
    "head_is_release",
    "publish",
    "tag_commit",
]

logger = logging.getLogger(__name__)

_PYPI = "https://pypi.org/simple"
DOCKER_PLATFORM = "linux/amd64"
"""The platform of the Docker image of a release."""


def _declared_indexes(release_path: Path) -> dict[str, str]:
    """Get the explicit package indexes of the release project, url by name."""
    pyproject = release_path / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return {
        index["name"]: index["url"].rstrip("/")
        for index in data.get("tool", {}).get("uv", {}).get("index", [])
        if index.get("explicit") and "name" in index and "url" in index
    }


def _local_packages(release_path: Path) -> list[str]:
    lock = release_path / "uv.lock"
    if not lock.is_file():
        return []
    registries = {_PYPI, *_declared_indexes(release_path).values()}
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    names = []
    for package in data.get("package", []):
        source = package.get("source", {})
        if source.get("registry", "").rstrip("/") in registries or source in (
            {"editable": "."},
            {"virtual": "."},
        ):
            continue
        names.append(package.get("name", "?"))
    return names


def _ok(runner: Runner, args: list[str], cwd: Path) -> bool:
    try:
        runner(args, cwd)
    except ReleaseError:
        return False
    return True


def _lock_problems(release_path: Path) -> list[str]:
    problems = []
    if not (release_path / "uv.lock").is_file():
        problems.append("no uv.lock")
    local = _local_packages(release_path)
    if local:
        problems.append(
            f"uv.lock has packages which are not from PyPI: {', '.join(local)}"
        )
    pyproject = release_path / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        indexes = _declared_indexes(release_path)
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        # a package pinned to an explicit index of the project, e.g. the builds
        # of libroadrunner, is a package of a registry; anything else is local
        local = [
            name
            for name, source in sources.items()
            if not (isinstance(source, dict) and source.keys() == {"index"})
            or source["index"] not in indexes
        ]
        if local:
            problems.append(f"pyproject.toml has tool.uv.sources: {', '.join(local)}")
    return problems


def _tool_problems(release_path: Path, runner: Runner) -> list[str]:
    problems = []
    if not _ok(runner, ["docker", "info"], release_path):
        problems.append("docker is not available (docker info failed)")
    if not _ok(runner, ["docker", "buildx", "version"], release_path):
        problems.append("docker buildx is not available (docker buildx version failed)")
    if not _ok(runner, ["gh", "auth", "status"], release_path):
        problems.append("gh is not logged in (gh auth status failed)")
    return problems


def _refuse(problems: list[str]) -> None:
    if problems:
        raise ReleaseError("cannot publish:\n- " + "\n- ".join(problems))


def check_tools(release_path: Path, runner: Runner = run) -> None:
    """Check that the tools of the publishing steps are ready.

    The check needs no build, so a release runs it before the build.

    Args:
        release_path: the checkout of the release repository.
        runner: runs the tool checks.

    Raises:
        ReleaseError: if Docker or its buildx plugin is not available or `gh`
            is not logged in; one message lists every problem.
    """
    _refuse(_tool_problems(release_path, runner))


def check_lock(release_path: Path) -> None:
    """Check that the built project depends on released packages only.

    Args:
        release_path: the checkout of the release repository, after the build.

    Raises:
        ReleaseError: if there is no `uv.lock`, the lock has a package which is
            neither from PyPI nor from an explicit index of the project, or the
            project has a source other than such an index (`tool.uv.sources`);
            one message lists everything.
    """
    _refuse(_lock_problems(release_path))


def head_is_release(path: Path, version: str, runner: Runner = run) -> bool:
    """Tell whether the release of a version is committed at `HEAD`.

    Args:
        path: the checkout of the release repository.
        version: the released version.
        runner: runs git.

    Returns:
        `True` if the working tree is clean and the `release.json` of `HEAD` is
        the manifest of `version`.
    """
    if runner(["git", "status", "--porcelain"], path).strip():
        return False
    try:
        text = runner(["git", "show", f"HEAD:{MANIFEST_FILE}"], path)
        return json.loads(text).get("version") == version
    except ReleaseError, ValueError, AttributeError:
        return False


def tag_commit(path: Path, version: str, runner: Runner = run) -> str | None:
    """Get the commit of the tag of a version.

    Args:
        path: the checkout of the release repository.
        version: the version, the name of the tag.
        runner: runs git.

    Returns:
        The commit of `refs/tags/<version>`, `None` without the tag.
    """
    if not runner(["git", "tag", "--list", version], path).strip():
        return None
    return runner(["git", "rev-list", "-n", "1", f"refs/tags/{version}"], path).strip()


def _commit(path: Path, version: str, runner: Runner) -> bool:
    if head_is_release(path, version, runner):
        return False
    runner(["git", "add", "-A"], path)
    runner(["git", "commit", "-m", f"Release {version}"], path)
    return True


def _tag(path: Path, version: str, runner: Runner) -> bool:
    tagged = tag_commit(path, version, runner)
    if tagged is not None:
        if tagged != runner(["git", "rev-parse", "HEAD"], path).strip():
            raise ReleaseError(f"tag {version} exists and is not at HEAD")
        return False
    runner(["git", "tag", version], path)
    return True


def _push(path: Path, version: str, runner: Runner) -> bool:
    head = runner(["git", "rev-parse", "HEAD"], path).strip()
    listing = runner(
        ["git", "ls-remote", "origin", "refs/heads/main", f"refs/tags/{version}"],
        path,
    )
    remote = {
        ref: sha
        for sha, ref in (line.split() for line in listing.splitlines() if line.strip())
    }
    if (
        remote.get("refs/heads/main") == head
        and remote.get(f"refs/tags/{version}") == head
    ):
        return False
    runner(["git", "push", "origin", "main", version], path, stream=True)
    return True


def _pages(path: Path, settings: ReleaseSettings, runner: Runner) -> bool:
    """Make GitHub Pages deploy the site from the docs workflow.

    A new site is created with the build type `workflow`, an existing site which
    is built from a branch is switched to it.
    """
    endpoint = f"repos/{settings.repository}/pages"
    try:
        answer = runner(["gh", "api", endpoint], path)
    except ReleaseError as err:
        if "HTTP 404" not in str(err) and "Not Found" not in str(err):
            raise
        runner(["gh", "api", "-X", "POST", endpoint, "-f", "build_type=workflow"], path)
        return True
    try:
        build_type = json.loads(answer).get("build_type")
    except (ValueError, AttributeError) as err:
        raise ReleaseError(f"'gh api {endpoint}' gave no Pages site: {err}") from err
    if build_type == "workflow":
        return False
    runner(["gh", "api", "-X", "PUT", endpoint, "-f", "build_type=workflow"], path)
    return True


def _manifest(runner: Runner, reference: str, path: Path) -> str | None:
    try:
        return runner(["docker", "manifest", "inspect", reference], path)
    except ReleaseError:
        return None


def _image(path: Path, settings: ReleaseSettings, version: str, runner: Runner) -> bool:
    """Build the image of the version, before anything is published.

    A Dockerfile or a lock which does not build fails the release before the
    commit, the tag and the push; a pushed tag cannot be moved. The image is
    built only while the registry lacks the version (a local image of the
    version may be of an earlier build, the layer cache keeps the build cheap),
    for the platform of the hosted runners and of most servers, whatever the
    machine of the release is.
    """
    image = settings.docker_image
    versioned, latest = f"{image}:{version}", f"{image}:latest"
    if _manifest(runner, versioned, path) is not None:
        return False
    runner(
        ["docker", "build", "--platform", DOCKER_PLATFORM,
         "-t", versioned, "-t", latest, "."],
        path,
        stream=True,
    )  # fmt: skip
    return True


def _docker(
    path: Path, settings: ReleaseSettings, version: str, runner: Runner
) -> bool:
    """Push the image of the version and point `latest` at it.

    The image of the version on the registry is the release: the local image
    (`_image`) is pushed only while the registry lacks it, and once it is there
    `latest` is tagged on the registry from it.
    """
    image = settings.docker_image
    versioned, latest = f"{image}:{version}", f"{image}:latest"
    remote_version = _manifest(runner, versioned, path)
    if remote_version is None:
        runner(["docker", "push", versioned], path, stream=True)
        runner(["docker", "push", latest], path, stream=True)
        return True
    if remote_version == _manifest(runner, latest, path):
        return False
    runner(
        ["docker", "buildx", "imagetools", "create", "-t", latest, versioned],
        path,
        stream=True,
    )
    return True


def _github_release(
    path: Path, settings: ReleaseSettings, version: str, archive: Path, runner: Runner
) -> bool:
    repository = settings.repository
    try:
        view = runner(
            [
                "gh",
                "release",
                "view",
                version,
                "--repo",
                repository,
                "--json",
                "assets",
            ],
            path,
        )
    except ReleaseError:
        runner(
            [
                "gh", "release", "create", version,
                "--repo", repository,
                "--title", f"{settings.repo_name} {version}",
                "--notes-file", f"release-notes/{version}.md",
                str(archive),
            ],
            path,
        )  # fmt: skip
        return True
    assets = {a.get("name") for a in json.loads(view or "{}").get("assets", [])}
    if archive.name in assets:
        return False
    runner(
        ["gh", "release", "upload", version, str(archive), "--repo", repository,
         "--clobber"],
        path,
    )  # fmt: skip
    return True


def publish(
    settings: ReleaseSettings,
    release_path: Path,
    version: str,
    archive: Path,
    runner: Runner = run,
) -> list[str]:
    """Publish a built release; every step already done is skipped.

    The Docker image is built first, so an image which does not build stops
    the release before anything is published. GitHub Pages is set up before the
    push, so the docs workflow which the push starts deploys the site.

    Args:
        settings: the release settings of the model.
        release_path: the checkout of the release repository.
        version: the released version.
        archive: the source archive attached to the GitHub release.
        runner: runs git, gh and docker.

    Returns:
        The names of the steps which ran: `image`, `commit`, `tag`, `pages`,
        `push`, `docker`, `github_release`.

    Raises:
        ReleaseError: if a command fails or the tag exists at another commit.
    """
    steps = {
        "image": (
            "Docker image build",
            lambda: _image(release_path, settings, version, runner),
        ),
        "commit": ("Commit", lambda: _commit(release_path, version, runner)),
        "tag": ("Tag", lambda: _tag(release_path, version, runner)),
        "pages": ("GitHub Pages", lambda: _pages(release_path, settings, runner)),
        "push": ("Push", lambda: _push(release_path, version, runner)),
        "docker": (
            "Docker image push",
            lambda: _docker(release_path, settings, version, runner),
        ),
        "github_release": (
            "GitHub release",
            lambda: _github_release(release_path, settings, version, archive, runner),
        ),
    }
    done = []
    for name, (label, step) in steps.items():
        stage(label)
        if step():
            done.append(name)
        else:
            logger.info("%s: done already", label)
    return done
