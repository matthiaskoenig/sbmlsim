"""The release of a model repository: the build into an output and its publishing.

A release has two steps:

1. The build (`prepare_release` and `build_release`, the kind `release` of the
   run layer, `sbmlsim.run.release_kinds`) clones the release repository into
   `<out>/repository` (`REPOSITORY_DIR`) and builds the release there: the
   export of the model package, the project files, the models, simulations,
   fit reports and the assessment of the credibility, the release notes, the
   docs and the manifest. It writes nothing outside of the output, the
   checkout of the release repository it may clone from is only read, and it
   commits nothing, so the build can be reviewed.
2. The publishing (`publish_release`, `python -m sbmlsim.release publish`), a
   local step: the build of the output, which must still be the build of the
   source (`matches`), is committed, tagged and pushed, the site is deployed
   with GitHub Pages, the Docker image is built and pushed and the GitHub
   release is created. A publishing which failed half way is finished by
   running it again.

Git, uv, the analyses of the model, docker and gh run through a `Runner`, so
the tests replace them.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from sbmlsim.release.build import BuildResult, build
from sbmlsim.release.docs import write_docs
from sbmlsim.release.manifest import (
    Manifest,
    changed_by_build,
    check_file_sizes,
    check_not_ignored,
    finish_build,
    matches,
    prune,
    start_build,
    stop_build,
    write_manifest,
)
from sbmlsim.release.notes import fits_prefix, write_release_notes
from sbmlsim.release.project import project_context, write_project_files
from sbmlsim.release.publish import check_lock, check_tools, head_is_release, tag_commit
from sbmlsim.release.publish import publish as publish_steps
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import (
    ReleaseError,
    ReleaseRepo,
    Runner,
    SourceRepo,
    check_version,
    export_package,
    parse_version,
    release_repo,
    run,
    source_framework,
    source_repo,
    stage,
    using_runner,
)

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = [
    "REPOSITORY_DIR",
    "PreparedRelease",
    "PublishableBuild",
    "Publisher",
    "ReleaseBuild",
    "build_release",
    "build_summary",
    "check_build",
    "execute_release",
    "prepare_release",
    "publish_release",
    "publishable",
    "release_settings",
]

#: the folder of the release repository in the output of a build
REPOSITORY_DIR: Final = "repository"

Publisher = Callable[[ReleaseSettings, Path, str, Path, Runner], list[str]]
"""Publishes a built release, see `sbmlsim.release.publish.publish`."""


@dataclass(frozen=True)
class PreparedRelease:
    """A release which passed its checks and is ready to be built.

    Attributes:
        settings: the complete release settings of the project.
        source: the clean source repository.
        target: the checkout of the release repository in the output.
    """

    settings: ReleaseSettings
    source: SourceRepo
    target: ReleaseRepo


@dataclass(frozen=True)
class ReleaseBuild:
    """The build of a release.

    Attributes:
        manifest: the manifest of the build.
        build: what the analyses of the build wrote and ran.
    """

    manifest: Manifest
    build: BuildResult


def release_settings(project: Project) -> ReleaseSettings:
    """Get the complete release settings of a project.

    Raises:
        ReleaseError: if the project has no release settings or they lack a
            field.
    """
    settings = project.release
    if settings is None:
        raise ReleaseError(f"The project '{project.name}' has no release settings.")
    missing = settings.missing()
    if missing:
        raise ReleaseError(
            f"The release settings of the project '{project.name}' lack "
            f"{', '.join(missing)}."
        )
    return settings


def _archive_name(settings: ReleaseSettings, project: Project) -> str:
    """Get the name of the COMBINE archive which the GitHub release attaches."""
    if settings.archive is None:
        raise ReleaseError(
            f"The release settings of '{project.name}' name no archive, the GitHub "
            "release attaches its COMBINE archive."
        )
    return PurePosixPath(settings.archive).name


def _check_fits(project: Project) -> None:
    """Check that every adopted fit has its record in the model package."""
    try:
        fit_ids = list(project.adopted_fits().values())
    except ValueError as err:
        raise ReleaseError(str(err)) from err
    if fit_ids:
        fits_prefix(project)
    for fit_id in fit_ids:
        try:
            project.adopted_record(fit_id)
        except (ValueError, FileNotFoundError) as err:
            raise ReleaseError(str(err)) from err


def _resumable(path: Path, version: str, runner: Runner) -> bool:
    """Tell whether the release of the version is committed, only publishing remains.

    `HEAD` is the release commit of the version (see `head_is_release`) and the
    tag of the version is missing or at `HEAD`.
    """
    if not head_is_release(path, version, runner):
        return False
    tagged = tag_commit(path, version, runner)
    return (
        tagged is None or tagged == runner(["git", "rev-parse", "HEAD"], path).strip()
    )


def _check_ahead(target: ReleaseRepo, resume: bool) -> None:
    """Refuse local commits which `origin/main` lacks, but a resumed release commit."""
    if target.ahead > (1 if resume else 0):
        commits = "commit" if target.ahead == 1 else "commits"
        raise ReleaseError(
            f"{target.path} is ahead of origin/main by {target.ahead} {commits}; "
            "only the release commit of this version may wait for the push, push "
            "or reset the others first"
        )


def prepare_release(
    project: Project,
    version: str,
    out: Path,
    *,
    checkout: Path | None = None,
    runner: Runner = run,
) -> PreparedRelease:
    """Check a release and clone its repository into the output.

    The checks, before anything is built: the version is `X.Y.Z`, the release
    settings of the project are complete, the source repository is clean
    (`source_repo`), the release repository in `out/repository` (cloned when
    missing, see `release_repo`) is on `main`, up to date and without local
    commits, the version is greater than the last release (`check_version`),
    every adopted fit has its record, and the release repository is clean or
    changed only by an earlier build of this version or by an unfinished
    build (`changed_by_build`), e.g. of a resumed run.

    Args:
        project: the project which is released, with its release settings.
        version: the version of the release, `X.Y.Z`.
        out: the output of the build.
        checkout: a checkout of the release repository which is cloned instead
            of the repository on GitHub; only read.
        runner: runs git.

    Returns:
        The prepared release.

    Raises:
        ReleaseError: if a check fails; the message is for the user.
    """
    parse_version(version)
    settings = release_settings(project)
    source = source_repo(project, runner)
    target = release_repo(settings, out / REPOSITORY_DIR, runner, checkout=checkout)
    path = target.path
    if _resumable(path, version, runner) and target.ahead:
        raise ReleaseError(
            f"The release {version} is committed in {path} already, publish it "
            "(python -m sbmlsim.release publish)."
        )
    _check_ahead(target, resume=False)
    check_version(version, target.last_tag)
    _check_fits(project)
    if not changed_by_build(path, version, runner):
        raise ReleaseError(
            "the release repository has changes which no release made, "
            f"discard them first (see git -C {path} status)"
        )
    return PreparedRelease(settings=settings, source=source, target=target)


def build_release(
    project: Project,
    reference: str,
    version: str,
    prepared: PreparedRelease,
    *,
    framework_path: Path | None = None,
    workers: int | None = None,
    runner: Runner = run,
) -> ReleaseBuild:
    """Build a prepared release in its repository and write its manifest.

    The stages write, in this order: the export of the package, the project
    files, the build (`uv.lock`, the specs of its steps, models, simulations,
    fit reports, credibility), the release notes and the docs. A written file
    which git ignores is refused (`check_not_ignored`), the commit would lack
    it, and so is a file larger than GitHub accepts (`check_file_sizes`).
    `prune` then removes every file which no stage wrote, and the manifest
    `release.json` is written last, so it hashes the final tree which
    `matches` compares.

    A build which failed or was interrupted is rebuilt by the next run: the
    build marks its start in the git folder and removes the mark after the
    manifest (`start_build`, `finish_build`); a build which fails or is
    interrupted records the paths it changed in the mark (`stop_build`), so a
    file added by hand afterwards is still refused.

    Args:
        project: the project which is released.
        reference: the project as `module:attribute`, which the specs of the
            steps name.
        version: the version of the release.
        prepared: the checked release, see `prepare_release`.
        framework_path: a local checkout of the framework instead of the
            released one; such a build is not published.
        workers: the number of workers of the simulations.
        runner: runs git, uv and the analyses of the model.

    Returns:
        The manifest and what the analyses wrote and ran.

    Raises:
        ReleaseError: if a stage fails; the message is for the user.
    """
    settings, source, target = prepared.settings, prepared.source, prepared.target
    path = target.path
    context = project_context(
        project, reference, settings, source, version, framework_path
    )
    start_build(path, version, runner)
    try:
        stage(f"Export of {context.package} into {path}")
        written: list[Path] = export_package(project, settings, source, path, runner)
        stage("Project files")
        written += write_project_files(context, path)
        result = build(
            project, reference, path, runner, workers, source_commit=source.commit
        )
        written += result.written
        context = dataclasses.replace(context, built_steps=tuple(result.steps))
        stage("Release notes")
        written.append(write_release_notes(context, target, runner))
        stage("Docs and README")
        written += write_docs(context, path)
        stage("Prune and manifest")
        check_not_ignored(path, written, runner)
        check_file_sizes(path, written)
        prune(path, written, runner)
        manifest = write_manifest(
            path,
            version,
            source,
            written,
            framework=settings.framework,
            core_libraries=settings.core_libraries,
        )
    except BaseException:
        # a failed or interrupted build records what it changed, the next run
        # rebuilds it; recording must not hide the failure
        with contextlib.suppress(Exception):
            stop_build(path, runner)
        raise
    finish_build(path, runner)
    return ReleaseBuild(manifest=manifest, build=result)


def build_summary(release: ReleaseBuild, path: Path) -> dict[str, Any]:
    """Get the summary of a build: what it built and whether it can be published.

    Args:
        release: the build.
        path: the release repository.

    Returns:
        The repository relative to the output, the version, the source commit,
        the framework and its version, the number of files, the steps which
        ran, the outputs of the factory which differ from the source package
        and are part of the release (`changed`), the skipped steps with their
        reason (`skipped`) and whether the lock of the build can be published
        (`publishable`).
    """
    manifest = release.manifest
    try:
        check_lock(path)
        publishable = True
    except ReleaseError:
        publishable = False
    return {
        "repository": REPOSITORY_DIR,
        "version": manifest.version,
        "source_commit": manifest.source_commit,
        "framework": manifest.framework,
        "framework_version": manifest.framework_version,
        "files": len(manifest.files),
        "steps": [step.name for step in release.build.steps],
        "changed": list(release.build.changed),
        "skipped": [dict(entry) for entry in release.build.skipped],
        "publishable": publishable,
    }


@dataclass(frozen=True)
class PublishableBuild:
    """The build of a release in an output which can be published.

    Attributes:
        settings: the complete release settings of the project.
        path: the release repository of the output, `out/repository`.
        archive: the COMBINE archive the GitHub release attaches.
        resumed: whether `HEAD` is the release commit of the version, i.e. an
            earlier publishing failed half way and resumes.
    """

    settings: ReleaseSettings
    path: Path
    archive: Path
    resumed: bool


def check_build(
    project: Project, version: str, out: Path, *, runner: Runner = run
) -> PublishableBuild:
    """Check that an output holds the build of a release which can be published.

    The checks: the version is `X.Y.Z`, the release settings are complete and
    name the archive, which the GitHub release attaches, the output has a
    release repository, the source repository is clean and the release
    repository of the output is on `main` and up to date without local
    commits. If `HEAD` is the release commit of the version (clean, with its
    manifest) and its tag is missing or at `HEAD`, an earlier publishing
    failed half way and resumes, whatever the source commit is now; otherwise
    the version must be greater than the last release and the release
    repository must hold the build of this version from the current source
    commit and framework, unchanged since (`matches`). The lock must have
    released packages only (`check_lock`). The tools of the publishing are
    not checked, see `publish_release`, and nothing is changed.

    Args:
        project: the project which is released.
        version: the version of the release.
        out: the output of the build, see `build_release`.
        runner: runs git.

    Returns:
        The build.

    Raises:
        ReleaseError: naming the first check which fails.
    """
    parse_version(version)
    settings = release_settings(project)
    archive = _archive_name(settings, project)
    path = out / REPOSITORY_DIR
    if not (path / ".git").exists():
        raise ReleaseError(
            f"{out} holds no build of a release, build it first with the kind "
            "release (python -m sbmlsim.release build)."
        )
    source = source_repo(project, runner)
    target = release_repo(settings, path, runner)
    resume = _resumable(path, version, runner)
    _check_ahead(target, resume)
    if not resume:
        check_version(version, target.last_tag)
        framework_version = source_framework(source, settings.framework)
        if not matches(
            path, version, source.commit, settings.framework, framework_version, runner
        ):
            raise ReleaseError(
                f"{path} holds no unchanged build of the release {version} of the "
                f"source commit {source.commit[:10]} with {settings.framework} "
                f"{framework_version}; build it again"
            )
    check_lock(path)
    return PublishableBuild(
        settings=settings, path=path, archive=path / archive, resumed=resume
    )


def publishable(
    project: Project, version: str, out: Path, *, runner: Runner = run
) -> str | None:
    """Tell whether an output holds the build of a release which can be published.

    Args:
        project: the project which is released.
        version: the version of the release.
        out: the output of the build.
        runner: runs git.

    Returns:
        `None` for a build which can be published, else why not, the message
        of the check which fails, see `check_build`.
    """
    try:
        check_build(project, version, out, runner=runner)
    except ReleaseError as err:
        return str(err)
    return None


def publish_release(
    project: Project,
    version: str,
    out: Path,
    *,
    runner: Runner = run,
    publisher: Publisher = publish_steps,
) -> list[str]:
    """Publish the build of a release in an output.

    The build is checked (`check_build`), then the tools of the publishing
    (`check_tools`), a stale marker of a build is removed and the release is
    published (`publisher`).

    Args:
        project: the project which is released.
        version: the version of the release.
        out: the output of the build, see `build_release`.
        runner: runs git, docker and gh.
        publisher: publishes the built release.

    Returns:
        The steps of the publishing which ran.

    Raises:
        ReleaseError: if a check or a step fails; the message is for the user.
    """
    build = check_build(project, version, out, runner=runner)
    check_tools(build.path, runner)
    finish_build(build.path, runner)  # the tree is a finished build, a marker is stale
    stage(f"Publishing {build.settings.repository} {version}")
    return publisher(build.settings, build.path, version, build.archive, runner)


def execute_release(
    project: str,
    version: str,
    out: Path,
    *,
    repository: Path | None = None,
    framework_path: Path | None = None,
    workers: int | None = None,
    resume: bool = False,
    runner: Runner = run,
) -> dict[str, Any]:
    """Build a release into an output with the kind `release` of the run layer.

    `sbmlsim.run.execute` of the kind with the commands run by `runner`
    (`sbmlsim.release.source.using_runner`), so the output has the records of
    a run; `python -m sbmlsim.release build` calls it.

    Args:
        project: the project, `module:attribute`.
        version: the version of the release.
        out: the output of the build.
        repository: a checkout of the release repository which is cloned
            instead of the repository on GitHub; only read.
        framework_path: a local checkout of the framework instead of the
            released one.
        workers: the number of workers of the simulations.
        resume: build again into the output of an earlier build of the spec.
        runner: runs git, uv and the analyses of the build.

    Returns:
        The summary of the run, see `build_summary`.

    Raises:
        ConfigurationError: for an invalid spec, an output in use or a
            refused release.
        ReleaseError: if a stage of the build fails.
    """
    from sbmlsim.run import execute

    spec: dict[str, Any] = {"project": project, "version": version}
    if repository is not None:
        spec["repository"] = str(Path(repository).resolve())
    if framework_path is not None:
        spec["framework_path"] = str(Path(framework_path).resolve())
    with using_runner(runner):
        return execute("release", spec, out, workers=workers, resume=resume)
