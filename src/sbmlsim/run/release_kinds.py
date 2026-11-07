"""The kind of the release of a project as a repository of its own.

`release` (`ReleaseSpec`) builds the release `version` of a project into
`out/repository` (`sbmlsim.release.workflow.REPOSITORY_DIR`), a clone of the
release repository (`ReleaseSettings.repository`, or the checkout
`repository`, which is only read): the export of the model package, the
project files, the models, simulations, fit reports and the assessment of
the credibility, each an analysis of the run layer in the release repository
with its spec in `specs/`, the release notes, the docs and the manifest, see
`sbmlsim.release.workflow.build_release`. `framework_path` builds with a local
checkout of the framework instead of the released one, a build which is not
published. The kind commits nothing and writes nothing outside of the out:
publishing the build (commit, tag, push, Pages, Docker image, GitHub release)
is the local step `python -m sbmlsim.release publish`, never a kind.

A check which fails before the build, e.g. a source repository with changes or
a version which is not greater than the last release, is a
`ConfigurationError`; a resume rebuilds the release in the repository of the
out, which only a build of the same version may have changed.

The release is imported when the kind plans or runs, not when this module is
imported.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import field_validator

from sbmlsim.run.kinds import (
    ConfigurationError,
    Kind,
    Spec,
    load_project_ref,
    register,
)

if TYPE_CHECKING:
    from sbmlsim.project import Project
    from sbmlsim.run.context import RunContext

__all__ = ["RELEASE", "ReleaseSpec"]

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class ReleaseSpec(Spec):
    """The spec of the build of a release of a project, see the module.

    Attributes:
        project: the project, `module:attribute`, with its release settings.
        version: the version of the release, `X.Y.Z`.
        repository: a checkout of the release repository which is cloned
            instead of the repository on GitHub; only read.
        framework_path: a local checkout of the framework of the release
            (`ReleaseSettings.framework`) instead of the released one; such a
            build is not published.
    """

    project: str
    version: str
    repository: Path | None = None
    framework_path: Path | None = None

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        """Check that the version is `X.Y.Z`.

        Raises:
            ValueError: for another version.
        """
        if not _VERSION.match(value):
            raise ValueError(f"'{value}' is not a version 'X.Y.Z'")
        return value


def _project(spec: ReleaseSpec) -> Project:
    """Load the project of a spec and check its release settings.

    Raises:
        ConfigurationError: for a project which is not loaded, has no package
            or incomplete release settings.
    """
    from sbmlsim.release.source import ReleaseError
    from sbmlsim.release.workflow import release_settings

    project = load_project_ref(spec.project)
    if not project.package:
        raise ConfigurationError(
            f"The project '{spec.project}' has no package, a release exports it."
        )
    try:
        release_settings(project)
    except ReleaseError as err:
        raise ConfigurationError(str(err)) from err
    return project


def _plan_release(spec: ReleaseSpec) -> ReleaseSpec:
    """Check the project and its release settings, the spec is planned as given."""
    _project(spec)
    return spec


def _run_release(spec: ReleaseSpec, ctx: RunContext) -> dict[str, Any]:
    """Build the release into the out.

    Raises:
        ConfigurationError: if a check before the build fails.
        ReleaseError: if a stage of the build fails.
    """
    from sbmlsim.release import source
    from sbmlsim.release.workflow import (
        build_release,
        build_summary,
        prepare_release,
    )

    project = _project(spec)
    # the runner of `sbmlsim.release.workflow.execute_release`, the real one
    # without
    runner = source.current_runner()
    try:
        prepared = prepare_release(
            project, spec.version, ctx.out, checkout=spec.repository, runner=runner
        )
    except source.ReleaseError as err:
        raise ConfigurationError(f"The release is refused: {err}") from err
    built = build_release(
        project,
        spec.project,
        spec.version,
        prepared,
        framework_path=spec.framework_path,
        workers=ctx.workers,
        runner=runner,
    )
    return build_summary(built, prepared.target.path)


#: the build of a release of a project into the out
RELEASE: Kind[ReleaseSpec] = Kind(
    name="release", spec=ReleaseSpec, run=_run_release, plan=_plan_release
)

register(RELEASE)
