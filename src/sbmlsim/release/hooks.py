"""What a framework adds to the release of a model package.

The release of sbmlsim builds a release repository of every project: the
models, the simulations, the reports of the adopted fits and the assessment of
the credibility, the release notes and the pages of the site. A framework on
top of sbmlsim, e.g. one of pharmacokinetic models, adds the content of its
domain through a `ReleaseHook`, which `ReleaseSettings.hook` names
(`sbmlsim.project.Project.release_hook`):

- `templates`: directories of templates which are searched before the ones of
  sbmlsim (`sbmlsim/resources/release`): a template of the same path replaces
  the one of sbmlsim, another one adds to them, e.g. a page of the site or a
  part which a template of sbmlsim includes when it exists. The page of the
  credibility includes, in the section of every context of use (`context`
  the context), `docs/credibility-acceptance.md.j2` right after its
  acceptance table (e.g. the tables of the rules of an extension) and
  `docs/credibility-context.md.j2` at its end, and
  `docs/credibility-sections.md.j2` after its sections of the core, see
  `CREDIBILITY_PARTS`.
- `pages`: pages of the site of its own, `docs/<page>.md.j2` of its templates,
  with their titles in the navigation, before the release notes.
- `values`: further values of the pages of the site and of the README.
- `credibility`: the data of the page of the credibility, e.g. the evidence of
  the evidence extensions of the framework, see
  `sbmlsim.credibility.extensions`.
- `data_study`: the study a file of the data of the package belongs to, which
  the release notes list as added, removed or changed.
- `notes_sections`: sections of markdown at the end of the release notes.

Every method has a default, so a hook overrides only what it adds. The module
imports nothing of the release at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from sbmlsim.credibility.assessment import CredibilityAssessment
    from sbmlsim.release.project import ProjectContext
    from sbmlsim.release.source import ReleaseRepo, Runner

__all__ = ["CREDIBILITY_PARTS", "ReleaseHook"]

#: the parts of the page of the credibility which a hook may add, by where
#: they are included: in the section of every context of use after its
#: acceptance table and at its end, and after the sections of the core
CREDIBILITY_PARTS: Final = {
    "acceptance": "docs/credibility-acceptance.md.j2",
    "context": "docs/credibility-context.md.j2",
    "sections": "docs/credibility-sections.md.j2",
}


class ReleaseHook:
    """What a framework adds to the release of a model package, see the module.

    The base class adds nothing; a framework derives from it and overrides
    what it adds.
    """

    def templates(self) -> Sequence[Path | Traversable]:
        """Get the directories of the templates of the hook.

        Returns:
            The directories, searched in their order before the templates of
            sbmlsim; e.g. `importlib.resources.files("framework") /
            "resources/release"`.
        """
        return ()

    def pages(self, context: ProjectContext) -> Sequence[tuple[str, str]]:
        """Get the pages the hook adds to the site.

        Args:
            context: the project context of the release.

        Returns:
            The pages as `(page, title)`, the template of a page
            `docs/<page>.md.j2` of the templates of the hook; they come before
            the release notes in the navigation.
        """
        return ()

    def values(self, context: ProjectContext, release_path: Path) -> Mapping[str, Any]:
        """Get further values of the templates of the site and of the README.

        Args:
            context: the project context of the release.
            release_path: the built release repository.

        Returns:
            The values by their name; a value of the release of the same name
            is replaced.
        """
        return {}

    def credibility(
        self, assessment: CredibilityAssessment, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get the data of the page of the credibility.

        Args:
            assessment: the assessment of the contexts of use of the release.
            data: the data of the page as sbmlsim shows it, see
                `sbmlsim.release.docs`; the context of use `i` is
                `data["contexts"][i]` and `assessment.contexts[i]`.

        Returns:
            The data of the page, e.g. `data` with the evidence of the
            extensions of the framework.
        """
        return data

    def data_study(self, data_dir: PurePosixPath, path: PurePosixPath) -> str | None:
        """Get the study which a file of the data of the package belongs to.

        The release notes key the files by their path in the release
        repository, so files of two data folders belong to one study only if
        this method gives them one name.

        Args:
            data_dir: the data folder of the file relative to the root of the
                package, one of the data paths of the project
                (`sbmlsim.project.Project.data_path`), e.g. `data`.
            path: the path of the file relative to `data_dir`.

        Returns:
            The study, by default the folder of the file relative to the root
            of the package, e.g. `data/drug/Study2000`; `None` for a file which
            belongs to no study, by default a file at the top of the data
            folder.
        """
        if len(path.parts) < 2:
            return None
        return (data_dir / path.parent).as_posix()

    def notes_sections(
        self, context: ProjectContext, release: ReleaseRepo, runner: Runner
    ) -> Sequence[str]:
        """Get the sections of the release notes the hook adds.

        Args:
            context: the project context of the release.
            release: the release repository with the new build in its working
                tree and the previous release as `last_tag`.
            runner: runs git.

        Returns:
            The sections as markdown, each starting with its heading of the
            second level, at the end of the release notes.
        """
        return ()
