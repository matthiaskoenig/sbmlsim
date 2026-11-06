"""The release metadata of a model: what its package does not know.

A model package sets `sbmlsim.project.Project.release` to a `ReleaseSettings`.
The module uses the standard library only, so importing the project of a model
package stays light.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields

__all__ = ["Author", "ReleaseSettings"]

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Author:
    """An author of the citation of a model.

    Attributes:
        name: the name as `"Family, Given"`, as Zenodo wants it.
        orcid: the ORCID of the author.
        affiliation: the affiliation of the author.
    """

    name: str
    orcid: str | None = None
    affiliation: str | None = None

    @property
    def citation_name(self) -> str:
        """The name with the initial of the given name, `"Family, G."`."""
        family, _, given = self.name.partition(",")
        given = given.strip()
        if not given:
            return family.strip()
        return f"{family.strip()}, {given[0]}."

    @property
    def display_name(self) -> str:
        """The name as `"Given Family"`."""
        family, _, given = self.name.partition(",")
        return f"{given.strip()} {family.strip()}".strip()


@dataclass(frozen=True, kw_only=True)
class ReleaseSettings:
    """The metadata of the release of a model.

    All fields are keyword-only; the sequences are stored as tuples, so a list
    can be passed and the settings are hashable.

    Attributes:
        title: the title of the release: the Zenodo record, the citation, the
            README and the docs site.
        repository: the GitHub repository as `owner/name`.
        doi: the concept DOI of Zenodo.
        authors: the authors of the citation, in order.
        keywords: the keywords of the Zenodo record.
        docker_image: the Docker image of the model.
        funding: the funding text.
        description: one paragraph about the model.
        exclude: paths of the model folder which are not released.
        first_year: the year of the first release, the start of the copyright.
    """

    title: str
    repository: str
    doi: str
    authors: Sequence[Author]
    keywords: Sequence[str]
    docker_image: str
    funding: str
    description: str
    exclude: Sequence[str] = ("TODO.md", "literature", "thesis")
    first_year: int

    def __post_init__(self) -> None:
        """Normalize the sequences and check the form of the repository."""
        for name in ("authors", "keywords", "exclude"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not _REPOSITORY.match(self.repository):
            raise ValueError(
                f"The repository is 'owner/name', not '{self.repository}'."
            )

    @property
    def owner(self) -> str:
        """The owner of the repository."""
        return self.repository.split("/")[0]

    @property
    def repo_name(self) -> str:
        """The name of the repository."""
        return self.repository.split("/")[1]

    @property
    def site_url(self) -> str:
        """The URL of the docs site on GitHub pages."""
        return f"https://{self.owner}.github.io/{self.repo_name}/"

    @property
    def author_names(self) -> str:
        """The authors as `"Given Family"`, the last joined with `and`."""
        names = [author.display_name for author in self.authors]
        if len(names) <= 1:
            return "".join(names)
        return ", ".join(names[:-1]) + " and " + names[-1]

    @property
    def raw_url(self) -> str:
        """The base URL of the raw files of the main branch."""
        return f"https://raw.githubusercontent.com/{self.repository}/main/"

    def missing(self) -> list[str]:
        """Get the names of the fields a release lacks, sorted.

        A field is missing if it is empty; `exclude` may be empty (everything is
        released), `first_year` is missing unless it is a positive int.
        """
        names = [
            f.name
            for f in fields(self)
            if f.name not in ("exclude", "first_year") and not getattr(self, f.name)
        ]
        year = self.first_year
        if isinstance(year, bool) or not isinstance(year, int) or year < 1:
            names.append("first_year")
        return sorted(names)
