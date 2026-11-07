"""The license texts of a model, e.g. for its COMBINE archive.

A model is published under two licenses: the models and the documentation
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), the
source code under [MIT](https://opensource.org/license/MIT). The texts are
resources of the package; the text of CC BY-SA 4.0 is the same for every
model, the text of MIT starts with the copyright of the model, e.g.
`Copyright (c) 2024-2025 Matthias König`, which is the only
part in which the MIT texts of two models differ. The texts are byte for byte
the license files of the models of pkdb_models, `src/sbmlsim/resources/licenses/`,
whose line endings git never converts (`.gitattributes`).

`write_license` writes a text as `cc-by-sa-4.0.txt` or `mit.txt` into a
directory, from where the factory of a model adds it to the COMBINE archive.
"""

from enum import StrEnum
from importlib.resources import files
from pathlib import Path

__all__ = [
    "License",
    "license_text",
    "write_license",
]


class License(StrEnum):
    """A license of a model, its value is the name of its file without `.txt`."""

    CC_BY_SA_4_0 = "cc-by-sa-4.0"
    """Creative Commons Attribution-ShareAlike 4.0 International, the models
    and the documentation."""

    MIT = "mit"
    """The MIT license, the source code; its text starts with the copyright."""

    @property
    def filename(self) -> str:
        """The name of the file of the license text, e.g. `mit.txt`."""
        return f"{self.value}.txt"


_MIT_COPYRIGHT_PLACEHOLDER = "Copyright (c) <year> <copyright holders>"
"""The first line of the MIT resource, which `license_text` replaces."""


def license_text(license: License | str, copyright: str | None = None) -> str:
    """Get the text of a license.

    Args:
        license: the license, or its value (`"mit"`).
        copyright: the year or years and the holders of the copyright, e.g.
            `2024-2025 Matthias König`, which make the first line of the MIT
            text `Copyright (c) 2024-2025 Matthias König`. Required for MIT,
            not allowed for CC BY-SA 4.0, whose text has no copyright line.

    Returns:
        The text of the license, every line ends with a line feed.

    Raises:
        ValueError: for a missing or a superfluous copyright, or a copyright
            which is empty, spans several lines or starts with `Copyright`
            itself.
    """
    license = License(license)
    text = (
        files("sbmlsim")
        .joinpath("resources", "licenses", license.filename)
        .read_text(encoding="utf-8")
    )
    if license is not License.MIT:
        if copyright is not None:
            raise ValueError(
                f"The text of {license.value} has no copyright line, "
                f"the copyright '{copyright}' cannot be set."
            )
        return text

    if copyright is None:
        raise ValueError(
            "The text of mit starts with the copyright line, a copyright such "
            "as '2024 Matthias König' is required."
        )
    if not copyright.strip() or "\n" in copyright or "\r" in copyright:
        raise ValueError(
            f"The copyright has to be a single line which is not empty: '{copyright}'."
        )
    if copyright.lower().startswith("copyright"):
        raise ValueError(
            f"The copyright is the year and the holders, the text adds "
            f"'Copyright (c)' itself: '{copyright}'."
        )
    return text.replace(_MIT_COPYRIGHT_PLACEHOLDER, f"Copyright (c) {copyright}", 1)


def write_license(
    directory: Path, license: License | str, copyright: str | None = None
) -> Path:
    """Write the text of a license into a directory.

    The file is named after the license (`cc-by-sa-4.0.txt`, `mit.txt`) and is
    written in UTF-8 with line feeds as line endings on every platform, so it
    is the same file byte for byte wherever a model is created; an existing
    file is replaced.

    Args:
        directory: the directory of the file, which has to exist, e.g. the
            folder whose files the factory of a model packs into its COMBINE
            archive.
        license: the license, or its value (`"mit"`).
        copyright: the year or years and the holders of the copyright, required
            for MIT, see `license_text`.

    Returns:
        The path of the written file.
    """
    license = License(license)
    path = directory / license.filename
    path.write_text(license_text(license, copyright), encoding="utf-8", newline="\n")
    return path
