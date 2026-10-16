"""Format-independent document model of a report.

A report is a `Document` of blocks (`Paragraph`, `Equation`, `Table`, `Figure`,
`Section`) whose text is a sequence of inline spans (`Text`, `Code`, `Math`,
`Cite`, `Ref`). The renderers of `sbmlsim.report.render` turn the same document
into HTML, Typst or LaTeX. Equations carry their LaTeX (MathML of the HTML
report, LaTeX report) and their Typst source (Typst report), tables and figures
are numbered in document order and referenced by their label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sbmlsim.report.references import Reference

__all__ = [
    "Block",
    "Cite",
    "Code",
    "Document",
    "Equation",
    "Figure",
    "Inline",
    "Math",
    "Paragraph",
    "Ref",
    "Section",
    "Table",
    "Text",
    "p",
]


@dataclass(frozen=True)
class Text:
    """Plain text, escaped by the renderers."""

    text: str


@dataclass(frozen=True)
class Code:
    """Literal text in a monospace font, e.g. an identifier of the model."""

    text: str


@dataclass(frozen=True)
class Math:
    """Inline formula given in LaTeX (HTML, LaTeX) and in Typst."""

    latex: str
    typst: str


@dataclass(frozen=True)
class Cite:
    """Citation of the references with the given keys."""

    keys: tuple[str, ...]


@dataclass(frozen=True)
class Ref:
    """Reference to the table, figure, equation or section with this label."""

    label: str


Inline = Text | Code | Math | Cite | Ref


@dataclass(frozen=True)
class Paragraph:
    """Paragraph of inline spans."""

    content: tuple[Inline, ...]
    key: str = ""


@dataclass(frozen=True)
class Equation:
    """Display equation given in LaTeX and in Typst."""

    latex: str
    typst: str
    label: str = ""
    key: str = ""


@dataclass(frozen=True)
class Table:
    """Table with a caption and a header.

    `align` has one character per column (`l`, `r` or `c`); empty means the
    first column left and all others right.
    """

    caption: tuple[Inline, ...]
    header: tuple[str, ...]
    rows: tuple[tuple[str | Code | Math, ...], ...]
    label: str
    align: str = ""
    key: str = ""

    def __post_init__(self) -> None:
        """Check that alignment and rows match the columns."""
        n = len(self.header)
        if self.align and (len(self.align) != n or set(self.align) - set("lrc")):
            raise ValueError(
                f"align '{self.align}' of table '{self.label}' must be one of "
                f"l, r, c per column ({n} columns)"
            )
        for row in self.rows:
            if len(row) != n:
                raise ValueError(
                    f"row of table '{self.label}' has {len(row)} cells, the header {n}"
                )

    @property
    def alignment(self) -> str:
        """Alignment characters, one per column."""
        if self.align:
            return self.align
        return "l" + "r" * (len(self.header) - 1) if self.header else ""


@dataclass(frozen=True)
class Figure:
    """Figure with a caption; `path` is relative to the output directory."""

    path: str
    caption: tuple[Inline, ...]
    label: str
    key: str = ""


@dataclass(frozen=True)
class Section:
    """Section with a title and blocks; `level` 1 is a top level section."""

    title: str
    blocks: tuple[Block, ...]
    level: int = 1
    label: str = ""
    key: str = ""


Block = Paragraph | Equation | Table | Figure | Section


@dataclass(frozen=True)
class Document:
    """Document with a title, blocks and the references they cite.

    `references` are the cited references in the order of their first citation
    (`sbmlsim.report.references.bibliography`); the renderers write the list of
    references only when there are any.
    """

    title: str
    blocks: tuple[Block, ...]
    references: tuple[Reference, ...] = ()


def p(*parts: str | Inline, key: str = "") -> Paragraph:
    """Create a paragraph, a string part is a `Text`.

    Args:
        *parts: strings and inline spans.
        key: key of the paragraph.

    Returns:
        The paragraph.
    """
    return Paragraph(
        tuple(Text(part) if isinstance(part, str) else part for part in parts),
        key=key,
    )
