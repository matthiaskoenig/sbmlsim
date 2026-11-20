"""The notices of the components of a web app as the plain text of a NOTICE file.

The notices of a component are markdown (`sbmlsim_core.THIRD_PARTY` is a document of a heading,
a table of the libraries which are linked and the license text of each of them), and the page of
an app shows them as they are, in a block of preformatted text: a renderer of markdown is no
dependency of a package which writes one HTML file and no part of a page which loads nothing from
the network, and the markup of such a document in preformatted text reads as the markup and not
as the document (`# Third-party notices`, a row `|---|---|` under the header of a table, the
fences of the license texts).

`plain_notices` is the one conversion the page needs: it keeps every line of a license text
verbatim, since that is what a notice is, and renders the markup around it as a NOTICE file has
it. It is not a renderer of markdown: it reads the constructs such a document is written with
(headings, one table, fenced blocks, inline code and links) and leaves every other line as it is.
"""

import re

__all__ = ["plain_notices"]


#: The fence of a block of preformatted text, whose lines are kept verbatim.
FENCE = "```"

#: The characters a heading of the first two levels is underlined with, by its level.
RULES = {1: "=", 2: "-"}

#: The indent of the fields of a row of a table.
FIELD_INDENT = "  "

#: A heading: its level and its text.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

#: The row which separates the header of a table from its body, e.g. `|---|---|`.
_SEPARATOR = re.compile(r"^\|[\s:|-]*\|$")

#: A span of inline code, whose backticks are dropped.
_CODE = re.compile(r"`([^`]*)`")

#: A link, which becomes its text and its target.
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")


def _inline(text: str) -> str:
    """Render the inline markup of a line.

    Args:
        text: the line, outside of a fenced block.

    Returns:
        the line with the backticks of its code dropped and every link as its text and its target.
    """
    return _CODE.sub(r"\1", _LINK.sub(r"\1 (\2)", text))


def _cells(row: str) -> list[str]:
    """Read the cells of a row of a table.

    Args:
        row: the row, which starts and ends with a pipe.

    Returns:
        the cells, their inline markup rendered and their padding removed.
    """
    return [_inline(cell.strip()) for cell in row.strip().strip("|").split("|")]


def _table(rows: list[list[str]]) -> list[str]:
    """Render a table as a block per row.

    Columns of a width which fits a terminal do not fit the panel of a page, which wraps them
    into each other; a block per row, its first cell its heading and every other cell a field of
    `<header>: <cell>`, reads at any width and loses nothing of the table.

    Args:
        rows: the rows, the header first, each a list of cells.

    Returns:
        the lines of the table, a blank line between its blocks.
    """
    if len(rows) < 2:
        return [FIELD_INDENT.join(row) for row in rows]
    header, *body = rows
    lines: list[str] = []
    for row in body:
        if lines:
            lines.append("")
        lines.append(row[0] if row else "")
        for index, cell in enumerate(row[1:], start=1):
            if cell:
                name = header[index] if index < len(header) else ""
                lines.append(
                    f"{FIELD_INDENT}{name}: {cell}" if name else f"{FIELD_INDENT}{cell}"
                )
    return lines


def plain_notices(text: str) -> str:
    """Render the notices of a component as plain text.

    A heading of the first two levels is underlined, a heading below them is its text alone, the
    rows of a table are aligned columns under an underlined header, the fences of a block of
    preformatted text are dropped and its lines are kept verbatim, and the backticks and the
    links of every other line are rendered. A line which is none of these is kept as it is, so a
    document of paragraphs comes back as its paragraphs.

    Args:
        text: the notices, markdown.

    Returns:
        the notices as the plain text of a NOTICE file, without a trailing newline.
    """
    lines: list[str] = []
    rows: list[list[str]] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(FENCE):
            fenced = not fenced
            continue
        if fenced:
            lines.append(line)
            continue
        if line.startswith("|"):
            if not _SEPARATOR.match(line):
                rows.append(_cells(line))
            continue
        if rows:
            lines.extend(_table(rows))
            rows = []
        heading = _HEADING.match(line)
        if heading is None:
            lines.append(_inline(line))
            continue
        level = len(heading.group(1))
        title = _inline(heading.group(2))
        lines.append(title)
        rule = RULES.get(level)
        if rule is not None and title:
            lines.append(rule * len(title))
    if rows:
        lines.extend(_table(rows))
    return "\n".join(lines).strip("\n")
