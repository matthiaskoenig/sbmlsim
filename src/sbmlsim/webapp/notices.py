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

#: The columns of a table are separated by this in the plain text.
COLUMN_GAP = "  "

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
    """Render a table as aligned columns.

    The header is underlined over the width of the table, so the columns are read as columns in
    preformatted text; a row of more cells than the header has keeps them.

    Args:
        rows: the rows, the header first, each a list of cells.

    Returns:
        the lines of the table.
    """
    if not rows:
        return []
    widths: list[int] = []
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))
            else:
                widths.append(len(cell))
    written = [
        COLUMN_GAP.join(
            cell.ljust(widths[index]) if index + 1 < len(row) else cell
            for index, cell in enumerate(row)
        ).rstrip()
        for row in rows
    ]
    rule = "-" * max(len(line) for line in written)
    return [written[0], rule, *written[1:]]


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
