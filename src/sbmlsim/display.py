"""Console output of scripts: sections, key/value blocks and links to results.

The output of a run, e.g. of simulation experiments or of a fit, is a sequence
of sections with aligned key/value lines. A result which is a file, e.g. a
report, is a link: the terminal opens it with one click.

    from sbmlsim import display

    display.section("Report", icon=display.ICON_REPORT)
    display.key_values({"experiments": 3})
    display.link("report", "results/index.html")

The output goes to the console of the package, see `sbmlsim.console`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.style import Style
from rich.table import Table
from rich.text import Text

from sbmlsim.console import console

#: width of the keys of a key/value block
KEY_WIDTH = 18

#: icon of the section of a report
ICON_REPORT = ":clipboard:"


def section(title: str, icon: str | None = None) -> None:
    """Start a section of the output.

    A blank line separates the section from whatever came before it, so the
    sections are told apart whether the previous one ended in a table or in a
    key/value block.

    Args:
        title: title of the section.
        icon: emoji in front of the title, e.g. `ICON_REPORT`.
    """
    prefix = f"{icon} " if icon else ""
    console.line()
    console.rule(f"{prefix}[bold]{title}", align="left", style="white")


def key_values(items: Mapping[str, Any]) -> None:
    """Print aligned key/value lines, the smallest section of the output."""
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    table.add_column("key", style="bold", width=KEY_WIDTH)
    table.add_column("value", overflow="fold")
    for key, value in items.items():
        table.add_row(key, str(value))
    console.print(table)


def link(key: str, path: Path | str) -> None:
    """Print a link to a file or a directory, which the terminal opens with a click.

    The link is a hyperlink of the terminal (OSC 8), which VS Code, iTerm2,
    Windows Terminal, GNOME Terminal and kitty open with a click, whatever the
    width of the terminal. Its text is the `file://` URI of the path on a
    single line, so that a terminal without hyperlinks still recognizes it,
    and it is what ends up in a log file. The URI has forward slashes and a
    drive is `file:///C:/...`; a windows path with backslashes is not a link
    a terminal opens.

    Args:
        key: what the link points to, in front of it.
        path: path of the file or the directory, relative paths are resolved.
    """
    uri = Path(path).resolve().as_uri()
    # the two spaces are the padding of the columns of `key_values`, so that
    # the link starts where the values of a key/value block start
    text = Text.assemble((f"{key:<{KEY_WIDTH}}", "bold"), "  ", (uri, Style(link=uri)))
    console.print(text, soft_wrap=True)
