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

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
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

#: the folders a run writes into and the folders they end up as, see
#: `relocated`
_RELOCATIONS: ContextVar[tuple[tuple[tuple[Path, Path], ...], ...]] = ContextVar(
    "relocations", default=()
)


@contextmanager
def relocated(moves: Mapping[Path | str, Path | str]) -> Generator[None]:
    """Let the links printed in the block name the folders the output ends up in.

    A command which writes into a staging folder and moves it into its place
    when the run succeeded (e.g. the staged swap of the commands of pkpdsim)
    prints its links while the output is still in the staging folder, which no
    longer exists once it is moved (#669). Within the block a link to a path
    in one of the folders of `moves` names the path in the folder it is moved
    to; the deepest folder wins, so a folder of the result which is moved up
    out of the staging folder is given before the staging folder itself.
    Blocks nest, the inner moves are applied first.

    Args:
        moves: the folders the run writes into, by the folder each ends up as.
    """
    resolved = tuple(
        sorted(
            ((Path(old).resolve(), Path(new).resolve()) for old, new in moves.items()),
            key=lambda move: len(move[0].parts),
            reverse=True,
        )
    )
    token = _RELOCATIONS.set((resolved, *_RELOCATIONS.get()))
    try:
        yield
    finally:
        _RELOCATIONS.reset(token)


def _final(path: Path) -> Path:
    """Get the path a resolved path ends up at after the moves of `relocated`."""
    for moves in _RELOCATIONS.get():
        for old, new in moves:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
    return path


def section(title: str, icon: str | None = None) -> None:
    """Start a section of the output.

    A blank line separates the section from whatever came before it, so the
    sections are told apart whether the previous one ended in a table or in a
    key/value block.

    Args:
        title: title of the section.
        icon: emoji in front of the title, e.g. `ICON_REPORT`; left out when the
            console does not write UTF-8.
    """
    # an emoji cannot be written to a stream of another encoding, e.g. the
    # cp1252 of a redirected stdout on windows, and would raise
    can_encode = console.encoding.lower().replace("-", "").startswith("utf")
    prefix = f"{icon} " if icon and can_encode else ""
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
        path: path of the file or the directory, relative paths are resolved;
            a path in a folder of `relocated` names the folder it ends up as.
    """
    uri = _final(Path(path).resolve()).as_uri()
    # the two spaces are the padding of the columns of `key_values`, so that
    # the link starts where the values of a key/value block start
    text = Text.assemble((f"{key:<{KEY_WIDTH}}", "bold"), "  ", (uri, Style(link=uri)))
    console.print(text, soft_wrap=True)
