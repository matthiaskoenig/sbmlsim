"""Test the console output of the scripts."""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sbmlsim import display
from sbmlsim.fit import display as fit_display


def _terminal(monkeypatch: pytest.MonkeyPatch, width: int) -> StringIO:
    """Print to a terminal of the given width, with its control codes.

    The terminal is one with virtual terminal sequences, i.e. on windows the
    Windows Terminal or VS Code rather than the legacy console, which rich
    assumes on a windows runner and which has no hyperlinks.
    """
    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=True, width=width, legacy_windows=False
    )
    monkeypatch.setattr(display, "console", console)
    return buffer


def test_link_is_a_hyperlink_of_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal gets a hyperlink (OSC 8) to the file, so it opens with a click."""
    buffer = _terminal(monkeypatch, width=200)
    display.link("report", "results/index.html")
    uri = Path("results/index.html").resolve().as_uri()
    out = buffer.getvalue()
    assert "\x1b]8;" in out
    assert uri in out


def test_a_long_link_is_a_single_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """A link longer than the terminal is not broken into lines."""
    buffer = _terminal(monkeypatch, width=40)
    path = Path("results") / ("simulation_" * 10) / "index.html"
    display.link("report", path)
    out = buffer.getvalue()
    assert out.count("\n") == 1
    assert path.resolve().as_uri() in out


def test_link_without_a_terminal_is_the_uri(capsys: pytest.CaptureFixture[str]) -> None:
    """Without a terminal, e.g. in a log file, the link is its URI."""
    display.link("figures", "results/_figures")
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert out.split() == ["figures", Path("results/_figures").resolve().as_uri()]


def test_a_link_is_aligned_with_the_values(capsys: pytest.CaptureFixture[str]) -> None:
    """The link starts in the column of the values of a key/value block."""
    display.key_values({"experiments": 3})
    display.link("report", "index.html")
    values, link = capsys.readouterr().out.splitlines()
    assert link.index("file://") == values.index("3")


def test_the_fit_output_is_the_output_of_the_scripts() -> None:
    """The sections, blocks and links of a fit are those of `sbmlsim.display`."""
    assert fit_display.section is display.section
    assert fit_display.key_values is display.key_values
    assert fit_display.link is display.link
    assert fit_display.ICON_REPORT == display.ICON_REPORT


def test_section_icon_is_left_out_without_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream which is not UTF-8 (cp1252 of a redirected windows stdout) gets no emoji."""
    import io

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    console = Console(file=stream, width=40, legacy_windows=False)
    monkeypatch.setattr(display, "console", console)
    display.section("Report", icon=display.ICON_REPORT)
    stream.flush()
    assert b"Report" in raw.getvalue()


def _plain(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Print to a file, without the control codes of a terminal: a link once."""
    buffer = StringIO()
    monkeypatch.setattr(display, "console", Console(file=buffer, width=400))
    return buffer


def test_a_link_into_a_staging_folder_names_the_folder_it_ends_up_as(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A link printed while the output is staged names its final folder (#669)."""
    buffer = _plain(monkeypatch)
    staging = tmp_path / ".all.0123"
    target = tmp_path / "all"
    with display.relocated({staging: target}):
        display.link("report", staging / "index.html")
        display.link("figures", staging / "_figures")
        display.link("other", tmp_path / "other" / "index.html")
    display.link("after", staging / "index.html")

    out = buffer.getvalue()
    assert (target / "index.html").resolve().as_uri() in out
    assert (target / "_figures").resolve().as_uri() in out
    assert (tmp_path / "other" / "index.html").resolve().as_uri() in out
    # outside of the block a link is the path it is given
    assert out.count(staging.resolve().as_uri()) == 1


def test_the_deepest_folder_of_a_relocation_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A result folder moved up out of the staging folder is that folder's target.

    The records of a fit are written into `<staging>/<fit id>` and moved up into
    the staging folder, which becomes `<fit id>` itself.
    """
    buffer = _plain(monkeypatch)
    staging = tmp_path / ".PK_1.0123"
    target = tmp_path / "PK_1"
    with display.relocated({staging: target, staging / "PK_1": target}):
        display.link("report", staging / "PK_1" / "index.html")
        display.link("results", staging / "index.html")

    out = buffer.getvalue()
    assert out.count((target / "index.html").resolve().as_uri()) == 2
    assert "PK_1/PK_1" not in out
