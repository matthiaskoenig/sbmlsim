"""Test the console output of the scripts."""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sbmlsim import display
from sbmlsim.fit import display as fit_display


def _terminal(monkeypatch: pytest.MonkeyPatch, width: int) -> StringIO:
    """Print to a terminal of the given width, with its control codes."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=width)
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
