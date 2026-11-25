"""Test the figures of the reports: saved with their spec and inlined."""

import json
from pathlib import Path

import pytest

from sbmlsim.plot import Chart
from sbmlsim.report.figures import (
    FIGURE_SCRIPT,
    inline_figure,
    save_figure,
    write_figure_script,
)


def _chart() -> Chart:
    chart = Chart(title="a </script> title")
    chart.plot().curve([0.0, 1.0], [1.0, 2.0], label="curve")
    return chart


def test_a_figure_is_saved_with_its_spec(tmp_path: Path) -> None:
    pytest.importorskip("sbmlsim_core")
    written = save_figure(_chart(), tmp_path / "f.svg")
    assert written == [tmp_path / "f.svg", tmp_path / "f.json"]
    spec = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert spec["contract"] == "figure/1"
    assert not (tmp_path / "f.png").exists()


def test_a_png_copy_with_a_resolution(tmp_path: Path) -> None:
    pytest.importorskip("sbmlsim_core")
    written = save_figure(_chart(), tmp_path / "f.svg", png_dpi=100.0)
    assert written[-1] == tmp_path / "f.png"
    png = save_figure(_chart(), tmp_path / "g.png", png_dpi=100.0)
    assert png == [tmp_path / "g.png", tmp_path / "g.json"]


def test_another_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="svg or png"):
        save_figure(_chart(), tmp_path / "f.pdf")


def test_an_inline_figure_is_its_svg_and_its_spec(tmp_path: Path) -> None:
    pytest.importorskip("sbmlsim_core")
    save_figure(_chart(), tmp_path / "f.svg", png_dpi=72.0)
    inline = inline_figure(tmp_path / "f.svg")
    assert inline["svg"].startswith("<svg")
    # the page gives inline SVG its namespace, it names no URL
    assert "http" not in inline["svg"]
    assert "</script>" not in inline["spec_text"]
    assert json.loads(inline["spec_text"].replace("<\\/", "</"))["title"]
    # a PNG and a figure without its spec are shown as images
    assert inline_figure(tmp_path / "f.png") == {}
    (tmp_path / "f.json").unlink()
    assert inline_figure(tmp_path / "f.svg") == {}


def test_the_figure_script_is_written_once(tmp_path: Path) -> None:
    """The script of a report directory is written when it is missing or other."""
    pytest.importorskip("sbmlsim_core")
    path = write_figure_script(tmp_path)
    assert path == tmp_path / FIGURE_SCRIPT
    text = path.read_text(encoding="utf-8")
    assert "hydrateFigures" in text and "atob" in text
    before = path.stat().st_mtime_ns
    assert write_figure_script(tmp_path) == path
    assert path.stat().st_mtime_ns == before
    path.write_text("stale", encoding="utf-8")
    write_figure_script(tmp_path)
    assert path.read_text(encoding="utf-8") == text
