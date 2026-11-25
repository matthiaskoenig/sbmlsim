"""The normalization of the colours of a style."""

import pytest

from sbmlsim.plot.colors import CSS4, CYCLE, TABLEAU, color_alpha, normalize_color


def test_tables_equal_matplotlib() -> None:
    colors = pytest.importorskip("matplotlib.colors")
    for name in CSS4:
        assert normalize_color(name) == colors.to_hex(name), name
    assert len(CSS4) == len(colors.CSS4_COLORS)
    for name in TABLEAU:
        assert normalize_color(f"tab:{name}") == colors.to_hex(f"tab:{name}")
    for k in range(10):
        assert normalize_color(f"C{k}") == colors.to_hex(f"C{k}") == CYCLE[k]


def test_case_padding_and_none() -> None:
    assert normalize_color("  DimGrey ") == "#696969"
    assert normalize_color("TAB:Red") == "#d62728"
    assert normalize_color("none") is None and normalize_color(None) is None


def test_hex_and_alpha() -> None:
    assert normalize_color("#F0a") == "#ff00aa"
    assert normalize_color("#ff000080") == "#ff0000"
    assert color_alpha("#ff000080") == ("#ff0000", 128 / 255)
    assert color_alpha("#f008")[0] == "#ff0000"
    assert color_alpha("red") == ("#ff0000", 1.0)
    assert color_alpha((1.0, 0.0, 0.0, 0.5)) == ("#ff0000", 0.5)
    assert normalize_color((0.0, 0.0, 1.0)) == "#0000ff"


def test_unknown_colours_raise() -> None:
    for bad in ("notacolour", "#12", "#gggggg", "C10", (2.0, 0.0, 0.0)):
        with pytest.raises(ValueError, match="unknown colour"):
            normalize_color(bad)
