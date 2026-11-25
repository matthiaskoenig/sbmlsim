"""The normalization of the colours of a style."""

import pytest

from sbmlsim.plot._xkcd_colors import XKCD
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


def test_shorthands_equal_matplotlib() -> None:
    colors = pytest.importorskip("matplotlib.colors")
    for name in "bgrcmykw":
        assert normalize_color(name) == colors.to_hex(name), name
    for grey in ("0", "0.4", "0.75", "1", "1.0", ".5"):
        assert normalize_color(grey) == colors.to_hex(grey), grey
    for name in XKCD:
        assert normalize_color(f"xkcd:{name}") == colors.to_hex(f"xkcd:{name}"), name
    assert len(XKCD) == len(colors.XKCD_COLORS)
    assert normalize_color("XKCD:Sky Blue") == colors.to_hex("xkcd:sky blue")


def test_grey_levels_outside_of_0_and_1_raise() -> None:
    for bad in ("1.5", "-0.1", "nan"):
        with pytest.raises(ValueError, match="unknown colour"):
            normalize_color(bad)


PARSED = [
    ("k", 1.0),
    ("r", 0.5),
    ("0.4", 0.3),
    ("tab:red", 1.0),
    ("C3", 0.25),
    ("xkcd:sky blue", 1.0),
    ("dimgrey", 0.8),
    ("none", 1.0),
    ("#ff0000", 0.5),
    ("#ff000080", 1.0),
    ("#FF0000", 1.0),
]


@pytest.mark.parametrize(("color", "alpha"), PARSED)
def test_parse_color_equals_matplotlib(color: str, alpha: float) -> None:
    colors = pytest.importorskip("matplotlib.colors")
    from sbmlsim.plot.plotting import ColorType

    parsed = ColorType.parse_color(color, alpha)
    assert parsed is not None
    if color.startswith("#"):
        expected = color + f"{round(alpha * 255):02x}" if len(color) == 7 else color
    else:
        expected = colors.to_hex(colors.to_rgba(color, alpha), keep_alpha=True)
    assert parsed.color == expected


def test_parse_color_refuses_short_hex_and_takes_nothing() -> None:
    from sbmlsim.plot.plotting import ColorType

    assert ColorType.parse_color(None) is None
    assert ColorType.parse_color("") is None
    with pytest.raises(ValueError, match="Incorrect hex color"):
        ColorType.parse_color("#f00")
    with pytest.raises(ValueError, match="unknown colour"):
        ColorType.parse_color("notacolour")


def test_plotting_imports_no_matplotlib() -> None:
    import ast
    from pathlib import Path

    import sbmlsim.plot.plotting as plotting

    tree = ast.parse(Path(plotting.__file__).read_text())
    imported = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    ]
    assert not [m for m in imported if m and m.startswith("matplotlib")]
