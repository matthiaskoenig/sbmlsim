"""The colours of the figure model, `ColorType`, read as matplotlib reads them.

The normalization of the colours is the one of sbmlsim-core
(`sbmlsim_core.chart.colors`), whose tests compare it with matplotlib.
"""

import pytest

from sbmlsim.plot import colors as sbmlsim_colors


def test_colors_are_the_ones_of_sbmlsim_core() -> None:
    from sbmlsim_core.chart import colors

    assert sbmlsim_colors.normalize_color is colors.normalize_color
    assert sbmlsim_colors.color_alpha is colors.color_alpha
    assert sbmlsim_colors.normalize_color("tab:red") == "#d62728"


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

    tree = ast.parse(Path(plotting.__file__).read_text(encoding="utf-8"))
    imported = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    ]
    assert not [m for m in imported if m and m.startswith("matplotlib")]
