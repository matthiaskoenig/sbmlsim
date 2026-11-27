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


#: a colour, its alpha and the colour of the figure model: a name is read as
#: `matplotlib.colors.to_hex(to_rgba(color, alpha), keep_alpha=True)` of
#: matplotlib 3.11 reads it, a hex string keeps its digits and gets the alpha
#: appended when it has none
PARSED = [
    ("k", 1.0, "#000000ff"),
    ("r", 0.5, "#ff000080"),
    ("0.4", 0.3, "#6666664c"),
    ("tab:red", 1.0, "#d62728ff"),
    ("C3", 0.25, "#d6272840"),
    ("xkcd:sky blue", 1.0, "#75bbfdff"),
    ("dimgrey", 0.8, "#696969cc"),
    ("none", 1.0, "#00000000"),
    ("#ff0000", 0.5, "#ff000080"),
    ("#ff000080", 1.0, "#ff000080"),
    ("#FF0000", 1.0, "#FF0000ff"),
]


@pytest.mark.parametrize(("color", "alpha", "expected"), PARSED)
def test_parse_color_reads_a_colour_as_matplotlib(
    color: str, alpha: float, expected: str
) -> None:
    from sbmlsim.plot.plotting import ColorType

    parsed = ColorType.parse_color(color, alpha)
    assert parsed is not None
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
