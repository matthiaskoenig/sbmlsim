import pytest

from sbmlsim.plot.plotting import Style


color_data = [
    ("r", "#ff0000ff", 1.0),
    ("g", "#008000ff", 1.0),
    ("b", "#0000ffff", 1.0),
    ("white", "#ffffffff", 1.0),
    ("black", "#000000ff", 1.0),
    ("b", "#0000ff00", 0.0),
    ("#1234ff44", "#1234ff44", 1.0),
    ("#1234ff", "#1234ffff", 1.0),
    ("#56784a", "#56784a00", 0.0),
]


@pytest.mark.parametrize("color, hex, alpha", color_data)
def test_parse_color(color, hex, alpha):
    c = Style.parse_color(color, alpha=alpha)
    assert c.color == hex
