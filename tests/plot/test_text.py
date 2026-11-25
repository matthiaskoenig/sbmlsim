"""Test the width of a text as the renderer measures it."""

import pytest

from sbmlsim.plot import Chart
from sbmlsim.plot.text import text_width


def test_the_width_of_a_text_is_the_one_of_the_renderer() -> None:
    """A legend entry is its swatch and its label: the label is measured alike."""
    pytest.importorskip("sbmlsim_core")
    from sbmlsim.engine import import_core

    labels = [
        "i",
        "W",
        "GMFE = 1.26, AFE = 1.01, 2-fold = 93.9%",
        "R² = 0.940",
        "√(f(x) · y) ± ×/÷ µ",
    ]
    rest = []
    for label in labels:
        chart = Chart(legend_position="outside")
        chart.plot().curve([0, 1], [0, 1], label=label, legend=label)
        geometry = import_core().figure_geometry(chart.to_json())
        [entry] = geometry["legend_entries"]
        rest.append(entry["rect"]["width"] - text_width(label))
    assert max(rest) - min(rest) == pytest.approx(0.0, abs=1e-6)


def test_the_widest_line_is_the_width() -> None:
    assert text_width("ab\nabcd") == text_width("abcd")
    assert text_width("x", size=22.0) == pytest.approx(2 * text_width("x"))
