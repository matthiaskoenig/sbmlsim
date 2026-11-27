"""The colours of a style as matplotlib names them, `sbmlsim_core.chart.colors`.

The colour names of matplotlib (CSS, base, tableau, xkcd, the cycle `C0` to
`C9`, hex, grey levels and tuples of floats) normalized to `#rrggbb` without
matplotlib, which the chart builder of sbmlsim-core does; this module
re-exports it under its name in sbmlsim, read on first use as
`sbmlsim.plot.chart` does.
"""

from typing import TYPE_CHECKING, Any

from sbmlsim.plot.chart import lazy_attribute

if TYPE_CHECKING:
    from sbmlsim_core.chart.colors import (
        BASE,
        CSS4,
        CYCLE,
        TABLEAU,
        color_alpha,
        normalize_color,
    )

__all__ = ["BASE", "CSS4", "CYCLE", "TABLEAU", "color_alpha", "normalize_color"]


def __getattr__(name: str) -> Any:
    """Get a name of the colours, importing sbmlsim-core on first use."""
    return lazy_attribute(
        globals(), "sbmlsim_core.chart.colors", __all__, __name__, name
    )


def __dir__() -> list[str]:
    """Get the names of the module, those of the colours included."""
    return sorted({*globals(), *__all__})
