"""The matplotlib objects of the colours of points, for the serializers until they are removed.

`sbmlsim.plot.points` knows no matplotlib; the two serializers which still draw
with it get the norms, colour maps and the colour cycle from here.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    LogNorm,
    Normalize,
    to_hex,
)

from sbmlsim.plot.points import PointStyle, point_ramp


def cycle_color(position: int) -> str:
    """Get the colour of the colour cycle of matplotlib at a position, as a hex string."""
    return to_hex(f"C{position}")


def style_norm(styles: PointStyle) -> Normalize:
    """Get the norm of the colour bar of a point style."""
    scale = styles.scale
    if scale is None:
        return Normalize(0.0, 1.0)
    if scale.categorical:
        n = round(scale.high - scale.low)
        return BoundaryNorm(np.arange(n + 1) - 0.5, n)
    if scale.log:
        return LogNorm(scale.low, scale.high)
    return Normalize(scale.low, scale.high)


def style_colormap(styles: PointStyle) -> Colormap:
    """Get the colour map of the colour bar of a point style."""
    return ListedColormap(styles.ramp)


def style_ticks(styles: PointStyle) -> list[str] | None:
    """Get the tick labels of a discrete colour bar, `None` for one of values."""
    scale = styles.scale
    return scale.labels if scale is not None and scale.categorical else None


def point_colormap(color: str | None) -> Colormap:
    """Get the colour map of the colour bar of a dimension, see `point_ramp`."""
    return ListedColormap(point_ramp(color))
