"""The curves of a result of a scan, without the padding of ragged results.

The values of a result of a scan have the dimensions of the scan first and
the time last, see `sbmlsim.result.scan`; in the ragged layout every
simulation keeps its own time points and one with fewer points is padded with
`NaN`. A figure draws the first point of a scan, and the points whose x is
`NaN` are the padding; a `NaN` of y is a gap of the data and is kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def first_curve(values: np.ndarray | None) -> np.ndarray | None:
    """Get the values of the first point of a scan.

    Args:
        values: the values, a dimension per dimension of the scan and the
            time last.

    Returns:
        The values of the first point over the time, `None` without values.
    """
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim <= 1:
        return array
    return array.reshape(-1, array.shape[-1])[0]


def without_padding(x: Any, *others: Any) -> tuple[Any, ...]:
    """Drop the points of a curve whose x is the padding of a ragged result.

    Args:
        x: the x values.
        others: arrays of the same points, e.g. y and the errors, or `None`.

    Returns:
        The x values and the others without the padded points, `None` stays.
        The values are typed `Any`: they are handed to the functions of
        matplotlib and plotly, which take any array.
    """
    if x is None:
        return (x, *others)
    xs = np.asarray(x)
    if not np.issubdtype(xs.dtype, np.floating):
        return (x, *others)
    keep = ~np.isnan(xs)
    if keep.all():
        return (x, *others)
    return (
        xs[keep],
        *[
            None
            if other is None
            else (np.asarray(other)[keep] if np.size(other) == keep.size else other)
            for other in others
        ],
    )
