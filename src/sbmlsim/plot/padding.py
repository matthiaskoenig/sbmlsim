"""The values of a curve as one line, without the padding of ragged results.

The values of a curve are the labelled arrays of its data, see
`Data.get_data`; they are broadcast by the names of their dimensions and must
leave one dimension, along which the line is drawn: the time of a timecourse,
the points of a ragged result or the rows of a dataset. A dimension of a scan
is selected with `Data(sel=...)`. In the ragged layout every simulation keeps
its own time points and one with fewer points is padded with `NaN`: the
points whose x is `NaN` are the padding, a `NaN` of y is a gap of the data and
is kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.data import ROW
from sbmlsim.result.scan import POINT, TIME


def line_values(sid: str, *arrays: xr.DataArray | None) -> tuple[Any, ...]:
    """Get the values of a curve as one line, without the padding.

    Args:
        sid: the id of the curve, for the errors.
        arrays: the x, y and error arrays of the curve, `None` for a missing
            one.

    Returns:
        The values of every array along the one dimension which is left,
        `None` stays. The values are typed `Any`: they are handed to the
        functions of matplotlib and plotly, which take any array.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension,
            or more than one dimension is left after broadcasting.
    """
    given = [a for a in arrays if a is not None]
    if not given:
        return tuple(arrays)
    try:
        aligned = xr.align(*given, join="exact")
    except ValueError as err:
        raise ValueError(
            f"The data of the curve '{sid}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = iter(xr.broadcast(*aligned))
    lines = [None if a is None else next(broadcast) for a in arrays]
    dims = next(line for line in lines if line is not None).dims
    if len(dims) > 1:
        scan = [str(d) for d in dims if d not in (TIME, POINT, ROW)] or [
            str(d) for d in dims
        ]
        raise ValueError(
            f"The curve '{sid}' has the dimensions {[str(d) for d in dims]}; a curve "
            f"draws one line: select one label of {scan} with Data(sel=...)."
        )
    return without_padding(
        *[None if line is None else np.asarray(line.values) for line in lines]
    )


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
