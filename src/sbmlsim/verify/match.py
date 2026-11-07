"""Match the rows of the results of two dumps and measure their difference."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np

from sbmlsim.verify.dump import TIME_DIMENSIONS, Dump

__all__ = [
    "SCALE_FLOOR",
    "TIME_RESOLUTION",
    "ArrayDifference",
    "Columns",
    "InterpolationRange",
    "PointMatch",
    "RowMatch",
    "columns",
    "equal",
    "interpolated_pairs",
    "interpolation_envelope",
    "matched_rows",
    "point_matches",
    "point_times",
    "relative_difference",
    "relative_differences",
    "scaled_difference",
    "task_columns",
    "time_axis",
]


def time_axis(dims: Sequence[str]) -> int | None:
    """Get the position of the time (or scan point) dimension."""
    for k, dim in enumerate(dims):
        if dim in TIME_DIMENSIONS:
            return k
    return None


def point_times(
    dump: Dump, sid: str, task: str, dims: Sequence[str], shape: tuple[int, ...]
) -> np.ndarray | None:
    """Get the output time of every value of a variable of a task, in its layout.

    A grid has one vector of times (`("time",)`) for every point of a scan; the
    ragged layout (`_point`) has the times of every point, over the dimensions
    of the variable, padded with NaN to the longest point.

    Args:
        dump: the dump.
        sid: the experiment.
        task: the task.
        dims: the dimensions of the variable.
        shape: the shape of the variable.

    Returns:
        The times in the shape of the variable, or `None` if the task has no
        times or they do not fit the variable.
    """
    entry = dump["results"].get((sid, task, "time"))
    axis = time_axis(dims)
    if entry is None or axis is None:
        return None
    tdims, values = entry
    times = values.astype(float)
    if len(tdims) == 1 and time_axis(tdims) == 0 and times.size == shape[axis]:
        expand = [1] * len(shape)
        expand[axis] = -1
        return np.broadcast_to(times.reshape(expand), shape)
    if sorted(tdims) != sorted(dims):
        return None
    times = np.transpose(times, [list(tdims).index(d) for d in dims])
    return times if times.shape == shape else None


class RowMatch(NamedTuple):
    """The rows of the output of a task in two dumps which are compared."""

    #: the rows of the old and of the new output which are compared
    old: np.ndarray
    new: np.ndarray
    #: rows left out at a time which the other output has fewer times (the
    #: states before a change)
    ignored_old: int
    ignored_new: int
    #: rows of the old output without a time in the new output
    unmatched: int
    #: rows of the new output without a time in the old output
    unmatched_new: int
    #: the times of the old output which the new output does not have, and
    #: the other way round
    times_old: np.ndarray
    times_new: np.ndarray
    #: the rows of the old output at a time which the new output does not have
    #: but whose segment contains it, compared with the new output interpolated
    #: at their times (`interpolation_envelope`), and the other way round
    interpolated_old: np.ndarray = np.arange(0)
    interpolated_new: np.ndarray = np.arange(0)


#: the smallest scale of an array relative to the largest scale of its variable in
#: the tasks of its experiment, see `relative_differences`
SCALE_FLOOR = 1e-12

#: the resolution of the output times relative to the largest time: two times
#: closer than it are one time, the same time computed in another order; two
#: times further apart are two times of the steps of the integrator, which are
#: compared by interpolation
TIME_RESOLUTION = 1e-12


def _runs(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the first row and the number of rows of every time of an output."""
    starts = np.flatnonzero(np.append(True, np.diff(times) != 0))
    return starts, np.diff(np.append(starts, times.size))


def _last_rows(ends: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Get the last `counts[k]` rows before `ends[k]` of every time, in order."""
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(ends - counts, counts) + offsets


def matched_rows(old_times: np.ndarray, new_times: np.ndarray) -> RowMatch:
    """Match the rows of the output times of a task in two dumps.

    The times are matched relative to the largest time (`TIME_RESOLUTION`).
    The rows of a time which an output has more than once (a change: the state
    before it, then the state after it) are matched from the last one, the k-th
    last old row with the k-th last new row, and the surplus rows of the output
    with more of them are left out. So every row of the common times of two dumps
    is compared, and where one dump has fewer rows at a change than the other
    the state after the change is the one compared.

    The steps of the integrator differ between two runs whose model or
    tolerances differ, so most times of an adaptive output are in one dump
    only. A row at such a time is compared with the other output interpolated
    at its time (`interpolated_old`, `interpolated_new`) if the time lies
    between two rows of the other output (a segment between its changes,
    `interpolation_envelope`) and occurs once in its own output (the states
    before and after a change which the other output does not have are not
    one value of it); every other row of a time only one output has is
    unmatched.

    Args:
        old_times: the output times of the old dump.
        new_times: the output times of the new dump.

    Returns:
        The rows which are compared and the rows which are not.
    """
    if old_times.shape == new_times.shape and np.array_equal(old_times, new_times):
        rows = np.arange(old_times.size)
        none = np.zeros(0)
        return RowMatch(rows, rows, 0, 0, 0, 0, none, none)
    if old_times.size == 0 or new_times.size == 0:
        empty = np.arange(0)
        return RowMatch(
            empty,
            empty,
            0,
            0,
            int(old_times.size),
            int(new_times.size),
            np.unique(old_times),
            np.unique(new_times),
        )
    starts_a, counts_a = _runs(old_times)
    starts_b, counts_b = _runs(new_times)
    a, b = old_times[starts_a], new_times[starts_b]
    j = np.clip(np.searchsorted(b, a), 0, b.size - 1)
    previous = np.clip(j - 1, 0, b.size - 1)
    j = np.where(np.abs(b[previous] - a) < np.abs(b[j] - a), previous, j)
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    common = np.abs(b[j] - a) < TIME_RESOLUTION * scale
    # a new time is matched with one old time only, the first close to it
    first = np.zeros(a.size, dtype=bool)
    first[np.unique(np.where(common, j, -1), return_index=True)[1]] = True
    ka = np.flatnonzero(common & first)
    kb = j[ka]
    n = np.minimum(counts_a[ka], counts_b[kb])
    only_a, only_b = np.ones(a.size, dtype=bool), np.ones(b.size, dtype=bool)
    only_a[ka], only_b[kb] = False, False
    # a time of one output only which occurs once in it and lies between two
    # rows of the other output is compared by interpolation
    inside_a = only_a & (counts_a == 1) & (a > b[0]) & (a < b[-1])
    inside_b = only_b & (counts_b == 1) & (b > a[0]) & (b < a[-1])
    only_a &= ~inside_a
    only_b &= ~inside_b
    return RowMatch(
        _last_rows(starts_a[ka] + counts_a[ka], n),
        _last_rows(starts_b[kb] + counts_b[kb], n),
        int(np.sum(counts_a[ka] - n)),
        int(np.sum(counts_b[kb] - n)),
        int(np.sum(counts_a[only_a])),
        int(np.sum(counts_b[only_b])),
        a[only_a],
        b[only_b],
        starts_a[inside_a],
        starts_b[inside_b],
    )


class InterpolationRange(NamedTuple):
    """The range of a trajectory at times, from the interpolants of its steps."""

    #: the lowest and the highest value of the range, (times, columns) each
    low: np.ndarray
    high: np.ndarray
    #: the spread of the interpolants, the resolution of the interpolation
    spread: np.ndarray


def interpolation_envelope(
    times: np.ndarray, values: np.ndarray, at: np.ndarray
) -> InterpolationRange:
    """Get the range of the trajectory of an output at times, from its interpolants.

    The rows of an adaptive output are the steps of the integrator, which are
    too far apart for one interpolant to reach the threshold of a comparison:
    between two runs of hctz whose steps differ (the relative tolerance 1e-9
    and 1e-10), the linear interpolant is up to 4e-2 of the scale of an array
    off the other run, the monotone cubic one 4e-3, the cubic spline swings
    at a sharp transient (the concentration in the urine after it was emptied).
    Three interpolants of every segment between the changes of the output
    (the rows between two repeated times), the linear one, the monotone cubic
    one (PCHIP) and the cubic spline (not-a-knot), span the envelope of the
    trajectory, and their spread is the uncertainty of the interpolation: the
    range is the envelope widened by its width on both sides. A value within
    it is no difference which the steps of the output can tell, and the
    spread is the resolution which the interpolation reaches: it depends on
    the density of the steps only, about 1e-2 of the scale of an oral dose on
    20 steps over 48 hr and up to 4e-2 on the steps of hctz, so `compare`
    reports an array whose resolution is coarser than its threshold as
    unresolved, never as equal. A column with a value which is not finite in a
    segment is interpolated linearly there.

    Args:
        times: the output times, ascending, a time of a change twice.
        values: the values of the rows, (rows, columns).
        at: the times, each between two rows of a segment.

    Returns:
        The range and the spread of the interpolants, (times, columns) each.
    """
    from scipy.interpolate import CubicSpline, PchipInterpolator

    # the row after every time and the segment of the rows, which a repeated
    # time starts
    after = np.searchsorted(times, at, side="right")
    segment = np.cumsum(np.append(0, np.diff(times) == 0))
    lowest = np.full((at.size, values.shape[1]), np.nan)
    highest = np.full((at.size, values.shape[1]), np.nan)
    spread = np.full((at.size, values.shape[1]), np.nan)
    for s in np.unique(segment[after]):
        rows = np.flatnonzero(segment == s)
        queries = np.flatnonzero(segment[after] == s)
        t, v, q = times[rows], values[rows], at[queries]
        k = np.searchsorted(t, q, side="right")
        weight = ((q - t[k - 1]) / (t[k] - t[k - 1]))[:, None]
        linear = v[k - 1] + weight * (v[k] - v[k - 1])
        low, high = linear.copy(), linear.copy()
        finite = np.all(np.isfinite(v), axis=0)
        if t.size > 2 and finite.any():
            for interpolant in (PchipInterpolator, CubicSpline):
                curve = interpolant(t, v[:, finite], axis=0)(q)
                low[:, finite] = np.minimum(low[:, finite], curve)
                high[:, finite] = np.maximum(high[:, finite], curve)
        width = high - low
        lowest[queries], highest[queries] = low - width, high + width
        spread[queries] = width
    return InterpolationRange(lowest, highest, spread)


def _squeezed(
    values: np.ndarray, dims: list[str], other: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Leave out the dimensions of size 1 which the other array does not have."""
    drop = [
        k for k, d in enumerate(dims) if d not in other and values.shape[1 + k] == 1
    ]
    if not drop:
        return values, dims
    kept = [d for k, d in enumerate(dims) if k not in drop]
    return np.squeeze(values, axis=tuple(1 + k for k in drop)), kept


def _aligned(
    dims_a: Sequence[str], a: np.ndarray, dims_b: Sequence[str], b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Get two arrays with the time first and the other dimensions in the order of `a`.

    Both arrays have a time dimension. A dimension of size 1 which only one array
    has is left out (a scan of one point is a simulation).

    Returns:
        The arrays as floats, or `None` if their other dimensions differ.
    """
    axis_a, axis_b = time_axis(dims_a), time_axis(dims_b)
    assert axis_a is not None and axis_b is not None
    x = np.moveaxis(a.astype(float), axis_a, 0)
    y = np.moveaxis(b.astype(float), axis_b, 0)
    rest_a = [str(d) for k, d in enumerate(dims_a) if k != axis_a]
    rest_b = [str(d) for k, d in enumerate(dims_b) if k != axis_b]
    x, rest_a = _squeezed(x, rest_a, rest_b)
    y, rest_b = _squeezed(y, rest_b, rest_a)
    if sorted(rest_a) != sorted(rest_b):
        return None
    return x, np.transpose(y, [0, *(1 + rest_b.index(d) for d in rest_a)])


class Columns(NamedTuple):
    """A variable of a task in two dumps, one column per point of a scan.

    The time is the first dimension, the points are the other dimensions of the
    variable, matched by name and flattened in the order of the old dump.
    """

    #: the values of the old and of the new dump, (rows, points)
    old: np.ndarray
    new: np.ndarray
    #: the output time of every value
    old_times: np.ndarray
    new_times: np.ndarray


def columns(old: Dump, new: Dump, key: tuple[str, str, str]) -> Columns | None:
    """Get a variable of a task in two dumps with the times of every value.

    Returns:
        The columns, or `None` if the dimensions of the variables differ or a
        dump has no times of the task which fit its variable.
    """
    (dims_a, a), (dims_b, b) = old["results"][key], new["results"][key]
    sid, task, _ = key
    ta = point_times(old, sid, task, dims_a, a.shape)
    tb = point_times(new, sid, task, dims_b, b.shape)
    if ta is None or tb is None:
        return None
    values = _aligned(dims_a, a, dims_b, b)
    times = _aligned(dims_a, ta, dims_b, tb)
    if values is None or times is None or values[0].shape[1:] != values[1].shape[1:]:
        return None
    x, y, tx, ty = (v.reshape(v.shape[0], -1) for v in (*values, *times))
    return Columns(x, y, tx, ty)


def task_columns(old: Dump, new: Dump, sid: str, task: str) -> Columns | None:
    """Get the output times of every point of a task in two dumps.

    The times are those of a variable of the task (the first by name), so
    that the points are matched by the names of their dimensions as the
    variables are; a task with the times only compares their arrays.

    Returns:
        The times as the columns (values and times), or `None`.
    """
    variables = sorted(
        v
        for (s, t, v) in old["results"]
        if s == sid and t == task and v != "time" and (s, t, v) in new["results"]
    )
    for variable in variables:
        found = columns(old, new, (sid, task, variable))
        if found is not None:
            return Columns(found.old_times, found.new_times, *found[2:])
    key = (sid, task, "time")
    if key in old["results"] and key in new["results"]:
        return columns(old, new, key)
    return None


class PointMatch(NamedTuple):
    """The rows of points of a task which are compared, see `point_matches`."""

    #: the points (columns) with these times
    points: np.ndarray
    #: the rows of the old and of the new values which are compared
    old: np.ndarray
    new: np.ndarray
    #: the match of the times of the points
    match: RowMatch
    #: the rows of the old and of the new values with a time (not padding)
    rows_old: np.ndarray = np.arange(0)
    rows_new: np.ndarray = np.arange(0)


def _same_columns(times: np.ndarray) -> bool:
    """Check that every point has the times of the first, NaN equal to NaN."""
    first = times[:, :1]
    return bool(np.all((times == first) | (np.isnan(times) & np.isnan(first))))


def point_matches(old_times: np.ndarray, new_times: np.ndarray) -> list[PointMatch]:
    """Match the rows of every point of a task by its own times.

    The points of a grid have the same times and one match; a point of a
    ragged result is matched by its own times, without the NaN which pad it to
    the longest point. Points with the same times in both dumps share a match.

    Args:
        old_times: the times of the old dump, (rows, points).
        new_times: the times of the new dump, (rows, points).

    Returns:
        The matches.
    """
    points = old_times.shape[1]
    if _same_columns(old_times) and _same_columns(new_times):
        groups = [list(range(points))]
    else:
        by_times: dict[bytes, list[int]] = {}
        for p in range(points):
            key = old_times[:, p].tobytes() + new_times[:, p].tobytes()
            by_times.setdefault(key, []).append(p)
        groups = list(by_times.values())
    matches: list[PointMatch] = []
    for group in groups:
        ta, tb = old_times[:, group[0]], new_times[:, group[0]]
        rows_a, rows_b = np.flatnonzero(~np.isnan(ta)), np.flatnonzero(~np.isnan(tb))
        match = matched_rows(ta[rows_a], tb[rows_b])
        matches.append(
            PointMatch(
                np.array(group),
                rows_a[match.old],
                rows_b[match.new],
                match,
                rows_a,
                rows_b,
            )
        )
    return matches


def interpolated_pairs(
    found: Columns, match: PointMatch
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get the rows of times only one dump has with the other dump at their times.

    A row of the old dump is paired with the nearest value of the range of
    the interpolants of the new dump at its time (`interpolation_envelope`),
    a row of the new dump with the nearest value of the range of the old
    dump, so the difference of a pair is the distance from the range, zero
    within it. The old value comes first, the scale of a comparison, the
    spread of the interpolants at the row third, the resolution of the pair.

    Args:
        found: the variable of a task in the two dumps.
        match: the match of the rows of points with the same times.

    Returns:
        The old and the new values and the spread, (rows, points) each.
    """
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    first = match.points[0]
    sides = (
        (found.old, found.old_times, match.rows_old, match.match.interpolated_old),
        (found.new, found.new_times, match.rows_new, match.match.interpolated_new),
    )
    for k, (values, times, rows, interpolated) in enumerate(sides):
        if not interpolated.size:
            continue
        other_values, other_times, other_rows, _ = sides[1 - k]
        own_rows = rows[interpolated]
        own = values[np.ix_(own_rows, match.points)]
        span = interpolation_envelope(
            other_times[other_rows, first],
            other_values[np.ix_(other_rows, match.points)],
            times[own_rows, first],
        )
        nearest = np.clip(own, span.low, span.high)
        pairs.append(
            (own, nearest, span.spread) if k == 0 else (nearest, own, span.spread)
        )
    return pairs


def relative_difference(old: Dump, new: Dump, key: tuple[str, str, str]) -> float:
    """Get the difference of a result array on the matched rows.

    The time is moved first, the other dimensions are matched by name (a
    dimension of size 1 which only one array has is left out), and the rows of
    every point are matched by its own times (`point_matches`); a row at a
    time which only one dump has is compared with the other dump interpolated
    at its time, its distance from the range of the interpolants
    (`interpolated_pairs`). The difference is max|new - old| / max|old|; other
    dimensions and NaN at different positions are an infinite difference. A
    numeric array without a time, e.g. the values at the end of the points of
    a scan, is compared on the same dimensions relative to its scale, any
    other array exactly; the variable `time` is compared as the output times
    of the points of the task.
    """
    return relative_differences(old, new, [key])[key].difference


class ArrayDifference(NamedTuple):
    """The difference of an array of two dumps and the resolution it was found at."""

    #: max|new - old| / max|old| on the matched and the interpolated rows
    difference: float
    #: the largest spread of the interpolants of the interpolated rows relative
    #: to the same scale, 0 without interpolated rows: a difference below it
    #: cannot be told from the interpolation
    resolution: float


def relative_differences(
    old: Dump, new: Dump, keys: Iterable[tuple[str, str, str]]
) -> dict[tuple[str, str, str], ArrayDifference]:
    """Get the difference of every result array of keys, see `relative_difference`.

    The arrays of a task with the same output times are compared together, so
    that the interpolants of a segment are built once for all of them. An array
    is compared relative to its scale, but at least to `SCALE_FLOOR` of the
    largest scale of its variable in the tasks of its experiment among the
    keys: a variable which is zero in a task, e.g. the drug of a control
    without a dose, is the noise of the integrator there (1e-24 against 1e-4
    of the doses in hctz), which differs with every other step. The resolution
    of an array is the largest spread of the interpolants of its interpolated
    rows (`interpolation_envelope`) relative to the same scale.

    Args:
        old: the old dump.
        new: the new dump.
        keys: the arrays, (experiment, task, variable).

    Returns:
        The difference and the resolution of every array.
    """
    differences: dict[tuple[str, str, str], ArrayDifference] = {}
    batches: dict[tuple[object, ...], list[tuple[tuple[str, str, str], Columns]]] = {}
    #: the NaN at different positions, the scale, the difference and the
    #: spread of the interpolants of an array
    compared: dict[tuple[str, str, str], tuple[bool, float, float, float]] = {}
    for key in keys:
        found = _compared_columns(old, new, key)
        if not isinstance(found, Columns):
            differences[key] = ArrayDifference(found, 0.0)
            continue
        batch = (
            key[0],
            key[1],
            found.old_times.shape,
            found.old_times.tobytes(),
            found.new_times.shape,
            found.new_times.tobytes(),
        )
        batches.setdefault(batch, []).append((key, found))
    for entries in batches.values():
        stacked = Columns(
            *(
                np.hstack([getattr(f, field) for _, f in entries])
                for field in Columns._fields
            )
        )
        width = stacked.old.shape[1]
        scale, difference = np.zeros(width), np.zeros(width)
        spread = np.zeros(width)
        mismatch = np.zeros(width, dtype=bool)
        for match in point_matches(stacked.old_times, stacked.new_times):
            exact = (
                stacked.old[np.ix_(match.old, match.points)],
                stacked.new[np.ix_(match.new, match.points)],
                np.zeros((match.old.size, match.points.size)),
            )
            for x, y, w in (exact, *interpolated_pairs(stacked, match)):
                if not x.shape[0]:
                    continue
                points = match.points
                mismatch[points] |= np.any(np.isnan(x) != np.isnan(y), axis=0)
                x, y = np.nan_to_num(x), np.nan_to_num(y)
                scale[points] = np.maximum(scale[points], np.max(np.abs(x), axis=0))
                difference[points] = np.maximum(
                    difference[points], np.max(np.abs(x - y), axis=0)
                )
                spread[points] = np.maximum(
                    spread[points], np.max(np.nan_to_num(w), axis=0)
                )
        first = 0
        for key, found in entries:
            last = first + found.old.shape[1]
            compared[key] = (
                bool(mismatch[first:last].any()),
                float(np.max(scale[first:last], initial=0.0)),
                float(np.max(difference[first:last], initial=0.0)),
                float(np.max(spread[first:last], initial=0.0)),
            )
            first = last
    variable_scales: dict[tuple[str, str], float] = {}
    for (sid, _, variable), (_, s, _, _) in compared.items():
        variable_scales[sid, variable] = max(
            variable_scales.get((sid, variable), 0.0), s
        )
    for key, (nan_mismatch, s, d, w) in compared.items():
        s = max(s, SCALE_FLOOR * variable_scales[key[0], key[2]])
        if nan_mismatch:
            differences[key] = ArrayDifference(math.inf, 0.0)
        elif s > 1e-300:
            differences[key] = ArrayDifference(d / s, w / s)
        else:
            differences[key] = ArrayDifference(d, w)
    return differences


def _compared_columns(
    old: Dump, new: Dump, key: tuple[str, str, str]
) -> Columns | float:
    """Get the columns of an array with times, or its difference without them.

    Returns:
        The columns, see `relative_difference`, or the difference of an array
        which is not compared on its rows: a numeric array without a time
        relative to its scale, any other array exactly, other dimensions or no
        times an infinite difference.
    """
    (dims_a, a), (dims_b, b) = old["results"][key], new["results"][key]
    sid, task, variable = key
    if variable == "time":
        # the time of a grid is stored once, ("time",), the one of a ragged
        # result for every point: compare the output times of the task
        found = task_columns(old, new, sid, task)
    else:
        numeric = a.dtype.kind in "fiub" and b.dtype.kind in "fiub"
        timeless = time_axis(dims_a) is None and time_axis(dims_b) is None
        if numeric and timeless:
            if tuple(dims_a) != tuple(dims_b) or a.shape != b.shape:
                return math.inf
            return scaled_difference([(a, b)])
        if not numeric or time_axis(dims_a) is None or time_axis(dims_b) is None:
            same = a.shape == b.shape and np.array_equal(a, b)
            return 0.0 if same else math.inf
        found = columns(old, new, key)
    return math.inf if found is None else found


def scaled_difference(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    """Get max|new - old| / max|old| over pairs of old and new arrays.

    NaN at different positions is an infinite difference.
    """
    scale = difference = 0.0
    for x, y in pairs:
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return math.inf
        x, y = np.nan_to_num(x), np.nan_to_num(y)
        if x.size:
            scale = max(scale, float(np.max(np.abs(x))))
            difference = max(difference, float(np.max(np.abs(x - y))))
    return difference / scale if scale > 1e-300 else difference


def equal(a: object, b: object) -> bool:
    """Check that two values (arrays, lists, dicts) are equal, NaN equal to NaN."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype.kind != y.dtype.kind:
            return False
        return bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
    return a == b
