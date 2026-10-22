"""Result of a single timecourse simulation.

The simulator answers every `Simulation` with a `TimecourseResult`: the
array of the selections which roadrunner returns, with a row per time point and
a column per selection, and the names of the columns.
`sbmlsim.simulator.Simulator.run` places the results of the simulations of a
scan into a `ScanResult`.

It is deliberately not a data frame. A fit simulates its groups on every
evaluation of the residuals and a scan simulates every combination of its
dimensions, so building a frame per simulation, concatenating the frames and
looking their columns up took a third of the time of the residuals and half of
the time of a scan; the columns of a `TimecourseResult` are views into the
array roadrunner returned.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True, eq=False)
class TimecourseResult:
    """Values of the selections of a timecourse simulation.

    Attributes:
        columns: names of the columns, i.e., the timecourse selections.
        values: values with a row per time point and a column per selection.
    """

    columns: tuple[str, ...]
    values: np.ndarray
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the shape and index the columns by name.

        Raises:
            ValueError: if the values are not 2 dimensional or do not have one
                column per name.
        """
        if self.values.ndim != 2:
            raise ValueError(
                f"The values of a timecourse must be 2 dimensional, "
                f"but have the shape {self.values.shape}."
            )
        if self.values.shape[1] != len(self.columns):
            raise ValueError(
                f"The values have {self.values.shape[1]} columns, but "
                f"{len(self.columns)} columns are named: {self.columns}."
            )
        # a name which appears twice refers to its first column
        index: dict[str, int] = {}
        for k, column in enumerate(self.columns):
            index.setdefault(column, k)
        object.__setattr__(self, "_index", index)

    @classmethod
    def concatenate(cls, results: Sequence["TimecourseResult"]) -> "TimecourseResult":
        """Concatenate the results of consecutive timecourses.

        Args:
            results: results with the same columns, in the order of their time.

        Returns:
            The rows of all results.

        Raises:
            ValueError: if there are no results or their columns differ.
        """
        if not results:
            raise ValueError("No results to concatenate.")
        if len(results) == 1:
            return results[0]
        columns = results[0].columns
        for result in results[1:]:
            if result.columns != columns:
                raise ValueError(
                    f"Results with different columns cannot be concatenated: "
                    f"{columns} and {result.columns}."
                )
        return cls(
            columns=columns,
            values=np.concatenate([result.values for result in results], axis=0),
        )

    def __getitem__(self, key: str) -> np.ndarray:
        """Get the values of a column.

        Args:
            key: name of the column.

        Returns:
            The values of the column, a view into `values`.

        Raises:
            KeyError: if there is no column of that name.
        """
        try:
            k = self._index[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not a column of the result, the columns are "
                f"{self.columns}."
            ) from None
        return self.values[:, k]

    def __len__(self) -> int:
        """Get the number of time points."""
        return int(self.values.shape[0])

    @property
    def time(self) -> np.ndarray:
        """Get the time points, i.e., the column `time`."""
        return self["time"]


class GridWeights(NamedTuple):
    """The linear interpolation from a common time axis onto a grid."""

    lower: np.ndarray
    upper: np.ndarray
    weight: np.ndarray
    valid: np.ndarray


def grid_weights(
    time: np.ndarray,
    grid: np.ndarray,
    tolerance: float = 0.0,
    changes: np.ndarray | None = None,
    left: np.ndarray | None = None,
) -> GridWeights:
    """Get the indices and weights which interpolate `time` onto `grid` once.

    The output of a simulation has two time points at the time of a change
    after the start, the states before and after the change, and two at the
    time of an event of the model, the states before and after the event
    (`sbmlsim.model.event_rows`): a jump. Two time points of an event which are
    a little apart, as the exact output of `sbmlsim.simulator.executor` has
    them, are a jump as well. A chain of jumps starts at a jump and has the
    time points after it at most `tolerance` apart, e.g. a change and an
    event closer than the resolution of the time and the first steps of
    CVODE after the last jump. A time of the grid sees:

    - before a chain, the value interpolated toward the state before it,
      also within `tolerance` before it if its first jump is a change, and
      whether or not a step of the integrator is that close to it;
    - at a change, at an event and inside a chain, the state after the last
      jump of the chain, not a step of the integrator after it;
    - at most `tolerance` before a chain whose first jump is an event, the
      state after the chain: CVODE locates an event a little after its
      time, at the end of the interval in which it finds the root of the
      trigger.

    A time of the grid in `left` sees the state before the jump at it
    instead, the state before the chain, e.g. a data point of a fit which
    takes the state before a change; by default the first of two equal
    times of the grid, so a grid of the time points of a timecourse gives
    every row of it. The fit interpolates its data by the same rule.

    Args:
        time: the time points, sorted; the ones which are not finite are no
            time points.
        grid: the times of the result.
        tolerance: the resolution of the time of the simulation, see
            `sbmlsim.simulator.plan.time_tolerance`; `0` takes only a time
            point which appears twice as a jump.
        changes: the times of the changes after the start in the time of the
            time points, `None` for none.
        left: whether a time of the grid takes the state before a jump at
            it, a mask of the grid; `None` for the first of two equal times.

    Returns:
        The indices and weights.
    """
    grid = np.asarray(grid, dtype=float)
    finite = np.flatnonzero(np.isfinite(time))
    t = time[finite]
    if t.size == 0:
        zeros = np.zeros(grid.size, dtype=int)
        return GridWeights(zeros, zeros, np.zeros(grid.size), np.zeros(grid.size, bool))
    n = t.size
    valid = (grid >= t[0]) & (grid <= t[-1])
    if left is None:
        left = np.zeros(grid.size, dtype=bool)
        left[:-1] = grid[:-1] == grid[1:]
    changes = np.empty(0) if changes is None else np.asarray(changes, dtype=float)
    rows = np.arange(n)
    # the jumps: a time point which appears twice, its second point
    twice = np.zeros(n, dtype=bool)
    twice[1:] = t[1:] == t[:-1]
    # the chains: time points at most the tolerance apart, which start at
    # their first jump, the steps of the integrator before it are no part of
    # them; the first and the last point of the chain of every time point
    split = np.diff(t) > tolerance
    close = np.concatenate([[0], np.cumsum(split)])
    first_jump = np.full(close[-1] + 1, n)
    np.minimum.at(first_jump, close[twice], rows[twice] - 1)
    split |= rows[1:] == first_jump[close[1:]]
    chain = np.concatenate([[0], np.cumsum(split)])
    first = np.flatnonzero(np.concatenate([[True], split]))[chain]
    last = np.flatnonzero(np.concatenate([split, [True]]))[chain]
    # the state after the last jump of a chain, or after the chain of an
    # event a little apart
    last_jump = np.full(chain[-1] + 1, -1)
    np.maximum.at(last_jump, chain[twice], rows[twice])
    end = np.where(last_jump[chain] >= 0, last_jump[chain], last)

    # the last time point at or before every time of the grid
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, n - 1)
    upper = np.minimum(i + 1, n - 1)
    span = t[upper] - t[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(span > 0.0, (grid - t[i]) / span, 0.0)
    at = t[i] == grid
    own = at & (twice[i] | np.isin(t[i], changes))
    inside = ~own & (end[i] > i)
    following = upper
    ahead = (
        ~own
        & ~inside
        & (following > i)
        & (first[following] == following)
        & (last[following] > following)
        & (t[following] - grid <= tolerance)
        & ~np.isin(t[following], changes)
    )
    # at a time point: the state after the last jump of its chain, the time
    # point itself if it is after it or its chain has no jump
    at_own = np.where(last_jump[chain[i]] >= 0, np.maximum(end[i], i), i)
    to = np.where(own, at_own, np.where(inside, end[i], end[following]))
    after = ~left & (own | inside | ahead)
    # the state before the jump at a time of the grid: the first point of a
    # time which appears twice, or of the chain the time is inside
    before = left & ((at & twice[i]) | inside)
    to = np.where(
        before,
        np.where(at & twice[i], np.searchsorted(t, grid, side="left"), first[i]),
        to,
    )
    fixed = after | before
    lower = np.where(fixed, to, i)
    upper = np.where(fixed, to, upper)
    weight = np.where(fixed, 0.0, weight)
    return GridWeights(finite[lower], finite[upper], weight, valid)


def apply_weights(w: GridWeights, values: np.ndarray) -> np.ndarray:
    """Interpolate the last axis of values with the weights, `NaN` outside.

    Values without a time point, e.g. of a ragged scan whose points all
    failed, are `NaN` at every time of the grid.
    """
    if values.shape[-1] == 0:
        return np.full((*values.shape[:-1], w.weight.size), np.nan)
    lower, upper = values[..., w.lower], values[..., w.upper]
    with np.errstate(invalid="ignore"):
        out = lower * (1.0 - w.weight) + upper * w.weight
    # An exact endpoint needs only that endpoint: NaN or infinity in its
    # unused neighbor must not contaminate the value through multiplication by 0.
    np.copyto(out, lower, where=w.weight == 0.0)
    np.copyto(out, upper, where=w.weight == 1.0)
    return np.where(w.valid, out, np.nan)


def interpolate(
    time: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    tolerance: float = 0.0,
    changes: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a timecourse linearly onto a grid of times.

    The time points of a simulation increase, a time of a change after the
    start appears twice, with the states before and after the change, and so
    does an event of the model. A time of the grid before a change is
    interpolated toward the state before it and the time of a change sees
    the state after it, the first of two equal times of the grid the state
    before it, see `grid_weights`. The padding (`NaN`) and the steady state
    after the end (`inf`) are no time points of the interpolation.

    Args:
        time: the time points.
        values: the values, a row per time point, one or two dimensional.
        grid: the times of the result.
        tolerance: time points at most this apart are a jump, see
            `grid_weights`.
        changes: the times of the changes after the start, see
            `grid_weights`.

    Returns:
        The values at the times of the grid, a row per time; `NaN` outside of
        the finite time points and for a timecourse without any.
    """
    weights = grid_weights(
        np.asarray(time, dtype=float),
        np.asarray(grid, dtype=float),
        tolerance,
        changes,
    )
    return np.moveaxis(apply_weights(weights, np.moveaxis(values, 0, -1)), -1, 0)


def without_states_before(
    time: np.ndarray, values: np.ndarray, changes: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Leave out the state before every change after the start.

    The output of a simulation has the state before and the state after a
    change at its time; a timecourse of samples, e.g. of pkpdutils, has a
    time once, the state after the change. The first of the two rows at the
    time of a change is left out, the two rows of an event of the model stay.

    Args:
        time: the time points `(n, n_rows)`, padded with `NaN`.
        values: the timecourses, of the shape of `time`.
        changes: the times of the changes after the start of every
            timecourse, in the time of the time points.

    Returns:
        The time points and the timecourses without the states before the
        changes, padded with `NaN`, which is the padding of every time point
        which is not finite as well.
    """
    keep = np.isfinite(time)
    for r, at in enumerate(changes):
        if len(at) == 0:
            continue
        row = time[r]
        first = np.ones(row.size, dtype=bool)
        first[1:] = row[1:] != row[:-1]
        twice = np.zeros(row.size, dtype=bool)
        twice[:-1] = row[:-1] == row[1:]
        keep[r] &= ~(first & twice & np.isin(row, at))
    counts = keep.sum(axis=1)
    out_time = np.full((time.shape[0], int(counts.max(initial=0))), np.nan)
    out_values = np.full(out_time.shape, np.nan)
    for r in range(time.shape[0]):
        out_time[r, : counts[r]] = time[r, keep[r]]
        out_values[r, : counts[r]] = values[r, keep[r]]
    return out_time, out_values
