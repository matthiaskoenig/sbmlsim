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
    time: np.ndarray, grid: np.ndarray, tolerance: float = 0.0
) -> GridWeights:
    """Get the indices and weights which interpolate `time` onto `grid` once.

    Two time points at most `tolerance` apart are a jump: an event of the
    model, whose states before and after it the steps of the integrator have
    at almost one time, or a duplicated final time, the state after a change
    at the end. A time of the grid at a jump sees the state after it: a time
    inside it, at one of its points or at most `tolerance` before it, since
    roadrunner locates an event only to the tolerance of its roots, on
    either side of its time. The fit interpolates its data by the same rule,
    see `sbmlsim.fit.optimization._interpolate_after_jumps`.

    Args:
        time: the time points, sorted; the ones which are not finite are no
            time points.
        grid: the times of the result.
        tolerance: the resolution of the time of the simulation, see
            `sbmlsim.simulator.plan.time_tolerance`; `0` takes only a time
            point which appears twice as a jump.

    Returns:
        The indices and weights.
    """
    finite = np.flatnonzero(np.isfinite(time))
    t = time[finite]
    if t.size == 0:
        zeros = np.zeros(grid.size, dtype=int)
        return GridWeights(zeros, zeros, np.zeros(grid.size), np.zeros(grid.size, bool))
    valid = (grid >= t[0]) & (grid <= t[-1])
    low = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, max(t.size - 2, 0))
    up = np.minimum(low + 1, t.size - 1)
    span = t[up] - t[low]
    jump = (span <= tolerance) & (up > low)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(jump, 1.0, (grid - t[low]) / np.where(jump, 1.0, span))
    weight = np.where(up > low, weight, 0.0)
    # a time at most the tolerance before the first point of a jump
    after = np.minimum(up + 1, t.size - 1)
    ahead = (
        ~jump
        & (after > up)
        & (t[after] - t[up] <= tolerance)
        & (t[up] - grid <= tolerance)
    )
    low = np.where(ahead, up, low)
    up = np.where(ahead, after, up)
    weight = np.where(ahead, 1.0, weight)
    return GridWeights(finite[low], finite[up], weight, valid)


def apply_weights(w: GridWeights, values: np.ndarray) -> np.ndarray:
    """Interpolate the last axis of values with the weights, `NaN` outside.

    Values without a time point, e.g. of a ragged scan whose points all
    failed, are `NaN` at every time of the grid.
    """
    if values.shape[-1] == 0:
        return np.full((*values.shape[:-1], w.weight.size), np.nan)
    out = values[..., w.lower] * (1.0 - w.weight) + values[..., w.upper] * w.weight
    return np.where(w.valid, out, np.nan)


def interpolate(
    time: np.ndarray, values: np.ndarray, grid: np.ndarray, tolerance: float = 0.0
) -> np.ndarray:
    """Interpolate a timecourse linearly onto a grid of times.

    The time points of a simulation increase and a time of a change appears
    once, with the state after the change, so the value at the time of a
    change is the value after it, also at a duplicated time, and so is the
    value at an event of the model, whose states before and after it are
    closer than the `tolerance` (`grid_weights`). The padding (`NaN`) and the
    steady state after the end (`inf`) are no time points of the
    interpolation.

    Args:
        time: the time points.
        values: the values, a row per time point, one or two dimensional.
        grid: the times of the result.
        tolerance: two time points at most this apart are a jump, see
            `grid_weights`.

    Returns:
        The values at the times of the grid, a row per time; `NaN` outside of
        the finite time points and for a timecourse without any.
    """
    weights = grid_weights(
        np.asarray(time, dtype=float), np.asarray(grid, dtype=float), tolerance
    )
    return np.moveaxis(apply_weights(weights, np.moveaxis(values, 0, -1)), -1, 0)
