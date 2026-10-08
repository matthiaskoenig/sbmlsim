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


def interpolate(time: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a timecourse linearly onto a grid of times.

    The time points of a simulation increase and a time of a change appears
    once, with the state after the change, so the value at the time of a
    change is the value after it. The padding (`NaN`) and the steady state
    after the end (`inf`) are no time points of the interpolation.

    Args:
        time: the time points.
        values: the values, a row per time point, one or two dimensional.
        grid: the times of the result.

    Returns:
        The values at the times of the grid, a row per time; `NaN` outside of
        the finite time points and for a timecourse without any.
    """
    grid = np.asarray(grid, dtype=float)
    mask = np.isfinite(time)
    out = np.full((grid.size, *values.shape[1:]), np.nan)
    if not mask.any():
        return out
    t, v = time[mask], values[mask]
    if v.ndim == 1:
        return np.interp(grid, t, v, left=np.nan, right=np.nan)
    for j in range(v.shape[1]):
        out[:, j] = np.interp(grid, t, v[:, j], left=np.nan, right=np.nan)
    return out
