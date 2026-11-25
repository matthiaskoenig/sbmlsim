"""Helpers for numerical comparison of data.

Allows to tests semi-automatically for problems with the various models.
Used to benchmark the simulation results.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.console import console
from sbmlsim.plot.chart import Chart, ChartAxis
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)

#: the diverging color scale of the differences to the tolerances
DIFF_SCALE = "coolwarm"

#: the lower end of the logarithmic axes of the differences
LOG_BOTTOM = 1e-10

#: the most rows of the results whose tick labels the heatmap shows
MAX_HEATMAP_TICKS = 30


def within_tolerance(
    expected: Any,
    observed: Any,
    abs_tol: float,
    rel_tol: float,
) -> Any:
    """Check which values agree within an absolute and a relative tolerance.

    The criterion of the SBML Test Suite, see
    [the README of the semantic cases](https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/semantic/README.md):
    an observed value `u` agrees with the expected value `c` if

        |c - u| <= abs_tol + rel_tol * |c|

    The tolerance is relative to the expected value, not to the difference of
    the two, so a value which is expected to be zero is only allowed to differ
    by the absolute tolerance.

    Args:
        expected: the correct values, an array or a DataFrame.
        observed: the values which are checked, of the same shape.
        abs_tol: absolute tolerance.
        rel_tol: relative tolerance.

    Returns:
        Booleans of the shape of the input, `True` where both values are finite
        and agree.
    """
    # an undefined or infinite value never agrees, not even with itself;
    # subtracting infinities gives NaN rather than an error
    with np.errstate(invalid="ignore"):
        difference = np.abs(expected - observed)
        return (
            np.isfinite(expected)
            & np.isfinite(observed)
            & (difference <= (abs_tol + rel_tol * np.abs(expected)))
        )


def _largest(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Get the larger of two values, `NaN` where either of them is `NaN`."""
    return a.where((a >= b) | a.isna(), b)


def get_files_by_extension(
    base_path: Path, extension: str = ".json"
) -> dict[str, Path]:
    """Get all files by given extension.

    Simulation definitions are json files.

    Args:
        base_path: the directory which is searched, with its subdirectories.
        extension: the extension of the files.

    Returns:
        The path of every file by its name without the extension.
    """
    files = [f for f in base_path.glob("**/*") if f.is_file() and f.suffix == extension]
    offset = len(extension)
    return {f.name[:-offset]: f for f in files}


def to_dataframe(data: pd.DataFrame | xr.Dataset, label: str) -> pd.DataFrame:
    """Get the table of a result, one row per time point.

    A data frame is returned as it is. A dataset, e.g. `ScanResult.ds` of a
    simulation or of one point of a scan, is a column per variable and the
    time as a column; the dimension `_point` of the steps of the integrator,
    the labels of the dimensions and the changed values are no columns, and
    the padding of a point of a ragged scan, the rows without a time, is no
    row.

    Args:
        data: the result.
        label: the name of the result in the messages.

    Returns:
        The table.

    Raises:
        ValueError: if the dataset has a dimension of a scan.
    """
    if isinstance(data, pd.DataFrame):
        return data
    dims = [str(dim) for dim in data.dims if dim not in (TIME, POINT)]
    if dims:
        raise ValueError(
            f"The dataset '{label}' has the dimensions {dims} of a scan, compare "
            f"one point at a time, e.g. ds.isel({dims[0]}=0)."
        )
    df = data.reset_coords(drop=True).to_dataframe().reset_index()
    if POINT in data.dims and TIME in df.columns:
        df = df[df[TIME].notna()].reset_index(drop=True)
    return df.drop(columns=[POINT], errors="ignore")


class DataSetsComparison:
    """Comparing multiple simulation results.

    A result is a data frame or a dataset, e.g. `ScanResult.ds`, see
    `to_dataframe`. Only the subset of identical columns are compared. In the
    beginning a matching of column names is performed to find the subset of
    columns which can be compared.

    Rows are compared in their order, independent of their pandas index. The
    first result is the reference for every other result. With several
    candidates, the differences show the largest deviation at each value.
    The simulations must contain a "time" column unless `time_column=False`.
    """

    tol_abs = 1e-4  # absolute tolerance for comparison
    tol_rel = 1e-4  # relative tolerance for comparison
    eps_plot = 1e-5 * tol_abs  # tolerance for plotting

    @timeit
    def __init__(
        self,
        dfs_dict: Mapping[str, pd.DataFrame | xr.Dataset],
        columns_filter=None,
        time_column: bool = True,
        title: str | None = None,
        selections: Mapping[str, list[str]] | None = None,
        factors: Mapping[str, list[float]] | None = None,
    ):
        """Initialize the comparison.

        :param dfs_dict: data dictionary d[simulator_key] = result, a data
            frame or a dataset, e.g. `ScanResult.ds`, see `to_dataframe`
        :param columns_filter: function which returns True if in Set or False
            if should be filtered.
        :param time_column: flag to check for time column
        :param title: the title of the report, the labels by default
        :param selections: label -> the columns of the result which are
            compared, renamed to the columns of the second result
        :param factors: label -> the factor of every selected column
        """
        self.columns_filter = columns_filter
        frames: dict[str, pd.DataFrame] = {
            label: to_dataframe(data, label).reset_index(drop=True).copy()
            for label, data in dfs_dict.items()
        }
        if len(frames) < 2:
            raise ValueError("A comparison requires at least two results.")

        # check that identical number of rows (mostly timepoints)
        nrow = len(next(iter(frames.values())))
        for label, df in frames.items():
            if len(df) != nrow:
                raise ValueError(
                    f"DataFrame have different length (number of rows): "
                    f"{len(df)} != {nrow} ({label})"
                )
        if nrow == 0:
            raise ValueError("A comparison requires at least one row.")

        # check that time column exist in data frames
        if time_column:
            for label, df in frames.items():
                if TIME not in df.columns:
                    raise ValueError(f"'time' column must exist in data ({label})")

        # handle selections replacements
        if selections:
            if factors is None:
                factors = {}
            # the columns are named as in the second selection
            colnames = next(iter(selections.values()))
            if len(selections) > 1:
                colnames = list(selections.values())[1]
            for key, sel_keys in selections.items():
                console.log("***", key, "***")
                df = frames[key]
                # get subset
                df_new = df[sel_keys].copy()
                # apply factors
                fs = factors.get(key, [1.0] * len(sel_keys))
                if len(fs) != len(sel_keys) or len(sel_keys) != len(colnames):
                    raise ValueError(
                        f"Selections and factors of '{key}' must have "
                        f"{len(colnames)} entries."
                    )
                for k, sel in enumerate(sel_keys):
                    console.log(f"scaling: '{sel}' * {fs[k]}")
                    df_new[sel] = df_new[sel] * float(fs[k])

                # do renaming
                df_new = df_new.rename(
                    columns=dict(zip(sel_keys, colnames, strict=True))
                )
                # store updated df
                console.log(df_new.head())
                frames[key] = df_new

        # get the subset of columns to compare
        columns, self.col_intersection, self.col_union = self._process_columns(frames)
        if time_column and TIME not in self.col_intersection:
            raise ValueError(
                "The 'time' column must be numeric and selected in every result."
            )

        # filtered columns
        if columns_filter:
            columns = [col for col in columns if columns_filter(col)]
        if not columns:
            raise ValueError("The results have no numeric columns to compare.")
        self.columns = columns
        logger.info("Comparing: %s", self.columns)

        # get common subset of data
        self.dfs, self.labels = self._filter_dfs(frames, self.columns)

        # set title
        self.title = title if title else " | ".join(self.labels)

        # calculate difference
        (
            self.diff,
            self.diff_abs,
            self.diff_rel,
            self.diff_tol,
            self.diff_tol_bool,
        ) = self.df_diff()

    @classmethod
    def _process_columns(cls, dataframes):
        """Get the intersection and union of columns.

        :param dataframes:
        :return:
        """
        # set of columns from the individual dataframes
        col_union = None
        col_intersection = None
        for _path, df in dataframes.items():
            # get all numeric columns
            num_df = df.select_dtypes(
                include="number", exclude=["complex", "timedelta"]
            )
            if len(num_df.columns) < len(df.columns):
                logger.warning(
                    "Non-numeric columns in DataFrame: %s",
                    set(df.columns) - set(num_df.columns),
                )

            cols = set(num_df.columns)

            if col_union is None or col_intersection is None:
                col_union = cols
                col_intersection = cols
            else:
                col_union = col_union.union(cols)
                col_intersection = col_intersection.intersection(cols)
        if col_union is None or col_intersection is None:
            raise ValueError("No DataFrames to compare.")

        logger.info("Column Union #: %s", len(col_union))
        logger.info("Column Intersection #: %s", len(col_intersection))

        columns = list(col_intersection.copy())
        if TIME in columns:
            columns.remove(TIME)
            columns = [TIME, *sorted(columns)]
        else:
            columns = sorted(columns)

        return columns, col_intersection, col_union

    @classmethod
    def _filter_dfs(cls, dataframes, columns):
        """Filter the dataframes using the column ids occurring in all datasets.

        The common set of columns is used for comparison.

        :param dataframes:
        :param columns:
        :return: List[pd.DataFrame], List[str], list of dataframes and simulator labels.
        """
        dfs = []
        labels = []
        for label, df in dataframes.items():
            try:
                df_filtered = df[columns]
            except KeyError as err:
                logger.error(
                    "Some keys from '%s' do not exist in DataFrame columns '%s'",
                    columns,
                    df.columns,
                )
                raise ValueError from err
            dfs.append(df_filtered)
            labels.append(label)

        return dfs, labels

    def df_diff(self):
        """Dataframe of all differences between the files.

        https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/semantic/README.md
        Let the following variables be defined:

        * `abs_tol` stand for the absolute tolerance for a tests case,
        * `rel_tol` stand for the relative tolerance for a tests case,
        * `c_ij` stand for the expected correct value for row `i`, column
          `j`, of the result data set for the tests case
        * `u_ij` stand for the corresponding value produced by a given
          software simulation system run by the user

        These absolute and relative tolerances are used in the following way:
        a data point `u_ij` is considered to be within tolerances
        if and only if the following expression is true:

        |c_ij - u_ij| <= (abs_tol + rel_tol * |c_ij|)

        """
        c = self.dfs[0]
        diff, diff_abs, diff_rel, diff_tol, diff_tol_bool = self._pair_diff(
            c, self.dfs[1]
        )
        for candidate in self.dfs[2:]:
            other, other_abs, other_rel, other_tol, other_bool = self._pair_diff(
                c, candidate
            )
            # An invalid value remains invalid even when another candidate
            # matches it. Each diagnostic retains its largest deviation.
            replace = other_abs.isna() | (diff_abs.notna() & (other_abs > diff_abs))
            diff = diff.where(~replace, other)
            diff_abs = _largest(diff_abs, other_abs)
            diff_rel = _largest(diff_rel, other_rel)
            diff_tol = _largest(diff_tol, other_tol)
            diff_tol_bool = diff_tol_bool | other_bool

        return diff, diff_abs, diff_rel, diff_tol, diff_tol_bool

    def _pair_diff(
        self, c: pd.DataFrame, u: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compare one candidate with the reference in row order.

        Args:
            c: the reference.
            u: the candidate, with the rows and columns of the reference.

        Returns:
            The difference, its absolute value, the relative difference, the
            difference beyond the tolerance (positive where the values differ)
            and the booleans which are `True` where the values differ.
        """
        # Integer subtraction can wrap, especially for unsigned data.
        c, u = c.astype(float), u.astype(float)
        # difference
        diff = c - u

        # absolute differences between all data frames
        diff_abs = diff.abs()

        # relative differences between data frames
        diff_rel = 2 * diff_abs / (c.abs() + u.abs())
        diff_rel = diff_rel.mask((c == 0) & (u == 0), 0.0)

        # difference based on tolerance
        # |c_ij - u_ij| <= (abs_tol + rel_tol * |c_ij|)

        # > 0 if difference
        diff_tol = (c - u).abs() - (self.tol_abs + self.tol_rel * c.abs())

        # boolean matrix: True if different, False if identical
        diff_tol_bool = ~within_tolerance(
            expected=c, observed=u, abs_tol=self.tol_abs, rel_tol=self.tol_rel
        )

        return diff, diff_abs, diff_rel, diff_tol, diff_tol_bool

    def is_equal(self):
        """Check if DataFrames are identical within numerical tolerance."""
        return not self.diff_tol_bool.any(axis=None)

    def __str__(self) -> str:
        """Get string."""
        return f"{self.__class__.__name__} ({self.labels})"

    def __repr__(self):
        """Get representation."""
        return f"{self.__class__.__name__} ({self.labels})"

    @timeit
    def report_str(self) -> str:
        """Get report as string."""
        lines = [
            "-" * 80,
            str(self),
            str(self.title),
            "-" * 80,
            "# Elements (Nt, Nx)",
            str(self.diff.shape),
            "# Maximum column difference (above eps)",
        ]
        diff_max = self.diff_abs.max()
        diff_0 = self.diff_abs.iloc[0]
        diff_rel_max = self.diff_rel.max()
        diff_rel_0 = self.diff_rel.iloc[0]

        diff_info = pd.concat([diff_0, diff_rel_0, diff_max, diff_rel_max], axis=1)
        diff_info.columns = ["Delta_abs_0", "Delta_rel_0", "Delta_max", "Delta_rel_max"]

        diff_info = diff_info[diff_max >= DataSetsComparison.tol_abs]
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            lines.append(
                str(diff_info.sort_values(by=["Delta_rel_max"], ascending=False))
            )

        lines.append("# Maximum initial column difference")
        lines.append(str(self.diff.iloc[0].abs().max()))

        lines.append("# Maximum element difference")
        lines.append(str(self.diff.abs().max().max()))

        lines.append(
            "# Datasets are equal (|c_ij - u_ij| <= (tol_abs + tol_rel * |c_ij|))"
        )
        lines.append(str(self.is_equal()).upper())
        lines.append("-" * 80)
        if not self.is_equal():
            logger.warning("Datasets are not equal !")

        return "\n".join([str(item) for item in lines])

    @timeit
    def report(self) -> Chart:
        """Log the report and draw the figure of the differences, see `plot_diff`."""
        console.log(self.report_str())
        return self.plot_diff()

    @timeit
    def plot_diff(self) -> Chart:
        """Draw the differences of the results.

        Four panels: of the columns whose absolute difference reaches
        `eps_plot`, the differences to the tolerances over the rows of the
        results, the differences above the tolerances as a heatmap of the
        columns over the rows, on a diverging scale centered at 0 (a value
        within the tolerances has no color) with its colorbar at the right
        edge, and the absolute and relative differences over the rows on a
        logarithmic axis with the absolute tolerance.

        Returns:
            The chart, drawn by the renderer of sbmlsim-core.
        """
        diff_abs = self.diff_abs.copy()
        diff_rel = self.diff_rel.copy()
        diff_tol = self.diff_tol.copy()
        diff_tol[diff_tol < 0] = np.nan  # remove the identical tolerance
        plotted = [
            cid
            for cid, top in diff_abs.max().items()
            if top >= DataSetsComparison.eps_plot
        ]
        labelled = len(plotted) > 0

        # the columns with a difference above the tolerances
        df_diff_bool: pd.DataFrame = self.diff_tol_bool.copy()
        col_drops = [c for c in df_diff_bool.columns if df_diff_bool[c].sum() == 0]
        df_diff: pd.DataFrame = self.diff_tol.copy().drop(labels=col_drops, axis=1)
        df_diff[df_diff < 0] = np.nan

        chart = Chart(
            864,
            864,
            title=self.title,
            rows=2,
            columns=2,
            fonts={"legend": 8},
        )
        title = f"equal = {str(self.is_equal()).upper()}"
        if df_diff.empty:
            # every value is within the tolerances, there is nothing to map
            empty = ChartAxis(min=0.0, max=1.0, ticks_visible=False)
            heat = chart.plot(0, 1, title=title, x=empty, y=empty, legend=False)
            heat.text(0.5, 0.5, "every value within the tolerances", anchor="middle")
        else:
            # the rows of a long result are too many for their tick labels
            heat = chart.plot(
                0,
                1,
                title=title,
                x=ChartAxis(
                    label="time index",
                    ticks_visible=len(df_diff.index) <= MAX_HEATMAP_TICKS,
                ),
                legend=False,
            )
            values = np.abs(df_diff.to_numpy(dtype=float))
            finite = values[np.isfinite(values)]
            vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
            heat.heatmap(
                [str(c) for c in df_diff.columns],
                [str(i) for i in df_diff.index],
                df_diff.T.to_numpy(dtype=float),
                scale=DIFF_SCALE,
                low=-vmax,
                high=vmax,
                center=0.0,
            )
            heat.colorbar(
                DIFF_SCALE, -vmax, vmax, label="Tolerance difference", center=0.0
            )
        index = ChartAxis(label="time index")
        tolerance = chart.plot(
            0,
            0,
            x=index,
            y=ChartAxis(label="Tolerance difference"),
            legend=labelled,
        )
        bottom = DataSetsComparison.tol_abs
        # the axes end at the decade above the differences and the tolerance
        largest = [
            float(np.nanmax(df[plotted].to_numpy(dtype=float), initial=0.0))
            for df in (diff_abs, diff_rel)
        ]
        tops = [10.0 ** np.ceil(np.log10(max(10 * bottom, v))) for v in largest]
        logs = [
            chart.plot(
                1,
                column,
                x=index,
                y=ChartAxis(label=label, scale="log", min=LOG_BOTTOM, max=top),
                legend=labelled,
            )
            for column, (label, top) in enumerate(
                zip(["Absolute difference", "Relative difference"], tops, strict=True)
            )
        ]
        x = np.asarray(diff_abs.index, dtype=float)
        for cid in plotted:
            tolerance.curve(x, diff_tol[cid].to_numpy(dtype=float), label=str(cid))
            logs[0].curve(x, diff_abs[cid].to_numpy(dtype=float), label=str(cid))
            logs[1].curve(x, diff_rel[cid].to_numpy(dtype=float), label=str(cid))
        tolerance.rule(0.0, line="dashed", width=1.0, color="#000000")
        for plot in logs:
            plot.rule(bottom, line="dashed", width=1.0, color="#000000")
        return chart
