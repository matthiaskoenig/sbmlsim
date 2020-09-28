"""
Helpers for numerical comparison of simulation results between different simulators.
Allows to test semi-automatically for problems with the various models.

Used to benchmark the simulation results.
"""

import logging
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


def get_files_by_extension(base_path, extension=".json") -> List[str]:
    """Get all simulation definitions in the test directory.

    Simulation definitions are json files.
    """
    # get all files with extension in given path
    files = [f for f in base_path.glob("**/*") if f.is_file() and f.suffix == extension]
    keys = [f.name[:-5] for f in files]

    return dict(zip(keys, files))


class DataSetsComparison(object):
    """Comparing multiple simulation results.

    Only the subset of identical columns are compared. In the beginning a matching of column
    names is performed to find the subset of columns which can be compared.

    The simulations must contain a "time" column with identical time points.
    """

    eps = 1e-6  # tolerance for comparison
    eps_plot = 1e-9  # tolerance for plotting

    @timeit
    def __init__(self, dfs_dict, columns_filter=None, title: str = None):
        """Initialize the comparison.

        :param dfs_dict: data dictionary d[simulator_key] = df_result
        :param columns_filter: function which returns True if in Set or False if should be filtered.
        """
        self.title = title
        self.columns_filter = columns_filter

        # check that identical number of timepoints
        Nt = 0
        for label, df in dfs_dict.items():
            if Nt == 0:
                Nt = len(df)

            if len(df) != Nt:
                raise ValueError(
                    f"DataFrame have different length (number of rows): "
                    f"{len(df)} != {Nt} ({label})"
                )

        # check that time column exist in data frames
        for label, df in dfs_dict.items():
            if "time" not in df.columns:
                raise ValueError("'time' column must exist in data ({})".format(label))

        # get the subset of columns to compare
        columns, self.col_intersection, self.col_union = self._process_columns(dfs_dict)

        # filtered columns
        if columns_filter:
            columns = [col for col in columns if columns_filter(col)]
        self.columns = columns

        # get common subset of data
        self.dfs, self.labels = self._filter_dfs(dfs_dict, self.columns)
        if self.title is None:
            self.title = " | ".join(self.labels)

        # calculate difference
        self.diff, self.diff_abs, self.diff_rel = self.df_diff()

    @classmethod
    def _process_columns(cls, dataframes):
        """Get the intersection and union of columns.

        :param dataframes:
        :return:
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

        # set of columns from the individual dataframes
        col_union = None
        col_intersection = None
        for path, df in dataframes.items():
            # get all numeric columns
            num_df = df.select_dtypes(include=numerics)
            if len(num_df.columns) < len(df.columns):
                logger.warning(
                    f"Non-numeric columns in DataFrame: {set(df.columns)-set(num_df.columns)}"
                )

            cols = set(num_df.columns)

            if not col_union or not col_intersection:
                col_union = cols
                col_intersection = cols
            else:
                col_union = col_union.union(cols)
                col_intersection = col_intersection.intersection(cols)

        logger.info(f"Column Union #: {len(col_union)}")
        logger.info(f"Column Intersection #: {len(col_intersection)}")

        columns = list(col_intersection.copy())
        columns.remove("time")
        columns = ["time"] + sorted(columns)

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
            except KeyError:
                logger.error(
                    f"Some keys from '{columns}' do not exist in DataFrame columns "
                    f"'{df.columns}'"
                )
                raise ValueError
            dfs.append(df_filtered)
            labels.append(label)

        return dfs, labels

    def df_diff(self):
        """ DataFrame of all differences between the files."""
        # TODO: update to multiple comparison, i.e. between more then 2 simulators

        diff = self.dfs[0] - self.dfs[1]

        # absolute differences between all data frames
        diff_abs = diff.abs()

        # relative differences between all data frames
        diff_rel = 2 * diff_abs / (self.dfs[0].abs() + self.dfs[1].abs())
        diff_rel[diff_rel.isnull()] = 0.0

        return diff, diff_abs, diff_rel

    def is_equal(self):
        """ Check if DataFrames are identical within numerical tolerance."""
        return abs(self.diff.abs().max().max()) <= DataSetsComparison.eps

    def __str__(self):
        return f"{self.__class__.__name__} ({self.labels})"

    def __repr__(self):
        return f"{self.__class__.__name__} [{self.id}] ({self.labels})"

    @timeit
    def report_str(self):
        """

        :return:
        """
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

        # TODO: fixme
        diff_info = diff_info[diff_max >= DataSetsComparison.eps]
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            lines.append(
                str(diff_info.sort_values(by=["Delta_rel_max"], ascending=False))
            )

        lines.append("# Maximum initial column difference")
        lines.append(str(self.diff.iloc[0].abs().max()))

        lines.append("# Maximum element difference")
        lines.append(str(self.diff.abs().max().max()))

        lines.append(f"# Datasets are equal (diff <= eps={self.eps})")
        lines.append(str(self.is_equal()).upper())
        lines.append("-" * 80)
        if not (self.is_equal()):
            logging.warning("Datasets are not equal !")

        return "\n".join([str(item) for item in lines])

    @timeit
    def report(self):
        # print report
        print(self.report_str())

        # plot figure
        f = self.plot_diff()
        return f

    @timeit
    def plot_diff(self):
        """Plots lines for entries which are above epsilon treshold."""

        # FIXME: only plot the top differences, otherwise plotting takes
        # very long
        # filter data
        diff_abs = self.diff_abs.copy()
        diff_rel = self.diff_rel.copy()
        diff_max = diff_abs.max()
        column_index = diff_max >= DataSetsComparison.eps_plot
        # column_index = diff_max >= DataSetsComparison.eps

        # print(column_index)
        diff_abs = diff_abs.transpose()
        diff_abs = diff_abs[column_index]
        diff_abs = diff_abs.transpose()

        # plot all overview
        f1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        f1.subplots_adjust(wspace=0.3)

        for cid in diff_abs.columns:
            for ax in (ax1, ax2):
                ax.plot(diff_abs[cid], label=cid)

            for ax in (ax3, ax4):
                ax.plot(diff_rel[cid], label=cid)

        for ax in (ax1, ax2):
            ax.set_title(f"{self.title}")
            ax.set_ylabel(f"Absolute difference")

        for ax in (ax3, ax4):
            ax.set_ylabel(f"Relative difference")
            ax.set_xlabel("time index")

        for ax in (ax2, ax4):
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-10)

            if ax.get_ylim()[1] < 10 * DataSetsComparison.eps:
                ax.set_ylim(top=10 * DataSetsComparison.eps)

        for ax in (ax1, ax3):
            ax.set_ylim(bottom=0)

        for ax in (ax1, ax2, ax3, ax4):
            ax.axhline(DataSetsComparison.eps, color="black", linestyle="--")

            # ax.legend()
            # ax.set_xlim(right=ax.get_xlim()[1] * 2)

        return f1
