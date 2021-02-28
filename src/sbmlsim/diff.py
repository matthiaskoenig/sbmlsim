"""Helpers for numerical comparison of simulation results between different simulators.

Allows to test semi-automatically for problems with the various models.
Used to benchmark the simulation results.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


def get_files_by_extension(base_path: Path, extension: str = ".json") -> Dict[str, str]:
    """Get all files by given extension.

    Simulation definitions are json files.
    """
    # get all files with extension in given path
    files = [f for f in base_path.glob("**/*") if f.is_file() and f.suffix == extension]
    offset = len(extension)
    keys = [f.name[:-offset] for f in files]

    return dict(zip(keys, files))  # type: ignore


class DataSetsComparison(object):
    """Comparing multiple simulation results.

    Only the subset of identical columns are compared. In the beginning a matching of column
    names is performed to find the subset of columns which can be compared.

    The simulations must contain a "time" column with identical time points.
    """

    tol_abs = 1e-4  # absolute tolerance for comparison
    tol_rel = 1e-4  # relative tolerance for comparison
    eps_plot = 1e-5 * tol_abs  # tolerance for plotting

    @timeit
    def __init__(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        columns_filter=None,
        time_column: bool = True,
        title: str = None,
        selections: Dict[str, str] = None,
        factors: Dict[str, float] = None,
    ):
        """Initialize the comparison.

        :param dfs_dict: data dictionary d[simulator_key] = df_result
        :param columns_filter: function which returns True if in Set or False if should be filtered.
        :param time_column: flag to check for time column
        """
        self.columns_filter = columns_filter

        # check that identical number of rows (mostly timepoints)
        nrow = 0
        for label, df in dfs_dict.items():
            if nrow == 0:
                nrow = len(df)

            if len(df) != nrow:
                raise ValueError(
                    f"DataFrame have different length (number of rows): "
                    f"{len(df)} != {nrow} ({label})"
                )

        # check that time column exist in data frames
        if time_column:
            for label, df in dfs_dict.items():
                if "time" not in df.columns:
                    raise ValueError(f"'time' column must exist in data ({label})")

        # handle selections replacements
        pd.set_option("display.max_columns", None)
        if selections:
            if factors is None:
                factors = {}
            # use the first keys for comparison
            colnames = list(selections.values())[1]
            for key, sel_keys in selections.items():
                print("***", key, "***")
                df = dfs_dict[key]
                # get subset
                df_new = df[sel_keys]
                # apply factors
                fs = factors.get(key, [1.0] * len(sel_keys))
                for k, sel in enumerate(sel_keys):
                    print(f"scaling: '{sel}' * {fs[k]}")  # type: ignore
                    # df_new[sel] = fs[k] * df_new[sel]
                    df_new.loc[:, sel] *= fs[k]  # type: ignore

                # do renaming
                df_new = df_new.rename(columns=dict(zip(sel_keys, colnames)))
                # store updated df
                print(df_new.head())
                dfs_dict[key] = df_new

        # get the subset of columns to compare
        columns, self.col_intersection, self.col_union = self._process_columns(dfs_dict)

        # filtered columns
        if columns_filter:
            columns = [col for col in columns if columns_filter(col)]
        self.columns = columns
        logger.info(f"Comparing: {self.columns}")

        # get common subset of data
        self.dfs, self.labels = self._filter_dfs(dfs_dict, self.columns)

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
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

        # set of columns from the individual dataframes
        col_union = None
        col_intersection = None
        for _path, df in dataframes.items():
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
        """Dataframe of all differences between the files.

        https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/semantic/README.md
        Let the following variables be defined:

        * `abs_tol` stand for the absolute tolerance for a test case,
        * `rel_tol` stand for the relative tolerance for a test case,
        * `c_ij` stand for the expected correct value for row `i`, column `j`, of the result data set for the test case
        * `u_ij` stand for the corresponding value produced by a given software simulation system run by the user

        These absolute and relative tolerances are used in the following way:
        a data point `u_ij` is considered to be within tolerances
        if and only if the following expression is true:

        |c_ij - u_ij| <= (abs_tol + rel_tol * |c_ij|)

        """
        c = self.dfs[0]
        u = self.dfs[1]

        # difference
        diff = c - u

        # absolute differences between all data frames
        diff_abs = diff.abs()

        # relative differences between data frames
        diff_rel = 2 * diff_abs / (self.dfs[0].abs() + self.dfs[1].abs())
        diff_rel[diff_rel.isnull()] = 0.0

        # difference based on tolerance
        # |c_ij - u_ij| <= (abs_tol + rel_tol * |c_ij|)

        # > 0 if difference
        diff_tol = (c - u).abs() - (self.tol_abs + self.tol_rel * c.abs())

        # boolean matrix: True if difference, False if identical
        diff_tol_bool = diff_tol > 0

        return diff, diff_abs, diff_rel, diff_tol, diff_tol_bool

    def is_equal(self):
        """Check if DataFrames are identical within numerical tolerance."""
        return not self.diff_tol_bool.any(axis=None)

    def __str__(self) -> str:
        """Get string."""
        return f"{self.__class__.__name__} ({self.labels})"

    def __repr__(self):
        """Get representation."""
        return f"{self.__class__.__name__} [{self.id}] ({self.labels})"

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

        # TODO: fixme
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
            logging.warning("Datasets are not equal !")

        return "\n".join([str(item) for item in lines])

    @timeit
    def report(self):
        """Report."""
        # print report
        print(self.report_str())

        # plot figure
        f = self.plot_diff()
        return f

    @timeit
    def plot_diff(self):
        """Plot lines for entries which are above epsilon treshold."""

        import seaborn as sns

        # FIXME: only plot the top differences, otherwise plotting takes
        # very long
        # filter data
        diff_abs = self.diff_abs.copy()
        diff_rel = self.diff_rel.copy()
        diff_tol = self.diff_tol.copy()
        diff_max = diff_abs.max()
        column_index = diff_max >= DataSetsComparison.eps_plot
        # column_index = diff_max >= DataSetsComparison.eps

        # print(column_index)
        diff_abs = diff_abs.transpose()
        diff_abs = diff_abs[column_index]
        diff_abs = diff_abs.transpose()

        # plot all overview
        f1, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(20, 4.5))
        f1.subplots_adjust(wspace=0.35)
        f1.suptitle(self.title, fontsize=14, fontweight="bold")

        sns.heatmap(data=self.diff_tol_bool, cmap="Blues", vmin=0, vmax=1, ax=ax1)
        ax1.set_title(f"equal = {str(self.is_equal()).upper()}", fontweight="bold")
        ax1.set_ylabel("Tolerance difference", fontweight="bold")

        # sns.heatmap(data=self.diff_tol, center=0, ax=ax2)

        for cid in diff_abs.columns:
            ax2.plot(diff_tol[cid], label=cid)
            ax3.plot(diff_abs[cid], label=cid)
            ax4.plot(diff_rel[cid], label=cid)

        ax2.set_ylabel("Tolerance difference", fontweight="bold")
        ax3.set_ylabel("Absolute difference", fontweight="bold")
        ax4.set_ylabel("Relative difference", fontweight="bold")

        for ax in (ax3, ax4):
            ax.set_xlabel("time index", fontweight="bold")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-10)

            if ax.get_ylim()[1] < 10 * DataSetsComparison.tol_abs:
                ax.set_ylim(top=10 * DataSetsComparison.tol_abs)

        ax2.axhline(0.0, color="black", linestyle="--")
        for ax in (ax3, ax4):
            ax.axhline(DataSetsComparison.tol_abs, color="black", linestyle="--")

            # ax.legend()
            # ax.set_xlim(right=ax.get_xlim()[1] * 2)

        # ax3.imshow(self.diff_tol, cmap='Greys')

        return f1
