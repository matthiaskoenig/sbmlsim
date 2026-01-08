"""Plotting functionality for sensitivity analysis.

FIXME: use patchcollection
https://stackoverflow.com/questions/59381273/heatmap-with-circles-indicating-size-of-population
"""
import xarray as xr
from typing import Optional

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sbmlutils.console import console


def heatmap(
    df: pd.DataFrame,
    parameter_labels: Optional[dict[str, str]] = None,
    output_labels: Optional[dict[str, str]] = None,
    cutoff: float=0.01,
    annotate_values=True,
    cluster_rows: bool = True, # cluster parameters
    cluster_cols: bool = False, # cluster outputs
    transpose: bool=False,
    title: Optional[str] = None,
):
    """Creates heatmap of model sensitivity"""

    def calculate_mask(df, cutoff=0.01):
        """Calculates a boolean mask DataFrame for the heatmap based on cutoff."""
        mask = np.empty(shape=df.shape, dtype="bool")
        for index, value in np.ndenumerate(df):
            if np.abs(value) < cutoff:
                mask[index] = True
            else:
                mask[index] = False
        return pd.DataFrame(data=mask, columns=df.columns, index=df.index)

    def calculate_subset(df, cutoff=0.01) -> pd.DataFrame:
        """Calculates subset of data frame consisting of rows where at least
        one value is above cutoff."""
        return df[(df.abs() >= cutoff).any(axis=1)]


    # filter rows
    # X.drop(pk_exclude, axis=1, inplace=True)

    if cutoff > 0:
        df_subset = calculate_subset(df, cutoff=cutoff)
        df_subset_mask = calculate_mask(df_subset, cutoff)

    # outputs
    xticklabels = [qid for qid in df_subset.columns]
    if output_labels:
        console.print(output_labels)
        xticklabels = [output_labels[qid] for qid in xticklabels]

    # parameters
    yticklabels = [pid for pid in df_subset.index]
    if parameter_labels:
        yticklabels = [f"{pid}: {parameter_labels[pid]}" for pid in yticklabels]

    n_outputs = df_subset.shape[1]
    n_parameters = df_subset.shape[0]
    figsize = (int(n_outputs/n_parameters*30), 15)

    colorbar_range = 2.0

    # plot heatmap
    cg = sns.clustermap(
        df_subset,
        center=0,
        vmin=-colorbar_range,
        vmax=colorbar_range,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="seismic",
        # cbar_pos=(0.0, 0.0, 0.6, 0.05), #  (left, bottom, width, height),
        cbar_pos=(0.0, 0.4, 0.03, 0.2),  # (left, bottom, width, height),
        cbar_kws={
            "orientation": "vertical",
            # "label": "sensitivity"
        },
        annot=annotate_values,
        fmt="1.2f",
        annot_kws={"size": 11},
        mask=df_subset_mask,
        col_cluster=False,
        row_cluster=True,
        method="single",
        figsize=figsize,
    )
    plt.setp(
        cg.ax_heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        size=20,
    )
    label_fontsize=10
    plt.setp(cg.ax_heatmap.get_yticklabels(), size=label_fontsize)
    plt.setp(cg.ax_heatmap.get_xticklabels(), size=label_fontsize)
    cg.ax_cbar.tick_params(labelsize=label_fontsize)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    if title:
        plt.suptitle(title)

    # for label in cg.ax_heatmap.get_xticklabels():
    #     label.set_bbox(dict(facecolor='tab:blue', edgecolor='black', alpha=0.8))
    #
    # for label in cg.ax_heatmap.get_yticklabels():
    #     label.set_bbox(dict(facecolor='tab:orange', edgecolor='black', alpha=0.8))

    # create custom legend containing yticklabels and their description
    # handles = [t.get_text() for t in ax.ax_heatmap.get_yticklabels()]
    # labels = [pnames[pid]["label"] for pid in handles]
    #
    # # FIXME: update after defining labels
    # idx = [pnames[pid]["idx"] for pid in handles]
    # # idx = [k for k, pid in enumerate(handles)]
    #
    # labels = [label for _, label in sorted(zip(idx, labels))]
    # handles = [f"{handle}:" for _, handle in sorted(zip(idx, handles))]
    # handles = [handle.replace("_", "\_") for handle in handles]

    # mid = int(np.ceil(len(handles) / 2))
    # legend1 = plt.legend(
    #     handles[:mid],
    #     labels[:mid],
    #     handler_map={str: LegendTitle({"fontsize": 16})},
    #     fontsize=16,
    #     frameon=False,
    #     bbox_to_anchor=(1.2, -0.6),
    #     loc="upper left",
    #     handlelength=14,
    # )
    # legend2 = plt.legend(
    #     handles[mid:],
    #     labels[mid:],
    #     handler_map={str: LegendTitle({"fontsize": 16})},
    #     fontsize=16,
    #     frameon=False,
    #     bbox_to_anchor=(13, -0.6),
    #     loc="upper left",
    #     handlelength=19,
    # )
    # plt.gca().add_artist(legend1)

    # plt.savefig(
    #     results_dir / "parameter.sensitivity_cluster.png", dpi=300, bbox_inches="tight"
    # )
    # plt.savefig(results_dir / "parameter.sensitivity_cluster.svg", bbox_inches="tight")

    # plt.show()
