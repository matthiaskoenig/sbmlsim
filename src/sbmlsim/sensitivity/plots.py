"""Plotting functionality for sensitivity analysis."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pymetadata.console import console


def heatmap(
    df: pd.DataFrame,
    parameter_labels: Optional[dict[str, str]] = None,
    output_labels: Optional[dict[str, str]] = None,
    cutoff: float = 0.1,
    annotate_values=True,
    cluster_rows: bool = True,  # cluster parameters
    cluster_cols: bool = False,  # cluster outputs
    title: Optional[str] = None,
    cmap: str = "seismic",
    vcenter: float = 0.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    fig_path: Optional[Path] = None,
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
    else:
        df_subset = df
    df_subset_mask = calculate_mask(df_subset, cutoff)

    # outputs
    xticklabels = [qid for qid in df_subset.columns]
    if output_labels:
        console.print(output_labels)
        xticklabels = [output_labels[qid] for qid in xticklabels]

    # parameters
    yticklabels = [pid for pid in df_subset.index]
    if parameter_labels:
        yticklabels = [parameter_labels[pid] for pid in yticklabels]

    n_outputs = df_subset.shape[1]
    n_parameters = df_subset.shape[0]
    # (width, height)
    figsize = (15, int(n_parameters / n_outputs * 15))

    # plot heatmap
    cg = sns.clustermap(
        df_subset,
        center=vcenter,
        vmin=vmin,
        vmax=vmax,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        cbar_pos=(0.0, 0.4, 0.03, 0.2),  # (left, bottom, width, height),
        cbar_kws={
            "orientation": "vertical",
            # "label": "sensitivity"
        },
        annot=annotate_values,
        fmt="1.2f",
        annot_kws={"size": 11},
        mask=df_subset_mask,
        col_cluster=cluster_cols,
        row_cluster=cluster_rows,
        method="single",
        figsize=figsize,
    )
    plt.setp(
        cg.ax_heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        size=20,
    )
    label_fontsize = 15
    plt.setp(cg.ax_heatmap.get_yticklabels(), size=label_fontsize, weight="bold")
    plt.setp(cg.ax_heatmap.get_xticklabels(), size=label_fontsize, weight="bold")
    cg.ax_cbar.tick_params(labelsize=label_fontsize)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    if title:
        plt.suptitle(title, fontsize=40, fontweight="bold")

    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_S1_ST_indices(
    sa,  # SensitivityAnalysis,
    fig_path: Path,
):
    """Barplots for the S1 and ST indices."""
    parameter_labels: dict[str, str] = {p.uid: p.uid for p in sa.parameters}
    output_labels: dict[str, str] = {q.uid: q.name for q in sa.outputs}

    for group in sa.groups:
        gid = group.uid
        ymax = sa.sensitivity[gid]["ST"].max(dim=None)
        ymin = sa.sensitivity[gid]["S1"].min(dim=None)

        for ko, output in enumerate(sa.outputs):
            f_path = fig_path.parent / f"{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"

            S1 = sa.sensitivity[gid]["S1"][:, ko]
            ST = sa.sensitivity[gid]["ST"][:, ko]
            S1_conf = sa.sensitivity[gid]["S1_conf"][:, ko]
            ST_conf = sa.sensitivity[gid]["ST_conf"][:, ko]
            S1_ST_barplot(
                S1=S1,
                ST=ST,
                S1_conf=S1_conf,
                ST_conf=ST_conf,
                title=f"{output_labels[output.uid]} ({group.name})",
                fig_path=f_path,
                parameter_labels=parameter_labels,
                ymax=np.max([1.05, ymax]),
                ymin=np.min([-0.05, ymin]),
            )


def S1_ST_barplot(
    S1, ST, S1_conf, ST_conf,
    parameter_labels: dict[str, str],
    fig_path: Optional[Path] = None,
    title: Optional[str] = None,
    ymax: float = 1.1,
    ymin: float = -0.1,
):
    # width
    figsize = (15, 3)
    label_fontsize = 15

    categories: list[str] = list(parameter_labels.values())
    f, ax = plt.subplots(figsize=figsize)

    ax.bar(categories, ST, label='ST',
           color="black",
           alpha=1.0,
           edgecolor="black",
           yerr=ST_conf, capsize=5
           )

    ax.bar(categories, S1, label='S1', color="tab:blue",
           edgecolor="black", yerr=S1_conf, capsize=5)

    # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel('Sensitivity', fontsize=label_fontsize, fontweight="bold")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.grid(True, axis="y")
    ax.tick_params(axis='x', labelrotation=90)
    # ax.tick_params(axis='x', labelweight='bold')
    ax.legend()

    if title:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
