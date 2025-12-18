"""Plotting functionality for sensitivity analysis."""
import xarray as xr


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def heatmap(da: xr.DataArray, cutoff: float=0.01, annotate_values=True, transpose: bool=False):
    """Creates heatmap of model sensitivity"""

    def calculate_mask(df, cutoff=0.01):
        """Calculates a boolean mask DataFrame for the heatmap based on cutoff."""
        mask = np.empty(shape=df.shape, dtype="bool")
        for index, value in np.ndenumerate(df):
            if np.abs(value) < cutoff:
                mask[index] = True
            else:
                mask[index] = False
        return pd.DataFrame(data=mask, columns=df.COLUMNS, index=df.index)

    def calculate_subset(df, cutoff=0.01):
        """Calculates subset of data frame consisting of rows where at least
        one value is above cutoff."""
        return df[(df.abs() >= cutoff).any(axis=1)]



    # filter rows
    # X.drop(pk_exclude, axis=1, inplace=True)

    # if cutoff > 0:
    # X_subset = calculate_subset(X, cutoff=cutoff)
    # X_subset_mask = calculate_mask(X_subset, cutoff)
    da_subset = da

    # yticklabels = ["{}".format(pid) for pid in X_subset.index]
    # xticklabels = ["{}".format(pnames[pid]["label"]) for pid in X_subset.COLUMNS]

    xticklabels = da.coords[da.dims[1]]
    yticklabels = da.coords[da.dims[0]]

    # plot heatmap
    ax = sns.clustermap(
        da_subset,
        center=0,
        # vmin=-0.2,
        # vmax=0.2,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="seismic",
        cbar_pos=(0.05, 0.25, 0.03, 0.4),
        annot=annotate_values,
        fmt="1.2f",
        annot_kws={"size": 13},
        # mask=X_subset_mask,
        col_cluster=False,
        method="single",
        figsize=(20, 20),
    )
    plt.setp(
        ax.ax_heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        size=20,
    )
    plt.setp(ax.ax_heatmap.get_yticklabels(), size=20)
    ax.ax_cbar.tick_params(labelsize=20)
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_col_dendrogram.set_visible(False)

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

    plt.show()
