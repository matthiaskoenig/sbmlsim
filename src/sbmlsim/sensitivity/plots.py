"""Plotting functionality for sensitivity analysis."""

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult


def _heatmap(
    df: pd.DataFrame,
    parameter_labels: dict[str, str] | None = None,
    output_labels: dict[str, str] | None = None,
    cutoff: float | None = 0.1,
    annotate_values=True,
    cluster_rows: bool = True,  # cluster parameters
    cluster_cols: bool = False,  # cluster outputs
    title: str | None = None,
    cmap: str = "seismic",
    vcenter: float = 0.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    fig_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Creates heatmap of model sensitivity.

    The figure is saved to `fig_path` at the resolution `dpi` if one is given and
    is returned. It is closed, i.e., it is not held by pyplot and no window is
    opened; a caller which wants to display it does so itself.
    """

    def calculate_mask(df, cutoff: float | None = 0.01) -> pd.DataFrame:
        """Calculates a boolean mask DataFrame for the heatmap based on cutoff.

        The masked values are removed.
        """
        mask = np.empty(shape=df.shape, dtype="bool")
        for index, value in np.ndenumerate(df):
            if cutoff is not None:
                if np.abs(value) < cutoff:
                    mask[index] = True
                else:
                    mask[index] = False
            else:
                mask[index] = False
        return pd.DataFrame(data=mask, columns=df.columns, index=df.index)

    def calculate_subset(df, cutoff=0.01) -> pd.DataFrame:
        """Calculate the rows of the data frame with at least one value above the cutoff."""
        return df[(df.abs() >= cutoff).any(axis=1)]

    # filter rows
    # X.drop(pk_exclude, axis=1, inplace=True)

    df_subset = calculate_subset(df, cutoff=cutoff) if cutoff and cutoff > 0 else df
    df_subset_mask = calculate_mask(df_subset, cutoff)

    # outputs
    xticklabels = list(df_subset.columns)
    if output_labels:
        xticklabels = [output_labels[qid] for qid in xticklabels]

    # parameters
    yticklabels = list(df_subset.index)
    if parameter_labels:
        yticklabels = [parameter_labels[pid] for pid in yticklabels]

    n_outputs = df_subset.shape[1]
    n_parameters = df_subset.shape[0]
    # (width, height)
    figsize = (15, int(n_parameters / n_outputs * 15))

    # plot heatmap
    with warnings.catch_warnings():
        # seaborn 0.13.2 calls `Colormap.set_bad` for `center`, which
        # matplotlib 3.11 deprecates
        warnings.filterwarnings(
            "ignore",
            message="The set_bad function will be deprecated",
            category=PendingDeprecationWarning,
        )
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
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(cg.figure)
    return cg.figure


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
            f_path = (
                fig_path.parent
                / f"{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"
            )

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
                dpi=sa.dpi,
            )


def S1_ST_barplot(  # noqa: D103 -- documented below the signature
    S1,
    ST,
    S1_conf,
    ST_conf,
    parameter_labels: dict[str, str],
    fig_path: Path | None = None,
    title: str | None = None,
    ymax: float = 1.1,
    ymin: float = -0.1,
    dpi: int = 300,
) -> Figure:
    # width
    figsize = (15, 3)
    label_fontsize = 15

    categories: list[str] = list(parameter_labels.values())
    f, ax = plt.subplots(figsize=figsize)

    ax.bar(
        categories,
        ST,
        label="ST",
        color="black",
        alpha=1.0,
        edgecolor="black",
        yerr=ST_conf,
        capsize=5,
    )

    ax.bar(
        categories,
        S1,
        label="S1",
        color="tab:blue",
        edgecolor="black",
        yerr=S1_conf,
        capsize=5,
    )

    # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("Sensitivity", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", labelrotation=90)
    # ax.tick_params(axis='x', labelweight='bold')
    ax.legend()

    if title:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    if fig_path:
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(f)
    return f


def _selected(
    data: xr.DataArray, selection: Mapping[str, Any], keep: set[str]
) -> xr.DataArray:
    """Select labels and check that only the dimensions of `keep` remain.

    Args:
        data: the indices.
        selection: the labels to select, by dimension.
        keep: the dimensions which the plot draws.

    Returns:
        The selected data.

    Raises:
        ValueError: if a dimension outside `keep` has more than one label.
    """
    selected = data.sel(selection)
    for dim in selected.dims:
        if str(dim) not in keep and selected.sizes[dim] > 1:
            raise ValueError(
                f"Choose one label of '{dim}' with {dim}=..., it has "
                f"{selected.sizes[dim]}."
            )
    return selected.squeeze([d for d in selected.dims if str(d) not in keep])


def _label(name: str, unit: str | None) -> str:
    """Get an axis label with the unit if one is known."""
    return f"{name} [{unit}]" if unit and unit != "dimensionless" else name


def plot_heatmap(
    result: SensitivityResult,
    index: str,
    *,
    observables: Sequence[str] | None = None,
    cutoff: float | None = 0.1,
    cluster_rows: bool = True,
    title: str | None = None,
    cmap: str = "seismic",
    vmin: float | None = None,
    vmax: float | None = None,
    path: Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw an index of the scalar observables over the parameters.

    Args:
        result: the sensitivity result.
        index: the index, e.g. `ST` or `normalized`.
        observables: the observables, every scalar one with the index by default.
        cutoff: parameters whose values are all below it are left out.
        cluster_rows: whether the parameters are clustered.
        title: the title of the figure.
        cmap: the color map.
        vmin: the lower end of the color scale, `-2` for `normalized`, else `0`.
        vmax: the upper end of the color scale, `2` for `normalized`, else `1`.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure.

    Raises:
        ValueError: if a dimension with several labels is not chosen.
    """
    data = _selected(
        result.index(index, observables), selection, {PARAMETER, "observable"}
    )
    df = pd.DataFrame(
        data.transpose(PARAMETER, "observable").values,
        index=data[PARAMETER].values,
        columns=data["observable"].values,
    )
    normalized = index == "normalized"
    low = (-2.0 if normalized else 0.0) if vmin is None else vmin
    high = (2.0 if normalized else 1.0) if vmax is None else vmax
    return _heatmap(
        df=df,
        cutoff=cutoff,
        cluster_rows=cluster_rows,
        title=title,
        cmap=cmap,
        vcenter=(low + high) / 2,
        vmin=low,
        vmax=high,
        fig_path=path,
        dpi=dpi,
    )


def plot_indices(
    result: SensitivityResult,
    observable: str,
    *,
    path: Path | None = None,
    **selection: Any,
) -> Figure:
    """Draw the S1 and ST indices of an observable with their intervals.

    Args:
        result: the result of a Sobol or FAST analysis.
        observable: the observable.
        path: where the figure is saved, if given.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure, a pair of bars per parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen.
    """
    keep = {PARAMETER}
    values = {
        key: _selected(result[f"{observable}.{key}"], selection, keep)
        for key in ("S1", "ST", "S1_conf", "ST_conf")
        if f"{observable}.{key}" in result
    }
    parameters = result.parameters
    x = np.arange(len(parameters))
    width = 0.4
    figure = Figure(
        figsize=(max(6.0, 0.6 * len(parameters) + 2), 4), layout="constrained"
    )
    ax = figure.subplots()
    for offset, key, color in (
        (-width / 2, "S1", "tab:blue"),
        (width / 2, "ST", "black"),
    ):
        if key not in values:
            continue
        ax.bar(
            x + offset,
            values[key].values,
            width,
            label=key,
            color=color,
            edgecolor="black",
            yerr=values[f"{key}_conf"].values if f"{key}_conf" in values else None,
            capsize=4,
        )
    ax.set_xticks(x, parameters, rotation=90)
    ax.set_xlabel("Parameter")
    ax.set_ylabel(_label("Sensitivity", result.units.get(f"{observable}.ST")))
    ax.set_title(observable)
    ax.grid(True, axis="y")
    ax.legend()
    if path:
        figure.savefig(path, dpi=150)
    return figure


def plot_morris(
    result: SensitivityResult,
    observable: str,
    *,
    path: Path | None = None,
    **selection: Any,
) -> Figure:
    """Draw `mu_star` against `sigma` of an observable, a point per parameter.

    Args:
        result: the result of a Morris analysis.
        observable: the observable.
        path: where the figure is saved, if given.
        **selection: the label of every other dimension.

    Returns:
        The figure, every point labelled with its parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen.
    """
    mu_star = _selected(result[f"{observable}.mu_star"], selection, {PARAMETER})
    sigma = _selected(result[f"{observable}.sigma"], selection, {PARAMETER})
    figure = Figure(figsize=(5, 4.5), layout="constrained")
    ax = figure.subplots()
    ax.scatter(mu_star.values, sigma.values, color="tab:blue", edgecolor="black")
    for name, x, y in zip(result.parameters, mu_star.values, sigma.values, strict=True):
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
    unit = result.units.get(f"{observable}.mu_star")
    ax.set_xlabel(_label("mu_star", unit))
    ax.set_ylabel(_label("sigma", result.units.get(f"{observable}.sigma")))
    ax.set_title(observable)
    ax.grid(True)
    if path:
        figure.savefig(path, dpi=150)
    return figure
