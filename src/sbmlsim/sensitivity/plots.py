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
from matplotlib.patches import Rectangle

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
    vcenter: float | None = 0.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    fig_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Creates heatmap of model sensitivity.

    The cutoff applies to the finite values only: a row with an undefined value
    (`NaN`) is kept and the undefined cells are drawn in light grey. An axis is
    clustered only if it has at least two entries, on the values with `NaN`
    replaced by 0. The size of the figure follows the number of rows and columns.

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
        return df[(df.abs() >= cutoff).any(axis=1) | df.isna().any(axis=1)]

    # filter rows
    # X.drop(pk_exclude, axis=1, inplace=True)

    df_subset = calculate_subset(df, cutoff=cutoff) if cutoff and cutoff > 0 else df
    undefined = df_subset.isna()
    df_subset_mask = calculate_mask(df_subset, cutoff) | undefined
    df_subset = df_subset.fillna(0.0)

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
    figsize = (
        float(np.clip(0.9 * n_outputs + 4, 6, 24)),
        float(np.clip(0.6 * n_parameters + 3, 4, 40)),
    )

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
            col_cluster=cluster_cols and n_outputs > 1,
            row_cluster=cluster_rows and n_parameters > 1,
            method="single",
            figsize=figsize,
        )
    ordered = undefined.loc[cg.data2d.index, cg.data2d.columns].to_numpy()
    for row, col in np.argwhere(ordered):
        cg.ax_heatmap.add_patch(
            Rectangle((col, row), 1, 1, facecolor="lightgrey", edgecolor="none")
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
    # the space of the hidden dendrograms goes to the heatmap
    box = cg.ax_heatmap.get_position()
    width = cg.figure.get_figwidth()
    left = min(box.x0, max(0.06, min(0.1, 0.03 + 0.7 / width)))
    top = 0.97 if not title else 0.9
    cg.ax_heatmap.set_position((left, box.y0, box.x1 - left, top - box.y0))

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


SIGNED_INDICES = ("raw", "normalized", "mu")
"""The indices which have a sign, drawn on a diverging color map around 0."""

BOUNDED_INDICES = ("S1", "ST")
"""The indices in the interval 0 to 1."""

INDEX_NAMES = {"sobol": "Sobol index", "fast": "FAST index"}


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
        ValueError: if a dimension or a label does not exist, or if a dimension
            outside `keep` has more than one label.
    """
    selected = data
    for dim, label in selection.items():
        if dim not in selected.dims:
            available = [str(d) for d in data.dims if str(d) not in keep]
            raise ValueError(
                f"Unknown selection '{dim}', the dimensions to select are {available}."
            )
        try:
            selected = selected.sel({dim: label})
        except KeyError as err:
            labels = data[dim].values.tolist()
            raise ValueError(
                f"Unknown label {label!r} of '{dim}', the labels are {labels}."
            ) from err
    for dim in selected.dims:
        if str(dim) not in keep and selected.sizes[dim] > 1:
            raise ValueError(
                f"Choose one label of '{dim}' with {dim}=..., it has "
                f"{selected.sizes[dim]}."
            )
    return selected.squeeze([d for d in selected.dims if str(d) not in keep])


def _check_observable(result: SensitivityResult, observable: str) -> None:
    """Raise a `ValueError` naming the observables if `observable` is unknown."""
    if observable not in result.observables:
        raise ValueError(
            f"Unknown observable '{observable}', the observables are "
            f"{result.observables}."
        )


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
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    path: Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw an index of the scalar observables over the parameters.

    The cutoff applies to the finite values. An undefined value (`NaN`, e.g. a
    normalized index of a zero reference) is never dropped by it: its row stays
    and its cells are drawn masked in light grey. An axis with a single entry is
    not clustered. The figure grows with the number of parameters and observables.

    Args:
        result: the sensitivity result.
        index: the index, e.g. `ST` or `normalized`.
        observables: the observables, every scalar one with the index by default.
        cutoff: parameters whose finite values are all below it are left out.
        cluster_rows: whether the parameters are clustered.
        title: the title of the figure.
        cmap: the color map; `seismic` centered at 0 for the signed indices
            (`raw`, `normalized`, `mu`), else `Reds` from 0.
        vmin: the lower end of the color scale: `-2` for `normalized`, the
            largest magnitude of the data (negated) for the other signed
            indices, else `0`.
        vmax: the upper end of the color scale: `2` for `normalized`, the largest
            magnitude of the data for the other signed indices, `1` for `S1` and
            `ST`, else the largest value of the data.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure.

    Raises:
        ValueError: if a dimension with several labels is not chosen, or an
            index, an observable, a dimension or a label does not exist.
    """
    for observable in observables or ():
        _check_observable(result, observable)
    try:
        stacked = result.index(index, observables)
    except KeyError as err:
        raise ValueError(
            f"No scalar observable has the index '{index}', the observables are "
            f"{result.observables}."
        ) from err
    data = _selected(stacked, selection, {PARAMETER, "observable"})
    df = pd.DataFrame(
        data.transpose(PARAMETER, "observable").values,
        index=data[PARAMETER].values,
        columns=data["observable"].values,
    )
    finite = np.abs(df.to_numpy(dtype=float))
    top = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
    top = top if top > 0 else 1.0
    signed = index in SIGNED_INDICES
    if signed:
        extent = 2.0 if index == "normalized" else top
        low, high = -extent, extent
        vcenter: float | None = 0.0
    else:
        low, high = 0.0, 1.0 if index in BOUNDED_INDICES else top
        vcenter = None
    low = low if vmin is None else vmin
    high = high if vmax is None else vmax
    return _heatmap(
        df=df,
        cutoff=cutoff,
        cluster_rows=cluster_rows,
        title=title,
        cmap=cmap or ("seismic" if signed else "Reds"),
        vcenter=vcenter,
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
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw the S1 and ST indices of an observable with their intervals.

    Args:
        result: the result of a Sobol or FAST analysis.
        observable: the observable.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure, a pair of bars per parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen, or the
            observable, a dimension or a label does not exist.
    """
    _check_observable(result, observable)
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
    ax.set_ylabel(INDEX_NAMES.get(result.method, "Sensitivity index"))
    ax.set_title(observable)
    ax.grid(True, axis="y")
    ax.legend()
    if path:
        figure.savefig(path, dpi=dpi)
    return figure


def plot_morris(
    result: SensitivityResult,
    observable: str,
    *,
    path: Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw `mu_star` against `sigma` of an observable, a point per parameter.

    Args:
        result: the result of a Morris analysis.
        observable: the observable.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension.

    Returns:
        The figure, every point labelled with its parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen, or the
            observable, a dimension or a label does not exist.
    """
    _check_observable(result, observable)
    mu_star = _selected(result[f"{observable}.mu_star"], selection, {PARAMETER})
    sigma = _selected(result[f"{observable}.sigma"], selection, {PARAMETER})
    figure = Figure(figsize=(5, 4.5), layout="constrained")
    ax = figure.subplots()
    ax.scatter(mu_star.values, sigma.values, color="tab:blue", edgecolor="black")
    notes = [
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
        for name, x, y in zip(
            result.parameters, mu_star.values, sigma.values, strict=True
        )
    ]
    ax.margins(0.1)
    # widen the limits until no label touches or crosses the frame
    figure.canvas.draw()
    frame = ax.get_window_extent()
    pad = 6.0
    to_data = ax.transData.inverted()
    x1, y1 = to_data.transform((frame.x1 - pad, frame.y1 - pad))
    boxes = [n.get_window_extent() for n in notes]
    if boxes:
        edge_x, edge_y = to_data.transform(
            (max(b.x1 for b in boxes), max(b.y1 for b in boxes))
        )
        ax.set_xlim(right=ax.get_xlim()[1] + max(0.0, float(edge_x - x1)))
        ax.set_ylim(top=ax.get_ylim()[1] + max(0.0, float(edge_y - y1)))
    unit = result.units.get(f"{observable}.mu_star")
    ax.set_xlabel(_label("mu_star", unit))
    ax.set_ylabel(_label("sigma", result.units.get(f"{observable}.sigma")))
    ax.set_title(observable)
    ax.grid(True)
    if path:
        figure.savefig(path, dpi=dpi)
    return figure
