"""Console output of a parameter fit.

The output of a fit is a sequence of sections which say what is being fitted:
the problem, the parameters which are optimized, the settings of the fit, the
data it uses and how that data is split into training and validation data.
Every section is rendered here, so the fit runner, the command line tools and
an interactive session produce the same output.

    from sbmlsim.fit import display

    display.section("Fit problem 'PK'")
    display.key_values({"strategy": "ALL", "runs": 4})

The sections, the key/value blocks and the links are those of
`sbmlsim.display`, which this module provides as well.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.measure import Measurement
from rich.table import Table

from sbmlsim.console import console

# the output of the fit is made of the sections, blocks and links of
# `sbmlsim.display`, which are provided here as well
from sbmlsim.display import ICON_REPORT as ICON_REPORT
from sbmlsim.display import KEY_WIDTH as KEY_WIDTH
from sbmlsim.display import key_values as key_values
from sbmlsim.display import link as link
from sbmlsim.display import section
from sbmlsim.fit.derived import HookSummary, ParameterGroup, group_parameters
from sbmlsim.fit.objects import FitParameter, MappingKind
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.parameter_mapping import CoverageRow, has_renamed_targets

#: icon of every section, so the sections of a fit are told apart at a glance
ICON_FIT = ":wrench:"
ICON_PARAMETERS = ":control_knobs:"
ICON_SETTINGS = ":gear:"
ICON_DATA = ":bar_chart:"
ICON_OPTIMIZATION = ":rocket:"
ICON_IDENTIFIABILITY = ":mag:"

#: color of every identifiability, see `sbmlsim.fit.identifiability`
IDENTIFIABILITY_STYLES: dict[str, str] = {
    "identifiable": "green",
    "non_identifiable_lower": "orange3",
    "non_identifiable_upper": "orange3",
    "non_identifiable": "red",
    "structural": "magenta",
}

#: color of every kind of data
KIND_STYLES: dict[str, str] = {
    MappingKind.TRAINING.value: "green",
    MappingKind.VALIDATION.value: "blue",
    MappingKind.OUTLIER.value: "orange3",
    MappingKind.EXCLUDED.value: "grey35",
}


def _table(*columns: str, title: str | None = None) -> Table:
    """Create the table style shared by the sections."""
    table = Table(
        box=box.SIMPLE_HEAD,
        title=title,
        title_justify="left",
        header_style="bold",
        pad_edge=False,
        show_edge=False,
    )
    for column in columns:
        table.add_column(column)
    return table


def _number(value: float | None) -> str:
    """Format a number of a table, `-` if there is none."""
    return "-" if value is None else f"{value:.4g}"


def parameters_table(parameters: Iterable[FitParameter]) -> Table:
    """Get the table of the parameters which are optimized.

    The target is shown only when some parameter writes an entity of another
    name, so an ordinary fit does not get a column which repeats its ids.
    """
    parameters = list(parameters)
    versioned = has_renamed_targets(parameters)
    columns = ["parameter"]
    if versioned:
        columns.append("target")
    columns.extend(["start", "lower", "upper", "unit"])
    table = _table(*columns)
    for p in parameters:
        row = [p.pid]
        if versioned:
            row.append(p.target_id)
        row.extend(
            [
                _number(p.start_value),
                _number(p.lower_bound),
                _number(p.upper_bound),
                p.unit or "[dim]model[/dim]",
            ]
        )
        table.add_row(*row)
    return table


#: the unit of the elements of a network, as the tables show it
ELEMENT_UNIT_LABEL = "dimensionless"


def hooks_table(summaries: Iterable[HookSummary]) -> Table:
    """Get the table of the hooks of a problem, e.g. its networks."""
    table = _table("network", "pattern", "layers", "targets")
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.kind,
            summary.description,
            ", ".join(summary.targets),
        )
    return table


def _array_values(members: Sequence[FitParameter]) -> list[str]:
    """Get the minimum, the maximum and the norm of the start values of an array."""
    values = np.asarray([p.start_value for p in members], dtype=float)
    if values.size == 0:
        return ["-", "-", "-"]
    return [
        _number(float(values.min())),
        _number(float(values.max())),
        _number(float(np.linalg.norm(values))),
    ]


def groups_table(
    groups: Sequence[tuple[ParameterGroup, Sequence[FitParameter]]],
) -> Table:
    """Get the table of the arrays of the networks, one row per array.

    An array is shown with the number of its elements, the number of them
    which are estimated and the minimum, the maximum and the norm of the
    start values of the estimated elements, and the bounds when the elements
    agree on them.
    """
    table = _table(
        "array", "elements", "estimated", "min", "max", "norm", "lower", "upper"
    )
    for group, members in groups:
        lower = {p.lower_bound for p in members}
        upper = {p.upper_bound for p in members}
        table.add_row(
            group.label,
            str(len(group.ids)),
            str(len(members)),
            *_array_values(members),
            _number(lower.pop()) if len(lower) == 1 else "-",
            _number(upper.pop()) if len(upper) == 1 else "-",
        )
    return table


def coverage_table(rows: Sequence[CoverageRow]) -> Table:
    """Get the table of the simulations every parameter applies to.

    A simulation no version of a target reaches keeps the value of the model,
    which is right where the parameter has no meaning, e.g. an absorption rate
    on intravenous data; the table makes it a fact which is read and not one
    which is discovered later. A parameter whose selector matches nothing
    covers no simulation at all: it stays in the parameter vector without
    ever changing the model, which is a silent trap for a mistyped filter, so
    such a row is styled in bold red rather than as an ordinary line.
    """
    table = _table("parameter", "target", "simulations", "not covered")
    for row in rows:
        uncovered = ", ".join(row.uncovered_groups)
        no_coverage = row.n_covered == 0
        style = "bold red" if no_coverage else None
        table.add_row(
            row.pid,
            row.target,
            f"{row.n_covered} of {row.n_groups}",
            (
                f"[bold red]{uncovered}[/bold red]"
                if no_coverage
                else (f"[dim]{uncovered}[/dim]" if uncovered else "-")
            ),
            style=style,
        )
    return table


def settings_table(settings: FitSettings) -> Table:
    """Get the table of the settings of a fit."""
    weighting_curves = (
        ", ".join(w.name for w in settings.weighting_curves)
        if settings.weighting_curves
        else "none"
    )
    table = _table("setting", "value")
    for key, value in [
        ("residual", settings.residual.name),
        ("loss function", settings.loss_function.name),
        ("weighting curves", weighting_curves),
        ("weighting points", settings.weighting_points.name),
        ("relative tolerance", f"{settings.relative_tolerance:.1e}"),
        ("absolute tolerance", f"{settings.absolute_tolerance:.1e}"),
        ("variable step size", str(settings.variable_step_size)),
    ]:
        table.add_row(key, value)
    return table


def _kind(kind: str) -> str:
    """Format the kind of a fit mapping with its color."""
    style = KIND_STYLES.get(kind)
    return f"[{style}]{kind}[/{style}]" if style else kind


def data_summary_table(df: pd.DataFrame) -> Table:
    """Get the table of the fit mappings per experiment and kind.

    Args:
        df: metadata table of `sbmlsim.fit.helpers.MappingSelection`.

    Returns:
        One row per simulation experiment with the number of mappings of every
        kind, and a row with the totals.
    """
    kinds = [kind.value for kind in MappingKind]
    table = _table("experiment", *(_kind(kind) for kind in kinds), "mappings")

    if "kind" not in df.columns or "experiment" not in df.columns:
        table.add_row("[dim]unknown[/dim]", *(["-"] * len(kinds)), str(len(df)))
        return table

    counts = df.groupby(["experiment", "kind"]).size().unstack(fill_value=0)
    for experiment in counts.index:
        row = [str(int(counts.get(kind, {}).get(experiment, 0))) for kind in kinds]
        table.add_row(str(experiment), *row, str(sum(int(v) for v in row)))

    totals = [str(int((df["kind"] == kind).sum())) for kind in kinds]
    table.add_section()
    table.add_row(
        "[bold]total[/bold]",
        *(f"[bold]{value}[/bold]" for value in totals),
        f"[bold]{len(df)}[/bold]",
    )
    return table


#: the columns of the data table every selection has, the rest is the metadata
DATA_COLUMNS = ["experiment", "fm_key", "yid", "kind"]


def data_table(df: pd.DataFrame) -> Table:
    """Get the table of the single fit mappings.

    Args:
        df: metadata table of `sbmlsim.fit.helpers.MappingSelection`.

    Returns:
        One row per fit mapping with its experiment, its observable, its kind
        and the fields of its metadata, which describe the curve.
    """
    metadata_columns = [c for c in df.columns if c not in DATA_COLUMNS]
    table = _table("experiment", "mapping", "observable", "kind", *metadata_columns)
    for column in table.columns:
        column.no_wrap = True
    for _, row in df.iterrows():
        kind = str(row.get("kind", ""))
        # the excluded data is no part of the fit, the complete row is grey
        excluded = kind == MappingKind.EXCLUDED.value
        table.add_row(
            str(row.get("experiment", "")),
            str(row.get("fm_key", "")),
            str(row.get("yid", "")),
            _kind(kind),
            *(_cell(row[c]) for c in metadata_columns),
            style=KIND_STYLES[MappingKind.EXCLUDED.value] if excluded else None,
        )
    return table


def _cell(value: Any) -> str:
    """Format a metadata value of the data table."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[dim]-[/dim]"
    return str(value)


def print_parameters(
    parameters: Iterable[FitParameter],
    coverage: Sequence[CoverageRow] | None = None,
    hooks: Iterable[HookSummary] | None = None,
) -> None:
    """Print the section of the parameters which are optimized.

    The elements of a network are not listed one by one: the networks are
    printed with their pattern, their layers and their targets, and their
    arrays with the number of elements, the estimated ones and the range and
    the norm of the start values.

    Args:
        parameters: parameters of the fit.
        coverage: what every parameter reaches, from
            `sbmlsim.fit.parameter_mapping.ParameterMapping.coverage`. The
            coverage table is only printed when some parameter does not reach
            every simulation, so an ordinary fit is not given an all-`-`
            table. The elements of the networks are left out of it.
        hooks: the summaries of the hooks of the problem, see
            `sbmlsim.fit.derived.hook_summaries`.
    """
    parameters = list(parameters)
    summaries = list(hooks or [])
    single, groups = group_parameters(parameters, summaries)
    section(f"Parameters ({len(parameters)})", icon=ICON_PARAMETERS)
    if single:
        console.print(parameters_table(single))
    if summaries:
        print_wide(hooks_table(summaries))
        print_wide(groups_table(groups))
    grouped = {p.pid for _, members in groups for p in members}
    rows = [row for row in (coverage or []) if row.pid not in grouped]
    if rows and any(row.uncovered_groups for row in rows):
        console.print(coverage_table(rows))


def print_settings(settings: FitSettings) -> None:
    """Print the section of the settings of a fit."""
    section("Settings", icon=ICON_SETTINGS)
    console.print(settings_table(settings))


def print_data(df: pd.DataFrame, detail: bool = True) -> None:
    """Print the section of the data of a fit.

    Args:
        df: metadata table of `sbmlsim.fit.helpers.MappingSelection`.
        detail: list the single fit mappings, not only the counts.
    """
    section(f"Data ({len(df)} fit mappings)", icon=ICON_DATA)
    console.print(data_summary_table(df))
    if detail:
        console.line()
        print_wide(data_table(df))


def print_wide(table: Table) -> None:
    """Print a table at its full width, the columns are not truncated.

    A table with many columns, e.g. the data table with its metadata, is wider
    than the console; the console would shorten the cells to `…`, so the table
    is printed at the width it needs and the terminal wraps the lines.
    """
    width = Measurement.get(console, console.options.update_width(10_000), table)
    if width.maximum <= console.width:
        console.print(table)
        return
    # `console.print(width=...)` is capped at the width of the console, so the
    # console is widened for the table and restored afterwards
    console_width = console._width
    console.width = width.maximum
    try:
        console.print(table, crop=False)
    finally:
        console._width = console_width


def identifiability_table(df: pd.DataFrame) -> Table:
    """Get the table of a profile likelihood analysis.

    Args:
        df: summary of an `IdentifiabilityResult`, one row per parameter.

    Returns:
        The table with the value, the confidence interval and the
        classification of every parameter; an open side of an interval is
        shown as the bound of the parameter with a `<` or `>`.
    """
    table = _table(
        "parameter", "value", "ci lower", "ci upper", "unit", "identifiability"
    )
    for row in df.to_dict(orient="records"):
        ci_lower = (
            f"< {row['lower_bound']:.4g}"
            if pd.isna(row["ci_lower"])
            else _number(float(row["ci_lower"]))
        )
        ci_upper = (
            f"> {row['upper_bound']:.4g}"
            if pd.isna(row["ci_upper"])
            else _number(float(row["ci_upper"]))
        )
        identifiability = str(row["identifiability"])
        style = IDENTIFIABILITY_STYLES.get(identifiability)
        table.add_row(
            str(row["parameter"]),
            _number(float(row["value"])),
            ci_lower,
            ci_upper,
            str(row["unit"]) if row["unit"] else "[dim]model[/dim]",
            f"[{style}]{identifiability}[/{style}]" if style else identifiability,
        )
    return table


def print_identifiability(df: pd.DataFrame) -> None:
    """Print the table of a profile likelihood analysis."""
    console.print(identifiability_table(df))
