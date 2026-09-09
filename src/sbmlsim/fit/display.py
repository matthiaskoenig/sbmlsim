"""Console output of a parameter fit.

The output of a fit is a sequence of sections which say what is being fitted:
the problem, the parameters which are optimized, the settings of the fit, the
data it uses and how that data is split into training and validation data.
Every section is rendered here, so the fit runner, the command line tools and
an interactive session produce the same output.

    from sbmlsim.fit import display

    display.section("Fit problem 'PK'")
    display.key_values({"strategy": "ALL", "runs": 4})
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.table import Table

from sbmlsim.console import console
from sbmlsim.fit.objects import FitParameter, MappingKind
from sbmlsim.fit.options import FitSettings

#: width of the keys of a key/value block
KEY_WIDTH = 18

#: icon of every section, so the sections of a fit are told apart at a glance
ICON_FIT = ":wrench:"
ICON_PARAMETERS = ":control_knobs:"
ICON_SETTINGS = ":gear:"
ICON_DATA = ":bar_chart:"
ICON_OPTIMIZATION = ":rocket:"
ICON_REPORT = ":clipboard:"
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
}


def section(title: str, icon: str | None = None) -> None:
    """Start a section of the output.

    A blank line separates the section from whatever came before it, so the
    sections are told apart whether the previous one ended in a table or in a
    key/value block.

    Args:
        title: title of the section.
        icon: emoji in front of the title, e.g. `ICON_PARAMETERS`.
    """
    prefix = f"{icon} " if icon else ""
    console.line()
    console.rule(f"{prefix}[bold]{title}", align="left", style="white")


def key_values(items: Mapping[str, Any]) -> None:
    """Print aligned key/value lines, the smallest section of the output."""
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    table.add_column("key", style="bold", width=KEY_WIDTH)
    table.add_column("value", overflow="fold")
    for key, value in items.items():
        table.add_row(key, str(value))
    console.print(table)


def link(key: str, path: Path | str) -> None:
    """Print a file link, on a single line so that the terminal can open it.

    The path is a `file://` URI, i.e., it has forward slashes and a drive is
    `file:///C:/...`; a windows path with backslashes is not a link a terminal
    opens.

    Args:
        key: what the link points to, in front of it.
        path: path of the file, relative paths are resolved.
    """
    console.print(
        f"[bold]{key:<{KEY_WIDTH}}[/bold] {Path(path).resolve().as_uri()}",
        soft_wrap=True,
    )


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
    """Get the table of the parameters which are optimized."""
    table = _table("parameter", "start", "lower", "upper", "unit")
    for p in parameters:
        table.add_row(
            p.pid,
            _number(p.start_value),
            _number(p.lower_bound),
            _number(p.upper_bound),
            p.unit or "[dim]model[/dim]",
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
        df: metadata table of `sbmlsim.fit.helpers.filtered_mapping_collections`.

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


def data_table(df: pd.DataFrame) -> Table:
    """Get the table of the single fit mappings.

    Args:
        df: metadata table of `sbmlsim.fit.helpers.filtered_mapping_collections`.

    Returns:
        One row per fit mapping with its experiment, its observable and its
        kind. The remaining metadata describes the curve and is in the report.
    """
    table = _table("experiment", "mapping", "observable", "kind")
    for _, row in df.iterrows():
        table.add_row(
            str(row.get("experiment", "")),
            str(row.get("fm_key", "")),
            str(row.get("yid", "")),
            _kind(str(row.get("kind", ""))),
        )
    return table


def print_parameters(parameters: Iterable[FitParameter]) -> None:
    """Print the section of the parameters which are optimized."""
    parameters = list(parameters)
    section(f"Parameters ({len(parameters)})", icon=ICON_PARAMETERS)
    console.print(parameters_table(parameters))


def print_settings(settings: FitSettings) -> None:
    """Print the section of the settings of a fit."""
    section("Settings", icon=ICON_SETTINGS)
    console.print(settings_table(settings))


def print_data(df: pd.DataFrame, detail: bool = True) -> None:
    """Print the section of the data of a fit.

    Args:
        df: metadata table of `sbmlsim.fit.helpers.filtered_mapping_collections`.
        detail: list the single fit mappings, not only the counts.
    """
    section(f"Data ({len(df)} fit mappings)", icon=ICON_DATA)
    console.print(data_summary_table(df))
    if detail:
        console.line()
        console.print(data_table(df))


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
