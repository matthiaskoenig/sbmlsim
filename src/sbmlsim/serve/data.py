"""The study data behind the curves of a panel.

`panel_data(entry, panel, x_unit)` reads the study data a panel declares
(`sbmlsim.panel.PanelData`) out of the data of the project of its model and
gives one `PanelSeries` per declaration: the measured points a reader sees
under the simulated curves.

A series names its dataset by the id of a figure or a table of a study, e.g.
`Beermann1976_Fig2`, which `sbmlsim.data.load_pkdb_dataframe` resolves below
the `data_path` of the project. The id is no path and a `PanelData` refuses one
(`sbmlsim.panel`), so a declaration can never name a file outside of that data.

The rows of a series are the ones its `sel` selects, by the text of the value
of every column which selects them, which a dataset of several series needs,
and its columns `x` and `y` are read with the units of the dataset
(`sbmlsim.data.DataSet.from_df`). The values and their error are multiplied by
the `conversion` the series declares, e.g. by the inverse of a molecular weight
for the mass of a study which the model carries as a substance, and are
delivered in the unit the series declares, else in the unit of the curve it
belongs under (`PanelData.curve`, its own id without), so the points and the
curve share the axis of the panel. Several series may belong under one curve,
e.g. two studies, and are delivered in that one unit. A row whose time or value
is no number is no point of the series, so the answer carries no `NaN`, which
JSON has not.

The times are delivered in `x_unit`, the unit the x of the panel is delivered
in: the caller resolves it, and the one caller of the service is the route,
which takes it from `sbmlsim.serve.runner.PanelRunner.x_unit`, the unit a run
of the same panel reports in `units["x"]`. The points of a reader are therefore
on the axis of its lines, whatever the time unit of the dataset is; a model
whose time has no unit (an empty `x_unit`) delivers the times of the dataset as
they are.

Every problem of a declaration is a `ValueError` which names the series, its
dataset and its columns or units and never a path of the service, which the
route answers with its own fixed message (`sbmlsim.serve.app`). Nothing here
loads a model: the data of a panel is read from the files of its project alone,
and only the `x_unit` of a panel which declares none needs the model, which the
runner keeps warm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from pint.errors import DimensionalityError, UndefinedUnitError

from sbmlsim.data import UNCERTAINTY_SUFFIXES, DataSet, load_pkdb_dataframe
from sbmlsim.units import Quantity, ureg

if TYPE_CHECKING:
    import pandas as pd

    from sbmlsim.panel import Panel, PanelData
    from sbmlsim.serve.runner import ModelEntry

logger = logging.getLogger(__name__)

__all__ = ["COUNT", "PanelSeries", "panel_data"]

#: the column of the number of subjects behind a point, as a table of a study
#: carries it; a series of a dataset without it has no counts
COUNT: Final = "count"


@dataclass(frozen=True)
class PanelSeries:
    """One series of the study data of a panel, see the module.

    Attributes:
        id: the id of the series, the one the panel declares; the curve it
            belongs under is `PanelData.curve` of that declaration.
        label: the series as it is shown, e.g. the study it comes from.
        x: the times of the points, in the unit `x_unit` names, which is the
            unit of the x of a simulation of the same panel.
        y: the values of the points, in the unit `unit` names.
        yerr: the error of every value, `None` for a point without one; `None`
            for a series which declares no column of the error.
        x_unit: the unit of the times, `None` for a model whose time has no
            unit, whose times are the ones of the dataset.
        unit: the unit of the values, `None` for a series which names no curve
            and whose column has no unit.
        count: the number of subjects behind every point, `None` for a dataset
            without the column `COUNT` and for one whose points do not all
            carry it.
    """

    id: str
    label: str
    x: list[float]
    y: list[float]
    yerr: list[float | None] | None
    x_unit: str | None
    unit: str | None
    count: list[int] | None


def panel_data(entry: ModelEntry, panel: Panel, x_unit: str) -> list[PanelSeries]:
    """Read the study data a panel declares, see the module.

    Args:
        entry: the model of the panel, whose project holds the data.
        panel: the panel.
        x_unit: the unit the times are delivered in, the unit the x of the
            panel is delivered in: `sbmlsim.serve.runner.PanelRunner.x_unit`
            of the same panel, so the points are on the axis of the curves.
            An empty unit, i.e. a model whose time has no unit, delivers the
            times of the dataset as they are.

    Returns:
        One series per declaration of the panel, in the order the panel
        declares them; none for a panel without study data.

    Raises:
        FileNotFoundError: for a dataset which is no file of the data of the
            project.
        ValueError: naming the series and what is wrong with it, for a dataset
            of a project without a `data_path`, a selection which no row has,
            a column the dataset has not or which carries no numbers, a
            conversion which is no quantity, a series whose times cannot be
            delivered in `x_unit` and one which cannot be delivered in the
            unit it is declared in.
    """
    return [_series(entry, panel, declared, x_unit) for declared in panel.data]


def _series(
    entry: ModelEntry, panel: Panel, declared: PanelData, x_unit: str
) -> PanelSeries:
    """Read one series of the study data of a panel, see `panel_data`."""
    frame = load_pkdb_dataframe(declared.dataset, data_path=entry.project.data_path)
    dset = DataSet.from_df(_selected(frame, declared))
    conversion = _conversion(declared)
    times, time_unit = _delivered(dset, declared, declared.x, x_unit or None)
    values, unit = _delivered(
        dset, declared, declared.y, _unit_of_series(panel, declared), conversion
    )
    keep = np.isfinite(times) & np.isfinite(values)
    errors: list[float | None] | None = None
    if declared.yerr is not None:
        deviations, _ = _delivered(dset, declared, declared.yerr, unit, conversion)
        errors = [
            float(value) if np.isfinite(value) else None for value in deviations[keep]
        ]
    logger.debug(
        "The series '%s' of the dataset '%s' has %d points in '%s'.",
        declared.id,
        declared.dataset,
        int(keep.sum()),
        unit,
    )
    return PanelSeries(
        id=declared.id,
        label=declared.label,
        x=[float(value) for value in times[keep]],
        y=[float(value) for value in values[keep]],
        yerr=errors,
        x_unit=time_unit,
        unit=unit,
        count=_counts(dset, keep),
    )


def _selected(frame: pd.DataFrame, declared: PanelData) -> pd.DataFrame:
    """Get the rows of a dataset which are the series, see the module.

    Args:
        frame: the dataset as it was read.
        declared: the declaration of the series.

    Returns:
        The rows whose columns have the values of `sel`, every row of the
        dataset for a series without a selection.

    Raises:
        ValueError: naming the series and its dataset, if a column is not one
            of the dataset or no row is left.
    """
    rows = frame
    for column, value in (declared.sel or {}).items():
        if column not in frame.columns:
            raise ValueError(
                f"The series '{declared.id}' selects rows by the column "
                f"'{column}', which the dataset '{declared.dataset}' has not: "
                f"{sorted(str(name) for name in frame.columns)}."
            )
        rows = rows[rows[column].astype(str) == value]
    if rows.empty:
        if declared.sel:
            raise ValueError(
                f"The series '{declared.id}' selects {dict(declared.sel)}, which "
                f"no row of the dataset '{declared.dataset}' has."
            )
        raise ValueError(
            f"The dataset '{declared.dataset}' of the series '{declared.id}' has "
            f"no rows."
        )
    return rows


def _conversion(declared: PanelData) -> Quantity | None:
    """Get the quantity the values of a series are multiplied by, none without.

    Raises:
        ValueError: naming the series and the expression, for a conversion
            which the registry of the package does not read as a quantity.
    """
    if declared.conversion is None:
        return None
    try:
        return ureg.Quantity(declared.conversion)
    except Exception as err:
        raise ValueError(
            f"The conversion '{declared.conversion}' of the series "
            f"'{declared.id}' is no quantity: {err}"
        ) from err


def _unit_of_series(panel: Panel, declared: PanelData) -> str | None:
    """Get the unit a series is delivered in, see the module.

    Args:
        panel: the panel of the series.
        declared: the declaration of the series.

    Returns:
        The unit the series declares, else the unit of the curve it belongs
        under (`PanelData.curve_id`), `None` for a series which names no curve
        of the panel or one without a unit, which is delivered in the unit of
        its dataset.
    """
    if declared.unit is not None:
        return declared.unit
    wanted = declared.curve_id
    return next((curve.unit for curve in panel.curves if curve.id == wanted), None)


def _delivered(
    dset: DataSet,
    declared: PanelData,
    column: str,
    target: str | None,
    conversion: Quantity | None = None,
) -> tuple[np.ndarray, str | None]:
    """Get the values of a column of a series and the unit they are in.

    Args:
        dset: the rows of the series with the units of their columns.
        declared: the declaration of the series.
        column: the column which is read.
        target: the unit the values are delivered in, their own without.
        conversion: the quantity the values are multiplied by, none without.

    Returns:
        The values and their unit, `None` for a column without one which is
        delivered as it is.

    Raises:
        ValueError: naming the series, for a column the dataset has not, one
            which carries no numbers, one without a unit which is converted,
            and for values which cannot be delivered in `target`.
    """
    values = _values(dset, declared, column)
    unit = _unit_of(dset, column)
    if conversion is None and target is None:
        return values, unit
    if unit is None:
        raise ValueError(
            f"The column '{column}' of the dataset '{declared.dataset}' has no "
            f"unit, which the series '{declared.id}' needs to be converted."
        )
    quantity = ureg.Quantity(values, unit)
    if conversion is not None:
        quantity = quantity * conversion
    if target is None:
        return np.asarray(quantity.magnitude, dtype=float), _unit_text(quantity)
    try:
        quantity = quantity.to(target)
    except (DimensionalityError, UndefinedUnitError) as err:
        measured = unit if conversion is None else f"{unit} * {declared.conversion}"
        raise ValueError(
            f"The series '{declared.id}' is measured in '{measured}' and cannot "
            f"be delivered in '{target}': {err}"
        ) from err
    return np.asarray(quantity.magnitude, dtype=float), target


def _values(dset: DataSet, declared: PanelData, column: str) -> np.ndarray:
    """Get the numbers of a column of a series.

    Raises:
        ValueError: naming the series and its dataset, for a column the
            dataset has not and for one which carries values which are no
            numbers.
    """
    if column not in dset.columns:
        raise ValueError(
            f"The series '{declared.id}' reads the column '{column}', which the "
            f"dataset '{declared.dataset}' has not: "
            f"{sorted(str(name) for name in dset.columns)}."
        )
    try:
        return np.asarray(dset[column].to_numpy(), dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"The column '{column}' of the dataset '{declared.dataset}', which "
            f"the series '{declared.id}' reads, has values which are no numbers: "
            f"{err}"
        ) from err


def _unit_of(dset: DataSet, column: str) -> str | None:
    """Get the unit of a column of a dataset, the one of its quantity for an error.

    Args:
        dset: the dataset.
        column: the column.

    Returns:
        The unit of the column, the unit of the quantity of a column of its
        uncertainty (`cp_sd` is in the unit of `cp`), `None` for a column
        without one.
    """
    uinfo = dset.uinfo
    if column in uinfo:
        return str(uinfo[column]) or None
    base, _, suffix = column.rpartition("_")
    if base and suffix in UNCERTAINTY_SUFFIXES and base in uinfo:
        return str(uinfo[base]) or None
    return None


def _unit_text(quantity: Quantity) -> str:
    """Get the unit of a quantity as a dataset writes it, e.g. `micromole/liter`."""
    return str(quantity.units).replace("**", "^").replace(" ", "")


def _counts(dset: DataSet, keep: np.ndarray) -> list[int] | None:
    """Get the number of subjects behind every point of a series, none without.

    Args:
        dset: the rows of the series.
        keep: the rows which are points of the series.

    Returns:
        The counts of these rows, `None` for a dataset without the column
        `COUNT`, for a series without a point and for one whose points do not
        all carry a count, which is no uncertainty of a point.
    """
    if COUNT not in dset.columns:
        return None
    counts = np.asarray(dset[COUNT].to_numpy(), dtype=float)[keep]
    if counts.size == 0 or not bool(np.isfinite(counts).all()):
        return None
    return [int(count) for count in counts]
