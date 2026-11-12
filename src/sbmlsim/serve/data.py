"""The study data behind the curves of a panel.

`panel_data(entry, panel, units)` reads the study data a panel declares
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
delivered in the unit of the series (`sbmlsim.panel.Panel.series_unit`: its
own, else the one of the curve it belongs under), so the points and the curve
share the axis of the panel. Several series may belong under one curve, e.g.
two studies, and are delivered in that one unit. A row whose time or value is
no number is no point of the series, so the answer carries no `NaN`, which JSON
has not, and the error of a value is delivered in the unit of that value.

The units a series is delivered in are the `PanelUnits` of its panel, which
the caller resolves and hands over: the one caller of the service is the route,
which takes them from `sbmlsim.serve.runner.PanelRunner.units` of the same
panel, where the model resolves a unit id of its own (`mmole_per_l`) as it does
for a run. The times are therefore on the axis of the lines of the panel and
the values in the unit of their curve, whatever the units of the dataset are,
and nothing of a declaration is parsed twice.

A series whose curve declares no unit is delivered in the unit the model gives
that curve, the unit a run of the panel delivers the curve in
(`sbmlsim.serve.runner.PanelUnits`), and only a series which belongs under no
curve of the panel at all is delivered in the unit of its own dataset. Where
the panel has no unit, a curve the model gives none and a model whose time has
no unit, the values are delivered as they are and the dataset must carry no
unit either, which keeps the two sides of one axis one scale (`_plain`).

Every problem of a declaration is a `ValueError` which names the series, its
dataset and its columns or units and never a path of the service, which the
route answers with its own fixed message (`sbmlsim.serve.app`). Nothing here
loads a model or parses a unit of a declaration: the data of a panel is read
from the files of its project alone and is delivered in the units its caller
resolved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from tokenize import TokenError
from typing import TYPE_CHECKING, Final

import numpy as np
from pint.errors import PintError

from sbmlsim.data import DataSet, load_pkdb_dataframe
from sbmlsim.units import Quantity, ureg

if TYPE_CHECKING:
    import pandas as pd

    from sbmlsim.panel import Panel, PanelData
    from sbmlsim.serve.runner import DeliveredUnit, ModelEntry, PanelUnits

logger = logging.getLogger(__name__)

__all__ = ["COUNT", "PanelSeries", "panel_data"]

#: the column of the number of subjects behind a point, as a table of a study
#: carries it; a series of a dataset without it has no counts
COUNT: Final = "count"

#: the suffixes of the columns of the error of a column, which a dataset
#: measures in the unit of that column (`cp_sd` is in the unit of `cp`); the
#: relative `cv` of `sbmlsim.data.UNCERTAINTY_SUFFIXES` is no such column and
#: carries no unit of its own, see `_unit_of`
_ERROR_SUFFIXES: Final = ("sd", "se", "min", "max")

#: the errors the parsing of a quantity raises: the ones of pint, whose
#: `UndefinedUnitError` is an `AttributeError`, and the ones of its parser, see
#: `sbmlsim.units.short_unit`
_QUANTITY_ERRORS: Final = (PintError, TokenError, ValueError, TypeError, AttributeError)


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
        x_unit: the unit of the times, `None` for times without one, i.e. a
            model whose time has no unit and a dataset whose time column has
            none either, whose times are delivered as they are. It is the
            `units["x"]` a simulation of the same panel reports, which is
            `null` for no unit as well (`sbmlsim.serve.payload`).
        unit: the unit of the values, the one of the curve the series belongs
            under, `None` for a curve the model gives no unit and for a series
            which is delivered in the unit of its dataset and whose column has
            none. It is the unit that curve is drawn in, the `units[<curve>]`
            a simulation of the same panel reports.
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


def panel_data(entry: ModelEntry, panel: Panel, units: PanelUnits) -> list[PanelSeries]:
    """Read the study data a panel declares, see the module.

    Args:
        entry: the model of the panel, whose project holds the data.
        panel: the panel.
        units: the units the panel is delivered in, resolved in the registry
            of its model: `sbmlsim.serve.runner.PanelRunner.units` of the same
            panel, so the times are on the axis of its curves and the values
            in the units of the curves they belong under.

    Returns:
        One series per declaration of the panel, in the order the panel
        declares them; none for a panel without study data.

    Raises:
        FileNotFoundError: for a dataset which is no file of the data of the
            project.
        KeyError: for `units` of another panel, which has no unit of a series
            of this one.
        ValueError: naming the series and what is wrong with it, for a dataset
            of a project without a `data_path`, a selection which no row has,
            a column the dataset has not or which carries no numbers, a
            conversion which is no quantity, a series whose times cannot be
            delivered in the unit of the x and one which cannot be delivered
            in the unit it is declared in.
    """
    return [
        _series(entry, declared, units.x, units.series[declared.id])
        for declared in panel.data
    ]


def _series(
    entry: ModelEntry,
    declared: PanelData,
    x: DeliveredUnit,
    target: DeliveredUnit | None,
) -> PanelSeries:
    """Read one series of the study data of a panel, see `panel_data`.

    Args:
        entry: the model of the panel, whose project holds the data.
        declared: the declaration of the series.
        x: the unit the times are delivered in, the one of the x of the panel.
        target: the unit the values are delivered in, the unit of its dataset
            without.

    Returns:
        The series.

    Raises:
        ValueError: naming the series and what is wrong with it, see
            `panel_data`.
    """
    frame = load_pkdb_dataframe(declared.dataset, data_path=entry.project.data_path)
    dset = DataSet.from_df(_selected(frame, declared))
    conversion = _conversion(declared)
    times, time_unit = _delivered(dset, declared, declared.x, x)
    values, unit = _delivered(dset, declared, declared.y, target, conversion)
    keep = np.isfinite(times) & np.isfinite(values)
    errors: list[float | None] | None = None
    if declared.yerr is not None:
        deviations, error_unit = _delivered(
            dset, declared, declared.yerr, target, conversion
        )
        if error_unit != unit:
            deviations = _rescaled(declared, deviations, error_unit, unit)
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
        count=_counts(dset, declared, keep),
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
    except _QUANTITY_ERRORS as err:
        raise ValueError(
            f"The conversion '{declared.conversion}' of the series "
            f"'{declared.id}' is no quantity: {err}"
        ) from err


def _delivered(
    dset: DataSet,
    declared: PanelData,
    column: str,
    target: DeliveredUnit | None,
    conversion: Quantity | None = None,
) -> tuple[np.ndarray, str | None]:
    """Get the values of a column of a series and the unit they are in.

    Args:
        dset: the rows of the series with the units of their columns.
        declared: the declaration of the series.
        column: the column which is read.
        target: the unit the values are delivered in, resolved by the caller,
            their own without; a target of no unit
            (`sbmlsim.serve.runner.UNITLESS`) delivers them as they are, see
            `_plain`.
        conversion: the quantity the values are multiplied by, none without.

    Returns:
        The values and their unit: the text of `target`, the unit of the
        column with the `conversion` applied to it, or the unit of the column
        as it is, `None` for a column which has none.

    Raises:
        ValueError: naming the series, for a column the dataset has not, one
            which carries no numbers, one without a unit which is converted,
            one which carries a unit the panel has not (`_plain`) and for
            values which cannot be delivered in `target`.
    """
    values = _values(dset, declared, column)
    unit = _unit_of(dset, column)
    if target is not None and target.unit is None:
        return _plain(declared, column, values, unit, conversion), None
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
        quantity = quantity.to(target.unit)
    except PintError as err:
        measured = unit if conversion is None else f"{unit} * {declared.conversion}"
        raise ValueError(
            f"The series '{declared.id}' is measured in '{measured}' and cannot "
            f"be delivered in '{target.text}': {err}"
        ) from err
    return np.asarray(quantity.magnitude, dtype=float), target.text


def _plain(
    declared: PanelData,
    column: str,
    values: np.ndarray,
    unit: str | None,
    conversion: Quantity | None,
) -> np.ndarray:
    """Get the values of a column which the panel delivers without a unit.

    The panel delivers the column without a unit, which is a curve whose
    selection the model gives none and a model whose time has none
    (`sbmlsim.serve.runner.UNITLESS`), so the dataset must carry none either:
    there is nothing the values could be converted into, and a dataset which
    does carry one is no one scale with the curve or the axis it is drawn on.

    Args:
        declared: the declaration of the series.
        column: the column which is read.
        values: the values of the column.
        unit: the unit of the column, `None` for none.
        conversion: the quantity the values are multiplied by, none without.

    Returns:
        The values as they are.

    Raises:
        ValueError: naming the series, its column and the unit, for a column
            which carries a unit and for a series which declares a conversion.
    """
    if unit is not None:
        raise ValueError(
            f"The column '{column}' of the series '{declared.id}' is measured "
            f"in '{unit}' and the panel delivers it without a unit, which its "
            f"model gives none either, so the two are no one scale."
        )
    if conversion is not None:
        raise ValueError(
            f"The series '{declared.id}' is converted by "
            f"'{declared.conversion}' and the panel delivers its column "
            f"'{column}' without a unit, so there is nothing to convert."
        )
    return values


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
        absolute error (`cp_sd` is in the unit of `cp`, see `_ERROR_SUFFIXES`),
        `None` for a column without one, a relative `cp_cv` among them.
    """
    uinfo = dset.uinfo
    if column in uinfo:
        return str(uinfo[column]) or None
    base, _, suffix = column.rpartition("_")
    if base and suffix in _ERROR_SUFFIXES and base in uinfo:
        return str(uinfo[base]) or None
    return None


def _rescaled(
    declared: PanelData, values: np.ndarray, measured: str | None, target: str | None
) -> np.ndarray:
    """Get the error of a series in the unit its values are delivered in.

    A dataset may measure the error of a column in another unit than the
    column itself (`cp` in `mg/l` and `cp_sd` in `ug/l`), which is one error
    bar only in the unit of the values.

    Both units are read in the registry of the package, which is where they
    come from: a series which is delivered in a unit of its panel takes that
    unit for its values and for their error alike (`_delivered`), so the two
    differ only for a series which is delivered in the units of its dataset.

    Args:
        declared: the declaration of the series.
        values: the error of every value.
        measured: the unit of the error, `None` for none.
        target: the unit of the values, `None` for none.

    Returns:
        The error in the unit of the values.

    Raises:
        ValueError: naming the series and both units, for an error which has
            no unit, for values which have none and for an error which cannot
            be converted into the unit of the values.
    """
    if measured is None or target is None:
        raise ValueError(
            f"The error '{declared.yerr}' of the series '{declared.id}' is "
            f"measured in {_named(measured)} and its values '{declared.y}' in "
            f"{_named(target)}, which are no one error bar."
        )
    try:
        quantity = ureg.Quantity(values, measured).to(target)
    except PintError as err:
        raise ValueError(
            f"The error '{declared.yerr}' of the series '{declared.id}' is "
            f"measured in '{measured}' and cannot be delivered in the '{target}' "
            f"of its values '{declared.y}': {err}"
        ) from err
    return np.asarray(quantity.magnitude, dtype=float)


def _named(unit: str | None) -> str:
    """Get a unit as a message of this module names it, `no unit` for none."""
    return f"'{unit}'" if unit is not None else "no unit"


def _unit_text(quantity: Quantity) -> str:
    """Get the unit of a quantity as a dataset writes it, e.g. `micromole/liter`."""
    return str(quantity.units).replace("**", "^").replace(" ", "")


def _counts(dset: DataSet, declared: PanelData, keep: np.ndarray) -> list[int] | None:
    """Get the number of subjects behind every point of a series, none without.

    Args:
        dset: the rows of the series.
        declared: the declaration of the series.
        keep: the rows which are points of the series.

    Returns:
        The counts of these rows, `None` for a dataset without the column
        `COUNT`, for a series without a point and for one whose points do not
        all carry a count, which is no uncertainty of a point and which is
        reported.

    Raises:
        ValueError: naming the series, for a column `COUNT` which carries
            values which are no numbers, see `_values`.
    """
    if COUNT not in dset.columns:
        return None
    counts = _values(dset, declared, COUNT)[keep]
    if counts.size == 0:
        return None
    if not bool(np.isfinite(counts).all()):
        logger.warning(
            "The series '%s' of the dataset '%s' has points without a count in "
            "the column '%s', so it carries none.",
            declared.id,
            declared.dataset,
            COUNT,
        )
        return None
    return [int(count) for count in counts]
