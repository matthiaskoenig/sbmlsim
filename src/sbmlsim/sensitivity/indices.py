"""The sensitivity analyses: indices computed on the result of a scan.

A scan with a design of `sbmlsim.simulation.sampling` (`local`, `sobol`,
`fast`, `morris`) runs with `Simulator.run` and its observables; an analysis
reads the record of the design from the result, moves the dimension of the
design to the front of every kept observable and computes the indices of
every remaining array element: every label of the other dimensions of the
scan (doses, conditions) and every time point of a timecourse on a grid. A
ragged timecourse has no common time points; run the scan with `time=`.

- `local`: central differences at the reference, `raw` (the change of the
  observable per change of the parameter) and `normalized` (`d ln y / d ln p`);
- `sobol`: the first order and total Sobol indices `S1`, `ST` (and `S2`)
  with their confidence intervals, SALib;
- `fast`: `S1`, `ST` of the extended FAST, SALib;
- `morris`: the elementary effects `mu`, `mu_star`, `sigma`, `mu_star_conf`,
  SALib on the unit cube the record recreates.

An element whose points contain a failed simulation (`NaN`) has `NaN`
indices, one warning counts them.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.result import ScanResult
from sbmlsim.result.scan import POINT
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2, SensitivityResult
from sbmlsim.simulation.sampling import Design
from sbmlsim.units import ureg

logger = logging.getLogger(__name__)


def design_of(
    result: ScanResult, methods: Collection[str], dim: str | None = None
) -> tuple[str, Design]:
    """Find the dimension of a design of a method in a result.

    Args:
        result: the result of a scan with a design of `sbmlsim.simulation.sampling`.
        methods: the methods an analysis takes, e.g. `{"sobol"}`.
        dim: the id of the dimension, needed when there are several.

    Returns:
        The id of the dimension and its record.

    Raises:
        ValueError: if no dimension has a design of the methods, several have
            and `dim` is not given, or `dim` has none.
    """
    found = {
        str(d["id"]): Design.from_dict(d["design"])
        for d in result.ds.attrs.get("scan", {}).get("dimensions", [])
        if d.get("design") and d["design"]["method"] in methods
    }
    if dim is not None:
        if dim not in found:
            raise ValueError(
                f"The dimension '{dim}' has no design of {sorted(methods)}: "
                f"{sorted(found)}."
            )
        return dim, found[dim]
    if not found:
        raise ValueError(
            f"The result has no dimension with a design of {sorted(methods)}; "
            f"create one with sbmlsim.simulation.sampling."
        )
    if len(found) > 1:
        raise ValueError(
            f"The result has the designs {sorted(found)} of {sorted(methods)}; "
            f"choose one with dim=."
        )
    ((name, design),) = found.items()
    return name, design


def _observables(result: ScanResult, observables: Sequence[str] | None) -> list[str]:
    """Get the observables of an analysis; a ragged timecourse raises.

    Args:
        result: the result of the scan.
        observables: the observables, every variable of the result by default.

    Returns:
        The kept observables.

    Raises:
        ValueError: if an observable is no variable of the result or a ragged
            timecourse.
    """
    names = list(observables) if observables is not None else list(result.variables)
    for name in names:
        if name not in result.ds.data_vars:
            raise ValueError(
                f"'{name}' is no variable of the result: {list(result.variables)}."
            )
        if POINT in result.ds[name].dims:
            raise ValueError(
                f"'{name}' is a ragged timecourse, which has no common time points; "
                f"run the scan with time= or a simulation with steps or times."
            )
    return names


def _moved(result: ScanResult, name: str, dim: str) -> tuple[np.ndarray, list[str]]:
    """Get the values of a variable with the axis of the design first.

    Args:
        result: the result of the scan.
        name: the variable.
        dim: the dimension of the design.

    Returns:
        The values and the ids of the other dimensions in their order.
    """
    data = result.ds[name]
    others = [str(d) for d in data.dims if d != dim]
    values = np.asarray(data.transpose(dim, *others).values, dtype=float)
    return values, others


def _assemble(
    result: ScanResult,
    dim: str,
    method: str,
    options: dict[str, Any],
    parameters: list[str],
    indices: dict[str, tuple[list[str], np.ndarray]],
    units: dict[str, str],
) -> SensitivityResult:
    """Build the result: a variable per observable and index, the coords of the scan.

    Args:
        result: the result of the scan.
        dim: the dimension of the design.
        method: the method.
        options: the options of the analysis.
        parameters: the labels of the dimension `parameter`.
        indices: the dims and values of every variable `<observable>.<index>`.
        units: the unit of every variable.

    Returns:
        The sensitivity result.
    """
    used = {d for dims, _ in indices.values() for d in dims}
    coords: dict[str, Any] = {PARAMETER: parameters}
    if PARAMETER_2 in used:
        coords[PARAMETER_2] = parameters
    for name in used - {PARAMETER, PARAMETER_2}:
        if name in result.ds.coords:
            coords[name] = result.ds.coords[name].values
    ds = xr.Dataset(
        {name: (dims, values) for name, (dims, values) in indices.items()},
        coords=coords,
        attrs={
            "units": units,
            "method": method,
            "options": options,
            "dim": dim,
            "scan": result.ds.attrs.get("scan", {}),
        },
    )
    return SensitivityResult(ds)


def _unit_ratio(numerator: str, denominator: str) -> str:
    """Get the unit of a ratio of two units.

    Args:
        numerator: the unit of the numerator.
        denominator: the unit of the denominator.

    Returns:
        The unit of the ratio, `""` where one is unknown.
    """
    if not numerator or not denominator:
        return ""
    return str(ureg.Unit(numerator) / ureg.Unit(denominator))


def local(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute the local sensitivities of a scan with a local design, see the module.

    `raw = (y(+) - y(-)) / (2 delta p_ref)` and `normalized = (y(+) - y(-)) /
    (2 delta y_ref)`, with `y_ref` the value at the reference; a zero `y_ref`
    or a zero reference of a parameter gives `NaN`. A variable has one unit, so
    `raw` has one only when all targets share a unit (else `""`).

    Args:
        result: the result of a scan with a `sampling.local` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.

    Returns:
        The indices `raw` and `normalized` of every observable.

    Raises:
        ValueError: if the result has no local design or several without
            `dim`, or an observable is a ragged timecourse.
    """
    dim, design = design_of(result, {"local"}, dim)
    delta = float(design.options["delta"])
    targets = [str(t) for t in design.options["targets"]]
    labels = [str(label) for label in result.ds[dim].values.tolist()]
    reference = labels.index("reference")
    target_units = {str(design.references[t]["unit"]) for t in targets}
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        y_ref = values[reference]
        raw = []
        normalized = []
        for target in targets:
            up = values[labels.index(f"{target}+")]
            down = values[labels.index(f"{target}-")]
            p_ref = float(design.references[target]["value"])
            with np.errstate(divide="ignore", invalid="ignore"):
                if p_ref != 0.0:
                    raw.append((up - down) / (2.0 * delta * p_ref))
                else:
                    raw.append(np.full_like(up, np.nan))
                normalized.append(
                    np.where(y_ref != 0.0, (up - down) / (2.0 * delta * y_ref), np.nan)
                )
        dims = [PARAMETER, *others]
        indices[f"{name}.raw"] = (dims, np.stack(raw))
        indices[f"{name}.normalized"] = (dims, np.stack(normalized))
        unit = result.units.get(name, "")
        shared = next(iter(target_units)) if len(target_units) == 1 else ""
        units[f"{name}.raw"] = _unit_ratio(unit, shared)
        units[f"{name}.normalized"] = "dimensionless"
    options = {"delta": delta, "targets": targets}
    return _assemble(result, dim, "local", options, targets, indices, units)
