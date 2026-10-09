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
import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.result import ScanResult
from sbmlsim.result.scan import POINT
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2, SensitivityResult
from sbmlsim.simulation.sampling import Design
from sbmlsim.simulation.sampling.designs import _problem, unit_cube
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


def _unit_points(
    result: ScanResult, dim: str, design: Design
) -> tuple[list[str], np.ndarray]:
    """Get the parameters and the unit cube of a SALib design, checked against the result.

    Args:
        result: the result of the scan.
        dim: the dimension of the design.
        design: the record of the design.

    Returns:
        The targets of the design and its unit cube, a row per point.

    Raises:
        ValueError: if the dimension has another number of points than the cube.
    """
    parameters = list(design.distributions)
    cube = unit_cube(design, len(parameters))
    points = result.ds.sizes[dim]
    if cube.shape[0] != points:
        raise ValueError(
            f"The design of the dimension '{dim}' has {cube.shape[0]} points by its "
            f"record, the result has {points} points."
        )
    return parameters, cube


def _global(
    result: ScanResult,
    methods: set[str],
    dim: str | None,
    observables: Sequence[str] | None,
    analyze: Callable[[Design, np.ndarray, np.ndarray], dict[str, np.ndarray]],
    keys: Sequence[str],
    unitless: Collection[str],
    constant: float,
    pair_keys: Sequence[str] = (),
) -> SensitivityResult:
    """Compute the indices of a SALib method for every element, see the module.

    Args:
        result: the result of the scan.
        methods: the methods of the design.
        dim: the dimension of the design.
        observables: the observables.
        analyze: `analyze(design, cube, y)` gives the indices of one element, `y`
            the values of its points; a `d` vector per key and a `d x d` matrix
            per pair key (second order, only when the design has it).
        keys: the indices over the parameters.
        unitless: the indices which have no unit (the others have the unit of
            the observable).
        constant: the value of every index of a constant element.
        pair_keys: the indices over pairs of parameters, present when the
            design has second order indices.

    Returns:
        The sensitivity result.
    """
    dim, design = design_of(result, methods, dim)
    parameters, cube = _unit_points(result, dim, design)
    d = len(parameters)
    pairs = list(pair_keys) if design.options.get("second_order") else []
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    failed = 0
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        flat = values.reshape(values.shape[0], -1)
        out: dict[str, np.ndarray] = {
            key: np.full((d, flat.shape[1]), np.nan) for key in keys
        }
        out.update({key: np.full((d, d, flat.shape[1]), np.nan) for key in pairs})
        for k in range(flat.shape[1]):
            y = flat[:, k]
            if not np.isfinite(y).all():
                failed += 1
                continue
            if np.ptp(y) == 0.0:
                for key in out:
                    out[key][..., k] = constant
                continue
            found = analyze(design, cube, y)
            for key in out:
                out[key][..., k] = np.asarray(found[key], dtype=float)
        for key, array in out.items():
            lead = [PARAMETER, PARAMETER_2] if key in pairs else [PARAMETER]
            shape = array.shape[: len(lead)] + values.shape[1:]
            indices[f"{name}.{key}"] = ([*lead, *others], array.reshape(shape))
            base = key.removesuffix("_conf")
            units[f"{name}.{key}"] = (
                "dimensionless" if base in unitless else result.units.get(name, "")
            )
    if failed:
        logger.warning(
            "%s elements of the result contain a failed simulation (NaN); their "
            "indices are NaN.",
            failed,
        )
    return _assemble(
        result, dim, design.method, dict(design.options), parameters, indices, units
    )


def sobol(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
) -> SensitivityResult:
    """Compute the Sobol indices of a scan with a Sobol design, see the module.

    The indices are `S1` and `ST` (and `S2` over `(parameter, parameter_2)` when
    the design has second order) with their bootstrap intervals `<index>_conf`,
    computed by SALib on the unit cube the record recreates, seeded with the
    seed of the design. A constant element has `NaN` indices.

    Args:
        result: the result of a scan with a `sampling.sobol` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence intervals.
        num_resamples: the number of bootstrap resamples.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no Sobol design or several without `dim`,
            an observable is a ragged timecourse or the dimension has another
            number of points than its record.
    """
    from SALib.analyze import sobol as analyzer

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        found = analyzer.analyze(
            _problem(cube.shape[1]),
            y,
            calc_second_order=bool(design.options["second_order"]),
            num_resamples=num_resamples,
            conf_level=conf_level,
            print_to_console=False,
            seed=design.options["seed"],
        )
        return {
            key: found[key]
            for key in ("S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf")
            if key in found
        }

    return _global(
        result,
        {"sobol"},
        dim,
        observables,
        analyze,
        ("S1", "S1_conf", "ST", "ST_conf"),
        {"S1", "ST", "S2"},
        np.nan,
        ("S2", "S2_conf"),
    )


def fast(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
) -> SensitivityResult:
    """Compute the FAST indices of a scan with a FAST design, see the module.

    The indices are `S1` and `ST` with `S1_conf` and `ST_conf`, computed by
    SALib on the unit cube the record recreates. A constant element has `NaN`
    indices. SALib documents that the bootstrap intervals of FAST are
    unreliable, so `S1_conf` and `ST_conf` are indicative only.

    Args:
        result: the result of a scan with a `sampling.fast` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence intervals.
        num_resamples: the number of bootstrap resamples.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no FAST design or several without `dim`,
            an observable is a ragged timecourse or the dimension has another
            number of points than its record.
    """
    from SALib.analyze import fast as analyzer

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        with warnings.catch_warnings():
            # SALib warns on every call that the bootstrap intervals are unreliable,
            # which the docstring of `fast` says
            warnings.filterwarnings(
                "ignore", message="FAST confidence intervals", category=UserWarning
            )
            found = analyzer.analyze(
                _problem(cube.shape[1]),
                y,
                M=int(design.options["m"]),
                num_resamples=num_resamples,
                conf_level=conf_level,
                print_to_console=False,
                seed=design.options["seed"],
            )
        return {key: found[key] for key in ("S1", "S1_conf", "ST", "ST_conf")}

    return _global(
        result,
        {"fast"},
        dim,
        observables,
        analyze,
        ("S1", "S1_conf", "ST", "ST_conf"),
        {"S1", "ST"},
        np.nan,
    )


def morris(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
) -> SensitivityResult:
    """Compute the elementary effects of a scan with a Morris design, see the module.

    The indices are `mu`, `mu_star`, `sigma` and `mu_star_conf`, computed by
    SALib on the unit cube the record recreates (`scaled=False`). They are the
    change of the observable per step of the unit cube, so they have the unit
    of the observable. A constant element has zero effects.

    Args:
        result: the result of a scan with a `sampling.morris` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence interval.
        num_resamples: the number of bootstrap resamples.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no Morris design or several without `dim`,
            an observable is a ragged timecourse or the dimension has another
            number of points than its record.
    """
    from SALib.analyze import morris as analyzer

    keys = ("mu", "mu_star", "sigma", "mu_star_conf")

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        found = analyzer.analyze(
            _problem(cube.shape[1]),
            cube,
            y,
            num_resamples=num_resamples,
            conf_level=conf_level,
            scaled=False,
            print_to_console=False,
            num_levels=int(design.options["levels"]),
            seed=design.options["seed"],
        )
        return {key: found[key] for key in keys}

    return _global(result, {"morris"}, dim, observables, analyze, keys, set(), 0.0)
