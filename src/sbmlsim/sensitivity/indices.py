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
  with their confidence intervals;
- `fast`: `S1`, `ST` of the extended FAST;
- `morris`: the elementary effects `mu`, `mu_star`, `sigma`, `mu_star_conf`
  on the unit cube the record recreates.

The three global analyses estimate every element of an observable in one pass
with `sbmlsim.sensitivity.estimators`, the estimators of SALib over the
elements; SALib itself draws the points of a design
(`sbmlsim.simulation.sampling`) and is the oracle of the tests.

An element whose points contain a failed simulation (`NaN`) has `NaN`
indices, one warning counts them. An element of a global analysis which is
constant, i.e. whose values vary by no more than the error of the integrator,
has `NaN` indices of variance (Sobol, FAST) and zero effects (Morris), see
`constant_tolerance`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.result import ScanResult
from sbmlsim.result.scan import POINT
from sbmlsim.sensitivity.estimators import (
    fast_indices,
    morris_effects,
    sobol_indices,
)
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2, SensitivityResult
from sbmlsim.simulation.sampling import Design
from sbmlsim.simulation.sampling.designs import unit_cube
from sbmlsim.units import ureg

logger = logging.getLogger(__name__)

#: the relative tolerance of the integrator of a result which does not record
#: it, the default of `sbmlsim.simulator.Simulator`
RELATIVE_TOLERANCE = 1e-10

#: the smallest relative tolerance of the integrator a tolerance is derived
#: from: below it the error of the integrator does not decrease any more (about
#: `5e-9` of a value at `1e-10` and `1e-8` at `1e-12`, double precision)
SMALLEST_RELATIVE_TOLERANCE = 1e-10

#: the tolerance of a constant element in relative tolerances of the
#: integrator: the error of a value is up to about 50 relative tolerances
#: (`5e-9` at `1e-10`, the saturated and the conserved values of the chain
#: S1 -> S2 -> S3), the factor keeps a margin of 20 over it
CONSTANT_FACTOR = 1000.0


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
    units = {**units, PARAMETER: ""}
    if PARAMETER_2 in used:
        coords[PARAMETER_2] = parameters
        units[PARAMETER_2] = ""
    for name, coord in result.ds.coords.items():
        # the labels of the other dimensions, the values a dimension changes
        # along them and the time, not the values along the design
        if set(coord.dims) <= used - {PARAMETER, PARAMETER_2}:
            coords[str(name)] = (coord.dims, coord.values)
            units[str(name)] = result.units.get(str(name), "")
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
    """Get the unit of a ratio of two units in its short form, e.g. `h*mg/l`.

    Args:
        numerator: the unit of the numerator.
        denominator: the unit of the denominator.

    Returns:
        The unit of the ratio, `""` where one is unknown.
    """
    if not numerator or not denominator:
        return ""
    unit = ureg.Unit(numerator) / ureg.Unit(denominator)
    return f"{unit:~C}" if not unit.dimensionless else "dimensionless"


def _warn_failed(failed: int) -> None:
    """Log one warning which counts the elements with a failed simulation.

    Args:
        failed: the number of elements whose points contain a `NaN`.
    """
    if failed:
        logger.warning(
            "%s elements of the result contain a failed simulation (NaN); the "
            "indices which use it are NaN.",
            failed,
        )


def local(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute the local sensitivities of a scan with a local design, see the module.

    `raw = (y(+) - y(-)) / (2 delta p_ref)` in the unit of the observable per
    unit of the parameter and `normalized = (y(+) - y(-)) / (2 delta y_ref)`,
    with `y_ref` the value at the reference. A zero `y_ref` gives a `NaN`
    `normalized`; a parameter whose reference is zero is not moved by the
    design, so both of its indices are `NaN`. A variable has one unit, so `raw`
    has one only when all targets share a unit (else `""`). An element whose
    points contain a failed simulation (`NaN`) has `NaN` indices where they
    use such a point, one warning counts the elements.

    A step which the design clamped to the domain of its target (`sbmlsim.domains`,
    the `clamped` of its record) is differentiated over the two values the
    design reached instead of `2 delta p_ref`, so the difference of a clamped
    parameter is the one-sided difference it is; a target which the design
    could not move at all has `NaN` indices.

    Args:
        result: the result of a scan with a `sampling.local` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.

    Returns:
        The indices `raw` and `normalized` of every observable.

    Raises:
        ValueError: if the result has no local design or several without
            `dim`, an observable is a ragged timecourse, or the labels of the
            dimension are not the ones of the design (a cut result).
    """
    dim, design = design_of(result, {"local"}, dim)
    delta = float(design.options["delta"])
    targets = [str(t) for t in design.options["targets"]]
    clamped: dict[str, tuple[float, float]] = {
        str(target): (float(span[0]), float(span[1]))
        for target, span in (design.options.get("clamped") or {}).items()
    }
    labels = [str(label) for label in result.ds[dim].values.tolist()]
    expected = ["reference", *(f"{t}{sign}" for t in targets for sign in "+-")]
    if sorted(labels) != sorted(expected):
        raise ValueError(
            f"The labels of the dimension '{dim}' must be {expected}, the points of "
            f"its design, not {labels}; the result was cut or its labels changed."
        )
    reference = labels.index("reference")
    target_units = {str(design.references[t]["unit"]) for t in targets}
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    failed = 0
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        failed += int((~np.isfinite(values)).any(axis=0).sum())
        y_ref = values[reference]
        raw = []
        normalized = []
        for target in targets:
            up = values[labels.index(f"{target}+")]
            down = values[labels.index(f"{target}-")]
            p_ref = float(design.references[target]["value"])
            # the step of a target whose domain clamped it ends at the values
            # of the record, so the difference is taken over those
            dp, dy = 2.0 * delta * p_ref, 2.0 * delta * y_ref
            span = clamped.get(target)
            if span is not None:
                dp = span[0] - span[1]
                dy = dp * y_ref / p_ref if p_ref != 0.0 else 0.0
            if p_ref == 0.0 or dp == 0.0:
                # the design does not move a parameter whose reference is zero
                # and none whose domain clamped both of its steps to one value
                raw.append(np.full_like(up, np.nan))
                normalized.append(np.full_like(up, np.nan))
                continue
            # a non-finite value at the reference or a moved point is a failed point
            usable = np.isfinite(up) & np.isfinite(down) & np.isfinite(y_ref)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                raw.append(np.where(usable, (up - down) / dp, np.nan))
                normalized.append(
                    np.where(usable & (y_ref != 0.0), (up - down) / dy, np.nan)
                )
        dims = [PARAMETER, *others]
        indices[f"{name}.raw"] = (dims, np.stack(raw))
        indices[f"{name}.normalized"] = (dims, np.stack(normalized))
        unit = result.units.get(name, "")
        shared = next(iter(target_units)) if len(target_units) == 1 else ""
        units[f"{name}.raw"] = _unit_ratio(unit, shared)
        units[f"{name}.normalized"] = "dimensionless"
    _warn_failed(failed)
    options: dict[str, Any] = {"delta": delta, "targets": targets}
    if clamped:
        options["clamped"] = {t: list(span) for t, span in clamped.items()}
        logger.warning(
            "The steps of the parameters %s were clamped to their domain, so "
            "their differences are the one-sided differences of the values the "
            "design reached.",
            sorted(clamped),
        )
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
        ValueError: if the dimension has another number of points than the cube
            or its labels are not `0..n-1`, the rows of the cube.
    """
    parameters = list(design.distributions)
    cube = unit_cube(design, len(parameters))
    points = result.ds.sizes[dim]
    if cube.shape[0] != points:
        raise ValueError(
            f"The design of the dimension '{dim}' has {cube.shape[0]} points by its "
            f"record, the result has {points} points."
        )
    labels = np.asarray(result.ds[dim].values)
    if not np.array_equal(np.sort(labels), np.arange(points)):
        raise ValueError(
            f"The labels of the dimension '{dim}' must be 0..{points - 1}, the rows "
            f"of its design; the result was cut or its labels changed."
        )
    return parameters, cube


def _elements(flat: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Get the elements an analysis estimates and which of them vary.

    Args:
        flat: the values, a row per point and a column per element.
        tolerance: the tolerance of a constant element, see
            `constant_tolerance`.

    Returns:
        The positions of the elements of no failed point, and which of them
        have values which differ by more than the tolerance times their
        largest magnitude; the others count as constant.
    """
    usable = np.flatnonzero(np.isfinite(flat).all(axis=0))
    columns = flat[:, usable]
    span = np.ptp(columns, axis=0)
    return usable, span > tolerance * np.max(np.abs(columns), axis=0, initial=0.0)


def _check(conf_level: float, num_resamples: int) -> None:
    """Check the arguments of the bootstrap of an analysis.

    Args:
        conf_level: the level of the confidence intervals.
        num_resamples: the number of bootstrap resamples.

    Raises:
        ValueError: if the level is not in (0, 1) or there is no resample.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), not {conf_level}.")
    if num_resamples < 1:
        raise ValueError(f"num_resamples must be at least 1, not {num_resamples}.")


def constant_tolerance(result: ScanResult, tolerance: float | None = None) -> float:
    """Get the relative range up to which an element of a result is constant.

    An element of a global analysis is constant when the range of its values
    is at most `tolerance * max|y|`: the integrator solves a value only up to
    its error, so an element which is constant except for that error (a
    saturated or a conserved value, a steady state) has no variance a Sobol or
    FAST index could share out. The tolerance is `CONSTANT_FACTOR` times the
    relative tolerance of the integrator, which the result records in
    `attrs["integrator_settings"]`, at least `SMALLEST_RELATIVE_TOLERANCE`;
    the default of `Simulator`, `1e-10`, gives `1e-7`. A result without the
    record takes `RELATIVE_TOLERANCE`. A value near zero, at the level of the
    absolute tolerance (a species which has decayed), varies by its error
    relative to itself and is not caught.

    Args:
        result: the result of the scan.
        tolerance: the tolerance, which wins over the record; `0` keeps every
            element which is not exactly constant.

    Returns:
        The tolerance.

    Raises:
        ValueError: if the tolerance is negative or not finite.
    """
    if tolerance is not None:
        if not (np.isfinite(tolerance) and tolerance >= 0.0):
            raise ValueError(
                f"The tolerance of a constant element is a number of at least 0, "
                f"not {tolerance}."
            )
        return float(tolerance)
    settings = result.ds.attrs.get("integrator_settings") or {}
    rtol = settings.get("relative_tolerance", RELATIVE_TOLERANCE)
    if isinstance(rtol, bool) or not isinstance(rtol, int | float):
        rtol = RELATIVE_TOLERANCE
    return CONSTANT_FACTOR * max(float(rtol), SMALLEST_RELATIVE_TOLERANCE)


def _global(
    result: ScanResult,
    methods: set[str],
    dim: str | None,
    observables: Sequence[str] | None,
    analyze: Callable[[Design, np.ndarray, np.ndarray, int], dict[str, np.ndarray]],
    *,
    keys: Sequence[str],
    dimensionless: bool,
    constant: float,
    bootstrap: Callable[[int], int],
    conf_level: float,
    num_resamples: int,
    tolerance: float | None,
    pair_keys: Sequence[str] = (),
) -> SensitivityResult:
    """Compute the indices of a method for every element, see the module.

    Every element of an observable is estimated in one pass, with the same
    bootstrap resamples, so its indices and intervals do not depend on the
    other elements. The points are in the order of the rows of the cube, i.e.
    of the labels `0..n-1` of the dimension. An element whose values differ by
    no more than the tolerance of `constant_tolerance` times their largest
    magnitude counts as constant and is not estimated, and neither is one
    whose points contain a failed simulation.

    Args:
        result: the result of the scan.
        methods: the methods of the design.
        dim: the dimension of the design.
        observables: the observables.
        analyze: `analyze(design, cube, y, seed)` gives the indices of the
            elements of `y`, the values of the points of every element in a
            column of its own; a `(d, elements)` array per key and a
            `(d, d, elements)` array per pair key (second order, only when the
            design has it).
        keys: the indices over the parameters.
        dimensionless: whether the indices are dimensionless (else they have
            the unit of the observable).
        constant: the value of every index of a constant element.
        bootstrap: the seed of the bootstrap of the intervals from the seed of
            the design.
        conf_level: the level of the confidence intervals.
        num_resamples: the number of bootstrap resamples.
        tolerance: the tolerance of a constant element, see
            `constant_tolerance`.
        pair_keys: the indices over pairs of parameters, present when the
            design has second order indices.

    Returns:
        The sensitivity result.
    """
    _check(conf_level, num_resamples)
    tolerance = constant_tolerance(result, tolerance)
    dim, design = design_of(result, methods, dim)
    parameters, cube = _unit_points(result, dim, design)
    seed = bootstrap(int(design.options["seed"]))
    d = len(parameters)
    pairs = list(pair_keys) if design.options.get("second_order") else []
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    failed = 0
    order = np.argsort(np.asarray(result.ds[dim].values))
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        flat = values[order].reshape(values.shape[0], -1)
        out: dict[str, np.ndarray] = {
            key: np.full((d, flat.shape[1]), np.nan) for key in keys
        }
        out.update({key: np.full((d, d, flat.shape[1]), np.nan) for key in pairs})
        # the elements which are estimated: the ones of no failed point whose
        # values are not constant; a failed element keeps its NaN indices
        usable, varying = _elements(flat, tolerance)
        failed += int(flat.shape[1] - usable.size)
        for key in out:
            out[key][..., usable[~varying]] = constant
        estimated = usable[varying]
        if estimated.size:
            found = analyze(design, cube, flat[:, estimated], seed)
            for key in out:
                out[key][..., estimated] = np.asarray(found[key], dtype=float)
        for key, array in out.items():
            lead = [PARAMETER, PARAMETER_2] if key in pairs else [PARAMETER]
            shape = array.shape[: len(lead)] + values.shape[1:]
            indices[f"{name}.{key}"] = ([*lead, *others], array.reshape(shape))
            units[f"{name}.{key}"] = (
                "dimensionless" if dimensionless else result.units.get(name, "")
            )
    _warn_failed(failed)
    options = {
        **design.options,
        "conf_level": conf_level,
        "num_resamples": num_resamples,
        "bootstrap_seed": seed,
        "tolerance": tolerance,
    }
    return _assemble(result, dim, design.method, options, parameters, indices, units)


def sobol(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
    tolerance: float | None = None,
) -> SensitivityResult:
    """Compute the Sobol indices of a scan with a Sobol design, see the module.

    The indices are `S1` and `ST` (and `S2` over `(parameter, parameter_2)` when
    the design has second order) with their bootstrap intervals `<index>_conf`,
    the estimators of Saltelli on the unit cube the record recreates, see
    `sbmlsim.sensitivity.estimators`. The bootstrap is seeded with the seed of
    the design and its resamples are the ones of every element, so the
    intervals are reproducible for every seed, and the global random state is
    not touched. `S2` is the upper triangle: `NaN` on and below the diagonal,
    so `S2` of `x1` and `x3` is at `(x1, x3)` only. A constant element (within
    `tolerance`) has `NaN` indices. The options of the result are those of the
    design and the arguments of the analysis (`conf_level`, `num_resamples`,
    `bootstrap_seed`, `tolerance`).

    Args:
        result: the result of a scan with a `sampling.sobol` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence intervals, in (0, 1).
        num_resamples: the number of bootstrap resamples, at least 1.
        tolerance: the relative range up to which an element is constant,
            derived from the relative tolerance of the integrator by default,
            see `constant_tolerance`.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no Sobol design or several without `dim`,
            an observable is a ragged timecourse, the dimension has another
            number of points than its record or labels other than `0..n-1`, or
            an argument is out of range.
    """

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray, seed: int
    ) -> dict[str, np.ndarray]:
        return sobol_indices(
            y,
            d=cube.shape[1],
            second_order=bool(design.options["second_order"]),
            conf_level=conf_level,
            num_resamples=num_resamples,
            seed=seed,
        )

    return _global(
        result,
        {"sobol"},
        dim,
        observables,
        analyze,
        keys=("S1", "S1_conf", "ST", "ST_conf"),
        dimensionless=True,
        constant=np.nan,
        bootstrap=int,
        conf_level=conf_level,
        num_resamples=num_resamples,
        tolerance=tolerance,
        pair_keys=("S2", "S2_conf"),
    )


def _fast_seed(seed: int) -> int:
    """Get the non-zero seed of the bootstrap of FAST from the seed of the design.

    SALib seeded its generator only for a seed which is true, so the seed 0 of
    a design would have left the global random state in charge. The estimators
    of sbmlsim seed a generator of their own, which 0 is a seed of as well, and
    the derived seed is kept so the intervals of a result do not change.

    Args:
        seed: the seed of the design.

    Returns:
        A seed in `[1, 2**31)`.
    """
    return int(np.random.default_rng(seed).integers(1, 2**31))


def fast(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
    tolerance: float | None = None,
) -> SensitivityResult:
    """Compute the FAST indices of a scan with a FAST design, see the module.

    The indices are `S1` and `ST` with `S1_conf` and `ST_conf`, the spectral
    estimators of the extended FAST on the unit cube the record recreates, see
    `sbmlsim.sensitivity.estimators`. The windows of the bootstrap are drawn
    from a generator of a seed derived from the seed of the design
    (`bootstrap_seed`, which is never 0, where SALib leaves its generator
    alone) and are the ones of every element, so the intervals are
    reproducible for every seed, and the global random state is not touched. A
    constant element (within `tolerance`) has `NaN` indices. SALib documents
    that the bootstrap intervals of FAST are unreliable, so `S1_conf` and
    `ST_conf` are indicative only, and they are most of the cost of a FAST
    analysis of many elements: one transform per parameter and resample.

    Args:
        result: the result of a scan with a `sampling.fast` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence intervals, in (0, 1).
        num_resamples: the number of bootstrap resamples, at least 1.
        tolerance: the relative range up to which an element is constant,
            derived from the relative tolerance of the integrator by default,
            see `constant_tolerance`.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no FAST design or several without `dim`,
            an observable is a ragged timecourse, the dimension has another
            number of points than its record or labels other than `0..n-1`, or
            an argument is out of range.
    """

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray, seed: int
    ) -> dict[str, np.ndarray]:
        return fast_indices(
            y,
            d=cube.shape[1],
            m=int(design.options["m"]),
            conf_level=conf_level,
            num_resamples=num_resamples,
            seed=seed,
        )

    return _global(
        result,
        {"fast"},
        dim,
        observables,
        analyze,
        keys=("S1", "S1_conf", "ST", "ST_conf"),
        dimensionless=True,
        constant=np.nan,
        bootstrap=_fast_seed,
        conf_level=conf_level,
        num_resamples=num_resamples,
        tolerance=tolerance,
    )


def morris(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
    tolerance: float | None = None,
) -> SensitivityResult:
    """Compute the elementary effects of a scan with a Morris design, see the module.

    The indices are `mu`, `mu_star`, `sigma` and `mu_star_conf`, the
    elementary effects of the trajectories of the unit cube the record
    recreates, unscaled, see `sbmlsim.sensitivity.estimators`. They are the
    change of the observable divided by the jump `levels / (2 (levels - 1))` of
    the grid of levels of SALib on `[0, 1]`, i.e. per unit of that grid, so they have
    the unit of the observable: an effect is `2 (levels - 1) / levels` times the
    change of the observable over one step of a trajectory, which moves the
    probability of one parameter by `1/2` (`1.5` times for 4 levels). A
    constant element (within `tolerance`) has zero effects.

    Args:
        result: the result of a scan with a `sampling.morris` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.
        conf_level: the level of the confidence interval, in (0, 1).
        num_resamples: the number of bootstrap resamples, at least 1.
        tolerance: the relative range up to which an element is constant,
            derived from the relative tolerance of the integrator by default,
            see `constant_tolerance`.

    Returns:
        The indices of every observable.

    Raises:
        ValueError: if the result has no Morris design or several without `dim`,
            an observable is a ragged timecourse, the dimension has another
            number of points than its record or labels other than `0..n-1`, or
            an argument is out of range.
    """
    keys = ("mu", "mu_star", "sigma", "mu_star_conf")

    def analyze(
        design: Design, cube: np.ndarray, y: np.ndarray, seed: int
    ) -> dict[str, np.ndarray]:
        return morris_effects(
            cube,
            y,
            levels=int(design.options["levels"]),
            conf_level=conf_level,
            num_resamples=num_resamples,
            seed=seed,
        )

    return _global(
        result,
        {"morris"},
        dim,
        observables,
        analyze,
        keys=keys,
        dimensionless=False,
        constant=0.0,
        bootstrap=int,
        conf_level=conf_level,
        num_resamples=num_resamples,
        tolerance=tolerance,
    )
