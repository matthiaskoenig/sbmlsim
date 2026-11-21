"""The local sensitivities of a simulation from the forward sensitivities.

`local` of `sbmlsim.sensitivity.indices` takes central differences of the
points of a `sampling.local` design: two simulations per parameter, whose
difference is accurate to the step of the design and to the error of the
integrations it takes the difference of. `forward` reads the derivatives the
engine core integrated along with the states instead (CVODES, `CV_STAGGERED`,
`Simulator.sensitivities`): one simulation of `k` parameters answers what
`2 k + 1` simulations of a design answer, with no step to choose and to the
tolerance of the integration.

The indices are the ones of `local`, `raw` and `normalized`, over
`(parameter, time)`:

- `raw` is `dy/dp` at every output time, in the unit of the selection per unit
  of the parameter;
- `normalized` is `d ln y / d ln p`, `dy/dp * p / y`, dimensionless, `NaN`
  where the selection is zero at a time point.

`p` is the value of the parameter at the start of the simulation, which the
run delivers as a column of its own, so a pre-initialization of the simulation
or a change of the model is the reference the indices are normalized by.

What it does not do, and `local` does: the derivatives are the ones of the
selections of the model, not of the observables of a scan, since the
derivative of a reduction (`max`, the parameters of a PK analysis) is not one
of the engine; and only the engine core carries them, for a simulation of one
interval without changes, without a presimulation and of a model without
events (the engine refuses the others). The central differences of a design
remain the general path.

A parameter which enters the model as an exponent of a state which reaches
zero, e.g. the Hill coefficient of a repressilator, has no finite derivative
there and the integration of the sensitivities fails: such a parameter is one
for a difference of simulations and not for this path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from sbmlsim.result.scan import TIME
from sbmlsim.sensitivity.indices import _unit_ratio
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult

if TYPE_CHECKING:  # pragma: no cover - the types of the caller
    from sbmlsim.simulation import Simulation
    from sbmlsim.simulator import Simulator
    from sbmlsim.simulator.plan import Plan
    from sbmlsim.simulator.simulator import ModelLike

#: the name of the method in the result
METHOD = "forward"


def forward(
    simulator: Simulator,
    model: ModelLike,
    simulation: Simulation | Plan,
    parameters: Sequence[str],
    *,
    selections: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute the local sensitivities of a simulation exactly, see the module.

    Args:
        simulator: the simulator, of the engine core
            (`Simulator(engine="core")`).
        model: the model, see `Simulator.load`.
        simulation: the simulation or a plan of it; its output times are the
            time points of the indices, so a simulation of `times=` or
            `steps=` gives a grid and one of the steps of the integrator gives
            its steps.
        parameters: the ids the derivatives are taken in, in their order: a
            parameter of the model, a compartment, or a species as its amount
            (`S`) or its concentration (`[S]`).
        selections: the selections the indices are computed for, the ones of
            the model by default.

    Returns:
        The indices `raw` and `normalized` of every selection over
        `(parameter, time)`.

    Raises:
        ValueError: if the simulation does not fit the model, no parameter is
            given, or a parameter or a selection is not one of the model.
        NotImplementedError: if the simulator is not of the engine core, or
            for a simulation or a model whose sensitivities the engine does
            not carry.
        RuntimeError: if the engine fails to integrate.
    """
    ids = list(parameters)
    if not ids:
        raise ValueError("The forward sensitivities need at least one parameter.")
    loaded = simulator.load(model)
    wanted = (
        [str(s) for s in selections]
        if selections is not None
        else [str(s) for s in (loaded.selections or []) if s != TIME]
    )
    for names, what in ((wanted, "selections"), (ids, "parameters")):
        missing = [name for name in names if not loaded.has_selection(name)]
        if missing:
            raise ValueError(
                f"The {what} {missing} are not selections of the model "
                f"'{loaded.sid or loaded.source}'."
            )
    # the parameters are columns of the run as well, so their values at the
    # start of the simulation, the point the derivatives are taken at, are the
    # ones of the engine and not of a model this process reads
    columns = list(dict.fromkeys([TIME, *wanted, *ids]))
    sensitivities = simulator.sensitivities(loaded, simulation, ids, selections=columns)
    result = sensitivities.result
    references = {pid: float(result[pid][0]) for pid in ids}
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    parameter_units = {loaded.uinfo.get(pid, "") or "" for pid in ids}
    for name in wanted:
        y = result[name]
        raw = np.stack([sensitivities[name, pid] for pid in ids])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            factors = np.array([references[pid] for pid in ids])[:, None]
            normalized = np.where(y != 0.0, raw * factors / y, np.nan)
        indices[f"{name}.raw"] = ([PARAMETER, TIME], raw)
        indices[f"{name}.normalized"] = ([PARAMETER, TIME], normalized)
        unit = loaded.uinfo.get(name, "") or ""
        shared = next(iter(parameter_units)) if len(parameter_units) == 1 else ""
        units[f"{name}.raw"] = _unit_ratio(unit, shared)
        units[f"{name}.normalized"] = "dimensionless"
    options: dict[str, Any] = {
        "parameters": ids,
        "references": [references[pid] for pid in ids],
        "selections": wanted,
    }
    ds = xr.Dataset(
        {name: (dims, values) for name, (dims, values) in indices.items()},
        coords={PARAMETER: ids, TIME: result.time},
        attrs={
            "units": {
                **units,
                PARAMETER: "",
                TIME: loaded.uinfo.get(TIME, "") or "",
            },
            "method": METHOD,
            "options": options,
            "dim": None,
            "scan": {},
        },
    )
    return SensitivityResult(ds)
