"""The references of the targets of a design.

The reference of a target is the value the model gives it after the
pre-initialization of a simulation: the changes of the model and of the
simulation are set, and the initial assignments which read them are evaluated
again. A relative distribution is centred on it, and a local design varies
around it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import libsbml

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import Number
from sbmlsim.simulator.plan import preinit_targets
from sbmlsim.simulator.simulator import ModelLike, Simulator
from sbmlsim.units import ureg

#: a reference below this magnitude is zero for `parameters_of`
ZERO: float = 1e-8


def _loaded(model: ModelLike) -> RoadrunnerSBMLModel:
    """Get a loaded model; a loaded one is used as it is, with its settings."""
    if isinstance(model, RoadrunnerSBMLModel) and model.r is not None:
        return model
    return Simulator().load(model)


def _nothing(sid: str) -> bool:
    """Exclude nothing."""
    return False


def references(
    model: ModelLike,
    targets: Sequence[str],
    simulation: Simulation | None = None,
) -> dict[str, Number]:
    """Get the value every target has after the pre-initialization.

    The simulation is compiled with the changes of the model as defaults and
    the model is initialized with its plan, which every simulation of the
    model does anyway, so the model is left initialized.

    Args:
        model: the model, loaded or a source.
        targets: the targets, selections of roadrunner, e.g. `k1`, `S` or `[S]`.
        simulation: the simulation whose pre-initialization counts,
            `Simulation(end=1)` by default.

    Returns:
        Target -> its value, a quantity in the unit of the target in the model
        or a float without a unit.

    Raises:
        ValueError: if a target is no selection of the model.
    """
    loaded = _loaded(model)
    plan = Simulator().compile(loaded, simulation or Simulation(end=1.0))
    loaded.initialize(preinit_targets(plan))
    values: dict[str, Number] = {}
    for target in targets:
        if not loaded.has_selection(target):
            raise ValueError(f"'{target}' is no target of the model.")
        value = float(loaded.r_loaded.getValue(target))
        unit = loaded.uinfo.get(target, "") or ""
        values[target] = ureg.Quantity(value, unit) if unit else value
    return values


def parameters_of(
    model: ModelLike,
    *,
    species: bool = False,
    exclude: Callable[[str], bool] | Collection[str] | None = None,
    exclude_zero: bool = True,
    simulation: Simulation | None = None,
) -> list[str]:
    """List the constant parameters of a model, which an analysis of all parameters varies.

    Args:
        model: the model, loaded or a source.
        species: list the species as well (their amounts).
        exclude: ids, or a function which is true for an id, to leave out.
        exclude_zero: leave out a target whose reference is below `ZERO` in
            magnitude, which a relative change does not change.
        simulation: the simulation whose pre-initialization gives the
            references of `exclude_zero`.

    Returns:
        The ids, sorted, without the helper parameters sbmlsim adds.
    """
    loaded = _loaded(model)
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(loaded.r_loaded.getSBML())
    sbml_model: libsbml.Model = doc.getModel()
    ids = [p.getId() for p in sbml_model.getListOfParameters() if p.getConstant()]
    if species:
        ids.extend(s.getId() for s in sbml_model.getListOfSpecies())
    helpers = set(loaded.parameters)
    excluded: Callable[[str], bool]
    if exclude is None:
        excluded = _nothing
    elif isinstance(exclude, Collection):
        excluded = set(exclude).__contains__
    else:
        excluded = exclude
    ids = sorted(sid for sid in ids if sid not in helpers and not excluded(sid))
    if exclude_zero:
        refs = references(loaded, ids, simulation)
        ids = [
            sid
            for sid in ids
            if abs(float(getattr(refs[sid], "magnitude", refs[sid]))) >= ZERO
        ]
    return ids
