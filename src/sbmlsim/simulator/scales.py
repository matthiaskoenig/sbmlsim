"""The typical magnitudes of the states of a model, which scale their absolute tolerances.

An absolute tolerance relative to the scale of every state
(`AbsoluteTolerance(relative=..., scales=...)`, see
`sbmlsim.model.tolerances`) gives a small state the same relative accuracy as
a large one (#435). The initial value of a state is its scale when no scale
is given, which leaves a state which starts at 0, e.g. a drug before its
dose, at the tolerance of its kind. `typical_scales` gives the largest
magnitude of every state in test simulations, e.g. the standard doses of a
model, which includes its initial value:

```python
scales = typical_scales(simulator, model, [oral_dose, iv_dose])
tolerance = AbsoluteTolerance(relative=1e-10, scales=scales)
```

The scales are computed once and stored with the model, e.g. in the settings
of the simulator of its project as JSON (`AbsoluteTolerance.to_dict`), so
that the tolerances of a simulation never depend on its own scenario. Fixed
scales of a standard dose keep the relative accuracy of the states within
about `1e-7` from a hundredth to a hundred times the dose, see #435.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sbmlsim.engine import Engine
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.scan import Scan
from sbmlsim.simulator.simulator import ModelLike, Simulator


def typical_scales(
    simulator: Simulator,
    model: ModelLike,
    simulations: Simulation | Scan | Iterable[Simulation | Scan],
) -> dict[str, float]:
    """Get the largest magnitude of every state of a model in test simulations.

    The states are the ones roadrunner integrates, the targets of the rate
    rules and the independent floating species, whose scale is their
    amount, as the tolerances of `AbsoluteTolerance` are amounts. Every
    simulation, or every point of a scan, is simulated with the simulator in
    the steps of the integrator, and the scale of a state is the largest
    absolute value it takes in any of them, the start included.

    Args:
        simulator: the simulator, of any engine.
        model: the model.
        simulations: the test simulations, a simulation or a scan or several.

    Returns:
        The scale of every state which is not 0 in all simulations, by id, in
        the order of the states.

    Raises:
        ValueError: without a simulation.
    """
    if isinstance(simulations, Simulation | Scan):
        simulations = [simulations]
    simulations = list(simulations)
    if not simulations:
        raise ValueError("The scales of the states need at least one simulation.")
    loaded: SBMLModel = simulator.load(model)
    states = load_model(loaded, Engine.ROADRUNNER).state_ids()
    largest = np.zeros(len(states))
    for simulation in simulations:
        result = simulator.run(loaded, simulation, keep=states)
        for k, sid in enumerate(states):
            values = np.abs(np.asarray(result.ds[sid].values, dtype=float))
            values = values[np.isfinite(values)]
            if values.size:
                largest[k] = max(largest[k], float(values.max()))
    return {sid: float(v) for sid, v in zip(states, largest, strict=True) if v > 0}
