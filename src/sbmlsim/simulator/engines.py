"""The executor of the engine of a model.

`execute_plan` runs a plan on a loaded model with the executor of its engine:
`sbmlsim.simulator.executor` for roadrunner and
`sbmlsim.simulator.executor_copasi` for COPASI, which is imported when a
model of COPASI runs, so a run of roadrunner never imports COPASI. Every
simulation of the package outside the executors runs through it: the
simulator, the workers of a scan and the evaluation of a fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from sbmlsim.engine import Engine
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.model_sbml import SBMLModel
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator import executor
from sbmlsim.simulator.plan import Plan

if TYPE_CHECKING:
    from sbmlsim.model.model_copasi import CopasiSBMLModel


def execute_plan(
    plan: Plan, model: SBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a loaded model with the executor of its engine.

    Args:
        plan: the plan, compiled against the model.
        model: the loaded model, of roadrunner or of COPASI.
        selections: the columns of the output, the time first.

    Returns:
        The output of the simulation.

    Raises:
        RuntimeError: if the engine fails to integrate.
        SteadyStateError: if a presimulation finds no steady state.
        NotImplementedError: for a feature the engine does not simulate, the
            same for every plan of the model, see
            `sbmlsim.simulator.executor_copasi`.
    """
    if model.engine is Engine.COPASI:
        # imports COPASI, which only a model of COPASI needs
        from sbmlsim.simulator import executor_copasi

        return executor_copasi.execute(plan, cast("CopasiSBMLModel", model), selections)
    return executor.execute(plan, cast(RoadrunnerSBMLModel, model), selections)
