"""The executor of the engine of a model.

`execute_plan` runs a plan on a loaded model with the executor of its engine:
`sbmlsim.simulator.executor` for roadrunner,
`sbmlsim.simulator.executor_copasi` for COPASI and
`sbmlsim.simulator.executor_amici` for AMICI, which are imported when a model
of their engine runs, so a run of roadrunner never imports COPASI or AMICI.
Every simulation of the package outside the executors runs through it: the
simulator, the workers of a scan and the evaluation of a fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sbmlsim.engine import Engine
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.model_sbml import SBMLModel
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator import executor
from sbmlsim.simulator.plan import Plan

if TYPE_CHECKING:
    from sbmlsim.model.model_amici import AmiciSBMLModel
    from sbmlsim.model.model_copasi import CopasiSBMLModel


def execute_plan(
    plan: Plan, model: SBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a loaded model with the executor of its engine.

    Args:
        plan: the plan, compiled against the model.
        model: the loaded model, of roadrunner, COPASI or AMICI.
        selections: the columns of the output, the time first.

    Returns:
        The output of the simulation.

    Raises:
        RuntimeError: if the engine fails to integrate.
        SteadyStateError: if a presimulation finds no steady state.
        NotImplementedError: for a feature the engine does not simulate, the
            same for every plan of the model, see `check_plan`.
        TypeError: if the model is not a model of its engine.
        ValueError: for an engine without an executor.
    """
    if model.engine is Engine.ROADRUNNER:
        return executor.execute(plan, _roadrunner(model), selections)
    if model.engine is Engine.COPASI:
        # imports COPASI, which only a model of COPASI needs
        from sbmlsim.simulator import executor_copasi

        return executor_copasi.execute(plan, _copasi(model), selections)
    if model.engine is Engine.AMICI:
        # imports AMICI, which only a model of AMICI needs
        from sbmlsim.simulator import executor_amici

        return executor_amici.execute(plan, _amici(model), selections)
    raise ValueError(f"The engine '{model.engine}' has no executor.")


def check_plan(plan: Plan, model: SBMLModel) -> None:
    """Check that the engine of a model simulates a plan, before it runs.

    A feature the engine does not simulate is the same for every point of a
    plan, so a run checks it once before it runs a point, serially or in a
    pool; the executor checks every plan as well. roadrunner simulates every
    plan, COPASI refuses a delayed event of the model in a plan which sets
    values between two runs, see
    `sbmlsim.simulator.executor_copasi.check_supported`, and AMICI a change
    of a compartment it keeps out of its states, see
    `sbmlsim.simulator.executor_amici.check_supported`.

    Args:
        plan: the plan, compiled against the model, with the values of a point.
        model: the loaded model.

    Raises:
        NotImplementedError: for a feature the engine does not simulate.
        TypeError: if the model is not a model of its engine.
        ValueError: for an engine without an executor.
    """
    if model.engine is Engine.ROADRUNNER:
        return
    if model.engine is Engine.COPASI:
        from sbmlsim.simulator import executor_copasi

        executor_copasi.check_supported(plan, _copasi(model))
        return
    if model.engine is Engine.AMICI:
        from sbmlsim.simulator import executor_amici

        executor_amici.check_supported(plan, _amici(model))
        return
    raise ValueError(f"The engine '{model.engine}' has no executor.")


def _roadrunner(model: SBMLModel) -> RoadrunnerSBMLModel:
    """Get a model of the engine roadrunner as its class."""
    if not isinstance(model, RoadrunnerSBMLModel):
        raise TypeError(
            f"The model '{model.label}' of the engine roadrunner is no "
            f"`RoadrunnerSBMLModel`."
        )
    return model


def _copasi(model: SBMLModel) -> CopasiSBMLModel:
    """Get a model of the engine COPASI as its class, imports COPASI."""
    from sbmlsim.model.model_copasi import CopasiSBMLModel

    if not isinstance(model, CopasiSBMLModel):
        raise TypeError(
            f"The model '{model.label}' of the engine COPASI is no `CopasiSBMLModel`."
        )
    return model


def _amici(model: SBMLModel) -> AmiciSBMLModel:
    """Get a model of the engine AMICI as its class.

    The module of the model does not import AMICI, a model of AMICI which is
    loaded has imported it.
    """
    from sbmlsim.model.model_amici import AmiciSBMLModel

    if not isinstance(model, AmiciSBMLModel):
        raise TypeError(
            f"The model '{model.label}' of the engine AMICI is no `AmiciSBMLModel`."
        )
    return model
