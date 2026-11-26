"""Independent evaluations of a fit problem in a pool of processes.

The evaluations of the credibility of a project - a fit at the draws of its
parameters, the subjects of a predictive check - are many evaluations of one
problem which do not depend on each other. One after the other they used one
core (sbmlsim#664). `map_evaluations` runs them in a pool: every worker
receives the problem, or the report which holds it, once when it starts and
initializes it with the settings of the caller (a pickled problem is its
definition and not its resolved data, see `OptimizationProblem.__getstate__`),
then evaluates the tasks it is given. The results are the ones of the calls in
this process, in the order of the tasks, since an evaluation is a function of
its task alone.

A run of the tasks in this process takes the object of the caller as it is: a
run of few tasks, one worker, or a run inside a worker of a pool, which starts
no pool of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from sbmlsim import parallel
from sbmlsim.fit.options import FitSettings

__all__ = ["chunks", "map_evaluations"]

#: the tasks of every worker, so that a slow task does not keep the others
#: waiting at the end of a run
TASKS_PER_WORKER = 4

#: what a worker evaluates with: the object of the caller with its problem
#: initialized, and the state the tasks share
_HELD: tuple[Any, Any] | None = None


def _hold(owner: Any, settings: FitSettings | None, state: Any) -> None:
    """Initialize the problem of the object a worker received and keep it."""
    global _HELD
    from sbmlsim.fit.optimization import OptimizationProblem

    problem = owner if isinstance(owner, OptimizationProblem) else owner.problem
    problem.initialize(settings if settings is not None else FitSettings())
    _HELD = (owner, state)


def _call(item: tuple[Callable[[Any, Any, Any], Any], Any]) -> Any:
    """Evaluate a task in a worker with what the worker holds."""
    function, task = item
    if _HELD is None:
        raise RuntimeError("The worker holds no problem to evaluate a task with.")
    owner, state = _HELD
    return function(owner, state, task)


def chunks(n: int, workers: int) -> list[range]:
    """Split `n` evaluations into tasks, `TASKS_PER_WORKER` per worker.

    Args:
        n: the number of evaluations.
        workers: the workers of the pool.

    Returns:
        The ranges of the evaluations of the tasks, consecutive.
    """
    if n <= 0:
        return []
    size = max(1, math.ceil(n / max(1, workers * TASKS_PER_WORKER)))
    return [range(start, min(start + size, n)) for start in range(0, n, size)]


def map_evaluations[T](
    owner: Any,
    settings: FitSettings | None,
    function: Callable[[Any, Any, Any], T],
    tasks: Sequence[Any],
    workers: int,
    state: Any = None,
) -> list[T]:
    """Evaluate tasks of a problem, in a pool of processes if it is worth one.

    Args:
        owner: the problem, initialized, or the report which holds it.
        settings: the settings the problem is initialized with in a worker.
        function: a module level function `function(owner, state, task)`,
            which a worker imports.
        tasks: the tasks, picklable.
        workers: the processes, at most one per task; one runs the tasks in
            this process.
        state: what the tasks share, sent to every worker once.

    Returns:
        The result of every task, in the order of the tasks.
    """
    n_workers = min(workers, len(tasks))
    if n_workers <= 1 or parallel.in_worker():
        return [function(owner, state, task) for task in tasks]
    executor = parallel.start_pool(
        n_workers, initializer=_hold, initargs=(owner, settings, state)
    )
    try:
        return list(executor.map(_call, [(function, task) for task in tasks]))
    finally:
        parallel.stop(executor)
