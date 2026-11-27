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

`map_evaluations` starts a pool per call and its workers initialize the
problem of the call. An `EvaluationPool` is started once for several problems
and calls, e.g. for the fits of every subset of a credibility assessment and
for both its uncertainty and its variability: its workers receive the problems
by key when they start and initialize each one once, the first time a task
needs it, or when the pool warms them up. A pool which is started before the
caller initializes its own problems (`warm`) initializes them in the workers at
the same time.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from sbmlsim import parallel
from sbmlsim.fit.options import FitSettings

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from sbmlsim.fit.optimization import OptimizationProblem

__all__ = [
    "EvaluationPool",
    "chunks",
    "evaluates_on_threads",
    "map_evaluations",
    "problem_modules",
]

#: the tasks of every worker, so that a slow task does not keep the others
#: waiting at the end of a run
TASKS_PER_WORKER = 4

#: what a worker evaluates with: the object of the caller with its problem
#: initialized, and the state the tasks share
_HELD: tuple[Any, Any] | None = None


def problem_modules(problems: Iterable[OptimizationProblem]) -> list[str]:
    """Get the modules the forkserver imports once for the workers of problems.

    Under the start method `forkserver` every worker imports sbmlsim and the
    modules of the experiments of a problem again when it receives the
    problem, which costs more than a short optimization, and 0.7 s of every
    worker of the credibility assessment of the HCTZ model of pkdb_models.
    The forkserver imports them once and the workers inherit them; the
    experiments of a script (`__main__`) are imported by every worker.

    Args:
        problems: the problems of the workers.

    Returns:
        The modules, sorted.
    """
    modules = {"sbmlsim.fit.optimization"}
    for problem in problems:
        for mapping_collection in problem.mapping_collections:
            module = getattr(mapping_collection.experiment_class, "__module__", None)
            if module and module != "__main__":
                modules.add(module)
    return sorted(modules)


def _problem_of(owner: Any) -> OptimizationProblem:
    """Get the problem of an object of `map_evaluations`, a problem or a report."""
    from sbmlsim.fit.optimization import OptimizationProblem

    return owner if isinstance(owner, OptimizationProblem) else owner.problem


def _hold(owner: Any, settings: FitSettings | None, state: Any) -> None:
    """Initialize the problem of the object a worker received and keep it."""
    global _HELD
    _problem_of(owner).initialize(settings if settings is not None else FitSettings())
    _HELD = (owner, state)


#: the problems of a worker of an `EvaluationPool` by their key, each with
#: the settings it is initialized with, as the worker received them
_PROBLEMS: dict[str, tuple[OptimizationProblem, FitSettings]] = {}

#: the keys of the problems of `_PROBLEMS` the worker initialized
_INITIALIZED: set[str] = set()


def _receive(payload: bytes) -> None:
    """Keep the problems of an `EvaluationPool`, the initializer of its workers.

    The problems are definitions (`OptimizationProblem.__getstate__`), which
    are initialized when a task needs them, see `_problem`.
    """
    _PROBLEMS.clear()
    _PROBLEMS.update(pickle.loads(payload))
    _INITIALIZED.clear()


def _problem(key: str) -> OptimizationProblem:
    """Get a problem of the worker, initialized the first time it is needed."""
    problem, settings = _PROBLEMS[key]
    if key not in _INITIALIZED:
        problem.initialize(settings)
        _INITIALIZED.add(key)
    return problem


def _warm(keys: Sequence[str]) -> None:
    """Initialize the problems of the worker, a task of `EvaluationPool`."""
    for key in keys:
        _problem(key)


def _call_problem(
    item: tuple[str, Callable[[Any, Any, Any], Any], Any, Any],
) -> Any:
    """Evaluate a task of an `EvaluationPool` with the problem of its key."""
    key, function, state, task = item
    return function(_problem(key), state, task)


class EvaluationPool:
    """A pool whose workers hold several problems and evaluate tasks of each.

    The workers receive the problems when they start, pickled once, and
    initialize a problem the first time a task needs it, so a problem is
    initialized once per worker for all calls of `map`. With `warm` every
    worker is given a task which initializes all problems as soon as it
    started, while the caller goes on, e.g. initializes its own problems: the
    tasks of the first `map` find the problems initialized.

    The results of a task are the ones of the call in this process, since an
    evaluation is a function of its task alone; a pool of one worker, or a
    pool created in a worker, starts no processes (`active` is false) and the
    caller evaluates in its own process, see `map_evaluations`.

    Attributes:
        workers: the number of worker processes.
        keys: the keys of the problems.
    """

    def __init__(
        self,
        problems: Mapping[str, tuple[OptimizationProblem, FitSettings]],
        workers: int,
        *,
        warm: bool = True,
    ) -> None:
        """Start the pool.

        Args:
            problems: the problems by their key, each with the settings it is
                initialized with in a worker; a problem is sent as its
                definition, initialized or not.
            workers: the number of worker processes; one, or a pool in a
                worker, starts none.
            warm: initialize every problem in every worker right after the
                start, in the background.

        Raises:
            RuntimeError: see `sbmlsim.parallel.start_pool`.
        """
        self.workers = max(1, workers)
        self.keys: list[str] = list(problems)
        self._executor: ProcessPoolExecutor | None = None
        if self.workers <= 1 or parallel.in_worker() or not self.keys:
            return
        payload = pickle.dumps(dict(problems))
        self._executor = parallel.start_pool(
            self.workers,
            preload=problem_modules(problem for problem, _ in problems.values()),
            initializer=_receive,
            initargs=(payload,),
        )
        if warm:
            # a worker which takes a second of these finds its problems
            # initialized, and one which takes none initializes them with its
            # first task
            for _ in range(self.workers):
                self._executor.submit(_warm, self.keys)

    @property
    def active(self) -> bool:
        """Whether the pool has worker processes, which `map` evaluates in."""
        return self._executor is not None

    def map[T](
        self,
        key: str,
        function: Callable[[Any, Any, Any], T],
        tasks: Sequence[Any],
        state: Any = None,
    ) -> list[T]:
        """Evaluate tasks of a problem of the pool in its workers.

        Args:
            key: the key of the problem.
            function: a module level function `function(problem, state,
                task)`, which a worker imports, called with the initialized
                problem of the key.
            tasks: the tasks, picklable.
            state: what the tasks share, sent with every task.

        Returns:
            The result of every task, in the order of the tasks.

        Raises:
            ValueError: if the pool is not active or has no problem of the key.
        """
        if self._executor is None:
            raise ValueError("The pool has no workers, evaluate in the process.")
        if key not in self.keys:
            raise ValueError(f"The pool has no problem '{key}', it has {self.keys}.")
        return list(
            self._executor.map(
                _call_problem, [(key, function, state, task) for task in tasks]
            )
        )

    def close(self) -> None:
        """Stop the workers; a closed pool is not active."""
        if self._executor is not None:
            executor, self._executor = self._executor, None
            parallel.stop(executor)

    def __enter__(self) -> Self:
        """Enter the context, which closes the pool when it exits."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the pool."""
        self.close()


def _call(item: tuple[Callable[[Any, Any, Any], Any], Any]) -> Any:
    """Evaluate a task in a worker with what the worker holds."""
    function, task = item
    if _HELD is None:
        raise RuntimeError("The worker holds no problem to evaluate a task with.")
    owner, state = _HELD
    return function(owner, state, task)


def evaluates_on_threads(problem: OptimizationProblem) -> bool:
    """Whether an evaluation of a problem runs its simulations on threads.

    The engine core simulates the groups of fit mappings of one evaluation as
    one batch on threads of its own (`OptimizationProblem._batched_evaluation`),
    so the evaluations of a problem in this process use the CPUs of the
    process, and a pool would only add the start of its workers, each of
    which initializes the problem, i.e. loads its experiments and models
    again. The other engines simulate one plan per call, and a pool is what
    makes their evaluations parallel.

    Args:
        problem: the problem.

    Returns:
        Whether the engine of the problem is the engine core.
    """
    from sbmlsim.engine import Engine

    return problem.engine is Engine.CORE


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
    *,
    pool: EvaluationPool | None = None,
    key: str | None = None,
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
        pool: a started pool which holds the problem as `key`, whose workers
            evaluate the tasks instead of a pool of this call; the owner must
            then be the problem, which the function gets in the workers. An
            inactive pool evaluates in this process.
        key: the key of the problem in `pool`.

    Returns:
        The result of every task, in the order of the tasks.
    """
    if pool is not None and pool.active and len(tasks) > 1:
        if key is None:
            raise ValueError("A pool evaluates the problem of a key, none is given.")
        return pool.map(key, function, tasks, state)
    n_workers = min(workers, len(tasks))
    if pool is not None or n_workers <= 1 or parallel.in_worker():
        return [function(owner, state, task) for task in tasks]
    executor = parallel.start_pool(
        n_workers,
        preload=problem_modules([_problem_of(owner)]),
        initializer=_hold,
        initargs=(owner, settings, state),
    )
    try:
        return list(executor.map(_call, [(function, task) for task in tasks]))
    finally:
        parallel.stop(executor)
