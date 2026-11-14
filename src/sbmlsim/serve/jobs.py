"""The simulation experiments of a model as jobs of the service.

A panel is a simulation of a second (`sbmlsim.serve.runner`), which a request
waits for. The simulation experiments of a model are not: a study of the HCTZ
example runs its scans, evaluates its data and renders its figures, which is
minutes and not seconds. They are therefore submitted and not awaited: a
`JobStore` takes the experiments a request names, answers the `Job` it created
and runs it in a thread of its own, and the caller follows that job by its id
until it is `done` or `failed` and then fetches the figures it collected.

A `Job` is the record of one such run and nothing else: its id, the model and
the experiments it names, its `status`, when it was created and, for a run
which failed, a `message` which names neither a path nor an exception. The
record is the answer of the service, so it carries nothing of the host, and
the traceback of a failure is logged instead (`logger.exception`).

A store runs one job at a time, in one `ThreadPoolExecutor` of a single
worker: a run is `Project.run_experiments`, which builds an
`ExperimentRunner` and loads roadrunner models of its own in the thread of
the job, and roadrunner models are not thread safe. Two runs therefore never
load or simulate at once. A run is not serialized against the panels of the
service, though: the lock of `sbmlsim.serve.runner.PanelRunner` covers its own
warm models and the job does not take it, so a job may load a model in its
thread while a panel request loads or simulates the same SBML in the thread
pool of the server. Taking that lock for the length of a run would block every
panel of the model for minutes, which is the opposite of what the service is
for.

The jobs which wait are `queued` and the store keeps at most `max_jobs` of
them in flight, counting the queued and the running one (`IN_FLIGHT`): a
submission above that is a `TooManyJobs`, which the service answers with a
429, so the queue of a public endpoint is bounded as every request of it is.
What a finished job keeps is bounded as well: the store holds at most
`max_finished` records and drops the oldest one with its directory when a job
finishes above that (`_drop_old`), so a service which runs for weeks does not
fill its disk with the reports of runs nobody reads any more.

A run itself is serial: `n_workers=1` and `scan_workers=1`, so a job never
starts a process pool from a thread of a server, where a pool would fork or
spawn a copy of the whole service per worker and where its workers would
outlive nothing but the request which started them. The experiments draw
`matplotlib.figure.Figure` objects and never ones of pyplot (see the
conventions of the package), which is what makes the run safe in a thread at
all.

A run writes into a directory of its own below the root of the store, which is
created with the first job the store takes and readable by its owner alone, so
a store which is never used leaves nothing behind. It is where the service
serves the figures from: `figures(job_id)` lists the static figures the run
collected (`sbmlsim.project.FIGURES_DIR`) as paths relative to that directory,
and `figure(job_id, path)` resolves such a path against it and answers `None`
for everything which is not a file inside it, a path of `..` and a symbolic
link which leaves it included, so the route of the figures of a job cannot
reach a file of the host.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from sbmlsim.project import FIGURES_DIR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sbmlsim.serve.runner import ModelEntry

logger = logging.getLogger(__name__)

__all__ = [
    "IN_FLIGHT",
    "MAX_EXPERIMENTS",
    "MAX_FINISHED",
    "MAX_JOBS",
    "MODE",
    "RUN_FAILED",
    "Job",
    "JobStatus",
    "JobStore",
    "TooManyJobs",
    "UnknownExperiment",
    "check_experiments",
]

#: the status of a job: it is `queued` until the one worker of its store takes
#: it, `running` while its experiments run and `done` or `failed` after
type JobStatus = Literal["queued", "running", "done", "failed"]

#: the statuses of a job which is not finished, which `JobStore.max_jobs`
#: counts, see the module
IN_FLIGHT: Final = frozenset({"queued", "running"})

#: the jobs a store keeps in flight by default, see `JobStore`
MAX_JOBS: Final = 8

#: the finished jobs a store keeps by default: their records and their
#: directories, which hold the report, the datasets and the figures of a run.
#: The oldest one is dropped when a job finishes above it, see `JobStore`
MAX_FINISHED: Final = 32

#: the mode of the directory of the jobs and of every job below it: a run
#: writes the data of a model into it, and a temporary directory of the system
#: is read by every user of the host otherwise
MODE: Final = 0o700

#: the experiments one submission may name, the bound of its body; a name is a
#: group of the project or an experiment class of it, and the group `all` is
#: every experiment of the model in one name
MAX_EXPERIMENTS: Final = 16

#: the message of a job whose run raised, which names neither the experiments,
#: nor a path, nor the exception; the traceback is logged
RUN_FAILED: Final = "The experiments did not run."


class UnknownExperiment(ValueError):
    """An experiment name which the model of a submission does not have.

    It is the one error of a submission which a request is answered with (a
    422), as `sbmlsim.serve.runner.UnknownKnob` is the one of a run, so a
    service answers it and no other `ValueError` of the project, which would
    carry the inside of the service into a response.
    """


class TooManyJobs(RuntimeError):
    """A submission to a store which holds `max_jobs` jobs in flight."""


@dataclass(frozen=True)
class Job:
    """One run of the simulation experiments of a model, see the module.

    Attributes:
        id: the id of the job, the one a request follows it by; it is a path
            component of its directory and is created by the store.
        model: the id of the model whose experiments run, the one the
            submission named.
        experiments: the names of the groups and experiment classes the
            submission named, in its order.
        status: where the job is, see `JobStatus`.
        created: when the job was submitted, an ISO 8601 timestamp in UTC.
        message: why a job failed, a message which names no path and carries
            no traceback; empty for every other status.
    """

    id: str
    model: str
    experiments: tuple[str, ...]
    status: JobStatus
    created: str
    message: str = ""


def check_experiments(model: ModelEntry, experiments: Sequence[str]) -> None:
    """Check the names of a submission against the experiments of a model.

    It is the check which decides whether a job is created at all, so a
    submission which names something the model does not have costs the import
    of its experiments and nothing more.

    The message names the unknown names and the groups and experiment classes
    the model has, as the knobs of a panel are named: it is what the caller may
    ask for and carries no path and nothing of the project behind the model.

    Args:
        model: the model of the submission.
        experiments: the names of groups or of experiment classes.

    Raises:
        UnknownExperiment: naming the names and the ones the model has, if one
            of them is neither a group nor an experiment class of it.
        ValueError: if the model has no simulation experiments at all, which is
            a failure of the service and no error of the request.
    """
    try:
        model.project.experiment_classes(list(experiments))
    except KeyError:
        groups = model.project.load_experiments()
        known = sorted(
            {*groups, *(cls.__name__ for group in groups.values() for cls in group)}
        )
        unknown = sorted(set(experiments).difference(known))
        raise UnknownExperiment(
            f"{unknown} are no experiments of the model '{model.id}', "
            f"which has {known}."
        ) from None


class JobStore:
    """The jobs of the service and the one worker which runs them, see the module."""

    def __init__(
        self,
        root: Path,
        max_jobs: int = MAX_JOBS,
        max_finished: int = MAX_FINISHED,
    ) -> None:
        """Create a store which runs its jobs below a directory.

        Args:
            root: the directory of the jobs; it is created with the first job
                the store takes, and every job writes into a directory of its
                own below it.
            max_jobs: the jobs the store keeps in flight, the queued and the
                running one together; a submission above it raises
                `TooManyJobs`.
            max_finished: the finished jobs the store keeps; the oldest one is
                dropped with its directory when a job finishes above it, see
                the module.

        Raises:
            ValueError: for a store of fewer than one job or one which keeps
                no finished job.
        """
        if max_jobs < 1:
            raise ValueError(f"A store runs at least one job, not {max_jobs}.")
        if max_finished < 1:
            raise ValueError(
                f"A store keeps at least one finished job, not {max_finished}."
            )
        self._root = Path(root)
        self._max_jobs = max_jobs
        self._max_finished = max_finished
        #: the jobs by their id, the record of every one which was submitted
        self._jobs: dict[str, Job] = {}
        #: the lock of every read and write of the records; it is never held
        #: over a run, which is the work of the worker
        self._lock = threading.Lock()
        #: the one worker of the store, see the module
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sbmlsim-job"
        )

    def __repr__(self) -> str:
        """Get the representation, which carries no record."""
        return (
            f"JobStore(root={self._root}, max_jobs={self._max_jobs}, "
            f"max_finished={self._max_finished})"
        )

    def __len__(self) -> int:
        """Get the number of jobs the store holds, finished ones included."""
        with self._lock:
            return len(self._jobs)

    @property
    def root(self) -> Path:
        """The directory of the jobs, see `__init__`."""
        return self._root

    @property
    def max_jobs(self) -> int:
        """The jobs the store keeps in flight, see `__init__`."""
        return self._max_jobs

    @property
    def max_finished(self) -> int:
        """The finished jobs the store keeps, see `__init__`."""
        return self._max_finished

    def close(self, *, wait: bool = True) -> None:
        """Stop the worker of the store: no job of it runs afterwards.

        The jobs which are queued are cancelled and stay `queued`, and the job
        which runs finishes, so the directory of the store is complete when
        this returns and the caller may remove it. A store which was closed
        answers its records and refuses a submission.

        Args:
            wait: wait for the job which runs; without it a run which is under
                way keeps writing into its directory.
        """
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def submit(self, model: ModelEntry, experiments: Sequence[str]) -> Job:
        """Submit the experiments of a model as a job, see the module.

        The names are checked against the experiments of the model before
        anything is created (`check_experiments`), so a submission which names
        something else leaves neither a record nor a directory behind.

        Args:
            model: the model of the service whose experiments run; its project
                runs them and `model.id` is the `model` of the job.
            experiments: the names of groups or of experiment classes.

        Returns:
            The job as it was created, `queued`; its run has not started.

        Raises:
            UnknownExperiment: for a name the model does not have, see
                `check_experiments`.
            TooManyJobs: if the store holds `max_jobs` jobs in flight.
            RuntimeError: if the store was closed.
            OSError: if the directory of the job cannot be created.
        """
        names = tuple(experiments)
        check_experiments(model, names)
        job = Job(
            id=uuid.uuid4().hex,
            model=model.id,
            experiments=names,
            status="queued",
            created=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            in_flight = sum(1 for one in self._jobs.values() if one.status in IN_FLIGHT)
            if in_flight >= self._max_jobs:
                raise TooManyJobs(
                    f"The service runs {in_flight} of {self._max_jobs} jobs."
                )
            self._jobs[job.id] = job
        directory = self._root / job.id
        try:
            # the root is created with the first job, so a store which is
            # never used leaves nothing behind, see the module
            self._root.mkdir(parents=True, exist_ok=True, mode=MODE)
            directory.mkdir(exist_ok=True, mode=MODE)
            self._executor.submit(self._run, job.id, model, names, directory)
        except BaseException:
            # a job which was not taken is no job: it must not count against
            # the jobs in flight of every submission after it and it leaves no
            # directory of its own behind either
            with self._lock:
                self._jobs.pop(job.id, None)
            shutil.rmtree(directory, ignore_errors=True)
            raise
        logger.info(
            "The job '%s' of the model '%s' runs the experiments %s.",
            job.id,
            model.id,
            list(names),
        )
        return job

    def job(self, job_id: str) -> Job | None:
        """Get the record of a job as it is now.

        Args:
            job_id: the id of the job.

        Returns:
            The job, or `None` for an id the store does not hold.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def directory(self, job_id: str) -> Path | None:
        """Get the directory a job writes into.

        Args:
            job_id: the id of the job.

        Returns:
            The directory, or `None` for an id the store does not hold: an id
            of a request never reaches the filesystem.
        """
        with self._lock:
            if job_id not in self._jobs:
                return None
        return self._root / job_id

    def figures(self, job_id: str) -> tuple[str, ...]:
        """List the static figures a job collected, in the order of their names.

        Args:
            job_id: the id of the job.

        Returns:
            The paths of the figures relative to the directory of the job,
            which `figure` takes, e.g. `_figures/Study_Fig1.svg`; none for a
            job which has not run or which wrote none.
        """
        directory = self.directory(job_id)
        if directory is None:
            return ()
        collected = directory / FIGURES_DIR
        if not collected.is_dir():
            return ()
        return tuple(
            sorted(
                path.relative_to(directory).as_posix()
                for path in collected.iterdir()
                if path.is_file()
            )
        )

    def figure(self, job_id: str, path: str) -> Path | None:
        """Resolve a path of a request against the directory of a job.

        The path and the directory are both resolved and the path is answered
        only when it is a file of that directory: a path of `..`, an absolute
        path and a symbolic link which leaves the directory are `None` and are
        a 404 of the service, see the module.

        Args:
            job_id: the id of the job.
            path: the path of the file relative to the directory of the job,
                as `figures` lists it.

        Returns:
            The file, or `None` for an id the store does not hold and for a
            path which is no file inside the directory of its job.
        """
        directory = self.directory(job_id)
        if directory is None:
            return None
        try:
            root = directory.resolve()
            file = (directory / path).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                return None
        except OSError:
            # a path which the filesystem cannot resolve, e.g. one which is
            # longer than it takes or a loop of symbolic links
            return None
        return file

    def _run(
        self,
        job_id: str,
        model: ModelEntry,
        experiments: tuple[str, ...],
        directory: Path,
    ) -> None:
        """Run the experiments of a job in the worker of the store.

        Nothing of a run leaves this method: its outcome is the record of the
        job (`_outcome`) and a `BaseException`, which ends the worker, is
        recorded as a failure before it is raised on, so a job never stays
        `running` and holds its slot in flight for the life of the process.
        The finished jobs are pruned afterwards, see the module.

        Args:
            job_id: the id of the job.
            model: the model whose experiments run.
            experiments: the names of the experiments.
            directory: the directory of the job, which exists.
        """
        self._set(job_id, "running", "")
        try:
            status, message = self._outcome(job_id, model, experiments, directory)
        except BaseException:
            logger.exception(
                "The job '%s' of the model '%s' was interrupted.", job_id, model.id
            )
            self._set(job_id, "failed", RUN_FAILED)
            self._drop_old()
            raise
        self._set(job_id, status, message)
        self._drop_old()

    def _outcome(
        self,
        job_id: str,
        model: ModelEntry,
        experiments: tuple[str, ...],
        directory: Path,
    ) -> tuple[JobStatus, str]:
        """Run the experiments of a job and answer what became of them.

        The run is serial (`n_workers=1`, `scan_workers=1`): a job runs in a
        thread of a server, which never starts a process pool, see the module.
        It raises for nothing (`raise_on_failure=False`): an experiment or a
        figure which failed is counted in the message of the job and a run
        which raised is `RUN_FAILED`. The reasons are logged, with the
        traceback of a run which raised, and are never answered.

        Args:
            job_id: the id of the job.
            model: the model whose experiments run.
            experiments: the names of the experiments.
            directory: the directory of the job, which exists.

        Returns:
            The status the job reached and its message, empty for a run in
            which nothing failed.
        """
        try:
            results = model.project.run_experiments(
                list(experiments),
                directory,
                n_workers=1,
                scan_workers=1,
                progress=False,
                raise_on_failure=False,
                collect_figures=True,
            )
        except Exception:
            logger.exception(
                "The job '%s' of the model '%s' did not run.", job_id, model.id
            )
            return "failed", RUN_FAILED
        failed = [result for result in results if result.failed]
        for result in failed:
            # the error is `None` for an experiment whose figures alone
            # failed, which is why both are logged: it is the one place the
            # reason is kept, the message of the job carries none
            logger.error(
                "The experiment '%s' of the job '%s' failed with the error %s "
                "and the figures %s.",
                type(result.experiment).__name__,
                job_id,
                result.error,
                result.failed_figures,
            )
        if failed:
            return "failed", f"{len(failed)} of {len(results)} experiments failed."
        return "done", ""

    def _drop_old(self) -> None:
        """Drop the finished jobs over `max_finished`, the oldest ones first.

        A record is dropped with the directory of its job, so the reports, the
        datasets and the figures of a run are kept as long as the record which
        names them is. The directories are removed outside the lock, which is
        never held over the filesystem.
        """
        with self._lock:
            finished = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status not in IN_FLIGHT
            ]
            dropped = finished[: max(0, len(finished) - self._max_finished)]
            for job_id in dropped:
                del self._jobs[job_id]
        for job_id in dropped:
            shutil.rmtree(self._root / job_id, ignore_errors=True)
            logger.debug("The job '%s' and its directory were dropped.", job_id)

    def _set(self, job_id: str, status: JobStatus, message: str) -> None:
        """Record the status of a job, which is its record from now on.

        Args:
            job_id: the id of the job.
            status: the status it reached.
            message: why it failed, empty for every other status.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, status=status, message=message)
