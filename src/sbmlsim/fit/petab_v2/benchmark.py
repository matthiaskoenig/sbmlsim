"""The problems of the PEtab benchmark collection.

The [benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)
is a set of published parameter estimation problems, each in
`problems/<name>/v1` in PEtab v1 with the simulations of its measurements at
the nominal parameters, `simulations.tsv`, and some also in
`problems/<name>/v2` in PEtab v2. `sbmlsim` reads PEtab v2 only: a problem is
its encoding in PEtab v2 where the collection has one, otherwise it is
converted from PEtab v1 with `petab.v2.petab1to2`; its simulations are
converted with it, see `sbmlsim.fit.petab_v2.conversion`. The problem is read
with `PetabReader`, simulated at the nominal values of its parameter table
and compared with the simulations of the collection; where AMICI states the
log-likelihood of the problem (`tests/benchmark_models/benchmark_models.yaml`
of AMICI), it is compared too. The problems of `EXCLUDED_PROBLEMS` take too
long and are left out.

| status | meaning |
| --- | --- |
| `pass` | the simulations and the log-likelihood agree |
| `tolerance` | a simulation or the log-likelihood is outside of its tolerance |
| `conversion` | `petab1to2` cannot convert the problem |
| `error` | the problem cannot be read or simulated |

The collection has no releases, so it is pinned by a commit, the references
of AMICI by another. `BenchmarkCollection` is a commit of the collection with
the download, the conversion and the cache.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import tempfile
import time
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sbmlsim.comparison.diff import within_tolerance
from sbmlsim.fit.options import DataAtChangeType, FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2.conversion import (
    SIMULATIONS_FILE,
    convert_simulations,
    convert_v1,
    measurement_table,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.testsuite import cache

logger = logging.getLogger(__name__)

#: commit of the collection the tests run against
BENCHMARK_COMMIT = "c44d5ede9bf979db5aa8ae4e94fc31fba9776675"

#: archive of a commit of the collection
BENCHMARK_URL = (
    "https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/archive/"
    "{commit}.zip"
)

#: the environment variable which points at the collection when it is not in
#: the cache, i.e. at a directory with `problems` and `v2`
BENCHMARK_PATH_VARIABLE = "SBMLSIM_BENCHMARK_PATH"

#: commit of AMICI whose log-likelihoods of the problems are the references,
#: the release v1.2.0
AMICI_COMMIT = "b15fed818801c8eea6fbbfadb18f0ab39c5922b8"

#: the problems of the collection which are not run, with the reason; the
#: script, the tests and `BenchmarkCollection.run` leave them out unless they
#: are named
EXCLUDED_PROBLEMS: dict[str, str] = {
    "Froehlich_CellSystems2018": "9169 experiments with a pre-equilibration "
    "each take longer than the 600 s a problem gets (28 s per experiment to "
    "the steady state)",
}

#: the log-likelihoods of AMICI at a commit
AMICI_URL = (
    "https://raw.githubusercontent.com/AMICI-dev/AMICI/{commit}/tests/"
    "benchmark_models/benchmark_models.yaml"
)

#: the directories and files of the collection in the cache
PROBLEMS = "problems"
V2 = "v2"
AMICI_FILE = "amici.yaml"

#: the seconds a problem gets in the script and in the tests
BENCHMARK_TIMEOUT = 600.0

#: absolute and relative tolerance of a simulation, see `within_tolerance`
SIMULATION_TOLERANCE = 1e-3

#: absolute and relative tolerance of the log-likelihood
LLH_ABSOLUTE_TOLERANCE = 1e-3
LLH_RELATIVE_TOLERANCE = 1e-6

#: the settings a problem is simulated with. The parameters are evaluated and
#: not searched, so the scale is linear, which every bound allows. The
#: integrator is tighter than the tolerances of the comparison; tighter still,
#: CVODE fails in `Isensee_JCB2018` and `Raimundez_PCB2020`. A measurement at
#: the time of a change is the state after it (PEtab)
BENCHMARK_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-8,
    data_at_change=DataAtChangeType.AFTER,
)

#: the settings of the problems which need a tighter integrator than
#: `BENCHMARK_SETTINGS`: `EarlyEffector` of `Crauste_CellSystems2017` grows
#: towards a blow up in finite time, at the tolerances of `BENCHMARK_SETTINGS`
#: its simulations are off by up to 1.7 %, at these they agree with the
#: collection
PROBLEM_SETTINGS: dict[str, FitSettings] = {
    "Crauste_CellSystems2017": FitSettings(
        parameter_scale=ParameterScaleType.LINEAR,
        absolute_tolerance=1e-14,
        relative_tolerance=1e-12,
        data_at_change=DataAtChangeType.AFTER,
    ),
}

#: the columns which identify the simulation of a measurement
_KEY_COLUMNS: tuple[str, ...] = ("observableId", "experimentId", "time")

#: the columns of the experiment and the observable, which a table of the
#: simulations may name otherwise, see `_same_or_renamed`
_ID_COLUMNS: tuple[str, ...] = ("observableId", "experimentId")

#: the names of the columns of PEtab v1 some tables of the simulations of the
#: collection use, which are renamed before the conversion
_V1_ALIASES: dict[str, str] = {
    "simulationCondition": "simulationConditionId",
    "preequilibrationCondition": "preequilibrationConditionId",
}

#: the directory of a problem in PEtab v1 and in PEtab v2 in the collection
V1_DIR = "v1"
V2_DIR = "v2"


class BenchmarkStatus(StrEnum):
    """The outcome of a problem."""

    PASS = "pass"
    TOLERANCE = "tolerance"
    CONVERSION = "conversion"
    ERROR = "error"


@dataclass(frozen=True)
class BenchmarkResult:
    """The outcome of a problem.

    Attributes:
        name: the name of the problem.
        status: the outcome.
        message: what failed, or a note on what was not compared.
        max_difference: the largest absolute difference of a simulation to
            the collection, `None` when nothing was compared.
        n_simulations: the number of simulations which were compared.
        llh: the log-likelihood at the nominal values, `None` when it was
            not calculated.
        llh_reference: the log-likelihood of AMICI it was compared with.
        timings: the seconds it took to `read`, `initialize`, `simulate` and
            calculate the `llh`.
    """

    name: str
    status: BenchmarkStatus
    message: str = ""
    max_difference: float | None = None
    n_simulations: int = 0
    llh: float | None = None
    llh_reference: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check whether the problem passes."""
        return self.status == BenchmarkStatus.PASS

    @property
    def key(self) -> str:
        """Get the key of the problem in the baseline, its name."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "max_difference": self.max_difference,
            "n_simulations": self.n_simulations,
            "llh": self.llh,
            "llh_reference": self.llh_reference,
            "timings": dict(self.timings),
        }


def _key_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Get the columns which identify the simulation of a measurement."""
    columns: dict[str, pd.Series] = {}
    for column in _KEY_COLUMNS:
        if column not in table:
            columns[column] = pd.Series([""] * len(table), index=table.index)
        elif column == "time":
            columns[column] = table[column].astype(float)
        else:
            columns[column] = table[column].astype(object).fillna("").astype(str)
    return pd.DataFrame(columns)


def _keys(table: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Get the key of the simulation of every row of a table."""
    return list(_key_frame(table).itertuples(index=False, name=None))


def _same_or_renamed(ids: np.ndarray, other: np.ndarray) -> bool:
    """Check whether two columns of ids agree row by row, up to a renaming.

    The ids are renamed when the columns have no id in common and every id of
    `other` stands for one id of `ids`, e.g. an observable whose simulations
    are written per parameters and conditions (`Isensee_JCB2018`).
    """
    if (ids == other).all():
        return True
    if set(ids) & set(other):
        return False
    original: dict[str, str] = {}
    return all(original.setdefault(b, a) == a for a, b in zip(ids, other, strict=True))


def _in_order(measurements: pd.DataFrame, expected: pd.DataFrame) -> bool:
    """Check whether the simulations are row by row the ones of the measurements.

    The times agree row by row, and so do the observables and the experiments,
    up to a renaming after the simulations were written, see
    `_same_or_renamed`.
    """
    if len(measurements) != len(expected):
        return False
    keys, other = _key_frame(measurements), _key_frame(expected)
    if not np.allclose(
        keys["time"].to_numpy(), other["time"].to_numpy(), equal_nan=True
    ):
        return False
    return all(
        _same_or_renamed(keys[column].to_numpy(), other[column].to_numpy())
        for column in _ID_COLUMNS
    )


def _with_observables_of(
    measurements: pd.DataFrame, expected: pd.DataFrame
) -> pd.DataFrame:
    """Rename the observables of the simulations to the ones of the measurements.

    A table of the simulations may name an observable differently: without
    its prefix (`Raia_CancerResearch2011`) or with its parameters and
    conditions (`Isensee_JCB2018`). An id which is no observable of the
    measurements is the observable whose id it starts or ends with, if there
    is exactly one.

    Args:
        measurements: the measurement table of the problem.
        expected: the table of the simulations of the collection.

    Returns:
        The table of the simulations with the observables of the measurements.
    """
    observables = set(measurements["observableId"].astype(str))
    renamed: dict[str, str] = {}
    for sid in set(expected["observableId"].astype(str)) - observables:
        candidates = [o for o in observables if sid.startswith(o) or o.endswith(sid)]
        if len(candidates) == 1:
            renamed[sid] = candidates[0]
    if not renamed:
        return expected
    expected = expected.copy()
    expected["observableId"] = expected["observableId"].astype(str).replace(renamed)
    return expected


def _match(
    measurements: pd.DataFrame, expected: pd.DataFrame
) -> tuple[list[int], list[float]]:
    """Find the simulation of the collection of every measurement.

    Args:
        measurements: the measurement table of the problem.
        expected: the table of the simulations of the collection.

    Returns:
        The rows of the measurements which have a simulation and their
        simulations.
    """
    values = expected["simulation"].to_numpy(dtype=float)
    if _in_order(measurements, expected):
        return list(range(len(measurements))), list(values)
    available: dict[tuple[Any, ...], deque[float]] = defaultdict(deque)
    expected = _with_observables_of(measurements, expected)
    for key, value in zip(_keys(expected), values, strict=True):
        available[key].append(value)
    rows: list[int] = []
    references: list[float] = []
    for i, key in enumerate(_keys(measurements)):
        if available[key]:
            rows.append(i)
            references.append(available[key].popleft())
    return rows, references


class ProblemTimeout(Exception):
    """A problem took longer than the time it was given."""


def _timeout(signum: int, frame: FrameType | None) -> None:
    """Stop the problem which takes too long."""
    raise ProblemTimeout("the timer of the process expired")


@dataclass(frozen=True)
class BenchmarkProblem:
    """A problem of the benchmark collection.

    Attributes:
        name: the name of the problem.
        source_dir: the directory of the problem in the collection, with its
            encodings `v1` and, for some problems, `v2`.
        v2_dir: the directory of the problem in PEtab v2 which is run: the
            encoding of the collection in PEtab v2 or the one converted from
            PEtab v1, with the simulations of the collection converted to
            PEtab v2, `SIMULATIONS_FILE`, see `BenchmarkCollection.convert`.
    """

    name: str
    source_dir: Path
    v2_dir: Path

    @property
    def native(self) -> bool:
        """Check whether the collection has the problem in PEtab v2."""
        return (self.source_dir / V2_DIR).is_dir()

    @property
    def conversion_error(self) -> Path:
        """Get the file which holds why the conversion failed."""
        return self.v2_dir.parent / f"{self.name}.error"

    def run(
        self,
        llh_reference: float | None = None,
        settings: FitSettings | None = None,
        timeout: float | None = None,
    ) -> BenchmarkResult:
        """Read, simulate and compare the problem.

        Args:
            llh_reference: the log-likelihood of AMICI, which is compared.
            settings: the settings the problem is simulated with, by default
                the ones of `PROBLEM_SETTINGS` or `BENCHMARK_SETTINGS`.
            timeout: the seconds the problem may take, a problem which takes
                longer is an error. The time is measured with the timer
                `ITIMER_REAL` of the process, so a time limit needs the main
                thread of a POSIX system, on Windows the problem runs without
                one and a warning says so; a call into roadrunner finishes
                before the problem stops.

        Returns:
            The outcome.
        """
        if settings is None:
            settings = PROBLEM_SETTINGS.get(self.name, BENCHMARK_SETTINGS)
        if not self.v2_dir.is_dir():
            message = (
                self.conversion_error.read_text(encoding="utf-8").strip()
                if self.conversion_error.is_file()
                else "the problem was not converted to PEtab v2"
            )
            return BenchmarkResult(
                name=self.name, status=BenchmarkStatus.CONVERSION, message=message
            )
        timings: dict[str, float] = {}
        previous = None
        if timeout is not None and not hasattr(signal, "setitimer"):
            logger.warning(
                "'%s': the time limit of %s s needs the timer of a POSIX "
                "system, the problem runs without one.",
                self.name,
                timeout,
            )
            timeout = None
        if timeout is not None:
            previous = signal.signal(signal.SIGALRM, _timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return self._run(llh_reference, settings, timings)
        except ProblemTimeout as err:
            return BenchmarkResult(
                name=self.name,
                status=BenchmarkStatus.ERROR,
                message=f"ProblemTimeout: the problem takes more than {timeout} s "
                f"({err})",
                timings=timings,
            )
        except Exception as err:  # every failure is the outcome of the problem
            logger.debug("Problem '%s' failed", self.name, exc_info=True)
            return BenchmarkResult(
                name=self.name,
                status=BenchmarkStatus.ERROR,
                message=f"{type(err).__name__}: {err}",
                timings=timings,
            )
        finally:
            if timeout is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(
                    signal.SIGALRM, previous if previous is not None else signal.SIG_DFL
                )

    def _run(
        self,
        llh_reference: float | None,
        settings: FitSettings,
        timings: dict[str, float],
    ) -> BenchmarkResult:
        """Run the problem, see `run`, filling in the timings."""
        start = time.perf_counter()
        yaml_file = next(iter(sorted(self.v2_dir.glob("*.yaml"))))
        reader = PetabReader.from_yaml(yaml_file)
        reader.derived_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_"))
        problem = reader.to_optimization_problem(opid=self.name)
        timings["read"] = time.perf_counter() - start

        start = time.perf_counter()
        problem.initialize(settings)
        timings["initialize"] = time.perf_counter() - start

        nominal = reader.nominal_parameters(problem)
        start = time.perf_counter()
        evaluations = problem.evaluations(nominal.x(problem.pids), problem.indices())
        predictions = {k: e.prediction for k, e in evaluations.items()}
        timings["simulate"] = time.perf_counter() - start

        start = time.perf_counter()
        llh = log_likelihood(problem, nominal, evaluations=evaluations)
        timings["llh"] = time.perf_counter() - start

        index = {key: k for k, key in enumerate(problem.mapping_keys)}
        simulated = np.array(
            [
                predictions[index[key]][position]
                for key, position in reader.measurement_rows()
            ],
            dtype=float,
        )
        failures, notes, max_difference, compared = self._compare(simulated)

        if llh_reference is not None and not within_tolerance(
            llh_reference, llh, LLH_ABSOLUTE_TOLERANCE, LLH_RELATIVE_TOLERANCE
        ):
            failures.append(
                f"llh {llh:.10g} differs from the llh {llh_reference:.10g} of AMICI"
            )
        return BenchmarkResult(
            name=self.name,
            status=BenchmarkStatus.TOLERANCE if failures else BenchmarkStatus.PASS,
            message="; ".join(failures + notes),
            max_difference=max_difference,
            n_simulations=compared,
            llh=llh,
            llh_reference=llh_reference,
            timings=timings,
        )

    def _compare(
        self, simulated: np.ndarray
    ) -> tuple[list[str], list[str], float | None, int]:
        """Compare the simulation of every measurement with the collection.

        The simulations of the collection are converted to PEtab v2 with the
        problem. A table of the simulations which has the observables and the
        times of the measurements row by row is in their order, even where
        the problem renamed its observables or experiments afterwards, e.g.
        an encoding in PEtab v2 of the collection; otherwise it may have
        another order and more rows, and a simulation is found by the
        observable, the experiment and the time of its measurement, the
        replicates in their order.

        Args:
            simulated: the simulation of every measurement, in the order of
                the measurement table.

        Returns:
            The failures, the notes, the largest difference, `None` without a
            comparison, and the number of simulations which were compared.
        """
        simulations = self.v2_dir / SIMULATIONS_FILE
        if not simulations.is_file():
            return [], ["the collection has no simulations of the problem"], None, 0
        measurements = measurement_table(self.v2_dir)
        expected = pd.read_csv(simulations, sep="\t")
        if len(measurements) != len(simulated):
            raise ValueError(
                f"The problem has '{len(simulated)}' measurements, the "
                f"collection '{len(measurements)}'."
            )
        rows, references = _match(measurements, expected)
        notes: list[str] = []
        failures: list[str] = []
        if len(rows) < len(simulated):
            failures.append(
                f"'{len(simulated) - len(rows)}' of the '{len(simulated)}' "
                f"measurements have no simulation in the collection"
            )
        if not rows:
            return failures, notes, None, 0

        values = simulated[rows]
        reference = np.asarray(references, dtype=float)
        difference = np.abs(values - reference)
        agree = within_tolerance(
            reference, values, SIMULATION_TOLERANCE, SIMULATION_TOLERANCE
        )
        # a simulation which is `nan` in both is not a difference
        agree |= np.isnan(values) & np.isnan(reference)
        if not np.all(agree):
            table = measurements.iloc[rows]
            keys = _key_frame(table)
            failed = defaultdict(int)
            for (observable_id, experiment_id), ok in zip(
                zip(keys["observableId"], keys["experimentId"], strict=True),
                agree,
                strict=True,
            ):
                if not ok:
                    failed[(observable_id, experiment_id)] += 1
            failures.append(
                f"'{int(np.sum(~agree))}' of '{len(rows)}' simulations differ: "
                + ", ".join(
                    f"{observable_id} in {experiment_id or 'no experiment'} ({count})"
                    for (observable_id, experiment_id), count in sorted(failed.items())
                )
            )
        finite = difference[np.isfinite(difference)]
        max_difference = float(np.max(finite)) if finite.size else None
        return failures, notes, max_difference, len(rows)


@dataclass(frozen=True)
class BenchmarkCollection:
    """A commit of the benchmark collection, converted to PEtab v2.

    Attributes:
        path: the directory with the problems of the collection in
            `problems`, the problems in PEtab v2 which are run in `v2` and
            the references of AMICI.
        commit: the commit of the collection.
    """

    path: Path
    commit: str

    @staticmethod
    def cache_path(commit: str = BENCHMARK_COMMIT) -> Path:
        """Get the directory a commit of the collection is unpacked into.

        `SBMLSIM_BENCHMARK_PATH` overrides it. Otherwise it is
        `sbmlsim/petab-benchmark/<commit>` in the user cache.

        Args:
            commit: the commit of the collection.

        Returns:
            The directory of the collection.
        """
        return cache.cache_path(BENCHMARK_PATH_VARIABLE, "petab-benchmark", commit)

    @classmethod
    def cached(cls, commit: str = BENCHMARK_COMMIT) -> BenchmarkCollection | None:
        """Get a commit of the collection if it is already on this machine.

        Args:
            commit: the commit of the collection.

        Returns:
            The collection, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(commit)
        return cls(path=path, commit=commit) if (path / PROBLEMS).is_dir() else None

    @classmethod
    def load(cls, commit: str = BENCHMARK_COMMIT) -> BenchmarkCollection:
        """Get a commit of the collection, downloading and converting it.

        The problems are downloaded once, converted once and the references
        of AMICI are fetched once.

        Args:
            commit: the commit of the collection.

        Returns:
            The collection in the cache.

        Raises:
            OSError: if the commit or the references cannot be downloaded.
        """
        path = cls.cache_path(commit)
        if not (path / PROBLEMS).is_dir():
            url = BENCHMARK_URL.format(commit=commit)
            logger.info("Downloading the PEtab benchmark collection '%s'", commit)
            cache.fetch(url, path / PROBLEMS, select=cls._problems_dir)
        elif not cache.is_overridden(BENCHMARK_PATH_VARIABLE):
            cache.remove_stale(path / PROBLEMS)
        collection = cls(path=path, commit=commit)
        if not (path / AMICI_FILE).is_file():
            _fetch_file(AMICI_URL.format(commit=AMICI_COMMIT), path / AMICI_FILE)
        collection.convert()
        return collection

    @staticmethod
    def _problems_dir(staging: Path) -> Path:
        """Get the directory of the problems of an unpacked archive.

        Raises:
            OSError: if the archive holds no such directory.
        """
        for candidate in sorted(staging.glob(f"*/{PROBLEMS}")):
            if candidate.is_dir():
                return candidate
        raise OSError(
            f"No directory '{PROBLEMS}' in the unpacked collection '{staging}'"
        )

    def problem_names(self, excluded: bool = False) -> list[str]:
        """Get the names of the problems, sorted.

        Args:
            excluded: whether the problems of `EXCLUDED_PROBLEMS` are listed,
                which are neither converted nor run by default.
        """
        directory = self.path / PROBLEMS
        if not directory.is_dir():
            return []
        return sorted(
            p.name
            for p in directory.iterdir()
            if ((p / V1_DIR).is_dir() or (p / V2_DIR).is_dir())
            and (excluded or p.name not in EXCLUDED_PROBLEMS)
        )

    def problem(self, name: str) -> BenchmarkProblem:
        """Get a problem of the collection by its name."""
        return BenchmarkProblem(
            name=name,
            source_dir=self.path / PROBLEMS / name,
            v2_dir=self.path / V2 / name,
        )

    def problems(self) -> Iterator[BenchmarkProblem]:
        """Iterate the problems of the collection."""
        for name in self.problem_names():
            yield self.problem(name)

    def references(self) -> dict[str, float]:
        """Get the log-likelihoods of AMICI by the name of the problem."""
        path = self.path / AMICI_FILE
        if not path.is_file():
            return {}
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            name: float(entry["llh"])
            for name, entry in content.items()
            if isinstance(entry, dict) and entry.get("llh") is not None
        }

    def convert(self, names: Sequence[str] | None = None) -> None:
        """Write the problems in PEtab v2 which are not written yet.

        A problem is its encoding in PEtab v2 of the collection or, without
        one, its encoding in PEtab v1 converted with `petab1to2`; the
        simulations of the collection are converted from PEtab v1 in both
        cases, see `sbmlsim.fit.petab_v2.conversion`. A problem which cannot
        be converted gets the file `v2/<name>.error` with the reason, see
        `BenchmarkProblem.conversion_error`.

        Args:
            names: the problems, all of them by default.
        """
        for name in names if names is not None else self.problem_names():
            problem = self.problem(name)
            if problem.v2_dir.is_dir() or problem.conversion_error.is_file():
                continue
            _convert(problem)

    def run(
        self,
        names: Sequence[str] | None = None,
        progress: Callable[[BenchmarkResult], None] | None = None,
    ) -> list[BenchmarkResult]:
        """Run the problems of the collection.

        Args:
            names: the problems, all of them by default.
            progress: called with the result of every problem.

        Returns:
            The results in the order of the names.
        """
        references = self.references()
        results: list[BenchmarkResult] = []
        for name in names if names is not None else self.problem_names():
            result = self.problem(name).run(llh_reference=references.get(name))
            if progress is not None:
                progress(result)
            results.append(result)
        return results


def _convert(problem: BenchmarkProblem) -> None:
    """Write a problem in PEtab v2 into a staging directory and move it."""
    problem.v2_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{problem.name}.", dir=problem.v2_dir.parent)
    )
    output = staging / problem.name
    try:
        v1_dir = problem.source_dir / V1_DIR
        v1_yaml = _yaml_file(v1_dir) if v1_dir.is_dir() else None
        simulation_files = _simulation_files(v1_yaml) if v1_yaml is not None else []
        if problem.native:
            shutil.copytree(problem.source_dir / V2_DIR, output)
            if v1_yaml is not None and simulation_files:
                convert_simulations(v1_yaml, simulation_files).to_csv(
                    output / SIMULATIONS_FILE, sep="\t", index=False
                )
        elif v1_yaml is not None:
            convert_v1(v1_yaml, output, simulation_files)
        else:
            raise ValueError(f"The problem '{problem.name}' has no encoding.")
        os.replace(output, problem.v2_dir)
    except Exception as err:  # the conversion of the problem is its outcome
        logger.warning("The problem '%s' is not converted: %s", problem.name, err)
        problem.conversion_error.write_text(
            f"{type(err).__name__}: {err}\n", encoding="utf-8"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _yaml_file(directory: Path) -> Path:
    """Get the YAML file of a problem.

    Raises:
        ValueError: if the directory has no YAML file.
    """
    yaml_files = sorted(directory.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"The problem '{directory}' has no YAML file.")
    return yaml_files[0]


def _simulation_files(v1_yaml: Path) -> list[Path]:
    """Get the tables of the simulations of a problem of PEtab v1.

    The tables are repaired in place, in the cache, before they are
    converted: a column which some tables name otherwise is renamed, see
    `_V1_ALIASES`, and the conditions of a table which names conditions the
    problem renamed afterwards (`Fujita_SciSignal2010`) are the ones of the
    measurements, when the table has the observables and the times of the
    measurements row by row.
    """
    v1_dir = v1_yaml.parent
    paths = sorted(v1_dir.glob("*simulation*.tsv"))
    measurements = _v1_table(v1_yaml, "measurement_files") if len(paths) == 1 else None
    for path in paths:
        table = pd.read_csv(path, sep="\t")
        aliases = {
            a: c for a, c in _V1_ALIASES.items() if a in table and c not in table
        }
        renamed = table.rename(columns=aliases)
        repaired = (
            renamed
            if measurements is None
            else _with_conditions_of(
                measurements, renamed, _v1_table(v1_yaml, "condition_files")
            )
        )
        if aliases or repaired is not renamed:
            repaired.to_csv(path, sep="\t", index=False)
    return paths


#: the columns of the conditions of a measurement in PEtab v1
_V1_CONDITIONS: tuple[str, ...] = (
    "simulationConditionId",
    "preequilibrationConditionId",
)


def _v1_table(v1_yaml: Path, key: str) -> pd.DataFrame | None:
    """Get the tables of an entry of a problem of PEtab v1 as one table.

    Args:
        v1_yaml: the YAML file of the problem.
        key: the entry of its problems, e.g. `measurement_files`.

    Returns:
        The tables, `None` without one.
    """
    config: dict[str, Any] = yaml.safe_load(v1_yaml.read_text(encoding="utf-8"))
    names = [
        name for problem in config.get("problems", []) for name in problem.get(key, [])
    ]
    if not names:
        return None
    return pd.concat(
        [pd.read_csv(v1_yaml.parent / name, sep="\t") for name in names],
        ignore_index=True,
    )


def _with_conditions_of(
    measurements: pd.DataFrame,
    simulations: pd.DataFrame,
    conditions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Give simulations of renamed conditions the conditions of the measurements.

    Args:
        measurements: the measurements of the problem.
        simulations: its simulations, which may name conditions the problem
            has no more.
        conditions: the conditions of the problem.

    Returns:
        The simulations, with the conditions of the measurements when some of
        their conditions are unknown and they have the observables and the
        times of the measurements row by row; else the table as it is.
    """
    known = set() if conditions is None else set(conditions["conditionId"].astype(str))
    columns = [c for c in _V1_CONDITIONS if c in simulations]
    used = {
        str(value)
        for column in columns
        for value in simulations[column].dropna()
        if str(value)
    }
    if used <= known or len(simulations) != len(measurements):
        return simulations
    aligned = np.allclose(
        simulations["time"].to_numpy(dtype=float),
        measurements["time"].to_numpy(dtype=float),
        equal_nan=True,
    ) and _same_or_renamed(
        measurements["observableId"].astype(str).to_numpy(),
        simulations["observableId"].astype(str).to_numpy(),
    )
    if not aligned:
        return simulations
    repaired = simulations.copy()
    for column in _V1_CONDITIONS:
        if column in measurements:
            repaired[column] = measurements[column].to_numpy()
        elif column in repaired:
            repaired = repaired.drop(columns=column)
    return repaired


def _fetch_file(url: str, path: Path) -> None:
    """Download a file and move it into place, so it is complete or missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        urllib.request.urlretrieve(url, staging)
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)
