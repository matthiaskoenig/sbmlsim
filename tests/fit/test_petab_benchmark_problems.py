"""Tests of reading and running a problem of the PEtab benchmark collection.

The problems are written by the tests, nothing is downloaded: the problem of
PEtab v2 is the one the reader tests write, with the simulations the
collection compares with, converted to PEtab v2 as `BenchmarkCollection`
converts them; a problem of PEtab v1 is converted as the collection is.
"""

import gc
import logging
import signal
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sbmlsim.fit.petab_v2.benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_SETTINGS,
    EXCLUDED_PROBLEMS,
    BenchmarkCollection,
    BenchmarkProblem,
    BenchmarkStatus,
)
from sbmlsim.fit.petab_v2.conversion import SIMULATIONS_FILE
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.testsuite import cache
from tests.fit.petab_v1 import VALUES, write_v1_problem
from tests.fit.test_petab_v2_reader import write_problem

NAME = "Lotka_Volterra"


def _problem(collection: Path, name: str = NAME) -> BenchmarkProblem:
    """Write a problem of a collection whose simulations are its result.

    Args:
        collection: directory of the collection.
        name: name of the problem.

    Returns:
        The problem.
    """
    v2_dir = collection / "v2" / name
    # the experiments differ, the prey starts higher in `e2`
    write_problem(
        v2_dir,
        {"prey_o": "prey"},
        {"e1": None, "e2": "c2"},
        conditions=[("c2", "prey", "3.0")],
    )
    # the problem of the collection, whose encoding is the one above
    (collection / "problems" / name / "v2").mkdir(parents=True)

    reader = PetabReader.from_yaml(v2_dir / "problem.yaml")
    problem = reader.to_optimization_problem()
    problem.initialize(BENCHMARK_SETTINGS)
    nominal = reader.nominal_parameters(problem)
    predictions = problem.predictions(nominal.x(problem.pids))
    index = {key: k for k, key in enumerate(problem.mapping_keys)}
    measurements = pd.read_csv(v2_dir / "measurements.tsv", sep="\t")
    simulations = measurements.rename(columns={"measurement": "simulation"})
    simulations["simulation"] = [
        predictions[index[key]][position] for key, position in reader.measurement_rows()
    ]
    simulations.to_csv(v2_dir / SIMULATIONS_FILE, sep="\t", index=False)
    return BenchmarkProblem(
        name=name, source_dir=collection / "problems" / name, v2_dir=v2_dir
    )


def _simulations(problem: BenchmarkProblem) -> pd.DataFrame:
    return pd.read_csv(problem.v2_dir / SIMULATIONS_FILE, sep="\t")


def _write(problem: BenchmarkProblem, simulations: pd.DataFrame) -> None:
    simulations.to_csv(problem.v2_dir / SIMULATIONS_FILE, sep="\t", index=False)


def _llh(problem: BenchmarkProblem) -> float:
    """Get the log-likelihood of a problem at its nominal values."""
    reader = PetabReader.from_yaml(problem.v2_dir / "problem.yaml")
    optimization_problem = reader.to_optimization_problem()
    optimization_problem.initialize(BENCHMARK_SETTINGS)
    return log_likelihood(
        optimization_problem, reader.nominal_parameters(optimization_problem)
    )


def test_a_problem_passes(tmp_path: Path) -> None:
    """A problem whose simulations agree with the collection passes."""
    result = _problem(tmp_path).run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.name == NAME
    assert result.n_simulations == 20
    assert result.max_difference is not None
    assert result.max_difference < 1e-9
    assert set(result.timings) == {"read", "initialize", "simulate", "llh"}
    assert result.llh is not None
    assert result.llh_reference is None


def test_a_simulation_outside_of_the_tolerance(tmp_path: Path) -> None:
    """A simulation which differs is named with its observable and experiment."""
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    simulations.loc[12, "simulation"] += 1.0
    _write(problem, simulations)
    result = problem.run()
    assert result.status is BenchmarkStatus.TOLERANCE
    assert "prey_o" in result.message
    assert "e2" in result.message
    assert result.max_difference == pytest.approx(1.0, rel=1e-6)


def test_the_simulations_are_found_by_their_keys(tmp_path: Path) -> None:
    """The table of the simulations may have another order and more rows."""
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    extra = simulations.iloc[[0]].assign(time=0.5)
    _write(problem, pd.concat([simulations.iloc[::-1], extra], ignore_index=True))
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_the_log_likelihood_of_amici(tmp_path: Path) -> None:
    """The log-likelihood is compared with the reference of AMICI."""
    problem = _problem(tmp_path)
    llh = _llh(problem)
    result = problem.run(llh_reference=llh)
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.llh_reference == llh

    result = problem.run(llh_reference=llh + 1.0)
    assert result.status is BenchmarkStatus.TOLERANCE
    assert "llh" in result.message


def test_a_problem_which_was_not_converted(tmp_path: Path) -> None:
    """A problem which `petab1to2` cannot convert is the status `conversion`."""
    problem = _problem(tmp_path)
    error = problem.v2_dir.parent / f"{NAME}.error"
    for path in problem.v2_dir.iterdir():
        path.unlink()
    problem.v2_dir.rmdir()
    error.write_text("ValueError: the conversion failed", encoding="utf-8")
    result = problem.run()
    assert result.status is BenchmarkStatus.CONVERSION
    assert "the conversion failed" in result.message


def test_a_problem_which_cannot_be_read(tmp_path: Path) -> None:
    """An error while the problem is read or simulated is the status `error`."""
    problem = _problem(tmp_path)
    (problem.v2_dir / "observables.tsv").write_text("nothing\n", encoding="utf-8")
    result = problem.run()
    assert result.status is BenchmarkStatus.ERROR
    assert result.message


def test_the_collection(tmp_path: Path) -> None:
    """The collection is its problems and the references of AMICI."""
    first = _problem(tmp_path, "A_First")
    _problem(tmp_path, "B_Second")
    (tmp_path / "amici.yaml").write_text(
        yaml.safe_dump({"A_First": {"llh": _llh(first), "t_sim": 0.1}}),
        encoding="utf-8",
    )
    collection = BenchmarkCollection(path=tmp_path, commit="abc")
    assert collection.problem_names() == ["A_First", "B_Second"]
    assert set(collection.references()) == {"A_First"}
    results = collection.run(["A_First"])
    assert [result.name for result in results] == ["A_First"]
    assert results[0].status is BenchmarkStatus.PASS, results[0].message
    assert results[0].llh_reference is not None


def test_the_cache_is_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache holds one directory per commit, the variable overrides it."""
    monkeypatch.delenv("SBMLSIM_BENCHMARK_PATH", raising=False)
    path = BenchmarkCollection.cache_path(BENCHMARK_COMMIT)
    assert path.parent.name == "petab-benchmark"
    assert path.name == BENCHMARK_COMMIT
    assert path.parent.parent == cache.cache_root()
    monkeypatch.setenv("SBMLSIM_BENCHMARK_PATH", "/somewhere")
    assert BenchmarkCollection.cache_path(BENCHMARK_COMMIT) == Path("/somewhere")


def test_simulations_without_a_table_of_the_collection(tmp_path: Path) -> None:
    """A problem without simulations of the collection passes with a note."""
    problem = _problem(tmp_path)
    (problem.v2_dir / SIMULATIONS_FILE).unlink()
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert "no simulations" in result.message
    assert result.n_simulations == 0


def test_simulations_of_renamed_experiments(tmp_path: Path) -> None:
    """A table of the simulations in the order of the measurements is used as is.

    The encoding of PEtab v2 of the collection names its experiments
    otherwise than the conversion of its problem of PEtab v1, whose
    simulations are converted (`Boehm_JProteomeRes2014` has none), and some
    problems renamed their conditions after the simulations were written.
    """
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    simulations["experimentId"] = simulations["experimentId"].map(
        {"e1": "model1_data1", "e2": "model1_data2"}
    )
    _write(problem, simulations)
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_simulations_of_the_experiments_in_another_order(tmp_path: Path) -> None:
    """The observables and the times agree row by row, the experiments not.

    `Liu_IFACPapersOnLine2025` lists the simulations of its second condition
    first; the experiments are the ones of the measurements, so they are
    found by their keys and not by their position.
    """
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    first, second = simulations.iloc[:10], simulations.iloc[10:]
    _write(problem, pd.concat([second, first], ignore_index=True))
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message


def test_simulations_of_renamed_observables(tmp_path: Path) -> None:
    """A table of the simulations whose observables were renamed row by row.

    `Raia_CancerResearch2011` drops the prefix of its observables in the table
    of the simulations, `Isensee_JCB2018` adds the parameters and conditions.
    """
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    simulations["observableId"] = [
        f"prey__{experiment}" for experiment in simulations["experimentId"]
    ]
    _write(problem, simulations)
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_simulations_of_renamed_observables_in_another_order(tmp_path: Path) -> None:
    """A renamed observable is found by the observable it starts with."""
    problem = _problem(tmp_path)
    simulations = _simulations(problem)
    simulations["observableId"] = [
        f"prey_o__{experiment}" for experiment in simulations["experimentId"]
    ]
    _write(problem, simulations.iloc[::-1])
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def _v1_collection(collection: Path, name: str = "Decay") -> Path:
    """Write a problem of PEtab v1 of a collection, see `write_v1_problem`."""
    v1_dir = collection / "problems" / name / "v1"
    write_v1_problem(v1_dir)
    (v1_dir / "_simulations.tsv").rename(v1_dir / "simulations.tsv")
    return v1_dir


def test_a_problem_of_petab_v1_is_converted_with_its_simulations(
    tmp_path: Path,
) -> None:
    """The problem and its simulations are written in PEtab v2 and pass."""
    _v1_collection(tmp_path)
    collection = BenchmarkCollection(path=tmp_path, commit="abc")
    collection.convert()
    problem = collection.problem("Decay")
    assert not problem.native
    config = yaml.safe_load(
        (problem.v2_dir / "problem.yaml").read_text(encoding="utf-8")
    )
    assert config["format_version"] == "2.0.0"
    simulations = _simulations(problem)
    assert "simulationConditionId" not in simulations
    assert simulations["simulation"].tolist() == pytest.approx(list(VALUES))
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 2


def test_the_encoding_of_petab_v2_of_the_collection_is_run(tmp_path: Path) -> None:
    """A problem with an encoding in PEtab v2 is that encoding, not converted.

    Its simulations are converted from PEtab v1, which the collection has
    them in.
    """
    v1_dir = _v1_collection(tmp_path)
    native = write_problem(
        tmp_path / "native",
        {"prey_o": "prey"},
        {"e1": None},
    ).parent
    native.rename(v1_dir.parent / "v2")
    collection = BenchmarkCollection(path=tmp_path, commit="abc")
    collection.convert()
    problem = collection.problem("Decay")
    assert problem.native
    assert (problem.v2_dir / "measurements.tsv").read_text(encoding="utf-8") == (
        v1_dir.parent / "v2" / "measurements.tsv"
    ).read_text(encoding="utf-8")
    assert _simulations(problem)["simulation"].tolist() == pytest.approx(list(VALUES))


def test_simulations_of_conditions_which_were_renamed(tmp_path: Path) -> None:
    """The conditions of simulations the problem renamed are the measurements' ones.

    `Fujita_SciSignal2010` renamed its conditions after the simulations were
    written, which `petab1to2` cannot convert.
    """
    v1_dir = _v1_collection(tmp_path)
    simulations = pd.read_csv(v1_dir / "simulations.tsv", sep="\t")
    simulations["simulationConditionId"] = "model1_data1"
    simulations.to_csv(v1_dir / "simulations.tsv", sep="\t", index=False)
    collection = BenchmarkCollection(path=tmp_path, commit="abc")
    collection.convert()
    result = collection.problem("Decay").run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 2


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason="the time limit needs a POSIX timer"
)
def test_a_problem_which_takes_too_long(tmp_path: Path) -> None:
    """A problem which takes longer than its time is the status `error`."""
    problem = _problem(tmp_path)
    # the timer raises the timeout wherever the problem is, also between the
    # `open` of a reader and the `with` which closes it (`load_yaml` of petab
    # reads the YAML of a problem through `pandas.io.common.get_handle`) or in
    # a walk of a directory: such a handle is unreachable instead of closed,
    # and its finalization raises a `ResourceWarning`, which `filterwarnings =
    # error` turns into an unraisable exception of this test or of whichever
    # test runs when the garbage is collected. The handles of the interrupted
    # run are collected here while the warning is ignored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        result = problem.run(timeout=1e-4)
        gc.collect()
    assert result.status is BenchmarkStatus.ERROR
    assert result.message.startswith("ProblemTimeout")


def test_a_time_limit_without_a_posix_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the timer of the process, e.g. on Windows, the problem has no limit."""
    problem = _problem(tmp_path)
    monkeypatch.delattr(signal, "setitimer", raising=False)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.benchmark"):
        result = problem.run(timeout=1e-4)
    assert result.status is BenchmarkStatus.PASS, result.message
    assert any("time limit" in r.getMessage() for r in caplog.records)


def test_an_excluded_problem_is_not_listed(tmp_path: Path) -> None:
    """A problem which takes too long is left out unless it is asked for."""
    _problem(tmp_path, "A_First")
    (tmp_path / "problems" / "Froehlich_CellSystems2018" / "v1").mkdir(parents=True)
    collection = BenchmarkCollection(path=tmp_path, commit="abc")
    assert "Froehlich_CellSystems2018" in EXCLUDED_PROBLEMS
    assert collection.problem_names() == ["A_First"]
    assert collection.problem_names(excluded=True) == [
        "A_First",
        "Froehlich_CellSystems2018",
    ]
