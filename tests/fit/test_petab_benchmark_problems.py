"""Tests of reading and running a problem of the PEtab benchmark collection.

The problems are written by the tests, nothing is downloaded: the problem of
PEtab v2 is the one the reader tests write, and the tables of PEtab v1 are
the measurements and the simulations the collection compares with.
"""

import logging
import signal
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sbmlsim.fit.petab_v2.benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_SETTINGS,
    BenchmarkCollection,
    BenchmarkProblem,
    BenchmarkStatus,
    llh_is_comparable,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.testsuite import cache
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
    v1_dir = collection / "problems" / name / "v1"
    v1_dir.mkdir(parents=True)

    measurements = pd.read_csv(v2_dir / "measurements.tsv", sep="\t")
    v1 = measurements.rename(columns={"experimentId": "simulationConditionId"})
    v1 = v1.drop(columns=["noiseParameters"])
    v1.to_csv(v1_dir / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"observableId": "prey_o", "observableTransformation": "lin"}]
    ).to_csv(v1_dir / "observables.tsv", sep="\t", index=False)

    reader = PetabReader.from_yaml(v2_dir / "problem.yaml")
    problem = reader.to_optimization_problem()
    problem.initialize(BENCHMARK_SETTINGS)
    nominal = reader.nominal_parameters(problem)
    predictions = problem.predictions(nominal.x(problem.pids))
    index = {key: k for k, key in enumerate(problem.mapping_keys)}
    simulations = v1.rename(columns={"measurement": "simulation"})
    simulations["simulation"] = [
        predictions[index[key]][position] for key, position in reader.measurement_rows()
    ]
    simulations.to_csv(v1_dir / "simulations.tsv", sep="\t", index=False)
    return BenchmarkProblem(name=name, v1_dir=v1_dir, v2_dir=v2_dir)


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
    """A simulation which differs is named with its observable and condition."""
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    simulations.loc[12, "simulation"] += 1.0
    simulations.to_csv(problem.v1_dir / "simulations.tsv", sep="\t", index=False)
    result = problem.run()
    assert result.status is BenchmarkStatus.TOLERANCE
    assert "prey_o" in result.message
    assert "e2" in result.message
    assert result.max_difference == pytest.approx(1.0, rel=1e-6)


def test_the_simulations_are_found_by_their_keys(tmp_path: Path) -> None:
    """The table of the simulations may have another order and more rows."""
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    extra = simulations.iloc[[0]].assign(time=0.5)
    simulations = pd.concat([simulations.iloc[::-1], extra], ignore_index=True)
    simulations.to_csv(problem.v1_dir / "simulations.tsv", sep="\t", index=False)
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


def test_a_log10_observable_has_no_comparable_llh(tmp_path: Path) -> None:
    """The conversion to PEtab v2 changes the density of a log10 observable."""
    problem = _problem(tmp_path)
    assert llh_is_comparable(problem.v1_dir)
    pd.DataFrame(
        [{"observableId": "prey_o", "observableTransformation": "log10"}]
    ).to_csv(problem.v1_dir / "observables.tsv", sep="\t", index=False)
    assert not llh_is_comparable(problem.v1_dir)
    result = problem.run(llh_reference=1e6)
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.llh_reference is None


def test_a_problem_which_was_not_converted(tmp_path: Path) -> None:
    """A problem which `petab1to2` cannot convert is the status `conversion`."""
    problem = _problem(tmp_path)
    error = problem.v2_dir.parent / f"{NAME}.error"
    for path in problem.v2_dir.iterdir():
        path.unlink()
    problem.v2_dir.rmdir()
    error.write_text("ValueError: the conversion failed")
    result = problem.run()
    assert result.status is BenchmarkStatus.CONVERSION
    assert "the conversion failed" in result.message


def test_a_problem_which_cannot_be_read(tmp_path: Path) -> None:
    """An error while the problem is read or simulated is the status `error`."""
    problem = _problem(tmp_path)
    (problem.v2_dir / "observables.tsv").write_text("nothing\n")
    result = problem.run()
    assert result.status is BenchmarkStatus.ERROR
    assert result.message


def test_the_collection(tmp_path: Path) -> None:
    """The collection is its problems and the references of AMICI."""
    first = _problem(tmp_path, "A_First")
    _problem(tmp_path, "B_Second")
    (tmp_path / "amici.yaml").write_text(
        yaml.safe_dump({"A_First": {"llh": _llh(first), "t_sim": 0.1}})
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


def test_simulations_of_renamed_conditions(tmp_path: Path) -> None:
    """A table of the simulations in the order of the measurements is used as is.

    Some problems of the collection renamed their conditions after the
    simulations were written (`Fujita_SciSignal2010`).
    """
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    simulations["simulationConditionId"] = simulations["simulationConditionId"].map(
        {"e1": "model1_data1", "e2": "model1_data2"}
    )
    simulations.to_csv(problem.v1_dir / "simulations.tsv", sep="\t", index=False)
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_simulations_with_the_values_of_the_parameters(tmp_path: Path) -> None:
    """The observable parameters of a simulation may be their values.

    `Elowitz_Nature2000` writes the values of the observable parameters into
    the table of the simulations, which is in another order here.
    """
    problem = _problem(tmp_path)
    measurements = pd.read_csv(problem.v1_dir / "measurements.tsv", sep="\t")
    measurements["observableParameters"] = "scale;offset"
    measurements.to_csv(problem.v1_dir / "measurements.tsv", sep="\t", index=False)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    simulations["observableParameters"] = "1.5;-0.3"
    simulations.iloc[::-1].to_csv(
        problem.v1_dir / "simulations.tsv", sep="\t", index=False
    )
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_simulations_of_the_conditions_in_another_order(tmp_path: Path) -> None:
    """The observables and the times agree row by row, the conditions not.

    `Liu_IFACPapersOnLine2025` lists the simulations of its second condition
    first; the conditions are the ones of the measurements, so they are found
    by their keys and not by their position.
    """
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    first, second = simulations.iloc[:10], simulations.iloc[10:]
    simulations = pd.concat([second, first], ignore_index=True)
    simulations.to_csv(problem.v1_dir / "simulations.tsv", sep="\t", index=False)
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message


def test_simulations_of_renamed_observables(tmp_path: Path) -> None:
    """A table of the simulations whose observables were renamed row by row.

    `Raia_CancerResearch2011` drops the prefix of its observables in the table
    of the simulations, `Isensee_JCB2018` adds the parameters and conditions.
    """
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    simulations["observableId"] = [
        f"prey__{condition}" for condition in simulations["simulationConditionId"]
    ]
    simulations.to_csv(problem.v1_dir / "simulations.tsv", sep="\t", index=False)
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


def test_simulations_of_renamed_observables_in_another_order(tmp_path: Path) -> None:
    """A renamed observable is found by the observable it starts with."""
    problem = _problem(tmp_path)
    simulations = pd.read_csv(problem.v1_dir / "simulations.tsv", sep="\t")
    simulations["observableId"] = [
        f"prey_o__{condition}" for condition in simulations["simulationConditionId"]
    ]
    simulations.iloc[::-1].to_csv(
        problem.v1_dir / "simulations.tsv", sep="\t", index=False
    )
    result = problem.run()
    assert result.status is BenchmarkStatus.PASS, result.message
    assert result.n_simulations == 20


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason="the time limit needs a POSIX timer"
)
def test_a_problem_which_takes_too_long(tmp_path: Path) -> None:
    """A problem which takes longer than its time is the status `error`."""
    result = _problem(tmp_path).run(timeout=1e-4)
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
