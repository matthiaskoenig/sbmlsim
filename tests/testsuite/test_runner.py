"""Tests of running cases of the SBML Test Suite in parallel processes."""

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import AbstractModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulator import Simulator
from sbmlsim.testsuite import SemanticSuite, run_case, run_suite, write_submission
from sbmlsim.testsuite.cases import SemanticCase
from sbmlsim.testsuite.runner import CaseResult, CaseStatus, map_cases, simulate_case

SETTINGS = """start: 0
duration: 2.0
steps: 20
variables: S1, S2
absolute: 1e-06
relative: 1e-06
amount: S1, S2
concentration:
"""

MODEL_M = """(*
componentTags: Compartment, Species, Reaction, Parameter
testTags:      Amount
testType:      TimeCourse
*)
"""

#: S1 -> S2 with the rate k1 * S1 on amounts, i.e. S1 = exp(-k1 t)
MODEL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="decay">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialAmount="1" hasOnlySubstanceUnits="true"
        boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
        boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="J1" reversible="false">
        <listOfReactants>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S2" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k1</ci><ci>S1</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _case(path: Path, cid: str, model: str = MODEL_SBML, k1: float = 0.5) -> None:
    """Write a case of the decay of S1 into S2 with its analytic results."""
    directory = path / cid
    directory.mkdir(parents=True)
    (directory / f"{cid}-settings.txt").write_text(SETTINGS)
    (directory / f"{cid}-model.m").write_text(MODEL_M)
    (directory / f"{cid}-sbml-l3v2.xml").write_text(
        model.replace('value="0.5"', f'value="{k1}"')
    )
    time = np.linspace(0.0, 2.0, 21)
    s1 = np.exp(-k1 * time)
    rows = "".join(f"{t},{a},{1 - a}\n" for t, a in zip(time, s1, strict=True))
    (directory / f"{cid}-results.csv").write_text("time,S1,S2\n" + rows)


def _suite(path: Path) -> SemanticSuite:
    """Get a suite of three cases which pass and one which cannot be read."""
    for cid in ("00001", "00002", "00003"):
        _case(path, cid)
    _case(path, "00004", model="<sbml/>")
    return SemanticSuite(version="test", path=path)


def test_the_cases_run_in_parallel_as_in_one_process(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    serial = run_suite(suite, workers=1)
    parallel = run_suite(suite, workers=2)

    assert [(r.cid, r.status) for r in parallel] == [(r.cid, r.status) for r in serial]
    assert [r.status for r in parallel] == [
        CaseStatus.PASS,
        CaseStatus.PASS,
        CaseStatus.PASS,
        CaseStatus.NOT_READ,
    ]


def test_a_case_is_simulated_with_a_simulator_and_its_model(tmp_path: Path) -> None:
    """The model of a case gets the selections and the output of its settings."""
    _case(tmp_path, "00001")
    case = SemanticCase.from_directory(tmp_path / "00001")
    assert case is not None
    simulator = Simulator(
        n_workers=1, absolute_tolerance=1e-12, relative_tolerance=1e-12
    )
    model = simulator.load(AbstractModel(source=case.model_path))

    observed = simulate_case(case, simulator, model)

    assert observed.columns == ("time", "S1", "S2")
    np.testing.assert_allclose(observed.time, np.linspace(0.0, 2.0, 21))
    np.testing.assert_allclose(observed["S1"], np.exp(-0.5 * observed.time), rtol=1e-8)


#: the model of case 01757: two events at t0 with delays, `P1` is 1.5, 5 from
#: 1.3 on and 25 from 2.5 on
EVENTS_SETTINGS = """start: 0
duration: 10
steps: 10
variables: P1
absolute: 0.0001
relative: 0.0001
amount:
concentration:
"""


def _events_case(path: Path) -> SemanticCase:
    """Write a case of two events at t0 with delays, as case 01757."""
    from tests.simulator.models import EVENTS_AT_T0, sbml

    directory = path / "01757"
    directory.mkdir(parents=True)
    (directory / "01757-settings.txt").write_text(EVENTS_SETTINGS)
    (directory / "01757-model.m").write_text(MODEL_M)
    (directory / "01757-sbml-l3v2.xml").write_text(sbml(EVENTS_AT_T0))
    p1 = [1.5, 1.5, 5.0] + [25.0] * 8
    rows = "".join(f"{t},{v}\n" for t, v in zip(range(11), p1, strict=True))
    (directory / "01757-results.csv").write_text("time,P1\n" + rows)
    case = SemanticCase.from_directory(directory)
    assert case is not None
    return case


def test_a_case_is_simulated_twice_on_its_model(tmp_path: Path) -> None:
    """The second simulation of the loaded model of a case is the first (#305).

    The events at t0 of the model fire on every simulation of it, so the case
    passes the repeat check.
    """
    result = run_case(_events_case(tmp_path))
    assert result.status is CaseStatus.PASS, result.message


def test_a_second_simulation_which_differs_is_a_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case whose second simulation differs from its first fails with `repeat`."""
    import sbmlsim.testsuite.runner as runner

    _case(tmp_path, "00001")
    case = SemanticCase.from_directory(tmp_path / "00001")
    assert case is not None
    calls: list[int] = []

    def simulate(*args: object) -> TimecourseResult:
        result = simulate_case(*args)  # ty: ignore[invalid-argument-type]
        calls.append(1)
        if len(calls) == 2:
            values = np.array(result.values, dtype=float)
            values[-1, 1] = np.nextafter(values[-1, 1], np.inf)
            return TimecourseResult(columns=result.columns, values=values)
        return result

    monkeypatch.setattr(runner, "simulate_case", simulate)
    result = run_case(case)
    assert len(calls) == 2
    assert result.status is CaseStatus.REPEAT
    assert "second simulation" in result.message


def _delayed_event_rule_case(path: Path) -> SemanticCase:
    """Write a case of an event with a delay and a species of a rule.

    Its expected results are the simulation of a fresh model, the case is
    about the simulations after it on the loaded model.
    """
    from tests.simulator.models import DELAYED_EVENT_RULE, sbml

    directory = path / "00622"
    directory.mkdir(parents=True)
    (directory / "00622-settings.txt").write_text(
        EVENTS_SETTINGS.replace("variables: P1", "variables: A, R")
    )
    (directory / "00622-model.m").write_text(MODEL_M)
    (directory / "00622-sbml-l3v2.xml").write_text(sbml(DELAYED_EVENT_RULE))
    case = SemanticCase.from_directory(directory)
    assert case is not None
    simulator = Simulator(
        n_workers=1, absolute_tolerance=1e-12, relative_tolerance=1e-12
    )
    result = simulate_case(
        case, simulator, simulator.load(AbstractModel(source=case.model_path))
    )
    rows = "".join(f"{t},{a},{r}\n" for t, a, r in np.asarray(result.values))
    (directory / "00622-results.csv").write_text("time,A,R\n" + rows)
    return case


def test_a_case_is_simulated_after_a_shorter_simulation_and_to_a_steady_state(
    tmp_path: Path,
) -> None:
    """The repeat checks which integrate beyond the simulation before pass (#305).

    A simulation after a shorter one and a steady state regenerate a model
    with an event with a delay, which raised for a species of a rule.
    """
    result = run_case(_delayed_event_rule_case(tmp_path))
    assert result.status is CaseStatus.PASS, result.message


@pytest.mark.parametrize(
    ("call", "message"),
    [(4, "after a shorter one"), (6, "steady state")],
    ids=["after a shorter one", "a steady state"],
)
def test_a_simulation_beyond_the_one_before_which_differs_is_a_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: int, message: str
) -> None:
    """A simulation after a shorter one or to a steady state which differs is a `repeat`.

    The simulations of the case: the first, the same again, the shorter one,
    the full one after it, the steady state of a fresh model and the one of
    the loaded model.
    """
    import sbmlsim.testsuite.runner as runner

    # a decay which reaches its steady state within the duration of the case
    _case(tmp_path, "00001", k1=50.0)
    case = SemanticCase.from_directory(tmp_path / "00001")
    assert case is not None
    calls: list[int] = []
    simulate = Simulator.simulate

    def changed(self: Simulator, *args: Any, **kwargs: Any) -> TimecourseResult:
        calls.append(1)
        result = simulate(self, *args, **kwargs)
        if len(calls) == call:
            values = np.array(result.values, dtype=float)
            values[-1, 1] = np.nextafter(values[-1, 1], np.inf)
            return TimecourseResult(columns=result.columns, values=values)
        return result

    monkeypatch.setattr(runner.Simulator, "simulate", changed)
    result = run_case(case)
    # the check stops at the simulation which differs
    assert len(calls) == call
    assert result.status is CaseStatus.REPEAT
    assert message in result.message


def _interrupted(case: SemanticCase) -> CaseResult:
    """Ctrl-C while a case runs; the worker hands it to the parent."""
    raise KeyboardInterrupt


def test_an_interrupted_run_stops_the_pool(tmp_path: Path) -> None:
    """The workers ignore Ctrl-C, the parent stops them and does not wait for them."""
    cases = list(_suite(tmp_path).cases())
    executor = parallel.pool(2)
    # every worker is started
    list(executor.map(time.sleep, [0.2, 0.2]))
    processes = list(executor._processes.values())
    with pytest.raises(KeyboardInterrupt):
        map_cases(_interrupted, cases, workers=2)
    assert parallel._POOLS == {}
    assert len(processes) == 2
    assert not any(process.is_alive() for process in processes)


def test_the_submission_holds_the_cases_which_could_be_simulated(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "suite")
    path = write_submission(suite, tmp_path / "submission", workers=2)

    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == [
            "00001.csv",
            "00002.csv",
            "00003.csv",
            "manifest.json",
        ]
        header, *rows = zf.read("00002.csv").decode().splitlines()
        manifest = json.loads(zf.read("manifest.json"))

    assert header == "time,S1,S2"
    assert len(rows) == 21
    assert manifest["n_cases"] == 4
    assert manifest["n_submitted"] == 3
    assert manifest["suite_version"] == "test"
