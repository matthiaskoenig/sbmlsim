"""Tests of running cases of the SBML Test Suite in parallel processes."""

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.testsuite import SemanticSuite, run_suite, write_submission
from sbmlsim.testsuite.cases import SemanticCase
from sbmlsim.testsuite.runner import CaseResult, CaseStatus, map_cases

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


def _case(path: Path, cid: str, model: str = MODEL_SBML) -> None:
    """Write a case of the decay of S1 into S2 with its analytic results."""
    directory = path / cid
    directory.mkdir(parents=True)
    (directory / f"{cid}-settings.txt").write_text(SETTINGS)
    (directory / f"{cid}-model.m").write_text(MODEL_M)
    (directory / f"{cid}-sbml-l3v2.xml").write_text(model)
    time = np.linspace(0.0, 2.0, 21)
    s1 = np.exp(-0.5 * time)
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
