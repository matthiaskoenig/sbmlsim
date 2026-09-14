"""Tests of reading a case of the SBML Test Suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.testsuite.cases import SemanticCase, SemanticSuite
from sbmlsim.testsuite.comparison import compare_case

SETTINGS = """start: 0
duration: 2.0
steps: 50
variables: S1, S2
absolute: 0.0001
relative: 0.0001
amount: S1
concentration: S2
"""

MODEL_M = """(*

category:      Test
synopsis:      Basic single forward reaction with two species.
componentTags: Compartment, Species, Reaction, Parameter
testTags:      Amount, NonConstantParameter
testType:      TimeCourse
levels:        2.1, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment.
*)
"""


def _case(path: Path, cid: str = "00001", settings: str = SETTINGS) -> Path:
    """Write a case directory with its metadata and an l3v2 and an l2v1 model."""
    directory = path / cid
    directory.mkdir(parents=True)
    (directory / f"{cid}-settings.txt").write_text(settings)
    (directory / f"{cid}-model.m").write_text(MODEL_M)
    (directory / f"{cid}-sbml-l3v2.xml").write_text("<sbml/>")
    (directory / f"{cid}-sbml-l2v1.xml").write_text("<sbml/>")
    (directory / f"{cid}-results.csv").write_text(
        "time,S1,S2\n0,1.0,0.0\n1,0.5,0.5\n2,0.25,0.75\n"
    )
    return directory


def test_a_case_is_read_from_its_directory(tmp_path: Path) -> None:
    """The settings and the tags of a case are read."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None

    assert case.cid == "00001"
    assert (case.start, case.duration, case.steps) == (0.0, 2.0, 50)
    assert case.variables == ["S1", "S2"]
    assert case.absolute_tolerance == 1e-4
    assert case.relative_tolerance == 1e-4
    assert case.component_tags == {"Compartment", "Species", "Reaction", "Parameter"}
    assert case.test_tags == {"Amount", "NonConstantParameter"}
    assert case.test_type == "TimeCourse"
    assert case.generated_by == "Analytic"
    assert not case.packages


def test_the_newest_encoding_is_simulated(tmp_path: Path) -> None:
    """A case is run once, in the newest encoding it provides."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None
    assert case.model_path.name == "00001-sbml-l3v2.xml"
    assert case.encoding == "l3v2"


def test_a_concentration_is_selected_as_a_concentration(tmp_path: Path) -> None:
    """The amount and concentration lists decide the roadrunner selection."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None
    assert case.selections == ["time", "S1", "[S2]"]


def test_a_case_without_a_duration_is_not_a_timecourse(tmp_path: Path) -> None:
    """The flux balance cases have no duration and are not run."""
    settings = (
        "start:\nduration:\nsteps:\nvariables: R01\nabsolute: 0.001\nrelative: 0.001\n"
    )
    assert SemanticCase.from_directory(_case(tmp_path, settings=settings)) is None


def test_a_directory_without_settings_is_not_a_case(tmp_path: Path) -> None:
    """A directory which is not a case is skipped rather than raising."""
    (tmp_path / "00002").mkdir()
    assert SemanticCase.from_directory(tmp_path / "00002") is None


def test_the_time_column_is_normalized(tmp_path: Path) -> None:
    """Some cases spell the time column `Time`, which is renamed."""
    directory = _case(tmp_path)
    (directory / "00001-results.csv").write_text("Time,S1,S2\n0,1.0,0.0\n")
    case = SemanticCase.from_directory(directory)
    assert case is not None
    assert "time" in case.expected().columns


def test_the_suite_iterates_its_cases(tmp_path: Path) -> None:
    """A suite yields the timecourse cases of its directory, by identifier."""
    _case(tmp_path, cid="00002")
    _case(tmp_path, cid="00001")
    (tmp_path / "not-a-case").mkdir()
    suite = SemanticSuite(path=tmp_path, version="0.0.0")

    assert [case.cid for case in suite.cases()] == ["00001", "00002"]
    assert len(suite) == 2
    assert suite.case("00001").cid == "00001"
    with pytest.raises(ValueError, match="no timecourse case"):
        suite.case("00003")


def test_the_cache_is_per_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """A release is cached under its version, and can be pointed elsewhere."""
    monkeypatch.delenv("SBMLSIM_TEST_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert SemanticSuite.cache_path("3.5.0") == Path(
        "/tmp/cache/sbmlsim/test-suite/3.5.0/semantic"
    )

    monkeypatch.setenv("SBMLSIM_TEST_SUITE_PATH", "/elsewhere/semantic")
    assert SemanticSuite.cache_path("3.5.0") == Path("/elsewhere/semantic")


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def _observed(s1: list[float], s2: list[float]) -> pd.DataFrame:
    """Build the results of a simulation of the fixture case."""
    return pd.DataFrame({"time": [0.0, 1.0, 2.0], "S1": s1, "[S2]": s2})


def test_a_simulation_within_the_tolerances_passes(tmp_path: Path) -> None:
    """The expected results reproduced exactly are within the tolerances."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None
    comparison = compare_case(case, _observed([1.0, 0.5, 0.25], [0.0, 0.5, 0.75]))

    assert comparison.valid
    assert comparison.n_violations == 0
    assert comparison.n_points == 6


def test_a_deviation_beyond_the_tolerances_fails(tmp_path: Path) -> None:
    """A point outside `abs_tol + rel_tol * |c|` is a violation and is named."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None
    comparison = compare_case(case, _observed([1.0, 0.5, 0.9], [0.0, 0.5, 0.75]))

    assert not comparison.valid
    assert comparison.n_violations == 1
    assert comparison.worst_variable == "S1"
    assert comparison.worst_expected == 0.25
    assert comparison.worst_observed == 0.9
    assert comparison.worst_time == 2.0
    assert "outside the tolerances" in comparison.summary


def test_an_undefined_value_agrees_with_an_undefined_value(tmp_path: Path) -> None:
    """A value which is expected to be undefined may come out undefined."""
    directory = _case(tmp_path)
    (directory / "00001-results.csv").write_text(
        "time,S1,S2\n0,1.0,0.0\n1,nan,0.5\n2,0.25,0.75\n"
    )
    case = SemanticCase.from_directory(directory)
    assert case is not None

    assert compare_case(case, _observed([1.0, np.nan, 0.25], [0.0, 0.5, 0.75])).valid
    # a number where nothing is defined is still wrong
    assert not compare_case(case, _observed([1.0, 7.0, 0.25], [0.0, 0.5, 0.75])).valid


def test_a_variable_which_was_not_simulated_is_reported(tmp_path: Path) -> None:
    """A missing variable is its own outcome, not a tolerance violation."""
    case = SemanticCase.from_directory(_case(tmp_path))
    assert case is not None
    comparison = compare_case(
        case, pd.DataFrame({"time": [0.0, 1.0, 2.0], "S1": [1.0, 0.5, 0.25]})
    )

    assert not comparison.valid
    assert comparison.missing == ["S2"]
    assert "not simulated" in comparison.summary
