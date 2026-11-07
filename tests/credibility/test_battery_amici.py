"""The check `cross_simulator` of the battery with AMICI, which the extra `amici` installs.

AMICI compiles a variant in a worker with a deadline
(`sbmlsim.credibility.battery.with_deadline`); a model AMICI does not
simulate as roadrunner and a compilation which takes too long are not
compared. The tests skip without AMICI; the job `amici engine` of the
continuous integration runs them.
"""

from pathlib import Path

import antimony
import pytest

from sbmlsim.credibility import BatteryStatus, run_battery
from sbmlsim.credibility.verification import CROSS_SIMULATOR
from sbmlsim.engine import using_amici_cache
from sbmlsim.project import CredibilitySettings, Project

pytest.importorskip("amici")

#: a source of A at its steady state which flows into the sink B
STEADY = """
model steady
  compartment V = 1;
  species A in V = 2; species B in V = 0;
  k0 = 1; k = 0.5;
  SOURCE: => A; k0;
  FLOW: A => B; k * A;
end
"""

#: a model with a delayed event, which AMICI does not compile
DELAYED = """
model delayed
  compartment V = 1;
  species A in V = 2; species B in V = 0;
  E: at 5 after (time >= 0): B = 5;
end
"""


def _project(tmp_path: Path, text: str, **settings: object) -> Project:
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(text) < 0:
        raise ValueError(antimony.getLastError())
    path = tmp_path / "model.xml"
    path.write_text(
        antimony.getSBMLString(antimony.getMainModuleName()), encoding="utf-8"
    )
    return Project(
        name="amici",
        root=tmp_path,
        variants={"default": path},
        credibility_settings=CredibilitySettings(
            battery_settings={"engines": ["amici"], **settings}
        ),
    )


def test_amici_gives_the_results_of_roadrunner(tmp_path: Path) -> None:
    result = run_battery(_project(tmp_path, STEADY), rebuild=False).result(
        CROSS_SIMULATOR
    )
    assert result.status is BatteryStatus.PASSED, result.problems
    assert result.cases == 1
    assert result.statement.startswith("AMICI gives the results of roadrunner")


def test_a_model_amici_does_not_simulate(tmp_path: Path) -> None:
    result = run_battery(_project(tmp_path, DELAYED), rebuild=False).result(
        CROSS_SIMULATOR
    )
    assert result.status is BatteryStatus.NOT_APPLICABLE
    assert "default: not compared with AMICI: The model" in result.statement
    assert "the delay of the event 'E'" in result.statement


def test_a_compilation_which_takes_too_long(tmp_path: Path) -> None:
    """A model which is not in the cache compiles for longer than a millisecond."""
    project = _project(tmp_path, STEADY, compile_timeout=0.001)
    with using_amici_cache(str(tmp_path / "amici")):
        result = run_battery(project, rebuild=False).result(CROSS_SIMULATOR)
    assert result.status is BatteryStatus.NOT_APPLICABLE
    assert result.statement == (
        "No case was compared. default: not compared with AMICI: compiling "
        "took longer than 0.001 s."
    )
