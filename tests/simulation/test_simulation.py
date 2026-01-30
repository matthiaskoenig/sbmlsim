"""Test simulations."""

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.resources import REPRESSILATOR_SBML


def test_timecourse_simulation() -> None:
    """Run timecourse simulation."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    tc = Timecourse(start=0, end=100, steps=100)
    s = simulator.run_timecourse(TimecourseSim(tc))
    assert s is not None

    tc = Timecourse(start=0, end=100, steps=100, changes={"PX": 10.0})
    xres = simulator.run_timecourse(TimecourseSim(tc))
    assert xres is not None
    assert hasattr(xres, "_time")
    assert len(xres._time) == 101
    assert xres["[PX]"][0] == 10.0

    tcsim = TimecourseSim(
        timecourses=[Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})]
    )
    xres = simulator.run_timecourse(tcsim)
    assert xres is not None


def test_timecourse_concat() -> None:
    """Reuse of timecourses."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)
    tc = Timecourse(start=0, end=50, steps=100, changes={"X": 10})

    xres = simulator.run_timecourse(simulation=TimecourseSim([tc] * 3))
    assert xres._time.values[-1] == 150.0
    assert len(xres._time) == 3 * 101
    assert xres["[X]"].values[0] == 10.0
    assert xres["[X]"].values[101] == 10.0
    assert xres["[X]"].values[202] == 10.0


def test_timecourse_empty() -> None:
    """Reuse of timecourses."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)
    tc = Timecourse(start=0, end=50, steps=100, changes={"X": 10})

    tcsim = TimecourseSim([None, tc, None])
    xres = simulator.run_timecourse(
        simulation=tcsim,
    )
    assert len(tcsim.timecourses) == 1
    assert xres._time.values[-1] == 50.0
    assert len(xres._time) == 101


def test_timecourse_discard() -> None:
    """Test discarding pre-simulation."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    xres = simulator.run_timecourse(
        simulation=TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=100,
                    steps=100,
                    discard=True,
                    changes={
                        "[X]": 20.0,
                        "[Y]": 20.0,
                        "[Z]": 20.0,
                    },
                ),
                Timecourse(start=0, end=100, steps=100),
            ]
        )
    )
    assert len(xres._time) == 101
    assert xres._time.values[0] == 0.0
    assert xres._time.values[-1] == 100.0
