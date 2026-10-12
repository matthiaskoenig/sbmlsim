"""Reserved result names do not collide with scanned model targets."""

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import OnError
from tests.simulator.models import sbml


def _model(target: str = "status") -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(
        source=sbml(f"model status_model\n S=1; {target}=1; S'={target};\nend"),
        selections=["time", "S"],
    )


@pytest.mark.parametrize("on_error", ["raise", "flag"])
@pytest.mark.parametrize("target", ["status", "_point", "statistic"])
def test_a_scanned_reserved_parameter_is_qualified(
    on_error: OnError, target: str
) -> None:
    scan = Scan(
        Simulation(end=1, steps=1), [Dimension("rate", values={target: [1, 2]})]
    )
    result = Simulator(n_workers=1).run(_model(target), scan, on_error=on_error)
    assert result[f"rate.{target}"].values.tolist() == [1, 2]
    assert result["S"].values.tolist() == [[1, 2], [1, 3]]
    if on_error == "flag":
        assert result["status"].values.tolist() == [0, 0]
    else:
        assert "status" not in result


def test_a_coordinate_cannot_hide_a_qualified_target() -> None:
    scan = Scan(
        Simulation(end=1, steps=1),
        [
            Dimension(
                "rate", values={"status": [1, 2]}, coordinates={"rate.status": [3, 4]}
            )
        ],
    )
    with pytest.raises(ValueError, match=r"rate\.status.*changed target"):
        Simulator(n_workers=1).run(_model(), scan, on_error="flag")


def test_a_dimension_cannot_hide_a_qualified_target() -> None:
    scan = Scan(
        Simulation(end=1, steps=1),
        [
            Dimension("rate", values={"status": [1, 2]}),
            Dimension("rate.status", values={"S": [1, 2]}),
        ],
    )
    with pytest.raises(ValueError, match=r"rate\.status.*hide a dimension"):
        Simulator(n_workers=1).run(_model(), scan, on_error="flag")
