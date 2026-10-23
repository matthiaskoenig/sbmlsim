"""The sources of an uncertainty analysis."""

import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulation.sampling import LogNormal
from sbmlsim.uncertainty import Covariance, Uncertain, UncertaintyModel


def _simulation(**preinit: object) -> Simulation:
    return Simulation(end=24, preinit_changes=preinit)  # ty: ignore[invalid-argument-type]


def test_the_inputs_of_a_simulation_and_the_parameters() -> None:
    dose = Uncertain(Q(100, "mg"), cv=0.1)
    simulation = Simulation(
        end=48,
        preinit_changes={"BW": Uncertain(Q(75, "kg"), cv=0.15, kind="variability")},
        changes=[Change([0, 24], {"PODOSE": dose})],
    )
    uq = UncertaintyModel(parameters={"ka": LogNormal(median=1.0, cv=0.3)})
    variables = uq.variables({"s": simulation})
    assert [(v.key, v.targets, v.source, v.kind) for v in variables] == [
        ("BW", ("BW",), "input", "variability"),
        ("PODOSE", ("PODOSE",), "input", "uncertainty"),
        ("ka", ("ka",), "parameter", "uncertainty"),
    ]
    assert variables[1].distribution is dose.distribution
    assert UncertaintyModel(inputs=False).variables({"s": simulation}) == []


def test_one_variable_by_object_and_by_name() -> None:
    shared = Uncertain(0.3, cv=0.2)
    named = [Uncertain(1.0, cv=0.1, name="scale"), Uncertain(1.0, cv=0.1, name="scale")]
    simulation = _simulation(k1=shared, k2=shared, f1=named[0], f2=named[1])
    variables = UncertaintyModel().variables({"s": simulation})
    assert [(v.key, v.targets) for v in variables] == [
        ("k1", ("k1", "k2")),
        ("scale", ("f1", "f2")),
    ]


def test_the_inputs_of_scenarios_are_paired_by_key() -> None:
    dose = Uncertain(Q(100, "mg"), cv=0.1)
    weight = Uncertain(Q(75, "kg"), cv=0.15)
    variables = UncertaintyModel().variables(
        {"a": _simulation(PODOSE=dose), "b": _simulation(PODOSE=dose, BW=weight)}
    )
    assert [v.key for v in variables] == ["PODOSE", "BW"]


def test_two_values_of_one_name_must_agree() -> None:
    simulation = _simulation(
        f1=Uncertain(1.0, cv=0.1, name="scale"), f2=Uncertain(1.0, cv=0.2, name="scale")
    )
    with pytest.raises(ValueError, match="'scale'"):
        UncertaintyModel().variables({"s": simulation})


def test_a_target_set_uncertain_and_certain_raises() -> None:
    """A draw replaces a target wherever the simulation sets it."""
    simulation = Simulation(
        end=48,
        changes=[
            Change(0, {"PODOSE": Uncertain(Q(100, "mg"), cv=0.1)}),
            Change(24, {"PODOSE": Q(50, "mg")}),
        ],
    )
    with pytest.raises(ValueError, match=r"'PODOSE'.*24"):
        UncertaintyModel().variables({"s": simulation})
    twice = Simulation(
        end=48,
        changes=[
            Change(0, {"PODOSE": Uncertain(Q(100, "mg"), cv=0.1)}),
            Change(24, {"PODOSE": Uncertain(Q(100, "mg"), cv=0.1)}),
        ],
    )
    with pytest.raises(ValueError, match="'PODOSE'"):
        UncertaintyModel().variables({"s": twice})


def test_a_parameter_which_is_an_input_raises() -> None:
    simulation = _simulation(ka=Uncertain(1.0, cv=0.1))
    with pytest.raises(ValueError, match="'ka'"):
        UncertaintyModel(parameters={"ka": Uncertain(1.0, cv=0.2)}).variables(
            {"s": simulation}
        )


def test_from_covariance() -> None:
    omega = Covariance(["CL", "V"], matrix=[[0.09, 0.03], [0.03, 0.04]], scale="log")
    uq = UncertaintyModel.from_covariance({"CL": Q(10, "l/hr"), "V": Q(50, "l")}, omega)
    assert uq.covariance == (omega,)
    cl = uq.parameters["CL"]
    assert isinstance(cl, LogNormal)
    linear = Covariance(["CL", "V"], matrix=[[4.0, 1.2], [1.2, 9.0]])
    normal = UncertaintyModel.from_covariance(
        {"CL": 10.0, "V": 50.0}, linear, shape="normal"
    )
    assert normal.parameters["V"].sd == pytest.approx(3.0)  # ty: ignore[unresolved-attribute]
