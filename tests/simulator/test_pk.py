"""The PK analysis of a chunk with pkpdutils."""

import numpy as np
import pkpdutils as pk
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation, SteadyState
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.pk import DoseSpec, PKNode, compile_pk, doses_of, evaluate_pk
from sbmlsim.simulator.plan import Plan
from sbmlsim.units import ureg
from tests.simulator.models import sbml_pk

KA, KE, V = 1.0, 0.2, 10.0
DOSE = DoseSpec(target="PODOSE", amount=0.0, unit="mg")


def concentration(t: np.ndarray, dose: float, at: float = 0.0) -> np.ndarray:
    tau = np.clip(t - at, 0.0, None)
    return np.where(
        t >= at,
        dose * KA / (V * (KA - KE)) * (np.exp(-KE * tau) - np.exp(-KA * tau)),
        0.0,
    )


@pytest.fixture(scope="module")
def plans() -> dict[str, Plan]:
    simulator = Simulator()
    model = simulator.load(sbml_pk())

    def plan(simulation: Simulation) -> Plan:
        return simulator.compile(model, simulation)

    return {
        "single": plan(
            Simulation(end=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})])
        ),
        "late": plan(Simulation(end=48, changes=[Change(10, {"PODOSE": 50.0})])),
        "multiple": plan(
            Simulation(end=48, changes=[Change([0, 24], {"PODOSE": Q(100, "mg")})])
        ),
        "preinit": plan(
            Simulation(
                end=48,
                preinit_changes={"PODOSE": 5.0},
                changes=[Change(0, {"PODOSE": 100.0})],
            )
        ),
        "shifted": plan(
            Simulation(end=48, time_shift=-12, changes=[Change(0, {"PODOSE": 100.0})])
        ),
        "formula": plan(Simulation(end=48, changes=[Change(0, {"PODOSE": "2 * ka"})])),
        "none": plan(Simulation(end=48)),
        "presimulation": plan(
            Simulation(
                end=48,
                presimulation=SteadyState(preinit_changes={"PODOSE": 3.0}),
                changes=[Change(0, {"PODOSE": 100.0})],
            )
        ),
    }


def test_the_doses_are_the_values_the_plan_assigns(plans: dict[str, Plan]) -> None:
    times, amounts = doses_of(plans["single"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]
    times, amounts = doses_of(plans["late"], DOSE)
    assert times.tolist() == [10.0] and amounts.tolist() == [50.0]
    times, amounts = doses_of(plans["multiple"], DOSE)
    assert times.tolist() == [0.0, 24.0] and amounts.tolist() == [100.0, 100.0]


def test_a_change_at_the_start_replaces_the_value_before_the_initialization(
    plans: dict[str, Plan],
) -> None:
    times, amounts = doses_of(plans["preinit"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]


def test_the_dose_times_are_shifted_like_the_result(plans: dict[str, Plan]) -> None:
    times, _ = doses_of(plans["shifted"], DOSE)
    assert times.tolist() == [-12.0]


def test_the_values_of_a_presimulation_are_no_doses(plans: dict[str, Plan]) -> None:
    times, amounts = doses_of(plans["presimulation"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]


def test_no_dose_and_a_fixed_dose(plans: dict[str, Plan]) -> None:
    times, amounts = doses_of(plans["none"], DOSE)
    assert times.size == 0 and amounts.size == 0
    fixed = DoseSpec(target=None, amount=7.0, unit="mg")
    times, amounts = doses_of(plans["none"], fixed)
    assert times.tolist() == [0.0] and amounts.tolist() == [7.0]


def test_a_formula_dose_raises(plans: dict[str, Plan]) -> None:
    with pytest.raises(ValueError, match="formula"):
        doses_of(plans["formula"], DOSE)


def _compile(observable: PK, *plan_list: Plan) -> tuple[PKNode, dict[str, str]]:
    dose_unit = "mg" if isinstance(observable.dose, str) else None
    return compile_pk(
        observable, unit="mg/l", time_unit="hr", dose_unit=dose_unit, plans=plan_list
    )


def test_the_parameters_and_their_units_come_from_pkpdutils(
    plans: dict[str, Plan],
) -> None:
    node, units = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f", "flags"):
        assert f"c.{parameter}" in units
    assert ureg.Quantity(1.0, units["c.cmax"]).to("mg/l").magnitude == pytest.approx(
        1.0
    )
    assert ureg.Quantity(1.0, units["c.tmax"]).to("hr").magnitude == pytest.approx(1.0)
    assert node.outputs == tuple(units)


def test_a_multiple_dosing_adds_the_parameters_of_its_interval(
    plans: dict[str, Plan],
) -> None:
    _, single = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    _, both = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"], plans["multiple"]
    )
    assert "c.tau" not in single
    assert "c.tau" in both and set(single) < set(both)


def test_without_a_dose_the_parameters_of_the_dose_are_left_out(
    plans: dict[str, Plan],
) -> None:
    _, units = _compile(PK("c", "[C]"), plans["single"])
    assert "c.cmax" in units and "c.cl_f" not in units


def test_the_parameters_to_keep_and_the_flags(plans: dict[str, Plan]) -> None:
    node, units = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral", parameters=("cmax",)),
        plans["single"],
    )
    assert node.parameters == ("cmax", "flags")
    assert list(units) == ["c.cmax", "c.flags"]
    with pytest.raises(ValueError, match="nope"):
        _compile(
            PK("c", "[C]", dose="PODOSE", route="oral", parameters=("nope",)),
            plans["single"],
        )


@pytest.mark.parametrize(
    ("route", "match"), [("intranasal", "route"), ("iv_infusion", "infusion")]
)
def test_a_route_pkpdutils_cannot_take_raises(
    plans: dict[str, Plan], route: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _compile(PK("c", "[C]", dose="PODOSE", route=route), plans["single"])


def test_a_timecourse_or_a_dose_without_a_unit_raises(plans: dict[str, Plan]) -> None:
    with pytest.raises(ValueError, match="unit"):
        compile_pk(
            PK("c", "[C]"),
            unit="",
            time_unit="hr",
            dose_unit=None,
            plans=[plans["single"]],
        )
    with pytest.raises(ValueError, match="unit"):
        compile_pk(
            PK("c", "[C]", dose="PODOSE", route="oral"),
            unit="mg/l",
            time_unit="hr",
            dose_unit="",
            plans=[plans["single"]],
        )


def _direct(
    time: np.ndarray,
    values: np.ndarray,
    dose_times: np.ndarray,
    amounts: np.ndarray,
) -> pk.NCAResult:
    timecourses = pk.Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("_sim",),
        dose={"amount": amounts, "time": dose_times, "unit": "mg"},
        route="oral",
    )
    return pk.nca(timecourses)


def test_the_analysis_equals_pkpdutils_on_the_same_arrays(
    plans: dict[str, Plan],
) -> None:
    node, _ = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    t = np.linspace(0.0, 48.0, 97)
    time = np.vstack([t, t])
    values = np.vstack([concentration(t, 100.0), concentration(t, 50.0, at=10.0)])
    out = evaluate_pk(node, time, values, [plans["single"], plans["late"]])
    direct = _direct(
        time, values, np.array([[0.0], [10.0]]), np.array([[100.0], [50.0]])
    )
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f", "flags"):
        np.testing.assert_allclose(
            out[f"c.{parameter}"][:, 0], np.asarray(direct.ds[parameter].values, float)
        )
    assert out["c.auc_inf_obs"][0, 0] == pytest.approx(100.0 / (V * KE), rel=1e-2)
    assert out["c.thalf"][0, 0] == pytest.approx(np.log(2.0) / KE, rel=1e-2)


def test_ragged_rows_are_analysed_without_their_padding(
    plans: dict[str, Plan],
) -> None:
    node, _ = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    t = np.linspace(0.0, 48.0, 97)
    short = np.full(97, np.nan)
    short[:65] = t[:65]
    time = np.vstack([t, short])
    values = concentration(np.where(np.isfinite(time), time, 0.0), 100.0)
    values[1, 65:] = np.nan
    out = evaluate_pk(node, time, values, [plans["single"], plans["single"]])
    alone = evaluate_pk(node, short[None, :65], values[1:, :65], [plans["single"]])
    np.testing.assert_allclose(out["c.cmax"][1], alone["c.cmax"][0])
    np.testing.assert_allclose(out["c.auc_last"][1], alone["c.auc_last"][0])


def test_points_with_different_dosings_are_analysed_on_their_own(
    plans: dict[str, Plan],
) -> None:
    node, _ = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"], plans["multiple"]
    )
    t = np.linspace(0.0, 48.0, 97)
    single = concentration(t, 100.0)
    multiple = single + concentration(t, 100.0, at=24.0)
    out = evaluate_pk(
        node,
        np.vstack([t, t]),
        np.vstack([single, multiple]),
        [plans["single"], plans["multiple"]],
    )
    direct = _direct(
        t[None, :],
        multiple[None, :],
        np.array([[0.0, 24.0]]),
        np.array([[100.0, 100.0]]),
    )
    assert out["c.tau"][1, 0] == pytest.approx(float(direct.ds["tau"].values[0]))
    assert np.isnan(out["c.tau"][0, 0])
    assert out["c.cmax"][0, 0] == pytest.approx(single.max(), rel=1e-12)
