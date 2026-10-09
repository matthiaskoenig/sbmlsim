"""The observables of a run, compiled against its model and evaluated."""

import pickle

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import PK, Change, Custom, Formula, ObservableKind, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.observables import (
    FormulaNode,
    ObservableError,
    compile_observables,
    identity_graph,
)
from sbmlsim.simulator.plan import Plan
from sbmlsim.units import ureg
from tests.simulator.models import auc_of_c, doubled, sbml, sbml_pk

SCALAR, TIMECOURSE = ObservableKind.SCALAR, ObservableKind.TIMECOURSE


@pytest.fixture(scope="module")
def pk_model() -> RoadrunnerSBMLModel:
    return Simulator().load(sbml_pk())


@pytest.fixture(scope="module")
def plan(pk_model: RoadrunnerSBMLModel) -> Plan:
    simulation = Simulation(end=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})])
    return Simulator().compile(pk_model, simulation)


def _equal(unit: str, expected: str) -> bool:
    return ureg.Quantity(1.0, unit).to(expected).magnitude == pytest.approx(1.0)


def test_has_selection(pk_model: RoadrunnerSBMLModel) -> None:
    for name in ("time", "C", "[C]", "ke", "V", "absorption"):
        assert pk_model.has_selection(name)
    assert not pk_model.has_selection("nope")


def test_the_kinds_follow_what_a_formula_reads(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            Formula("cmax", "max(c)"),
            Formula("rel", "c / cmax"),
            Formula("two", "2"),
            Formula("late", "at(c, 10) + cmax"),
        ],
        pk_model,
    )
    assert graph.kinds["c"] is TIMECOURSE and graph.kinds["rel"] is TIMECOURSE
    assert graph.kinds["cmax"] is SCALAR and graph.kinds["two"] is SCALAR
    assert graph.kinds["late"] is SCALAR
    assert graph.timecourses == ("c", "rel") and graph.scalars == (
        "cmax",
        "two",
        "late",
    )
    assert graph.selections == ("[C]",)


def test_the_units_are_derived(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            Formula("rel", "c / max(c)"),
            Formula("exposure", "mean(c) * time"),
        ],
        pk_model,
    )
    assert _equal(graph.units["c"], "mg/l")
    assert _equal(graph.units["rel"], "dimensionless")
    assert _equal(graph.units["exposure"], "mg*hr/l")


def test_a_declared_unit_converts_the_values(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables([Formula("c", "[C]", unit="ng/ml")], pk_model)
    (node,) = graph.nodes
    assert graph.units["c"] == "ng/ml"
    assert isinstance(node, FormulaNode)
    assert node.factor == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="cannot be converted"):
        compile_observables([Formula("c", "[C]", unit="hr")], pk_model)


def test_a_unit_which_cannot_be_derived_is_declared(
    pk_model: RoadrunnerSBMLModel,
) -> None:
    with pytest.raises(ValueError, match="unit="):
        compile_observables([Formula("high", "piecewise(1, [C] > 2, 0)")], pk_model)
    graph = compile_observables(
        [Formula("high", "piecewise(1, [C] > 2, 0)", unit="dimensionless")], pk_model
    )
    assert graph.units["high"] == "dimensionless"


def test_the_symbols_are_checked(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="'nope'"):
        compile_observables([Formula("x", "nope * 2")], pk_model)
    with pytest.raises(ValueError, match="selections of the model"):
        compile_observables([Formula("ke", "2")], pk_model)
    with pytest.raises(ValueError, match="Two observables"):
        compile_observables([Formula("x", "1"), Formula("x", "2")], pk_model)


def test_a_cycle_is_reported(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="cycle: a -> b -> a"):
        compile_observables([Formula("a", "b + 1"), Formula("b", "a * 2")], pk_model)
    with pytest.raises(ValueError, match="cycle"):
        compile_observables([Formula("a", "a + 1")], pk_model)


def test_the_time_of_at_is_no_timecourse(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="at"):
        compile_observables([Formula("x", "at([C], [C])")], pk_model)


def test_keep_and_the_observables_it_needs(pk_model: RoadrunnerSBMLModel) -> None:
    observables = [
        Formula("c", "[C]"),
        Formula("cmax", "max(c)"),
        Formula("rel", "c / cmax"),
        Formula("unused", "ke * 2"),
    ]
    graph = compile_observables(observables, pk_model, keep=["rel"])
    assert graph.keep == ("rel",)
    assert [node.id for node in graph.nodes] == ["c", "cmax", "rel"]
    assert graph.selections == ("[C]",)
    with pytest.raises(ValueError, match="'nope'"):
        compile_observables(observables, pk_model, keep=["nope"])
    with pytest.raises(TypeError):
        compile_observables(observables, pk_model, keep="rel")


def test_the_id_of_a_pk_observable_keeps_its_parameters(
    pk_model: RoadrunnerSBMLModel, plan: Plan
) -> None:
    observables = [
        PK("p", "[C]", dose="PODOSE", route="oral"),
        Formula("ratio", "p.auc_inf_obs / p.cmax"),
    ]
    graph = compile_observables(observables, pk_model, keep=["p"], plans=[plan])
    assert "p.cmax" in graph.keep and "p.flags" in graph.keep
    assert "ratio" not in graph.keep
    assert graph.doses == ("PODOSE",)
    full = compile_observables(observables, pk_model, plans=[plan])
    assert _equal(full.units["ratio"], "hr")
    with pytest.raises(ValueError, match=r"p\.nope"):
        compile_observables(
            [*observables, Formula("x", "p.nope")], pk_model, plans=[plan]
        )


def test_the_graph_of_a_run_without_observables(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(None, pk_model)
    expected = tuple(s for s in pk_model.selections or [] if s != "time")
    assert graph.selections == expected and graph.keep == expected
    assert graph.nodes == () and graph.scalars == ()
    assert identity_graph(["[C]", "ke"], keep=["ke"]).keep == ("ke",)


T = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, np.nan]])
C = np.array([[0.0, 4.0, 2.0, 1.0], [0.0, 8.0, 4.0, np.nan]])


def test_the_graph_evaluates_on_the_arrays_of_a_chunk(
    pk_model: RoadrunnerSBMLModel, plan: Plan
) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]", unit="ng/ml"),
            Formula("cmax", "max([C])"),
            Formula("rel", "[C] / cmax"),
            Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
            Custom("twice", doubled, "mg/l", symbols=["[C]"], kind=TIMECOURSE),
        ],
        pk_model,
    )
    out = graph.evaluate(T, {"[C]": C}, [plan, plan])
    np.testing.assert_allclose(out["c"], C * 1000.0)
    np.testing.assert_allclose(out["cmax"], [4.0, 8.0])
    np.testing.assert_allclose(out["rel"], C / np.array([[4.0], [8.0]]))
    np.testing.assert_allclose(
        out["auc"], [np.trapezoid(C[0], T[0]), np.trapezoid(C[1, :3], T[1, :3])]
    )
    np.testing.assert_allclose(out["twice"][:, :3], 2.0 * C[:, :3])
    assert np.isnan(out["twice"][1, 3])


def test_a_division_by_zero_gives_inf_without_a_warning(
    pk_model: RoadrunnerSBMLModel,
) -> None:
    graph = compile_observables([Formula("x", "1 / at([C], 0)")], pk_model)
    out = graph.evaluate(T, {"[C]": C}, [])
    assert np.isinf(out["x"]).all()


def test_a_custom_which_fails_names_its_row(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"])], pk_model
    )
    broken = {"[C]": np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, np.nan]])}
    out = graph.evaluate(T, broken, [])
    assert np.isfinite(out["auc"]).all()
    with pytest.raises(ObservableError) as info:
        graph.evaluate(T, {}, [])
    assert info.value.row == 0
    assert "auc" in info.value.message


def test_a_pk_observable_in_the_graph(
    pk_model: RoadrunnerSBMLModel, plan: Plan
) -> None:
    graph = compile_observables(
        [
            PK("p", "[C]", dose="PODOSE", route="oral"),
            Formula("ratio", "p.auc_inf_obs / p.cmax"),
        ],
        pk_model,
        plans=[plan],
    )
    t = np.linspace(0.0, 48.0, 97)[None, :]
    c = 12.5 * (np.exp(-0.2 * t) - np.exp(-t))
    out = graph.evaluate(t, {"[C]": c}, [plan])
    assert out["ratio"][0] == pytest.approx(out["p.auc_inf_obs"][0] / out["p.cmax"][0])


def test_the_graph_pickles(pk_model: RoadrunnerSBMLModel, plan: Plan) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            PK("p", "[C]", dose="PODOSE", route="oral"),
            Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
        ],
        pk_model,
        plans=[plan],
    )
    again = pickle.loads(pickle.dumps(graph))
    assert again.keep == graph.keep and again.units == graph.units


def test_a_model_without_units_is_dimensionless() -> None:
    model = Simulator().load(sbml())
    graph = compile_observables([Formula("a", "[A] + 1")], model)
    assert graph.units["a"] == "dimensionless"
