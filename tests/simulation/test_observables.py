"""The definitions of the observables."""

import json
import pickle

import pytest
from pkpdutils import NCAOptions

from sbmlsim import Q
from sbmlsim.simulation import PK, Custom, Formula, Observable, ObservableKind
from tests.simulator.models import auc_of_c


def test_a_formula_reads_its_symbols() -> None:
    formula = Formula("ins_rel", "ins / at(ins, 0) + max([glc])")
    assert isinstance(formula, Observable)
    assert formula.reads == ("[glc]", "ins")


@pytest.mark.parametrize("formula", ["", "max(", "at(x)", "x +"])
def test_an_invalid_formula_raises(formula: str) -> None:
    with pytest.raises(ValueError):
        Formula("f", formula)


@pytest.mark.parametrize(
    "name", ["", "1a", "a.b", "[X]", "time", "status", "_point", "statistic"]
)
def test_an_invalid_id_raises(name: str) -> None:
    with pytest.raises(ValueError):
        Formula(name, "1")


def test_a_unit_pint_does_not_know_raises() -> None:
    with pytest.raises(ValueError, match="unit"):
        Formula("f", "1", unit="furlongs_per_blob")


def test_a_pk_observable() -> None:
    pk = PK(
        "hctz",
        "[Cve_hctz]",
        dose="PODOSE_hctz",
        route="oral",
        parameters=["cmax", "auc_inf_obs"],  # ty: ignore[invalid-argument-type]
    )
    assert pk.parameters == ("cmax", "auc_inf_obs")
    assert pk.reads == ("[Cve_hctz]",)


def test_a_dose_needs_its_route() -> None:
    with pytest.raises(ValueError, match="route"):
        PK("p", "[C]", dose="PODOSE")


def test_a_fixed_dose_is_one_amount() -> None:
    with pytest.raises(ValueError, match="one amount"):
        PK("p", "[C]", dose=Q([1.0, 2.0], "mg"), route="oral")


def test_the_parameters_are_a_sequence_of_names() -> None:
    with pytest.raises(TypeError):
        PK("p", "[C]", parameters="cmax")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        PK("p", "[C]", parameters=["cmax", "cmax"])  # ty: ignore[invalid-argument-type]


def test_a_custom_observable() -> None:
    custom = Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"])  # ty: ignore[invalid-argument-type]
    assert custom.kind is ObservableKind.SCALAR
    assert custom.symbols == ("[C]",)
    assert custom.reads == ("[C]",)


def test_a_lambda_or_a_closure_is_refused() -> None:
    with pytest.raises(ValueError, match="module"):
        Custom("r", lambda t, v: 1.0, "dimensionless", symbols=[])  # ty: ignore[invalid-argument-type]

    def inner(t: object, v: object) -> float:
        return 1.0

    with pytest.raises(ValueError, match="module"):
        Custom("r", inner, "dimensionless", symbols=[])  # ty: ignore[invalid-argument-type]


def test_the_definitions_pickle_and_serialize() -> None:
    observables: list[Observable] = [
        Formula("f", "max([C])", unit="mg/l"),
        PK("p", "[C]", dose=Q(10.0, "mg"), route="oral", options=NCAOptions()),
        PK("q", "[C]", dose="PODOSE", route="iv_bolus", parameters=["cmax"]),  # ty: ignore[invalid-argument-type]
        Custom("c", auc_of_c, "mg*hr/l", symbols=["[C]"]),  # ty: ignore[invalid-argument-type]
    ]
    for observable in observables:
        again = pickle.loads(pickle.dumps(observable))
        assert again.to_dict() == observable.to_dict()
    data = json.loads(json.dumps([o.to_dict() for o in observables]))
    assert data[0] == {
        "type": "Formula",
        "id": "f",
        "formula": "max([C])",
        "unit": "mg/l",
    }
    assert data[1]["dose"] == {"value": 10.0, "unit": "milligram"}
    assert data[3]["function"] == "tests.simulator.models:auc_of_c"
