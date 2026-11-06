"""The filters of the fit mappings and the selection of the data of a project."""

import copy
import dataclasses
import pickle
from enum import StrEnum
from typing import Any, override

import pytest

from sbmlsim.experiment import ExperimentRunner
from sbmlsim.fit import FitMapping, FitMappingCollection, MappingKind
from sbmlsim.fit.filters import (
    KeyFilter,
    filter_keys,
    metadata_filter,
    named,
    named_filter,
    observable_filter,
    observable_id,
    study,
)
from sbmlsim.fit.helpers import select_mappings
from sbmlsim.fit.objects import MappingMetaData
from tests.project_toy import PROJECT, TOY_PATH
from tests.project_toy.experiments import ToyDecay


class Route(StrEnum):
    IV = "iv"
    PO = "po"
    SC = "sc"


class Dosing(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"


class Tissue(StrEnum):
    PLASMA = "plasma"
    URINE = "urine"


class Site(StrEnum):
    VEIN = "vein"
    NOT_APPLICABLE = "not applicable"


@dataclasses.dataclass(kw_only=True)
class StudyMetaData(MappingMetaData):
    """A generic metadata of the curves of a study."""

    route: Route
    dosing: Dosing = Dosing.SINGLE
    tissue: Tissue = Tissue.PLASMA
    site: Site | None = None


@dataclasses.dataclass(kw_only=True)
class SiteMetaData(StudyMetaData):
    """A metadata whose site follows the tissue when it is not stated."""

    @override
    def filter_value(self, name: str) -> Any:
        if name == "site" and self.site is None:
            return Site.VEIN if self.tissue is Tissue.PLASMA else Site.NOT_APPLICABLE
        return super().filter_value(name)


@pytest.fixture(scope="module")
def fit_mapping() -> FitMapping:
    """Get the fit mapping of the toy decay with a metadata of the study."""
    runner = ExperimentRunner(
        experiment_classes=[ToyDecay], base_path=TOY_PATH, data_path=TOY_PATH
    )
    mapping = copy.copy(runner.experiments["ToyDecay"]._fit_mappings["fm_decay_A"])
    mapping.metadata = SiteMetaData(route=Route.IV)
    return mapping


def with_metadata(fit_mapping: FitMapping, **changes: object) -> FitMapping:
    """A copy of the mapping whose metadata differs in some fields."""
    other = copy.copy(fit_mapping)
    other.metadata = dataclasses.replace(fit_mapping.metadata, **changes)  # ty: ignore[invalid-argument-type]
    return other


def test_the_observable_and_the_study_of_a_mapping(fit_mapping: FitMapping) -> None:
    assert observable_id(fit_mapping) == "A"
    assert study(fit_mapping) == "ToyDecay"


def test_the_filter_value_is_the_field() -> None:
    assert StudyMetaData(route=Route.PO).filter_value("route") is Route.PO


def test_a_mapping_with_an_allowed_value_passes(fit_mapping: FitMapping) -> None:
    control = metadata_filter(StudyMetaData, route={Route.IV, Route.PO})
    assert control("fm_decay_A", fit_mapping)
    assert not control("fm_decay_A", with_metadata(fit_mapping, route=Route.SC))


def test_every_clause_must_hold(fit_mapping: FitMapping) -> None:
    control = metadata_filter(StudyMetaData, route={Route.IV}, tissue={Tissue.PLASMA})
    assert control("fm_decay_A", fit_mapping)
    assert not control("fm_decay_A", with_metadata(fit_mapping, route=Route.PO))


def test_an_excluded_value_fails(fit_mapping: FitMapping) -> None:
    single = metadata_filter(StudyMetaData, excluded={"dosing": {Dosing.MULTIPLE}})
    assert single("fm_decay_A", fit_mapping)
    assert not single("fm_decay_A", with_metadata(fit_mapping, dosing=Dosing.MULTIPLE))


def test_an_unknown_field_fails_when_the_filter_is_built() -> None:
    with pytest.raises(TypeError, match=r"StudyMetaData has no field.*diplotype"):
        metadata_filter(StudyMetaData, diplotype={"*1/*1"})
    with pytest.raises(TypeError, match=r"has no field.*tisue"):
        metadata_filter(StudyMetaData, excluded={"tisue": {Tissue.URINE}})


def test_a_filter_needs_a_clause_and_values() -> None:
    with pytest.raises(ValueError, match="at least one clause"):
        metadata_filter(StudyMetaData)
    with pytest.raises(ValueError, match=r"selects nothing.*route"):
        metadata_filter(StudyMetaData, route=set())


def test_a_single_value_is_not_a_collection() -> None:
    with pytest.raises(TypeError, match=r"route.*collection"):
        metadata_filter(StudyMetaData, route=Route.IV)
    with pytest.raises(TypeError, match=r"dosing.*collection"):
        metadata_filter(StudyMetaData, excluded={"dosing": Dosing.SINGLE})
    with pytest.raises(TypeError, match=r"route.*collection"):
        metadata_filter(StudyMetaData, route=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"observables.*collection"):
        observable_filter("Cve_hctz")


def test_an_empty_exclusion_or_observable_set_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"excludes nothing.*dosing"):
        metadata_filter(StudyMetaData, excluded={"dosing": set()})
    with pytest.raises(ValueError, match=r"observables.*selects nothing"):
        observable_filter([])


def test_the_metadata_class_is_not_an_instance() -> None:
    with pytest.raises(TypeError, match="class, not an instance"):
        metadata_filter(StudyMetaData(route=Route.IV), route={Route.IV})  # ty: ignore[invalid-argument-type]


def test_the_metadata_class_must_be_a_dataclass() -> None:
    class OtherMetaData:
        route: Route

    with pytest.raises(TypeError, match="OtherMetaData is not a dataclass"):
        metadata_filter(OtherMetaData, route={Route.IV})  # ty: ignore[invalid-argument-type]
    # a class derived from the metadata without fields of its own has none
    with pytest.raises(TypeError, match=r"MappingMetaData has no fields \['route'\]"):
        metadata_filter(MappingMetaData, route={Route.IV})


def test_the_filter_checks_the_type_of_the_metadata(fit_mapping: FitMapping) -> None:
    other = dataclasses.make_dataclass(
        "Other", [("route", Route)], bases=(MappingMetaData,), kw_only=True
    )
    with pytest.raises(TypeError, match="Other required on"):
        metadata_filter(other, route={Route.IV})("fm_decay_A", fit_mapping)  # ty: ignore[invalid-argument-type]


def test_a_subclass_passes_the_filter_of_its_base(fit_mapping: FitMapping) -> None:
    assert metadata_filter(StudyMetaData, route={Route.IV})("k", fit_mapping)


def test_the_filter_says_what_it_selects() -> None:
    control = metadata_filter(
        StudyMetaData,
        route={Route.PO, Route.IV},
        excluded={"dosing": {Dosing.MULTIPLE}},
    )
    assert repr(control) == (
        "MetadataFilter(StudyMetaData: route in [IV, PO] and dosing not in [MULTIPLE])"
    )


def test_the_filters_pickle_and_compare() -> None:
    for f in (
        metadata_filter(StudyMetaData, route={Route.IV}),
        observable_filter({"A"}),
        named_filter({"ToyDecay": {"fm_decay_A"}}),
        filter_keys({"fm_decay_A"}),
    ):
        assert pickle.loads(pickle.dumps(f)) == f


def test_the_filter_compares_the_filter_value(fit_mapping: FitMapping) -> None:
    """A metadata class decides what a filter compares, e.g. a resolved site."""
    vein = metadata_filter(SiteMetaData, site={Site.VEIN})
    not_applicable = metadata_filter(SiteMetaData, site={Site.NOT_APPLICABLE})
    assert vein("k", fit_mapping)
    assert not not_applicable("k", fit_mapping)
    urine = with_metadata(fit_mapping, tissue=Tissue.URINE)
    assert not_applicable("k", urine)
    assert not vein("k", urine)
    # a stated site is the site
    stated = with_metadata(urine, site=Site.VEIN)
    assert vein("k", stated)


def test_the_observable_filter(fit_mapping: FitMapping) -> None:
    pk_filter = observable_filter({"A", "B"})
    assert pk_filter("fm_decay_A", fit_mapping)
    assert not observable_filter({"B"})("fm_decay_A", fit_mapping)
    assert repr(pk_filter) == "ObservableFilter(A, B)"


def test_named_selects_the_keys_of_their_study(fit_mapping: FitMapping) -> None:
    assert named({"ToyDecay": {"fm_decay_A"}}, "fm_decay_A", fit_mapping)
    assert not named({"Other": {"fm_decay_A"}}, "fm_decay_A", fit_mapping)
    assert named_filter({"ToyDecay": {"fm_decay_A"}})("fm_decay_A", fit_mapping)
    assert not named_filter({"Other": {"fm_decay_A"}})("fm_decay_A", fit_mapping)


def test_named_filter_is_sorted() -> None:
    f = named_filter({"B": {"y", "x"}, "A": {"z"}})
    assert f.mappings == (("A", frozenset({"z"})), ("B", frozenset({"x", "y"})))


def test_the_key_filter_selects_a_key_in_every_study(fit_mapping: FitMapping) -> None:
    f = filter_keys(["fm_decay_A"])
    assert isinstance(f, KeyFilter)
    assert f("fm_decay_A", fit_mapping)
    assert not f("fm_other", fit_mapping)
    assert repr(f) == "KeyFilter(fm_decay_A)"


def training(collections: dict[str, list[FitMappingCollection]]) -> list[list[str]]:
    return [
        c.mappings for c in collections["ToyDecay"] if c.kind is MappingKind.TRAINING
    ]


def test_the_mappings_of_a_project_are_selected_by_the_filters() -> None:
    selected = select_mappings(PROJECT, [ToyDecay], [observable_filter({"A"})])
    assert training(selected) == [["fm_decay_A"]]
    excluded = select_mappings(PROJECT, [ToyDecay], [observable_filter({"B"})])
    assert training(excluded) == []


def test_outliers_and_validation_are_passed_on() -> None:
    outlier = select_mappings(PROJECT, [ToyDecay], outliers={"fm_decay_A"})
    assert training(outlier) == []
    validation = select_mappings(PROJECT, [ToyDecay], validation={"fm_decay_A"})
    assert [c.kind for c in validation["ToyDecay"]] == [MappingKind.VALIDATION]
    with pytest.raises(ValueError, match="no_mapping"):
        select_mappings(PROJECT, [ToyDecay], validation={"no_mapping"})
