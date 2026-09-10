"""Test the selection of the data of a fit."""

import pytest

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.studies import Beermann1976, Patel1984
from sbmlsim.fit import FitMapping, MappingKind
from sbmlsim.fit.helpers import (
    FitMappings,
    MappingSelection,
    mapping_kinds_info,
    select_mapping_collections,
)


@pytest.fixture(scope="module")
def fit_mappings() -> FitMappings:
    """The fit mappings of two studies, loaded once."""
    return FitMappings(
        experiment_classes=[Beermann1976, Patel1984],
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
    )


def is_urine(key: str, fit_mapping: FitMapping) -> bool:
    """Accept the urine data."""
    return "urine" in key


def is_iv(key: str, fit_mapping: FitMapping) -> bool:
    """Accept the iv data."""
    return "_iv" in key


def test_keys(fit_mappings: FitMappings) -> None:
    """The keys are the fit mappings of every experiment."""
    assert set(fit_mappings.keys) == {"Beermann1976", "Patel1984"}
    assert "fm_hctz_iv1_5_urine" in fit_mappings.keys["Beermann1976"]
    assert fit_mappings.keys["Patel1984"]


def test_no_filters_is_all_training(fit_mappings: FitMappings) -> None:
    """Without filters, outliers and validation every mapping is training data."""
    selection = fit_mappings.select(print_info=False)
    assert isinstance(selection, MappingSelection)
    assert set(selection.df["kind"]) == {MappingKind.TRAINING.value}
    assert len(selection.df) == sum(len(keys) for keys in fit_mappings.keys.values())

    collections = selection.collections
    assert set(collections) == {"Beermann1976", "Patel1984"}
    for sid, keys in fit_mappings.keys.items():
        (collection,) = collections[sid]
        assert collection.kind is MappingKind.TRAINING
        assert collection.mappings == keys
        assert collection.use_mapping_weights


def test_filters_exclude(fit_mappings: FitMappings) -> None:
    """A mapping which fails a filter is excluded, the rest is training data."""
    selection = fit_mappings.select(filters=[is_urine, is_iv], print_info=False)
    df = selection.df
    training = df[df["kind"] == MappingKind.TRAINING.value]
    excluded = df[df["kind"] == MappingKind.EXCLUDED.value]
    assert len(training) + len(excluded) == len(df)
    assert all("urine" in key and "_iv" in key for key in training["fm_key"])
    assert all("urine" not in key or "_iv" not in key for key in excluded["fm_key"])
    # a study whose data is excluded completely still has its collection
    assert set(selection.collections) == {"Beermann1976", "Patel1984"}
    (patel,) = selection.collections["Patel1984"]
    assert patel.kind is MappingKind.EXCLUDED


def test_single_filter(fit_mappings: FitMappings) -> None:
    """A single filter is accepted like a list of filters."""
    selection = fit_mappings.select(filters=is_urine, print_info=False)
    assert MappingKind.EXCLUDED.value in set(selection.df["kind"])


def test_outliers(fit_mappings: FitMappings) -> None:
    """The outliers are part of the training data which is not fitted."""
    selection = fit_mappings.select(outliers={"fm_hctz_iv1_5_urine"}, print_info=False)
    assert selection.kind("Beermann1976", "fm_hctz_iv1_5_urine") is MappingKind.OUTLIER
    kinds = selection.kinds_of("Beermann1976")
    assert list(kinds.values()).count(MappingKind.OUTLIER) == 1
    (training, outlier) = selection.collections["Beermann1976"]
    assert training.kind is MappingKind.TRAINING
    assert outlier.kind is MappingKind.OUTLIER
    assert outlier.mappings == ["fm_hctz_iv1_5_urine"]


def test_outlier_of_excluded_data_stays_excluded(fit_mappings: FitMappings) -> None:
    """An outlier is a decision about the training data, not about excluded data."""
    selection = fit_mappings.select(
        filters=is_iv, outliers={"fm_hctz_iv1_5_urine"}, print_info=False
    )
    assert selection.kind("Beermann1976", "fm_hctz_iv1_5_urine") is MappingKind.OUTLIER

    selection = fit_mappings.select(
        filters=lambda key, fm: not is_iv(key, fm),
        outliers={"fm_hctz_iv1_5_urine"},
        print_info=False,
    )
    assert selection.kind("Beermann1976", "fm_hctz_iv1_5_urine") is MappingKind.EXCLUDED


def test_validation_keys(fit_mappings: FitMappings) -> None:
    """The validation data is named by its keys."""
    selection = fit_mappings.select(
        validation={"fm_hctz_iv1_5_urine"}, print_info=False
    )
    assert (
        selection.kind("Beermann1976", "fm_hctz_iv1_5_urine") is MappingKind.VALIDATION
    )
    assert selection.df["kind"].value_counts()[MappingKind.VALIDATION.value] == 1


def test_validation_filter(fit_mappings: FitMappings) -> None:
    """The validation data is selected by a filter."""
    selection = fit_mappings.select(validation=is_iv, print_info=False)
    df = selection.df
    validation = df[df["kind"] == MappingKind.VALIDATION.value]
    assert len(validation) > 0
    assert all("_iv" in key for key in validation["fm_key"])
    assert all(
        "_iv" not in key
        for key in df[df["kind"] == MappingKind.TRAINING.value]["fm_key"]
    )


def test_precedence(fit_mappings: FitMappings) -> None:
    """Excluded beats outlier, outlier beats validation, validation beats training."""
    key = "fm_hctz_iv1_5_urine"
    # outlier and validation: outlier
    selection = fit_mappings.select(outliers={key}, validation={key}, print_info=False)
    assert selection.kind("Beermann1976", key) is MappingKind.OUTLIER
    # excluded, outlier and validation: excluded
    selection = fit_mappings.select(
        filters=lambda k, fm: k != key,
        outliers={key},
        validation={key},
        print_info=False,
    )
    assert selection.kind("Beermann1976", key) is MappingKind.EXCLUDED


def test_unknown_keys_raise(fit_mappings: FitMappings) -> None:
    """A key which is no fit mapping of any experiment is a typo."""
    with pytest.raises(ValueError, match="fm_missing"):
        fit_mappings.select(outliers={"fm_missing"}, print_info=False)
    with pytest.raises(ValueError, match="fm_missing"):
        fit_mappings.select(validation={"fm_missing"}, print_info=False)


def test_every_mapping_once(fit_mappings: FitMappings) -> None:
    """Every fit mapping is in exactly one collection."""
    selection = fit_mappings.select(
        filters=is_urine,
        outliers={"fm_hctz_iv1_5_urine"},
        validation=is_iv,
        print_info=False,
    )
    for sid, keys in fit_mappings.keys.items():
        selected = [m for c in selection.collections[sid] for m in c.mappings]
        assert sorted(selected) == sorted(keys)
        # the collections are ordered by kind
        kinds = [c.kind for c in selection.collections[sid]]
        assert kinds == sorted(kinds, key=list(MappingKind).index)


def test_select_mapping_collections() -> None:
    """The function is the selection in one call."""
    collections = select_mapping_collections(
        experiment_classes=[Beermann1976],
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        filters=is_urine,
        validation={"fm_hctz_iv1_5_urine"},
        print_info=False,
    )
    kinds = {c.kind for c in collections["Beermann1976"]}
    assert MappingKind.VALIDATION in kinds
    assert MappingKind.TRAINING in kinds


def test_mapping_kinds_info(fit_mappings: FitMappings) -> None:
    """The overview counts the mappings per kind."""
    selection = fit_mappings.select(outliers={"fm_hctz_iv1_5_urine"}, print_info=False)
    info = mapping_kinds_info(selection.df)
    assert info.startswith("mappings")
    assert "1 outlier" in info
