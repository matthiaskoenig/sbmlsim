"""Test the training, validation and outlier data of a fit."""

import numpy as np
import pytest

from sbmlsim.fit import FitMappingCollection, FitSettings, MappingKind
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.helpers import mapping_kinds_info
from sbmlsim.fit.metrics import FitMetrics
from sbmlsim.fit.objects import EVALUATED_KINDS, UNUSED_KINDS, FitMapping
from sbmlsim.fit.optimization import OptimizationProblem


def test_collection_default_kind() -> None:
    """The data of a fit experiment is training data by default."""
    from examples.hctz_fitting.experiments.studies import Beermann1976

    collection = FitMappingCollection(experiment=Beermann1976, mappings=["a"])
    assert collection.kind is MappingKind.TRAINING


def test_collection_kind() -> None:
    """The kind classifies the selected data of a fit experiment."""
    from examples.hctz_fitting.experiments.studies import Beermann1976

    collection = FitMappingCollection(
        experiment=Beermann1976, mappings=["a"], kind=MappingKind.VALIDATION
    )
    assert collection.kind is MappingKind.VALIDATION
    assert "validation" in str(collection)


def test_reduce_keeps_the_kinds_apart() -> None:
    """The training and the validation data of an experiment are not combined."""
    from examples.hctz_fitting.experiments.studies import Beermann1976

    reduced = FitMappingCollection.reduce(
        [
            FitMappingCollection(experiment=Beermann1976, mappings=["a"]),
            FitMappingCollection(experiment=Beermann1976, mappings=["b"]),
            FitMappingCollection(
                experiment=Beermann1976,
                mappings=["c"],
                kind=MappingKind.VALIDATION,
            ),
        ]
    )
    assert len(reduced) == 2
    assert reduced[0].mappings == ["a", "b"]
    assert reduced[0].kind is MappingKind.TRAINING
    assert reduced[1].mappings == ["c"]
    assert reduced[1].kind is MappingKind.VALIDATION


def test_metadata_describes_the_curve() -> None:
    """The metadata of a mapping does not decide what a fit does with it."""
    from sbmlsim.fit import MappingMetaData

    assert not hasattr(MappingMetaData(), "kind")
    assert not hasattr(FitMapping.__new__(FitMapping), "kind")


def test_kinds_of_the_problem(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The problem knows which of its mappings are fitted."""
    op = op_hctz_pk
    op.initialize(fit_settings)

    counts = op.mapping_counts()
    assert counts[MappingKind.TRAINING] > 0
    assert counts[MappingKind.VALIDATION] > 0
    # the data a fit does not use is not part of the problem at all
    for kind in UNUSED_KINDS:
        assert kind not in counts
        assert all(mapping_kind is not kind for mapping_kind in op.mapping_kinds)

    assert len(op.mapping_kinds) == len(op.mapping_keys)
    assert len(op.training_indices) == counts[MappingKind.TRAINING]
    assert len(op.validation_indices) == counts[MappingKind.VALIDATION]
    assert set(op.indices()) == set(op.training_indices) | set(op.validation_indices)


def test_optimization_uses_only_training(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The residuals of the optimizer only cover the training data."""
    op = op_hctz_pk
    op.initialize(fit_settings)
    xlog = np.log10(op.xmodel)

    residuals = op.residuals(xlog)
    n_training = sum(len(op.y_references[k]) for k in op.training_indices)
    assert len(residuals) == n_training

    # the complete data covers the validation mappings as well
    complete = op.residuals(xlog, complete_data=True)
    assert len(complete["y_obsip"]) == len(op.mapping_keys)


def test_problem_without_training_data(
    definition_hctz_pkiv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A problem needs at least one fit experiment which is fitted."""
    validation_only = [
        FitMappingCollection(
            experiment=collection.experiment_class,
            mappings=list(collection.mappings),
            use_mapping_weights=True,
            kind=MappingKind.VALIDATION,
        )
        for collection in definition_hctz_pkiv.collections()
    ]
    problem = definition_hctz_pkiv.problem(
        opid="validation_only", mapping_collections=validation_only
    )
    with pytest.raises(ValueError, match="no training data"):
        problem.initialize(fit_settings)


def test_metrics_per_kind(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The metrics are calculated for the training and the validation data."""
    op = op_hctz_pk
    op.initialize(fit_settings)
    metrics = FitMetrics(problem=op, parameter_set=op.parameter_set_model())

    df = metrics.summary_df()
    assert list(df.kind) == ["training", "validation", "all"]
    assert df.n.iloc[2] == df.n.iloc[0] + df.n.iloc[1]

    # the cost is the objective of the optimization, i.e. the training data
    assert np.isfinite(df.cost.iloc[0])
    assert np.isnan(df.cost.iloc[1])
    assert np.isnan(df.cost.iloc[2])

    # the kind of every mapping and every data point is reported
    assert set(metrics.mappings_df().kind) == {"training", "validation"}
    assert set(metrics.datapoints_df().kind) == {"training", "validation"}


def test_metrics_unknown_kind(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A kind without data points is reported."""
    op = op_hctz_pkiv
    op.initialize(fit_settings)
    metrics = FitMetrics(problem=op, parameter_set=op.parameter_set_model())
    with pytest.raises(ValueError, match="no data points of kind"):
        metrics.summary(kind=MappingKind.VALIDATION)


def test_metrics_only_training(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A problem with only training data has a single summary row."""
    op = op_hctz_pkiv
    op.initialize(fit_settings)
    metrics = FitMetrics(problem=op, parameter_set=op.parameter_set_model())

    df = metrics.summary_df()
    assert list(df.kind) == ["training"]
    assert np.isfinite(df.cost.iloc[0])


def test_mapping_kinds_info() -> None:
    """The overview of the data counts the mappings per kind."""
    import pandas as pd

    df = pd.DataFrame({"kind": ["training", "training", "validation", "outlier"]})
    info = mapping_kinds_info(df)
    assert "4 (2 training, 1 validation, 1 outlier)" in info

    # a table without the kind reports the number of mappings
    assert "2" in mapping_kinds_info(pd.DataFrame({"fm_key": ["a", "b"]}))


def test_excluded_is_not_an_outlier() -> None:
    """The two kinds a fit does not use say different things.

    An outlier is a decision about the data, i.e. the data is not usable. An
    exclusion is a decision about the model, i.e. the model does not describe
    what was measured. Both are unused, and a fit which drops data for the two
    reasons should say which is which.
    """
    assert MappingKind.OUTLIER is not MappingKind.EXCLUDED
    assert set(UNUSED_KINDS) == {MappingKind.OUTLIER, MappingKind.EXCLUDED}
    assert not set(UNUSED_KINDS) & set(EVALUATED_KINDS)
    assert set(EVALUATED_KINDS) | set(UNUSED_KINDS) == set(MappingKind)
