"""Tests of the predictions and the noise models of a resolved problem."""

import dataclasses

import numpy as np
import pytest

from examples.hctz_fitting.experiments.studies import Beermann1976
from sbmlsim.fit import FitMapping, FitSettings
from sbmlsim.fit.objects import NoiseDistribution, NoiseModel
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ResidualType


@pytest.fixture
def settings_tight() -> FitSettings:
    """Get settings with which two simulations of a problem agree.

    With a variable step size the output grid of a simulation is the steps of
    the integrator, which are not the same in two runs, and the data is
    interpolated on it: two simulations then differ by `1e-6` however tight
    the tolerances are. With a fixed grid they agree to the tolerances.
    """
    return FitSettings(
        residual=ResidualType.ABSOLUTE,
        variable_step_size=False,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )


def test_a_fit_mapping_has_no_noise_model_by_default(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The noise model is optional, a fit does not need one."""
    op_hctz_iv.initialize(fit_settings)
    assert op_hctz_iv.noise_models == [None] * len(op_hctz_iv.mapping_keys)


def test_the_noise_model_of_a_fit_mapping_is_resolved(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The problem holds the noise model of every mapping it resolves."""
    noise = NoiseModel(formula="0.05", distribution=NoiseDistribution.LAPLACE)
    fit_mappings = Beermann1976.fit_mappings

    def with_noise(self: Beermann1976) -> dict[str, FitMapping]:
        mappings = fit_mappings(self)
        mappings["fm_hctz_iv1_5_urine"].noise = noise
        return mappings

    monkeypatch.setattr(Beermann1976, "fit_mappings", with_noise)
    op_hctz_iv.initialize(fit_settings)

    k = op_hctz_iv.mapping_keys.index("fm_hctz_iv1_5_urine")
    assert op_hctz_iv.noise_models[k] is noise
    others = [n for i, n in enumerate(op_hctz_iv.noise_models) if i != k]
    assert others == [None] * (len(op_hctz_iv.mapping_keys) - 1)


def test_the_noise_models_are_resolved_again(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A second initialization does not append to the first."""
    op_hctz_iv.initialize(fit_settings)
    op_hctz_iv.initialize(fit_settings, force=True)
    assert len(op_hctz_iv.noise_models) == len(op_hctz_iv.mapping_keys)


def test_predictions_are_the_simulation_at_the_data(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The predictions are what the residuals are calculated from."""
    op_hctz_pk.initialize(settings_tight)
    x = np.asarray(op_hctz_pk.x0, dtype=float)

    # all mappings, which is what the complete data of the residuals simulates
    indices = op_hctz_pk.indices()
    predictions = op_hctz_pk.predictions(x, indices=indices)
    assert sorted(predictions) == indices

    data = op_hctz_pk.residuals(op_hctz_pk.to_scale(x), complete_data=True)
    assert isinstance(data, dict)
    for k, prediction in predictions.items():
        assert prediction.shape == np.shape(op_hctz_pk.y_references[k])
        assert prediction == pytest.approx(
            np.asarray(data["y_obsip"][k]), rel=1e-8, abs=1e-12
        )


def test_predictions_of_the_training_data_by_default(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """Without indices the training data is simulated, as a fit does."""
    op_hctz_pk.initialize(settings_tight)
    x = np.asarray(op_hctz_pk.x0, dtype=float)

    predictions = op_hctz_pk.predictions(x)
    assert sorted(predictions) == op_hctz_pk.training_indices
    assert op_hctz_pk.validation_indices

    # the residuals of a fit without weights are `prediction - data`
    expected = np.concatenate(
        [
            predictions[k] - np.asarray(op_hctz_pk.y_references[k], dtype=float)
            for k in op_hctz_pk.training_indices
        ]
    )
    residuals = np.asarray(op_hctz_pk.residuals(op_hctz_pk.to_scale(x)), dtype=float)
    assert residuals == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_predictions_of_selected_mappings(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The mappings are selected by their indices."""
    op_hctz_pk.initialize(fit_settings)
    x = np.asarray(op_hctz_pk.x0, dtype=float)
    indices = op_hctz_pk.validation_indices
    assert indices
    assert sorted(op_hctz_pk.predictions(x, indices=indices)) == indices


def test_predictions_are_not_shifted_to_the_baseline(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The residual of the settings does not change what is simulated."""
    x = np.asarray(op_hctz_pk.x0, dtype=float)
    op_hctz_pk.initialize(settings_tight)
    absolute = op_hctz_pk.predictions(x)
    op_hctz_pk.initialize(
        dataclasses.replace(settings_tight, residual=ResidualType.ABSOLUTE_TO_BASELINE)
    )
    baseline = op_hctz_pk.predictions(x)

    assert sorted(absolute) == sorted(baseline)
    # a curve which does not start at zero is where a shift would show
    assert any(prediction[0] != 0.0 for prediction in absolute.values())
    for k, prediction in absolute.items():
        assert baseline[k] == pytest.approx(prediction, rel=1e-8, abs=1e-12)


def test_predictions_require_an_initialized_problem(
    op_hctz_iv: OptimizationProblem,
) -> None:
    """A problem which is not initialized has no simulator."""
    with pytest.raises(ValueError, match="initialized"):
        op_hctz_iv.predictions(np.asarray(op_hctz_iv.x0, dtype=float))


def test_predictions_of_a_failed_simulation_raise(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulation which failed has no prediction, and the error says so."""
    op_hctz_iv.initialize(fit_settings)
    simulator = op_hctz_iv.runner_initialized.simulator
    assert simulator is not None

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CVODE failed")

    monkeypatch.setattr(simulator, "_timecourses", fail)
    with pytest.raises(ValueError, match="failed"):
        op_hctz_iv.predictions(np.asarray(op_hctz_iv.x0, dtype=float))
