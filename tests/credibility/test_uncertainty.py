"""The parameter uncertainty of the decisive predictions (`sbmlsim.credibility.uncertainty`).

The draws are tested on records built by hand and the propagation on the
adopted fit of the session fixture `credibility_project`.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.credibility import (
    FitPropagation,
    PropagationSettings,
    evaluate_fits,
    propagate_fits,
)
from sbmlsim.credibility.uncertainty import (
    parameter_draws,
    rule_probability,
    sensitivity_indices,
    uncertainty_record_from_dict,
)
from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.objects import ParameterScaleType
from sbmlsim.fit.records import ParameterUncertainty, UncertaintyMethod
from sbmlsim.project import Project

DATA = Path(__file__).parents[1] / "data" / "credibility"
SETTINGS = PropagationSettings(draws=256, seed=1)


def _parameter(pid: str, value: float) -> ParameterUncertainty:
    """Get a parameter on a log scale with a profile interval of 2-fold."""
    return ParameterUncertainty(
        pid=pid,
        value=value,
        unit="1/min",
        lower_bound=value / 100,
        upper_bound=value * 100,
        scale=ParameterScaleType.LOG10,
        alpha=0.95,
        z=1.96,
        method=UncertaintyMethod.PROFILE,
        ci_lower=value / 2,
        ci_upper=value * 2,
        fit_id="PK_20261001_120000__abcde",
        cost=1.0,
        record="PK_20261001_120000__abcde",
    )


def _fisher(correlation: float, matrix: np.ndarray | None = None) -> FisherInformation:
    covariance = np.array([[1.0, correlation], [correlation, 1.0]]) * 0.01
    return FisherInformation(
        opid="PK",
        sid="best",
        pids=["a", "b"],
        values=np.array([1.0, 2.0]),
        scale=ParameterScaleType.LOG10,
        matrix=np.linalg.inv(covariance) if matrix is None else matrix,
        cost=1.0,
        n=100,
        scales=[ParameterScaleType.LOG10, ParameterScaleType.LOG10],
    )


def test_the_draws_have_the_fisher_correlation() -> None:
    parameters = [_parameter("a", 1.0), _parameter("b", 2.0)]
    settings = PropagationSettings(draws=1024, seed=1)
    draws = parameter_draws(
        "PK_20261001_120000__abcde", parameters, _fisher(0.8), settings
    )
    assert draws.correlated
    assert draws.subset == "PK"
    assert draws.pids == ("a", "b")
    assert draws.values.shape == (1024, 2)
    logs = np.log(draws.values)
    assert np.corrcoef(logs.T)[0, 1] == pytest.approx(0.8, abs=0.05)
    # the marginals are those of the profile, a median at the value
    assert np.median(draws.values, axis=0) == pytest.approx([1.0, 2.0], rel=0.05)
    assert np.std(logs[:, 0]) == pytest.approx(math.log(2) / 1.96, rel=0.1)
    # the same seed, the same draws
    again = parameter_draws(
        "PK_20261001_120000__abcde", parameters, _fisher(0.8), settings
    )
    assert np.array_equal(again.values, draws.values)


def test_independent_draws() -> None:
    parameters = [_parameter("a", 1.0), _parameter("b", 2.0)]
    singular = _fisher(0.0, matrix=np.array([[1.0, 1.0], [1.0, 1.0]]))
    draws = parameter_draws("PK_20261001_120000__abcde", parameters, singular, SETTINGS)
    assert not draws.correlated
    assert "rank 1 of 2" in draws.note
    none = parameter_draws("PK_20261001_120000__abcde", parameters, None, SETTINGS)
    assert "no Fisher information" in none.note
    other = parameter_draws(
        "PK_20261001_120000__abcde", [_parameter("a", 1.0)], _fisher(0.5), SETTINGS
    )
    assert "other parameters" in other.note


def test_the_settings() -> None:
    with pytest.raises(ValueError, match="fewer than two draws"):
        PropagationSettings(draws=1)
    with pytest.raises(ValueError, match="not increasing"):
        PropagationSettings(quantiles=(0.9, 0.1))
    with pytest.raises(ValueError, match="threshold"):
        PropagationSettings(threshold=1.0)
    with pytest.raises(ValueError, match="level"):
        PropagationSettings(level=0.0)


def test_the_sensitivity_indices() -> None:
    parameters = [_parameter("a", 1.0), _parameter("b", 2.0)]
    settings = PropagationSettings(draws=512, seed=2)
    draws = parameter_draws("PK_20261001_120000__abcde", parameters, None, settings)
    y = np.log(draws.values[:, 0])
    y[:3] = np.nan
    a, b = sensitivity_indices("Cve", "quantity", y, {"PK": draws})
    assert (a["parameter"], b["parameter"]) == ("a", "b")
    assert (a["index"] or 0.0) > 0.8
    assert (b["index"] or 0.0) < 0.05
    (constant, _) = sensitivity_indices("C", "quantity", np.ones(512), {"PK": draws})
    assert constant["index"] is None


def test_a_rule_probability() -> None:
    accepted = rule_probability("GMFE", "<= 1.5", [True] * 256, SETTINGS, 1)
    assert (accepted["estimate"], accepted["decision"]) == (1.0, "accepted")
    rejected = rule_probability("GMFE", "<= 1.5", [True, False] * 128, SETTINGS, 1)
    assert rejected["decision"] == "rejected"
    failed = rule_probability("GMFE", "<= 1.5", [True] * 31 + [None], SETTINGS, 1)
    assert (failed["n"], failed["n_failed"]) == (31, 1)
    nothing = rule_probability("GMFE", "<= 1.5", [None] * 4, SETTINGS, 1)
    assert nothing["decision"] == "not_evaluable"
    assert nothing["estimate"] is None


def test_the_propagation_of_the_adopted_fit(credibility_project: Project) -> None:
    fits = evaluate_fits(credibility_project)
    settings = PropagationSettings(draws=8, seed=3)
    propagation = propagate_fits(credibility_project, fits, settings)
    assert isinstance(propagation, FitPropagation)
    assert propagation.failed == {"DECAY": 0}
    (draws,) = propagation.draws.values()
    assert draws.pids == ("k",)
    assert draws.note == "independent draws, the fit has no Fisher information"
    predictions = propagation.predictions
    assert list(predictions.index) == list(fits.table.index)
    assert list(predictions.columns) == list(range(8))
    values = predictions.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    # the draws spread the predictions around the adopted ones
    ipred = fits.table["IPRED"].to_numpy(dtype=float)
    later = ipred < 0.99
    assert (values[later].std(axis=1) > 0).all()


def test_the_record_of_pkpdsim() -> None:
    """The uncertainty record of pkpdsim 0.19 is read, a record of 0.18 has none."""
    data = json.loads((DATA / "context-0.19-uncertainty.json").read_text("utf-8"))
    record = uncertainty_record_from_dict(data["uncertainty"])
    assert record is not None
    assert record == data["uncertainty"]
    assert [f["fit"] for f in record["fits"]] == ["DECAY"]
    without = {k: v for k, v in data["uncertainty"].items() if k != "sensitivity"}
    read = uncertainty_record_from_dict(without)
    assert read is not None
    assert read["sensitivity"] == []
    old = json.loads((DATA / "context-0.18.json").read_text("utf-8"))
    assert uncertainty_record_from_dict(old.get("uncertainty")) is None
