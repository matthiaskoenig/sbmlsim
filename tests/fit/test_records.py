"""The records of the adopted fits: adopting a fit, its parameters and their uncertainty.

The fit is the one of the toy project, run by the `fit` kind with its
profiles, its Fisher information and its report, and adopted into its out.
"""

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from sbmlutils.factory import Parameter, create_model

from sbmlsim import run
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.records import (
    RECORD_FILES,
    AdoptedRecord,
    AdoptedRecords,
    ParameterUncertainty,
    UncertaintyMethod,
    adopt_fit,
    convert_value,
)
from sbmlsim.report.render import compile_pdf
from sbmlsim.simulation.sampling import (
    UNCERTAINTY_DIMENSION,
    Truncated,
    Uniform,
    distribution_from_formula,
    parameter_distributions,
    uncertainty_dimension,
)
from tests.project_toy import PROJECT
from tests.project_toy.factory import U, toy_model

#: the fit of the toy project with what an adopted fit needs
TOY_FIT = {
    "project": "tests.project_toy:PROJECT",
    "subset": "DECAY",
    "repeats": 2,
    "seed": 1,
    "identifiability": True,
    "profile": {"max_points": 5},
    "report": True,
    "mapping_figures": False,
    "adopt": True,
}


@pytest.fixture(scope="module")
def toy_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of the adopted fit of the toy project."""
    out = tmp_path_factory.mktemp("toy") / "out"
    run.execute("fit", TOY_FIT, out, workers=1)
    return out


@pytest.fixture(scope="module")
def toy_fit(toy_out: Path) -> Path:
    """Get the records of the fit, with its report."""
    (fit_id,) = json.loads((toy_out / "adopted.json").read_text(encoding="utf-8"))[
        "fits"
    ].values()
    return toy_out / fit_id


@pytest.fixture(scope="module")
def fits_dir(toy_out: Path) -> Path:
    """Get the directory of the adopted fit."""
    return toy_out / "fits"


@pytest.fixture(scope="module")
def toy_with_uncertainty(
    toy_fit: Path, fits_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Create the toy model whose rate is the value of the fit with its uncertainty."""
    records = AdoptedRecords(fits_dir, [toy_fit.name])
    model = toy_model(0.2)
    model.parameters = [
        Parameter("k", **records.parameter("k", unit=U.per_min), name="rate")
    ]
    path = tmp_path_factory.mktemp("model") / "toy.xml"
    create_model(
        model=model,
        filepath=path,
        sbml_level=3,
        sbml_version=2,
        validate=False,
        create_markdown=False,
        create_antimony=False,
    )
    return path


def uncertainty(**changes: object) -> ParameterUncertainty:
    """Get the uncertainty of a parameter with a closed profile in log10."""
    fields: dict[str, object] = {
        "pid": "k",
        "value": 1.0,
        "unit": "1/min",
        "lower_bound": 0.01,
        "upper_bound": 10.0,
        "scale": ParameterScaleType.LOG10,
        "alpha": 0.95,
        "z": 1.959963984540054,
        "method": UncertaintyMethod.PROFILE,
        "ci_lower": 0.5,
        "ci_upper": 2.0,
        "fit_id": "FIT_1",
        "cost": 1.0,
        "record": "FIT_1/",
    }
    fields.update(changes)
    return ParameterUncertainty(**fields)  # ty: ignore[invalid-argument-type]


# ----------------------------------------------------------------------------
# adopting a fit
# ----------------------------------------------------------------------------
def test_the_fit_kind_adopts_the_record(toy_fit: Path, fits_dir: Path) -> None:
    record = fits_dir / toy_fit.name
    names = {p.name for p in record.iterdir()}
    assert {"parameters.json", "identifiability.json", "fisher.json"} <= names
    assert names <= set(RECORD_FILES) | {"plots"}


def test_adopt_copies_the_record(toy_fit: Path, tmp_path: Path) -> None:
    record = adopt_fit(toy_fit, tmp_path)
    assert record == tmp_path / toy_fit.name
    names = {p.name for p in record.iterdir()}
    assert {"parameters.json", "identifiability.json", "fisher.json"} <= names
    assert names <= set(RECORD_FILES) | {"plots"}


REPORT_FILES = ["report.pdf", "report.typ", "references.bib", "references.csl"]


def test_adopt_copies_the_report(toy_fit: Path, tmp_path: Path) -> None:
    record = adopt_fit(toy_fit, tmp_path)
    for name in REPORT_FILES:
        assert (record / name).exists(), name
    # the LaTeX sources stay in the results of the fit
    for name in ("report.tex", "report_standalone.tex"):
        assert (toy_fit / name).exists(), name
        assert not (record / name).exists(), name
    referenced = set(
        re.findall(
            r"plots/[\w.\-]+\.(?:png|svg|pdf)",
            (record / "report.typ").read_text(encoding="utf-8"),
        )
    )
    assert referenced
    kept = {p.relative_to(record).as_posix() for p in (record / "plots").iterdir()}
    # the figures the Typst source shows, nothing else of the plots, no PNG
    # copies of the LaTeX source
    assert kept == referenced
    assert list((toy_fit / "plots").glob("*.png"))
    assert not list((record / "plots").glob("*.png"))


def test_the_report_compiles_in_the_record(toy_fit: Path, tmp_path: Path) -> None:
    record = adopt_fit(toy_fit, tmp_path)
    pdf = compile_pdf(record / "report.typ", tmp_path / "recompiled.pdf")
    assert pdf.stat().st_size > 0


def test_adopt_skips_a_missing_report(toy_fit: Path, tmp_path: Path) -> None:
    fit = tmp_path / "old" / toy_fit.name
    shutil.copytree(toy_fit, fit)
    for name in REPORT_FILES:
        (fit / name).unlink(missing_ok=True)
    record = adopt_fit(fit, tmp_path / "fits")
    assert (record / "parameters.json").exists()
    assert not (record / "report.typ").exists()
    assert not (record / "plots").exists()


def test_adopt_refuses_an_adopted_fit_unless_forced(
    toy_fit: Path, tmp_path: Path
) -> None:
    adopt_fit(toy_fit, tmp_path)
    with pytest.raises(FileExistsError, match="adopted already"):
        adopt_fit(toy_fit, tmp_path)
    assert adopt_fit(toy_fit, tmp_path, force=True).exists()


def test_adopt_refuses_a_fit_without_uncertainty(toy_fit: Path, tmp_path: Path) -> None:
    fit = tmp_path / "FIT"
    fit.mkdir()
    for name in ("parameters.json", "optimization_result.json"):
        shutil.copy2(toy_fit / name, fit / name)
    with pytest.raises(FileNotFoundError, match="neither a profile"):
        adopt_fit(fit, tmp_path / "fits")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adopt_fit(tmp_path / "missing", tmp_path / "fits")
    (fit / "parameters.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"parameters\.json"):
        adopt_fit(fit, tmp_path / "fits")


# ----------------------------------------------------------------------------
# the record and the distribution of a parameter
# ----------------------------------------------------------------------------
def test_the_record_of_the_toy_fit(fits_dir: Path, toy_fit: Path) -> None:
    record = AdoptedRecord.load(fits_dir / toy_fit.name)
    assert record.fit_id == toy_fit.name
    k = record.parameters["k"]
    best = json.loads((toy_fit / "parameters.json").read_text(encoding="utf-8"))[
        "sets"
    ][0]
    assert k.value == best["values"]["k"]
    assert k.method == UncertaintyMethod.PROFILE
    assert (k.lower_bound, k.upper_bound) == (0.01, 1.0)
    assert k.standard_error is not None
    assert k.fisher_ci is not None
    assert k.n_data == 5
    assert k.identifiability is not None
    # the data of the toy model do not close the profile within its bounds
    expected = "uniform(" if k.sigma() is None else "lognormal("
    assert k.distribution().startswith(expected)
    distribution_from_formula(k.distribution())


def test_a_record_without_profiles_falls_back_on_the_fisher_information(
    fits_dir: Path, toy_fit: Path, tmp_path: Path
) -> None:
    record_dir = tmp_path / toy_fit.name
    shutil.copytree(fits_dir / toy_fit.name, record_dir)
    (record_dir / "identifiability.json").unlink()
    k = AdoptedRecord.load(record_dir).parameters["k"]
    assert k.method == UncertaintyMethod.FISHER
    assert k.identifiability is None
    assert (k.ci_lower, k.ci_upper) == k.fisher_ci
    assert k.sigma() == k.standard_error
    assert k.distribution().startswith("lognormal(")
    (record_dir / "fisher.json").unlink()
    with pytest.raises(FileNotFoundError, match="neither a profile"):
        AdoptedRecord.load(record_dir)


def test_the_distribution_of_a_closed_profile() -> None:
    p = uncertainty()
    sigma = math.log10(2.0) / p.z
    assert p.sigma() == pytest.approx(sigma)
    formula = p.distribution()
    assert formula == (f"lognormal(0.0, {sigma * math.log(10)!r}, 0.01, 10.0)")
    # the 95 % interval of the draws is the profile interval
    d = distribution_from_formula(formula)
    lower, upper = d.ppf(np.array([0.025, 0.975]))
    assert (lower, upper) == pytest.approx((0.5, 2.0), rel=1e-6)


def test_an_asymmetric_profile_takes_the_mean_of_its_half_widths() -> None:
    p = uncertainty(ci_lower=0.1, ci_upper=2.0)
    assert p.sigma() == pytest.approx((1.0 + math.log10(2.0)) / 2 / p.z)


def test_an_open_side_is_truncated_at_the_bound() -> None:
    p = uncertainty(value=3.0, upper_bound=3.0, ci_lower=2.6, ci_upper=None)
    assert p.sigma() == pytest.approx(math.log10(3.0 / 2.6) / p.z)
    assert p.interval() == (2.6, 3.0)
    d = distribution_from_formula(p.distribution())
    assert isinstance(d, Truncated)
    sigma = p.sigma()
    assert sigma is not None
    draws = np.asarray(d.ppf(np.linspace(0.01, 0.99, 50)), dtype=float)
    assert np.all(draws <= 3.0)
    # the upper half of the lognormal is cut off: the median of the draws is
    # its lower quartile
    assert float(np.median(draws)) == pytest.approx(
        3.0 * 10 ** (-0.6745 * sigma), rel=1e-2
    )


def test_a_profile_open_on_both_sides_is_uniform_in_the_bounds() -> None:
    p = uncertainty(ci_lower=None, ci_upper=None, identifiability="non_identifiable")
    assert p.sigma() is None
    assert p.distribution() == "uniform(0.01, 10.0)"
    assert isinstance(distribution_from_formula(p.distribution()), Uniform)


def test_a_linear_scale_gives_a_truncated_normal() -> None:
    p = uncertainty(scale=ParameterScaleType.LINEAR, ci_lower=0.8, ci_upper=1.2)
    assert p.distribution().startswith("normal(1.0, ")
    assert p.sigma() == pytest.approx(0.2 / p.z)
    d = distribution_from_formula(p.distribution())
    assert isinstance(d, Truncated)


def test_the_fisher_fallback_takes_the_standard_error() -> None:
    p = uncertainty(method=UncertaintyMethod.FISHER, standard_error=0.1)
    assert p.sigma() == 0.1
    assert p.distribution() == (f"lognormal(0.0, {0.1 * math.log(10)!r}, 0.01, 10.0)")
    assert p.standard_error_model() == pytest.approx(math.log(10) * 0.1)


def test_the_notes_have_the_provenance() -> None:
    notes = uncertainty(ci_upper=None).notes()
    for text in ["FIT_1", "profile likelihood", "[0.5, 10]", "open sides", "lognormal"]:
        assert text in notes


def test_a_value_is_converted_into_another_unit() -> None:
    assert convert_value(12.0, "1/hr", "1/min") == pytest.approx(0.2)
    assert convert_value(2.0, None, "1/min") == 2.0
    assert convert_value(2.0, "1/min", None) == 2.0
    assert convert_value(2.0, "1/min", "1/min") == 2.0


# ----------------------------------------------------------------------------
# the records of a project
# ----------------------------------------------------------------------------
def test_the_records_give_the_arguments_of_a_parameter(
    fits_dir: Path, toy_fit: Path
) -> None:
    records = AdoptedRecords(fits_dir, [toy_fit.name])
    kwargs = records.parameter("k", unit=U.per_min)
    assert set(kwargs) == {"value", "uncertainties", "unit"}
    assert kwargs["value"] == records["k"].value
    assert set(records.parameter("k")) == {"value", "uncertainties"}
    with pytest.raises(KeyError, match="No adopted fit has the parameter 'x'"):
        records.parameter("x")


def test_a_parameter_of_two_fits_raises(fits_dir: Path, toy_fit: Path) -> None:
    shutil.copytree(fits_dir / toy_fit.name, fits_dir / "COPY", dirs_exist_ok=True)
    try:
        with pytest.raises(ValueError, match="one adopted fit"):
            AdoptedRecords(fits_dir, [toy_fit.name, "COPY"])
    finally:
        shutil.rmtree(fits_dir / "COPY")


def test_a_fit_which_is_not_adopted_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fits/FIT_X"):
        AdoptedRecords(tmp_path, ["FIT_X"])


def test_the_records_of_a_project(toy_out: Path, toy_fit: Path) -> None:
    project = PROJECT.replace(
        fits_dir=toy_out / "fits", adopted=toy_out / "adopted.json"
    )
    records = project.fit_records()
    assert list(records.records) == [toy_fit.name]
    assert records["k"].fit_id == toy_fit.name
    # a project without adopted fits has no records
    assert PROJECT.replace(fits_dir=None).fit_records().parameters == {}
    with pytest.raises(ValueError, match="no directory of fit records"):
        PROJECT.replace(fits_dir=None, adopted=toy_out / "adopted.json").fit_records()


def test_the_model_has_the_uncertainty(toy_with_uncertainty: Path) -> None:
    text = toy_with_uncertainty.read_text(encoding="utf-8")
    assert "distrib:uncertainty" in text
    assert 'distrib:type="confidenceInterval"' in text
    assert 'distrib:type="standardError"' in text
    distributions = parameter_distributions(toy_with_uncertainty)
    assert set(distributions) == {"k"}
    # the profile of the toy data is open: uniform in the bounds of the fit
    assert isinstance(distributions["k"], Truncated | Uniform)


def test_the_draws_of_the_model(toy_with_uncertainty: Path) -> None:
    dimension = uncertainty_dimension(toy_with_uncertainty, 8, seed=1)
    assert dimension.id == UNCERTAINTY_DIMENSION
    values = np.asarray(dimension.values["k"], dtype=float)
    assert values.shape == (8,)
    assert np.all((values >= 0.01) & (values <= 1.0))
    again = uncertainty_dimension(toy_with_uncertainty, 8, seed=1)
    assert np.array_equal(values, np.asarray(again.values["k"], dtype=float))


def test_a_model_without_distributions_cannot_be_drawn() -> None:
    with pytest.raises(ValueError, match="No parameter"):
        uncertainty_dimension(PROJECT.model_path, 3, seed=1)
