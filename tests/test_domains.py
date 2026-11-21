"""The domains of the quantities of a model, from distrib."""

import math
from pathlib import Path

import pytest

from sbmlsim.domains import Domain, Values, domains
from tests.simulator.models import FRACTION, sbml, with_domains


@pytest.fixture(scope="module")
def sbml_with_domains() -> str:
    return with_domains(
        sbml(FRACTION),
        {
            "f_renal": Domain(0.0, 1.0),
            "stage": Domain(0.0, 3.0, Values.INTEGER),
            "C": Domain(0.0, math.inf),
        },
    )


def test_the_domains_of_a_model_are_read_by_id(sbml_with_domains: str) -> None:
    assert domains(sbml_with_domains) == {
        "f_renal": Domain(0.0, 1.0),
        "stage": Domain(0.0, 3.0, Values.INTEGER),
        "C": Domain(0.0, math.inf),
    }


def test_a_model_without_domains_states_none() -> None:
    assert domains(sbml(FRACTION)) == {}


def test_the_domains_are_read_from_a_file(
    sbml_with_domains: str, tmp_path: Path
) -> None:
    path = tmp_path / "fraction.xml"
    path.write_text(sbml_with_domains, encoding="utf-8")
    assert domains(path)["f_renal"] == Domain(0.0, 1.0)


def test_a_path_which_is_no_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        domains(tmp_path / "missing.xml")


def test_an_sbml_which_is_no_model_raises() -> None:
    with pytest.raises(ValueError, match="no SBML model"):
        domains("<sbml xmlns='http://www.sbml.org/sbml/level3/version2/core'/>")


@pytest.mark.parametrize(
    ("domain", "value", "clipped"),
    [
        (Domain(0.0, 1.0), 1.081, 1.0),
        (Domain(0.0, 1.0), -0.2, 0.0),
        (Domain(0.0, 1.0), 0.5, 0.5),
        (Domain(0.0, math.inf), 1e6, 1e6),
        (Domain(0.0, math.inf), math.inf, math.inf),
        (Domain(0.0, 3.0, Values.INTEGER), 2.3, 2.0),
        (Domain(0.0, 3.0, Values.INTEGER), 3.4, 3.0),
        (Domain(0.5, 3.0, Values.INTEGER), 0.6, 1.0),
        (Domain(0, 1, Values.BOOLEAN), 1.15, 1.0),
    ],
)
def test_a_value_is_clipped_onto_the_domain(
    domain: Domain, value: float, clipped: float
) -> None:
    assert domain.clip(value) == clipped


def test_an_interval_is_clipped_on_each_side() -> None:
    assert Domain(0.0, 1.0).clip(-0.1, 1.2) == (0.0, 1.0)


def test_an_interval_whose_bounds_are_reversed_is_kept() -> None:
    assert Domain(0.0, 1.0).clip(1.2, -0.1) == (1.2, -0.1)


def test_a_value_which_is_no_number_cannot_be_clipped() -> None:
    with pytest.raises(ValueError, match="not a number"):
        Domain(0.0, 1.0).clip(math.nan)


@pytest.mark.parametrize(
    ("domain", "value", "contained"),
    [
        (Domain(0.0, 1.0), 1.0, True),
        (Domain(0.0, 1.0), 1.0001, False),
        (Domain(0.0, math.inf), math.inf, False),
        (Domain(0.0, 1.0), math.nan, False),
        (Domain(0.0, 3.0, Values.INTEGER), 2.0, True),
        (Domain(0.0, 3.0, Values.INTEGER), 2.3, False),
    ],
)
def test_the_values_a_domain_contains(
    domain: Domain, value: float, contained: bool
) -> None:
    assert domain.contains(value) is contained


@pytest.mark.parametrize(
    ("domain", "text"),
    [
        (Domain(0.0, 1.0), "[0, 1]"),
        (Domain(0.0, math.inf), "[0, inf)"),
        (Domain(-math.inf, math.inf), "(-inf, inf)"),
        (Domain(0.0, 3.0, Values.INTEGER), "[0, 3] integer"),
        (Domain(0, 1, Values.BOOLEAN), "[0, 1] boolean"),
        (Domain(0.5, 1.5), "[0.5, 1.5]"),
    ],
)
def test_a_domain_is_an_interval(domain: Domain, text: str) -> None:
    assert str(domain) == text
