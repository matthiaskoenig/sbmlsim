"""The domains of the quantities of a model, from distrib."""

import logging
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
    """Which `sampling.set_relative_bounds` builds for a negative value."""
    assert Domain(0.0, 1.0).clip(1.2, -0.1) == (1.2, -0.1)


def test_an_integer_domain_whose_interval_holds_no_integer_is_refused() -> None:
    """A domain which accepts no value is a mistake where it is written."""
    with pytest.raises(ValueError, match=r"\[0\.2, 0\.8\] holds no integer"):
        Domain(0.2, 0.8, Values.INTEGER)


def test_an_integer_domain_of_one_integer_is_a_domain() -> None:
    assert Domain(0.2, 1.8, Values.INTEGER).clip(0.5, 1.5) == (1.0, 1.0)


def test_an_interval_of_a_boolean_domain_is_widened_onto_its_values() -> None:
    """A fractional span of a switch is widened onto the values of the switch.

    `clip(0.85, 1.15)` of a boolean domain gave `(0.85, 1.0)`, a pair whose
    lower bound the domain itself rejects (matthiaskoenig/pkpdsim#230).
    """
    boolean = Domain(0.0, 1.0, Values.BOOLEAN)

    low, high = boolean.clip(0.85, 1.15)

    assert (low, high) == (0.0, 1.0)
    assert boolean.contains(low)
    assert boolean.contains(high)


def test_an_interval_of_an_integer_domain_is_widened_onto_its_values() -> None:
    domain = Domain(0.0, 4.0, Values.INTEGER)

    low, high = domain.clip(0.3, 3.7)

    assert (low, high) == (0.0, 4.0)
    assert domain.contains(low)
    assert domain.contains(high)


def test_widening_an_interval_keeps_both_bounds_inside_the_domain() -> None:
    """A domain whose own bounds are not integral is not left by the widening."""
    domain = Domain(0.5, 3.5, Values.INTEGER)

    low, high = domain.clip(0.6, 3.4)

    assert (low, high) == (1.0, 3.0)
    assert domain.contains(low)
    assert domain.contains(high)


def test_widening_an_interval_onto_the_values_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="sbmlsim.domains"):
        Domain(0.0, 1.0, Values.BOOLEAN).clip(0.85, 1.15)

    assert "widened to (0.0, 1.0)" in caplog.text


def test_an_integral_interval_of_an_integer_domain_is_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="sbmlsim.domains"):
        assert Domain(0.0, 4.0, Values.INTEGER).clip(1.0, 3.0) == (1.0, 3.0)

    assert [record.getMessage() for record in caplog.records] == []


def test_an_interval_of_a_continuous_domain_is_not_widened() -> None:
    assert Domain(0.0, 1.0).clip(0.2, 0.7) == (0.2, 0.7)


def test_a_reversed_interval_of_an_integer_domain_stays_reversed() -> None:
    """The contract of a reversed interval holds for an integer domain too.

    `sampling.set_relative_bounds` builds an inverted pair for a negative
    value on purpose and `sampling.distributions` refuses it by name later, so
    the widening must not repair such a pair into an interval a sampler would
    accept.
    """
    assert Domain(0.0, 4.0, Values.INTEGER).clip(1.5, 0.5) == (1.5, 0.5)


def test_an_infinite_bound_of_an_integer_domain_is_not_widened() -> None:
    domain = Domain(0.0, math.inf, Values.INTEGER)

    assert domain.clip(0.3, math.inf) == (0.0, math.inf)


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
