"""The grammar of the distrib formulas of the fitted parameters."""

import math

import pytest

from sbmlsim import distrib


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("normal(0.3, 0.05, 0.01, 0.6)", (0.01, 0.6)),
        ("lognormal(-6.2, 0.3, 1e-5, 0.1)", (1e-5, 0.1)),
        ("uniform(0.0, 1.5)", (0.0, 1.5)),
        ("normal(0.3, 0.05)", (-math.inf, math.inf)),
        ("lognormal(-6.2, 0.3)", (0.0, math.inf)),
        ("gamma(1, 2)", (-math.inf, math.inf)),
        ("normal(a, 0.1)", (-math.inf, math.inf)),
        ("normal(0.3)", (-math.inf, math.inf)),
    ],
)
def test_the_support_of_a_distribution(
    formula: str, expected: tuple[float, float]
) -> None:
    """Bounds of a truncation or uniform, lognormal positive, else unbounded."""
    assert distrib.support(formula) == expected


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        ("normal", "no function of numbers"),
        ("normal(a, 1)", "no numbers"),
        ("uniform(0, 1, 2, 3)", "not supported"),
        ("beta(1, 2)", "not supported"),
    ],
)
def test_a_formula_outside_the_grammar_is_refused(formula: str, match: str) -> None:
    """The parser names what is wrong."""
    with pytest.raises(ValueError, match=match):
        distrib.parse(formula)


def test_the_bounds_of_a_formula() -> None:
    """The last two of four arguments, the two of uniform, none of a plain normal."""
    assert distrib.parse("lognormal(-1.2, 0.3, 0.01, 1.0)").bounds == (0.01, 1.0)
    assert distrib.parse("uniform(0.1, 0.4)").bounds == (0.1, 0.4)
    assert distrib.parse("normal(0.5, 0.1)").bounds is None
