"""The polynomial chaos expansion."""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.chaos import (
    MAX_TERMS,
    POINTS_PER_TERM,
    TRUNCATIONS,
    PolynomialChaos,
    choose_truncation,
    design_matrix,
    expansion,
    fit_chaos,
    legendre_values,
    pce,
    supported,
)
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2
from sbmlsim.simulation import (
    Custom,
    Dimension,
    Formula,
    Observable,
    Scan,
    Simulation,
    sampling,
)
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import CHAIN, ISHIGAMI, s2_end_fails_for_high_s1_and_k1
from tests.simulator.models import sbml

PI = np.pi
BOUNDS = {"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI), "x3": Uniform(-PI, PI)}
OBSERVABLES: list[Observable] = [Formula("y_max", "max(y)")]

# the analytic indices of the Ishigami function with a = 7 and b = 0.1
S1_ISHIGAMI = [0.3139, 0.4424, 0.0]
ST_ISHIGAMI = [0.5576, 0.4424, 0.2437]


def _ishigami(u: np.ndarray) -> np.ndarray:
    """The Ishigami function of the points of the unit cube."""
    x = -PI + 2 * PI * np.asarray(u, dtype=float)
    return (
        np.sin(x[:, 0])
        + 7.0 * np.sin(x[:, 1]) ** 2
        + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
    )


def _run(design: Dimension, *others: Dimension) -> ScanResult:
    """Run the Ishigami model over a design."""
    model = Simulator().load(sbml(ISHIGAMI))
    return Simulator(n_workers=1).run(
        model, Scan(Simulation(end=1, steps=1), [*others, design]), OBSERVABLES
    )


def test_the_terms_of_an_expansion() -> None:
    # every multi-index of the degree, (p + d)! / (p! d!) of them
    assert expansion(3, 0).tolist() == [[0, 0, 0]]
    assert expansion(1, 3).tolist() == [[0], [1], [2], [3]]
    assert expansion(3, 4).shape == (35, 3)
    assert expansion(5, 3).shape == (56, 5)
    # ordered by the total degree, the constant first
    terms = expansion(3, 3)
    assert terms[0].tolist() == [0, 0, 0]
    assert np.all(np.diff(terms.sum(axis=1)) >= 0)
    # every term of a degree once
    assert len({tuple(row) for row in terms.tolist()}) == terms.shape[0]


def test_the_truncation_of_an_expansion() -> None:
    # the hyperbolic norm drops the terms of many inputs at a high degree
    full = expansion(4, 5)
    sparse = expansion(4, 5, q=0.5)
    assert sparse.shape[0] < full.shape[0]
    for row in sparse:
        assert np.sum(row.astype(float) ** 0.5) ** 2.0 <= 5.0 + 1e-9
    # every term of one input is kept, whatever q
    assert np.count_nonzero(np.count_nonzero(sparse, axis=1) <= 1) == 4 * 5 + 1
    # the interaction order caps the number of inputs of a term
    capped = expansion(4, 5, max_interaction=2)
    assert np.all(np.count_nonzero(capped, axis=1) <= 2)
    assert capped.shape[0] < full.shape[0]


def test_the_arguments_of_a_basis_are_checked() -> None:
    for arguments in ({"d": 0, "degree": 2}, {"d": 2, "degree": -1}):
        with pytest.raises(ValueError):
            expansion(**arguments)
    with pytest.raises(ValueError, match="q must be in"):
        expansion(2, 2, q=0.0)
    with pytest.raises(ValueError, match="interaction order"):
        expansion(2, 2, max_interaction=0)
    with pytest.raises(ValueError, match=f"more than {MAX_TERMS}"):
        expansion(20, 10)


def test_the_basis_is_orthonormal() -> None:
    # the integrals of the products over the unit cube, by many points
    u = np.random.default_rng(3).random((200000, 2))
    x = design_matrix(u, expansion(2, 4))
    gram = x.T @ x / u.shape[0]
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=0.02)
    # the first term is the constant one
    np.testing.assert_allclose(x[:, 0], 1.0)


def test_the_values_of_the_polynomials() -> None:
    u = np.array([[0.0], [0.5], [1.0]])
    values = legendre_values(u, 3)
    assert values.shape == (3, 1, 4)
    # P_k(2u - 1) normalized, i.e. sqrt(2k + 1) at u = 1
    np.testing.assert_allclose(values[2, 0], np.sqrt(2 * np.arange(4) + 1.0))
    np.testing.assert_allclose(values[1, 0], [1.0, 0.0, -np.sqrt(5) / 2, 0.0])
    np.testing.assert_allclose(
        values[0, 0], np.sqrt(2 * np.arange(4) + 1.0) * [1, -1, 1, -1]
    )


def test_the_design_matrix_checks_its_arguments() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        design_matrix(np.zeros(4), expansion(1, 2))
    with pytest.raises(ValueError, match="2 inputs"):
        design_matrix(np.zeros((4, 2)), expansion(3, 2))


def test_an_expansion_of_a_polynomial_is_exact() -> None:
    # y = 1 + 2 u1 + 3 u2^2 - u1 u2 is in the basis of the degree 2
    rng = np.random.default_rng(5)
    u = rng.random((200, 2))
    y = 1.0 + 2.0 * u[:, 0] + 3.0 * u[:, 1] ** 2 - u[:, 0] * u[:, 1]
    chaos = fit_chaos(u, y[:, None], degree=3)
    assert chaos.loo[0] < 1e-20
    assert chaos.degree.tolist() == [2]
    np.testing.assert_allclose(chaos.predict(u).ravel(), y, atol=1e-12)
    # the moments of the expansion are the ones of the distribution of the
    # inputs, not of the sample: 1 + 2/2 + 3/3 - 1/4 for a uniform unit cube
    np.testing.assert_allclose(chaos.mean, 2.75, rtol=1e-10)
    assert chaos.variance[0] == pytest.approx(y.var(), rel=0.1)


def test_the_indices_of_the_ishigami_function() -> None:
    u = np.random.default_rng(1).random((2000, 3))
    chaos = fit_chaos(
        u, _ishigami(u)[:, None], degree=10, parameters=["x1", "x2", "x3"]
    )
    assert chaos.parameters == ("x1", "x2", "x3")
    assert chaos.loo[0] < 1e-4
    np.testing.assert_allclose(chaos.first_order().ravel(), S1_ISHIGAMI, atol=1e-3)
    np.testing.assert_allclose(chaos.total_order().ravel(), ST_ISHIGAMI, atol=1e-3)
    np.testing.assert_allclose(chaos.mean, 3.5, atol=1e-3)
    # the interaction of x1 and x3 is the rest of their total indices
    pairs = chaos.second_order()[:, :, 0]
    assert pairs[0, 2] == pytest.approx(0.2437, abs=1e-3)
    assert pairs[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert pairs[1, 2] == pytest.approx(0.0, abs=1e-3)
    # the lower triangle and the diagonal have no pairs
    assert np.all(np.isnan(np.tril(np.ones_like(pairs)) * pairs + np.nan * pairs * 0))


def test_an_expansion_fits_every_element_at_once() -> None:
    # two elements of different degrees, each fitted with its own basis
    rng = np.random.default_rng(7)
    u = rng.random((300, 2))
    linear = 2.0 * u[:, 0] - u[:, 1]
    cubic = u[:, 0] ** 3 + u[:, 1]
    chaos = fit_chaos(u, np.column_stack([linear, cubic]), degree=4)
    assert chaos.degree[0] < chaos.degree[1] == 3
    np.testing.assert_allclose(
        chaos.predict(u), np.column_stack([linear, cubic]), atol=1e-12
    )
    # the terms above the degree of an element are zero
    totals = expansion(2, 4).sum(axis=1)
    for element, degree in enumerate(chaos.degree):
        assert np.all(chaos.coefficients[totals > degree, element] == 0.0)


def test_an_expansion_checks_its_arguments() -> None:
    u = np.random.default_rng(2).random((40, 2))
    y = u[:, :1]
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_chaos(u, y.ravel())
    with pytest.raises(ValueError, match="40 rows"):
        fit_chaos(u, y[:20])
    with pytest.raises(ValueError, match="at least 1"):
        fit_chaos(u, y, degree=0)
    with pytest.raises(ValueError, match="2 inputs"):
        fit_chaos(u, y, parameters=["a"])
    with pytest.raises(ValueError, match="No degree of the basis"):
        fit_chaos(u[:2], y[:2])


def test_the_analysis_of_a_scan() -> None:
    res = _run(sampling.lhs(BOUNDS, 600, seed=1))
    s = pce(res, degree=9)
    assert s.method == "pce" and s.ds.attrs["options"]["degree"] == 9
    assert list(s.ds["y_max.S1"].dims) == [PARAMETER]
    np.testing.assert_allclose(s["y_max.S1"].values, S1_ISHIGAMI, atol=0.02)
    np.testing.assert_allclose(s["y_max.ST"].values, ST_ISHIGAMI, atol=0.02)
    # the quality of the fit and the moments of the expansion
    assert float(s["y_max.loo"]) < 0.01
    # the degree is chosen from the leave-one-out error, not given
    assert 6 <= int(s["y_max.degree"]) <= 9
    assert float(s["y_max.mean"]) == pytest.approx(3.5, abs=0.05)
    assert float(s["y_max.sd"]) == pytest.approx(np.sqrt(13.84), abs=0.2)
    assert s.units["y_max.S1"] == "dimensionless"
    assert s.units["y_max.mean"] == s.units["y_max.sd"] == res.units["y_max"]
    assert "y_max.S2" not in s


def test_the_second_order_of_a_scan() -> None:
    res = _run(sampling.random(BOUNDS, 600, seed=2))
    s = pce(res, degree=9, second_order=True)
    pairs = s["y_max.S2"]
    assert list(pairs.dims) == [PARAMETER, PARAMETER_2]
    assert float(pairs[0, 2]) == pytest.approx(0.2437, abs=0.02)
    assert np.isnan(float(pairs[1, 0])) and np.isnan(float(pairs[0, 0]))


def test_the_indices_agree_with_a_sobol_analysis() -> None:
    # the same model, a tenth of the simulations of the Sobol design
    sobol_result = _run(sampling.sobol(BOUNDS, 4096, seed=1))
    estimated = sensitivity.sobol(sobol_result)
    expanded = pce(_run(sampling.lhs(BOUNDS, 800, seed=1)), degree=9)
    for key in ("S1", "ST"):
        np.testing.assert_allclose(
            expanded[f"y_max.{key}"].values,
            estimated[f"y_max.{key}"].values,
            atol=0.02,
        )


def test_the_expansion_of_a_sobol_design() -> None:
    # the points of a Saltelli matrix are a sample of the inputs as well
    res = _run(sampling.sobol(BOUNDS, 256, seed=3))
    s = pce(res, degree=8)
    np.testing.assert_allclose(s["y_max.S1"].values, S1_ISHIGAMI, atol=0.05)


def test_a_timecourse_has_an_expansion_per_time_point() -> None:
    model = Simulator().load(sbml(CHAIN))
    design = sampling.lhs(
        {"k1": Uniform(0.5, 1.5), "k2": Uniform(0.5, 1.5)}, 120, seed=4
    )
    res = Simulator(n_workers=1).run(model, Scan(Simulation(end=5, steps=20), [design]))
    s = pce(res, observables=["[S3]"], degree=5)
    assert list(s.ds["[S3].S1"].dims) == [PARAMETER, "time"]
    assert list(s.ds["[S3].loo"].dims) == ["time"]
    # the first time point is the constant initial state, the later ones vary
    assert np.allclose(s["[S3].S1"].values[:, 0], 0.0)
    assert float(s["[S3].ST"].values[0, -1]) > 0.1
    assert np.nanmax(s["[S3].loo"].values) < 1e-3


def test_a_constant_element_has_zero_indices() -> None:
    model = Simulator().load(sbml(CHAIN))
    design = sampling.lhs({"k1": Uniform(0.5, 1.5)}, 60, seed=5)
    res = Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=2, steps=2), [design]),
        [Formula("total", "max([S1] + [S2] + [S3])")],
    )
    s = pce(res, degree=3)
    assert s["total.S1"].values.tolist() == [0.0]
    assert s["total.ST"].values.tolist() == [0.0]
    assert float(s["total.loo"]) == 0.0
    assert float(s["total.sd"]) == 0.0
    assert float(s["total.mean"]) == pytest.approx(1.0, abs=1e-9)


def test_a_failed_point_leaves_no_indices(caplog: pytest.LogCaptureFixture) -> None:
    model = Simulator().load(sbml(CHAIN))
    design = sampling.lhs(
        {"[S1]": Uniform(1.0, 10.0), "k1": Uniform(0.5, 1.5)}, 60, seed=6
    )
    res = Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=2, steps=2), [design]),
        [
            Custom(
                "s2_end",
                s2_end_fails_for_high_s1_and_k1,
                "dimensionless",
                symbols=["[S1]", "[S2]", "k1"],
            )
        ],
        on_error="flag",
    )
    s = pce(res, degree=3)
    assert "contain a failed simulation" in caplog.text
    assert np.all(np.isnan(s["s2_end.S1"].values))
    assert np.isnan(float(s["s2_end.loo"]))


def test_a_correlated_design_is_refused() -> None:
    res = _run(
        sampling.lhs(
            {"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI)},
            60,
            seed=7,
            correlation=[[1.0, 0.5], [0.5, 1.0]],
        )
    )
    with pytest.raises(ValueError, match="has a correlation"):
        pce(res)


def test_a_design_of_another_method_is_refused() -> None:
    res = _run(sampling.fast(BOUNDS, 65, seed=1))
    with pytest.raises(ValueError, match="no dimension with a design"):
        pce(res)


def test_too_few_points_are_reported() -> None:
    res = _run(sampling.lhs(BOUNDS, 6, seed=8))
    with pytest.raises(ValueError, match="No degree of the basis"):
        pce(res, degree=6)


def test_the_points_which_support_a_basis() -> None:
    """A basis needs `POINTS_PER_TERM` points per term and a degree of freedom."""
    assert supported(int(POINTS_PER_TERM * 10), 10)
    assert not supported(int(POINTS_PER_TERM * 10) - 1, 10)
    # a basis of one term needs more than two points, whatever the budget says
    assert not supported(2, 1)
    assert supported(3, 1)


def test_one_decomposition_serves_every_candidate_degree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bases of the candidates are prefixes, so their QR is one QR.

    The fit of a candidate is the leading block of the decomposition of the
    largest basis the points support, which is what makes the expansion of a
    model of twenty parameters affordable: the decomposition is the cost of
    the method and it was taken once per candidate degree.
    """
    rng = np.random.default_rng(3)
    u = rng.random((400, 4))
    y = np.column_stack([np.sin(3 * u[:, 0]) + u[:, 1] ** 2, u.sum(axis=1)])
    calls: list[tuple[int, ...]] = []
    qr = np.linalg.qr

    def counted(a: np.ndarray) -> object:
        calls.append(a.shape)
        return qr(a)

    monkeypatch.setattr(np.linalg, "qr", counted)
    chaos = fit_chaos(u, y, degree=4)
    assert len(calls) == 1, calls
    # of the whole basis, which the 400 points support
    assert calls[0] == (400, chaos.multi_indices.shape[0])
    # every candidate was still tried: the second element is linear in its
    # inputs and keeps the degree 1, the first one needs more
    assert chaos.degree[1] == 1
    assert chaos.degree[0] > 1

    # of the largest candidate the points support where they support not every
    # one of them, and once again
    calls.clear()
    few = fit_chaos(u[:60], y[:60], degree=4)
    assert len(calls) == 1, calls
    terms = calls[0][1]
    assert supported(60, terms)
    assert terms < few.multi_indices.shape[0]
    assert few.degree.max() <= 2


def test_the_expansion_is_a_surrogate_of_the_model() -> None:
    # the expansion predicts points which were not simulated
    rng = np.random.default_rng(9)
    u = rng.random((1500, 3))
    chaos = fit_chaos(u, _ishigami(u)[:, None], degree=10)
    assert isinstance(chaos, PolynomialChaos)
    other = rng.random((500, 3))
    predicted = chaos.predict(other).ravel()
    exact = _ishigami(other)
    assert np.max(np.abs(predicted - exact)) < 0.05 * np.ptp(exact)


def test_a_chosen_truncation_is_the_richest_basis_the_points_support() -> None:
    """The exponent falls until the basis fits, the interactions are kept."""
    # the 22 parameters of the HCTZ model of pkdb_models on 2048 points: the
    # full basis of the degree 4 has 14950 terms, the hyperbolic one 782
    q, max_interaction = choose_truncation(22, 2048, 4)
    assert (q, max_interaction) == (0.75, None)
    terms = len(expansion(22, 4, q=q, max_interaction=max_interaction))
    assert terms == 782
    assert terms <= 2048 // POINTS_PER_TERM
    assert len(expansion(22, 4)) > 2048 // POINTS_PER_TERM


def test_a_chosen_truncation_falls_back_to_the_additive_basis() -> None:
    """A sample which supports no interaction gets the terms of one input each."""
    q, max_interaction = choose_truncation(22, 512, 5)
    assert (q, max_interaction) == TRUNCATIONS[-1] == (1.0, 1)
    assert len(expansion(22, 5, q=q, max_interaction=max_interaction)) == 22 * 5 + 1


def test_a_chosen_truncation_is_fitted_and_gives_the_indices() -> None:
    """The chosen basis of the Ishigami function finds its inputs."""
    rng = np.random.default_rng(3)
    u = rng.random((600, 3))
    q, max_interaction = choose_truncation(3, 600, 5)
    chaos = fit_chaos(
        u, _ishigami(u)[:, None], degree=5, q=q, max_interaction=max_interaction
    )
    total = chaos.total_order().ravel()
    # x3 enters the Ishigami function only through its interaction with x1
    assert total[0] > total[1] > total[2] > 0.0


def test_a_sample_which_supports_no_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="the additive basis, has 7 terms"):
        choose_truncation(2, 4, 3)


@pytest.mark.parametrize(
    ("d", "n", "degree", "message"),
    [
        (0, 10, 2, "at least one input"),
        (2, 0, 2, "at least one point"),
        (2, 10, -1, "must not be negative"),
    ],
)
def test_a_chosen_truncation_checks_its_arguments(
    d: int, n: int, degree: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        choose_truncation(d, n, degree)
