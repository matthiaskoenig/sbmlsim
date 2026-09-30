"""Tests of the report of a hybrid fit: one row per array of a network."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from sbmlsim.fit import FitParameter
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.helpers import filter_keys
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.objects import describe_array
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.sciml import network_fit_parameters
from tests.sciml.hybrid import feed_forward
from tests.sciml.test_fit import SETTINGS, _before, _problem


def _report(
    tmp_path: Path,
    fisher: bool = False,
    mechanistic: bool = True,
    data_points: int | None = None,
) -> tuple[FitReport, dict]:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    problem = _problem([_before(network)], elements)
    if not mechanistic:
        problem = OptimizationProblem(
            opid="hybrid",
            mapping_collections=problem.mapping_collections,
            fit_parameters=elements,
            base_path=problem.base_path,
            data_path=problem.data_path,
            hybridizations=problem.hybridizations,
        )
    problem.initialize(SETTINGS)
    values = dict(zip(problem.pids, np.asarray(problem.x0, dtype=float), strict=True))
    # one element at its bound
    values["net1__layer1__bias__0"] = 4.99
    pset = ParameterSet(sid="nominal", values=values)
    fim = fisher_information(problem, SETTINGS, pset) if fisher else None
    if fim is not None and data_points is not None:
        fim = dataclasses.replace(fim, n=data_points)
    report = FitReport(problem, SETTINGS, pset, fisher=fim, mapping_figures=False)
    context = report.html_context(tmp_path, "report")
    return report, context


def test_the_overview_shows_the_network_and_its_arrays(tmp_path: Path) -> None:
    report, context = _report(tmp_path)
    assert [row["pid"] for row in context["parameters"]] == ["alpha", "beta"]
    (hook,) = context["hooks"]
    assert hook == {
        "name": "net1",
        "kind": "pre_initialization",
        "description": "layer1 (Linear), layer2 (Linear)",
        "targets": "gamma",
    }
    arrays = {row["label"]: row for row in context["arrays"]}
    assert set(arrays) == {
        "net1.layer1.weight",
        "net1.layer1.bias",
        "net1.layer2.weight",
        "net1.layer2.bias",
    }
    row = arrays["net1.layer1.weight"]
    assert (row["elements"], row["estimated"], row["lower"], row["upper"]) == (
        6,
        6,
        "-5",
        "5",
    )
    (values,) = row["set_values"]
    assert len(values) == 3 and all(value != "-" for value in values)
    # the warnings of an array are under the table of the arrays
    assert context["bound_warnings"] == []
    assert context["array_bound_warnings"] == [
        "nominal: !1 of the 3 elements of 'net1.layer1.bias' within 5% of a bound!"
    ]
    assert "net1" in report.fit_info()["networks"]
    # one collection per simulation, one class
    assert report.fit_info()["experiments"] == "LotkaVolterra"
    path = report.create(tmp_path / "out", name="report")
    html = (path / "index.html").read_text()
    assert "net1.layer1.weight" in html
    assert "net1__layer1__weight__0_0" not in html
    text = (path / "report.txt").read_text()
    assert "1 of the 3 elements of 'net1.layer1.bias'" in text


def test_a_fit_of_elements_only_has_no_parameters_table(tmp_path: Path) -> None:
    """Without parameters which are no elements the report has no empty table."""
    report, context = _report(tmp_path, mechanistic=False)
    assert context["parameters"] == []
    assert context["array_bound_warnings"] == [
        "nominal: !1 of the 3 elements of 'net1.layer1.bias' within 5% of a bound!"
    ]
    path = report.create(tmp_path / "out", name="report")
    html = (path / "index.html").read_text()
    assert "<h3>Parameters</h3>" not in html
    networks = html.index("<h3>Networks</h3>")
    assert html.index("of &#39;net1.layer1.bias&#39; within 5%", networks) > networks
    text = (path / "report.txt").read_text()
    assert "Empty DataFrame" not in text
    assert "1 of the 3 elements of 'net1.layer1.bias'" in text


def test_the_fisher_table_has_one_row_per_array(tmp_path: Path) -> None:
    _, context = _report(tmp_path, fisher=True)
    fisher = context["fisher"]
    labels = [row[0] for row in fisher["rows"]]
    assert labels[:2] == ["alpha", "beta"]
    assert "net1.layer1.weight (6 elements)" in labels
    assert len(labels) == 2 + 4
    assert fisher["pids"] == ["alpha", "beta"]
    assert len(fisher["correlation"]) == 2
    assert "13 elements are left out" in fisher["note"]


def test_the_fisher_table_without_errors_shows_no_nan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without degrees of freedom there is no error, which is `-` and not `nan`."""
    with caplog.at_level(logging.WARNING):
        _, context = _report(tmp_path, fisher=True, data_points=15)
    fisher = context["fisher"]
    se = [column["name"] for column in fisher["columns"]].index("se")
    rows = {row[0]: row for row in fisher["rows"]}
    assert rows["alpha"][se] == "-"
    assert rows["net1.layer1.weight (6 elements)"][se] == "-"
    assert not any("nan" in cell for row in fisher["rows"] for cell in row)


def test_bound_warnings_count_the_elements_of_a_group() -> None:
    parameters = [FitParameter(f"w{k}", 0.0, -1.0, 1.0) for k in range(3)]
    x = np.array([0.99, -0.99, 0.0])
    scales = [ParameterScaleType.LINEAR] * 3
    assert bound_warnings(parameters, x, scales) == [
        "!Optimal parameter 'w0' within 5% of upper bound!",
        "!Optimal parameter 'w1' within 5% of lower bound!",
    ]
    assert bound_warnings(
        parameters, x, scales, groups={"net.w": ["w0", "w1", "w2"]}
    ) == ["!2 of the 3 elements of 'net.w' within 5% of a bound!"]
    assert bound_warnings(
        parameters[:1], x[:1], scales[:1], groups={"net.b": ["w0"]}
    ) == ["!1 of the 1 element of 'net.b' within 5% of a bound!"]


def _result(problem: OptimizationProblem, x: np.ndarray) -> OptimizationResult:
    """Get the result of a fit with one run at `x`."""
    return OptimizationResult(
        parameters=problem.parameters,
        fits=[
            OptimizeResult(
                x=x, cost=1.0, status=1, success=True, duration=0.1, x0=x.copy()
            )
        ],
        trajectories=[[1.0]],
        settings=SETTINGS,
        opid=problem.opid,
    )


def test_the_text_report_has_one_row_per_array(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    x[problem.pids.index("net1__layer1__bias__0")] = 4.99
    pset = ParameterSet(sid="fit", values=dict(zip(problem.pids, x, strict=True)))
    report = FitReport(
        problem,
        SETTINGS,
        pset,
        opt_result=_result(problem, x),
        mapping_figures=False,
    )
    path = report.create(tmp_path / "out", name="report")
    text = (path / "report.txt").read_text()
    # no element is listed, neither of the problem, nor of the sets, nor of the fit
    assert "net1__layer" not in text
    for label in ("net1.layer1.weight", "net1.layer2.bias"):
        # the problem, the parameter set and the optimal parameters
        assert text.count(f"{label}:") == 3
    assert "alpha" in text and "beta" in text
    assert "1 of the 3 elements of 'net1.layer1.bias'" in text
    # the problem alone, before the report
    assert "net1__layer" not in str(problem)
    assert "net1.layer1.weight: 6 of 6 elements estimated" in str(problem)


def test_an_array_of_one_element_is_described_in_the_singular() -> None:
    element = FitParameter("b0", 0.5, -1.0, 1.0, unit="dimensionless")
    assert describe_array("net.b", 1, [element], [0.5]) == (
        "net.b: 1 of 1 element estimated, min 0.5, max 0.5, norm 0.5, bounds [-1, 1]"
    )
    assert describe_array("net.w", 3, [], []) == "net.w: 0 of 3 elements estimated"


def test_the_bound_warnings_count_the_elements_of_versioned_parameters() -> None:
    parameters = [
        FitParameter("w_a", 0.0, -1.0, 1.0, target="w"),
        FitParameter("w_b", 0.0, -1.0, 1.0, target="w"),
        FitParameter("v", 0.0, -1.0, 1.0),
        FitParameter("u", 0.0, -1.0, 1.0),
    ]
    x = np.array([0.99, 0.99, 0.99, 0.0])
    scales = [ParameterScaleType.LINEAR] * 4
    # w is one element, at its bound in two versions; x is an element which
    # the fit does not estimate
    assert bound_warnings(
        parameters, x, scales, groups={"net.w": ["w", "v", "u", "x"]}
    ) == ["!2 of the 4 elements (3 estimated) of 'net.w' within 5% of a bound!"]


def test_a_versioned_element_is_one_element_of_its_array(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    target = "net1__layer2__bias__0"
    versions = [
        FitParameter(
            f"bias_{sid}",
            0.1,
            0.1,
            5.0,
            unit="dimensionless",
            scale=scale,
            target=f"sciml:{target}",
            mappings=filter_keys([f"prey_{sid}", f"predator_{sid}"]),
        )
        for sid, scale in (
            ("e1", ParameterScaleType.LINEAR),
            ("e2", ParameterScaleType.LOG10),
        )
    ]
    elements = [p for p in elements if p.pid != target] + versions
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    pset = ParameterSet(sid="fit", values=dict(zip(problem.pids, x, strict=True)))
    fim = fisher_information(problem, SETTINGS, pset)
    report = FitReport(problem, SETTINGS, pset, fisher=fim, mapping_figures=False)
    context = report.html_context(tmp_path, "report")
    labels = [row[0] for row in context["fisher"]["rows"]]
    assert "net1.layer2.bias (1 element, 2 parameters)" in labels
    # the versions are searched on different scales
    assert context["fisher"]["rows"][-1][2] == "mixed"
    rows = {row["label"]: row for row in context["arrays"]}
    row = rows["net1.layer2.bias"]
    assert (row["elements"], row["estimated"]) == (1, 1)
    assert report.parameter_groups()["net1.layer2.bias"] == (target,)
    assert {p["pid"] for p in context["parameters"]} == {"alpha", "beta"}


def test_the_profiles_default_to_the_parameters_which_are_no_elements() -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    values = dict(zip(problem.pids, np.asarray(problem.x0, dtype=float), strict=True))
    pset = ParameterSet(sid="nominal", values=values)
    profile_settings = ProfileSettings(max_points=3, reoptimize=False)

    result = profile_likelihood(
        problem,
        SETTINGS,
        pset,
        profile_settings,
        serial=True,
        show_progress=False,
    )
    assert list(result.profiles) == ["alpha", "beta"]

    # the scan ran on the linear scale: the values are the values of the
    # parameters, within their bounds, and the optimum is the fitted value
    for pid, (lower, upper) in {"alpha": (0.0, 15.0), "beta": (0.0, 15.0)}.items():
        profile = result.profiles[pid]
        assert profile.value_optimum == pytest.approx(values[pid])
        assert np.all(profile.values >= lower) and np.all(profile.values <= upper)
        assert len(profile.values) > 1
        # the path of every parameter is on the linear scale as well
        assert np.all(np.abs(profile.paths) < 100.0)

    # an element is profiled when it is named
    element = "net1__layer1__bias__0"
    named = profile_likelihood(
        problem,
        SETTINGS,
        pset,
        profile_settings,
        pids=["alpha", element],
        serial=True,
        show_progress=False,
    )
    assert list(named.profiles) == ["alpha", element]
    profile = named.profiles[element]
    assert profile.value_optimum == pytest.approx(values[element])
    assert np.all(np.abs(profile.values) <= 5.0)

    # nothing to profile: no parameter named, or only elements in the problem
    with pytest.raises(ValueError, match="`pids` names none"):
        profile_likelihood(problem, SETTINGS, pset, profile_settings, pids=[])
    only_elements = OptimizationProblem(
        opid="elements",
        mapping_collections=problem.mapping_collections,
        fit_parameters=elements,
        base_path=problem.base_path,
        data_path=problem.data_path,
        hybridizations=problem.hybridizations,
    )
    with pytest.raises(ValueError, match="all elements of networks"):
        profile_likelihood(
            only_elements,
            SETTINGS,
            ParameterSet(sid="nominal", values=values),
            profile_settings,
        )
