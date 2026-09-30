"""Tests of the report of a hybrid fit: one row per array of a network."""

from pathlib import Path

import numpy as np

from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import bound_warnings
from sbmlsim.sciml import network_fit_parameters
from tests.sciml.hybrid import feed_forward
from tests.sciml.test_fit import SETTINGS, _before, _problem


def _report(tmp_path: Path, fisher: bool = False) -> tuple[FitReport, dict]:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    values = dict(zip(problem.pids, np.asarray(problem.x0, dtype=float), strict=True))
    # one element at its bound
    values["net1__layer1__bias__0"] = 4.99
    pset = ParameterSet(sid="nominal", values=values)
    fim = fisher_information(problem, SETTINGS, pset) if fisher else None
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
    assert context["bound_warnings"] == [
        "nominal: !1 of the 3 elements of 'net1.layer1.bias' within 5% of a bound!"
    ]
    assert "net1" in report.fit_info()["networks"]
    path = report.create(tmp_path / "out", name="report")
    html = (path / "index.html").read_text()
    assert "net1.layer1.weight" in html
    assert "net1__layer1__weight__0_0" not in html
    text = (path / "report.txt").read_text()
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


def test_bound_warnings_count_the_elements_of_a_group() -> None:
    from sbmlsim.fit import FitParameter
    from sbmlsim.fit.options import ParameterScaleType

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
