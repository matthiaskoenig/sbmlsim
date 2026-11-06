"""The table of the parameters of the sensitivity analyses, on the toy model.

The toy model is the decay of A with the rate `k` in the compartment `Vc`.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim import Q
from sbmlsim.sensitivity.parameters import (
    COLUMNS,
    ParameterType,
    SensitivityParameter,
    build_values,
    design_factors,
    distributions,
    free_parameters,
    parameters_from_sbml,
    parameters_to_df,
    relative_design,
    set_bounds,
    set_relative_bounds,
    write_parameters,
)
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulation.sampling import Uniform, lhs
from tests.project_toy import PROJECT

MODEL_PATH = PROJECT.model_path


def test_the_constant_parameters_and_compartments_of_the_model() -> None:
    parameters = {p.uid: p for p in parameters_from_sbml(MODEL_PATH)}
    assert set(parameters) == {"k", "Vc"}
    assert parameters["k"].value == pytest.approx(0.2)
    assert parameters["k"].name == "rate of the decay"
    assert parameters["Vc"].value == pytest.approx(1.0)
    assert math.isnan(parameters["k"].lower_bound)
    assert parameters["k"].type is ParameterType.NA


def test_parameters_are_excluded() -> None:
    assert [p.uid for p in parameters_from_sbml(MODEL_PATH, exclude_ids={"Vc"})] == [
        "k"
    ]


def test_the_bounds_of_the_analysis() -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    set_bounds(
        parameters,
        [("k", 0.1, 0.3, ParameterType.FIT), ("unknown", 0, 1, ParameterType.DATA)],
    )
    set_relative_bounds(parameters, 0.15)
    by_uid = {p.uid: p for p in parameters}
    assert (by_uid["k"].lower_bound, by_uid["k"].upper_bound) == (0.1, 0.3)
    assert by_uid["k"].type is ParameterType.FIT
    assert by_uid["Vc"].lower_bound == pytest.approx(0.85)
    assert by_uid["Vc"].upper_bound == pytest.approx(1.15)


def test_the_table_is_sorted_by_type_and_id() -> None:
    parameters = [
        SensitivityParameter(uid="b", name="b", type=ParameterType.SCALING),
        SensitivityParameter(uid="a", name="a", type=ParameterType.SCALING),
        SensitivityParameter(uid="c", name="c", type=ParameterType.FIT),
    ]
    df = parameters_to_df(parameters)
    assert tuple(df.columns) == COLUMNS
    assert df["uid"].tolist() == ["c", "a", "b"]
    assert df["type"].tolist() == ["fitted", "scaling", "scaling"]
    assert parameters_to_df(parameters, sort=False)["uid"].tolist() == ["b", "a", "c"]


def test_the_table_is_written_as_tsv_and_latex(tmp_path: Path) -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    set_relative_bounds(parameters, 0.15)
    df = write_parameters(parameters, tmp_path / "sensitivity")
    read = pd.read_csv(tmp_path / "sensitivity" / "parameters.tsv", sep="\t")
    assert tuple(read.columns) == COLUMNS
    assert np.allclose(read["value"], df["value"])
    tex = (tmp_path / "sensitivity" / "parameters.tex").read_text(encoding="utf-8")
    assert r"lower\_bound" in tex
    assert "_bound" not in tex.replace(r"\_bound", "")


def test_the_distributions_of_a_global_analysis() -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    set_relative_bounds(parameters, 0.1)
    result = distributions(parameters)
    assert set(result) == {"k", "Vc"}
    assert isinstance(result["k"], Uniform)
    assert result["k"].lower == pytest.approx(0.18)


def test_a_parameter_without_bounds_has_no_distribution() -> None:
    with pytest.raises(ValueError, match="k"):
        distributions(parameters_from_sbml(MODEL_PATH))


def test_a_negative_value_has_the_bounds_of_sbmlsim_08() -> None:
    """The relative bounds of a negative value are reversed, as in sbmlsim 0.8."""
    parameters = [SensitivityParameter(uid="dE", name="dE", value=-2.0)]
    set_relative_bounds(parameters, 0.15)
    assert parameters[0].lower_bound == pytest.approx(-1.7)
    assert parameters[0].upper_bound == pytest.approx(-2.3)


def test_reversed_bounds_have_no_distribution() -> None:
    parameters = [
        SensitivityParameter(
            uid="k", name="k", value=1.0, lower_bound=0.9, upper_bound=1.1
        ),
        SensitivityParameter(uid="dE", name="dE", value=-2.0),
    ]
    set_relative_bounds(parameters, 0.15)
    with pytest.raises(ValueError, match=r"'dE' \(-1.7 > -2.3\)"):
        distributions(parameters)


def _parameters(*uids: str) -> list[SensitivityParameter]:
    return [SensitivityParameter(uid=uid, name=uid, value=1.0) for uid in uids]


def test_free_parameters_without_set_targets_keeps_all() -> None:
    parameters = _parameters("Ka", "CL", "f_renal_function")
    simulation = Simulation(end=10, steps=5, preinit_changes={"PODOSE": Q(1, "mg")})
    assert free_parameters(parameters, [simulation]) == parameters
    assert free_parameters(parameters, []) == parameters


def test_free_parameters_leaves_out_a_preinit_change() -> None:
    parameters = _parameters("Ka", "CL", "f_renal_function")
    simulation = Simulation(
        end=10, steps=5, preinit_changes={"f_renal_function": 0.5, "PODOSE": Q(1, "mg")}
    )
    free = free_parameters(parameters, [simulation])
    assert [p.uid for p in free] == ["Ka", "CL"]


def test_free_parameters_leaves_out_a_change() -> None:
    parameters = _parameters("Ka", "CL", "f_renal_function")
    simulation = Simulation(
        end=10, steps=5, changes=[Change([5.0], {"CL": Q(2.0, "1/hr")})]
    )
    free = free_parameters(parameters, [simulation])
    assert [p.uid for p in free] == ["Ka", "f_renal_function"]


def test_free_parameters_unites_the_simulations_and_keeps_the_order() -> None:
    parameters = _parameters("d", "c", "b", "a")
    simulations = [
        Simulation(end=10, steps=5, preinit_changes={"c": 1.0}),
        Simulation(end=10, steps=5, changes=[Change([1.0], {"a": 1.0})]),
    ]
    assert [p.uid for p in free_parameters(parameters, simulations)] == ["d", "b"]
    # a generator of simulations is consumed once
    assert [p.uid for p in free_parameters(parameters, iter(simulations))] == ["d", "b"]


def test_free_parameters_logs_the_removed_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    simulation = Simulation(end=10, steps=5, preinit_changes={"CL": 1.0})
    with caplog.at_level("INFO", logger="sbmlsim.sensitivity.parameters"):
        free_parameters(_parameters("Ka", "CL"), [simulation])
    assert "Removing fixed parameter: CL" in caplog.text
    assert "Ka" not in caplog.text


def test_free_parameters_leaves_out_a_target_of_the_presimulation() -> None:
    simulation = Simulation(
        end=10, steps=5, presimulation=SteadyState(preinit_changes={"CL": 1.0})
    )
    free = free_parameters(_parameters("Ka", "CL"), [simulation])
    assert [p.uid for p in free] == ["Ka"]


def test_free_parameters_leaves_out_a_change_at_several_times() -> None:
    simulation = Simulation(
        end=48, steps=5, changes=[Change([0.0, 24.0], {"CL": Q(2.0, "1/hr")})]
    )
    free = free_parameters(_parameters("Ka", "CL"), [simulation])
    assert [p.uid for p in free] == ["Ka"]


def test_free_parameters_compares_the_ids_as_written() -> None:
    simulation = Simulation(end=10, steps=5, changes=[Change([1.0], {"[S]": 1.0})])
    free = free_parameters(_parameters("S"), [simulation])
    assert [p.uid for p in free] == ["S"]


def derived_model(path: Path) -> Path:
    """Write the toy model with the derived parameter `k2 = 2 * k` into `path`."""
    import libsbml

    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    k2 = model.createParameter()
    k2.setId("k2")
    k2.setConstant(True)
    k2.setValue(0.0)
    assignment = model.createInitialAssignment()
    assignment.setSymbol("k2")
    assignment.setMath(libsbml.parseL3Formula("2 * k"))
    target = path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(target))
    return target


def test_derived_parameters_can_be_excluded(tmp_path: Path) -> None:
    path = derived_model(tmp_path)
    assert {p.uid for p in parameters_from_sbml(path)} == {"k", "Vc", "k2"}
    kept = parameters_from_sbml(path, exclude_derived=True)
    assert {p.uid for p in kept} == {"k", "Vc"}
    both = parameters_from_sbml(path, exclude_ids={"Vc"}, exclude_derived=True)
    assert [p.uid for p in both] == ["k"]


def test_the_table_has_the_values_of_other_variants(tmp_path: Path) -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    builds = {"slow": build_values(PROJECT.variant_path("slow"), ["k", "Vc", "x"])}
    assert builds == {"slow": {"k": pytest.approx(0.1), "Vc": pytest.approx(1.0)}}
    df = write_parameters(parameters, tmp_path, builds)
    assert list(df.columns)[:4] == ["uid", "name", "value", "value_slow"]
    table = df.set_index("uid")
    assert table.loc["k", "value_slow"] == pytest.approx(0.1)
    assert table.loc["k", "value"] == pytest.approx(0.2)


def test_the_factors_of_a_variant() -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    assert design_factors(parameters, {"k": 0.1, "Vc": 1.0}) == {
        "k": pytest.approx(0.5)
    }
    with pytest.raises(ValueError, match="'Vc' has no value in the variant"):
        design_factors(parameters, {"k": 0.1})
    zero = [SensitivityParameter(uid="k", name="k", value=0.0)]
    with pytest.raises(ValueError, match=r"'k' \(0 in the table, 0.1 in the variant\)"):
        design_factors(zero, {"k": 0.1})
    negative = [SensitivityParameter(uid="k", name="k", value=1.0)]
    with pytest.raises(ValueError, match="'k'"):
        design_factors(negative, {"k": -1.0})


def test_a_relative_design_scales_the_values_and_the_bounds() -> None:
    parameters = parameters_from_sbml(MODEL_PATH)
    set_relative_bounds(parameters, 0.15)
    design = lhs(distributions(parameters), n=4, seed=1)
    scaled = relative_design(design, {"k": 0.5, "other": 2.0})
    np.testing.assert_allclose(scaled.values["k"], 0.5 * np.asarray(design.values["k"]))
    np.testing.assert_array_equal(scaled.values["Vc"], design.values["Vc"])
    np.testing.assert_array_equal(scaled.labels, design.labels)
    assert scaled.design is not None and design.design is not None
    k = scaled.design.distributions["k"]
    assert (k["lower"], k["upper"]) == pytest.approx((0.085, 0.115))
    assert scaled.design.distributions["Vc"] == design.design.distributions["Vc"]
    assert relative_design(design, {"other": 2.0}) is design
