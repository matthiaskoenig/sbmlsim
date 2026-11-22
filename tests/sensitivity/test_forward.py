"""The local sensitivities from the forward sensitivities of sbmlsim-core."""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.model.model_sbml import SBMLModel
from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML
from sbmlsim.result import TimecourseResult, TimecourseSensitivities
from sbmlsim.sensitivity.forward import forward
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult
from sbmlsim.simulation import Scan, Simulation, sampling
from sbmlsim.simulator import Simulator

pytest.importorskip("sbmlsim_core")

#: the parameters of the repressilator the engine differentiates in; `n`, a
#: Hill exponent of a state which starts at zero, has no finite derivative
#: there, see the module of `sbmlsim.sensitivity.forward`
TARGETS = ["tau_prot", "ps_a"]

#: the end and the output times of the simulations of the tests
END = 20.0
STEPS = 5


@pytest.fixture(scope="module")
def simulator() -> Simulator:
    """Get a simulator of sbmlsim-core at tight tolerances."""
    return Simulator(engine="core", relative_tolerance=1e-10, absolute_tolerance=1e-12)


@pytest.fixture(scope="module")
def model(simulator: Simulator) -> SBMLModel:
    """Get the repressilator in sbmlsim-core."""
    return simulator.load(REPRESSILATOR_SBML)


def test_the_result_has_the_indices_of_every_selection(simulator, model) -> None:
    result = forward(
        simulator,
        model,
        Simulation(end=END, steps=STEPS),
        TARGETS,
        selections=["PX", "X"],
    )
    assert result.method == "forward"
    assert result.parameters == TARGETS
    assert set(result.ds.data_vars) == {
        "PX.raw",
        "PX.normalized",
        "X.raw",
        "X.normalized",
    }
    assert result.ds["PX.raw"].dims == (PARAMETER, "time")
    assert result.ds["PX.raw"].shape == (2, STEPS + 1)
    assert result.units["PX.normalized"] == "dimensionless"
    assert result.units["time"] == "min"
    assert np.allclose(result.ds["time"].values, np.linspace(0.0, END, STEPS + 1))


def test_the_selections_are_the_ones_of_the_model(simulator, model) -> None:
    result = forward(simulator, model, Simulation(end=END, steps=STEPS), ["ps_a"])
    names = {str(name).rsplit(".", 1)[0] for name in result.ds.data_vars}
    assert names == {s for s in model.selections if s != "time"}
    assert "time.raw" not in result.ds.data_vars


def test_the_indices_are_the_central_differences_of_a_local_design(
    simulator, model
) -> None:
    """The exact derivative is the limit of the difference of the design."""
    simulation = Simulation(end=END, steps=STEPS)
    exact = forward(simulator, model, simulation, TARGETS, selections=["PX"])
    grid = np.linspace(0.0, END, STEPS + 1)
    errors = []
    for delta in (1e-2, 1e-4):
        design = sampling.local(TARGETS, delta, model=model, simulation=simulation)
        scan = Scan(simulation, [design])
        differences = sensitivity.local(
            simulator.run(model, scan, keep=["PX"], time=grid)
        )
        for name in ("PX.raw", "PX.normalized"):
            a = differences.ds[name].values[:, -1]
            b = exact.ds[name].values[:, -1]
            errors.append(float(np.max(np.abs(a / b - 1.0))))
    # the difference of the design approaches the exact derivative, so its
    # error at the smaller step is the smaller one
    assert max(errors[:2]) < 1e-4
    assert max(errors[2:]) < 1e-6
    assert max(errors[2:]) < max(errors[:2])


def test_a_normalized_index_is_the_raw_one_over_the_reference(simulator, model) -> None:
    result = forward(
        simulator, model, Simulation(end=END, steps=STEPS), ["ps_a"], selections=["PX"]
    )
    values = simulator.simulate(model, Simulation(end=END, steps=STEPS))
    reference = float(values["ps_a"][0])
    raw = result.ds["PX.raw"].values[0]
    normalized = result.ds["PX.normalized"].values[0]
    y = values["PX"]
    assert y[0] == 0.0
    # the selection is zero at the start, where there is no relative change
    assert np.isnan(normalized[0])
    assert np.allclose(normalized[1:], raw[1:] * reference / y[1:], rtol=1e-6)
    assert result.ds.attrs["options"]["references"] == [reference]


def test_the_reference_is_the_value_after_the_pre_initialization(
    simulator, model
) -> None:
    changed = 2.0 * float(simulator.simulate(model, Simulation(end=1.0))["ps_a"][0])
    result = forward(
        simulator,
        model,
        Simulation(end=END, steps=STEPS, preinit_changes={"ps_a": changed}),
        ["ps_a"],
        selections=["PX"],
    )
    assert result.ds.attrs["options"]["references"] == [changed]


def test_the_derivative_of_a_parameter_in_itself_is_one(simulator, model) -> None:
    result = forward(
        simulator,
        model,
        Simulation(end=END, steps=STEPS),
        ["ps_a"],
        selections=["ps_a"],
    )
    assert np.allclose(result.ds["ps_a.raw"].values, 1.0)
    assert np.allclose(result.ds["ps_a.normalized"].values, 1.0)


def test_the_unit_of_a_raw_index_is_the_ratio_of_the_units(simulator) -> None:
    """The demo model, whose entities carry units; the repressilator has none."""
    demo = simulator.load(DEMO_SBML)
    simulation = Simulation(end=1.0, steps=STEPS)
    one = forward(simulator, demo, simulation, ["Vmax_bA"], selections=["e__A"])
    # the unit of the selection (`mole`) per unit of the parameter (`mol/s`)
    assert one.units["e__A.raw"] == "s"
    assert one.units["e__A.normalized"] == "dimensionless"
    several = forward(
        simulator, demo, simulation, ["Vmax_bA", "Km_A"], selections=["e__A"]
    )
    # the parameters of a variable do not share a unit, so it has none
    assert several.units["e__A.raw"] == ""


def test_the_indices_of_a_model_without_units_have_none(simulator, model) -> None:
    result = forward(
        simulator, model, Simulation(end=END, steps=STEPS), TARGETS, selections=["PX"]
    )
    assert result.units["PX.raw"] == ""


def test_the_analysis_checks_its_arguments(simulator, model) -> None:
    simulation = Simulation(end=END, steps=STEPS)
    with pytest.raises(ValueError, match="at least one parameter"):
        forward(simulator, model, simulation, [])
    with pytest.raises(ValueError, match="not selections of the model"):
        forward(simulator, model, simulation, ["ps_a"], selections=["nothing"])
    with pytest.raises(ValueError, match="parameters of the sensitivities repeat"):
        forward(simulator, model, simulation, ["ps_a", "ps_a"], selections=["PX"])


def test_another_engine_has_no_forward_sensitivities(model) -> None:
    roadrunner = Simulator(engine="roadrunner")
    with pytest.raises(NotImplementedError, match="sbmlsim-core"):
        forward(roadrunner, REPRESSILATOR_SBML, Simulation(end=END), ["ps_a"])
    with pytest.raises(NotImplementedError, match="sbmlsim-core"):
        roadrunner.sensitivities(REPRESSILATOR_SBML, Simulation(end=END), ["ps_a"])


def test_a_parameter_without_a_finite_derivative_is_reported(simulator, model) -> None:
    """The exponent of a state which starts at zero, see the module."""
    with pytest.raises(RuntimeError, match="integration failed"):
        forward(simulator, model, Simulation(end=END, steps=STEPS), ["n"])


def test_the_output_carries_the_derivatives_of_every_column(simulator, model) -> None:
    sensitivities = simulator.sensitivities(
        model, Simulation(end=END, steps=STEPS), TARGETS
    )
    result = sensitivities.result
    assert sensitivities.parameters == tuple(TARGETS)
    assert sensitivities.values.shape == (*result.values.shape, len(TARGETS))
    # the time is no function of a parameter
    assert np.all(sensitivities["time", "ps_a"] == 0.0)
    assert np.array_equal(
        sensitivities["PX", "ps_a"],
        sensitivities.values[:, result.columns.index("PX"), 1],
    )
    with pytest.raises(KeyError, match="not a column of the result"):
        sensitivities["nothing", "ps_a"]
    with pytest.raises(KeyError, match="No parameter 'nothing'"):
        sensitivities["PX", "nothing"]


def test_the_derivatives_are_checked_against_the_result() -> None:
    result = TimecourseResult(columns=("time", "PX"), values=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="the result and the parameters"):
        TimecourseSensitivities(
            result=result, parameters=("k",), values=np.zeros((3, 2, 2))
        )
    assert TimecourseSensitivities(
        result=result, parameters=("k",), values=np.zeros((3, 2, 1))
    ).values.shape == (3, 2, 1)


def test_the_selections_of_the_run_are_given(simulator, model) -> None:
    sensitivities = simulator.sensitivities(
        model, Simulation(end=END, steps=STEPS), ["ps_a"], selections=["PX"]
    )
    assert sensitivities.result.columns == ("time", "PX")


def test_the_result_is_a_sensitivity_result(simulator, model, tmp_path) -> None:
    """The methods of a `SensitivityResult` apply, the variables being timecourses."""
    result = forward(
        simulator,
        model,
        Simulation(end=END, steps=STEPS),
        TARGETS,
        selections=["PX", "PY"],
    )
    assert result.observables == ["PX", "PY"]
    at_end = result.sel(time=END).index("normalized")
    assert at_end.dims == (PARAMETER, "observable")
    assert at_end.shape == (len(TARGETS), 2)
    assert np.array_equal(
        at_end.sel(observable="PX").values,
        result.ds["PX.normalized"].values[:, -1],
    )
    frame = result.to_dataframe("PX.raw")
    assert list(frame.index) == TARGETS
    path = tmp_path / "forward.nc"
    result.to_netcdf(path)
    back = SensitivityResult.from_netcdf(path)
    assert back.method == "forward"
    assert back.parameters == TARGETS
    assert back.units["time"] == "min"
    assert np.array_equal(
        back.ds["PX.raw"].values, result.ds["PX.raw"].values, equal_nan=True
    )


def test_a_parameter_of_no_selection_is_reported(simulator, model) -> None:
    with pytest.raises(ValueError, match="parameters \\['nothing'\\] are not"):
        forward(
            simulator,
            model,
            Simulation(end=END, steps=STEPS),
            ["nothing"],
            selections=["PX"],
        )
