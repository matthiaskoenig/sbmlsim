"""Serialization preserves declared behavior and rejects unsupported values."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.model import AbstractModel
from sbmlsim.plot import plotting
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.serialization import to_json
from sbmlsim.task import Task


def test_numpy_scalars_remain_json_numbers() -> None:
    """Numeric table values must not become strings when stored."""
    encoded = to_json(
        {"integer": np.int64(3), "float": np.float32(1.25), "bool": np.bool_(True)}
    )
    assert json.loads(encoded) == {"integer": 3, "float": 1.25, "bool": True}


def test_different_formula_parameters_have_different_definitions() -> None:
    """The stored formula contains the numbers which determine its result."""
    source = Data("X", task="task")
    left = Data("scaled", function="x*k", variables={"x": source}, parameters={"k": 2})
    right = Data("scaled", function="x*k", variables={"x": source}, parameters={"k": 3})
    assert to_json(left) != to_json(right)
    definition = json.loads(to_json(left))
    assert definition["parameters"] == {"k": 2}
    assert definition["sid"] == left.sid


def test_model_and_task_definitions_keep_configuration(tmp_path: Path) -> None:
    """Selections, resource directory, and task labels survive serialization."""
    model = AbstractModel(
        REPRESSILATOR_SBML, base_path=tmp_path, selections=["time", "X"]
    )
    definition = json.loads(to_json(model))
    assert definition["selections"] == ["time", "X"]
    assert definition["base_path"] == str(tmp_path)
    assert definition["source"]["path"] is not None
    assert definition["source"]["content"] is None
    task = Task("model", "simulation", sid="custom", name="Custom task")
    assert json.loads(to_json(task)) == {
        "model": "model",
        "simulation": "simulation",
        "sid": "custom",
        "name": "Custom task",
    }


def test_supported_values_keep_existing_representations() -> None:
    """Paths and quantities use explicit codecs, including array magnitudes."""

    @dataclass
    class Values:
        path: Path
        quantity: object

    value = Values(Path("model.xml"), Q(np.array([1, 2]), "mg"))
    assert json.loads(to_json(value)) == {
        "path": "model.xml",
        "quantity": {"_magnitude": [1, 2], "_units": "milligram"},
    }


@pytest.mark.parametrize("value", [{1, 2}, complex(1, 2), object()])
def test_unsupported_values_raise_instead_of_losing_type(value: object) -> None:
    with pytest.raises(TypeError, match="is not JSON serializable"):
        to_json(value)


def test_arbitrary_object_attributes_are_not_exported() -> None:
    class InternalState:
        def __init__(self) -> None:
            self.secret = "private"

    with pytest.raises(TypeError, match="InternalState is not JSON serializable"):
        to_json(InternalState())


def test_declared_plot_objects_serialize_without_reflection() -> None:
    style = plotting.Style(sid="style")
    plot = plotting.Plot(sid="plot")
    subplot = plotting.SubPlot(plot, row=1, col=2)
    style_info = json.loads(to_json(style))
    subplot_info = json.loads(to_json(subplot))
    assert style_info["sid"] == "style"
    assert style_info["line"] is not None
    assert subplot_info["plot"]["sid"] == "plot"
    assert (subplot_info["row"], subplot_info["col"]) == (1, 2)


@pytest.mark.parametrize("existing", [None, '{"kept": true}'])
def test_a_value_which_cannot_be_encoded_leaves_the_file_as_it_was(
    existing: str | None, tmp_path: Path
) -> None:
    path = tmp_path / "values.json"
    if existing is not None:
        path.write_text(existing, encoding="utf-8")
    with pytest.raises(TypeError, match="set is not JSON serializable"):
        to_json({"a": [0, 1, 2, 3, 4], "b": {1, 2}}, path=path)
    if existing is None:
        assert not path.exists()
    else:
        assert path.read_text(encoding="utf-8") == existing
