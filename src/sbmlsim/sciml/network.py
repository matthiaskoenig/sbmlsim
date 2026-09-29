"""The architecture and the arrays of a neural network.

`Network` holds the architecture of a network as the `NNModel` of
`petab_sciml`, i.e. the content of the NN YAML, and the arrays of its layers
in the PyTorch layout. `Network.forward` evaluates it with numpy.

Every element of an array has an id, `<net>__<layer>__<array>__<index>` with
the PyTorch index of the element and `_` between the axes, e.g.
`net1__layer1__weight__0_1`. The id is a valid SBML `SId`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from petab_sciml import ArrayData, ArrayDataStandard, NNModel, NNModelStandard

from sbmlsim.sciml.backend import ALL_BACKENDS, BackendKind, NumpyBackend
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
from sbmlsim.sciml.interpreter import (
    CALL_FUNCTION,
    CALL_METHOD,
    CALL_MODULE,
    evaluate,
)
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS, ArraySpec

logger = logging.getLogger(__name__)

#: the arrays of a network: layer id -> array name -> values
NetworkParameters = dict[str, dict[str, np.ndarray]]

#: separator of the parts of an id
ID_SEPARATOR = "__"

#: separator of the axes of the index of an id
INDEX_SEPARATOR = "_"


#: the characters of an id which are not part of an SBML `SId`
_NOT_SID = re.compile(r"[^A-Za-z0-9_]")


def element_id(network: str, layer: str, array: str, index: tuple[int, ...]) -> str:
    """Get the id of an element of an array.

    Args:
        network: id of the network.
        layer: id of the layer.
        array: name of the array, e.g. `weight`.
        index: the PyTorch index of the element.

    Returns:
        The id, e.g. `net1__layer1__weight__0_1`. A character which is not
        part of an SBML `SId` is replaced by `_`, e.g. the dot in the id of a
        layer of a nested module.
    """
    sid = ID_SEPARATOR.join(
        [network, layer, array, INDEX_SEPARATOR.join(str(i) for i in index)]
    )
    return _NOT_SID.sub("_", sid)


def copy_parameters(
    parameters: Mapping[str, Mapping[str, np.ndarray]],
) -> NetworkParameters:
    """Copy the arrays of a network.

    Args:
        parameters: the arrays, layer id -> array name -> values.

    Returns:
        A copy which shares no array with the original.
    """
    return {
        layer: {name: np.array(array, dtype=float) for name, array in arrays.items()}
        for layer, arrays in parameters.items()
    }


def load_array_data(path: Path) -> ArrayData:
    """Read an array file.

    Args:
        path: the HDF5 file.

    Returns:
        The arrays of the file.

    Raises:
        NetworkImportError: if the file does not exist or is not an array
            file.
    """
    if not path.is_file():
        raise NetworkImportError(f"The array file '{path}' does not exist")
    data = ArrayDataStandard.load_data(str(path))
    if not isinstance(data, ArrayData):
        raise NetworkImportError(f"'{path}' is not an array file")
    return data


@dataclass
class Network:
    """The architecture and the arrays of one network.

    Attributes:
        sid: id of the network.
        model: the architecture, i.e. the content of the NN YAML.
        parameters: the nominal values of the arrays in the PyTorch layout,
            layer id -> array name -> values.
    """

    sid: str
    model: NNModel
    parameters: NetworkParameters = field(default_factory=dict)

    @classmethod
    def from_files(
        cls, yaml_path: Path, array_path: Path | None = None, sid: str | None = None
    ) -> Network:
        """Read a network from its NN YAML and its array file.

        The arrays of a file with `metadata/pytorch_format` false are stored
        in the column major layout, i.e. with the axes in the reverse order,
        and are permuted into the PyTorch layout.

        Args:
            yaml_path: the NN YAML.
            array_path: the HDF5 file with the arrays of the network. Without
                it the network has no values, which a problem sets.
            sid: id of the network, the `nn_model_id` of the YAML when `None`.
                The arrays are read from the group of this id.

        Returns:
            The network.

        Raises:
            NetworkImportError: if a file does not exist, if the array file
                has no arrays for the network, or if an array does not belong
                to a layer or does not have the shape of the layer.
            UnsupportedLayerError: if the network has a layer without an
                implementation.
        """
        if not yaml_path.is_file():
            raise NetworkImportError(f"The NN YAML '{yaml_path}' does not exist")
        model = NNModelStandard.load_data(str(yaml_path))
        if not isinstance(model, NNModel):
            raise NetworkImportError(f"'{yaml_path}' is not a NN YAML")
        if sid is not None:
            model = model.model_copy(update={"nn_model_id": sid})
        network = cls(sid=model.nn_model_id, model=model)
        if array_path is not None:
            network.parameters = network.read_arrays(array_path)
        network.check_arrays(network.parameters, complete=False)
        return network

    def read_arrays(self, array_path: Path) -> NetworkParameters:
        """Read the arrays of the network from an array file.

        An empty array is an array the file does not provide, which is how a
        file leaves out the arrays a problem sets.

        Args:
            array_path: the HDF5 file.

        Returns:
            The arrays in the PyTorch layout.

        Raises:
            NetworkImportError: if the file does not exist or has no arrays
                for the network.
        """
        data = load_array_data(array_path)
        if self.sid not in data.parameters:
            raise NetworkImportError(
                f"Network '{self.sid}': the array file '{array_path}' has no "
                f"parameters of the network, it has {sorted(data.parameters)}"
            )
        pytorch_format = data.metadata.pytorch_format
        if not pytorch_format:
            logger.info(
                "Network '%s': the arrays of '%s' are permuted into the PyTorch layout",
                self.sid,
                array_path,
            )
        parameters: NetworkParameters = {}
        for layer, arrays in data.parameters[self.sid].items():
            for name, values in arrays.items():
                array = np.asarray(values, dtype=float)
                if array.size == 0:
                    continue
                if not pytorch_format:
                    array = np.ascontiguousarray(array.T)
                parameters.setdefault(layer, {})[name] = array
        return parameters

    def array_specs(self) -> dict[str, dict[str, ArraySpec]]:
        """Get the arrays of every layer of the network.

        Returns:
            The arrays of the layers, layer id -> array name -> shape and
            kind, in the order of the layers.

        Raises:
            UnsupportedLayerError: if a layer has no implementation.
        """
        specs: dict[str, dict[str, ArraySpec]] = {}
        for layer in self.model.layers:
            layer_type = LAYERS.get(layer.layer_type)
            if layer_type is None:
                raise UnsupportedLayerError(
                    self.sid,
                    layer.layer_id,
                    layer.layer_type,
                    "the layer is not implemented",
                )
            specs[layer.layer_id] = layer_type.arrays(layer.args or {})
        return specs

    def used_layers(self) -> list[str]:
        """Get the ids of the layers the forward pass calls, in its order."""
        used: list[str] = []
        for node in self.model.forward:
            if node.op == CALL_MODULE and node.target not in used:
                used.append(node.target)
        return used

    def backends(self) -> frozenset[BackendKind]:
        """Get the backends which evaluate every node of the forward pass.

        Returns:
            The backends all layers and functions of the forward pass support.

        Raises:
            UnsupportedLayerError: if a node has no implementation.
        """
        layers = {layer.layer_id: layer for layer in self.model.layers}
        backends = ALL_BACKENDS
        for node in self.model.forward:
            if node.op == CALL_MODULE:
                name = layers[node.target].layer_type
                supported = LAYERS.get(name)
            elif node.op in (CALL_FUNCTION, CALL_METHOD):
                name = node.target
                supported = FUNCTIONS.get(name)
            else:
                continue
            if supported is None:
                raise UnsupportedLayerError(
                    self.sid, node.name, name, "it is not implemented"
                )
            backends = backends & supported.backends
        return backends

    def check_arrays(
        self, parameters: Mapping[str, Mapping[str, np.ndarray]], complete: bool = True
    ) -> None:
        """Check arrays against the architecture.

        Args:
            parameters: the arrays, layer id -> array name -> values.
            complete: whether every required array of a layer of the forward
                pass must be there.

        Raises:
            NetworkImportError: if an array does not belong to a layer of the
                network, does not have the shape of the layer, holds a value
                which is not finite or, with `complete`, is missing.
        """
        specs = self.array_specs()
        for layer, arrays in parameters.items():
            if layer not in specs:
                raise NetworkImportError(
                    f"Network '{self.sid}': arrays are given for the layer "
                    f"'{layer}', the layers are {sorted(specs)}"
                )
            for name, array in arrays.items():
                if name not in specs[layer]:
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' is not an array of the layer, the arrays "
                        f"are {sorted(specs[layer])}"
                    )
                shape = specs[layer][name].shape
                if np.shape(array) != shape:
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' has the shape {np.shape(array)}, the "
                        f"layer needs {shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' holds values which are not finite"
                    )
        if not complete:
            return
        for layer in self.used_layers():
            for name, spec in specs[layer].items():
                if spec.required and name not in parameters.get(layer, {}):
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' has no values. A network is not "
                        f"initialized with random values, the array file or "
                        f"the problem must provide them"
                    )

    def forward(
        self, *inputs: np.ndarray, parameters: NetworkParameters | None = None
    ) -> tuple[np.ndarray, ...]:
        """Evaluate the network with numpy, in evaluation mode.

        Args:
            *inputs: the inputs in the PyTorch layout, one per input of the
                forward pass.
            parameters: the arrays the network is evaluated with, the nominal
                values when `None`.

        Returns:
            The outputs of the network.

        Raises:
            NetworkImportError: if an array of a layer is missing or has the
                wrong shape.
            UnsupportedLayerError: if a layer or function has no
                implementation.
            ValueError: if the inputs do not fit the network.
        """
        arrays = self.parameters if parameters is None else parameters
        self.check_arrays(arrays)
        return evaluate(self.model, arrays, inputs, NumpyBackend())

    def parameter_ids(self) -> dict[str, tuple[str, str, tuple[int, ...]]]:
        """Get the ids of the elements of the arrays which are parameters.

        The ids follow from the architecture, so they exist for an array
        without values as well. The running statistics of a normalization
        layer are not parameters.

        Returns:
            id of the element -> layer id, array name and PyTorch index, in
            the order of the layers, the arrays and the row major order of the
            elements.

        Raises:
            NetworkImportError: if two elements have the same id, which the
                ids of two layers such as `block.0` and `block_0` cause.
        """
        ids: dict[str, tuple[str, str, tuple[int, ...]]] = {}
        for layer, specs in self.array_specs().items():
            for name, spec in specs.items():
                if not spec.trainable:
                    continue
                for index in np.ndindex(spec.shape):
                    sid = element_id(self.sid, layer, name, index)
                    if sid in ids:
                        raise NetworkImportError(
                            f"Network '{self.sid}': the elements {ids[sid]} "
                            f"and {(layer, name, index)} have the id '{sid}'"
                        )
                    ids[sid] = (layer, name, index)
        return ids

    def with_values(self, values: Mapping[str, float]) -> NetworkParameters:
        """Get the arrays with the values of some elements replaced.

        Args:
            values: id of the element -> value.

        Returns:
            A copy of the nominal values with the elements replaced. The
            network is not changed.

        Raises:
            KeyError: if an id is not the id of an element of the network.
            NetworkImportError: if an element of an array without nominal
                values is set.
        """
        ids = self.parameter_ids()
        parameters = copy_parameters(self.parameters)
        for sid, value in values.items():
            if sid not in ids:
                raise KeyError(
                    f"Network '{self.sid}': '{sid}' is not the id of an element"
                )
            layer, name, index = ids[sid]
            if name not in parameters.get(layer, {}):
                raise NetworkImportError(
                    f"Network '{self.sid}', layer '{layer}': the array "
                    f"'{name}' has no values, '{sid}' cannot be set"
                )
            parameters[layer][name][index] = value
        return parameters
