"""The nominal values and the fit parameters of a network.

A problem describes the elements of a network in groups: an entry is given
for the network (`net1`), for a layer (`net1.layer1`) or for an array
(`net1.layer1.weight`), and the more specific entry wins. This is how a
problem sets the elements of one layer to `0.0` while the other layers keep
the values of the array file, and how it estimates one layer and freezes the
others.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from sbmlsim.fit.objects import FitParameter
from sbmlsim.sciml.errors import NetworkImportError
from sbmlsim.sciml.network import (
    Network,
    NetworkParameters,
    copy_parameters,
    element_id,
)

logger = logging.getLogger(__name__)

#: separator of the network, the layer and the array in the key of an entry
KEY_SEPARATOR = "."

#: the unit of the elements of a network
ELEMENT_UNIT = "dimensionless"


def covered_arrays(network: Network, key: str) -> tuple[int, list[tuple[str, str]]]:
    """Get the arrays an entry covers.

    Args:
        network: the network.
        key: the network (`net1`), a layer (`net1.layer1`) or an array
            (`net1.layer1.weight`).

    Returns:
        How specific the entry is (0 for the network, 1 for a layer, 2 for an
        array) and the arrays as pairs of layer id and array name. Only the
        arrays which are parameters are covered, i.e. not the running
        statistics of a normalization layer.

    Raises:
        KeyError: if the key does not name the network, one of its layers or
            one of their arrays.
    """
    specs = {
        layer: [name for name, spec in arrays.items() if spec.trainable]
        for layer, arrays in network.array_specs().items()
    }
    if key == network.sid:
        return 0, [(layer, name) for layer, names in specs.items() for name in names]

    prefix = network.sid + KEY_SEPARATOR
    if key.startswith(prefix):
        rest = key[len(prefix) :]
        # the id of a layer may hold the separator, so the layer is tried first
        if rest in specs:
            return 1, [(rest, name) for name in specs[rest]]
        layer, _, name = rest.rpartition(KEY_SEPARATOR)
        if layer in specs and name in specs[layer]:
            return 2, [(layer, name)]
    raise KeyError(
        f"Network '{network.sid}': '{key}' is not the network, a layer or an "
        f"array of it. The layers and their arrays are {specs}"
    )


def resolve_entries[T](
    network: Network, entries: Mapping[str, T]
) -> dict[tuple[str, str], T]:
    """Resolve the entries of a problem to the arrays of a network.

    Args:
        network: the network.
        entries: key of the entry -> value, see `covered_arrays`.

    Returns:
        layer id and array name -> the value of the most specific entry which
        covers the array. An array no entry covers is not part of it.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
    """
    covered = {key: covered_arrays(network, key) for key in entries}
    resolved: dict[tuple[str, str], T] = {}
    # the network first and the arrays last, so the more specific entry wins
    for key in sorted(entries, key=lambda key: covered[key][0]):
        for array in covered[key][1]:
            resolved[array] = entries[key]
    return resolved


def nominal_parameters(
    network: Network, values: Mapping[str, float] | None = None
) -> NetworkParameters:
    """Get the nominal values of the arrays of a network.

    Args:
        network: the network with the values of its array file.
        values: key of the entry -> value of every element the entry covers,
            see `covered_arrays`. An array no entry covers keeps the values of
            the array file.

    Returns:
        The arrays in the PyTorch layout. The network is not changed.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
        NetworkImportError: if a value is not finite, or if an array of a
            layer of the forward pass has values neither in the array file
            nor in `values`.
    """
    parameters = copy_parameters(network.parameters)
    specs = network.array_specs()
    for (layer, name), value in resolve_entries(network, values or {}).items():
        if not np.isfinite(value):
            raise NetworkImportError(
                f"Network '{network.sid}', layer '{layer}': the value "
                f"'{value}' of the array '{name}' is not finite"
            )
        parameters.setdefault(layer, {})[name] = np.full(
            specs[layer][name].shape, float(value)
        )
    network.check_arrays(parameters, complete=True)
    return parameters


def network_fit_parameters(
    network: Network,
    estimate: Mapping[str, bool],
    bounds: Mapping[str, tuple[float, float]],
    values: Mapping[str, float] | None = None,
) -> list[FitParameter]:
    """Create the fit parameters of the estimated elements of a network.

    `estimate`, `bounds` and `values` are given for the network, for a layer
    or for an array, and the more specific entry wins, see `covered_arrays`.

    Args:
        network: the network with the values of its array file.
        estimate: key of the entry -> whether the elements are estimated. An
            element no entry covers is not estimated.
        bounds: key of the entry -> lower and upper bound of the elements. An
            estimated element no entry covers is not bounded.
        values: key of the entry -> nominal value of the elements, which
            replaces the values of the array file.

    Returns:
        One parameter per estimated element, named by the id of the element,
        with the nominal value as start value and the unit `dimensionless`, in the
        order of `Network.parameter_ids`.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
        NetworkImportError: if an estimated element has no nominal value.
        ValueError: if a nominal value is outside of its bounds.
    """
    parameters = nominal_parameters(network, values)
    estimated = resolve_entries(network, estimate)
    bounded = resolve_entries(network, bounds)

    fit_parameters: list[FitParameter] = []
    for layer, name, index in network.parameter_ids().values():
        if not estimated.get((layer, name), False):
            continue
        if name not in parameters.get(layer, {}):
            raise NetworkImportError(
                f"Network '{network.sid}', layer '{layer}': the array "
                f"'{name}' is estimated and has no nominal values"
            )
        lower, upper = bounded.get((layer, name), (-np.inf, np.inf))
        fit_parameters.append(
            FitParameter(
                pid=element_id(network.sid, layer, name, index),
                start_value=float(parameters[layer][name][index]),
                lower_bound=lower,
                upper_bound=upper,
                unit=ELEMENT_UNIT,
            )
        )
    logger.info(
        "Network '%s': %d of %d elements are estimated",
        network.sid,
        len(fit_parameters),
        len(network.parameter_ids()),
    )
    return fit_parameters
