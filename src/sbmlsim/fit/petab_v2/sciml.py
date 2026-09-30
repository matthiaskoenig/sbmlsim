"""The neural networks of a PEtab SciML problem.

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) is the extension
`sciml` of PEtab v2 for hybrid problems, in which a model is combined with
neural networks. This module translates the extension into the objects of
`sbmlsim.sciml`, which know nothing of PEtab:

| PEtab SciML | `sbmlsim` |
| --- | --- |
| a network of the block `neural_networks` with its array files | `Network` |
| `pre_initialization`, the hybridization table and the observables | `Hybridization` and its `NetworkPattern` |
| the rows of the mapping table | the ids of the inputs, outputs and arrays |
| the rows of the parameter table of a network | `FitParameter` per element |
| an array of a condition | an array of the simulation of an experiment |

The module imports `sbmlsim.sciml` and with it `petab_sciml`, which is the
extra `sciml`. `sbmlsim.fit.petab_v2.reader` imports it only for a problem
with the extension.

`petab.v2.Problem.from_yaml` of `petab` 0.9.0 reads the networks of a problem
through PyTorch. `torch` is no dependency of `sbmlsim`, so the tables of a
problem are read by `petab` without the block of the extension, and the
networks, the hybridization tables and the array files are read here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import sympy
from petab.v2 import Problem as PetabProblem
from petab.v2.extensions.sciml import HybridizationTable, SciMLConfig
from petab_sciml.constants import ALL_CONDITION_IDS, ARRAY

from sbmlsim.fit.objects import EXTERNAL_PREFIX, FitParameter
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.mathml import expression_to_formula, formula_symbols
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    input_shapes,
    output_shapes,
)
from sbmlsim.sciml.network import (
    Network,
    input_id,
    load_array_data,
    output_id,
)
from sbmlsim.sciml.parameters import (
    ELEMENT_UNIT,
    network_fit_parameters,
    nominal_parameters,
)

logger = logging.getLogger(__name__)

#: the format of a network which is read
YAML_FORMAT = "yaml"

#: a `modelEntityId` of the mapping table which names a part of a network
ENTITY = re.compile(
    r"(?P<network>[^.\[\]\s]+)\.(?P<kind>inputs|outputs|parameters)(?P<rest>.*)"
)

#: the indices of an input or output, e.g. `[0][1]`
INDICES = re.compile(r"(?:\[\d+\])+")

#: the layer and the array of the parameters, e.g. `[layer1].weight`
LAYER = re.compile(r"\[(?P<layer>[^\]]+)\](?:\.(?P<array>[^.\[\]\s]+))?")

#: the scales of the column `parameterScale`, which PEtab v2 dropped and the
#: problems of PEtab SciML carry
SCALES: dict[str, ParameterScaleType] = {
    "lin": ParameterScaleType.LINEAR,
    "log": ParameterScaleType.LOG,
    "log10": ParameterScaleType.LOG10,
}


class SciMLProblemError(ValueError):
    """A PEtab SciML problem which cannot be read.

    Attributes:
        gap: id of the gap the problem runs into, see
            `sbmlsim.fit.petab_v2.gaps`, `None` for a problem which is not
            valid.
    """

    def __init__(self, message: str, gap: str | None = None) -> None:
        """Initialize the error.

        Args:
            message: what cannot be read.
            gap: id of the gap the problem runs into.
        """
        self.gap = gap
        super().__init__(message if gap is None else f"{message} (gap '{gap}')")


@dataclass(frozen=True)
class NetworkEntity:
    """A part of a network a row of the mapping table names.

    Attributes:
        petab_id: the `petabEntityId` of the row.
        network: id of the network.
        kind: `inputs`, `outputs` or `parameters`.
        k: position of the input or output, `None` for parameters.
        index: index of the element, `None` for an input which is an array
            and for parameters.
        key: the key of the parameters, e.g. `net1.layer1.weight`, see
            `sbmlsim.sciml.parameters`, `None` for an input or output.
    """

    petab_id: str
    network: str
    kind: str
    k: int | None = None
    index: tuple[int, ...] | None = None
    key: str | None = None


def parse_entity(petab_id: str, model_entity_id: str) -> NetworkEntity | None:
    """Read a row of the mapping table which names a part of a network.

    Args:
        petab_id: the `petabEntityId` of the row.
        model_entity_id: the `modelEntityId`, e.g. `net1.inputs[0][1]`,
            `net1.inputs[0]`, `net1.outputs[0][0]`, `net1.parameters`,
            `net1.parameters[layer1]` or `net1.parameters[layer1].weight`.

    Returns:
        The part of the network, `None` for a row which names an entity of a
        model.

    Raises:
        SciMLProblemError: if the row names a part of a network and is not
            one of the forms.
    """
    match = ENTITY.fullmatch(model_entity_id.strip())
    if match is None:
        return None
    network, kind, rest = match.group("network", "kind", "rest")
    error = SciMLProblemError(
        f"The mapping of '{petab_id}' to '{model_entity_id}' is not a part of a "
        f"network: an input is '{network}.inputs[0][1]' or '{network}.inputs[0]', "
        f"an output '{network}.outputs[0][0]' and the parameters "
        f"'{network}.parameters', '{network}.parameters[layer1]' or "
        f"'{network}.parameters[layer1].weight'"
    )
    if kind == "parameters":
        if not rest:
            return NetworkEntity(petab_id, network, kind, key=network)
        layer = LAYER.fullmatch(rest)
        if layer is None:
            raise error
        key = f"{network}.{layer.group('layer')}"
        if layer.group("array"):
            key = f"{key}.{layer.group('array')}"
        return NetworkEntity(petab_id, network, kind, key=key)

    if INDICES.fullmatch(rest) is None:
        raise error
    numbers = [int(number) for number in re.findall(r"\d+", rest)]
    index = tuple(numbers[1:]) if len(numbers) > 1 else None
    if kind == "outputs" and index is None:
        raise error
    return NetworkEntity(petab_id, network, kind, k=numbers[0], index=index)


def read_sciml_config(extensions: Mapping[str, Any] | None) -> SciMLConfig | None:
    """Get the block of the extension `sciml` of a problem.

    Args:
        extensions: the blocks of the extensions of the YAML of the problem.

    Returns:
        The configuration, `None` for a problem without the extension.

    Raises:
        SciMLProblemError: if the block is not a block of the extension.
    """
    block = (extensions or {}).get("sciml")
    if block is None:
        return None
    if isinstance(block, SciMLConfig):
        return block
    try:
        return SciMLConfig(**dict(block))
    except (TypeError, ValueError) as err:
        raise SciMLProblemError(
            f"The block 'sciml' of the problem is not valid: {err}"
        ) from err


class SciMLReader:
    """Read the networks of a PEtab SciML problem.

    Attributes:
        networks: the networks by their id, with the nominal values of the
            problem.
        entities: the parts of the networks the mapping table names, by their
            `petabEntityId`.
    """

    def __init__(
        self,
        petab_problem: PetabProblem,
        config: SciMLConfig,
        base_path: Path,
        simulations: Mapping[str, list[str]],
    ) -> None:
        """Read the networks, the hybridization tables and the array files.

        Args:
            petab_problem: the problem, read without the block of the
                extension.
            config: the block of the extension.
            base_path: directory the files of the problem are relative to.
            simulations: id of the simulation of every experiment -> the ids
                of the conditions the simulation starts with.

        Raises:
            SciMLProblemError: if the problem has not one model, if a network
                is not in the format `YAML`, if a row of the mapping table
                names a network the problem does not have, if a row of the
                hybridization table cannot be read, or if a parameter of a
                network has a prior.
            NetworkImportError: if a network or its arrays cannot be read.
        """
        self.petab_problem = petab_problem
        self.config = config
        self.base_path = Path(base_path)
        self.simulations = {key: list(ids) for key, ids in simulations.items()}
        if len(petab_problem.models) != 1:
            raise SciMLProblemError(
                f"A problem with networks has one model, but the problem has "
                f"the models {[m.model_id for m in petab_problem.models]}"
            )
        self.model = petab_problem.models[0]
        self.model_id: str = self.model.model_id
        self._parameters = {p.id: p for p in petab_problem.parameters}

        self._arrays = [
            load_array_data(self.base_path / str(path)) for path in config.array_files
        ]
        self.entities: dict[str, NetworkEntity] = {}
        for mapping in petab_problem.mappings:
            entity = parse_entity(mapping.petab_id, mapping.model_id or "")
            if entity is None:
                continue
            if entity.network not in (config.neural_networks or {}):
                raise SciMLProblemError(
                    f"The mapping of '{entity.petab_id}' names the network "
                    f"'{entity.network}', the networks of the problem are "
                    f"{sorted(config.neural_networks or {})}"
                )
            self.entities[entity.petab_id] = entity

        self._hybridization: dict[str, sympy.Basic] = {}
        for path in config.hybridization_files:
            for row in HybridizationTable.from_tsv(str(path), self.base_path).elements:
                if row.target_id in self._hybridization:
                    raise SciMLProblemError(
                        f"The hybridization table assigns '{row.target_id}' twice"
                    )
                if row.target_value is None:
                    raise SciMLProblemError(
                        f"The hybridization table assigns '{row.target_id}' no value"
                    )
                self._hybridization[row.target_id] = row.target_value

        self.networks: dict[str, Network] = {
            sid: self._network(sid) for sid in (config.neural_networks or {})
        }
        #: the inputs of every network by their id
        self.inputs: dict[str, dict[str, NetworkInput]] = {
            sid: {
                input_id(sid, entity.k or 0, entity.index): self._input(entity)
                for entity in self._entities(sid, "inputs")
            }
            for sid in self.networks
        }

    # --- THE NETWORKS ---

    def _entities(self, network: str, kind: str) -> list[NetworkEntity]:
        """Get the parts of a kind of a network, in the order of the table."""
        return [
            entity
            for entity in self.entities.values()
            if entity.network == network and entity.kind == kind
        ]

    def _network(self, sid: str) -> Network:
        """Read a network with the nominal values of the problem.

        Args:
            sid: id of the network.

        Returns:
            The network with the values of its array file, replaced by the
            values of the rows of the parameter table which are numbers.

        Raises:
            SciMLProblemError: if the network is not in the format `YAML`,
                or if a parameter of the network has a prior.
            NetworkImportError: if the network or its arrays cannot be read.
        """
        network_config = (self.config.neural_networks or {})[sid]
        if network_config.format.lower() != YAML_FORMAT:
            raise SciMLProblemError(
                f"Network '{sid}': the format '{network_config.format}' is not "
                f"read, only the format 'YAML'",
                gap="sciml-model-format",
            )
        network = Network.from_files(
            self.base_path / str(network_config.location), sid=sid
        )
        for path, data in zip(self.config.array_files, self._arrays, strict=True):
            if sid in data.parameters:
                network = replace(
                    network, parameters=network.read_arrays(self.base_path / str(path))
                )
        values: dict[str, float] = {}
        for entity in self._entities(sid, "parameters"):
            parameter = self._parameters.get(entity.petab_id)
            if parameter is None or entity.key is None:
                continue
            if getattr(parameter, "prior_distribution", None) is not None:
                raise SciMLProblemError(
                    f"Network '{sid}': the parameter '{parameter.id}' has the "
                    f"prior '{parameter.prior_distribution}', priors of the "
                    f"parameters of a network are not read",
                    gap="sciml-priors",
                )
            value = parameter.nominal_value
            if value is not None and value != ARRAY:
                values[entity.key] = float(value)
        return replace(network, parameters=nominal_parameters(network, values))

    def is_pre_initialization(self, sid: str) -> bool:
        """Check whether a network runs before the simulation."""
        return bool((self.config.neural_networks or {})[sid].pre_initialization)

    @property
    def parameter_ids(self) -> set[str]:
        """Get the ids of the parameter table which are arrays of networks."""
        return {e.petab_id for e in self.entities.values() if e.kind == "parameters"}

    @property
    def input_ids(self) -> set[str]:
        """Get the ids which are inputs of networks."""
        return {e.petab_id for e in self.entities.values() if e.kind == "inputs"}

    @property
    def condition_ids(self) -> set[str]:
        """Get the conditions the array files have arrays for."""
        return {
            condition
            for data in self._arrays
            for arrays in data.inputs.values()
            for condition in arrays
        } - {ALL_CONDITION_IDS}

    # --- THE HYBRIDIZATIONS ---

    def _conditions(self, by_condition: Mapping[str, Any], what: str) -> dict[str, Any]:
        """Translate the values of conditions into values of simulations.

        Args:
            by_condition: id of the condition -> value, `ALL_CONDITION_IDS`
                for the value of every condition.
            what: what the values are, for the message.

        Returns:
            id of the simulation -> value, `ALL_CONDITIONS` for the value of
            every simulation.

        Raises:
            SciMLProblemError: if a simulation starts with two conditions
                which have a value.
        """
        values: dict[str, Any] = {}
        if ALL_CONDITION_IDS in by_condition:
            values[ALL_CONDITIONS] = by_condition[ALL_CONDITION_IDS]
        for simulation, conditions in self.simulations.items():
            hits = [c for c in conditions if c in by_condition]
            if len(hits) > 1:
                raise SciMLProblemError(
                    f"The experiment '{simulation}' starts with the conditions "
                    f"{hits}, which all set {what}"
                )
            if hits:
                values[simulation] = by_condition[hits[0]]
        return values

    def _input(self, entity: NetworkEntity) -> NetworkInput:
        """Get an input of a network from the tables of the problem.

        The value of an input is, in this order, the row of the hybridization
        table, the changes of the conditions, and the parameter of the
        parameter table of its id.

        Args:
            entity: the input.

        Returns:
            The input.

        Raises:
            SciMLProblemError: if the input has no value, or if an array has
                no values in the array files.
        """
        petab_id = entity.petab_id
        value = self._hybridization.get(petab_id)
        if value is not None and str(value) == ARRAY:
            arrays = [
                data.inputs[petab_id]
                for data in self._arrays
                if petab_id in data.inputs
            ]
            if len(arrays) != 1:
                raise SciMLProblemError(
                    f"Network '{entity.network}': the input '{petab_id}' is an "
                    f"array, and {len(arrays)} array files have values for it"
                )
            return NetworkInput(
                arrays=self._conditions(
                    {
                        condition: np.asarray(array, dtype=float)
                        for condition, array in arrays[0].items()
                    },
                    f"the input '{petab_id}'",
                )
            )
        if value is not None:
            return NetworkInput(formula=expression_to_formula(value))

        by_condition = {
            condition.id: expression_to_formula(change.target_value)
            for condition in self.petab_problem.conditions
            for change in condition.changes
            if change.target_id == petab_id
        }
        if by_condition:
            formulas = self._conditions(by_condition, f"the input '{petab_id}'")
            if petab_id in self._parameters:
                formulas.setdefault(ALL_CONDITIONS, petab_id)
            return NetworkInput(formulas=formulas)
        if petab_id in self._parameters:
            return NetworkInput(formula=petab_id)
        raise SciMLProblemError(
            f"Network '{entity.network}': the input '{petab_id}' has no value, it "
            f"is neither assigned by the hybridization table or a condition nor "
            f"a parameter of the parameter table"
        )

    def _constants(self, inputs: Mapping[str, NetworkInput]) -> dict[str, float]:
        """Get the values of the symbols of inputs which are not in the model.

        Args:
            inputs: the inputs of a network.

        Returns:
            id -> nominal value of the parameters of the parameter table
            which the formulas use, which are not entities of the model and
            which are not estimated.
        """
        constants: dict[str, float] = {}
        for symbol in sorted(_symbols(inputs)):
            parameter = self._parameters.get(symbol)
            if parameter is None or self.model.has_entity_with_id(symbol):
                continue
            if parameter.estimate and not self.is_compiled_symbol(symbol):
                continue
            constants[symbol] = _nominal(parameter)
        return constants

    def is_compiled_symbol(self, symbol: str) -> bool:
        """Check whether a symbol is used by a network which is compiled.

        Such a symbol is a parameter of the model with the networks, so a
        parameter of the fit of its id is an entity of the model.

        Args:
            symbol: id of a parameter of the parameter table.
        """
        return any(
            symbol in _symbols(inputs)
            for sid, inputs in self.inputs.items()
            if not self.is_pre_initialization(sid)
        )

    def hybridizations(self) -> list[Hybridization]:
        """Get the hybridizations of the networks of the problem.

        The pattern of a network is `PRE_INITIALIZATION` when the problem
        says so. Otherwise an output which an observable uses is
        `OBSERVABLE` and an output which the hybridization table assigns to
        an entity of the model is `RHS`, and a network with outputs of both
        kinds has two hybridizations.

        Returns:
            The hybridizations, in the order of the networks of the problem.

        Raises:
            SciMLProblemError: if an input has no value, if an output is
                assigned to two targets or to a target and an observable, or
                if a network has no output which is used.
            NetworkHybridizationError: if the inputs and outputs do not fit
                a network.
        """
        observed = {
            str(symbol)
            for observable in self.petab_problem.observables
            for symbol in observable.formula.free_symbols
        }
        targets: dict[str, str] = {}
        for target, value in self._hybridization.items():
            output = self.entities.get(str(value))
            if output is None or output.kind != "outputs":
                continue
            if output.petab_id in targets:
                raise SciMLProblemError(
                    f"Network '{output.network}': the hybridization table assigns "
                    f"the output '{output.petab_id}' to '{targets[output.petab_id]}' "
                    f"and to '{target}'"
                )
            targets[output.petab_id] = target
        for target, value in self._hybridization.items():
            if target in self.entities or str(value) in targets:
                continue
            raise SciMLProblemError(
                f"The hybridization table assigns '{target}' the value '{value}'. "
                f"A row assigns a value to an input of a network or an output of "
                f"a network to an entity of the model"
            )

        hybridizations: list[Hybridization] = []
        for sid, network in self.networks.items():
            inputs = self.inputs[sid]
            outputs: dict[NetworkPattern, dict[str, str]] = {}
            shapes = output_shapes(network, input_shapes(network, inputs))
            for entity in self._entities(sid, "outputs"):
                key = output_id(sid, entity.k or 0, _index(entity, shapes))
                if entity.petab_id in targets:
                    pattern = (
                        NetworkPattern.PRE_INITIALIZATION
                        if self.is_pre_initialization(sid)
                        else NetworkPattern.RHS
                    )
                    target = self._target(targets[entity.petab_id], pattern)
                    outputs.setdefault(pattern, {})[key] = target
                if entity.petab_id in observed:
                    if self.is_pre_initialization(sid):
                        raise SciMLProblemError(
                            f"Network '{sid}': the output '{entity.petab_id}' is "
                            f"used by an observable, but the network runs "
                            f"before the simulation"
                        )
                    if entity.petab_id in targets:
                        raise SciMLProblemError(
                            f"Network '{sid}': the output '{entity.petab_id}' is "
                            f"used by an observable and assigned to "
                            f"'{targets[entity.petab_id]}' by the hybridization "
                            f"table. An output sets an entity of the model or "
                            f"is a symbol of an observable"
                        )
                    outputs.setdefault(NetworkPattern.OBSERVABLE, {})[key] = (
                        entity.petab_id
                    )
            if not outputs:
                raise SciMLProblemError(
                    f"Network '{sid}': no output of the network is used, neither "
                    f"by the hybridization table nor by an observable"
                )
            frozen = set(network.parameter_ids()) - {
                p.pid for p in self.network_parameters(sid)
            }
            hybridizations.extend(
                Hybridization(
                    network=network,
                    pattern=pattern,
                    model=self.model_id,
                    inputs=inputs,
                    outputs=pattern_outputs,
                    frozen=frozen,
                    constants=self._constants(inputs),
                )
                for pattern, pattern_outputs in outputs.items()
            )
        return hybridizations

    def _target(self, target: str, pattern: NetworkPattern) -> str:
        """Get the target of an output as `sbmlsim` names it.

        A network which runs before the simulation sets the initial value of
        a species as the model means it, i.e. the concentration of a species
        which is not amount based, which is the selection `[S]`.

        Args:
            target: the `targetId` of the hybridization table.
            pattern: the pattern of the hybridization.

        Returns:
            The entity, or the selection of the concentration of a species.
        """
        sbml_model = getattr(self.model, "sbml_model", None)
        if pattern is not NetworkPattern.PRE_INITIALIZATION or sbml_model is None:
            return target
        species = sbml_model.getSpecies(target)
        if species is not None and not species.getHasOnlySubstanceUnits():
            return f"[{target}]"
        return target

    # --- THE PARAMETERS ---

    def network_parameters(self, sid: str) -> list[FitParameter]:
        """Get the parameters of the fit which are elements of a network.

        Args:
            sid: id of the network.

        Returns:
            One parameter per estimated element, see
            `sbmlsim.sciml.parameters.network_fit_parameters`. The elements of
            a network which runs before the simulation are not entities of
            the model.
        """
        estimate: dict[str, bool] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for entity in self._entities(sid, "parameters"):
            parameter = self._parameters.get(entity.petab_id)
            if parameter is None or entity.key is None:
                continue
            estimate[entity.key] = bool(parameter.estimate)
            bounds[entity.key] = (
                -np.inf if parameter.lb is None else float(parameter.lb),
                np.inf if parameter.ub is None else float(parameter.ub),
            )
        return network_fit_parameters(
            self.networks[sid],
            estimate=estimate,
            bounds=bounds,
            external=self.is_pre_initialization(sid),
        )

    def fit_parameters(
        self, unit_of: Callable[[str], str | None]
    ) -> list[FitParameter]:
        """Get the parameters of the fit which the networks add.

        These are the estimated elements of the networks and the estimated
        parameters of the parameter table which are inputs of networks and
        not entities of the model.

        Args:
            unit_of: gets the unit of an entity of the model.

        Returns:
            The parameters, the inputs first and the elements in the order of
            the networks.
        """
        parameters: list[FitParameter] = []
        symbols = {
            symbol
            for hybridization in self.hybridizations()
            for symbol in _symbols(hybridization.inputs)
        }
        for symbol in sorted(symbols):
            parameter = self._parameters.get(symbol)
            if (
                parameter is None
                or not parameter.estimate
                or self.model.has_entity_with_id(symbol)
            ):
                continue
            parameters.append(
                FitParameter(
                    pid=symbol,
                    start_value=_nominal(parameter),
                    lower_bound=-np.inf
                    if parameter.lb is None
                    else float(parameter.lb),
                    upper_bound=np.inf if parameter.ub is None else float(parameter.ub),
                    unit=unit_of(symbol) or ELEMENT_UNIT,
                    target=None
                    if self.is_compiled_symbol(symbol)
                    else f"{EXTERNAL_PREFIX}{symbol}",
                    scale=parameter_scale(parameter) or ParameterScaleType.LINEAR,
                )
            )
        for sid in self.networks:
            parameters.extend(self.network_parameters(sid))
        return parameters


def _symbols(inputs: Mapping[str, NetworkInput]) -> set[str]:
    """Get the symbols of the formulas of inputs."""
    return {
        symbol
        for network_input in inputs.values()
        for formula in network_input.all_formulas()
        for symbol in formula_symbols(formula)
    }


def _index(entity: NetworkEntity, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Get the index of an output of PEtab SciML in the output of the network.

    The index of PEtab SciML names an element of the output of one sample,
    the output of the network may have leading axes of the length one, i.e.
    the axis of the batch: `outputs[0][0]` of an output of the shape `(1, 1)`
    is the element `(0, 0)`.

    Args:
        entity: the output.
        shapes: the shapes of the outputs of the network.

    Returns:
        The index with a zero for every leading axis of the length one which
        it does not name. An index which does not fit is returned as it is,
        the hybridization names it.
    """
    index = entity.index or ()
    k = entity.k or 0
    if k >= len(shapes):
        return index
    missing = len(shapes[k]) - len(index)
    if missing > 0 and all(n == 1 for n in shapes[k][:missing]):
        return (0,) * missing + index
    return index


def _nominal(parameter: Any) -> float:
    """Get the nominal value of a parameter which is the value of an input.

    Args:
        parameter: parameter of the parameter table.

    Returns:
        The nominal value.

    Raises:
        SciMLProblemError: if the nominal value is not a finite number.
    """
    value = parameter.nominal_value
    if isinstance(value, int | float) and np.isfinite(value):
        return float(value)
    raise SciMLProblemError(
        f"The parameter '{parameter.id}' is the value of an input of a network "
        f"and has the nominal value '{value}', which is not a finite number"
    )


def parameter_scale(parameter: Any) -> ParameterScaleType | None:
    """Get the scale of a parameter of the parameter table.

    PEtab v2 has no scale of a parameter, it is a property of the
    optimization. The problems of PEtab SciML carry the column
    `parameterScale` of PEtab v1, which is read when it is there.

    Args:
        parameter: parameter of the parameter table.

    Returns:
        The scale, `None` for a parameter without one.

    Raises:
        SciMLProblemError: if the scale is not `lin`, `log` or `log10`.
    """
    extra = getattr(parameter, "model_extra", None) or {}
    value = extra.get("parameterScale")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if str(value) not in SCALES:
        raise SciMLProblemError(
            f"The parameter '{parameter.id}' has the scale '{value}', which is "
            f"not one of {sorted(SCALES)}"
        )
    return SCALES[str(value)]
