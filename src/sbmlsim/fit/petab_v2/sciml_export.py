"""Write the neural networks of a hybrid problem as PEtab SciML.

The inverse of `sbmlsim.fit.petab_v2.sciml`: the hybridizations of an
`OptimizationProblem` become the block `sciml` of the problem, the
hybridization table, the rows of the mapping table and of the parameter table
which name the networks, the NN YAML of every network and one array file per
network with its arrays and the arrays of its inputs.

| `sbmlsim` | PEtab SciML |
| --- | --- |
| `Network` | the NN YAML and the arrays of the array file |
| `Hybridization.pattern` | `pre_initialization` of the network, and the observables |
| an input which is a formula | a row of the hybridization table |
| an input which is a parameter of its own | the parameter is the `petabEntityId` of the input |
| an input with a formula per condition | a change of the condition of every experiment |
| the formula of the other conditions | a change of the conditions without a formula of their own, and the block `sbmlsim` |
| an input which is arrays | `array` in the hybridization table, the arrays in the array file by condition |
| an output of `RHS` or `PRE_INITIALIZATION` | a row of the hybridization table which assigns the target |
| an output of `OBSERVABLE` | the symbol of the observable formula, mapped to the output |
| `frozen` and the bounds of the elements | the most general rows of the parameter table |
| `constants` | rows of the parameter table which are not estimated |

The linter of `petab` counts a parameter of the parameter table which the
model does not have, i.e. a constant of a hybridization or an estimated
parameter which is no entity of the model, as used where the mapping table
names it or a condition uses it, not where the hybridization table uses it.
An input whose formula is such a parameter, which no other formula uses, is
therefore the parameter (its `petabEntityId`), and an input whose formula
uses such a parameter otherwise is a change of the condition of every
experiment. The formula of an input for the conditions without a formula of
their own is written into the conditions of those experiments and into the
block `sbmlsim`, from which the reader gives the input back as it was.

The module imports `sbmlsim.sciml` and with it `petab_sciml`, which is the
extra `sciml`. `sbmlsim.fit.petab_v2.export` imports it only for a problem
with hybridizations.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import petab.v2 as petab_v2
import sympy
from petab.v2.extensions.sciml import Hybridization as HybridizationRow
from petab.v2.extensions.sciml import (
    HybridizationTable,
    NeuralNetConfig,
    SciMLConfig,
)
from petab.v2.math import petab_math_str
from petab_sciml import ArrayData, ArrayDataStandard, Metadata, NNModelStandard
from petab_sciml.constants import ALL_CONDITION_IDS, ARRAY

from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.export import period_condition_id
from sbmlsim.fit.petab_v2.sciml import YAML_FORMAT
from sbmlsim.mathml import formula_expression, formula_symbols
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    entity_of,
)
from sbmlsim.sciml.network import Network, parse_io_id
from sbmlsim.sciml.parameters import ELEMENT_UNIT

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)

#: the file of the hybridization table
HYBRIDIZATION_FILE = "hybridization.tsv"

#: the suffix of the `petabEntityId` of the parameters of a network, a
#: layer or an array, e.g. `net1__parameters`, `net1__layer1__parameters`
PARAMETERS_SUFFIX = "parameters"


def network_yaml(sid: str) -> str:
    """Get the file of the NN YAML of a network."""
    return f"{sid}.yaml"


def network_arrays(sid: str) -> str:
    """Get the array file of a network, with its arrays and its inputs."""
    return f"{sid}_arrays.hdf5"


def petab_math(formula: str) -> str:
    """Get the math of PEtab of an L3 formula of SBML.

    Args:
        formula: the formula, e.g. `alpha + (prey - 1.3)`.

    Returns:
        The expression in the math of PEtab, in which `log` is the natural
        logarithm. A decimal of the formula is written as a decimal, not as
        the fraction the parser reads it as.

    Raises:
        ValueError: if the formula is not valid math.
    """
    expression = formula_expression(formula)

    def is_decimal(atom: sympy.Basic) -> bool:
        if not isinstance(atom, sympy.Rational) or atom.is_Integer:
            return False
        q = int(atom.q)
        while q % 10 == 0:
            q //= 10
        return q == 1

    expression = expression.replace(is_decimal, lambda atom: sympy.Float(atom))
    return petab_math_str(expression)


def petab_index(index: tuple[int, ...], shape: tuple[int, ...]) -> str:
    """Get the index of an element of an output as PEtab SciML writes it.

    Args:
        index: the index of the element in the output.
        shape: the shape of the output.

    Returns:
        The index, e.g. `[0][1]`, without the leading axes of length one,
        which are the axes of the batch, see `sbmlsim.fit.petab_v2.sciml`.
    """
    leading = 0
    while leading < len(shape) - 1 and shape[leading] == 1 and index[leading] == 0:
        leading += 1
    return "".join(f"[{i}]" for i in index[leading:])


def parameters_id(
    network: str, layer: str | None = None, array: str | None = None
) -> str:
    """Get the `petabEntityId` of the parameters of a network, a layer or an array."""
    parts = [network, layer, array, PARAMETERS_SUFFIX]
    return "__".join(part for part in parts if part)


def model_entity_id(
    network: str, layer: str | None = None, array: str | None = None
) -> str:
    """Get the `modelEntityId` of the parameters of a network, a layer or an array."""
    if layer is None:
        return f"{network}.parameters"
    if array is None:
        return f"{network}.parameters[{layer}]"
    return f"{network}.parameters[{layer}].{array}"


@dataclass
class ArrayRow:
    """The row of the parameter table an array of a network is described by."""

    estimate: bool
    lower: float
    upper: float

    def key(self) -> tuple[bool, float, float]:
        """Get the row as a hashable key."""
        return (self.estimate, self.lower, self.upper)


@dataclass
class ExportedNetwork:
    """A network of the problem with what is written for it.

    Attributes:
        network: the network.
        hybridizations: its hybridizations, one or two.
        pre_initialization: whether it runs before the simulation.
        arrays: the array file of the network, with the arrays of its inputs.
        parameter_rows: the rows of the parameter table, `petabEntityId` ->
            `modelEntityId` and the row.
    """

    network: Network
    hybridizations: list[Hybridization]
    pre_initialization: bool
    arrays: ArrayData
    parameter_rows: dict[str, tuple[str, ArrayRow]] = field(default_factory=dict)


class SciMLExporter:
    """Write the networks of an optimization problem as PEtab SciML.

    The exporter is created by `PetabExporter` for a problem with
    hybridizations and fills the tables of the problem it builds. It
    validates the hybridizations against the fit parameters of the problem:
    every element of a network which is a fit parameter has the value of the
    network as its start value, the linear scale and the unit
    `dimensionless`, and the elements of one array are either all estimated
    with one pair of bounds or all frozen.

    Attributes:
        problem: the problem which is exported.
        simulation_ids: id of the simulation of a fit mapping -> ids of its
            PEtab experiments.
        networks: the networks by their id.
        element_ids: the ids of the elements of all networks, which are no
            rows of the parameter table and no part of the `sbmlsim` block.
        constants: the constants of the hybridizations which are rows of
            the parameter table, id -> value.
        input_ids: id of an input -> its `petabEntityId`, which is the
            parameter of an input whose formula is a parameter of its own
            and the id of the input otherwise.
        input_formulas: id of an input which is written as changes of the
            conditions -> id of the simulation -> the formula.
        fallbacks: id of an input -> its formula for the simulations
            without a formula of their own, in the math of PEtab, which
            the conditions repeat and the block `sbmlsim` carries.
        hybridization_rows: the rows of the hybridization table.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        hybridizations: Sequence[Any],
        simulation_ids: Mapping[str, Sequence[str]],
    ) -> None:
        """Initialize the exporter.

        Args:
            problem: the initialized problem.
            hybridizations: the hybridizations of the problem.
            simulation_ids: id of the simulation of a fit mapping (the
                condition of the inputs) -> ids of its experiments of PEtab: a
                simulation whose fit mappings are in several collections is
                several experiments, which have the same inputs.

        Raises:
            ValueError: if a hybridization is not a `Hybridization` of
                `sbmlsim.sciml`, if a network is before the simulation and in
                the model, if two hybridizations of a network differ in the
                network, the model or the inputs, if an element which is a
                fit parameter differs from the network, if the elements of an
                array are not described by one row, if two hybridizations
                give a constant different values, or if an input has no
                formula for a simulation of the problem.
        """
        self.problem = problem
        self.simulation_ids = {key: list(ids) for key, ids in simulation_ids.items()}
        by_network: dict[str, list[Hybridization]] = {}
        for hybridization in hybridizations:
            if not isinstance(hybridization, Hybridization):
                raise ValueError(
                    f"'{problem.opid}': the hybridization '{hybridization}' is not "
                    f"a `Hybridization` of `sbmlsim.sciml`, only networks are "
                    f"written as PEtab SciML."
                )
            by_network.setdefault(hybridization.network.sid, []).append(hybridization)

        parameters = {p.pid: p for p in problem.parameters}
        self.networks: dict[str, ExportedNetwork] = {}
        for sid, group in by_network.items():
            first = group[0]
            for other in group[1:]:
                if other.network != first.network:
                    raise ValueError(
                        f"'{problem.opid}': the network '{sid}' has two "
                        f"hybridizations with different networks, a network of "
                        f"a problem is one network."
                    )
                if other.model != first.model or other.inputs != first.inputs:
                    raise ValueError(
                        f"'{problem.opid}': the network '{sid}' has two "
                        f"hybridizations which differ in the model or the inputs."
                    )
            patterns = {h.pattern for h in group}
            pre = NetworkPattern.PRE_INITIALIZATION in patterns
            if pre and len(patterns) > 1:
                raise ValueError(
                    f"'{problem.opid}': the network '{sid}' runs before the "
                    f"simulation and is in the model, a network of PEtab SciML "
                    f"is one of the two."
                )
            frozen: set[str] = set()
            for h in group:
                frozen |= set(h.frozen)
            self.networks[sid] = ExportedNetwork(
                network=first.network,
                hybridizations=list(group),
                pre_initialization=pre,
                arrays=ArrayData(
                    metadata=Metadata(pytorch_format=True),
                    parameters={
                        sid: {
                            layer: {
                                name: np.asarray(array, dtype=float)
                                for name, array in arrays.items()
                            }
                            for layer, arrays in first.network.parameters.items()
                        }
                    },
                ),
                parameter_rows=self._parameter_rows(first.network, frozen, parameters),
            )
        self.element_ids: set[str] = {
            sid
            for exported in self.networks.values()
            for sid in exported.network.parameter_ids()
        }
        self.constants: dict[str, float] = {}
        for exported in self.networks.values():
            for h in exported.hybridizations:
                for key, value in h.constants.items():
                    if key in parameters:
                        continue
                    if key in self.constants and self.constants[key] != value:
                        raise ValueError(
                            f"'{problem.opid}': the constant '{key}' has the values "
                            f"'{self.constants[key]}' and '{value}' in two "
                            f"hybridizations."
                        )
                    self.constants[key] = value
        self.input_ids: dict[str, str] = {}
        self.input_formulas: dict[str, dict[str, str]] = {}
        self.fallbacks: dict[str, str] = {}
        #: the inputs which are rows of the hybridization table
        self._row_inputs: set[str] = set()
        self._place_inputs(parameters)
        # built once: the arrays of the inputs go into the array files while
        # the rows are built
        self.hybridization_rows: list[HybridizationRow] = self._hybridization_rows()

    # --- THE INPUTS ---

    def _inputs(self) -> list[tuple[str, str, NetworkInput]]:
        """Get the inputs of all networks as network, id and input."""
        return [
            (sid, key, network_input)
            for sid, exported in self.networks.items()
            for key, network_input in exported.hybridizations[0].inputs.items()
        ]

    def _place_inputs(self, parameters: Mapping[str, FitParameter]) -> None:
        """Decide where every input of a formula is written, see the module.

        Args:
            parameters: the fit parameters of the problem by their id.

        Raises:
            ValueError: if an input has no formula for a simulation of the
                problem.
        """
        simulations = list(self.simulation_ids)
        # the parameters of the parameter table which the model does not
        # have, and how many formulas of inputs use each of them
        own = {
            key
            for exported in self.networks.values()
            for h in exported.hybridizations
            for key in h.constants
        } | {
            p.pid
            for p in parameters.values()
            if p.is_external and p.pid not in self.element_ids
        }
        uses: Counter[str] = Counter(
            symbol
            for _, _, network_input in self._inputs()
            for formula in network_input.all_formulas()
            for symbol in formula_symbols(formula) & own
        )
        for sid, key, network_input in self._inputs():
            if network_input.arrays is not None:
                self._drop_unknown(sid, key, network_input.arrays, "arrays")
                continue
            if network_input.formula is not None:
                formulas: dict[str, str] = {}
                fallback: str | None = network_input.formula
            else:
                formulas = dict(network_input.formulas or {})
                fallback = formulas.pop(ALL_CONDITIONS, None)
                self._drop_unknown(sid, key, formulas, "formulas")
            if not formulas and fallback is not None:
                if fallback in own and uses[fallback] == 1:
                    # the parameter of the input, as PEtab SciML writes it
                    self.input_ids[key] = fallback
                    continue
                if not formula_symbols(fallback) & own:
                    self._row_inputs.add(key)
                    continue
            by_simulation: dict[str, str] = {}
            for simulation in simulations:
                formula = formulas.get(simulation, fallback)
                if formula is None:
                    raise ValueError(
                        f"'{self.problem.opid}': the input '{key}' of the network "
                        f"'{sid}' has no formula for the simulation "
                        f"'{simulation}', it has formulas for {sorted(formulas)}."
                    )
                by_simulation[simulation] = formula
            self.input_formulas[key] = by_simulation
            if fallback is not None:
                self.fallbacks[key] = petab_math(fallback)

    def _drop_unknown(
        self, sid: str, key: str, values: Mapping[str, Any], what: str
    ) -> None:
        """Log the simulations of an input which the problem does not have.

        A selection of the data can leave out a simulation which an input
        names; the problem has no experiment for it, so its values are not
        written.

        Args:
            sid: id of the network.
            key: id of the input.
            values: id of the simulation -> the formula or the array.
            what: what the values are, for the message.
        """
        unknown = sorted(set(values) - set(self.simulation_ids) - {ALL_CONDITIONS})
        if unknown:
            logger.warning(
                "'%s': the input '%s' of the network '%s' has %s for the "
                "simulations %s, which the problem does not have; they are not "
                "written.",
                self.problem.opid,
                key,
                sid,
                what,
                unknown,
            )

    # --- THE PARAMETER ROWS ---

    def _parameter_rows(
        self,
        network: Network,
        frozen: set[str],
        parameters: Mapping[str, FitParameter],
    ) -> dict[str, tuple[str, ArrayRow]]:
        """Get the most general rows which describe the elements of a network.

        Args:
            network: the network.
            frozen: the ids of the frozen elements.
            parameters: the fit parameters of the problem by their id.

        Returns:
            `petabEntityId` -> `modelEntityId` and the row: one row for the
            network with the most common description of its arrays, and rows
            for the layers or the arrays which differ from it.

        Raises:
            ValueError: if an element which is a fit parameter does not carry
                the value of the network, the linear scale and the unit
                `dimensionless`, if an element is neither frozen nor a fit
                parameter, or if the elements of one array differ in whether
                they are estimated or in their bounds.
        """
        sid = network.sid
        used = set(network.used_layers())
        rows: dict[tuple[str, str], ArrayRow] = {}
        for element, (layer, name, index) in network.parameter_ids().items():
            if element in frozen or layer not in used:
                row = ArrayRow(False, -np.inf, np.inf)
            elif element in parameters:
                parameter = parameters[element]
                value = float(network.parameters[layer][name][index])
                if parameter.start_value != value:
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' starts "
                        f"from {parameter.start_value}, but the network '{sid}' "
                        f"carries {value}. The values of a network are the start "
                        f"values of its elements, set them on the network."
                    )
                if parameter.scale not in (None, ParameterScaleType.LINEAR):
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' has the "
                        f"scale '{parameter.scale.name}', the elements of a "
                        f"network are on the linear scale."
                    )
                if parameter.unit not in (None, ELEMENT_UNIT):
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' has the "
                        f"unit '{parameter.unit}', the elements of a network are "
                        f"'{ELEMENT_UNIT}'."
                    )
                row = ArrayRow(True, parameter.lower_bound, parameter.upper_bound)
            else:
                raise ValueError(
                    f"'{self.problem.opid}': the element '{element}' of the "
                    f"network '{sid}' is neither frozen nor a parameter of the fit."
                )
            first = rows.setdefault((layer, name), row)
            if first.key() != row.key():
                raise ValueError(
                    f"'{self.problem.opid}': the elements of the array '{name}' "
                    f"of the layer '{layer}' of the network '{sid}' differ in "
                    f"whether they are estimated or in their bounds "
                    f"({first} and {row} for '{element}'). A row of the parameter "
                    f"table of PEtab SciML describes an array as a whole "
                    f"(gap 'sciml-partial-array')."
                )

        # the most common row is the row of the network, the layers and the
        # arrays which differ get rows of their own
        counts = Counter(row.key() for row in rows.values())
        network_key = counts.most_common(1)[0][0]
        result: dict[str, tuple[str, ArrayRow]] = {
            parameters_id(sid): (model_entity_id(sid), ArrayRow(*network_key))
        }
        layers: dict[str, list[tuple[str, ArrayRow]]] = {}
        for (layer, name), row in rows.items():
            layers.setdefault(layer, []).append((name, row))
        for layer, arrays in layers.items():
            keys = {row.key() for _, row in arrays}
            if keys == {network_key}:
                continue
            if len(keys) == 1:
                result[parameters_id(sid, layer)] = (
                    model_entity_id(sid, layer),
                    ArrayRow(*next(iter(keys))),
                )
                continue
            for name, row in arrays:
                if row.key() != network_key:
                    result[parameters_id(sid, layer, name)] = (
                        model_entity_id(sid, layer, name),
                        row,
                    )
        return result

    # --- THE TABLES ---

    def conditions_of(self, simulation: str) -> list[str]:
        """Get the ids of the conditions of the first periods of a simulation.

        Args:
            simulation: id of the simulation.

        Returns:
            The id of the condition of the first period of every experiment of
            the simulation, which `PetabExporter._periods` gives it; none for
            a simulation none of whose fit mappings is written.
        """
        return [
            period_condition_id(experiment_id, 0)
            for experiment_id in self.simulation_ids.get(simulation, [])
        ]

    def input_changes(self, simulation: str) -> list[petab_v2.Change]:
        """Get the changes of the condition of a simulation which set inputs.

        Args:
            simulation: id of the simulation.

        Returns:
            One change per input which is written as changes of the
            conditions, see `_place_inputs`.
        """
        return [
            petab_v2.Change(target_id=key, target_value=petab_math(formula))
            for key, by_simulation in self.input_formulas.items()
            if (formula := by_simulation.get(simulation)) is not None
        ]

    def needs_condition(self, simulation: str) -> bool:
        """Check whether the first period of an experiment needs a condition.

        An input which differs between the conditions needs the condition of
        the experiment: its formula is a change of it, its arrays are keyed
        by it in the array file.

        Args:
            simulation: id of the simulation.
        """
        return any(
            network_input.is_conditional
            for exported in self.networks.values()
            for network_input in exported.hybridizations[0].inputs.values()
        )

    def mapping_rows(self) -> list[petab_v2.Mapping]:
        """Get the rows of the mapping table which name parts of the networks.

        Returns:
            A row per input, per used output and per row of the parameter
            table of the networks. The `petabEntityId` of an input and of an
            output which sets an entity is its id in `sbmlsim`, the one of an
            input which is a constant of its own is the constant, the one of
            an output an observable uses is its target, i.e. the symbol of
            the observable formula.
        """
        rows: list[petab_v2.Mapping] = []
        for sid, exported in self.networks.items():
            hybridization = exported.hybridizations[0]
            for key in hybridization.inputs:
                k, index = parse_io_id(sid, "input", key)
                indices = "" if index is None else "".join(f"[{i}]" for i in index)
                rows.append(
                    petab_v2.Mapping(
                        petab_id=self.input_ids.get(key, key),
                        model_id=f"{sid}.inputs[{k}]{indices}",
                    )
                )
            shapes = hybridization.output_shapes()
            for h in exported.hybridizations:
                for key, target in h.outputs.items():
                    k, index = parse_io_id(sid, "output", key)
                    petab_id = target if h.pattern is NetworkPattern.OBSERVABLE else key
                    indices = petab_index(index or (), shapes[k])
                    rows.append(
                        petab_v2.Mapping(
                            petab_id=petab_id, model_id=f"{sid}.outputs[{k}]{indices}"
                        )
                    )
            for petab_id, (model_id, _) in exported.parameter_rows.items():
                rows.append(petab_v2.Mapping(petab_id=petab_id, model_id=model_id))
        return rows

    def _hybridization_rows(self) -> list[HybridizationRow]:
        """Get the rows of the hybridization table.

        Returns:
            A row per input which is a formula for every condition and not
            written otherwise, see `_place_inputs` (`targetValue` is the
            math), per input of arrays (`targetValue` is `array`,
            the arrays go into the array file of the network keyed by the
            conditions), and a row per output of a network before the
            simulation or in the right hand side, which assigns the output
            to its target.
        """
        rows: list[HybridizationRow] = []
        for exported in self.networks.values():
            hybridization = exported.hybridizations[0]
            for key, network_input in hybridization.inputs.items():
                if key in self._row_inputs and network_input.formula is not None:
                    rows.append(
                        HybridizationRow(
                            target_id=key,
                            target_value=petab_math(network_input.formula),
                        )
                    )
                elif network_input.arrays is not None:
                    rows.append(HybridizationRow(target_id=key, target_value=ARRAY))
                    exported.arrays.inputs[key] = {
                        condition_id: np.asarray(array, dtype=float)
                        for condition, array in network_input.arrays.items()
                        if condition == ALL_CONDITIONS
                        or condition in self.simulation_ids
                        for condition_id in (
                            [ALL_CONDITION_IDS]
                            if condition == ALL_CONDITIONS
                            else self.conditions_of(condition)
                        )
                    }
            for h in exported.hybridizations:
                if h.pattern is NetworkPattern.OBSERVABLE:
                    continue
                for key, target in h.outputs.items():
                    rows.append(
                        HybridizationRow(target_id=entity_of(target), target_value=key)
                    )
        return rows

    def parameter_rows(self) -> list[petab_v2.Parameter]:
        """Get the rows of the parameter table of the networks and the constants.

        Returns:
            The rows of the arrays of every network with the nominal value
            `array`, and one row per constant of the hybridizations which is
            not estimated.
        """
        rows: list[petab_v2.Parameter] = []
        for exported in self.networks.values():
            for petab_id, (_, row) in exported.parameter_rows.items():
                rows.append(
                    petab_v2.Parameter(
                        id=petab_id,
                        lb=row.lower if row.estimate else None,
                        ub=row.upper if row.estimate else None,
                        nominal_value=ARRAY,
                        estimate=row.estimate,
                    )
                )
        for key, value in self.constants.items():
            rows.append(
                petab_v2.Parameter(
                    id=key, lb=None, ub=None, nominal_value=value, estimate=False
                )
            )
        return rows

    def config(self) -> SciMLConfig:
        """Get the block of the extension `sciml` of the problem."""
        return SciMLConfig(
            required=True,
            array_files=[Path(network_arrays(sid)) for sid in self.networks],
            hybridization_files=[Path(HYBRIDIZATION_FILE)],
            neural_networks={
                sid: NeuralNetConfig(
                    location=Path(network_yaml(sid)),
                    pre_initialization=exported.pre_initialization,
                    format=YAML_FORMAT.upper(),
                )
                for sid, exported in self.networks.items()
            },
        )

    def write(self, output_dir: Path) -> None:
        """Write the files of the networks into the directory of the problem.

        Args:
            output_dir: the directory of the problem. The NN YAML and the
                array file of every network and the hybridization table are
                written next to `problem.yaml`.
        """
        output_dir = Path(output_dir)
        for sid, exported in self.networks.items():
            NNModelStandard.save_data(
                exported.network.model, str(output_dir / network_yaml(sid))
            )
            ArrayDataStandard.save_data(
                exported.arrays, str(output_dir / network_arrays(sid))
            )
        HybridizationTable(self.hybridization_rows).to_tsv(
            output_dir / HYBRIDIZATION_FILE
        )
        logger.info(
            "The networks %s are written as PEtab SciML into '%s'",
            sorted(self.networks),
            output_dir,
        )
