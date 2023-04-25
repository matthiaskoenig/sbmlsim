from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple, Any

import numpy as np
import pandas as pd
import libsbml
from petab.conditions import get_condition_df
import uuid


class Change:
    """Assignment of value to a target id in the model.

        ${parameterId}
            The values will override any parameter values specified in the model.

        ${speciesId}
            If a species ID is provided, it is interpreted as the initial
            concentration/amount of that species and will override the initial
            concentration/amount given in the SBML model or given by
            a preequilibration condition. If NaN is provided for a condition, the result
            of the preequilibration (or initial concentration/amount from the SBML model,
            if no preequilibration is defined) is used.

        ${compartmentId}
            If a compartment ID is provided, it is interpreted as the initial
            compartment size.
    """
    def __init__(self,
                 target_id: str,
                 value: float,
                 unit: Optional[str]
                 ):
        self.target_id: str = target_id
        self.value: float = value
        self.unit: str = unit


class Condition:
    """Collection of assignments with a given id."""

    def __init__(self,
            sid: str,
            name: Optional[str],
            changes: Optional[List[Change]]
    ):
        self.sid: str = sid
        self.name: Optional[str] = name
        if changes is None:
            changes = []
        self.changes: List[Change] = changes

    @classmethod
    def parse_conditions_from_file(cls, conditions_path: Path) -> List[Condition]:
        """Parse conditions from file."""
        df = get_condition_df(condition_file=str(conditions_path))
        return cls.parse_conditions(df)

    @staticmethod
    def parse_conditions(df: pd.DataFrame) -> List[Condition]:
        """Parse conditions from DataFrame."""
        conditions: List[Condition] = []
        columns = df.columns
        target_ids = [col for col in columns if col not in {"conditionName"}]
        for condition_id, row in df.iterrows():
            changes: List[Change] = []
            for tid in target_ids:
                changes.append(
                    Change(
                        target_id=tid,
                        value=row[tid],
                        unit=None,
                    )
                )
            condition = Condition(
                sid=str(condition_id),
                name=row["conditionName"] if "conditionName" in columns else None,
                changes=changes
            )
            conditions.append(condition)

        return conditions


class SimulateSBML:
    """Class for simulating an SBML model."""

    def __init__(self, sbml_path, conditions: List[Condition], results_dir: Path,
                 absolute_tolerance: float=1E-8, relative_tolerance=1E-8):
        """

        :param sbml_path: Path to SBML model.
        :param results_dir: Path to results dir and intermediate results,
        :param absolute_tolerance: absolute tolerance for simulation
        :param relative_tolerance: relatvie tolerance for simulation
        :param conditions: conditions to simulate
        """

        self.sbml_path: Path = sbml_path
        self.conditions: Dict[str, Condition] = {c.sid: c for c in conditions}
        self.results_dir = results_dir
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

        # process SBML information for unifying simulations
        sbml_data = self.parse_sbml(sbml_path=self.sbml_path)
        self.mid: str = sbml_data[0]
        self.species: Set[str] = sbml_data[1]
        self.compartments: Set[str] = sbml_data[2]
        self.parameters: Set[str] = sbml_data[3]
        self.has_only_substance: Dict[str, bool] = sbml_data[4]
        self.species_compartments: Dict[str, str] = sbml_data[5]
        self.sid2name: Dict[str, str] = sbml_data[6]

    @staticmethod
    def parse_sbml(sbml_path: Path) -> Tuple[Any]:
        """Parses the identifiers."""
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
        model: libsbml.Model = doc.getModel()
        species: Set[str] = set()
        parameters: Set[str] = set()
        compartments: Set[str] = set()
        has_only_substance: Dict[str, bool] = {}
        species_compartments: Dict[str, str] = {}
        sid2name: Dict[str, str] = {}
        mid = str(uuid.uuid4())

        if model:
            if model.isSetId():
                mid = model.getId()
            s: libsbml.Species
            for s in model.getListOfSpecies():
                sid = s.getId()
                has_only_substance[sid] = s.getHasOnlySubstanceUnits()
                species_compartments[sid] = s.getCompartment()
                sid2name[sid] = s.getName() if s.isSetName() else s.getId()

            for p in model.getListOfParameters():
                sid2name[p.getId()] = p.getName() if p.isSetName() else p.getId()
            for c in model.getListOfCompartments():
                sid2name[c.getId()] = c.getName() if c.isSetName() else c.getId()

            species = {s.getId() for s in model.getListOfSpecies()}
            parameters = {p.getId() for p in model.getListOfParameters()}
            compartments = {c.getId() for c in model.getListOfCompartments()}

        return (
            mid,
            species,
            compartments,
            parameters,
            has_only_substance,
            species_compartments,
            sid2name,
        )

    def simulate_condition(self, condition: Condition, timepoints: List[float]):
        pass
