"""Tools and helpers to handle parameters for sensitivity analysis."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Iterable

import libsbml
import numpy as np
import pandas as pd
import roadrunner
from pydantic import BaseModel, Field, ConfigDict
from pymetadata.console import console
from sbmlutils.report.units import udef_to_string


class ParameterType(str, Enum):
    """Types of model parameters."""
    DATA = "data"
    SCALING = "scaling"
    NA = "na"
    FIT = "fitted"


class SensitivityParameter(BaseModel):
    """Parameter for SensitivityAnalysis."""
    model_config = ConfigDict(use_enum_values=True)

    uid: str
    name: str
    value: float = Field(default=np.nan)
    lower_bound: float = Field(default=np.nan)
    upper_bound: float = Field(default=np.nan)
    unit: Optional[str] = None
    type: ParameterType = ParameterType.NA
    reference: str = ""

    def __hash__(self):
        return hash(self.uid)

    @staticmethod
    def parameters_set_bounds(parameters: Iterable[SensitivityParameter],
                              bounds: Iterable[tuple]) -> None:
        """Set bounds for sensitivity analysis."""

        parameters_d = {p.uid: p for p in parameters}

        for (key, lb, ub, ptype) in bounds:
            if key not in parameters_d:
                console.print(f"unused bounds definition: {key} = [{lb}, {ub}]")
            else:
                p = parameters_d[key]
                p.lower_bound = lb
                p.upper_bound = ub
                p.type = ptype

    @staticmethod
    def parameters_to_df(parameters: Iterable[SensitivityParameter],
                         sort: bool = True) -> pd.DataFrame:
        """Create parameter table from parameters."""
        items = []
        for item in parameters:
            d_item = item.model_dump()
            # better printing of type
            d_item["type"] = d_item["type"].value
            items.append(d_item)

        df = pd.DataFrame(items)
        if sort:
            df.sort_values(by=["type", "uid"], ascending=True, inplace=True,
                           ignore_index=True)
        return df

    @classmethod
    def parameter_to_latex(
        cls,
        tex_path: Path,
        parameters: list[SensitivityParameter],
    ) -> None:
        """Latex parameter table."""
        df = cls.parameters_to_df(parameters)
        tex_str = df.to_latex(
            None, index=False, float_format="{:.3g}".format
        )
        tex_str = tex_str.replace("_", r"\_")

        with open(tex_path, 'w') as f:
            f.write(tex_str)

    @staticmethod
    def parameters_from_sbml(
        sbml_path: Path,
        exclude_ids: Optional[set[str]] = None,
        exclude_na: bool = True,
        exclude_zero: bool = True,
    ) -> list[SensitivityParameter]:
        """Retrieve parameters from SBML model for the sensitivity analysis.

        Constant parameters, constant compartments and constant species are returned.

        :sbml_path: Path to the SBML file.
        :param exclude_ids: ids to exclude,
        :param exclude_na: whether to exclude NA values
        :return: dict[id, name]
        """
        r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(sbml_path))
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
        sbml_model: libsbml.Model = doc.getModel()
        parameters = []

        if not exclude_ids:
            exclude_ids = set()

        def parameter_from_sbase(sbase: libsbml.SBase) -> SensitivityParameter:
            """Create parameter from SBase for sensitivity analysis."""
            uid = sbase.getId()

            name = sbase.getName() if sbase.isSetName() else uid
            udef: libsbml.UnitDefinition = sbase.getDerivedUnitDefinition()
            unit: str = udef_to_string(udef, model=None, format="str")

            # handle the species concentration
            ruid = uid
            if (sbase.getTypeCode() == libsbml.SpeciesType) and (
                sbase.getHasOnlySubstanceUnits()):
                ruid = f"[{uid}]"

            value = r.getValue(ruid)

            parameter = SensitivityParameter(
                uid=uid,
                name=name,
                value=value,
                unit=unit,
                lower_bound=np.nan,
                upper_bound=np.nan,
            )

            return parameter

        # constant parameters
        p: libsbml.Parameter
        for p in sbml_model.getListOfParameters():
            sid = p.getId()
            if p.getConstant() is True:
                if exclude_na and np.isnan(p.getValue()):
                    exclude_ids.add(sid)
                if exclude_zero and np.isclose(r.getValue(sid), 0.0):
                    exclude_ids.add(sid)
                parameters.append(parameter_from_sbase(p))

        # constant compartments
        c: libsbml.Compartment
        for c in sbml_model.getListOfCompartments():
            sid = c.getId()
            if c.getConstant() is True:
                if exclude_na and np.isnan(c.getSize()):
                    exclude_ids.add(sid)
                if exclude_zero and np.isclose(r.getValue(sid), 0.0):
                    exclude_ids.add(sid)
                parameters.append(parameter_from_sbase(c))

        # constant species or boundaryCondition == True
        s: libsbml.Species
        for s in sbml_model.getListOfSpecies():
            sid = s.getId()

            if exclude_na:
                if not s.isSetInitialAmount() and not s.isSetInitialConcentration():
                    exclude_ids.add(sid)
                elif s.isSetInitialAmount() and np.isnan(s.getInitialAmount()):
                    exclude_ids.add(sid)
                elif s.isSetInitialConcentration() and np.isnan(
                    s.getInitialConcentration()):
                    exclude_ids.add(sid)
            if exclude_zero:
                if s.isSetInitialAmount() and np.isclose(s.getInitialAmount(), 0.0):
                    exclude_ids.add(sid)
                elif s.isSetInitialConcentration() and np.isclose(
                    s.getInitialConcentration(), 0.0):
                    exclude_ids.add(sid)

            if s.getConstant() is True or s.getBoundaryCondition() is True:
                parameters.append(parameter_from_sbase(s))

        # remove excluded ids
        parameters_filtered: list[SensitivityParameter] = []
        parameters_excluded: list[SensitivityParameter] = []

        sp: SensitivityParameter
        for sp in parameters:
            if sp.uid in exclude_ids:
                parameters_excluded.append(sp)
            else:
                parameters_filtered.append(sp)

        console.print(f"Excluded parameters: {[sp.uid for sp in parameters_excluded]}")

        return parameters_filtered
