"""Tools and helpers to handle parameters for sensitivity analysis."""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import libsbml
import numpy as np
from sbmlutils.console import console
from sbmlutils.factory import ValueWithUnit



@dataclass
class SensitivityParameter:
    """Parameter for SensitivityAnalysis"""
    uid: str
    name: str
    unit: Optional[str] = None
    lower_bound: float = np.nan
    upper_bound: float = np.nan

    def __hash__(self):
        return hash(self.uid)


def parameters_for_sensitivity_analysis(
    sbml_path: Path,
    exclude_ids: Optional[set[str]] = None,
    exclude_na: bool = True,
) -> list[SensitivityParameter]:
    """Retrieve parameters from model for the sensitivity analysis.

    Constant parameters, constant compartments and constant species are returned.

    :sbml_path: Path to the SBML file.
    :param exclude_ids: ids to exclude,
    :param exclude_na: whether to exclude NA values
    :return: dict[id, name]
    """

    def parameter_from_sbase(sbase: libsbml.SBase) -> SensitivityParameter:
        """Create parameter from SBase for sensitivity analysis."""
        uid = sbase.getId()
        name = sbase.getName() if sbase.isSetName() else uid
        udef: libsbml.UnitDefinition = sbase.getDerivedUnitDefinition()
        unit: str = libsbml.UnitDefinition.printUnits(ud=udef, compact=True)

        # FIXME: get bound information from SBML model or table
        parameter = SensitivityParameter(uid=uid, name=name, unit=unit)

        return parameter


    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    sbml_model: libsbml.Model = doc.getModel()

    parameters = []
    excluded: list[SensitivityParameter] = []

    # constant parameters
    p: libsbml.Parameter
    for p in sbml_model.getListOfParameters():
        sid = p.getId()
        if p.getConstant() is True:
            if exclude_na and np.isnan(p.getValue()):
                exclude_ids.add(sid)
            parameters.append(parameter_from_sbase(p))

    # constant compartments
    c: libsbml.Compartment
    for c in sbml_model.getListOfCompartments():
        sid = c.getId()
        if c.getConstant() is True:
            if exclude_na and np.isnan(c.getSize()):
                exclude_ids.add(sid)
            parameters.append(parameter_from_sbase(c))

    # constant species or boundaryCondition == True
    s: libsbml.Species
    for s in sbml_model.getListOfSpecies():
        sid = s.getId()
        name = s.getName() if s.isSetName() else sid
        if exclude_na:
            if not s.isSetInitialAmount() and not s.isSetInitialConcentration():
                exclude_ids.add(sid)
            elif s.isSetInitialAmount() and np.isnan(s.getInitialAmount()):
                exclude_ids.add(sid)
            elif s.isSetInitialConcentration() and np.isnan(s.getInitialConcentration()):
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

    console.print(f"Excluded parameters: {parameters_excluded}")

    return parameters_filtered


if __name__ == "__main__":
    model_path = Path(__file__).parent / "models" / "losartan" / "losartan_body_flat.xml"
    parameters: list[SensitivityParameter] = parameters_for_sensitivity_analysis(
        sbml_path=model_path,
        exclude_ids = {
            "conversion_min_per_day",  # constant conversion factor
            "Mr_los",  # molecular weight
        }
    )

    console.print("finished")
    console.print(parameters)
