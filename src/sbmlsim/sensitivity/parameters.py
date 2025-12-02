"""Tools and helpers to handle parameters for senitivity analysis."""
from pathlib import Path
from typing import Optional

import libsbml
import numpy as np
from sbmlutils.console import console

def bounds_for_parameters():
    """Retrieve the parameter bounds from the model."""



def parameters_for_sensitivity_analysis(
    sbml_path: Path,
    exclude_ids: Optional[set[str]] = None,
    exclude_na: bool = True,
) -> dict[str, str]:
    """Retrieve parameters from model for the sensitivity analysis.

    Constant parameters, constant compartments and constant species are returned.

    :sbml_path: Path to the SBML file.
    :param exclude_ids: ids to exclude,
    :param exclude_na: whether to exclude NA values
    :return: dict[id, name]
    """


    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    sbml_model: libsbml.Model = doc.getModel()

    id2name: dict[str, str] = {}
    excluded: set[str] = set()

    # constant parameters
    p: libsbml.Parameter
    for p in sbml_model.getListOfParameters():
        sid = p.getId()
        if p.getConstant() is True:
            if exclude_na and np.isnan(p.getValue()):
                excluded.add(sid)
                continue
            id2name[sid] = p.getName() if p.isSetName() else sid

    # constant compartments
    c: libsbml.Compartment
    for c in sbml_model.getListOfCompartments():
        sid = c.getId()
        if c.getConstant() is True:
            if exclude_na and np.isnan(c.getSize()):
                excluded.add(sid)
                continue
            id2name[sid] = c.getName() if c.isSetName() else sid

    # constant species or boundaryCondition == True
    s: libsbml.Species
    for s in sbml_model.getListOfSpecies():
        sid = s.getId()
        name = s.getName() if s.isSetName() else sid
        if exclude_na:
            if not s.isSetInitialAmount() and not s.isSetInitialConcentration():
                excluded.add(sid)
                continue
            if s.isSetInitialAmount() and np.isnan(s.getInitialAmount()):
                excluded.add(sid)
                continue
            if s.isSetInitialConcentration() and np.isnan(s.getInitialConcentration()):
                excluded.add(sid)
                continue

        if s.getConstant() is True or s.getBoundaryCondition() is True:
            id2name[sid] = name

    # remove excluded ids
    if exclude_ids:
        for sid in exclude_ids:
            if sid in id2name:
                excluded.add(sid)
                id2name.pop(sid)

    console.print(f"Excluded parameters: {excluded}")

    return id2name

if __name__ == "__main__":
    model_path = Path(__file__).parent / "models" / "losartan" / "losartan_body_flat.xml"
    id2name = parameters_for_sensitivity_analysis(
        sbml_path=model_path,
        exclude_ids = {
            "conversion_min_per_day",  # constant conversion factor
            "Mr_los",  # molecular weight
        }
    )
    console.print(id2name)
