# -*- coding: utf-8 -*-
"""
Converting SED-ML to a simulation experiment.
Reading SED-ML file and encoding as simulation experiment.

This module implements SED-ML support for sbmlsim.

----------------
Overview SED-ML
----------------
SED-ML is build of main classes
    the Model Class,
    the Simulation Class,
    the Task Class,
    the DataGenerator Class,
    and the Output Class.

The Model Class
    The Model class is used to reference the models used in the simulation experiment.
    SED-ML itself is independent of the model encoding underlying the models. The only
    requirement is that the model needs to be referenced by using an unambiguous identifier
    which allows for finding it, for example using a MIRIAM URI. To specify the language in
    which the model is encoded, a set of predefined language URNs is provided.
    The SED-ML Change class allows the application of changes to the referenced models,
    including changes on the XML attributes, e.g. changing the value of an observable,
    computing the change of a value using mathematics, or general changes on any XML element
    of the model representation that is addressable by XPath expressions, e.g. substituting
    a piece of XML by an updated one.

TODO: DATA CLASS


The Simulation Class
    The Simulation class defines the simulation settings and the steps taken during simulation.
    These include the particular type of simulation and the algorithm used for the execution of
    the simulation; preferably an unambiguous reference to such an algorithm should be given,
    using a controlled vocabulary, or ontologies. One example for an ontology of simulation
    algorithms is the Kinetic Simulation Algorithm Ontology KiSAO. Further information encodable
    in the Simulation class includes the step size, simulation duration, and other
    simulation-type dependent information.

The Task Class
    SED-ML makes use of the notion of a Task class to combine a defined model (from the Model class)
    and a defined simulation setting (from the Simulation class). A task always holds one reference each.
    To refer to a specific model and to a specific simulation, the corresponding IDs are used.

The DataGenerator Class
    The raw simulation result sometimes does not correspond to the desired output of the simulation,
    e.g. one might want to normalise a plot before output, or apply post-processing like mean-value calculation.
    The DataGenerator class allows for the encoding of such post-processings which need to be applied to the
    simulation result before output. To define data generators, any addressable variable or parameter
    of any defined model (from instances of the Model class) may be referenced, and new entities might
    be specified using MathML definitions.

The Output Class
    The Output class defines the output of the simulation, in the sense that it specifies what shall be
    plotted in the output. To do so, an output type is defined, e.g. 2D-plot, 3D-plot or data table,
    and the according axes or columns are all assigned to one of the formerly specified instances
    of the DataGenerator class.

For information about SED-ML please refer to http://www.sed-ml.org/
and the SED-ML specification.

------------------------------------
SED-ML in tellurium: Implementation
------------------------------------
SED-ML support in tellurium is based on Combine Archives.
The SED-ML files in the Archive can be executed and stored with results.

----------------------------------------
SED-ML in tellurium: Supported Features
----------------------------------------
Tellurium supports SED-ML L1V3 with SBML as model format.

SBML models are fully supported, whereas for CellML models only basic support
is implemented (when additional support is requested it be implemented).
CellML models are transformed to SBML models which results in different XPath expressions,
so that targets, selections cannot be easily resolved in the CellMl-SBML.

Supported input for SED-ML are either SED-ML files ('.sedml' extension),
SED-ML XML strings or combine archives ('.sedx'|'.omex' extension).
Executable python code is generated from the SED-ML which allows the
execution of the defined simulation experiment.

In the current implementation all SED-ML constructs with exception of
XML transformation changes of the model
    - Change.RemoveXML
    - Change.AddXML
    - Change.ChangeXML
are supported.

"""
import re
import requests
import logging
import sys
import platform
import tempfile
import shutil
import traceback
import os.path
import warnings
import datetime
import zipfile
import numpy as np
from collections import namedtuple
from pathlib import Path

import libsedml
import importlib
importlib.reload(libsedml)

from sbmlsim.combine import biomodels
from sbmlsim.combine import omex
from sbmlsim.combine.sedml.utils import SEDMLTools
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import load_model
from sbmlsim.combine.sedml.mathml import evaluableMathML

logger = logging.getLogger(__file__)
'''
def experiment_from_omex(omex_path: Path):
    """Create SimulationExperiments from all SED-ML files."""
    tmp_dir = tempfile.mkdtemp()
    try:
        omex.extractCombineArchive(omex_path, directory=tmp_dir, method="zip")
        locations = omex.getLocationsByFormat(omex_path, "sed-ml")
        sedml_files = [os.path.join(tmp_dir, loc) for loc in locations]

        for k, sedml_file in enumerate(sedml_files):
            pystr = sedmlToPython(sedml_file)
            factory = SEDMLCodeFactory(inputStr, workingDir=workingDir)
            factory.to
            pycode[locations[k]] = pystr

    finally:
        shutil.rmtree(tmp_dir)
    return pycode
'''


class SEDMLModelParser(object):
    """ Code Factory generating executable code."""

    def __init__(self, doc: libsedml.SedDocument, working_dir: Path):
        self.doc = doc
        self.working_dir = working_dir
        model_sources, model_changes = SEDMLTools.resolve_model_changes(self.doc)

        self.model_sources = model_sources
        self.model_changes = model_changes
        self.models = {}

        for sed_model in doc.getListOfModels():  # type: libsedml.SedModel
            mid = sed_model.getId()  # type: str
            model_result = self.parse_model(sed_model, working_dir=working_dir)
            self.models[mid] = model_result['model']

    def parse_model(self, sed_model: libsedml.SedModel, working_dir: Path):
        """ Python code for SedModel.

        :param sed_model: SedModel instance
        :type sed_model: SedModel
        :return: python str
        :rtype: str
        """
        mid = sed_model.getId()
        language = sed_model.getLanguage()
        source = self.model_sources[mid]

        if not language:
            warnings.warn("No model language specified, defaulting to SBML for: {}".format(source))

        def is_urn():
            return source.lower().startswith('urn')

        def is_http():
            return source.lower().startswith('http') or source.startswith('HTTP')

        # read SBML
        if 'sbml' in language or len(language) == 0:
            sbml_str = None
            if is_urn():
                sbml_str = biomodels.from_urn(source)
            elif is_http():
                sbml_str = biomodels.from_url(source)
            if sbml_str:
                model = load_model(sbml_str)
            else:
                # load file, by resolving path relative to working dir
                # FIXME: absolute paths?
                sbml_path = os.path.join(working_dir, source)
                model = load_model(sbml_path)

        # read CellML
        elif 'cellml' in language:
            warnings.warn("CellML model encountered, sbmlsim does not support CellML".format(language))
            raise ValueError("CellML models not supported yet")
        # other
        else:
            warnings.warn("Unsupported model language: '{}'.".format(language))

        # apply model changes
        for change in self.model_changes[mid]:
            self._apply_model_change(model, change)

        return {
            'model': model,
            'mid': mid,
            'language': language,
        }

    @staticmethod
    def set_xpath_value(xpath: str, value: float, model):
        """ Creates python line for given xpath target and value.
        :param xpath:
        :type xpath:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        target = SEDMLModelParser._resolve_xpath(xpath)
        if target:
            if target.type == "concentration":
                # initial concentration
                expr = f'init([{target.id}])'
            elif target.type == "amount":
                # initial amount
                expr = f'init({target.id})'
            else:
                # other (parameter, flux, ...)
                expr = target.id
            print(f"{expr} = {value}")
            model[expr] = value
        else:
            logger.error(f"Unsupported target xpath: {xpath}")


    @staticmethod
    def _resolve_xpath(xpath: str):
        """ Resolve the target from the xpath expression.

        A single target in the model corresponding to the modelId is resolved.
        Currently, the model is not used for xpath resolution.

        :param xpath: xpath expression.
        :type xpath: str
        :param modelId: id of model in which xpath should be resolved
        :type modelId: str
        :return: single target of xpath expression
        :rtype: Target (namedtuple: id type)
        """
        # TODO: via better xpath expression
        #   get type from the SBML document for the given id.
        #   The xpath expression can be very general and does not need to contain the full
        #   xml path
        #   For instance:
        #   /sbml:sbml/sbml:model/descendant::*[@id='S1']
        #   has to resolve to species.
        # TODO: figure out concentration or amount (from SBML document)
        # FIXME: getting of sids, pids not very robust, handle more cases (rules, reactions, ...)

        Target = namedtuple('Target', 'id type')

        def getId(xpath):
            xpath = xpath.replace('"', "'")
            match = re.findall(r"id='(.*?)'", xpath)
            if (match is None) or (len(match) is 0):
                logger.warn("Xpath could not be resolved: {}".format(xpath))
            return match[0]

        # parameter value change
        if ("model" in xpath) and ("parameter" in xpath):
            return Target(getId(xpath), 'parameter')
        # species concentration change
        elif ("model" in xpath) and ("species" in xpath):
            return Target(getId(xpath), 'concentration')
        # other
        elif ("model" in xpath) and ("id" in xpath):
            return Target(getId(xpath), 'other')
        # cannot be parsed
        else:
            raise ValueError("Unsupported target in xpath: {}".format(xpath))

    @staticmethod
    def selectionFromVariable(var, model):
        """ Resolves the selection for the given variable.

        First checks if the variable is a symbol and returns the symbol.
        If no symbol is set the xpath of the target is resolved
        and used in the selection

        :param var: variable to resolve
        :type var: SedVariable
        :return: a single selection
        :rtype: Selection (namedtuple: id type)
        """
        Selection = namedtuple('Selection', 'id type')

        # parse symbol expression
        if var.isSetSymbol():
            cvs = var.getSymbol()
            astr = cvs.rsplit("symbol:")
            sid = astr[1]
            return Selection(sid, 'symbol')
        # use xpath
        elif var.isSetTarget():
            xpath = var.getTarget()
            target = SEDMLModelParser._resolveXPath(xpath, model)
            return Selection(target.id, target.type)

        else:
            warnings.warn("Unrecognized Selection in variable")
            return None

    def _apply_model_change(self, model, change):
        """ Creates the apply change python string for given model and change.

        Currently only a very limited subset of model changes is supported.
        Namely changes of parameters and concentrations within a SedChangeAttribute.

        :param model: given model
        :type model: SedModel
        :param change: model change
        :type change: SedChange
        :return:
        :rtype: str
        """
        xpath = change.getTarget()

        if change.getTypeCode() == libsedml.SEDML_CHANGE_ATTRIBUTE:
            # resolve target change
            value = float(change.getNewValue())
            SEDMLModelParser.set_xpath_value(xpath, value, model=model)

        elif change.getTypeCode() == libsedml.SEDML_CHANGE_COMPUTECHANGE:
            # calculate the value
            variables = {}
            for par in change.getListOfParameters():  # type: libsedml.SedParameter
                variables[par.getId()] = par.getValue()

            for var in change.getListOfVariables():  # type: libsedml.SedVariable
                vid = var.getId()
                selection = SEDMLModelParser.selectionFromVariable(var, model)
                expr = selection.id
                if selection.type == "concentration":
                    expr = f"init([{selection.id}])"
                elif selection.type == "amount":
                    expr = f"init({selection.id})"
                variables[vid] = model[expr]

            # value is calculated with the current state of model
            value = evaluableMathML(change.getMath(), variables=variables)
            SEDMLModelParser.set_xpath_value(xpath, value, model=model)

        elif change.getTypeCode() in [libsedml.SEDML_CHANGE_REMOVEXML,
                                      libsedml.SEDML_CHANGE_ADDXML,
                                      libsedml.SEDML_CHANGE_CHANGEXML]:
            logger.error(f"Unsupported change: {change.getElementName()}")
        else:
            logger.error(f"Unknown change: {change.getElementName()}")


if __name__ == "__main__":
    from pathlib import Path
    base_path = Path(__file__).parent
    sedml_path = base_path / "experiments" / "repressilator_sedml.xml"
    results = SEDMLTools.read_sedml_document(str(sedml_path), working_dir=base_path)
    doc = results['doc']
    sed_parser = SEDMLModelParser(doc, working_dir=base_path)

