# -*- coding: utf-8 -*-
"""
SED-ML support for sbmlsim
==========================

This modules parses SED-ML based simulation experiments in the sbmlsim
SimulationExperiment format and executes them.

Overview SED-ML
----------------
SED-ML is build of the main classes
- DataDescription
- Model
- Simulation
- Task
- DataGenerator
- Output

DataDescription
---------------
The DataDescription allows to reference external data, and contains a
description on how to access the data, in what format it is, and what subset
of data to extract.

Model
-----
The Model class is used to reference the models used in the simulation
experiment. SED-ML itself is independent of the model encoding underlying the
models. The only requirement is that the model needs to be referenced by
using an unambiguous identifier which allows for finding it, for example
using a MIRIAM URI. To specify the language in which the model is encoded,
a set of predefined language URNs is provided. The SED-ML Change class allows
the application of changes to the referenced models, including changes on the
XML attributes, e.g. changing the value of an observable, computing the change
of a value using mathematics, or general changes on any XML element
of the model representation that is addressable by XPath expressions,
e.g. substituting a piece of XML by an updated one.

Simulation
----------
The Simulation class defines the simulation settings and the steps taken
during simulation. These include the particular type of simulation and the
algorithm used for the execution of the simulation; preferably an unambiguous
reference to such an algorithm should be given, using a controlled vocabulary,
or ontologies. One example for an ontology of simulation algorithms is the
Kinetic Simulation Algorithm Ontology KiSAO. Further information encodable
in the Simulation class includes the step size, simulation duration, and other
simulation-type dependent information.

Task
----
SED-ML makes use of the notion of a Task class to combine a defined model
(from the Model class) and a defined simulation setting
(from the Simulation class). A task always holds one reference each.
To refer to a specific model and to a specific simulation, the corresponding
IDs are used.

DataGenerator
-------------
The raw simulation result sometimes does not correspond to the desired output
of the simulation, e.g. one might want to normalise a plot before output,
or apply post-processing like mean-value calculation.
The DataGenerator class allows for the encoding of such post-processings
which need to be applied to the simulation result before output.
To define data generators, any addressable variable or parameter of any
defined model (from instances of the Model class) may be referenced,
and new entities might be specified using MathML definitions.

Output
-------
The Output class defines the output of the simulation, in the sense that it
specifies what shall be plotted in the output. To do so, an output type is
defined, e.g. 2D-plot, 3D-plot or data table, and the according axes or
columns are all assigned to one of the formerly specified instances of the
DataGenerator class.

For information about SED-ML please refer to http://www.sed-ml.org/
and the SED-ML specification.

SED-ML in sbmlsim: Supported Features
=====================================
sbmlsim supports SED-ML L1V4 with SBML as model format.
SBML models are fully supported

Supported input for SED-ML are either SED-ML files ('.sedml' extension),
SED-ML XML strings or combine archives ('.sedx'|'.omex' extension).

In the current implementation all SED-ML constructs with exception of
XML transformation changes of the model, i.e.,
- Change.RemoveXML
- Change.AddXML
- Change.ChangeXML
are supported.
"""
import re
import logging
import warnings
import numpy as np
from typing import Dict, List
from collections import namedtuple
from pathlib import Path
import libsedml
import importlib

from sbmlsim.models import model_resources
from sbmlsim.combine.sedml.data import DataDescriptionParser
from sbmlsim.combine.sedml.utils import SEDMLTools
from sbmlsim.models.model import AbstractModel

from pint import UnitRegistry

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


class SEDMLParser(object):
    """ Parsing SED-ML in internal format."""

    def __init__(self, sed_doc: libsedml.SedDocument, working_dir: Path):
        self.sed_doc = sed_doc  # type: libsedml.SedDocument
        self.working_dir = working_dir

        # unit registry to handle units throughout the simulation
        self.ureg = UnitRegistry()

        # --- Models ---
        self.models = {}
        # resolve original model source and changes
        model_sources, model_changes = self.resolve_model_changes()
        for sed_model in self.sed_doc.getListOfModels():  # type: libsedml.SedModel
            mid = sed_model.getId()
            source = model_sources[mid]
            sed_changes = model_changes[mid]
            self.models[mid] = self.parse_model(sed_model, source=source, sed_changes=sed_changes)

        '''
        # --- DataDescriptions ---
        for sed_data_description in sed_doc.getListOfDataDescriptions():  # type: libsedml.SedDataDescription
            did = sed_data_description.getId()
            self.data_descriptions[did] = self.parse_data_description(sed_data_description)

        # parse tasks
        for sed_task in sed_doc.getListOfTasks():  # type: libsedml.SedTask
            tid = sed_task.getId()
            self.tasks[tid] = self.parse_tasks(sed_task)
        '''

    # --- MODELS ---
    def parse_model(self, sed_model: libsedml.SedModel,
                    source: str,
                    sed_changes: List[libsedml.SedChange]) -> AbstractModel:
        """ Convert SedModel to AbstractModel.

        :param sed_model:
        :return:
        """
        # TODO: resolve changes
        changes = []
        for sed_change in sed_changes:
            self.parse_change(model, sed_change)

        mid = sed_model.getId()
        model = AbstractModel(
            source=source,
            sid=mid,
            name=sed_model.getName(),
            language=sed_model.getLanguage(),
            base_path=self.working_dir,
            changes=None,
            selections=None,
            ureg=self.ureg
        )


        return model

    def resolve_model_changes(self):
        """Resolves the original model sources and full change lists.

         Going through the tree of model upwards until root is reached and
         collecting changes on the way (example models m* and changes c*)
         m1 (source) -> m2 (c1, c2) -> m3 (c3, c4)
         resolves to
         m1 (source) []
         m2 (source) [c1,c2]
         m3 (source) [c1,c2,c3,c4]
         The order of changes is important (at least between nodes on different
         levels of hierarchies), because later changes of derived models could
         reverse earlier changes.

         Uses recursive search strategy, which should be okay as long as the
         model tree hierarchy is not getting to deep.
         """
        def findSource(mid, changes):
            """
            Recursive search for original model and store the
            changes which have to be applied in the list of changes

            :param mid:
            :param changes:
            :return:
            """
            # mid is node above
            if mid in model_sources and not model_sources[mid] == mid:
                # add changes for node
                for c in model_changes[mid]:
                    changes.append(c)
                # keep looking deeper
                return findSource(model_sources[mid], changes)
            # the source is no longer a key in the sources, it is the source
            return mid, changes

        # store original source and changes for model
        model_sources = {}
        model_changes = {}

        # collect direct source and changes
        for m in self.sed_doc.getListOfModels():  # type: libsedml.SedModel
            mid = m.getId()
            source = m.getSource()
            model_sources[mid] = source
            changes = []
            # store the changes unique for this model
            for c in m.getListOfChanges():
                changes.append(c)
            model_changes[mid] = changes

        # resolve source and changes recursively
        all_changes = {}
        mids = [m.getId() for m in self.sed_doc.getListOfModels()]
        for mid in mids:
            source, changes = findSource(mid, changes=list())
            model_sources[mid] = source
            all_changes[mid] = changes[::-1]

        return model_sources, all_changes

    def parse_change(self, sed_change: libsedml.SedChange) -> Dict:
        """ Parses the change.

        Currently only a limited subset of model changes is supported.
        Namely changes of parameters and concentrations within a
        SedChangeAttribute.
        """
        xpath = sed_change.getTarget()

        if sed_change.getTypeCode() == libsedml.SEDML_CHANGE_ATTRIBUTE:
            # resolve target change
            value = float(sed_change.getNewValue())

            SEDMLParser.set_xpath_value(xpath, value, model=model)


        elif sed_change.getTypeCode() == libsedml.SEDML_CHANGE_COMPUTECHANGE:
            # calculate the value
            variables = {}
            for par in sed_change.getListOfParameters():  # type: libsedml.SedParameter
                variables[par.getId()] = par.getValue()

            for var in sed_change.getListOfVariables():  # type: libsedml.SedVariable
                vid = var.getId()
                selection = SEDMLParser.selectionFromVariable(var, model)
                expr = selection.id
                if selection.type == "concentration":
                    expr = f"init([{selection.id}])"
                elif selection.type == "amount":
                    expr = f"init({selection.id})"
                variables[vid] = model[expr]

            # value is calculated with the current state of model
            value = evaluableMathML(sed_change.getMath(), variables=variables)
            SEDMLParser.set_xpath_value(xpath, value, model=model)

        elif sed_change.getTypeCode() in [libsedml.SEDML_CHANGE_REMOVEXML,
                                          libsedml.SEDML_CHANGE_ADDXML,
                                          libsedml.SEDML_CHANGE_CHANGEXML]:
            logger.error(f"Unsupported change: {sed_change.getElementName()}")
        else:
            logger.error(f"Unknown change: {sed_change.getElementName()}")


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
        target = SEDMLParser._resolve_xpath(xpath)
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






    def parse_data_description(self, dataDescription):
        """Parse DataDescription.

        :param dataDescription: SedModel instance
        :type dataDescription: DataDescription
        :return: python str
        :rtype: str
        """
        lines = []
        data_sources = DataDescriptionParser.parse(dataDescription, self.workingDir)

        # FIXME: still needed
        # for sid, data in data_sources.items():
        #    # handle the 1D shapes
        #    if len(data.shape) == 1:
        #        data = np.reshape(data.values, (data.shape[0], 1))

        return data_sources


if __name__ == "__main__":

    base_path = Path(__file__).parent
    sedml_path = base_path / "experiments" / "repressilator_sedml.xml"
    results = SEDMLTools.read_sedml_document(str(sedml_path), working_dir=base_path)
    doc = results['doc']
    sed_parser = SEDMLParser(doc, working_dir=base_path)

