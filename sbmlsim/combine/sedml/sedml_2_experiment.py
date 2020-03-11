"""
Converting SED-ML to a simulation experiment.
Reading SED-ML file and encoding as simulation experiment.
"""
import sys
import platform
import tempfile
import shutil
import traceback
import os.path
import warnings
import datetime
import zipfile
import re
import numpy as np
from collections import namedtuple
import jinja2
from pathlib import Path

import libsedml
import importlib
importlib.reload(libsedml)

from sbmlsim.combine import biomodels
import re
import requests

from sbmlsim.combine import omex
from sbmlsim.experiment import SimulationExperiment
from .mathml import evaluableMathML




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

class SEDMLModelParser(object):




class SEDMLCodeFactory(object):
    """ Code Factory generating executable code."""

    def parse_models(self, model: libsedml.SedModel, working_dir: Path):
        """ Python code for SedModel.

        :param model: SedModel instance
        :type model: SedModel
        :return: python str
        :rtype: str
        """
        mid = model.getId()
        language = model.getLanguage()
        source = self.model_sources[mid]

        if not language:
            warnings.warn("No model language specified, defaulting to SBML for: {}".format(source))

        def is_urn():
            return source.lower().startswith('urn')

        def is_http():
            return source.lower().startswith('http') or source.startswith('HTTP')

        # read SBML
        if 'sbml' in language or len(language) == 0:
            if is_urn():
                sbml_str = biomodels.from_urn(source)
            elif is_http():
                sbml_str = biomodels.from_url(source)
            else:
                # load file
                path = os.path.join(working_dir, source)

        # read CellML
        elif 'cellml' in language:
            warnings.warn("CellML model encountered, sbmlsim does not support CellML".format(language))
            raise ValueError
        # other
        else:
            warnings.warn("Unsupported model language: '{}'.".format(language))

        # TODO: load models


        # apply model changes
        for change in self.model_changes[mid]:
            lines.extend(SEDMLCodeFactory.modelChangeToPython(model, change))

        return model