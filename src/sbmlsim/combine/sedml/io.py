"""
Template functions to run the example cases.
"""
import importlib
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Union
from xml.etree import ElementTree

import libcombine
import libsedml

from sbmlsim.combine.omex import Omex

logger = logging.getLogger(__name__)


INPUT_TYPE_STR = "SEDML_STRING"
INPUT_TYPE_FILE_SEDML = "SEDML_FILE"
INPUT_TYPE_FILE_COMBINE = "COMBINE_FILE"  # includes .sedx archives


def check_sedml(sed_doc: libsedml.SedDocument) -> libsedml.SedErrorLog:
    """Checks the SedDocument for errors.

    :param sed_doc: SedDocument.
    """
    errorlog: libsedml.SedErrorLog = sed_doc.getErrorLog()
    msg = errorlog.toString()
    if sed_doc.getErrorLog().getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_ERROR) > 0:
        # FIXME: workaround for https://github.com/fbergmann/libSEDML/issues/47
        logger.warning(msg)
        # raise IOError(msg)
    if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_FATAL) > 0:
        # raise IOError(msg)
        logger.warning(msg)
    if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_WARNING) > 0:
        logger.warning(msg)
    if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_SCHEMA_ERROR) > 0:
        logger.warning(msg)
    if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_GENERAL_WARNING) > 0:
        logger.warning(msg)
    if errorlog.getNumErrors() > 0:
        logger.error(f"errors: {msg}")
    return errorlog


# FIXME: use proper results data structure
# FIXME: support execution of multiple SED-ML files
# FIXME: cleanup of function

def read_sedml(source: Union[Path, str], working_dir: Path = None) -> Dict:
    """Parses SedMLDocument from given input.

    :return: dictionary of SedDocument, input_type and working directory.
    """
    if isinstance(source, str) and not Path(source).exists:
        logger.info("SED-ML from string")
        input_type = INPUT_TYPE_STR
        try:
            # check if XML can be parsed
            ElementTree.fromstring(source)
            # is parsable xml string
        except ElementTree.ParseError as err:
            logger.error(f"SED-ML string is not valid XML: '{source}'")
            raise err

        sed_doc: libsedml.SedDocument = libsedml.readSedMLFromString(source)
        if sed_doc is None:
            raise IOError("SED-ML could not be read.")

        if working_dir is None:
            working_dir = Path.cwd()

    else:
        file_path = Path(source)
        file_stem, file_suffix = file_path.stem, file_path.suffix

        if zipfile.is_zipfile(file_path):
            logger.info(f"SED-ML from archive: {file_path}")
            input_type = INPUT_TYPE_FILE_COMBINE
            omex_path = file_path

            # in case of an archive a working directory is created
            # in which the files are extracted
            extract_dir: Path
            if working_dir is None:
                extract_dir = omex_path.parent / f"_sbmlsim_{file_stem}"
            else:
                extract_dir = working_dir
            logger.debug(f"extracting archive to '{extract_dir}'")

            # extract archive to working directory
            importlib.reload(libcombine)
            omex = Omex(omex_path=file_path, working_dir=working_dir)
            omex.extract()
            for location, master in omex.locations_by_format("sed-ml"):
                print("SED-ML location: ", location)
                if master:
                    sedml_path = extract_dir / location
                    break
            else:
                logger.error("No SED-ML file with master flag found.")

            importlib.reload(libsedml)
            sed_doc = libsedml.readSedMLFromFile(str(sedml_path))

            # we have to work relative to the SED-ML file
            # FIXME: add execution directory (clear separation between where the files are
            # and where SED-ML is resolving files
            working_dir = sedml_path.parent

        elif file_path.exists():
            logger.info(f"SED-ML from file: {file_path}")
            input_type = INPUT_TYPE_FILE_SEDML
            if file_suffix not in [".sedml", ".xml"]:
                raise IOError(
                    f"SEDML file must have [.sedml|.xml] extension:" f"'{source}'"
                )

            sed_doc = libsedml.readSedMLFromFile(str(source))

            # working directory is where the sedml file is
            if working_dir is None:
                working_dir = file_path.parent

        # check document
        errorlog = check_sedml(sed_doc)

        return sed_doc, errorlog, working_dir, input_type
