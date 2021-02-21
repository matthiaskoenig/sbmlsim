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


logger = logging.getLogger(__name__)


INPUT_TYPE_STR = "SEDML_STRING"
INPUT_TYPE_FILE_SEDML = "SEDML_FILE"
INPUT_TYPE_FILE_COMBINE = "COMBINE_FILE"  # includes .sedx archives


def check_sedml(doc: libsedml.SedDocument) -> str:
    """Checks the SedDocument for errors.

    :param doc: SedDocument.
    """
    errorlog = doc.getErrorLog()
    msg = errorlog.toString()
    if doc.getErrorLog().getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_ERROR) > 0:
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
    print(f"errors: {msg}")
    return str(msg)


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

        doc = libsedml.readSedMLFromString(source)

        if working_dir is None:
            working_dir = Path.cwd()

    else:
        file_path = Path(source)
        file_stem, file_suffix = file_path.stem, file_path.suffix

        if zipfile.is_zipfile(file_path):
            print(f"SED-ML from archive: {file_path}")
            input_type = INPUT_TYPE_FILE_COMBINE
            omex_path = file_path

            # in case of an archive a working directory is created
            # in which the files are extracted
            if working_dir is None:
                extract_dir = omex_path.parent / f"_sbmlsim_{file_stem}"
            else:
                extract_dir = working_dir
            print(f"extracting archive to '{extract_dir}'")

            # extract archive to working directory
            importlib.reload(libcombine)
            libcombine.CombineArchive.extractArchive(str(omex_path), str(extract_dir))
            sedml_files = libcombine.CombineArchive.filePathsFromExtractedArchive(
                str(extract_dir), filetype="sed-ml"
            )
            if len(sedml_files) == 0:
                raise IOError(f"No SEDML files found in archive: {omex_path}")
            elif len(sedml_files) > 1:
                logger.warning(
                    "More than one sedml file in archive, only "
                    f"processing first file."
                )

            sedml_path = extract_dir / sedml_files[0]
            if not file_path.exists():
                raise IOError(f"SED-ML file does not exist: {sedml_path}")

            importlib.reload(libsedml)
            doc = libsedml.readSedMLFromFile(str(sedml_path))

            # we have to work relative to the SED-ML file
            working_dir = sedml_path.parent

        elif file_path.exists():
            print(f"SED-ML from file: {file_path}")
            input_type = INPUT_TYPE_FILE_SEDML
            if file_suffix not in [".sedml", ".xml"]:
                raise IOError(
                    f"SEDML file must have [.sedml|.xml] extension:" f"'{source}'"
                )

            doc = libsedml.readSedMLFromFile(str(source))

            # working directory is where the sedml file is
            if working_dir is None:
                working_dir = file_path.parent

        # check document
        check_sedml(doc)
        print

        return doc, working_dir, input_type
