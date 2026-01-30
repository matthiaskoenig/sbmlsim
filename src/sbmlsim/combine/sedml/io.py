"""Template functions to run the example cases."""

import importlib
import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree

import libsedml
from pymetadata import omex as pyomex
from pymetadata import log


logger = log.get_logger(__name__)


def check_sedml_doc(sed_doc: libsedml.SedDocument) -> libsedml.SedErrorLog:
    """Check SedDocument for errors.

    Logs errors and warnings

    :param sed_doc: SedDocument.
    :return SedErrorLog.
    """
    errorlog: libsedml.SedErrorLog = sed_doc.getErrorLog()
    msg = errorlog.toString()
    if sed_doc.getErrorLog().getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_ERROR) > 0:
        logger.error(msg)
    if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_FATAL) > 0:
        logger.error(msg)
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


class SEDMLInputType(Enum):
    """Types of SED-ML input.

    SED-ML can be read from string, file or a COMBINE archive (or zip archives).
    """

    SEDML_STRING = 0
    SEDML_FILE = 1
    OMEX = 2


class SEDMLReader:
    """Class for reading SED-ML document from various sources.

    SED-ML can be provided as string, file or as file in a COMBINE
    archive.

    Execution must be performed where the master SED-ML is located.
    """

    def __init__(self, source: Union[Path, str], working_dir: Path = None):
        """Initialize SEDMLReader."""
        self.source: Union[Path, str] = source
        self.exec_dir: Path = os.getcwd()
        self.working_dir: Path = working_dir
        self.input_type: Optional[SEDMLInputType] = None
        self.error_log: Optional[libsedml.SedErrorLog] = None
        self.sed_doc: Optional[libsedml.SedDocument] = None

        # read document
        self.sed_doc, self.input_type = self.read_sedml()

        # check document
        if self.sed_doc:
            self.error_log: Optional[libsedml.SedErrorLog] = check_sedml_doc(
                self.sed_doc
            )

    def __repr__(self) -> None:
        """Get string representation."""

        source_str = (
            self.source
            if self.input_type is not SEDMLInputType.SEDML_STRING
            else "string"
        )
        return f"<SEDMLReader(source={source_str}, input_type={self.input_type}), exec_dir={self.exec_dir}>"

    def __str__(self) -> str:
        """Get string."""
        source_str = (
            self.source
            if self.input_type is not SEDMLInputType.SEDML_STRING
            else "string"
        )
        info = [
            "SEDMLReader(",
            f"\tinput_type: {self.input_type}",
            f"\tsource: {source_str}",
            f"\texec_dir: {self.exec_dir}",
            ")",
        ]
        return "\n".join(info)

    def read_sedml(self) -> Tuple[libsedml.SedDocument, SEDMLInputType]:
        """Read SedMLDocument.

        Sets the instance variables as a result.
        """
        sed_doc: libsedml.SedDocument
        input_type: SEDMLInputType

        if isinstance(self.source, str) and "<sedML" in self.source:
            logger.warning("SED-ML from string")
            sedml_str: str = self.source
            input_type = SEDMLInputType.SEDML_STRING
            try:
                # check if XML can be parsed
                ElementTree.fromstring(sedml_str)
                # is parsable xml string
            except ElementTree.ParseError as err:
                logger.error(f"SED-ML string is not valid XML: '{sedml_str}'")
                raise err

            sed_doc: libsedml.SedDocument = libsedml.readSedMLFromString(sedml_str)
        else:
            file_path = Path(self.source)
            if not file_path.exists():
                raise IOError(f"SED-ML file/archive does not exist: {file_path}")

            _, file_suffix = file_path.stem, file_path.suffix

            if zipfile.is_zipfile(file_path):
                logger.warning(f"SED-ML from archive: {file_path}")
                input_type = SEDMLInputType.OMEX

                # in case of an archive a working directory is created
                # in which the files are extracted
                omex = pyomex.Omex.from_omex(omex_path=file_path)
                sedml_entries = omex.entries_by_format(format_key="sed-ml")
                for entry in sedml_entries:
                    logger.info("SED-ML location: ", entry.location)
                    if entry.master:
                        sedml_path = omex.get_path(entry.location)
                        break
                else:
                    logger.error(
                        f"No SED-ML file with master flag found in archive: "
                        f"'{file_path}'. Using first file."
                    )
                    if len(sedml_entries) > 0:
                        sedml_path = omex.get_path(sedml_entries[0].location)
                    else:
                        raise ValueError("No SED-ML in archive.")

                importlib.reload(libsedml)
                self.exec_dir = sedml_path.parent
                sed_doc = libsedml.readSedMLFromFile(str(sedml_path))

            else:
                logger.warning(f"SED-ML from file: {file_path}")
                input_type = SEDMLInputType.SEDML_FILE
                if file_suffix not in [".sedml", ".xml"]:
                    logger.error(
                        f"SEDML should have [.sedml|.xml] extension: '{file_path}'"
                    )

                self.exec_dir = file_path.parent
                sed_doc = libsedml.readSedMLFromFile(str(file_path))

        if sed_doc is None:
            raise IOError("SED-ML could not be read.")

        # FIXME: figure out the working dir, i.e. relative to the SED-ML files
        return sed_doc, input_type
