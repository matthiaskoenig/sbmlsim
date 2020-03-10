"""
Template functions to run the example cases.
"""
import os
import warnings
import libsedml
import zipfile

import importlib
import libcombine
importlib.reload(libcombine)
import libsedml
importlib.reload(libsedml)

class SEDMLTools(object):
    """ Helper functions to work with sedml. """

    INPUT_TYPE_STR = 'SEDML_STRING'
    INPUT_TYPE_FILE_SEDML = 'SEDML_FILE'
    INPUT_TYPE_FILE_COMBINE = 'COMBINE_FILE'  # includes .sedx archives

    @classmethod
    def checkSEDMLDocument(cls, doc):
        """ Checks the SedDocument for errors.
        Raises IOError if error exists.
        :param doc:
        :type doc:
        """
        errorlog = doc.getErrorLog()
        msg = errorlog.toString()
        if doc.getErrorLog().getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_ERROR) > 0:
            # FIXME: workaround for https://github.com/fbergmann/libSEDML/issues/47
            warnings.warn(msg)
            # raise IOError(msg)
        if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_FATAL) > 0:
            # raise IOError(msg)
            warnings.warn(msg)
        if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_WARNING) > 0:
            warnings.warn(msg)
        if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_SCHEMA_ERROR) > 0:
            warnings.warn(msg)
        if errorlog.getNumFailsWithSeverity(libsedml.LIBSEDML_SEV_GENERAL_WARNING) > 0:
            warnings.warn(msg)

    @classmethod
    def readSEDMLDocument(cls, inputStr, workingDir):
        """ Parses SedMLDocument from given input.

        :return: dictionary of SedDocument, inputType and working directory.
        :rtype: {doc, inputType, workingDir}
        """

        # SEDML-String
        if not os.path.exists(inputStr):
            try:
                from xml.etree import ElementTree
                x = ElementTree.fromstring(inputStr)
                # is parsable xml string
                doc = libsedml.readSedMLFromString(inputStr)
                inputType = cls.INPUT_TYPE_STR
                if workingDir is None:
                    workingDir = os.getcwd()

            except ElementTree.ParseError:
                if not os.path.exists(inputStr):
                    raise IOError("SED-ML String is not valid XML:", inputStr)

        # SEDML-File
        else:
            filename, extension = os.path.splitext(os.path.basename(inputStr))

            # Archive
            if zipfile.is_zipfile(inputStr):
                omexPath = inputStr
                inputType = cls.INPUT_TYPE_FILE_COMBINE

                # in case of sedx and combine a working directory is created
                # in which the files are extracted
                if workingDir is None:
                    extractDir = os.path.join(os.path.dirname(os.path.realpath(omexPath)), '_te_{}'.format(filename))
                else:
                    extractDir = workingDir

                # TODO: refactor this
                importlib.reload(libcombine)
                # extract the archive to working directory
                libcombine.CombineArchive.extractArchive(omexPath, extractDir)
                # get SEDML files from archive
                sedmlFiles = libcombine.CombineArchive.filePathsFromExtractedArchive(extractDir, filetype='sed-ml')
                importlib.reload(libsedml)

                if len(sedmlFiles) == 0:
                    raise IOError("No SEDML files found in archive.")

                # FIXME: there could be multiple SEDML files in archive (currently only first used)
                # analogue to executeOMEX
                if len(sedmlFiles) > 1:
                    warnings.warn("More than one sedml file in archive, only processing first one.")

                sedmlFile = sedmlFiles[0]
                doc = libsedml.readSedMLFromFile(sedmlFile)
                # we have to work relative to the SED-ML file
                workingDir = os.path.dirname(sedmlFile)

                cls.checkSEDMLDocument(doc)

            # SEDML single file
            elif os.path.isfile(inputStr):
                if extension not in [".sedml", '.xml']:
                    raise IOError("SEDML file should have [.sedml|.xml] extension:", inputStr)
                inputType = cls.INPUT_TYPE_FILE_SEDML
                doc = libsedml.readSedMLFromFile(inputStr)
                cls.checkSEDMLDocument(doc)
                # working directory is where the sedml file is
                if workingDir is None:
                    workingDir = os.path.dirname(os.path.realpath(inputStr))

        return {'doc': doc,
                'inputType': inputType,
                'workingDir': workingDir}
