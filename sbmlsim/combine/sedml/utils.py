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
    def check_sedml_document(cls, doc: libsedml.SedDocument):
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
    def read_sedml_document(cls, input: str, working_dir):
        """ Parses SedMLDocument from given input.

        :return: dictionary of SedDocument, inputType and working directory.
        :rtype: {doc, inputType, workingDir}
        """

        # SEDML-String
        if not os.path.exists(input):
            try:
                from xml.etree import ElementTree
                x = ElementTree.fromstring(input)
                # is parsable xml string
                doc = libsedml.readSedMLFromString(input)
                inputType = cls.INPUT_TYPE_STR
                if working_dir is None:
                    working_dir = os.getcwd()

            except ElementTree.ParseError:
                if not os.path.exists(input):
                    raise IOError("SED-ML String is not valid XML:", input)

        # SEDML-File
        else:
            filename, extension = os.path.splitext(os.path.basename(input))

            # Archive
            if zipfile.is_zipfile(input):
                omexPath = input
                inputType = cls.INPUT_TYPE_FILE_COMBINE

                # in case of sedx and combine a working directory is created
                # in which the files are extracted
                if working_dir is None:
                    extractDir = os.path.join(os.path.dirname(os.path.realpath(omexPath)), '_te_{}'.format(filename))
                else:
                    extractDir = working_dir

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
                working_dir = os.path.dirname(sedmlFile)

                cls.check_sedml_document(doc)

            # SEDML single file
            elif os.path.isfile(input):
                if extension not in [".sedml", '.xml']:
                    raise IOError("SEDML file should have [.sedml|.xml] extension:", input)
                inputType = cls.INPUT_TYPE_FILE_SEDML
                doc = libsedml.readSedMLFromFile(input)
                cls.check_sedml_document(doc)
                # working directory is where the sedml file is
                if working_dir is None:
                    working_dir = os.path.dirname(os.path.realpath(input))

        return {'doc': doc,
                'inputType': inputType,
                'workingDir': working_dir}

    @staticmethod
    def resolve_model_changes(doc: libsedml.SedDocument):
        """ Resolves the original source model and full change lists for models.

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

         Uses recursive search strategy, which should be okay as long as the model tree hierarchy is
         not getting to big.
         """
        # initial dicts (handle source & change information for single node)
        model_sources = {}
        model_changes = {}

        for m in doc.getListOfModels():  # type: libsedml.SedModel
            mid = m.getId()
            source = m.getSource()
            model_sources[mid] = source
            changes = []
            # store the changes unique for this model
            for c in m.getListOfChanges():
                changes.append(c)
            model_changes[mid] = changes

        # recursive search for original model and store the
        # changes which have to be applied in the list of changes
        def findSource(mid, changes):
            # mid is node above
            if mid in model_sources and not model_sources[mid] == mid:
                # add changes for node
                for c in model_changes[mid]:
                    changes.append(c)
                # keep looking deeper
                return findSource(model_sources[mid], changes)
            # the source is no longer a key in the sources, it is the source
            return mid, changes

        all_changes = {}

        mids = [m.getId() for m in doc.getListOfModels()]
        for mid in mids:
            source, changes = findSource(mid, changes=list())
            model_sources[mid] = source
            all_changes[mid] = changes[::-1]

        return model_sources, all_changes
