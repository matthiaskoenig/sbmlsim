import importlib
import logging
import warnings
from enum import Enum
from pathlib import Path

import libnuml
import libsbml
import libsedml
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class NumlParser(object):
    """Helper class for parsing Numl data files."""

    class Library(Enum):
        LIBNUML = 1
        LIBSEDML = 2

    @classmethod
    def read_numl_document(cls, path):
        """Helper to read external numl document and check for errors

        :param path: path of file
        :return:
        """
        importlib.reload(libnuml)
        doc_numl = libnuml.readNUMLFromFile(path)  # type: libnuml.NUMLDocument

        # check for errors
        errorlog = doc_numl.getErrorLog()
        msg = "NUML ERROR in '{}': {}".format(path, errorlog.toString())
        if errorlog.getNumFailsWithSeverity(libnuml.LIBNUML_SEV_ERROR) > 0:
            raise IOError(msg)
        if errorlog.getNumFailsWithSeverity(libnuml.LIBNUML_SEV_FATAL) > 0:
            raise IOError(msg)
        if errorlog.getNumFailsWithSeverity(libnuml.LIBNUML_SEV_WARNING) > 0:
            warnings.warn(msg)
        if errorlog.getNumFailsWithSeverity(libnuml.LIBNUML_SEV_SCHEMA_ERROR) > 0:
            warnings.warn(msg)
        if errorlog.getNumFailsWithSeverity(libnuml.LIBNUML_SEV_GENERAL_WARNING) > 0:
            warnings.warn(msg)

        importlib.reload(libsbml)
        return doc_numl

    @classmethod
    def load_numl_data(cls, path) -> pd.DataFrame:
        """Reading NuML data from file.

        This loads the complete numl data.
        For more information see: https://github.com/numl/numl

        :param path: NuML path
        :return: data
        """
        importlib.reload(libnuml)
        path_str = path
        if isinstance(path, Path):
            path_str = str(path)

        doc_numl = NumlParser.read_numl_document(path_str)

        # reads all the resultComponents from the numl file
        results = []

        Nrc = doc_numl.getNumResultComponents()
        rcs = doc_numl.getResultComponents()

        logger.info("\nNumResultComponents:", Nrc)
        for k in range(Nrc):
            rc = rcs.get(k)  # parse ResultComponent
            rc_id = rc.getId()

            # dimension info
            description = rc.getDimensionDescription()
            data_types = cls.parse_dimension_description(description)

            # data
            dimension = rc.getDimension()
            assert isinstance(dimension, libnuml.Dimension)
            data = [
                cls._parse_dimension(dimension.get(k)) for k in range(dimension.size())
            ]

            # create data frame
            flat_data = []
            for entry in data:
                for part in entry:
                    flat_data.append(part)

            # column ids from DimensionDescription
            column_ids = []
            for entry in data_types:
                for cid, dtype in entry.items():
                    column_ids.append(cid)

            df = pd.DataFrame(flat_data, columns=column_ids)

            # convert data types to actual data types
            for entry in data_types:
                for cid, dtype in entry.items():
                    if dtype == "double":
                        df[cid] = df[cid].astype(np.float64)
                    elif dtype == "string":
                        df[cid] = df[cid].astype(str)

            # convert all the individual columns to the corresponding data types
            # df = df.apply(pd.to_numeric, errors="ignore")

            results.append([rc_id, df, data_types])

        return results

    @classmethod
    def parse_dimension_description(
        cls, description, library: Library = Library.LIBNUML
    ):
        """Parses the given dimension description.

        Returns dictionary of { key: dtype }

        :param description:
        :return:
        """
        if library == cls.Library.LIBNUML:
            importlib.reload(libnuml)
            assert description.getTypeCode() == libnuml.NUML_DIMENSIONDESCRIPTION
        elif library == cls.Library.LIBSEDML:
            importlib.reload(libsedml)
            assert description.getTypeCode() == libsedml.NUML_DIMENSIONDESCRIPTION

        info = [
            cls._parse_description(description.get(k), library=library)
            for k in range(description.size())
        ]

        flat_info = []
        for entry in info:
            for part in entry:
                flat_info.append(part)

        return flat_info

    @classmethod
    def _parse_description(
        cls, d, info=None, entry=None, library: Library = Library.LIBNUML
    ):
        """Parses the recursive DimensionDescription, TupleDescription, AtomicDescription.

        This gets the dimension information from NuML.

          <dimensionDescription>
            <compositeDescription indexType="double" id="time" name="time">
              <compositeDescription indexType="string" id="SpeciesIds" name="SpeciesIds">
                <atomicDescription valueType="double" id="Concentrations" name="Concentrations" />
              </compositeDescription>
            </compositeDescription>
          </dimensionDescription>

        :param d:
        :param info:
        :return:
        """
        type_code = d.getTypeCode()
        if library == cls.Library.LIBNUML:
            importlib.reload(libnuml)
            assert type_code in [
                libnuml.NUML_COMPOSITEDESCRIPTION,
                libnuml.NUML_ATOMICDESCRIPTION,
                libnuml.NUML_TUPLEDESCRIPTION,
            ]
            # if type_code == libnuml.NUML_COMPOSITEDESCRIPTION:
            #    d = libnuml.CompositeDescription(d)
        elif library == cls.Library.LIBSEDML:
            importlib.reload(libsedml)
            assert type_code in [
                libsedml.NUML_COMPOSITEDESCRIPTION,
                libsedml.NUML_ATOMICDESCRIPTION,
                libsedml.NUML_TUPLEDESCRIPTION,
            ]
            # if type_code == libsedml.NUML_COMPOSITEDESCRIPTION:
            #    d = libnuml.CompositeDescription(d)

        if info is None:
            info = []
        if entry is None:
            entry = []
        print("typecode:", type_code)
        print("type:", type(d))
        print("object", object)

        if (
            library == cls.Library.LIBNUML
            and type_code == libnuml.NUML_COMPOSITEDESCRIPTION
        ) or (
            library == cls.Library.LIBSEDML
            and type_code == libsedml.NUML_COMPOSITEDESCRIPTION
        ):

            content = {d.getId(): d.getIndexType()}
            info.append(content)
            # print('\t* CompositeDescription:', content)
            if d.isContentCompositeDescription():
                for k in range(d.size()):
                    info = cls._parse_description(
                        d.getCompositeDescription(k), info, list(entry), library=library
                    )
            elif d.isContentAtomicDescription():
                info = cls._parse_description(
                    d.getAtomicDescription(), info, entry, library=library
                )

        elif (
            library == cls.Library.LIBNUML
            and type_code == libnuml.NUML_ATOMICDESCRIPTION
        ) or (
            library == cls.Library.LIBSEDML
            and type_code == libsedml.NUML_ATOMICDESCRIPTION
        ):
            content = {d.getId(): d.getValueType()}
            info.append(content)
            # print('\t* AtomicDescription:', content)

        elif (
            library == cls.Library.LIBNUML
            and type_code == libnuml.NUML_TUPLEDESCRIPTION
        ) or (
            library == cls.Library.LIBSEDML
            and type_code == libsedml.NUML_TUPLEDESCRIPTION
        ):
            tuple_des = d.getTupleDescription()
            Natomic = d.size()
            valueTypes = []
            for k in range(Natomic):
                atomic = tuple_des.getAtomicDescription(k)
                valueTypes.append(atomic.getValueType())

            info.append(valueTypes)
            # print('\t* TupleDescription:', valueTypes)

        else:
            raise NotImplementedError("Type code: {}".format(type_code))

        return info

    @classmethod
    def _parse_dimension(
        cls, d, data=None, entry=None, library: Library = Library.LIBNUML
    ):
        """Parses the recursive CompositeValue, Tuple, AtomicValue.

        This gets the actual data from NuML.

        :param d:
        :param data:
        :return:
        """
        if library == cls.Library.LIBNUML:
            importlib.reload(libnuml)
        elif library == cls.Library.LIBSEDML:
            importlib.reload(libsedml)
        if data is None:
            data = []
        if entry is None:
            entry = []

        type_code = d.getTypeCode()
        # print('typecode:', type_code)

        if (
            library == cls.Library.LIBNUML and type_code == libnuml.NUML_COMPOSITEVALUE
        ) or (
            library == cls.Library.LIBSEDML
            and type_code == libsedml.NUML_COMPOSITEVALUE
        ):

            indexValue = d.getIndexValue()
            entry.append(indexValue)
            # print('\t* CompositeValue:', indexValue)

            if d.isContentCompositeValue():
                for k in range(d.size()):
                    # make copy, so every entry is own entry
                    data = cls._parse_dimension(
                        d.getCompositeValue(k), data, list(entry)
                    )
            elif d.isContentAtomicValue():
                data = cls._parse_dimension(d.getAtomicValue(), data, entry)

        elif (
            library == cls.Library.LIBNUML and type_code == libnuml.NUML_ATOMICVALUE
        ) or (
            library == cls.Library.LIBSEDML and type_code == libsedml.NUML_ATOMICVALUE
        ):
            # Data is converted to correct
            # value = d.getDoubleValue()
            value = d.getValue()
            entry.append(value)
            # entry finished, we are appending
            data.append(entry)
            # print('\t* AtomicValue:', value)

        elif (library == cls.Library.LIBNUML and type_code == libnuml.NUML_TUPLE) or (
            library == cls.Library.LIBSEDML and type_code == libsedml.NUML_TUPLE
        ):
            tuple = d.getTuple()
            Natomic = d.size()
            values = []
            for k in range(Natomic):
                atomic = tuple.getAtomicValue(k)
                values.append(atomic.getDoubleValue())

            data.append(values)
            # print('\t* TupleDescription:', values)

        else:
            raise NotImplementedError

        return data
