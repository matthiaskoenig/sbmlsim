"""
Reading NUML, CSV and TSV data from DataDescriptions
"""
import http.client as httplib
import importlib
import logging
import os
import tempfile
from typing import Dict

import libsbml
import libsedml
import pandas as pd

from .numl import NumlParser


logger = logging.getLogger("sedml-data")


class DataDescriptionParser(object):
    """ Class for parsing DataDescriptions. """

    FORMAT_URN = "urn:sedml:format:"
    FORMAT_NUML = "urn:sedml:format:numl"
    FORMAT_CSV = "urn:sedml:format:csv"
    FORMAT_TSV = "urn:sedml:format:tsv"

    # supported formats
    SUPPORTED_FORMATS = [FORMAT_NUML, FORMAT_CSV, FORMAT_TSV]

    @classmethod
    def parse(
        cls, dd: libsedml.SedDataDescription, workingDir=None
    ) -> Dict[str, pd.Series]:
        """Parses single DataDescription.

        Returns dictionary of data sources {DataSource.id, slice_data}

        :param dd: SED-ML DataDescription
        :param workingDir: workingDir relative to which the sources are resolved
        :return:
        """
        importlib.reload(libsedml)
        assert dd.getTypeCode() == libsedml.SEDML_DATA_DESCRIPTION

        did = dd.getId()
        name = dd.getName()
        source = dd.getSource()

        # -------------------------------
        # Resolve source
        # -------------------------------
        # FIXME: this must work for absolute paths and URL paths
        if workingDir is None:
            workingDir = "."

        # TODO: refactor in general resource module (for resolving anyURI and resource)
        tmp_file = None
        if source.startswith("http") or source.startswith("HTTP"):
            conn = httplib.HTTPConnection(source)
            conn.request("GET", "")
            r1 = conn.getresponse()
            # print(r1.status, r1.reason)
            data = r1.read()
            conn.close()
            try:
                file_str = str(data.decode("utf-8"))
            except:
                # FIXME: too broad
                file_str = str(data)

            tmp_file = tempfile.NamedTemporaryFile("w")
            tmp_file.write(file_str)
            source_path = tmp_file.name
        else:
            source_path = os.path.join(workingDir, source)

        # -------------------------------
        # Find the format
        # -------------------------------
        format = None
        if hasattr(dd, "getFormat"):
            format = dd.getFormat()
        format = cls._determine_format(source_path=source_path, format=format)

        # log data description
        logger.info("-" * 80)
        logger.info("DataDescription: :", dd)
        logger.info("\tid:", did)
        logger.info("\tname:", name)
        logger.info("\tsource", source)
        logger.info("\tformat", format)

        # -------------------------------
        # Parse DimensionDescription
        # -------------------------------
        # FIXME: uses the data_types to check the actual data type
        dim_description = dd.getDimensionDescription()
        data_types = None
        if dim_description is not None:
            data_types = NumlParser.parse_dimension_description(
                dim_description, library=NumlParser.Library.LIBSEDML
            )

        # -------------------------------
        # Load complete data
        # -------------------------------
        data = None
        if format == cls.FORMAT_CSV:
            data = cls._load_csv(path=source_path)
        elif format == cls.FORMAT_TSV:
            data = cls._load_tsv(path=source_path)
        elif format == cls.FORMAT_NUML:
            data = NumlParser.load_numl_data(path=source_path)

        # log data
        logger.info("-" * 80)
        logger.info("Data")
        logger.info("-" * 80)
        if format in [cls.FORMAT_CSV, cls.FORMAT_TSV]:
            logger.info(data.head(10))
        elif format == cls.FORMAT_NUML:
            # multiple result components via id
            for result in data:
                logger.info(result[0])  # rc id
                logger.info(result[1].head(10))  # DataFrame
        logger.info("-" * 80)

        # -------------------------------
        # Process DataSources
        # -------------------------------
        data_sources = {}
        for k, ds in enumerate(dd.getListOfDataSources()):

            dsid = ds.getId()

            # log DataSource
            logger.info("\n\t*** DataSource:", ds)
            logger.info("\t\tid:", ds.getId())
            logger.info("\t\tname:", ds.getName())
            logger.info("\t\tindexSet:", ds.getIndexSet())
            logger.info("\t\tslices")

            # CSV/TSV
            if format in [cls.FORMAT_CSV, cls.FORMAT_TSV]:
                if len(ds.getIndexSet()) > 0:
                    # if index set we return the index
                    data_sources[dsid] = pd.Series(data.index.tolist())
                else:
                    sids = []
                    for slice in ds.getListOfSlices():
                        # FIXME: this does not handle multiple slices for rows
                        # print('\t\t\treference={}; value={}'.format(slice.getReference(), slice.getValue()))
                        sids.append(slice.getValue())

                    # slice values are columns from data frame
                    try:
                        data_sources[dsid] = data[sids].values
                    except KeyError as e:
                        # something does not fit between data and data sources
                        print("-" * 80)
                        print("Format:", format)
                        print("Source:", source_path)
                        print("-" * 80)
                        print(data)
                        print("-" * 80)
                        raise

            # NUML
            elif format == cls.FORMAT_NUML:
                # Using the first results component only in SED-ML L1V3
                rc_id, rc, data_types = data[0]

                index_set = ds.getIndexSet()
                if ds.getIndexSet() and len(ds.getIndexSet()) != 0:
                    # data via indexSet
                    data_source = rc[index_set].drop_duplicates()
                    data_sources[dsid] = data_source
                else:
                    # data via slices
                    for slice in ds.getListOfSlices():
                        reference = slice.getReference()
                        value = slice.getValue()
                        df = rc.loc[rc[reference] == value]
                        # select last column with values
                        data_sources[dsid] = df.iloc[:, -1]

        # log data sources
        logger.info("-" * 80)
        logger.info("DataSources")
        logger.info("-" * 80)
        for key, value in data_sources.items():
            logger.info("{} : {}; shape={}".format(key, type(value), value.shape))
        logger.info("-" * 80)

        # cleanup
        # FIXME: handle in finally
        if tmp_file is not None:
            os.remove(tmp_file)

        importlib.reload(libsbml)

        return data_sources

    @classmethod
    def _determine_format(cls, source_path, format=None):
        """

        :param source_path: path of file
        :param format: format given in the DataDescription
        :return:
        """
        if format is None or format == "":
            is_xml = False
            with open(source_path) as unknown_file:
                start_str = unknown_file.read(1024)
                start_str = start_str.strip()
                if start_str.startswith("<"):
                    is_xml = True

            if is_xml:
                # xml format is numl
                format = cls.FORMAT_NUML  # defaults to numl
            else:
                # format is either csv or tsv
                df_csv = cls._load_csv(source_path)
                df_tsv = cls._load_tsv(source_path)
                if df_csv.shape[1] >= df_tsv.shape[1]:
                    format = cls.FORMAT_CSV
                else:
                    format = cls.FORMAT_TSV

        # base format
        if format.startswith(cls.FORMAT_NUML):
            format = cls.FORMAT_NUML

        # check supported formats
        if format not in cls.SUPPORTED_FORMATS:
            raise NotImplementedError(
                "Format '{}' not supported for DataDescription. Format must be in: {}".format(
                    format, cls.SUPPORTED_FORMATS
                )
            )

        return format

    @classmethod
    def _load_csv(cls, path):
        """Read CSV data from file.

        :param path: path of file
        :return: returns pandas DataFrame with data
        """
        return cls._load_sv(path, separator=",")

    @classmethod
    def _load_tsv(cls, path):
        """Read TSV data from file.

        :param path: path of file
        :return: returns pandas DataFrame with data
        """
        return cls._load_sv(path, separator="\t")

    @classmethod
    def _load_sv(cls, path, separator):
        """Helper function for loading data file from given source.

        CSV files must have a header. Handles file and online resources.

        :param path: path of file.
        :return: pandas data frame
        """
        df = pd.read_csv(
            path,
            sep=separator,
            index_col=False,
            skip_blank_lines=True,
            quotechar='"',
            comment="#",
            skipinitialspace=True,
            na_values="nan",
        )
        return df
